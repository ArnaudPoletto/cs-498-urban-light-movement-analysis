import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import torch
import torchvision
from torch import nn
import supervision as sv
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large

from src.utils.random_utils import set_seed
from src.utils.video_utils import get_video, get_video_frame_iterator

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
GENERATED_PATH = str(GLOBAL_DIR / "generated") + "/"
GENERATED_MOTION_PATH = f"{GENERATED_PATH}motion/"
SCENES_PATH = f"{DATA_PATH}ldr/processed/"
MOTION_PATH = f"{DATA_PATH}motion/"
GSAM_PATH = f"{MOTION_PATH}gsam/"
GROUNDING_DINO_CONFIG_PATH = f"{GSAM_PATH}GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = f"{GSAM_PATH}groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_b"
SAM_CHECKPOINT_PATH = f"{GSAM_PATH}sam_vit_b_01ec64.pth"

FRAME_STEP = 25
SIZE_FACTOR = 1
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
CLASSES = [
    "buildings",
    "car",
    "bus",
    "person",
    "forest",
    "trees",
    "grass",
    "bushes",
    "sky",
]
TYPE_COLORS = {
    "background": (0, 0, 0),
    "vehicle": (255, 0, 0),
    "person": (0, 180, 180),
    "vegetation": (0, 110, 51),
}
CLASS_TO_TYPES = {
    "background": "background",
    "buildings": "background",
    "car": "vehicle",
    "bus": "vehicle",
    "person": "person",
    "forest": "vegetation",
    "trees": "vegetation",
    "grass": "vegetation",
    "bushes": "vegetation",
    "sky": "background",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def save_bounding_boxes(
    video_name: str,
    rgb_image: np.ndarray,
    detections: torch.Tensor,
) -> None:
    """
    Show the bounding boxes of the detections.

    Args:
        video_name (str): The video name
        rgb_image (np.ndarray): The image
        detections (torch.Tensor): The detections
    """
    # Get image and labels
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    # Show image with bounding boxes
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(
        scene=rgb_image.copy(), detections=detections, labels=labels
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(annotated_image)
    plt.title(f"{video_name} Bounding Boxes")
    plt.axis("off")
    plt.tight_layout()

    # save image
    bounding_boxes_path = f"{GENERATED_MOTION_PATH}bounding_boxes/"
    os.makedirs(bounding_boxes_path, exist_ok=True)
    bounding_boxes_file_path = f"{bounding_boxes_path}{video_name}.png"
    plt.savefig(bounding_boxes_file_path)
    plt.close()

    print(f"💾 Bounding boxes saved at {bounding_boxes_file_path}")


def get_segmentation_mask_color(
    segmentation_mask: np.ndarray,
    class_dict: dict,
) -> np.ndarray:
    """
    Get the segmentation mask with colors.

    Args:
        segmentation_mask (np.ndarray): The segmentation mask
        class_dict (dict): The class dictionary

    Returns:
        np.ndarray: The segmentation mask with colors
    """
    # Get segmentation mask with colors
    segmentation_mask_colored = np.vectorize(
        lambda x: TYPE_COLORS[CLASS_TO_TYPES[class_dict[x]]]
    )(segmentation_mask)
    segmentation_mask_colored = np.array(segmentation_mask_colored, dtype=np.uint8)

    return segmentation_mask_colored


def save_flow_segmentation(
    video_name: str,
    rgb_frame: np.ndarray,
    segmentation_masks: np.ndarray,
    class_ids: np.ndarray,
    class_dict: dict,
    optical_flow: np.ndarray,
    step: int = 10,
    alpha: float = 0.5,
) -> None:
    """
    Show the optical flow with the segmentation masks.

    Args:
        video_name (str): The video name
        rgb_frame (np.ndarray): The image
        segmentation_masks (np.ndarray): The segmentation masks
        class_ids (np.ndarray): The class ids
        class_dict (dict): The class dictionary
        optical_flow (np.ndarray): The optical flow
        step (int, optional): The step, defaults to 10
        alpha (float, optional): The alpha, defaults to 0.5
    """
    # Tensor of masks into single mask with class ids
    all_segmentation_masks = np.zeros_like(segmentation_masks[0]).astype(np.int32)
    for i in range(segmentation_masks.shape[0]):
        segmentation_mask = segmentation_masks[i]
        class_id = class_ids[i]
        all_segmentation_masks[segmentation_mask] = class_id + 1

    # Map the segmentation classes to the corresponding colors
    segmentation_colors = get_segmentation_mask_color(
        all_segmentation_masks, class_dict
    )
    segmentation_colors = segmentation_colors.transpose(1, 2, 0)
    rgb_mixed = np.where(
        segmentation_colors.sum(axis=2)[..., np.newaxis] == 0,
        rgb_frame,
        rgb_frame * (1 - alpha) + segmentation_colors * alpha,
    )
    rgb_mixed = rgb_mixed.astype(np.uint8)

    # Downsample the flow for a cleaner quiver plot
    optical_flow_downsampled = optical_flow[::step, ::step]
    y, x = np.mgrid[0 : rgb_mixed.shape[0] : step, 0 : rgb_mixed.shape[1] : step]

    # Get the colors for the downsampled segmentation at the points where we will place the quiver arrows
    colors_for_quiver = segmentation_colors[y.flatten(), x.flatten()]

    # Normalize the color values
    quiver_colors = colors_for_quiver / 255.0

    # Plot the flow vectors
    plt.figure(figsize=(10, 6))
    plt.quiver(
        x.flatten(),
        y.flatten(),
        optical_flow_downsampled[..., 0].flatten(),
        optical_flow_downsampled[..., 1].flatten(),
        angles="xy",
        scale_units="xy",
        scale=0.25,
        color=quiver_colors,
        width=0.0015,
    )
    plt.imshow(rgb_mixed, interpolation="none")
    plt.title(f"{video_name} Segmented Optical Flow")
    plt.axis("off")
    plt.tight_layout()

    # Legend with mpatches
    legend_patches = []
    for class_name, color in TYPE_COLORS.items():
        color = np.array(color) / 255.0
        legend_patches.append(mpatches.Patch(color=color, label=class_name))
    plt.gcf().legend(handles=legend_patches, loc="lower center", ncol=2)

    # Save image
    flow_segmentation_path = f"{GENERATED_MOTION_PATH}flow_segmentation/"
    os.makedirs(flow_segmentation_path, exist_ok=True)
    flow_segmentation_file_path = f"{flow_segmentation_path}{video_name}.png"
    plt.savefig(flow_segmentation_file_path)
    plt.close()

    print(f"💾 Flow segmentation saved at {flow_segmentation_file_path}")


def get_gdino_model() -> nn.Module:
    """
    Get the grounding dino model.

    Returns:
        nn.Module: The grounding dino model
    """
    gdino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    )

    return gdino_model


def get_sam_predictor() -> nn.Module:
    """
    Get the segment anything model.

    Returns:
        nn.Module: The segment anything model
    """
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    return sam_predictor


def get_optical_flow_model() -> nn.Module:
    """
    Get the optical flow model.

    Returns:
        nn.Module: The optical flow model
    """

    return raft_large(weights="C_T_SKHT_V2").to(DEVICE)


def apply_nms_processing(
    detections: torch.Tensor, nms_threshold: float = NMS_THRESHOLD
) -> torch.Tensor:
    """
    Apply non-maximum suppression to the detections.

    Args:
        detections (torch.Tensor): The detections
        nms_threshold (float, optional): The nms threshold, defaults to NMS_THRESHOLD

    Returns:
        torch.Tensor: The detections
    """
    nms_idx = (
        torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold,
        )
        .numpy()
        .tolist()
    )

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = [
        class_id if class_id is not None else 0
        for class_id in detections.class_id[nms_idx]
    ]

    return detections


def get_segmentation(
    sam_predictor: nn.Module, image: np.ndarray, detections: torch.Tensor
) -> torch.Tensor:
    """
    Get the segmentation masks from an image.

    Args:
        sam_predictor (nn.Module): The sam predictor
        image (np.ndarray): The image
        detections (torch.Tensor): The detections

    Returns:
        torch.Tensor: The detections
    """
    # Get masks of shape (n_boxes, h, w)
    sam_predictor.set_image(image)
    segmentation_masks = []
    for box in detections.xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        segmentation_masks.append(masks[index])
    segmentation_masks = np.array(segmentation_masks)

    detections.mask = segmentation_masks

    return detections


def get_class_dict():
    """
    Get the class dictionary.

    Returns:
        dict: The class dictionary
    """
    class_dict = {0: "background"}
    for i, class_name in enumerate(CLASSES):
        class_dict[i + 1] = class_name

    return class_dict


def downscale_image(image: np.ndarray, size_factor: float = SIZE_FACTOR) -> np.ndarray:
    """
    Downscale an image.

    Args:
        image (np.ndarray): The image
        size_factor (int, optional): The size factor, defaults to SIZE_FACTOR

    Returns:
        np.ndarray: The downscaled image
    """
    image = cv2.resize(
        image,
        (image.shape[1] // size_factor, image.shape[0] // size_factor),
    )

    return image


def upscale_masks(
    detections: np.ndarray, size_factor: float = SIZE_FACTOR
) -> np.ndarray:
    """
    Upscale the masks of the detections.

    Args:
        detections (np.ndarray): The detections
        size_factor (int, optional): The size factor, defaults to SIZE_FACTOR

    Returns:
        np.ndarray: The upscaled masks
    """
    masks = detections.mask

    # Upscale all masks independently
    scaled_masks = np.zeros(
        (masks.shape[0], masks.shape[1] * size_factor, masks.shape[2] * size_factor),
        dtype=np.uint8,
    )
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.uint8)
        scaled_masks[i] = cv2.resize(
            mask, (mask.shape[1] * size_factor, mask.shape[0] * size_factor)
        )

    detections.mask = scaled_masks.astype(bool)

    return detections


def get_segmentation_masks_from(
    gdino_model: nn.Module,
    sam_predictor: nn.Module,
    rgb_image: np.ndarray,
    size_factor: float = SIZE_FACTOR,
) -> torch.Tensor:
    """
    Get the segmentation masks from an image.

    Args:
        gdino_model (nn.Module): The gdino model
        sam_predictor (nn.Module): The sam predictor
        rgb_image (np.ndarray): The image
        size_factor (int, optional): The size factor, defaults to SIZE_FACTOR

    Returns:
        torch.Tensor: The detections
    """
    # Make image divisible by size_factor
    h, w = rgb_image.shape[:2]
    h = h // size_factor * size_factor
    w = w // size_factor * size_factor
    rgb_image = rgb_image[:h, :w, :]

    # Downscale image and detections
    rgb_image = downscale_image(rgb_image, size_factor)
    detections = gdino_model.predict_with_classes(
        image=rgb_image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    detections = apply_nms_processing(detections)

    # Get and upscale segmentation mask
    detections = get_segmentation(sam_predictor, rgb_image, detections)
    detections = upscale_masks(detections, size_factor)

    # Get class dictionary

    return detections


def get_flow(
    optical_flow_model: nn.Module,
    rgb_frame1: np.ndarray,
    rgb_frame2: np.ndarray,
    shape_factor: int = 2,
    multiple: int = 8,
) -> np.ndarray:
    """
    Get the optical flow between two frames.

    Args:
        optical_flow_model (nn.Module): The optical flow model
        rgb_frame1 (np.ndarray): The first frame
        rgb_frame2 (np.ndarray): The second frame
        shape_factor (int, optional): The shape factor, defaults to 2
        multiple (int, optional): The multiple, defaults to 8

    Returns:
        np.ndarray: The optical flow
    """
    optical_flow_model.eval()

    old_shape = (rgb_frame1.shape[0], rgb_frame1.shape[1])
    new_shape = (
        (old_shape[0] // (shape_factor * multiple)) * multiple,
        (old_shape[1] // (shape_factor * multiple)) * multiple,
    )

    # Get tensor from frame
    frame1_tensor = (
        torch.from_numpy(rgb_frame1).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    )
    frame2_tensor = (
        torch.from_numpy(rgb_frame2).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    )
    transforms = T.Compose(
        [
            T.Resize(size=(new_shape[0], new_shape[1])),
            T.Lambda(lambda x: x / 255.0),  # [0, 255] to [0, 1]
            T.Normalize(mean=[0.5], std=[0.5]),  # [0, 1] to [-1, 1]
        ]
    )
    frame1_tensor = transforms(frame1_tensor)
    frame2_tensor = transforms(frame2_tensor)

    with torch.no_grad():
        optical_flow = optical_flow_model(frame1_tensor, frame2_tensor)
        optical_flow = optical_flow[0].squeeze().cpu().numpy()

        # Resize flow to original shape
        optical_flow = optical_flow.transpose(1, 2, 0)
        optical_flow = cv2.resize(optical_flow, (old_shape[1], old_shape[0]))

    # Free GPU memory
    del frame1_tensor
    del frame2_tensor
    torch.cuda.empty_cache()

    return optical_flow


def analyze_scene(
    video_path: str,
    gdino_model: nn.Module,
    sam_predictor: nn.Module,
    optical_flow_model: nn.Module,
    frame_step: int = 1,
    split: bool = False,
    masked: bool = False,
    reframed: bool = False,
):
    """
    Analyze a scene.

    Args:
        video_path (str): The path to the video
        gdino_model (nn.Module): The gdino model
        sam_predictor (nn.Module): The sam predictor
        optical_flow_model (nn.Module): The optical flow model
        frame_step (int, optional): The frame step, defaults to 1
        split (bool, optional): Whether to split the frame in two, defaults to False
        masked (bool, optional): Whether to apply the mask to the frame, defaults to False
        reframed (bool, optional): Whether to reframe the frame, defaults to False

    Returns:
        dict: The object class dictionary
        dict: The mean class counts
    """
    video_name = video_path.split("/")[-1].split(".")[0]
    video = get_video(video_path)
    frame_it = get_video_frame_iterator(
        video, frame_step=frame_step, split=split, masked=masked, reframed=reframed
    )

    class_magnitudes = {}
    class_counts = {}

    previous_rgb_frame = None
    saved_plots = False
    n_frames = 0
    for bgr_frame, _ in tqdm(frame_it):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # BGR to RGB

        # Reframe shape like previous frame if needed
        if (
            previous_rgb_frame is not None
            and rgb_frame.shape != previous_rgb_frame.shape
        ):
            rgb_frame = cv2.resize(
                rgb_frame, (previous_rgb_frame.shape[1], previous_rgb_frame.shape[0])
            )

        # Get statistics
        if previous_rgb_frame is not None:
            # Get segmentation mask
            detections = get_segmentation_masks_from(
                gdino_model, sam_predictor, previous_rgb_frame
            )
            segmentation_masks = detections.mask
            class_ids = (
                detections.class_id
            )  # Get class ids, all objects detected by grounding dino (no background)
            class_dict = get_class_dict()  # Key 0 is background! (ids shifted by 1)
            types = [
                CLASS_TO_TYPES[class_dict[class_id + 1]] for class_id in class_ids
            ]  # Get types of objects detected by grounding dino

            # Get optical flow
            flow = get_flow(optical_flow_model, previous_rgb_frame, rgb_frame)
            magnitudes = np.linalg.norm(flow, axis=2)

            # Add magnitudes and count to corresponding class
            # Only for classes that are not vegetation
            class_dict = get_class_dict()
            for i, type in enumerate(types):
                if type == "vegetation":
                    continue

                # Get mean magnitude of detected object
                segmentation_mask = segmentation_masks[i]
                magnitudes_for_class = magnitudes[segmentation_mask]
                mean_magnitude = np.mean(magnitudes_for_class)

                # Add to class
                if type not in class_magnitudes:
                    class_magnitudes[type] = 0
                    class_counts[type] = 0
                class_magnitudes[type] += mean_magnitude
                class_counts[type] += 1

            # Add magnitudes for vegetation class, vegetation acts as a single mask
            # Filter out masks that are not vegetation
            vegetation_mask = segmentation_masks[np.array(types) == "vegetation"]
            # Merge all vegetation masks
            vegetation_mask = np.any(vegetation_mask, axis=0)

            # Get mean magnitude of vegetation
            magnitudes_for_class = magnitudes[vegetation_mask]
            mean_magnitude = np.mean(magnitudes_for_class)

            # Add to class
            if "vegetation" not in class_magnitudes:
                class_magnitudes["vegetation"] = 0
                class_counts["vegetation"] = 0
            class_magnitudes["vegetation"] += mean_magnitude
            class_counts["vegetation"] += 1

            # Show frames
            if not saved_plots:
                save_bounding_boxes(video_name, previous_rgb_frame, detections)
                save_flow_segmentation(
                    video_name,
                    previous_rgb_frame,
                    segmentation_masks,
                    class_ids,
                    class_dict,
                    flow,
                )
                saved_plots = True

        previous_rgb_frame = rgb_frame
        n_frames += 1

    # Get magnitudes for each type, normalized by number of frames
    classes = ["vegetation", "person", "vehicle"]
    class_magnitudes_keys = np.array(
        [e for e in class_magnitudes.keys() if e in classes]
    )
    class_magnitudes_values = np.array(
        [class_magnitudes[key] for key in class_magnitudes_keys]
    )
    class_magnitudes_values = class_magnitudes_values / n_frames

    # Get mean number of objects per class
    mean_class_counts = {k: v / n_frames for k, v in class_counts.items()}

    # Get dicts
    object_class_dict = {
        k: v for k, v in zip(class_magnitudes_keys, class_magnitudes_values)
    }

    return object_class_dict, mean_class_counts


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--part", type=int, choices=[1, 2, 3], default=3, help="The scenes part to analyze")
    args = parser.parse_args()
    part = args.part

    print(f"📝 Analyzing scenes part {part}.")

    set_seed(SEED)

    gdino_model = get_gdino_model()
    sam_predictor = get_sam_predictor()
    optical_flow_model = get_optical_flow_model()

    final_statistics = {}

    scene_paths = [
        os.path.join(SCENES_PATH, scene)
        for scene in os.listdir(SCENES_PATH)
        if f"P{part}" in scene
    ]
    for scene_path in tqdm(scene_paths, desc="▶️  Analyzing scenes"):
        object_class_dict, mean_class_counts = analyze_scene(
            scene_path,
            gdino_model,
            sam_predictor,
            optical_flow_model,
            frame_step=FRAME_STEP,
            split=False,
            masked=True,
            reframed=True,
        )

        final_statistics[scene_path.split("/")[-1]] = {
            "object_class_dict": object_class_dict,
            "mean_class_counts": mean_class_counts,
        }

        # Save statistics
        if not os.path.exists(MOTION_PATH):
            os.makedirs(MOTION_PATH)
        statistics_path = f"{MOTION_PATH}statistics_part{part}.npy"
        np.save(statistics_path, final_statistics)
        print(f"💾 Statistics saved at {statistics_path}")
