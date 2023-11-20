# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

import cv2
import numpy as np
import supervision as sv
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
GROUNDING_DINO_CONFIG_PATH = DATA_PATH + "gsam/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = DATA_PATH + "gsam/groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_b"
SAM_CHECKPOINT_PATH = DATA_PATH + "gsam/sam_vit_b_01ec64.pth"

CLASSES = [
    "buildings",
    "car", "bus",
    "person", 
    "forest", "trees", "grass", "bushes", 
    "sky", 
]
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
    "sky": "sky",
} 
TYPE_COLORS = {
    "background": (0, 0, 0),
    "vehicle": (255, 0, 0),
    "person": (255, 255, 0),
    "sky": (0, 0, 255),
    "vegetation": (0, 255, 0)
}
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
SIZE_FACTOR = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gdino_model():
    gdino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    return gdino_model

def get_sam_predictor():
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    return sam_predictor

def get_detections(
        gdino_model, 
        image, 
        classes = CLASSES,
        box_threshold = BOX_THRESHOLD, 
        text_threshold = TEXT_THRESHOLD,
    ):
    detections = gdino_model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return detections

def apply_nms_processing(detections, nms_threshold=NMS_THRESHOLD):
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        nms_threshold
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = [class_id if class_id is not None else 0 for class_id in detections.class_id[nms_idx]]

    return detections

def get_segmentation(
        sam_predictor,
        image,
        detections
    ):
    # Get masks of shape (n_boxes, h, w)
    sam_predictor.set_image(image)
    segmentation_masks = []
    for box in detections.xyxy:
        masks, scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        segmentation_masks.append(masks[index])
    segmentation_masks = np.array(segmentation_masks)

    detections.mask = segmentation_masks

    return detections

def get_class_dict():
    class_dict = {0: 'background'}
    for i, class_name in enumerate(CLASSES):
        class_dict[i+1] = class_name

    return class_dict

def downscale_image(bgr_image: np.ndarray, size_factor=SIZE_FACTOR):
    bgr_image = cv2.resize(bgr_image, (bgr_image.shape[1] // size_factor, bgr_image.shape[0] // size_factor))

    return bgr_image

def upscale_masks(detections: np.ndarray, size_factor=SIZE_FACTOR):
    masks = detections.mask

    # Upscale all masks independently
    scaled_masks = np.zeros((masks.shape[0], masks.shape[1] * size_factor, masks.shape[2] * size_factor), dtype=np.uint8)
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.uint8)
        scaled_masks[i] = cv2.resize(mask, (mask.shape[1] * size_factor, mask.shape[0] * size_factor))

    detections.mask = scaled_masks.astype(bool)
    return detections

def get_segmentation_masks_from(gdino_model, sam_predictor, bgr_image: np.ndarray, size_factor=SIZE_FACTOR):
    # Make image divisible by size_factor
    h, w = bgr_image.shape[:2]
    h = h // size_factor * size_factor
    w = w // size_factor * size_factor
    bgr_image = bgr_image[:h, :w, :]

    bgr_image = downscale_image(bgr_image, size_factor)

    # Read image and get detections
    detections = get_detections(gdino_model, bgr_image)
    detections = apply_nms_processing(detections)

    # Get and upscale segmentation mask
    detections = get_segmentation(
        sam_predictor, 
        cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), 
        detections
    )
    detections = upscale_masks(detections, size_factor)

    # Get class dictionary
    class_dict = get_class_dict()

    return detections, class_dict, CLASS_TO_TYPES

def get_segmentation_mask_color(segmentation_mask, class_dict):
    # Get segmentation mask with colors
    segmentation_mask_colored = np.vectorize(lambda x: TYPE_COLORS[CLASS_TO_TYPES[class_dict[x]]])(segmentation_mask)
    segmentation_mask_colored = np.array(segmentation_mask_colored, dtype=np.uint8)

    return segmentation_mask_colored

def show_segmentation_mask(bgr_image, segmentation_mask, class_dict, size_factor=SIZE_FACTOR):
    # Make image divisible by size_factor
    h, w = bgr_image.shape[:2]
    h = h // size_factor * size_factor
    w = w // size_factor * size_factor
    bgr_image = bgr_image[:h, :w, :]

    # Get segmentation mask with colors
    segmentation_mask_colored = get_segmentation_mask_color(segmentation_mask, class_dict)
    bgr_image = bgr_image.transpose(2, 0, 1)

    # Mix image and segmentation mask
    alpha = 0.5
    segmentation_mask_colored = cv2.addWeighted(bgr_image[::-1, :, :], alpha, segmentation_mask_colored, 1-alpha, 0)

    # Show image
    plt.figure(figsize=(15, 15))
    plt.imshow(segmentation_mask_colored.transpose(1, 2, 0))
    legend_patches = [Patch(color=np.array(TYPE_COLORS[type_name]) / 255, label=type_name) for type_name in TYPE_COLORS.keys()]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def show_bounding_boxes(bgr_image, detections):
    labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]

    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=bgr_image.copy(), detections=detections, labels=labels)

    plt.figure(figsize=(15, 15))
    plt.imshow(annotated_image)
    plt.show()
