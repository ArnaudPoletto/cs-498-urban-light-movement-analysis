# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from typing import List

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

from lumivid.sky_cloud_segmentation_sp.data import get_data, get_segment_data

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SCS_DATA_PATH = DATA_PATH + 'sky_cloud_segmentation/'

SKY_FINDER_SEG_TE_PATH = SCS_DATA_PATH + "sky_finder_segmented/test/"
SKY_FINDER_SEG_TE_INPUTS_PATH = SKY_FINDER_SEG_TE_PATH + "filtered_images/"
SKY_FINDER_SEG_TE_MASKS_PATH = SKY_FINDER_SEG_TE_PATH + "masks/"
SKY_FINDER_SEG_TE_LABELS_PATH = SKY_FINDER_SEG_TE_PATH + "labels/"

def evaluate_superpixels(
        model,
        scaler,
        image_width: int,
        image_height: int,
        n_segments: int = 100, 
        compactness: int = 10, 
        sigma: int = 0, 
        n_entropy_bins: int = 10, 
        n_color_bins: int = 10,
        visual_words: np.ndarray = None,
        patch_size: int = None,
        selected_features: List[str] = ['mean', 'std', 'entropy', 'color', 'bow', 'texture'],
        separate_clouds: bool = False
        ):

    # Get test data
    _, X, y = get_data(
        SKY_FINDER_SEG_TE_INPUTS_PATH,
        SKY_FINDER_SEG_TE_MASKS_PATH,
        SKY_FINDER_SEG_TE_LABELS_PATH,
        image_width, 
        image_height,
        n_segments = n_segments,
        compactness = compactness,
        sigma = sigma,
        n_entropy_bins = n_entropy_bins,
        n_color_bins = n_color_bins,
        visual_words = visual_words,
        patch_size = patch_size,
        selected_features = selected_features,
        separate_clouds = separate_clouds
    )

    # Scale features
    X = scaler.transform(X)

    # Predict superpixels
    y_pred = model.predict(X)

    # Evaluate
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    jac = jaccard_score(y, y_pred, average='weighted')

    return accuracy, f1, jac

def _get_data_paths():
    """
    Returns a list of input and mask paths for the sky finder dataset.

    Returns:
        input_paths: List of input paths.
        mask_paths: List of mask paths.
    """

    input_paths = []
    mask_paths = []
    label_paths = []

    # Sky finder segmented
    input_folder_paths = sorted([os.path.join(SKY_FINDER_SEG_TE_INPUTS_PATH, folder) for folder in os.listdir(SKY_FINDER_SEG_TE_INPUTS_PATH)])
    for folder_path in input_folder_paths:
        image_paths = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)])
        for image_path in image_paths:
            input_paths.append(image_path)
            folder = folder_path.split("/")[-1]
            mask_paths.append(f"{SKY_FINDER_SEG_TE_MASKS_PATH}{folder}.png")
            label_paths.append(image_path.replace(SKY_FINDER_SEG_TE_INPUTS_PATH, SKY_FINDER_SEG_TE_LABELS_PATH).replace(".jpg", ".png"))

    return input_paths, mask_paths, label_paths

def evaluate_pixels(
        model,
        scaler,
        image_width: int,
        image_height: int,
        n_segments: int = 100, 
        compactness: int = 10, 
        sigma: int = 0, 
        n_entropy_bins: int = 10, 
        n_color_bins: int = 10,
        visual_words: np.ndarray = None,
        patch_size: int = None,
        selected_features: List[str] = ['mean', 'std', 'entropy', 'color', 'bow', 'texture'],
        separate_clouds: bool = False
        ):
    
    input_paths, mask_paths, label_paths = _get_data_paths()

    accuracies = []
    f1s = []
    jacs = []
    for input_path, mask_path, label_path in tqdm(zip(input_paths, mask_paths, label_paths), total=len(input_paths), desc="▶️ Extracting features"):
        # Get image and mask
        image = np.array(Image.open(input_path).resize((image_width, image_height)))
        mask = np.array(Image.open(mask_path).resize((image_width, image_height)))
        label = np.array(Image.open(label_path).resize((image_width, image_height)))

        # Apply mask to image
        image = image * mask[:, :, np.newaxis]
        label = label * mask[:, :, np.newaxis]

        # Get label masks
        r_label = label[:, :, 0]
        g_label = label[:, :, 1]
        b_label = label[:, :, 2]
        ground_mask = np.where(r_label + g_label + b_label < 10, 1, 0).astype(bool)
        sky_mask = np.where(b_label - r_label - g_label > 128, 1, 0).astype(bool)
        light_cloud_mask = np.where(g_label - r_label - b_label > 128, 1, 0).astype(bool)
        thick_cloud_mask = np.where(r_label - g_label - b_label > 128, 1, 0).astype(bool)

        # Get superpixels
        segments, Xi, _ = get_segment_data(
            Image.fromarray(image),
            ground_mask,
            sky_mask,
            light_cloud_mask,
            thick_cloud_mask,
            n_segments = n_segments,
            compactness = compactness,
            sigma = sigma,
            n_entropy_bins = n_entropy_bins,
            n_color_bins = n_color_bins,
            visual_words = visual_words,
            patch_size = patch_size,
            selected_features = selected_features,
            separate_clouds = separate_clouds
            )
        
        # Skip if no superpixels
        if Xi.shape[0] == 0:
            continue
        
        # Scale features
        Xi = scaler.transform(Xi)

        # Predict superpixels
        y_pred = model.predict(Xi)

        # Construct prediction image
        y_pred_image = np.zeros((image_height, image_width))
        n_segments = np.max(segments)
        for i in range(n_segments):
            y_pred_image[segments == i+1] = y_pred[i] + 1
        # Mask out ground
        y_pred_image[ground_mask] = 0

        # Construct ground truth image
        y_target_image = np.zeros((image_height, image_width))
        y_target_image[sky_mask] = 1
        y_target_image[light_cloud_mask] = 2
        y_target_image[thick_cloud_mask] = 3 if separate_clouds else 2

        # Flatten images, remove ground
        y_pred_image = y_pred_image[~ground_mask]
        y_target_image = y_target_image[~ground_mask]

        # Evaluate
        accuracy = accuracy_score(y_target_image, y_pred_image)
        f1 = f1_score(y_target_image, y_pred_image, average='weighted')
        jac = jaccard_score(y_target_image, y_pred_image, average='weighted')

        accuracies.append(accuracy)
        f1s.append(f1)
        jacs.append(jac)

    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1s)
    mean_jac = np.mean(jacs)
    std_accuracy = np.std(accuracies)
    std_f1 = np.std(f1s)
    std_jac = np.std(jacs)

    return mean_accuracy, mean_f1, mean_jac, std_accuracy, std_f1, std_jac
