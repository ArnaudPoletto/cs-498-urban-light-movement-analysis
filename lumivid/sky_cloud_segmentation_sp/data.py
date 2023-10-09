# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from typing import List

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SCS_DATA_PATH = DATA_PATH + 'sky_cloud_segmentation/'

SKY_FINDER_SEG_PATH = SCS_DATA_PATH + "sky_finder_segmented/"
SKY_FINDER_SEG_INPUTS_PATH = SKY_FINDER_SEG_PATH + "filtered_images/"
SKY_FINDER_SEG_MASKS_PATH = SKY_FINDER_SEG_PATH + "masks/"

from lumivid.sky_cloud_segmentation_sp.visual_words import get_image_channels, get_visual_word_indices

def _get_superpixel_segments(image: np.ndarray, mask: np.ndarray, n_segments: int, compactness: float, sigma: float):
    """
    Apply superpixel segmentation to image.

    Args:
        image (np.ndarray): Image.
        n_segments (int): Number of segments.
        compactness (float): Compactness.
        sigma (float): Sigma.

    Returns:
        np.ndarray: Segments.
    """

    assert n_segments > 0, f"❌ Number of segments must be positive: {n_segments}"

    # Apply mask to image
    segments = slic(
        image = image, 
        mask = mask, 
        n_segments = n_segments, 
        compactness = compactness, 
        sigma = sigma
    )

    return segments

def get_segment_data(
        image: Image.Image, 
        ground_mask: Image.Image = None,
        sky_mask: Image.Image = None,
        cloud_mask: Image.Image = None, 
        n_segments: int = 100,
        compactness: float = 10,
        sigma: float = 0,
        n_entropy_bins: int = 10, 
        n_color_bins: int = 10,
        visual_words: np.ndarray = None,
        patch_size: int = None,
        selected_features: List[str] = ['mean', 'std', 'entropy', 'color', 'bow', 'texture']
        ):
    """
    Get features from image and segments.

    Args:
        image (Image.Image): Image.
        ground_mask (Image.Image): Ground mask.
        sky_mask (Image.Image): Sky mask.
        cloud_mask (Image.Image): Cloud mask.
        n_segments (int): Number of segments.
        compactness (float): Compactness.
        sigma (float): Sigma.
        n_entropy_bins (int): Number of entropy bins.
        n_color_bins (int): Number of color bins.

    Returns:
        np.ndarray: Input features.
        np.ndarray: Ground truth labels.
    """

    if visual_words is not None and patch_size is None:
        raise ValueError("❌ Patch size must be given if visual words are given.")

    # Apply mask to image if given
    if ground_mask is None:
        mask = None
        image = np.array(image)
    else:
        mask = ~np.array(ground_mask)
        image = np.array(image) * mask[:, :, np.newaxis]

    # Get grayscale image
    gray_image = np.mean(image, axis=-1).astype(np.uint8)

    # Get image features
    image_features = get_image_channels(image)
    n_channels = image_features.shape[-1]

    # Get segments
    segments = _get_superpixel_segments(image, mask, n_segments, compactness, sigma)

    # Get whole image statistics
    # Get image mean and standard deviation
    if 'mean' in selected_features:
        image_mean = image_features[mask].mean(axis=(0, 1))

    if 'std' in selected_features:
        image_std = image_features[mask].std(axis=(0, 1))

    # Get image color entropies
    if 'entropy' in selected_features:
        image_entropy = np.zeros(n_channels)
        for j in range(n_channels):
            value_range = (0, 255) if j <= 1 else (0., 2.)
            probs, _ = np.histogram(image_features[mask, j], bins=n_entropy_bins, density=True, range=value_range)
            probs = np.nan_to_num(probs)
            log_probs = np.log2(probs, where=probs!=0)
            entropy = -np.sum(probs * log_probs)
            image_entropy[j] = entropy if not np.isnan(entropy) else 0

    # Get image BOW histogram
    if 'bow' in selected_features and visual_words is not None:
        n_visual_words = visual_words.shape[0] if visual_words is not None else 0
        image_bow = get_visual_word_indices(image, visual_words, patch_size)

    # Get features for each segment
    n_segments = segments.max()
    features = []
    for i in range(n_segments):
        segment_mask = segments == i+1
        segment = image_features[segment_mask]

        # Create a list to hold selected features
        segment_features = []

        # Skip if segment is empty (should not happen)
        if len(segment) == 0:
            print(f"❌ Segment {i+1} is empty.")
            continue

        # Get mean and standard deviation
        if 'mean' in selected_features:
            segment_mean = image_mean - segment.mean(axis=0)
            segment_features += segment_mean.tolist()
        if 'std' in selected_features:
            segment_std = image_std - segment.std(axis=0)
            segment_features += segment_std.tolist()

        # Get color entropies
        if 'entropy' in selected_features:
            for j in range(n_channels):
                value_range = (0, 255) if j <= 1 else (0., 2.)
                probs, _ = np.histogram(segment[:, j], bins=n_entropy_bins, density=True, range=value_range)
                probs = np.nan_to_num(probs) # Replace nans by 0
                log_probs = np.log2(probs, where=probs!=0)
                entropy = -np.sum(probs * log_probs)
                entropy = entropy if not np.isnan(entropy) else 0
                entropy = image_entropy[j] - entropy
                segment_features.append(entropy)

        # Get color histograms
        if 'color' in selected_features:
            for j in range(n_channels):
                value_range = (0, 255) if j <= 1 else (0., 2.)
                hist, _ = np.histogram(segment[:, j], bins=n_color_bins, density=True, range=value_range)
                hist = np.nan_to_num(hist) # Replace nans by 0
                segment_features += hist.tolist()

        # Get BOW histogram
        if 'bow' in selected_features and visual_words is not None:
            segment_bow = image_bow[segments == i+1]
            segment_bow_hist, _ = np.histogram(segment_bow, bins=n_visual_words, density=True, range=(0, n_visual_words))
            segment_bow_hist = np.nan_to_num(segment_bow_hist) # Replace nans by 0
            segment_features += segment_bow_hist.tolist()

        # Get texture features
        if 'texture' in selected_features:
            # Extract the 2D superpixel from the original grayscale image
            superpixel = gray_image * (segment_mask.astype(np.uint8))

            # Calculating the bounding box of the superpixel to reduce the size
            x, y, w, h = cv2.boundingRect(segment_mask.astype(np.uint8))
            superpixel = superpixel[y:y+h, x:x+w]

            # Need at least 2 values to compute GLCM
            if len(np.unique(superpixel)) >= 2:
                # Calculate GLCM features only for the non-zero values
                distances = [1]
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                glcm = graycomatrix(superpixel, distances, angles, levels=256, symmetric=True, normed=True)
                
                contrast = np.mean(graycoprops(glcm, 'contrast'))
                disimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
                homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
                energy = np.mean(graycoprops(glcm, 'energy'))
                correlation = np.mean(graycoprops(glcm, 'correlation'))
                asm = np.mean(graycoprops(glcm, 'ASM'))

                texture_features = np.array([contrast, disimilarity, homogeneity, energy, correlation, asm])
                segment_features += texture_features.tolist()
            else:
                segment_features += [0] * 6

        features.append(segment_features)

    features = np.array(features)

    # Get ground truth labels if masks are given
    if ground_mask is None or sky_mask is None or cloud_mask is None:
        labels = None
    else:
        labels = np.zeros(n_segments)
        # Get most common label for each segment
        for i in range(n_segments):
            n_ground = ground_mask[segments == i+1].sum()
            n_sky = sky_mask[segments == i+1].sum()
            n_cloud = cloud_mask[segments == i+1].sum()
            labels[i] = np.argmax([n_ground, n_sky, n_cloud])

    return segments, features, labels

def get_data_paths():
    """
    Returns a list of input and mask paths for the sky finder dataset.

    Returns:
        input_paths: List of input paths.
        mask_paths: List of mask paths.
    """

    input_paths = []
    mask_paths = []

    # Sky finder segmented
    input_folder_paths = sorted([os.path.join(SKY_FINDER_SEG_INPUTS_PATH, folder) for folder in os.listdir(SKY_FINDER_SEG_INPUTS_PATH)])
    for folder_path in input_folder_paths:
        image_paths = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)])
        for image_path in image_paths:
            input_paths.append(image_path)
            mask_paths.append(image_path.replace(SKY_FINDER_SEG_INPUTS_PATH, SKY_FINDER_SEG_MASKS_PATH).replace(".jpg", ".png"))

    return input_paths, mask_paths

def get_data(
        image_width: int,
        image_height: int,
        n_segments: int = 100, 
        compactness: int = 10, 
        sigma: int = 0, 
        n_entropy_bins: int = 10, 
        n_color_bins: int = 10,
        visual_words: np.ndarray = None,
        patch_size: int = None,
        train_split: float = 0.8,
        selected_features: List[str] = ['mean', 'std', 'entropy', 'color', 'bow', 'texture']
        ):
    """
    Returns a dataset of features and labels.

    Args:
        n_segments: Number of segments for SLIC.
        compactness: Compactness parameter for SLIC.
        sigma: Sigma parameter for SLIC.
        n_entropy_bins: Number of bins for entropy.
        n_color_bins: Number of bins for color.

    Returns:
        dataset: Dataset of unnormalized features and labels.
    """

    input_paths, mask_paths = get_data_paths()

    X = np.array([])
    y = np.array([])
    for input_path, mask_path in tqdm(zip(input_paths, mask_paths), total=len(input_paths), desc="▶️ Extracting features"):
        # Get image and mask
        image = Image.open(input_path).resize((image_width, image_height))
        mask_image = Image.open(mask_path).resize((image_width, image_height))

        r_mask = np.array(mask_image)[:, :, 0]
        g_mask = np.array(mask_image)[:, :, 1]
        b_mask = np.array(mask_image)[:, :, 2]

        ground_mask = np.where(r_mask + g_mask + b_mask < 10, 1, 0).astype(bool)
        sky_mask = np.where(b_mask - r_mask - g_mask > 245, 1, 0).astype(bool)
        cloud_mask = np.where(ground_mask + sky_mask == 0, 1, 0).astype(bool)

        _, Xi, yi = get_segment_data(
            image,
            ground_mask,
            sky_mask,
            cloud_mask,
            n_segments = n_segments,
            compactness = compactness,
            sigma = sigma,
            n_entropy_bins = n_entropy_bins,
            n_color_bins = n_color_bins,
            visual_words = visual_words,
            patch_size = patch_size,
            selected_features = selected_features
            )
        
        X = np.concatenate((X, Xi), axis=0) if X.size else Xi
        y = np.concatenate((y, yi), axis=0) if y.size else yi

    # Split into train and test sets
    n_samples = X.shape[0]
    n_train_samples = int(n_samples * train_split)
    indices = np.arange(n_samples)
    np.random.shuffle(indices) # Shuffle indices
    X, y = X[indices], y[indices]
    X_train, y_train = X[:n_train_samples], y[:n_train_samples]
    X_test, y_test = X[n_train_samples:], y[n_train_samples:]

    return X_train, y_train, X_test, y_test