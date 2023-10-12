# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from typing import List, Tuple

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SCS_DATA_PATH = DATA_PATH + 'sky_cloud_segmentation/'

from lumivid.sky_cloud_segmentation_sp.visual_words import get_image_channels, get_visual_word_indices
from lumivid.utils.model_utils import split_data

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
    assert compactness >= 0, f"❌ Compactness must be non-negative: {compactness}"
    assert sigma >= 0, f"❌ Sigma must be non-negative: {sigma}"

    # Apply mask to image
    segments = slic(
        image = image, 
        mask = mask, 
        n_segments = n_segments, 
        compactness = compactness, 
        sigma = sigma
    )

    return segments

def _get_segments_mean(image: np.ndarray, mask: np.ndarray, segments: np.ndarray):
    """
    Get mean of each segment, substracted from image mean and not.

    Args:
        image (np.ndarray): Image.
        mask (np.ndarray): Mask.
        segments (np.ndarray): Segments.

    Returns:
        np.ndarray: Mean of each segment.
    """

    n_segments = segments.max()
    n_channels = image.shape[-1]
    segments_mean = np.zeros((n_segments, n_channels * 2))
    for j in range(n_channels):
        image_mean = image[mask, j].mean()
        for i in range(n_segments):
            segment_mean = image[segments == i+1, j].mean()

            segments_mean[i, 2*j] = image_mean - segment_mean
            segments_mean[i, 2*j+1] = segment_mean

    return segments_mean

def _get_segments_std(image: np.ndarray, mask: np.ndarray, segments: np.ndarray):
    """
    Get standard deviation of each segment, substracted from image standard deviation and not.

    Args:
        image (np.ndarray): Image.
        mask (np.ndarray): Mask.
        segments (np.ndarray): Segments.

    Returns:
        np.ndarray: Standard deviation of each segment.
    """

    n_segments = segments.max()
    n_channels = image.shape[-1]
    segments_std = np.zeros((n_segments, n_channels * 2))
    for j in range(n_channels):
        image_std = image[mask, j].std()
        for i in range(n_segments):
            segment_std = image[segments == i+1, j].std()

            segments_std[i, 2*j] = image_std - segment_std
            segments_std[i, 2*j+1] = segment_std

    return segments_std

def _get_segments_entropy(image: np.ndarray, segments: np.ndarray, n_entropy_bins: int) -> np.ndarray:
    """
    Get entropy of each segment.

    Args:
        image (np.ndarray): Image.
        segments (np.ndarray): Segments.
        n_entropy_bins (int): Number of bins for entropy.

    Returns:
        np.ndarray: Entropy of each segment.
    """

    n_segments = segments.max()
    n_channels = image.shape[-1]
    segments_entropy = np.zeros((n_segments, n_channels))
    for i in range(n_segments):
        for j in range(n_channels):
            probs, _ = np.histogram(image[segments == i+1, j], bins=n_entropy_bins, density=True)
            probs = np.clip(probs, a_min=1e-10, a_max=None)
            log_probs = np.log2(probs)
            entropy = -np.sum(probs * log_probs)
            segments_entropy[i, j] = entropy

    return segments_entropy

def _get_segments_color(image: np.ndarray, segments: np.ndarray, n_color_bins: int, value_ranges: List[Tuple[float, float]]) -> np.ndarray:
    """
    Get color histogram of each segment.

    Args:
        image (np.ndarray): Image.
        segments (np.ndarray): Segments.
        n_color_bins (int): Number of bins for color.

    Returns:
        np.ndarray: Color histogram of each segment.
    """

    n_segments = segments.max()
    n_channels = image.shape[-1]
    segments_color = np.zeros((n_segments, n_channels * n_color_bins))
    for i in range(n_segments):
        for j in range(n_channels):
            value_range = value_ranges[j]
            hist, _ = np.histogram(image[segments == i+1, j], bins=n_color_bins, density=True, range=value_range)
            hist = np.nan_to_num(hist) # Replace nans by 0
            segments_color[i, j*n_color_bins:(j+1)*n_color_bins] = hist

    return segments_color

def _get_segments_texture(gray_image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Get texture features of each segment.
    
    Args:
        gray_image (np.ndarray): Grayscale image.
        segments (np.ndarray): Segments.
        
    Returns:
        np.ndarray: Texture features of each segment.
    """

    n_segments = segments.max()
    segments_texture = np.zeros((n_segments, 6))
    for i in range(n_segments):
        segment_mask = segments == i+1
        segment = gray_image * (segment_mask.astype(np.uint8))

        # Calculating the bounding box of the superpixel to reduce the size
        x, y, w, h = cv2.boundingRect(segment_mask.astype(np.uint8))
        segment = segment[y:y+h, x:x+w]

        # Need at least 2 values to compute GLCM
        if len(np.unique(segment)) >= 2:
            # Calculate GLCM features only for the non-zero values
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(segment, distances, angles, levels=256, symmetric=True, normed=True)
            
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            disimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            energy = np.mean(graycoprops(glcm, 'energy'))
            correlation = np.mean(graycoprops(glcm, 'correlation'))
            asm = np.mean(graycoprops(glcm, 'ASM'))

            texture_features = np.array([contrast, disimilarity, homogeneity, energy, correlation, asm])
            segments_texture[i] = texture_features
        else:
            segments_texture[i] = np.array([0] * 6)

    return segments_texture

def _get_segments_bow(bow_image: np.ndarray, segments: np.ndarray, visual_words: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Get BOW histogram of each segment.

    Args:
        bow_image (np.ndarray): BOW image.
        segments (np.ndarray): Segments.
        visual_words (np.ndarray): Visual words.
        patch_size (int): Size of patches.

    Returns:
        np.ndarray: BOW histogram of each segment.
    """

    n_segments = segments.max()
    n_visual_words = visual_words.shape[0]
    segments_bow = np.zeros((n_segments, n_visual_words))
    for i in range(n_segments):
        segment_bow = bow_image[segments == i+1]
        segment_bow_hist, _ = np.histogram(segment_bow, bins=n_visual_words, density=True, range=(0, n_visual_words))
        segment_bow_hist = np.nan_to_num(segment_bow_hist) # Replace nans by 0
        segments_bow[i] = segment_bow_hist

    return segments_bow

def get_segment_data(
        rgb_image: Image.Image, 
        ground_mask: Image.Image = None,
        sky_mask: Image.Image = None,
        light_cloud_mask: Image.Image = None, 
        thick_cloud_mask: Image.Image = None,
        n_segments: int = 100,
        compactness: float = 10,
        sigma: float = 0,
        n_entropy_bins: int = 10, 
        n_color_bins: int = 10,
        visual_words: np.ndarray = None,
        patch_size: int = None,
        selected_features: List[str] = ['mean', 'std', 'entropy', 'color', 'bow', 'texture'],
        separate_clouds: bool = False
        ):
    """
    Get features from image and segments.

    Args:
        rgb_image: Image.
        ground_mask: Ground mask.
        sky_mask: Sky mask.
        light_cloud_mask: Light cloud mask.
        thick_cloud_mask: Thick cloud mask.
        n_segments: Number of segments for SLIC.
        compactness: Compactness parameter for SLIC.
        sigma: Sigma parameter for SLIC.
        n_entropy_bins: Number of bins for entropy.
        n_color_bins: Number of bins for color.
        visual_words: Visual words.
        patch_size: Size of patches.
        selected_features: List of selected features.
        separate_clouds: Whether to separate cloud classes in mask and ground truth labels.

    Returns:
        np.ndarray: Input features.
        np.ndarray: Ground truth labels.
    """

    if visual_words is not None and patch_size is None:
        raise ValueError("❌ Patch size must be given if visual words are given.")

    # Apply mask to image if given
    if ground_mask is None:
        mask = None
        rgb_image = np.array(rgb_image)
    else:
        mask = ~np.array(ground_mask)
        rgb_image = np.array(rgb_image) * mask[:, :, np.newaxis]

    # Apply superpixel segmentation on RGB image
    segments = _get_superpixel_segments(rgb_image, mask, n_segments, compactness, sigma)
    n_segments = segments.max() # Update number of segments

    # Get image features (new channels) and add segment features
    image, value_ranges = get_image_channels(rgb_image)
    features = np.zeros((n_segments, 0))

    # Get mean
    if 'mean' in selected_features:
        segments_mean = _get_segments_mean(image, mask, segments)
        features = np.concatenate((features, segments_mean), axis=1)

    # Get standard deviation
    if 'std' in selected_features:
        segments_std = _get_segments_std(image, mask, segments)
        features = np.concatenate((features, segments_std), axis=1)

    # Get entropy
    if 'entropy' in selected_features:
        segments_entropy = _get_segments_entropy(image, segments, n_entropy_bins)
        features = np.concatenate((features, segments_entropy), axis=1)

    # Get color histogram
    if 'color' in selected_features:
        segments_color = _get_segments_color(image, segments, n_color_bins, value_ranges)
        features = np.concatenate((features, segments_color), axis=1)

    # Get texture features
    if 'texture' in selected_features:
        gray_image = np.mean(rgb_image, axis=-1).astype(np.uint8)
        segments_texture = _get_segments_texture(gray_image, segments)
        features = np.concatenate((features, segments_texture), axis=1)

    if 'bow' in selected_features and visual_words is not None:
        bow_image = get_visual_word_indices(rgb_image, visual_words, patch_size)
        segments_bow = _get_segments_bow(bow_image, segments, visual_words, patch_size)
        features = np.concatenate((features, segments_bow), axis=1)


    # Ground truth labels
    if ground_mask is None or sky_mask is None or light_cloud_mask is None or thick_cloud_mask is None:
        return segments, features, None
    
    labels = np.zeros(n_segments)
    # Get most common label for each segment
    for i in range(n_segments):
        n_sky = sky_mask[segments == i+1].sum()
        n_light_cloud = light_cloud_mask[segments == i+1].sum()
        n_thick_cloud = thick_cloud_mask[segments == i+1].sum()

        label_classes = [n_sky, n_light_cloud, n_thick_cloud] if separate_clouds else [n_sky, n_light_cloud + n_thick_cloud]
        labels[i] = np.argmax(label_classes)

    return segments, features, labels

def _get_data_paths(inputs_path: str, masks_path: str, labels_path: str):
    """
    Returns a list of input and mask paths for the sky finder dataset.

    Returns:
        input_paths: List of input paths.
        mask_paths: List of mask paths.
        label_paths: List of label paths.
    """

    input_paths = []
    mask_paths = []
    label_paths = []

    # Sky finder segmented
    input_folder_paths = sorted([os.path.join(inputs_path, folder) for folder in os.listdir(inputs_path)])
    for folder_path in input_folder_paths:
        image_paths = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)])
        for image_path in image_paths:
            input_paths.append(image_path)
            folder = folder_path.split("/")[-1]
            mask_paths.append(f"{masks_path}{folder}.png")
            label_paths.append(image_path.replace(inputs_path, labels_path).replace(".jpg", ".png"))

    return input_paths, mask_paths, label_paths

def get_data(
        inputs_path: str,
        masks_path: str,
        labels_path: str,
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

    input_paths, mask_paths, label_paths = _get_data_paths(inputs_path, masks_path, labels_path)

    segmentss = []
    X = np.array([])
    y = np.array([])
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

        # Get features for each segment
        segments, Xi, yi = get_segment_data(
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
        
        segmentss.append(segments)
        X = np.concatenate((X, Xi), axis=0) if X.size else Xi
        y = np.concatenate((y, yi), axis=0) if y.size else yi

    return segmentss, X, y