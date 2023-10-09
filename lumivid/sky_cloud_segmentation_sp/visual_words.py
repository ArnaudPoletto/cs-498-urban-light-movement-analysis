# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from typing import List

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from skimage.util import view_as_windows

import torch

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SCS_DATA_PATH = DATA_PATH + 'sky_cloud_segmentation/'

SKY_FINDER_PATH = SCS_DATA_PATH + "sky_finder/"
SKY_FINDER_INPUTS_PATH = SKY_FINDER_PATH + "images/"
SKY_FINDER_MASKS_PATH = SKY_FINDER_PATH + "masks/"

PATCH_MEAN = [170.50721488, 42.75232128, 0.88888601]
PATCH_STD = [54.00342012, 41.27330411, 0.26348232]

def get_image_channels(image: np.ndarray) -> np.ndarray:
    """
    Get image features from RGB image.
    
    Args:
        image: RGB Image.
        
    Returns:
        np.ndarray: Image features.
    """

    # Get image features
    image_red = image[:, :, 0]
    image_saturation = np.array(Image.fromarray(image).convert('HSV'))[:, :, 1]
    image_blue = image[:, :, 2]
    image_r_over_b = np.divide(image_red, image_blue, out=np.zeros_like(image_red).astype(float), where=image_blue!=0)
    image = np.stack([image_red, image_saturation, image_r_over_b], axis=-1)

    return image

class SCRandomPatchesDataset(torch.utils.data.Dataset):
    """
    Dataset of random patches from sky/cloud images.
    
    Args:
        input_paths: List of paths to input images.
        mask_paths: List of paths to mask images.
        patch_size: Size of patches.
        n_patches: Number of patches to generate.
        
    Returns:
        torch.utils.data.Dataset: Dataset of random patches.
    """

    
    def _get_patches(self, args):
        image, mask, image_width, image_height, patch_size, n_tries, n_patches_per_image = args
        patches = []
        for _ in range(n_patches_per_image):
            patch = None
            tries = 0
            while patch is None and tries < n_tries:
                x = np.random.randint(patch_size//2, image_width + patch_size//2)
                y = np.random.randint(patch_size//2, image_height + patch_size//2)
                if mask[y, x]:
                    image_patch = image[y-patch_size//2:y+patch_size//2+1, x-patch_size//2:x+patch_size//2+1]
                    patch = (image_patch - PATCH_MEAN) / PATCH_STD
                tries += 1
            if patch is not None:
                patches.append(patch)
        return patches

    def get_patches(
            self, 
            input_paths: List[str],
            mask_paths: List[str],
            image_width: int,
            image_height: int,
            patch_size: int, 
            n_patches: int, 
            n_tries: int = 10
            ) -> List[np.ndarray]:

        assert n_patches > 0 and patch_size % 2 == 1 and n_tries > 0

        # Read images and masks
        images = [np.array(Image.open(path).resize((image_width, image_height))) for path in tqdm(input_paths, desc="▶️ Loading images")]
        masks = [(np.array(Image.open(path).convert("L").resize((image_width, image_height))) > 128) for path in tqdm(mask_paths, desc="▶️ Loading masks")]

        for image, mask in zip(images, masks):
            image[~mask] = 0

        # Add padding to images and masks
        images = [np.pad(image, ((patch_size//2, patch_size//2), (patch_size//2, patch_size//2), (0, 0)), mode="reflect") for image in images]
        masks = [np.pad(mask, ((patch_size//2, patch_size//2), (patch_size//2, patch_size//2)), mode="constant", constant_values=False) for mask in masks]

        # Generate patches using multiprocessing
        n_patches_per_image = n_patches // len(images)
        tasks = [(image, mask, image_width, image_height, patch_size, n_tries, n_patches_per_image) for image, mask in zip(images, masks)]

        from multiprocessing import Pool, cpu_count
        with Pool(cpu_count()) as pool:
            patches = []
            for patch_group in tqdm(pool.imap_unordered(self._get_patches, tasks), total=len(tasks), desc="▶️ Generating patches"):
                patches.extend(patch_group)

        patches = [patch for patch in patches if patch is not None]

        return np.array(patches)

    def __init__(
            self, 
            input_paths: List[str], 
            mask_paths: List[str], 
            image_width: int,
            image_height: int,
            patch_size: int, 
            n_patches: int
            ):
        self.patches = self.get_patches(input_paths, mask_paths, image_width, image_height, patch_size, n_patches)

        print(f"✅ Created dataset of {len(self.patches)} patches.")

    def __len__(self):

        return len(self.patches)

    def __getitem__(self, idx):
        
        return self.patches[idx]

def _get_dataset(image_width, image_height, patch_size, n_patches):
    """
    Get dataset of random patches from sky/cloud images.

    Returns:
        SCRandomPatchesDataset: Dataset of random patches.
    """

    input_paths = []
    mask_paths = []

    # Sky Finder
    input_folder_paths = sorted([os.path.join(SKY_FINDER_INPUTS_PATH, folder) for folder in os.listdir(SKY_FINDER_INPUTS_PATH)])
    for folder_path in input_folder_paths:
        image_paths = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)])
        for image_path in image_paths:
            input_paths.append(image_path)
            mask_paths.append(os.path.join(SKY_FINDER_MASKS_PATH, folder_path.split("/")[-1] + ".png"))
    assert len(input_paths) == len(mask_paths)

    dataset = SCRandomPatchesDataset(
        input_paths=input_paths,
        mask_paths=mask_paths,
        patch_size=patch_size,
        image_width=image_width,
        image_height=image_height,
        n_patches=n_patches
    )

    return dataset

def _get_X(dataset: SCRandomPatchesDataset):
    """
    Get X from dataset.

    Args:
        dataset: Dataset.

    Returns:
        torch.Tensor: X.
    """

    X = torch.tensor(np.array(dataset.patches)).reshape(-1, int(np.multiply.reduce(dataset.patches.shape[1:])))
    print(f"✅ Created X of shape {X.shape}.")

    return X

def _apply_kmeans(X: torch.Tensor, n_clusters: int):
    """
    Apply KMeans to dataset.

    Args:
        X: Dataset.
        n_clusters: Number of clusters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cluster centers and cluster labels.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(X)
    print(f"✅ Applied KMeans with {n_clusters} clusters.")

    return kmeans.cluster_centers_, kmeans.labels_

def _show_tsne(X: torch.Tensor, cluster_labels: np.ndarray):
    """
    Show t-SNE of dataset.
    
    Args:
        X: Dataset.
        cluster_labels: Cluster labels.
    """

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)

    # Plot t-SNE with cluster colors
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='jet', s=1)
    plt.colorbar()
    plt.title("Projection of patches using t-SNE")
    plt.show()

def get_visual_word_indices(image: np.ndarray, visual_words: np.ndarray, patch_size: int):
    """
    Get visual word indices from image.

    Args:
        image: Image.
        visual_words: Visual words.
        patch_size: Size of patches.

    Returns:
        np.ndarray: Visual word indices.
    """

    padding = patch_size // 2
    image_padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")

    # Efficient patch extraction using view_as_windows
    patches = view_as_windows(image_padded, (patch_size, patch_size, image.shape[2]))

    # Reshape patches and normalize
    patches = patches.reshape(-1, patch_size, patch_size, image.shape[2])
    patches = (patches - PATCH_MEAN) / PATCH_STD

    # Reshape patches to feature vectors
    patches_features = patches.reshape(patches.shape[0], -1)

    # Assign each patch to the nearest cluster
    cluster_indices, _ = pairwise_distances_argmin_min(patches_features, visual_words)
    cluster_indices = cluster_indices.reshape(image.shape[0], image.shape[1])

    return cluster_indices

def get_visual_words(
        image_width: int,
        image_height: int,
        patch_size: int, 
        n_patches: int, 
        n_visual_words: int,
        show_tsne: bool = False
        ) -> np.ndarray:
    """
    Get visual words from dataset.
    
    Returns:
        np.ndarray: Visual words.
    """

    dataset = _get_dataset(image_width, image_height, patch_size, n_patches)
    X = _get_X(dataset)
    visual_words, cluster_labels = _apply_kmeans(X, n_visual_words)

    if show_tsne:
        _show_tsne(X, cluster_labels)

    return visual_words


