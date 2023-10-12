# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from typing import Tuple, List

import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader

from albumentations import (
    CLAHE, ChannelShuffle, ColorJitter, Compose, ElasticTransform,
    GaussianBlur, GridDistortion, HorizontalFlip, HueSaturationValue,
    Normalize, OneOf, OpticalDistortion, RandomBrightness,
    Flip, RandomSunFlare,
    RandomGamma, RGBShift, Resize, ShiftScaleRotate, ToFloat,
)
from albumentations.pytorch import ToTensorV2

from lumivid.utils.data_utils import UnseededDataLoader, SeededDataLoader

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SCS_DATA_PATH = DATA_PATH + "sky_cloud_segmentation/"

SKY_FINDER_PATH = SCS_DATA_PATH + "sky_finder/"
SKY_FINDER_INPUTS_PATH = SKY_FINDER_PATH + "images/"
SKY_FINDER_MASKS_PATH = SKY_FINDER_PATH + "masks/"

SKY_FINDER_SEG_TR_PATH = SCS_DATA_PATH + "sky_finder_segmented/train/"
SKY_FINDER_SEG_TR_INPUTS_PATH = SKY_FINDER_SEG_TR_PATH + "filtered_images/"
SKY_FINDER_SEG_TR_MASKS_PATH = SKY_FINDER_SEG_TR_PATH + "masks/"
SKY_FINDER_SEG_TR_LABELS_PATH = SKY_FINDER_SEG_TR_PATH + "labels/"

SKY_FINDER_SEG_TE_PATH = SCS_DATA_PATH + "sky_finder_segmented/test/"
SKY_FINDER_SEG_TE_INPUTS_PATH = SKY_FINDER_SEG_TE_PATH + "filtered_images/"
SKY_FINDER_SEG_TE_MASKS_PATH = SKY_FINDER_SEG_TE_PATH + "masks/"
SKY_FINDER_SEG_TE_LABELS_PATH = SKY_FINDER_SEG_TE_PATH + "labels/"

N_DUPLICATES = 1000

N_TRAIN_WORKERS, N_VAL_WORKERS = 4, 0

IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42

class SCSegmentationDataset(torch.utils.data.Dataset):
    """
    Sky Cloud Segmentation Dataset.

    Args:
        input_paths (list): List of input images paths.
        mask_paths (list): List of mask images paths (same order as input_paths).

    Returns:
        torch.utils.data.Dataset: Dataset object.
    """

    def __init__(
            self, 
            input_paths: List[str], 
            mask_paths: List[str], 
            label_paths: List[str], 
            n_duplicates: int = 1,
            separate_clouds: bool = False,
            deterministic: bool = False):
        self.input_paths = input_paths * n_duplicates
        self.mask_paths = mask_paths * n_duplicates
        self.label_paths = label_paths * n_duplicates
        self.image_height = IMAGE_HEIGHT
        self.image_width = IMAGE_WIDTH
        self.separate_clouds = separate_clouds
        self.deterministic = deterministic

        # Torch transforms
        self.normalize_torch_transforms = Compose([
            ToFloat(max_value=1.0),  # Don't need to scale to [0, 1] since normalize does that
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()  # Convert to PyTorch tensor
        ], p=1.0)

        self.unnormalize_torch_transforms = Compose([
            ToFloat(max_value=1.0),  # Ensure image is in [0, 1] range
            ToTensorV2()  # Convert to PyTorch tensor
        ], p=1.0)
        
        # Common transforms
        deterministic_common_transforms = Compose([
            Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        ])

        undeterministic_common_transforms = Compose([
            Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            HorizontalFlip(p=0.5),
            Flip(p=0.5),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                ElasticTransform(p=0.3),
            ], p=0.8),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, value=(0, 0, 0), border_mode=0, p=0.8),
        ])

        self.common_transforms = deterministic_common_transforms if self.deterministic else undeterministic_common_transforms
        
        # Image transforms
        deterministic_image_transforms = Compose([])

        undeterministic_image_transforms = Compose([
            OneOf([
                RandomBrightness(limit=(0.0, 0.3), p=1.0),
                RandomGamma(gamma_limit=(25, 100), p=1.0),
            ], p=1.0),
            OneOf([
                GaussianBlur(blur_limit=(3, 7), p=1.0),
                CLAHE(p=1.0),
            ], p=0.5),
        ])

        self.image_transforms = deterministic_image_transforms if self.deterministic else undeterministic_image_transforms

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            int: Dataset length.
        """

        return len(self.input_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            input_image (torch.Tensor): Input image, with values in [0,1].
            mask_image (torch.Tensor): Mask image.
        """
        input_path = self.input_paths[idx]
        mask_path = self.mask_paths[idx]
        label_path = self.label_paths[idx]

        # Get image and mask
        image = np.array(Image.open(input_path).resize((self.image_width, self.image_height)))
        mask = np.array(Image.open(mask_path).resize((self.image_width, self.image_height)))
        label = np.array(Image.open(label_path).resize((self.image_width, self.image_height)))

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

        # Get layered label
        if self.separate_clouds:
            label = np.concatenate((
                ground_mask[:, :, np.newaxis].astype(int), 
                sky_mask[:, :, np.newaxis].astype(int), 
                light_cloud_mask[:, :, np.newaxis].astype(int), 
                thick_cloud_mask[:, :, np.newaxis].astype(int)
            ), axis=2)
        else:
            label = np.concatenate((
                ground_mask[:, :, np.newaxis].astype(int), 
                sky_mask[:, :, np.newaxis].astype(int), 
                (light_cloud_mask | thick_cloud_mask)[:, :, np.newaxis].astype(int)
            ), axis=2)

        # Apply common transforms
        data = self.common_transforms(image=image, mask=label)
        image = data['image']
        label = data['mask']

        # Empty pixels are labeled as ground
        empty_label = (label[:, :, 0] + label[:, :, 1] + label[:, :, 2] == 0)[:, :, np.newaxis]
        label = np.where(empty_label, [1, 0, 0, 0] if self.separate_clouds else [1, 0, 0], label)

        # Apply image transforms
        image = self.image_transforms(image=image)['image']

        # Apply torch transforms
        image = self.normalize_torch_transforms(image=image)['image']
        label = self.unnormalize_torch_transforms(image=label)['image']

        return image, label

    
def _get_data_paths(inputs_path: str, masks_path: str, labels_path: str) -> Tuple[List[str], List[str], List[str]]:
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

def get_dataloaders(batch_size: int, separate_clouds: bool = False, use_workers: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and val data loaders.

    Args:
        train_split (float): Train split.
        test_split (float): Test split.
        batch_size (int): Batch size.
        use_workers (bool, optional): Whether to use workers. Defaults to True.

    Returns:
        train_loader (torch.utils.data.DataLoader): Train data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
    """

    # Get data paths
    input_paths_tr, mask_paths_tr, label_paths_tr = _get_data_paths(SKY_FINDER_SEG_TR_INPUTS_PATH, SKY_FINDER_SEG_TR_MASKS_PATH, SKY_FINDER_SEG_TR_LABELS_PATH)
    input_paths_val, mask_paths_val, label_paths_val = _get_data_paths(SKY_FINDER_SEG_TE_INPUTS_PATH, SKY_FINDER_SEG_TE_MASKS_PATH, SKY_FINDER_SEG_TE_LABELS_PATH)

    # Create datasets
    train_dataset = SCSegmentationDataset(input_paths_tr, mask_paths_tr, label_paths_tr, n_duplicates=N_DUPLICATES, separate_clouds=separate_clouds, deterministic=False)
    val_dataset = SCSegmentationDataset(input_paths_val, mask_paths_val, label_paths_val, separate_clouds=separate_clouds, deterministic=True)

    # Create data loaders
    train_loader = UnseededDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=N_TRAIN_WORKERS if use_workers else 0)
    val_loader = SeededDataLoader(SEED, val_dataset, batch_size=batch_size, shuffle=False, num_workers=N_VAL_WORKERS if use_workers else 0)

    print("")
    print("➡️ Number of train images:", len(input_paths_tr))
    print("➡️ Number of val images:", len(input_paths_val))

    return train_loader, val_loader