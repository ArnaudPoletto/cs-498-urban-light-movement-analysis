# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from typing import Tuple, List

import os
import random
import numpy as np
from PIL import Image

from albumentations import (
    CLAHE, ChannelShuffle, ColorJitter, Compose, ElasticTransform,
    GaussianBlur, GridDistortion, HorizontalFlip, HueSaturationValue,
    Normalize, OneOf, OpticalDistortion, RandomBrightnessContrast,
    RandomGamma, RGBShift, Resize, ShiftScaleRotate, ToFloat,
)
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader

from lumivid.utils.data_utils import UnseededDataLoader, SeededDataLoader

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SGS_DATA_PATH = DATA_PATH + "sky_ground_segmentation/"

ADE20K_PATH = SGS_DATA_PATH + "ade20k_outdoors/"
ADE20K_INPUTS_PATH = ADE20K_PATH + "images/"
ADE20K_MASKS_PATH = ADE20K_PATH + "masks/"

CITYSCAPES_PATH = SGS_DATA_PATH + "cityscapes/"
CITYSCAPES_INPUTS_PATH = CITYSCAPES_PATH + "images/"
CITYSCAPES_MASKS_PATH = CITYSCAPES_PATH + "masks/"

MAPILLARY_PATH = SGS_DATA_PATH + "mapillary_vistas/"
MAPILLARY_INPUTS_PATH = MAPILLARY_PATH + "images/"
MAPILLARY_MASKS_PATH = MAPILLARY_PATH + "masks/"

SKY_FINDER_PATH = SGS_DATA_PATH + "sky_finder/"
SKY_FINDER_INPUTS_PATH = SKY_FINDER_PATH + "images/"
SKY_FINDER_MASKS_PATH = SKY_FINDER_PATH + "masks/"
SKY_FINDER_STACK = 4

SUN_PATH = SGS_DATA_PATH + "sun2012/"
SUN_INPUTS_PATH = SUN_PATH + "images/"
SUN_MASKS_PATH = SUN_PATH + "masks/"

SWIMSEG_PATH = SGS_DATA_PATH + "swimseg/"
SWIMSEG_INPUTS_PATH = SWIMSEG_PATH + "images/"
SWIMSEG_MASKS_PATH = SWIMSEG_PATH + "masks/"
SWIMSEG_STACK = 10

IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_TRAIN_WORKERS, N_TEST_WORKERS, N_VAL_WORKERS = 8, 8, 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

class SGSegmentationDataset(torch.utils.data.Dataset):
    """
    Sky Ground Segmentation Dataset.

    Args:
        input_paths (list): List of input images paths.
        mask_paths (list): List of mask images paths (same order as input_paths).

    Returns:
        torch.utils.data.Dataset: Dataset object.
    """
    
    def __init__(self, input_paths: List[str], mask_paths: List[str], deterministic: bool = False):
        self.input_paths = input_paths
        self.mask_paths = mask_paths
        self.deterministic = deterministic

        # Torch transforms
        self.normalize_torch_transforms = Compose([
            ToFloat(max_value=1.0),  # Don't need to scale to [0, 1] since normalize does that
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()  # Convert to PyTorch tensor
        ], p=1.0)

        self.unnormalize_torch_transforms = Compose([
            ToFloat(max_value=255.0),  # Ensure image is in [0, 1] range
            ToTensorV2()  # Convert to PyTorch tensor
        ], p=1.0)
        
        # Common transforms
        deterministic_common_transforms = Compose([
            Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        ])

        undeterministic_common_transforms = Compose([
            Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            HorizontalFlip(p=0.5),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                ElasticTransform(p=0.3),
            ], p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, value=(0, 0, 0), border_mode=0, p=0.5),
        ])

        self.common_transforms = deterministic_common_transforms if self.deterministic else undeterministic_common_transforms
        
        # Image transforms
        deterministic_image_transforms = Compose([])

        undeterministic_image_transforms = Compose([
            OneOf([
                RandomBrightnessContrast(p=1.0),
                HueSaturationValue(p=1.0),
                RandomGamma(p=1.0),
                ColorJitter(p=1.0),
            ], p=0.3),
            OneOf([
                GaussianBlur(blur_limit=(3, 7), p=1.0),
                CLAHE(p=1.0),
            ], p=0.3),
            OneOf([
                ChannelShuffle(p=1.0),
                RGBShift(p=1.0),
            ], p=0.3),
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

        # Read image and mask
        input_image = np.array(Image.open(self.input_paths[idx]))
        mask_image = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        # Repeat input image channel 3 times if it only has 1 channel
        if len(input_image.shape) == 2:
            input_image = np.repeat(input_image[:, :, np.newaxis], 3, axis=2)
        elif input_image.shape[2] != 3:
            input_image = input_image.repeat(3, axis=2)

        # Apply transforms
        augmented = self.common_transforms(image=input_image, mask=mask_image)
        input_image, mask_image = augmented['image'], augmented['mask']
        input_image = self.image_transforms(image=input_image)['image']
        input_image = self.normalize_torch_transforms(image=input_image)['image']
        mask_image = self.unnormalize_torch_transforms(image=mask_image)['image']

        # Make the mask binary and 2 dimensional
        mask_image[mask_image > 0.5] = 1.
        mask_image[mask_image <= 0.5] = 0.
        mask_image = torch.cat((1. - mask_image, mask_image), dim=0)

        return input_image, mask_image
    
def get_dataloaders(train_split: float, test_split: float, batch_size: int, use_workers: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, test and validation data loaders.

    Args:
        train_split (float): Train split.
        test_split (float): Test split.
        batch_size (int): Batch size.
        use_workers (bool, optional): Whether to use workers. Defaults to True.

    Returns:
        train_loader (torch.utils.data.DataLoader): Train data loader.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
    """

    input_paths = []
    mask_paths = []
    prev_data_length = 0

    # Ade20k
    input_paths += sorted([os.path.join(ADE20K_INPUTS_PATH, filename) for filename in os.listdir(ADE20K_INPUTS_PATH)])
    mask_paths += sorted([os.path.join(ADE20K_MASKS_PATH, filename) for filename in os.listdir(ADE20K_MASKS_PATH)])
    assert len(input_paths) == len(mask_paths)
    print("➡️ Number of Ade20k images:", len(input_paths))
    prev_data_length += len(input_paths)

    # Cityscapes
    input_paths += sorted([os.path.join(CITYSCAPES_INPUTS_PATH, filename) for filename in os.listdir(CITYSCAPES_INPUTS_PATH)])
    mask_paths += sorted([os.path.join(CITYSCAPES_MASKS_PATH, filename) for filename in os.listdir(CITYSCAPES_MASKS_PATH)])
    assert len(input_paths) == len(mask_paths)
    print("➡️ Number of Cityscapes images:", len(input_paths) - prev_data_length)
    prev_data_length += len(input_paths) - prev_data_length

    # Mapillary Vistas
    input_paths += sorted([os.path.join(MAPILLARY_INPUTS_PATH, filename) for filename in os.listdir(MAPILLARY_INPUTS_PATH)])
    mask_paths += sorted([os.path.join(MAPILLARY_MASKS_PATH, filename) for filename in os.listdir(MAPILLARY_MASKS_PATH)])
    assert len(input_paths) == len(mask_paths)
    print("➡️ Number of Mapillary Vistas images:", len(input_paths) - prev_data_length)
    prev_data_length += len(input_paths) - prev_data_length

    # Sky Finder
    input_folder_paths = sorted([os.path.join(SKY_FINDER_INPUTS_PATH, folder) for folder in os.listdir(SKY_FINDER_INPUTS_PATH)] * SKY_FINDER_STACK)
    for folder_path in input_folder_paths:
        image_paths = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)])
        for image_path in image_paths:
            input_paths.append(image_path)
            mask_paths.append(os.path.join(SKY_FINDER_MASKS_PATH, folder_path.split("/")[-1] + ".png"))
    assert len(input_paths) == len(mask_paths)
    print("➡️ Number of SkyFinder images:", len(input_paths) - prev_data_length)
    prev_data_length += len(input_paths) - prev_data_length

    # Sun2019
    input_paths += sorted([os.path.join(SUN_INPUTS_PATH, filename) for filename in os.listdir(SUN_INPUTS_PATH)])
    mask_paths += sorted([os.path.join(SUN_MASKS_PATH, filename) for filename in os.listdir(SUN_MASKS_PATH)])
    assert len(input_paths) == len(mask_paths)
    print("➡️ Number of Sun2019 images:", len(input_paths) - prev_data_length)
    prev_data_length += len(input_paths) - prev_data_length

    # Swimseg
    swimseg_input_paths = sorted([os.path.join(SWIMSEG_INPUTS_PATH, filename) for filename in os.listdir(SWIMSEG_INPUTS_PATH)] * SWIMSEG_STACK)
    input_paths += swimseg_input_paths
    mask_paths += [os.path.join(SWIMSEG_MASKS_PATH, "mask.png")] * len(swimseg_input_paths)
    assert len(input_paths) == len(mask_paths)
    print("➡️ Number of Swimseg images:", len(input_paths) - prev_data_length)

    input_paths = np.array(input_paths)
    mask_paths = np.array(mask_paths)

    # Train test val split
    train_n = int(len(input_paths) * train_split)
    test_n = int(len(input_paths) * test_split)

    train_indices = np.random.choice(len(input_paths), train_n, replace=False)
    test_val_indices = np.setdiff1d(np.arange(len(input_paths)), train_indices)
    test_indices = np.random.choice(test_val_indices, test_n, replace=False)
    val_indices = np.setdiff1d(test_val_indices, test_indices)

    train_input_paths = input_paths[train_indices]
    train_mask_paths = mask_paths[train_indices]
    test_input_paths = input_paths[test_indices]
    test_mask_paths = mask_paths[test_indices]
    val_input_paths = input_paths[val_indices]
    val_mask_paths = mask_paths[val_indices]

    # Data loaders
    train_dataset = SGSegmentationDataset(train_input_paths, train_mask_paths, deterministic=False)
    train_loader = UnseededDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=N_TRAIN_WORKERS if use_workers else 0)

    test_dataset = SGSegmentationDataset(test_input_paths, test_mask_paths, deterministic=True)
    test_loader = SeededDataLoader(SEED, test_dataset, batch_size=batch_size, shuffle=False, num_workers=N_TEST_WORKERS if use_workers else 0)

    val_dataset = SGSegmentationDataset(val_input_paths, val_mask_paths, deterministic=True)
    val_loader = SeededDataLoader(SEED, val_dataset, batch_size=batch_size, shuffle=False, num_workers=N_VAL_WORKERS if use_workers else 0)

    print("")
    print("➡️ Number of train images:", len(train_input_paths))
    print("➡️ Number of test images:", len(test_input_paths))
    print("➡️ Number of val images:", len(val_input_paths))

    return train_loader, test_loader, val_loader