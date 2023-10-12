# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

import os
import numpy as np
from tqdm import tqdm
from ranger_adabelief import RangerAdaBelief # https://github.com/juntang-zhuang/Adabelief-Optimizer

from lumivid.utils.model_utils import train
from lumivid.utils.random_utils import set_seed
from lumivid.sky_cloud_segmentation_dl.dataset import get_dataloaders

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import resnet101
import torchvision.models.segmentation as models
from torchvision.models._utils import IntermediateLayerGetter

DATA_PATH = str(GLOBAL_DIR / 'data') + '/'
SCS_DATA_PATH = DATA_PATH + "sky_cloud_segmentation/"

BYOL_MODEL_PATH = SCS_DATA_PATH + "models/byol.pth"

EPOCHS = 2
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
VALIDATION_STEPS = 200
LEARNING_RATE = 0.001
EPSILON = 1e-12
BETAS = (0.95, 0.999)
WEIGHT_DECAY = 0.01
STEP_SIZE = 50
GAMMA = 0.97

CLASS_WEIGHTS = torch.tensor([[2.14396795, 4.16733798, 3.40583472]])
CLASS_WEIGHTS = CLASS_WEIGHTS.view(1, 3, 1, 1)

SEPARATE_CLOUDS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 27

def get_model(n_classes: int = 3, byol_pretrained: bool = False) -> nn.Module:
    """
    Get model.

    Args:
        n_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
        model (nn.Module): Model.
    """

    model = models.deeplabv3_resnet101(
        weights='COCO_WITH_VOC_LABELS_V1',
        weights_backbone='IMAGENET1K_V2',
        progress=True, 
        num_classes=21, 
        aux_loss=True
    ).to(DEVICE)

    # Change the number of classes in the classifier to 2
    in_features = model.classifier[4].in_channels
    new_classifier = nn.Sequential(
        nn.Conv2d(in_features, n_classes, kernel_size=1, stride=1)
    ).to(DEVICE)
    model.classifier[4] = new_classifier

    """
    def modify_backbone(model):
            model.layer3[0].conv2.stride = (2, 2)
            model.layer3[0].downsample[0].stride = (2, 2)
            for i in range(1, 23):
                model.layer3[i].conv2.padding = (1, 1)
                model.layer3[i].conv2.dilation = (1, 1)

            model.layer4[0].conv2.stride = (2, 2)
            model.layer4[0].conv2.padding = (1, 1)
            model.layer4[0].conv2.dilation = (1, 1)
            model.layer4[0].downsample[0].stride = (2, 2)
            for i in range(1, 3):
                model.layer4[i].conv2.padding = (1, 1)
                model.layer4[i].conv2.dilation = (1, 1)
    modify_backbone(model.backbone)
    """

    if byol_pretrained:
        # Load weights
        assert os.path.exists(BYOL_MODEL_PATH), "❌ BYOL model path does not exist."
        state_dict = torch.load(BYOL_MODEL_PATH)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.backbone.state_dict()}
        model.backbone.load_state_dict(filtered_state_dict, strict=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Loaded {'BYOL ' if byol_pretrained else ''}pretrained model deeplabv3resnet101 with {n_params:,} learnable parameters.")

    return model

def get_criterion(class_weights: torch.Tensor = CLASS_WEIGHTS) -> nn.Module:
    """
    Get criterion.

    Args:
        class_weights (torch.Tensor, optional): Class weights. Defaults to CLASS_WEIGHTS.
    
    Returns:
        criterion (nn.Module): Criterion.
    """

    criterion = nn.BCEWithLogitsLoss(weight=class_weights.to(DEVICE))

    print(f"✅ Loaded weighted binary cross entropy loss.")

    return criterion

def get_optimizer(
        optimizer_type: str,
        model: nn.Module,
        learning_rate: float = LEARNING_RATE,
        epsilon: float = EPSILON,
        betas: tuple = BETAS,
        weight_decay: float = WEIGHT_DECAY
        ) -> optim.Optimizer:
    """
    Get optimizer.

    Args:
        optimizer_type (str): Optimizer type.
        model (nn.Module): Model.
        learning_rate (float, optional): Learning rate. Defaults to LEARNING_RATE.
        epsilon (float, optional): Epsilon. Defaults to EPSILON.
        betas (tuple, optional): Betas. Defaults to BETAS.
        weight_decay (float, optional): Weight decay. Defaults to WEIGHT_DECAY.

    Returns:
        optimizer (optim.Optimizer): Optimizer.
    """

    assert optimizer_type in ['adam', 'adamw', 'ranger'], "❌ Invalid optimizer type."

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon, betas=betas, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon, betas=betas, weight_decay=weight_decay)
    elif optimizer_type == 'ranger':
        optimizer = RangerAdaBelief(model.parameters(), lr=learning_rate, eps=epsilon, betas=betas, weight_decay=weight_decay)

    print(f"✅ Loaded {optimizer_type} optimizer with learning rate {learning_rate} and weight decay {weight_decay}.")

    return optimizer

def get_scheduler(optimizer: optim.Optimizer, step_size: int = STEP_SIZE, gamma: float = GAMMA) -> optim.lr_scheduler._LRScheduler:
    """
    Get scheduler.

    Args:
        optimizer (optim.Optimizer): Optimizer.

    Returns:
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler.
    """
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print(f"✅ Loaded scheduler with step size {step_size} and gamma {gamma}.")

    return scheduler

def ask_optimizer():
    """
    Ask user for optimizer type.
    
    Returns:
        optimizer_type (str): Optimizer type.
    """

    int_to_optimizer = {
        '0': 'adam',
        '1': 'adamw',
        '2': 'ranger'
    }

    optimizer_type = input("❓ Which optimizer would you like to use?\n" + ''.join([f'\t{i}: {optimizer}\n' for i, optimizer in int_to_optimizer.items()]))

    return int_to_optimizer[optimizer_type]

def ask_byol():
    """
    Ask user to perform byol self-supervised learning instead of default learning.
    
    Returns:
        pretrained (bool): Whether to perform byol self-supervised learning.
    """

    pretrained = input("❓ Would you like to use byol pretrained weights? (y/n)\n")
    assert pretrained in ['y', 'n'], "❌ Invalid pretrained."

    return pretrained == 'y'

if __name__ == '__main__':
    # Set seed for deterministic results
    set_seed(SEED)

    optimizer_type = ask_optimizer()
    byol_pretrained = ask_byol()

    train_loader, val_loader = get_dataloaders(BATCH_SIZE, separate_clouds=SEPARATE_CLOUDS)

    model = get_model(byol_pretrained=byol_pretrained)
    
    optimizer = get_optimizer(optimizer_type, model)
    criterion = get_criterion()
    scheduler = get_scheduler(optimizer)

    info_file_path = str(GLOBAL_DIR / 'data' / 'sky_cloud_segmentation' / 'models' / f"deeplabv3resnet101_{optimizer_type}{'_byol' if byol_pretrained else ''}_info.csv")
    model_save_path = str(GLOBAL_DIR / 'data' / 'sky_cloud_segmentation' / 'models' / f"deeplabv3resnet101_{optimizer_type}_{'byol' if byol_pretrained else ''}.pth")
    train(
        model = model, 
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler,
        epochs = EPOCHS,
        accumulation_steps = ACCUMULATION_STEPS,
        validation_steps = VALIDATION_STEPS,
        info_file_path = info_file_path,
        model_save_path = model_save_path,
        )