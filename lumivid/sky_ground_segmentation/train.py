# Ensure that the current working directory is this file
import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / '..' / '..'
sys.path.append(str(GLOBAL_DIR))

from ranger_adabelief import RangerAdaBelief # https://github.com/juntang-zhuang/Adabelief-Optimizer

from lumivid.utils.model_utils import train
from lumivid.utils.random_utils import set_seed
from lumivid.sky_ground_segmentation.dataset import get_dataloaders

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models.segmentation as models

TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT = 0.9, 0.095, 0.005
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

CLASS_WEIGHTS = torch.tensor([[1.68795506, 2.45358333]])
CLASS_WEIGHTS = CLASS_WEIGHTS.view(1, 2, 1, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 27

def get_model(model_type: str, n_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Get model.

    Args:
        model_type (str): Model type.
        n_classes (int, optional): Number of classes. Defaults to 2.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.

    Returns:
        model (nn.Module): Model.
    """
    
    assert model_type in ['deeplabv3resnet101', 'deeplabv3resnet50', 'deeplabv3mobilenetv3large'], f"❌ Invalid model type: {model_type}."

    # Get model type
    if model_type == 'deeplabv3resnet101':
        model = models.deeplabv3_resnet101
    elif model_type == 'deeplabv3resnet50':
        model = models.deeplabv3_resnet50
    elif model_type == 'deeplabv3mobilenetv3large':
        model = models.deeplabv3_mobilenet_v3_large

    model = model(
        weights='COCO_WITH_VOC_LABELS_V1' if pretrained else None,
        weights_backbone='IMAGENET1K_V2' if pretrained else None,
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Loaded {'pretrained ' if pretrained else ''}model {model_type} with {n_params:,} learnable parameters.")

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

def ask_model():
    """
    Ask user for model type.
    
    Returns:
        model_type (str): Model type.
    """

    int_to_model = {
        '0': 'deeplabv3resnet101',
        '1': 'deeplabv3resnet50',
        '2': 'deeplabv3mobilenetv3large'
    }

    model_type = input("❓ Which model would you like to use?\n" + ''.join([f'\t{i}: {model}\n' for i, model in int_to_model.items()]))
    assert model_type in ['0', '1', '2'], "❌ Invalid model type."

    return int_to_model[model_type]

def ask_pretrained():
    """
    Ask user for pretrained.
    
    Returns:
        pretrained (bool): Whether to use pretrained weights.
    """

    pretrained = input("❓ Would you like to use pretrained weights? (y/n)\n")
    assert pretrained in ['y', 'n'], "❌ Invalid pretrained."

    return pretrained == 'y'


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

if __name__ == '__main__':
    # Set seed for deterministic results
    set_seed(SEED)

    model_type = ask_model()
    pretrained = ask_pretrained()
    optimizer_type = ask_optimizer()

    train_loader, _, val_loader = get_dataloaders(TRAIN_SPLIT, TEST_SPLIT, BATCH_SIZE)

    model = get_model(model_type, pretrained=pretrained)

    criterion = get_criterion()
    optimizer = get_optimizer(optimizer_type, model)
    scheduler = get_scheduler(optimizer)

    info_file_path = str(GLOBAL_DIR / 'data' / 'sky_ground_segmentation' / 'models' / f'{model_type}_{optimizer_type}_info.csv')
    model_save_path = str(GLOBAL_DIR / 'data' / 'sky_ground_segmentation' / 'models' / f'{model_type}_{optimizer_type}.pth')
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

