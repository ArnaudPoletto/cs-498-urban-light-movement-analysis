from typing import List

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

def validate(
        model: nn.Module, 
        val_loader: DataLoader, 
        criterion: nn.Module) -> float:
    """
    Validate model.

    Args:
        model (nn.Module): Model.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Criterion.

    Returns:
        val_loss (float): Validation loss.
    """

    model.eval()

    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            input, mask = data

            output = model(input)
            output = output['out']
            loss = criterion(output, mask)
            val_loss += loss.item()
            
    model.train()

    return val_loss / len(val_loader)

def train(
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler._LRScheduler,
        epochs: int,
        accumulation_steps: int,
        validation_steps: int,
        info_file_path: str,
        model_save_path: str,
        ) -> None:
    """
    Train model.

    Args:
        model (nn.Module): Model.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (nn.Module): Criterion.
        optimizer (optim.Optimizer): Optimizer.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler.
        epochs (int): Number of epochs.
        accumulation_steps (int): Number of steps to accumulate gradients.
        validation_steps (int): Number of steps to validate.
        info_file_path (str): Path to file to write training info to.
        model_save_path (str): Path to file to save model to.
    """

    print(f"🚀 Training model for {epochs} epochs...")
    
    best_val_loss = float("inf")
    train_loss = 0

    # Refresh info file
    if os.path.exists(info_file_path):
        open(info_file_path, 'w').close()

    with open(info_file_path, "a") as f:
        f.write(f"epoch,step,validation_loss,train_loss\n")

    scaler = GradScaler()

    model.train()

    for epoch in range(epochs):
        bar = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            inputs, masks = data

            # Forward pass
            with autocast():
                try:
                    outputs = model(inputs)
                    outputs = outputs['out']
                except Exception as e:
                    print(f"❌ {e}")
                    continue
                loss = criterion(outputs, masks)

            # Backward pass
            scaler.scale(loss).backward()
            train_loss += loss.item()

            # Update weights every ACCUMULATION_STEPS
            if (i+1) % accumulation_steps == 0:
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Decay learning rate
                scheduler.step()

                # Show training loss
                bar.set_description(f"➡️ Training loss: {train_loss / accumulation_steps:.5f}")
                train_loss = 0

            # Validate every VALIDATION_STEPS
            if (i+1) % validation_steps == 0 or i == 0:
                # Validate
                val_loss = validate(model, val_loader, criterion)

                # Print validation info to file
                with open(info_file_path, "a") as f:
                    f.write(f"{epoch},{i+1},{val_loss},{loss.item()}\n")

                # Save model for best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    bar.clear()
                    print(f"✅ Saved model with validation loss {val_loss:.5f}.")
                    
            bar.update(1)

def load_model(model: nn.Module, model_save_path: str, force: bool = False) -> None:
    """
    Load model from path.

    Args:
        model (nn.Module): Model.
        model_save_path (str): Path to model.
        force (bool): Whether to force loading the model.
    """

    if os.path.exists(model_save_path) or force:
        model.load_state_dict(torch.load(model_save_path))
        print(f"✅ Loaded model from {model_save_path}")
    else:
        print("❌ Model not loaded.")

def get_class_weights(dataloader, n_classes=2, n_samples=1000):
    """
    Get class weights for a given dataloader.

    Args:
        dataloader: Dataloader to get class weights for.
        n_samples: Number of samples to use for computing class weights.

    Returns:
        Class weights.
    """

    class_weights = np.zeros(n_classes)
    total_samples = min(n_samples, len(dataloader))

    bar = tqdm(total=total_samples)
    for i, (_, masks) in enumerate(dataloader):
        masks = masks.cpu().squeeze().numpy()[1]
        for j in range(n_classes):
            class_weights[j] += np.sum(masks == j)

        bar.update(1)
        
        if i == total_samples - 1:
            break

    # Avoid division by zero
    class_weights += 1e-10

    class_weights = class_weights / np.sum(class_weights)
    class_weights = 1 / class_weights

    return class_weights

def show_learning_curves(
    model_file_dir: str,
    default_max_steps: int,
    y_lim = (0.001, None),
    color_dict: dict = {},
    window_size: int = 4,
    ):
    """
    Show learning curves for models.

    Args:
        model_file_dir (str): Path to directory containing model files.
        default_max_steps (int): Default maximum number of steps.
        y_lim (tuple): Y-axis limits.
        color_dict (dict): Dictionary mapping model names to colors.
        window_size (int): Window size for smoothing.
    """

    def smooth_moving_avg(data, window_size: int) -> List[float]:
        """
        Smooth data using moving average.

        Args:
            data (list): Data to smooth.
            window_size (int): Window size.

        Returns:
            moving_avgs (list): Smoothed data.
        """
        
        moving_avgs = []
        for i, _ in enumerate(data, 1):
            if i < window_size:
                avg = sum(data[:i]) / i
            else:
                avg = sum(data[i-window_size:i]) / window_size
            moving_avgs.append(avg)

        return moving_avgs


    model_file_paths = sorted([os.path.join(model_file_dir, filename) for filename in os.listdir(model_file_dir) if filename.endswith(".csv")])
    validation_lossess = []
    for model_file_path in model_file_paths:
        with open(model_file_path, "r") as f:
            name = model_file_path.split("/")[-1].split(".")[0]
            name = name.replace('_info', '')

            lines = f.readlines()[1:]
            epochs = [int(line.split(",")[0]) for line in lines]
            steps = [int(line.split(",")[1]) for line in lines]
            max_steps = max(steps)
            steps = [step * default_max_steps / max_steps for step in steps] # Normalize steps
            idxs = [epoch * default_max_steps + step for epoch, step in zip(epochs, steps)]
            validation_losses = [float(line.split(",")[2]) for line in lines]
            smoothed_validation_losses = smooth_moving_avg(validation_losses, window_size)

            validation_lossess.append((name, idxs, smoothed_validation_losses))

    plt.figure(figsize=(10, 6))
    for i, (name, idxs, validation_losses) in enumerate(validation_lossess):
        color = color_dict[name] if name in color_dict else plt.get_cmap('tab20')(i)
        plt.plot(idxs, validation_losses, label=name, color=color)
    plt.yscale("log")
    plt.title("Validation Losses")
    plt.xlabel("Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.xlim(0, None)
    plt.ylim(y_lim)
    plt.legend()
    plt.show()