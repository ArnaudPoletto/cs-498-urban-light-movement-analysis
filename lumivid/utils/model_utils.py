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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            # Move to GPU
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)

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

            # Move to GPU
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE)

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

def get_class_weights(dataloader, n_classes=2, n_samples=1000, binary: bool = False):
    """
    Get class weights for a given dataloader.

    Args:
        dataloader: Dataloader to get class weights for.
        n_classes: Number of classes.
        n_samples: Number of samples to use for computing class weights.
        binary: If True, the data should be a stack of binary masks.

    Returns:
        Class weights.
    """

    class_weights = np.zeros(n_classes)
    total_samples = min(n_samples, len(dataloader))

    bar = tqdm(total=total_samples)
    for i, (_, masks) in enumerate(dataloader):
        masks = masks.cpu().squeeze().numpy()

        if binary:
            class_weights += np.sum(masks, axis=(0, 2, 3))  # Sum over batch and spatial dimensions
        else:
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

def split_data(X: np.ndarray, y: np.ndarray, train_split: int, test_split: int = None):
    """
    Split data into training, test, and validation (only if specified) sets.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Labels.
        train_split (int): Percentage of data to use for training.
        test_split (int): Percentage of data to use for testing.

    Returns:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        y_test (np.ndarray): Testing labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
    """
    assert train_split > 0 and train_split < 1, "❌ Train split must be between 0 and 1."
    assert test_split is None or (test_split > 0 and test_split < 1), "❌ Test split must be between 0 and 1."
    assert test_split is None or train_split + test_split <= 1, "❌ Train split + test split must be less than or equal to 1."

    # Get number of samples for each set
    n_samples = X.shape[0]
    n_train_samples = int(n_samples * train_split)
    n_test_samples = n_samples - n_train_samples if test_split is None else int(n_samples * test_split)
    n_val_samples = n_samples - n_train_samples - n_test_samples if test_split is None else 0

    # Shuffle data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    if n_val_samples == 0:
        X_train, y_train = X[:n_train_samples], y[:n_train_samples]
        X_test, y_test = X[n_train_samples:], y[n_train_samples:]
        return X_train, y_train, X_test, y_test, None, None
    else:
        X_train, y_train = X[:n_train_samples], y[:n_train_samples]
        X_test, y_test = X[n_train_samples:n_train_samples+n_test_samples], y[n_train_samples:n_train_samples+n_test_samples]
        X_val, y_val = X[n_train_samples+n_test_samples:], y[n_train_samples+n_test_samples:]
        return X_train, y_train, X_test, y_test, X_val, y_val

def show_learning_curves(
    model_file_dir: str,
    default_max_steps: int,
    y_lim = (0.001, None),
    color_dict: dict = {},
    name_dict: dict = {},
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
            file_name = model_file_path.split("/")[-1].split(".")[0]
            name = name_dict[file_name] if file_name in name_dict else file_name

            lines = f.readlines()[1:]
            epochs = [int(line.split(",")[0]) for line in lines]
            steps = [int(line.split(",")[1]) for line in lines]
            max_steps = max(steps)
            steps = [step * default_max_steps / max_steps for step in steps] # Normalize steps
            idxs = [epoch * default_max_steps + step for epoch, step in zip(epochs, steps)]
            validation_losses = [float(line.split(",")[2]) for line in lines]
            smoothed_validation_losses = smooth_moving_avg(validation_losses, window_size)

            validation_lossess.append((file_name, name, idxs, smoothed_validation_losses))

    plt.figure(figsize=(10, 6))
    for i, (file_name, name, idxs, validation_losses) in enumerate(validation_lossess):
        color = color_dict[file_name] if file_name in color_dict else plt.get_cmap('tab20')(i)
        plt.plot(idxs, validation_losses, label=name, color=color)
    plt.yscale("log")
    plt.title("Validation Losses")
    plt.xlabel("Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.xlim(0, None)
    plt.ylim(y_lim)
    plt.legend()
    plt.show()