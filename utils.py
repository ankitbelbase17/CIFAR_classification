"""
utils.py - Utility functions for training, checkpointing, and visualization
"""

import torch
import torch.nn as nn
import os
import json
import yaml
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_wandb(config: Dict[str, Any], project_name: str = "cv_classification"):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=project_name,
        config=config,
        name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=[config['model_name'], config.get('dataset', 'custom')]
    )
    print(f"✓ WandB initialized: {wandb.run.name}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    iteration: int,
    best_val_acc: float,
    save_dir: str,
    model_name: str,
    is_best: bool = False
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        iteration: Current iteration
        best_val_acc: Best validation accuracy so far
        save_dir: Directory to save checkpoint
        model_name: Name of the model
        is_best: Whether this is the best model so far
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'model_name': model_name
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(save_dir, f'{model_name}_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save checkpoint at this iteration
    iter_path = os.path.join(save_dir, f'{model_name}_iter_{iteration}.pth')
    torch.save(checkpoint, iter_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, f'{model_name}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved with val_acc: {best_val_acc:.4f}")
    
    print(f"✓ Checkpoint saved at iteration {iteration}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load model to
    
    Returns:
        Dictionary with checkpoint info
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return {'epoch': 0, 'iteration': 0, 'best_val_acc': 0.0}
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']}")
    print(f"  Best val acc: {checkpoint.get('best_val_acc', 0.0):.4f}")
    
    return checkpoint


def log_to_wandb(metrics: Dict[str, float], step: int, images: Optional[Dict] = None):
    """
    Log metrics and images to WandB
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        images: Optional dictionary of images to log
    """
    log_dict = {**metrics, 'step': step}
    
    if images:
        wandb_images = {}
        for key, img_data in images.items():
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.cpu().numpy()
            wandb_images[key] = wandb.Image(img_data)
        log_dict.update(wandb_images)
    
    wandb.log(log_dict, step=step)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str
):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {save_path}")


def visualize_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: list,
    num_images: int = 8,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize model predictions
    
    Args:
        images: Batch of images (B, C, H, W)
        true_labels: True labels (B,)
        pred_labels: Predicted labels (B,)
        class_names: List of class names
        num_images: Number of images to visualize
        save_path: Optional path to save visualization
    
    Returns:
        Visualization as numpy array
    """
    num_images = min(num_images, len(images))
    
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx in range(num_images):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        true_label = class_names[true_labels[idx].item()]
        pred_label = class_names[pred_labels[idx].item()]
        
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(
            f'True: {true_label}\nPred: {pred_label}',
            color=color,
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to array for WandB
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return img_array


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed}")


def get_device() -> torch.device:
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device


def create_experiment_dir(base_dir: str, model_name: str) -> str:
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'{model_name}_{timestamp}')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    print(f"✓ Experiment directory: {exp_dir}")
    
    return exp_dir


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test device detection
    device = get_device()
    
    # Test seed setting
    set_seed(42)
    
    # Test experiment directory creation
    exp_dir = create_experiment_dir('experiments', 'test_model')
    
    print("\n✓ All utility tests passed")