"""
train.py - Training script for image classification models
Supports CIFAR-10 dataset with DINOv2, MAE, SwinV2, ViT, CLIP, VLA, Custom models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
import os

from model import get_model, xavier_init
from dataloader import get_dataloaders
from utils import (
    load_config, save_checkpoint, load_checkpoint,
    setup_wandb, log_to_wandb, plot_training_curves,
    visualize_predictions, count_parameters,
    set_seed, get_device, create_experiment_dir
)
from metrics import calculate_metrics


def validate_batch(
    model: nn.Module,
    test_loader,
    criterion,
    device
) -> tuple:
    """Validate on a single batch"""
    model.eval()
    
    # Get one batch
    test_iter = iter(test_loader)
    try:
        batch = next(test_iter)
        if len(batch) == 3:
            images, labels, metadata = batch
        else:
            images, labels = batch
    except StopIteration:
        return 0.0, 0.0
    
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        with autocast(dtype=torch.float16):
            logits, aux_outputs = model(images)
            loss = criterion(logits, labels)
        
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        acc = 100. * correct / total
    
    model.train()
    return loss.item(), acc


def train_epoch(
    model: nn.Module,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int,
    global_iter: int,
    config: dict,
    exp_dir: str
) -> tuple:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:
            images, labels, metadata = batch
        else:
            images, labels = batch
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training with FP16
        with autocast(dtype=torch.float16):
            logits, aux_outputs = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
        
        global_iter += 1
        
        # Validate on a single batch using test_loader
        val_loss, val_acc = validate_batch(
            model, test_loader, criterion, device)
        
        # Log to WandB every iteration
        if global_iter % config.get('log_interval', 10) == 0:
            metrics = {
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'train/loss': loss.item(),
                'train/accuracy': 100. * correct / total,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }
            log_to_wandb(metrics, global_iter)
        
        # Visualize predictions and log to WandB
        if global_iter % config.get('viz_interval', 100) == 0:
            with torch.no_grad():
                viz_img = visualize_predictions(
                    images[:8], labels[:8], predicted[:8],
                    train_loader.dataset.classes,
                    num_images=8
                )
                log_to_wandb({}, global_iter, {'train/predictions': viz_img})
        
        # Save checkpoint every 250 iterations
        if global_iter % 250 == 0:
            save_checkpoint(
                model, optimizer, None, epoch, global_iter,
                0.0,  # Will be updated with validation
                os.path.join(exp_dir, 'checkpoints'),
                config['model_name']
            )
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, global_iter


def validate(
    model: nn.Module,
    test_loader,
    criterion,
    device,
    class_names: list
) -> tuple:
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Validation'):
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                
            images, labels = images.to(device), labels.to(device)
            
            with autocast(dtype=torch.float16):
                logits, aux_outputs = model(images)
                loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
    
    val_loss = running_loss / len(test_loader)
    val_acc = 100. * correct / total
    
    # Calculate detailed metrics
    all_logits = torch.cat(all_logits, dim=0)
    metrics = calculate_metrics(
        all_labels, all_predictions, all_logits, class_names
    )
    
    return val_loss, val_acc, metrics


def main(args):
    # Load configuration
    config = load_config(args.config) if args.config else {}
    config['model_name'] = args.model_name
    config['data_dir'] = args.data_dir
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Get device
    device = get_device()
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.model_name)
    
    # Initialize WandB
    setup_wandb(config, project_name="cv_classification")
    
    # Load data
    print("\nLoading CIFAR-10 data...")
    train_loader, test_loader, class_names = get_dataloaders(
        root_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True
    )
    
    # Create model
    print(f"\nCreating {args.model_name} model...")
    model = get_model(
        args.model_name,
        num_classes=len(class_names),
        class_names=class_names,
        freeze_backbone=args.freeze_backbone
    )
    
    # Apply Xavier initialization if no checkpoint
    if not args.resume:
        print("Applying Xavier initialization to classifier...")
        if hasattr(model, 'classifier'):
            model.classifier.apply(xavier_init)
    
    model = model.to(device)
    
    # Print parameter count
    param_counts = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Mixed precision scaler for FP16
    scaler = GradScaler()
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_iter = 0
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint_path = os.path.join(
            exp_dir, 'checkpoints', f'{args.model_name}_latest.pth'
        )
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            start_epoch = checkpoint['epoch']
            global_iter = checkpoint['iteration']
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch}, iteration {global_iter}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting fresh")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc, global_iter = train_epoch(
            model, train_loader, test_loader, criterion, optimizer, scaler,
            device, epoch, global_iter, config, exp_dir
        )
        
        # Validate on test set
        test_loss, test_acc, test_metrics = validate(
            model, test_loader, criterion, device, class_names
        )
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(test_acc)
        
        # Log epoch metrics to WandB
        epoch_metrics = {
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'train/epoch_accuracy': train_acc,
            'val/loss': test_loss,
            'val/accuracy': test_acc,
            **{f'val/{k}': v for k, v in test_metrics.items()}
        }
        log_to_wandb(epoch_metrics, global_iter)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print(f"  Test F1: {test_metrics['f1_macro']:.4f}")
        
        # Save best model
        is_best = test_acc > best_val_acc
        if is_best:
            best_val_acc = test_acc
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_iter,
            best_val_acc,
            os.path.join(exp_dir, 'checkpoints'),
            config['model_name'],
            is_best=is_best
        )
    
    # Plot training curves
    plot_path = os.path.join(exp_dir, 'visualizations', 'training_curves.png')
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, plot_path
    )
    
    # Final test evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)
    
    test_loss, test_acc, test_metrics = validate(
        model, test_loader, criterion, device, class_names
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
    
    # Log final test metrics to WandB
    test_wandb_metrics = {
        'test/loss': test_loss,
        'test/accuracy': test_acc,
        **{f'test/{k}': v for k, v in test_metrics.items()}
    }
    log_to_wandb(test_wandb_metrics, global_iter)
    
    print(f"\n✓ Training completed! Best val accuracy: {best_val_acc:.2f}%")
    print(f"✓ Results saved to: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['dinov2', 'mae', 'swin', 'vit', 'clip', 'vla', 'custom'],
                        help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory for experiments')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    main(args)