"""
visualize_attention_activation.py - Visualize attention maps and activation maps
For transformer-based models with attention mechanisms
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse
import os
from typing import Tuple, List

from model import get_model
from utils import load_checkpoint, get_device
from transformers import AutoImageProcessor


def visualize_attention_maps(
    attention_weights: torch.Tensor,
    image: torch.Tensor,
    save_path: str,
    layer_idx: int = -1,
    head_idx: int = 0,
    image_size: Tuple[int, int] = (224, 224)
):
    """
    Visualize attention maps from transformer layers
    
    Args:
        attention_weights: Attention weights tensor (num_layers, num_heads, seq_len, seq_len)
        image: Original image tensor (C, H, W)
        save_path: Path to save visualization
        layer_idx: Which layer to visualize (-1 for last layer)
        head_idx: Which attention head to visualize
        image_size: Size of input image
    """
    # Get attention from specified layer and head
    attn = attention_weights[layer_idx][0, head_idx].cpu().detach().numpy()
    
    # Get attention from CLS token to all patches
    cls_attn = attn[0, 1:]  # Skip CLS token itself
    
    # Reshape to spatial dimensions
    num_patches = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(num_patches, num_patches)
    
    # Resize to image size
    attn_map = np.array(Image.fromarray(attn_map).resize(image_size, Image.BILINEAR))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # Prepare image
    img = image.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attn_map, cmap='jet', alpha=0.8)
    axes[1].set_title(f'Attention Map (Layer {layer_idx}, Head {head_idx})', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attn_map, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention visualization saved to {save_path}")


def visualize_all_attention_heads(
    attention_weights: torch.Tensor,
    image: torch.Tensor,
    save_path: str,
    layer_idx: int = -1,
    max_heads: int = 12,
    image_size: Tuple[int, int] = (224, 224)
):
    """
    Visualize attention maps from all heads in a layer
    
    Args:
        attention_weights: Attention weights
        image: Original image
        save_path: Save path
        layer_idx: Layer to visualize
        max_heads: Maximum number of heads to show
        image_size: Image size
    """
    attn = attention_weights[layer_idx][0].cpu().detach().numpy()
    num_heads = min(attn.shape[0], max_heads)
    
    # Calculate grid size
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        # Get CLS token attention
        cls_attn = attn[head_idx, 0, 1:]
        
        # Reshape to spatial
        num_patches = int(np.sqrt(len(cls_attn)))
        attn_map = cls_attn.reshape(num_patches, num_patches)
        
        # Resize
        attn_map = np.array(Image.fromarray(attn_map).resize(image_size, Image.BILINEAR))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        
        # Prepare image
        img = image.cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[head_idx].imshow(img)
        axes[head_idx].imshow(attn_map, cmap='jet', alpha=0.5)
        axes[head_idx].set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
        axes[head_idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'All Attention Heads (Layer {layer_idx})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Multi-head attention visualization saved to {save_path}")


def visualize_activation_maps(
    hidden_states: Tuple[torch.Tensor],
    image: torch.Tensor,
    save_path: str,
    layer_indices: List[int] = None,
    num_features: int = 64
):
    """
    Visualize activation maps from hidden states
    
    Args:
        hidden_states: Tuple of hidden states from model
        image: Original image
        save_path: Save path
        layer_indices: Which layers to visualize
        num_features: Number of feature channels to show
    """
    if layer_indices is None:
        # Visualize first, middle, and last layers
        num_layers = len(hidden_states)
        layer_indices = [0, num_layers // 2, num_layers - 1]
    
    fig, axes = plt.subplots(len(layer_indices), num_features // 8, 
                             figsize=(20, len(layer_indices) * 2.5))
    
    if len(layer_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, layer_num in enumerate(layer_indices):
        activations = hidden_states[layer_num][0].cpu().detach().numpy()
        
        # For transformers, activations are (seq_len, hidden_dim)
        # Need to reshape to spatial format
        seq_len, hidden_dim = activations.shape
        
        if seq_len > 1:  # Has patches
            num_patches = int(np.sqrt(seq_len - 1))  # Exclude CLS token
            patch_activations = activations[1:, :]  # Skip CLS token
            
            # Sample feature channels
            channel_indices = np.linspace(0, hidden_dim - 1, num_features // 8, dtype=int)
            
            for feat_idx, channel_idx in enumerate(channel_indices):
                # Get feature map
                feat_map = patch_activations[:, channel_idx]
                feat_map = feat_map.reshape(num_patches, num_patches)
                
                # Normalize
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                
                # Plot
                im = axes[layer_idx, feat_idx].imshow(feat_map, cmap='viridis')
                axes[layer_idx, feat_idx].axis('off')
                
                if layer_idx == 0:
                    axes[layer_idx, feat_idx].set_title(f'F{channel_idx}', fontsize=8)
        
        # Add layer label
        axes[layer_idx, 0].set_ylabel(f'Layer {layer_num}', 
                                       fontsize=10, fontweight='bold', rotation=0,
                                       labelpad=40, va='center')
    
    plt.suptitle('Activation Maps Across Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Activation maps visualization saved to {save_path}")


def visualize_cls_token_evolution(
    hidden_states: Tuple[torch.Tensor],
    save_path: str
):
    """
    Visualize how CLS token embedding evolves through layers
    
    Args:
        hidden_states: Hidden states from all layers
        save_path: Save path
    """
    cls_tokens = []
    
    for layer_hidden in hidden_states:
        cls_token = layer_hidden[0, 0, :].cpu().detach().numpy()
        cls_tokens.append(cls_token)
    
    cls_tokens = np.array(cls_tokens)  # (num_layers, hidden_dim)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Heatmap of CLS token across layers
    im1 = ax1.imshow(cls_tokens, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_xlabel('Feature Dimension', fontsize=12)
    ax1.set_ylabel('Layer', fontsize=12)
    ax1.set_title('CLS Token Evolution', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Norm of CLS token per layer
    cls_norms = np.linalg.norm(cls_tokens, axis=1)
    ax2.plot(range(len(cls_norms)), cls_norms, marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('L2 Norm', fontsize=12)
    ax2.set_title('CLS Token L2 Norm per Layer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CLS token evolution visualization saved to {save_path}")


def main(args):
    # Get device
    device = get_device()
    
    # Load processor
    processor_map = {
        'dinov2': 'facebook/dinov2-base',
        'mae': 'facebook/vit-mae-base',
        'swin': 'microsoft/swinv2-base-patch4-window8-256',
        'vit': 'google/vit-base-patch16-224',
        'clip': 'openai/clip-vit-base-patch32',
        'vla': 'google/siglip-base-patch16-224'
    }
    
    processor = AutoImageProcessor.from_pretrained(
        processor_map[args.model_name.lower()]
    )
    
    # Load image
    image = Image.open(args.image_path).convert('RGB')
    processed = processor(images=image, return_tensors="pt")
    pixel_values = processed['pixel_values'].to(device)
    
    # Create model
    model = get_model(args.model_name, num_classes=args.num_classes)
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device=device)
    
    model.eval()
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        logits, aux_outputs = model(pixel_values)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize based on what's available
    if aux_outputs.get('attentions') is not None:
        print("Visualizing attention maps...")
        
        # Single attention head
        visualize_attention_maps(
            aux_outputs['attentions'],
            pixel_values[0],
            os.path.join(args.output_dir, 'attention_single.png'),
            layer_idx=args.layer_idx,
            head_idx=args.head_idx
        )
        
        # All attention heads
        visualize_all_attention_heads(
            aux_outputs['attentions'],
            pixel_values[0],
            os.path.join(args.output_dir, 'attention_all_heads.png'),
            layer_idx=args.layer_idx
        )
    
    if aux_outputs.get('hidden_states') is not None:
        print("Visualizing activation maps...")
        
        # Activation maps
        visualize_activation_maps(
            aux_outputs['hidden_states'],
            pixel_values[0],
            os.path.join(args.output_dir, 'activation_maps.png')
        )
        
        # CLS token evolution
        if aux_outputs['hidden_states'][0].shape[1] > 1:
            visualize_cls_token_evolution(
                aux_outputs['hidden_states'],
                os.path.join(args.output_dir, 'cls_token_evolution.png')
            )
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize attention and activation maps'
    )
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['dinov2', 'mae', 'swin', 'vit', 'clip', 'vla'],
                        help='Model architecture')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes (for model creation)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory')
    parser.add_argument('--layer_idx', type=int, default=-1,
                        help='Layer index to visualize (-1 for last)')
    parser.add_argument('--head_idx', type=int, default=0,
                        help='Attention head index to visualize')
    
    args = parser.parse_args()
    main(args)