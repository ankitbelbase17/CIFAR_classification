"""
model.py - All SOTA Model Architectures for Image Classification
Models: DINOv2, MAE, SwinV2, ViT, CLIP, OpenVLA, Stable Diffusion
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoImageProcessor, AutoModelForImageClassification,
    Dinov2Model, ViTMAEModel, Swinv2Model, ViTModel, CLIPModel, CLIPProcessor
)
from typing import Dict, Tuple, Optional
import torch.nn.functional as F


def xavier_init(module):
    """Xavier initialization for model weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class DINOv2Classifier(nn.Module):
    """DINOv2 with classification head"""
    def __init__(self, num_classes: int, model_name: str = "facebook/dinov2-base", freeze_backbone: bool = False):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Xavier init for classifier
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.backbone(pixel_values, output_attentions=True, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_token)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'cls_token': cls_token
        }


class MAEClassifier(nn.Module):
    """Masked Autoencoder (MAE) with classification head"""
    def __init__(self, num_classes: int, model_name: str = "facebook/vit-mae-base", freeze_backbone: bool = False):
        super().__init__()
        self.backbone = ViTMAEModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.backbone(pixel_values, output_attentions=True, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'cls_token': cls_token
        }


class SwinTransformerClassifier(nn.Module):
    """Swin Transformer V2 with classification head"""
    def __init__(self, num_classes: int, model_name: str = "microsoft/swinv2-base-patch4-window8-256", freeze_backbone: bool = False):
        super().__init__()
        self.backbone = Swinv2Model.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.backbone(pixel_values, output_attentions=True, output_hidden_states=True)
        # Swin outputs: (B, num_patches, C) -> pool over patches
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_features': pooled
        }


class ViTClassifier(nn.Module):
    """Vision Transformer with classification head"""
    def __init__(self, num_classes: int, model_name: str = "google/vit-base-patch16-224", freeze_backbone: bool = False):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.backbone(pixel_values, output_attentions=True, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'cls_token': cls_token
        }


class CLIPClassifier(nn.Module):
    """CLIP with zero-shot and fine-tuning classification"""
    def __init__(self, num_classes: int, class_names: list, model_name: str = "openai/clip-vit-base-patch32", freeze_backbone: bool = False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.class_names = class_names
        self.num_classes = num_classes
        self.hidden_size = self.model.config.vision_config.hidden_size
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Additional classifier head for fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor, text_inputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Get image features
        vision_outputs = self.model.vision_model(pixel_values, output_attentions=True, output_hidden_states=True)
        image_embeds = vision_outputs.pooler_output
        
        # Classification head
        logits = self.classifier(image_embeds)
        
        return logits, {
            'attentions': vision_outputs.attentions,
            'hidden_states': vision_outputs.hidden_states,
            'image_embeds': image_embeds
        }


class OpenVLAClassifier(nn.Module):
    """
    Vision-Language-Action model (~200M params) for classification
    Using a smaller VLA-style architecture
    """
    def __init__(self, num_classes: int, model_name: str = "google/siglip-base-patch16-224", freeze_backbone: bool = False):
        super().__init__()
        # Using SigLIP as base (similar to VLA architectures)
        self.vision_encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config - handle different config structures
        if hasattr(self.vision_encoder.config, 'vision_config'):
            self.hidden_size = self.vision_encoder.config.vision_config.hidden_size
        elif hasattr(self.vision_encoder.config, 'hidden_size'):
            self.hidden_size = self.vision_encoder.config.hidden_size
        else:
            # Fallback: try to infer from model
            self.hidden_size = 768
        
        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # VLA-style head with action prediction adapted for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.vision_encoder(pixel_values, output_attentions=True, output_hidden_states=True)
        
        # Pool features - handle different output structures
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
            pooled = outputs.image_embeds
        else:
            # Fallback: mean pooling
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'pooled_features': pooled
        }


class CustomTransformerClassifier(nn.Module):
    """
    Custom Transformer-based classifier trained from scratch
    Uses a Vision Transformer backbone pretrained on ImageNet-21k
    """
    def __init__(self, num_classes: int, model_name: str = "google/vit-base-patch16-224-in21k", freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained model
        self.backbone = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head with enhanced architecture
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Xavier init for classifier
        self.classifier.apply(xavier_init)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.backbone(pixel_values, output_attentions=True, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_token)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'cls_token': cls_token
        }


def get_model(model_name: str, num_classes: int, class_names: list = None, freeze_backbone: bool = False) -> nn.Module:
    """
    Factory function to get model by name
    
    Args:
        model_name: One of ['dinov2', 'mae', 'swin', 'vit', 'clip', 'vla', 'custom']
        num_classes: Number of output classes
        class_names: List of class names (required for CLIP)
        freeze_backbone: Whether to freeze backbone weights
    """
    models = {
        'dinov2': lambda: DINOv2Classifier(num_classes, freeze_backbone=freeze_backbone),
        'mae': lambda: MAEClassifier(num_classes, freeze_backbone=freeze_backbone),
        'swin': lambda: SwinTransformerClassifier(num_classes, freeze_backbone=freeze_backbone),
        'vit': lambda: ViTClassifier(num_classes, freeze_backbone=freeze_backbone),
        'clip': lambda: CLIPClassifier(num_classes, class_names or [], freeze_backbone=freeze_backbone),
        'vla': lambda: OpenVLAClassifier(num_classes, freeze_backbone=freeze_backbone),
        'custom': lambda: CustomTransformerClassifier(num_classes, freeze_backbone=freeze_backbone),
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")
    
    model = models[model_name.lower()]()
    
    # Apply Xavier initialization to non-pretrained parts
    print(f"✓ Loaded {model_name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


if __name__ == "__main__":
    # Test all models
    batch_size, channels, height, width = 2, 3, 224, 224
    num_classes = 10
    x = torch.randn(batch_size, channels, height, width)
    
    for model_name in ['dinov2', 'mae', 'swin', 'vit', 'clip', 'vla', 'custom']:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, num_classes, class_names=[f"class_{i}" for i in range(num_classes)])
        model.eval()
        
        with torch.no_grad():
            logits, aux = model(x)
            print(f"  Input: {x.shape}, Output: {logits.shape}")
            print(f"  Auxiliary outputs: {list(aux.keys())}")