"""
tests/test_model.py - Unit tests for model architectures
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    get_model, DINOv2Classifier, MAEClassifier,
    SwinTransformerClassifier, ViTClassifier,
    CLIPClassifier, OpenVLAClassifier, CustomTransformerClassifier, xavier_init
)


class TestModels(unittest.TestCase):
    """Test cases for all model architectures"""
    
    def setUp(self):
        """Set up test parameters"""
        self.batch_size = 2
        self.num_classes = 10
        self.image_size = 224
        self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        
    def test_dinov2_forward(self):
        """Test DINOv2 forward pass"""
        model = get_model('dinov2', self.num_classes)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('attentions', aux)
        self.assertIn('hidden_states', aux)
        print("✓ DINOv2 forward pass test passed")
    
    def test_mae_forward(self):
        """Test MAE forward pass"""
        model = get_model('mae', self.num_classes)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('attentions', aux)
        print("✓ MAE forward pass test passed")
    
    def test_swin_forward(self):
        """Test Swin Transformer forward pass"""
        model = get_model('swin', self.num_classes)
        # Swin requires specific image size
        x = torch.randn(self.batch_size, 3, 256, 256)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        print("✓ Swin Transformer forward pass test passed")
    
    def test_vit_forward(self):
        """Test ViT forward pass"""
        model = get_model('vit', self.num_classes)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('attentions', aux)
        print("✓ ViT forward pass test passed")
    
    def test_clip_forward(self):
        """Test CLIP forward pass"""
        model = get_model('clip', self.num_classes, self.class_names)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('image_embeds', aux)
        print("✓ CLIP forward pass test passed")
    
    def test_vla_forward(self):
        """Test VLA forward pass"""
        model = get_model('vla', self.num_classes)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        print("✓ VLA forward pass test passed")
    
    def test_custom_transformer_forward(self):
        """Test Custom Transformer forward pass"""
        model = get_model('custom', self.num_classes)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        logits, aux = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIn('attentions', aux)
        self.assertIn('hidden_states', aux)
        self.assertIn('cls_token', aux)
        print("✓ Custom Transformer forward pass test passed")
    
    def test_xavier_initialization(self):
        """Test Xavier initialization"""
        model = get_model('vit', self.num_classes)
        
        # Check if classifier has Xavier initialized weights
        for module in model.classifier.modules():
            if isinstance(module, torch.nn.Linear):
                # Check weight variance is reasonable
                var = module.weight.var().item()
                self.assertGreater(var, 0)
                self.assertLess(var, 1)
        
        print("✓ Xavier initialization test passed")
    
    def test_model_freeze_backbone(self):
        """Test freezing backbone"""
        model = get_model('dinov2', self.num_classes, freeze_backbone=True)
        
        # Check backbone is frozen
        for param in model.backbone.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check classifier is trainable
        for param in model.classifier.parameters():
            self.assertTrue(param.requires_grad)
        
        print("✓ Freeze backbone test passed")
    
    def test_parameter_count(self):
        """Test parameter counting"""
        from utils import count_parameters
        
        model = get_model('vit', self.num_classes)
        params = count_parameters(model)
        
        self.assertGreater(params['total'], 0)
        self.assertEqual(params['total'], params['trainable'] + params['frozen'])
        
        print(f"✓ Parameter count test passed (Total: {params['total']:,})")
    
    def test_gradient_flow(self):
        """Test gradient flow through models"""
        model = get_model('vit', self.num_classes)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        target = torch.randint(0, self.num_classes, (self.batch_size,))
        
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients)
        print("✓ Gradient flow test passed")
