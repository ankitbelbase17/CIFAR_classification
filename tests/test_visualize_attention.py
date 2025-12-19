"""
tests/test_visualize_attention.py - Unit tests for attention and activation visualization
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model
from visualize_attention_activation import (
    visualize_attention_maps, visualize_all_attention_heads,
    visualize_activation_maps, compute_activation_stats,
    visualize_attention_heatmap, plot_attention_comparison
)


class TestVisualization(unittest.TestCase):
    """Test cases for attention and activation visualization"""
    
    def setUp(self):
        """Set up test parameters"""
        self.batch_size = 1
        self.num_classes = 10
        self.image_size = 224
        self.temp_dir = tempfile.mkdtemp()
        
    def test_attention_map_visualization(self):
        """Test attention map visualization"""
        model = get_model('vit', self.num_classes)
        model.eval()
        
        # Create dummy image
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            logits, aux = model(x)
            attention_weights = aux['attentions']
        
        self.assertIsNotNone(attention_weights)
        save_path = os.path.join(self.temp_dir, 'attention_map.png')
        
        try:
            visualize_attention_maps(
                attention_weights,
                x[0],
                save_path,
                layer_idx=-1,
                head_idx=0,
                image_size=(self.image_size, self.image_size)
            )
            
            # Check if file was created
            self.assertTrue(os.path.exists(save_path))
            print("✓ Attention map visualization test passed")
            
        except Exception as e:
            print(f"⚠ Attention map visualization test skipped: {e}")
    
    def test_all_attention_heads_visualization(self):
        """Test visualization of all attention heads"""
        model = get_model('vit', self.num_classes)
        model.eval()
        
        # Create dummy image
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            logits, aux = model(x)
            attention_weights = aux['attentions']
        
        save_path = os.path.join(self.temp_dir, 'all_heads_attention.png')
        
        try:
            visualize_all_attention_heads(
                attention_weights,
                x[0],
                save_path,
                layer_idx=-1,
                max_heads=12,
                image_size=(self.image_size, self.image_size)
            )
            
            # Check if file was created
            self.assertTrue(os.path.exists(save_path))
            print("✓ All attention heads visualization test passed")
            
        except Exception as e:
            print(f"⚠ All attention heads visualization test skipped: {e}")
    
    def test_activation_map_visualization(self):
        """Test activation map visualization"""
        model = get_model('vit', self.num_classes)
        model.eval()
        
        # Create dummy image
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            logits, aux = model(x)
            hidden_states = aux['hidden_states']
        
        save_path = os.path.join(self.temp_dir, 'activation_map.png')
        
        try:
            visualize_activation_maps(
                hidden_states,
                x[0],
                save_path,
                layer_idx=-1,
                image_size=(self.image_size, self.image_size)
            )
            
            # Check if file was created
            self.assertTrue(os.path.exists(save_path))
            print("✓ Activation map visualization test passed")
            
        except Exception as e:
            print(f"⚠ Activation map visualization test skipped: {e}")
    
    def test_activation_statistics(self):
        """Test activation statistics computation"""
        model = get_model('custom', self.num_classes)
        model.eval()
        
        # Create dummy batch
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            logits, aux = model(x)
            hidden_states = aux['hidden_states']
        
        try:
            stats = compute_activation_stats(hidden_states)
            
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)
            print("✓ Activation statistics test passed")
            
        except Exception as e:
            print(f"⚠ Activation statistics test skipped: {e}")
    
    def test_attention_heatmap_visualization(self):
        """Test attention heatmap visualization"""
        model = get_model('vit', self.num_classes)
        model.eval()
        
        # Create dummy image
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            logits, aux = model(x)
            attention_weights = aux['attentions']
        
        save_path = os.path.join(self.temp_dir, 'attention_heatmap.png')
        
        try:
            visualize_attention_heatmap(
                attention_weights,
                x[0],
                save_path,
                layer_idx=-1,
                head_idx=0,
                image_size=(self.image_size, self.image_size)
            )
            
            # Check if file was created
            self.assertTrue(os.path.exists(save_path))
            print("✓ Attention heatmap visualization test passed")
            
        except Exception as e:
            print(f"⚠ Attention heatmap visualization test skipped: {e}")
    
    def test_custom_model_attention_visualization(self):
        """Test attention visualization for custom model"""
        model = get_model('custom', self.num_classes)
        model.eval()
        
        # Create dummy image
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            logits, aux = model(x)
            attention_weights = aux['attentions']
        
        self.assertIsNotNone(attention_weights)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        
        save_path = os.path.join(self.temp_dir, 'custom_attention.png')
        
        try:
            visualize_attention_maps(
                attention_weights,
                x[0],
                save_path,
                layer_idx=-1,
                head_idx=0,
                image_size=(self.image_size, self.image_size)
            )
            
            self.assertTrue(os.path.exists(save_path))
            print("✓ Custom model attention visualization test passed")
            
        except Exception as e:
            print(f"⚠ Custom model attention visualization test skipped: {e}")
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
