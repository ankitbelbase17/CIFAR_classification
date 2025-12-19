"""
tests/test_train.py - Unit tests for training pipeline
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import tempfile
import shutil
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model
from utils import save_checkpoint, load_checkpoint, count_parameters
from metrics import calculate_metrics


class TestTraining(unittest.TestCase):
    """Test cases for training pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.device = torch.device('cpu')
        
    def test_loss_calculation(self):
        """Test loss calculation"""
        criterion = torch.nn.CrossEntropyLoss()
        
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        
        loss = criterion(logits, labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        print("✓ Loss calculation test passed")
    
    def test_optimizer_step(self):
        """Test optimizer step"""
        model = get_model('vit', 10)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Get initial param values
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward and backward
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 10, (2,))
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        changed = False
        for p_initial, p_current in zip(initial_params, model.parameters()):
            if not torch.equal(p_initial, p_current):
                changed = True
                break
        
        self.assertTrue(changed)
        print("✓ Optimizer step test passed")
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create model and optimizer
            model = get_model('vit', 10)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, None,
                epoch=5, iteration=100, best_val_acc=0.85,
                save_dir=temp_dir, model_name='test_model'
            )
            
            # Create new model and load
            new_model = get_model('vit', 10)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            checkpoint_path = os.path.join(temp_dir, 'test_model_latest.pth')
            checkpoint = load_checkpoint(checkpoint_path, new_model, new_optimizer)
            
            self.assertEqual(checkpoint['epoch'], 5)
            self.assertEqual(checkpoint['iteration'], 100)
            print("✓ Checkpoint save/load test passed")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_mixed_precision_training(self):
        """Test mixed precision training"""
        model = get_model('vit', 10)
        scaler = GradScaler()
        
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 10, (2,))
        
        with autocast():
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, target)
        
        scaler.scale(loss).backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grads)
        
        print("✓ Mixed precision training test passed")
    
    def test_custom_model_training(self):
        """Test custom model in training loop"""
        model = get_model('custom', 10)
        model.train()
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Simulate training step
        x = torch.randn(4, 3, 224, 224)
        target = torch.randint(0, 10, (4,))
        
        logits, _ = model(x)
        loss = criterion(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss is a scalar
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        
        print("✓ Custom model training test passed")
    
    def test_all_models_forward_pass(self):
        """Test forward pass for all models"""
        model_names = ['dinov2', 'mae', 'swin', 'vit', 'clip', 'vla', 'custom']
        num_classes = 10
        batch_size = 2
        
        for model_name in model_names:
            with self.subTest(model=model_name):
                model = get_model(model_name, num_classes, 
                                 class_names=[f'class_{i}' for i in range(num_classes)])
                model.eval()
                
                # Different models may need different image sizes
                if model_name == 'swin':
                    x = torch.randn(batch_size, 3, 256, 256)
                else:
                    x = torch.randn(batch_size, 3, 224, 224)
                
                with torch.no_grad():
                    logits, aux = model(x)
                
                self.assertEqual(logits.shape, (batch_size, num_classes))
                print(f"✓ {model_name} forward pass test passed")


if __name__ == "__main__":
    unittest.main()
