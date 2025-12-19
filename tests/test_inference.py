"""
tests/test_inference.py - Unit tests for inference pipeline
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os
import tempfile
from PIL import Image
from transformers import AutoImageProcessor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_model


class TestInference(unittest.TestCase):
    """Test cases for inference pipeline"""
    
    def test_single_image_inference(self):
        """Test single image inference"""
        # Create dummy image
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img = Image.new('RGB', (224, 224), color='red')
        img.save(temp_file.name)
        
        try:
            model = get_model('vit', 10)
            model.eval()
            
            # Load and process image
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
            
            image = Image.open(temp_file.name)
            processed = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                logits, _ = model(processed['pixel_values'])
            
            self.assertEqual(logits.shape[0], 1)
            self.assertEqual(logits.shape[1], 10)
            print("✓ Single image inference test passed")
            
        finally:
            os.unlink(temp_file.name)
    
    def test_batch_inference(self):
        """Test batch inference"""
        model = get_model('vit', 10)
        model.eval()
        
        batch = torch.randn(8, 3, 224, 224)
        
        with torch.no_grad():
            logits, _ = model(batch)
        
        self.assertEqual(logits.shape, (8, 10))
        print("✓ Batch inference test passed")
    
    def test_top_k_predictions(self):
        """Test top-k predictions"""
        model = get_model('vit', 10)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            logits, _ = model(x)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs[0], k=5)
        
        self.assertEqual(len(top_probs), 5)
        self.assertEqual(len(top_indices), 5)
        self.assertTrue(torch.all(top_probs[:-1] >= top_probs[1:]))  # Check sorted
        print("✓ Top-k predictions test passed")
    
    def test_confidence_scores(self):
        """Test confidence score calculation"""
        model = get_model('vit', 10)
        model.eval()
        
        x = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            logits, _ = model(x)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        self.assertTrue(torch.all(confidences >= 0))
        self.assertTrue(torch.all(confidences <= 1))
        self.assertTrue(torch.all(torch.abs(probs.sum(dim=1) - 1) < 1e-5))
        print("✓ Confidence scores test passed")
    
    def test_custom_model_inference(self):
        """Test custom model inference"""
        model = get_model('custom', 10)
        model.eval()
        
        x = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            logits, aux = model(x)
            probs = F.softmax(logits, dim=1)
        
        self.assertEqual(logits.shape, (4, 10))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))
        print("✓ Custom model inference test passed")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
