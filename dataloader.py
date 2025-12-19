"""
dataloader.py - Dataset and DataLoader for CIFAR-10 image classification
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor
from PIL import Image
import os
from typing import Tuple, Optional, Dict, List
import numpy as np


class CIFAR10Dataset(Dataset):
    """CIFAR-10 image classification dataset"""
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        model_name: str = 'dinov2',
        download: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.model_name = model_name
        
        # Load image processor based on model
        self.processor = self._get_processor(model_name)
        
        # Load CIFAR-10 dataset
        is_train = (split == 'train')
        self.dataset = datasets.CIFAR10(
            root=root_dir,
            train=is_train,
            download=download,
            transform=None
        )
        
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Loaded {len(self.dataset)} images from CIFAR-10 {split} split")
        print(f"Classes: {self.classes}")
        
    def _get_processor(self, model_name: str):
        """Get appropriate image processor for model"""
        processor_map = {
            'dinov2': 'facebook/dinov2-base',
            'mae': 'facebook/vit-mae-base',
            'swin': 'microsoft/swinv2-base-patch4-window8-256',
            'vit': 'google/vit-base-patch16-224',
            'clip': 'openai/clip-vit-base-patch32',
            'vla': 'google/siglip-base-patch16-224',
            'custom': 'google/vit-base-patch16-224-in21k'
        }
        
        model_id = processor_map.get(model_name.lower(), 'google/vit-base-patch16-224')
        return AutoImageProcessor.from_pretrained(model_id)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        # Get raw image array directly from CIFAR-10 data (not through __getitem__)
        image_array = self.dataset.data[idx]
        label = self.dataset.targets[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        
        # Process with model-specific processor
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        
        # Return image, label, and metadata
        metadata = {
            'idx': idx,
            'class_name': self.classes[label]
        }
        
        return pixel_values, label, metadata


class AugmentedCIFAR10Dataset(CIFAR10Dataset):
    """CIFAR-10 dataset with advanced augmentations for training"""
    def __init__(self, *args, **kwargs):
        # Define augmentations before calling super
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ])
        
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        # Get raw image array directly from CIFAR-10 data (not through __getitem__)
        image_array = self.dataset.data[idx]
        label = self.dataset.targets[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        
        # Apply augmentations for training
        if self.split == 'train':
            image = self.augment_transform(image)
        
        # Process with model-specific processor
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        
        metadata = {
            'idx': idx,
            'class_name': self.classes[label]
        }
        
        return pixel_values, label, metadata


def get_dataloaders(
    root_dir: str,
    model_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and test dataloaders for CIFAR-10
    
    Args:
        root_dir: Root directory for CIFAR-10 dataset
        model_name: Model name for processor selection
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_augmentation: Whether to use data augmentation
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, test_loader, class_names
    """
    
    DatasetClass = AugmentedCIFAR10Dataset if use_augmentation else CIFAR10Dataset
    
    # Create datasets
    train_dataset = DatasetClass(
        root_dir=root_dir,
        split='train',
        model_name=model_name,
        download=True
    )
    
    test_dataset = CIFAR10Dataset(
        root_dir=root_dir,
        split='test',
        model_name=model_name,
        download=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n{'='*50}")
    print(f"CIFAR-10 DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    print(f"  Classes: {len(train_dataset.classes)} ({', '.join(train_dataset.classes)})")
    print(f"{'='*50}\n")
    
    return train_loader, test_loader, train_dataset.classes


def collate_fn(batch):
    """Custom collate function for batching"""
    images, labels, metadata = zip(*batch)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels, metadata


if __name__ == "__main__":
    # Test CIFAR-10 dataloader
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        print("Testing CIFAR-10 dataloaders...")
        train_loader, test_loader, classes = get_dataloaders(
            root_dir=temp_dir,
            model_name='dinov2',
            batch_size=32,
            num_workers=0,
            use_augmentation=True
        )
        
        # Test batch
        images, labels, metadata = next(iter(train_loader))
        print(f"\nBatch test:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {classes}")
        print(f"  Number of classes: {len(classes)}")
        print(f"  Metadata sample: {metadata[0]}")
        print(f"\n✓ CIFAR-10 dataloader working correctly!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    try:
        for split in ['train', 'val', 'test']:
            for class_name in ['cat', 'dog']:
                class_dir = os.path.join(temp_dir, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Create dummy images
                for i in range(5):
                    img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                    img.save(os.path.join(class_dir, f'img_{i}.jpg'))
        
        # Test dataloaders
        print("Testing dataloaders...")
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            root_dir=temp_dir,
            model_name='dinov2',
            batch_size=2,
            num_workers=0
        )
        
        # Test batch
        images, labels, metadata = next(iter(train_loader))
        print(f"\nBatch test:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {classes}")
        print(f"  Metadata: {metadata[0]}")
        
    finally:
        shutil.rmtree(temp_dir)
        print("\nCleaned up temporary directory")