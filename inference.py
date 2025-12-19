"""
inference.py - Inference script for trained models
Supports batch inference and single image prediction
"""

import torch
import torch.nn.functional as F
import argparse
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

from model import get_model
from dataloader import get_dataloaders, ImageClassificationDataset
from utils import load_checkpoint, get_device, visualize_predictions
from metrics import calculate_metrics


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    processor,
    class_names: list,
    device: str,
    top_k: int = 5
) -> dict:
    """
    Predict class for a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        processor: Image processor
        class_names: List of class names
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and probabilities
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    processed = processor(images=image, return_tensors="pt")
    pixel_values = processed['pixel_values'].to(device)
    
    # Inference
    with torch.no_grad():
        logits, aux_outputs = model(pixel_values)
        probs = F.softmax(logits, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(class_names)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': class_names[idx.item()],
            'class_idx': idx.item(),
            'probability': prob.item()
        })
    
    return {
        'image_path': image_path,
        'predictions': predictions,
        'top_prediction': predictions[0]
    }


def batch_inference(
    model: torch.nn.Module,
    dataloader,
    device: str,
    class_names: list,
    save_results: bool = True,
    output_dir: str = None
) -> dict:
    """
    Run inference on a batch of images
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: Device to run inference on
        class_names: List of class names
        save_results: Whether to save results to file
        output_dir: Directory to save results
    
    Returns:
        Dictionary with all predictions and metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    all_metadata = []
    
    print("Running batch inference...")
    with torch.no_grad():
        for images, labels, metadata in tqdm(dataloader):
            images = images.to(device)
            
            logits, aux_outputs = model(images)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu())
            all_metadata.extend(metadata)
    
    # Concatenate all logits
    all_logits = torch.cat(all_logits, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_labels, all_predictions, all_logits, class_names
    )
    
    # Prepare results
    results = {
        'num_samples': len(all_predictions),
        'metrics': metrics,
        'predictions': []
    }
    
    # Add individual predictions
    for i in range(len(all_predictions)):
        pred_probs = F.softmax(all_logits[i], dim=0)
        top_prob, top_idx = torch.max(pred_probs, 0)
        
        results['predictions'].append({
            'image_path': all_metadata[i]['image_path'],
            'true_label': class_names[all_labels[i]],
            'true_label_idx': int(all_labels[i]),
            'predicted_label': class_names[all_predictions[i]],
            'predicted_label_idx': int(all_predictions[i]),
            'confidence': float(top_prob.item()),
            'correct': all_labels[i] == all_predictions[i]
        })
    
    # Save results to file
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'inference_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Inference Summary")
    print("="*50)
    print(f"Total samples: {results['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print("="*50)
    
    return results


def main(args):
    # Get device
    device = get_device()
    
    # Load class names (from training)
    if args.mode == 'batch':
        # Load from dataset
        dataset = ImageClassificationDataset(
            root_dir=args.data_dir,
            split=args.split,
            model_name=args.model_name
        )
        class_names = dataset.classes
        processor = dataset.processor
    else:
        # Load from class_names file or use default
        if args.class_names_file and os.path.exists(args.class_names_file):
            with open(args.class_names_file, 'r') as f:
                class_names = json.load(f)
        else:
            raise ValueError("For single image inference, provide --class_names_file")
        
        from transformers import AutoImageProcessor
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
    
    # Create model
    print(f"\nLoading {args.model_name} model...")
    model = get_model(
        args.model_name,
        num_classes=len(class_names),
        class_names=class_names
    )
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")
    
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Run inference based on mode
    if args.mode == 'single':
        # Single image inference
        result = predict_single_image(
            model, args.image_path, processor,
            class_names, device, top_k=args.top_k
        )
        
        print("\n" + "="*50)
        print(f"Predictions for: {args.image_path}")
        print("="*50)
        for i, pred in enumerate(result['predictions'], 1):
            print(f"{i}. {pred['class']}: {pred['probability']:.4f}")
        print("="*50)
        
        # Save result
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 'prediction.json')
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Result saved to: {output_path}")
    
    else:
        # Batch inference
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        results = batch_inference(
            model, dataloader, device, class_names,
            save_results=True, output_dir=args.output_dir
        )
        
        # Visualize some predictions
        if args.visualize:
            images, labels, metadata = next(iter(dataloader))
            images = images.to(device)
            
            with torch.no_grad():
                logits, _ = model(images)
                _, predicted = torch.max(logits, 1)
            
            viz_path = os.path.join(args.output_dir, 'predictions_viz.png')
            visualize_predictions(
                images[:16], labels[:16], predicted[:16],
                class_names, num_images=16, save_path=viz_path
            )
            print(f"✓ Visualization saved to: {viz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['dinov2', 'mae', 'swin', 'vit', 'clip', 'vla'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'batch'],
                        help='Inference mode')
    
    # Single image mode arguments
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image (for single mode)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--class_names_file', type=str, default=None,
                        help='Path to JSON file with class names')
    
    # Batch mode arguments
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory (for batch mode)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of predictions')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'single' and not args.image_path:
        parser.error("--image_path required for single mode")
    if args.mode == 'batch' and not args.data_dir:
        parser.error("--data_dir required for batch mode")
    
    main(args)