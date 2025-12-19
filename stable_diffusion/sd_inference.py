
# ==============================================================================
# stable_diffusion/sd_inference.py - Inference script for SD classifier
# ==============================================================================

"""
Inference script for Stable Diffusion classifier
"""

import argparse
import json
import os
from PIL import Image
from tqdm import tqdm


def inference_single_image(
    classifier: StableDiffusionClassifier,
    image_path: str,
    class_names: List[str],
    method: str = "clip"
) -> Dict:
    """Run inference on single image"""
    image = Image.open(image_path).convert('RGB')
    results = classifier.classify(image, class_names, method=method)
    
    print(f"\nImage: {image_path}")
    print(f"Method: {results['method']}")
    print("\nTop predictions:")
    for i, pred in enumerate(results['predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['probability']:.4f}")
    
    return results


def inference_batch(
    classifier: StableDiffusionClassifier,
    image_dir: str,
    class_names: List[str],
    method: str = "clip",
    output_file: str = None
) -> List[Dict]:
    """Run inference on batch of images"""
    results_list = []
    
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            results = classifier.classify(image, class_names, method=method)
            
            results_list.append({
                'image_path': image_path,
                'predictions': results['predictions'],
                'method': results['method']
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(
        description='Stable Diffusion Zero-Shot Classification'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'batch'],
                        help='Inference mode')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory with images for batch mode')
    parser.add_argument('--class_names', type=str, nargs='+', required=True,
                        help='List of class names')
    parser.add_argument('--method', type=str, default='clip',
                        choices=['clip', 'reconstruction', 'ensemble'],
                        help='Classification method')
    parser.add_argument('--model_id', type=str,
                        default='stabilityai/stable-diffusion-2-1-base',
                        help='Stable Diffusion model ID')
    parser.add_argument('--output_file', type=str,
                        default='sd_inference_results.json',
                        help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'single' and not args.image_path:
        parser.error("--image_path required for single mode")
    if args.mode == 'batch' and not args.image_dir:
        parser.error("--image_dir required for batch mode")
    
    # Initialize classifier
    print("Initializing Stable Diffusion classifier...")
    classifier = StableDiffusionClassifier(
        model_id=args.model_id,
        device=args.device
    )
    
    # Run inference
    if args.mode == 'single':
        results = inference_single_image(
            classifier, args.image_path, args.class_names, args.method
        )
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    else:
        results = inference_batch(
            classifier, args.image_dir, args.class_names,
            args.method, args.output_file
        )


if __name__ == "__main__":
    main()