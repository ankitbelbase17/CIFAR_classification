
# Computer Vision Classification Project

A comprehensive deep learning project for image classification using state-of-the-art models.

## Features

- **Multiple SOTA Models**: DINOv2, MAE, Swin Transformer V2, ViT, CLIP, OpenVLA
- **Stable Diffusion**: Zero-shot classification (inference-only)
- **Production Ready**: Complete training, inference, and evaluation pipelines
- **WandB Integration**: Real-time training monitoring and visualization
- **Checkpointing**: Auto-save every 250 iterations with best model tracking
- **Mixed Precision**: Faster training with automatic mixed precision
- **Comprehensive Testing**: Unit tests for all components
- **Visualization**: Attention maps and activation map visualization

## Installation

```bash
# Clone repository
git clone <your-repo>
cd cv_classification_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to WandB (optional but recommended)
wandb login
```

## Project Structure

```
cv_classification_project/
├── model.py                              # All model architectures
├── dataloader.py                         # Dataset and DataLoader
├── train.py                              # Training script
├── inference.py                          # Inference script
├── metrics.py                            # Evaluation metrics
├── utils.py                              # Utility functions
├── visualize_attention_activation.py     # Visualization tools
├── config.yaml                           # Configuration file
├── requirements.txt                      # Python dependencies
├── scripts/
│   ├── train.sh                          # Training shell script
│   ├── inference.sh                      # Inference shell script
│   └── metrics.sh                        # Metrics shell script
├── tests/
│   ├── test_model.py                     # Model tests
│   ├── test_train.py                     # Training tests
│   └── test_inference.py                 # Inference tests
└── stable_diffusion/
    ├── sd_classifier.py                  # SD classifier
    ├── sd_inference.py                   # SD inference script
    └── sd_inference.sh                   # SD shell script
```

## Quick Start

### 1. Prepare Dataset

Organize your dataset in ImageFolder format:
```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### 2. Train a Model

```bash
# Train DINOv2
bash scripts/train.sh dinov2 ./data ./experiments 32 50 1e-4

# Train with different models
bash scripts/train.sh vit ./data
bash scripts/train.sh swin ./data
bash scripts/train.sh clip ./data
```

### 3. Run Inference

```bash
# Batch inference
bash scripts/inference.sh dinov2 ./experiments/checkpoints/dinov2_best.pth batch ./data

# Single image inference
bash scripts/inference.sh vit ./experiments/checkpoints/vit_best.pth single ./data ./image.jpg
```

### 4. Evaluate Metrics

```bash
bash scripts/metrics.sh dinov2 ./experiments/checkpoints/dinov2_best.pth ./data
```

### 5. Visualize Attention Maps

```bash
python visualize_attention_activation.py \
    --model_name dinov2 \
    --image_path ./test_image.jpg \
    --checkpoint ./experiments/checkpoints/dinov2_best.pth \
    --output_dir ./visualizations
```

### 6. Stable Diffusion Classification

```bash
# Zero-shot classification (no training needed!)
bash stable_diffusion/sd_inference.sh single ./test_image.jpg "" ./sd_results.json clip
```

## Training Features

- **Automatic Checkpointing**: Saves every 250 iterations
- **WandB Logging**: Real-time metrics and prediction visualizations
- **Resume Training**: Automatically loads latest checkpoint
- **Mixed Precision**: Faster training with AMP
- **Xavier Initialization**: Proper weight initialization
- **Gradient Clipping**: Stable training
- **Learning Rate Scheduling**: Cosine annealing
- **Data Augmentation**: Advanced augmentations for better generalization

## Models

### 1. DINOv2 (Meta)
- Self-supervised vision transformer
- Excellent for transfer learning
- ~86M parameters

### 2. Masked Autoencoder (MAE)
- Self-supervised pre-training
- Strong feature representations
- ~86M parameters

### 3. Swin Transformer V2
- Hierarchical vision transformer
- Efficient for various image sizes
- ~88M parameters

### 4. Vision Transformer (ViT)
- Original transformer for images
- Strong performance on many tasks
- ~86M parameters

### 5. CLIP (OpenAI)
- Vision-language model
- Zero-shot capabilities
- ~149M parameters

### 6. OpenVLA
- Vision-Language-Action model
- ~200M parameters
- Inspired by robotics applications

### 7. Stable Diffusion (Inference-Only)
- Zero-shot classification
- No training required
- Uses CLIP and diffusion features

## Testing

```bash
# Run all tests
python tests/test_model.py
python tests/test_train.py
python tests/test_inference.py

# Run specific test
python -m pytest tests/test_model.py::TestModels::test_dinov2_forward
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation
- Logging frequency
- Class names

## WandB Logging

The project logs:
- Training/validation loss and accuracy
- Learning rate
- Prediction visualizations
- Confusion matrices
- Per-class metrics
- Training curves

## Tips

1. **Start with frozen backbone** for faster initial training:
   ```bash
   python train.py --model_name dinov2 --freeze_backbone ...
   ```

2. **Use smaller batch size** if running out of memory:
   ```bash
   bash scripts/train.sh dinov2 ./data ./experiments 16
   ```

3. **Monitor training** with WandB:
   - Visit https://wandb.ai after training starts

4. **Try ensemble methods** for better accuracy:
   - Train multiple models and average predictions

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use smaller model
- **Slow training**: Enable mixed precision, increase num_workers
- **Poor accuracy**: Check data quality, try data augmentation, increase epochs

## Citation

If you use this codebase, please cite the respective model papers:
- DINOv2: https://arxiv.org/abs/2304.07193
- MAE: https://arxiv.org/abs/2111.06377
- Swin Transformer: https://arxiv.org/abs/2103.14030
- ViT: https://arxiv.org/abs/2010.11929
- CLIP: https://arxiv.org/abs/2103.00020

## License

MIT License
