
# Computer Vision Classification Project

A comprehensive deep learning project for image classification using DINOv2 and Swin Transformer V2 - two state-of-the-art vision transformers for high-performance image classification.

## Model Performance Comparison

| Metric | Swin | DINOv2 |
|--------|------|--------|
| Accuracy | 0.9644 | 0.9721 |
| Precision (Macro) | 0.99509 | 0.97893 |
| Recall (Macro) | 0.9812 | 0.9786 |
| F1 Score (Macro) | 0.9812 | 0.97862 |
| Matthews Correlation Coefficient | 0.97912 | 0.97625 |
| Cohen's Kappa | 0.97911 | 0.97622 |
| ROC-AUC (Macro) | 0.99936 | 0.99892 |

## Features

- **Two SOTA Vision Transformers**: DINOv2, Swin Transformer V2
- **Production Ready**: Complete training, inference, and evaluation pipelines
- **WandB Integration**: Real-time training monitoring and visualization
- **Checkpointing**: Auto-save every 250 iterations with best model tracking
- **Mixed Precision**: Faster training with automatic mixed precision
- **Comprehensive Testing**: Unit tests for all components
- **Visualization**: Attention maps and activation map visualization

## Installation

```bash
# Clone repository
git clone <this-repo>
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

# Train Swin Transformer V2
bash scripts/train.sh swin ./data ./experiments 32 50 1e-4
```

### 3. Run Inference

```bash
# Batch inference with DINOv2
bash scripts/inference.sh dinov2 ./experiments/checkpoints/dinov2_best.pth batch ./data

# Single image inference with Swin
bash scripts/inference.sh swin ./experiments/checkpoints/swin_best.pth single ./data ./image.jpg
```

### 4. Evaluate Metrics

```bash
bash scripts/metrics.sh dinov2 ./experiments/checkpoints/dinov2_best.pth ./data
bash scripts/metrics.sh swin ./experiments/checkpoints/swin_best.pth ./data
```

### 5. Visualize Attention Maps

```bash
python visualize_attention_activation.py \
    --model_name dinov2 \
    --image_path ./test_image.jpg \
    --checkpoint ./experiments/checkpoints/dinov2_best.pth \
    --output_dir ./visualizations

python visualize_attention_activation.py \
    --model_name swin \
    --image_path ./test_image.jpg \
    --checkpoint ./experiments/checkpoints/swin_best.pth \
    --output_dir ./visualizations
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

DINOv2 is a self-supervised vision transformer developed by Meta that provides exceptional transfer learning capabilities for image classification tasks. It learns rich visual representations through self-supervised learning without requiring labeled data for pre-training.

**Key Characteristics:**
- Self-supervised pre-training using contrastive learning
- Excellent zero-shot and few-shot capabilities
- Strong feature representations for downstream tasks
- ~86M parameters
- Uses CLS token for classification
- Architecture: Vision Transformer base model

**Performance:**
- Achieves 97.21% accuracy on the benchmark dataset
- Macro F1 Score: 0.97862
- ROC-AUC: 0.99892
- Strong performance across all metrics

### 2. Swin Transformer V2 (Microsoft)

Swin Transformer V2 is a hierarchical vision transformer that processes images at multiple scales, making it efficient for various input image sizes and computational budgets. The hierarchical design with shifted windows provides a good balance between accuracy and efficiency.

**Key Characteristics:**
- Hierarchical multi-scale architecture
- Shifted window-based attention mechanism
- Efficient computation with linear complexity relative to input size
- ~88M parameters
- Uses mean pooling over patches for classification
- Architecture: Hierarchical transformer with 4 stages

**Performance:**
- Achieves 96.44% accuracy on the benchmark dataset
- Macro F1 Score: 0.9812
- ROC-AUC: 0.99936
- Highest precision (0.99509) among the two models
- Best ROC-AUC performance

## Testing

```bash
# Run model tests
python tests/test_model.py

# Run training tests
python tests/test_train.py

# Run inference tests
python tests/test_inference.py

# Run specific test
python -m pytest tests/test_model.py::TestModels::test_dinov2_forward
python -m pytest tests/test_model.py::TestModels::test_swin_forward
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture (dinov2 or swin)
- Training hyperparameters
- Data augmentation
- Logging frequency
- Class names

## Model Comparison

### DINOv2 vs Swin Transformer V2

**DINOv2 Advantages:**
- Higher overall accuracy (97.21% vs 96.44%)
- Better macro recall performance
- Self-supervised pre-training provides strong representations
- Excellent for transfer learning tasks
- Better generalization on unseen classes

**Swin Transformer V2 Advantages:**
- Superior precision metrics (0.99509)
- Highest ROC-AUC score (0.99936)
- More efficient with hierarchical architecture
- Better for scenarios where false positives are critical
- Shifted window attention reduces computational cost

### When to Use Which Model

**Choose DINOv2 when:**
- Maximizing overall accuracy is the priority
- Transfer learning on downstream tasks is required
- Working with limited labeled data
- Feature extraction quality is important

**Choose Swin Transformer V2 when:**
- Minimizing false positives is critical
- Computational efficiency is important
- Working with various image resolutions
- High precision predictions are needed

## WandB Logging

The project logs:
- Training/validation loss and accuracy
- Learning rate
- Prediction visualizations
- Confusion matrices
- Per-class metrics
- Training curves

## Implementation Details

### DINOv2 Implementation
The DINOv2 classifier uses the base model from Meta with:
- Classification head: LayerNorm → Linear (768→384) → GELU → Dropout(0.1) → Linear (384→num_classes)
- Xavier initialization for all linear layers
- CLS token extraction for classification
- Layer normalization before the classification layers

### Swin Transformer V2 Implementation
The Swin Transformer V2 classifier uses the base model from Microsoft with:
- Classification head: LayerNorm → Linear (768→384) → GELU → Dropout(0.1) → Linear (384→num_classes)
- Xavier initialization for all linear layers
- Mean pooling over all patches for classification
- Layer normalization before the classification layers

Both models support:
- Backbone freezing for faster training or feature extraction
- Output of attention maps and hidden states for visualization
- Mixed precision training with automatic mixed precision (AMP)

## Citation

If you use this codebase, please cite the respective model papers:
- DINOv2: https://arxiv.org/abs/2304.07193
- Swin Transformer V2: https://arxiv.org/abs/2111.06377

## License

MIT License
