
# ==============================================================================
# metrics.sh - Metrics evaluation script
# Usage: bash scripts/metrics.sh <model_name> <checkpoint_path>
# Example: bash scripts/metrics.sh vit ./experiments/vit_best.pth
# ==============================================================================

#!/bin/bash

MODEL_NAME=${1:-dinov2}
CHECKPOINT=${2:-./experiments/checkpoints/${MODEL_NAME}_best.pth}
DATA_DIR=${3:-./data}
OUTPUT_DIR=${4:-./metrics_results}
BATCH_SIZE=${5:-32}

echo "========================================"
echo "Evaluating Metrics - ${MODEL_NAME} Model"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint '${CHECKPOINT}' does not exist"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory '${DATA_DIR}' does not exist"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run metrics evaluation
python -c "
import torch
import json
import os
from model import get_model
from dataloader import get_dataloaders
from utils import load_checkpoint, get_device
from metrics import (
    calculate_metrics, plot_confusion_matrix,
    generate_classification_report, plot_per_class_metrics,
    save_metrics_json, generate_metrics_summary
)
from tqdm import tqdm

# Setup
device = get_device()
model_name = '${MODEL_NAME}'
checkpoint_path = '${CHECKPOINT}'
data_dir = '${DATA_DIR}'
output_dir = '${OUTPUT_DIR}'

# Load data
print('Loading data...')
_, _, test_loader, class_names = get_dataloaders(
    root_dir=data_dir,
    model_name=model_name,
    batch_size=${BATCH_SIZE},
    num_workers=4
)

# Load model
print(f'Loading {model_name} model...')
model = get_model(model_name, num_classes=len(class_names), class_names=class_names)
model = model.to(device)
load_checkpoint(checkpoint_path, model, device=device)
model.eval()

# Collect predictions
print('Running inference on test set...')
all_predictions = []
all_labels = []
all_logits = []

with torch.no_grad():
    for images, labels, _ in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        logits, _ = model(images)
        _, predicted = torch.max(logits, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_logits.append(logits.cpu())

all_logits = torch.cat(all_logits, dim=0)

# Calculate metrics
print('Calculating metrics...')
metrics = calculate_metrics(all_labels, all_predictions, all_logits, class_names)

# Generate visualizations
print('Generating visualizations...')
plot_confusion_matrix(
    all_labels, all_predictions, class_names,
    os.path.join(output_dir, 'confusion_matrix.png'),
    normalize=True
)

plot_confusion_matrix(
    all_labels, all_predictions, class_names,
    os.path.join(output_dir, 'confusion_matrix_counts.png'),
    normalize=False
)

generate_classification_report(
    all_labels, all_predictions, class_names,
    os.path.join(output_dir, 'classification_report.txt')
)

plot_per_class_metrics(
    metrics, class_names,
    os.path.join(output_dir, 'per_class_metrics.png')
)

save_metrics_json(metrics, os.path.join(output_dir, 'metrics.json'))

# Print summary
df_overall, df_per_class = generate_metrics_summary(metrics, class_names)

print(f'\n✓ All metrics saved to: {output_dir}')
"

echo "Metrics evaluation completed for ${MODEL_NAME}"
