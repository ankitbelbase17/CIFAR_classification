
# ==============================================================================
# inference.sh - Inference script with model selection
# Usage: bash scripts/inference.sh <model_name> <checkpoint_path> <mode>
# Example: bash scripts/inference.sh dinov2 ./experiments/dinov2_best.pth batch
# ==============================================================================

#!/bin/bash

MODEL_NAME=${1:-dinov2}
CHECKPOINT=${2:-./experiments/checkpoints/${MODEL_NAME}_best.pth}
MODE=${3:-batch}
DATA_DIR=${4:-./data}
IMAGE_PATH=${5:-./test_image.jpg}
OUTPUT_DIR=${6:-./inference_results}
BATCH_SIZE=${7:-32}

echo "========================================"
echo "Running Inference - ${MODEL_NAME} Model"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Mode: ${MODE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint '${CHECKPOINT}' does not exist"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

if [ "${MODE}" == "batch" ]; then
    # Batch inference
    if [ ! -d "${DATA_DIR}" ]; then
        echo "Error: Data directory '${DATA_DIR}' does not exist"
        exit 1
    fi
    
    python inference.py \
        --model_name ${MODEL_NAME} \
        --checkpoint ${CHECKPOINT} \
        --mode batch \
        --data_dir ${DATA_DIR} \
        --split test \
        --batch_size ${BATCH_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --visualize
        
elif [ "${MODE}" == "single" ]; then
    # Single image inference
    if [ ! -f "${IMAGE_PATH}" ]; then
        echo "Error: Image '${IMAGE_PATH}' does not exist"
        exit 1
    fi
    
    python inference.py \
        --model_name ${MODEL_NAME} \
        --checkpoint ${CHECKPOINT} \
        --mode single \
        --image_path ${IMAGE_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --top_k 5
else
    echo "Error: Invalid mode '${MODE}'. Use 'batch' or 'single'"
    exit 1
fi

echo "Inference completed for ${MODEL_NAME}"
