# ==============================================================================
# train.sh - Training script with model selection
# Usage: bash scripts/train.sh <model_name>
# Example: bash scripts/train.sh dinov2
# ==============================================================================

MODEL_NAME=${1:-dinov2}
DATA_DIR=${2:-./data}
OUTPUT_DIR=${3:-./experiments}
BATCH_SIZE=${4:-32}
EPOCHS=${5:-50}
LR=${6:-1e-4}
NUM_WORKERS=${7:-4}

echo "========================================"
echo "Training ${MODEL_NAME} Model"
echo "========================================"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LR}"
echo "========================================"

# Check if model name is valid
valid_models=("dinov2" "mae" "swin" "vit" "clip" "vla")
if [[ ! " ${valid_models[@]} " =~ " ${MODEL_NAME} " ]]; then
    echo "Error: Invalid model name '${MODEL_NAME}'"
    echo "Valid models: ${valid_models[@]}"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory '${DATA_DIR}' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run training
python train.py \
    --model_name ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --resume

echo "Training completed for ${MODEL_NAME}"
