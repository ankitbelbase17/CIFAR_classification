
MODE=${1:-single}
IMAGE_PATH=${2:-./test_image.jpg}
IMAGE_DIR=${3:-./test_images}
OUTPUT_FILE=${4:-./sd_results.json}
METHOD=${5:-clip}

CLASS_NAMES="cat dog bird car airplane"

echo "========================================"
echo "Stable Diffusion Zero-Shot Classification"
echo "========================================"
echo "Mode: ${MODE}"
echo "Method: ${METHOD}"
echo "Output: ${OUTPUT_FILE}"
echo "========================================"

if [ "${MODE}" == "single" ]; then
    if [ ! -f "${IMAGE_PATH}" ]; then
        echo "Error: Image file '${IMAGE_PATH}' does not exist"
        exit 1
    fi
    
    python stable_diffusion/sd_inference.py \
        --mode single \
        --image_path ${IMAGE_PATH} \
        --class_names ${CLASS_NAMES} \
        --method ${METHOD} \
        --output_file ${OUTPUT_FILE}

elif [ "${MODE}" == "batch" ]; then
    if [ ! -d "${IMAGE_DIR}" ]; then
        echo "Error: Image directory '${IMAGE_DIR}' does not exist"
        exit 1
    fi
    
    python stable_diffusion/sd_inference.py \
        --mode batch \
        --image_dir ${IMAGE_DIR} \
        --class_names ${CLASS_NAMES} \
        --method ${METHOD} \
        --output_file ${OUTPUT_FILE}

else
    echo "Error: Invalid mode '${MODE}'. Use 'single' or 'batch'"
    exit 1
fi

echo "Stable Diffusion inference completed!"
