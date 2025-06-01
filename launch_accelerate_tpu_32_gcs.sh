#!/bin/bash
# Launch VoRA training on 32 TPU cores using Accelerate with GCS support

set -e

# Default configuration files
ACCELERATE_CONFIG="configs/accelerate_tpu_32.yaml"
TRAINING_CONFIG="configs/pretrain_tpu_32_gcs.yaml"
DEBUG=false
export PJRT_DEVICE="TPU"
export TPU_NAME="ye-tpu-vora"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --config FILE         Training configuration file (default: $TRAINING_CONFIG)"
    echo "  -a, --accelerate FILE     Accelerate configuration file (default: $ACCELERATE_CONFIG)"
    echo "  -d, --debug              Enable debug mode with verbose logging"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Use default configs"
    echo "  $0 -c configs/my_config.yaml               # Custom training config"
    echo "  $0 -c my_config.yaml -a my_accelerate.yaml # Custom both configs"
    echo "  $0 --config configs/debug.yaml --debug     # Debug mode"
    echo ""
    echo "GCS Setup:"
    echo "  1. pip install -r requirements_gcs.txt"
    echo "  2. gcloud auth application-default login"
    echo "  3. Edit your config file with GCS bucket paths"
    echo "  4. Run this script"
    echo ""
    echo "Config File Requirements:"
    echo "  Training config should contain GCS paths like:"
    echo "    output_dir: gs://your-bucket/output/"
    echo "    model.aux_vision: gs://your-bucket/models/image-encoder.pt"
    echo "    data.train.data_fetch.data_paths[].anno_path: gs://your-bucket/data.jsonl"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        -a|--accelerate)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "[INFO] VoRA TPU-32 Training with Accelerate + GCS"
echo "==============================================="
echo "[INFO] Training config: $TRAINING_CONFIG"
echo "[INFO] Accelerate config: $ACCELERATE_CONFIG"
if [[ "$DEBUG" == "true" ]]; then
    echo "[INFO] Debug mode: ENABLED"
fi
echo ""

# Set environment variables
export XLA_USE_BF16=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if [[ "$DEBUG" == "true" ]]; then
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export CUDA_LAUNCH_BLOCKING=1
    echo "[DEBUG] Environment variables set for debugging"
fi

# Ensure Google Cloud credentials are available
if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]] && [[ -z "$GCLOUD_PROJECT" ]]; then
    echo "[WARNING] No Google Cloud credentials found."
    echo "[INFO] Make sure to authenticate with: gcloud auth application-default login"
    echo "[INFO] Or set GOOGLE_APPLICATION_CREDENTIALS environment variable"
fi

# Check if gsutil is available for better performance
if ! command -v gsutil &> /dev/null; then
    echo "[WARNING] gsutil not found. Install Google Cloud SDK for better performance."
    echo "[INFO] Falling back to Python client for GCS operations."
else
    echo "[INFO] gsutil found - will use for optimal GCS performance"
fi

# Validate configuration files
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    echo "[ERROR] Accelerate config not found: $ACCELERATE_CONFIG"
    echo "[INFO] Available configs:"
    ls -la configs/ | grep ".yaml$" || echo "No YAML configs found"
    exit 1
fi

if [[ ! -f "$TRAINING_CONFIG" ]]; then
    echo "[ERROR] Training config not found: $TRAINING_CONFIG"
    echo "[INFO] Available configs:"
    ls -la configs/ | grep ".yaml$" || echo "No YAML configs found"
    exit 1
fi

echo "[SUCCESS] Configuration files validated"

# Check if training script exists
TRAINING_SCRIPT="train/train_tpu_32_accelerate_gcs.py"
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "[ERROR] GCS training script not found: $TRAINING_SCRIPT"
    echo "[INFO] Available training scripts:"
    ls -la train/*.py 2>/dev/null || echo "No training scripts found"
    exit 1
fi

echo "[SUCCESS] Training script found: $TRAINING_SCRIPT"
echo ""

# Show what will happen
echo "[INFO] Training will:"
echo "       • Use accelerate config: $ACCELERATE_CONFIG"
echo "       • Use training config: $TRAINING_CONFIG (passed as argument to Python script)"
echo "       • Command: accelerate launch --config_file $ACCELERATE_CONFIG $TRAINING_SCRIPT $TRAINING_CONFIG"
echo "       • Automatically download GCS datasets and models"
echo "       • Cache data locally for performance"
echo "       • Upload checkpoints to GCS buckets"
echo "       • Upload final model to GCS bucket"
echo "       • Clean up temporary files"
echo ""

if [[ "$DEBUG" == "true" ]]; then
    echo "[DEBUG] Accelerate config contents:"
    cat "$ACCELERATE_CONFIG" | head -10
    echo "..."
    echo ""
    echo "[DEBUG] Training config (first 20 lines):"
    head -20 "$TRAINING_CONFIG"
    echo "..."
    echo ""
fi

echo "[INFO] Launching training with Accelerate + GCS..."

# Build the command
CMD=("accelerate" "launch" "--config_file" "$ACCELERATE_CONFIG" "$TRAINING_SCRIPT" "$TRAINING_CONFIG")

if [[ "$DEBUG" == "true" ]]; then
    echo "[DEBUG] Full command: ${CMD[*]}"
    echo "[DEBUG] Working directory: $(pwd)"
    echo "[DEBUG] Python path: $PYTHONPATH"
    echo ""
fi

# Execute the training command
if "${CMD[@]}"; then
    echo ""
    echo "[SUCCESS] GCS-enabled training completed successfully!"
    echo "[INFO] Check your GCS bucket for saved models and checkpoints"
    echo "[INFO] Training config used: $TRAINING_CONFIG"
else
    exit_code=$?
    echo ""
    echo "[ERROR] Training failed with exit code: $exit_code"
    echo "[INFO] Check the logs above for error details"
    echo "[INFO] Common issues:"
    echo "       • GCS authentication problems"
    echo "       • Bucket permissions or non-existent buckets"
    echo "       • Invalid configuration file"
    echo "       • TPU resource allocation issues"
    exit $exit_code
fi
