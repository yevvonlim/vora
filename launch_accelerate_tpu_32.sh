#!/bin/bash
# Launch VoRA training on 32 TPU cores using Accelerate

set -e

# Default configuration files
ACCELERATE_CONFIG="configs/accelerate_tpu_32.yaml"
TRAINING_CONFIG="configs/pretrain_tpu_32_cores.yaml"
DEBUG=false

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
    echo "Note: This script is for local/regular training."
    echo "For GCS bucket support, use: launch_accelerate_tpu_32_gcs.sh"
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

echo "[INFO] VoRA TPU-32 Training with Accelerate"
echo "========================================="
echo "[INFO] Training config: $TRAINING_CONFIG"
echo "[INFO] Accelerate config: $ACCELERATE_CONFIG"
if [[ "$DEBUG" == "true" ]]; then
    echo "[INFO] Debug mode: ENABLED"
fi
echo ""

# Set TPU environment variables
export XLA_USE_BF16=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if [[ "$DEBUG" == "true" ]]; then
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export CUDA_LAUNCH_BLOCKING=1
    echo "[DEBUG] Environment variables set for debugging"
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
TRAINING_SCRIPT="train/train_tpu_32_accelerate.py"
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "[ERROR] Training script not found: $TRAINING_SCRIPT"
    echo "[INFO] Available training scripts:"
    ls -la train/*.py 2>/dev/null || echo "No training scripts found"
    exit 1
fi

echo "[SUCCESS] Training script found: $TRAINING_SCRIPT"
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

echo "[INFO] Launching training with Accelerate..."

# Build the command
CMD=("accelerate" "launch" "--config_file" "$ACCELERATE_CONFIG" "$TRAINING_SCRIPT" "$TRAINING_CONFIG")

if [[ "$DEBUG" == "true" ]]; then
    echo "[DEBUG] Full command: ${CMD[*]}"
    echo "[DEBUG] Working directory: $(pwd)"
    echo ""
fi

# Execute the training command
if "${CMD[@]}"; then
    echo ""
    echo "[SUCCESS] Training completed successfully!"
    echo "[INFO] Training config used: $TRAINING_CONFIG"
else
    exit_code=$?
    echo ""
    echo "[ERROR] Training failed with exit code: $exit_code"
    echo "[INFO] Check the logs above for error details"
    exit $exit_code
fi
