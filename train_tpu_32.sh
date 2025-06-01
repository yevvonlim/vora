#!/bin/bash
# VoRA TPU-32 Training Script

set -e

CONFIG_FILE="configs/pretrain_tpu_32_cores.yaml"

echo "[INFO] VoRA TPU-32 Training Setup"
echo "======================================"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "[SUCCESS] Configuration file found: $CONFIG_FILE"

# Set TPU environment variables
export XLA_USE_BF16=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
export XLA_USE_SPMD=1
export TPU_NUM_DEVICES=32
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

echo "[INFO] Environment configured for 32 TPU cores"

if [[ ! -f "train/train_tpu_32.py" ]]; then
    echo "[ERROR] Training script not found: train/train_tpu_32.py"
    exit 1
fi

echo "[INFO] Starting training with 32 TPU cores..."
python3 train/train_tpu_32.py

echo "[SUCCESS] Training script completed!"
