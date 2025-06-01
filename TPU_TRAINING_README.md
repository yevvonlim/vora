# VoRA TPU Training with 32 Cores

This directory contains optimized training scripts for running VoRA on 32 TPU cores.

## Files

- `train/train_tpu_32.py` - Main TPU training script optimized for 32 cores
- `train/launch_tpu_32.py` - Launcher script with environment setup
- `configs/pretrain_tpu_32_cores.yaml` - Configuration optimized for 32 TPU cores
- `train_tpu_32.sh` - Simple bash script to launch training

## Quick Start

### Setup Environment

```bash
export TPU_NAME=your-tpu-name
export TPU_ZONE=us-central1-f
export PROJECT_ID=your-project-id
```

### Run Training

Simplest way:
```bash
./train_tpu_32.sh
```

With launcher:
```bash
python train/launch_tpu_32.py --config configs/pretrain_tpu_32_cores.yaml
```

Direct:
```bash
python train/train_tpu_32.py
```

## Key Features

- Optimized for 32 TPU cores
- BF16 training through XLA
- SPMD parallelization
- Async checkpointing
- TPU metrics monitoring
- Automatic batch size optimization

## Configuration

The config file sets:
- Batch size: 4 per device (128 total)
- Gradient accumulation: 8 steps (effective batch size 1024)
- Learning rate: 0.0002 with warmup
- Checkpointing every 2000 steps

## Troubleshooting

Check environment:
```bash
python train/launch_tpu_32.py --check-only
```

Check TPU status:
```bash
gcloud compute tpus list --zone=$TPU_ZONE
```

Test XLA:
```bash
python -c "import torch_xla; print('Success!')"
```
