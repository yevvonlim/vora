# VoRA TPU Training with GCS Bucket Support

This guide explains how to set up and run VoRA training on 32 TPU cores with Google Cloud Storage (GCS) bucket support for datasets, model files, logging, and checkpoints.

## Overview

The GCS-enabled training system automatically:
- ⬇️ **Downloads** datasets and model files from GCS buckets to local storage
- 💾 **Caches** data locally during training for performance
- ⬆️ **Uploads** checkpoints and final models to GCS buckets
- 🧹 **Cleans up** temporary files after training

## Quick Start

### 1. Prerequisites

**Install GCS dependencies:**
```bash
pip install -r requirements_gcs.txt
```

**Authenticate with Google Cloud:**
```bash
# Option 1: User account (for development)
gcloud auth application-default login

# Option 2: Service account (for production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

**Install Google Cloud SDK (recommended for performance):**
```bash
# Ubuntu/Debian
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 2. Setup Your GCS Buckets

Create buckets for different purposes:
```bash
# Data bucket
gsutil mb gs://your-data-bucket

# Model bucket  
gsutil mb gs://your-model-bucket

# Training output bucket
gsutil mb gs://your-training-bucket
```

### 3. Upload Your Data

**Upload dataset:**
```bash
# Upload annotations
gsutil cp your_annotations.jsonl gs://your-data-bucket/datasets/annotations/

# Upload images directory
gsutil -m cp -r your_images/ gs://your-data-bucket/datasets/images/
```

**Upload model files:**
```bash
# Upload aux vision model
gsutil cp image-encoder.pt gs://your-model-bucket/aux-vision/

# Upload HuggingFace transform model directory
gsutil -m cp -r hf-transform-model/ gs://your-model-bucket/hf-image-transform/
```

### 4. Configure Training

Edit `configs/pretrain_tpu_32_gcs.yaml`:
```yaml
# Update these paths with your bucket names
output_dir: gs://your-training-bucket/output/vora-tpu32-run

model:
  aux_vision: "gs://your-model-bucket/aux-vision/image-encoder.pt"
  # Optional: pretrained model
  # pretrained: "gs://your-model-bucket/pretrained/checkpoint-1000"

data:
  train:
    data_fetch:
      data_paths: [
        {
          "anno_path": "gs://your-data-bucket/datasets/annotations/data_utf8.jsonl",
          "image_folder": "gs://your-data-bucket/datasets/images"
        }
      ]
    data_preprocess:
      frames_ops:
        HFImageTransform:
          path: gs://your-model-bucket/hf-image-transform
```

### 5. Launch Training

```bash
./launch_accelerate_tpu_32_gcs.sh
```

## File Structure

```
├── train/
│   └── train_tpu_32_accelerate_gcs.py  # GCS-enabled training script
├── configs/
│   ├── accelerate_tpu_32.yaml          # Accelerate config
│   └── pretrain_tpu_32_gcs.yaml        # GCS training config
├── launch_accelerate_tpu_32_gcs.sh     # GCS launch script
├── requirements_gcs.txt                # GCS dependencies
└── GCS_TRAINING_SETUP.md              # This guide
```

## GCS Path Support

The training system supports GCS paths for:

### 📋 **Dataset Paths**
- `data.train.data_fetch.data_paths[].anno_path`
- `data.train.data_fetch.data_paths[].image_folder`

### 🤖 **Model Paths**
- `model.aux_vision` - Auxiliary vision model file
- `model.pretrained` - Pretrained model checkpoint directory
- `data.train.data_preprocess.frames_ops.HFImageTransform.path` - HuggingFace image transform

### 💾 **Output Paths**
- `output_dir` - Training output directory for checkpoints and final model

## How GCS Integration Works

### 📀 **Download Phase (Training Start)**
1. Main process detects GCS paths in configuration
2. Creates temporary directory: `/tmp/vora_gcs_XXXXXX`
3. Downloads all GCS resources to local storage
4. Updates configuration to use local paths
5. Other TPU cores wait for download completion

### 🚀 **Training Phase**
- Training runs normally using local cached files
- Faster I/O performance compared to direct GCS access
- No network dependencies during training

### ⬆️ **Save Phase (Checkpoints & Final Model)**
1. Model saved to local temporary directory first
2. Main process uploads to GCS bucket
3. Local files cleaned up to save disk space
4. Other TPU cores wait for upload completion

### 🧹 **Cleanup Phase (Training End)**
- All temporary directories removed
- Only final model remains in GCS bucket

## Performance Considerations

### ⚡ **Optimization Tips**

1. **Use gsutil for large files:** Install Google Cloud SDK for 10x faster transfers
2. **Regional co-location:** Place TPU and GCS bucket in same region
3. **Parallel uploads:** Large checkpoints uploaded in parallel chunks
4. **Local caching:** Data downloaded once, used throughout training

### 📊 **Expected Performance**

| Operation | With gsutil | Python client only |
|-----------|-------------|--------------------|
| **Dataset download** | 2-5 min | 10-20 min |
| **Model download** | 30 sec | 2-3 min |
| **Checkpoint save** | 30 sec | 1-2 min |
| **Training I/O** | Local speed | Local speed |

## Monitoring and Logging

### 📜 **Log Messages**
The training script provides detailed GCS operation logs:
```
[INFO] Setting up GCS paths in temporary directory: /tmp/vora_gcs_abc123
[INFO] Downloading gs://bucket/data.jsonl to /tmp/vora_gcs_abc123/annotations_0.jsonl
[INFO] Successfully downloaded gs://bucket/data.jsonl
[INFO] Uploading /tmp/model_save_xyz to gs://bucket/output/checkpoint-1000
[INFO] Successfully uploaded to gs://bucket/output/checkpoint-1000
```

### 📉 **Weights & Biases Integration**
W&B logging works normally with GCS:
- Metrics logged in real-time
- Model artifacts can be saved to W&B and/or GCS
- Training curves and metrics tracked as usual

## Troubleshooting

### ⚠️ **Common Issues**

#### **Authentication Error**
```
PermissionDenied: 403 Access denied
```
**Solution:**
```bash
gcloud auth application-default login
# or
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

#### **Bucket Not Found**
```
NotFound: 404 Bucket does not exist
```
**Solution:**
```bash
# Create bucket
gsutil mb gs://your-bucket-name

# Or check bucket name in config
```

#### **gsutil Not Found**
```
gsutil failed, using Python client
```
**Solution:** Install Google Cloud SDK (recommended) or ignore (fallback works)

#### **Slow Downloads**
- Ensure TPU and GCS bucket are in same region
- Install Google Cloud SDK for gsutil
- Check network connectivity

#### **Out of Disk Space**
```
OSError: [Errno 28] No space left on device
```
**Solution:**
- Use smaller datasets for testing
- Increase TPU VM disk size
- Check temporary directory cleanup

### 🔍 **Debug Commands**

```bash
# Test GCS access
gsutil ls gs://your-bucket/

# Check file sizes
gsutil du -s gs://your-bucket/datasets/

# Test authentication
gcloud auth application-default print-access-token

# Monitor disk space
df -h

# Check temporary directories
ls -la /tmp/vora_gcs_*
```

## Advanced Configuration

### 🗺️ **Multi-Region Setup**
For multi-region training:
```yaml
# Use regional buckets
output_dir: gs://bucket-us-central1/output/
data:
  train:
    data_fetch:
      data_paths:
        - anno_path: gs://data-us-central1/annotations.jsonl
          image_folder: gs://data-us-central1/images/
```

### 🔐 **Security Best Practices**

1. **Use service accounts** for production
2. **Limit bucket permissions** (read-only for data, write for output)
3. **Enable bucket versioning** for important data
4. **Use IAM conditions** for time-limited access

Example service account permissions:
```json
{
  "bindings": [
    {
      "role": "roles/storage.objectViewer",
      "members": ["serviceAccount:training@project.iam.gserviceaccount.com"],
      "condition": {
        "title": "Data bucket read",
        "expression": "resource.name.startsWith('projects/_/buckets/your-data-bucket/')"
      }
    },
    {
      "role": "roles/storage.objectCreator",
      "members": ["serviceAccount:training@project.iam.gserviceaccount.com"],
      "condition": {
        "title": "Output bucket write", 
        "expression": "resource.name.startsWith('projects/_/buckets/your-training-bucket/')"
      }
    }
  ]
}
```

### 👥 **Team Collaboration**

For team training workflows:
```yaml
# Use shared buckets with organized structure
output_dir: gs://team-training-bucket/experiments/{user}/{timestamp}/
data:
  train:
    data_fetch:
      data_paths:
        - anno_path: gs://team-data-bucket/datasets/v2/annotations.jsonl
          image_folder: gs://team-data-bucket/datasets/v2/images/
```

## Cost Optimization

### 💰 **Storage Costs**
- Use **Standard** storage for active training data
- Move old experiments to **Coldline** or **Archive** storage
- Enable **lifecycle policies** for automatic transitions

### 🔄 **Transfer Costs**
- Keep data and compute in **same region**
- Use **gsutil** for efficient transfers
- Consider **preemptible TPUs** for cost savings

Example lifecycle policy:
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 30, "matchesPrefix": ["experiments/"]}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 365, "matchesPrefix": ["temp/"]}
      }
    ]
  }
}
```

## Next Steps

1. **Start with small datasets** to test the setup
2. **Monitor first training run** for any issues
3. **Scale up** to full datasets once validated
4. **Set up monitoring** and alerting for production use
5. **Implement backup strategies** for important models

For additional help, check the main TPU training documentation or reach out to the team!
