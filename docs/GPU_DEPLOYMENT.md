# GPU-Enabled Cloud Run Deployment Guide

This guide explains how to deploy the Mannequin Segmenter microservice to Google Cloud Run with GPU acceleration for optimal EVF-SAM model performance.

## 🚀 Quick Start

Run the automated setup script:

```bash
chmod +x scripts/setup-gcp.sh
./scripts/setup-gcp.sh
```

## 📋 Prerequisites

- **Google Cloud SDK** installed and configured
- **Docker** installed and running
- **GCP Project** with billing enabled
- **GitHub repository** with admin access

## 🔧 Infrastructure Overview

### GPU Configuration
- **GPU Type**: NVIDIA L4 (optimized for ML inference)
- **GPU Count**: 1 per instance
- **Memory**: 8Gi per instance
- **CPU**: 4 cores per instance
- **Auto-scaling**: 0-5 instances

### Regional Setup
- **Region**: `eu-central1` (Frankfurt)
- **Artifact Registry**: `eu-central1-docker.pkg.dev`

## 🏗️ Manual Setup Steps

### 1. Enable GCP APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  containerregistry.googleapis.com \
  compute.googleapis.com
```

### 2. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create mannequin-segmenter-sa \
  --display-name="Mannequin Segmenter Service Account"

# Grant required roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:mannequin-segmenter-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:mannequin-segmenter-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:mannequin-segmenter-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.accessor"
```

### 3. Create Artifact Registry

```bash
gcloud artifacts repositories create mannequin-segmenter-repo \
  --repository-format=docker \
  --location=eu-central1 \
  --description="Repository for mannequin segmenter container images"
```

### 4. Setup Secrets

```bash
# Create secrets for AWS credentials
echo "YOUR_AWS_ACCESS_KEY" | gcloud secrets create aws-access-key --data-file=-
echo "YOUR_AWS_SECRET_KEY" | gcloud secrets create aws-secret-key --data-file=-
echo "YOUR_S3_BUCKET_NAME" | gcloud secrets create aws-s3-bucket --data-file=-
echo "YOUR_S3_REGION" | gcloud secrets create aws-s3-region --data-file=-
```

### 5. Deploy with GPU Support

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker eu-central1-docker.pkg.dev

# Build and push GPU-optimized image
docker build -f Dockerfile.gpu -t eu-central1-docker.pkg.dev/YOUR_PROJECT_ID/mannequin-segmenter-repo/mannequin-segmenter:latest .
docker push eu-central1-docker.pkg.dev/YOUR_PROJECT_ID/mannequin-segmenter-repo/mannequin-segmenter:latest

# Deploy to Cloud Run with GPU
gcloud run deploy mannequin-segmenter \
  --image eu-central1-docker.pkg.dev/YOUR_PROJECT_ID/mannequin-segmenter-repo/mannequin-segmenter:latest \
  --platform managed \
  --region eu-central1 \
  --allow-unauthenticated \
  --port 5001 \
  --memory 8Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --timeout 900 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars="EVFSAM_PROMPT_MODE=both,FLASK_ENV=production,PYTHONPATH=/app,CUDA_VISIBLE_DEVICES=0" \
  --set-secrets="AWS_ACCESS_KEY_ID=aws-access-key:latest,AWS_SECRET_ACCESS_KEY=aws-secret-key:latest,AWS_S3_BUCKET_NAME=aws-s3-bucket:latest,AWS_S3_REGION=aws-s3-region:latest"
```

## 🔑 GitHub Secrets Configuration

Add these secrets to your GitHub repository (`Settings` → `Secrets and variables` → `Actions`):

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `GCP_PROJECT_ID` | Your GCP project ID | `my-ml-project-123456` |
| `GCP_SA_KEY` | Service account JSON key | `{"type": "service_account", ...}` |

## 🐳 Docker Configuration

### GPU-Optimized Dockerfile

The `Dockerfile.gpu` includes:
- **NVIDIA CUDA 11.8** runtime base image
- **PyTorch with CUDA support** (cu118)
- **Optimized dependencies** for GPU inference
- **CUDA environment variables** properly configured

### Key Optimizations
- Uses CUDA-specific PyTorch build for better GPU utilization
- Sets `CUDA_VISIBLE_DEVICES=0` for single GPU access
- Configures NVIDIA driver capabilities for compute workloads

## 📊 Performance Considerations

### GPU Benefits
- **~10-50x faster** inference compared to CPU-only deployment
- **Lower latency** for real-time image segmentation
- **Better throughput** for batch processing

### Cost Optimization
- **Min instances: 0** - scales to zero when not in use
- **Max instances: 5** - prevents runaway costs
- **Concurrency: 1** - optimal for GPU workloads
- **15-minute timeout** - sufficient for model loading and inference

## 🔍 Monitoring and Debugging

### Cloud Run Logs
```bash
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=mannequin-segmenter" --limit 50
```

### GPU Utilization
Monitor GPU usage in the Cloud Run metrics dashboard:
- GPU utilization percentage
- GPU memory usage
- Instance scaling events

### Health Check
```bash
curl https://YOUR_SERVICE_URL/health
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Quota Exceeded**
   ```bash
   gcloud compute quotas list --filter="metric:nvidia_l4_gpus"
   ```
   Request quota increase if needed.

2. **CUDA Out of Memory**
   - Increase memory allocation (`--memory 16Gi`)
   - Reduce batch size in application code

3. **Cold Start Timeout**
   - Increase timeout (`--timeout 1200`)
   - Consider using min-instances > 0 for critical workloads

4. **Image Build Failures**
   - Ensure CUDA compatibility between base image and PyTorch version
   - Check Dockerfile.gpu for proper CUDA environment setup

## 📈 Scaling Configuration

### Auto-scaling Behavior
- **Scale to zero**: Instances automatically terminate after 15 minutes of inactivity
- **Scale up**: New instances spin up within 30-60 seconds under load
- **GPU allocation**: Each instance gets 1 dedicated NVIDIA L4 GPU

### Custom Scaling
```bash
# For high-traffic scenarios
gcloud run services update mannequin-segmenter \
  --min-instances 2 \
  --max-instances 10 \
  --region eu-central1
```

## 💰 Cost Estimation

| Resource | Cost (per hour) | Notes |
|----------|----------------|-------|
| NVIDIA L4 GPU | ~$0.60 | Per GPU per hour |
| 8Gi Memory | ~$0.08 | Per Gi per hour |
| 4 vCPU | ~$0.24 | Per vCPU per hour |
| **Total** | **~$0.92/hour** | Per active instance |

With scale-to-zero, you only pay when the service is actively processing requests.

## 🔗 Additional Resources

- [Cloud Run GPU Documentation](https://cloud.google.com/run/docs/configuring/services/gpu)
- [NVIDIA L4 GPU Specifications](https://www.nvidia.com/en-us/data-center/l4/)
- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [EVF-SAM Model Documentation](https://github.com/hustvl/EVF-SAM) 