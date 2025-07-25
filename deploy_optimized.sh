#!/bin/bash

# Optimized Deployment Script for Mannequin Segmenter
# Targets 6-second completion time for 100 concurrent requests

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"$(gcloud config get-value project 2>/dev/null)"}
SERVICE_NAME="mannequin-segmenter"
REGION="europe-west4"
REPOSITORY_NAME="mannequin-segmenter-repo"

echo "🚀 Deploying OPTIMIZED Mannequin Segmenter for 6-second target..."

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID not set. Please set it using:"
    echo "   export PROJECT_ID=your-actual-project-id"
    echo "   or use: gcloud config set project your-project-id"
    exit 1
fi

echo "📋 Project ID: $PROJECT_ID"

# Configure Docker for Artifact Registry
echo "📦 Configuring Docker..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Build optimized GPU image
echo "🔨 Building optimized GPU image..."
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${SERVICE_NAME}:optimized-$(date +%Y%m%d-%H%M%S)"
docker build -f Dockerfile.gpu -t "$IMAGE_URL" .

# Push image
echo "📤 Pushing image to Artifact Registry..."
docker push "$IMAGE_URL"

# Deploy with optimized settings
echo "🚀 Deploying to Cloud Run with OPTIMIZED settings..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_URL" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --port 5001 \
    --memory 32Gi \
    --cpu 8.0 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --timeout 300 \
    --concurrency 20 \
    --min-instances 3 \
    --max-instances 3 \
    --cpu-boost \
    --execution-environment gen2 \
    --set-env-vars="ENVIRONMENT=production,PYTHONPATH=/app,CUDA_VISIBLE_DEVICES=0" \
    --set-secrets="AWS_ACCESS_KEY_ID=aws-access-key:latest" \
    --set-secrets="AWS_SECRET_ACCESS_KEY=aws-secret-key:latest" \
    --set-secrets="AWS_S3_BUCKET_NAME=aws-s3-bucket:latest" \
    --set-secrets="AWS_S3_REGION=aws-s3-region:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)')

echo "✅ Deployment completed!"
echo "🌐 Service URL: $SERVICE_URL"

# Test the deployment
echo "🧪 Testing deployment..."
if curl -f "$SERVICE_URL/health" >/dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed. Check Cloud Run logs."
fi

echo ""
echo "🎯 OPTIMIZATION SUMMARY:"
echo "   • Concurrency: 20 (optimized for GPU)"
echo "   • Min instances: 3 (always warm)"
echo "   • Max instances: 3 (no scaling, consistent performance)"
echo "   • Memory: 32Gi with NVIDIA L4 GPU"
echo "   • CPU: 8.0 with CPU boost enabled"
echo "   • Preload app: enabled"
echo "   • Reduced logging overhead"
echo "   • Optimized timeouts"
echo ""
echo "📊 Expected performance:"
echo "   • Target: 100 requests in 6 seconds"
echo "   • Throughput: ~16.7 requests/second"
echo "   • Average response time: <6 seconds"
echo ""
echo "🧪 Run load test: python load_test.py" 