"""
FastAPI Optimized Mannequin Segmenter API
Performance-optimized implementation with async batch processing for 6s SLA and 50 concurrency
"""

import asyncio
import time
import torch
import numpy as np
import boto3
import io
import uuid
from datetime import datetime
from collections import deque
from typing import List, Dict, Any, Optional
import concurrent.futures
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import aiohttp
import aiofiles
from pydantic import BaseModel
from PIL import Image
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

from tools.logger import AppLogger
from tools.env_utils import get_env_variable
from tools.BirefNet import BiRefNetSegmenter
from request_limiter import limit_concurrent_requests, get_limiter_status
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Prometheus metrics for observability
REQUEST_COUNT = Counter('mannequin_segmenter_requests_total', 'Total requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('mannequin_segmenter_request_duration_seconds', 'Request duration')
BATCH_SIZE_GAUGE = Gauge('mannequin_segmenter_batch_size', 'Current batch size')
GPU_MEMORY_GAUGE = Gauge('mannequin_segmenter_gpu_memory_allocated_bytes', 'GPU memory allocated')
QUEUE_SIZE_GAUGE = Gauge('mannequin_segmenter_queue_size', 'Number of requests in queue')

# Global queue for batch processing
request_queue = deque()
BATCH_LIMIT = 10  # Maximum images per batch - reduced for single worker optimization
PAD_TIME = 0.010  # 10ms wait time to collect batch
PROCESSING_TIMEOUT = 60  # Maximum processing time per batch

# Global variables for app components
api_logger = None
s3_client = None
inferencer = None


class ImageRequest(BaseModel):
    image_url: str


class BatchImageRequest(BaseModel):
    image_urls: List[str]


class InferenceItem:
    """Container for inference request with future for async completion"""
    def __init__(self, image_data: np.ndarray, future: asyncio.Future):
        self.image_data = image_data
        self.future = future
        self.timestamp = time.time()


async def download_image_async(session: aiohttp.ClientSession, url: str) -> Optional[np.ndarray]:
    """Download and preprocess image asynchronously"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                image_data = await response.read()
                # Convert to PIL Image and then to numpy array
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return np.array(image)
            else:
                api_logger.log(f"Failed to download image: HTTP {response.status}")
                return None
    except Exception as e:
        api_logger.log(f"Error downloading image {url}: {e}")
        return None


async def batch_worker():
    """Background worker that processes batches of images for optimal GPU utilization"""
    global request_queue, inferencer
    
    while True:
        await asyncio.sleep(PAD_TIME)
        
        if not request_queue:
            continue
            
        # Collect batch of requests
        batch_items = []
        start_time = time.time()
        
        # Collect up to BATCH_LIMIT items or until timeout
        while len(batch_items) < BATCH_LIMIT and request_queue and (time.time() - start_time) < PAD_TIME:
            batch_items.append(request_queue.popleft())
        
        if not batch_items:
            continue
            
        BATCH_SIZE_GAUGE.set(len(batch_items))
        QUEUE_SIZE_GAUGE.set(len(request_queue))
        
        try:
            # Extract image data for batch processing
            images = [item.image_data for item in batch_items]
            futures = [item.future for item in batch_items]
            
            api_logger.log(f"Processing batch of {len(images)} images")
            
            # Process batch using the optimized segmenter
            batch_start = time.time()
            processed_images = []
            
            # TRUE BATCH PROCESSING - process all images in single GPU forward pass
            if len(images) == 1:
                # Single image optimization - use existing method
                try:
                    img_bgr = images[0][:, :, ::-1]
                    result = inferencer.process_image_array(img_bgr, plot=False)
                    processed_images.append(result)
                    api_logger.log(f"✅ Single image processed successfully")
                except Exception as e:
                    api_logger.log(f"Error processing single image: {e}")
                    processed_images.append(None)
            else:
                # REAL BATCH PROCESSING - multiple images in one GPU forward pass
                    try:
                    # Convert RGB to BGR for all images
                    batch_images_bgr = [img[:, :, ::-1] for img in images]
                    
                    api_logger.log(f"🚀 Using TRUE BATCH PROCESSING for {len(batch_images_bgr)} images")
                    
                    # Use new batch processing method
                    batch_results = inferencer.process_image_arrays_batch(batch_images_bgr, plot=False)
                    processed_images.extend(batch_results)
                    
                    api_logger.log(f"✅ Batch processing completed: {len([r for r in batch_results if r is not None])}/{len(batch_results)} successful")
                    
                    except Exception as e:
                    api_logger.log(f"Batch processing failed, falling back to sequential: {e}")
                
                    # Fallback to sequential processing
                for img_array in images:
                        try:
                            img_bgr = img_array[:, :, ::-1]
                            result = inferencer.process_image_array(img_bgr, plot=False)
                    processed_images.append(result)
                        except Exception as single_e:
                            api_logger.log(f"Error processing image in fallback: {single_e}")
                            processed_images.append(None)
            
            batch_duration = time.time() - batch_start
            api_logger.log(f"Batch processing completed in {batch_duration:.3f}s")
            
            # Update GPU memory metrics
            if torch.cuda.is_available():
                GPU_MEMORY_GAUGE.set(torch.cuda.memory_allocated())
            
            # Set results for all futures
            for future, result in zip(futures, processed_images):
                if not future.cancelled():
                    future.set_result(result)
                    
        except Exception as e:
            api_logger.log(f"Batch processing error: {e}")
            # Set error for all futures
            for item in batch_items:
                if not item.future.cancelled():
                    item.future.set_exception(e)


async def upload_to_s3_async(processed_img: np.ndarray, s3_client, aws_s3_bucket_name: str, aws_s3_region: str) -> Optional[str]:
    """Upload processed image to S3 asynchronously"""
    try:
        # Convert to PIL and prepare for upload
        vis_pil = Image.fromarray(processed_img.astype(np.uint8))
        buff = io.BytesIO()
        vis_pil.save(buff, format="JPEG")
        buff.seek(0)

        filename = f"{uuid.uuid4()}.jpg"
        today = datetime.utcnow()
        date_prefix = today.strftime("%Y/%m/%d")
        s3_key = f"{date_prefix}/{filename}"

        # Upload to S3
        s3_client.upload_fileobj(
            buff,
            aws_s3_bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )

        s3_url = f"https://{aws_s3_bucket_name}.s3.{aws_s3_region}.amazonaws.com/{s3_key}"
        return s3_url
        
    except Exception as e:
        api_logger.log(f"S3 upload error: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup app components"""
    global api_logger, s3_client, inferencer
    
    # Initialize logger
    api_logger = AppLogger()
    api_logger.log("Starting FastAPI Mannequin Segmenter")
    
    # AWS S3 Configuration
    aws_access_key_id = get_env_variable("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = get_env_variable("AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket_name = get_env_variable("AWS_S3_BUCKET_NAME")
    aws_s3_region = get_env_variable("AWS_S3_REGION")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_s3_region
    )
    
    # Initialize the optimized BiRefNet segmenter
    try:
        inferencer = BiRefNetSegmenter(
            model_path="models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt",  # ✅ NEW BIREFNET_LITE MODEL FROM S3
            model_name="zhengpeng7/BiRefNet_lite",  # ✅ Use BiRefNet_lite
            precision="fp16",
            mask_threshold=0.5
        )
        api_logger.log("BiRefNet_lite segmenter initialized successfully")
        
        # Start background batch worker
        asyncio.create_task(batch_worker())
        api_logger.log("Background batch worker started")
        
    except Exception as e:
        api_logger.log(f"Model loading failed: {e}")
        inferencer = None
    
    yield
    
    # Cleanup
    api_logger.log("Shutting down FastAPI Mannequin Segmenter")


# Create FastAPI app with optimized settings
app = FastAPI(
    title="Mannequin Segmenter API",
    description="Performance-optimized mannequin segmentation with async batch processing",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    
    # Enhanced GPU detection
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None
    }
    
    if torch.cuda.is_available():
        gpu_info["current_device"] = torch.cuda.get_device_name(0)
        gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "mannequin-segmenter-api",
        "version": "2.0.0",
        "gpu_info": gpu_info,
        "model_loaded": inferencer is not None,
        "model_device": str(inferencer.device) if inferencer else "not_loaded"
    }


@app.get("/status")
async def status():
    """Get current server load and performance metrics"""
    limiter_status = get_limiter_status()
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_reserved": torch.cuda.memory_reserved(),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory
        }
    
    return {
        'service': 'mannequin-segmenter-api',
        'timestamp': datetime.utcnow().isoformat(),
        'request_limiter': limiter_status,
        'performance': {
            'queue_size': len(request_queue),
            'gpu_info': gpu_info,
            'model_loaded': inferencer is not None
        },
        'recommendation': {
            'load_level': 'low' if limiter_status['load_percentage'] < 50 else 'high' if limiter_status['load_percentage'] < 90 else 'critical',
            'can_accept_requests': limiter_status['slots_available'] > 0
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/infer")
async def infer(request: ImageRequest):
    """Single image inference with async batch processing"""
    REQUEST_COUNT.labels(endpoint="/infer", method="POST").inc()
    
    with REQUEST_DURATION.time():
        if inferencer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Download image asynchronously
        async with aiohttp.ClientSession() as session:
            image_data = await download_image_async(session, request.image_url)
            
        if image_data is None:
            raise HTTPException(status_code=400, detail="Failed to download image")
        
        # Create future for async result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Add to batch queue
        inference_item = InferenceItem(image_data, future)
        request_queue.append(inference_item)
        
        # Wait for batch processing result
        try:
            processed_image = await asyncio.wait_for(future, timeout=PROCESSING_TIMEOUT)
            
            if processed_image is None:
                raise HTTPException(status_code=500, detail="Image processing failed")
            
            # Upload to S3
            aws_s3_bucket_name = get_env_variable("AWS_S3_BUCKET_NAME")
            aws_s3_region = get_env_variable("AWS_S3_REGION")
            
            s3_url = await upload_to_s3_async(processed_image, s3_client, aws_s3_bucket_name, aws_s3_region)
            
            if s3_url is None:
                raise HTTPException(status_code=500, detail="S3 upload failed")
            
            return {"visualization_url": s3_url}
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Processing timeout")


@app.post("/batch_infer")
async def batch_infer(request: BatchImageRequest):
    """Batch image inference with optimized processing"""
    REQUEST_COUNT.labels(endpoint="/batch_infer", method="POST").inc()
    
    with REQUEST_DURATION.time():
        if inferencer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if len(request.image_urls) > 20:
            raise HTTPException(status_code=400, detail="Maximum batch size is 20 images")
        
        # Download all images in parallel
        async with aiohttp.ClientSession() as session:
            download_tasks = [download_image_async(session, url) for url in request.image_urls]
            downloaded_images = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Filter successful downloads
        valid_images = []
        valid_indices = []
        for i, img in enumerate(downloaded_images):
            if isinstance(img, np.ndarray):
                valid_images.append(img)
                valid_indices.append(i)
        
        if not valid_images:
            raise HTTPException(status_code=400, detail="No images could be downloaded")
        
        # Create futures for batch processing
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in valid_images]
        
        # Add all to batch queue
        for img, future in zip(valid_images, futures):
            inference_item = InferenceItem(img, future)
            request_queue.append(inference_item)
        
        # Wait for all results
        try:
            processed_images = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True), 
                timeout=PROCESSING_TIMEOUT
            )
            
            # Upload successful results to S3 in parallel
            aws_s3_bucket_name = get_env_variable("AWS_S3_BUCKET_NAME")
            aws_s3_region = get_env_variable("AWS_S3_REGION")
            
            upload_tasks = []
            for processed_img in processed_images:
                if isinstance(processed_img, np.ndarray):
                    upload_tasks.append(upload_to_s3_async(processed_img, s3_client, aws_s3_bucket_name, aws_s3_region))
                else:
                    # Create a simple async coroutine that returns None for failed processing
                    async def failed_upload():
                        return None
                    upload_tasks.append(failed_upload())
            
            s3_urls = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Build response
            successful_count = sum(1 for url in s3_urls if isinstance(url, str))
            
            return {
                "batch_size": len(request.image_urls),
                "successful_count": successful_count,
                "failed_count": len(request.image_urls) - successful_count,
                "visualization_urls": [url if isinstance(url, str) else None for url in s3_urls],
                "input_urls": request.image_urls
            }
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Batch processing timeout")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 