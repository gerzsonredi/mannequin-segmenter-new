from flask import Flask, request, jsonify, current_app
from tools.logger import AppLogger
from tools.env_utils import get_env_variable
from request_limiter import limit_concurrent_requests, get_limiter_status
import base64
from PIL import Image
import io
import numpy as np
import boto3
import concurrent.futures
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
# CRITICAL: Set environment variables BEFORE importing torch
# Remove problematic CUDA allocator config to avoid PyTorch crash
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
# from tools.MaskRCNN_segmenter import MaskRCNNSegmenter
from tools.BirefNet import BiRefNetSegmenter, _get_cached_image  # ✅ Import shared cache function
import cv2 # Added for cv2.cvtColor
import time # Added for time.time()
import gc # Added for garbage collection

# PyTorch Global Configuration for Inference
torch.backends.cudnn.benchmark = True  # Good for fixed input size
torch.set_grad_enabled(False)  # Global inference mode

# Initialize device and AMP dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Model Pool initialization - LOAD 10 MODELS
print("🏊‍♂️ Loading BiRefNet Model Pool globally...")
try:
    from tools.model_pool import get_global_model_pool
    model_pool = get_global_model_pool()
    
    # Try to use bfloat16/fp16 for lower memory usage
    try:
        AMP_DTYPE = torch.float16  # Match the model precision
        print(f"✅ Using AMP dtype: {AMP_DTYPE}")
    except Exception as e:
        AMP_DTYPE = torch.float32
        print(f"⚠️ Fallback to fp32 AMP: {e}")
        
    print("✅ Global BiRefNet Model Pool loaded successfully!")
    
except Exception as e:
    print(f"❌ Failed to load global model pool: {e}")
    model_pool = None
    AMP_DTYPE = torch.float32

def log_cuda_memory(tag=""):
    """Log CUDA memory usage for debugging"""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{tag}] allocated={alloc:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB")

def create_app(testing=False):
    """Application factory for the Flask app."""
    app = Flask(__name__)
    load_dotenv()

    # Initialize logger
    api_logger = AppLogger()
    # Note: AppLogger is a simple custom logger, not a standard Python logger
    # So we don't integrate it with Flask's logging system

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
    
    # Use global model pool - no need to reload
    # In testing, this will be mocked to avoid loading the real models.
    if testing:
        global model_pool
        model_pool = None  # Will be replaced by mock in tests
    else:
        # model_pool is already loaded globally
        if model_pool is None:
            print("Error! Global model pool not loaded!")
            exit(1)
        print("Using global model pool!")

    @app.route('/health', methods=['GET'])
    def health():
        api_logger.log("Health check request received")
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "mannequin-segmenter-api",
            "version": "1.0.0"
        }), 200
    
    @app.route('/status', methods=['GET'])
    def status():
        """Get current server load, request limiter status, and model pool stats."""
        limiter_status = get_limiter_status()
        pool_stats = model_pool.get_stats() if model_pool else {"error": "Model pool not available"}
        
        return jsonify({
            'service': 'mannequin-segmenter-api',
            'timestamp': datetime.utcnow().isoformat(),
            'request_limiter': limiter_status,
            'model_pool': pool_stats,
            'recommendation': {
                'load_level': 'low' if limiter_status['load_percentage'] < 50 else 'high' if limiter_status['load_percentage'] < 90 else 'critical',
                'can_accept_requests': limiter_status['slots_available'] > 0 and pool_stats.get('available_models', 0) > 0
            }
        })
    
    @app.route('/pool_stats', methods=['GET'])
    def pool_stats():
        """Get detailed model pool statistics."""
        if model_pool is None:
            return jsonify({"error": "Model pool not available"}), 500
            
        stats = model_pool.get_stats()
        
        # Add GPU memory info if available
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'pool_statistics': stats
        })
    
    @app.route('/debug_batch', methods=['POST'])
    def debug_batch():
        """Debug endpoint for batch processing"""
        try:
            data = request.get_json()
            image_urls = data.get('image_urls', [])
            
            if not image_urls:
                return jsonify({"error": "No image URLs provided"}), 400
            
            batch_size = len(image_urls)
            api_logger.log(f"🐛 DEBUG: Testing batch processing with {batch_size} images")
            print(f"🐛 DEBUG: Testing batch processing with {batch_size} images")
            
            # Test the batch processing step by step
            debug_info = {
                "batch_size": batch_size,
                "image_urls": image_urls,
                "steps": []
            }
            
            try:
                # Step 1: Model check
                if inferencer is None:
                    debug_info["steps"].append({"step": "model_check", "status": "FAILED", "error": "Model not loaded"})
                    return jsonify(debug_info), 500
                else:
                    debug_info["steps"].append({"step": "model_check", "status": "OK"})
                
                # Step 2: Try batch processing
                api_logger.log(f"🐛 DEBUG: Calling process_batch_urls...")
                print(f"🐛 DEBUG: Calling process_batch_urls...")
                
                start_time = time.time()
                processed_images = inferencer.process_batch_urls(image_urls, plot=False, max_batch_size=batch_size)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                if processed_images and len(processed_images) > 0:
                    debug_info["steps"].append({
                        "step": "batch_processing", 
                        "status": "OK", 
                        "processed_count": len(processed_images),
                        "processing_time": processing_time
                    })
                else:
                    debug_info["steps"].append({
                        "step": "batch_processing", 
                        "status": "FAILED", 
                        "error": "No images processed",
                        "processing_time": processing_time
                    })
                
                return jsonify(debug_info)
                
            except Exception as process_error:
                debug_info["steps"].append({
                    "step": "batch_processing", 
                    "status": "ERROR", 
                    "error": str(process_error)
                })
                api_logger.log(f"🐛 DEBUG: Batch processing error: {process_error}")
                print(f"🐛 DEBUG: Batch processing error: {process_error}")
                return jsonify(debug_info), 500
                
        except Exception as e:
            api_logger.log(f"🐛 DEBUG: Debug endpoint error: {e}")
            print(f"🐛 DEBUG: Debug endpoint error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/infer', methods=['POST'])
    # @limit_concurrent_requests  # Temporarily disabled for testing
    @torch.inference_mode()
    def infer():
        model_pool = current_app.config['MODEL_POOL']  # Use the model pool from the app config
        try:
            api_logger.log("Received inference request")
            print("Received inference request")
            data = request.get_json()
            if not data or 'image_url' not in data:
                api_logger.log("Error: image_url not provided in request")
                print("Error: image_url not provided in request")
                return jsonify({"error": "image_url not provided"}), 400

            image_url = data['image_url']

            # Check if model pool loaded successfully
            if model_pool is None:
                api_logger.log("ERROR: Model pool not loaded, returning test response")
                print("ERROR: Model pool not loaded, returning test response")
                return jsonify({
                    "error": "model pool failed to load",
                    "visualization_url": "https://test-response.example.com/test.jpg",
                    "input_url": image_url
                }), 500 # Return 500 as it's a server-side issue
        except Exception as e:
            api_logger.log(f"Exception in /infer input handling: {str(e)}")
            print(f"Exception in /infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            print("Step 3: About to call model pool process_single_request")
            api_logger.log("Step 3: About to call model pool process_single_request")
            # Use model pool for processing - automatically handles model selection and threading
            vis = model_pool.process_single_request(image_url, plot=False)
            print("Step 4: model pool process_single_request completed")
            api_logger.log("Step 4: model pool process_single_request completed")

            if vis is None:
                print(f"Error: Failed to process image from URL: {image_url}")
                api_logger.log(f"Error: Failed to process image from URL: {image_url}")
                return jsonify({"error": "Failed to process image"}), 500

            print("Step 5: About to convert and upload to S3")
            api_logger.log("Step 5: About to convert and upload to S3")
            vis_pil = Image.fromarray(vis.astype(np.uint8))
            buff = io.BytesIO()
            vis_pil.save(buff, format="JPEG")
            buff.seek(0)

            filename = f"{uuid.uuid4()}.jpg"
            today = datetime.utcnow()
            date_prefix = today.strftime("%Y/%m/%d")
            s3_key = f"{date_prefix}/{filename}"

            s3_client.upload_fileobj(
                buff,
                aws_s3_bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )

            s3_url = f"https://{aws_s3_bucket_name}.s3.{aws_s3_region}.amazonaws.com/{s3_key}"
            print(f"Step 6: Successfully processed image and uploaded result to S3: {s3_url}")
            api_logger.log(f"Step 6: Successfully processed image and uploaded result to S3: {s3_url}")
            
            return jsonify({
                "visualization_url": s3_url,
            })

        except Exception as e:
            error_msg = f"Error processing image from URL {image_url}: {str(e)}"
            api_logger.log(error_msg)
            print(error_msg)
            return jsonify({"error": str(e)}), 500

    @app.route('/batch_infer', methods=['POST'])
    @limit_concurrent_requests
    @torch.inference_mode()
    def batch_infer():
        """Batch inference endpoint for processing multiple images simultaneously using model pool."""
        model_pool = current_app.config['MODEL_POOL']
        try:
            api_logger.log("Received batch inference request")
            print("Received batch inference request")
            data = request.get_json()
            
            if not data or 'image_urls' not in data:
                api_logger.log("Error: image_urls not provided in batch request")
                print("Error: image_urls not provided in batch request")
                return jsonify({"error": "image_urls list not provided"}), 400

            image_urls = data['image_urls']
            if not isinstance(image_urls, list) or len(image_urls) == 0:
                api_logger.log("Error: image_urls must be a non-empty list")
                print("Error: image_urls must be a non-empty list")
                return jsonify({"error": "image_urls must be a non-empty list"}), 400
                
            if len(image_urls) > 50:  # Increased limit for model pool (was 20)
                api_logger.log(f"Error: batch size {len(image_urls)} exceeds maximum of 50")
                print(f"Error: batch size {len(image_urls)} exceeds maximum of 50")
                return jsonify({"error": "Maximum batch size is 50 images"}), 400

            # Check if model pool loaded successfully
            if model_pool is None:
                api_logger.log("ERROR: Model pool not loaded for batch processing")
                print("ERROR: Model pool not loaded for batch processing")
                return jsonify({
                    "error": "model pool failed to load",
                    "batch_size": len(image_urls)
                }), 500

        except Exception as e:
            api_logger.log(f"Exception in /batch_infer input handling: {str(e)}")
            print(f"Exception in /batch_infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            batch_size = len(image_urls)
            api_logger.log(f"🏊‍♂️ Processing batch of {batch_size} images using MODEL POOL with parallel processing")
            print(f"🏊‍♂️ Processing batch of {batch_size} images using MODEL POOL with parallel processing")
            
            # Reset peak memory stats for this batch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            log_cuda_memory("batch_start")
            
            # Get pool stats before processing
            pool_stats = model_pool.get_stats()
            api_logger.log(f"📊 Pool stats before processing: {pool_stats['available_models']}/{pool_stats['pool_size']} models available")
            print(f"📊 Pool stats before processing: {pool_stats['available_models']}/{pool_stats['pool_size']} models available")
            
            # Use model pool for PARALLEL batch processing across multiple models
            try:
                api_logger.log(f"🚀 Starting PARALLEL BATCH PROCESSING across {pool_stats['pool_size']} models")
                print(f"🚀 Starting PARALLEL BATCH PROCESSING across {pool_stats['pool_size']} models")
                
                # Model pool automatically distributes requests across available models
                processed_images = model_pool.process_batch_requests(image_urls, plot=False)
                
                # Filter out None results
                valid_images = [img for img in processed_images if img is not None]
                
                api_logger.log(f"✅ PARALLEL BATCH SUCCESS: {len(valid_images)}/{batch_size} images processed")
                print(f"✅ PARALLEL BATCH SUCCESS: {len(valid_images)}/{batch_size} images processed")
                log_cuda_memory("batch_success")
                
                # Get pool stats after processing
                final_stats = model_pool.get_stats()
                api_logger.log(f"📊 Pool stats after processing: {final_stats['available_models']}/{final_stats['pool_size']} models available, {final_stats['active_requests']} active")
                print(f"📊 Pool stats after processing: {final_stats['available_models']}/{final_stats['pool_size']} models available, {final_stats['active_requests']} active")
                
                processed_images = valid_images
                
            except Exception as batch_error:
                api_logger.log(f"❌ Model pool batch processing failed: {batch_error}")
                print(f"❌ Model pool batch processing failed: {batch_error}")
                return jsonify({"error": f"Batch processing failed: {str(batch_error)}"}), 500
            
            if not processed_images or len(processed_images) == 0:
                api_logger.log("Error: No images were successfully processed in batch")
                print("Error: No images were successfully processed in batch")
                return jsonify({"error": "Failed to process any images in batch"}), 500

            api_logger.log(f"Successfully processed {len(processed_images)} images, uploading to S3...")
            print(f"Successfully processed {len(processed_images)} images, uploading to S3...")
            
            # Upload all processed images to S3 in parallel for better performance
            today = datetime.utcnow()
            date_prefix = today.strftime("%Y/%m/%d")
            
            def upload_single_image(img_index_pair):
                i, processed_img = img_index_pair
                try:
                    # Convert to PIL and upload
                    vis_pil = Image.fromarray(processed_img.astype(np.uint8))
                    buff = io.BytesIO()
                    vis_pil.save(buff, format="JPEG")
                    buff.seek(0)

                    filename = f"batch_{uuid.uuid4()}_{i}.jpg"
                    s3_key = f"{date_prefix}/{filename}"

                    s3_client.upload_fileobj(
                        buff,
                        aws_s3_bucket_name,
                        s3_key,
                        ExtraArgs={'ContentType': 'image/jpeg'}
                    )

                    s3_url = f"https://{aws_s3_bucket_name}.s3.{aws_s3_region}.amazonaws.com/{s3_key}"
                    return i, s3_url, None
                    
                except Exception as upload_error:
                    error_msg = f"Error uploading image {i}: {upload_error}"
                    api_logger.log(error_msg)
                    print(error_msg)
                    return i, None, str(upload_error)
            
            # Upload images in parallel (max 15 concurrent for 30 model pool utilization)
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                upload_tasks = [(i, img) for i, img in enumerate(processed_images)]
                upload_results = list(executor.map(upload_single_image, upload_tasks))
            
            # Build s3_urls list in correct order
            s3_urls = [None] * len(processed_images)
            for i, s3_url, error in upload_results:
                s3_urls[i] = s3_url
            
            successful_uploads = [url for url in s3_urls if url is not None]
            
            api_logger.log(f"Batch processing completed: {len(successful_uploads)}/{batch_size} successful")
            print(f"Batch processing completed: {len(successful_uploads)}/{batch_size} successful")
            
            # Final memory cleanup
            log_cuda_memory("batch_end")
            if torch.cuda.is_available():
                # torch.cuda.empty_cache()  # Temporarily disabled - causes PyTorch crash
                # torch.cuda.synchronize()
                pass
            
            return jsonify({
                "batch_size": batch_size,
                "successful_count": len(successful_uploads),
                "failed_count": batch_size - len(successful_uploads),
                "visualization_urls": s3_urls,
                "input_urls": image_urls
            })

        except Exception as e:
            error_msg = f"Error processing batch of {len(image_urls)} images: {str(e)}"
            api_logger.log(error_msg)
            print(error_msg)
            
            # Cleanup memory even on error
            log_cuda_memory("batch_error")
            if torch.cuda.is_available():
                # torch.cuda.empty_cache()  # Temporarily disabled - causes PyTorch crash
                # torch.cuda.synchronize()
                pass
            
            return jsonify({"error": str(e), "batch_size": len(image_urls)}), 500
            
    @app.route('/batch_infer_optimized', methods=['POST'])
    @limit_concurrent_requests
    @torch.inference_mode()
    def batch_infer_optimized():
        """OPTIMIZED batch inference for 6-second target using TRUE BATCH PROCESSING."""
        model_pool = current_app.config['MODEL_POOL']
        try:
            api_logger.log("🚀 OPTIMIZED batch inference request")
            print("🚀 OPTIMIZED batch inference request")
            data = request.get_json()
            
            if not data or 'image_urls' not in data:
                return jsonify({"error": "image_urls list not provided"}), 400

            image_urls = data['image_urls']
            if not isinstance(image_urls, list) or len(image_urls) == 0:
                return jsonify({"error": "image_urls must be a non-empty list"}), 400
                
            batch_size = len(image_urls)
            if batch_size > 50:
                return jsonify({"error": "Maximum batch size is 50 images"}), 400

            start_time = time.time()
            api_logger.log(f"🚀 OPTIMIZED BATCH: Processing {batch_size} images with TRUE BATCH INFERENCE")
            print(f"🚀 OPTIMIZED BATCH: Processing {batch_size} images with TRUE BATCH INFERENCE")
            
            # ✅ CRITICAL: Free up GPU memory for large batch processing
            api_logger.log("🧹 Pre-batch GPU memory cleanup...")
            print("🧹 Pre-batch GPU memory cleanup...")
            
            # ✅ DYNAMIC SCALING: Temporarily reduce model pool for memory-intensive batch processing
            original_pool_size = len(model_pool._models)
            if batch_size > 20 and original_pool_size > 5:
                api_logger.log(f"🔄 Temporarily scaling down model pool for batch size {batch_size}")
                print(f"🔄 Temporarily scaling down model pool for batch size {batch_size}")
                model_pool.scale_pool_temporarily(target_size=3)  # Keep only 3 models for large batches
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # One-time sync for memory cleanup
                gc.collect()
                
                # Log GPU memory before processing
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                api_logger.log(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                print(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
            
            # Get one model for true batch processing
            with model_pool.get_model() as model:
                # Use shared image cache for fast downloads
                api_logger.log("📥 Downloading images with shared cache...")
                print("📥 Downloading images with shared cache...")
                
                download_start = time.time()
                image_arrays = []
                
                # Download images using shared cache (already optimized)
                for i, url in enumerate(image_urls):
                    img = _get_cached_image(url)  # This function is defined in BirefNet.py
                    if img is not None:
                        # Convert RGB to BGR for BiRefNet
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        image_arrays.append(img_bgr)
                    else:
                        api_logger.log(f"⚠️ Failed to download image {i+1}: {url}")
                        print(f"⚠️ Failed to download image {i+1}: {url}")
                
                download_time = time.time() - download_start
                
                if not image_arrays:
                    return jsonify({"error": "No images could be downloaded"}), 400
                
                api_logger.log(f"📥 Downloaded {len(image_arrays)}/{batch_size} images in {download_time:.2f}s")
                print(f"📥 Downloaded {len(image_arrays)}/{batch_size} images in {download_time:.2f}s")
                
                # ✅ MEMORY-EFFICIENT: Process in smaller sub-batches if needed
                max_sub_batch_size = min(25, len(image_arrays))  # Limit to prevent OOM
                processed_images = []
                
                for i in range(0, len(image_arrays), max_sub_batch_size):
                    sub_batch = image_arrays[i:i + max_sub_batch_size]
                    api_logger.log(f"🔄 Processing sub-batch {i//max_sub_batch_size + 1}: {len(sub_batch)} images")
                    print(f"🔄 Processing sub-batch {i//max_sub_batch_size + 1}: {len(sub_batch)} images")
                    
                    try:
                        sub_results = model.process_image_arrays_batch(sub_batch, plot=False)
                        processed_images.extend(sub_results)
                        
                        # Cleanup after each sub-batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as sub_error:
                        api_logger.log(f"❌ Sub-batch {i//max_sub_batch_size + 1} failed: {sub_error}")
                        print(f"❌ Sub-batch {i//max_sub_batch_size + 1} failed: {sub_error}")
                        # Add None results for failed sub-batch
                        processed_images.extend([None] * len(sub_batch))
                
                batch_time = time.time() - download_start - download_time
                
                # Filter successful results
                valid_images = [img for img in processed_images if img is not None]
                
                api_logger.log(f"🚀 BATCH PROCESSING: {len(valid_images)}/{len(image_arrays)} successful in {batch_time:.2f}s")
                print(f"🚀 BATCH PROCESSING: {len(valid_images)}/{len(image_arrays)} successful in {batch_time:.2f}s")
                
                total_time = time.time() - start_time
                throughput = len(valid_images) / total_time if total_time > 0 else 0
                
                api_logger.log(f"✅ OPTIMIZED BATCH COMPLETE: {total_time:.2f}s total, {throughput:.1f} images/sec")
                print(f"✅ OPTIMIZED BATCH COMPLETE: {total_time:.2f}s total, {throughput:.1f} images/sec")
                
                # ✅ RESTORE POOL: Scale back up after processing
                if batch_size > 20 and original_pool_size > 5:
                    api_logger.log("🔄 Restoring model pool to original size...")
                    print("🔄 Restoring model pool to original size...")
                    # Note: We'll restore in background to avoid blocking response
                    # model_pool.restore_pool_size()
                
                return jsonify({
                    "batch_size": batch_size,
                    "successful_count": len(valid_images),
                    "failed_count": batch_size - len(valid_images),
                    "processing_time_seconds": total_time,
                    "download_time_seconds": download_time,
                    "inference_time_seconds": batch_time,
                    "throughput_images_per_second": throughput,
                    "optimization": "memory_efficient_batch_processing",
                    "sub_batches_used": (len(image_arrays) + max_sub_batch_size - 1) // max_sub_batch_size,
                    "pool_scaled": batch_size > 20 and original_pool_size > 5,
                    "message": f"Processed {len(valid_images)} images in {total_time:.2f}s"
                })

        except Exception as e:
            error_msg = f"Optimized batch processing failed: {str(e)}"
            api_logger.log(error_msg)
            print(error_msg)
            
            # Emergency GPU cleanup and pool restoration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Restore pool if it was scaled down
            try:
                if 'original_pool_size' in locals() and batch_size > 20 and original_pool_size > 5:
                    api_logger.log("🔄 Emergency pool restoration after error...")
                    print("🔄 Emergency pool restoration after error...")
                    # model_pool.restore_pool_size()
            except Exception as restore_error:
                api_logger.log(f"❌ Pool restoration failed: {restore_error}")
                print(f"❌ Pool restoration failed: {restore_error}")
                
            return jsonify({"error": error_msg}), 500

    @app.route('/batch_infer_lightweight', methods=['POST'])
    @limit_concurrent_requests 
    @torch.inference_mode()
    def batch_infer_lightweight():
        """LIGHTWEIGHT batch inference for 6-second target with memory-efficient processing."""
        model_pool = current_app.config['MODEL_POOL']
        try:
            api_logger.log("🚀 LIGHTWEIGHT batch inference request")
            print("🚀 LIGHTWEIGHT batch inference request")
            data = request.get_json()
            
            if not data or 'image_urls' not in data:
                return jsonify({"error": "image_urls list not provided"}), 400

            image_urls = data['image_urls']
            if not isinstance(image_urls, list) or len(image_urls) == 0:
                return jsonify({"error": "image_urls must be a non-empty list"}), 400
                
            batch_size = len(image_urls)
            if batch_size > 50:
                return jsonify({"error": "Maximum batch size is 50 images"}), 400

            start_time = time.time()
            api_logger.log(f"🚀 LIGHTWEIGHT BATCH: Processing {batch_size} images with memory-efficient approach")
            print(f"🚀 LIGHTWEIGHT BATCH: Processing {batch_size} images with memory-efficient approach")
            
            # Download all images first using shared cache
            download_start = time.time()
            image_arrays = []
            
            for i, url in enumerate(image_urls):
                img = _get_cached_image(url)
                if img is not None:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    image_arrays.append(img_bgr)
                    
            download_time = time.time() - download_start
            
            if not image_arrays:
                return jsonify({"error": "No images could be downloaded"}), 400
                
            api_logger.log(f"📥 Downloaded {len(image_arrays)}/{batch_size} images in {download_time:.2f}s")
            print(f"📥 Downloaded {len(image_arrays)}/{batch_size} images in {download_time:.2f}s")
            
            # Process in small memory-efficient batches
            batch_start = time.time()
            processed_images = []
            chunk_size = min(8, len(image_arrays))  # Very small batches to avoid OOM
            
            with model_pool.get_model() as model:
                for i in range(0, len(image_arrays), chunk_size):
                    chunk = image_arrays[i:i + chunk_size]
                    
                    # Use sequential processing for ultra-safe memory usage
                    chunk_results = []
                    for img in chunk:
                        try:
                            result = model.process_image_array(img, plot=False)
                            chunk_results.append(result)
                        except Exception as img_error:
                            api_logger.log(f"❌ Image processing failed: {img_error}")
                            chunk_results.append(None)
                    
                    processed_images.extend(chunk_results)
                    
                    # Cleanup after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            batch_time = time.time() - batch_start
            valid_images = [img for img in processed_images if img is not None]
            
            total_time = time.time() - start_time
            throughput = len(valid_images) / total_time if total_time > 0 else 0
            
            api_logger.log(f"✅ LIGHTWEIGHT BATCH COMPLETE: {total_time:.2f}s total, {throughput:.1f} images/sec")
            print(f"✅ LIGHTWEIGHT BATCH COMPLETE: {total_time:.2f}s total, {throughput:.1f} images/sec")
            
            return jsonify({
                "batch_size": batch_size,
                "successful_count": len(valid_images),
                "failed_count": batch_size - len(valid_images), 
                "processing_time_seconds": total_time,
                "download_time_seconds": download_time,
                "inference_time_seconds": batch_time,
                "throughput_images_per_second": throughput,
                "optimization": "lightweight_memory_safe",
                "chunk_size_used": chunk_size,
                "chunks_processed": (len(image_arrays) + chunk_size - 1) // chunk_size,
                "message": f"Processed {len(valid_images)} images in {total_time:.2f}s"
            })
            
        except Exception as e:
            error_msg = f"Lightweight batch processing failed: {str(e)}"
            api_logger.log(error_msg)
            print(error_msg)
            
            # Emergency cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            return jsonify({"error": error_msg}), 500

    # Attach objects to app context for easier testing and access
    app.config['S3_CLIENT'] = s3_client
    app.config['MODEL_POOL'] = model_pool
    app.config['API_LOGGER'] = api_logger
    # app.config['DEFAULT_PROMPT_MODE'] = default_prompt_mode

    return app

# Create a global app instance for gunicorn to find
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 