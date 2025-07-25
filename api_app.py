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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
# from tools.MaskRCNN_segmenter import MaskRCNNSegmenter
from tools.BirefNet import BiRefNetSegmenter

# PyTorch Global Configuration for Inference
torch.backends.cudnn.benchmark = True  # Good for fixed input size
torch.set_grad_enabled(False)  # Global inference mode

# Initialize device and AMP dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model initialization - LOAD ONCE
print("🤖 Loading BiRefNet model globally...")
try:
    inferencer = BiRefNetSegmenter(
        model_path="artifacts/20250703_190222/checkpoint.pt",
        model_name="zhengpeng7/BiRefNet",
        precision="fp16",  # Use fp16 for memory efficiency
        mask_threshold=0.5
    )
    
    # Try to use bfloat16/fp16 for lower memory usage
    try:
        # inferencer model should already be in fp16 from precision setting
        AMP_DTYPE = torch.float16  # Match the model precision
        print(f"✅ Using AMP dtype: {AMP_DTYPE}")
    except Exception as e:
        AMP_DTYPE = torch.float32
        print(f"⚠️ Fallback to fp32 AMP: {e}")
        
    print("✅ Global BiRefNet model loaded successfully!")
    
except Exception as e:
    print(f"❌ Failed to load global model: {e}")
    inferencer = None
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
    
    # Use global inferencer - no need to reload
    # In testing, this will be mocked to avoid loading the real model.
    if testing:
        global inferencer
        inferencer = None  # Will be replaced by mock in tests
    else:
        # inferencer is already loaded globally
        if inferencer is None:
            print("Error! Global inferencer not loaded!")
            exit(1)
        print("Using global inferencer!")

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
        """Get current server load and request limiter status."""
        limiter_status = get_limiter_status()
        return jsonify({
            'service': 'mannequin-segmenter-api',
            'timestamp': datetime.utcnow().isoformat(),
            'request_limiter': limiter_status,
            'recommendation': {
                'load_level': 'low' if limiter_status['load_percentage'] < 50 else 'high' if limiter_status['load_percentage'] < 90 else 'critical',
                'can_accept_requests': limiter_status['slots_available'] > 0
            }
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
        inferencer = current_app.config['INFERENCER'] # Use the inferencer from the app config
        try:
            api_logger.log("Received inference request")
            print("Received inference request")
            data = request.get_json()
            if not data or 'image_url' not in data:
                api_logger.log("Error: image_url not provided in request")
                print("Error: image_url not provided in request")
                return jsonify({"error": "image_url not provided"}), 400

            image_url = data['image_url']

            # Check if model loaded successfully
            if inferencer is None:
                api_logger.log("ERROR: Model not loaded, returning test response")
                print("ERROR: Model not loaded, returning test response")
                return jsonify({
                    "error": "model failed to load",
                    "visualization_url": "https://test-response.example.com/test.jpg",
                    "input_url": image_url
                }), 500 # Return 500 as it's a server-side issue
        except Exception as e:
            api_logger.log(f"Exception in /infer input handling: {str(e)}")
            print(f"Exception in /infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            print("Step 3: About to call process_image_url")
            api_logger.log("Step 3: About to call process_image_url")
            # vis = inferencer.process_image_url(image_url, plot=False, prompt_mode=prompt_mode)
            vis = inferencer.process_image_url(image_url)
            print("Step 4: process_image_url completed")
            api_logger.log("Step 4: process_image_url completed")

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
        """Batch inference endpoint for processing multiple images simultaneously."""
        inferencer = current_app.config['INFERENCER']
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
                
            if len(image_urls) > 20:  # Limit batch size
                api_logger.log(f"Error: batch size {len(image_urls)} exceeds maximum of 20")
                print(f"Error: batch size {len(image_urls)} exceeds maximum of 20")
                return jsonify({"error": "Maximum batch size is 20 images"}), 400

            # Check if model loaded successfully
            if inferencer is None:
                api_logger.log("ERROR: Model not loaded for batch processing")
                print("ERROR: Model not loaded for batch processing")
                return jsonify({
                    "error": "model failed to load",
                    "batch_size": len(image_urls)
                }), 500

        except Exception as e:
            api_logger.log(f"Exception in /batch_infer input handling: {str(e)}")
            print(f"Exception in /batch_infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            batch_size = len(image_urls)
            api_logger.log(f"Processing batch of {batch_size} images")
            print(f"Processing batch of {batch_size} images")
            
            # Reset peak memory stats for this batch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            log_cuda_memory("batch_start")
            
            # Smart batch processing with memory-aware fallback
            processed_images = None
            try:
                api_logger.log(f"🚀 Attempting TRUE BATCH PROCESSING for {batch_size} images")
                print(f"🚀 Attempting TRUE BATCH PROCESSING for {batch_size} images")
                
                # Try true batch processing first
                processed_images = inferencer.process_batch_urls(image_urls, plot=False, max_batch_size=batch_size)
                
                api_logger.log(f"✅ TRUE BATCH SUCCESS: {len(processed_images) if processed_images else 0} processed images")
                print(f"✅ TRUE BATCH SUCCESS: {len(processed_images) if processed_images else 0} processed images")
                log_cuda_memory("batch_success")
                
            except Exception as batch_error:
                api_logger.log(f"🔄 Batch processing failed, falling back to parallel single processing: {batch_error}")
                print(f"🔄 Batch processing failed, falling back to parallel single processing: {batch_error}")
                
                # Fallback: Process images individually but efficiently
                processed_images = []
                for i, url in enumerate(image_urls):
                    try:
                        api_logger.log(f"🔄 Processing image {i+1}/{batch_size} individually")
                        single_result = inferencer.process_image_url(url, plot=False)
                        if single_result is not None:
                            processed_images.append(single_result)
                        else:
                            api_logger.log(f"⚠️ Image {i+1} processing returned None")
                    except Exception as single_error:
                        api_logger.log(f"❌ Failed to process image {i+1}: {single_error}")
                        
                api_logger.log(f"🔄 FALLBACK COMPLETE: {len(processed_images)}/{batch_size} images processed")
                print(f"🔄 FALLBACK COMPLETE: {len(processed_images)}/{batch_size} images processed")
                log_cuda_memory("fallback_complete")
            
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
            
            # Upload images in parallel (max 3 concurrent to be conservative)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
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
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return jsonify({"error": str(e), "batch_size": len(image_urls)}), 500
            
    # Attach objects to app context for easier testing and access
    app.config['S3_CLIENT'] = s3_client
    app.config['INFERENCER'] = inferencer
    app.config['API_LOGGER'] = api_logger
    # app.config['DEFAULT_PROMPT_MODE'] = default_prompt_mode

    return app

# Create a global app instance for gunicorn to find
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 