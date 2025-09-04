from flask import Flask, request, jsonify, current_app
from tools.logger import AppLogger
from tools.env_utils import get_env_variable
from request_limiter import limit_concurrent_requests, get_limiter_status
import base64
from PIL import Image
import io
import numpy as np
from google.cloud import storage
import json
import concurrent.futures
import threading
from dotenv import load_dotenv
import os
import uuid
import random
import string
from datetime import datetime
# CRITICAL: Set environment variables BEFORE importing torch
# Remove problematic CUDA allocator config to avoid PyTorch crash
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
# from tools.MaskRCNN_segmenter import MaskRCNNSegmenter
# BiRefNet dependency removed - using BiSeNet v1 instead
import cv2 # Added for cv2.cvtColor
import time # Added for time.time()
import gc # Added for garbage collection

# 🧵 GLOBAL CPU THREADING OPTIMIZATION - BEFORE MODEL LOADING
force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
if force_cpu or not torch.cuda.is_available():
    cpu_count = os.cpu_count() or 4
    print(f"🧵 GLOBAL CPU OPTIMIZATION: Setting up {cpu_count} threads for maximum performance")
    
    # Set PyTorch threading
    torch.set_num_threads(cpu_count)
    
    # Set environment variables for all multi-threading libraries
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_count)
    os.environ['BLAS_NUM_THREADS'] = str(cpu_count)
    os.environ['TORCH_THREADS'] = str(cpu_count)
    
    print(f"✅ Global threading configured: {torch.get_num_threads()} PyTorch threads")

# PyTorch Global Configuration for Inference
torch.backends.cudnn.benchmark = True  # Good for fixed input size
torch.set_grad_enabled(False)  # Global inference mode

# Initialize device and AMP dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Single Model initialization - BISENET V1 ARCHITECTURE (NEW HIGH-PERFORMANCE MODEL)
print("🚀 Loading BiSeNet v1 Model globally (CPU optimized)...")

# Add startup status tracking for Cloud Run readiness
startup_status = {
    "model_loading": True,
    "model_loaded": False,
    "ready": False,
    "start_time": time.time()
}

try:
    from tools.BiSeNet_v1 import BiSeNetV1Segmenter, _get_cached_image
    
    # Initialize single BiSeNet v1 model per instance (concurrency=1)
    models_bucket = get_env_variable("GCP_MODELS_BUCKET") or "artifactsredi"
    model_path = f"{models_bucket}/models/mannequin_segmenter_bisenet/20250728/checkpoint-4.pt"
    single_model = BiSeNetV1Segmenter(
        model_path=model_path,
        model_name="BiSeNet v1 (2-class)",
        precision="fp32",  # CPU uses fp32
        vis_save_dir="infer"
    )
    
    # Set AMP dtype based on device
    if single_model.device.type == 'cuda':
        AMP_DTYPE = torch.float16
    else:
        AMP_DTYPE = torch.float32
        
    print(f"✅ BiSeNet v1 model loaded on {single_model.device}")
    print(f"✅ Using AMP dtype: {AMP_DTYPE}")
    print("🏗️ Architecture: 0-100 instances × 1 BiSeNet model = 100 parallel capacity")
    print(f"📊 Model size: {single_model.get_model_info()['parameters']:,} parameters")
    
    # 🚨 CRITICAL: Log connection limits and threading config at startup
    print("\n🔍 STARTUP DIAGNOSTICS:")
    print(f"   🧵 PyTorch threads: {torch.get_num_threads()}")
    print(f"   🌐 OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS', 'default')}")
    print(f"   🔄 Force CPU mode: {force_cpu}")
    print(f"   ⚡ CUDA available: {torch.cuda.is_available()}")
    print("   🚨 LOCK-FREE MODE: Global image cache lock DISABLED to prevent serialization!")
    print("   📡 Per-request downloads: Each instance downloads separately (no shared cache)")
    
    # Update startup status
    startup_status["model_loading"] = False
    startup_status["model_loaded"] = True
    startup_status["ready"] = True
    startup_time = time.time() - startup_status["start_time"]
    print(f"🚀 Startup completed in {startup_time:.2f}s")
    
except Exception as e:
    print(f"❌ Failed to load single model: {e}")
    single_model = None
    AMP_DTYPE = torch.float32
    startup_status["model_loading"] = False
    startup_status["model_loaded"] = False
    startup_status["ready"] = False

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

    # GCP Cloud Storage Configuration
    gcp_project_id = get_env_variable("GCP_PROJECT_ID")
    gcp_sa_key_b64 = get_env_variable("GCP_SA_KEY")
    gcp_bucket_name = get_env_variable("GCP_BUCKET_NAME") or "pictures-not-public"

    # Initialize GCS client with service account key
    if gcp_sa_key_b64:
        try:
            # Decode base64 service account key
            gcp_sa_key_json = base64.b64decode(gcp_sa_key_b64).decode('utf-8')
            gcp_sa_key = json.loads(gcp_sa_key_json)
            
            # Create GCS client with service account credentials
            gcs_client = storage.Client.from_service_account_info(gcp_sa_key, project=gcp_project_id)
            gcs_bucket = gcs_client.bucket(gcp_bucket_name)
        except Exception as e:
            print(f"❌ Failed to initialize GCS client: {e}")
            gcs_client = None
            gcs_bucket = None
    else:
        print("⚠️ No GCP_SA_KEY provided, GCS uploads will be disabled")
        gcs_client = None
        gcs_bucket = None
    
    # Use global single model - no need to reload
    # In testing, this will be mocked to avoid loading the real model.
    if testing:
        global single_model
        single_model = None  # Will be replaced by mock in tests
    else:
        # single_model is already loaded globally
        if single_model is None:
            print("Error! Global single model not loaded!")
            exit(1)
        print("Using global single model!")

    @app.route('/health', methods=['GET'])
    def health():
        """Fast health check endpoint for Cloud Run readiness probe."""
        current_time = time.time()
        uptime = current_time - startup_status["start_time"]
        
        # Always respond quickly, even during startup
        health_data = {
            "status": "healthy" if startup_status["ready"] else "starting",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "mannequin-segmenter-api",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "startup_status": {
                "model_loading": startup_status["model_loading"],
                "model_loaded": startup_status["model_loaded"],
                "ready": startup_status["ready"]
            }
        }
        
        # Log health check (but not too frequently to avoid spam)
        if int(uptime) % 10 == 0 or not startup_status["ready"]:
            api_logger.log(f"Health check: {health_data['status']} (uptime: {uptime:.1f}s)")
        
        # Return 200 even during startup to keep Cloud Run happy
        # Cloud Run needs to see 200 responses to consider the container ready
        return jsonify(health_data), 200

    @app.route('/cpu_diagnostics', methods=['GET'])
    def cpu_diagnostics():
        """Detailed CPU and threading diagnostics for Cloud Run"""
        import threading
        import multiprocessing
        
        diagnostics = {
            "system_info": {
                "cpu_count_os": os.cpu_count(),
                "cpu_count_multiprocessing": multiprocessing.cpu_count(),
                "active_threads": threading.active_count(),
                "current_thread": threading.current_thread().name
            },
            "pytorch_config": {
                "torch_threads": torch.get_num_threads(),
                "torch_device": str(device),
                "torch_version": torch.__version__
            },
            "environment_variables": {
                "OMP_NUM_THREADS": os.environ.get('OMP_NUM_THREADS', 'not_set'),
                "MKL_NUM_THREADS": os.environ.get('MKL_NUM_THREADS', 'not_set'),
                "NUMEXPR_MAX_THREADS": os.environ.get('NUMEXPR_MAX_THREADS', 'not_set'),
                "BLAS_NUM_THREADS": os.environ.get('BLAS_NUM_THREADS', 'not_set'),
                "TORCH_THREADS": os.environ.get('TORCH_THREADS', 'not_set'),
                "FORCE_CPU": os.environ.get('FORCE_CPU', 'not_set'),
                "ENVIRONMENT": os.environ.get('ENVIRONMENT', 'not_set')
            },
            "model_info": single_model.get_model_info() if single_model else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify(diagnostics)

    @app.route('/readiness', methods=['GET'])
    def readiness():
        """Kubernetes-style readiness probe - only returns 200 when fully ready."""
        if startup_status["ready"] and single_model is not None:
            return jsonify({
                "ready": True,
                "model_loaded": True,
                "timestamp": datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                "ready": False,
                "model_loaded": startup_status["model_loaded"],
                "model_loading": startup_status["model_loading"],
                "timestamp": datetime.utcnow().isoformat()
            }), 503  # Service Unavailable
    
    @app.route('/status', methods=['GET'])
    def status():
        """Get current server load, request limiter status, and model pool stats."""
        limiter_status = get_limiter_status()
        if single_model:
            model_info = single_model.get_model_info()
            model_stats = {
                "model_loaded": True,
                "device": model_info["device"],
                "architecture": f"CPU Horizontal Scaling (0-60 instances) - {model_info['architecture']}",
                "model_name": model_info["model_name"],
                "parameters": model_info["parameters"]
            }
        else:
            model_stats = {"error": "Single model not available"}
        
        return jsonify({
            'service': 'mannequin-segmenter-api',
            'timestamp': datetime.utcnow().isoformat(),
            'request_limiter': limiter_status,
            'model_info': model_stats,
            'recommendation': {
                'load_level': 'low' if limiter_status['load_percentage'] < 50 else 'high' if limiter_status['load_percentage'] < 90 else 'critical',
                'can_accept_requests': limiter_status['slots_available'] > 0 and pool_stats.get('available_models', 0) > 0
            }
        })
    
    @app.route('/pool_stats', methods=['GET'])
    def pool_stats():
        """Get single model statistics for horizontal scaling architecture."""
        if single_model is None:
            return jsonify({"error": "Single model not available"}), 500
            
        if single_model:
            model_info = single_model.get_model_info()
            stats = {
                "architecture": "CPU Horizontal Scaling (BiSeNet v1)",
                "model_loaded": True,
                "device": model_info["device"],
                "model_name": model_info["model_name"],
                "precision": model_info["precision"],
                "image_size": model_info["image_size"],
                "parameters": model_info["parameters"],
                "instances": "0-60 instances × 1 concurrent = 60 parallel capacity",
                "memory_per_instance": "4Gi",
                "cpu_per_instance": "2",
                "model_architecture": model_info["architecture"]
            }
        else:
            stats = {"error": "Single model not available"}
        
        # Add GPU memory if available
        if single_model and single_model.device.type == 'cuda':
            try:
                stats["gpu_memory"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                }
            except:
                pass
        
        return jsonify({
            'timestamp': datetime.utcnow().isoformat(),
            'model_statistics': stats
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
        request_start_time = time.time()
        timing_data = {}
        
        try:
            # ⚡ SPEED: Fast request parsing
            parse_start = time.time()
            print("Step 1: Fast request parsing...")
            data = request.get_json()
            if not data or 'image_url' not in data:
                return jsonify({"error": "image_url not provided"}), 400

            image_url = data['image_url']
            upload_gcs = data.get('upload_gcs', True)
            parse_time = time.time() - parse_start
            timing_data['request_parsing'] = parse_time
            print(f"   ✅ Request parsing: {parse_time:.3f}s")

            # ⚡ SPEED: Quick model check
            model_check_start = time.time()
            if single_model is None:
                return jsonify({"error": "model not loaded"}), 500
            model_check_time = time.time() - model_check_start
            timing_data['model_check'] = model_check_time
            print(f"   ✅ Model check: {model_check_time:.3f}s")
            
        except Exception as e:
            api_logger.log(f"Exception in /infer input handling: {str(e)}")
            print(f"Exception in /infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            print("Step 2: Starting detailed inference timing...")
            
            # ⚡ SPEED: Minimal logging for performance
            inference_start = time.time()
            
            # Use single model for processing - simple and efficient for 1 concurrent request per instance
            print(f"   📸 Processing image: {image_url}")
            result = single_model.process_image_url(image_url, plot=False)
            
            # Handle new tuple return (result, download_time)
            if isinstance(result, tuple):
                vis, input_download_time = result
                timing_data['input_download'] = input_download_time
            else:
                # Backward compatibility
                vis = result
                timing_data['input_download'] = 0.0
            
            inference_time = time.time() - inference_start
            timing_data['model_inference'] = inference_time
            print(f"   ✅ Model inference: {inference_time:.3f}s")
            print(f"   📥 Input download: {timing_data['input_download']:.3f}s")

            if vis is None:
                error_msg = f"Error: Failed to process image from URL: {image_url}"
                print(error_msg)
                api_logger.log(error_msg)
                return jsonify({"error": "Failed to process image"}), 500

            # Prepare response data first
            instance_completed_time = time.time()
            total_time = instance_completed_time - request_start_time
            timing_data['total_request'] = total_time
            timing_data['instance_completed_timestamp'] = instance_completed_time
            
            if upload_gcs:
                print("Step 3: Preparing async GCS upload...")
                api_logger.log("Step 3: Preparing async GCS upload")
                
                if gcs_bucket is None:
                    error_msg = "GCS client not initialized"
                    print(f"❌ {error_msg}")
                    api_logger.log(error_msg)
                    return jsonify({"error": error_msg}), 500
                
                # Handle both PIL Image and numpy array
                conversion_start = time.time()
                if isinstance(vis, Image.Image):
                    vis_pil = vis
                    print("   ✅ Using PIL Image directly")
                else:
                    vis_pil = Image.fromarray(vis.astype(np.uint8))
                    print("   🔄 Converting numpy array to PIL")
                
                buff = io.BytesIO()
                vis_pil.save(buff, format="JPEG")
                buff.seek(0)
                conversion_time = time.time() - conversion_start
                timing_data['image_conversion'] = conversion_time
                print(f"   ✅ Image conversion: {conversion_time:.3f}s")

                # Generate unique filename with UUID + timestamp + random suffix for maximum uniqueness
                random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                timestamp = int(time.time() * 1000)  # milliseconds
                filename = f"{uuid.uuid4()}_{timestamp}_{random_suffix}.jpg"
                today = datetime.utcnow()
                date_prefix = today.strftime("%Y/%m/%d")
                gcs_key = f"removed_mannequin/{date_prefix}/{filename}"
                gcs_url = f"https://storage.googleapis.com/{gcp_bucket_name}/{gcs_key}"

                # Start async upload in background thread
                def async_upload():
                    try:
                        upload_start = time.time()
                        blob = gcs_bucket.blob(gcs_key)
                        blob.upload_from_file(buff, content_type='image/jpeg')
                        upload_time = time.time() - upload_start
                        print(f"   🚀 Background GCS upload completed: {upload_time:.3f}s")
                        api_logger.log(f"Background GCS upload completed: {upload_time:.3f}s for {gcs_url}")
                    except Exception as e:
                        print(f"   ❌ Background GCS upload failed: {e}")
                        api_logger.log(f"Background GCS upload failed: {e}")

                # Start upload in background thread (non-blocking)
                upload_thread = threading.Thread(target=async_upload, daemon=True)
                upload_thread.start()
                
                print(f"Step 4: Request completed in {total_time:.3f}s (GCS upload started in background)")
                print(f"   📊 TIMING BREAKDOWN:")
                print(f"      Request parsing: {timing_data['request_parsing']:.3f}s ({timing_data['request_parsing']/total_time*100:.1f}%)")
                print(f"      Model check: {timing_data['model_check']:.3f}s ({timing_data['model_check']/total_time*100:.1f}%)")
                print(f"      Model inference: {timing_data['model_inference']:.3f}s ({timing_data['model_inference']/total_time*100:.1f}%)")
                print(f"      Input download: {timing_data['input_download']:.3f}s ({timing_data['input_download']/total_time*100:.1f}%)")
                print(f"      Image conversion: {timing_data['image_conversion']:.3f}s ({timing_data['image_conversion']/total_time*100:.1f}%)")
                print(f"      Total (before async upload): {total_time:.3f}s")
                print(f"   🚀 Response ready, GCS upload running in background: {gcs_url}")
                print(f"   ⏰ Instance completed at: {instance_completed_time:.6f}")
                
                api_logger.log(f"Request completed: {total_time:.3f}s (inference: {timing_data['model_inference']:.3f}s, input_download: {timing_data['input_download']:.3f}s)")
                
                return jsonify({
                    "visualization_url": gcs_url,
                    "timing": timing_data
                })
            else:
                # ⚡ SPEED: Minimal response for fast processing (no GCS upload)
                print(f"Step 3: Fast response (no GCS) - {total_time:.3f}s total")
                print(f"   📊 TIMING BREAKDOWN:")
                print(f"      Request parsing: {timing_data['request_parsing']:.3f}s ({timing_data['request_parsing']/total_time*100:.1f}%)")
                print(f"      Model check: {timing_data['model_check']:.3f}s ({timing_data['model_check']/total_time*100:.1f}%)")
                print(f"      Model inference: {timing_data['model_inference']:.3f}s ({timing_data['model_inference']/total_time*100:.1f}%)")
                print(f"      Input download: {timing_data['input_download']:.3f}s ({timing_data['input_download']/total_time*100:.1f}%)")
                print(f"      Total: {total_time:.3f}s")
                
                return jsonify({
                    "success": True,
                    "inference_completed": True,
                    "timing": timing_data
                })

        except Exception as e:
            total_time = time.time() - request_start_time
            error_msg = f"Error processing image from URL {image_url}: {str(e)} (after {total_time:.3f}s)"
            api_logger.log(error_msg)
            print(error_msg)
            return jsonify({"error": str(e), "timing": timing_data}), 500

    @app.route('/batch_infer', methods=['POST'])
    @torch.inference_mode()
    def batch_infer():
        """Batch inference endpoint - processes images sequentially on single instance for horizontal scaling."""
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

            # Check if single model loaded successfully
            if single_model is None:
                api_logger.log("ERROR: Single model not loaded for batch processing")
                print("ERROR: Single model not loaded for batch processing")
                return jsonify({
                    "error": "single model failed to load",
                    "batch_size": len(image_urls)
                }), 500

        except Exception as e:
            api_logger.log(f"Exception in /batch_infer input handling: {str(e)}")
            print(f"Exception in /batch_infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            batch_size = len(image_urls)
            api_logger.log(f"📋 Processing batch of {batch_size} images SEQUENTIALLY on single model (horizontal scaling)")
            print(f"📋 Processing batch of {batch_size} images SEQUENTIALLY on single model (horizontal scaling)")
            
            # Sequential processing with single model
            try:
                api_logger.log(f"🚀 Starting SEQUENTIAL BATCH PROCESSING on {single_model.device}")
                print(f"🚀 Starting SEQUENTIAL BATCH PROCESSING on {single_model.device}")
                
                processed_images = []
                for i, image_url in enumerate(image_urls):
                    try:
                        print(f"  Processing image {i+1}/{batch_size}: {image_url}")
                        result = single_model.process_image_url(image_url, plot=False)
                        processed_images.append(result)
                    except Exception as img_error:
                        print(f"  ❌ Failed to process image {i+1}: {img_error}")
                        processed_images.append(None)
                
                # Filter out None results
                valid_images = [img for img in processed_images if img is not None]
                
                api_logger.log(f"✅ SEQUENTIAL BATCH SUCCESS: {len(valid_images)}/{batch_size} images processed")
                print(f"✅ SEQUENTIAL BATCH SUCCESS: {len(valid_images)}/{batch_size} images processed")
                
                # Single model processing complete
                api_logger.log(f"📊 Single model batch processing complete on {single_model.device}")
                print(f"📊 Single model batch processing complete on {single_model.device}")
                
                processed_images = valid_images
                
            except Exception as batch_error:
                api_logger.log(f"❌ Single model batch processing failed: {batch_error}")
                print(f"❌ Single model batch processing failed: {batch_error}")
                return jsonify({"error": f"Batch processing failed: {str(batch_error)}"}), 500
            
            if not processed_images or len(processed_images) == 0:
                api_logger.log("Error: No images were successfully processed in batch")
                print("Error: No images were successfully processed in batch")
                return jsonify({"error": "Failed to process any images in batch"}), 500

            api_logger.log(f"Successfully processed {len(processed_images)} images, uploading to GCS...")
            print(f"Successfully processed {len(processed_images)} images, uploading to GCS...")
            
            # Upload all processed images to GCS in parallel for better performance
            today = datetime.utcnow()
            date_prefix = today.strftime("%Y/%m/%d")
            
            def upload_single_image(img_index_pair):
                i, processed_img = img_index_pair
                try:
                    if gcs_bucket is None:
                        return i, None, "GCS client not initialized"
                    
                    # Convert to PIL and upload
                    vis_pil = Image.fromarray(processed_img.astype(np.uint8))
                    buff = io.BytesIO()
                    vis_pil.save(buff, format="JPEG")
                    buff.seek(0)

                    filename = f"batch_{uuid.uuid4()}_{i}.jpg"
                    gcs_key = f"removed_mannequin/{date_prefix}/{filename}"

                    # Upload to GCS
                    blob = gcs_bucket.blob(gcs_key)
                    blob.upload_from_file(buff, content_type='image/jpeg')

                    gcs_url = f"https://storage.googleapis.com/{gcp_bucket_name}/{gcs_key}"
                    return i, gcs_url, None
                    
                except Exception as upload_error:
                    error_msg = f"Error uploading image {i}: {upload_error}"
                    api_logger.log(error_msg)
                    print(error_msg)
                    return i, None, str(upload_error)
            
            # Upload images in parallel (max 15 concurrent for 30 model pool utilization)
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                upload_tasks = [(i, img) for i, img in enumerate(processed_images)]
                upload_results = list(executor.map(upload_single_image, upload_tasks))
            
            # Build gcs_urls list in correct order
            gcs_urls = [None] * len(processed_images)
            for i, gcs_url, error in upload_results:
                gcs_urls[i] = gcs_url
            
            successful_uploads = [url for url in gcs_urls if url is not None]
            
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
                "visualization_urls": gcs_urls,
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
    def batch_infer_optimized():
        """Optimized batch processing not needed in horizontal scaling mode."""
        return jsonify({
            "info": "Optimized batch processing not needed in horizontal scaling mode",
            "recommendation": "Send individual requests to /infer - load balancer will distribute across 60 instances",
            "architecture": "0-60 instances × 1 concurrent = 60 parallel capacity",
            "alternative": "Use /batch_infer for sequential processing on single instance"
        }), 200
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
                    img = _get_cached_image(url)  # Image caching function  # Image caching function from BiSeNet_v1.py
                    if img is not None:
                        # Convert RGB to BGR for BiSeNet
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
    def batch_infer_lightweight():
        """Lightweight batch processing not needed in horizontal scaling mode."""
        return jsonify({
            "info": "Lightweight batch processing not needed in horizontal scaling mode",
            "recommendation": "Send individual requests to /infer - load balancer will distribute across 60 instances",
            "architecture": "0-60 instances × 1 concurrent = 60 parallel capacity",
            "alternative": "Use /batch_infer for sequential processing on single instance"
        }), 200
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
    app.config['GCS_CLIENT'] = gcs_client
    app.config['SINGLE_MODEL'] = single_model
    app.config['API_LOGGER'] = api_logger
    # app.config['DEFAULT_PROMPT_MODE'] = default_prompt_mode

    return app

# Create a global app instance for gunicorn to find
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 