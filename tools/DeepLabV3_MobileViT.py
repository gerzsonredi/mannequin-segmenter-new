#!/usr/bin/env python3
"""
DeepLabV3-MobileViT Mannequin Segmenter
Fast and lightweight model for mannequin segmentation
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
import io
import boto3
import threading
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Handle both relative and absolute imports for local testing
try:
    from .logger import AppLogger
    from .env_utils import get_env_variable
except ImportError:
    # Fallback for local testing
    from logger import AppLogger
    from env_utils import get_env_variable

# ⚡ SPEED: Global cache for image downloads (DISABLED in production)
ENABLE_IMAGE_CACHE = os.getenv('ENVIRONMENT', 'development').lower() != 'production'
_image_download_cache = {} if ENABLE_IMAGE_CACHE else None
_cache_lock = threading.Lock() if ENABLE_IMAGE_CACHE else None

class DeepLabV3MobileViTSegmenter:
    """
    Fast DeepLabV3-MobileViT based mannequin segmenter
    Much faster than BiRefNet, optimized for CPU inference with caching
    """
    
    def __init__(self, 
                 model_path: str = "models/mannequin_segmenter_deeplabv3_mobilevit/checkpoint_20250726.pt",
                 model_name: str = "apple/deeplabv3-mobilevit-xx-small",
                 image_size: int = 512,
                 precision: str = "fp32",
                 vis_save_dir: str = "infer"):
        
        self.model_path = model_path
        self.model_name = model_name
        self.image_size = image_size
        self.precision = precision
        self.vis_save_dir = vis_save_dir
        
        # Initialize logger
        self.logger = AppLogger()
        
        # Device setup
        self.logger.log("Initializing DeepLabV3-MobileViT Segmenter")
        print("Initializing DeepLabV3-MobileViT Segmenter")
        
        # Check if CPU is forced via environment variable
        force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        
        if force_cpu:
            self.device = torch.device("cpu")
            self.logger.log("🔧 FORCE_CPU enabled - Using CPU for DeepLabV3-MobileViT")
            print("🔧 FORCE_CPU enabled - Using CPU for DeepLabV3-MobileViT")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.log(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            self.logger.log("💻 Using CPU (no GPU available)")
            print("💻 Using CPU (no GPU available)")
        
        # Optimize CPU threading for maximum single-request performance
        if self.device.type == 'cpu':
            cpu_count = os.cpu_count() or 4
            # Use all available CPU cores for maximum single request speed
            max_threads = cpu_count  # Use all available cores
            torch.set_num_threads(max_threads)
            
            # Set environment variables for multi-threading libraries
            os.environ['OMP_NUM_THREADS'] = str(max_threads)
            os.environ['MKL_NUM_THREADS'] = str(max_threads) 
            os.environ['NUMEXPR_MAX_THREADS'] = str(max_threads)
            os.environ['BLAS_NUM_THREADS'] = str(max_threads)
            
            self.logger.log(f"🧵 CPU multi-threading optimized: {torch.get_num_threads()} threads across {cpu_count} cores")
            print(f"🧵 CPU multi-threading optimized: {torch.get_num_threads()} threads across {cpu_count} cores")
        
        # Load model and processor
        try:
            self.logger.log(f"Loading DeepLabV3-MobileViT model: {model_name}")
            print(f"Loading DeepLabV3-MobileViT model: {model_name}")
            
            # Load processor
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            print("✅ AutoImageProcessor loaded")
            
            # Load base model
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            print("✅ Base model loaded")
            
            # Try to load custom checkpoint
            checkpoint_loaded = False
            if os.path.exists(model_path):
                try:
                    self.logger.log(f"Loading custom checkpoint from: {model_path}")
                    print(f"Loading custom checkpoint from: {model_path}")
                    
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint, strict=False)
                    checkpoint_loaded = True
                    
                    self.logger.log("✅ Custom checkpoint loaded successfully")
                    print("✅ Custom checkpoint loaded successfully")
                    
                except Exception as e:
                    self.logger.log(f"⚠️ Failed to load local checkpoint: {e}")
                    print(f"⚠️ Failed to load local checkpoint: {e}")
            
            # Try to download from S3 if local file not found
            if not checkpoint_loaded:
                try:
                    self.logger.log("Attempting to download checkpoint from S3...")
                    print("Attempting to download checkpoint from S3...")
                    
                    s3_path = f"s3://artifactsredi/models/mannequin_segmenter_deeplabv3_mobilevit/checkpoint_20250726.pt"
                    
                    # Download from S3
                    if self._download_model_from_s3(s3_path, model_path):
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                        self.model.load_state_dict(checkpoint, strict=False)
                        checkpoint_loaded = True
                        
                        self.logger.log("✅ S3 checkpoint downloaded and loaded")
                        print("✅ S3 checkpoint downloaded and loaded")
                    
                except Exception as s3_error:
                    self.logger.log(f"⚠️ S3 download failed: {s3_error}")
                    print(f"⚠️ S3 download failed: {s3_error}")
            
            if not checkpoint_loaded:
                self.logger.log("ℹ️ Using base pretrained model (no custom checkpoint)")
                print("ℹ️ Using base pretrained model (no custom checkpoint)")
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Optimize model for CPU performance
            if self.device.type == 'cpu':
                # Convert to channels_last memory format for better CPU performance
                self.model = self.model.to(memory_format=torch.channels_last)
                self.logger.log("✅ Model optimized for CPU with channels_last memory format")
                print("✅ Model optimized for CPU with channels_last memory format")
            
            # Set precision
            if self.precision == "fp16" and self.device.type == 'cuda':
                self.model = self.model.half()
                self.logger.log("✅ Model set to FP16 precision")
                print("✅ Model set to FP16 precision")
            
            # Create output directory
            os.makedirs(vis_save_dir, exist_ok=True)
            
            self.logger.log(f"✅ DeepLabV3-MobileViT initialized on {self.device}")
            print(f"✅ DeepLabV3-MobileViT initialized on {self.device}")
            
        except Exception as e:
            self.logger.log(f"❌ Failed to initialize DeepLabV3-MobileViT: {e}")
            print(f"❌ Failed to initialize DeepLabV3-MobileViT: {e}")
            raise e
    
    def _download_model_from_s3(self, s3_path: str, local_path: str) -> bool:
        """Download model from S3"""
        try:
            # Parse S3 path
            s3_path = s3_path.replace("s3://", "")
            bucket_name = s3_path.split("/")[0]
            key = "/".join(s3_path.split("/")[1:])
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=get_env_variable('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=get_env_variable('AWS_SECRET_ACCESS_KEY'),
                region_name=get_env_variable('AWS_S3_REGION') or 'eu-central-1'
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            s3_client.download_file(bucket_name, key, local_path)
            
            self.logger.log(f"✅ Model downloaded from S3: {s3_path}")
            print(f"✅ Model downloaded from S3: {s3_path}")
            return True
            
        except Exception as e:
            self.logger.log(f"❌ S3 download failed: {e}")
            print(f"❌ S3 download failed: {e}")
            return False
    
    def _preprocess_image(self, image_url: str) -> torch.Tensor:
        """Download and preprocess image with SPEED optimizations + caching"""
        try:
            # ⚡ SPEED: Check cache first (only in development)
            global _image_download_cache, _cache_lock, ENABLE_IMAGE_CACHE
            
            if ENABLE_IMAGE_CACHE and _image_download_cache is not None and _cache_lock is not None:
                with _cache_lock:
                    if image_url in _image_download_cache:
                        print("   💾 Using cached image (development only)")
                        cached_data = _image_download_cache[image_url]
                        image = Image.open(io.BytesIO(cached_data)).convert('RGB')
                    else:
                        print("   🌐 Downloading image (caching enabled)...")
                        # ⚡ SPEED: Fast image download
                        import requests
                        session = requests.Session()
                        session.headers.update({
                            'Accept-Encoding': 'gzip, deflate',
                            'Connection': 'keep-alive',
                            'User-Agent': 'mannequin-segmenter/1.0'
                        })
                        
                        response = session.get(image_url, timeout=10, stream=True)  # Reduced timeout
                        response.raise_for_status()
                        
                        # Cache the raw image data
                        image_data = response.content
                        _image_download_cache[image_url] = image_data
                        
                        # Limit cache size (keep only last 5 images)
                        if len(_image_download_cache) > 5:
                            oldest_key = next(iter(_image_download_cache))
                            del _image_download_cache[oldest_key]
                        
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                # Production mode: Always download fresh (no cache)
                print("   🌐 Downloading image (production - no cache)...")
                import requests
                session = requests.Session()
                session.headers.update({
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'User-Agent': 'mannequin-segmenter/1.0'
                })
                
                response = session.get(image_url, timeout=10, stream=True)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # ⚡ SPEED: Direct resize with PIL (faster than AutoImageProcessor for resize)
            image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            
            # ⚡ SPEED: Fast numpy conversion
            import numpy as np
            img_array = np.array(image_resized, dtype=np.float32) / 255.0
            
            # ⚡ SPEED: Direct tensor creation (faster than AutoImageProcessor)
            pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # CHW format
            
            # Move to device and optimize
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            
            # Enable channels_last for CPU performance
            if self.device.type == 'cpu':
                pixel_values = pixel_values.to(memory_format=torch.channels_last)
            
            if self.precision == "fp16" and self.device.type == 'cuda':
                pixel_values = pixel_values.half()
            
            return pixel_values, image
            
        except Exception as e:
            self.logger.log(f"❌ Preprocessing failed: {e}")
            print(f"❌ Preprocessing failed: {e}")
            return None, None
    
    def _run_inference(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run model inference with CPU multi-threading optimization"""
        try:
            with torch.inference_mode():
                # Enable CPU optimizations
                torch.set_grad_enabled(False)
                
                # Forward pass with optimized threading
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits  # (1, num_classes, H, W)
                
                # Apply softmax to get probabilities - CPU optimized
                probs = torch.softmax(logits, dim=1)
                
                # Get mannequin mask (assuming class 1 is mannequin)
                # You may need to adjust the class index based on your training
                mannequin_probs = probs[:, 1:2, :, :]  # Keep batch and channel dims
                
                return mannequin_probs
                
        except Exception as e:
            self.logger.log(f"❌ Inference failed: {e}")
            print(f"❌ Inference failed: {e}")
            return None
    
    def _postprocess_mask(self, mask_tensor: torch.Tensor, original_image: Image.Image) -> np.ndarray:
        """Convert mask tensor to final visualization - CPU optimized"""
        try:
            # Get mask on CPU and resize using PIL (faster than PyTorch on CPU)
            mask = mask_tensor.squeeze().cpu().numpy()  # Remove batch and channel dims
            
            # Convert mask to PIL Image for fast resize
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_resized_pil = mask_pil.resize(original_image.size, Image.BILINEAR)
            mask_resized = np.array(mask_resized_pil).astype(np.float32) / 255.0
            
            # Apply threshold
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Convert original image to numpy
            orig_array = np.array(original_image)
            
            # Create visualization (mannequin + transparent background) - vectorized operations
            vis = orig_array.copy()
            background_mask = mask_binary == 0
            vis[background_mask] = (vis[background_mask] * 0.3).astype(np.uint8)  # Darken background
            
            return vis
            
        except Exception as e:
            self.logger.log(f"❌ Postprocessing failed: {e}")
            print(f"❌ Postprocessing failed: {e}")
            return None
    
    def process_image_url(self, image_url: str, plot: bool = False) -> np.ndarray:
        """Process image from URL with SPEED optimizations"""
        try:
            # ⚡ SPEED: Minimal logging for performance
            print("Step 2: Fast preprocessing...")
            
            # Preprocess
            pixel_values, original_image = self._preprocess_image(image_url)
            if pixel_values is None:
                return None
            
            print("Step 3: Running inference...")
            # Inference
            mask_tensor = self._run_inference(pixel_values)
            if mask_tensor is None:
                return None
            
            print("Step 4: Fast postprocessing...")
            # Postprocess
            visualization = self._postprocess_mask(mask_tensor, original_image)
            if visualization is None:
                return None
            
            print("✅ Processing complete")
            
            return visualization
            
        except Exception as e:
            self.logger.log(f"❌ Process image failed: {e}")
            print(f"❌ Process image failed: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "precision": self.precision,
            "image_size": self.image_size,
            "architecture": "DeepLabV3-MobileViT",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        } 