import os
import sys
import cv2
import torch
import numpy as np
import requests
import io
import gc
import concurrent.futures
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageSegmentation, AutoProcessor
from torchvision.transforms import functional as F
from dotenv import load_dotenv
from typing import List
import time
import uuid
import threading
from functools import lru_cache
from io import BytesIO
import boto3
import tempfile
import traceback
import copy  # 🚀 For model cloning in cache

# Import utilities from tools package
try:
    # Try relative import first
    from .logger import AppLogger
except (ImportError, ValueError):
    try:
        # Try absolute import from tools
        from tools.logger import AppLogger
    except ImportError:
        # Try direct import (when tools in PYTHONPATH)
        from logger import AppLogger

os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"

# ✅ SHARED IMAGE CACHE to eliminate duplicate downloads across model instances
_image_cache = {}
_cache_lock = threading.Lock()

# 🚀 SHARED MODEL CACHE to avoid re-downloading BiRefNet_lite for each instance
_model_cache = {}  # Global cache for loaded models
_model_cache_lock = threading.Lock()  # Thread-safe access to model cache

def _get_cached_model(model_name: str, device: torch.device):
    """
    Get cached model or download and cache if not available.
    Returns a fresh clone of the cached model.
    """
    cache_key = f"{model_name}_{device.type}"
    
    with _model_cache_lock:
        if cache_key not in _model_cache:
            print(f"🔽 First-time download: {model_name} for {device.type}")
            # Download and cache the model
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            
            base_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                config={'model_type': 'custom_segmentation_model'}
            )
            
            # Store in cache (on CPU to save GPU memory)
            _model_cache[cache_key] = base_model.cpu()
            print(f"✅ Model cached: {cache_key}")
        else:
            print(f"🎯 Using cached model: {cache_key}")
    
    # Return a fresh clone from cache (without re-downloading)
    cached_model = _model_cache[cache_key]
    
    # Use copy.deepcopy to create a true independent clone
    cloned_model = copy.deepcopy(cached_model)
    
    return cloned_model.to(device)

def _get_cached_image(image_url: str):
    """Get image from shared cache or download if not cached."""
    with _cache_lock:
        if image_url in _image_cache:
            return _image_cache[image_url]
    
    # Download only if not cached
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        img_data = response.content
        
        # Convert to opencv format
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cache the result
        with _cache_lock:
            _image_cache[image_url] = img
            # Simple cache size management (keep only last 50 images)
            if len(_image_cache) > 50:
                oldest_key = next(iter(_image_cache))
                del _image_cache[oldest_key]
        
        return img
        
    except Exception as e:
        print(f"⚠️ Image download failed: {e} (URL: {image_url})")
        return None

def _download_model_from_s3(bucket_name: str, s3_key: str, local_path: str) -> bool:
    """
    Download custom trained model from S3 bucket.
    
    Args:
        bucket_name: S3 bucket name (e.g., 'artifactsredi')
        s3_key: S3 object key (e.g., 'models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt')
        local_path: Local file path to save the model
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"📥 Downloading custom model from S3: s3://{bucket_name}/{s3_key}")
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the model file
        s3_client.download_file(bucket_name, s3_key, local_path)
        
        print(f"✅ Model downloaded successfully to: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model from S3: {e}")
        print(f"   Bucket: {bucket_name}")
        print(f"   Key: {s3_key}")
        print(f"   Local path: {local_path}")
        return False

class BiRefNetSegmenter:
    def __init__(
        self,
        model_path: str = None,
        model_name: str = "zhengpeng7/BiRefNet_lite",  # ✅ Use BiRefNet_lite model
        precision: str = "fp16",
        vis_save_dir: str = "infer",
        thickness_threshold: int = 200,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize BiRefNet Segmenter for mannequin detection and removal.
        
        Args:
            model_path: Path to the trained BiRefNet model weights (checkpoint.pt)
            model_name: HuggingFace model name for BiRefNet (default: BiRefNet_lite)
            precision: Model precision ("fp32", "fp16", "bf16")
            vis_save_dir: Directory for saving visualization outputs
            thickness_threshold: Threshold for removing thin artifacts
            mask_threshold: Threshold for mask binarization
        """
        os.makedirs(vis_save_dir, exist_ok=True)
        self.vis_save_dir = vis_save_dir
        self.precision = precision
        self.dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
        self.thickness_threshold = thickness_threshold
        self.mask_threshold = mask_threshold
        self.model_name = model_name

        # Initialize logger
        self.logger = AppLogger()
        
        # Enhanced Device setup with detailed detection
        self.logger.log("Initializing BiRefNet Segmenter")
        print("Initializing BiRefNet Segmenter")
        
        # Check if CPU is forced via environment variable
        force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        
        if force_cpu:
            self.device = torch.device("cpu")
            self.logger.log("🔧 FORCE_CPU enabled - Using CPU for horizontal scaling")
            print("🔧 FORCE_CPU enabled - Using CPU for horizontal scaling")
        elif torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_id = 0
            self.device = torch.device(f"cuda:{device_id}")
            device_name = torch.cuda.get_device_name(device_id)
            memory_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            
            self.logger.log(f"Using GPU: {device_name} (Device {device_id}/{device_count})")
            self.logger.log(f"Total GPU Memory: {memory_gb:.1f}GB")
            print(f"Using GPU: {device_name} (Device {device_id}/{device_count})")
            print(f"Total GPU Memory: {memory_gb:.1f}GB")
            
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
            self.logger.log("Using CPU (no GPU acceleration available)")
            print("Using CPU (no GPU acceleration available)")
        
        # 🚀 Load BiRefNet_lite model using shared cache
        try:
            self.logger.log(f"Loading BiRefNet_lite model: {model_name}")
            print(f"Loading BiRefNet_lite model: {model_name}")

            # Use cached model to avoid re-downloading for each instance
            self.model = _get_cached_model(model_name, self.device)
            self.model.eval()
            
            # Apply half precision if using CUDA and fp16
            if self.device.type == 'cuda' and self.precision == 'fp16':
                self.model.half()
                self.logger.log("Applied fp16 precision")
                print("Applied fp16 precision")
            elif self.precision == 'fp16' and self.device.type == 'cpu':
                self.logger.log("⚠️ fp16 not supported on CPU, using fp32")
                print("⚠️ fp16 not supported on CPU, using fp32")
                self.precision = 'fp32'  # Override to fp32 for CPU
                self.dtype = torch.float32
                
        except Exception as e:
            self.logger.log(f"Error loading BiRefNet_lite model: {e}")
            print(f"Error loading BiRefNet_lite model: {e}")
            raise e
        
        # ✅ 2. Load custom checkpoint from S3 if provided
        if model_path:
            self.logger.log(f"🎯 Attempting to load custom checkpoint: {model_path}")
            print(f"🎯 Attempting to load custom checkpoint: {model_path}")
            
            # If model doesn't exist locally, try downloading from S3
            if not os.path.exists(model_path):
                self.logger.log(f"📥 Checkpoint not found locally, attempting S3 download...")
                print(f"📥 Checkpoint not found locally, attempting S3 download...")
                
                # S3 configuration for the new model
                s3_bucket = "artifactsredi"
                s3_key = "models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt"
                
                # Download from S3
                download_success = _download_model_from_s3(s3_bucket, s3_key, model_path)
                
                if not download_success:
                    self.logger.log("❌ Failed to download checkpoint from S3, using pretrained weights")
                    print("❌ Failed to download checkpoint from S3, using pretrained weights")
                    model_path = None  # Fall back to pretrained

            # ✅ 3. Load custom trained weights
            if model_path and os.path.exists(model_path):
                try:
                    self.logger.log(f"🚀 Loading custom checkpoint from: {model_path}")
                    print(f"🚀 Loading custom checkpoint from: {model_path}")
                    
                    # Try loading with weights_only=False first (full compatibility)
                    try:
                        checkpoint = torch.load(
                            model_path, 
                            map_location=self.device, 
                            weights_only=False
                        )
                    except Exception as weights_error:
                        self.logger.log(f"⚠️ Standard loading failed, trying with safe globals: {weights_error}")
                        print(f"⚠️ Standard loading failed, trying with safe globals: {weights_error}")
                        
                        # Try with weights_only=True and proper safe globals
                        try:
                            # Add safe globals for numpy compatibility
                            import numpy as np
                            safe_globals = [
                                np.core.multiarray.scalar,
                                np.core.multiarray._reconstruct,
                                np.ndarray,
                                np.dtype,
                                np.core.multiarray.scalar
                            ]
                            
                            # Try with safe globals context manager
                            with torch.serialization.safe_globals(safe_globals):
                                checkpoint = torch.load(
                                    model_path, 
                                    map_location=self.device, 
                                    weights_only=True
                                )
                                
                        except Exception as safe_error:
                            self.logger.log(f"⚠️ Safe loading also failed: {safe_error}")
                            print(f"⚠️ Safe loading also failed: {safe_error}")
                            
                            # Final fallback: try weights_only=False explicitly
                            try:
                                checkpoint = torch.load(
                                    model_path, 
                                    map_location=self.device, 
                                    weights_only=False
                                )
                                self.logger.log("✅ Fallback loading with weights_only=False succeeded")
                                print("✅ Fallback loading with weights_only=False succeeded")
                            except Exception as final_error:
                                self.logger.log(f"❌ All loading methods failed: {final_error}")
                                print(f"❌ All loading methods failed: {final_error}")
                                raise final_error
                    
                    # Load model state dict (following the user's example)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Direct state dict loading
                        self.model.load_state_dict(checkpoint)
                    
                    self.logger.log("✅ Custom trained weights loaded successfully")
                    print("✅ Custom trained weights loaded successfully")
                    
                except Exception as e:
                    self.logger.log(f"❌ Error loading custom checkpoint: {e}")
                    print(f"❌ Error loading custom checkpoint: {e}")
                    print(f"❌ Traceback: {traceback.format_exc()}")
                    self.logger.log("⚠️ Continuing with pretrained weights")
                    print("⚠️ Continuing with pretrained weights")
            else:
                self.logger.log("⚠️ No valid checkpoint path, using pretrained BiRefNet_lite")
                print("⚠️ No valid checkpoint path, using pretrained BiRefNet_lite")
        else:
            self.logger.log("ℹ️ No checkpoint path provided, using pretrained BiRefNet_lite")
            print("ℹ️ No checkpoint path provided, using pretrained BiRefNet_lite")
        
        # Skip processor initialization - use manual preprocessing
        self.processor = None
        self.logger.log("Using manual preprocessing (matching notebook approach)")
        print("Using manual preprocessing (matching notebook approach)")
        
        self.logger.log("BiRefNet_lite Segmenter initialized successfully")
        print("BiRefNet_lite Segmenter initialized successfully")

    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "precision": self.precision,
                "parameters": total_params,
                "trainable_parameters": trainable_params,
                "architecture": "BiRefNet_lite for mannequin segmentation",
                "image_size": 512,  # Fixed input size for BiRefNet
                "thickness_threshold": self.thickness_threshold,
                "mask_threshold": self.mask_threshold
            }
        except Exception as e:
            self.logger.log(f"Error getting model info: {e}")
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "precision": self.precision,
                "parameters": "unknown",
                "trainable_parameters": "unknown",
                "architecture": "BiRefNet_lite for mannequin segmentation",
                "image_size": 512,
                "thickness_threshold": self.thickness_threshold,
                "mask_threshold": self.mask_threshold,
                "error": str(e)
            }

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for BiRefNet inference.
        
        Args:
            img: Input image in RGB format
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Resize and normalize (using PIL for consistency with notebook)
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((512, 512), Image.Resampling.BILINEAR)
            
            # Convert to tensor and normalize
            img_tensor = torch.tensor(np.array(pil_img), dtype=self.dtype) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
            
            # Move to device
            img_tensor = img_tensor.to(self.device)
            
            # ✅ MINIMAL LOGGING for parallel performance
            # self.logger.log(f"Input image shape: {img.shape}, resized to: {pil_img.size}")
            # print(f"Input image shape: {img.shape}, resized to: {pil_img.size}")
            
            # Convert to appropriate precision and memory layout
            img_tensor = img_tensor.to(dtype=self.dtype)
            if torch.cuda.is_available():
                img_tensor = img_tensor.to(memory_format=torch.channels_last)
            
            # ✅ MINIMAL LOGGING for parallel performance  
            # self.logger.log(f"Preprocessing output shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            # print(f"Preprocessing output shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            
            return img_tensor
            
        except Exception as e:
            self.logger.log(f"Error in _preprocess_image: {e}")
            print(f"Error in _preprocess_image: {e}")
            raise

    def _preprocess_batch_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess multiple images for batch BiRefNet inference.
        
        Args:
            images: List of input images in BGR format
            
        Returns:
            Batch tensor with shape [batch_size, 3, 512, 512]
        """
        try:
            preprocessed_images = []
            
            for i, img in enumerate(images):
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and resize to 512x512
                pil_img = Image.fromarray(img_rgb).convert("RGB").resize((512, 512))
                
                # Convert to tensor (no unsqueeze - we'll stack later)
                img_tensor = F.to_tensor(pil_img).to(self.device)
            
            # Apply half precision if needed
            if self.precision == 'fp16' and self.device.type == 'cuda':
                img_tensor = img_tensor.half()
            
                preprocessed_images.append(img_tensor)
            
            # Stack all images into a batch tensor
            batch_tensor = torch.stack(preprocessed_images, dim=0)
            
            # ✅ MINIMAL LOGGING for parallel performance
            # self.logger.log(f"Batch preprocessing output shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
            # print(f"Batch preprocessing output shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
            
            return batch_tensor
            
        except Exception as e:
            self.logger.log(f"Error in _preprocess_batch_images: {e}")
            print(f"Error in _preprocess_batch_images: {e}")
            raise

    def _extract_birefnet_output(self, logits):
        """
        Extract the correct output tensor from BiRefNet's complex output structure.
        Handles both single image and batch processing cases.
        Based on the notebook's extract_birefnet_output function.
        """
        # Strategy 1: Direct tensor
        if hasattr(logits, 'shape'):
            # Check if this is batch or single
            if len(logits.shape) >= 3:  # Batch case: [batch, H, W] or [batch, C, H, W]
                return logits
            elif len(logits.shape) == 2:  # Single case: [H, W]
                return logits

        # Strategy 2: Look for correct tensor shape (batch or single 512x512)
        def find_target_tensor(obj):
            if hasattr(obj, 'shape') and len(obj.shape) >= 2:
                # For batch: [batch_size, 512, 512] or [batch_size, 1, 512, 512]
                if len(obj.shape) >= 3 and obj.shape[-2:] == (512, 512):
                    return obj
                # For single: [512, 512] or [1, 512, 512]
                elif obj.shape[-2:] == (512, 512):
                    return obj
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    result = find_target_tensor(item)
                    if result is not None:
                        return result
            return None

        result = find_target_tensor(logits)
        if result is not None:
            return result

        # Strategy 3: Take the last tensor we can find
        def find_any_tensor(obj):
            if hasattr(obj, 'shape'):
                return obj
            elif isinstance(obj, (list, tuple)):
                for item in reversed(obj):  # Try from the end
                    result = find_any_tensor(item)
                    if result is not None:
                        return result
            return None

        return find_any_tensor(logits)

    def _run_inference(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run BiRefNet inference on preprocessed image.
        
        Args:
            img_tensor: Preprocessed image tensor
            
        Returns:
            Segmentation mask tensor
        """
        with torch.no_grad(), torch.inference_mode():
            try:
                # Use autocast for fp16 performance optimization
                autocast_dtype = torch.float16 if torch.cuda.is_available() and self.precision == 'fp16' else None
                with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    outputs = self.model(img_tensor)
                
                # Use the extraction function from the notebook
                extracted_logits = self._extract_birefnet_output(outputs)
                
                if extracted_logits is None:
                    self.logger.log("Warning: Could not extract valid output from BiRefNet")
                    print("Warning: Could not extract valid output from BiRefNet")
                    # Return empty mask as fallback
                    return torch.zeros((512, 512), device=self.device, dtype=img_tensor.dtype)
                
                # Apply sigmoid to get probabilities
                mask = torch.sigmoid(extracted_logits)
                
                # Remove any extra dimensions and ensure 2D output
                while mask.ndim > 2:
                    mask = mask.squeeze(0)
                
                # CRITICAL: Move to CPU immediately to free GPU memory
                mask_cpu = mask.detach().cpu()
                
                # ✅ MINIMAL LOGGING for parallel performance
                # self.logger.log(f"Inference output shape: {mask_cpu.shape}, dtype: {mask_cpu.dtype}")
                # print(f"Inference output shape: {mask_cpu.shape}, dtype: {mask_cpu.dtype}")
                
                # Explicit GPU memory cleanup - only delete if variables exist
                try:
                    del outputs, extracted_logits, mask
                except NameError:
                    pass  # Variables may not exist if there was an earlier exception
                
                # GPU memory cleanup after inference (NO SYNCHRONIZE for parallel execution!)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
                
                return mask_cpu
                
            except Exception as e:
                import traceback
                full_error = traceback.format_exc()
                self.logger.log(f"Error during BiRefNet inference: {e}")
                self.logger.log(f"Full stack trace: {full_error}")
                print(f"Error during BiRefNet inference: {e}")
                print(f"Full stack trace: {full_error}")
                
                # Only try to log outputs if the variable exists
                try:
                    self.logger.log(f"Model output type: {type(outputs)}")
                    print(f"Model output type: {type(outputs)}")
                    if isinstance(outputs, (list, tuple)):
                        self.logger.log(f"Output length: {len(outputs)}")
                        print(f"Output length: {len(outputs)}")
                        for i, item in enumerate(outputs[:3]):  # Show first 3 items
                            self.logger.log(f"Item {i} type: {type(item)}")
                            print(f"Item {i} type: {type(item)}")
                            if hasattr(item, 'shape'):
                                self.logger.log(f"Item {i} shape: {item.shape}")
                                print(f"Item {i} shape: {item.shape}")
                except NameError:
                    self.logger.log("Model outputs variable not defined due to early failure")
                    print("Model outputs variable not defined due to early failure")
                
                # GPU memory cleanup even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # torch.cuda.synchronize()
                    
                raise

    def _run_batch_inference(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run BiRefNet inference on batch of preprocessed images.
        
        Args:
            batch_tensor: Batch tensor with shape [batch_size, 3, 512, 512]
            
        Returns:
            Batch segmentation masks tensor [batch_size, 512, 512]
        """
        with torch.no_grad(), torch.inference_mode():
            try:
                # Use autocast for fp16 performance optimization
                autocast_dtype = torch.float16 if torch.cuda.is_available() and self.precision == 'fp16' else None
                
                with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    # TRUE BATCH INFERENCE - process all images in one forward pass
                    outputs = self.model(batch_tensor)
                
                # Debug: Log model output info
                self.logger.log(f"🔍 Model output type: {type(outputs)}")
                print(f"🔍 Model output type: {type(outputs)}")
                if hasattr(outputs, 'shape'):
                    self.logger.log(f"🔍 Model output shape: {outputs.shape}")
                    print(f"🔍 Model output shape: {outputs.shape}")
                elif isinstance(outputs, (list, tuple)):
                    self.logger.log(f"🔍 Model output list length: {len(outputs)}")
                    print(f"🔍 Model output list length: {len(outputs)}")
                    for i, item in enumerate(outputs[:3]):  # Show first 3 items
                        if hasattr(item, 'shape'):
                            self.logger.log(f"🔍 Item {i} shape: {item.shape}")
                            print(f"🔍 Item {i} shape: {item.shape}")
                
                # Use the extraction function from the notebook
                extracted_logits = self._extract_birefnet_output(outputs)
                
                if extracted_logits is None:
                    self.logger.log("❌ Warning: Could not extract valid output from BiRefNet batch")
                    print("❌ Warning: Could not extract valid output from BiRefNet batch")
                    # Return empty masks as fallback - CRITICAL: on CPU
                    batch_size = batch_tensor.shape[0]
                    return torch.zeros((batch_size, 512, 512), dtype=torch.float32)  # CPU tensor
                
                self.logger.log(f"🔍 Extracted logits shape: {extracted_logits.shape}")
                print(f"🔍 Extracted logits shape: {extracted_logits.shape}")
                
                # Apply sigmoid to get probabilities
                batch_masks = torch.sigmoid(extracted_logits)
                
                # Ensure correct shape: [batch_size, 512, 512]
                original_shape = batch_masks.shape
                if batch_masks.ndim == 4 and batch_masks.shape[1] == 1:
                    # Remove channel dimension: [batch, 1, H, W] -> [batch, H, W]
                    batch_masks = batch_masks.squeeze(1)
                elif batch_masks.ndim == 3:
                    # Already correct: [batch, H, W]
                    pass
                elif batch_masks.ndim == 2:
                    # Single image case: [H, W] -> [1, H, W]
                    batch_masks = batch_masks.unsqueeze(0)
                else:
                    self.logger.log(f"❌ Unexpected batch mask shape: {batch_masks.shape}")
                    print(f"❌ Unexpected batch mask shape: {batch_masks.shape}")
                    # Try to reshape to expected format
                    if batch_masks.numel() == batch_tensor.shape[0] * 512 * 512:
                        batch_masks = batch_masks.view(batch_tensor.shape[0], 512, 512)
                    else:
                        raise ValueError(f"Cannot reshape {batch_masks.shape} to batch format")
                
                # CRITICAL: Move to CPU immediately to free GPU memory
                batch_masks_cpu = batch_masks.detach().cpu()
                
                self.logger.log(f"✅ Batch inference output: {original_shape} -> {batch_masks_cpu.shape}, dtype: {batch_masks_cpu.dtype}")
                print(f"✅ Batch inference output: {original_shape} -> {batch_masks_cpu.shape}, dtype: {batch_masks_cpu.dtype}")
                
                # Explicit GPU memory cleanup - only delete if variables exist
                try:
                    del outputs, extracted_logits, batch_masks, batch_tensor
                except NameError:
                    pass  # Variables may not exist if there was an earlier exception
                
                return batch_masks_cpu
                
            except Exception as e:
                self.logger.log(f"Error during BiRefNet batch inference: {e}")
                print(f"Error during BiRefNet batch inference: {e}")
                
                # GPU memory cleanup even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # torch.cuda.synchronize()
                
                raise e

    def _extract_mannequin_masks(self, mask_tensor: torch.Tensor, img_shape: tuple) -> np.ndarray:
        """
        Extract mannequin masks from BiRefNet predictions.
        
        Args:
            mask_tensor: BiRefNet output mask tensor
            img_shape: Original image shape (H, W)
            
        Returns:
            Binary mannequin mask
        """
        try:
            # Convert to numpy and ensure proper format
            mask_np = mask_tensor.numpy()
            
            # ✅ MINIMAL LOGGING for parallel performance
            # self.logger.log(f"Initial mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            # print(f"Initial mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            
            # Remove extra dimensions
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            if mask_np.ndim != 2:
                self.logger.log(f"Warning: Unexpected mask dimensions: {mask_np.shape}")
                print(f"Warning: Unexpected mask dimensions: {mask_np.shape}")
                return np.zeros(img_shape, dtype=bool)
            
            # ✅ MINIMAL LOGGING for parallel performance
            # self.logger.log(f"Processed mask shape: {mask_np.shape}")
            # print(f"Processed mask shape: {mask_np.shape}")
            
            # Resize to original image shape
            if mask_np.shape != img_shape:
                mask_resized = cv2.resize(mask_np, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask_np
            
            # Ensure values are in [0, 1] range
            mask_resized = np.clip(mask_resized, 0, 1)
            
            # Binarize mask
            binary_mask = mask_resized > self.mask_threshold
            
            # ✅ MINIMAL LOGGING for parallel performance
            # msg1 = f"Final mask shape: {mask_resized.shape}, threshold: {self.mask_threshold}"
            # msg2 = f"Final mask: {np.sum(binary_mask)} mannequin pixels / {binary_mask.size} total"
            # self.logger.log(msg1)
            # print(msg1)
            # self.logger.log(msg2)
            # print(msg2)
            
            return binary_mask
            
        except Exception as e:
            self.logger.log(f"Error in _extract_mannequin_masks: {e}")
            print(f"Error in _extract_mannequin_masks: {e}")
            # Return empty mask as fallback
            return np.zeros(img_shape, dtype=bool)

    def apply_masks_to_remove_unwanted_areas(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply mask to remove unwanted areas (mannequins) from image.
        
        Args:
            img: Original image
            mask: Binary mask of areas to remove
            
        Returns:
            Processed image with unwanted areas replaced by white
        """
        processed_img = img.copy()
        mask_bool = mask.astype(bool)
        
        if processed_img.ndim == 3:
            processed_img[mask_bool] = [255, 255, 255]
        else:
            processed_img[mask_bool] = 255
            
        return processed_img

    def _filter_components(
        self,
        mask: np.ndarray,
        min_size: int = 400,
        keep_largest: bool = False,
        connectivity: int = 8,
    ) -> np.ndarray:
        """
        Remove small connected components *or* keep only the largest one.

        Args:
            mask: binary mask (bool / 0-1) where True==foreground.
            min_size: minimum pixel area to keep (ignored if keep_largest=True).
            keep_largest: if True keep only the biggest component, otherwise drop
                          everything smaller than min_size.
            connectivity: 4 or 8 (8 gives diagonals connectivity).

        Returns:
            Cleaned binary mask (same shape as input).
        """
        # Make sure we have uint8 {0,1} for OpenCV
        mask_uint8 = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=connectivity
        )

        # stats[:, cv2.CC_STAT_AREA] => pixel count per label (label 0 is background)
        if keep_largest and num_labels > 1:
            # Choose foreground label with largest area
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = labels == largest_label
        else:
            # Keep everything >= min_size
            keep = stats[:, cv2.CC_STAT_AREA] >= min_size
            keep[0] = False  # never keep background
            cleaned = keep[labels]

        return cleaned.astype(bool)

    def remove_thin_stripes(
        self,
        img: np.ndarray,
        thickness_threshold: int = None,
        method: str = "morphology"
    ) -> np.ndarray:
        """
        Remove thin artifacts and stripes from processed image.
        Standard image cleaning implementation.
        """
        if thickness_threshold is None:
            thickness_threshold = self.thickness_threshold
            
        cleaned_img = img.copy()
        
        if method == "morphology":
            if img.ndim == 3:
                non_white_mask = ~np.all(img == [255, 255, 255], axis=2)
            else:
                non_white_mask = img != 255
                
            kernel_size = max(1, thickness_threshold // 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened_mask = cv2.morphologyEx(non_white_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            removed_areas = non_white_mask & ~opened_mask.astype(bool)
            # removed_areas = self._filter_components(
            #     removed_areas,
            #     min_size=2000,
            #     keep_largest=False,   # change to True if you prefer “largest only”
            # )
            
            if cleaned_img.ndim == 3:
                cleaned_img[removed_areas] = [255, 255, 255]
            else:
                cleaned_img[removed_areas] = 255
                
        elif method == "connected_components":
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                non_white_mask = gray != 255
            else:
                non_white_mask = img != 255
                
            num_labels, labels = cv2.connectedComponents(non_white_mask.astype(np.uint8))
            
            for label in range(1, num_labels):
                component_mask = labels == label
                area = np.sum(component_mask)
                
                if area < thickness_threshold * thickness_threshold:
                    if cleaned_img.ndim == 3:
                        cleaned_img[component_mask] = [255, 255, 255]
                    else:
                        cleaned_img[component_mask] = 255
                        
        return cleaned_img

    def process_image_url(self, image_url: str, plot: bool = True) -> np.ndarray:
        """
        Process an image from URL to remove mannequins using BiRefNet.
        
        Args:
            image_url: URL of the image to process
            plot: Whether to show visualization plots
            
        Returns:
            Processed image with mannequins removed
        """
        # ✅ USE SHARED CACHE - No more redundant downloads or file I/O!
        img = _get_cached_image(image_url)
        if img is None:
            return None

        # Preprocess image for BiRefNet
        img_tensor = self._preprocess_image(img)
        
        # Run inference (minimal logging for parallel performance)
        mask_tensor = self._run_inference(img_tensor)
        
        # Extract mannequin masks
        mannequin_mask = self._extract_mannequin_masks(mask_tensor, img.shape[:2])
        
        # Apply masks to remove unwanted areas
        processed_img = self.apply_masks_to_remove_unwanted_areas(img, mannequin_mask)
        
        # Remove thin artifacts
        cleaned_img = self.remove_thin_stripes(
            processed_img,
            thickness_threshold=self.thickness_threshold,
            method="morphology"
        )



        # Visualization
        if plot:
            plt.figure(figsize=(20, 4))
            
            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original image")
            plt.axis("off")
            
            plt.subplot(1, 4, 2)
            plt.imshow(mannequin_mask, cmap='gray')
            plt.title("Mannequin mask")
            plt.axis("off")
            
            plt.subplot(1, 4, 3)
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            plt.title("After masking")
            plt.axis("off")
            
            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
            plt.title(f"After cleaning (threshold={self.thickness_threshold})")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()

        # Cleanup - no files to remove since we use shared cache
            
        # GPU memory cleanup after processing (NO SYNCHRONIZE for parallel execution!)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
        # Periodic aggressive cache clearing to prevent memory fragmentation
        if not hasattr(self, '_image_counter'):
            self._image_counter = 0
        self._image_counter += 1
        
        # Every 50 images, do aggressive cleanup (reduced frequency for performance)
        if self._image_counter % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Only call ipc_collect occasionally as it's expensive
                if self._image_counter % 100 == 0:
                    torch.cuda.ipc_collect()
            gc.collect()
            
        return cleaned_img

    def process_image_array(self, img: np.ndarray, plot: bool = True) -> np.ndarray:
        """
        Process an image array to remove mannequins using BiRefNet.
        
        Args:
            img: Input image as numpy array in BGR format
            plot: Whether to show visualization plots
            
        Returns:
            Processed image with mannequins removed
        """
        self.logger.log("Processing image array...")
        print("Processing image array...")
        
        # Preprocess image for BiRefNet
        img_tensor = self._preprocess_image(img)
        
        # Run inference
        self.logger.log("Running BiRefNet inference...")
        print("Running BiRefNet inference...")
        mask_tensor = self._run_inference(img_tensor)
        
        # Extract mannequin masks
        mannequin_mask = self._extract_mannequin_masks(mask_tensor, img.shape[:2])
        
        # Apply masks to remove unwanted areas
        processed_img = self.apply_masks_to_remove_unwanted_areas(img, mannequin_mask)
        
        # Remove thin artifacts
        cleaned_img = self.remove_thin_stripes(
            processed_img,
            thickness_threshold=self.thickness_threshold,
            method="morphology"
        )

        # Visualization
        if plot:
            plt.figure(figsize=(20, 4))
            
            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original image")
            plt.axis("off")
            
            plt.subplot(1, 4, 2)
            plt.imshow(mannequin_mask, cmap='gray')
            plt.title("Mannequin mask")
            plt.axis("off")
            
            plt.subplot(1, 4, 3)
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            plt.title("After masking")
            plt.axis("off")
            
            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
            plt.title(f"After cleaning (threshold={self.thickness_threshold})")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            
        # GPU memory cleanup after processing
        # Memory cleanup after batch processing (NO SYNCHRONIZE for parallel execution!)  
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
        self.logger.log("Finished processing image array")
        print("Finished processing image array")
        return cleaned_img

    def process_batch_urls(self, image_urls, plot=False, max_batch_size=20):
        """
        Process a batch of image URLs using proven single image processing.
        
        Args:
            image_urls: List of image URLs to process
            plot: Whether to show plots for visualization
            max_batch_size: Maximum number of images to process in one batch
            
        Returns:
            List of processed images (numpy arrays)
        """
        self.logger.log(f"Starting batch processing of {len(image_urls)} images")
        print(f"Starting batch processing of {len(image_urls)} images")
        
        if len(image_urls) > max_batch_size:
            self.logger.log(f"Warning: Batch size {len(image_urls)} exceeds maximum {max_batch_size}, truncating")
            print(f"Warning: Batch size {len(image_urls)} exceeds maximum {max_batch_size}, truncating")
            image_urls = image_urls[:max_batch_size]
        
        processed_images = []
        failed_count = 0
        
        try:
            # Download all images first
            self.logger.log("Step 1: Downloading all images...")
            print("Step 1: Downloading all images...")
            
            downloaded_images = []
            valid_indices = []
            
            # Download images in parallel for better performance
            def download_single_image(url_index_pair):
                i, url = url_index_pair
                try:
                    self.logger.log(f"Downloading image {i+1}/{len(image_urls)}: {url}")
                    print(f"Downloading image {i+1}/{len(image_urls)}: {url}")
                    
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    return i, image, None
                    
                except Exception as e:
                    error_msg = f"Failed to download image {i+1}: {str(e)}"
                    self.logger.log(error_msg)
                    print(error_msg)
                    return i, None, str(e)
            
            # Download images in parallel (max 3 concurrent to be conservative)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                download_tasks = [(i, url) for i, url in enumerate(image_urls)]
                download_results = list(executor.map(download_single_image, download_tasks))
            
            # Process download results in original order
            for i, image, error in download_results:
                if image is not None:
                    downloaded_images.append(image)
                    valid_indices.append(i)
                else:
                    failed_count += 1
            
            if not downloaded_images:
                self.logger.log("No images successfully downloaded")
                print("No images successfully downloaded")
                return []
            
            self.logger.log(f"Successfully downloaded {len(downloaded_images)} images, processing batch...")
            print(f"Successfully downloaded {len(downloaded_images)} images, processing batch...")
            
            # Process images in batch
            batch_tensors = []
            original_sizes = []
            
            # Use new TRUE BATCH PROCESSING for optimal GPU utilization
            self.logger.log(f"Step 2: Processing {len(downloaded_images)} images using TRUE BATCH PROCESSING...")
            print(f"Step 2: Processing {len(downloaded_images)} images using TRUE BATCH PROCESSING...")
            
            try:
                # Convert PIL images to numpy arrays (BGR format for BiRefNet)
                image_arrays = []
                for image in downloaded_images:
                    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_arrays.append(image_np)
                
                # Use the new batch processing method for true GPU parallelization
                batch_results = self.process_image_arrays_batch(image_arrays, plot=False)
                
                # Count successful results
                for i, result_img in enumerate(batch_results):
                    if result_img is not None:
                        processed_images.append(result_img)
                        self.logger.log(f"Successfully processed image {i+1}/{len(downloaded_images)} in batch")
                        print(f"Successfully processed image {i+1}/{len(downloaded_images)} in batch")
                    else:
                        self.logger.log(f"Failed to process image {i+1} in batch: result was None")
                        print(f"Failed to process image {i+1} in batch: result was None")
                        failed_count += 1
                        
            except Exception as batch_error:
                self.logger.log(f"Batch processing failed, falling back to sequential: {batch_error}")
                print(f"Batch processing failed, falling back to sequential: {batch_error}")
                
                # Fallback to sequential processing
            for i, image in enumerate(downloaded_images):
                try:
                    self.logger.log(f"Processing image {i+1}/{len(downloaded_images)} (sequential fallback)")
                    print(f"Processing image {i+1}/{len(downloaded_images)} (sequential fallback)")
                    
                    # Convert PIL image to numpy array (BGR format for BiRefNet)
                    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Use the proven single image processing method
                    result_img = self.process_image_array(image_np, plot=False)
                    
                    if result_img is not None:
                        processed_images.append(result_img)
                        self.logger.log(f"Successfully processed image {i+1}/{len(downloaded_images)} (fallback)")
                        print(f"Successfully processed image {i+1}/{len(downloaded_images)} (fallback)")
                    else:
                        self.logger.log(f"Failed to process image {i+1}: process_image_array returned None")
                        print(f"Failed to process image {i+1}: process_image_array returned None")
                        failed_count += 1
                    
                except Exception as e:
                    self.logger.log(f"Failed to process image {i+1} (fallback): {str(e)}")
                    print(f"Failed to process image {i+1} (fallback): {str(e)}")
                    failed_count += 1
                    continue
            
            success_count = len(processed_images)
            total_requested = len(image_urls)
            
            self.logger.log(f"Batch processing completed: {success_count}/{total_requested} successful, {failed_count} failed")
            print(f"Batch processing completed: {success_count}/{total_requested} successful, {failed_count} failed")
            
            # Final GPU memory cleanup (NO SYNCHRONIZE for parallel execution!)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
            return processed_images
            
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.logger.log(error_msg)
            print(error_msg)
            
            # Cleanup GPU memory even on error (NO SYNCHRONIZE for parallel execution!)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
            return processed_images  # Return whatever we managed to process
    
    def _clean_mask(self, mask):
        """Clean mask by removing thin artifacts."""
        import cv2
        
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Remove thin structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Opening to remove thin connections
        mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find contours and filter by area
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean mask
        clean_mask = np.zeros_like(mask_uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.thickness_threshold:  # Keep only substantial areas
                cv2.fillPoly(clean_mask, [contour], 255)
        
        return (clean_mask / 255.0).astype(np.float32)
    
    def _create_clothing_preserved_result(self, image, mask):
        """Create result where mannequin body is white, clothing remains visible."""
        # Invert mask so clothing areas are preserved
        clothing_mask = 1.0 - mask
        
        # Create white background
        white_background = np.ones_like(image) * 255
        
        # Apply mask: white where mannequin body, original image where clothing
        result = image * clothing_mask[..., np.newaxis] + white_background * mask[..., np.newaxis]
        
        return result.astype(np.uint8)

    def process_image_arrays_batch(self, image_arrays: List[np.ndarray], plot: bool = False) -> List[np.ndarray]:
        """
        Process multiple image arrays using true batch inference for optimal GPU utilization.
        
        Args:
            image_arrays: List of images in BGR format
            plot: Whether to show plots for visualization
            
        Returns:
            List of processed images (numpy arrays)
        """
        try:
            if not image_arrays:
                return []
            
            batch_size = len(image_arrays)
            self.logger.log(f"🚀 TRUE BATCH PROCESSING: {batch_size} images in single GPU forward pass")
            print(f"🚀 TRUE BATCH PROCESSING: {batch_size} images in single GPU forward pass")
            
            # Step 1: Batch preprocessing - all images to one tensor
            batch_start = time.time()
            batch_tensor = self._preprocess_batch_images(image_arrays)
            preprocess_time = time.time() - batch_start
            
            # Step 2: Batch inference - single model.forward() call
            inference_start = time.time()
            batch_masks = self._run_batch_inference(batch_tensor)
            inference_time = time.time() - inference_start
            
            # Step 3: Individual postprocessing for each image
            postprocess_start = time.time()
            processed_images = []
            
            for i, (img, mask_tensor) in enumerate(zip(image_arrays, batch_masks)):
                try:
                    # Extract mannequin masks
                    mannequin_mask = self._extract_mannequin_masks(mask_tensor, img.shape[:2])
                    
                    # Apply masks to remove unwanted areas
                    processed_img = self.apply_masks_to_remove_unwanted_areas(img, mannequin_mask)
                    
                    # Remove thin artifacts
                    cleaned_img = self.remove_thin_stripes(
                        processed_img,
                        thickness_threshold=self.thickness_threshold,
                        method="morphology"
                    )
                    
                    processed_images.append(cleaned_img)
                    
                except Exception as e:
                    self.logger.log(f"Error processing image {i+1} in batch: {e}")
                    print(f"Error processing image {i+1} in batch: {e}")
                    processed_images.append(None)
            
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - batch_start
            
            # Performance logging
            self.logger.log(f"⚡ BATCH PERFORMANCE BREAKDOWN:")
            self.logger.log(f"   Preprocessing: {preprocess_time:.3f}s ({preprocess_time/total_time*100:.1f}%)")
            self.logger.log(f"   Inference: {inference_time:.3f}s ({inference_time/total_time*100:.1f}%)")
            self.logger.log(f"   Postprocessing: {postprocess_time:.3f}s ({postprocess_time/total_time*100:.1f}%)")
            self.logger.log(f"   Total: {total_time:.3f}s")
            self.logger.log(f"   Throughput: {batch_size/total_time:.2f} images/second")
            self.logger.log(f"   Time per image: {total_time/batch_size:.3f}s")
            
            print(f"⚡ BATCH PERFORMANCE BREAKDOWN:")
            print(f"   Preprocessing: {preprocess_time:.3f}s ({preprocess_time/total_time*100:.1f}%)")
            print(f"   Inference: {inference_time:.3f}s ({inference_time/total_time*100:.1f}%)")
            print(f"   Postprocessing: {postprocess_time:.3f}s ({postprocess_time/total_time*100:.1f}%)")
            print(f"   Total: {total_time:.3f}s")
            print(f"   Throughput: {batch_size/total_time:.2f} images/second")
            print(f"   Time per image: {total_time/batch_size:.3f}s")
            
            # GPU memory cleanup (NO SYNCHRONIZE for parallel execution!)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
            return processed_images
            
        except Exception as e:
            self.logger.log(f"Error in batch processing: {e}")
            print(f"Error in batch processing: {e}")
            
            # GPU memory cleanup even on error (NO SYNCHRONIZE for parallel execution!)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
            # Fallback to sequential processing
            self.logger.log("Falling back to sequential processing...")
            print("Falling back to sequential processing...")
            return [self.process_image_array(img, plot=False) for img in image_arrays]