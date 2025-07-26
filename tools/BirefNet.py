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

class BiRefNetSegmenter:
    def __init__(
        self,
        model_path: str = None,
        model_name: str = "zhengpeng7/BiRefNet",
        precision: str = "fp16",
        vis_save_dir: str = "infer",
        thickness_threshold: int = 200,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize BiRefNet Segmenter for mannequin detection and removal.
        
        Args:
            model_path: Path to the trained BiRefNet model weights (checkpoint.pt)
            model_name: HuggingFace model name for BiRefNet
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
        
        # Set performance environment variables programmatically (fallback if not set)
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # Temporarily disabled - causes PyTorch crash
        if not os.getenv('OMP_NUM_THREADS'):
            os.environ['OMP_NUM_THREADS'] = '8'
        if not os.getenv('MKL_NUM_THREADS'):
            os.environ['MKL_NUM_THREADS'] = '8'
        
        # Detailed GPU detection
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        self.logger.log(f"CUDA available: {cuda_available}")
        self.logger.log(f"MPS available: {mps_available}")
        print(f"CUDA available: {cuda_available}")
        print(f"MPS available: {mps_available}")
        
        if cuda_available:
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            self.logger.log(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # GPU Memory Management Optimization
            torch.cuda.set_per_process_memory_fraction(0.9)  # Leave space for driver
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
            self.logger.log("GPU memory management optimizations enabled")
            
        elif mps_available:
            self.device = torch.device("mps")
            self.logger.log("Using MPS (Apple Silicon GPU)")
            print("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            self.logger.log("Using CPU (no GPU acceleration available)")
            print("Using CPU (no GPU acceleration available)")
        
        # Initialize BiRefNet model
        try:
            self.logger.log(f"Loading BiRefNet model: {model_name}")
            print(f"Loading BiRefNet model: {model_name}")

            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                config={'model_type': 'custom_segmentation_model'}
            )
            self.model.to(self.device)
            
            # Apply performance optimizations
            if torch.cuda.is_available():
                # Use channels_last memory format for better GPU utilization
                self.model = self.model.to(memory_format=torch.channels_last)
                
                # DISABLED: torch.compile conflicts with BiRefNet's tensor mutations
                # The model mutates input tensors which causes "skipping cudagraphs due to mutated inputs"
                # try:
                #     self.model = torch.compile(self.model, mode="reduce-overhead")
                #     self.logger.log("Model compiled successfully with torch.compile")
                #     print("Model compiled successfully with torch.compile")
                # except Exception as compile_error:
                #     self.logger.log(f"Model compilation failed: {compile_error}")
                #     print(f"Model compilation failed: {compile_error}")
                self.logger.log("torch.compile disabled - conflicts with BiRefNet tensor mutations")
                print("torch.compile disabled - conflicts with BiRefNet tensor mutations")
            
            self.model.eval()
            # Ensure no gradients are computed for inference
            self.model.requires_grad_(False)
            
            # Apply half precision if using CUDA and fp16 (matching notebook)
            if self.device.type == 'cuda' and self.precision == 'fp16':
                self.model.half()
                
        except Exception as e:
            self.logger.log(f"Error loading BiRefNet model: {e}")
            print(f"Error loading BiRefNet model: {e}")
            raise e
        
        # Load custom checkpoint if provided
        if model_path and os.path.exists(model_path):
            self.logger.log(f"Loading custom BiRefNet weights from: {model_path}")
            print(f"Loading custom BiRefNet weights from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.logger.log("Custom weights loaded successfully")
                print("Custom weights loaded successfully")
            except Exception as e:
                self.logger.log(f"Error loading custom weights: {e}")
                print(f"Error loading custom weights: {e}")
                self.logger.log("Continuing with pretrained weights")
                print("Continuing with pretrained weights")
        else:
            self.logger.log("No custom weights provided, using pretrained model")
            print("No custom weights provided, using pretrained model")
        
        # Skip processor initialization - use manual preprocessing like notebook
        self.processor = None
        self.logger.log("Using manual preprocessing (matching notebook approach)")
        print("Using manual preprocessing (matching notebook approach)")
        
        self.logger.log("BiRefNet Segmenter initialized successfully")
        print("BiRefNet Segmenter initialized successfully")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for BiRefNet inference.
        Based on the notebook preprocessing approach.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize to 512x512 (matching notebook)
            pil_img = Image.fromarray(img_rgb).convert("RGB").resize((512, 512))
            
            self.logger.log(f"Input image shape: {img.shape}, resized to: {pil_img.size}")
            print(f"Input image shape: {img.shape}, resized to: {pil_img.size}")
            
            # Convert to tensor (matching notebook - no explicit normalization)
            img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(self.device)
            
            # Apply half precision if needed
            if self.precision == 'fp16' and self.device.type == 'cuda':
                img_tensor = img_tensor.half()
            
            self.logger.log(f"Preprocessing output shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            print(f"Preprocessing output shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            
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
            batch_tensors = []
            
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
                
                batch_tensors.append(img_tensor)
            
            # Stack all tensors into a batch
            batch_tensor = torch.stack(batch_tensors, dim=0)  # [batch_size, 3, 512, 512]
            
            self.logger.log(f"Batch preprocessing output shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
            print(f"Batch preprocessing output shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
            
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
                
                self.logger.log(f"Inference output shape: {mask_cpu.shape}, dtype: {mask_cpu.dtype}")
                print(f"Inference output shape: {mask_cpu.shape}, dtype: {mask_cpu.dtype}")
                
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
            # Convert tensor to numpy with proper type handling
            # Tensor should already be on CPU from inference functions
            if mask_tensor.dtype == torch.float16:
                mask_np = mask_tensor.float().numpy()
            else:
                mask_np = mask_tensor.numpy()
            
            self.logger.log(f"Initial mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            print(f"Initial mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            
            # Ensure 2D mask
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            if mask_np.ndim != 2:
                self.logger.log(f"Warning: Unexpected mask dimensions: {mask_np.shape}")
                print(f"Warning: Unexpected mask dimensions: {mask_np.shape}")
                return np.zeros(img_shape, dtype=bool)
            
            self.logger.log(f"Processed mask shape: {mask_np.shape}")
            print(f"Processed mask shape: {mask_np.shape}")
            
            # Resize to original image shape
            if mask_np.shape != img_shape:
                mask_resized = cv2.resize(mask_np, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask_np
            
            # Ensure values are in [0, 1] range
            mask_resized = np.clip(mask_resized, 0, 1)
            
            # Binarize mask
            binary_mask = mask_resized > self.mask_threshold
            
            msg1 = f"Final mask shape: {mask_resized.shape}, threshold: {self.mask_threshold}"
            self.logger.log(msg1)
            print(msg1)
            
            msg2 = f"Mask pixels above threshold: {binary_mask.sum()}"
            self.logger.log(msg2)
            print(msg2)
            
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
        # ✅ FIXED: Generate unique filename per request to avoid race conditions
        unique_id = uuid.uuid4().hex
        thread_id = threading.current_thread().ident
        fname = os.path.join(self.vis_save_dir, f"temp_image_{thread_id}_{unique_id}.jpg")
        
        try:
            # Download image
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            with open(fname, "wb") as f:
                f.write(response.content)
            
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise ValueError("Failed to load the image!")
                
            msg = f"Downloaded and loaded image from URL: {image_url}"
            self.logger.log(msg)
            print(msg)
            
        except Exception as e:
            msg = f"Image download/load error: {e} (URL: {image_url})"
            self.logger.log(msg)
            print(msg)
            if os.path.exists(fname):
                os.remove(fname)
            return None

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

        # Cleanup
        if os.path.exists(fname):
            os.remove(fname)
            
        # GPU memory cleanup after processing (NO SYNCHRONIZE for parallel execution!)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            
        # Periodic aggressive cache clearing to prevent memory fragmentation
        if not hasattr(self, '_image_counter'):
            self._image_counter = 0
        self._image_counter += 1
        
        # Every 10 images, do aggressive cleanup (NO SYNCHRONIZE for parallel execution!)
        if self._image_counter % 10 == 0:
            self.logger.log(f"🧹 Aggressive memory cleanup after {self._image_counter} images")
            print(f"🧹 Aggressive memory cleanup after {self._image_counter} images")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # torch.cuda.synchronize()  # ❌ REMOVED: Blocks parallel execution!
            gc.collect()
            
        msg = f"Finished processing image from URL: {image_url}"
        self.logger.log(msg)
        print(msg)
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