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

# Import utilities from tools package
from .logger import AppLogger

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
                
                # Compile model for optimized execution (PyTorch 2.0+)
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self.logger.log("Model compiled successfully with torch.compile")
                    print("Model compiled successfully with torch.compile")
                except Exception as compile_error:
                    self.logger.log(f"Model compilation failed: {compile_error}")
                    print(f"Model compilation failed: {compile_error}")
            
            self.model.eval()
            
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

    def _extract_birefnet_output(self, logits):
        """
        Extract the correct output tensor from BiRefNet's complex output structure.
        Based on the notebook's extract_birefnet_output function.
        """
        # Strategy 1: Direct tensor
        if hasattr(logits, 'shape'):
            return logits

        # Strategy 2: Look for 512x512 tensor (matching notebook training size)
        def find_512x512_tensor(obj):
            if hasattr(obj, 'shape') and len(obj.shape) >= 2 and obj.shape[-2:] == (512, 512):
                return obj
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    result = find_512x512_tensor(item)
                    if result is not None:
                        return result
            return None

        result = find_512x512_tensor(logits)
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
                
                self.logger.log(f"Inference output shape: {mask.shape}, dtype: {mask.dtype}")
                print(f"Inference output shape: {mask.shape}, dtype: {mask.dtype}")
                
                # GPU memory cleanup after inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                return mask
                
            except Exception as e:
                self.logger.log(f"Error during BiRefNet inference: {e}")
                print(f"Error during BiRefNet inference: {e}")
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
                
                # GPU memory cleanup even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                raise

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
            if mask_tensor.dtype == torch.float16:
                mask_np = mask_tensor.detach().cpu().float().numpy()
            else:
                mask_np = mask_tensor.detach().cpu().numpy()
            
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
        Same implementation as EVF-SAM for consistency.
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
        fname = os.path.join(self.vis_save_dir, "temp_image.jpg")
        
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
            
        # GPU memory cleanup after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            
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
            
            # Process images using proven single image processing method
            self.logger.log(f"Step 2: Processing {len(downloaded_images)} images using single processing...")
            print(f"Step 2: Processing {len(downloaded_images)} images using single processing...")
            
            for i, image in enumerate(downloaded_images):
                try:
                    self.logger.log(f"Processing image {i+1}/{len(downloaded_images)}")
                    print(f"Processing image {i+1}/{len(downloaded_images)}")
                    
                    # Convert PIL image to numpy array (BGR format for BiRefNet)
                    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Use the proven single image processing method
                    result_img = self.process_image_array(image_np, plot=False)
                    
                    if result_img is not None:
                        processed_images.append(result_img)
                        self.logger.log(f"Successfully processed image {i+1}/{len(downloaded_images)}")
                        print(f"Successfully processed image {i+1}/{len(downloaded_images)}")
                    else:
                        self.logger.log(f"Failed to process image {i+1}: process_image_array returned None")
                        print(f"Failed to process image {i+1}: process_image_array returned None")
                        failed_count += 1
                    
                except Exception as e:
                    self.logger.log(f"Failed to process image {i+1}: {str(e)}")
                    print(f"Failed to process image {i+1}: {str(e)}")
                    failed_count += 1
                    continue
            
            success_count = len(processed_images)
            total_requested = len(image_urls)
            
            self.logger.log(f"Batch processing completed: {success_count}/{total_requested} successful, {failed_count} failed")
            print(f"Batch processing completed: {success_count}/{total_requested} successful, {failed_count} failed")
            
            # Final GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return processed_images
            
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.logger.log(error_msg)
            print(error_msg)
            
            # Cleanup GPU memory even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
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