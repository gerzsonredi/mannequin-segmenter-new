import os
import sys
import cv2
import torch
import numpy as np
import requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageSegmentation, AutoProcessor
from torchvision.transforms import functional as F
from dotenv import load_dotenv

# Import utilities from tools package
from .logger import EVFSAMLogger

os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"

class BiRefNetSegmenter:
    def __init__(
        self,
        model_path: str = None,
        model_name: str = "zhengpeng7/BiRefNet",
        precision: str = "fp16",
        vis_save_dir: str = "infer",
        thickness_threshold: int = 40,
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
        self.logger = EVFSAMLogger()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log("Initializing BiRefNet Segmenter")
        print("Initializing BiRefNet Segmenter")
        
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
            self.model.eval()
            
            # Apply half precision if using CUDA and fp16
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
        
        # Initialize processor if available
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.logger.log("Processor loaded successfully")
            print("Processor loaded successfully")
        except:
            self.logger.log("No processor available, using manual preprocessing")
            print("No processor available, using manual preprocessing")
            self.processor = None
        
        self.logger.log("BiRefNet Segmenter initialized successfully")
        print("BiRefNet Segmenter initialized successfully")

    @staticmethod
    def _resize_with_padding(img: np.ndarray, target_size: int):
        """
        Resize image with padding to maintain aspect ratio.
        Same implementation as EVF-SAM for consistency.
        """
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        pad_top = (target_size - nh) // 2
        pad_bottom = target_size - nh - pad_top
        pad_left = (target_size - nw) // 2
        pad_right = target_size - nw - pad_left
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return img_padded, (pad_top, pad_bottom, pad_left, pad_right), (nh, nw)

    @staticmethod
    def _mask_to_original(mask: np.ndarray, pad: tuple, orig_shape: tuple) -> np.ndarray:
        """
        Transform mask back to original image dimensions.
        Same implementation as EVF-SAM for consistency.
        """
        pt, pb, pl, pr = pad
        mask_cropped = mask[pt:mask.shape[0] - pb, pl:mask.shape[1] - pr]
        return cv2.resize(mask_cropped.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    def _is_main_color_brown_or_beige(self, img: np.ndarray) -> bool:
        """
        Detect if a significant portion of the image is brown or beige.
        Same implementation as EVF-SAM for consistency.
        """
        # Convert to HSV for better color detection
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define brown and beige color ranges in HSV
        lower_brown = np.array([10, 60, 20])
        upper_brown = np.array([25, 255, 200])
        lower_beige = np.array([15, 10, 180])
        upper_beige = np.array([35, 100, 255])

        # Create masks for brown and beige
        mask_brown = cv2.inRange(hsv_img, lower_brown, upper_brown)
        mask_beige = cv2.inRange(hsv_img, lower_beige, upper_beige)
        mask_combined = cv2.bitwise_or(mask_brown, mask_beige)

        # Calculate the percentage of brown/beige pixels
        total_pixels = img.shape[0] * img.shape[1]
        brown_beige_pixels = np.count_nonzero(mask_combined)
        fraction = brown_beige_pixels / total_pixels

        msg = f"Brown/beige pixel fraction: {fraction:.3f} (brown: {np.count_nonzero(mask_brown)}, beige: {np.count_nonzero(mask_beige)}, total: {total_pixels})"
        self.logger.log(msg)
        print(msg)

        threshold = 0.10
        result = fraction > threshold
        msg2 = f"Is main color brown/beige: {result} (threshold: {threshold})"
        self.logger.log(msg2)
        print(msg2)
        return result

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for BiRefNet inference.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        if self.processor is not None:
            # Use the processor if available
            inputs = self.processor(pil_img, return_tensors="pt")
            img_tensor = inputs["pixel_values"].to(self.device)
            if self.precision == 'fp16' and self.device.type == 'cuda':
                img_tensor = img_tensor.half()
        else:
            # Manual preprocessing
            # Resize to standard BiRefNet input size (typically 1024x1024)
            pil_img = pil_img.resize((1024, 1024), Image.BILINEAR)
            
            # Convert to tensor and normalize
            img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(self.device)
            
            # Standard ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            if self.precision == 'fp16' and self.device.type == 'cuda':
                img_tensor = img_tensor.half()
                mean = mean.half()
                std = std.half()
            
            img_tensor = (img_tensor - mean) / std
        
        return img_tensor

    def _run_inference(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run BiRefNet inference on preprocessed image.
        
        Args:
            img_tensor: Preprocessed image tensor
            
        Returns:
            Segmentation mask tensor
        """
        with torch.no_grad():
            if self.processor is not None:
                # Use model with processor
                outputs = self.model(img_tensor)
                # BiRefNet typically returns logits that need sigmoid
                if hasattr(outputs, 'logits'):
                    mask = torch.sigmoid(outputs.logits)
                else:
                    mask = torch.sigmoid(outputs)
            else:
                # Direct model inference
                outputs = self.model(img_tensor)
                mask = torch.sigmoid(outputs)
        
        return mask.squeeze()  # Remove batch dimension

    def _extract_mannequin_masks(self, mask_tensor: torch.Tensor, img_shape: tuple) -> np.ndarray:
        """
        Extract mannequin masks from BiRefNet predictions.
        
        Args:
            mask_tensor: BiRefNet output mask tensor
            img_shape: Original image shape (H, W)
            
        Returns:
            Binary mannequin mask
        """
        # Convert tensor to numpy
        mask_np = mask_tensor.detach().cpu().numpy()
        
        # Handle different output formats
        if mask_np.ndim == 3:
            # If multiple channels, take the first one or average
            if mask_np.shape[0] > 1:
                mask_np = mask_np.mean(axis=0)
            else:
                mask_np = mask_np[0]
        
        # Resize to original image shape
        mask_resized = cv2.resize(mask_np, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Binarize mask
        binary_mask = mask_resized > self.mask_threshold
        
        msg1 = f"Mask shape: {mask_resized.shape}, threshold: {self.mask_threshold}"
        self.logger.log(msg1)
        print(msg1)
        
        msg2 = f"Mask pixels above threshold: {binary_mask.sum()}"
        self.logger.log(msg2)
        print(msg2)
        
        return binary_mask

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
            
        self.logger.log("Finished processing image array")
        print("Finished processing image array")
        return cleaned_img