import os
import sys
import cv2
import torch
import numpy as np
import requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

from .logger import EVFSAMLogger
from .s3_utils import download_evf_sam_from_s3


class MaskRCNNSegmenter:
    def __init__(
        self,
        model_path: str = None,
        precision: str = "fp32",
        vis_save_dir: str = "infer",
        thickness_threshold: int = 40,
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize MaskRCNN Segmenter for mannequin detection and removal.
        
        Args:
            model_path: Path to the trained MaskRCNN model weights
            precision: Model precision ("fp32", "fp16", "bf16")
            vis_save_dir: Directory for saving visualization outputs
            thickness_threshold: Threshold for removing thin artifacts
            confidence_threshold: Confidence threshold for detections
            mask_threshold: Threshold for mask binarization
        """
        os.makedirs(vis_save_dir, exist_ok=True)
        self.vis_save_dir = vis_save_dir
        self.precision = precision
        self.dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
        self.thickness_threshold = thickness_threshold
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold

        # Initialize logger
        self.logger = EVFSAMLogger()
        
        # Force CPU usage to avoid segmentation faults (similar to EVF-SAM)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log("Initializing MaskRCNN Segmenter")
        
        # Initialize MaskRCNN model
        self.model = maskrcnn_resnet50_fpn_v2(num_classes=2)  # 2 classes: background, clothing
        
        if model_path and os.path.exists(model_path):
            self.logger.log(f"Loading MaskRCNN weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.logger.log("No model weights provided, using pretrained backbone")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.log("MaskRCNN Segmenter initialized successfully")

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

        self.logger.log(
            f"Brown/beige pixel fraction: {fraction:.3f} "
            f"(brown: {np.count_nonzero(mask_brown)}, beige: {np.count_nonzero(mask_beige)}, total: {total_pixels})"
        )

        threshold = 0.10
        result = fraction > threshold
        self.logger.log(f"Is main color brown/beige: {result} (threshold: {threshold})")
        return result

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for MaskRCNN inference.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to tensor
        pil_img = Image.fromarray(img_rgb)
        
        # Convert to tensor and normalize to [0, 1]
        img_tensor = F.to_tensor(pil_img)
        
        return img_tensor.unsqueeze(0).to(self.device)

    def _run_inference(self, img_tensor: torch.Tensor) -> dict:
        """
        Run MaskRCNN inference on preprocessed image.
        
        Args:
            img_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary containing predictions
        """
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        return predictions[0]  # Get first (and only) image predictions

    def _extract_mannequin_masks(self, predictions: dict, img_shape: tuple) -> np.ndarray:
        """
        Extract mannequin masks from MaskRCNN predictions.
        
        Args:
            predictions: MaskRCNN output dictionary
            img_shape: Original image shape (H, W)
            
        Returns:
            Combined mannequin mask
        """
        masks = predictions['masks']
        scores = predictions['scores']
        labels = predictions['labels']
        
        # Filter predictions by confidence threshold
        valid_predictions = scores > self.confidence_threshold
        
        if not valid_predictions.any():
            self.logger.log("No valid predictions found above confidence threshold")
            return np.zeros(img_shape, dtype=bool)
        
        masks = masks[valid_predictions]
        scores = scores[valid_predictions]
        labels = labels[valid_predictions]
        
        self.logger.log(f"Found {len(masks)} valid predictions")
        
        # Combine all masks (assuming all detections are mannequins/clothing we want to remove)
        combined_mask = np.zeros(img_shape, dtype=bool)
        
        for i, mask in enumerate(masks):
            # Binarize mask
            binary_mask = (mask.squeeze().cpu().numpy() > self.mask_threshold)
            combined_mask = combined_mask | binary_mask
            
            self.logger.log(f"Mask {i}: score={scores[i]:.3f}, label={labels[i]}, pixels={binary_mask.sum()}")
        
        self.logger.log(f"Combined mask pixels: {combined_mask.sum()}")
        return combined_mask

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
        Process an image from URL to remove mannequins using MaskRCNN.
        
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
                
            self.logger.log(f"Downloaded and loaded image from URL: {image_url}")
            
        except Exception as e:
            self.logger.log(f"Image download/load error: {e} (URL: {image_url})")
            if os.path.exists(fname):
                os.remove(fname)
            return None

        # Preprocess image for MaskRCNN
        img_tensor = self._preprocess_image(img)
        
        # Run inference
        self.logger.log("Running MaskRCNN inference...")
        predictions = self._run_inference(img_tensor)
        
        # Extract mannequin masks
        mannequin_mask = self._extract_mannequin_masks(predictions, img.shape[:2])
        
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
            
        self.logger.log(f"Finished processing image from URL: {image_url}")
        return cleaned_img

    def process_image_array(self, img: np.ndarray, plot: bool = True) -> np.ndarray:
        """
        Process an image array to remove mannequins using MaskRCNN.
        
        Args:
            img: Input image as numpy array in BGR format
            plot: Whether to show visualization plots
            
        Returns:
            Processed image with mannequins removed
        """
        self.logger.log("Processing image array...")
        
        # Preprocess image for MaskRCNN
        img_tensor = self._preprocess_image(img)
        
        # Run inference
        self.logger.log("Running MaskRCNN inference...")
        predictions = self._run_inference(img_tensor)
        
        # Extract mannequin masks
        mannequin_mask = self._extract_mannequin_masks(predictions, img.shape[:2])
        
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
        return cleaned_img