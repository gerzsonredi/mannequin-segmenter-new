import os
import sys
import cv2
import torch
import numpy as np
import requests
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Import utilities from tools package
from tools.logger import EVFSAMLogger
from tools.s3_utils import download_evf_sam_from_s3

# Load environment variables from .env file
load_dotenv()

# Download EVF-SAM before importing model
evf_sam_path = download_evf_sam_from_s3()
sys.path.append(evf_sam_path)
from model.evf_sam2 import EvfSam2Model

# Initialize global logger
evfsam_logger = EVFSAMLogger()

class EVFSAMSingleImageInferencer:
    def __init__(
        self,
        model_version: str = "YxZhang/evf-sam2-multitask",
        model_type: str = "sam2",
        precision: str = "fp16",
        vis_save_dir: str = "infer",
        use_bnb: bool = False,
        thickness_threshold: int = 50,
    ):
        os.makedirs(vis_save_dir, exist_ok=True)
        self.vis_save_dir = vis_save_dir
        self.precision = precision
        self.dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
        self.model_version = model_version
        self.use_bnb = use_bnb
        self.thickness_threshold = thickness_threshold

        # Force CPU usage to avoid segmentation faults
        self.device = torch.device("cpu")
        evfsam_logger.log("Forcing CPU usage to avoid segmentation faults")

        kwargs = dict(torch_dtype=self.dtype, low_cpu_mem_usage=True)
        if use_bnb:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            )
            kwargs["torch_dtype"] = torch.float16

        self.model = EvfSam2Model.from_pretrained(model_version, **kwargs)
        if not use_bnb:
            self.model = self.model.to(self.device)
        self.model.eval()
        # Remove model compilation for CPU usage
        # if self.device.type == 'cuda':
        #     self.model = torch.compile(self.model, mode="max-autotune")
        self.tokenizer = AutoTokenizer.from_pretrained(model_version, padding_side="right", use_fast=False)

        self.prompts1 = ["mark beige mannequin under the T-shirt"]
        self.prompts3 = ["mark only the vertical white-beige tube under the T-shirt"]

    @staticmethod
    def _resize_with_padding(img: np.ndarray, target_size: int):
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
        pt, pb, pl, pr = pad
        mask_cropped = mask[pt:mask.shape[0] - pb, pl:mask.shape[1] - pr]
        return cv2.resize(mask_cropped.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    def apply_masks_to_remove_unwanted_areas(
        self,
        img: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        mask3: np.ndarray = None
    ) -> np.ndarray:
        processed_img = img.copy()
        mask1_bool = mask1.astype(bool)
        mask2_bool = mask2.astype(bool)
        mask3_bool = mask3.astype(bool) if mask3 is not None else np.zeros_like(mask1_bool, dtype=bool)
        combined_mask = mask1_bool | mask2_bool | mask3_bool
        if processed_img.ndim == 3:
            processed_img[combined_mask] = [255, 255, 255]
        else:
            processed_img[combined_mask] = 255
        return processed_img

    def remove_thin_stripes(
        self,
        img: np.ndarray,
        thickness_threshold: int = 10,
        method: str = "morphology"
    ) -> np.ndarray:
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
        fname = os.path.join(self.vis_save_dir, "temp_image.jpg")
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            with open(fname, "wb") as f:
                f.write(response.content)
            img = cv2.imread(fname)
            if img is None:
                raise ValueError("Failed to load the image!")
            evfsam_logger.log(f"Downloaded and loaded image from URL: {image_url}")
        except Exception as e:
            evfsam_logger.log(f"Image download/load error: {e} (URL: {image_url})")
            if os.path.exists(fname):
                os.remove(fname)
            return None

        sam_img, sam_pad, _ = self._resize_with_padding(img, 1024)
        beit_img, _, _ = self._resize_with_padding(img, 224)

        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        sam_np = cv2.cvtColor(sam_img, cv2.COLOR_BGR2RGB)
        sam_tensor = (torch.from_numpy(sam_np).permute(2, 0, 1) - pixel_mean) / pixel_std
        sam_tensor = sam_tensor.to(dtype=self.dtype, device=self.device).unsqueeze(0)

        beit_np = cv2.cvtColor(beit_img, cv2.COLOR_BGR2RGB)
        beit_tensor = (torch.from_numpy(beit_np).permute(2, 0, 1) / 255.0 - 0.5) / 0.5
        beit_tensor = beit_tensor.to(dtype=self.dtype, device=self.device).unsqueeze(0)

        orig_sizes = [sam_img.shape[:2]]
        resize_info = [None]

        input_ids1 = self.tokenizer(self.prompts1, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        input_ids3 = self.tokenizer(self.prompts3, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(autocast_device, dtype=self.dtype):
            pred_mask1_logits = self.model.inference(
                sam_tensor, beit_tensor, input_ids1, resize_list=resize_info, original_size_list=orig_sizes
            )
            pred_mask3_logits = self.model.inference(
                sam_tensor, beit_tensor, input_ids3, resize_list=resize_info, original_size_list=orig_sizes
            )

        threshold = 0.3

        probs1 = torch.sigmoid(pred_mask1_logits[0])
        mask1 = (probs1.detach().cpu().numpy() > threshold)
        evfsam_logger.log(f"Pred mask1 logits min: {pred_mask1_logits[0].min():.4f}, max: {pred_mask1_logits[0].max():.4f}")
        evfsam_logger.log(f"Mask1 probs min: {probs1.min():.4f}, max: {probs1.max():.4f}")
        evfsam_logger.log(f"Mask1 pixels > {threshold}: {mask1.sum()}")

        mask2 = np.zeros_like(mask1, dtype=bool)

        probs3 = torch.sigmoid(pred_mask3_logits[0])
        mask3 = (probs3.detach().cpu().numpy() > threshold)
        evfsam_logger.log(f"Pred mask3 logits min: {pred_mask3_logits[0].min():.4f}, max: {pred_mask3_logits[0].max():.4f}")
        evfsam_logger.log(f"Mask3 probs min: {probs3.min():.4f}, max: {probs3.max():.4f}")
        evfsam_logger.log(f"Mask3 pixels > {threshold}: {mask3.sum()}")

        final_mask1 = self._mask_to_original(mask1, sam_pad, img.shape[:2])
        final_mask2 = self._mask_to_original(mask2, sam_pad, img.shape[:2])
        final_mask3 = self._mask_to_original(mask3, sam_pad, img.shape[:2])

        evfsam_logger.log(f"Final mask1 sum: {final_mask1.sum()}")
        evfsam_logger.log(f"Final mask2 sum: {final_mask2.sum()}")
        evfsam_logger.log(f"Final mask3 sum: {final_mask3.sum()}")

        processed_img = self.apply_masks_to_remove_unwanted_areas(img, final_mask1, final_mask2, final_mask3)

        cleaned_img = self.remove_thin_stripes(
            processed_img,
            thickness_threshold=self.thickness_threshold,
            method="morphology"
        )

        if plot:
            plt.figure(figsize=(20, 4))
            plt.subplot(1, 5, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original image")
            plt.axis("off")
            plt.subplot(1, 5, 2)
            combined_mask_display = final_mask1 | final_mask2 | final_mask3
            plt.imshow(combined_mask_display, cmap='gray')
            plt.title("Combined mask")
            plt.axis("off")
            plt.subplot(1, 5, 3)
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            plt.title("After masking")
            plt.axis("off")
            plt.subplot(1, 5, 4)
            plt.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
            plt.title(f"After cleaning (threshold={self.thickness_threshold})")
            plt.axis("off")
            plt.subplot(1, 5, 5)
            plt.imshow(final_mask3, cmap='gray')
            plt.title("Tube mask")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        if os.path.exists(fname):
            os.remove(fname)
        evfsam_logger.log(f"Finished processing image from URL: {image_url}")
        return cleaned_img
