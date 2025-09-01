#!/usr/bin/env python3
"""
DeepLabV3-MobileViT Mannequin Segmenter
Fast and lightweight model for mannequin segmentation
BACKGROUND WHITE OUT: Only mannequin area is preserved, background is white.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
import io
from google.cloud import storage
import json
import base64
import threading
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Handle both relative and absolute imports for local testing
try:
    from .logger import AppLogger
    from .env_utils import get_env_variable
except ImportError:
    from logger import AppLogger
    from env_utils import get_env_variable

ENABLE_IMAGE_CACHE = os.getenv('ENVIRONMENT', 'development').lower() != 'production'
_image_download_cache = {} if ENABLE_IMAGE_CACHE else None
_cache_lock = threading.Lock() if ENABLE_IMAGE_CACHE else None

def apply_white_background(original_image: Image.Image, pred_mask: np.ndarray) -> Image.Image:
    """
    Keep only the mannequin (pred_mask==1), turn all background pixels to white.
    """
    print(f"   📐 Original: {original_image.size}, Mask: {pred_mask.shape}")
    print(f"   📊 Mask values: {pred_mask.min()}-{pred_mask.max()}, Mannequin pixels: {np.sum(pred_mask > 0):,}")
    
    img_np = np.array(original_image)
    if len(img_np.shape) == 2:  # grayscale
        img_np = np.stack([img_np]*3, axis=-1)
    mask = (pred_mask > 0).astype(np.uint8)
    if mask.shape != img_np.shape[:2]:
        print(f"   🔄 Resizing mask: {mask.shape} → {img_np.shape[:2]}")
        mask = np.array(Image.fromarray(mask).resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST))
    
    mannequin_pixels = np.sum(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage = mannequin_pixels / total_pixels * 100
    print(f"   🎯 Final coverage: {coverage:.1f}% ({mannequin_pixels:,}/{total_pixels:,} pixels)")
    
    mask_3ch = np.stack([mask]*3, axis=-1)
    white_bg = np.ones_like(img_np, dtype=np.uint8) * 255
    out = img_np * mask_3ch + white_bg * (1 - mask_3ch)
    
    print(f"   🤍 White background applied successfully!")
    return Image.fromarray(out)

class DeepLabV3MobileViTSegmenter:
    """
    DeepLabV3-MobileViT mannequin segmenter with WHITE BACKGROUND postprocessing.
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
        self.logger = AppLogger()
        print("Initializing DeepLabV3-MobileViT Segmenter")
        force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        if force_cpu:
            self.device = torch.device("cpu")
            print("🔧 FORCE_CPU enabled - Using CPU for DeepLabV3-MobileViT")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU (no GPU available)")
        if self.device.type == 'cpu':
            cpu_count = os.cpu_count() or 4
            max_threads = cpu_count
            torch.set_num_threads(max_threads)
            os.environ['OMP_NUM_THREADS'] = str(max_threads)
            os.environ['MKL_NUM_THREADS'] = str(max_threads) 
            os.environ['NUMEXPR_MAX_THREADS'] = str(max_threads)
            os.environ['BLAS_NUM_THREADS'] = str(max_threads)
            print(f"🧵 CPU multi-threading optimized: {torch.get_num_threads()} threads across {cpu_count} cores")
        import random, time
        delay = random.uniform(0.5, 3.0)
        print(f"⏳ Anti-rate-limit delay: {delay:.1f}s")
        time.sleep(delay)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
            print("✅ AutoImageProcessor loaded (offline)")
        except:
            print("⚠️ Offline processor failed, trying online...")
            time.sleep(random.uniform(1, 3))
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            print("✅ AutoImageProcessor loaded (online)")
        try:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, local_files_only=True)
            print("✅ Base model loaded (offline)")
        except:
            print("⚠️ Offline model failed, trying online...")
            time.sleep(random.uniform(1, 3))
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            print("✅ Base model loaded (online)")
        checkpoint_loaded = False
        if os.path.exists(model_path):
            try:
                print(f"Loading custom checkpoint from: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint, strict=False)
                checkpoint_loaded = True
                print("✅ Custom checkpoint loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load local checkpoint: {e}")
        if not checkpoint_loaded:
            try:
                print("Attempting to download checkpoint from GCS...")
                gcs_path = f"artifactsredi/models/mannequin_segmenter_deeplabv3_mobilevit/checkpoint_20250726.pt"
                if self._download_model_from_gcs(gcs_path, model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint, strict=False)
                    checkpoint_loaded = True
                    print("✅ GCS checkpoint downloaded and loaded")
            except Exception as gcs_error:
                print(f"⚠️ GCS download failed: {gcs_error}")
        if not checkpoint_loaded:
            print("ℹ️ Using base pretrained model (no custom checkpoint)")
        
        self.model = self.model.to(self.device)
        # USER'S WORKING VALIDATION CODE used eval() mode!
        print("🔧 Using EVAL mode like in user's working validation...")
        self.model.eval()  # Like in validation
        print("✅ Model set to eval mode")
        if self.device.type == 'cpu':
            self.model = self.model.to(memory_format=torch.channels_last)
            print("✅ Model optimized for CPU with channels_last memory format")
        if self.precision == "fp16" and self.device.type == 'cuda':
            self.model = self.model.half()
            print("✅ Model set to FP16 precision")
        os.makedirs(vis_save_dir, exist_ok=True)
        print(f"✅ DeepLabV3-MobileViT initialized on {self.device}")

    def _download_model_from_gcs(self, gcs_path: str, local_path: str) -> bool:
        try:
            # Parse GCS path: bucket/key
            parts = gcs_path.split('/', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid GCS path format: {gcs_path}")
            
            bucket_name, blob_path = parts
            
            # Initialize GCS client
            gcp_project_id = get_env_variable("GCP_PROJECT_ID")
            gcp_sa_key_b64 = get_env_variable("GCP_SA_KEY")
            
            if not gcp_sa_key_b64:
                raise ValueError("GCP_SA_KEY environment variable not found")
            
            # Decode base64 service account key
            gcp_sa_key_json = base64.b64decode(gcp_sa_key_b64).decode('utf-8')
            gcp_sa_key = json.loads(gcp_sa_key_json)
            
            # Create GCS client with service account credentials
            gcs_client = storage.Client.from_service_account_info(gcp_sa_key, project=gcp_project_id)
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"Model file not found in GCS: {bucket_name}/{blob_path}")
            
            # Download to local file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"✅ Model downloaded from GCS: {gcs_path}")
            return True
        except Exception as e:
            print(f"❌ GCS download failed: {e}")
            return False

    def _preprocess_image(self, image_url: str):
        try:
            global _image_download_cache, _cache_lock, ENABLE_IMAGE_CACHE
            if ENABLE_IMAGE_CACHE and _image_download_cache is not None and _cache_lock is not None:
                with _cache_lock:
                    if image_url in _image_download_cache:
                        print("   💾 Using cached image (development only)")
                        cached_data = _image_download_cache[image_url]
                        image = Image.open(io.BytesIO(cached_data)).convert('RGB')
                    else:
                        print("   🌐 Downloading image (caching enabled)...")
                        session = requests.Session()
                        session.headers.update({
                            'Accept-Encoding': 'gzip, deflate',
                            'Connection': 'keep-alive',
                            'User-Agent': 'mannequin-segmenter/1.0'
                        })
                        response = session.get(image_url, timeout=10, stream=True)
                        response.raise_for_status()
                        image_data = response.content
                        _image_download_cache[image_url] = image_data
                        if len(_image_download_cache) > 5:
                            oldest_key = next(iter(_image_download_cache))
                            del _image_download_cache[oldest_key]
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                print("   🌐 Downloading image (production - no cache)...")
                session = requests.Session()
                session.headers.update({
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'User-Agent': 'mannequin-segmenter/1.0'
                })
                response = session.get(image_url, timeout=10, stream=True)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_array = np.array(image_resized, dtype=np.float32) / 255.0
            pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            if self.device.type == 'cpu':
                pixel_values = pixel_values.to(memory_format=torch.channels_last)
            if self.precision == "fp16" and self.device.type == 'cuda':
                pixel_values = pixel_values.half()
            return pixel_values, image
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None, None

    def _draw_full_mask_to_console(self, pred_mask: np.ndarray):
        """Draw the FULL mask pixel-by-pixel to console for debugging"""
        try:
            h, w = pred_mask.shape
            print(f"   📐 Mask dimensions: {h}x{w}")
            
            # For large masks, downsample to fit console (50x50 max)
            if h > 50 or w > 50:
                step_h = max(1, h // 50)
                step_w = max(1, w // 50)
                display_mask = pred_mask[::step_h, ::step_w]
                print(f"   📉 Downsampled to: {display_mask.shape[0]}x{display_mask.shape[1]} (step: {step_h}x{step_w})")
            else:
                display_mask = pred_mask
            
            # Character mapping for different classes
            char_map = {
                0: '  ',    # Background = spaces
                1: '██',    # Class 1 = full blocks
                2: '▓▓',    # Class 2 = dark shade
                3: '▒▒',    # Class 3 = medium shade
                4: '░░',    # Class 4 = light shade
                5: '••',    # Class 5 = bullets
                6: '++',    # Class 6 = plus
                7: 'XX',    # Class 7 = X
                8: '##',    # Class 8 = hash
                9: '$$',    # Class 9 = dollar
                10: '%%',   # Class 10 = percent
                11: '&&',   # Class 11 = ampersand
                12: '@@',   # Class 12 = at
                13: '!!',   # Class 13 = exclamation
                14: '??',   # Class 14 = question
                15: '🟥',   # Class 15 (person) = red square
                16: '77',   # Class 16 = 7
                17: '88',   # Class 17 = 8
                18: '99',   # Class 18 = 9
                19: 'OO',   # Class 19 = O
                20: 'UU'    # Class 20 = U
            }
            
            # Draw border
            border_width = display_mask.shape[1] * 2 + 4
            print("   " + "─" * border_width)
            
            # Draw each row
            for row_idx, row in enumerate(display_mask):
                line = f"   │"
                for col_idx, pixel in enumerate(row):
                    char = char_map.get(pixel, f"{pixel%10}{pixel%10}")
                    line += char
                line += "│"
                print(line)
            
            print("   " + "─" * border_width)
            
            # Legend for detected classes
            detected_classes = np.unique(display_mask)
            legend_items = []
            for cls in detected_classes:
                char = char_map.get(cls, f"{cls%10}{cls%10}")
                legend_items.append(f"[{char}]=class{cls}")
            
            print(f"   📋 Legend: {', '.join(legend_items)}")
            
            # Statistics
            for cls in detected_classes:
                count = np.sum(display_mask == cls)
                percentage = count / display_mask.size * 100
                print(f"   📊 Class {cls}: {count} pixels ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"   ❌ Console mask visualization failed: {e}")

    def _run_inference(self, pixel_values: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """EXACT copy of user's working validation code"""
        try:
            # USER'S WORKING CODE: model.eval() + torch.no_grad()
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                print(f"   🔍 Model logits shape: {logits.shape}")
                
                # USER'S EXACT METHOD: F.interpolate + argmax
                print(f"   🔧 USER's WORKING METHOD: Upsampling {logits.shape[2:]} → {target_size}")
                logits_upsampled = F.interpolate(
                    logits, size=target_size, mode="bilinear", align_corners=False
                )
                print(f"   ✅ Upsampled shape: {logits_upsampled.shape}")
                
                # USER'S EXACT ARGMAX: torch.argmax(logits_upsampled, dim=1).squeeze()
                pred_mask = torch.argmax(logits_upsampled, dim=1).squeeze().cpu().numpy()
                print(f"   📊 Final prediction shape: {pred_mask.shape}")
                
                # USER'S DEBUG: show unique values (like in validation)
                unique_classes, counts = np.unique(pred_mask, return_counts=True)
                print(f"   🎯 Pred mask unique: {list(zip(unique_classes, counts))}")
                
                # USER REQUESTED: Draw FULL mask pixel-by-pixel to console
                print("   🖼️ FULL MASK VISUALIZATION (pixel-level):")
                self._draw_full_mask_to_console(pred_mask)
                
                return pred_mask
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return None

    def process_image_url(self, image_url: str, plot: bool = False) -> Image.Image:
        """
        Processes an image from URL, returns a PIL.Image with mannequin preserved,
        background set to white.
        """
        try:
            print("Step 2: Fast preprocessing...")
            pixel_values, original_image = self._preprocess_image(image_url)
            if pixel_values is None:
                return None
            print("Step 3: Running inference...")
            # Pass target size for F.interpolate (USER's training method)
            target_size = (self.image_size, self.image_size)  # Training size first
            pred_mask = self._run_inference(pixel_values, target_size)
            if pred_mask is None:
                return None
            # Now resize to ORIGINAL image size (not training size)
            if pred_mask.shape != original_image.size[::-1]:
                print(f"   🔄 Final resize: {pred_mask.shape} → {original_image.size[::-1]}")
                pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8), mode="L")
                pred_mask_pil = pred_mask_pil.resize(original_image.size, Image.NEAREST)
                pred_mask = np.array(pred_mask_pil)
            # SMART class detection - try multiple likely mannequin classes
            unique_classes, counts = np.unique(pred_mask, return_counts=True)
            class_info = list(zip(unique_classes, counts))
            print(f"   📊 Final class distribution: {class_info}")
            
            # Try person class (15) and mannequin class (1) first
            best_class = None
            best_pixels = 0
            for cls in [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]:
                if cls in unique_classes:
                    pixels = counts[unique_classes == cls][0]
                    if pixels > best_pixels and pixels > 1000:  # Minimum threshold
                        best_pixels = pixels
                        best_class = cls
                        
            if best_class is None:
                print("   ⚠️ No significant class found, using largest non-background")
                non_bg_classes = [(cls, cnt) for cls, cnt in class_info if cls != 0]
                if non_bg_classes:
                    best_class = max(non_bg_classes, key=lambda x: x[1])[0]
                    best_pixels = max(non_bg_classes, key=lambda x: x[1])[1]
                else:
                    best_class = 1  # Fallback
                    
            print(f"   🎯 Using class {best_class} as mannequin ({best_pixels:,} pixels)")
            mannequin_mask = (pred_mask == best_class).astype(np.uint8)
            print("Step 4: White background postprocessing...")
            result_img = apply_white_background(original_image, mannequin_mask)
            print("✅ Processing complete")
            if plot:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,8))
                plt.imshow(result_img)
                plt.axis('off')
                plt.title("Background white, mannequin preserved")
                plt.show()
            return result_img
        except Exception as e:
            print(f"❌ Process image failed: {e}")
            return None

    def get_model_info(self) -> dict:
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

# --- Példa használat ---
# segmenter = DeepLabV3MobileViTSegmenter()
# url = "https://example.com/your_image.jpg"
# out_img = segmenter.process_image_url(url, plot=True)
# out_img.save("output_whitebg.png")
 