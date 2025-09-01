#!/usr/bin/env python3
"""
🎯 BiSeNet v1 Mannequin Segmentation Implementation
Replaces BiRefNet with BiSeNet v1 for better performance
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import requests
from google.cloud import storage
import json
import base64

import io
import subprocess
import cv2
import threading

# --- NEW: Load .env or local_test.env automatically ---
try:
    from dotenv import load_dotenv
    dotenv_loaded = False
    for env_file in [".env", "local_test.env"]:
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
            print(f"✅ Loaded environment variables from {env_file}")
            
            # DEBUG: Show first 4 characters of GCP credentials
            gcp_project = os.environ.get('GCP_PROJECT_ID', '')
            gcp_sa_key = os.environ.get('GCP_SA_KEY', '')
            print(f"🔑 DEBUG GCP_PROJECT_ID: {gcp_project}")
            print(f"🔑 DEBUG GCP_SA_KEY: {gcp_sa_key[:4]}*** (length: {len(gcp_sa_key)})")
            
            dotenv_loaded = True
            break
    if not dotenv_loaded:
        print("⚠️ No .env or local_test.env found, using system environment only")
        # DEBUG: Show system environment GCP credentials
        gcp_project = os.environ.get('GCP_PROJECT_ID', '')
        gcp_sa_key = os.environ.get('GCP_SA_KEY', '')
        print(f"🔑 DEBUG System GCP_PROJECT_ID: {gcp_project}")
        print(f"🔑 DEBUG System GCP_SA_KEY: {gcp_sa_key[:4]}*** (length: {len(gcp_sa_key)})")
except ImportError:
    print("⚠️ python-dotenv not installed, skipping .env loading")
# --- END NEW ---

try:
    from tools.env_utils import get_env_variable
except ImportError:
    def get_env_variable(name, default=None):
        return os.environ.get(name, default)

class BiSeNetV1Segmenter:
    """BiSeNet v1 mannequin segmentation model"""
    
    def __init__(self, model_path=None, model_name="BiSeNetV1", image_size=512, 
                 precision="fp32", vis_save_dir="infer"):
        """
        Initialize BiSeNet v1 segmenter
        
        Args:
            model_path: GCS path to model checkpoint
            model_name: Model identifier 
            image_size: Input image size
            precision: Model precision (fp32/fp16)
            vis_save_dir: Directory for saving visualizations
        """
        print("Initializing BiSeNet v1 Segmenter")
        
        # Configuration
        self.model_name = model_name
        self.image_size = image_size
        self.precision = precision
        self.vis_save_dir = vis_save_dir
        self.num_classes = 2
        
        # Device configuration
        force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        if force_cpu:
            self.device = torch.device("cpu")
            print("🔧 FORCE_CPU enabled - Using CPU for BiSeNet v1")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CPU threading optimization
        if self.device.type == 'cpu':
            cpu_count = os.cpu_count() or 2
            print(f"🧵 CPU multi-threading optimized: {cpu_count} threads across {cpu_count} cores")
            torch.set_num_threads(cpu_count)
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_count)
            os.environ['BLAS_NUM_THREADS'] = str(cpu_count)
        
        # Setup BiSeNet repository
        self._setup_bisenet_repo()
        
        # Load model
        self._load_model(model_path)
        
        # Image transforms - BiSeNet expects fixed 512x512 input
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        print(f"✅ BiSeNet v1 initialized on {self.device}")
    
    def _setup_bisenet_repo(self):
        """Setup BiSeNet repository - should be pre-cloned in Docker build"""
        bisenet_path = "BiSeNet"
        
        # Check if BiSeNet directory exists (should be pre-cloned in Dockerfile)
        if os.path.exists(bisenet_path):
            print(f"✅ Found pre-built BiSeNet directory: {bisenet_path}")
        else:
            print("⚠️ BiSeNet directory not found - this should not happen in production!")
            print("🔧 Attempting to clone BiSeNet (fallback for local development)...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/CoinCheung/BiSeNet.git"
                ], check=True, capture_output=True)
                print("✅ BiSeNet repository cloned successfully")
            except Exception as e:
                print(f"❌ Failed to clone BiSeNet: {e}")
                raise Exception(f"BiSeNet setup failed: {e}")
        
        # Add to Python path
        if bisenet_path not in sys.path:
            sys.path.append(bisenet_path)
        
        try:
            from lib.models.bisenetv1 import BiSeNetV1
            self.BiSeNetV1 = BiSeNetV1
            print("✅ BiSeNet v1 imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import BiSeNet v1: {e}")
            raise
    
    def _download_model_from_gcs(self, gcs_path):
        """Download model checkpoint from GCS"""
        if not gcs_path:
            raise ValueError("GCS model path is required")
        
        print(f"📥 Downloading model from GCS: {gcs_path}")
        
        # Parse GCS path: bucket/key
        parts = gcs_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path format: {gcs_path}")
        
        bucket_name, blob_path = parts
        
        # DEBUG: Show parsed bucket and key
        print(f"🪣 DEBUG Parsed bucket: '{bucket_name}'")
        print(f"🔑 DEBUG Parsed blob path: '{blob_path}'")
        
        try:
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
            
            # List files in the model directory to debug
            print(f"📋 Listing files in model directory...")
            folder_prefix = blob_path.rsplit('/', 1)[0] + '/'
            blobs = list(bucket.list_blobs(prefix=folder_prefix, max_results=20))
            
            if blobs:
                print(f"📄 Found {len(blobs)} files in folder:")
                for blob in blobs:
                    print(f"   📄 {blob.name} (size: {blob.size:,} bytes)")
            else:
                print(f"📭 Folder '{folder_prefix}' is empty")
            
            # Download the specific blob
            print(f"⬇️ Attempting download: {bucket_name}/{blob_path}")
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"Model file not found in GCS: {bucket_name}/{blob_path}")
            
            # Download to memory buffer
            buffer = io.BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)
            
            # Validate file size
            buffer.seek(0, 2)  # Seek to end
            file_size = buffer.tell()
            buffer.seek(0)  # Reset to beginning
            
            print(f"📦 Downloaded model: {file_size:,} bytes")
            if file_size < 1000:  # Less than 1KB is suspicious
                raise ValueError(f"Downloaded file too small: {file_size} bytes")
            
            print("✅ Model downloaded from GCS successfully")
            return buffer
            
        except Exception as e:
            print(f"❌ GCS download failed: {e}")
            raise
    
    def _load_model(self, model_path):
        """Load BiSeNet v1 model and checkpoint"""
        print(f"🔧 Loading BiSeNet v1 model...")
        
        # Initialize model
        self.model = self.BiSeNetV1(n_classes=self.num_classes)
        
        if model_path:
            # Download and load checkpoint
            checkpoint_buffer = self._download_model_from_gcs(model_path)
            checkpoint = torch.load(checkpoint_buffer, map_location=self.device)
            
            print(f"Loading custom checkpoint from: {model_path}")
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Custom checkpoint loaded successfully")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Custom checkpoint loaded successfully (direct state dict)")
        else:
            print("⚠️ No checkpoint provided, using random weights")
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Memory optimization for CPU
        if self.device.type == 'cpu':
            self.model = self.model.to(memory_format=torch.channels_last)
            print("✅ Model optimized for CPU with channels_last memory format")
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "architecture": "BiSeNet v1",
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": total_params,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "precision": self.precision
        }
    
    def _preprocess_image(self, image_pil):
        """Preprocess PIL image for BiSeNet v1"""
        # Ensure RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Store original size for later mask resizing
        self.original_size = image_pil.size  # (width, height)
        
        # Apply transforms (resize to 512x512 + normalize)
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        if self.device.type == 'cpu':
            input_tensor = input_tensor.to(memory_format=torch.channels_last)
        
        return input_tensor
    
    def _run_inference(self, input_tensor):
        """Run BiSeNet v1 inference"""
        with torch.no_grad():
            # BiSeNet returns tuple, we need first output
            logits = self.model(input_tensor)[0]  # (1, num_classes, H, W)
            
            # Upsample to input image size
            logits_upsampled = F.interpolate(
                logits, 
                size=(self.image_size, self.image_size), 
                mode="bilinear", 
                align_corners=False
            )
            
            # Get prediction mask
            pred_mask = torch.argmax(logits_upsampled, dim=1).squeeze().cpu().numpy()
            
            return pred_mask
    
    def _postprocess_mask(self, pred_mask):
        """Postprocess prediction mask for mannequin segmentation"""
        # Debug: show class distribution
        unique_classes, counts = np.unique(pred_mask, return_counts=True)
        print(f"🎯 Classes found: {list(zip(unique_classes, counts))}")
        
        # Show all classes with more than 100 pixels for better visibility
        significant_classes = [(cls, cnt) for cls, cnt in zip(unique_classes, counts) if cnt > 100]
        print(f"🔍 Significant classes (>100 pixels): {significant_classes}")
        
        # Based on training: target > 0 means mannequin (all non-background classes)
        # Create binary mask where ANY non-background class = mannequin
        binary_mask = (pred_mask > 0).astype(np.uint8) * 255
        
        mannequin_pixels = np.sum(binary_mask > 0)
        print(f"🎯 Total mannequin pixels: {mannequin_pixels} (all non-background classes)")
        
        # Convert to PIL and resize back to original dimensions
        mask_pil = Image.fromarray(binary_mask, mode='L')
        mask_resized = mask_pil.resize(self.original_size, Image.LANCZOS)
        
        return mask_resized
    
    def _apply_white_background(self, image_pil, mask_pil):
        """Apply white background to mannequin areas"""
        # Convert to numpy arrays
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)
        
        # Create white background where mask is True (mannequin pixels)
        white_background = np.ones_like(image_np) * 255
        
        # Apply mask: white where mannequin, original where background
        mask_3d = np.stack([mask_np, mask_np, mask_np], axis=2) / 255.0
        result_np = image_np * (1 - mask_3d) + white_background * mask_3d
        
        result_pil = Image.fromarray(result_np.astype(np.uint8))
        return result_pil
    
    def process_image_url(self, image_url, plot=False):
        """Process image from URL and return result with white background"""
        try:
            # Download image with detailed timing
            print(f"📸 Processing image: {image_url}")
            
            # Detailed HTTP timing
            import time
            download_start = time.time()
            
            # Create optimized session for image downloads with proper connection pooling
            session = requests.Session()
            session.headers.update({
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'User-Agent': 'mannequin-segmenter/1.0'
            })
            
            # Configure session with connection pooling
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Retry strategy for network resilience
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],  # Updated API
                backoff_factor=0.3  # 0.3, 0.6, 1.2 seconds
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,  # Connection pool size
                pool_maxsize=20      # Max connections per pool
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # DNS + Connection + TLS timing with optimized session
            response = session.get(image_url, timeout=(5, 25), stream=True)  # (connect, read) timeout
            first_byte_time = time.time() - download_start
            
            response.raise_for_status()
            
            # Download content timing
            content_start = time.time()
            content = response.content
            download_complete_time = time.time() - download_start
            content_download_time = time.time() - content_start
            
            print(f"   ⏱️  HTTP First Byte: {first_byte_time:.3f}s")
            print(f"   ⏱️  Content Download: {content_download_time:.3f}s") 
            print(f"   ⏱️  Total Download: {download_complete_time:.3f}s")
            print(f"   📦 Content Size: {len(content):,} bytes")
            print(f"   🔄 Retry attempts: {getattr(response.raw, '_original_response', 'N/A')}")
            print(f"   🌐 Final URL: {response.url}")
            
            # Load image timing
            image_decode_start = time.time()
            image_pil = Image.open(io.BytesIO(content))
            original_size = image_pil.size
            image_decode_time = time.time() - image_decode_start
            
            print(f"   ⏱️  Image Decode: {image_decode_time:.3f}s")
            
            print(f"   📐 Original image size: {original_size}")
            
            # Preprocess
            input_tensor = self._preprocess_image(image_pil)
            
            # Run inference
            start_time = time.time()
            pred_mask = self._run_inference(input_tensor)
            inference_time = time.time() - start_time
            
            print(f"   🧠 Inference completed in {inference_time:.3f}s")
            
            # Postprocess
            mask_pil = self._postprocess_mask(pred_mask)
            
            # Apply white background
            result_pil = self._apply_white_background(image_pil, mask_pil)
            
            print(f"   ✅ Processing completed successfully")
            
            return result_pil
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            return None
        finally:
            # Clean up session if it exists
            if 'session' in locals():
                session.close()

# 🚀 Image caching system for shared image downloads
_image_cache = {}
_cache_lock = threading.Lock()

def _get_cached_image(image_url: str):
    """
    Download image without caching to avoid lock-based serialization.
    
    🚨 CRITICAL FIX: Removed global cache lock that was serializing all downloads!
    With concurrency=1 and 50 instances, separate downloads per instance are better.
    """
    cache_start = time.time()
    
    # Always download (no cache to prevent lock contention)
    download_start = time.time()
    try:
        print(f"        📥 Downloading (LOCK-FREE): {image_url}")
        
        # Use optimized session without global locks
        session = requests.Session()
        session.headers.update({
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'User-Agent': 'mannequin-segmenter-lockfree/1.0'
        })
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Updated API
            backoff_factor=0.3
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,
            pool_maxsize=10
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        response = session.get(image_url, timeout=(5, 15))
        response.raise_for_status()
        download_time = time.time() - download_start
        print(f"        ⬇️ Download complete: {download_time:.3f}s")
        
        decode_start = time.time()
        img_data = response.content
        
        # Convert to opencv format
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        decode_time = time.time() - decode_start
        print(f"        🔄 Image decode: {decode_time:.3f}s")
        
        total_time = time.time() - cache_start
        print(f"        ✅ Total download (LOCK-FREE): {total_time:.3f}s")
        
        # Clean up session
        session.close()
        
        return img
        
    except Exception as e:
        total_time = time.time() - cache_start
        print(f"        ⚠️ Image download failed after {total_time:.3f}s: {e} (URL: {image_url})")
        if 'session' in locals():
            session.close()
        return None


# Test function for local development
if __name__ == "__main__":
    # Test BiSeNet v1 segmenter
    print("🚀 Starting BiSeNet v1 test...")
    
    segmenter = BiSeNetV1Segmenter(
        model_path="artifactsredi/models/mannequin_segmenter_bisenet/20250728/checkpoint-4.pt",
        model_name="BiSeNet v1 (2-class)",
        image_size=512,
        precision="fp32",
        vis_save_dir="infer"
    )
    
    print(f"✅ Model loaded: {segmenter.get_model_info()}")
    
    # Test with a sample image
    test_url = "https://media.remix.eu/files/20-2025/Roklya-Atos-Lombardini-131973196b.jpg"
    print(f"🖼️ Processing test image: {test_url}")
    
    result = segmenter.process_image_url(test_url)
    
    if result:
        print("✅ BiSeNet v1 test completed successfully - Image uploaded to GCS!")
    else:
        print("❌ BiSeNet v1 test failed") 