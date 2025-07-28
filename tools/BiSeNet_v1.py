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
import boto3
from botocore.exceptions import ClientError
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
            
            # DEBUG: Show first 4 characters of AWS credentials
            aws_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
            aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
            aws_region = os.environ.get('AWS_S3_REGION', '')
            print(f"🔑 DEBUG AWS_ACCESS_KEY_ID: {aws_key[:4]}*** (length: {len(aws_key)})")
            print(f"🔑 DEBUG AWS_SECRET_ACCESS_KEY: {aws_secret[:4]}*** (length: {len(aws_secret)})")
            print(f"🔑 DEBUG AWS_S3_REGION: {aws_region}")
            
            dotenv_loaded = True
            break
    if not dotenv_loaded:
        print("⚠️ No .env or local_test.env found, using system environment only")
        # DEBUG: Show system environment AWS credentials
        aws_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
        aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
        aws_region = os.environ.get('AWS_S3_REGION', '')
        print(f"🔑 DEBUG System AWS_ACCESS_KEY_ID: {aws_key[:4]}*** (length: {len(aws_key)})")
        print(f"🔑 DEBUG System AWS_SECRET_ACCESS_KEY: {aws_secret[:4]}*** (length: {len(aws_secret)})")
        print(f"🔑 DEBUG System AWS_S3_REGION: {aws_region}")
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
            model_path: S3 path to model checkpoint
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
        """Setup BiSeNet repository if not exists"""
        bisenet_path = "BiSeNet"
        
        if not os.path.exists(bisenet_path):
            print("📥 Cloning BiSeNet repository...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/CoinCheung/BiSeNet.git"
                ], check=True, capture_output=True)
                print("✅ BiSeNet repository cloned")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to clone BiSeNet: {e}")
                raise
        
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
    
    def _download_model_from_s3(self, s3_path):
        """Download model checkpoint from S3"""
        if not s3_path:
            raise ValueError("S3 model path is required")
        
        print(f"📥 Downloading model from S3: {s3_path}")
        
        # Parse S3 path: s3://bucket/key or bucket/key
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]  # Remove s3:// prefix
        
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        bucket, key = parts
        
        # DEBUG: Show parsed bucket and key
        print(f"🪣 DEBUG Parsed bucket: '{bucket}'")
        print(f"🔑 DEBUG Parsed key: '{key}'")
        
        try:
            # Initialize S3 client - try multiple regions
            access_key = get_env_variable('AWS_ACCESS_KEY_ID')
            secret_key = get_env_variable('AWS_SECRET_ACCESS_KEY')
            regions_to_try = [
                get_env_variable('AWS_S3_REGION') or 'eu-central-1',
                'us-east-1',  # Default AWS region
                'eu-west-1',  # Common EU region
                'us-west-2'   # Common US region
            ]
            
            # Remove duplicates while preserving order
            regions_to_try = list(dict.fromkeys(regions_to_try))
            
            buffer = None
            last_error = None
            
            for region in regions_to_try:
                try:
                    print(f"🌍 Trying region: {region}")
                    
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        region_name=region
                    )
                    
                    # TEST: Try to list bucket contents first to check permissions
                    print(f"📋 Testing bucket access in {region}...")
                    try:
                        folder_prefix = key.rsplit('/', 1)[0] + '/'
                        print(f"🔍 Listing with prefix: '{folder_prefix}'")
                        
                        # FIRST: Compare with a working directory (DeepLabV3)
                        print(f"🆚 COMPARISON: Listing DeepLabV3 models first...")
                        try:
                            comparison_response = s3_client.list_objects_v2(
                                Bucket=bucket,
                                Prefix="models/mannequin_segmenter_deeplabv3_mobilevit/",
                                MaxKeys=10
                            )
                            if 'Contents' in comparison_response:
                                print(f"✅ DeepLabV3 folder has {len(comparison_response['Contents'])} objects:")
                                for obj in comparison_response['Contents'][:3]:  # Show first 3
                                    print(f"   📄 {obj['Key']} (size: {obj['Size']} bytes)")
                            else:
                                print(f"❌ DeepLabV3 folder is empty")
                        except Exception as e:
                            print(f"❌ DeepLabV3 comparison failed: {e}")
                        
                        # NOW: List our target directory with multiple strategies
                        print(f"🎯 NOW checking BiSeNet directory...")
                        
                        # Strategy 1: With trailing slash
                        response = s3_client.list_objects_v2(
                            Bucket=bucket,
                            Prefix=folder_prefix,
                            MaxKeys=50
                        )
                        
                        if 'Contents' in response:
                            print(f"✅ BiSeNet folder has {len(response['Contents'])} objects (with slash):")
                            for obj in response['Contents']:
                                print(f"   📄 {obj['Key']} (size: {obj['Size']} bytes)")
                                if obj['Key'] == key:
                                    print(f"   🎯 TARGET FILE FOUND: {key}")
                        else:
                            print(f"📂 BiSeNet folder appears empty (with slash)")
                            
                        # Strategy 2: Without trailing slash
                        response2 = s3_client.list_objects_v2(
                            Bucket=bucket,
                            Prefix=key.rsplit('/', 1)[0],
                            MaxKeys=50
                        )
                        
                        if 'Contents' in response2:
                            print(f"📦 BiSeNet folder has {len(response2['Contents'])} objects (without slash):")
                            for obj in response2['Contents']:
                                if obj['Key'] not in [o['Key'] for o in response.get('Contents', [])]:
                                    print(f"   📄 {obj['Key']} (size: {obj['Size']} bytes)")
                        
                        # Strategy 3: List versions (in case of versioning)
                        print(f"🔄 Checking for object versions...")
                        try:
                            versions_response = s3_client.list_object_versions(
                                Bucket=bucket,
                                Prefix=folder_prefix,
                                MaxKeys=10
                            )
                            if 'Versions' in versions_response:
                                print(f"📝 Found {len(versions_response['Versions'])} versions:")
                                for ver in versions_response['Versions']:
                                    print(f"   📄 {ver['Key']} (version: {ver['VersionId'][:8]}...)")
                        except Exception as e:
                            print(f"⚠️ Versioning check failed: {e}")
                            
                    except ClientError as list_error:
                        print(f"❌ Cannot list bucket in {region}: {list_error.response['Error']['Code']}")
                        continue
                    
                    # Skip listing check and try direct download (encrypted files may not show in list)
                    print(f"🔍 Attempting direct download from {region}...")
                    
                    # Download to memory
                    buffer = io.BytesIO()
                    s3_client.download_fileobj(bucket, key, buffer)
                    buffer.seek(0)
                    
                    file_size = buffer.tell()
                    print(f"✅ Successfully downloaded from region {region} - Size: {file_size:,} bytes")
                    break
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    print(f"❌ Region {region} failed: {error_code} - {e.response['Error']['Message']}")
                    last_error = e
                    continue
            
            if buffer is None:
                raise last_error or Exception("Failed to download from any region")
            
            print("✅ Model downloaded from S3 successfully")
            return buffer
            
        except ClientError as e:
            print(f"❌ S3 download failed: {e}")
            raise
        except Exception as e:
            print(f"❌ S3 client setup failed: {e}")
            raise
    
    def _load_model(self, model_path):
        """Load BiSeNet v1 model and checkpoint"""
        print(f"🔧 Loading BiSeNet v1 model...")
        
        # Initialize model
        self.model = self.BiSeNetV1(n_classes=self.num_classes)
        
        if model_path:
            # Download and load checkpoint
            checkpoint_buffer = self._download_model_from_s3(model_path)
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
            # Download image
            print(f"📸 Processing image: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Load image
            image_pil = Image.open(io.BytesIO(response.content))
            original_size = image_pil.size
            
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

# 🚀 Image caching system for shared image downloads
_image_cache = {}
_cache_lock = threading.Lock()

def _get_cached_image(image_url: str):
    """Get image from shared cache or download if not cached."""
    cache_start = time.time()
    
    with _cache_lock:
        if image_url in _image_cache:
            cache_hit_time = time.time() - cache_start
            print(f"        🎯 Cache HIT: {cache_hit_time:.3f}s")
            return _image_cache[image_url]
    
    # Download only if not cached
    download_start = time.time()
    try:
        print(f"        📥 Downloading: {image_url}")
        response = requests.get(image_url, timeout=15)
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
        
        # Cache the result
        cache_store_start = time.time()
        with _cache_lock:
            _image_cache[image_url] = img
            # Simple cache size management (keep only last 50 images)
            if len(_image_cache) > 50:
                oldest_key = next(iter(_image_cache))
                del _image_cache[oldest_key]
        cache_store_time = time.time() - cache_store_start
        print(f"        💾 Cache store: {cache_store_time:.3f}s")
        
        total_time = time.time() - cache_start
        print(f"        ✅ Total download+cache: {total_time:.3f}s")
        
        return img
        
    except Exception as e:
        total_time = time.time() - cache_start
        print(f"        ⚠️ Image download failed after {total_time:.3f}s: {e} (URL: {image_url})")
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
        print("✅ BiSeNet v1 test completed successfully - Image uploaded to S3!")
    else:
        print("❌ BiSeNet v1 test failed") 