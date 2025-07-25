#!/usr/bin/env python3
"""
Local True Batch Processing Test for BiRefNet
"""

import sys
import os
import numpy as np
import time
from PIL import Image
import requests
import io

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

try:
    from BirefNet import BiRefNetSegmenter
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the project root directory and have all dependencies installed:")
    print("   pip install torch torchvision transformers pillow requests python-dotenv opencv-python")
    print("\n🔧 If you're still getting import errors, try:")
    print("   export PYTHONPATH=$PYTHONPATH:$(pwd)/tools")
    print("   python local_batch_test.py")
    sys.exit(1)

# Check if we have GPU support
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - will run on CPU (slower)")
except ImportError:
    print("⚠️  PyTorch not found - will try to proceed anyway")

def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image_np = np.array(pil_image)
    return image_np[:, :, ::-1]  # BGR

def main():
    print("🚀 Local BiRefNet True Batch Processing Test")
    print("=" * 60)
    
    TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
    BATCH_SIZE = 5

    # 1. Download test image
    print("📥 Downloading test image...")
    try:
        img = download_image(TEST_IMAGE_URL)
        print(f"✅ Downloaded image shape: {img.shape}")
    except Exception as e:
        print(f"❌ Failed to download image: {e}")
        return

    # 2. Prepare batch
    batch = [img.copy() for _ in range(BATCH_SIZE)]
    print(f"🖼️  Batch size: {len(batch)}")

    # 3. Load model
    print("\n🤖 Loading BiRefNetSegmenter...")
    try:
        segmenter = BiRefNetSegmenter(precision="fp16", thickness_threshold=200)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 4. Test single image processing (baseline)
    print("\n🧪 Testing single image processing (baseline)...")
    try:
        start = time.time()
        single_result = segmenter.process_image_array(img.copy(), plot=False)
        single_time = time.time() - start
        print(f"✅ Single image time: {single_time:.2f}s")
        print(f"   Result shape: {single_result.shape}")
    except Exception as e:
        print(f"❌ Single image processing failed: {e}")
        return

    # 5. True batch processing
    print("\n🚀 Running true batch processing...")
    try:
        start = time.time()
        results = segmenter.process_image_arrays_batch(batch, plot=False)
        batch_time = time.time() - start
        
        print(f"\n📊 BATCH PROCESSING RESULTS:")
        print(f"   Total batch time: {batch_time:.2f}s")
        print(f"   Time per image: {batch_time/BATCH_SIZE:.2f}s")
        print(f"   Speedup vs single: {single_time/(batch_time/BATCH_SIZE):.1f}x")
        
        # Check results
        successful = 0
        for i, res in enumerate(results):
            if res is not None:
                print(f"   Image {i+1}: {res.shape} ✅")
                successful += 1
            else:
                print(f"   Image {i+1}: FAILED ❌")
        
        success_rate = successful / BATCH_SIZE * 100
        print(f"\n🎯 Success rate: {successful}/{BATCH_SIZE} ({success_rate:.0f}%)")
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   Single image: {single_time:.2f}s per image")
        print(f"   Batch processing: {batch_time/BATCH_SIZE:.2f}s per image")
        if batch_time/BATCH_SIZE < single_time:
            print(f"   🎉 Batch is {single_time/(batch_time/BATCH_SIZE):.1f}x FASTER!")
        else:
            print(f"   ⚠️  Batch is {(batch_time/BATCH_SIZE)/single_time:.1f}x slower (unexpected)")
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 