#!/usr/bin/env python3
"""
Debug Batch Processing Locally
Test the new batch processing implementation directly without going through the API
"""

import sys
import os
import numpy as np
import time
from PIL import Image
import requests
import io

# Add tools to path
sys.path.append('tools')

from BirefNet import BiRefNetSegmenter

def download_test_image():
    """Download test image for local testing"""
    url = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
    
    try:
        print(f"📥 Downloading test image...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to PIL Image then to numpy
        pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_np = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        image_bgr = image_np[:, :, ::-1]
        
        print(f"✅ Image downloaded: {image_bgr.shape}")
        return image_bgr
    except Exception as e:
        print(f"❌ Failed to download image: {e}")
        return None

def test_single_processing(segmenter, image):
    """Test single image processing"""
    print(f"\n🧪 Testing single image processing...")
    try:
        start_time = time.time()
        result = segmenter.process_image_array(image, plot=False)
        end_time = time.time()
        
        if result is not None:
            print(f"✅ Single processing: {(end_time - start_time)*1000:.0f}ms")
            print(f"   Result shape: {result.shape}")
            return True
        else:
            print(f"❌ Single processing failed: result is None")
            return False
    except Exception as e:
        print(f"❌ Single processing error: {e}")
        return False

def test_batch_processing(segmenter, images):
    """Test new batch processing"""
    print(f"\n🚀 Testing TRUE BATCH PROCESSING with {len(images)} images...")
    try:
        start_time = time.time()
        results = segmenter.process_image_arrays_batch(images, plot=False)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        successful_results = [r for r in results if r is not None]
        
        print(f"✅ Batch processing: {total_time:.0f}ms total")
        print(f"   Successful: {len(successful_results)}/{len(images)}")
        print(f"   Time per image: {total_time/len(images):.0f}ms")
        
        for i, result in enumerate(results):
            if result is not None:
                print(f"   Image {i+1}: {result.shape}")
            else:
                print(f"   Image {i+1}: FAILED")
        
        return len(successful_results) == len(images)
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🐛 LOCAL BATCH PROCESSING DEBUG")
    print("=" * 50)
    
    # Download test image
    test_image = download_test_image()
    if test_image is None:
        print("❌ Cannot proceed without test image")
        return
    
    # Initialize segmenter
    print(f"\n🤖 Initializing BiRefNet segmenter...")
    try:
        segmenter = BiRefNetSegmenter(
            precision="fp16",
            thickness_threshold=200
        )
        print(f"✅ Segmenter initialized")
    except Exception as e:
        print(f"❌ Failed to initialize segmenter: {e}")
        return
    
    # Test single processing
    single_success = test_single_processing(segmenter, test_image)
    
    if not single_success:
        print("❌ Single processing failed, skipping batch tests")
        return
    
    # Test batch processing with different sizes
    batch_sizes = [2, 3, 5]
    
    for batch_size in batch_sizes:
        # Create batch by duplicating the test image
        batch_images = [test_image.copy() for _ in range(batch_size)]
        
        batch_success = test_batch_processing(segmenter, batch_images)
        
        if batch_success:
            print(f"✅ Batch size {batch_size}: SUCCESS")
        else:
            print(f"❌ Batch size {batch_size}: FAILED")
        
        time.sleep(1)  # Brief pause
    
    print(f"\n🎯 Debug completed!")

if __name__ == "__main__":
    main() 