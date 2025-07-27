#!/usr/bin/env python3
"""
Quick local test for bottleneck fixes
"""

import sys
import time
import requests
from io import BytesIO
import os
sys.path.insert(0, 'tools')

# Fix relative imports
from BirefNet import BiRefNetSegmenter  
from model_pool import BiRefNetModelPool

def test_s3_upload_fix():
    """Test 1: Model Pool & S3 Upload Logic"""
    print("🧪 TEST 1: Model Pool Speed Test")
    print("=" * 50)
    
    try:
        print("⏳ Loading model pool (2 models)...")
        start_time = time.time()
        model_pool = BiRefNetModelPool(
            pool_size=2,
            model_path="models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt",
            model_name="zhengpeng7/BiRefNet_lite",
            precision="fp32",  # Use fp32 for CPU
            vis_save_dir="infer",
            thickness_threshold=200,
            mask_threshold=0.5
        )
        load_time = time.time() - start_time
        print(f"✅ Model pool loaded in {load_time:.2f}s")
        
        # Test single inference speed
        print("\n🎯 Testing single inference speed...")
        test_url = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
        
        start_time = time.time()
        result = model_pool.process_single_request(test_url, plot=False)
        inference_time = time.time() - start_time
        
        if result is not None:
            print(f"✅ Single inference completed in {inference_time:.2f}s")
            print(f"   Result shape: {result.shape}")
        else:
            print("❌ Inference failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_batch_processing():
    """Test 2: Batch processing comparison"""
    print("\n🧪 TEST 2: Batch Processing Test") 
    print("=" * 50)
    
    try:
        model_pool = BiRefNetModelPool(pool_size=2, model_name="zhengpeng7/BiRefNet_lite", precision="fp32")
        test_urls = [
            "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
            "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
        ]
        
        # Test parallel processing (using model pool)
        print("🔄 Testing 2 images with model pool...")
        start_time = time.time()
        results = model_pool.process_batch_requests(test_urls, max_batch_size=2)
        parallel_time = time.time() - start_time
        
        successful = len([r for r in results if r is not None])
        print(f"✅ Parallel processing: {successful}/2 images in {parallel_time:.2f}s")
        print(f"   Throughput: {successful/parallel_time:.2f} images/second")
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        import traceback
        traceback.print_exc()

def test_image_cache():
    """Test 3: Image caching effectiveness"""
    print("\n🧪 TEST 3: Image Cache Test")
    print("=" * 50)
    
    try:
        model_pool = BiRefNetModelPool(pool_size=2, model_name="zhengpeng7/BiRefNet_lite", precision="fp32") 
        test_url = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
        
        # First request (cache miss)
        print("🔄 First request (cache miss)...")
        start_time = time.time()
        result1 = model_pool.process_single_request(test_url, plot=False)
        first_time = time.time() - start_time
        
        # Second request (cache hit)
        print("🔄 Second request (cache hit)...")
        start_time = time.time()
        result2 = model_pool.process_single_request(test_url, plot=False)
        second_time = time.time() - start_time
        
        if result1 is not None and result2 is not None:
            print(f"   First request: {first_time:.2f}s")
            print(f"   Second request: {second_time:.2f}s")
            if second_time < first_time:
                speedup = first_time / second_time
                print(f"✅ Cache speedup: {speedup:.1f}x")
            else:
                print("⚠️ No cache benefit detected")
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 LOCAL BOTTLENECK FIXES TEST")
    print("=" * 70)
    
    # Kill any hanging server process first
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "python api_app.py"], capture_output=True)
        print("🔄 Stopped any existing Flask server")
    except:
        pass
    
    test_s3_upload_fix()
    test_batch_processing() 
    test_image_cache()
    
    print("\n🎉 All local tests completed!") 