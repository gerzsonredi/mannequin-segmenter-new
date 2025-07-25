#!/usr/bin/env python3
"""
Test True Batch Processing - Testing the new batch processing implementation
Verifies that multiple images are processed in a single GPU forward pass
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"

async def test_single_image():
    """Test single image processing to establish baseline"""
    print(f"🧪 Testing single image processing (baseline)...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"image_url": TEST_IMAGE_URL}
            
            start_time = time.time()
            async with session.post(f"{SERVICE_URL}/infer", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                end_time = time.time()
                
                if response.status == 200:
                    result = await response.json()
                    response_time = (end_time - start_time) * 1000
                    
                    print(f"✅ Single image: {response_time:.0f}ms")
                    print(f"   URL: {result.get('visualization_url', 'None')}")
                    return response_time, True
                else:
                    print(f"❌ Single image failed: {response.status}")
                    return 0, False
    except Exception as e:
        print(f"❌ Single image error: {e}")
        return 0, False

async def test_batch_processing(batch_size: int):
    """Test true batch processing with multiple images"""
    print(f"\n🚀 Testing TRUE BATCH PROCESSING with {batch_size} images...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"image_urls": [TEST_IMAGE_URL] * batch_size}
            
            start_time = time.time()
            async with session.post(f"{SERVICE_URL}/batch_infer", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    successful = result.get('successful_count', 0)
                    total = result.get('batch_size', 0)
                    
                    print(f"✅ Batch {batch_size}: {response_time:.0f}ms total")
                    print(f"   Success: {successful}/{total}")
                    print(f"   Time per image: {response_time/batch_size:.0f}ms")
                    print(f"   URLs: {len(result.get('visualization_urls', []))}")
                    
                    return response_time, successful == total, successful
                else:
                    error_text = await response.text()
                    print(f"❌ Batch {batch_size} failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return response_time, False, 0
    except Exception as e:
        print(f"❌ Batch {batch_size} error: {e}")
        return 0, False, 0

async def test_concurrent_single_requests(num_requests: int):
    """Test multiple concurrent single requests for comparison"""
    print(f"\n🔄 Testing {num_requests} concurrent single requests...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"image_url": TEST_IMAGE_URL}
            
            # Create all tasks
            async def single_request():
                async with session.post(f"{SERVICE_URL}/infer", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    return response.status == 200
            
            start_time = time.time()
            tasks = [single_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000
            successful = sum(1 for r in results if r is True)
            
            print(f"✅ Concurrent {num_requests}: {response_time:.0f}ms total")
            print(f"   Success: {successful}/{num_requests}")
            print(f"   Time per image: {response_time/num_requests:.0f}ms")
            
            return response_time, successful == num_requests, successful
    except Exception as e:
        print(f"❌ Concurrent {num_requests} error: {e}")
        return 0, False, 0

async def main():
    """Main test function"""
    print("🚀 TRUE BATCH PROCESSING TEST")
    print("=" * 60)
    
    # Test 1: Single image baseline
    single_time, single_success = await test_single_image()
    if not single_success:
        print("❌ Single image test failed, aborting")
        return
    
    # Test 2-4: Batch processing with different sizes
    batch_results = []
    for batch_size in [2, 3, 5]:
        batch_time, batch_success, batch_count = await test_batch_processing(batch_size)
        batch_results.append((batch_size, batch_time, batch_success, batch_count))
        await asyncio.sleep(2)  # Brief pause between tests
    
    # Test 5: Concurrent single requests for comparison
    concurrent_time, concurrent_success, concurrent_count = await test_concurrent_single_requests(5)
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print(f"📈 Baseline (Single image): {single_time:.0f}ms")
    
    print(f"\n🚀 TRUE BATCH PROCESSING:")
    for batch_size, batch_time, success, count in batch_results:
        if success:
            time_per_image = batch_time / batch_size
            efficiency = single_time / time_per_image if time_per_image > 0 else 0
            print(f"   {batch_size} images: {batch_time:.0f}ms total, {time_per_image:.0f}ms/image ({efficiency:.1f}x faster)")
        else:
            print(f"   {batch_size} images: FAILED")
    
    print(f"\n🔄 Concurrent Single Requests:")
    if concurrent_success:
        concurrent_time_per_image = concurrent_time / 5
        concurrent_efficiency = single_time / concurrent_time_per_image if concurrent_time_per_image > 0 else 0
        print(f"   5 concurrent: {concurrent_time:.0f}ms total, {concurrent_time_per_image:.0f}ms/image ({concurrent_efficiency:.1f}x faster)")
    else:
        print(f"   5 concurrent: FAILED")
    
    # Find best performing batch size
    successful_batches = [(size, time/size) for size, time, success, _ in batch_results if success]
    if successful_batches:
        best_size, best_time_per_image = min(successful_batches, key=lambda x: x[1])
        best_efficiency = single_time / best_time_per_image
        
        print(f"\n🎯 BEST PERFORMANCE:")
        print(f"   Batch size: {best_size} images")
        print(f"   Time per image: {best_time_per_image:.0f}ms")
        print(f"   Efficiency gain: {best_efficiency:.1f}x faster than single")
        print(f"   Throughput: {1000/best_time_per_image:.2f} images/second")
        
        if best_efficiency > 3:
            print(f"   🎉 EXCELLENT: True batch processing provides major speedup!")
        elif best_efficiency > 2:
            print(f"   ✅ GOOD: Significant batch processing benefit")
        elif best_efficiency > 1.2:
            print(f"   ⚠️  MODERATE: Some batch benefit")
        else:
            print(f"   ❌ POOR: Batch processing not providing expected speedup")
    else:
        print(f"\n❌ No successful batch processing - all batch sizes failed")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"true_batch_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "single_baseline": {"time_ms": single_time, "success": single_success},
            "batch_results": [
                {"batch_size": size, "time_ms": time, "success": success, "successful_count": count}
                for size, time, success, count in batch_results
            ],
            "concurrent_comparison": {"time_ms": concurrent_time, "success": concurrent_success, "successful_count": concurrent_count}
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 