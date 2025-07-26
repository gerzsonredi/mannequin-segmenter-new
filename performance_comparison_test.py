#!/usr/bin/env python3
"""
Model Pool Performance Comparison Test
Compares batch vs single request performance with 50 images each
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import statistics

# Test Configuration
BASE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
NUM_IMAGES = 50

# Test image URLs (mix of different sizes and types)
TEST_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80",
    "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", 
    "https://images.unsplash.com/photo-1558769132-cb1aea458c5e?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    "https://images.unsplash.com/photo-1596495578065-6e0763fa1178?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",
    "https://images.unsplash.com/photo-1594736797933-d0601ba2f4b7?ixlib=rb-4.0.3&auto=format&fit=crop&w=700&q=80",
    "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",
    "https://images.unsplash.com/photo-1574015974293-817f0ebebb74?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
    "https://images.unsplash.com/photo-1581803118522-7b72a50f7e9f?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",
    "https://images.unsplash.com/photo-1520006403909-838d6b92c22e?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
] * 5  # Repeat to get 50 images

async def get_pool_stats(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Get current model pool statistics."""
    try:
        async with session.get(f"{BASE_URL}/pool_stats", timeout=30) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def get_server_status(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Get current server status."""
    try:
        async with session.get(f"{BASE_URL}/status", timeout=30) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def test_batch_processing(session: aiohttp.ClientSession, image_urls: List[str]) -> Dict[str, Any]:
    """Test batch processing with all images at once."""
    print(f"\n🏊‍♂️ TESTING BATCH PROCESSING: {len(image_urls)} images")
    print("=" * 80)
    
    # Get initial pool stats
    initial_stats = await get_pool_stats(session)
    print(f"📊 Initial Pool Stats: {initial_stats.get('pool_statistics', {}).get('available_models', 'unknown')}/30 models available")
    
    start_time = time.time()
    
    try:
        payload = {"image_urls": image_urls}
        async with session.post(
            f"{BASE_URL}/batch_infer", 
            json=payload,
            timeout=aiohttp.ClientTimeout(total=1800)  # 30 minutes timeout
        ) as response:
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                successful_count = result.get("successful_count", 0)
                failed_count = result.get("failed_count", 0)
                
                print(f"✅ BATCH PROCESSING COMPLETED")
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Successful: {successful_count}/{len(image_urls)}")
                print(f"   Failed: {failed_count}")
                print(f"   Throughput: {successful_count/duration:.2f} images/second")
                
                # Get final pool stats  
                final_stats = await get_pool_stats(session)
                print(f"📊 Final Pool Stats: {final_stats.get('pool_statistics', {}).get('available_models', 'unknown')}/30 models available")
                
                return {
                    "success": True,
                    "duration": duration,
                    "successful_count": successful_count,
                    "failed_count": failed_count,
                    "throughput": successful_count/duration,
                    "images_per_second": successful_count/duration,
                    "total_images": len(image_urls),
                    "initial_stats": initial_stats,
                    "final_stats": final_stats
                }
            else:
                error_text = await response.text()
                print(f"❌ BATCH PROCESSING FAILED: HTTP {response.status}")
                print(f"   Error: {error_text[:200]}")
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"HTTP {response.status}: {error_text[:200]}",
                    "total_images": len(image_urls)
                }
                
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ BATCH PROCESSING EXCEPTION: {e}")
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
            "total_images": len(image_urls)
        }

async def single_image_request(session: aiohttp.ClientSession, image_url: str, index: int) -> Dict[str, Any]:
    """Process a single image request."""
    start_time = time.time()
    
    try:
        payload = {"image_url": image_url, "upload_to_s3": False}
        async with session.post(
            f"{BASE_URL}/infer",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes per image
        ) as response:
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "duration": duration,
                    "index": index,
                    "url": image_url
                }
            else:
                error_text = await response.text()
                return {
                    "success": False,
                    "duration": duration,
                    "index": index,
                    "url": image_url,
                    "error": f"HTTP {response.status}: {error_text[:100]}"
                }
                
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        return {
            "success": False,
            "duration": duration,
            "index": index,
            "url": image_url,
            "error": str(e)
        }

async def test_parallel_single_requests(session: aiohttp.ClientSession, image_urls: List[str]) -> Dict[str, Any]:
    """Test parallel single requests."""
    print(f"\n🔥 TESTING PARALLEL SINGLE REQUESTS: {len(image_urls)} images")
    print("=" * 80)
    
    # Get initial pool stats
    initial_stats = await get_pool_stats(session)
    print(f"📊 Initial Pool Stats: {initial_stats.get('pool_statistics', {}).get('available_models', 'unknown')}/30 models available")
    
    start_time = time.time()
    
    # Create tasks for all requests
    tasks = [
        single_image_request(session, url, i) 
        for i, url in enumerate(image_urls)
    ]
    
    print(f"🚀 Launching {len(tasks)} parallel requests...")
    
    # Execute all requests in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Analyze results
    successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed_results = [r for r in results if isinstance(r, dict) and not r.get("success")]
    exception_results = [r for r in results if not isinstance(r, dict)]
    
    successful_count = len(successful_results)
    failed_count = len(failed_results) + len(exception_results)
    
    # Calculate timing statistics
    if successful_results:
        durations = [r["duration"] for r in successful_results]
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        median_duration = statistics.median(durations)
    else:
        avg_duration = min_duration = max_duration = median_duration = 0
    
    print(f"✅ PARALLEL SINGLE REQUESTS COMPLETED")
    print(f"   Total Duration: {total_duration:.2f} seconds")
    print(f"   Successful: {successful_count}/{len(image_urls)}")
    print(f"   Failed: {failed_count}")
    print(f"   Throughput: {successful_count/total_duration:.2f} images/second")
    print(f"   Individual Request Stats:")
    print(f"     - Average: {avg_duration:.2f}s")
    print(f"     - Median: {median_duration:.2f}s") 
    print(f"     - Min: {min_duration:.2f}s")
    print(f"     - Max: {max_duration:.2f}s")
    
    # Get final pool stats
    final_stats = await get_pool_stats(session)
    print(f"📊 Final Pool Stats: {final_stats.get('pool_statistics', {}).get('available_models', 'unknown')}/30 models available")
    
    return {
        "success": True,
        "total_duration": total_duration,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "throughput": successful_count/total_duration,
        "images_per_second": successful_count/total_duration,
        "individual_stats": {
            "avg_duration": avg_duration,
            "median_duration": median_duration,
            "min_duration": min_duration,
            "max_duration": max_duration
        },
        "total_images": len(image_urls),
        "initial_stats": initial_stats,
        "final_stats": final_stats,
        "failed_requests": failed_results[:5]  # Show first 5 failures for debugging
    }

async def wait_for_service_ready(session: aiohttp.ClientSession) -> bool:
    """Wait for the service to be ready."""
    print("🔍 Checking if service is ready...")
    
    for attempt in range(10):
        try:
            async with session.get(f"{BASE_URL}/health", timeout=30) as response:
                if response.status == 200:
                    print("✅ Service is ready!")
                    return True
        except Exception as e:
            print(f"⏳ Attempt {attempt + 1}/10: {e}")
            
        await asyncio.sleep(10)
    
    print("❌ Service not ready after 10 attempts")
    return False

async def main():
    """Main test function."""
    print("🏊‍♂️ MODEL POOL PERFORMANCE COMPARISON TEST")
    print("=" * 80)
    print(f"🎯 Target: {BASE_URL}")
    print(f"📦 Test Images: {NUM_IMAGES}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Prepare test images
    test_urls = TEST_IMAGE_URLS[:NUM_IMAGES]
    
    # Create session with appropriate timeouts and limits
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Wait for service to be ready
        if not await wait_for_service_ready(session):
            return
        
        # Get initial server status
        server_status = await get_server_status(session)
        print(f"\n📊 Initial Server Status:")
        if "model_pool" in server_status:
            pool_info = server_status["model_pool"]
            print(f"   Model Pool: {pool_info.get('available_models', 'unknown')}/{pool_info.get('pool_size', 'unknown')} available")
            print(f"   Active Requests: {pool_info.get('active_requests', 'unknown')}")
        
        results = {}
        
        # Test 1: Batch Processing
        print(f"\n" + "="*80)
        print("TEST 1: BATCH PROCESSING")
        print("="*80)
        results["batch"] = await test_batch_processing(session, test_urls)
        
        # Wait a bit between tests
        print(f"\n⏳ Waiting 30 seconds between tests...")
        await asyncio.sleep(30)
        
        # Test 2: Parallel Single Requests  
        print(f"\n" + "="*80)
        print("TEST 2: PARALLEL SINGLE REQUESTS")
        print("="*80)
        results["parallel_single"] = await test_parallel_single_requests(session, test_urls)
        
        # Final Analysis
        print(f"\n" + "="*80)
        print("📊 PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        
        batch_result = results["batch"]
        single_result = results["parallel_single"]
        
        if batch_result.get("success") and single_result.get("success"):
            batch_time = batch_result["duration"]
            single_time = single_result["total_duration"]
            batch_throughput = batch_result["throughput"]
            single_throughput = single_result["throughput"]
            
            print(f"🏊‍♂️ BATCH PROCESSING:")
            print(f"   Duration: {batch_time:.2f} seconds")
            print(f"   Throughput: {batch_throughput:.2f} images/second")
            print(f"   Success Rate: {batch_result['successful_count']}/{NUM_IMAGES}")
            
            print(f"\n🔥 PARALLEL SINGLE REQUESTS:")
            print(f"   Duration: {single_time:.2f} seconds") 
            print(f"   Throughput: {single_throughput:.2f} images/second")
            print(f"   Success Rate: {single_result['successful_count']}/{NUM_IMAGES}")
            print(f"   Avg Request Time: {single_result['individual_stats']['avg_duration']:.2f}s")
            
            # Winner analysis
            print(f"\n🏆 WINNER:")
            if batch_time < single_time:
                speedup = single_time / batch_time
                print(f"   🏊‍♂️ BATCH PROCESSING wins by {speedup:.2f}x ({batch_time:.2f}s vs {single_time:.2f}s)")
            else:
                speedup = batch_time / single_time
                print(f"   🔥 PARALLEL SINGLE REQUESTS wins by {speedup:.2f}x ({single_time:.2f}s vs {batch_time:.2f}s)")
            
            print(f"\n💡 EFFICIENCY ANALYSIS:")
            throughput_diff = abs(batch_throughput - single_throughput)
            throughput_ratio = max(batch_throughput, single_throughput) / min(batch_throughput, single_throughput)
            print(f"   Throughput difference: {throughput_diff:.2f} images/second")
            print(f"   Throughput ratio: {throughput_ratio:.2f}x")
            
        else:
            print("❌ One or both tests failed, cannot compare results")
            if not batch_result.get("success"):
                print(f"   Batch test error: {batch_result.get('error', 'unknown')}")
            if not single_result.get("success"):
                print(f"   Single test error: {single_result.get('error', 'unknown')}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "test_config": {
                    "num_images": NUM_IMAGES,
                    "base_url": BASE_URL
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(main()) 