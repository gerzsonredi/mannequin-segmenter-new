#!/usr/bin/env python3
"""
Cloud Run Throughput Test - Test the deployed 60-model service
Measure real-world throughput with concurrent requests to Cloud Run
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
import numpy as np

# Test configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"

# Progressive test configurations
TEST_CONFIGS = [
    {"concurrent": 10, "total_images": 20, "name": "Light Load"},
    {"concurrent": 30, "total_images": 60, "name": "Medium Load"}, 
    {"concurrent": 50, "total_images": 100, "name": "Heavy Load"},
    {"concurrent": 60, "total_images": 120, "name": "Max Load"}
]

TIMEOUT = 120  # seconds per request

async def send_inference_request(session, request_id):
    """Send a single /infer request"""
    start_time = time.time()
    
    try:
        payload = {
            "image_url": TEST_IMAGE_URL,
            "upload_s3": False  # Focus on inference speed, not S3 upload
        }
        
        async with session.post(
            f"{SERVICE_URL}/infer",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            
            duration = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                
                return {
                    "request_id": request_id,
                    "duration": duration,
                    "success": True,
                    "status_code": response.status
                }
            else:
                error_text = await response.text()
                
                return {
                    "request_id": request_id,
                    "duration": duration,
                    "success": False,
                    "status_code": response.status,
                    "error": error_text[:100]
                }
                
    except Exception as e:
        duration = time.time() - start_time
        
        return {
            "request_id": request_id,
            "duration": duration,
            "success": False,
            "error": str(e)
        }

async def monitor_service_stats(session, test_name):
    """Monitor pool statistics during the test"""
    try:
        while True:
            try:
                async with session.get(f"{SERVICE_URL}/pool_stats", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        pool_data = await response.json()
                        available = pool_data.get('available_models', '?')
                        total = pool_data.get('total_models', '?')
                        active = pool_data.get('active_models', '?')
                        
                        print(f"📊 [{test_name}] Pool: {available}/{total} available, {active} active")
                    else:
                        print(f"⚠️  [{test_name}] Pool stats unavailable")
            except:
                pass  # Ignore errors during monitoring
            
            await asyncio.sleep(3)  # Check every 3 seconds
            
    except asyncio.CancelledError:
        print(f"📊 [{test_name}] Pool monitoring stopped")

async def test_throughput_config(config):
    """Test a specific concurrent/total configuration"""
    concurrent = config["concurrent"]
    total_images = config["total_images"]
    test_name = config["name"]
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING: {test_name}")
    print(f"   Concurrent requests: {concurrent}")
    print(f"   Total images: {total_images}")
    print(f"   Expected batches: {total_images // concurrent}")
    print(f"{'='*70}")
    
    # Check service health first
    print("🔍 Checking service health...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVICE_URL}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Service healthy: {health_data.get('status', 'unknown')}")
                else:
                    print(f"⚠️  Service health check failed: {response.status}")
                    return None
    except Exception as e:
        print(f"❌ Cannot reach service: {e}")
        return None
    
    # Create session with appropriate limits
    connector = aiohttp.TCPConnector(
        limit=concurrent + 10,  # Total connection pool
        limit_per_host=concurrent + 5,  # Connections per host
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        
        # Start monitoring in background
        monitor_task = asyncio.create_task(monitor_service_stats(session, test_name))
        
        try:
            # Process images in batches of 'concurrent' size
            all_results = []
            batches = []
            
            # Split total images into batches
            for i in range(0, total_images, concurrent):
                batch_size = min(concurrent, total_images - i)
                batch_ids = list(range(i + 1, i + batch_size + 1))
                batches.append(batch_ids)
            
            print(f"🚀 Processing {len(batches)} batches...")
            
            overall_start = time.time()
            
            for batch_num, batch_ids in enumerate(batches):
                print(f"\n🔄 Batch {batch_num + 1}/{len(batches)}: {len(batch_ids)} requests")
                batch_start = time.time()
                
                # Create concurrent requests for this batch
                tasks = []
                for request_id in batch_ids:
                    task = asyncio.create_task(send_inference_request(session, request_id))
                    tasks.append(task)
                
                # Wait for all requests in this batch to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results (handle exceptions)
                processed_batch_results = []
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        processed_batch_results.append({
                            "request_id": batch_ids[i],
                            "duration": time.time() - batch_start,
                            "success": False,
                            "error": str(result)
                        })
                    else:
                        processed_batch_results.append(result)
                
                all_results.extend(processed_batch_results)
                
                # Batch summary
                batch_duration = time.time() - batch_start
                successful_in_batch = sum(1 for r in processed_batch_results if r["success"])
                batch_throughput = successful_in_batch / batch_duration if batch_duration > 0 else 0
                
                print(f"   ✅ Batch {batch_num + 1}: {successful_in_batch}/{len(batch_ids)} successful in {batch_duration:.2f}s")
                print(f"   📈 Batch throughput: {batch_throughput:.2f} images/second")
                
                # Brief pause between batches to avoid overwhelming
                if batch_num < len(batches) - 1:
                    await asyncio.sleep(1)
            
            overall_duration = time.time() - overall_start
            
        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    # Analyze overall results
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]
    
    if successful_results:
        durations = [r["duration"] for r in successful_results]
        overall_throughput = len(successful_results) / overall_duration
        avg_duration = sum(durations) / len(durations)
        
        print(f"\n🏆 {test_name.upper()} RESULTS:")
        print(f"   ✅ Successful: {len(successful_results)}/{total_images}")
        print(f"   ❌ Failed: {len(failed_results)}")
        print(f"   ⏱️  Total duration: {overall_duration:.2f}s")
        print(f"   📈 Overall throughput: {overall_throughput:.2f} images/second")
        print(f"   🕐 Avg request time: {avg_duration:.2f}s")
        print(f"   📊 Request times: min={min(durations):.2f}s, max={max(durations):.2f}s")
        
        # Calculate efficiency vs theoretical max
        theoretical_max_throughput = 60 / avg_duration  # If all 60 models working optimally
        efficiency = (overall_throughput / theoretical_max_throughput) * 100 if theoretical_max_throughput > 0 else 0
        
        print(f"   💡 Pool efficiency: {efficiency:.1f}% of theoretical max")
        
        return {
            "test_name": test_name,
            "concurrent": concurrent,
            "total_images": total_images,
            "successful": len(successful_results),
            "failed": len(failed_results),
            "duration": overall_duration,
            "throughput": overall_throughput,
            "avg_request_time": avg_duration,
            "efficiency": efficiency,
            "individual_times": durations
        }
    else:
        print(f"\n❌ {test_name.upper()} FAILED!")
        for result in failed_results[:5]:  # Show first 5 errors
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        return {
            "test_name": test_name,
            "concurrent": concurrent,
            "total_images": total_images,
            "successful": 0,
            "failed": len(failed_results),
            "duration": overall_duration,
            "throughput": 0,
            "avg_request_time": 0,
            "efficiency": 0,
            "error": "All requests failed"
        }

def analyze_cloud_run_performance(all_results):
    """Analyze Cloud Run performance across different loads"""
    print(f"\n{'='*70}")
    print(f"📊 CLOUD RUN PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    successful_tests = [r for r in all_results if r["successful"] > 0]
    
    if not successful_tests:
        print("❌ No successful tests to analyze")
        return
    
    print(f"📈 THROUGHPUT SCALING:")
    
    for result in successful_tests:
        concurrent = result["concurrent"]
        throughput = result["throughput"]
        efficiency = result["efficiency"]
        
        if efficiency > 80:
            status = "🎉 Excellent"
        elif efficiency > 60:
            status = "👍 Good"
        elif efficiency > 40:
            status = "⚠️  Fair"
        else:
            status = "❌ Poor"
        
        print(f"   {concurrent} concurrent: {throughput:.2f} img/s ({efficiency:.1f}% efficiency) {status}")
    
    # Find best performance
    best_result = max(successful_tests, key=lambda x: x["throughput"])
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Configuration: {best_result['test_name']} ({best_result['concurrent']} concurrent)")
    print(f"   Throughput: {best_result['throughput']:.2f} images/second")
    print(f"   Efficiency: {best_result['efficiency']:.1f}% of theoretical max")
    
    # 50 images in 6 seconds target check
    target_throughput = 50 / 6  # 8.33 images/second needed
    
    print(f"\n🎯 TARGET ANALYSIS (50 images in 6 seconds):")
    print(f"   Required throughput: {target_throughput:.2f} images/second")
    print(f"   Best achieved: {best_result['throughput']:.2f} images/second")
    
    if best_result['throughput'] >= target_throughput:
        print(f"   ✅ TARGET ACHIEVED! ({best_result['throughput']/target_throughput:.1f}x better than needed)")
    else:
        improvement_needed = target_throughput / best_result['throughput']
        print(f"   🔄 Need {improvement_needed:.1f}x improvement to reach target")
    
    # Project actual 50-image performance
    projected_50_time = 50 / best_result['throughput']
    print(f"   📊 Projected 50 images time: {projected_50_time:.1f}s")

async def main():
    print("🚀 CLOUD RUN THROUGHPUT TEST")
    print("Testing deployed 60-model BiRefNet_lite service")
    print("=" * 70)
    
    all_results = []
    
    # Test each configuration
    for config in TEST_CONFIGS:
        try:
            result = await test_throughput_config(config)
            if result:
                all_results.append(result)
            
            # Pause between tests
            if config != TEST_CONFIGS[-1]:
                print(f"\n⏸️  Pausing 5 seconds before next test...")
                await asyncio.sleep(5)
                
        except Exception as e:
            print(f"❌ Test {config['name']} failed: {e}")
            all_results.append({
                "test_name": config["name"],
                "concurrent": config["concurrent"],
                "total_images": config["total_images"],
                "successful": 0,
                "failed": config["total_images"],
                "duration": 0,
                "throughput": 0,
                "efficiency": 0,
                "error": str(e)
            })
    
    # Analyze results
    analyze_cloud_run_performance(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cloud_run_throughput_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "test_type": "cloud_run_throughput",
            "timestamp": timestamp,
            "service_url": SERVICE_URL,
            "test_image": TEST_IMAGE_URL,
            "configurations": TEST_CONFIGS,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎉 CLOUD RUN THROUGHPUT TEST COMPLETED")
    print(f"{'='*70}")
    
    successful_tests = sum(1 for r in all_results if r["successful"] > 0)
    total_tests = len(all_results)
    
    print(f"✅ Successful test configurations: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        best_throughput = max((r["throughput"] for r in all_results if r["throughput"] > 0), default=0)
        print(f"🏆 Best throughput achieved: {best_throughput:.2f} images/second")
        print(f"🎯 60-model Cloud Run service performance verified!")
    else:
        print(f"❌ All tests failed. Check service deployment.")

if __name__ == "__main__":
    asyncio.run(main()) 