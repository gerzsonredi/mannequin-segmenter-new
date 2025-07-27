#!/usr/bin/env python3
"""
Test 60 concurrent single requests to maximize model pool utilization
NOT using batch processing - each request goes to /infer endpoint
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime

# Configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
NUM_REQUESTS = 60
TIMEOUT = 120  # 2 minutes timeout

async def send_single_request(session, request_id):
    """Send a single inference request"""
    try:
        start_time = time.time()
        
        payload = {
            "image_url": TEST_IMAGE_URL,
            "upload_s3": False  # Use the S3 upload fix for faster processing
        }
        
        async with session.post(
            f"{SERVICE_URL}/infer",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                success = result.get('success', False) or result.get('inference_completed', False)
                return {
                    'request_id': request_id,
                    'success': success,
                    'duration': duration,
                    'status_code': response.status,
                    'response': result
                }
            else:
                error_text = await response.text()
                return {
                    'request_id': request_id,
                    'success': False,
                    'duration': duration,
                    'status_code': response.status,
                    'error': error_text
                }
                
    except Exception as e:
        return {
            'request_id': request_id,
            'success': False,
            'duration': time.time() - start_time if 'start_time' in locals() else 0,
            'error': str(e)
        }

async def test_60_concurrent_requests():
    """Test 60 concurrent single requests"""
    print(f"🧪 TESTING {NUM_REQUESTS} CONCURRENT SINGLE REQUESTS")
    print("=" * 60)
    print(f"🎯 Target: Maximize 60-model pool utilization")
    print(f"🔗 Service: {SERVICE_URL}")
    print(f"🖼️  Image: {TEST_IMAGE_URL}")
    print(f"⚡ S3 Upload: Disabled (using fix)")
    print("")
    
    # Configure aiohttp with high limits
    connector = aiohttp.TCPConnector(
        limit=100,  # Total connection pool size
        limit_per_host=100,  # Connections per host
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print(f"🚀 Sending {NUM_REQUESTS} concurrent requests...")
        overall_start = time.time()
        
        # Create all tasks
        tasks = [
            send_single_request(session, i+1) 
            for i in range(NUM_REQUESTS)
        ]
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        overall_end = time.time()
        total_time = overall_end - overall_start
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
        failed_requests = [r for r in results if not (isinstance(r, dict) and r.get('success', False))]
        
        success_count = len(successful_requests)
        failure_count = len(failed_requests)
        
        # Calculate performance metrics
        if successful_requests:
            durations = [r['duration'] for r in successful_requests]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0
        
        throughput = success_count / total_time if total_time > 0 else 0
        
        # Print results
        print(f"")
        print(f"📊 RESULTS:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful: {success_count}/{NUM_REQUESTS} ({success_count/NUM_REQUESTS*100:.1f}%)")
        print(f"   Failed: {failure_count}")
        print(f"   Throughput: {throughput:.2f} images/second")
        print(f"")
        print(f"📈 TIMING ANALYSIS:")
        print(f"   Average request: {avg_duration:.2f}s")
        print(f"   Fastest request: {min_duration:.2f}s")
        print(f"   Slowest request: {max_duration:.2f}s")
        print(f"")
        
        # Target analysis
        target_time = 6.0  # Target: process in 6 seconds
        if total_time <= target_time:
            print(f"🎉 TARGET ACHIEVED! {NUM_REQUESTS} images in {total_time:.2f}s (≤ {target_time}s target)")
        else:
            improvement_needed = total_time - target_time
            speedup_needed = total_time / target_time
            print(f"⚠️ Target missed: {NUM_REQUESTS} images in {total_time:.2f}s")
            print(f"   Target: ≤ {target_time}s")
            print(f"   Need {improvement_needed:.2f}s faster ({speedup_needed:.1f}x speedup)")
        
        print(f"")
        print(f"🔍 MODEL POOL UTILIZATION ANALYSIS:")
        if success_count == NUM_REQUESTS and total_time <= target_time:
            print(f"   ✅ Perfect utilization achieved!")
        elif success_count == NUM_REQUESTS:
            print(f"   ✅ All models working, but need speed optimization")
        else:
            print(f"   ⚠️  Only {success_count}/{NUM_REQUESTS} models utilized")
            print(f"   🔧 Possible concurrency/threading limits")
        
        # Error analysis
        if failed_requests:
            print(f"")
            print(f"❌ FAILURE ANALYSIS:")
            error_types = {}
            for req in failed_requests:
                if isinstance(req, dict):
                    error = req.get('error', 'Unknown error')
                    status = req.get('status_code', 'Unknown status')
                    error_key = f"{status}: {error}"
                else:
                    error_key = f"Exception: {str(req)}"
                
                error_types[error_key] = error_types.get(error_key, 0) + 1
            
            for error, count in error_types.items():
                print(f"   {error}: {count} occurrences")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"concurrent_60_singles_results_{timestamp}.json"
        
        results_data = {
            "test_config": {
                "num_requests": NUM_REQUESTS,
                "service_url": SERVICE_URL,
                "test_image": TEST_IMAGE_URL,
                "upload_s3": False,
                "timestamp": timestamp
            },
            "performance": {
                "total_time": total_time,
                "success_count": success_count,
                "failure_count": failure_count,
                "throughput": throughput,
                "avg_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "target_achieved": total_time <= target_time
            },
            "detailed_results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"")
        print(f"💾 Results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(test_60_concurrent_requests()) 