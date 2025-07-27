#!/usr/bin/env python3
"""
Quick concurrent test for optimized DeepLabV3-MobileViT
"""

import asyncio
import aiohttp
import time

async def send_request(session, request_id):
    """Send single request"""
    try:
        start_time = time.time()
        
        payload = {
            "image_url": "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
            "upload_s3": False
        }
        
        async with session.post(
            "http://localhost:5001/infer",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            duration = time.time() - start_time
            
            if response.status == 200:
                result = await response.json()
                success = result.get('success', False) or result.get('inference_completed', False)
                return {'id': request_id, 'success': success, 'duration': duration}
            else:
                return {'id': request_id, 'success': False, 'duration': duration, 'error': response.status}
                
    except Exception as e:
        return {'id': request_id, 'success': False, 'duration': 0, 'error': str(e)}

async def test_concurrent(num_requests=10):
    """Test concurrent requests"""
    print(f"🚀 Testing {num_requests} concurrent requests...")
    
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=20)
    timeout = aiohttp.ClientTimeout(total=120)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        overall_start = time.time()
        
        # Create all tasks
        tasks = [send_request(session, i+1) for i in range(num_requests)]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - overall_start
        
        # Analyze results
        successful = [r for r in results if r.get('success', False)]
        success_count = len(successful)
        
        if successful:
            durations = [r['duration'] for r in successful]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0
        
        throughput = success_count / total_time if total_time > 0 else 0
        
        print(f"\n📊 RESULTS:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful: {success_count}/{num_requests}")
        print(f"   Throughput: {throughput:.2f} img/s")
        print(f"   Avg request: {avg_duration:.2f}s")
        print(f"   Min: {min_duration:.2f}s, Max: {max_duration:.2f}s")
        
        # Cloud Run projection
        cloud_run_time = avg_duration  # Each instance handles 1 request
        images_in_6s = 6 / cloud_run_time if cloud_run_time > 0 else 0
        
        print(f"\n🎯 CLOUD RUN PROJECTION (60 instances):")
        print(f"   Single instance time: {cloud_run_time:.2f}s")
        print(f"   Max images in 6s: {min(60, images_in_6s):.0f} images")
        
        if images_in_6s >= 50:
            print(f"   🎉 TARGET ACHIEVABLE! (50+ images in 6s)")
        else:
            print(f"   ⚠️ Target: {images_in_6s:.0f} images (need {50/images_in_6s:.1f}x improvement)")

if __name__ == "__main__":
    asyncio.run(test_concurrent(10)) 