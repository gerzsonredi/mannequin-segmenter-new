#!/usr/bin/env python3
"""
Test 20 concurrent requests to deployed Cloud Run service
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime

# Deployed service URL
SERVICE_URL = "https://mannequin-segmenter-234382015820.europe-west4.run.app"

# Test images - mix of Remix and other images
TEST_IMAGES = [
    "https://media.remix.eu/files/20-2025/Roklya-Atos-Lombardini-131973196b.jpg",  # Remix image
]

async def process_single_image(session, image_url, request_id):
    """Process a single image and return results"""
    print(f"📤 Request {request_id:2d}: Starting...")
    
    payload = {
        "image_url": image_url,
        "upload_s3": True
    }
    
    start_time = time.time()
    try:
        async with session.post(
            f"{SERVICE_URL}/infer",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)  # 2 minutes timeout
        ) as response:
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                data = await response.json()
                s3_url = data.get('visualization_url', 'No S3 URL')
                print(f"✅ Request {request_id:2d}: SUCCESS in {duration:.2f}s")
                return {
                    "id": request_id,
                    "success": True,
                    "duration": duration,
                    "s3_url": s3_url,
                    "image_url": image_url
                }
            else:
                error_text = await response.text()
                print(f"❌ Request {request_id:2d}: FAILED {response.status} in {duration:.2f}s")
                return {
                    "id": request_id,
                    "success": False,
                    "duration": duration,
                    "error": f"HTTP {response.status}: {error_text}",
                    "image_url": image_url
                }
                
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Request {request_id:2d}: ERROR in {duration:.2f}s - {e}")
        return {
            "id": request_id,
            "success": False,
            "duration": duration,
            "error": str(e),
            "image_url": image_url
        }

async def run_concurrent_test(num_requests=60):
    """Run concurrent tests"""
    print(f"🚀 STARTING {num_requests} CONCURRENT REQUESTS")
    print("=" * 70)
    print(f"🎯 Service: {SERVICE_URL}")
    print(f"🕐 Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Create tasks for concurrent execution
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            # Cycle through test images
            image_url = TEST_IMAGES[i % len(TEST_IMAGES)]
            task = process_single_image(session, image_url, i + 1)
            tasks.append(task)
        
        # Start all requests at the same time
        print(f"🚦 Launching {num_requests} concurrent requests...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
    
    # Analyze results
    print("\n" + "=" * 70)
    print("📊 RESULTS ANALYSIS")
    print("=" * 70)
    
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
    exceptions = [r for r in results if not isinstance(r, dict)]
    
    print(f"✅ Successful: {len(successful)}/{num_requests}")
    print(f"❌ Failed: {len(failed)}/{num_requests}")
    print(f"💥 Exceptions: {len(exceptions)}/{num_requests}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    if successful:
        durations = [r["duration"] for r in successful]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        print(f"📈 Average duration: {avg_duration:.2f}s")
        print(f"🏃 Fastest: {min_duration:.2f}s")
        print(f"🐌 Slowest: {max_duration:.2f}s")
        print(f"🚀 Throughput: {len(successful)/total_time:.2f} images/second")
    
    # Show individual results
    print("\n" + "=" * 70)
    print("📋 INDIVIDUAL RESULTS")
    print("=" * 70)
    
    for result in results:
        if isinstance(result, dict):
            if result["success"]:
                print(f"✅ #{result['id']:2d}: {result['duration']:5.2f}s - {result['s3_url'][:50]}...")
            else:
                print(f"❌ #{result['id']:2d}: {result['duration']:5.2f}s - {result['error'][:50]}...")
        else:
            print(f"💥 Exception: {str(result)[:60]}...")
    
    # Show S3 URLs for successful results
    if successful:
        print("\n" + "=" * 70)
        print("🔗 S3 RESULT URLS")
        print("=" * 70)
        for result in successful[:5]:  # Show first 5 URLs
            print(f"🖼️  #{result['id']:2d}: {result['s3_url']}")
        if len(successful) > 5:
            print(f"... and {len(successful) - 5} more successful results")

async def main():
    """Main function"""
    try:
        await run_concurrent_test(60)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 