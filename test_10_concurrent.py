#!/usr/bin/env python3
"""
Test 10 concurrent requests to deployed Cloud Run service - complete results
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime

# Deployed service URL
SERVICE_URL = "https://mannequin-segmenter-234382015820.europe-west4.run.app"

# Test images
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
            timeout=aiohttp.ClientTimeout(total=180)  # 3 minutes timeout
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

async def main():
    """Run 10 concurrent tests and show complete results"""
    print("🚀 STARTING 10 CONCURRENT REQUESTS - COMPLETE TEST")
    print("=" * 70)
    print(f"🎯 Service: {SERVICE_URL}")
    print(f"🕐 Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Create tasks for concurrent execution
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i in range(10):
            # Cycle through test images
            image_url = TEST_IMAGES[i % len(TEST_IMAGES)]
            task = process_single_image(session, image_url, i + 1)
            tasks.append(task)
        
        # Start all requests at the same time
        print("🚦 Launching 10 concurrent requests...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
    
    # Analyze results
    print("\n" + "=" * 70)
    print("📊 COMPLETE RESULTS ANALYSIS")
    print("=" * 70)
    
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
    exceptions = [r for r in results if not isinstance(r, dict)]
    
    print(f"✅ Successful: {len(successful)}/10")
    print(f"❌ Failed: {len(failed)}/10")
    print(f"💥 Exceptions: {len(exceptions)}/10")
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
    
    # Show ALL individual results
    print("\n" + "=" * 70)
    print("📋 ALL INDIVIDUAL RESULTS")
    print("=" * 70)
    
    # Sort by request ID
    sorted_results = sorted([r for r in results if isinstance(r, dict)], key=lambda x: x["id"])
    
    for result in sorted_results:
        if result["success"]:
            print(f"✅ #{result['id']:2d}: {result['duration']:5.2f}s - SUCCESS")
        else:
            print(f"❌ #{result['id']:2d}: {result['duration']:5.2f}s - {result['error'][:40]}...")
    
    # Show all S3 URLs
    if successful:
        print("\n" + "=" * 70)
        print("🔗 ALL S3 RESULT URLS")
        print("=" * 70)
        for result in sorted([r for r in successful], key=lambda x: x["id"]):
            print(f"🖼️  #{result['id']:2d}: {result['s3_url']}")

if __name__ == "__main__":
    asyncio.run(main()) 