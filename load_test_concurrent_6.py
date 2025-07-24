#!/usr/bin/env python3
"""
Minimal Load Test - 6 Concurrent Requests
Testing the lowest safe concurrency for GPU/Cloud Run.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any

SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer"
WORKING_IMAGE = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
CONCURRENT_REQUESTS = 20
TIMEOUT = 600

PAYLOAD = {
    "image_url": WORKING_IMAGE
}

async def send_minimal_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    start_time = time.time()
    try:
        async with session.post(
            SERVICE_URL,
            json=PAYLOAD,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            if response.status == 200:
                result = await response.json()
                return {
                    "request_id": request_id,
                    "success": True,
                    "response_time_ms": response_time,
                    "status_code": response.status,
                    "visualization_url": result.get("visualization_url", ""),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time_ms": response_time,
                    "status_code": response.status,
                    "error": error_text,
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        return {
            "request_id": request_id,
            "success": False,
            "response_time_ms": response_time,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def run_minimal_load_test():
    print(f"🟢 MINIMAL LOAD TEST (6 concurrent requests)")
    print("=" * 60)
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🖼️  Test image: {WORKING_IMAGE}")
    print(f"🎯 Concurrent requests: {CONCURRENT_REQUESTS}")
    print("=" * 60)

    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, keepalive_timeout=300)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        headers={"Content-Type": "application/json"}
    ) as session:
        overall_start = time.time()
        completed = 0
        results = []
        print(f"⏰ Starting {CONCURRENT_REQUESTS} requests at {datetime.now()}")
        tasks = [send_minimal_request(session, i + 1) for i in range(CONCURRENT_REQUESTS)]
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            results.append(result)
            elapsed = time.time() - overall_start
            if completed % 2 == 0 or completed <= 3:
                success_count = sum(1 for r in results if r.get("success", False))
                avg_time = sum(r.get("response_time_ms", 0) for r in results) / len(results)
                print(f"📊 {completed}/{CONCURRENT_REQUESTS} | Success: {success_count} | Avg: {avg_time:.0f}ms | Elapsed: {elapsed:.1f}s")
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        print(f"\n✅ All requests completed in {overall_duration:.2f} seconds")
        print("=" * 60)
        return results, overall_duration

def analyze_minimal_results(results: List[Dict[str, Any]], overall_duration: float):
    print(f"📊 MINIMAL LOAD TEST RESULTS")
    print("=" * 60)
    total_requests = len(results)
    successful_requests = [r for r in results if r.get("success", False)]
    failed_requests = [r for r in results if not r.get("success", False)]
    success_rate = len(successful_requests) / total_requests
    print(f"📈 OVERALL RESULTS:")
    print(f"   Total Requests: {total_requests}")
    print(f"   Successful: {len(successful_requests)} ({success_rate*100:.1f}%)")
    print(f"   Failed: {len(failed_requests)} ({(1-success_rate)*100:.1f}%)")
    print(f"   Total Duration: {overall_duration:.2f}s")
    if successful_requests:
        response_times = [r["response_time_ms"] for r in successful_requests]
        response_times_sec = [t/1000 for t in response_times]
        avg_time = statistics.mean(response_times_sec)
        stdev_time = statistics.stdev(response_times_sec) if len(response_times_sec) > 1 else 0
        print(f"\n⚡ SUCCESSFUL REQUESTS ANALYSIS:")
        print(f"   Average: {avg_time:.1f}s (±{stdev_time:.1f}s)")
        print(f"   Range: {min(response_times_sec):.1f}s - {max(response_times_sec):.1f}s")
    print("=" * 60)

async def main():
    results, duration = await run_minimal_load_test()
    analyze_minimal_results(results, duration)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")
        import traceback
        traceback.print_exc() 