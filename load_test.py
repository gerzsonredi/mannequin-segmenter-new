#!/usr/bin/env python3
"""
Load Testing Script for Mannequin Segmentation Service
Sends 100 concurrent requests to test auto-scaling and performance.
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
CONCURRENT_REQUESTS = 100
TIMEOUT = 300  # 5 minutes timeout per request

# Request payload
PAYLOAD = {
    "image_url": TEST_IMAGE_URL,
    "prompt_mode": "both"
}

async def send_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    """Send a single inference request and measure response time."""
    start_time = time.time()
    
    try:
        async with session.post(
            SERVICE_URL,
            json=PAYLOAD,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            response_time = time.time() - start_time
            response_data = await response.text()
            
            result = {
                "request_id": request_id,
                "status_code": response.status,
                "response_time": response_time,
                "success": response.status == 200,
                "response_size": len(response_data),
                "timestamp": datetime.now().isoformat()
            }
            
            if response.status == 200:
                try:
                    json_data = json.loads(response_data)
                    result["visualization_url"] = json_data.get("visualization_url", "N/A")
                except json.JSONDecodeError:
                    result["error"] = "Invalid JSON response"
            else:
                result["error"] = response_data[:200]  # First 200 chars of error
                
            return result
            
    except asyncio.TimeoutError:
        return {
            "request_id": request_id,
            "status_code": 0,
            "response_time": time.time() - start_time,
            "success": False,
            "error": "Request timeout",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "status_code": 0,
            "response_time": time.time() - start_time,
            "success": False,
            "error": str(e)[:200],
            "timestamp": datetime.now().isoformat()
        }

async def run_load_test() -> List[Dict[str, Any]]:
    """Run the load test with concurrent requests."""
    print(f"🚀 Starting load test: {CONCURRENT_REQUESTS} concurrent requests")
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🖼️  Test image: {TEST_IMAGE_URL}")
    print(f"⏱️  Timeout: {TIMEOUT}s per request")
    print("=" * 80)
    
    # Create aiohttp session
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=100)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    ) as session:
        
        # Record overall start time
        overall_start = time.time()
        
        # Create all tasks
        tasks = [
            send_request(session, i + 1) 
            for i in range(CONCURRENT_REQUESTS)
        ]
        
        print(f"⏰ Starting {CONCURRENT_REQUESTS} requests at {datetime.now()}")
        print("🚀 Sending ALL REQUESTS CONCURRENTLY...")
        print("⌛ Waiting for responses... (this will show progress)")
        
        # Track completed requests for progress
        completed = 0
        results = []
        
        # Start all tasks concurrently using asyncio.as_completed for progress tracking
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            results.append(result)
            
            # Show progress every 10 completed requests
            if completed % 10 == 0 or completed <= 5:
                elapsed = time.time() - overall_start
                print(f"📊 Progress: {completed}/{CONCURRENT_REQUESTS} completed ({completed/CONCURRENT_REQUESTS*100:.1f}%) - {elapsed:.1f}s elapsed")
        
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"✅ All requests completed in {overall_duration:.2f} seconds")
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "request_id": i + 1,
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results, overall_duration

def analyze_results(results: List[Dict[str, Any]], overall_duration: float):
    """Analyze and display test results."""
    print("\n" + "=" * 80)
    print("📊 LOAD TEST RESULTS")
    print("=" * 80)
    
    # Basic stats
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    failed_requests = total_requests - successful_requests
    
    # Response times
    response_times = [r["response_time"] for r in results if r["success"]]
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95_idx = int(len(sorted_times) * 0.95)
        p95 = sorted_times[p95_idx] if sorted_times else 0
        p99_idx = int(len(sorted_times) * 0.99)
        p99 = sorted_times[p99_idx] if sorted_times else 0
    else:
        avg_response_time = min_response_time = max_response_time = p50 = p95 = p99 = 0
    
    # Success rate
    success_rate = (successful_requests / total_requests) * 100
    
    # Requests per second
    rps = total_requests / overall_duration if overall_duration > 0 else 0
    
    print(f"📈 Overall Performance:")
    print(f"   • Total requests: {total_requests}")
    print(f"   • Successful: {successful_requests}")
    print(f"   • Failed: {failed_requests}")
    print(f"   • Success rate: {success_rate:.1f}%")
    print(f"   • Total duration: {overall_duration:.2f}s")
    print(f"   • Requests/second: {rps:.2f}")
    
    print(f"\n⏱️  Response Times:")
    print(f"   • Average: {avg_response_time:.2f}s")
    print(f"   • Minimum: {min_response_time:.2f}s")
    print(f"   • Maximum: {max_response_time:.2f}s")
    print(f"   • P50 (median): {p50:.2f}s")
    print(f"   • P95: {p95:.2f}s")
    print(f"   • P99: {p99:.2f}s")
    
    # Error analysis
    if failed_requests > 0:
        print(f"\n❌ Error Analysis:")
        error_counts = {}
        for result in results:
            if not result["success"]:
                error = result.get("error", "Unknown error")
                error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            print(f"   • {error}: {count} occurrences")
    
    # Timeline analysis (first 10 and last 10 requests)
    print(f"\n⏰ Timeline Analysis:")
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        first_response = min(r["response_time"] for r in successful_results)
        print(f"   • First successful response: {first_response:.2f}s")
        print(f"   • Auto-scaling performance: {'Good' if first_response < 60 else 'Slow'}")

async def main():
    """Main function to run the load test."""
    try:
        results, overall_duration = await run_load_test()
        analyze_results(results, overall_duration)
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "total_requests": CONCURRENT_REQUESTS,
                    "target_url": SERVICE_URL,
                    "test_image": TEST_IMAGE_URL,
                    "overall_duration": overall_duration,
                    "timestamp": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp...")
        import subprocess
        subprocess.run(["pip", "install", "aiohttp"])
        import aiohttp
    
    # Run the load test
    asyncio.run(main()) 