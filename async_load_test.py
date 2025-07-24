#!/usr/bin/env python3
"""
Asynchronous Load Testing Script for Mannequin Segmentation Service
Simulates real-world usage with 2-second intervals and variable batch processing.
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Configuration - REALISTIC USAGE PATTERN
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer_async"
STATUS_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer_status"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
TOTAL_IMAGES = 100
INTERVAL_SECONDS = 2.0  # Send requests every 2 seconds
TIMEOUT = 300
TARGET_TIME = 6.0  # Target completion time in seconds

async def send_async_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    """Send a single async inference request."""
    start_time = time.time()
    
    payload = {
        "image_url": TEST_IMAGE_URL
    }
    
    try:
        async with session.post(
            SERVICE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            response_data = await response.text()
            
            if response.status == 200:
                try:
                    json_data = json.loads(response_data)
                    request_id_from_api = json_data.get("request_id")
                    
                    return {
                        "request_id": request_id,
                        "api_request_id": request_id_from_api,
                        "status_code": response.status,
                        "queue_time": time.time() - start_time,
                        "status": "queued",
                        "timestamp": datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    return {
                        "request_id": request_id,
                        "status_code": response.status,
                        "queue_time": time.time() - start_time,
                        "error": "Invalid JSON response",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "request_id": request_id,
                    "status_code": response.status,
                    "queue_time": time.time() - start_time,
                    "error": response_data[:200],
                    "timestamp": datetime.now().isoformat()
                }
                
    except Exception as e:
        return {
            "request_id": request_id,
            "status_code": 0,
            "queue_time": time.time() - start_time,
            "error": str(e)[:200],
            "timestamp": datetime.now().isoformat()
        }

async def check_request_status(session: aiohttp.ClientSession, api_request_id: str, max_retries: int = 30) -> Dict[str, Any]:
    """Check status of async request until completion."""
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            async with session.get(
                f"{STATUS_URL}/{api_request_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_data = await response.text()
                
                if response.status == 200:
                    try:
                        json_data = json.loads(response_data)
                        
                        # Check if processing is complete
                        if "visualization_url" in json_data or "error" in json_data:
                            return {
                                "api_request_id": api_request_id,
                                "status": "completed",
                                "response_time": time.time() - start_time,
                                "result": json_data,
                                "attempts": attempt + 1
                            }
                        else:
                            # Still processing
                            await asyncio.sleep(1)  # Wait 1 second before next check
                            continue
                            
                    except json.JSONDecodeError:
                        return {
                            "api_request_id": api_request_id,
                            "status": "error",
                            "response_time": time.time() - start_time,
                            "error": "Invalid JSON response",
                            "attempts": attempt + 1
                        }
                else:
                    return {
                        "api_request_id": api_request_id,
                        "status": "error",
                        "response_time": time.time() - start_time,
                        "error": f"HTTP {response.status}: {response_data[:200]}",
                        "attempts": attempt + 1
                    }
                    
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "api_request_id": api_request_id,
                    "status": "error",
                    "response_time": time.time() - start_time,
                    "error": str(e),
                    "attempts": attempt + 1
                }
            await asyncio.sleep(1)
    
    return {
        "api_request_id": api_request_id,
        "status": "timeout",
        "response_time": time.time() - start_time,
        "error": "Max retries exceeded",
        "attempts": max_retries
    }

async def run_async_load_test() -> List[Dict[str, Any]]:
    """Run the async load test with realistic timing."""
    print(f"🚀 Starting ASYNC load test: {TOTAL_IMAGES} images every {INTERVAL_SECONDS}s")
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🖼️  Test image: {TEST_IMAGE_URL}")
    print(f"⏱️  Interval: {INTERVAL_SECONDS}s between requests")
    print(f"🎯 Target completion time: {TARGET_TIME}s")
    print("=" * 80)
    
    # Create aiohttp session
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    ) as session:
        
        overall_start = time.time()
        all_results = []
        
        print(f"⏰ Starting requests at {datetime.now()}")
        print("🔄 Sending requests with 2-second intervals...")
        
        # Send requests with intervals
        for i in range(TOTAL_IMAGES):
            request_start = time.time()
            
            # Send async request
            queue_result = await send_async_request(session, i + 1)
            
            if queue_result.get("status") == "queued" and "api_request_id" in queue_result:
                # Check status until completion
                status_result = await check_request_status(session, queue_result["api_request_id"])
                
                # Combine results
                combined_result = {
                    "request_id": queue_result["request_id"],
                    "api_request_id": queue_result["api_request_id"],
                    "queue_time": queue_result["queue_time"],
                    "processing_time": status_result["response_time"],
                    "total_time": queue_result["queue_time"] + status_result["response_time"],
                    "status": status_result["status"],
                    "success": status_result["status"] == "completed" and "visualization_url" in status_result.get("result", {}),
                    "attempts": status_result.get("attempts", 0),
                    "timestamp": queue_result["timestamp"]
                }
                
                if "result" in status_result:
                    if "visualization_url" in status_result["result"]:
                        combined_result["visualization_url"] = status_result["result"]["visualization_url"]
                    if "error" in status_result["result"]:
                        combined_result["error"] = status_result["result"]["error"]
                
                all_results.append(combined_result)
                
                # Show progress
                if (i + 1) % 10 == 0 or i < 5:
                    elapsed = time.time() - overall_start
                    completed = len([r for r in all_results if r["success"]])
                    print(f"📊 Progress: {i + 1}/{TOTAL_IMAGES} requests sent, {completed} completed - {elapsed:.1f}s elapsed")
            else:
                # Queue failed
                all_results.append({
                    "request_id": queue_result["request_id"],
                    "queue_time": queue_result["queue_time"],
                    "processing_time": 0,
                    "total_time": queue_result["queue_time"],
                    "status": "queue_failed",
                    "success": False,
                    "error": queue_result.get("error", "Unknown error"),
                    "timestamp": queue_result["timestamp"]
                })
            
            # Wait for next interval (except for last request)
            if i < TOTAL_IMAGES - 1:
                elapsed = time.time() - request_start
                wait_time = max(0, INTERVAL_SECONDS - elapsed)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
        
        # Wait for remaining requests to complete
        print("⌛ Waiting for remaining requests to complete...")
        max_wait_time = 60  # Wait up to 60 seconds for completion
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait_time:
            pending = len([r for r in all_results if r["status"] == "processing"])
            if pending == 0:
                break
            
            print(f"⏳ {pending} requests still processing...")
            await asyncio.sleep(5)
        
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"✅ All requests completed in {overall_duration:.2f} seconds")
        
        # Check if target was met
        if overall_duration <= TARGET_TIME:
            print(f"🎉 TARGET ACHIEVED! Completed in {overall_duration:.2f}s (target: {TARGET_TIME}s)")
        else:
            print(f"❌ TARGET MISSED! Completed in {overall_duration:.2f}s (target: {TARGET_TIME}s)")
            print(f"   Performance gap: {overall_duration - TARGET_TIME:.2f}s")
        
        return all_results, overall_duration

def analyze_async_results(results: List[Dict[str, Any]], overall_duration: float):
    """Analyze and display async test results."""
    print("\n" + "=" * 80)
    print("📊 ASYNC LOAD TEST RESULTS")
    print("=" * 80)
    
    # Basic stats
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    failed_requests = total_requests - successful_requests
    
    # Timing analysis
    queue_times = [r["queue_time"] for r in results if r["success"]]
    processing_times = [r["processing_time"] for r in results if r["success"]]
    total_times = [r["total_time"] for r in results if r["success"]]
    
    if total_times:
        avg_total_time = sum(total_times) / len(total_times)
        avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate percentiles
        sorted_times = sorted(total_times)
        p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95_idx = int(len(sorted_times) * 0.95)
        p95 = sorted_times[p95_idx] if sorted_times else 0
        p99_idx = int(len(sorted_times) * 0.99)
        p99 = sorted_times[p99_idx] if sorted_times else 0
    else:
        avg_total_time = avg_queue_time = avg_processing_time = p50 = p95 = p99 = 0
    
    # Success rate
    success_rate = (successful_requests / total_requests) * 100
    
    # Throughput
    requests_per_second = total_requests / overall_duration if overall_duration > 0 else 0
    
    print(f"📈 Overall Performance:")
    print(f"   • Total requests: {total_requests}")
    print(f"   • Successful: {successful_requests}")
    print(f"   • Failed: {failed_requests}")
    print(f"   • Success rate: {success_rate:.1f}%")
    print(f"   • Total duration: {overall_duration:.2f}s")
    print(f"   • Requests/second: {requests_per_second:.2f}")
    
    print(f"\n⏱️  Timing Analysis:")
    print(f"   • Average total time: {avg_total_time:.2f}s")
    print(f"   • Average queue time: {avg_queue_time:.2f}s")
    print(f"   • Average processing time: {avg_processing_time:.2f}s")
    print(f"   • P50 (median): {p50:.2f}s")
    print(f"   • P95: {p95:.2f}s")
    print(f"   • P99: {p99:.2f}s")
    
    # Batch analysis
    status_counts = {}
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n🔄 Status Analysis:")
    for status, count in status_counts.items():
        print(f"   • {status}: {count} requests")
    
    # Performance comparison
    print(f"\n🎯 Performance Analysis:")
    if overall_duration <= TARGET_TIME:
        print(f"   ✅ Target achieved: {overall_duration:.2f}s ≤ {TARGET_TIME}s")
        print(f"   🚀 Performance: {requests_per_second:.1f} requests/second")
    else:
        print(f"   ❌ Target missed: {overall_duration:.2f}s > {TARGET_TIME}s")
        print(f"   📈 Needed: {TOTAL_IMAGES/TARGET_TIME:.1f} requests/second")
        print(f"   📊 Current: {requests_per_second:.1f} requests/second")

async def main():
    """Main function to run the async load test."""
    try:
        results, overall_duration = await run_async_load_test()
        analyze_async_results(results, overall_duration)
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"async_load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "total_requests": TOTAL_IMAGES,
                    "interval_seconds": INTERVAL_SECONDS,
                    "target_url": SERVICE_URL,
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
    
    # Run the async load test
    asyncio.run(main()) 