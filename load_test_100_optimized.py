#!/usr/bin/env python3
"""
Optimized 100 Concurrent Request Load Test
Based on real performance data: ~2.7s per request
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Configuration based on real performance data
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer"
WORKING_IMAGE = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
CONCURRENT_REQUESTS = 100
TIMEOUT = 600  # 10 minutes timeout (realistic for GPU processing)

# Expected performance based on measurements
EXPECTED_AVG_TIME = 2659  # ms from benchmark
EXPECTED_MIN_TIME = 2100  # ms 
EXPECTED_MAX_TIME = 3200  # ms

# Request payload
PAYLOAD = {
    "image_url": WORKING_IMAGE
}

async def send_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    """Send a single inference request with detailed timing."""
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
                
    except asyncio.TimeoutError:
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        return {
            "request_id": request_id,
            "success": False,
            "response_time_ms": response_time,
            "error": "Timeout",
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

async def run_load_test() -> List[Dict[str, Any]]:
    """Run the optimized load test with 100 concurrent requests."""
    print(f"🚀 OPTIMIZED 100 CONCURRENT REQUEST LOAD TEST")
    print("=" * 70)
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🖼️  Test image: {WORKING_IMAGE}")
    print(f"⏱️  Timeout: {TIMEOUT}s per request")
    print(f"📊 Expected avg time: {EXPECTED_AVG_TIME}ms")
    print(f"🎯 Concurrent requests: {CONCURRENT_REQUESTS}")
    print("=" * 70)

    # Optimized connector settings
    connector = aiohttp.TCPConnector(
        limit=200,           # Total connection pool size
        limit_per_host=150,  # Connections per host (higher for single endpoint)
        keepalive_timeout=300  # Keep connections alive
    )
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    ) as session:
        
        # Performance tracking
        overall_start = time.time()
        completed = 0
        results = []
        
        print(f"⏰ Starting {CONCURRENT_REQUESTS} requests at {datetime.now()}")
        print("🚀 Sending ALL REQUESTS CONCURRENTLY...")
        print("⌛ Real-time progress tracking:")
        
        # Create all tasks
        tasks = [
            send_request(session, i + 1)
            for i in range(CONCURRENT_REQUESTS)
        ]
        
        # Track completion in real-time
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            results.append(result)
            
            elapsed = time.time() - overall_start
            
            # Progress updates
            if completed % 5 == 0 or completed <= 10:
                success_count = sum(1 for r in results if r.get("success", False))
                avg_time = sum(r.get("response_time_ms", 0) for r in results) / len(results)
                
                print(f"📊 {completed:3d}/{CONCURRENT_REQUESTS} | "
                      f"Success: {success_count:3d} | "
                      f"Avg: {avg_time:4.0f}ms | "
                      f"Elapsed: {elapsed:5.1f}s")

        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"\n✅ All requests completed in {overall_duration:.2f} seconds")
        print("=" * 70)
        
        return results, overall_duration

def analyze_performance(results: List[Dict[str, Any]], overall_duration: float):
    """Analyze and display comprehensive performance results."""
    print(f"📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
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
        # Response time analysis
        response_times = [r["response_time_ms"] for r in successful_requests]
        
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        median_time = sorted(response_times)[len(response_times)//2]
        stdev_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        cv_percent = (stdev_time / avg_time * 100) if avg_time > 0 else 0
        
        print(f"\n⚡ RESPONSE TIME ANALYSIS:")
        print(f"   Average: {avg_time:.1f}ms (±{stdev_time:.1f}ms)")
        print(f"   Median: {median_time:.1f}ms")
        print(f"   Range: {min_time:.1f}ms - {max_time:.1f}ms")
        print(f"   Std Dev: {stdev_time:.1f}ms ({cv_percent:.1f}% CV)")
        print(f"   95th percentile: {sorted(response_times)[int(len(response_times)*0.95)]:.1f}ms")
        
        # Performance vs expectations
        print(f"\n🎯 PERFORMANCE vs EXPECTATIONS:")
        print(f"   Expected avg: {EXPECTED_AVG_TIME}ms")
        print(f"   Actual avg: {avg_time:.1f}ms")
        
        if avg_time <= EXPECTED_AVG_TIME * 1.2:
            print(f"   ✅ Performance: EXCELLENT (within 20% of expected)")
        elif avg_time <= EXPECTED_AVG_TIME * 1.5:
            print(f"   ⚠️  Performance: GOOD (within 50% of expected)")
        else:
            print(f"   ❌ Performance: NEEDS IMPROVEMENT (>50% slower)")
        
        # Throughput analysis
        requests_per_second = total_requests / overall_duration
        print(f"\n🚀 THROUGHPUT ANALYSIS:")
        print(f"   Requests/second: {requests_per_second:.2f}")
        print(f"   Requests/minute: {requests_per_second * 60:.1f}")
        
        # Auto-scaling analysis  
        print(f"\n🔄 AUTO-SCALING ANALYSIS:")
        if avg_time <= 3000:
            estimated_instances = min(5, max(1, int(CONCURRENT_REQUESTS / 20)))
            print(f"   Estimated instances used: ~{estimated_instances}")
            print(f"   GPU utilization: HIGH ✅")
        else:
            estimated_instances = min(10, max(3, int(CONCURRENT_REQUESTS / 15)))
            print(f"   Estimated instances used: ~{estimated_instances}")
            print(f"   GPU utilization: SATURATED ⚠️")
        
    # Error analysis
    if failed_requests:
        print(f"\n❌ ERROR ANALYSIS:")
        error_types = {}
        for req in failed_requests:
            error = req.get("error", "Unknown")
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in error_types.items():
            print(f"   {error}: {count} ({count/total_requests*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_100_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "concurrent_requests": CONCURRENT_REQUESTS,
                "timeout": TIMEOUT,
                "service_url": SERVICE_URL,
                "test_image": WORKING_IMAGE
            },
            "overall_stats": {
                "total_requests": total_requests,
                "successful_requests": len(successful_requests),
                "success_rate": success_rate,
                "total_duration_seconds": overall_duration
            },
            "raw_results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("=" * 70)

async def main():
    """Main execution function."""
    results, duration = await run_load_test()
    analyze_performance(results, duration)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")
        import traceback
        traceback.print_exc() 