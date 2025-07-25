#!/usr/bin/env python3
"""
Aggressive Scaling Test - Force Autoscaling with High Load
Tests autoscaling by sending many concurrent requests to /infer endpoint
Target: Force scaling from 1 to 3 instances (20 concurrent/instance = 60 total capacity)
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Configuration for aggressive scaling test
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer"
HEALTH_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/health"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"

# Aggressive test configuration
TOTAL_IMAGES = 100  # High load to force scaling
CONCURRENT_REQUESTS = 10  # High concurrency to trigger autoscaling
TIMEOUT = 180  # 3 minutes timeout for heavy load
BURST_SIZE = 10  # Send 10 requests every burst
BURST_INTERVAL = 0  # Wait 0.5s between bursts to create sustained load

async def send_single_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    """Send a single inference request with detailed timing"""
    start_time = time.time()
    
    payload = {"image_url": TEST_IMAGE_URL}
    
    try:
        async with session.post(
            SERVICE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms
            
            if response.status == 200:
                result = await response.json()
                return {
                    "request_id": request_id,
                    "success": True,
                    "response_time_ms": response_time,
                    "status_code": response.status,
                    "visualization_url": result.get("visualization_url"),
                    "timestamp": datetime.now().isoformat(),
                    "queue_time": response_time  # For analysis
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
        response_time = (end_time - start_time) * 1000  # ms
        return {
            "request_id": request_id,
            "success": False,
            "response_time_ms": response_time,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def test_health_check():
    """Quick health check before load test"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(HEALTH_URL, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Health check: {result.get('service', 'unknown')} - {result.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

async def run_aggressive_scaling_test():
    """Run aggressive load test to force autoscaling"""
    print(f"🚀 AGGRESSIVE SCALING TEST - Force 1→3 Instance Autoscaling")
    print("=" * 80)
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🎯 Total images: {TOTAL_IMAGES}")
    print(f"🔄 Max concurrent: {CONCURRENT_REQUESTS}")
    print(f"💥 Burst mode: {BURST_SIZE} requests every {BURST_INTERVAL}s")
    print(f"⏱️  Timeout: {TIMEOUT}s per request")
    print(f"📈 Expected scaling: 1→2→3 instances (20 concurrent/instance)")
    print("=" * 80)

    # Health check
    if not await test_health_check():
        print("❌ Health check failed, aborting test")
        return [], 0
    
    # Configure session for high concurrency
    connector = aiohttp.TCPConnector(
        limit=150,  # High connection limit
        limit_per_host=100,
        keepalive_timeout=180
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        headers={"Content-Type": "application/json"}
    ) as session:
        
        overall_start = time.time()
        completed = 0
        all_results = []
        
        print(f"⏰ Starting aggressive load test at {datetime.now()}")
        print(f"🔥 Sending {TOTAL_IMAGES} requests with {CONCURRENT_REQUESTS} max concurrent...")
        
        # Create semaphore for concurrent control
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        async def limited_request(request_id: int):
            async with semaphore:
                return await send_single_request(session, request_id)
        
        # Send requests in bursts to create sustained load
        all_tasks = []
        
        for burst_start in range(0, TOTAL_IMAGES, BURST_SIZE):
            burst_end = min(burst_start + BURST_SIZE, TOTAL_IMAGES)
            burst_size = burst_end - burst_start
            
            print(f"💥 Burst {burst_start//BURST_SIZE + 1}: Sending {burst_size} requests...")
            
            # Create tasks for this burst
            burst_tasks = [
                limited_request(i + 1) 
                for i in range(burst_start, burst_end)
            ]
            all_tasks.extend(burst_tasks)
            
            # Small delay between bursts to create sustained pressure
            if burst_end < TOTAL_IMAGES:
                await asyncio.sleep(BURST_INTERVAL)
        
        print(f"🚀 All {TOTAL_IMAGES} requests launched! Waiting for responses...")
        
        # Track completion with detailed progress
        start_tracking = time.time()
        for task in asyncio.as_completed(all_tasks):
            result = await task
            completed += 1
            all_results.append(result)
            
            elapsed = time.time() - overall_start
            
            # Detailed progress updates
            if completed % 10 == 0 or completed <= 10:
                success_count = sum(1 for r in all_results if r.get("success", False))
                successful_times = [r.get("response_time_ms", 0) for r in all_results if r.get("success", False)]
                
                if successful_times:
                    avg_time = statistics.mean(successful_times)
                    p95_time = sorted(successful_times)[int(len(successful_times) * 0.95)] if len(successful_times) > 20 else max(successful_times)
                    current_throughput = completed / elapsed if elapsed > 0 else 0
                    
                    print(f"📊 {completed:3d}/{TOTAL_IMAGES} | "
                          f"✅ {success_count:3d} | "
                          f"Avg: {avg_time:4.0f}ms | "
                          f"P95: {p95_time:4.0f}ms | "
                          f"Rate: {current_throughput:.1f}/s | "
                          f"⏱️ {elapsed:.0f}s")
                else:
                    print(f"📊 {completed:3d}/{TOTAL_IMAGES} | ❌ No successful requests yet | ⏱️ {elapsed:.0f}s")
        
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"\n✅ All requests completed in {overall_duration:.2f} seconds")
        print("=" * 80)
        
        return all_results, overall_duration

def analyze_scaling_results(results: List[Dict[str, Any]], overall_duration: float):
    """Analyze autoscaling performance and behavior"""
    print(f"📊 AUTOSCALING PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    total_requests = len(results)
    successful_requests = [r for r in results if r.get("success", False)]
    failed_requests = [r for r in results if not r.get("success", False)]
    
    success_rate = len(successful_requests) / total_requests if total_requests > 0 else 0
    
    print(f"📈 OVERALL PERFORMANCE:")
    print(f"   • Total requests: {total_requests}")
    print(f"   • Successful: {len(successful_requests)} ({success_rate*100:.1f}%)")
    print(f"   • Failed: {len(failed_requests)} ({(1-success_rate)*100:.1f}%)")
    print(f"   • Total duration: {overall_duration:.2f}s")
    
    if successful_requests:
        # Timing analysis with focus on autoscaling patterns
        response_times_ms = [r["response_time_ms"] for r in successful_requests]
        response_times_sec = [t/1000 for t in response_times_ms]
        
        avg_response_time = statistics.mean(response_times_sec)
        median_response_time = statistics.median(response_times_sec)
        stdev_response_time = statistics.stdev(response_times_sec) if len(response_times_sec) > 1 else 0
        
        # Calculate percentiles to understand scaling behavior
        sorted_times = sorted(response_times_sec)
        p50 = sorted_times[len(sorted_times) // 2]
        p95_idx = int(len(sorted_times) * 0.95)
        p95 = sorted_times[p95_idx] if sorted_times else 0
        p99_idx = int(len(sorted_times) * 0.99)
        p99 = sorted_times[p99_idx] if sorted_times else 0
        
        print(f"\n⚡ RESPONSE TIME ANALYSIS:")
        print(f"   • Average: {avg_response_time:.2f}s")
        print(f"   • Median (P50): {median_response_time:.2f}s") 
        print(f"   • P95: {p95:.2f}s")
        print(f"   • P99: {p99:.2f}s")
        print(f"   • Std deviation: {stdev_response_time:.2f}s")
        print(f"   • Range: {min(response_times_sec):.2f}s - {max(response_times_sec):.2f}s")
        
        # Throughput analysis
        requests_per_second = total_requests / overall_duration if overall_duration > 0 else 0
        successful_per_second = len(successful_requests) / overall_duration if overall_duration > 0 else 0
        
        print(f"\n🚀 THROUGHPUT ANALYSIS:")
        print(f"   • Overall throughput: {requests_per_second:.2f} requests/second")
        print(f"   • Successful throughput: {successful_per_second:.2f} images/second")
        print(f"   • Peak capacity utilization: {CONCURRENT_REQUESTS} concurrent")
        
        # Autoscaling behavior analysis
        print(f"\n📈 AUTOSCALING BEHAVIOR:")
        print(f"   • Target instances: 1-3 (20 concurrent each = 60 max capacity)")
        print(f"   • Load applied: {CONCURRENT_REQUESTS} concurrent requests")
        print(f"   • Expected scaling trigger: >{20} concurrent per instance")
        
        if successful_per_second >= 4.0:
            print(f"   🎉 EXCELLENT: {successful_per_second:.1f} img/s suggests multiple instances!")
            print(f"   ✅ Autoscaling appears to be working effectively")
        elif successful_per_second >= 2.0:
            print(f"   ✅ GOOD: {successful_per_second:.1f} img/s suggests some scaling")
            print(f"   ⚠️  May be 2 instances, check for further optimization")
        elif successful_per_second >= 1.0:
            print(f"   ⚠️  MODERATE: {successful_per_second:.1f} img/s suggests limited scaling")
            print(f"   🔧 May still be on 1 instance, need higher load")
        else:
            print(f"   ❌ POOR: {successful_per_second:.1f} img/s suggests no scaling")
            print(f"   🔧 Autoscaling may not be triggered or service overloaded")
            
        # Response time pattern analysis for scaling detection
        if p99 > avg_response_time * 3:
            print(f"   🔍 High P99/avg ratio ({p99/avg_response_time:.1f}x) suggests scaling events")
        
        if stdev_response_time > avg_response_time * 0.5:
            print(f"   🔍 High variability suggests cold starts during scaling")
    
    # Error analysis
    if failed_requests:
        print(f"\n❌ ERROR ANALYSIS:")
        error_types = {}
        for req in failed_requests:
            error = str(req.get("error", "Unknown"))
            status = req.get("status_code", "Unknown")
            error_key = f"{status}: {error[:50]}..."
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        for error, count in error_types.items():
            print(f"   {error}: {count} requests")
    
    # Final scaling assessment
    print(f"\n🎯 AUTOSCALING ASSESSMENT:")
    if success_rate >= 0.95 and successful_per_second >= 3.0:
        print(f"   🎉 EXCELLENT: Service autoscaled successfully under load!")
        print(f"   ✅ Handles {successful_per_second:.1f} img/s with {success_rate*100:.1f}% success")
        print(f"   📈 Likely scaled to 2-3 instances")
    elif success_rate >= 0.85 and successful_per_second >= 2.0:
        print(f"   ✅ GOOD: Decent autoscaling performance")
        print(f"   📈 Likely scaled to 2 instances")
    elif success_rate >= 0.70:
        print(f"   ⚠️  MODERATE: Some scaling, but room for improvement")
        print(f"   🔧 May need tuning or higher sustained load")
    else:
        print(f"   ❌ POOR: Autoscaling not effective under this load")
        print(f"   🔧 Service may be overwhelmed or scaling not working")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aggressive_scaling_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "total_images": TOTAL_IMAGES,
                "concurrent_requests": CONCURRENT_REQUESTS,
                "burst_size": BURST_SIZE,
                "burst_interval": BURST_INTERVAL,
                "timeout": TIMEOUT,
                "test_type": "aggressive_autoscaling"
            },
            "results": {
                "total_requests": total_requests,
                "successful_requests": len(successful_requests),
                "success_rate": success_rate,
                "total_duration_seconds": overall_duration,
                "throughput_requests_per_second": requests_per_second if 'requests_per_second' in locals() else 0,
                "throughput_images_per_second": successful_per_second if 'successful_per_second' in locals() else 0
            },
            "raw_results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("=" * 80)

async def main():
    """Main execution function"""
    results, duration = await run_aggressive_scaling_test()
    analyze_scaling_results(results, duration)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Aggressive scaling test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 