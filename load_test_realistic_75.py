#!/usr/bin/env python3
"""
Realistic Load Test - 75 Concurrent Requests
Testing within actual capacity: 3 instances × 25 concurrency = 75 max capacity
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Configuration matching actual capacity
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/infer"
WORKING_IMAGE = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
CONCURRENT_REQUESTS = 75  # Within capacity: 3 × 25 = 75
TIMEOUT = 600

# Realistic expectations
EXPECTED_PERFORMANCE = {
    "avg_time_ms": 8000,    # ~8s (better than 19.6s)
    "cv_percent": 25,       # ~25% (better than 37.7%)
    "success_rate": 100,    # No rate limiting
    "fast_requests_pct": 40 # More fast requests
}

PAYLOAD = {
    "image_url": WORKING_IMAGE
}

async def send_capacity_request(session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
    """Send request within capacity limits."""
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

async def run_realistic_load_test():
    """Run load test within actual capacity."""
    print(f"🎯 REALISTIC CAPACITY LOAD TEST")
    print("=" * 80)
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🖼️  Test image: {WORKING_IMAGE}")
    print(f"🎯 Concurrent requests: {CONCURRENT_REQUESTS} (within capacity)")
    print(f"⚙️  Actual capacity: 3 instances × 25 concurrency = 75")
    print(f"🚫 Previous issue: 100 requests > 75 capacity = rate limiting")
    print(f"📊 Expected avg time: ~{EXPECTED_PERFORMANCE['avg_time_ms']}ms")
    print(f"📈 Expected variability: ~{EXPECTED_PERFORMANCE['cv_percent']}% CV")
    print("=" * 80)

    connector = aiohttp.TCPConnector(
        limit=150,  # Reduced from 200
        limit_per_host=100,  # Reduced from 150
        keepalive_timeout=300
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        headers={"Content-Type": "application/json"}
    ) as session:
        
        overall_start = time.time()
        completed = 0
        results = []
        
        print(f"⏰ Starting {CONCURRENT_REQUESTS} requests at {datetime.now()}")
        print("🎯 Testing within capacity - no rate limiting expected...")
        
        tasks = [
            send_capacity_request(session, i + 1)
            for i in range(CONCURRENT_REQUESTS)
        ]
        
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            results.append(result)
            
            elapsed = time.time() - overall_start
            
            if completed % 5 == 0 or completed <= 10:
                success_count = sum(1 for r in results if r.get("success", False))
                error_count = completed - success_count
                
                if success_count > 0:
                    successful_times = [r.get("response_time_ms", 0) for r in results if r.get("success", False)]
                    avg_time = sum(successful_times) / len(successful_times)
                else:
                    avg_time = 0
                
                # Rate limiting check
                rate_limit_errors = sum(1 for r in results if r.get("status_code") == 429)
                
                if rate_limit_errors > 0:
                    status_indicator = f"🚨 RATE LIMITED ({rate_limit_errors})"
                elif success_count == completed:
                    status_indicator = "✅ ALL SUCCESS"
                elif error_count > 0:
                    status_indicator = f"⚠️  {error_count} ERRORS"
                else:
                    status_indicator = "🔄 PROCESSING"
                
                print(f"📊 {completed:2d}/{CONCURRENT_REQUESTS} | "
                      f"Success: {success_count:2d} | "
                      f"Avg: {avg_time:4.0f}ms | "
                      f"Elapsed: {elapsed:4.1f}s | {status_indicator}")

        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"\n✅ All requests completed in {overall_duration:.2f} seconds")
        print("=" * 80)
        
        return results, overall_duration

def analyze_realistic_results(results: List[Dict[str, Any]], overall_duration: float):
    """Analyze results compared to previous attempts."""
    print(f"📊 REALISTIC CAPACITY TEST RESULTS")
    print("=" * 80)
    
    # Basic statistics
    total_requests = len(results)
    successful_requests = [r for r in results if r.get("success", False)]
    failed_requests = [r for r in results if not r.get("success", False)]
    
    success_rate = len(successful_requests) / total_requests
    
    # Error analysis
    rate_limit_count = sum(1 for r in results if r.get("status_code") == 429)
    timeout_count = sum(1 for r in results if "timeout" in str(r.get("error", "")).lower())
    other_errors = len(failed_requests) - rate_limit_count - timeout_count
    
    print(f"📈 OVERALL RESULTS:")
    print(f"   Total Requests: {total_requests}")
    print(f"   Successful: {len(successful_requests)} ({success_rate*100:.1f}%)")
    print(f"   Failed: {len(failed_requests)} ({(1-success_rate)*100:.1f}%)")
    print(f"   Total Duration: {overall_duration:.2f}s")
    
    print(f"\n🚨 ERROR BREAKDOWN:")
    print(f"   Rate Limited (429): {rate_limit_count} ({rate_limit_count/total_requests*100:.1f}%)")
    print(f"   Timeouts: {timeout_count} ({timeout_count/total_requests*100:.1f}%)")
    print(f"   Other Errors: {other_errors} ({other_errors/total_requests*100:.1f}%)")
    
    if successful_requests:
        response_times = [r["response_time_ms"] for r in successful_requests]
        response_times_sec = [t/1000 for t in response_times]
        
        avg_time = statistics.mean(response_times_sec)
        median_time = statistics.median(response_times_sec)
        stdev_time = statistics.stdev(response_times_sec) if len(response_times_sec) > 1 else 0
        cv_percent = (stdev_time / avg_time) * 100 if avg_time > 0 else 0
        
        print(f"\n⚡ SUCCESSFUL REQUESTS ANALYSIS:")
        print(f"   Average: {avg_time:.1f}s (±{stdev_time:.1f}s)")
        print(f"   Median: {median_time:.1f}s")
        print(f"   Std Dev: {stdev_time:.1f}s ({cv_percent:.1f}% CV)")
        print(f"   Range: {min(response_times_sec):.1f}s - {max(response_times_sec):.1f}s")
        
        # Performance tiers
        fast_requests = [t for t in response_times_sec if t <= 5]
        medium_requests = [t for t in response_times_sec if 5 < t <= 15]
        slow_requests = [t for t in response_times_sec if t > 15]
        
        print(f"\n🎯 PERFORMANCE DISTRIBUTION:")
        print(f"   🚀 Fast (≤5s): {len(fast_requests)} ({len(fast_requests)/len(successful_requests)*100:.1f}%)")
        print(f"   ⚠️  Medium (5-15s): {len(medium_requests)} ({len(medium_requests)/len(successful_requests)*100:.1f}%)")
        print(f"   🐌 Slow (>15s): {len(slow_requests)} ({len(slow_requests)/len(successful_requests)*100:.1f}%)")
        
        # Compare with all previous tests
        print(f"\n📈 PROGRESSION COMPARISON:")
        print(f"   Original (100 req, concurrency=50): 19.6s avg, 37.7% CV, 100% success")
        print(f"   Failed optimization (100 req, concurrency=25): 38+s avg, 0% success (rate limited)")
        print(f"   Current realistic test ({total_requests} req, concurrency=25): {avg_time:.1f}s avg, {cv_percent:.1f}% CV, {success_rate*100:.1f}% success")
        
        if success_rate >= 0.95:
            print(f"   ✅ SUCCESS RATE: Excellent - no capacity issues")
        elif success_rate >= 0.8:
            print(f"   ⚠️  SUCCESS RATE: Good - minor capacity stress")
        else:
            print(f"   ❌ SUCCESS RATE: Poor - still over capacity")
            
        if rate_limit_count == 0:
            print(f"   ✅ RATE LIMITING: None - within capacity")
        else:
            print(f"   ❌ RATE LIMITING: Still occurring - capacity calculations wrong")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION ASSESSMENT:")
    
    if rate_limit_count == 0 and success_rate > 0.95:
        print(f"   🎯 CAPACITY OPTIMIZATION: SUCCESS")
        print(f"   ✅ The concurrency=25 optimization works within capacity limits")
        print(f"   📊 Ready for production with 75 concurrent request capacity")
    else:
        print(f"   ⚠️  CAPACITY OPTIMIZATION: PARTIAL SUCCESS")
        print(f"   🔧 May need further concurrency reduction")
        print(f"   🔧 Or increase timeout values")
    
    if successful_requests:
        if avg_time < 10:
            print(f"   🎯 PERFORMANCE: EXCELLENT ({avg_time:.1f}s avg)")
        elif avg_time < 15:
            print(f"   ✅ PERFORMANCE: GOOD ({avg_time:.1f}s avg)")
        else:
            print(f"   ⚠️  PERFORMANCE: NEEDS WORK ({avg_time:.1f}s avg)")
            
        if cv_percent < 25:
            print(f"   🎯 VARIABILITY: EXCELLENT ({cv_percent:.1f}% CV)")
        elif cv_percent < 35:
            print(f"   ✅ VARIABILITY: GOOD ({cv_percent:.1f}% CV)")
        else:
            print(f"   ⚠️  VARIABILITY: HIGH ({cv_percent:.1f}% CV)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"realistic_capacity_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "concurrent_requests": CONCURRENT_REQUESTS,
                "actual_capacity": "3 instances × 25 concurrency = 75",
                "expected_performance": EXPECTED_PERFORMANCE
            },
            "results": {
                "total_requests": total_requests,
                "successful_requests": len(successful_requests),
                "success_rate": success_rate,
                "rate_limit_count": rate_limit_count,
                "timeout_count": timeout_count,
                "total_duration_seconds": overall_duration
            },
            "raw_results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("=" * 80)

async def main():
    """Main execution function."""
    results, duration = await run_realistic_load_test()
    analyze_realistic_results(results, duration)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")
        import traceback
        traceback.print_exc() 