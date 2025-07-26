#!/usr/bin/env python3
"""
Batch Load Test - Testing Batch Processing Endpoint
Tests the new /batch_infer endpoint with optimal batch sizes for GPU utilization.
Each batch contains 10 images for optimal GPU memory usage.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
import torch

def log_cuda_memory(tag=""):
    """Log CUDA memory usage for debugging"""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{tag}] allocated={alloc:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB")

# Configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/batch_infer"
HEALTH_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/health"

# Test images (5 images for batch processing - optimized for GPU utilization)
TEST_IMAGES = [
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
]

# Test configuration - Optimized for your GPU specs (32Gi RAM, 8 CPU, 1x NVIDIA L4)
BATCH_SIZE = 5 # Images per batch (very conservative for testing)
CONCURRENT_BATCHES = 2  # Single batch for initial testing
TIMEOUT = 1200  # 20 minutes for batch processing (cold start can be slow)

# Health check timeout - increased for cold start
HEALTH_CHECK_TIMEOUT = 600  # 10 minutes for first health check

# Performance expectations
EXPECTED_IMPROVEMENTS = {
    "parallel_downloads": "5x faster image downloads (5 concurrent)",
    "batch_inference": "GPU optimization for 5 images at once (GPU memory optimized)", 
    "parallel_uploads": "5x faster S3 uploads (5 concurrent)",
    "total_speedup": "Expected 2-3x improvement vs sequential processing"
}

async def test_health_check():
    """Test if the service is healthy before running batch tests."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(HEALTH_URL, timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Health check passed: {result.get('service', 'unknown')} - {result.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

async def check_server_status():
    """Check current server load and request limiter status."""
    status_url = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app/status"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    limiter = result.get('request_limiter', {})
                    recommendation = result.get('recommendation', {})
                    
                    print(f"📊 Server Status:")
                    print(f"   Active requests: {limiter.get('active_requests', 0)}/{limiter.get('max_concurrent', 4)}")
                    print(f"   Load: {limiter.get('load_percentage', 0):.1f}%")
                    print(f"   Available slots: {limiter.get('slots_available', 0)}")
                    print(f"   Load level: {recommendation.get('load_level', 'unknown')}")
                    return True
                else:
                    print(f"⚠️ Status check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"⚠️ Status check error: {e}")
        return False

async def send_batch_request(session: aiohttp.ClientSession, batch_id: int, image_urls: List[str]) -> Dict[str, Any]:
    """Send a single batch inference request."""
    start_time = time.time()
    
    payload = {
        "image_urls": image_urls
    }
    
    try:
        async with session.post(
            SERVICE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if response.status == 200:
                result = await response.json()
                return {
                    "batch_id": batch_id,
                    "success": True,
                    "response_time_ms": response_time,
                    "status_code": response.status,
                    "batch_size": result.get("batch_size", 0),
                    "successful_count": result.get("successful_count", 0),
                    "failed_count": result.get("failed_count", 0),
                    "visualization_urls": result.get("visualization_urls", []),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                return {
                    "batch_id": batch_id,
                    "success": False,
                    "response_time_ms": response_time,
                    "status_code": response.status,
                    "error": error_text,
                    "batch_size": len(image_urls),
                    "timestamp": datetime.now().isoformat()
                }
                
    except Exception as e:
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        return {
            "batch_id": batch_id,
            "success": False,
            "response_time_ms": response_time,
            "error": str(e),
            "batch_size": len(image_urls),
            "timestamp": datetime.now().isoformat()
        }

async def run_batch_load_test():
    """Run the batch load test with concurrent batch requests."""
    print(f"🚀 BATCH PROCESSING LOAD TEST")
    print("=" * 80)
    print(f"📡 Target: {SERVICE_URL}")
    print(f"🎯 Concurrent batches: {CONCURRENT_BATCHES}")
    print(f"📦 Batch size: {BATCH_SIZE} images per batch")
    print(f"🖼️  Total images: {CONCURRENT_BATCHES * BATCH_SIZE}")
    print(f"⏱️  Timeout: {TIMEOUT}s per batch")
    print("=" * 80)

    # Health check first with extended timeout
    print("🔍 Performing health check (may take up to 60s for first request)...")
    if not await test_health_check():
        print("❌ Health check failed, aborting test")
        return [], 0
    
    # Check server status before starting
    print("📊 Checking server load before starting test...")
    await check_server_status()

    connector = aiohttp.TCPConnector(
        limit=50,  # Reasonable limit for batch requests
        limit_per_host=30,
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
        
        print(f"⏰ Starting {CONCURRENT_BATCHES} batch requests at {datetime.now()}")
        print(f"🔄 Each batch contains {BATCH_SIZE} images - testing app-level request limiting...")
        
        # Create batch tasks
        tasks = [
            send_batch_request(session, i + 1, TEST_IMAGES[:BATCH_SIZE])
            for i in range(CONCURRENT_BATCHES)
        ]
        
        # Track completion
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            results.append(result)
            
            elapsed = time.time() - overall_start
            
            if completed % 2 == 0 or completed <= 3:
                success_count = sum(1 for r in results if r.get("success", False))
                successful_times = [r.get("response_time_ms", 0) for r in results if r.get("success", False)]
                avg_time = statistics.mean(successful_times) if successful_times else 0
                
                # Calculate throughput
                total_images_processed = sum(r.get("successful_count", 0) for r in results if r.get("success", False))
                images_per_second = total_images_processed / elapsed if elapsed > 0 else 0
                
                print(f"📊 {completed:2d}/{CONCURRENT_BATCHES} batches | "
                      f"Success: {success_count:2d} | "
                      f"Avg: {avg_time:4.0f}ms | "
                      f"Images: {total_images_processed} | "
                      f"Throughput: {images_per_second:.1f} img/s")

        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        print(f"\n✅ All batch requests completed in {overall_duration:.2f} seconds")
        print("=" * 80)
        
        return results, overall_duration

def analyze_batch_results(results: List[Dict[str, Any]], overall_duration: float):
    """Analyze batch processing performance."""
    print(f"📊 BATCH PROCESSING RESULTS ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    total_batches = len(results)
    successful_batches = [r for r in results if r.get("success", False)]
    failed_batches = [r for r in results if not r.get("success", False)]
    
    success_rate = len(successful_batches) / total_batches if total_batches > 0 else 0
    
    print(f"📈 BATCH RESULTS:")
    print(f"   Total Batches: {total_batches}")
    print(f"   Successful Batches: {len(successful_batches)} ({success_rate*100:.1f}%)")
    print(f"   Failed Batches: {len(failed_batches)} ({(1-success_rate)*100:.1f}%)")
    print(f"   Total Duration: {overall_duration:.2f}s")
    
    if successful_batches:
        # Batch timing analysis
        batch_times = [r["response_time_ms"] for r in successful_batches]
        batch_times_sec = [t/1000 for t in batch_times]
        
        avg_batch_time = statistics.mean(batch_times_sec)
        stdev_batch_time = statistics.stdev(batch_times_sec) if len(batch_times_sec) > 1 else 0
        cv_percent = (stdev_batch_time / avg_batch_time) * 100 if avg_batch_time > 0 else 0
        
        print(f"\n⚡ BATCH TIMING ANALYSIS:")
        print(f"   Average batch time: {avg_batch_time:.1f}s (±{stdev_batch_time:.1f}s)")
        print(f"   Batch time range: {min(batch_times_sec):.1f}s - {max(batch_times_sec):.1f}s")
        print(f"   Batch variability: {cv_percent:.1f}% CV")
        
        # Image processing analysis
        total_images_requested = sum(r.get("batch_size", 0) for r in successful_batches)
        total_images_processed = sum(r.get("successful_count", 0) for r in successful_batches)
        total_images_failed = sum(r.get("failed_count", 0) for r in successful_batches)
        
        image_success_rate = total_images_processed / total_images_requested if total_images_requested > 0 else 0
        
        print(f"\n🖼️  IMAGE PROCESSING ANALYSIS:")
        print(f"   Total images requested: {total_images_requested}")
        print(f"   Total images processed: {total_images_processed}")
        print(f"   Total images failed: {total_images_failed}")
        print(f"   Image success rate: {image_success_rate*100:.1f}%")
        
        # Throughput analysis
        images_per_second = total_images_processed / overall_duration if overall_duration > 0 else 0
        avg_images_per_batch_per_second = (total_images_processed / len(successful_batches)) / avg_batch_time if avg_batch_time > 0 else 0
        
        print(f"\n🚀 THROUGHPUT ANALYSIS:")
        print(f"   Overall throughput: {images_per_second:.1f} images/second")
        print(f"   Avg batch throughput: {avg_images_per_batch_per_second:.1f} images/second/batch")
        print(f"   Time per image: {avg_batch_time/BATCH_SIZE:.1f}s per image")
        
        # Compare with single image processing
        estimated_single_time = 2.7  # From previous tests
        batch_efficiency = (estimated_single_time * BATCH_SIZE) / avg_batch_time
        
        print(f"\n📊 BATCH EFFICIENCY:")
        print(f"   Single image baseline: ~{estimated_single_time:.1f}s per image")
        print(f"   Batch processing: ~{avg_batch_time/BATCH_SIZE:.1f}s per image")
        print(f"   Efficiency gain: {batch_efficiency:.1f}x faster than individual processing")
        
        if batch_efficiency > 2:
            print(f"   🎯 EXCELLENT: Batch processing provides significant speedup!")
        elif batch_efficiency > 1.5:
            print(f"   ✅ GOOD: Batch processing provides meaningful speedup")
        elif batch_efficiency > 1:
            print(f"   ⚠️  MODERATE: Some batch benefit, but room for improvement")
        else:
            print(f"   ❌ POOR: Batch processing slower than individual - needs optimization")
    
    # Error analysis
    if failed_batches:
        print(f"\n❌ ERROR ANALYSIS:")
        error_types = {}
        for batch in failed_batches:
            error = str(batch.get("error", "Unknown"))
            status = batch.get("status_code", "Unknown")
            error_key = f"{status}: {error[:50]}..."
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        for error, count in error_types.items():
            print(f"   {error}: {count} batches")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if success_rate >= 0.95:
        print(f"   ✅ Batch reliability: Excellent")
    else:
        print(f"   🔧 Improve batch error handling and retry logic")
        
    if successful_batches:
        if cv_percent < 20:
            print(f"   ✅ Batch consistency: Excellent ({cv_percent:.1f}% variability)")
        else:
            print(f"   🔧 Reduce batch processing variability")
            
        if images_per_second > 2:
            print(f"   ✅ Throughput: Excellent ({images_per_second:.1f} img/s)")
        elif images_per_second > 1:
            print(f"   ⚠️  Throughput: Good but could be improved")
        else:
            print(f"   🔧 Optimize batch processing pipeline for better throughput")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_load_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "concurrent_batches": CONCURRENT_BATCHES,
                "batch_size": BATCH_SIZE,
                "total_images": CONCURRENT_BATCHES * BATCH_SIZE,
                "timeout": TIMEOUT
            },
            "results": {
                "total_batches": total_batches,
                "successful_batches": len(successful_batches),
                "batch_success_rate": success_rate,
                "total_duration_seconds": overall_duration
            },
            "raw_results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("=" * 80)

async def main():
    """Main execution function."""
    results, duration = await run_batch_load_test()
    analyze_batch_results(results, duration)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Batch load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Batch load test failed: {e}")
        import traceback
        traceback.print_exc() 