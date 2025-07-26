#!/usr/bin/env python3
"""
Batch Processing Optimizer Test
Configurable batch processing test to find optimal parameters for the Model Pool
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import statistics
import argparse
import sys

# Base Configuration
BASE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"

# Test image URLs with different sizes and complexities
TEST_IMAGE_SETS = {
    "small": [
        "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1558769132-cb1aea458c5e?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1581803118522-7b72a50f7e9f?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    ],
    "medium": [
        "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", 
        "https://images.unsplash.com/photo-1574015974293-817f0ebebb74?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1520006403909-838d6b92c22e?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
    ],
    "large": [
        "https://images.unsplash.com/photo-1594736797933-d0601ba2f4b7?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1596495578065-6e0763fa1178?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    ],
    "mixed": [
        "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", 
        "https://images.unsplash.com/photo-1594736797933-d0601ba2f4b7?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1558769132-cb1aea458c5e?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1574015974293-817f0ebebb74?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1581803118522-7b72a50f7e9f?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1596495578065-6e0763fa1178?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80",
        "https://images.unsplash.com/photo-1520006403909-838d6b92c22e?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80",
    ]
}

class BatchTestConfig:
    """Configuration for batch processing test."""
    
    def __init__(self):
        # Test parameters
        self.batch_size = 10
        self.concurrent_batches = 3
        self.total_images = 50
        self.image_set = "mixed"
        self.upload_s3 = False
        self.test_duration_minutes = None  # None = run until all images processed
        
        # Advanced parameters
        self.warmup_batches = 1
        self.cooldown_seconds = 10
        self.timeout_per_batch = 300  # 5 minutes
        self.detailed_logging = True
        self.save_results = True
        
        # Monitoring parameters
        self.monitor_pool_stats = True
        self.monitor_gpu_memory = True
        self.stats_interval = 5  # seconds
    
    def __str__(self):
        return f"""
🏊‍♂️ BATCH PROCESSING TEST CONFIGURATION
================================================================================
📦 Batch Parameters:
   - Batch Size: {self.batch_size} images per batch
   - Concurrent Batches: {self.concurrent_batches} batches running in parallel
   - Total Images: {self.total_images} images to process
   - Image Set: {self.image_set} ({len(TEST_IMAGE_SETS.get(self.image_set, []))} unique images)
   
⚙️ Processing Options:
   - Upload to S3: {'Yes' if self.upload_s3 else 'No'}
   - Test Duration: {f'{self.test_duration_minutes} minutes' if self.test_duration_minutes else 'Until completion'}
   - Timeout per Batch: {self.timeout_per_batch} seconds
   
🔧 Advanced Settings:
   - Warmup Batches: {self.warmup_batches}
   - Cooldown Between Tests: {self.cooldown_seconds} seconds
   - Detailed Logging: {'Yes' if self.detailed_logging else 'No'}
   
📊 Monitoring:
   - Pool Statistics: {'Yes' if self.monitor_pool_stats else 'No'}
   - GPU Memory: {'Yes' if self.monitor_gpu_memory else 'No'}
   - Stats Interval: {self.stats_interval} seconds
================================================================================
"""

async def get_pool_stats(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Get current model pool statistics."""
    try:
        async with session.get(f"{BASE_URL}/pool_stats", timeout=30) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def get_server_status(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Get current server status."""
    try:
        async with session.get(f"{BASE_URL}/status", timeout=30) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

def generate_image_urls(config: BatchTestConfig) -> List[str]:
    """Generate image URLs based on configuration."""
    image_set = TEST_IMAGE_SETS.get(config.image_set, TEST_IMAGE_SETS["mixed"])
    
    # Repeat images to reach total_images count
    images_needed = config.total_images
    repeated_images = []
    
    while len(repeated_images) < images_needed:
        repeated_images.extend(image_set)
    
    return repeated_images[:images_needed]

def chunk_images(image_urls: List[str], batch_size: int) -> List[List[str]]:
    """Split image URLs into batches."""
    return [image_urls[i:i + batch_size] for i in range(0, len(image_urls), batch_size)]

async def process_single_batch(
    session: aiohttp.ClientSession, 
    batch_images: List[str], 
    batch_id: int,
    config: BatchTestConfig
) -> Dict[str, Any]:
    """Process a single batch of images."""
    start_time = time.time()
    
    if config.detailed_logging:
        print(f"🏊‍♂️ Batch {batch_id}: Starting {len(batch_images)} images...")
    
    try:
        payload = {
            "image_urls": batch_images,
            "upload_to_s3": config.upload_s3
        }
        
        timeout = aiohttp.ClientTimeout(total=config.timeout_per_batch)
        async with session.post(f"{BASE_URL}/batch_infer", json=payload, timeout=timeout) as response:
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                successful_count = result.get("successful_count", 0)
                failed_count = result.get("failed_count", 0)
                
                if config.detailed_logging:
                    print(f"✅ Batch {batch_id}: {duration:.2f}s - {successful_count}/{len(batch_images)} success")
                
                return {
                    "batch_id": batch_id,
                    "success": True,
                    "duration": duration,
                    "successful_count": successful_count,
                    "failed_count": failed_count,
                    "batch_size": len(batch_images),
                    "throughput": successful_count / duration if duration > 0 else 0,
                    "response": result
                }
            else:
                error_text = await response.text()
                if config.detailed_logging:
                    print(f"❌ Batch {batch_id}: HTTP {response.status} - {error_text[:100]}")
                
                return {
                    "batch_id": batch_id,
                    "success": False,
                    "duration": duration,
                    "error": f"HTTP {response.status}: {error_text[:200]}",
                    "batch_size": len(batch_images)
                }
                
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        if config.detailed_logging:
            print(f"❌ Batch {batch_id}: Exception - {str(e)[:100]}")
        
        return {
            "batch_id": batch_id,
            "success": False,
            "duration": duration,
            "error": str(e),
            "batch_size": len(batch_images)
        }

async def monitor_stats(session: aiohttp.ClientSession, config: BatchTestConfig, stop_event: asyncio.Event):
    """Monitor pool and GPU stats during the test."""
    stats_history = []
    
    while not stop_event.is_set():
        try:
            pool_stats = await get_pool_stats(session)
            timestamp = time.time()
            
            if pool_stats and "pool_statistics" in pool_stats:
                stats_entry = {
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                    **pool_stats["pool_statistics"]
                }
                stats_history.append(stats_entry)
                
                if config.detailed_logging:
                    ps = pool_stats["pool_statistics"]
                    gpu = ps.get("gpu_memory", {})
                    print(f"📊 Pool: {ps.get('available_models', '?')}/30 available, "
                          f"{ps.get('active_requests', '?')} active, "
                          f"GPU: {gpu.get('allocated_gb', 0):.2f}GB")
            
            await asyncio.sleep(config.stats_interval)
            
        except Exception as e:
            if config.detailed_logging:
                print(f"⚠️ Stats monitoring error: {e}")
            await asyncio.sleep(config.stats_interval)
    
    return stats_history

async def run_warmup(session: aiohttp.ClientSession, config: BatchTestConfig):
    """Run warmup batches to prepare the model pool."""
    if config.warmup_batches <= 0:
        return
    
    print(f"\n🔥 Running {config.warmup_batches} warmup batch(es)...")
    
    warmup_images = TEST_IMAGE_SETS["small"][:min(5, config.batch_size)]
    
    for i in range(config.warmup_batches):
        result = await process_single_batch(session, warmup_images, f"warmup-{i+1}", config)
        if result["success"]:
            print(f"✅ Warmup {i+1}: {result['duration']:.2f}s")
        else:
            print(f"❌ Warmup {i+1}: Failed")
    
    print("🔥 Warmup completed, waiting before main test...")
    await asyncio.sleep(config.cooldown_seconds)

async def run_batch_test(config: BatchTestConfig) -> Dict[str, Any]:
    """Run the main batch processing test."""
    print(config)
    
    # Generate test images
    all_images = generate_image_urls(config)
    batches = chunk_images(all_images, config.batch_size)
    
    print(f"📦 Generated {len(batches)} batches from {len(all_images)} images")
    
    # Create session with increased limits for 40 concurrent batches
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    connector = aiohttp.TCPConnector(limit=60, limit_per_host=45)  # ✅ INCREASED: Support 40 concurrent batches
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Check service readiness
        print("🔍 Checking service status...")
        server_status = await get_server_status(session)
        
        if "error" in server_status:
            print(f"❌ Service not ready: {server_status['error']}")
            return {"error": "Service not ready", "status": server_status}
        
        pool_info = server_status.get("model_pool", {})
        print(f"✅ Service ready - Pool: {pool_info.get('available_models', '?')}/30 models available")
        
        # Run warmup
        await run_warmup(session, config)
        
        # Start monitoring
        stop_monitoring = asyncio.Event()
        monitoring_task = None
        
        if config.monitor_pool_stats:
            monitoring_task = asyncio.create_task(monitor_stats(session, config, stop_monitoring))
        
        # Main test execution
        print(f"\n🚀 STARTING BATCH PROCESSING TEST")
        print("="*80)
        
        test_start_time = time.time()
        results = []
        
        try:
            # Process batches with controlled concurrency
            semaphore = asyncio.Semaphore(config.concurrent_batches)
            
            async def process_batch_with_semaphore(batch_images, batch_id):
                async with semaphore:
                    return await process_single_batch(session, batch_images, batch_id, config)
            
            # Create all batch tasks
            batch_tasks = [
                process_batch_with_semaphore(batch_images, i+1)
                for i, batch_images in enumerate(batches)
            ]
            
            # Execute batches
            if config.test_duration_minutes:
                # Time-limited test
                try:
                    timeout_seconds = config.test_duration_minutes * 60
                    results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    print(f"⏰ Test timed out after {config.test_duration_minutes} minutes")
                    results = [{"error": "timeout"} for _ in batch_tasks]
            else:
                # Process all batches
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        finally:
            # Stop monitoring
            if monitoring_task:
                stop_monitoring.set()
                try:
                    stats_history = await asyncio.wait_for(monitoring_task, timeout=5.0)
                except asyncio.TimeoutError:
                    stats_history = []
            else:
                stats_history = []
        
        test_end_time = time.time()
        total_test_duration = test_end_time - test_start_time
        
        # Analyze results
        successful_batches = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_batches = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exception_batches = [r for r in results if not isinstance(r, dict)]
        
        total_successful_images = sum(r.get("successful_count", 0) for r in successful_batches)
        total_failed_images = sum(r.get("failed_count", 0) for r in successful_batches) + \
                             sum(r.get("batch_size", 0) for r in failed_batches) + \
                             len(exception_batches) * config.batch_size
        
        # Calculate performance metrics
        if successful_batches:
            batch_durations = [r["duration"] for r in successful_batches]
            avg_batch_duration = statistics.mean(batch_durations)
            min_batch_duration = min(batch_durations)
            max_batch_duration = max(batch_durations)
            median_batch_duration = statistics.median(batch_durations)
            
            batch_throughputs = [r["throughput"] for r in successful_batches]
            avg_throughput = statistics.mean(batch_throughputs)
        else:
            avg_batch_duration = min_batch_duration = max_batch_duration = median_batch_duration = 0
            avg_throughput = 0
        
        overall_throughput = total_successful_images / total_test_duration if total_test_duration > 0 else 0
        
        # Print results
        print(f"\n🏆 BATCH PROCESSING TEST RESULTS")
        print("="*80)
        print(f"⏱️  Total Duration: {total_test_duration:.2f} seconds")
        print(f"📦 Batches Processed: {len(successful_batches)}/{len(batches)} successful")
        print(f"🖼️  Images Processed: {total_successful_images}/{len(all_images)} successful")
        print(f"📈 Overall Throughput: {overall_throughput:.2f} images/second")
        print(f"📊 Average Batch Throughput: {avg_throughput:.2f} images/second")
        
        if successful_batches:
            print(f"\n📊 Batch Performance Statistics:")
            print(f"   - Average Duration: {avg_batch_duration:.2f}s")
            print(f"   - Median Duration: {median_batch_duration:.2f}s")
            print(f"   - Min Duration: {min_batch_duration:.2f}s")
            print(f"   - Max Duration: {max_batch_duration:.2f}s")
        
        # Get final pool stats
        final_pool_stats = await get_pool_stats(session)
        
        # Compile full results
        test_results = {
            "config": {
                "batch_size": config.batch_size,
                "concurrent_batches": config.concurrent_batches,
                "total_images": config.total_images,
                "image_set": config.image_set,
                "upload_s3": config.upload_s3
            },
            "summary": {
                "total_duration": total_test_duration,
                "successful_batches": len(successful_batches),
                "failed_batches": len(failed_batches),
                "total_successful_images": total_successful_images,
                "total_failed_images": total_failed_images,
                "overall_throughput": overall_throughput,
                "avg_batch_throughput": avg_throughput
            },
            "batch_stats": {
                "avg_duration": avg_batch_duration,
                "median_duration": median_batch_duration,
                "min_duration": min_batch_duration,
                "max_duration": max_batch_duration
            },
            "batches": results,
            "monitoring": stats_history,
            "final_pool_stats": final_pool_stats
        }
        
        # Save results
        if config.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_test_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "test_results": test_results
                }, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")
        
        return test_results

def create_config_from_args() -> BatchTestConfig:
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Batch Processing Optimizer Test")
    
    # Test parameters
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Number of images per batch (1-50)")
    parser.add_argument("--concurrent-batches", type=int, default=3,
                       help="Number of batches to run in parallel (1-40)")  # ✅ FIXED: Updated for 30 model pool
    parser.add_argument("--total-images", type=int, default=50,
                       help="Total number of images to process")
    parser.add_argument("--image-set", choices=list(TEST_IMAGE_SETS.keys()), 
                       default="mixed", help="Image set to use")
    parser.add_argument("--upload-s3", action="store_true",
                       help="Enable S3 upload for processed images")
    parser.add_argument("--duration", type=int, 
                       help="Test duration in minutes (default: until completion)")
    
    # Advanced parameters
    parser.add_argument("--warmup", type=int, default=1,
                       help="Number of warmup batches")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per batch in seconds")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce logging output")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")
    parser.add_argument("--no-monitoring", action="store_true",
                       help="Disable pool stats monitoring")
    
    args = parser.parse_args()
    
    config = BatchTestConfig()
    config.batch_size = max(1, min(50, args.batch_size))
    config.concurrent_batches = max(1, min(40, args.concurrent_batches))  # ✅ FIXED: 30 model pool + oversubscription
    config.total_images = max(1, args.total_images)
    config.image_set = args.image_set
    config.upload_s3 = args.upload_s3
    config.test_duration_minutes = args.duration
    config.warmup_batches = max(0, args.warmup)
    config.timeout_per_batch = max(60, args.timeout)
    config.detailed_logging = not args.quiet
    config.save_results = not args.no_save
    config.monitor_pool_stats = not args.no_monitoring
    
    return config

async def main():
    """Main function."""
    if len(sys.argv) > 1:
        config = create_config_from_args()
    else:
        # Interactive configuration
        print("🏊‍♂️ BATCH PROCESSING OPTIMIZER")
        print("="*80)
        print("Quick setup - press Enter for defaults:")
        
        config = BatchTestConfig()
        
        # Basic parameters
        batch_size = input(f"Batch size [{config.batch_size}]: ").strip()
        if batch_size:
            config.batch_size = max(1, min(50, int(batch_size)))
        
        concurrent = input(f"Concurrent batches [{config.concurrent_batches}]: ").strip()
        if concurrent:
            config.concurrent_batches = max(1, min(40, int(concurrent)))  # ✅ FIXED: 30 model pool + oversubscription
        
        total = input(f"Total images [{config.total_images}]: ").strip()
        if total:
            config.total_images = max(1, int(total))
        
        print(f"\nImage sets: {', '.join(TEST_IMAGE_SETS.keys())}")
        image_set = input(f"Image set [{config.image_set}]: ").strip()
        if image_set and image_set in TEST_IMAGE_SETS:
            config.image_set = image_set
        
        s3_upload = input("Upload to S3? [y/N]: ").strip().lower()
        config.upload_s3 = s3_upload in ['y', 'yes']
    
    # Run the test
    try:
        results = await run_batch_test(config)
        
        if "error" in results:
            print(f"\n❌ Test failed: {results['error']}")
            return 1
        
        print("\n✅ Test completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 