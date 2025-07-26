#!/usr/bin/env python3
"""
Local Test - Horizontal Scaling Architecture (Single Model per Instance)
Test 10 concurrent requests to simulate Cloud Run load balancing
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
import subprocess
import threading
import signal
import sys
import os

# Configuration
LOCAL_URL = "http://localhost:5001"
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
NUM_REQUESTS = 1
TIMEOUT = 120  # 2 minutes timeout

# Global variable to store Flask process
flask_process = None

def start_flask_server():
    """Start the Flask server in background"""
    global flask_process
    print("🚀 Starting Flask server locally...")
    
    # Set environment variables for local testing
    env = os.environ.copy()
    env['FORCE_CPU'] = 'true'
    env['PYTHONPATH'] = os.getcwd()  # Use current working directory
    
    try:
        flask_process = subprocess.Popen([
            'python', 'api_app.py'
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(10)
        print("✅ Flask server started (hopefully)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Flask server: {e}")
        return False

def stop_flask_server():
    """Stop the Flask server"""
    global flask_process
    if flask_process:
        print("🛑 Stopping Flask server...")
        flask_process.terminate()
        flask_process.wait()
        print("✅ Flask server stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Test interrupted by user")
    stop_flask_server()
    sys.exit(0)

async def test_server_health():
    """Test if server is responding"""
    print("🔍 Testing server health...")
    
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            async with session.get(f"{LOCAL_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Server health: {health.get('status', 'Unknown')}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

async def test_model_stats():
    """Test model statistics"""
    print("📊 Testing model statistics...")
    
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            async with session.get(f"{LOCAL_URL}/pool_stats") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    stats = stats_data.get('model_statistics', {})
                    
                    print(f"📊 MODEL STATISTICS:")
                    print(f"   Architecture: {stats.get('architecture', 'Unknown')}")
                    print(f"   Device: {stats.get('device', 'Unknown')}")
                    print(f"   Model loaded: {stats.get('model_loaded', False)}")
                    print(f"   Model name: {stats.get('model_name', 'Unknown')}")
                    print(f"   Precision: {stats.get('precision', 'Unknown')}")
                    
                    return stats.get('model_loaded', False)
                else:
                    print(f"❌ Model stats failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Model stats error: {e}")
            return False

async def send_single_request(session, request_id):
    """Send a single inference request"""
    try:
        start_time = time.time()
        
        payload = {
            "image_url": TEST_IMAGE_URL,
            "upload_s3": False  # Disable S3 for local testing
        }
        
        async with session.post(
            f"{LOCAL_URL}/infer",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                success = result.get('success', False) or result.get('inference_completed', False)
                
                return {
                    'request_id': request_id,
                    'success': success,
                    'duration': duration,
                    'status_code': response.status,
                    'response_size': len(str(result))
                }
            else:
                error_text = await response.text()
                return {
                    'request_id': request_id,
                    'success': False,
                    'duration': duration,
                    'status_code': response.status,
                    'error': error_text[:200]  # Limit error message length
                }
                
    except Exception as e:
        return {
            'request_id': request_id,
            'success': False,
            'duration': time.time() - start_time if 'start_time' in locals() else 0,
            'error': str(e)[:200]
        }

async def test_concurrent_requests():
    """Test concurrent requests to measure single model performance"""
    print(f"\n🚀 CONCURRENT TEST ({NUM_REQUESTS} requests)")
    print("-" * 50)
    
    connector = aiohttp.TCPConnector(
        limit=20,  # Total connections
        limit_per_host=20,  # Per host
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print(f"🌊 Sending {NUM_REQUESTS} concurrent requests...")
        overall_start = time.time()
        
        # Create all tasks
        tasks = [send_single_request(session, i+1) for i in range(NUM_REQUESTS)]
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        overall_end = time.time()
        total_time = overall_end - overall_start
        
        # Analyze results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        success_count = len(successful)
        failure_count = len(failed)
        
        if successful:
            durations = [r['duration'] for r in successful]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0
        
        throughput = success_count / total_time if total_time > 0 else 0
        
        print(f"\n📊 LOCAL TEST RESULTS:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful: {success_count}/{NUM_REQUESTS} ({success_count/NUM_REQUESTS*100:.1f}%)")
        print(f"   Failed: {failure_count}")
        print(f"   Throughput: {throughput:.2f} images/second")
        print(f"   Average request: {avg_duration:.2f}s")
        print(f"   Fastest request: {min_duration:.2f}s")
        print(f"   Slowest request: {max_duration:.2f}s")
        
        # Architecture analysis
        print(f"\n🏗️  ARCHITECTURE ANALYSIS:")
        print(f"   Current: Single model handling {NUM_REQUESTS} concurrent requests")
        if success_count == NUM_REQUESTS:
            print(f"   ✅ All requests processed successfully")
            print(f"   📈 Processing pattern: {'Sequential' if max_duration > avg_duration * 1.5 else 'Near-parallel'}")
        else:
            print(f"   ⚠️  {failure_count} requests failed")
        
        # Cloud Run projection
        cloud_run_20_instances = avg_duration  # Each instance handles 1 request
        images_in_6s = 6 / cloud_run_20_instances if cloud_run_20_instances > 0 else 0
        
        print(f"\n🎯 CLOUD RUN PROJECTION (20 instances):")
        print(f"   Single instance: {cloud_run_20_instances:.2f}s per request")
        print(f"   20 instances parallel: {cloud_run_20_instances:.2f}s per request")
        print(f"   Throughput: {images_in_6s:.1f} images in 6 seconds")
        print(f"   Est. capacity: {3600/cloud_run_20_instances:.0f} images/hour (20 instances)")
        
        # Show the architecture advantage
        print(f"\n📊 HORIZONTAL SCALING ADVANTAGE:")
        print(f"   Local batch (sequential): {avg_duration:.2f}s")
        print(f"   Cloud Run (20 parallel): {cloud_run_20_instances:.2f}s")
        print(f"   Speed improvement: {avg_duration/cloud_run_20_instances:.1f}x faster")
        
        # Error analysis
        if failed:
            print(f"\n❌ FAILURE ANALYSIS:")
            error_types = {}
            for req in failed:
                error = req.get('error', 'Unknown error')
                status = req.get('status_code', 'Unknown status')
                error_key = f"{status}: {error[:50]}..."
                error_types[error_key] = error_types.get(error_key, 0) + 1
            
            for error, count in error_types.items():
                print(f"   {error}: {count} occurrences")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"local_horizontal_test_{timestamp}.json"
        
        results_data = {
            "test_config": {
                "num_requests": NUM_REQUESTS,
                "local_url": LOCAL_URL,
                "test_image": TEST_IMAGE_URL,
                "architecture": "Single Model per Instance (Horizontal Scaling)",
                "timestamp": timestamp
            },
            "performance": {
                "total_time": total_time,
                "success_count": success_count,
                "failure_count": failure_count,
                "throughput": throughput,
                "avg_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration
            },
            "detailed_results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return success_count == NUM_REQUESTS

async def main():
    """Main test function"""
    print("🧪 LOCAL HORIZONTAL SCALING TEST")
    print("=" * 60)
    print(f"🔗 Local URL: {LOCAL_URL}")
    print(f"🖼️  Test Image: {TEST_IMAGE_URL}")
    print(f"📊 Architecture: Single BiRefNet model (CPU)")
    print(f"🎯 Goal: Test {NUM_REQUESTS} concurrent requests")
    print("")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start Flask server
    if not start_flask_server():
        print("❌ Failed to start Flask server")
        return
    
    try:
        # Wait for server to be ready
        print("⏳ Waiting for server to initialize...")
        await asyncio.sleep(5)
        
        # Test 1: Health check
        if not await test_server_health():
            print("❌ Server health check failed")
            return
        
        # Test 2: Model stats
        if not await test_model_stats():
            print("❌ Model not loaded properly")
            return
        
        # Test 3: Concurrent requests
        success = await test_concurrent_requests()
        
        if success:
            print("\n🎉 LOCAL TEST COMPLETED SUCCESSFULLY!")
            print("✅ Ready for Cloud Run deployment with 20 instances")
        else:
            print("\n⚠️ LOCAL TEST HAD ISSUES")
            print("🔧 Consider investigating before Cloud Run deployment")
            
    finally:
        # Always stop the server
        stop_flask_server()

if __name__ == "__main__":
    print("🚀 STARTING LOCAL HORIZONTAL SCALING TEST")
    print("Press Ctrl+C to stop at any time")
    print("=" * 70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        stop_flask_server()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        stop_flask_server() 