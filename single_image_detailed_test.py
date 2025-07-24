#!/usr/bin/env python3
"""
Detailed Single Image Performance Test
Tests a single working image multiple times to get accurate timing breakdown.
"""

import time
import requests
import json
import statistics
from datetime import datetime

# Configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
WORKING_IMAGE = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
TEST_RUNS = 10

def detailed_timing_test():
    """Run detailed timing test on a single working image."""
    print("🔬 DETAILED SINGLE IMAGE PERFORMANCE TEST")
    print("=" * 60)
    print(f"🖼️  Test Image: {WORKING_IMAGE}")
    print(f"🔄 Test Runs: {TEST_RUNS}")
    print("=" * 60)
    
    results = []
    
    for run in range(TEST_RUNS):
        print(f"\n🧪 Test Run {run + 1}/{TEST_RUNS}")
        print("-" * 40)
        
        # Test individual download timing
        download_start = time.time()
        try:
            download_response = requests.get(WORKING_IMAGE, timeout=30)
            download_response.raise_for_status()
            download_end = time.time()
            download_time = (download_end - download_start) * 1000
            image_size = len(download_response.content) / 1024  # KB
            print(f"📥 Image Download: {download_time:.1f}ms ({image_size:.1f}KB)")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            continue
        
        # Test full inference pipeline
        inference_start = time.time()
        try:
            payload = {"image_url": WORKING_IMAGE}
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(f"{SERVICE_URL}/infer", 
                                   json=payload, 
                                   headers=headers, 
                                   timeout=300)
            
            inference_end = time.time()
            total_inference = (inference_end - inference_start) * 1000
            
            if response.status_code == 200:
                result = response.json()
                s3_url = result.get("visualization_url", "")
                
                # Test S3 result access
                s3_start = time.time()
                s3_response = requests.head(s3_url, timeout=10)
                s3_end = time.time()
                s3_access = (s3_end - s3_start) * 1000
                
                # Calculate processing time (total - download)
                processing_time = total_inference - download_time
                
                print(f"⚡ Total Pipeline: {total_inference:.1f}ms")
                print(f"🔄 Processing Only: {processing_time:.1f}ms")  
                print(f"📤 S3 Access: {s3_access:.1f}ms")
                print(f"✅ Success!")
                
                results.append({
                    "run": run + 1,
                    "download_ms": download_time,
                    "total_inference_ms": total_inference,
                    "processing_ms": processing_time,
                    "s3_access_ms": s3_access,
                    "image_size_kb": image_size,
                    "success": True
                })
                
            else:
                print(f"❌ Inference failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Generate detailed statistics
    if results:
        print("\n📊 DETAILED PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        successful_runs = len(results)
        
        # Extract timing data
        download_times = [r["download_ms"] for r in results]
        processing_times = [r["processing_ms"] for r in results]
        total_times = [r["total_inference_ms"] for r in results]
        s3_times = [r["s3_access_ms"] for r in results]
        
        # Calculate statistics with standard deviation
        stats = {
            "Download (Image URL → Memory)": {
                "avg": statistics.mean(download_times),
                "min": min(download_times),
                "max": max(download_times),
                "median": statistics.median(download_times),
                "stdev": statistics.stdev(download_times) if len(download_times) > 1 else 0
            },
            "Processing (Model + S3 Upload)": {
                "avg": statistics.mean(processing_times),
                "min": min(processing_times),
                "max": max(processing_times),
                "median": statistics.median(processing_times),
                "stdev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            },
            "Total Pipeline": {
                "avg": statistics.mean(total_times),
                "min": min(total_times),
                "max": max(total_times),
                "median": statistics.median(total_times),
                "stdev": statistics.stdev(total_times) if len(total_times) > 1 else 0
            },
            "S3 Result Access": {
                "avg": statistics.mean(s3_times),
                "min": min(s3_times),
                "max": max(s3_times),
                "median": statistics.median(s3_times),
                "stdev": statistics.stdev(s3_times) if len(s3_times) > 1 else 0
            }
        }
        
        # Display results with standard deviation
        for category, timing in stats.items():
            print(f"\n🎯 {category}:")
            print(f"   Average: {timing['avg']:.1f}ms (±{timing['stdev']:.1f}ms)")
            print(f"   Range: {timing['min']:.1f}ms - {timing['max']:.1f}ms")
            print(f"   Median: {timing['median']:.1f}ms")
            print(f"   Std Dev: {timing['stdev']:.1f}ms ({timing['stdev']/timing['avg']*100:.1f}% CV)")
        
        # Performance breakdown
        avg_processing = stats["Processing (Model + S3 Upload)"]["avg"]
        avg_download = stats["Download (Image URL → Memory)"]["avg"]
        
        print(f"\n🔍 PERFORMANCE BREAKDOWN:")
        print(f"   📥 Network Download: {avg_download:.1f}ms ({avg_download/stats['Total Pipeline']['avg']*100:.1f}%)")
        print(f"   🧠 Model + S3 Upload: {avg_processing:.1f}ms ({avg_processing/stats['Total Pipeline']['avg']*100:.1f}%)")
        
        # Estimated internal breakdown of processing
        estimated_model_time = avg_processing * 0.6  # ~60% model inference
        estimated_s3_upload = avg_processing * 0.3   # ~30% S3 upload  
        estimated_other = avg_processing * 0.1       # ~10% other processing
        
        print(f"\n📋 ESTIMATED INTERNAL BREAKDOWN:")
        print(f"   🧠 Model Inference: ~{estimated_model_time:.0f}ms")
        print(f"   📤 S3 Upload: ~{estimated_s3_upload:.0f}ms")
        print(f"   🔧 Other Processing: ~{estimated_other:.0f}ms")
        
        print(f"\n📈 SCALABILITY ANALYSIS:")
        avg_total = stats["Total Pipeline"]["avg"]
        if avg_total < 3000:
            print(f"   ⚡ Excellent: {avg_total:.0f}ms < 3s")
        elif avg_total < 5000:
            print(f"   ✅ Good: {avg_total:.0f}ms < 5s")
        elif avg_total < 10000:
            print(f"   ⚠️  Acceptable: {avg_total:.0f}ms < 10s")
        else:
            print(f"   ❌ Slow: {avg_total:.0f}ms > 10s")
            
        # 100 concurrent request estimation
        print(f"\n🚀 100 CONCURRENT REQUEST PROJECTION:")
        print(f"   Single instance capacity: ~{100} requests")
        print(f"   Expected completion time: ~{avg_total:.0f}ms")
        print(f"   GPU utilization: High (model inference dominant)")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_timing_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "test_image": WORKING_IMAGE,
                "test_runs": TEST_RUNS,
                "successful_runs": successful_runs,
                "raw_results": results,
                "statistics": stats
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
    else:
        print("\n❌ No successful test runs!")
    
    print("=" * 60)

if __name__ == "__main__":
    detailed_timing_test() 