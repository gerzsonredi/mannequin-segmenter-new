#!/usr/bin/env python3
"""
Performance Benchmark for Mannequin Segmentation Pipeline
Measures detailed timing for each step from image URL to S3 response.
"""

import time
import requests
import json
import os
from datetime import datetime
from typing import Dict, List
import statistics

# Configuration
SERVICE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
TEST_IMAGES = [
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg",
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/DKB-Ruha-Midi-Koktélruha-Lila-Ujjatlan-Megkötős-Maxi-Alkalmi-132434085.jpg",
    "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Férfi-Cipő-Adidas-Stan-Smith-Fehér-Bézs-132434086.jpg"
]

def get_auth_token():
    """Get GCP auth token for authenticated requests."""
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"⚠️  Warning: Could not get auth token: {e}")
        return None

def benchmark_health_check() -> Dict:
    """Benchmark the health check endpoint."""
    print("🔍 Testing health check endpoint...")
    
    times = []
    for i in range(3):  # Multiple measurements
        start = time.time()
        
        try:
            token = get_auth_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
                
            response = requests.get(f"{SERVICE_URL}/health", 
                                  headers=headers, timeout=30)
            
            end = time.time()
            response_time = (end - start) * 1000
            times.append(response_time)
            
            print(f"  Health check {i+1}: {response_time:.1f}ms (Status: {response.status_code})")
            
        except Exception as e:
            print(f"  Health check {i+1}: FAILED - {e}")
            times.append(None)
    
    valid_times = [t for t in times if t is not None]
    if valid_times:
        return {
            "avg_ms": statistics.mean(valid_times),
            "min_ms": min(valid_times),
            "max_ms": max(valid_times),
            "success_rate": len(valid_times) / len(times)
        }
    else:
        return {"error": "All health checks failed"}

def benchmark_image_download(image_url: str) -> Dict:
    """Benchmark image download step separately."""
    print(f"📥 Testing image download: {image_url}")
    
    times = []
    sizes = []
    
    for i in range(3):
        start = time.time()
        
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            end = time.time()
            download_time = (end - start) * 1000
            file_size = len(response.content)
            
            times.append(download_time)
            sizes.append(file_size)
            
            print(f"  Download {i+1}: {download_time:.1f}ms ({file_size/1024:.1f}KB)")
            
        except Exception as e:
            print(f"  Download {i+1}: FAILED - {e}")
            times.append(None)
    
    valid_times = [t for t in times if t is not None]
    if valid_times and sizes:
        return {
            "avg_download_ms": statistics.mean(valid_times),
            "min_download_ms": min(valid_times),
            "max_download_ms": max(valid_times),
            "avg_size_kb": statistics.mean(sizes) / 1024,
            "success_rate": len(valid_times) / len(times)
        }
    else:
        return {"error": "All downloads failed"}

def benchmark_full_inference(image_url: str) -> Dict:
    """Benchmark the complete inference pipeline."""
    print(f"🔄 Testing full inference: {image_url}")
    
    payload = {
        "image_url": image_url
    }
    
    headers = {"Content-Type": "application/json"}
    token = get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Detailed timing
    start_total = time.time()
    
    try:
        # Phase 1: Request sending
        req_start = time.time()
        response = requests.post(f"{SERVICE_URL}/infer", 
                               json=payload, 
                               headers=headers, 
                               timeout=300)  # 5 min timeout
        req_end = time.time()
        
        total_time = (req_end - start_total) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse response
            visualization_url = result.get("visualization_url", "")
            
            # Test S3 result access
            s3_start = time.time()
            s3_response = requests.head(visualization_url, timeout=10)
            s3_end = time.time()
            s3_access_time = (s3_end - s3_start) * 1000
            
            print(f"  ✅ Total inference: {total_time:.1f}ms")
            print(f"  📤 S3 result access: {s3_access_time:.1f}ms")
            print(f"  🔗 Result URL: {visualization_url}")
            
            return {
                "success": True,
                "total_inference_ms": total_time,
                "s3_access_ms": s3_access_time,
                "visualization_url": visualization_url,
                "status_code": response.status_code
            }
        else:
            print(f"  ❌ Inference failed: {response.status_code}")
            print(f"  Response: {response.text}")
            
            return {
                "success": False,
                "total_inference_ms": total_time,
                "status_code": response.status_code,
                "error": response.text
            }
            
    except Exception as e:
        end_time = time.time()
        total_time = (end_time - start_total) * 1000
        
        print(f"  ❌ Inference exception: {e}")
        
        return {
            "success": False,
            "total_inference_ms": total_time,
            "error": str(e)
        }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark of the entire system."""
    print("🚀 Starting Comprehensive Performance Benchmark")
    print("=" * 80)
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print(f"🌐 Service URL: {SERVICE_URL}")
    print(f"🖼️  Test Images: {len(TEST_IMAGES)}")
    print("=" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "service_url": SERVICE_URL,
        "health_check": {},
        "image_downloads": [],
        "inferences": [],
        "summary": {}
    }
    
    # 1. Health Check Benchmark
    print("\n📊 PHASE 1: Health Check Performance")
    print("-" * 50)
    results["health_check"] = benchmark_health_check()
    
    # 2. Image Download Benchmark  
    print("\n📊 PHASE 2: Image Download Performance")
    print("-" * 50)
    for i, image_url in enumerate(TEST_IMAGES):
        download_result = benchmark_image_download(image_url)
        download_result["image_index"] = i
        download_result["image_url"] = image_url
        results["image_downloads"].append(download_result)
    
    # 3. Full Inference Benchmark
    print("\n📊 PHASE 3: Full Inference Performance")
    print("-" * 50)
    for i, image_url in enumerate(TEST_IMAGES):
        inference_result = benchmark_full_inference(image_url)
        inference_result["image_index"] = i  
        inference_result["image_url"] = image_url
        results["inferences"].append(inference_result)
        
        # Brief pause between tests
        time.sleep(2)
    
    # 4. Generate Summary
    print("\n📊 PHASE 4: Performance Summary")
    print("-" * 50)
    
    successful_inferences = [r for r in results["inferences"] if r.get("success")]
    
    if successful_inferences:
        inference_times = [r["total_inference_ms"] for r in successful_inferences]
        
        summary = {
            "total_tests": len(TEST_IMAGES),
            "successful_inferences": len(successful_inferences),
            "success_rate": len(successful_inferences) / len(TEST_IMAGES),
            "avg_inference_ms": statistics.mean(inference_times),
            "min_inference_ms": min(inference_times),
            "max_inference_ms": max(inference_times),
            "median_inference_ms": statistics.median(inference_times),
            "stdev_inference_ms": statistics.stdev(inference_times) if len(inference_times) > 1 else 0
        }
        
        # Download times
        successful_downloads = [r for r in results["image_downloads"] if "avg_download_ms" in r]
        if successful_downloads:
            download_times = [r["avg_download_ms"] for r in successful_downloads]
            summary.update({
                "avg_download_ms": statistics.mean(download_times),
                "min_download_ms": min(download_times),
                "max_download_ms": max(download_times)
            })
        
        print(f"✅ Success Rate: {summary['success_rate']*100:.1f}% ({summary['successful_inferences']}/{summary['total_tests']})")
        print(f"⚡ Average Inference: {summary['avg_inference_ms']:.1f}ms (±{summary['stdev_inference_ms']:.1f}ms)")
        print(f"📥 Average Download: {summary.get('avg_download_ms', 0):.1f}ms")
        print(f"🔄 Processing Time: {summary['avg_inference_ms'] - summary.get('avg_download_ms', 0):.1f}ms")
        print(f"📊 Range: {summary['min_inference_ms']:.1f}ms - {summary['max_inference_ms']:.1f}ms")
        print(f"📈 Std Dev: {summary['stdev_inference_ms']:.1f}ms ({summary['stdev_inference_ms']/summary['avg_inference_ms']*100:.1f}% CV)")
        
    else:
        summary = {
            "total_tests": len(TEST_IMAGES),
            "successful_inferences": 0,
            "success_rate": 0,
            "error": "All inferences failed"
        }
        print("❌ All inference tests failed!")
    
    results["summary"] = summary
    
    # 5. Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("=" * 80)
    
    return results

def analyze_pipeline_breakdown():
    """Analyze estimated breakdown of pipeline timing."""
    print("\n🔍 ESTIMATED PIPELINE BREAKDOWN")
    print("-" * 50)
    
    estimated_breakdown = {
        "1. Input Validation": "~1-5ms",
        "2. Image Download": "~100-2000ms (network dependent)",  
        "3. Image Loading": "~10-50ms",
        "4. Preprocessing": "~20-100ms (resize, tensor conversion)",
        "5. Model Inference": "~500-3000ms (GPU: ~500ms, CPU: ~3000ms)",
        "6. Mask Extraction": "~50-200ms", 
        "7. Mask Application": "~20-100ms",
        "8. Post-processing": "~100-500ms (morphology)",
        "9. Image Conversion": "~50-200ms (numpy→PIL→JPEG)",
        "10. S3 Upload": "~200-1000ms (network dependent)",
        "11. Response": "~1-10ms"
    }
    
    for step, timing in estimated_breakdown.items():
        print(f"{step}: {timing}")
    
    print("\n💡 Key Performance Factors:")
    print("  🌐 Network: Image download + S3 upload (~300-3000ms)")
    print("  🧠 Model: BiRefNet inference (~500-3000ms)")  
    print("  🖼️  Processing: Image operations (~200-800ms)")
    print("  ⚡ Total Expected: ~1200-6800ms per image")

if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    # Show estimated breakdown
    analyze_pipeline_breakdown()
    
    print(f"\n🎯 Benchmark completed successfully!")
    print(f"Check the saved JSON file for detailed results.") 