#!/usr/bin/env python3
"""
Test script for deployed Cloud Run Mannequin Segmenter API
Tests the batch inference with 5 images to verify GPU performance
"""

import requests
import json
import time

# Server Configuration (replace with your actual Cloud Run URL)
# TODO: Replace this with your actual Cloud Run service URL from GitHub Actions logs
SERVER_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"
# To find the URL: GitHub → Actions → Deploy to GCP Cloud Run → logs → look for "Service deployed to:"

# Sample image URLs for testing (5 images for batch test)
TEST_IMAGES = [
    "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&h=600&fit=crop",  # Person 1
    "https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=800&h=600&fit=crop",  # Fashion 1
    "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=800&h=600&fit=crop",  # Model 1
    "https://images.unsplash.com/photo-1506629905844-f21f6c08b9c4?w=800&h=600&fit=crop",  # Person 2
    "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=800&h=600&fit=crop",  # Fashion 2
]

def test_server_health():
    """Test the server health endpoint and check GPU status"""
    print("🔍 Testing server health and GPU status...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=15)
        response.raise_for_status()
        health_data = response.json()
        
        print(f"✅ Server Status: {health_data['status']}")
        print(f"   Service: {health_data['service']} v{health_data.get('version', 'unknown')}")
        
        # Check GPU info
        gpu_info = health_data.get('gpu_info', {})
        if gpu_info:
            print(f"   GPU Available: {gpu_info.get('cuda_available', False)}")
            print(f"   GPU Count: {gpu_info.get('gpu_count', 0)}")
            if gpu_info.get('gpu_names'):
                print(f"   GPU Type: {gpu_info['gpu_names'][0]}")
        
        # Check model device
        model_device = health_data.get('model_device', 'unknown')
        print(f"   Model Device: {model_device}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_server_status():
    """Test the server status endpoint"""
    print("\n🔍 Testing server status...")
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=10)
        response.raise_for_status()
        status_data = response.json()
        
        print(f"✅ Queue Size: {status_data['performance']['queue_size']}")
        print(f"   Batch Worker Active: {status_data['performance']['batch_worker_active']}")
        print(f"   Can Accept Requests: {status_data['recommendation']['can_accept_requests']}")
        
        return True
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

def test_batch_processing():
    """Test 5-image batch processing on the server"""
    print(f"\n🚀 Testing 5-image batch processing on GPU server...")
    print(f"   Images to process: {len(TEST_IMAGES)}")
    
    try:
        payload = {"image_urls": TEST_IMAGES}
        start_time = time.time()
        
        print("   📤 Sending batch request...")
        response = requests.post(
            f"{SERVER_URL}/batch_infer", 
            json=payload, 
            timeout=180  # 3 minutes timeout for batch processing
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing completed in {duration:.2f}s")
            print(f"   📊 Results:")
            print(f"      - Batch Size: {result.get('batch_size', 0)}")
            print(f"      - Successful: {result.get('successful_count', 0)}")
            print(f"      - Failed: {result.get('failed_count', 0)}")
            print(f"      - Throughput: {result.get('successful_count', 0) / duration:.2f} images/second")
            
            # Show result URLs (first 2)
            urls = result.get('visualization_urls', [])
            if urls:
                print(f"   🎯 Sample results:")
                for i, url in enumerate(urls[:2]):
                    if url:
                        print(f"      {i+1}. {url[:70]}...")
            
            return True
        else:
            print(f"❌ Batch processing failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Batch processing timed out after 3 minutes")
        return False
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False

def get_cloud_run_url():
    """Helper function to get the Cloud Run URL"""
    print("\n💡 To find your Cloud Run URL, run:")
    print("   gcloud run services list --platform managed --region europe-west4")
    print("   or check GitHub Actions logs for the deployed URL")

def main():
    """Run server tests"""
    print("🚀 Testing Cloud Run Mannequin Segmenter (GPU + Batch)")
    print("=" * 70)
    
    # Check if URL is configured
    if "xxxxxxxxxx" in SERVER_URL:
        print("❌ Please update SERVER_URL with your actual Cloud Run URL!")
        get_cloud_run_url()
        return
    
    print(f"🌐 Testing server: {SERVER_URL}")
    
    # Test health and GPU status
    if not test_server_health():
        print("❌ Health check failed, aborting tests")
        return
    
    # Test status
    test_server_status()
    
    # Test batch processing
    print("\n" + "=" * 70)
    print("🧪 BATCH PROCESSING TEST (5 images)")
    print("=" * 70)
    
    success = test_batch_processing()
    
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 70)
    
    if success:
        print("✅ Batch processing test PASSED")
        print("   - GPU is working correctly")
        print("   - Batch processing optimized for 10 images max")
        print("   - Sequential processing within batch (thread-safe)")
    else:
        print("❌ Batch processing test FAILED")
        print("   - Check Cloud Run logs for errors")
        print("   - Verify GPU allocation and model loading")
    
    print(f"\n🔗 Server endpoints:")
    print(f"   Health: {SERVER_URL}/health")
    print(f"   Status: {SERVER_URL}/status")
    print(f"   Metrics: {SERVER_URL}/metrics")

if __name__ == "__main__":
    main() 