#!/usr/bin/env python3
"""
Test script for local FastAPI Mannequin Segmenter API
Tests the inference endpoints with sample images
"""

import requests
import json
import time

# API Configuration
API_BASE = "http://localhost:5001"

# Sample image URLs for testing (public domain images)
SAMPLE_IMAGES = [
    "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&h=600&fit=crop",  # Person
    "https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=800&h=600&fit=crop",  # Fashion
    "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=800&h=600&fit=crop",  # Model
]

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        response.raise_for_status()
        health_data = response.json()
        print(f"✅ Health Status: {health_data['status']}")
        print(f"   Service: {health_data['service']} v{health_data['version']}")
        print(f"   GPU Available: {health_data['gpu_available']}")
        print(f"   Model Loaded: {health_data['model_loaded']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_status():
    """Test the status endpoint"""
    print("\n🔍 Testing status endpoint...")
    try:
        response = requests.get(f"{API_BASE}/status", timeout=10)
        response.raise_for_status()
        status_data = response.json()
        print(f"✅ Load Level: {status_data['recommendation']['load_level']}")
        print(f"   Queue Size: {status_data['performance']['queue_size']}")
        print(f"   Can Accept Requests: {status_data['recommendation']['can_accept_requests']}")
        return True
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

def test_inference(image_url, timeout=60):
    """Test single image inference"""
    print(f"\n🔍 Testing inference with image: {image_url[:50]}...")
    try:
        payload = {"image_url": image_url}
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/infer", 
            json=payload, 
            timeout=timeout
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference successful in {duration:.2f}s")
            print(f"   Result URL: {result.get('visualization_url', 'N/A')[:60]}...")
            return True
        else:
            print(f"❌ Inference failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Inference timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False

def test_batch_inference(image_urls, timeout=120):
    """Test batch image inference"""
    print(f"\n🔍 Testing batch inference with {len(image_urls)} images...")
    try:
        payload = {"image_urls": image_urls}
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/batch_infer", 
            json=payload, 
            timeout=timeout
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch inference successful in {duration:.2f}s")
            print(f"   Batch Size: {result.get('batch_size', 0)}")
            print(f"   Successful: {result.get('successful_count', 0)}")
            print(f"   Failed: {result.get('failed_count', 0)}")
            return True
        else:
            print(f"❌ Batch inference failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Batch inference timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Batch inference error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Local FastAPI Mannequin Segmenter Tests")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("❌ Health check failed, aborting tests")
        return
    
    # Test status
    test_status()
    
    # Test single inference
    print("\n" + "=" * 60)
    print("🧪 SINGLE INFERENCE TESTS")
    print("=" * 60)
    
    success_count = 0
    for i, image_url in enumerate(SAMPLE_IMAGES[:2], 1):  # Test first 2 images
        print(f"\n--- Test {i}/2 ---")
        if test_inference(image_url):
            success_count += 1
    
    print(f"\n📊 Single Inference Results: {success_count}/2 successful")
    
    # Test batch inference
    print("\n" + "=" * 60)
    print("🧪 BATCH INFERENCE TEST")
    print("=" * 60)
    
    test_batch_inference(SAMPLE_IMAGES[:2])  # Test batch with 2 images
    
    print("\n🎉 Testing completed!")
    print("\n💡 API Endpoints:")
    print(f"   Health: {API_BASE}/health")
    print(f"   Status: {API_BASE}/status")
    print(f"   Metrics: {API_BASE}/metrics")
    print(f"   Inference: {API_BASE}/infer")
    print(f"   Batch: {API_BASE}/batch_infer")

if __name__ == "__main__":
    main() 