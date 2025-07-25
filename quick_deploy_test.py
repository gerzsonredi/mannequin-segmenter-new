#!/usr/bin/env python3
"""
Quick deployment test - monitors when the fix is deployed and tests inference
"""
import requests
import time
import json
from datetime import datetime

BASE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"

def test_inference():
    """Test single inference to verify the PyTorch crash is fixed"""
    try:
        print("🔍 Testing single inference...")
        response = requests.post(
            f"{BASE_URL}/infer",
            json={
                "image_url": "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80",
                "upload_to_s3": False
            },
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ INFERENCE SUCCESS! Response: {len(str(data))} chars")
            return True
        else:
            print(f"❌ INFERENCE FAILED: {response.status_code}")
            try:
                error_data = response.json()
                if "captures_underway.empty()" in str(error_data):
                    print("🚨 PyTorch CUDA crash still happening!")
                else:
                    print(f"   Error: {error_data}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def check_health():
    """Quick health check"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        if response.status_code == 200:
            print(f"✅ Health check OK")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    print("🚀 MONITORING DEPLOYMENT & TESTING PyTorch CUDA FIX")
    print("=" * 60)
    
    attempt = 1
    max_attempts = 20  # ~10 minutes
    
    while attempt <= max_attempts:
        print(f"\n📡 Attempt {attempt}/{max_attempts} at {datetime.now().strftime('%H:%M:%S')}")
        
        # Check health first
        if check_health():
            # If health OK, test inference
            if test_inference():
                print(f"\n🎉 SUCCESS! PyTorch CUDA crash is FIXED!")
                print("🚀 Ready for batch testing!")
                break
            else:
                print("⚠️  Health OK but inference failed - checking again...")
        else:
            print("⚠️  Service not ready yet...")
        
        if attempt < max_attempts:
            print(f"⏳ Waiting 30s before next attempt...")
            time.sleep(30)
        
        attempt += 1
    
    if attempt > max_attempts:
        print(f"\n❌ TIMEOUT: Service not ready after {max_attempts} attempts")
        print("💡 Check Cloud Run logs or GitHub Actions for deployment status")

if __name__ == "__main__":
    main() 