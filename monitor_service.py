#!/usr/bin/env python3
"""
Service monitoring script - checks when the service is back online after deployment
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "https://mannequin-segmenter-o4c5wdhnoa-ez.a.run.app"

def check_health():
    """Check if service is responding"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.status_code
    except Exception as e:
        return False, str(e)

def check_status():
    """Check service status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get('request_limiter', {}).get('active_requests', 'unknown')
        return False, response.status_code
    except Exception as e:
        return False, str(e)

def test_infer():
    """Test inference endpoint"""
    try:
        payload = {
            "image_url": "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
        }
        response = requests.post(f"{BASE_URL}/infer", json=payload, timeout=30)
        if response.status_code == 200:
            return True, "SUCCESS"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 MONITORING SERVICE DEPLOYMENT")
    print("=" * 50)
    print(f"Target: {BASE_URL}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    attempt = 0
    while attempt < 60:  # Max 10 minutes (60 * 10s)
        attempt += 1
        now = datetime.now().strftime('%H:%M:%S')
        
        # Check health
        health_ok, health_result = check_health()
        health_status = "✅" if health_ok else "❌"
        
        # Check status  
        status_ok, status_result = check_status()
        status_status = "✅" if status_ok else "❌"
        
        print(f"[{now}] #{attempt:2d} | Health: {health_status} | Status: {status_status} | Active: {status_result if status_ok else 'N/A'}")
        
        if health_ok and status_ok:
            print()
            print("🎉 SERVICE IS BACK ONLINE! Testing inference...")
            
            # Test inference
            infer_ok, infer_result = test_infer()
            if infer_ok:
                print("✅ INFERENCE TEST PASSED!")
                print("🚀 Service is ready for batch testing!")
                return True
            else:
                print(f"❌ Inference test failed: {infer_result}")
                print("⏳ Service online but inference still has issues...")
        
        time.sleep(10)  # Wait 10 seconds
    
    print("\n⏰ Timeout reached. Service may need manual intervention.")
    return False

if __name__ == "__main__":
    main() 