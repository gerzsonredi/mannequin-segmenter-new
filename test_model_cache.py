#!/usr/bin/env python3
"""
Test script to verify the shared model cache is working correctly
"""

import sys
import os
sys.path.insert(0, 'tools')

from BirefNet import BiRefNetSegmenter
import time
import torch

def test_model_cache():
    print("🧪 TESTING SHARED MODEL CACHE")
    print("=" * 50)
    
    # Test 1: First model instance (should download)
    print("\n1️⃣ Creating first BiRefNetSegmenter instance...")
    start_time = time.time()
    
    try:
        segmenter1 = BiRefNetSegmenter(
            model_name="zhengpeng7/BiRefNet_lite",
            precision="fp32"  # Use fp32 for CPU testing
        )
        first_load_time = time.time() - start_time
        print(f"✅ First instance created in {first_load_time:.2f}s")
    except Exception as e:
        print(f"❌ First instance failed: {e}")
        return
    
    # Test 2: Second model instance (should use cache)
    print("\n2️⃣ Creating second BiRefNetSegmenter instance...")
    start_time = time.time()
    
    try:
        segmenter2 = BiRefNetSegmenter(
            model_name="zhengpeng7/BiRefNet_lite", 
            precision="fp32"
        )
        second_load_time = time.time() - start_time
        print(f"✅ Second instance created in {second_load_time:.2f}s")
    except Exception as e:
        print(f"❌ Second instance failed: {e}")
        return
    
    # Test 3: Performance comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   First load (download): {first_load_time:.2f}s")
    print(f"   Second load (cache):   {second_load_time:.2f}s")
    
    if second_load_time < first_load_time:
        speedup = first_load_time / second_load_time
        print(f"   🚀 Cache speedup: {speedup:.1f}x faster!")
    else:
        print(f"   ⚠️  Cache not faster (unexpected)")
    
    # Test 4: Verify they are separate instances
    print(f"\n🔍 INSTANCE VERIFICATION:")
    print(f"   Instance 1 ID: {id(segmenter1.model)}")
    print(f"   Instance 2 ID: {id(segmenter2.model)}")
    
    if id(segmenter1.model) != id(segmenter2.model):
        print(f"   ✅ Separate model instances (good!)")
    else:
        print(f"   ❌ Same model instance (sharing problem!)")
    
    # Test 5: Verify they have same weights
    state1 = segmenter1.model.state_dict()
    state2 = segmenter2.model.state_dict()
    
    weights_match = True
    for key in state1.keys():
        if not torch.equal(state1[key], state2[key]):
            weights_match = False
            break
    
    if weights_match:
        print(f"   ✅ Identical model weights (good!)")
    else:
        print(f"   ❌ Different model weights (problem!)")
    
    print(f"\n🎉 Model cache test completed!")

if __name__ == "__main__":
    test_model_cache() 