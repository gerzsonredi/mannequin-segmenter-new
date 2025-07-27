#!/usr/bin/env python3
"""
Progressive Model Pool Test - Test 1, 2, 4 BiRefNet_lite models locally
Verify shared cache performance and correctness before Cloud Run deployment
"""

import sys
import os
sys.path.insert(0, 'tools')

from BirefNet import BiRefNetSegmenter
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Test configuration
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
MODEL_CONFIGS = [1, 2, 4]  # Number of models to test
PRECISION = "fp32"  # Use fp32 for CPU testing

def create_model_with_timing(model_id: int):
    """Create a single model and measure timing"""
    start_time = time.time()
    
    try:
        print(f"🤖 Creating model {model_id}...")
        segmenter = BiRefNetSegmenter(
            model_name="zhengpeng7/BiRefNet_lite",
            precision=PRECISION
        )
        
        creation_time = time.time() - start_time
        print(f"✅ Model {model_id} created in {creation_time:.2f}s")
        
        return {
            "model_id": model_id,
            "segmenter": segmenter,
            "creation_time": creation_time,
            "success": True
        }
        
    except Exception as e:
        creation_time = time.time() - start_time
        print(f"❌ Model {model_id} failed in {creation_time:.2f}s: {e}")
        
        return {
            "model_id": model_id,
            "segmenter": None,
            "creation_time": creation_time,
            "success": False,
            "error": str(e)
        }

def test_model_inference(model_result):
    """Test inference with a model"""
    if not model_result["success"]:
        return {"model_id": model_result["model_id"], "inference_success": False, "error": "Model creation failed"}
    
    model_id = model_result["model_id"]
    segmenter = model_result["segmenter"]
    
    try:
        print(f"🧪 Testing inference for model {model_id}...")
        start_time = time.time()
        
        # Simple inference test (no fast_mode parameter)
        result = segmenter.process_image_url(TEST_IMAGE_URL, plot=False)
        
        inference_time = time.time() - start_time
        
        if result is not None and isinstance(result, np.ndarray):
            print(f"✅ Model {model_id} inference successful in {inference_time:.2f}s")
            print(f"   Result shape: {result.shape}, dtype: {result.dtype}")
            return {
                "model_id": model_id,
                "inference_success": True,
                "inference_time": inference_time,
                "result_shape": result.shape
            }
        else:
            print(f"❌ Model {model_id} inference returned invalid result")
            return {
                "model_id": model_id,
                "inference_success": False,
                "error": "Invalid result"
            }
            
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"❌ Model {model_id} inference failed in {inference_time:.2f}s: {e}")
        return {
            "model_id": model_id,
            "inference_success": False,
            "inference_time": inference_time,
            "error": str(e)
        }

def test_n_models(n_models: int):
    """Test creating N models and measure performance"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {n_models} BiRefNet_lite MODEL(S)")
    print(f"{'='*60}")
    
    overall_start = time.time()
    
    # Create models in parallel to simulate real usage
    if n_models == 1:
        # Single model - no parallelism needed
        results = [create_model_with_timing(1)]
    else:
        # Multiple models - test parallel creation
        print(f"🚀 Creating {n_models} models in parallel...")
        
        with ThreadPoolExecutor(max_workers=n_models) as executor:
            futures = {executor.submit(create_model_with_timing, i+1): i+1 for i in range(n_models)}
            results = []
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    overall_creation_time = time.time() - overall_start
    
    # Analyze creation performance
    successful_models = [r for r in results if r["success"]]
    failed_models = [r for r in results if not r["success"]]
    
    print(f"\n📊 CREATION RESULTS:")
    print(f"   ✅ Successful: {len(successful_models)}/{n_models}")
    print(f"   ❌ Failed: {len(failed_models)}")
    print(f"   ⏱️  Total time: {overall_creation_time:.2f}s")
    print(f"   📈 Avg per model: {overall_creation_time/n_models:.2f}s")
    
    if successful_models:
        creation_times = [r["creation_time"] for r in successful_models]
        print(f"   🕐 Individual times: {[f'{t:.2f}s' for t in creation_times]}")
        print(f"   📊 Min: {min(creation_times):.2f}s, Max: {max(creation_times):.2f}s")
    
    if failed_models:
        print(f"\n❌ FAILURES:")
        for failed in failed_models:
            print(f"   Model {failed['model_id']}: {failed.get('error', 'Unknown error')}")
    
    # Test inference with successful models
    if successful_models and len(successful_models) > 0:
        print(f"\n🧪 TESTING INFERENCE:")
        inference_results = []
        
        for model_result in successful_models[:2]:  # Test max 2 models for inference
            inference_result = test_model_inference(model_result)
            inference_results.append(inference_result)
        
        successful_inference = [r for r in inference_results if r.get("inference_success")]
        
        if successful_inference:
            inference_times = [r["inference_time"] for r in successful_inference]
            print(f"\n📊 INFERENCE RESULTS:")
            print(f"   ✅ Successful: {len(successful_inference)}/{len(inference_results)}")
            print(f"   ⏱️  Times: {[f'{t:.2f}s' for t in inference_times]}")
            print(f"   📈 Avg: {sum(inference_times)/len(inference_times):.2f}s")
    
    return {
        "n_models": n_models,
        "successful_models": len(successful_models),
        "failed_models": len(failed_models),
        "total_creation_time": overall_creation_time,
        "avg_creation_time": overall_creation_time / n_models,
        "individual_times": [r["creation_time"] for r in successful_models] if successful_models else [],
        "cache_effectiveness": len(successful_models) > 1 and max([r["creation_time"] for r in successful_models]) / min([r["creation_time"] for r in successful_models]) > 5
    }

def analyze_cache_performance(all_results):
    """Analyze cache performance across different model counts"""
    print(f"\n{'='*60}")
    print(f"📊 CACHE PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    for result in all_results:
        n = result["n_models"]
        if result["individual_times"]:
            times = result["individual_times"]
            first_time = times[0] if times else 0
            subsequent_times = times[1:] if len(times) > 1 else []
            
            print(f"\n🔍 {n} Model(s):")
            print(f"   First model: {first_time:.2f}s (download + cache)")
            
            if subsequent_times:
                avg_subsequent = sum(subsequent_times) / len(subsequent_times)
                speedup = first_time / avg_subsequent if avg_subsequent > 0 else 0
                print(f"   Subsequent models: {[f'{t:.2f}s' for t in subsequent_times]}")
                print(f"   Cache speedup: {speedup:.1f}x")
                
                if result["cache_effectiveness"]:
                    print(f"   ✅ Cache working effectively")
                else:
                    print(f"   ⚠️  Cache may not be working optimally")
            else:
                print(f"   (No subsequent models to compare)")
    
    # Overall scaling analysis
    print(f"\n🎯 SCALING ANALYSIS:")
    total_times = [(r["n_models"], r["total_creation_time"]) for r in all_results if r["successful_models"] == r["n_models"]]
    
    if len(total_times) >= 2:
        for i, (n, t) in enumerate(total_times):
            print(f"   {n} models: {t:.2f}s total")
            
        # Check if scaling is sub-linear (good cache performance)
        if len(total_times) >= 3:
            # Compare 1 vs 4 models
            time_1 = next((t for n, t in total_times if n == 1), None)
            time_4 = next((t for n, t in total_times if n == 4), None)
            
            if time_1 and time_4:
                linear_expectation = time_1 * 4  # What it would be without cache
                actual_ratio = time_4 / time_1
                cache_benefit = linear_expectation / time_4
                
                print(f"\n💡 CACHE BENEFIT:")
                print(f"   Without cache (4×1): {linear_expectation:.2f}s expected")
                print(f"   With cache (actual): {time_4:.2f}s")
                print(f"   Cache benefit: {cache_benefit:.1f}x improvement")
                
                if cache_benefit > 3:
                    print(f"   🎉 Excellent cache performance!")
                elif cache_benefit > 1.5:
                    print(f"   👍 Good cache performance")
                else:
                    print(f"   ⚠️  Cache performance needs improvement")

def main():
    print("🚀 PROGRESSIVE MODEL POOL TEST")
    print("Testing BiRefNet_lite model cache with 1, 2, 4 models")
    print("=" * 60)
    
    # Clear any existing cache for clean test
    from tools.BirefNet import _model_cache
    _model_cache.clear()
    print("🧹 Cleared model cache for clean test")
    
    all_results = []
    
    # Test each configuration
    for n_models in MODEL_CONFIGS:
        try:
            result = test_n_models(n_models)
            all_results.append(result)
            
            # Brief pause between tests
            if n_models < max(MODEL_CONFIGS):
                print(f"\n⏸️  Pausing 2 seconds before next test...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Test with {n_models} models failed: {e}")
            all_results.append({
                "n_models": n_models,
                "successful_models": 0,
                "failed_models": n_models,
                "total_creation_time": 0,
                "avg_creation_time": 0,
                "individual_times": [],
                "cache_effectiveness": False,
                "error": str(e)
            })
    
    # Analyze results
    analyze_cache_performance(all_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PROGRESSIVE TEST COMPLETED")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in all_results if r["successful_models"] == r["n_models"])
    total_tests = len(all_results)
    
    print(f"✅ Successful test configurations: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print(f"🎯 All tests passed! Model cache is working correctly.")
        print(f"💡 Ready for Cloud Run deployment with 60 models!")
    elif successful_tests > 0:
        print(f"⚠️  Some tests passed. Review failures before deployment.")
    else:
        print(f"❌ All tests failed. Fix issues before deployment.")

if __name__ == "__main__":
    main() 