#!/usr/bin/env python3
"""
Inference Throughput Test - Test parallel inference speed with 1, 2, 4 models
Measure how much faster we can process images with multiple models working in parallel
"""

import sys
import os
sys.path.insert(0, 'tools')

from BirefNet import BiRefNetSegmenter
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Test configuration
TEST_IMAGE_URL = "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
MODEL_CONFIGS = [1, 2, 4]  # Number of models to test
PRECISION = "fp32"  # Use fp32 for CPU testing
IMAGES_PER_TEST = 8  # Process 8 images with each configuration

def create_models(n_models: int):
    """Create N models quickly using cache"""
    print(f"🏗️ Creating {n_models} model(s)...")
    start_time = time.time()
    
    models = []
    for i in range(n_models):
        segmenter = BiRefNetSegmenter(
            model_name="zhengpeng7/BiRefNet_lite",
            precision=PRECISION
        )
        models.append(segmenter)
        print(f"   ✅ Model {i+1} ready")
    
    creation_time = time.time() - start_time
    print(f"🎯 {n_models} models created in {creation_time:.2f}s")
    return models

def single_inference_task(model_id: int, model: BiRefNetSegmenter, image_url: str):
    """Perform a single inference task"""
    start_time = time.time()
    
    try:
        result = model.process_image_url(image_url, plot=False)
        inference_time = time.time() - start_time
        
        if result is not None and isinstance(result, np.ndarray):
            return {
                "model_id": model_id,
                "success": True,
                "inference_time": inference_time,
                "result_shape": result.shape
            }
        else:
            return {
                "model_id": model_id,
                "success": False,
                "inference_time": inference_time,
                "error": "Invalid result"
            }
            
    except Exception as e:
        inference_time = time.time() - start_time
        return {
            "model_id": model_id,
            "success": False,
            "inference_time": inference_time,
            "error": str(e)
        }

def test_parallel_inference(models: list, n_images: int = IMAGES_PER_TEST):
    """Test parallel inference with multiple models"""
    n_models = len(models)
    print(f"\n🚀 TESTING PARALLEL INFERENCE:")
    print(f"   Models: {n_models}")
    print(f"   Images: {n_images}")
    print(f"   Strategy: Round-robin assignment")
    
    # Assign images to models in round-robin fashion
    tasks = []
    for i in range(n_images):
        model_idx = i % n_models
        model = models[model_idx]
        tasks.append((model_idx + 1, model, TEST_IMAGE_URL))
    
    print(f"   📋 Task distribution: {[f'Model {task[0]}' for task in tasks]}")
    
    # Execute all inference tasks in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=n_models) as executor:
        # Submit all tasks
        futures = {
            executor.submit(single_inference_task, task[0], task[1], task[2]): i 
            for i, task in enumerate(tasks)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                print(f"   ✅ Model {result['model_id']}: {result['inference_time']:.2f}s")
            else:
                print(f"   ❌ Model {result['model_id']}: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        inference_times = [r["inference_time"] for r in successful_results]
        avg_inference_time = sum(inference_times) / len(inference_times)
        throughput = len(successful_results) / total_time
        
        print(f"\n📊 PARALLEL INFERENCE RESULTS:")
        print(f"   ✅ Successful: {len(successful_results)}/{n_images}")
        print(f"   ⏱️  Total wall clock: {total_time:.2f}s")
        print(f"   📈 Throughput: {throughput:.2f} images/second")
        print(f"   🕐 Avg inference time: {avg_inference_time:.2f}s per image")
        print(f"   📊 Individual times: {[f'{t:.2f}s' for t in sorted(inference_times)]}")
        
        return {
            "n_models": n_models,
            "n_images": n_images,
            "successful_images": len(successful_results),
            "failed_images": len(failed_results),
            "total_time": total_time,
            "throughput": throughput,
            "avg_inference_time": avg_inference_time,
            "individual_times": inference_times
        }
    else:
        print(f"\n❌ ALL INFERENCE FAILED!")
        return {
            "n_models": n_models,
            "n_images": n_images,
            "successful_images": 0,
            "failed_images": len(failed_results),
            "total_time": total_time,
            "throughput": 0,
            "avg_inference_time": 0,
            "individual_times": []
        }

def test_sequential_inference(model: BiRefNetSegmenter, n_images: int = IMAGES_PER_TEST):
    """Test sequential inference with a single model for comparison"""
    print(f"\n🐌 TESTING SEQUENTIAL INFERENCE:")
    print(f"   Models: 1")
    print(f"   Images: {n_images}")
    print(f"   Strategy: One by one")
    
    start_time = time.time()
    results = []
    
    for i in range(n_images):
        print(f"   🔄 Processing image {i+1}/{n_images}...")
        result = single_inference_task(1, model, TEST_IMAGE_URL)
        results.append(result)
        
        if result["success"]:
            print(f"   ✅ Image {i+1}: {result['inference_time']:.2f}s")
        else:
            print(f"   ❌ Image {i+1}: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        inference_times = [r["inference_time"] for r in successful_results]
        avg_inference_time = sum(inference_times) / len(inference_times)
        throughput = len(successful_results) / total_time
        
        print(f"\n📊 SEQUENTIAL INFERENCE RESULTS:")
        print(f"   ✅ Successful: {len(successful_results)}/{n_images}")
        print(f"   ⏱️  Total wall clock: {total_time:.2f}s")
        print(f"   📈 Throughput: {throughput:.2f} images/second")
        print(f"   🕐 Avg inference time: {avg_inference_time:.2f}s per image")
        
        return {
            "n_models": 1,
            "n_images": n_images,
            "successful_images": len(successful_results),
            "total_time": total_time,
            "throughput": throughput,
            "avg_inference_time": avg_inference_time,
            "individual_times": inference_times
        }
    else:
        print(f"\n❌ ALL SEQUENTIAL INFERENCE FAILED!")
        return {
            "n_models": 1,
            "n_images": n_images,
            "successful_images": 0,
            "total_time": total_time,
            "throughput": 0,
            "avg_inference_time": 0,
            "individual_times": []
        }

def analyze_throughput_scaling(all_results):
    """Analyze how throughput scales with number of models"""
    print(f"\n{'='*60}")
    print(f"📈 THROUGHPUT SCALING ANALYSIS")
    print(f"{'='*60}")
    
    # Find baseline (1 model) performance
    baseline = next((r for r in all_results if r["n_models"] == 1), None)
    
    if not baseline or baseline["throughput"] == 0:
        print("❌ No valid baseline found")
        return
    
    baseline_throughput = baseline["throughput"]
    
    print(f"📊 SCALING RESULTS:")
    print(f"   Baseline (1 model): {baseline_throughput:.2f} images/sec")
    
    for result in all_results:
        if result["n_models"] == 1:
            continue
            
        n_models = result["n_models"]
        throughput = result["throughput"]
        speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0
        efficiency = speedup / n_models * 100  # Percentage of ideal scaling
        
        print(f"   {n_models} models: {throughput:.2f} images/sec ({speedup:.1f}x speedup, {efficiency:.1f}% efficiency)")
    
    # Theoretical vs actual comparison
    print(f"\n🎯 PARALLELIZATION EFFICIENCY:")
    
    for result in all_results:
        if result["n_models"] == 1:
            continue
            
        n_models = result["n_models"]
        actual_throughput = result["throughput"]
        theoretical_max = baseline_throughput * n_models
        efficiency = (actual_throughput / theoretical_max) * 100 if theoretical_max > 0 else 0
        
        if efficiency > 80:
            status = "🎉 Excellent"
        elif efficiency > 60:
            status = "👍 Good"
        elif efficiency > 40:
            status = "⚠️  Fair"
        else:
            status = "❌ Poor"
        
        print(f"   {n_models} models: {efficiency:.1f}% efficient {status}")
    
    # Projection to 60 models
    if len(all_results) >= 2:
        # Use highest efficiency model count for projection
        best_result = max((r for r in all_results if r["n_models"] > 1), 
                         key=lambda x: x["throughput"] / (baseline_throughput * x["n_models"]) if baseline_throughput > 0 else 0)
        
        if best_result["n_models"] > 1:
            efficiency = (best_result["throughput"] / (baseline_throughput * best_result["n_models"])) * 100
            projected_60_throughput = baseline_throughput * 60 * (efficiency / 100)
            projected_50_images_time = 50 / projected_60_throughput if projected_60_throughput > 0 else float('inf')
            
            print(f"\n🚀 PROJECTION TO 60 MODELS:")
            print(f"   Best efficiency: {efficiency:.1f}% ({best_result['n_models']} models)")
            print(f"   Projected 60-model throughput: {projected_60_throughput:.1f} images/sec")
            print(f"   Projected 50 images time: {projected_50_images_time:.1f}s")
            
            if projected_50_images_time <= 6:
                print(f"   🎯 TARGET ACHIEVABLE: 50 images in 6s!")
            else:
                improvement_needed = projected_50_images_time / 6
                print(f"   🔄 Need {improvement_needed:.1f}x more improvement for 6s target")

def main():
    print("🚀 INFERENCE THROUGHPUT TEST")
    print("Testing parallel inference performance with 1, 2, 4 models")
    print("=" * 60)
    
    # Clear cache for consistent testing
    from tools.BirefNet import _model_cache
    _model_cache.clear()
    print("🧹 Cleared model cache")
    
    all_results = []
    
    # Test each configuration
    for n_models in MODEL_CONFIGS:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING {n_models} MODEL(S) THROUGHPUT")
        print(f"{'='*60}")
        
        try:
            # Create models
            models = create_models(n_models)
            
            # Test parallel inference
            result = test_parallel_inference(models, IMAGES_PER_TEST)
            all_results.append(result)
            
            # Brief pause between tests
            if n_models < max(MODEL_CONFIGS):
                print(f"\n⏸️  Pausing 3 seconds before next test...")
                time.sleep(3)
                
        except Exception as e:
            print(f"❌ Test with {n_models} models failed: {e}")
            all_results.append({
                "n_models": n_models,
                "n_images": IMAGES_PER_TEST,
                "successful_images": 0,
                "total_time": 0,
                "throughput": 0,
                "avg_inference_time": 0,
                "error": str(e)
            })
    
    # Analyze throughput scaling
    analyze_throughput_scaling(all_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 THROUGHPUT TEST COMPLETED")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in all_results if r["successful_images"] > 0)
    total_tests = len(all_results)
    
    print(f"✅ Successful test configurations: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print(f"🎯 All tests passed! Parallel inference is working correctly.")
        
        best_throughput = max((r["throughput"] for r in all_results if r["throughput"] > 0), default=0)
        print(f"🏆 Best throughput achieved: {best_throughput:.2f} images/second")
    elif successful_tests > 0:
        print(f"⚠️  Some tests passed. Review failures.")
    else:
        print(f"❌ All tests failed. Fix issues before deployment.")

if __name__ == "__main__":
    main() 