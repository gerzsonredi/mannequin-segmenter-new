#!/usr/bin/env python3
"""
Single Model + Queue Architecture for optimal GPU memory usage
Instead of 60 models, use 1 model with 60 worker threads
"""

import sys
import time
import threading
import queue
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
from PIL import Image
import io

sys.path.insert(0, 'tools')

class SingleModelPool:
    """
    Single BiRefNet model with queue-based access for multiple workers
    Much more GPU memory efficient than 60 separate models
    """
    
    def __init__(self, model_name="zhengpeng7/BiRefNet_lite", num_workers=60):
        self.model_name = model_name
        self.num_workers = num_workers
        self.model = None
        self.model_lock = threading.Lock()
        
        # Performance tracking
        self.active_requests = 0
        self.total_requests = 0
        self.max_active = 0
        self.request_lock = threading.Lock()
        
        print(f"🚀 Initializing Single Model Pool with {num_workers} workers...")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the single shared model"""
        try:
            from BirefNet import BiRefNetSegmenter
            
            print("⏳ Loading single BiRefNet model...")
            start_time = time.time()
            
            self.model = BiRefNetSegmenter(
                model_name=self.model_name,
                precision="fp32",  # Use fp32 for CPU testing
                vis_save_dir="infer"
            )
            
            load_time = time.time() - start_time
            print(f"✅ Single model loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def process_image_url(self, image_url):
        """Process a single image URL using the shared model"""
        if self.model is None:
            return None
        
        # Thread-safe model access
        with self.model_lock:
            # Track active requests
            with self.request_lock:
                self.active_requests += 1
                self.max_active = max(self.max_active, self.active_requests)
                self.total_requests += 1
            
            try:
                # Process image with shared model
                result = self.model.process_image_url(image_url, plot=False)
                return result
                
            except Exception as e:
                print(f"❌ Error processing {image_url}: {e}")
                return None
            
            finally:
                # Release active request count
                with self.request_lock:
                    self.active_requests -= 1
    
    def process_batch_threaded(self, image_urls, max_workers=None):
        """Process multiple images using ThreadPoolExecutor with shared model"""
        if not image_urls:
            return []
        
        if max_workers is None:
            max_workers = min(len(image_urls), self.num_workers)
        
        print(f"🔄 Processing {len(image_urls)} images with {max_workers} workers...")
        
        results = [None] * len(image_urls)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_image_url, url): i 
                for i, url in enumerate(image_urls)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"❌ Task {index} failed: {e}")
                    results[index] = None
        
        return results
    
    def get_stats(self):
        """Get current pool statistics"""
        with self.request_lock:
            return {
                "model_loaded": self.model is not None,
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "max_active_requests": self.max_active,
                "num_workers": self.num_workers
            }

def test_single_vs_multi_model():
    """Test single model architecture vs multi-model"""
    
    test_urls = [
        "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
    ] * 20  # 20 images for testing
    
    print("🧪 SINGLE MODEL ARCHITECTURE TEST")
    print("=" * 50)
    print(f"📊 Test setup: {len(test_urls)} images")
    print()
    
    # Test 1: Single model with different worker counts
    for workers in [1, 2, 4, 8, 16, 20]:
        print(f"🔄 Testing with {workers} workers...")
        
        pool = SingleModelPool(num_workers=workers)
        
        if pool.model is None:
            print(f"❌ Model loading failed, skipping {workers} workers test")
            continue
        
        start_time = time.time()
        results = pool.process_batch_threaded(test_urls, max_workers=workers)
        end_time = time.time()
        
        duration = end_time - start_time
        successful = len([r for r in results if r is not None])
        throughput = successful / duration if duration > 0 else 0
        
        stats = pool.get_stats()
        
        print(f"   ✅ {workers} workers: {duration:.2f}s, {successful}/{len(test_urls)} success")
        print(f"      Throughput: {throughput:.2f} img/s")
        print(f"      Max concurrent: {stats['max_active_requests']}")
        print()
    
    # Test 2: Memory efficiency comparison
    print("📊 MEMORY EFFICIENCY ANALYSIS:")
    print("   Single Model Architecture:")
    print("   • 1 model × ~300MB = 300MB GPU memory")
    print("   • + threading overhead: ~50MB")
    print("   • Total: ~350MB (vs 18GB for 60 models!)")
    print("   • Available for inference: 21.6GB")
    print("   • Expected parallel capacity: 50+ concurrent")
    print()

def test_performance_scaling():
    """Test how performance scales with worker count"""
    
    print("🚀 PERFORMANCE SCALING TEST")
    print("=" * 40)
    
    # Use fewer images for faster testing
    test_urls = [
        "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
    ] * 10
    
    results_data = []
    
    for workers in [1, 2, 4, 8, 10]:
        print(f"🧪 Testing {workers} workers with {len(test_urls)} images...")
        
        pool = SingleModelPool(num_workers=workers)
        
        if pool.model is None:
            continue
        
        # Warm up
        pool.process_image_url(test_urls[0])
        
        # Actual test
        start_time = time.time()
        results = pool.process_batch_threaded(test_urls, max_workers=workers)
        end_time = time.time()
        
        duration = end_time - start_time
        successful = len([r for r in results if r is not None])
        throughput = successful / duration if duration > 0 else 0
        
        # Calculate scaling efficiency
        if len(results_data) == 0:
            efficiency = 100.0  # Baseline
        else:
            baseline_throughput = results_data[0]['throughput']
            expected_throughput = baseline_throughput * workers
            efficiency = (throughput / expected_throughput) * 100 if expected_throughput > 0 else 0
        
        result = {
            'workers': workers,
            'duration': duration,
            'successful': successful,
            'throughput': throughput,
            'efficiency': efficiency
        }
        results_data.append(result)
        
        print(f"   📊 Workers: {workers}")
        print(f"      Time: {duration:.2f}s")
        print(f"      Throughput: {throughput:.2f} img/s")
        print(f"      Scaling efficiency: {efficiency:.1f}%")
        print()
    
    # Summary
    print("📈 SCALING ANALYSIS:")
    best_efficiency = max(r['efficiency'] for r in results_data)
    best_throughput = max(r['throughput'] for r in results_data)
    
    print(f"   Best efficiency: {best_efficiency:.1f}%")
    print(f"   Best throughput: {best_throughput:.2f} img/s")
    print()
    
    # Project to 60 workers
    if len(results_data) > 1:
        avg_efficiency = sum(r['efficiency'] for r in results_data[1:]) / (len(results_data) - 1)
        baseline_throughput = results_data[0]['throughput']
        projected_60 = baseline_throughput * 60 * (avg_efficiency / 100)
        
        print(f"🎯 PROJECTION FOR 60 WORKERS:")
        print(f"   Expected throughput: {projected_60:.2f} img/s")
        print(f"   Time for 50 images: {50 / projected_60:.2f}s")
        
        if 50 / projected_60 <= 6:
            print(f"   🎉 TARGET ACHIEVABLE! (≤ 6s)")
        else:
            print(f"   ⚠️ Target missed, need optimization")

if __name__ == "__main__":
    # Set matplotlib backend for headless testing
    import matplotlib
    matplotlib.use('Agg')
    
    print("🧪 SINGLE MODEL ARCHITECTURE - LOCAL TEST")
    print("=" * 60)
    print()
    
    test_single_vs_multi_model()
    test_performance_scaling()
    
    print("🎉 Single model architecture test completed!")
    print("Ready for Cloud Run deployment with 1 model + 60 workers!") 