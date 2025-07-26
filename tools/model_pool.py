#!/usr/bin/env python3
"""
BiRefNet Model Pool Manager
Manages multiple BiRefNet model instances for parallel processing
"""

import threading
import queue
import time
import gc
import torch
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from .BirefNet import BiRefNetSegmenter

try:
    from .logger import AppLogger
except ImportError:
    try:
        from tools.logger import AppLogger
    except ImportError:
        from logger import AppLogger


class BiRefNetModelPool:
    """
    Thread-safe pool of BiRefNet models for parallel inference processing.
    
    Maintains multiple model instances and distributes requests across them
    to achieve higher throughput and better GPU utilization.
    """
    
    def __init__(self, pool_size: int = 10, **model_kwargs):
        """
        Initialize the model pool.
        
        Args:
            pool_size: Number of BiRefNet model instances to create
            **model_kwargs: Arguments passed to BiRefNetSegmenter constructor
        """
        self.pool_size = pool_size
        self.model_kwargs = model_kwargs
        self.logger = AppLogger()
        
        # Thread-safe model queue
        self._model_queue = queue.Queue(maxsize=pool_size)
        self._pool_lock = threading.Lock()
        self._initialized = False
        self._models = []
        
        # Performance tracking
        self._request_count = 0
        self._active_requests = 0
        self._max_active = 0
        
        self.logger.log(f"🏊‍♂️ Initializing BiRefNet Model Pool with {pool_size} instances...")
        print(f"🏊‍♂️ Initializing BiRefNet Model Pool with {pool_size} instances...")
    
    def initialize_pool(self):
        """Initialize all model instances in the pool."""
        if self._initialized:
            return
            
        with self._pool_lock:
            if self._initialized:  # Double-check pattern
                return
                
            self.logger.log("🚀 Creating model instances...")
            print("🚀 Creating model instances...")
            
            # Create model instances
            for i in range(self.pool_size):
                try:
                    self.logger.log(f"📦 Loading model {i+1}/{self.pool_size}...")
                    print(f"📦 Loading model {i+1}/{self.pool_size}...")
                    
                    model = BiRefNetSegmenter(**self.model_kwargs)
                    self._models.append(model)
                    self._model_queue.put(model)
                    
                    # Log GPU memory after each model
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        self.logger.log(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                        
                except Exception as e:
                    self.logger.log(f"❌ Failed to create model {i+1}: {e}")
                    print(f"❌ Failed to create model {i+1}: {e}")
                    # Continue with fewer models if some fail
                    break
            
            self._initialized = True
            actual_size = len(self._models)
            self.logger.log(f"✅ Model pool initialized with {actual_size}/{self.pool_size} models")
            print(f"✅ Model pool initialized with {actual_size}/{self.pool_size} models")
            
            # Final memory report
            if torch.cuda.is_available():
                total_allocated = torch.cuda.memory_allocated() / 1024**3
                total_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.log(f"🔥 TOTAL GPU Memory: {total_allocated:.2f}GB allocated, {total_reserved:.2f}GB reserved")
                print(f"🔥 TOTAL GPU Memory: {total_allocated:.2f}GB allocated, {total_reserved:.2f}GB reserved")
    
    @contextmanager
    def get_model(self, timeout: float = 300.0):
        """
        Get a model from the pool with automatic return.
        
        Args:
            timeout: Maximum time to wait for an available model
            
        Yields:
            BiRefNetSegmenter: Available model instance
        """
        if not self._initialized:
            self.initialize_pool()
        
        model = None
        start_time = time.time()
        
        try:
            # Get model from queue
            model = self._model_queue.get(timeout=timeout)
            
            # Track active requests
            with self._pool_lock:
                self._active_requests += 1
                self._max_active = max(self._max_active, self._active_requests)
                self._request_count += 1
            
            wait_time = time.time() - start_time
            if wait_time > 1.0:  # Log if we waited more than 1 second
                self.logger.log(f"⏱️ Waited {wait_time:.2f}s for available model")
                print(f"⏱️ Waited {wait_time:.2f}s for available model")
            
            yield model
            
        except queue.Empty:
            self.logger.log(f"❌ No model available within {timeout}s timeout")
            print(f"❌ No model available within {timeout}s timeout")
            raise TimeoutError(f"No model available within {timeout}s")
            
        finally:
            # Return model to queue
            if model is not None:
                self._model_queue.put(model)
                
                # Update active requests counter
                with self._pool_lock:
                    self._active_requests -= 1
    
    def process_single_request(self, image_url: str, **kwargs) -> Any:
        """
        Process a single image URL using an available model.
        
        Args:
            image_url: URL of the image to process
            **kwargs: Additional arguments for processing
            
        Returns:
            Processing result
        """
        with self.get_model() as model:
            return model.process_image_url(image_url, **kwargs)
    
    def process_batch_requests(self, image_urls: List[str], **kwargs) -> List[Any]:
        """
        Process multiple image URLs in parallel using the model pool.
        
        Args:
            image_urls: List of image URLs to process
            **kwargs: Additional arguments for processing
            
        Returns:
            List of processing results
        """
        if not image_urls:
            return []
        
        # ✅ ALWAYS USE PARALLEL PROCESSING - NO MORE SEQUENTIAL FALLBACK!
        # Even for 1-2 images, use ThreadPoolExecutor for consistency
        results = [None] * len(image_urls)
        max_workers = min(len(image_urls), self.pool_size)  # Use ALL available models
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_single_request, url, **kwargs): i 
                for i, url in enumerate(image_urls)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.log(f"❌ Error processing image {index}: {e}")
                    print(f"❌ Error processing image {index}: {e}")
                    results[index] = None
        
        return results
    
    def get_stats(self) -> dict:
        """Get pool performance statistics."""
        with self._pool_lock:
            return {
                "pool_size": len(self._models),
                "available_models": self._model_queue.qsize(),
                "active_requests": self._active_requests,
                "max_active_requests": self._max_active,
                "total_requests": self._request_count,
                "initialized": self._initialized
            }
    
    def cleanup(self):
        """Clean up model pool resources."""
        with self._pool_lock:
            self.logger.log("🧹 Cleaning up model pool...")
            print("🧹 Cleaning up model pool...")
            
            # Clear queue
            while not self._model_queue.empty():
                try:
                    self._model_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Clear model references
            self._models.clear()
            
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            self._initialized = False
            self.logger.log("✅ Model pool cleanup completed")
            print("✅ Model pool cleanup completed")

    def scale_pool_temporarily(self, target_size: int = 5):
        """
        Temporarily reduce pool size for memory-intensive operations.
        
        Args:
            target_size: Target number of models to keep active
        """
        if not self._initialized or target_size >= len(self._models):
            return
        
        with self._pool_lock:
            current_size = len(self._models)
            models_to_remove = current_size - target_size
            
            self.logger.log(f"🔄 Scaling down pool: {current_size} → {target_size} models")
            print(f"🔄 Scaling down pool: {current_size} → {target_size} models")
            
            # Remove models from the end of the list (they should be in queue)
            models_to_delete = self._models[target_size:]
            self._models = self._models[:target_size]
            
            # Clear the queue and rebuild with remaining models
            # Clear queue
            while not self._model_queue.empty():
                try:
                    self._model_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Re-add remaining models to queue
            for model in self._models:
                self._model_queue.put(model)
            
            # Delete removed models to free GPU memory
            for model in models_to_delete:
                del model
            
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Log memory after cleanup
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.log(f"📊 After scaling: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                print(f"📊 After scaling: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            self.logger.log(f"✅ Pool scaled down: removed {len(models_to_delete)} models")
            print(f"✅ Pool scaled down: removed {len(models_to_delete)} models")
    
    def restore_pool_size(self):
        """Restore pool to original size after temporary scaling."""
        if not self._initialized:
            return
            
        original_size = self.pool_size
        current_size = len(self._models)
        
        if current_size >= original_size:
            return
        
        with self._pool_lock:
            models_to_add = original_size - current_size
            
            self.logger.log(f"🔄 Restoring pool: {current_size} → {original_size} models")
            print(f"🔄 Restoring pool: {current_size} → {original_size} models")
            
            # Add models back
            for i in range(models_to_add):
                try:
                    model = BiRefNetSegmenter(**self.model_kwargs)
                    self._models.append(model)
                    self._model_queue.put(model)
                    
                    self.logger.log(f"📦 Restored model {current_size + i + 1}/{original_size}")
                    print(f"📦 Restored model {current_size + i + 1}/{original_size}")
                    
                except Exception as e:
                    self.logger.log(f"❌ Failed to restore model {current_size + i + 1}: {e}")
                    print(f"❌ Failed to restore model {current_size + i + 1}: {e}")
                    break
            
            self.logger.log(f"✅ Pool restored to {len(self._models)} models")
            print(f"✅ Pool restored to {len(self._models)} models")


# Global model pool instance
_global_model_pool: Optional[BiRefNetModelPool] = None
_pool_lock = threading.Lock()


def get_global_model_pool() -> BiRefNetModelPool:
    """Get or create the global model pool instance."""
    global _global_model_pool
    
    if _global_model_pool is None:
        with _pool_lock:
            if _global_model_pool is None:  # Double-check pattern
                _global_model_pool = BiRefNetModelPool(
                    pool_size=60,  # 🚀 INCREASED: 30 → 60 models for maximum parallelism
                    model_path="models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt", # NEW BIREFNET_LITE MODEL FROM S3
                    model_name="zhengpeng7/BiRefNet_lite", # Use BiRefNet_lite
                    precision="fp16",
                    vis_save_dir="infer",
                    thickness_threshold=200,
                    mask_threshold=0.5
                )
                _global_model_pool.initialize_pool()
    
    return _global_model_pool


def cleanup_global_pool():
    """Clean up the global model pool."""
    global _global_model_pool
    if _global_model_pool is not None:
        _global_model_pool.cleanup()
        _global_model_pool = None 