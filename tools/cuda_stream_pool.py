#!/usr/bin/env python3
"""
🚨 DEPRECATED: CUDA Stream Pool for BiRefNet Models
NOTE: This file is deprecated - the application now uses BiSeNet v1 with CPU optimization

CUDA Stream Pool for High-Performance BiRefNet Inference
Optimizes GPU memory usage and parallel processing for mannequin segmentation
"""

import asyncio
import threading
import time
import torch
from typing import List, Optional, Any

# Handle both relative and absolute imports for local testing
try:
    from .BirefNet import BiRefNetSegmenter
    from .logger import AppLogger
except ImportError:
    # Fallback for local testing
    from BirefNet import BiRefNetSegmenter
    from logger import AppLogger

class CUDAStreamPool:
    """
    Production CUDA Stream Pool
    - Single BiRefNet model (98% memory savings vs 60 models)
    - Multiple CUDA streams for parallel inference
    - Async lock-free access for maximum throughput
    """
    
    def __init__(self, 
                 model_path: str = "models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt",
                 model_name: str = "zhengpeng7/BiRefNet_lite",
                 num_streams: int = 60,
                 precision: str = "fp16",
                 vis_save_dir: str = "infer"):
        
        self.model_path = model_path
        self.model_name = model_name
        self.num_streams = num_streams
        self.precision = precision
        self.vis_save_dir = vis_save_dir
        
        self.model = None
        self.device = None
        self.streams = []
        self.stream_semaphore = None
        
        # Performance tracking
        self.active_requests = 0
        self.total_requests = 0
        self.max_active = 0
        self.request_lock = threading.Lock()
        
        # Logger
        self.logger = AppLogger()
        
        self.logger.log(f"🌊 Initializing CUDA Stream Pool: {num_streams} streams")
        print(f"🌊 Initializing CUDA Stream Pool: {num_streams} streams")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize single model and CUDA streams"""
        try:
            self.logger.log("⏳ Loading single BiRefNet model for stream pool...")
            print("⏳ Loading single BiRefNet model for stream pool...")
            
            start_time = time.time()
            
            # Initialize single shared model
            self.model = BiRefNetSegmenter(
                model_path=self.model_path,
                model_name=self.model_name,
                precision=self.precision,
                vis_save_dir=self.vis_save_dir,
                thickness_threshold=200,
                mask_threshold=0.5
            )
            
            self.device = self.model.device
            load_time = time.time() - start_time
            
            self.logger.log(f"✅ Model loaded in {load_time:.2f}s on {self.device}")
            print(f"✅ Model loaded in {load_time:.2f}s on {self.device}")
            
            # Initialize CUDA streams for parallel processing
            if self.device.type == 'cuda':
                self.logger.log(f"🌊 Creating {self.num_streams} CUDA streams...")
                print(f"🌊 Creating {self.num_streams} CUDA streams...")
                
                self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
                self.stream_semaphore = asyncio.Semaphore(self.num_streams)
                
                self.logger.log(f"✅ {len(self.streams)} CUDA streams ready")
                print(f"✅ {len(self.streams)} CUDA streams ready")
                
                # Log GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    self.logger.log(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            else:
                self.logger.log("⚠️ CPU mode - using async threading fallback")
                print("⚠️ CPU mode - using async threading fallback")
                self.streams = [None] * self.num_streams
                self.stream_semaphore = asyncio.Semaphore(self.num_streams)
            
            self.logger.log("✅ CUDA Stream Pool initialization complete")
            print("✅ CUDA Stream Pool initialization complete")
            
        except Exception as e:
            self.logger.log(f"❌ Failed to initialize CUDA Stream Pool: {e}")
            print(f"❌ Failed to initialize CUDA Stream Pool: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    async def _acquire_stream(self):
        """Acquire an available CUDA stream (async, lock-free)"""
        await self.stream_semaphore.acquire()
        
        # Round-robin stream selection
        with self.request_lock:
            stream_idx = self.total_requests % self.num_streams
            self.total_requests += 1
            self.active_requests += 1
            self.max_active = max(self.max_active, self.active_requests)
        
        return stream_idx, self.streams[stream_idx]
    
    def _release_stream(self):
        """Release stream (lock-free)"""
        with self.request_lock:
            self.active_requests -= 1
        self.stream_semaphore.release()
    
    def _process_with_stream(self, image_url: str, stream_idx: int, stream, **kwargs) -> Optional[Any]:
        """Process image using specific CUDA stream"""
        try:
            if self.device.type == 'cuda' and stream is not None:
                # Use CUDA stream for parallel GPU processing
                with torch.cuda.stream(stream):
                    torch.cuda.set_stream(stream)
                    
                    # Process on this specific stream
                    result = self.model.process_image_url(image_url, plot=False, **kwargs)
                    
                    # Synchronize stream to ensure completion
                    stream.synchronize()
                    
                    return result
            else:
                # CPU fallback
                result = self.model.process_image_url(image_url, plot=False, **kwargs)
                return result
                
        except Exception as e:
            self.logger.log(f"❌ Stream {stream_idx} error: {e}")
            print(f"❌ Stream {stream_idx} error processing {image_url}: {e}")
            return None
    
    async def process_image_async(self, image_url: str, **kwargs) -> Optional[Any]:
        """Async image processing with automatic stream management"""
        if self.model is None:
            return None
        
        # Acquire stream asynchronously
        stream_idx, stream = await self._acquire_stream()
        
        try:
            # Process in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._process_with_stream,
                image_url,
                stream_idx,
                stream,
                **kwargs
            )
            return result
            
        finally:
            # Always release stream
            self._release_stream()
    
    def process_image_sync(self, image_url: str, **kwargs) -> Optional[Any]:
        """Synchronous wrapper for compatibility"""
        if self.model is None:
            return None
        
        # For sync calls, use the model directly (fallback mode)
        try:
            return self.model.process_image_url(image_url, plot=False, **kwargs)
        except Exception as e:
            self.logger.log(f"❌ Sync processing error: {e}")
            return None
    
    async def process_batch_async(self, image_urls: List[str], **kwargs) -> List[Optional[Any]]:
        """Process multiple images concurrently using async streams"""
        if not image_urls or self.model is None:
            return []
        
        self.logger.log(f"🌊 Processing {len(image_urls)} images with {self.num_streams} streams")
        print(f"🌊 Processing {len(image_urls)} images with {self.num_streams} streams")
        
        start_time = time.time()
        
        # Create async tasks for all images
        tasks = [self.process_image_async(url, **kwargs) for url in image_urls]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.log(f"❌ Task {i+1} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        duration = time.time() - start_time
        successful = len([r for r in processed_results if r is not None])
        throughput = successful / duration if duration > 0 else 0
        
        self.logger.log(f"✅ Batch complete: {successful}/{len(image_urls)} in {duration:.2f}s ({throughput:.2f} img/s)")
        print(f"✅ Batch complete: {successful}/{len(image_urls)} in {duration:.2f}s ({throughput:.2f} img/s)")
        
        return processed_results
    
    def get_stats(self):
        """Get current pool statistics"""
        with self.request_lock:
            stats = {
                "pool_type": "CUDAStreamPool",
                "model_loaded": self.model is not None,
                "device": str(self.device) if self.device else "None",
                "num_streams": self.num_streams,
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "max_active_requests": self.max_active,
                "cuda_available": torch.cuda.is_available(),
                "streams_initialized": len(self.streams) > 0
            }
            
            # Add GPU memory stats if available
            if self.device and self.device.type == 'cuda':
                try:
                    stats["gpu_memory"] = {
                        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    }
                except:
                    pass
            
            return stats

# Global CUDA Stream Pool instance
_global_cuda_stream_pool = None
_cuda_pool_lock = threading.Lock()

def get_global_cuda_stream_pool() -> CUDAStreamPool:
    """Get or create global CUDA stream pool instance"""
    global _global_cuda_stream_pool
    
    with _cuda_pool_lock:
        if _global_cuda_stream_pool is None:
            _global_cuda_stream_pool = CUDAStreamPool(
                model_path="models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt",
                model_name="zhengpeng7/BiRefNet_lite",
                num_streams=60,  # 60 CUDA streams for maximum parallelism
                precision="fp16",
                vis_save_dir="infer"
            )
        
        return _global_cuda_stream_pool 