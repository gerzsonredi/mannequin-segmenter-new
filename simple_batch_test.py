#!/usr/bin/env python3
"""
Simple Batch Processing Test - Avoiding import complexities
Direct test of batch processing logic without complex module structure
"""

import os
import sys
import numpy as np
import time
from PIL import Image
import requests
import io

# Add project root and tools to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tools'))

def download_image(url):
    """Download test image"""
    print(f"📥 Downloading: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image_np = np.array(pil_image)
    return image_np[:, :, ::-1]  # Convert RGB to BGR

def test_imports():
    """Test if we can import the required modules"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU - will use CPU")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_batch_processing_concept():
    """Test the concept of batch processing with mock tensors"""
    try:
        import torch
        
        print("\n🧪 Testing batch processing concept with mock data...")
        
        # Simulate batch processing
        batch_size = 3
        
        # Create mock input batch [batch_size, 3, 512, 512]
        mock_batch = torch.randn(batch_size, 3, 512, 512)
        print(f"📦 Mock batch shape: {mock_batch.shape}")
        
        # Simulate batch inference timing
        start = time.time()
        
        # Mock processing (matrix operations to simulate GPU work)
        processed_batch = torch.sigmoid(mock_batch)  # Simulate inference
        result_masks = torch.mean(processed_batch, dim=1)  # Simulate mask extraction
        
        batch_time = time.time() - start
        
        print(f"✅ Batch processing simulation:")
        print(f"   Input shape: {mock_batch.shape}")
        print(f"   Output shape: {result_masks.shape}")
        print(f"   Batch time: {batch_time:.4f}s")
        print(f"   Time per image: {batch_time/batch_size:.4f}s")
        
        # Compare with sequential processing
        print("\n🔄 Comparing with sequential processing...")
        start = time.time()
        
        sequential_results = []
        for i in range(batch_size):
            single_input = mock_batch[i:i+1]  # [1, 3, 512, 512]
            single_output = torch.sigmoid(single_input)
            single_mask = torch.mean(single_output, dim=1)
            sequential_results.append(single_mask)
        
        sequential_time = time.time() - start
        
        print(f"✅ Sequential processing:")
        print(f"   Sequential time: {sequential_time:.4f}s")
        print(f"   Time per image: {sequential_time/batch_size:.4f}s")
        
        # Calculate speedup
        speedup = sequential_time / batch_time
        print(f"\n🚀 Theoretical speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print("🎉 Batch processing is faster!")
        else:
            print("⚠️  Sequential was faster (unexpected for GPU)")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False

def main():
    print("🚀 Simple Batch Processing Test")
    print("=" * 50)
    
    # Test 1: Check imports
    if not test_imports():
        print("❌ Cannot proceed without required imports")
        return
    
    # Test 2: Test batch processing concept
    if not test_batch_processing_concept():
        print("❌ Batch processing concept test failed")
        return
    
    # Test 3: Try to load real model (if possible)
    print("\n🤖 Testing BiRefNet import...")
    try:
        # Try different import methods
        import_success = False
        
        # Method 1: Direct import
        try:
            from BirefNet import BiRefNetSegmenter
            print("✅ Direct import successful")
            import_success = True
        except Exception as e1:
            print(f"❌ Direct import failed: {e1}")
            
            # Method 2: From tools
            try:
                from tools.BirefNet import BiRefNetSegmenter
                print("✅ Tools import successful")
                import_success = True
            except Exception as e2:
                print(f"❌ Tools import failed: {e2}")
        
        if import_success:
            print("🎯 BiRefNet import successful! You can run the full test.")
            
            # Try to initialize (but don't require it to work)
            try:
                print("🤖 Attempting to initialize BiRefNetSegmenter...")
                segmenter = BiRefNetSegmenter(precision="fp16", thickness_threshold=200)
                print("✅ BiRefNetSegmenter initialized successfully!")
                
                # If we get here, everything works
                print("\n🎉 SUCCESS: Full batch processing should work!")
                print("💡 You can now run: python local_batch_test.py")
                
            except Exception as e:
                print(f"⚠️  Model initialization failed: {e}")
                print("💡 This might be due to missing model files or GPU issues")
                
        else:
            print("❌ Cannot import BiRefNetSegmenter")
            print("💡 Check the import structure in tools/BirefNet.py")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 