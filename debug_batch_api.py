#!/usr/bin/env python3
"""
Debug script to test batch processing exactly as the API does
"""

import sys
import os
import traceback

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

# Import the same way as API
from tools.BirefNet import BiRefNetSegmenter
from tools.model_pool import BiRefNetModelPool

def main():
    print("🐛 DEBUG: Testing BiRefNet_lite with S3 checkpoint...")
    
    # Test image URLs (working Unsplash URLs)
    image_urls = [
        "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1558769132-cb1aea458c5e?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    ]
    
    try:
        print("🤖 Loading BiRefNet_lite Model Pool...")
        model_pool = BiRefNetModelPool(
            pool_size=2,  # Small pool for debugging
            model_path="models/birefnet_lite_mannequin_segmenter/checkpoint_20250726.pt",  # ✅ NEW BIREFNET_LITE MODEL FROM S3
            model_name="zhengpeng7/BiRefNet_lite",  # ✅ Use BiRefNet_lite
            precision="fp16",
            vis_save_dir="infer",
            thickness_threshold=200,
            mask_threshold=0.5
        )
        print("✅ BiRefNet_lite model pool loaded successfully!")
        
        # Test batch processing 
        print("🔄 Testing batch processing...")
        processed_images = []
        
        try:
            print("🔄 Calling process_batch_requests...")
            processed_images = model_pool.process_batch_requests(
                image_urls, 
                plot=False
            )
            print(f"✅ Batch processing completed: {len([x for x in processed_images if x is not None])}/{len(processed_images)} successful")
            
        except Exception as batch_error:
            print(f"❌ Batch processing failed: {batch_error}")
            print("🔄 Falling back to individual processing...")
            
            # Fallback to individual processing
            for i, url in enumerate(image_urls):
                try:
                    print(f"🔄 Processing image {i+1}/{len(image_urls)} individually...")
                    single_result = model_pool.process_single_request(url, plot=False)
                    if single_result is not None:
                        processed_images.append(single_result)
                        print(f"✅ Image {i+1} processed successfully")
                    else:
                        processed_images.append(None)
                        print(f"❌ Image {i+1} failed")
                except Exception as single_error:
                    print(f"❌ Individual processing failed for image {i+1}: {single_error}")
                    processed_images.append(None)
        
        # Final check
        if not processed_images or len(processed_images) == 0:
            print("\n❌ FINAL RESULT: No images were successfully processed!")
            print("🔍 This explains why API returns 'Failed to process any images in batch'")
        else:
            print(f"\n✅ FINAL RESULT: {len(processed_images)} images processed successfully!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 