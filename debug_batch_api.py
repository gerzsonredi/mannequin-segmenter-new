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

def main():
    print("🔍 DEBUG: Testing batch processing API logic...")
    
    # Test image URL
    image_urls = [
        "https://public-images-redivivum.s3.eu-central-1.amazonaws.com/Remix_data/Majka-teniska-Mustang-132434083b.jpg"
    ]
    
    try:
        print("🤖 Loading BiRefNetSegmenter (same as API)...")
        inferencer = BiRefNetSegmenter(
            model_path="artifacts/20250703_190222/checkpoint.pt",
            model_name="zhengpeng7/BiRefNet",
            precision="fp16",
            mask_threshold=0.5
        )
        print("✅ Model loaded successfully!")
        
        print(f"\n🚀 Testing batch processing with {len(image_urls)} images...")
        print(f"📸 Image URLs: {image_urls}")
        
        # Try the exact same call as API
        try:
            print("🔄 Calling process_batch_urls...")
            processed_images = inferencer.process_batch_urls(
                image_urls, 
                plot=False, 
                max_batch_size=len(image_urls)
            )
            
            print(f"✅ process_batch_urls returned: {type(processed_images)}")
            if processed_images:
                print(f"📊 Number of processed images: {len(processed_images)}")
                for i, img in enumerate(processed_images):
                    if img is not None:
                        print(f"   Image {i+1}: {img.shape} ✅")
                    else:
                        print(f"   Image {i+1}: None ❌")
            else:
                print("❌ processed_images is None or empty!")
                
        except Exception as batch_error:
            print(f"❌ BATCH ERROR: {batch_error}")
            traceback.print_exc()
            
            print("\n🔄 Trying fallback: individual processing...")
            processed_images = []
            for i, url in enumerate(image_urls):
                try:
                    print(f"🔄 Processing image {i+1}/{len(image_urls)} individually...")
                    single_result = inferencer.process_image_url(url, plot=False)
                    if single_result is not None:
                        processed_images.append(single_result)
                        print(f"   ✅ Image {i+1}: {single_result.shape}")
                    else:
                        print(f"   ❌ Image {i+1}: None")
                except Exception as single_error:
                    print(f"   ❌ Single image {i+1} error: {single_error}")
                    traceback.print_exc()
                    
            print(f"\n🔄 FALLBACK RESULT: {len(processed_images)}/{len(image_urls)} images processed")
        
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