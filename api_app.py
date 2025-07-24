from flask import Flask, request, jsonify, current_app
from tools.logger import AppLogger
from tools.env_utils import get_env_variable
import base64
from PIL import Image
import io
import numpy as np
import boto3
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
import torch
# from tools.MaskRCNN_segmenter import MaskRCNNSegmenter
from tools.BirefNet import BiRefNetSegmenter

def create_app(testing=False):
    """Application factory for the Flask app."""
    app = Flask(__name__)
    load_dotenv()

    # Initialize logger
    api_logger = AppLogger()
    # Note: AppLogger is a simple custom logger, not a standard Python logger
    # So we don't integrate it with Flask's logging system

    # AWS S3 Configuration
    aws_access_key_id = get_env_variable("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = get_env_variable("AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket_name = get_env_variable("AWS_S3_BUCKET_NAME")
    aws_s3_region = get_env_variable("AWS_S3_REGION")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_s3_region
    )
    
    # Initialize the inferencer.
    # In testing, this will be mocked to avoid loading the real model.
    if testing:
        inferencer = None  # Will be replaced by mock in tests
    else:
 
        try:

            # Initialize the segmenter
            inferencer = BiRefNetSegmenter(
                model_path="artifacts/20250703_190222/checkpoint.pt",
                model_name="zhengpeng7/BiRefNet",    # HuggingFace model
                precision="fp16",                    # fp16, fp32, or bf16
                mask_threshold=0.5
            )
            if inferencer is None:
                print("Error! Couldn't load inferencer!")
                exit(1)
            print("Inferencer successfully loaded!")

        except Exception as e:
            inferencer = None
            print(f"Model loading failed: {e}")

    @app.route('/health', methods=['GET'])
    def health():
        api_logger.log("Health check request received")
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "mannequin-segmenter-api",
            "version": "1.0.0"
        }), 200

    @app.route('/infer', methods=['POST'])
    def infer():
        inferencer = current_app.config['INFERENCER'] # Use the inferencer from the app config
        try:
            api_logger.log("Received inference request")
            print("Received inference request")
            data = request.get_json()
            if not data or 'image_url' not in data:
                api_logger.log("Error: image_url not provided in request")
                print("Error: image_url not provided in request")
                return jsonify({"error": "image_url not provided"}), 400

            image_url = data['image_url']

            # Check if model loaded successfully
            if inferencer is None:
                api_logger.log("ERROR: Model not loaded, returning test response")
                print("ERROR: Model not loaded, returning test response")
                return jsonify({
                    "error": "model failed to load",
                    "visualization_url": "https://test-response.example.com/test.jpg",
                    "input_url": image_url
                }), 500 # Return 500 as it's a server-side issue
        except Exception as e:
            api_logger.log(f"Exception in /infer input handling: {str(e)}")
            print(f"Exception in /infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            print("Step 3: About to call process_image_url")
            api_logger.log("Step 3: About to call process_image_url")
            # vis = inferencer.process_image_url(image_url, plot=False, prompt_mode=prompt_mode)
            vis = inferencer.process_image_url(image_url)
            print("Step 4: process_image_url completed")
            api_logger.log("Step 4: process_image_url completed")

            if vis is None:
                print(f"Error: Failed to process image from URL: {image_url}")
                api_logger.log(f"Error: Failed to process image from URL: {image_url}")
                return jsonify({"error": "Failed to process image"}), 500

            print("Step 5: About to convert and upload to S3")
            api_logger.log("Step 5: About to convert and upload to S3")
            vis_pil = Image.fromarray(vis.astype(np.uint8))
            buff = io.BytesIO()
            vis_pil.save(buff, format="JPEG")
            buff.seek(0)

            filename = f"{uuid.uuid4()}.jpg"
            today = datetime.utcnow()
            date_prefix = today.strftime("%Y/%m/%d")
            s3_key = f"{date_prefix}/{filename}"

            s3_client.upload_fileobj(
                buff,
                aws_s3_bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )

            s3_url = f"https://{aws_s3_bucket_name}.s3.{aws_s3_region}.amazonaws.com/{s3_key}"
            print(f"Step 6: Successfully processed image and uploaded result to S3: {s3_url}")
            api_logger.log(f"Step 6: Successfully processed image and uploaded result to S3: {s3_url}")
            
            return jsonify({
                "visualization_url": s3_url,
            })

        except Exception as e:
            error_msg = f"Error processing image from URL {image_url}: {str(e)}"
            api_logger.log(error_msg)
            print(error_msg)
            return jsonify({"error": str(e)}), 500

    @app.route('/batch_infer', methods=['POST'])
    def batch_infer():
        """Batch inference endpoint for processing multiple images simultaneously."""
        inferencer = current_app.config['INFERENCER']
        try:
            api_logger.log("Received batch inference request")
            print("Received batch inference request")
            data = request.get_json()
            
            if not data or 'image_urls' not in data:
                api_logger.log("Error: image_urls not provided in batch request")
                print("Error: image_urls not provided in batch request")
                return jsonify({"error": "image_urls list not provided"}), 400

            image_urls = data['image_urls']
            if not isinstance(image_urls, list) or len(image_urls) == 0:
                api_logger.log("Error: image_urls must be a non-empty list")
                print("Error: image_urls must be a non-empty list")
                return jsonify({"error": "image_urls must be a non-empty list"}), 400
                
            if len(image_urls) > 20:  # Limit batch size
                api_logger.log(f"Error: batch size {len(image_urls)} exceeds maximum of 20")
                print(f"Error: batch size {len(image_urls)} exceeds maximum of 20")
                return jsonify({"error": "Maximum batch size is 20 images"}), 400

            # Check if model loaded successfully
            if inferencer is None:
                api_logger.log("ERROR: Model not loaded for batch processing")
                print("ERROR: Model not loaded for batch processing")
                return jsonify({
                    "error": "model failed to load",
                    "batch_size": len(image_urls)
                }), 500

        except Exception as e:
            api_logger.log(f"Exception in /batch_infer input handling: {str(e)}")
            print(f"Exception in /batch_infer input handling: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
        try:
            batch_size = len(image_urls)
            api_logger.log(f"Processing batch of {batch_size} images")
            print(f"Processing batch of {batch_size} images")
            
            # Use the batch processing method
            processed_images = inferencer.process_batch_urls(image_urls, plot=False, max_batch_size=batch_size)
            
            if not processed_images or len(processed_images) == 0:
                api_logger.log("Error: No images were successfully processed in batch")
                print("Error: No images were successfully processed in batch")
                return jsonify({"error": "Failed to process any images in batch"}), 500

            api_logger.log(f"Successfully processed {len(processed_images)} images, uploading to S3...")
            print(f"Successfully processed {len(processed_images)} images, uploading to S3...")
            
            # Upload all processed images to S3
            s3_urls = []
            today = datetime.utcnow()
            date_prefix = today.strftime("%Y/%m/%d")
            
            for i, processed_img in enumerate(processed_images):
                try:
                    # Convert to PIL and upload
                    vis_pil = Image.fromarray(processed_img.astype(np.uint8))
                    buff = io.BytesIO()
                    vis_pil.save(buff, format="JPEG")
                    buff.seek(0)

                    filename = f"batch_{uuid.uuid4()}_{i}.jpg"
                    s3_key = f"{date_prefix}/{filename}"

                    s3_client.upload_fileobj(
                        buff,
                        aws_s3_bucket_name,
                        s3_key,
                        ExtraArgs={'ContentType': 'image/jpeg'}
                    )

                    s3_url = f"https://{aws_s3_bucket_name}.s3.{aws_s3_region}.amazonaws.com/{s3_key}"
                    s3_urls.append(s3_url)
                    
                except Exception as upload_error:
                    api_logger.log(f"Error uploading image {i}: {upload_error}")
                    print(f"Error uploading image {i}: {upload_error}")
                    s3_urls.append(None)  # Mark failed upload
            
            successful_uploads = [url for url in s3_urls if url is not None]
            
            api_logger.log(f"Batch processing completed: {len(successful_uploads)}/{batch_size} successful")
            print(f"Batch processing completed: {len(successful_uploads)}/{batch_size} successful")
            
            return jsonify({
                "batch_size": batch_size,
                "successful_count": len(successful_uploads),
                "failed_count": batch_size - len(successful_uploads),
                "visualization_urls": s3_urls,
                "input_urls": image_urls
            })

        except Exception as e:
            error_msg = f"Error processing batch of {len(image_urls)} images: {str(e)}"
            api_logger.log(error_msg)
            print(error_msg)
            return jsonify({"error": str(e), "batch_size": len(image_urls)}), 500
            
    # Attach objects to app context for easier testing and access
    app.config['S3_CLIENT'] = s3_client
    app.config['INFERENCER'] = inferencer
    app.config['API_LOGGER'] = api_logger
    # app.config['DEFAULT_PROMPT_MODE'] = default_prompt_mode

    return app

# Create a global app instance for gunicorn to find
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 