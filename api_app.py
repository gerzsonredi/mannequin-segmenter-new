from flask import Flask, request, jsonify, current_app
from evfsam import EVFSAMSingleImageInferencer
from tools.logger import EVFSAMLogger
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

    # Initialize logger using EVFSAMLogger from tools package
    api_logger = EVFSAMLogger()
    # Note: EVFSAMLogger is a simple custom logger, not a standard Python logger
    # So we don't integrate it with Flask's logging system

    # AWS S3 Configuration
    aws_access_key_id = get_env_variable("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = get_env_variable("AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket_name = get_env_variable("AWS_S3_BUCKET_NAME")
    aws_s3_region = get_env_variable("AWS_S3_REGION")
    
    # EVF-SAM Configuration - get default prompt mode from environment
    default_prompt_mode = get_env_variable("EVFSAM_PROMPT_MODE") or "both"
    if default_prompt_mode not in ["under", "above", "both"]:
        api_logger.log(f"WARNING: Invalid EVFSAM_PROMPT_MODE '{default_prompt_mode}', defaulting to 'both'")
        default_prompt_mode = "both"

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
        print("Loading EVF-SAM model...")
        api_logger.log(f"Starting API application - Loading EVF-SAM model with default prompt_mode: {default_prompt_mode}")
        try:
            
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # if device == "cuda":
            #     inferencer = EVFSAMSingleImageInferencer(use_bnb=False, precision="fp16")
            # else:
            #     inferencer = EVFSAMSingleImageInferencer(use_bnb=False, precision="fp32")
            # print("Model loaded.")
            # api_logger.log("EVF-SAM model loaded successfully")

            # # Initialize the segmenter
            # inferencer = MaskRCNNSegmenter(
            #     model_path="mannequin-segmenter/artifacts/20250703_132502/checkpoint.pt",
            #     confidence_threshold=0.5,
            #     mask_threshold=0.5
            # )

            # Initialize the segmenter
            inferencer = BiRefNetSegmenter(
                model_path="mannequin-segmenter/artifacts/birefnet/checkpoint.pt",  # Optional custom weights
                model_name="zhengpeng7/BiRefNet",    # HuggingFace model
                precision="fp16",                    # fp16, fp32, or bf16
                mask_threshold=0.5
            )

            print("Inferencer successfully loaded!")

        except Exception as e:
            api_logger.log(f"ERROR: Failed to load EVF-SAM model: {str(e)}")
            inferencer = None
            print(f"Model loading failed: {e}")

    @app.route('/health', methods=['GET'])
    def health():
        api_logger.log("Health check request received")
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "evf-sam-api",
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
            # Get prompt_mode from request or use default from config
            # prompt_mode = data.get('prompt_mode', current_app.config['DEFAULT_PROMPT_MODE'])
            
            # Validate prompt_mode
            # if prompt_mode not in ["under", "above", "both"]:
            #     api_logger.log(f"Error: Invalid prompt_mode '{prompt_mode}', must be 'under', 'above', or 'both'")
            #     return jsonify({"error": f"Invalid prompt_mode '{prompt_mode}'. Must be 'under', 'above', or 'both'"}), 400
            
            # api_logger.log(f"Processing image from URL: {image_url} with prompt_mode: {prompt_mode}")
            
            # Check if model loaded successfully
            if inferencer is None:
                api_logger.log("ERROR: Model not loaded, returning test response")
                print("ERROR: Model not loaded, returning test response")
                return jsonify({
                    "error": "EVF-SAM model failed to load",
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
            return jsonify({"error": str(e)}), 500
            
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