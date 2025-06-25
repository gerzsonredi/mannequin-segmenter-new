from flask import Flask, request, jsonify
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

load_dotenv()

app = Flask(__name__)

# Initialize logger using EVFSAMLogger from tools package
api_logger = EVFSAMLogger()

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = get_env_variable("AWS_S3_BUCKET_NAME")
AWS_S3_REGION = get_env_variable("AWS_S3_REGION")

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Initialize the inferencer. This will load the model into memory.
# It's placed here so it's only loaded once when the app starts.
print("Loading EVF-SAM model...")
api_logger.log("Starting API application - Loading EVF-SAM model...")
try:
    api_logger.log("Step 1: About to initialize EVFSAMSingleImageInferencer")
    inferencer = EVFSAMSingleImageInferencer(use_bnb=False, precision="fp32")
    api_logger.log("Step 2: EVFSAMSingleImageInferencer created successfully")
    print("Model loaded.")
    api_logger.log("EVF-SAM model loaded successfully")
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
    try:
        api_logger.log("Received inference request")
        data = request.get_json()
        if not data or 'image_url' not in data:
            api_logger.log("Error: image_url not provided in request")
            return jsonify({"error": "image_url not provided"}), 400

        image_url = data['image_url']
        api_logger.log(f"Processing image from URL: {image_url}")
        
        # Check if model loaded successfully
        if inferencer is None:
            api_logger.log("ERROR: Model not loaded, returning test response")
            return jsonify({
                "error": "EVF-SAM model failed to load",
                "visualization_url": "https://test-response.example.com/test.jpg",
                "input_url": image_url
            })
    except Exception as e:
        api_logger.log(f"Exception in /infer input handling: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    try:
        api_logger.log("Step 3: About to call process_image_url")
        # Process the image, but don't show the plot
        vis = inferencer.process_image_url(image_url, plot=False)  # Changed to False to avoid plot issues
        api_logger.log("Step 4: process_image_url completed")

        if vis is None:
            api_logger.log(f"Error: Failed to process image from URL: {image_url}")
            return jsonify({"error": "Failed to process image"}), 500

        api_logger.log("Step 5: About to convert and upload to S3")
        # Convert visualization image (numpy array) to a format that can be uploaded
        vis_pil = Image.fromarray(vis.astype(np.uint8))
        buff = io.BytesIO()
        vis_pil.save(buff, format="JPEG")
        buff.seek(0)

        # Create a unique filename
        filename = f"{uuid.uuid4()}.jpg"

        # Get current date as YYYY/MM/DD
        today = datetime.utcnow()
        date_prefix = today.strftime("%Y/%m/%d")
        s3_key = f"{date_prefix}/{filename}"

        # Upload to S3 with the date-based key
        s3_client.upload_fileobj(
            buff,
            AWS_S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )

        s3_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{s3_key}"
        api_logger.log(f"Step 6: Successfully processed image and uploaded result to S3: {s3_url}")
        
        return jsonify({
            "visualization_url": s3_url,
        })

    except Exception as e:
        error_msg = f"Error processing image from URL {image_url}: {str(e)}"
        api_logger.log(error_msg)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For development only - use gunicorn in production
    api_logger.log("Running in development mode - use gunicorn for production!")
    app.run(host='0.0.0.0', port=5001, debug=True) 