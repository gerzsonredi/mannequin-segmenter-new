# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, pycocotools, and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    build-essential \
    python3-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY EVF-SAM/requirements.txt /app/evf_sam_requirements.txt
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir flask torch boto3 python-dotenv 
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r evf_sam_requirements.txt

# Copy the entire application
COPY . /app/

# Create necessary directories if they don't exist
RUN mkdir -p /app/artifacts /app/tools /app/EVF-SAM

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api_app:app
ENV FLASK_ENV=production

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application with gunicorn using config file
CMD ["gunicorn", "--config", "gunicorn.conf.py", "api_app:app"]