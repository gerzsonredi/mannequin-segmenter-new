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

# Copy requirements files first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies in optimized order
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask torch boto3 python-dotenv && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . /app/

# Create necessary directories if they don't exist
RUN mkdir -p /app/artifacts /app/tools /app/infer

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api_app:app
ENV FLASK_ENV=production
ENV FORCE_CPU=true

# Cloud Run sets PORT automatically, but we default to 5001
ENV PORT=5001

# Expose the port
EXPOSE $PORT

# Optimize Python startup
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use gunicorn for production with optimized startup
CMD ["sh", "-c", "gunicorn --config gunicorn.conf.py --bind 0.0.0.0:$PORT api_app:app"]