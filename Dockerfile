# Use Python 3.11 slim base image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including git for BiSeNet and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (this layer gets cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone BiSeNet repository once during build time (this layer gets cached)
RUN git clone https://github.com/CoinCheung/BiSeNet.git && \
    echo "✅ BiSeNet repository cloned during build"

# Copy only specific files first (reduces cache invalidation)
COPY tools/ ./tools/
COPY api_app.py gunicorn.conf.py request_limiter.py ./

# Copy remaining files (this layer changes most often)
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:/app/BiSeNet
ENV FORCE_CPU=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --timeout 300 --worker-class sync api_app:app