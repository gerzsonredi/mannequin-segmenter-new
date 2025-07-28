# Gunicorn configuration for mannequin-segmenter API
# Optimized for BiSeNet v1 horizontal scaling with CPU-only processing

import os

# Application
port = os.environ.get("PORT", "5001")
bind = f"0.0.0.0:{port}"

# Timeouts - increased for model loading
timeout = 900  # Request timeout (15 minutes)
keepalive = 65 # Keep connections alive
graceful_timeout = 120  # Graceful shutdown timeout

# Worker processes - OPTIMIZED FOR HORIZONTAL SCALING (0-100 INSTANCES, MAX 50 CONCURRENT)
# Strategy: 1 worker + 1 thread per instance, each request uses ALL available CPUs
workers = 1  # Single worker per instance
worker_class = "sync"  # Sync worker for Flask  

# CORRECT APPROACH: 1 request per instance uses ALL available CPUs
# Each single request gets full CPU power (PyTorch uses all cores)  
# Better than 2 requests competing for the same CPU resources
threads = 1  # Single request per instance (no resource contention)
worker_connections = 5  # Keep low for single concurrent request

# Restart workers after more requests to allow model pool to be utilized longer
max_requests = 100
max_requests_jitter = 10

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr

# Process naming
proc_name = "mannequin-segmenter"

# Preload app for faster startup (but be careful with memory)
preload_app = False  # Keep False to avoid issues with model loading

# Security
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 16384

# Cloud Run optimizations
# Enable proper signal handling for graceful shutdown
enable_stdio_inheritance = True

print(f"🔧 Gunicorn configured for port {port}")
print(f"🔧 Workers: {workers}, Threads: {threads}")
print(f"🔧 Timeout: {timeout}s, Graceful timeout: {graceful_timeout}s") 
print(f"🧵 GUNICORN OPTIMIZATION: {threads} thread per instance (full CPU utilization per request)") 