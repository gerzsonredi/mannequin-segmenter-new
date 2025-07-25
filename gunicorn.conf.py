# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5001"
backlog = 2048

# Worker processes - OPTIMIZED FOR GPU MEMORY MANAGEMENT
# Strategy: 1 worker + 3 threads = max 3 concurrent requests per instance
workers = 1  # CRITICAL: Single worker to avoid multiple model loading on GPU
worker_class = "sync"  # Sync worker for Flask
threads = 3  # Aligned with Cloud Run concurrency=3
worker_connections = 20  # Adequate for async processing
timeout = 600  # 10 minutes for model inference
keepalive = 5

# Restart workers after fewer requests to prevent memory buildup
max_requests = 20  # Reduced from 1000 to prevent memory leaks
max_requests_jitter = 10  # Reduced jitter

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "mannequin-segmenter-api"

# Server mechanics
daemon = False
pidfile = None
tmp_upload_dir = None

# Worker timeout for graceful shutdown
graceful_timeout = 60  # Increased timeout for model cleanup

# Preload application for better memory usage
preload_app = False  # Changed to False to avoid model loading issues

# Environment variables - PERFORMANCE OPTIMIZED
raw_env = [
    f"PYTHONPATH={os.getenv('PYTHONPATH', '/app')}",
    # "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",  # Temporarily disabled - causes PyTorch crash
    "OMP_NUM_THREADS=8",  # Optimize CPU threads
    "MKL_NUM_THREADS=8",  # Intel MKL optimization
]

def when_ready(server):
    server.log.info("Mannequin Segmenter API server is ready. Listening on %s", server.address)

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal") 