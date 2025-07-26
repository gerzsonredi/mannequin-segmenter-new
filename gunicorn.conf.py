# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5001"
backlog = 2048

# Worker processes - OPTIMIZED FOR HORIZONTAL SCALING (60 INSTANCES)
# Strategy: 1 worker + 1 thread per instance, 60 instances = 60 concurrent capacity
workers = 1  # Single worker per instance
worker_class = "sync"  # Sync worker for Flask  
threads = 1  # Single thread per instance (concurrency=1 on Cloud Run)
worker_connections = 5  # Lower since only 1 concurrent request per instance
timeout = 600  # 10 minutes for model inference
keepalive = 5

# Restart workers after more requests to allow model pool to be utilized longer
max_requests = 100  # Increased from 20 to prevent frequent model pool reinitialization
max_requests_jitter = 20  # Proportional jitter

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

# Environment variables - MAXIMUM CPU UTILIZATION
raw_env = [
    f"PYTHONPATH={os.getenv('PYTHONPATH', '/app')}",
    # Note: Threading will be set dynamically by DeepLabV3_MobileViT based on available cores
    # Cloud Run has 2 CPUs, local machines may have more
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