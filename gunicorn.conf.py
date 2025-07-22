# Gunicorn configuration file
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5001"
backlog = 2048

# Worker processes - EXPLICITLY 1 WORKER FOR EVF-SAM MODEL
workers = 1  # CRITICAL: Only 1 worker to avoid memory conflicts with large model
worker_class = "sync"
worker_connections = 1000
timeout = 600  # 10 minutes for model inference (increased for CPU processing)
keepalive = 5

# Restart workers after fewer requests to prevent memory buildup
max_requests = 100  # Reduced from 1000 to prevent memory leaks
max_requests_jitter = 10  # Reduced jitter

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "evf-sam-api"

# Server mechanics
daemon = False
pidfile = None
tmp_upload_dir = None

# Worker timeout for graceful shutdown
graceful_timeout = 60  # Increased timeout for model cleanup

# Preload application for better memory usage
preload_app = False  # Changed to False to avoid model loading issues

# Environment variables
raw_env = [
    f"PYTHONPATH={os.getenv('PYTHONPATH', '/app')}",
]

def when_ready(server):
    server.log.info("EVF-SAM API server is ready. Listening on %s", server.address)

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal") 