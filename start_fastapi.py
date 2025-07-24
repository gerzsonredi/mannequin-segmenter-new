#!/usr/bin/env python3
"""
Startup script for optimized FastAPI Mannequin Segmenter
Supports both direct uvicorn and gunicorn execution
"""

import os
import sys
from fastapi_app import app

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5001"))
    workers = int(os.getenv("WORKERS", "1"))
    
    print(f"Starting FastAPI Mannequin Segmenter on {host}:{port}")
    print(f"Workers: {workers}")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    
    # Production settings for optimal performance
    uvicorn.run(
        "fastapi_app:app",
        host=host,
        port=port,
        workers=workers,
        loop="uvloop",  # High-performance event loop
        http="httptools",  # Fast HTTP implementation
        access_log=True,
        log_level="info"
    ) 