#!/usr/bin/env python3
"""
Application-level request limiter for additional protection.
Works alongside Cloud Run's built-in concurrency control.
"""

import threading
import time
from functools import wraps
from flask import jsonify
import logging

class RequestLimiter:
    def __init__(self, max_concurrent=4, queue_timeout=300):
        """
        Initialize request limiter.
        
        Args:
            max_concurrent: Maximum concurrent requests per instance
            queue_timeout: Maximum time to wait in queue (seconds)
        """
        self.max_concurrent = max_concurrent
        self.queue_timeout = queue_timeout
        self.active_requests = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def acquire_slot(self):
        """
        Try to acquire a processing slot.
        
        Returns:
            bool: True if slot acquired, False if should be queued
        """
        start_time = time.time()
        
        while time.time() - start_time < self.queue_timeout:
            with self.lock:
                if self.active_requests < self.max_concurrent:
                    self.active_requests += 1
                    self.logger.info(f"Slot acquired. Active requests: {self.active_requests}/{self.max_concurrent}")
                    return True
                    
            # Wait a bit before retrying
            time.sleep(0.1)
            
        # Timeout reached
        self.logger.warning(f"Request limiter timeout after {self.queue_timeout}s")
        return False
    
    def release_slot(self):
        """Release a processing slot."""
        with self.lock:
            if self.active_requests > 0:
                self.active_requests -= 1
                self.logger.info(f"Slot released. Active requests: {self.active_requests}/{self.max_concurrent}")
    
    def get_status(self):
        """Get current limiter status."""
        with self.lock:
            return {
                "active_requests": self.active_requests,
                "max_concurrent": self.max_concurrent,
                "slots_available": self.max_concurrent - self.active_requests,
                "load_percentage": (self.active_requests / self.max_concurrent) * 100
            }

# Global instance
request_limiter = RequestLimiter(max_concurrent=4, queue_timeout=300)

def limit_concurrent_requests(f):
    """
    Decorator to limit concurrent requests at application level.
    
    Usage:
        @app.route('/endpoint')
        @limit_concurrent_requests
        def my_endpoint():
            return "Hello"
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Try to acquire slot
        if not request_limiter.acquire_slot():
            return jsonify({
                "error": "Server overloaded",
                "message": "Too many concurrent requests. Please try again later.",
                "retry_after": 30,
                "status": request_limiter.get_status()
            }), 429  # Too Many Requests
        
        try:
            # Execute the actual function
            result = f(*args, **kwargs)
            return result
            
        finally:
            # Always release the slot
            request_limiter.release_slot()
    
    return decorated_function

def get_limiter_status():
    """Get current request limiter status."""
    return request_limiter.get_status() 