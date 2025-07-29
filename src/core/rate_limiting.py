"""
Advanced rate limiting and throttling middleware.
Supports sliding window, fixed window, and token bucket algorithms.
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import logging

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import starlette.middleware.base

from ..config.settings import settings
from .cache import cache

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Custom exception for rate limit violations."""
    
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)


class RateLimiter:
    """
    Advanced rate limiter with multiple algorithms and storage backends.
    Supports distributed rate limiting via Redis.
    """
    
    def __init__(self):
        self.local_storage: Dict[str, Any] = defaultdict(dict)
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _get_client_id(self, request: Request) -> str:
        """Generate unique client identifier."""
        # Priority: API key > JWT token > IP address
        client_id = "anonymous"
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            client_id = f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Check for Authorization header
        elif auth_header := request.headers.get("Authorization"):
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                client_id = f"token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        
        # Fall back to IP address
        else:
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                ip = forwarded_for.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else "unknown"
            client_id = f"ip:{ip}"
        
        return client_id
    
    def _get_rate_limit_key(self, client_id: str, endpoint: str) -> str:
        """Generate rate limit storage key."""
        return f"rate_limit:{client_id}:{endpoint}"
    
    async def _cleanup_local_storage(self):
        """Clean up expired entries from local storage."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_keys = []
        for key, data in self.local_storage.items():
            if isinstance(data, dict) and "expires" in data:
                if current_time > data["expires"]:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_storage[key]
        
        self._last_cleanup = current_time
        logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit entries")
    
    async def check_rate_limit(
        self,
        request: Request,
        max_requests: int,
        window_seconds: int,
        algorithm: str = "sliding_window"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be rate limited.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        client_id = self._get_client_id(request)
        endpoint = f"{request.method}:{request.url.path}"
        
        if algorithm == "sliding_window":
            return await self._sliding_window_check(
                client_id, endpoint, max_requests, window_seconds
            )
        elif algorithm == "fixed_window":
            return await self._fixed_window_check(
                client_id, endpoint, max_requests, window_seconds
            )
        elif algorithm == "token_bucket":
            return await self._token_bucket_check(
                client_id, endpoint, max_requests, window_seconds
            )
        else:
            raise ValueError(f"Unknown rate limiting algorithm: {algorithm}")
    
    async def _sliding_window_check(
        self, client_id: str, endpoint: str, max_requests: int, window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting algorithm."""
        key = self._get_rate_limit_key(client_id, endpoint)
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Try Redis first
        if cache.connected:
            try:
                # Use Redis sorted set for sliding window
                pipe = cache.client.pipeline()
                
                # Remove old entries
                pipe.zremrangebyscore(key, 0, window_start)
                
                # Count current requests
                pipe.zcard(key)
                
                # Add current request
                pipe.zadd(key, {str(current_time): current_time})
                
                # Set expiry
                pipe.expire(key, window_seconds + 60)
                
                results = await pipe.execute()
                current_count = results[1] + 1  # +1 for the request we just added
                
                is_allowed = current_count <= max_requests
                
                if not is_allowed:
                    # Remove the request we just added since it's not allowed
                    await cache.client.zrem(key, str(current_time))
                
                return is_allowed, {
                    "requests_made": current_count - (0 if is_allowed else 1),
                    "requests_remaining": max(0, max_requests - current_count + (0 if is_allowed else 1)),
                    "reset_time": int(current_time + window_seconds),
                    "retry_after": window_seconds if not is_allowed else 0
                }
            
            except Exception as e:
                logger.warning(f"Redis rate limiting failed: {e}, using local storage")
        
        # Fallback to local storage
        await self._cleanup_local_storage()
        
        if key not in self.local_storage:
            self.local_storage[key] = {
                "requests": deque(),
                "expires": current_time + window_seconds + 60
            }
        
        data = self.local_storage[key]
        requests = data["requests"]
        
        # Remove old requests
        while requests and requests[0] < window_start:
            requests.popleft()
        
        current_count = len(requests)
        is_allowed = current_count < max_requests
        
        if is_allowed:
            requests.append(current_time)
        
        return is_allowed, {
            "requests_made": current_count + (1 if is_allowed else 0),
            "requests_remaining": max(0, max_requests - current_count - (1 if is_allowed else 0)),
            "reset_time": int(current_time + window_seconds),
            "retry_after": window_seconds if not is_allowed else 0
        }
    
    async def _fixed_window_check(
        self, client_id: str, endpoint: str, max_requests: int, window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting algorithm."""
        current_time = time.time()
        window_start = int(current_time // window_seconds) * window_seconds
        key = f"{self._get_rate_limit_key(client_id, endpoint)}:{window_start}"
        
        # Try Redis first
        if cache.connected:
            try:
                current_count = await cache.client.incr(key)
                if current_count == 1:
                    await cache.client.expire(key, window_seconds + 60)
                
                is_allowed = current_count <= max_requests
                
                if not is_allowed:
                    await cache.client.decr(key)
                    current_count -= 1
                
                reset_time = window_start + window_seconds
                
                return is_allowed, {
                    "requests_made": current_count,
                    "requests_remaining": max(0, max_requests - current_count),
                    "reset_time": int(reset_time),
                    "retry_after": int(reset_time - current_time) if not is_allowed else 0
                }
            
            except Exception as e:
                logger.warning(f"Redis rate limiting failed: {e}, using local storage")
        
        # Fallback to local storage
        await self._cleanup_local_storage()
        
        if key not in self.local_storage:
            self.local_storage[key] = {
                "count": 0,
                "expires": window_start + window_seconds + 60
            }
        
        data = self.local_storage[key]
        current_count = data["count"]
        is_allowed = current_count < max_requests
        
        if is_allowed:
            data["count"] += 1
            current_count += 1
        
        reset_time = window_start + window_seconds
        
        return is_allowed, {
            "requests_made": current_count,
            "requests_remaining": max(0, max_requests - current_count),
            "reset_time": int(reset_time),
            "retry_after": int(reset_time - current_time) if not is_allowed else 0
        }
    
    async def _token_bucket_check(
        self, client_id: str, endpoint: str, max_requests: int, window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting algorithm."""
        key = self._get_rate_limit_key(client_id, endpoint)
        current_time = time.time()
        
        # Calculate refill rate (tokens per second)
        refill_rate = max_requests / window_seconds
        
        # Try Redis first
        if cache.connected:
            try:
                # Get current bucket state
                bucket_data = await cache.get("buckets", key)
                
                if bucket_data is None:
                    bucket_data = {
                        "tokens": max_requests,
                        "last_refill": current_time
                    }
                
                # Calculate tokens to add
                time_passed = current_time - bucket_data["last_refill"]
                tokens_to_add = time_passed * refill_rate
                bucket_data["tokens"] = min(max_requests, bucket_data["tokens"] + tokens_to_add)
                bucket_data["last_refill"] = current_time
                
                is_allowed = bucket_data["tokens"] >= 1
                
                if is_allowed:
                    bucket_data["tokens"] -= 1
                
                # Store updated bucket state
                await cache.set("buckets", key, bucket_data, window_seconds + 60)
                
                return is_allowed, {
                    "requests_made": max_requests - int(bucket_data["tokens"]),
                    "requests_remaining": int(bucket_data["tokens"]),
                    "reset_time": int(current_time + (max_requests - bucket_data["tokens"]) / refill_rate),
                    "retry_after": int(1 / refill_rate) if not is_allowed else 0
                }
            
            except Exception as e:
                logger.warning(f"Redis rate limiting failed: {e}, using local storage")
        
        # Fallback to local storage
        await self._cleanup_local_storage()
        
        if key not in self.local_storage:
            self.local_storage[key] = {
                "tokens": max_requests,
                "last_refill": current_time,
                "expires": current_time + window_seconds + 60
            }
        
        data = self.local_storage[key]
        
        # Calculate tokens to add
        time_passed = current_time - data["last_refill"]
        tokens_to_add = time_passed * refill_rate
        data["tokens"] = min(max_requests, data["tokens"] + tokens_to_add)
        data["last_refill"] = current_time
        
        is_allowed = data["tokens"] >= 1
        
        if is_allowed:
            data["tokens"] -= 1
        
        return is_allowed, {
            "requests_made": max_requests - int(data["tokens"]),
            "requests_remaining": int(data["tokens"]),
            "reset_time": int(current_time + (max_requests - data["tokens"]) / refill_rate),
            "retry_after": int(1 / refill_rate) if not is_allowed else 0
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    """
    FastAPI middleware for request rate limiting.
    Supports different limits for different endpoint categories.
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.rate_configs = {
            # Default limits
            "default": {
                "max_requests": settings.rate_limit_per_minute,
                "window_seconds": 60,
                "algorithm": "sliding_window"
            },
            # API-specific limits
            "api_heavy": {
                "max_requests": 10,
                "window_seconds": 60,
                "algorithm": "token_bucket"
            },
            "api_light": {
                "max_requests": 100,
                "window_seconds": 60,
                "algorithm": "sliding_window"
            },
            # Endpoint-specific limits
            "blog_creation": {
                "max_requests": 5,
                "window_seconds": 300,  # 5 minutes
                "algorithm": "token_bucket"
            },
            "health_check": {
                "max_requests": 1000,
                "window_seconds": 60,
                "algorithm": "fixed_window"
            }
        }
    
    def _get_rate_config(self, request: Request) -> Dict[str, Any]:
        """Determine rate limit configuration for request."""
        path = request.url.path
        method = request.method
        
        # Health check endpoints
        if path.startswith("/health") or path == "/":
            return self.rate_configs["health_check"]
        
        # Blog creation endpoints
        if path == "/api/blogs" and method == "POST":
            return self.rate_configs["blog_creation"]
        
        # Heavy API endpoints (content generation)
        if any(path.startswith(prefix) for prefix in ["/api/blogs", "/api/campaigns"]):
            if method in ["POST", "PUT"]:
                return self.rate_configs["api_heavy"]
            else:
                return self.rate_configs["api_light"]
        
        # Default limits
        return self.rate_configs["default"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Process request with rate limiting."""
        # Skip rate limiting for preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get rate limit configuration
        config = self._get_rate_config(request)
        
        try:
            # Check rate limit
            is_allowed, rate_info = await rate_limiter.check_rate_limit(
                request,
                config["max_requests"],
                config["window_seconds"],
                config["algorithm"]
            )
            
            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded for {rate_limiter._get_client_id(request)} "
                    f"on {request.method} {request.url.path}"
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": rate_info["retry_after"],
                        "reset_time": rate_info["reset_time"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(config["max_requests"]),
                        "X-RateLimit-Remaining": str(rate_info["requests_remaining"]),
                        "X-RateLimit-Reset": str(rate_info["reset_time"]),
                        "Retry-After": str(rate_info["retry_after"])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(config["max_requests"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["requests_remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
            
            return response
        
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue without rate limiting if there's an error
            return await call_next(request)


# Rate limiting decorators
def rate_limit(
    max_requests: int,
    window_seconds: int = 60,
    algorithm: str = "sliding_window",
    key_func: Optional[Callable] = None
):
    """
    Decorator for function-level rate limiting.
    
    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        algorithm: Rate limiting algorithm
        key_func: Custom key generation function
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Generate custom key if provided
            if key_func:
                custom_key = key_func(request, *args, **kwargs)
                # Temporarily modify the request path for rate limiting
                original_path = request.url.path
                request._url = request.url.replace(path=f"{original_path}:{custom_key}")
            
            # Check rate limit
            is_allowed, rate_info = await rate_limiter.check_rate_limit(
                request, max_requests, window_seconds, algorithm
            )
            
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": rate_info["retry_after"]
                    },
                    headers={
                        "Retry-After": str(rate_info["retry_after"])
                    }
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


# Rate limiting utilities
async def get_rate_limit_status(request: Request) -> Dict[str, Any]:
    """Get current rate limit status for a client."""
    client_id = rate_limiter._get_client_id(request)
    
    # Get status for different endpoint categories
    status_info = {}
    
    for category, config in {
        "default": {"max_requests": settings.rate_limit_per_minute, "window_seconds": 60},
        "blog_creation": {"max_requests": 5, "window_seconds": 300},
        "api_heavy": {"max_requests": 10, "window_seconds": 60}
    }.items():
        endpoint = f"status_check:{category}"
        _, rate_info = await rate_limiter.check_rate_limit(
            request, config["max_requests"], config["window_seconds"], "sliding_window"
        )
        
        # Don't actually consume a request for status check
        status_info[category] = {
            "limit": config["max_requests"],
            "remaining": rate_info["requests_remaining"],
            "reset_time": rate_info["reset_time"],
            "window_seconds": config["window_seconds"]
        }
    
    return {
        "client_id": client_id,
        "limits": status_info,
        "timestamp": int(time.time())
    }