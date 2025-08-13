"""
Enhanced Performance Optimization Middleware
Provides comprehensive API performance optimizations including:
- Response compression with multiple algorithms
- Intelligent caching strategies
- Rate limiting with Redis backend
- Connection pooling optimization
- Request/response metrics
"""

import asyncio
import gzip
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
import hashlib

from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseMiddleware
from starlette.responses import Response as StarletteResponse, JSONResponse

from ..services.redis_cache_service import get_api_cache
from ..services.enhanced_database_service import get_cached_db_service

logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for response compression."""
    min_size: int = 1000  # Minimum response size to compress
    compression_level: int = 6  # Compression level (1-9)
    enable_gzip: bool = True
    enable_brotli: bool = False  # Requires brotli package
    excluded_paths: List[str] = None
    excluded_content_types: List[str] = None

    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = ['/health', '/ping', '/metrics']
        if self.excluded_content_types is None:
            self.excluded_content_types = ['image/', 'video/', 'audio/']

@dataclass
class CacheConfig:
    """Configuration for response caching."""
    default_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000  # Maximum number of cached responses
    cache_control_header: str = "public, max-age=300"
    etag_enabled: bool = True
    excluded_paths: List[str] = None
    excluded_methods: List[str] = None
    cache_query_params: bool = True

    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = ['/api/v*/workflow', '/api/v*/campaigns', '/admin/']
        if self.excluded_methods is None:
            self.excluded_methods = ['POST', 'PUT', 'DELETE', 'PATCH']

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Allow burst up to this many requests
    enable_per_ip_limits: bool = True
    enable_per_user_limits: bool = True
    excluded_paths: List[str] = None

    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = ['/health', '/ping', '/docs', '/redoc', '/openapi.json']

class CompressionMiddleware(BaseHTTPMiddleware):
    """High-performance response compression middleware."""
    
    def __init__(self, app, config: CompressionConfig):
        super().__init__(app)
        self.config = config
        
        # Try to import brotli if enabled
        if config.enable_brotli:
            try:
                import brotli
                self.brotli = brotli
            except ImportError:
                logger.warning("Brotli compression requested but brotli package not installed")
                self.config.enable_brotli = False
                self.brotli = None
        else:
            self.brotli = None

    async def dispatch(self, request: Request, call_next):
        # Skip compression for excluded paths
        if self._should_skip_compression(request):
            return await call_next(request)

        response = await call_next(request)
        
        # Only compress successful responses
        if response.status_code >= 400:
            return response

        # Get response content
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # Skip compression if response is too small
        if len(response_body) < self.config.min_size:
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        # Skip compression for certain content types
        content_type = response.headers.get('content-type', '').lower()
        if any(excluded in content_type for excluded in self.config.excluded_content_types):
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        # Check what compression client accepts
        accept_encoding = request.headers.get('accept-encoding', '').lower()
        
        compressed_body = None
        encoding = None

        # Try brotli first (better compression)
        if self.config.enable_brotli and self.brotli and 'br' in accept_encoding:
            compressed_body = self.brotli.compress(response_body)
            encoding = 'br'
        # Try gzip
        elif self.config.enable_gzip and 'gzip' in accept_encoding:
            compressed_body = gzip.compress(response_body, compresslevel=self.config.compression_level)
            encoding = 'gzip'

        if compressed_body and len(compressed_body) < len(response_body):
            # Compression successful
            headers = dict(response.headers)
            headers['content-encoding'] = encoding
            headers['content-length'] = str(len(compressed_body))
            headers['vary'] = 'Accept-Encoding'
            
            logger.debug(f"Compressed response: {len(response_body)} -> {len(compressed_body)} bytes ({encoding})")
            
            return Response(
                content=compressed_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type
            )
        else:
            # Return uncompressed response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

    def _should_skip_compression(self, request: Request) -> bool:
        """Check if compression should be skipped for this request."""
        path = request.url.path
        return any(excluded in path for excluded in self.config.excluded_paths)

class CacheMiddleware(BaseHTTPMiddleware):
    """Intelligent API response caching middleware."""
    
    def __init__(self, app, config: CacheConfig):
        super().__init__(app)
        self.config = config
        self.cache_service = get_api_cache()
        self.cache_stats = {"hits": 0, "misses": 0, "total": 0}

    async def dispatch(self, request: Request, call_next):
        # Skip caching for excluded methods and paths
        if self._should_skip_cache(request):
            return await call_next(request)

        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get cached response
        self.cache_stats["total"] += 1
        cached_response = self.cache_service.get_api_response("api_cache", cache_key)
        
        if cached_response:
            self.cache_stats["hits"] += 1
            logger.debug(f"Cache hit for {request.url.path}")
            
            # Return cached response with appropriate headers
            headers = cached_response.get("headers", {})
            headers["x-cache"] = "HIT"
            headers["x-cache-key"] = cache_key[:16] + "..."
            
            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=headers
            )

        # Cache miss - execute request
        self.cache_stats["misses"] += 1
        start_time = time.time()
        response = await call_next(request)
        execution_time = time.time() - start_time

        # Only cache successful GET responses
        if (response.status_code == 200 and 
            request.method == "GET" and 
            hasattr(response, 'body')):
            
            # Get response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            try:
                # Try to parse as JSON for caching
                content = json.loads(response_body.decode())
                
                cache_data = {
                    "content": content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "cached_at": datetime.now().isoformat(),
                    "execution_time_ms": execution_time * 1000
                }
                
                # Add cache headers
                cache_data["headers"]["cache-control"] = self.config.cache_control_header
                cache_data["headers"]["x-cache"] = "MISS"
                
                # Generate ETag if enabled
                if self.config.etag_enabled:
                    etag = hashlib.md5(response_body).hexdigest()
                    cache_data["headers"]["etag"] = f'"{etag}"'
                
                # Cache the response
                self.cache_service.cache_api_response("api_cache", cache_key, cache_data, self.config.default_ttl)
                
                logger.debug(f"Cached response for {request.url.path}")
                
                # Return response with new headers
                return JSONResponse(
                    content=content,
                    status_code=response.status_code,
                    headers=cache_data["headers"]
                )
                
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug(f"Could not cache non-JSON response for {request.url.path}")
                pass

        return response

    def _should_skip_cache(self, request: Request) -> bool:
        """Check if caching should be skipped."""
        if request.method in self.config.excluded_methods:
            return True
            
        path = request.url.path
        return any(excluded in path for excluded in self.config.excluded_paths)

    def _generate_cache_key(self, request: Request) -> str:
        """Generate a unique cache key for the request."""
        key_components = [
            request.method,
            request.url.path,
        ]
        
        # Include query parameters if enabled
        if self.config.cache_query_params and request.query_params:
            sorted_params = sorted(request.query_params.items())
            params_str = "&".join([f"{k}={v}" for k, v in sorted_params])
            key_components.append(params_str)
        
        # Include relevant headers for cache variation
        cache_varying_headers = ['accept', 'accept-language', 'authorization']
        for header in cache_varying_headers:
            if header in request.headers:
                key_components.append(f"{header}:{request.headers[header]}")
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self.cache_stats["total"]
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "total": total,
            "hit_rate": (self.cache_stats["hits"] / total * 100) if total > 0 else 0.0,
            "cache_service_stats": self.cache_service.get_stats()
        }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiting middleware."""
    
    def __init__(self, app, config: RateLimitConfig):
        super().__init__(app)
        self.config = config
        self.cache_service = get_api_cache()

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if self._should_skip_rate_limit(request):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        
        # Check rate limits
        rate_limit_key = self._generate_rate_limit_key(client_ip, user_id)
        
        if await self._is_rate_limited(rate_limit_key):
            logger.warning(f"Rate limit exceeded for {client_ip} (user: {user_id})")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )

        # Record the request
        await self._record_request(rate_limit_key)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(rate_limit_key)
        reset_time = await self._get_reset_time(rate_limit_key)
        
        response.headers["x-rate-limit-limit"] = str(self.config.requests_per_minute)
        response.headers["x-rate-limit-remaining"] = str(remaining)
        response.headers["x-rate-limit-reset"] = str(reset_time)
        
        return response

    def _should_skip_rate_limit(self, request: Request) -> bool:
        """Check if rate limiting should be skipped."""
        path = request.url.path
        return any(excluded in path for excluded in self.config.excluded_paths)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
            
        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available."""
        # This would typically extract from JWT token or API key
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT here
            return hashlib.md5(auth_header.encode()).hexdigest()[:8]
        return None

    def _generate_rate_limit_key(self, client_ip: str, user_id: Optional[str]) -> str:
        """Generate rate limit key."""
        if user_id and self.config.enable_per_user_limits:
            return f"rate_limit:user:{user_id}"
        elif self.config.enable_per_ip_limits:
            return f"rate_limit:ip:{client_ip}"
        else:
            return f"rate_limit:global"

    async def _is_rate_limited(self, key: str) -> bool:
        """Check if the request should be rate limited."""
        # Simple sliding window implementation using Redis
        minute_key = f"{key}:minute"
        hour_key = f"{key}:hour"
        day_key = f"{key}:day"
        
        # Get current counts (in a real implementation, you'd use Redis operations)
        minute_count = await self._get_request_count(minute_key, 60)
        hour_count = await self._get_request_count(hour_key, 3600)
        day_count = await self._get_request_count(day_key, 86400)
        
        return (
            minute_count >= self.config.requests_per_minute or
            hour_count >= self.config.requests_per_hour or
            day_count >= self.config.requests_per_day
        )

    async def _get_request_count(self, key: str, window: int) -> int:
        """Get request count for a time window."""
        # This is a simplified implementation
        # In production, use Redis with proper sliding window
        cached_data = self.cache_service.get('rate_limits', key)
        if cached_data:
            return cached_data.get('count', 0)
        return 0

    async def _record_request(self, key: str):
        """Record a request for rate limiting."""
        # Simplified implementation
        minute_key = f"{key}:minute"
        current_data = self.cache_service.get('rate_limits', minute_key) or {'count': 0}
        current_data['count'] += 1
        self.cache_service.set('rate_limits', minute_key, current_data, ttl=60)

    async def _get_remaining_requests(self, key: str) -> int:
        """Get remaining requests for the current window."""
        minute_count = await self._get_request_count(f"{key}:minute", 60)
        return max(0, self.config.requests_per_minute - minute_count)

    async def _get_reset_time(self, key: str) -> int:
        """Get timestamp when rate limit resets."""
        return int(time.time()) + 60  # Reset in 60 seconds

class PerformanceMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect detailed performance metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.db_service = get_cached_db_service()
        self.metrics = {
            "total_requests": 0,
            "total_response_time": 0.0,
            "endpoint_metrics": {},
            "status_code_counts": {}
        }

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Record request start
        endpoint = f"{request.method} {request.url.path}"
        self.metrics["total_requests"] += 1
        
        if endpoint not in self.metrics["endpoint_metrics"]:
            self.metrics["endpoint_metrics"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "error_count": 0
            }

        try:
            response = await call_next(request)
            execution_time = time.time() - start_time
            
            # Update metrics
            self._update_endpoint_metrics(endpoint, execution_time, response.status_code)
            self._update_status_code_metrics(response.status_code)
            
            # Add performance headers
            response.headers["x-response-time"] = f"{execution_time * 1000:.2f}ms"
            response.headers["x-request-id"] = str(hash(f"{time.time()}{endpoint}"))[:16]
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_endpoint_metrics(endpoint, execution_time, 500, error=True)
            self._update_status_code_metrics(500)
            raise

    def _update_endpoint_metrics(self, endpoint: str, execution_time: float, status_code: int, error: bool = False):
        """Update metrics for a specific endpoint."""
        metrics = self.metrics["endpoint_metrics"][endpoint]
        metrics["count"] += 1
        metrics["total_time"] += execution_time
        metrics["min_time"] = min(metrics["min_time"], execution_time)
        metrics["max_time"] = max(metrics["max_time"], execution_time)
        
        if error or status_code >= 400:
            metrics["error_count"] += 1
        
        self.metrics["total_response_time"] += execution_time

    def _update_status_code_metrics(self, status_code: int):
        """Update status code metrics."""
        if status_code not in self.metrics["status_code_counts"]:
            self.metrics["status_code_counts"][status_code] = 0
        self.metrics["status_code_counts"][status_code] += 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        total_requests = self.metrics["total_requests"]
        
        endpoint_summary = {}
        for endpoint, metrics in self.metrics["endpoint_metrics"].items():
            avg_time = metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0
            endpoint_summary[endpoint] = {
                "count": metrics["count"],
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": metrics["min_time"] * 1000 if metrics["min_time"] != float('inf') else 0,
                "max_time_ms": metrics["max_time"] * 1000,
                "error_rate": (metrics["error_count"] / metrics["count"] * 100) if metrics["count"] > 0 else 0
            }
        
        return {
            "total_requests": total_requests,
            "avg_response_time_ms": (self.metrics["total_response_time"] / total_requests * 1000) if total_requests > 0 else 0,
            "status_codes": self.metrics["status_code_counts"],
            "endpoints": endpoint_summary,
            "timestamp": datetime.now().isoformat()
        }

def create_enhanced_performance_middleware(
    compression_config: CompressionConfig,
    cache_config: CacheConfig,
    rate_limit_config: RateLimitConfig,
    enable_compression: bool = True,
    enable_caching: bool = True,
    enable_rate_limiting: bool = True,
    enable_metrics: bool = True
):
    """
    Create a combined performance middleware with all optimizations.
    
    Args:
        compression_config: Configuration for response compression
        cache_config: Configuration for response caching
        rate_limit_config: Configuration for rate limiting
        enable_compression: Whether to enable compression
        enable_caching: Whether to enable caching
        enable_rate_limiting: Whether to enable rate limiting
        enable_metrics: Whether to enable performance metrics
        
    Returns:
        Combined middleware class
    """
    
    class EnhancedPerformanceMiddleware(BaseHTTPMiddleware):
        def __init__(self, app):
            super().__init__(app)
            
            # Initialize sub-middlewares
            self.middlewares = []
            
            if enable_metrics:
                self.metrics_middleware = PerformanceMetricsMiddleware(app)
                self.middlewares.append(self.metrics_middleware)
                
            if enable_rate_limiting:
                self.rate_limit_middleware = RateLimitMiddleware(app, rate_limit_config)
                self.middlewares.append(self.rate_limit_middleware)
                
            if enable_caching:
                self.cache_middleware = CacheMiddleware(app, cache_config)
                self.middlewares.append(self.cache_middleware)
                
            if enable_compression:
                self.compression_middleware = CompressionMiddleware(app, compression_config)
                self.middlewares.append(self.compression_middleware)

        async def dispatch(self, request: Request, call_next):
            # Apply middlewares in sequence
            current_call_next = call_next
            
            for middleware in reversed(self.middlewares):
                current_middleware = middleware
                current_call_next_temp = current_call_next
                
                async def create_wrapper(mid, next_call):
                    async def wrapper(req):
                        return await mid.dispatch(req, next_call)
                    return wrapper
                
                current_call_next = await create_wrapper(current_middleware, current_call_next_temp)
            
            return await current_call_next(request)
        
        def get_performance_summary(self) -> Dict[str, Any]:
            """Get comprehensive performance summary."""
            summary = {}
            
            if hasattr(self, 'metrics_middleware'):
                summary['metrics'] = self.metrics_middleware.get_metrics_summary()
                
            if hasattr(self, 'cache_middleware'):
                summary['cache'] = self.cache_middleware.get_cache_stats()
                
            return summary
    
    return EnhancedPerformanceMiddleware

# Helper function to create optimized configuration
def create_production_config():
    """Create production-optimized configuration."""
    return {
        "compression": CompressionConfig(
            min_size=500,
            compression_level=6,
            enable_gzip=True,
            enable_brotli=True,
            excluded_paths=['/health', '/ping', '/metrics', '/docs', '/redoc']
        ),
        "cache": CacheConfig(
            default_ttl=300,  # 5 minutes
            max_cache_size=2000,
            cache_control_header="public, max-age=300",
            etag_enabled=True,
            excluded_paths=['/api/v*/workflow', '/api/v*/campaigns']
        ),
        "rate_limit": RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=2000,
            requests_per_day=20000,
            burst_limit=20
        )
    }