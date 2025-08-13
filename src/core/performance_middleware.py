"""
API Performance Optimization Middleware
Provides request/response compression, caching headers, connection pooling,
and performance monitoring for FastAPI applications.
"""

import time
import gzip
import brotli
import zlib
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
import logging
import hashlib

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ..config.settings import settings
from .cache import cache

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for response compression."""
    min_size: int = 1000  # Minimum response size to compress (bytes)
    compression_level: int = 6  # Compression level (1-9)
    enable_gzip: bool = True
    enable_brotli: bool = True
    excluded_content_types: List[str] = None
    excluded_paths: List[str] = None

    def __post_init__(self):
        if self.excluded_content_types is None:
            self.excluded_content_types = [
                'image/', 'video/', 'audio/', 'font/',
                'application/octet-stream', 'application/zip'
            ]
        if self.excluded_paths is None:
            self.excluded_paths = ['/health', '/ping', '/metrics']


@dataclass
class CacheConfig:
    """Configuration for API response caching."""
    default_ttl: int = 300  # 5 minutes
    cache_control_header: str = "public, max-age=300"
    etag_enabled: bool = True
    vary_headers: List[str] = None
    cache_key_headers: List[str] = None
    excluded_methods: List[str] = None
    excluded_paths: List[str] = None

    def __post_init__(self):
        if self.vary_headers is None:
            self.vary_headers = ['Accept-Encoding', 'Authorization']
        if self.cache_key_headers is None:
            self.cache_key_headers = ['Accept', 'Authorization', 'Content-Type']
        if self.excluded_methods is None:
            self.excluded_methods = ['POST', 'PUT', 'DELETE', 'PATCH']
        if self.excluded_paths is None:
            self.excluded_paths = ['/api/v2/workflow', '/api/v2/campaigns']


class ResponseCompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for compressing API responses using gzip or brotli.
    Automatically detects best compression method based on Accept-Encoding header.
    """
    
    def __init__(self, app, config: Optional[CompressionConfig] = None):
        super().__init__(app)
        self.config = config or CompressionConfig()
    
    def _should_compress(self, request: Request, response: Response) -> bool:
        """Determine if response should be compressed."""
        # Check if path is excluded
        if any(request.url.path.startswith(excluded) for excluded in self.config.excluded_paths):
            return False
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if any(excluded in content_type for excluded in self.config.excluded_content_types):
            return False
        
        # Check if already compressed
        if response.headers.get('content-encoding'):
            return False
        
        return True
    
    def _get_best_encoding(self, accept_encoding: str) -> Optional[str]:
        """Get the best compression encoding based on client support."""
        accept_encoding = accept_encoding.lower()
        
        # Check for brotli support (better compression)
        if self.config.enable_brotli and 'br' in accept_encoding:
            return 'br'
        
        # Check for gzip support
        if self.config.enable_gzip and ('gzip' in accept_encoding or '*' in accept_encoding):
            return 'gzip'
        
        return None
    
    def _compress_data(self, data: bytes, encoding: str) -> bytes:
        """Compress data using specified encoding."""
        try:
            if encoding == 'br':
                return brotli.compress(data, quality=self.config.compression_level)
            elif encoding == 'gzip':
                return gzip.compress(data, compresslevel=self.config.compression_level)
            else:
                return data
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and compress response if appropriate."""
        start_time = time.time()
        
        # Get response from next middleware/route
        response = await call_next(request)
        
        # Check if we should compress
        if not self._should_compress(request, response):
            return response
        
        # Get client's accept-encoding
        accept_encoding = request.headers.get('accept-encoding', '')
        encoding = self._get_best_encoding(accept_encoding)
        
        if not encoding:
            return response
        
        # Get response body
        if hasattr(response, 'body'):
            body = response.body
        else:
            # For streaming responses, we need to read the content
            if isinstance(response, StreamingResponse):
                # Don't compress streaming responses to avoid memory issues
                return response
            
            # For other response types, try to get body
            try:
                body = response.body
            except AttributeError:
                return response
        
        # Check minimum size
        if len(body) < self.config.min_size:
            return response
        
        # Compress the body
        compressed_body = self._compress_data(body, encoding)
        
        # Calculate compression ratio
        original_size = len(body)
        compressed_size = len(compressed_body)
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        # Only use compression if it actually reduces size
        if compressed_size >= original_size:
            logger.debug(f"Compression ineffective for {request.url.path}")
            return response
        
        # Create new response with compressed body
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
        # Set compression headers
        compressed_response.headers['content-encoding'] = encoding
        compressed_response.headers['content-length'] = str(compressed_size)
        compressed_response.headers['x-compression-ratio'] = f"{compression_ratio:.1f}%"
        
        # Add performance header
        processing_time = (time.time() - start_time) * 1000
        compressed_response.headers['x-compression-time'] = f"{processing_time:.2f}ms"
        
        logger.debug(
            f"Compressed {request.url.path}: "
            f"{original_size}B → {compressed_size}B "
            f"({compression_ratio:.1f}% reduction) using {encoding}"
        )
        
        return compressed_response


class APICacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API response caching with ETag support and intelligent cache invalidation.
    """
    
    def __init__(self, app, config: Optional[CacheConfig] = None):
        super().__init__(app)
        self.config = config or CacheConfig()
    
    def _should_cache(self, request: Request) -> bool:
        """Determine if request should be cached."""
        # Only cache GET requests by default
        if request.method in self.config.excluded_methods:
            return False
        
        # Check if path is excluded
        if any(request.url.path.startswith(excluded) for excluded in self.config.excluded_paths):
            return False
        
        # Don't cache authenticated requests (unless specifically configured)
        if request.headers.get('authorization') and not settings.cache_authenticated_requests:
            return False
        
        return True
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method,
            str(request.url.path),
            str(request.url.query)
        ]
        
        # Add relevant headers to cache key
        for header_name in self.config.cache_key_headers:
            header_value = request.headers.get(header_name.lower())
            if header_value:
                key_parts.append(f"{header_name}:{header_value}")
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_etag(self, content: bytes) -> str:
        """Generate ETag for response content."""
        return f'"{hashlib.md5(content).hexdigest()}"'
    
    def _is_modified(self, request: Request, etag: str) -> bool:
        """Check if resource has been modified based on If-None-Match header."""
        if_none_match = request.headers.get('if-none-match')
        if if_none_match:
            # Handle both quoted and unquoted ETags
            client_etags = [tag.strip().strip('"') for tag in if_none_match.split(',')]
            server_etag = etag.strip('"')
            return server_etag not in client_etags
        return True
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with caching logic."""
        # Check if we should attempt caching
        if not self._should_cache(request):
            return await call_next(request)
        
        start_time = time.time()
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        try:
            cached_data = await cache.get("api_responses", cache_key)
            
            if cached_data:
                etag = cached_data.get('etag')
                
                # Check if client has current version
                if etag and not self._is_modified(request, etag):
                    # Return 304 Not Modified
                    response = Response(status_code=304)
                    response.headers['etag'] = etag
                    response.headers['cache-control'] = self.config.cache_control_header
                    response.headers['x-cache'] = 'HIT-304'
                    return response
                
                # Return cached content
                cached_response = Response(
                    content=cached_data['content'],
                    status_code=cached_data['status_code'],
                    headers=cached_data['headers']
                )
                
                # Add cache headers
                if etag:
                    cached_response.headers['etag'] = etag
                cached_response.headers['cache-control'] = self.config.cache_control_header
                cached_response.headers['x-cache'] = 'HIT'
                cached_response.headers['age'] = str(int(time.time() - cached_data['cached_at']))
                
                # Add vary headers
                if self.config.vary_headers:
                    cached_response.headers['vary'] = ', '.join(self.config.vary_headers)
                
                logger.debug(f"Cache hit for {request.url.path}")
                return cached_response
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        # Cache miss - get response from application
        response = await call_next(request)
        
        # Only cache successful responses
        if response.status_code == 200:
            try:
                # Get response content
                if hasattr(response, 'body'):
                    content = response.body
                elif hasattr(response, 'content'):
                    content = response.content
                else:
                    # Can't cache streaming or other complex responses
                    response.headers['x-cache'] = 'MISS-UNCACHEABLE'
                    return response
                
                # Generate ETag
                etag = self._generate_etag(content) if self.config.etag_enabled else None
                
                # Prepare cached data
                cache_data = {
                    'content': content,
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'etag': etag,
                    'cached_at': time.time()
                }
                
                # Cache the response
                await cache.set("api_responses", cache_key, cache_data, self.config.default_ttl)
                
                # Add cache headers to response
                if etag:
                    response.headers['etag'] = etag
                response.headers['cache-control'] = self.config.cache_control_header
                response.headers['x-cache'] = 'MISS'
                
                if self.config.vary_headers:
                    response.headers['vary'] = ', '.join(self.config.vary_headers)
                
                logger.debug(f"Cached response for {request.url.path}")
                
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
                response.headers['x-cache'] = 'MISS-ERROR'
        else:
            response.headers['x-cache'] = 'MISS-NON-200'
        
        return response


class ConnectionOptimizationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for optimizing connection handling and request processing.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with connection optimizations."""
        self.request_count += 1
        start_time = time.time()
        
        # Add request ID for tracking
        request_id = f"req_{int(time.time() * 1000)}_{self.request_count}"
        
        # Get response
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Add performance headers
        response.headers['x-request-id'] = request_id
        response.headers['x-processing-time'] = f"{processing_time:.2f}ms"
        response.headers['x-server-timing'] = f"total;dur={processing_time:.2f}"
        
        # Add connection optimization headers
        response.headers['connection'] = 'keep-alive'
        response.headers['keep-alive'] = 'timeout=30, max=100'
        
        # Log slow requests
        if processing_time > 1000:  # Log requests taking more than 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {processing_time:.2f}ms"
            )
        
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Combined performance middleware that includes compression, caching, and connection optimization.
    """
    
    def __init__(
        self,
        app,
        compression_config: Optional[CompressionConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        enable_compression: bool = True,
        enable_caching: bool = True,
        enable_connection_optimization: bool = True
    ):
        super().__init__(app)
        self.enable_compression = enable_compression
        self.enable_caching = enable_caching
        self.enable_connection_optimization = enable_connection_optimization
        
        # Initialize sub-middlewares
        if enable_compression:
            self.compression_middleware = ResponseCompressionMiddleware(app, compression_config)
        if enable_caching:
            self.cache_middleware = APICacheMiddleware(app, cache_config)
        if enable_connection_optimization:
            self.connection_middleware = ConnectionOptimizationMiddleware(app)
        
        # Performance metrics
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_saves = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through all performance optimizations."""
        start_time = time.time()
        self.total_requests += 1
        
        # Chain middlewares in order
        response = None
        
        # First: Connection optimization
        if self.enable_connection_optimization:
            response = await self.connection_middleware.dispatch(request, call_next)
        else:
            response = await call_next(request)
        
        # Second: Caching (only if we have the response from previous step)
        if self.enable_caching:
            # We need to create a call_next that returns our existing response
            async def return_response(req):
                return response
            response = await self.cache_middleware.dispatch(request, return_response)
        
        # Third: Compression (only if we have the response)
        if self.enable_compression:
            async def return_response(req):
                return response
            response = await self.compression_middleware.dispatch(request, return_response)
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        self.total_processing_time += processing_time
        
        # Track cache performance
        cache_header = response.headers.get('x-cache', '')
        if 'HIT' in cache_header:
            self.cache_hits += 1
        elif 'MISS' in cache_header:
            self.cache_misses += 1
        
        # Track compression
        if response.headers.get('content-encoding'):
            self.compression_saves += 1
        
        # Add overall performance metrics to response
        response.headers['x-performance-optimized'] = 'true'
        
        return response
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (self.cache_hits / max(self.cache_hits + self.cache_misses, 1)) * 100
        avg_processing_time = self.total_processing_time / max(self.total_requests, 1)
        compression_rate = (self.compression_saves / max(self.total_requests, 1)) * 100
        
        return {
            "total_requests": self.total_requests,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "compression_rate_percent": round(compression_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "compression_saves": self.compression_saves
        }


# Global performance middleware instance for statistics
global_performance_middleware = None


def create_performance_middleware(
    compression_config: Optional[CompressionConfig] = None,
    cache_config: Optional[CacheConfig] = None,
    enable_compression: bool = True,
    enable_caching: bool = True,
    enable_connection_optimization: bool = True
) -> PerformanceMiddleware:
    """Factory function to create performance middleware."""
    global global_performance_middleware
    
    global_performance_middleware = PerformanceMiddleware(
        None,  # App will be set by FastAPI
        compression_config,
        cache_config,
        enable_compression,
        enable_caching,
        enable_connection_optimization
    )
    
    return global_performance_middleware


async def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    if global_performance_middleware:
        return await global_performance_middleware.get_performance_stats()
    else:
        return {"error": "Performance middleware not initialized"}


# Cache invalidation utilities
async def invalidate_api_cache(pattern: Optional[str] = None) -> int:
    """Invalidate API response cache."""
    if pattern:
        cache_pattern = f"*api_responses*{pattern}*"
    else:
        cache_pattern = "*api_responses*"
    
    return await cache.invalidate_pattern(cache_pattern)


async def invalidate_cache_for_path(path: str) -> int:
    """Invalidate cache for a specific API path."""
    return await invalidate_api_cache(path)


async def warm_cache_for_endpoints(endpoints: List[str]) -> Dict[str, Any]:
    """Pre-warm cache for critical endpoints."""
    # This would make requests to critical endpoints to populate cache
    # Implementation would depend on your specific use case
    results = {"warmed": 0, "failed": 0}
    
    # Placeholder implementation
    logger.info(f"Cache warming requested for {len(endpoints)} endpoints")
    
    return results