"""
Redis Caching Service for Performance Optimization
Implements intelligent caching with TTL management, cache invalidation strategies,
and performance monitoring.
"""

import redis
import json
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import os
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    default_ttl: int = 3600  # 1 hour default
    max_retries: int = 3
    retry_delay: float = 0.1
    compression_threshold: int = 1024  # Compress data larger than 1KB
    max_key_length: int = 250

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return (self.hits / self.total_requests) * 100 if self.total_requests > 0 else 0.0

class RedisCacheService:
    """
    High-performance Redis caching service with:
    - Intelligent TTL management
    - Cache invalidation strategies
    - Performance monitoring
    - Automatic failover to local cache
    - Compression for large data
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.local_cache = {}  # Fallback cache
        self.stats = CacheStats()
        self._initialize_redis()
        
    def _initialize_redis(self):
        """Initialize Redis connection with fallback to local cache."""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,  # Handle binary data for compression
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache service initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache fallback: {str(e)}")
            self.redis_client = None

    def _generate_cache_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate a consistent cache key with namespace."""
        # Include kwargs in key generation for parameter-specific caching
        key_components = [namespace, key]
        
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = "&".join([f"{k}={v}" for k, v in sorted_kwargs])
            key_components.append(kwargs_str)
        
        full_key = ":".join(str(c) for c in key_components)
        
        # Hash long keys to prevent Redis key length issues
        if len(full_key) > self.config.max_key_length:
            hash_key = hashlib.md5(full_key.encode()).hexdigest()
            return f"{namespace}:hash:{hash_key}"
        
        return full_key

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if it exceeds threshold."""
        if len(data) > self.config.compression_threshold:
            import zlib
            return b"compressed:" + zlib.compress(data)
        return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it was compressed."""
        if data.startswith(b"compressed:"):
            import zlib
            return zlib.decompress(data[11:])  # Remove "compressed:" prefix
        return data

    @contextmanager
    def _measure_time(self):
        """Context manager to measure execution time."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            # Update rolling average
            total_time = self.stats.avg_response_time_ms * self.stats.total_requests + execution_time
            self.stats.total_requests += 1
            self.stats.avg_response_time_ms = total_time / self.stats.total_requests

    def get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """
        Get value from cache with performance tracking.
        
        Args:
            namespace: Cache namespace (e.g., 'agent_results', 'query_cache')
            key: Cache key
            **kwargs: Additional parameters for key generation
            
        Returns:
            Cached value or None if not found
        """
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        with self._measure_time():
            try:
                if self.redis_client:
                    # Try Redis first
                    data = self.redis_client.get(cache_key)
                    if data:
                        # Decompress and deserialize
                        decompressed_data = self._decompress_data(data)
                        value = json.loads(decompressed_data.decode())
                        self.stats.hits += 1
                        logger.debug(f"Cache hit for key: {cache_key}")
                        return value
                
                # Try local cache fallback
                if cache_key in self.local_cache:
                    entry = self.local_cache[cache_key]
                    if entry['expires_at'] > datetime.now():
                        self.stats.hits += 1
                        logger.debug(f"Local cache hit for key: {cache_key}")
                        return entry['value']
                    else:
                        # Remove expired entry
                        del self.local_cache[cache_key]
                
                self.stats.misses += 1
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Cache get error for key {cache_key}: {str(e)}")
                return None

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        ttl = ttl or self.config.default_ttl
        
        try:
            # Serialize and compress data
            serialized_data = json.dumps(value, default=str).encode()
            compressed_data = self._compress_data(serialized_data)
            
            if self.redis_client:
                # Store in Redis
                success = self.redis_client.setex(cache_key, ttl, compressed_data)
                if success:
                    logger.debug(f"Cached in Redis: {cache_key} (TTL: {ttl}s)")
                    return True
            
            # Store in local cache as fallback
            self.local_cache[cache_key] = {
                'value': value,
                'expires_at': datetime.now() + timedelta(seconds=ttl)
            }
            logger.debug(f"Cached locally: {cache_key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Cache set error for key {cache_key}: {str(e)}")
            return False

    def delete(self, namespace: str, key: str, **kwargs) -> bool:
        """Delete specific cache entry."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            deleted = 0
            
            if self.redis_client:
                deleted += self.redis_client.delete(cache_key)
            
            if cache_key in self.local_cache:
                del self.local_cache[cache_key]
                deleted += 1
            
            logger.debug(f"Deleted cache key: {cache_key}")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {str(e)}")
            return False

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all keys in a namespace."""
        try:
            deleted_count = 0
            
            if self.redis_client:
                # Find all keys with namespace prefix
                pattern = f"{namespace}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count += self.redis_client.delete(*keys)
            
            # Clear from local cache
            local_keys_to_delete = [k for k in self.local_cache.keys() if k.startswith(f"{namespace}:")]
            for key in local_keys_to_delete:
                del self.local_cache[key]
                deleted_count += 1
            
            logger.info(f"Invalidated {deleted_count} keys from namespace: {namespace}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Namespace invalidation error for {namespace}: {str(e)}")
            return 0

    def cached_function(self, namespace: str, ttl: Optional[int] = None):
        """
        Decorator for caching function results.
        
        Args:
            namespace: Cache namespace for the function
            ttl: Cache TTL in seconds
            
        Example:
            @cache_service.cached_function("agent_results", ttl=3600)
            def expensive_agent_operation(param1, param2):
                # Expensive operation here
                return result
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                func_name = func.__name__
                args_key = hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()
                
                # Try to get from cache first
                cached_result = self.get(namespace, f"{func_name}:{args_key}")
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(namespace, f"{func_name}:{args_key}", result, ttl)
                return result
            
            return wrapper
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            'hit_rate': self.stats.hit_rate,
            'total_requests': self.stats.total_requests,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'errors': self.stats.errors,
            'avg_response_time_ms': round(self.stats.avg_response_time_ms, 2),
            'redis_connected': self.redis_client is not None,
            'local_cache_size': len(self.local_cache)
        }

    def health_check(self) -> Dict[str, Any]:
        """Check cache service health."""
        health = {
            'status': 'unknown',
            'redis_connected': False,
            'local_cache_available': True,
            'stats': self.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if self.redis_client:
                # Test Redis connection
                start_time = time.time()
                self.redis_client.ping()
                response_time = (time.time() - start_time) * 1000
                
                health['redis_connected'] = True
                health['redis_response_time_ms'] = round(response_time, 2)
                health['status'] = 'healthy'
            else:
                health['status'] = 'local_only'
                health['message'] = 'Using local cache fallback'
                
        except Exception as e:
            health['status'] = 'redis_error'
            health['error'] = str(e)
            health['message'] = 'Redis connection failed, using local cache'
        
        return health

    def cleanup_expired(self) -> int:
        """Clean up expired entries from local cache."""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.local_cache.items():
            if entry['expires_at'] <= now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired local cache entries")
        return len(expired_keys)

# Specialized caching services for different use cases

class AgentResultCache(RedisCacheService):
    """Specialized cache for AI agent results with longer TTL."""
    
    def __init__(self):
        config = CacheConfig(default_ttl=7200)  # 2 hours for agent results
        super().__init__(config)
    
    def cache_agent_result(self, agent_name: str, input_hash: str, result: Any, 
                          execution_time_ms: int, ttl: Optional[int] = None) -> bool:
        """Cache agent result with metadata."""
        cache_data = {
            'result': result,
            'execution_time_ms': execution_time_ms,
            'cached_at': datetime.now().isoformat(),
            'agent_name': agent_name
        }
        return self.set('agent_results', f"{agent_name}:{input_hash}", cache_data, ttl)
    
    def get_agent_result(self, agent_name: str, input_hash: str) -> Optional[Dict]:
        """Get cached agent result with metadata."""
        return self.get('agent_results', f"{agent_name}:{input_hash}")

class QueryCache(RedisCacheService):
    """Specialized cache for database queries with shorter TTL."""
    
    def __init__(self):
        config = CacheConfig(default_ttl=1800)  # 30 minutes for queries
        super().__init__(config)
    
    def cache_query_result(self, query_hash: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache database query result."""
        return self.set('query_cache', query_hash, result, ttl)
    
    def get_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        return self.get('query_cache', query_hash)

class APIResponseCache(RedisCacheService):
    """Specialized cache for API responses with configurable TTL."""
    
    def __init__(self):
        config = CacheConfig(default_ttl=900)  # 15 minutes for API responses
        super().__init__(config)
    
    def cache_api_response(self, endpoint: str, params_hash: str, response: Any, 
                          ttl: Optional[int] = None) -> bool:
        """Cache API response."""
        return self.set('api_responses', f"{endpoint}:{params_hash}", response, ttl)
    
    def get_api_response(self, endpoint: str, params_hash: str) -> Optional[Any]:
        """Get cached API response."""
        return self.get('api_responses', f"{endpoint}:{params_hash}")

# Global cache service instances
_agent_cache = None
_query_cache = None
_api_cache = None
_main_cache = None

def get_agent_cache() -> AgentResultCache:
    """Get global agent result cache instance."""
    global _agent_cache
    if _agent_cache is None:
        _agent_cache = AgentResultCache()
    return _agent_cache

def get_query_cache() -> QueryCache:
    """Get global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache

def get_api_cache() -> APIResponseCache:
    """Get global API response cache instance."""
    global _api_cache
    if _api_cache is None:
        _api_cache = APIResponseCache()
    return _api_cache

def get_main_cache() -> RedisCacheService:
    """Get global main cache instance."""
    global _main_cache
    if _main_cache is None:
        _main_cache = RedisCacheService()
    return _main_cache