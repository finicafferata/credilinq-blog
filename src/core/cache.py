"""
Redis caching service for performance optimization.
Provides distributed caching with TTL, serialization, and fallback handling.
"""

import json
import pickle
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union, Callable
from functools import wraps
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from ..config.settings import settings

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass


class RedisCache:
    """
    Asynchronous Redis cache client with advanced features.
    Supports TTL, compression, serialization, and graceful fallbacks.
    """
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.connected = False
        self.fallback_cache: Dict[str, Dict] = {}  # In-memory fallback
        self.fallback_max_size = 1000
        
    async def initialize(self) -> bool:
        """Initialize Redis connection with health check."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory fallback cache")
            return False
            
        try:
            # Create Redis client with connection pool
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=10
            )
            
            # Test connection
            await self.client.ping()
            self.connected = True
            logger.info(f"✅ Redis connected: {settings.redis_host}:{settings.redis_port}")
            return True
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using fallback cache")
            self.connected = False
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.connected = False
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate namespaced cache key."""
        return f"{settings.cache_prefix}:{namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            json_str = json.dumps(value, default=str)
            return json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(namespace, key)
        
        if self.connected and self.client:
            try:
                data = await self.client.get(cache_key)
                if data:
                    return self._deserialize_value(data)
                return None
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                # Fall through to fallback cache
        
        # Fallback to in-memory cache
        if cache_key in self.fallback_cache:
            entry = self.fallback_cache[cache_key]
            if entry['expires'] > datetime.utcnow():
                return entry['value']
            else:
                del self.fallback_cache[cache_key]
        
        return None
    
    async def set(
        self, 
        namespace: str, 
        key: str, 
        value: Any, 
        ttl: int = 3600
    ) -> bool:
        """Set value in cache with TTL."""
        cache_key = self._generate_key(namespace, key)
        
        if self.connected and self.client:
            try:
                serialized = self._serialize_value(value)
                await self.client.setex(cache_key, ttl, serialized)
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                # Fall through to fallback cache
        
        # Fallback to in-memory cache
        if len(self.fallback_cache) >= self.fallback_max_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self.fallback_cache.keys(),
                key=lambda k: self.fallback_cache[k]['created']
            )[:100]
            for old_key in oldest_keys:
                del self.fallback_cache[old_key]
        
        self.fallback_cache[cache_key] = {
            'value': value,
            'created': datetime.utcnow(),
            'expires': datetime.utcnow() + timedelta(seconds=ttl)
        }
        return True
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(namespace, key)
        
        if self.connected and self.client:
            try:
                await self.client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        # Also remove from fallback cache
        self.fallback_cache.pop(cache_key, None)
        return True
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_key(namespace, key)
        
        if self.connected and self.client:
            try:
                return bool(await self.client.exists(cache_key))
            except Exception as e:
                logger.warning(f"Redis exists error: {e}")
        
        # Check fallback cache
        if cache_key in self.fallback_cache:
            entry = self.fallback_cache[cache_key]
            if entry['expires'] > datetime.utcnow():
                return True
            else:
                del self.fallback_cache[cache_key]
        
        return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        count = 0
        
        if self.connected and self.client:
            try:
                keys = await self.client.keys(pattern)
                if keys:
                    count = await self.client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis pattern invalidation error: {e}")
        
        # Handle fallback cache pattern matching
        fallback_keys = [
            key for key in self.fallback_cache.keys()
            if pattern.replace('*', '') in key
        ]
        for key in fallback_keys:
            del self.fallback_cache[key]
            count += 1
        
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'connected': self.connected,
            'fallback_entries': len(self.fallback_cache),
            'redis_available': REDIS_AVAILABLE
        }
        
        if self.connected and self.client:
            try:
                info = await self.client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_total_commands': info.get('total_commands_processed', 0)
                })
            except Exception as e:
                logger.warning(f"Redis stats error: {e}")
        
        return stats


# Global cache instance
cache = RedisCache()


def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_parts = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))
    
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
        else:
            key_parts.append(f"{k}:{hash(str(v))}")
    
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    namespace: str,
    ttl: int = 3600,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
        key_func: Custom key generation function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            # Try to get from cache
            result = await cache.get(namespace, cache_key)
            if result is not None:
                logger.debug(f"Cache hit: {namespace}:{cache_key}")
                return result
            
            # Execute function
            logger.debug(f"Cache miss: {namespace}:{cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(namespace, cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


# Common cache patterns
class CachePatterns:
    """Common caching patterns and utilities."""
    
    @staticmethod
    async def get_or_set(
        namespace: str,
        key: str,
        factory_func: Callable,
        ttl: int = 3600,
        *args,
        **kwargs
    ) -> Any:
        """Get from cache or execute factory function and cache result."""
        result = await cache.get(namespace, key)
        if result is not None:
            return result
        
        result = await factory_func(*args, **kwargs)
        await cache.set(namespace, key, result, ttl)
        return result
    
    @staticmethod
    async def invalidate_related(entity_type: str, entity_id: str):
        """Invalidate all cache entries related to an entity."""
        patterns = [
            f"{settings.cache_prefix}:*:{entity_type}:{entity_id}:*",
            f"{settings.cache_prefix}:*:{entity_type}s:*",  # Plural form
            f"{settings.cache_prefix}:lists:*{entity_type}*"
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            count = await cache.invalidate_pattern(pattern)
            total_invalidated += count
        
        logger.info(f"Invalidated {total_invalidated} cache entries for {entity_type}:{entity_id}")
        return total_invalidated


# Cache health check
async def cache_health_check() -> Dict[str, Any]:
    """Perform cache health check."""
    try:
        test_key = "health_check"
        test_value = {"timestamp": datetime.utcnow().isoformat()}
        
        # Test set/get/delete
        await cache.set("health", test_key, test_value, 60)
        retrieved = await cache.get("health", test_key)
        await cache.delete("health", test_key)
        
        success = retrieved is not None and retrieved["timestamp"] == test_value["timestamp"]
        
        stats = await cache.get_stats()
        
        return {
            "status": "healthy" if success else "degraded",
            "connected": cache.connected,
            "stats": stats,
            "test_passed": success
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False
        }