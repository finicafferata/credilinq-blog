"""
Query Result Caching Service
Provides intelligent caching for expensive database operations with
invalidation strategies and performance monitoring.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
import asyncio
import logging

from .cache import cache, CachePatterns
from .agent_performance import AgentExecutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for database query execution."""
    query_id: str
    query_type: str
    execution_time_ms: int
    cache_hit: bool
    result_size: int
    table_names: List[str]
    timestamp: datetime
    cost_estimate: Optional[float] = None
    row_count: Optional[int] = None


@dataclass
class CacheConfiguration:
    """Configuration for query caching."""
    ttl: int = 300  # 5 minutes default
    max_size: int = 1000  # Maximum cached items
    enable_compression: bool = True
    enable_metrics: bool = True
    invalidation_patterns: List[str] = None
    cache_key_fields: List[str] = None
    exclude_tables: List[str] = None

    def __post_init__(self):
        if self.invalidation_patterns is None:
            self.invalidation_patterns = []
        if self.cache_key_fields is None:
            self.cache_key_fields = []
        if self.exclude_tables is None:
            self.exclude_tables = ['agent_performance', 'agent_decisions', 'logs']


class QueryCacheManager:
    """
    Manages caching for expensive database queries with intelligent invalidation.
    """

    def __init__(self):
        self.query_metrics: List[QueryMetrics] = []
        self.cache_configs: Dict[str, CacheConfiguration] = {}
        self.invalidation_rules: Dict[str, List[str]] = {}
        
        # Default configurations for common query types
        self._setup_default_configurations()
        self._setup_invalidation_rules()

    def _setup_default_configurations(self):
        """Setup default cache configurations for different query types."""
        self.cache_configs = {
            # Analytics queries - longer cache time
            "analytics": CacheConfiguration(
                ttl=1800,  # 30 minutes
                max_size=100,
                enable_compression=True,
                invalidation_patterns=["blog_posts", "campaigns", "ci_*"]
            ),
            
            # Dashboard queries - medium cache time
            "dashboard": CacheConfiguration(
                ttl=600,  # 10 minutes
                max_size=200,
                enable_compression=True,
                invalidation_patterns=["*"]
            ),
            
            # List queries - short cache time
            "lists": CacheConfiguration(
                ttl=300,  # 5 minutes
                max_size=500,
                enable_compression=False,
                invalidation_patterns=["*"]
            ),
            
            # Search queries - very short cache time
            "search": CacheConfiguration(
                ttl=120,  # 2 minutes
                max_size=1000,
                enable_compression=False
            ),
            
            # Competitor intelligence - medium cache time
            "competitor_intelligence": CacheConfiguration(
                ttl=900,  # 15 minutes
                max_size=300,
                enable_compression=True,
                invalidation_patterns=["ci_*"]
            ),
            
            # Reports - long cache time
            "reports": CacheConfiguration(
                ttl=3600,  # 1 hour
                max_size=50,
                enable_compression=True,
                invalidation_patterns=["*"]
            )
        }

    def _setup_invalidation_rules(self):
        """Setup cache invalidation rules based on table changes."""
        self.invalidation_rules = {
            "blog_posts": [
                "analytics:*",
                "dashboard:*",
                "lists:blog*",
                "search:*"
            ],
            "campaigns": [
                "analytics:*",
                "dashboard:*",
                "lists:campaign*",
                "search:*"
            ],
            "campaign_tasks": [
                "analytics:*",
                "dashboard:*",
                "lists:campaign*"
            ],
            "ci_competitors": [
                "competitor_intelligence:*",
                "dashboard:*",
                "lists:competitor*"
            ],
            "ci_content_items": [
                "competitor_intelligence:*",
                "analytics:competitor*"
            ],
            "ci_social_posts": [
                "competitor_intelligence:*",
                "analytics:social*"
            ]
        }

    def get_cache_config(self, query_type: str) -> CacheConfiguration:
        """Get cache configuration for a query type."""
        return self.cache_configs.get(query_type, CacheConfiguration())

    def update_cache_config(self, query_type: str, config: CacheConfiguration):
        """Update cache configuration for a query type."""
        self.cache_configs[query_type] = config
        logger.info(f"Updated cache config for {query_type}: TTL={config.ttl}s")

    def generate_cache_key(
        self, 
        query_type: str, 
        query_params: Dict[str, Any],
        table_names: List[str] = None
    ) -> str:
        """Generate cache key for query."""
        config = self.get_cache_config(query_type)
        
        # Base key components
        key_components = [query_type]
        
        # Add table names if provided
        if table_names:
            key_components.extend(sorted(table_names))
        
        # Add query parameters
        if config.cache_key_fields:
            # Use only specified fields
            for field in config.cache_key_fields:
                if field in query_params:
                    key_components.append(f"{field}:{query_params[field]}")
        else:
            # Use all parameters
            for key, value in sorted(query_params.items()):
                if isinstance(value, (str, int, float, bool, type(None))):
                    key_components.append(f"{key}:{value}")
                else:
                    # Hash complex objects
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()[:8]
                    key_components.append(f"{key}:{value_hash}")
        
        key_string = ":".join(str(comp) for comp in key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get_cached_result(
        self,
        query_type: str,
        query_params: Dict[str, Any],
        table_names: List[str] = None
    ) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self.generate_cache_key(query_type, query_params, table_names)
        
        try:
            cached_data = await cache.get(f"query_cache:{query_type}", cache_key)
            
            if cached_data:
                # Check if cache is still valid
                cached_at = cached_data.get('cached_at', 0)
                config = self.get_cache_config(query_type)
                
                if time.time() - cached_at < config.ttl:
                    logger.debug(f"Query cache hit: {query_type}:{cache_key[:8]}")
                    return cached_data.get('result')
                else:
                    # Cache expired, remove it
                    await cache.delete(f"query_cache:{query_type}", cache_key)
            
            return None
        
        except Exception as e:
            logger.warning(f"Cache read error for {query_type}: {e}")
            return None

    async def cache_result(
        self,
        query_type: str,
        query_params: Dict[str, Any],
        result: Any,
        execution_time_ms: int,
        table_names: List[str] = None,
        row_count: Optional[int] = None
    ) -> bool:
        """Cache query result with metadata."""
        config = self.get_cache_config(query_type)
        
        # Skip caching if tables are excluded
        if table_names and config.exclude_tables:
            if any(table in config.exclude_tables for table in table_names):
                return False
        
        cache_key = self.generate_cache_key(query_type, query_params, table_names)
        
        try:
            # Prepare cache data
            cache_data = {
                'result': result,
                'cached_at': time.time(),
                'execution_time_ms': execution_time_ms,
                'query_type': query_type,
                'table_names': table_names or [],
                'row_count': row_count,
                'cache_key': cache_key
            }
            
            # Compress if enabled and result is large
            if config.enable_compression and len(str(result)) > 1000:
                cache_data['compressed'] = True
            
            # Store in cache
            success = await cache.set(
                f"query_cache:{query_type}",
                cache_key,
                cache_data,
                ttl=config.ttl
            )
            
            if success and config.enable_metrics:
                # Record metrics
                await self._record_query_metrics(
                    query_type, execution_time_ms, False, 
                    len(str(result)), table_names or [], row_count
                )
            
            logger.debug(f"Cached query result: {query_type}:{cache_key[:8]}")
            return success
        
        except Exception as e:
            logger.warning(f"Cache write error for {query_type}: {e}")
            return False

    async def _record_query_metrics(
        self,
        query_type: str,
        execution_time_ms: int,
        cache_hit: bool,
        result_size: int,
        table_names: List[str],
        row_count: Optional[int] = None
    ):
        """Record query performance metrics."""
        metrics = QueryMetrics(
            query_id=f"{query_type}_{int(time.time() * 1000)}",
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            cache_hit=cache_hit,
            result_size=result_size,
            table_names=table_names,
            timestamp=datetime.utcnow(),
            row_count=row_count
        )
        
        self.query_metrics.append(metrics)
        
        # Keep metrics history manageable
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-500:]
        
        # Store metrics in cache for analytics
        try:
            await cache.set(
                "query_metrics",
                f"{query_type}:{int(time.time())}",
                asdict(metrics),
                ttl=86400  # 24 hours
            )
        except Exception as e:
            logger.warning(f"Failed to store query metrics: {e}")

    async def invalidate_cache_for_table(self, table_name: str) -> int:
        """Invalidate cache entries affected by table changes."""
        invalidated_count = 0
        
        # Get invalidation patterns for this table
        patterns = self.invalidation_rules.get(table_name, [])
        
        for pattern in patterns:
            count = await cache.invalidate_pattern(f"*query_cache*{pattern}*")
            invalidated_count += count
        
        logger.info(f"Invalidated {invalidated_count} cache entries for table {table_name}")
        return invalidated_count

    async def invalidate_query_type(self, query_type: str) -> int:
        """Invalidate all cache entries for a specific query type."""
        pattern = f"*query_cache:{query_type}*"
        count = await cache.invalidate_pattern(pattern)
        logger.info(f"Invalidated {count} cache entries for query type {query_type}")
        return count

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_queries = len(self.query_metrics)
        cache_hits = sum(1 for m in self.query_metrics if m.cache_hit)
        
        if total_queries > 0:
            cache_hit_rate = cache_hits / total_queries * 100
            avg_execution_time = sum(m.execution_time_ms for m in self.query_metrics) / total_queries
        else:
            cache_hit_rate = 0
            avg_execution_time = 0
        
        # Get statistics by query type
        by_type = {}
        for metrics in self.query_metrics:
            query_type = metrics.query_type
            if query_type not in by_type:
                by_type[query_type] = {
                    'total_queries': 0,
                    'cache_hits': 0,
                    'avg_execution_time': 0,
                    'total_execution_time': 0
                }
            
            by_type[query_type]['total_queries'] += 1
            by_type[query_type]['total_execution_time'] += metrics.execution_time_ms
            
            if metrics.cache_hit:
                by_type[query_type]['cache_hits'] += 1
        
        # Calculate averages
        for query_type, stats in by_type.items():
            if stats['total_queries'] > 0:
                stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_queries']
                stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries'] * 100
        
        return {
            'overall': {
                'total_queries': total_queries,
                'cache_hits': cache_hits,
                'cache_hit_rate': round(cache_hit_rate, 2),
                'avg_execution_time_ms': round(avg_execution_time, 2)
            },
            'by_query_type': by_type,
            'cache_configs': {k: asdict(v) for k, v in self.cache_configs.items()}
        }


def cached_query(
    query_type: str,
    table_names: List[str] = None,
    cache_config: Optional[CacheConfiguration] = None,
    key_fields: List[str] = None
):
    """
    Decorator for caching database query results.
    
    Args:
        query_type: Type of query (analytics, dashboard, etc.)
        table_names: List of table names involved in the query
        cache_config: Custom cache configuration
        key_fields: Specific fields to use in cache key generation
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create cache manager
            if not hasattr(wrapper, '_cache_manager'):
                wrapper._cache_manager = QueryCacheManager()
            
            cache_manager = wrapper._cache_manager
            
            # Apply custom config if provided
            if cache_config:
                cache_manager.update_cache_config(query_type, cache_config)
            
            # Update key fields if provided
            if key_fields:
                config = cache_manager.get_cache_config(query_type)
                config.cache_key_fields = key_fields
                cache_manager.update_cache_config(query_type, config)
            
            # Try to get from cache
            query_params = {**kwargs}
            if args:
                query_params['_args'] = args
            
            cached_result = await cache_manager.get_cached_result(
                query_type, query_params, table_names
            )
            
            if cached_result is not None:
                # Record cache hit metrics
                await cache_manager._record_query_metrics(
                    query_type, 1, True, len(str(cached_result)), table_names or []
                )
                logger.debug(f"Query cache hit: {func.__name__}")
                return cached_result
            
            # Cache miss - execute query
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Determine row count if result is a list
            row_count = len(result) if isinstance(result, list) else None
            
            # Cache the result
            await cache_manager.cache_result(
                query_type, query_params, result, 
                execution_time_ms, table_names, row_count
            )
            
            logger.debug(f"Query executed and cached: {func.__name__} ({execution_time_ms}ms)")
            return result
        
        # Add helper methods
        wrapper.invalidate_cache = lambda: wrapper._cache_manager.invalidate_query_type(query_type)
        wrapper.get_cache_stats = lambda: wrapper._cache_manager.get_cache_statistics()
        
        return wrapper
    return decorator


# Global query cache manager
global_query_cache = QueryCacheManager()


async def invalidate_table_cache(table_name: str) -> int:
    """Invalidate cache for a specific table."""
    return await global_query_cache.invalidate_cache_for_table(table_name)


async def get_query_cache_stats() -> Dict[str, Any]:
    """Get global query cache statistics."""
    return await global_query_cache.get_cache_statistics()


async def warm_critical_caches():
    """Pre-warm cache for critical queries."""
    # This would make requests to critical endpoints/queries to populate cache
    # Implementation would depend on your specific critical queries
    logger.info("Cache warming requested for critical queries")
    
    # Placeholder for warming specific caches
    return {"warmed_caches": 0}


# Example usage decorators for common query types
def cached_analytics_query(table_names: List[str] = None):
    """Decorator for analytics queries with optimized caching."""
    return cached_query(
        query_type="analytics",
        table_names=table_names,
        cache_config=CacheConfiguration(ttl=1800, enable_compression=True)
    )


def cached_dashboard_query(table_names: List[str] = None):
    """Decorator for dashboard queries with medium caching."""
    return cached_query(
        query_type="dashboard", 
        table_names=table_names,
        cache_config=CacheConfiguration(ttl=600)
    )


def cached_list_query(table_names: List[str] = None):
    """Decorator for list queries with short caching."""
    return cached_query(
        query_type="lists",
        table_names=table_names, 
        cache_config=CacheConfiguration(ttl=300)
    )