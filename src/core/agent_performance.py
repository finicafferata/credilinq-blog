"""
Comprehensive Agent Performance Tracking System
Provides performance monitoring, caching, and optimization for AI agents.
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Union

from ..services.redis_cache_service import get_agent_cache, get_main_cache
from ..services.enhanced_database_service import get_cached_db_service

logger = logging.getLogger(__name__)

@dataclass
class AgentCacheConfig:
    """Configuration for agent caching behavior."""
    ttl: int = 3600  # 1 hour default
    enable_content_based_caching: bool = True
    cache_key_fields: Optional[List[str]] = None
    max_cache_entries: int = 1000
    invalidation_patterns: List[str] = None

    def __post_init__(self):
        if self.invalidation_patterns is None:
            self.invalidation_patterns = []

@dataclass
class AgentExecutionMetrics:
    """Metrics collected during agent execution."""
    agent_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cache_hit: bool = False
    error_occurred: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return not self.error_occurred

class AgentPerformanceTracker:
    """
    Centralized performance tracking and caching system for AI agents.
    """
    
    def __init__(self):
        self.cache_service = get_agent_cache()
        self.main_cache = get_main_cache()
        self.db_service = get_cached_db_service()
        self.cache_configs: Dict[str, AgentCacheConfig] = {}
        self.active_executions: Dict[str, AgentExecutionMetrics] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def configure_agent_cache(self, agent_name: str, config: AgentCacheConfig):
        """Configure caching for a specific agent."""
        self.cache_configs[agent_name] = config
        logger.info(f"Configured cache for agent {agent_name}: TTL={config.ttl}s")

    def start_execution(self, agent_name: str, input_data: Any, **metadata) -> str:
        """Start tracking an agent execution."""
        execution_id = f"{agent_name}_{int(time.time() * 1000)}_{hash(str(input_data)) % 10000}"
        
        metrics = AgentExecutionMetrics(
            agent_name=agent_name,
            execution_id=execution_id,
            start_time=datetime.now(),
            metadata=metadata
        )
        
        self.active_executions[execution_id] = metrics
        logger.debug(f"Started tracking execution {execution_id} for agent {agent_name}")
        return execution_id

    def end_execution(self, execution_id: str, result: Any = None, error: Optional[str] = None,
                     input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                     cost_usd: Optional[float] = None, quality_score: Optional[float] = None) -> Optional[AgentExecutionMetrics]:
        """End tracking an agent execution."""
        if execution_id not in self.active_executions:
            logger.warning(f"No active execution found for ID: {execution_id}")
            return None
        
        metrics = self.active_executions[execution_id]
        metrics.end_time = datetime.now()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        metrics.input_tokens = input_tokens
        metrics.output_tokens = output_tokens
        metrics.cost_usd = cost_usd
        
        if error:
            metrics.error_occurred = True
            metrics.error_message = error
        
        # Store in database asynchronously
        self.executor.submit(self._store_performance_metrics, metrics, quality_score)
        
        # Remove from active executions
        del self.active_executions[execution_id]
        
        logger.debug(f"Ended tracking execution {execution_id}: {metrics.duration_ms:.2f}ms")
        return metrics

    def _store_performance_metrics(self, metrics: AgentExecutionMetrics, quality_score: Optional[float] = None):
        """Store performance metrics in database."""
        try:
            from ..agents.core.database_service import AgentPerformanceMetrics as DBMetrics
            
            # Use real quality score if provided, otherwise use success-based fallback
            real_quality_score = quality_score if quality_score is not None else (8.0 if metrics.success else 2.0)
            
            # Convert 0-1 scale to 0-10 scale if needed (Quality Review Agent uses 0-1)
            if real_quality_score <= 1.0 and real_quality_score >= 0.0:
                real_quality_score = real_quality_score * 10.0
            
            db_metrics = DBMetrics(
                agent_type=metrics.agent_name,
                task_type="execution",
                execution_time_ms=int(metrics.duration_ms or 0),
                success_rate=1.0 if metrics.success else 0.0,
                quality_score=real_quality_score,  # Use real agent-calculated score
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                cost_usd=metrics.cost_usd,
                error_count=1 if metrics.error_occurred else 0
            )
            
            self.db_service.log_agent_performance(db_metrics)
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {str(e)}")

    def generate_cache_key(self, agent_name: str, input_data: Any, **kwargs) -> str:
        """Generate consistent cache key for agent execution."""
        config = self.cache_configs.get(agent_name, AgentCacheConfig())
        
        # Base key components
        key_data = {
            'agent': agent_name,
            'input': input_data
        }
        
        # Add specific fields if configured
        if config.cache_key_fields:
            for field in config.cache_key_fields:
                if field in kwargs:
                    key_data[field] = kwargs[field]
        
        # Generate hash for consistent key
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"agent_execution:{agent_name}:{key_hash}"

    def get_cached_result(self, agent_name: str, input_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result for agent execution."""
        config = self.cache_configs.get(agent_name, AgentCacheConfig())
        if not config.enable_content_based_caching:
            return None
        
        cache_key = self.generate_cache_key(agent_name, input_data, **kwargs)
        cached_result = self.cache_service.get('agent_results', cache_key)
        
        if cached_result:
            logger.debug(f"Cache hit for agent {agent_name}")
            return cached_result
        
        logger.debug(f"Cache miss for agent {agent_name}")
        return None

    def cache_result(self, agent_name: str, input_data: Any, result: Any, 
                    execution_time_ms: float, **kwargs) -> bool:
        """Cache agent execution result."""
        config = self.cache_configs.get(agent_name, AgentCacheConfig())
        if not config.enable_content_based_caching:
            return False
        
        cache_key = self.generate_cache_key(agent_name, input_data, **kwargs)
        cache_data = {
            'result': result,
            'execution_time_ms': execution_time_ms,
            'cached_at': datetime.now().isoformat(),
            'agent_name': agent_name,
            'metadata': kwargs
        }
        
        success = self.cache_service.cache_agent_result(
            agent_name, cache_key, cache_data, int(execution_time_ms), config.ttl
        )
        
        if success:
            logger.debug(f"Cached result for agent {agent_name}")
        
        return success

    async def get_agent_analytics(self, agent_name: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive analytics for an agent."""
        try:
            # Get performance data from database
            performance_data = self.db_service.get_agent_performance_analytics(agent_name, days)
            
            # Calculate aggregated metrics
            total_executions = len(performance_data)
            if total_executions == 0:
                return {
                    'agent_name': agent_name,
                    'total_executions': 0,
                    'avg_execution_time_ms': 0,
                    'success_rate': 0,
                    'total_cost_usd': 0,
                    'cache_stats': await self._get_agent_cache_stats(agent_name)
                }
            
            successful_executions = sum(1 for p in performance_data if p.get('success_rate', 0) > 0.5)
            total_time = sum(p.get('execution_time_ms', 0) for p in performance_data)
            total_cost = sum(p.get('cost_usd', 0) for p in performance_data if p.get('cost_usd'))
            
            return {
                'agent_name': agent_name,
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': (successful_executions / total_executions) * 100,
                'avg_execution_time_ms': total_time / total_executions,
                'total_execution_time_ms': total_time,
                'total_cost_usd': total_cost,
                'avg_cost_per_execution': total_cost / total_executions if total_cost > 0 else 0,
                'cache_stats': await self._get_agent_cache_stats(agent_name),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent analytics for {agent_name}: {str(e)}")
            return {'error': str(e)}

    async def _get_agent_cache_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get cache statistics for an agent."""
        try:
            cache_stats = self.cache_service.get_stats()
            return {
                'hit_rate': cache_stats.get('hit_rate', 0),
                'total_requests': cache_stats.get('total_requests', 0),
                'avg_response_time_ms': cache_stats.get('avg_response_time_ms', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats for {agent_name}: {str(e)}")
            return {}

    async def invalidate_agent_cache(self, agent_name: str, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries for an agent."""
        try:
            if pattern:
                # Invalidate specific pattern
                namespace = f"agent_results:{agent_name}:{pattern}"
                return self.cache_service.invalidate_namespace(namespace)
            else:
                # Invalidate all entries for agent
                namespace = f"agent_results:{agent_name}"
                return self.cache_service.invalidate_namespace(namespace)
                
        except Exception as e:
            logger.error(f"Failed to invalidate cache for {agent_name}: {str(e)}")
            return 0

    def update_cache_config(self, agent_name: str, config: AgentCacheConfig):
        """Update cache configuration for an agent."""
        self.cache_configs[agent_name] = config
        logger.info(f"Updated cache configuration for {agent_name}")

    async def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        try:
            # Get cache performance
            cache_performance = self.db_service.get_cache_performance()
            
            # Get active executions count
            active_count = len(self.active_executions)
            
            # Get database health
            db_health = self.db_service.health_check()
            
            return {
                'active_executions': active_count,
                'cache_performance': cache_performance,
                'database_health': db_health,
                'configured_agents': list(self.cache_configs.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system performance: {str(e)}")
            return {'error': str(e)}

def cached_agent_execution(
    agent_name: str,
    ttl: int = 3600,
    enable_caching: bool = True,
    cache_key_fields: Optional[List[str]] = None
):
    """
    Decorator for caching agent execution results with performance tracking.
    
    Args:
        agent_name: Name of the agent for tracking
        ttl: Cache TTL in seconds
        enable_caching: Whether to enable caching
        cache_key_fields: Specific fields to include in cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracker = global_performance_tracker
            
            # Configure caching if not already done
            if agent_name not in tracker.cache_configs:
                config = AgentCacheConfig(
                    ttl=ttl,
                    enable_content_based_caching=enable_caching,
                    cache_key_fields=cache_key_fields
                )
                tracker.configure_agent_cache(agent_name, config)
            
            # Extract input data (assume first arg or 'input_data' kwarg)
            input_data = args[0] if args else kwargs.get('input_data', {})
            
            # Try to get cached result first
            if enable_caching:
                cached_result = tracker.get_cached_result(agent_name, input_data, **kwargs)
                if cached_result:
                    logger.info(f"Returning cached result for agent {agent_name}")
                    return cached_result['result']
            
            # Start performance tracking
            execution_id = tracker.start_execution(agent_name, input_data, **kwargs)
            
            try:
                # Execute function
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Extract quality score from result if it's a ReviewAgentResult
                quality_score = None
                if hasattr(result, 'automated_score'):
                    quality_score = result.automated_score
                elif isinstance(result, dict) and 'automated_score' in result:
                    quality_score = result['automated_score']
                
                # End performance tracking
                tracker.end_execution(execution_id, result, quality_score=quality_score)
                
                # Cache the result
                if enable_caching:
                    tracker.cache_result(agent_name, input_data, result, execution_time_ms, **kwargs)
                
                return result
                
            except Exception as e:
                # End performance tracking with error
                tracker.end_execution(execution_id, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracker = global_performance_tracker
            
            # Configure caching if not already done
            if agent_name not in tracker.cache_configs:
                config = AgentCacheConfig(
                    ttl=ttl,
                    enable_content_based_caching=enable_caching,
                    cache_key_fields=cache_key_fields
                )
                tracker.configure_agent_cache(agent_name, config)
            
            # Extract input data
            input_data = args[0] if args else kwargs.get('input_data', {})
            
            # Try to get cached result first
            if enable_caching:
                cached_result = tracker.get_cached_result(agent_name, input_data, **kwargs)
                if cached_result:
                    logger.info(f"Returning cached result for agent {agent_name}")
                    return cached_result['result']
            
            # Start performance tracking
            execution_id = tracker.start_execution(agent_name, input_data, **kwargs)
            
            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Extract quality score from result if it's a ReviewAgentResult
                quality_score = None
                if hasattr(result, 'automated_score'):
                    quality_score = result.automated_score
                elif isinstance(result, dict) and 'automated_score' in result:
                    quality_score = result['automated_score']
                
                # End performance tracking
                tracker.end_execution(execution_id, result, quality_score=quality_score)
                
                # Cache the result
                if enable_caching:
                    tracker.cache_result(agent_name, input_data, result, execution_time_ms, **kwargs)
                
                return result
                
            except Exception as e:
                # End performance tracking with error
                tracker.end_execution(execution_id, error=str(e))
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global performance tracker instance
global_performance_tracker = AgentPerformanceTracker()

# Convenience functions
def get_performance_tracker() -> AgentPerformanceTracker:
    """Get the global performance tracker instance."""
    return global_performance_tracker

async def get_agent_performance(agent_name: str, days: int = 7) -> Dict[str, Any]:
    """Get performance analytics for a specific agent."""
    return await global_performance_tracker.get_agent_analytics(agent_name, days)

async def get_system_performance() -> Dict[str, Any]:
    """Get overall system performance metrics."""
    return await global_performance_tracker.get_system_performance()

def configure_agent_caching(agent_name: str, ttl: int = 3600, 
                           enable_caching: bool = True,
                           cache_key_fields: Optional[List[str]] = None):
    """Configure caching for a specific agent."""
    config = AgentCacheConfig(
        ttl=ttl,
        enable_content_based_caching=enable_caching,
        cache_key_fields=cache_key_fields
    )
    global_performance_tracker.configure_agent_cache(agent_name, config)