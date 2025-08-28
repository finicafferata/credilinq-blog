"""
Performance Optimization Module for LangGraph Agent System.

This module provides intelligent caching, parallel execution, and performance
monitoring capabilities for the enhanced LangGraph agent system.

Key features:
- Intelligent response caching with TTL and content-aware invalidation
- Parallel execution optimization with resource management
- Performance monitoring and metrics collection
- Retry logic with exponential backoff
- Database query optimization for workflow state management
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from collections import defaultdict, deque
import threading
import weakref

# Redis for distributed caching (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import logging
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Caching strategy options."""
    MEMORY_ONLY = "memory_only"
    REDIS_DISTRIBUTED = "redis_distributed"
    HYBRID = "hybrid"
    DISABLED = "disabled"

class ExecutionMode(Enum):
    """Execution mode options."""
    SEQUENTIAL = "sequential"
    PARALLEL_UNLIMITED = "parallel_unlimited"
    PARALLEL_LIMITED = "parallel_limited"
    ADAPTIVE = "adaptive"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    content_hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    parallel_executions: int = 0
    sequential_executions: int = 0
    retry_attempts: int = 0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

class IntelligentCache:
    """
    Intelligent caching system with TTL, content-aware invalidation,
    and distributed caching support.
    """
    
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.MEMORY_ONLY,
        default_ttl: int = 3600,  # 1 hour default
        max_memory_entries: int = 10000,
        redis_config: Optional[Dict[str, Any]] = None
    ):
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.max_memory_entries = max_memory_entries
        
        # Memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # Redis cache
        self._redis_client = None
        if strategy in [CacheStrategy.REDIS_DISTRIBUTED, CacheStrategy.HYBRID] and REDIS_AVAILABLE:
            self._initialize_redis(redis_config or {})
        
        # Cache statistics
        self._stats = PerformanceMetrics()
        
        # Background cleanup
        self._cleanup_task = None
        self._start_background_cleanup()
        
        logger.info(f"IntelligentCache initialized with strategy: {strategy.value}")
    
    def _initialize_redis(self, config: Dict[str, Any]):
        """Initialize Redis connection."""
        try:
            self._redis_client = redis.Redis(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                password=config.get("password"),
                decode_responses=True,
                socket_timeout=config.get("timeout", 5),
                retry_on_timeout=True
            )
            # Test connection
            self._redis_client.ping()
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed, falling back to memory cache: {e}")
            self._redis_client = None
    
    def _start_background_cleanup(self):
        """Start background cache cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Clean up every 5 minutes
                    self._cleanup_expired_entries()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop, skip background cleanup
            pass
    
    def _generate_cache_key(self, base_key: str, **kwargs) -> str:
        """Generate cache key with parameters."""
        if not kwargs:
            return base_key
        
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"{base_key}:{params_hash}"
    
    def _generate_content_hash(self, content: Any) -> str:
        """Generate content hash for cache invalidation."""
        try:
            content_str = json.dumps(content, sort_keys=True, default=str)
            return hashlib.md5(content_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(content).encode()).hexdigest()
    
    async def get(self, key: str, **kwargs) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_cache_key(key, **kwargs)
        
        try:
            # Try memory cache first
            with self._cache_lock:
                if cache_key in self._memory_cache:
                    entry = self._memory_cache[cache_key]
                    
                    # Check TTL
                    if self._is_expired(entry):
                        del self._memory_cache[cache_key]
                    else:
                        # Update access statistics
                        entry.access_count += 1
                        entry.last_accessed = datetime.utcnow()
                        self._stats.cache_hits += 1
                        logger.debug(f"Cache hit (memory): {cache_key}")
                        return entry.value
            
            # Try Redis cache if available
            if self._redis_client and self.strategy in [CacheStrategy.REDIS_DISTRIBUTED, CacheStrategy.HYBRID]:
                try:
                    cached_data = self._redis_client.get(f"langgraph:{cache_key}")
                    if cached_data:
                        entry_data = json.loads(cached_data)
                        
                        # Check TTL
                        created_at = datetime.fromisoformat(entry_data["created_at"])
                        if datetime.utcnow() - created_at < timedelta(seconds=entry_data["ttl_seconds"]):
                            # Cache to memory for faster access
                            if len(self._memory_cache) < self.max_memory_entries:
                                with self._cache_lock:
                                    self._memory_cache[cache_key] = CacheEntry(
                                        key=cache_key,
                                        value=entry_data["value"],
                                        created_at=created_at,
                                        ttl_seconds=entry_data["ttl_seconds"],
                                        content_hash=entry_data.get("content_hash")
                                    )
                            
                            self._stats.cache_hits += 1
                            logger.debug(f"Cache hit (redis): {cache_key}")
                            return entry_data["value"]
                        else:
                            # Expired, remove from Redis
                            self._redis_client.delete(f"langgraph:{cache_key}")
                except Exception as e:
                    logger.warning(f"Redis cache get error: {e}")
            
            # Cache miss
            self._stats.cache_misses += 1
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        """Set value in cache."""
        cache_key = self._generate_cache_key(key, **kwargs)
        ttl = ttl or self.default_ttl
        tags = tags or []
        
        try:
            content_hash = self._generate_content_hash(value)
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl,
                content_hash=content_hash,
                tags=tags
            )
            
            # Store in memory cache
            with self._cache_lock:
                # Evict old entries if at capacity
                if len(self._memory_cache) >= self.max_memory_entries:
                    self._evict_lru_entries(int(self.max_memory_entries * 0.1))
                
                self._memory_cache[cache_key] = entry
            
            # Store in Redis cache
            if self._redis_client and self.strategy in [CacheStrategy.REDIS_DISTRIBUTED, CacheStrategy.HYBRID]:
                try:
                    entry_data = {
                        "value": value,
                        "created_at": entry.created_at.isoformat(),
                        "ttl_seconds": ttl,
                        "content_hash": content_hash,
                        "tags": tags
                    }
                    
                    self._redis_client.setex(
                        f"langgraph:{cache_key}",
                        ttl,
                        json.dumps(entry_data, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Redis cache set error: {e}")
            
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def invalidate(self, key_pattern: str = None, tags: List[str] = None):
        """Invalidate cache entries by pattern or tags."""
        try:
            if key_pattern:
                # Invalidate by key pattern
                with self._cache_lock:
                    keys_to_remove = [k for k in self._memory_cache.keys() if key_pattern in k]
                    for key in keys_to_remove:
                        del self._memory_cache[key]
                
                if self._redis_client:
                    try:
                        pattern = f"langgraph:*{key_pattern}*"
                        keys = self._redis_client.keys(pattern)
                        if keys:
                            self._redis_client.delete(*keys)
                    except Exception as e:
                        logger.warning(f"Redis invalidation error: {e}")
            
            if tags:
                # Invalidate by tags
                with self._cache_lock:
                    keys_to_remove = []
                    for key, entry in self._memory_cache.items():
                        if any(tag in entry.tags for tag in tags):
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self._memory_cache[key]
            
            logger.info(f"Cache invalidated: pattern={key_pattern}, tags={tags}")
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() - entry.created_at > timedelta(seconds=entry.ttl_seconds)
    
    def _evict_lru_entries(self, count: int):
        """Evict least recently used entries."""
        if not self._memory_cache:
            return
        
        # Sort by last accessed time
        entries = list(self._memory_cache.items())
        entries.sort(key=lambda x: x[1].last_accessed or x[1].created_at)
        
        for i in range(min(count, len(entries))):
            key, _ = entries[i]
            del self._memory_cache[key]
    
    def _cleanup_expired_entries(self):
        """Clean up expired entries from memory cache."""
        with self._cache_lock:
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> PerformanceMetrics:
        """Get cache statistics."""
        total_requests = self._stats.cache_hits + self._stats.cache_misses
        self._stats.cache_hit_rate = self._stats.cache_hits / total_requests if total_requests > 0 else 0.0
        self._stats.last_updated = datetime.utcnow()
        return self._stats

class ParallelExecutionManager:
    """
    Manages parallel execution of agent tasks with resource optimization
    and adaptive scaling.
    """
    
    def __init__(
        self,
        max_workers: int = None,
        execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        self.max_workers = max_workers or min(32, (asyncio.cpu_count() or 1) + 4)
        self.execution_mode = execution_mode
        self.resource_limits = resource_limits or {}
        
        # Execution tracking
        self._active_tasks = 0
        self._task_queue = deque()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self._semaphore = asyncio.Semaphore(self.max_workers)
        
        # Performance tracking
        self._metrics = PerformanceMetrics()
        self._execution_history = deque(maxlen=100)
        
        logger.info(f"ParallelExecutionManager initialized: mode={execution_mode.value}, max_workers={self.max_workers}")
    
    async def execute_parallel(
        self,
        tasks: List[Callable],
        task_data: List[Any],
        timeout: Optional[float] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Execute multiple tasks in parallel with resource management."""
        if not tasks or not task_data:
            return []
        
        if len(tasks) != len(task_data):
            raise ValueError("Tasks and task_data must have the same length")
        
        max_concurrent = max_concurrent or self.max_workers
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_task(task, data, index):
            async with semaphore:
                start_time = time.time()
                try:
                    # Execute in thread pool for CPU-bound tasks
                    if asyncio.iscoroutinefunction(task):
                        result = await task(data)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(self._executor, task, data)
                    
                    execution_time = (time.time() - start_time) * 1000
                    self._record_execution(execution_time, success=True)
                    
                    return {"index": index, "result": result, "success": True, "error": None}
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    self._record_execution(execution_time, success=False)
                    logger.error(f"Parallel task {index} failed: {e}")
                    
                    return {"index": index, "result": None, "success": False, "error": str(e)}
        
        # Create task futures
        futures = [
            execute_single_task(task, data, i)
            for i, (task, data) in enumerate(zip(tasks, task_data))
        ]
        
        # Execute with timeout
        try:
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*futures, return_exceptions=True),
                    timeout=timeout
                )
            else:
                results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Sort results by original index
            sorted_results = [None] * len(results)
            for result in results:
                if isinstance(result, dict) and "index" in result:
                    sorted_results[result["index"]] = result
                elif isinstance(result, Exception):
                    logger.error(f"Parallel execution exception: {result}")
            
            self._metrics.parallel_executions += 1
            return sorted_results
            
        except asyncio.TimeoutError:
            logger.error(f"Parallel execution timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
    
    async def execute_with_dependencies(
        self,
        task_graph: Dict[str, Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute tasks with dependency resolution."""
        completed_tasks = {}
        in_progress_tasks = {}
        pending_tasks = dict(task_graph)
        
        async def can_execute_task(task_id: str, task_info: Dict[str, Any]) -> bool:
            dependencies = task_info.get("dependencies", [])
            return all(dep_id in completed_tasks for dep_id in dependencies)
        
        async def execute_ready_tasks():
            ready_tasks = []
            for task_id, task_info in list(pending_tasks.items()):
                if await can_execute_task(task_id, task_info):
                    ready_tasks.append((task_id, task_info))
                    del pending_tasks[task_id]
            
            if ready_tasks:
                # Execute ready tasks in parallel
                async def execute_task(task_id, task_info):
                    try:
                        task_func = task_info["task"]
                        task_data = task_info.get("data", {})
                        
                        # Add dependency results to task data
                        dependency_results = {}
                        for dep_id in task_info.get("dependencies", []):
                            if dep_id in completed_tasks:
                                dependency_results[dep_id] = completed_tasks[dep_id]
                        
                        task_data["dependency_results"] = dependency_results
                        
                        if asyncio.iscoroutinefunction(task_func):
                            result = await task_func(task_data)
                        else:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(self._executor, task_func, task_data)
                        
                        completed_tasks[task_id] = result
                        
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}")
                        completed_tasks[task_id] = {"error": str(e), "success": False}
                
                # Execute all ready tasks in parallel
                await asyncio.gather(*[execute_task(task_id, task_info) for task_id, task_info in ready_tasks])
        
        # Execute until all tasks are completed or no progress can be made
        start_time = time.time()
        while pending_tasks:
            initial_pending_count = len(pending_tasks)
            await execute_ready_tasks()
            
            # Check for progress
            if len(pending_tasks) == initial_pending_count:
                # No progress made - circular dependency or missing dependency
                logger.error(f"Dependency deadlock detected. Remaining tasks: {list(pending_tasks.keys())}")
                break
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.error(f"Dependency execution timed out after {timeout}s")
                break
        
        return completed_tasks
    
    def _record_execution(self, execution_time_ms: float, success: bool):
        """Record execution metrics."""
        self._metrics.execution_count += 1
        self._metrics.total_execution_time_ms += execution_time_ms
        self._metrics.avg_execution_time_ms = self._metrics.total_execution_time_ms / self._metrics.execution_count
        
        if not success:
            self._metrics.error_count += 1
        
        self._execution_history.append({
            "timestamp": datetime.utcnow(),
            "execution_time_ms": execution_time_ms,
            "success": success
        })
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get execution metrics."""
        self._metrics.last_updated = datetime.utcnow()
        return self._metrics
    
    def adaptive_scaling(self) -> int:
        """Calculate optimal worker count based on recent performance."""
        if len(self._execution_history) < 10:
            return self.max_workers
        
        recent_executions = list(self._execution_history)[-10:]
        avg_time = sum(e["execution_time_ms"] for e in recent_executions) / len(recent_executions)
        success_rate = sum(1 for e in recent_executions if e["success"]) / len(recent_executions)
        
        # Scale based on performance
        if avg_time < 1000 and success_rate > 0.95:  # Fast and reliable
            return min(self.max_workers * 2, 64)
        elif avg_time > 5000 or success_rate < 0.8:  # Slow or unreliable
            return max(self.max_workers // 2, 1)
        else:
            return self.max_workers

class RetryManager:
    """
    Advanced retry logic with exponential backoff and failure analysis.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self._retry_stats = defaultdict(int)
        self._failure_patterns = defaultdict(list)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_on_exceptions: Optional[Tuple] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        max_retries = max_retries or self.max_retries
        retry_on_exceptions = retry_on_exceptions or (Exception,)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                if attempt > 0:
                    self._retry_stats["successful_retries"] += 1
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except retry_on_exceptions as e:
                last_exception = e
                self._retry_stats["total_retries"] += 1
                self._failure_patterns[type(e).__name__].append(datetime.utcnow())
                
                if attempt < max_retries:
                    # Calculate delay
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    # Add jitter
                    if self.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Function failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Function failed after {max_retries + 1} attempts: {e}")
                    self._retry_stats["failed_after_retries"] += 1
                    raise
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failure patterns."""
        analysis = {
            "retry_stats": dict(self._retry_stats),
            "failure_patterns": {},
            "recommendations": []
        }
        
        # Analyze failure patterns
        for exception_type, timestamps in self._failure_patterns.items():
            recent_failures = [
                ts for ts in timestamps
                if datetime.utcnow() - ts < timedelta(hours=1)
            ]
            
            analysis["failure_patterns"][exception_type] = {
                "total_count": len(timestamps),
                "recent_count": len(recent_failures),
                "last_occurrence": max(timestamps).isoformat() if timestamps else None
            }
            
            # Generate recommendations
            if len(recent_failures) > 5:
                analysis["recommendations"].append(
                    f"High frequency of {exception_type} failures detected. Consider investigating root cause."
                )
        
        return analysis

class PerformanceOptimizer:
    """
    Main performance optimization coordinator that integrates caching,
    parallel execution, and monitoring.
    """
    
    def __init__(
        self,
        cache_config: Optional[Dict[str, Any]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize components
        self.cache = IntelligentCache(**(cache_config or {}))
        self.parallel_executor = ParallelExecutionManager(**(execution_config or {}))
        self.retry_manager = RetryManager(**(retry_config or {}))
        
        # Performance monitoring
        self._global_metrics = PerformanceMetrics()
        self._operation_timings = defaultdict(list)
        
        logger.info("PerformanceOptimizer initialized")
    
    async def cached_execute(
        self,
        func: Callable,
        cache_key: str,
        cache_ttl: int = 3600,
        cache_tags: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with intelligent caching."""
        # Try cache first
        cached_result = await self.cache.get(cache_key, **kwargs)
        if cached_result is not None:
            return cached_result
        
        # Execute function
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await self.cache.set(cache_key, result, ttl=cache_ttl, tags=cache_tags, **kwargs)
            
            execution_time = (time.time() - start_time) * 1000
            self._record_operation_timing("cached_execute", execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_operation_timing("cached_execute_failed", execution_time)
            raise
    
    async def parallel_cached_execute(
        self,
        operations: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Execute multiple operations in parallel with caching."""
        async def execute_operation(operation):
            func = operation["func"]
            cache_key = operation.get("cache_key")
            args = operation.get("args", [])
            kwargs = operation.get("kwargs", {})
            
            if cache_key:
                return await self.cached_execute(
                    func,
                    cache_key,
                    operation.get("cache_ttl", 3600),
                    operation.get("cache_tags"),
                    *args,
                    **kwargs
                )
            else:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        # Execute operations in parallel
        tasks = [execute_operation(op) for op in operations]
        functions = [execute_operation] * len(operations)
        
        results = await self.parallel_executor.execute_parallel(
            functions,
            operations,
            timeout=timeout,
            max_concurrent=max_concurrent
        )
        
        return [r["result"] if r and r["success"] else None for r in results]
    
    async def execute_with_full_optimization(
        self,
        func: Callable,
        cache_key: Optional[str] = None,
        retry_enabled: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with full optimization (caching, retry, monitoring)."""
        start_time = time.time()
        operation_name = getattr(func, "__name__", "unknown_operation")
        
        try:
            # Define execution function
            async def execute_func():
                if cache_key:
                    return await self.cached_execute(func, cache_key, cache_ttl, None, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
            
            # Execute with or without retry
            if retry_enabled:
                result = await self.retry_manager.execute_with_retry(
                    execute_func,
                    max_retries=max_retries
                )
            else:
                result = await execute_func()
            
            # Record success metrics
            execution_time = (time.time() - start_time) * 1000
            self._record_operation_timing(operation_name, execution_time)
            self._global_metrics.execution_count += 1
            self._global_metrics.total_execution_time_ms += execution_time
            
            return result
            
        except Exception as e:
            # Record failure metrics
            execution_time = (time.time() - start_time) * 1000
            self._record_operation_timing(f"{operation_name}_failed", execution_time)
            self._global_metrics.error_count += 1
            
            logger.error(f"Optimized execution failed for {operation_name}: {e}")
            raise
    
    async def optimize_database_queries(
        self,
        queries: List[Callable],
        query_data: List[Any],
        cache_results: bool = True,
        cache_ttl: int = 300  # 5 minutes for DB queries
    ) -> List[Any]:
        """Optimize database queries with parallel execution and caching."""
        if cache_results:
            # Create cache keys for DB queries
            operations = []
            for i, (query, data) in enumerate(zip(queries, query_data)):
                query_name = getattr(query, "__name__", f"query_{i}")
                cache_key = f"db_query:{query_name}"
                
                operations.append({
                    "func": query,
                    "args": [data],
                    "cache_key": cache_key,
                    "cache_ttl": cache_ttl,
                    "cache_tags": ["database_query"]
                })
            
            return await self.parallel_cached_execute(operations)
        else:
            return await self.parallel_executor.execute_parallel(queries, query_data)
    
    def _record_operation_timing(self, operation: str, timing_ms: float):
        """Record operation timing for analysis."""
        self._operation_timings[operation].append({
            "timestamp": datetime.utcnow(),
            "timing_ms": timing_ms
        })
        
        # Keep only recent timings (last 100 per operation)
        if len(self._operation_timings[operation]) > 100:
            self._operation_timings[operation] = self._operation_timings[operation][-100:]
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Update global metrics
        if self._global_metrics.execution_count > 0:
            self._global_metrics.avg_execution_time_ms = (
                self._global_metrics.total_execution_time_ms / self._global_metrics.execution_count
            )
        
        # Calculate operation statistics
        operation_stats = {}
        for operation, timings in self._operation_timings.items():
            if timings:
                times = [t["timing_ms"] for t in timings]
                operation_stats[operation] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "recent_trend": "improving" if len(times) > 10 and times[-5:] < times[-10:-5] else "stable"
                }
        
        return {
            "global_metrics": self._global_metrics,
            "cache_metrics": self.cache.get_stats(),
            "parallel_execution_metrics": self.parallel_executor.get_metrics(),
            "retry_analysis": self.retry_manager.get_failure_analysis(),
            "operation_statistics": operation_stats,
            "recommendations": self._generate_recommendations(operation_stats)
        }
    
    def _generate_recommendations(self, operation_stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Cache hit rate recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats.cache_hit_rate < 0.5:
            recommendations.append("Cache hit rate is low. Consider increasing TTL or reviewing cache keys.")
        
        # Slow operations recommendations
        for operation, stats in operation_stats.items():
            if stats["avg_ms"] > 5000:
                recommendations.append(f"Operation '{operation}' is slow (avg: {stats['avg_ms']:.2f}ms). Consider optimization.")
        
        # Error rate recommendations
        if self._global_metrics.error_count > 0:
            error_rate = self._global_metrics.error_count / max(self._global_metrics.execution_count, 1)
            if error_rate > 0.1:
                recommendations.append(f"High error rate detected ({error_rate:.2%}). Review error patterns.")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.cache, '_cleanup_task') and self.cache._cleanup_task:
            self.cache._cleanup_task.cancel()
        
        if hasattr(self.parallel_executor, '_executor'):
            self.parallel_executor._executor.shutdown(wait=True)
        
        logger.info("PerformanceOptimizer cleanup completed")

# Global instance for easy access
_global_optimizer = None

def get_performance_optimizer(**kwargs) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(**kwargs)
    return _global_optimizer

# Export key classes and functions
__all__ = [
    'PerformanceOptimizer',
    'IntelligentCache',
    'ParallelExecutionManager',
    'RetryManager',
    'CacheStrategy',
    'ExecutionMode',
    'PerformanceMetrics',
    'get_performance_optimizer'
]