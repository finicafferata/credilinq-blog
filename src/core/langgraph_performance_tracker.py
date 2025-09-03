"""
LangGraph Performance Tracking Integration
High-performance, non-blocking agent performance tracking for LangGraph workflows.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from functools import wraps
import logging

from ..config.database import db_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Real-time execution metrics for agent performance tracking"""
    agent_name: str
    agent_type: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    status: str = "running"
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    campaign_id: Optional[str] = None
    blog_post_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionMetrics:
    """Agent decision tracking with reasoning and confidence"""
    performance_id: str
    decision_point: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: str
    confidence_score: float
    alternatives_considered: List[str]
    execution_time_ms: int
    tokens_used: Optional[int] = None
    decision_latency_s: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncPerformanceTracker:
    """
    High-performance async performance tracker for LangGraph workflows.
    Uses background tasks to avoid blocking agent execution.
    """
    
    def __init__(self, batch_size: int = 50, flush_interval: float = 30.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.performance_queue: List[ExecutionMetrics] = []
        self.decision_queue: List[DecisionMetrics] = []
        self._background_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the background performance tracking task"""
        if self._running:
            return
        
        self._running = True
        self._background_task = asyncio.create_task(self._background_processor())
        logger.info("AsyncPerformanceTracker started")
    
    async def stop(self):
        """Stop background processing and flush remaining data"""
        self._running = False
        if self._background_task:
            await self._background_task
        
        # Flush any remaining data
        await self._flush_queues(force=True)
        logger.info("AsyncPerformanceTracker stopped")
    
    async def track_execution_start(self, 
                                   agent_name: str,
                                   agent_type: str,
                                   campaign_id: Optional[str] = None,
                                   blog_post_id: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an agent execution (non-blocking)"""
        execution_id = str(uuid.uuid4())
        
        metrics = ExecutionMetrics(
            agent_name=agent_name,
            agent_type=agent_type,
            execution_id=execution_id,
            start_time=datetime.now(timezone.utc),
            campaign_id=campaign_id,
            blog_post_id=blog_post_id,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.performance_queue.append(metrics)
        
        # Trigger immediate flush if queue is full
        if len(self.performance_queue) >= self.batch_size:
            asyncio.create_task(self._flush_queues())
        
        return execution_id
    
    async def track_execution_end(self,
                                 execution_id: str,
                                 status: str = "success",
                                 input_tokens: Optional[int] = None,
                                 output_tokens: Optional[int] = None,
                                 cost: Optional[float] = None,
                                 error_message: Optional[str] = None,
                                 error_code: Optional[str] = None,
                                 retry_count: int = 0):
        """Complete tracking an agent execution (non-blocking)"""
        async with self._lock:
            # Find and update the existing metrics
            for metrics in self.performance_queue:
                if metrics.execution_id == execution_id:
                    metrics.end_time = datetime.now(timezone.utc)
                    metrics.duration_ms = int((metrics.end_time - metrics.start_time).total_seconds() * 1000)
                    metrics.status = status
                    metrics.input_tokens = input_tokens
                    metrics.output_tokens = output_tokens
                    metrics.total_tokens = (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None
                    metrics.cost = cost
                    metrics.error_message = error_message
                    metrics.error_code = error_code
                    metrics.retry_count = retry_count
                    break
    
    async def track_decision(self,
                           execution_id: str,
                           decision_point: str,
                           input_data: Dict[str, Any],
                           output_data: Dict[str, Any],
                           reasoning: str,
                           confidence_score: float,
                           alternatives_considered: List[str],
                           execution_time_ms: int,
                           tokens_used: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Track an agent decision with reasoning (non-blocking)"""
        
        # Find the performance ID from execution_id
        performance_id = None
        async with self._lock:
            for metrics in self.performance_queue:
                if metrics.execution_id == execution_id:
                    performance_id = execution_id  # Use execution_id as performance_id for simplicity
                    break
        
        if not performance_id:
            logger.warning(f"No performance record found for execution_id: {execution_id}")
            return
        
        decision_metrics = DecisionMetrics(
            performance_id=performance_id,
            decision_point=decision_point,
            input_data=input_data,
            output_data=output_data,
            reasoning=reasoning,
            confidence_score=confidence_score,
            alternatives_considered=alternatives_considered,
            execution_time_ms=execution_time_ms,
            tokens_used=tokens_used,
            decision_latency_s=execution_time_ms / 1000.0,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.decision_queue.append(decision_metrics)
        
        # Trigger flush if needed
        if len(self.decision_queue) >= self.batch_size:
            asyncio.create_task(self._flush_queues())
    
    async def _background_processor(self):
        """Background task that periodically flushes queues"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_queues()
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
    
    async def _flush_queues(self, force: bool = False):
        """Flush pending metrics to database (async, non-blocking)"""
        if not force and len(self.performance_queue) < self.batch_size and len(self.decision_queue) < self.batch_size:
            return
        
        performance_batch = []
        decision_batch = []
        
        async with self._lock:
            performance_batch = self.performance_queue.copy()
            decision_batch = self.decision_queue.copy()
            self.performance_queue.clear()
            self.decision_queue.clear()
        
        if performance_batch:
            await self._save_performance_batch(performance_batch)
        
        if decision_batch:
            await self._save_decision_batch(decision_batch)
    
    async def _save_performance_batch(self, batch: List[ExecutionMetrics]):
        """Save performance metrics to database in batch"""
        try:
            async with asyncio.create_task(self._get_async_connection()) as conn:
                cur = conn.cursor()
                
                # Prepare batch insert
                values = []
                for metrics in batch:
                    values.append((
                        str(uuid.uuid4()),  # id
                        metrics.agent_name,
                        metrics.agent_type,
                        metrics.execution_id,
                        metrics.blog_post_id,
                        metrics.campaign_id,
                        metrics.start_time,
                        metrics.end_time,
                        metrics.duration_ms,
                        metrics.status,
                        metrics.input_tokens,
                        metrics.output_tokens,
                        metrics.total_tokens,
                        metrics.cost,
                        metrics.error_message,
                        metrics.error_code,
                        metrics.retry_count,
                        3,  # max_retries default
                        json.dumps(metrics.metadata) if metrics.metadata else None
                    ))
                
                # Batch insert performance records
                cur.executemany("""
                    INSERT INTO agent_performance (
                        id, agent_name, agent_type, execution_id, blog_post_id, campaign_id,
                        start_time, end_time, duration, status, input_tokens, output_tokens,
                        total_tokens, cost, error_message, error_code, retry_count,
                        max_retries, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (execution_id) DO UPDATE SET
                        end_time = EXCLUDED.end_time,
                        duration = EXCLUDED.duration,
                        status = EXCLUDED.status,
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        total_tokens = EXCLUDED.total_tokens,
                        cost = EXCLUDED.cost,
                        error_message = EXCLUDED.error_message,
                        error_code = EXCLUDED.error_code,
                        retry_count = EXCLUDED.retry_count,
                        metadata = EXCLUDED.metadata
                """, values)
                
                conn.commit()
                logger.debug(f"Saved {len(batch)} performance records to database")
                
        except Exception as e:
            logger.error(f"Error saving performance batch: {e}")
    
    async def _save_decision_batch(self, batch: List[DecisionMetrics]):
        """Save decision metrics to database in batch"""
        try:
            async with asyncio.create_task(self._get_async_connection()) as conn:
                cur = conn.cursor()
                
                # First, get performance IDs from execution IDs
                execution_ids = [dm.performance_id for dm in batch]
                placeholders = ','.join(['%s'] * len(execution_ids))
                
                cur.execute(f"""
                    SELECT id, execution_id FROM agent_performance 
                    WHERE execution_id IN ({placeholders})
                """, execution_ids)
                
                performance_id_map = {row[1]: row[0] for row in cur.fetchall()}
                
                # Prepare decision batch insert
                values = []
                for decision_metrics in batch:
                    performance_id = performance_id_map.get(decision_metrics.performance_id)
                    if not performance_id:
                        continue  # Skip if no performance record found
                    
                    values.append((
                        str(uuid.uuid4()),  # id
                        performance_id,
                        decision_metrics.decision_point,
                        json.dumps(decision_metrics.input_data),
                        json.dumps(decision_metrics.output_data),
                        decision_metrics.reasoning,
                        decision_metrics.confidence_score,
                        json.dumps(decision_metrics.alternatives_considered),
                        decision_metrics.execution_time_ms,
                        decision_metrics.tokens_used,
                        decision_metrics.decision_latency_s,
                        decision_metrics.timestamp,
                        json.dumps(decision_metrics.metadata) if decision_metrics.metadata else None
                    ))
                
                if values:
                    # Batch insert decision records
                    cur.executemany("""
                        INSERT INTO agent_decisions (
                            id, performance_id, decision_point, input_data, output_data,
                            reasoning, confidence_score, alternatives_considered,
                            execution_time, tokens_used, decision_latency, timestamp, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, values)
                    
                    conn.commit()
                    logger.debug(f"Saved {len(values)} decision records to database")
                
        except Exception as e:
            logger.error(f"Error saving decision batch: {e}")
    
    async def _get_async_connection(self):
        """Get async database connection (simulated with asyncio)"""
        # This is a placeholder - in real implementation, use async database driver
        # For now, wrap sync connection in executor
        def get_sync_connection():
            return db_config.get_db_connection()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_sync_connection)


# Global performance tracker instance
global_performance_tracker = AsyncPerformanceTracker()


def langgraph_performance_tracking(
    agent_name: str,
    agent_type: str = "unknown",
    track_decisions: bool = True,
    cost_calculation: bool = True
):
    """
    High-performance decorator for LangGraph node functions.
    Automatically tracks execution metrics without blocking workflow execution.
    
    Usage:
        @langgraph_performance_tracking("content_generator", "ai_content_generator")
        def generate_content_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Your LangGraph node logic here
            return state
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract campaign_id and blog_post_id from state if available
            state = args[0] if args and isinstance(args[0], dict) else {}
            campaign_id = state.get('campaign_id')
            blog_post_id = state.get('blog_post_id')
            
            # Start tracking
            execution_id = await global_performance_tracker.track_execution_start(
                agent_name=agent_name,
                agent_type=agent_type,
                campaign_id=campaign_id,
                blog_post_id=blog_post_id,
                metadata={
                    'function_name': func.__name__,
                    'is_langgraph_node': True,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
            )
            
            try:
                # Execute the original function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Track successful completion
                await global_performance_tracker.track_execution_end(
                    execution_id=execution_id,
                    status="success"
                )
                
                # Track decision if enabled and result contains decision data
                if track_decisions and isinstance(result, dict) and 'agent_decision' in result:
                    decision_data = result['agent_decision']
                    await global_performance_tracker.track_decision(
                        execution_id=execution_id,
                        decision_point=decision_data.get('decision_point', f"{agent_name}_execution"),
                        input_data={'state_keys': list(state.keys()) if state else []},
                        output_data={'result_keys': list(result.keys()) if isinstance(result, dict) else {}},
                        reasoning=decision_data.get('reasoning', f"Executed {func.__name__} successfully"),
                        confidence_score=decision_data.get('confidence_score', 0.8),
                        alternatives_considered=decision_data.get('alternatives_considered', []),
                        execution_time_ms=decision_data.get('execution_time_ms', 0)
                    )
                
                return result
                
            except Exception as e:
                # Track failure
                await global_performance_tracker.track_execution_end(
                    execution_id=execution_id,
                    status="failed",
                    error_message=str(e),
                    error_code=f"{agent_type.upper()}_{type(e).__name__.upper()}"
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async context
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Utility functions for cost calculation
def calculate_openai_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    """Calculate OpenAI API cost based on token usage"""
    pricing = {
        "gpt-4": {"input": 0.00003, "output": 0.00006},  # $0.03/$0.06 per 1K tokens
        "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},  # $0.0015/$0.002 per 1K tokens
        "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},  # $0.01/$0.03 per 1K tokens
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4"])
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return round(input_cost + output_cost, 6)


# Integration with BaseAgent
class PerformanceTrackingMixin:
    """Mixin for BaseAgent to automatically track performance"""
    
    async def execute_with_performance_tracking(self, input_data, context=None, **kwargs):
        """Execute agent with automatic performance tracking"""
        agent_name = getattr(self, 'agent_name', self.__class__.__name__)
        agent_type = getattr(self, 'metadata', None)
        agent_type = agent_type.agent_type.value if agent_type and hasattr(agent_type, 'agent_type') else 'unknown'
        
        # Extract context information
        campaign_id = getattr(context, 'metadata', {}).get('campaign_id') if context else None
        blog_post_id = getattr(context, 'metadata', {}).get('blog_post_id') if context else None
        
        # Start performance tracking
        execution_id = await global_performance_tracker.track_execution_start(
            agent_name=agent_name,
            agent_type=agent_type,
            campaign_id=campaign_id,
            blog_post_id=blog_post_id,
            metadata={
                'agent_class': self.__class__.__name__,
                'input_type': type(input_data).__name__,
                'has_context': context is not None
            }
        )
        
        try:
            # Execute the agent
            result = self.execute_safe(input_data, context, **kwargs)
            
            # Track successful execution
            await global_performance_tracker.track_execution_end(
                execution_id=execution_id,
                status="success"
            )
            
            # Track decisions from result
            if hasattr(result, 'decisions') and result.decisions:
                for decision in result.decisions:
                    await global_performance_tracker.track_decision(
                        execution_id=execution_id,
                        decision_point=decision.decision_point,
                        input_data={'input_summary': str(input_data)[:200]},
                        output_data={'decision_summary': decision.reasoning[:200]},
                        reasoning=decision.reasoning,
                        confidence_score=decision.confidence_score,
                        alternatives_considered=decision.alternatives_considered,
                        execution_time_ms=int(result.execution_time_ms or 0)
                    )
            
            return result
            
        except Exception as e:
            # Track failed execution
            await global_performance_tracker.track_execution_end(
                execution_id=execution_id,
                status="failed",
                error_message=str(e),
                error_code=f"{agent_type.upper()}_{type(e).__name__.upper()}"
            )
            raise