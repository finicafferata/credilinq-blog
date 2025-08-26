"""
Application metrics for Prometheus monitoring.
Provides custom metrics for AI agents, database operations, and business logic.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# HTTP request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# AI Agent metrics
agent_executions_total = Counter(
    'agent_executions_total',
    'Total AI agent executions',
    ['agent_name', 'status']
)

agent_execution_duration_seconds = Histogram(
    'agent_execution_duration_seconds',
    'AI agent execution duration in seconds',
    ['agent_name']
)

agent_execution_failures_total = Counter(
    'agent_execution_failures_total',
    'Total AI agent execution failures',
    ['agent_name', 'error_type']
)

agent_task_queue_size = Gauge(
    'agent_task_queue_size',
    'Current size of agent task queue',
    ['agent_name']
)

# Database metrics
database_connections_active = Gauge(
    'database_connections_active',
    'Active database connections'
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table']
)

database_operations_total = Counter(
    'database_operations_total',
    'Total database operations',
    ['operation', 'table', 'status']
)

# OpenAI API metrics
openai_api_requests_total = Counter(
    'openai_api_requests_total',
    'Total OpenAI API requests',
    ['model', 'status']
)

openai_api_duration_seconds = Histogram(
    'openai_api_duration_seconds',
    'OpenAI API request duration in seconds',
    ['model']
)

openai_api_errors_total = Counter(
    'openai_api_errors_total',
    'Total OpenAI API errors',
    ['model', 'error_type']
)

openai_tokens_used_total = Counter(
    'openai_tokens_used_total',
    'Total OpenAI tokens used',
    ['model', 'token_type']
)

# Business metrics
content_generation_total = Counter(
    'content_generation_total',
    'Total content generated',
    ['content_type', 'status']
)

campaign_executions_total = Counter(
    'campaign_executions_total',
    'Total campaign executions',
    ['campaign_type', 'status']
)

# Application info
app_info = Info(
    'app_info',
    'Application information'
)

# System metrics
system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

# Cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status']
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio'
)

# Security metrics
security_events_total = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

failed_auth_attempts_total = Counter(
    'failed_auth_attempts_total',
    'Total failed authentication attempts',
    ['source_ip', 'user_agent']
)

rate_limit_violations_total = Counter(
    'rate_limit_violations_total',
    'Total rate limit violations',
    ['source_ip']
)


class MetricsCollector:
    """
    Centralized metrics collection and management.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self._init_app_info()
    
    def _init_app_info(self):
        """Initialize application information metrics."""
        try:
            import os
            from ..config import settings
            
            app_info.info({
                'version': getattr(settings, 'api_version', 'unknown'),
                'environment': getattr(settings, 'environment', 'unknown'),
                'python_version': os.sys.version.split()[0],
                'start_time': str(int(self.start_time))
            })
        except Exception as e:
            logger.warning(f"Failed to initialize app info metrics: {e}")
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        try:
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record HTTP request metrics: {e}")
    
    def record_agent_execution(self, agent_name: str, duration: float, status: str = "success", error_type: str = None):
        """Record AI agent execution metrics."""
        try:
            agent_executions_total.labels(
                agent_name=agent_name,
                status=status
            ).inc()
            
            agent_execution_duration_seconds.labels(
                agent_name=agent_name
            ).observe(duration)
            
            if status == "failure" and error_type:
                agent_execution_failures_total.labels(
                    agent_name=agent_name,
                    error_type=error_type
                ).inc()
        except Exception as e:
            logger.error(f"Failed to record agent execution metrics: {e}")
    
    def update_agent_queue_size(self, agent_name: str, queue_size: int):
        """Update agent task queue size."""
        try:
            agent_task_queue_size.labels(agent_name=agent_name).set(queue_size)
        except Exception as e:
            logger.error(f"Failed to update agent queue size: {e}")
    
    def record_database_operation(self, operation: str, table: str, duration: float, status: str = "success"):
        """Record database operation metrics."""
        try:
            database_operations_total.labels(
                operation=operation,
                table=table,
                status=status
            ).inc()
            
            database_query_duration_seconds.labels(
                operation=operation,
                table=table
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record database operation metrics: {e}")
    
    def update_database_connections(self, active_connections: int):
        """Update active database connections count."""
        try:
            database_connections_active.set(active_connections)
        except Exception as e:
            logger.error(f"Failed to update database connections: {e}")
    
    def record_openai_request(self, model: str, duration: float, status: str = "success", 
                             error_type: str = None, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record OpenAI API request metrics."""
        try:
            openai_api_requests_total.labels(
                model=model,
                status=status
            ).inc()
            
            openai_api_duration_seconds.labels(model=model).observe(duration)
            
            if status == "error" and error_type:
                openai_api_errors_total.labels(
                    model=model,
                    error_type=error_type
                ).inc()
            
            if prompt_tokens > 0:
                openai_tokens_used_total.labels(
                    model=model,
                    token_type="prompt"
                ).inc(prompt_tokens)
            
            if completion_tokens > 0:
                openai_tokens_used_total.labels(
                    model=model,
                    token_type="completion"
                ).inc(completion_tokens)
        except Exception as e:
            logger.error(f"Failed to record OpenAI request metrics: {e}")
    
    def record_content_generation(self, content_type: str, status: str = "success"):
        """Record content generation metrics."""
        try:
            content_generation_total.labels(
                content_type=content_type,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record content generation metrics: {e}")
    
    def record_campaign_execution(self, campaign_type: str, status: str = "success"):
        """Record campaign execution metrics."""
        try:
            campaign_executions_total.labels(
                campaign_type=campaign_type,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record campaign execution metrics: {e}")
    
    def record_cache_operation(self, operation: str, status: str = "hit"):
        """Record cache operation metrics."""
        try:
            cache_operations_total.labels(
                operation=operation,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record cache operation metrics: {e}")
    
    def update_cache_hit_ratio(self, hit_ratio: float):
        """Update cache hit ratio."""
        try:
            cache_hit_ratio.set(hit_ratio)
        except Exception as e:
            logger.error(f"Failed to update cache hit ratio: {e}")
    
    def record_security_event(self, event_type: str, severity: str = "info"):
        """Record security event metrics."""
        try:
            security_events_total.labels(
                event_type=event_type,
                severity=severity
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record security event: {e}")
    
    def record_failed_auth(self, source_ip: str, user_agent: str = "unknown"):
        """Record failed authentication attempt."""
        try:
            # Truncate user agent to avoid cardinality explosion
            user_agent_truncated = user_agent[:50] if user_agent else "unknown"
            
            failed_auth_attempts_total.labels(
                source_ip=source_ip,
                user_agent=user_agent_truncated
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record failed auth attempt: {e}")
    
    def record_rate_limit_violation(self, source_ip: str):
        """Record rate limit violation."""
        try:
            rate_limit_violations_total.labels(source_ip=source_ip).inc()
        except Exception as e:
            logger.error(f"Failed to record rate limit violation: {e}")
    
    def update_system_metrics(self, memory_usage: int, cpu_usage: float):
        """Update system resource metrics."""
        try:
            system_memory_usage_bytes.set(memory_usage)
            system_cpu_usage_percent.set(cpu_usage)
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")


# Global metrics collector instance
metrics = MetricsCollector()


def track_execution_time(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to track execution time of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metrics
                if metric_name == "agent_execution":
                    metrics.record_agent_execution(
                        agent_name=labels.get("agent_name", "unknown"),
                        duration=duration,
                        status="success"
                    )
                elif metric_name == "database_operation":
                    metrics.record_database_operation(
                        operation=labels.get("operation", "unknown"),
                        table=labels.get("table", "unknown"),
                        duration=duration,
                        status="success"
                    )
                elif metric_name == "openai_request":
                    metrics.record_openai_request(
                        model=labels.get("model", "unknown"),
                        duration=duration,
                        status="success"
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure metrics
                if metric_name == "agent_execution":
                    metrics.record_agent_execution(
                        agent_name=labels.get("agent_name", "unknown"),
                        duration=duration,
                        status="failure",
                        error_type=type(e).__name__
                    )
                elif metric_name == "database_operation":
                    metrics.record_database_operation(
                        operation=labels.get("operation", "unknown"),
                        table=labels.get("table", "unknown"),
                        duration=duration,
                        status="failure"
                    )
                elif metric_name == "openai_request":
                    metrics.record_openai_request(
                        model=labels.get("model", "unknown"),
                        duration=duration,
                        status="error",
                        error_type=type(e).__name__
                    )
                
                raise
        return wrapper
    return decorator


def get_metrics_response():
    """Generate Prometheus metrics response."""
    try:
        return generate_latest(), CONTENT_TYPE_LATEST
    except Exception as e:
        logger.error(f"Failed to generate metrics response: {e}")
        return "# Error generating metrics\n", "text/plain"