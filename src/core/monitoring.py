"""
Performance monitoring and metrics collection system.
Provides comprehensive monitoring for API performance, system resources, and business metrics.
"""

import time
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging
import json

from ..config.settings import settings
from .cache import cache

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Centralized metrics collection system.
    Collects and aggregates performance metrics across the application.
    """
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._build_key(name, tags)
        self.counters[key] += value
        
        # Store timestamped value for trends
        self.metrics[key].append({
            'timestamp': time.time(),
            'value': value,
            'type': 'counter'
        })
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        key = self._build_key(name, tags)
        self.gauges[key] = value
        
        self.metrics[key].append({
            'timestamp': time.time(),
            'value': value,
            'type': 'gauge'
        })
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        key = self._build_key(name, tags)
        
        # Keep only recent values for histogram calculations
        if len(self.histograms[key]) >= 1000:
            self.histograms[key] = self.histograms[key][-500:]  # Keep last 500
        
        self.histograms[key].append(value)
        
        self.metrics[key].append({
            'timestamp': time.time(),
            'value': value,
            'type': 'histogram'
        })
    
    def _build_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Build metric key with tags."""
        if not tags:
            return name
        
        tag_string = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_string}"
    
    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        key = self._build_key(name, tags)
        
        if key not in self.metrics:
            return {"error": "Metric not found"}
        
        recent_values = [m['value'] for m in list(self.metrics[key])[-100:]]  # Last 100 values
        
        if not recent_values:
            return {"error": "No recent values"}
        
        return {
            'count': len(recent_values),
            'sum': sum(recent_values),
            'avg': sum(recent_values) / len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'last_value': recent_values[-1],
            'timestamp': time.time()
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics with summaries."""
        result = {
            'counters': {},
            'gauges': {},
            'histograms': {},
            'timestamp': time.time()
        }
        
        # Process counters
        for key, value in self.counters.items():
            result['counters'][key] = {
                'value': value,
                'summary': self.get_metric_summary(key.split(',')[0], self._parse_tags_from_key(key))
            }
        
        # Process gauges  
        for key, value in self.gauges.items():
            result['gauges'][key] = {
                'value': value,
                'summary': self.get_metric_summary(key.split(',')[0], self._parse_tags_from_key(key))
            }
        
        # Process histograms
        for key, values in self.histograms.items():
            if values:
                result['histograms'][key] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'p50': self._percentile(values, 0.5),
                    'p95': self._percentile(values, 0.95),
                    'p99': self._percentile(values, 0.99)
                }
        
        return result
    
    def _parse_tags_from_key(self, key: str) -> Optional[Dict[str, str]]:
        """Parse tags from metric key."""
        parts = key.split(',')
        if len(parts) == 1:
            return None
        
        tags = {}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                tags[k] = v
        
        return tags if tags else None
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]
    
    async def cleanup_old_metrics(self):
        """Clean up old metric data."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        for key in list(self.metrics.keys()):
            metrics_list = self.metrics[key]
            # Remove old entries
            while metrics_list and metrics_list[0]['timestamp'] < cutoff_time:
                metrics_list.popleft()
        
        self._last_cleanup = current_time


# Global metrics collector
metrics = MetricsCollector()


class PerformanceTracker:
    """
    Context manager for tracking operation performance.
    Automatically records timing and success/failure metrics.
    """
    
    def __init__(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.success: bool = True
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        self.start_time = time.time()
        metrics.increment_counter(f"{self.operation_name}.started", tags=self.tags)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is not None:
            self.success = False
            self.error = exc_val
            metrics.increment_counter(f"{self.operation_name}.failed", tags=self.tags)
        else:
            metrics.increment_counter(f"{self.operation_name}.completed", tags=self.tags)
        
        # Record timing
        metrics.record_histogram(f"{self.operation_name}.duration", duration, tags=self.tags)
        
        # Log slow operations
        if duration > 2.0:
            logger.warning(f"Slow operation {self.operation_name}: {duration:.2f}s")


@asynccontextmanager
async def async_performance_tracker(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Async context manager for performance tracking."""
    start_time = time.time()
    success = True
    error = None
    
    tags = tags or {}
    
    try:
        metrics.increment_counter(f"{operation_name}.started", tags=tags)
        yield
        metrics.increment_counter(f"{operation_name}.completed", tags=tags)
        
    except Exception as e:
        success = False
        error = e
        metrics.increment_counter(f"{operation_name}.failed", tags=tags)
        raise
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        metrics.record_histogram(f"{operation_name}.duration", duration, tags=tags)
        
        if duration > 2.0:
            logger.warning(f"Slow async operation {operation_name}: {duration:.2f}s")


class SystemMonitor:
    """
    System resource monitoring.
    Tracks CPU, memory, disk, and network usage.
    """
    
    def __init__(self):
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: int = 30):
        """Start system monitoring with specified interval."""
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started system monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped system monitoring")
    
    async def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.set_gauge("system.cpu.usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.set_gauge("system.memory.usage_percent", memory.percent)
            metrics.set_gauge("system.memory.available_gb", memory.available / (1024**3))
            metrics.set_gauge("system.memory.used_gb", memory.used / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.set_gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
            metrics.set_gauge("system.disk.free_gb", disk.free / (1024**3))
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.set_gauge("system.network.bytes_sent", network.bytes_sent)
            metrics.set_gauge("system.network.bytes_recv", network.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            metrics.set_gauge("process.memory_mb", process.memory_info().rss / (1024**2))
            metrics.set_gauge("process.cpu_percent", process.cpu_percent())
            metrics.set_gauge("process.threads", process.num_threads())
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")


# Global system monitor
system_monitor = SystemMonitor()


class AlertManager:
    """
    Simple alerting system for critical metrics.
    Can be extended to integrate with external alerting systems.
    """
    
    def __init__(self):
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.last_check = time.time()
    
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,  # "gt", "lt", "eq"
        threshold: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Add an alert rule."""
        self.alert_rules.append({
            'name': name,
            'metric_name': metric_name,
            'condition': condition,
            'threshold': threshold,
            'tags': tags or {}
        })
        logger.info(f"Added alert rule: {name}")
    
    async def check_alerts(self):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                metric_summary = metrics.get_metric_summary(
                    rule['metric_name'], 
                    rule['tags']
                )
                
                if 'error' in metric_summary:
                    continue
                
                current_value = metric_summary.get('last_value', 0)
                threshold = rule['threshold']
                condition = rule['condition']
                
                triggered = False
                if condition == 'gt' and current_value > threshold:
                    triggered = True
                elif condition == 'lt' and current_value < threshold:
                    triggered = True
                elif condition == 'eq' and current_value == threshold:
                    triggered = True
                
                alert_key = rule['name']
                
                if triggered:
                    if alert_key not in self.alerts:
                        # New alert
                        self.alerts[alert_key] = {
                            'rule': rule,
                            'current_value': current_value,
                            'triggered_at': current_time,
                            'last_updated': current_time
                        }
                        await self._send_alert(rule, current_value)
                    else:
                        # Update existing alert
                        self.alerts[alert_key]['current_value'] = current_value
                        self.alerts[alert_key]['last_updated'] = current_time
                else:
                    if alert_key in self.alerts:
                        # Alert resolved
                        await self._resolve_alert(rule, current_value)
                        del self.alerts[alert_key]
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    async def _send_alert(self, rule: Dict[str, Any], current_value: float):
        """Send alert notification."""
        message = (
            f"ALERT: {rule['name']} - "
            f"{rule['metric_name']} is {current_value} "
            f"({rule['condition']} {rule['threshold']})"
        )
        
        logger.warning(message)
        
        # Could integrate with external alerting systems here:
        # - Send to Slack/Discord
        # - Send email
        # - Create PagerDuty incident
        # - Send to monitoring service
    
    async def _resolve_alert(self, rule: Dict[str, Any], current_value: float):
        """Send alert resolution notification."""
        message = (
            f"RESOLVED: {rule['name']} - "
            f"{rule['metric_name']} is now {current_value}"
        )
        
        logger.info(message)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return list(self.alerts.values())


# Global alert manager
alert_manager = AlertManager()


# Monitoring decorators and utilities

def monitor_performance(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for monitoring function performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with async_performance_tracker(operation_name, tags):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with PerformanceTracker(operation_name, tags):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


async def setup_default_alerts():
    """Setup default alert rules for common issues."""
    # High CPU usage
    alert_manager.add_alert_rule(
        name="high_cpu_usage",
        metric_name="system.cpu.usage_percent",
        condition="gt",
        threshold=80.0
    )
    
    # High memory usage
    alert_manager.add_alert_rule(
        name="high_memory_usage",
        metric_name="system.memory.usage_percent",
        condition="gt",
        threshold=85.0
    )
    
    # Low disk space
    alert_manager.add_alert_rule(
        name="low_disk_space",
        metric_name="system.disk.usage_percent",
        condition="gt",
        threshold=90.0
    )
    
    # Slow API responses
    alert_manager.add_alert_rule(
        name="slow_api_responses",
        metric_name="api.request.duration",
        condition="gt",
        threshold=5.0
    )


async def start_monitoring():
    """Start all monitoring services."""
    logger.info("Starting performance monitoring...")
    
    # Start system monitoring
    await system_monitor.start_monitoring(interval=30)
    
    # Setup default alerts
    await setup_default_alerts()
    
    # Start alert checking loop
    asyncio.create_task(alert_check_loop())
    
    logger.info("Performance monitoring started")


async def stop_monitoring():
    """Stop all monitoring services."""
    logger.info("Stopping performance monitoring...")
    
    await system_monitor.stop_monitoring()
    
    logger.info("Performance monitoring stopped")


async def alert_check_loop():
    """Periodic alert checking loop."""
    while True:
        try:
            await alert_manager.check_alerts()
            await metrics.cleanup_old_metrics()
            await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Alert check loop error: {e}")
            await asyncio.sleep(60)


# API for getting monitoring data
async def get_monitoring_summary() -> Dict[str, Any]:
    """Get comprehensive monitoring summary."""
    return {
        'metrics': metrics.get_all_metrics(),
        'active_alerts': alert_manager.get_active_alerts(),
        'system_monitor_active': system_monitor.monitoring,
        'timestamp': time.time()
    }