"""
Workflow Monitoring and Metrics Collection System

Provides comprehensive monitoring, metrics collection, and observability
for LangGraph workflows with real-time performance tracking.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class WorkflowStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowMetric:
    name: str
    type: MetricType
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    workflow_id: str
    run_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    agent_executions: List[Dict[str, Any]] = field(default_factory=list)
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[WorkflowMetric] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    cost_tracking: Dict[str, Union[int, float]] = field(default_factory=dict)


class WorkflowMetricsCollector:
    """Centralized metrics collection system for workflow monitoring"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.completed_workflows: deque = deque(maxlen=max_history)
        self.metrics_buffer: List[WorkflowMetric] = []
        self.performance_stats = defaultdict(list)
        self.alert_thresholds = {
            'execution_time': 300.0,  # 5 minutes
            'error_rate': 0.1,        # 10%
            'queue_size': 100,
            'memory_usage': 0.8       # 80%
        }
    
    async def start_workflow(self, workflow_id: str, run_id: str) -> None:
        """Start tracking a new workflow execution"""
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            run_id=run_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        self.active_workflows[run_id] = execution
        
        await self.record_metric(
            name="workflow_started",
            type=MetricType.COUNTER,
            value=1,
            tags={"workflow_id": workflow_id, "run_id": run_id}
        )
        
        logger.info(f"Started tracking workflow {workflow_id} (run: {run_id})")
    
    async def complete_workflow(self, run_id: str, status: WorkflowStatus, 
                               final_state: Optional[Dict] = None) -> None:
        """Complete workflow tracking with final metrics"""
        if run_id not in self.active_workflows:
            logger.warning(f"Attempted to complete unknown workflow: {run_id}")
            return
        
        execution = self.active_workflows[run_id]
        execution.end_time = datetime.utcnow()
        execution.status = status
        execution.duration = (execution.end_time - execution.start_time).total_seconds()
        
        # Record completion metrics
        await self.record_metric(
            name="workflow_completed",
            type=MetricType.COUNTER,
            value=1,
            tags={
                "workflow_id": execution.workflow_id,
                "status": status.value,
                "duration_bucket": self._get_duration_bucket(execution.duration)
            }
        )
        
        await self.record_metric(
            name="workflow_duration",
            type=MetricType.TIMER,
            value=execution.duration,
            tags={"workflow_id": execution.workflow_id}
        )
        
        # Move to completed workflows
        self.completed_workflows.append(execution)
        del self.active_workflows[run_id]
        
        # Check for performance alerts
        await self._check_performance_alerts(execution)
        
        logger.info(
            f"Completed workflow {execution.workflow_id} "
            f"in {execution.duration:.2f}s with status {status.value}"
        )
    
    async def record_agent_execution(self, run_id: str, agent_name: str, 
                                   execution_data: Dict[str, Any]) -> None:
        """Record individual agent execution within a workflow"""
        if run_id not in self.active_workflows:
            return
        
        execution = self.active_workflows[run_id]
        agent_record = {
            "agent_name": agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "duration": execution_data.get("duration", 0),
            "tokens_used": execution_data.get("tokens_used", 0),
            "quality_score": execution_data.get("quality_score"),
            "status": execution_data.get("status", "completed"),
            "metadata": execution_data.get("metadata", {})
        }
        
        execution.agent_executions.append(agent_record)
        
        # Record agent-specific metrics
        await self.record_metric(
            name="agent_execution",
            type=MetricType.COUNTER,
            value=1,
            tags={
                "workflow_id": execution.workflow_id,
                "agent": agent_name,
                "status": agent_record["status"]
            }
        )
        
        if execution_data.get("tokens_used"):
            await self.record_metric(
                name="tokens_consumed",
                type=MetricType.COUNTER,
                value=execution_data["tokens_used"],
                tags={"agent": agent_name}
            )
    
    async def record_state_transition(self, run_id: str, from_state: str, 
                                    to_state: str, transition_data: Dict = None) -> None:
        """Record workflow state transitions"""
        if run_id not in self.active_workflows:
            return
        
        execution = self.active_workflows[run_id]
        transition = {
            "from_state": from_state,
            "to_state": to_state,
            "timestamp": datetime.utcnow().isoformat(),
            "data": transition_data or {}
        }
        
        execution.state_transitions.append(transition)
        
        await self.record_metric(
            name="state_transition",
            type=MetricType.COUNTER,
            value=1,
            tags={
                "workflow_id": execution.workflow_id,
                "from_state": from_state,
                "to_state": to_state
            }
        )
    
    async def record_error(self, run_id: str, error_type: str, 
                          error_message: str, agent_name: Optional[str] = None) -> None:
        """Record workflow errors for monitoring"""
        if run_id not in self.active_workflows:
            return
        
        execution = self.active_workflows[run_id]
        error_record = {
            "error_type": error_type,
            "message": error_message,
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        execution.errors.append(error_record)
        
        await self.record_metric(
            name="workflow_error",
            type=MetricType.COUNTER,
            value=1,
            tags={
                "workflow_id": execution.workflow_id,
                "error_type": error_type,
                "agent": agent_name or "unknown"
            }
        )
        
        logger.error(f"Workflow error in {execution.workflow_id}: {error_message}")
    
    async def record_metric(self, name: str, type: MetricType, value: Union[int, float],
                           tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric"""
        metric = WorkflowMetric(
            name=name,
            type=type,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
        
        # Update performance stats
        if name in ["workflow_duration", "agent_execution_time"]:
            self.performance_stats[name].append(value)
            if len(self.performance_stats[name]) > 100:
                self.performance_stats[name] = self.performance_stats[name][-100:]
    
    async def get_workflow_metrics(self, workflow_id: Optional[str] = None,
                                  time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive workflow metrics"""
        cutoff_time = None
        if time_range:
            cutoff_time = datetime.utcnow() - time_range
        
        # Filter workflows
        workflows = []
        for workflow in self.completed_workflows:
            if workflow_id and workflow.workflow_id != workflow_id:
                continue
            if cutoff_time and workflow.start_time < cutoff_time:
                continue
            workflows.append(workflow)
        
        # Add active workflows
        for workflow in self.active_workflows.values():
            if workflow_id and workflow.workflow_id != workflow_id:
                continue
            workflows.append(workflow)
        
        if not workflows:
            return {"message": "No workflows found for criteria"}
        
        # Calculate metrics
        total_workflows = len(workflows)
        completed_workflows = len([w for w in workflows if w.status == WorkflowStatus.COMPLETED])
        failed_workflows = len([w for w in workflows if w.status == WorkflowStatus.FAILED])
        
        durations = [w.duration for w in workflows if w.duration is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        total_agents = sum(len(w.agent_executions) for w in workflows)
        total_errors = sum(len(w.errors) for w in workflows)
        
        return {
            "total_workflows": total_workflows,
            "completed_workflows": completed_workflows,
            "failed_workflows": failed_workflows,
            "success_rate": completed_workflows / total_workflows if total_workflows > 0 else 0,
            "average_duration": avg_duration,
            "total_agent_executions": total_agents,
            "total_errors": total_errors,
            "error_rate": total_errors / total_agents if total_agents > 0 else 0,
            "workflows": [self._serialize_workflow(w) for w in workflows[-10:]]  # Last 10
        }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get system-wide performance summary"""
        active_count = len(self.active_workflows)
        
        # Calculate recent performance stats
        recent_durations = self.performance_stats.get("workflow_duration", [])
        avg_duration = sum(recent_durations) / len(recent_durations) if recent_durations else 0
        
        recent_workflows = list(self.completed_workflows)[-50:] if self.completed_workflows else []
        recent_success_rate = 0
        if recent_workflows:
            successful = len([w for w in recent_workflows if w.status == WorkflowStatus.COMPLETED])
            recent_success_rate = successful / len(recent_workflows)
        
        return {
            "active_workflows": active_count,
            "recent_average_duration": avg_duration,
            "recent_success_rate": recent_success_rate,
            "total_metrics_collected": len(self.metrics_buffer),
            "system_health": self._calculate_system_health(),
            "alerts": await self._get_active_alerts()
        }
    
    def _get_duration_bucket(self, duration: float) -> str:
        """Categorize workflow duration into buckets"""
        if duration < 30:
            return "fast"
        elif duration < 120:
            return "medium"
        elif duration < 300:
            return "slow"
        else:
            return "very_slow"
    
    async def _check_performance_alerts(self, execution: WorkflowExecution) -> None:
        """Check if execution triggers any performance alerts"""
        alerts = []
        
        # Check execution time
        if execution.duration and execution.duration > self.alert_thresholds['execution_time']:
            alerts.append({
                "type": "slow_execution",
                "message": f"Workflow {execution.workflow_id} took {execution.duration:.2f}s",
                "severity": "warning"
            })
        
        # Check error rate
        if execution.errors:
            alerts.append({
                "type": "workflow_errors",
                "message": f"Workflow {execution.workflow_id} had {len(execution.errors)} errors",
                "severity": "error"
            })
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Performance Alert: {alert['message']}")
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health score"""
        if not self.completed_workflows:
            return "unknown"
        
        recent_workflows = list(self.completed_workflows)[-20:]
        success_rate = len([w for w in recent_workflows 
                           if w.status == WorkflowStatus.COMPLETED]) / len(recent_workflows)
        
        if success_rate >= 0.95:
            return "excellent"
        elif success_rate >= 0.85:
            return "good"
        elif success_rate >= 0.7:
            return "fair"
        else:
            return "poor"
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active system alerts"""
        alerts = []
        
        # Check queue size
        if len(self.active_workflows) > self.alert_thresholds['queue_size']:
            alerts.append({
                "type": "high_queue_size",
                "message": f"High number of active workflows: {len(self.active_workflows)}",
                "severity": "warning"
            })
        
        return alerts
    
    def _serialize_workflow(self, workflow: WorkflowExecution) -> Dict[str, Any]:
        """Serialize workflow execution for JSON response"""
        return {
            "workflow_id": workflow.workflow_id,
            "run_id": workflow.run_id,
            "status": workflow.status.value,
            "start_time": workflow.start_time.isoformat(),
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "duration": workflow.duration,
            "agent_count": len(workflow.agent_executions),
            "error_count": len(workflow.errors),
            "quality_scores": workflow.quality_scores
        }


# Global metrics collector instance
workflow_metrics = WorkflowMetricsCollector()


class MetricsContextManager:
    """Context manager for automatic workflow metrics collection"""
    
    def __init__(self, workflow_id: str, run_id: Optional[str] = None):
        self.workflow_id = workflow_id
        self.run_id = run_id or f"{workflow_id}_{int(time.time())}"
    
    async def __aenter__(self):
        await workflow_metrics.start_workflow(self.workflow_id, self.run_id)
        return self.run_id
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await workflow_metrics.complete_workflow(self.run_id, WorkflowStatus.COMPLETED)
        else:
            await workflow_metrics.record_error(
                self.run_id, 
                exc_type.__name__ if exc_type else "Unknown",
                str(exc_val) if exc_val else "Unknown error"
            )
            await workflow_metrics.complete_workflow(self.run_id, WorkflowStatus.FAILED)


# Convenience functions for workflow monitoring
async def start_workflow_monitoring(workflow_id: str, run_id: Optional[str] = None) -> str:
    """Start monitoring a workflow execution"""
    run_id = run_id or f"{workflow_id}_{int(time.time())}"
    await workflow_metrics.start_workflow(workflow_id, run_id)
    return run_id


async def record_agent_metrics(run_id: str, agent_name: str, **kwargs) -> None:
    """Record agent execution metrics"""
    await workflow_metrics.record_agent_execution(run_id, agent_name, kwargs)


async def get_workflow_health() -> Dict[str, Any]:
    """Get overall workflow system health"""
    return await workflow_metrics.get_performance_summary()