#!/usr/bin/env python3
"""
Task Management System for Content Generation Workflows

This module provides comprehensive task management capabilities for content generation,
including task scheduling, dependency resolution, progress tracking, and performance monitoring.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque

from src.agents.workflow.content_generation_workflow import (
    ContentTask, ContentTaskStatus, ContentTaskPriority
)
from src.config.database import db_config

logger = logging.getLogger(__name__)

class TaskSchedulingStrategy(Enum):
    """Task scheduling strategies"""
    FIFO = "first_in_first_out"
    PRIORITY_BASED = "priority_based"
    DEADLINE_AWARE = "deadline_aware"
    DEPENDENCY_FIRST = "dependency_first"
    RESOURCE_OPTIMIZED = "resource_optimized"

class TaskExecutionMode(Enum):
    """Task execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class ResourceType(Enum):
    """Types of resources used for task execution"""
    AI_AGENT = "ai_agent"
    COMPUTE_SLOT = "compute_slot"
    API_QUOTA = "api_quota"
    REVIEW_SLOT = "review_slot"

@dataclass
class ResourceAllocation:
    """Resource allocation for task execution"""
    resource_type: ResourceType
    resource_id: str
    allocated_at: datetime
    expected_release_at: Optional[datetime] = None
    actual_release_at: Optional[datetime] = None
    utilization_percentage: float = 0.0

@dataclass
class TaskDependency:
    """Task dependency definition"""
    dependent_task_id: str
    prerequisite_task_id: str
    dependency_type: str  # 'completion', 'data', 'resource'
    created_at: datetime = field(default_factory=datetime.now)

@dataclass 
class TaskSchedule:
    """Schedule for task execution"""
    task_id: str
    scheduled_start: datetime
    estimated_duration_minutes: int
    estimated_completion: datetime
    resource_requirements: List[ResourceType]
    predecessor_tasks: List[str] = field(default_factory=list)
    successor_tasks: List[str] = field(default_factory=list)

@dataclass
class TaskPerformanceMetrics:
    """Performance metrics for individual tasks"""
    task_id: str
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    scheduled_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    estimated_duration_minutes: Optional[int] = None
    actual_duration_minutes: Optional[float] = None
    quality_score: Optional[float] = None
    retry_count: int = 0
    resource_usage: List[ResourceAllocation] = field(default_factory=list)
    performance_score: float = 0.0  # Overall performance score
    bottleneck_factors: List[str] = field(default_factory=list)

class TaskQueue:
    """Advanced task queue with priority and dependency management"""
    
    def __init__(self, strategy: TaskSchedulingStrategy = TaskSchedulingStrategy.PRIORITY_BASED):
        self.strategy = strategy
        self._queue = []  # Priority queue (heap)
        self._task_map: Dict[str, ContentTask] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # task_id -> prerequisite_task_ids
        self._dependents: Dict[str, Set[str]] = defaultdict(set)   # task_id -> dependent_task_ids
        self._completed_tasks: Set[str] = set()
        self._in_progress_tasks: Set[str] = set()
        
    def add_task(self, task: ContentTask, dependencies: List[str] = None) -> None:
        """Add a task to the queue with optional dependencies"""
        dependencies = dependencies or []
        
        # Store task
        self._task_map[task.task_id] = task
        
        # Set up dependencies
        for dep_id in dependencies:
            self._dependencies[task.task_id].add(dep_id)
            self._dependents[dep_id].add(task.task_id)
        
        # Add to priority queue if no unmet dependencies
        if self._are_dependencies_met(task.task_id):
            self._add_to_heap(task)
        
        logger.debug(f"Added task {task.task_id} to queue with {len(dependencies)} dependencies")
    
    def get_next_task(self) -> Optional[ContentTask]:
        """Get the next task to execute based on strategy"""
        while self._queue:
            priority, task_id = heapq.heappop(self._queue)
            
            if task_id not in self._task_map:
                continue
                
            task = self._task_map[task_id]
            
            # Double-check dependencies are still met
            if self._are_dependencies_met(task_id):
                self._in_progress_tasks.add(task_id)
                task.status = ContentTaskStatus.IN_PROGRESS
                return task
        
        return None
    
    def complete_task(self, task_id: str) -> List[ContentTask]:
        """Mark task as completed and return newly available tasks"""
        if task_id not in self._task_map:
            return []
        
        # Mark as completed
        self._completed_tasks.add(task_id)
        self._in_progress_tasks.discard(task_id)
        
        task = self._task_map[task_id]
        task.status = ContentTaskStatus.COMPLETED
        
        # Check which dependent tasks are now ready
        newly_available = []
        for dependent_id in self._dependents[task_id]:
            if self._are_dependencies_met(dependent_id) and dependent_id not in self._in_progress_tasks:
                dependent_task = self._task_map[dependent_id]
                self._add_to_heap(dependent_task)
                newly_available.append(dependent_task)
        
        logger.debug(f"Completed task {task_id}, unlocked {len(newly_available)} new tasks")
        return newly_available
    
    def fail_task(self, task_id: str) -> None:
        """Mark task as failed and handle dependent tasks"""
        if task_id not in self._task_map:
            return
        
        self._in_progress_tasks.discard(task_id)
        task = self._task_map[task_id]
        task.status = ContentTaskStatus.FAILED
        
        # Handle dependent tasks - they might need to be cancelled or rescheduled
        for dependent_id in self._dependents[task_id]:
            dependent_task = self._task_map[dependent_id]
            dependent_task.metadata['blocked_by_failed_dependency'] = task_id
        
        logger.warning(f"Failed task {task_id}, affected {len(self._dependents[task_id])} dependent tasks")
    
    def _are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies for a task are met"""
        dependencies = self._dependencies[task_id]
        return all(dep_id in self._completed_tasks for dep_id in dependencies)
    
    def _add_to_heap(self, task: ContentTask) -> None:
        """Add task to the priority heap based on scheduling strategy"""
        if self.strategy == TaskSchedulingStrategy.PRIORITY_BASED:
            priority = self._calculate_priority_score(task)
        elif self.strategy == TaskSchedulingStrategy.DEADLINE_AWARE:
            priority = self._calculate_deadline_score(task)
        elif self.strategy == TaskSchedulingStrategy.FIFO:
            priority = task.created_at.timestamp()
        else:
            priority = self._calculate_priority_score(task)
        
        heapq.heappush(self._queue, (priority, task.task_id))
    
    def _calculate_priority_score(self, task: ContentTask) -> float:
        """Calculate priority score for task (lower score = higher priority)"""
        base_priority = {
            ContentTaskPriority.URGENT: 1,
            ContentTaskPriority.HIGH: 2,
            ContentTaskPriority.MEDIUM: 3,
            ContentTaskPriority.LOW: 4
        }[task.priority]
        
        # Adjust based on content type importance
        content_type_weight = {
            'blog_posts': 0.8,
            'email_content': 0.9,
            'case_studies': 0.7,
            'social_posts': 1.0
        }.get(task.content_type.value, 1.0)
        
        return base_priority * content_type_weight
    
    def _calculate_deadline_score(self, task: ContentTask) -> float:
        """Calculate deadline-based priority score"""
        if not task.deadline:
            return 999999  # No deadline = lowest priority
        
        time_to_deadline = (task.deadline - datetime.now()).total_seconds()
        return max(time_to_deadline, 1)  # Ensure positive value
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of the task queue"""
        return {
            'total_tasks': len(self._task_map),
            'queued_tasks': len(self._queue),
            'in_progress_tasks': len(self._in_progress_tasks),
            'completed_tasks': len(self._completed_tasks),
            'failed_tasks': len([t for t in self._task_map.values() if t.status == ContentTaskStatus.FAILED]),
            'strategy': self.strategy.value
        }

class ResourceManager:
    """Manages resources for task execution"""
    
    def __init__(self, max_concurrent_ai_agents: int = 5, max_concurrent_review_slots: int = 2):
        self.max_concurrent_ai_agents = max_concurrent_ai_agents
        self.max_concurrent_review_slots = max_concurrent_review_slots
        
        # Resource tracking
        self.allocated_resources: Dict[str, ResourceAllocation] = {}
        self.resource_usage_history: List[ResourceAllocation] = []
        
        # Current usage counters
        self.current_ai_agents = 0
        self.current_review_slots = 0
    
    def can_allocate_resources(self, resource_types: List[ResourceType]) -> bool:
        """Check if required resources can be allocated"""
        ai_agents_needed = resource_types.count(ResourceType.AI_AGENT)
        review_slots_needed = resource_types.count(ResourceType.REVIEW_SLOT)
        
        return (
            self.current_ai_agents + ai_agents_needed <= self.max_concurrent_ai_agents and
            self.current_review_slots + review_slots_needed <= self.max_concurrent_review_slots
        )
    
    def allocate_resources(self, task_id: str, resource_types: List[ResourceType]) -> List[ResourceAllocation]:
        """Allocate resources for a task"""
        allocations = []
        
        for resource_type in resource_types:
            if resource_type == ResourceType.AI_AGENT and self.current_ai_agents < self.max_concurrent_ai_agents:
                allocation = ResourceAllocation(
                    resource_type=resource_type,
                    resource_id=f"ai_agent_{self.current_ai_agents}",
                    allocated_at=datetime.now()
                )
                allocations.append(allocation)
                self.allocated_resources[f"{task_id}_{resource_type.value}"] = allocation
                self.current_ai_agents += 1
                
            elif resource_type == ResourceType.REVIEW_SLOT and self.current_review_slots < self.max_concurrent_review_slots:
                allocation = ResourceAllocation(
                    resource_type=resource_type,
                    resource_id=f"review_slot_{self.current_review_slots}",
                    allocated_at=datetime.now()
                )
                allocations.append(allocation)
                self.allocated_resources[f"{task_id}_{resource_type.value}"] = allocation
                self.current_review_slots += 1
        
        return allocations
    
    def release_resources(self, task_id: str) -> None:
        """Release all resources allocated to a task"""
        keys_to_remove = []
        
        for key, allocation in self.allocated_resources.items():
            if key.startswith(task_id):
                allocation.actual_release_at = datetime.now()
                
                # Update usage counters
                if allocation.resource_type == ResourceType.AI_AGENT:
                    self.current_ai_agents = max(0, self.current_ai_agents - 1)
                elif allocation.resource_type == ResourceType.REVIEW_SLOT:
                    self.current_review_slots = max(0, self.current_review_slots - 1)
                
                # Archive allocation
                self.resource_usage_history.append(allocation)
                keys_to_remove.append(key)
        
        # Remove from active allocations
        for key in keys_to_remove:
            del self.allocated_resources[key]
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages"""
        return {
            'ai_agents': (self.current_ai_agents / self.max_concurrent_ai_agents) * 100,
            'review_slots': (self.current_review_slots / self.max_concurrent_review_slots) * 100
        }

class TaskManagementSystem:
    """
    Comprehensive task management system for content generation workflows
    
    Provides advanced task scheduling, dependency management, resource allocation,
    and performance monitoring for content generation tasks.
    """
    
    def __init__(self, scheduling_strategy: TaskSchedulingStrategy = TaskSchedulingStrategy.PRIORITY_BASED,
                 execution_mode: TaskExecutionMode = TaskExecutionMode.PARALLEL,
                 max_concurrent_tasks: int = 5):
        self.system_id = "task_management_system"
        self.scheduling_strategy = scheduling_strategy
        self.execution_mode = execution_mode
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Core components
        self.task_queue = TaskQueue(scheduling_strategy)
        self.resource_manager = ResourceManager(max_concurrent_tasks, 2)
        
        # Performance tracking
        self.task_metrics: Dict[str, TaskPerformanceMetrics] = {}
        self.system_metrics = {
            'total_tasks_processed': 0,
            'total_execution_time': 0.0,
            'average_task_duration': 0.0,
            'success_rate': 0.0,
            'throughput_tasks_per_hour': 0.0
        }
        
        # Execution control
        self.is_running = False
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Task callbacks
        self.task_callbacks: Dict[str, List[Callable]] = {
            'on_task_start': [],
            'on_task_complete': [],
            'on_task_fail': [],
            'on_queue_empty': []
        }
    
    async def submit_task_batch(self, tasks: List[ContentTask], 
                              dependencies: Dict[str, List[str]] = None) -> str:
        """
        Submit a batch of tasks for execution with dependencies
        """
        try:
            batch_id = str(uuid.uuid4())
            dependencies = dependencies or {}
            
            logger.info(f"Submitting task batch {batch_id} with {len(tasks)} tasks")
            
            # Add tasks to queue
            for task in tasks:
                task_dependencies = dependencies.get(task.task_id, [])
                
                # Initialize performance metrics
                self.task_metrics[task.task_id] = TaskPerformanceMetrics(
                    task_id=task.task_id,
                    estimated_duration_minutes=self._estimate_task_duration(task)
                )
                
                self.task_queue.add_task(task, task_dependencies)
            
            # Start processing if not already running
            if not self.is_running:
                await self.start_processing()
            
            return batch_id
            
        except Exception as e:
            logger.error(f"Error submitting task batch: {str(e)}")
            raise
    
    async def start_processing(self) -> None:
        """Start the task processing engine"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Started task processing engine")
        
        try:
            await self._process_tasks()
        except Exception as e:
            logger.error(f"Error in task processing: {str(e)}")
        finally:
            self.is_running = False
    
    async def stop_processing(self) -> None:
        """Stop the task processing engine"""
        self.is_running = False
        logger.info("Stopped task processing engine")
    
    async def _process_tasks(self) -> None:
        """Main task processing loop"""
        active_tasks = []
        
        while self.is_running:
            try:
                # Clean up completed tasks
                active_tasks = [t for t in active_tasks if not t.done()]
                
                # Check if we can start new tasks
                available_slots = self.max_concurrent_tasks - len(active_tasks)
                
                # Get next tasks to process
                for _ in range(available_slots):
                    next_task = self.task_queue.get_next_task()
                    if not next_task:
                        break
                    
                    # Check resource availability
                    required_resources = self._determine_required_resources(next_task)
                    if not self.resource_manager.can_allocate_resources(required_resources):
                        # Put task back in queue (this is simplified - real implementation would handle better)
                        continue
                    
                    # Start task processing
                    task_coroutine = self._execute_single_task(next_task)
                    task_future = asyncio.create_task(task_coroutine)
                    active_tasks.append(task_future)
                
                # If no active tasks and queue is empty, we're done
                if not active_tasks and not self.task_queue._queue:
                    await self._trigger_callbacks('on_queue_empty')
                    break
                
                # Wait a bit before checking again
                if active_tasks:
                    # Wait for at least one task to complete
                    done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    active_tasks = list(pending)
                else:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {str(e)}")
                await asyncio.sleep(1)
        
        # Wait for all remaining tasks to complete
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
    
    async def _execute_single_task(self, task: ContentTask) -> None:
        """Execute a single content generation task"""
        async with self.execution_semaphore:
            try:
                # Record start time
                metrics = self.task_metrics.get(task.task_id, TaskPerformanceMetrics(task.task_id))
                metrics.actual_start = datetime.now()
                
                # Allocate resources
                required_resources = self._determine_required_resources(task)
                resource_allocations = self.resource_manager.allocate_resources(task.task_id, required_resources)
                metrics.resource_usage = resource_allocations
                
                # Trigger start callbacks
                await self._trigger_callbacks('on_task_start', task)
                
                # Execute the task (this would integrate with the actual content generation)
                await self._perform_task_execution(task)
                
                # Record completion
                metrics.actual_completion = datetime.now()
                if metrics.actual_start:
                    duration = (metrics.actual_completion - metrics.actual_start).total_seconds() / 60
                    metrics.actual_duration_minutes = duration
                
                # Calculate performance score
                metrics.performance_score = self._calculate_performance_score(metrics)
                
                # Mark task as completed
                newly_available = self.task_queue.complete_task(task.task_id)
                
                # Update system metrics
                self._update_system_metrics(task, metrics)
                
                # Trigger completion callbacks
                await self._trigger_callbacks('on_task_complete', task, newly_available)
                
                logger.info(f"Successfully completed task {task.task_id}")
                
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {str(e)}")
                
                # Mark task as failed
                task.status = ContentTaskStatus.FAILED
                task.metadata['error'] = str(e)
                
                # Update metrics
                metrics = self.task_metrics.get(task.task_id, TaskPerformanceMetrics(task.task_id))
                metrics.actual_completion = datetime.now()
                
                self.task_queue.fail_task(task.task_id)
                
                # Trigger failure callbacks
                await self._trigger_callbacks('on_task_fail', task, str(e))
                
            finally:
                # Release resources
                self.resource_manager.release_resources(task.task_id)
    
    async def _perform_task_execution(self, task: ContentTask) -> None:
        """Perform the actual task execution (placeholder)"""
        # This would integrate with the AI Content Generator Agent
        # For now, simulate execution time based on content type
        
        execution_times = {
            'blog_posts': 2.0,  # 2 seconds
            'social_posts': 0.5,  # 0.5 seconds  
            'email_content': 1.0,  # 1 second
            'case_studies': 3.0,  # 3 seconds
        }
        
        execution_time = execution_times.get(task.content_type.value, 1.0)
        await asyncio.sleep(execution_time)
        
        # Simulate success/failure
        import random
        if random.random() < 0.95:  # 95% success rate
            task.status = ContentTaskStatus.COMPLETED
        else:
            raise Exception("Simulated task execution failure")
    
    def _estimate_task_duration(self, task: ContentTask) -> int:
        """Estimate task duration in minutes"""
        base_durations = {
            'blog_posts': 15,
            'social_posts': 5,
            'email_content': 10,
            'case_studies': 20,
            'newsletters': 12
        }
        
        base_duration = base_durations.get(task.content_type.value, 10)
        
        # Adjust based on word count
        if task.word_count:
            if task.word_count > 1000:
                base_duration *= 1.5
            elif task.word_count > 500:
                base_duration *= 1.2
        
        return int(base_duration)
    
    def _determine_required_resources(self, task: ContentTask) -> List[ResourceType]:
        """Determine required resources for a task"""
        resources = [ResourceType.AI_AGENT]
        
        # Add review slot for high-priority tasks or complex content
        if (task.priority in [ContentTaskPriority.HIGH, ContentTaskPriority.URGENT] or 
            task.content_type.value in ['case_studies', 'blog_posts']):
            resources.append(ResourceType.REVIEW_SLOT)
        
        return resources
    
    def _calculate_performance_score(self, metrics: TaskPerformanceMetrics) -> float:
        """Calculate overall performance score for a task"""
        score = 100.0
        
        # Adjust based on duration vs estimate
        if metrics.estimated_duration_minutes and metrics.actual_duration_minutes:
            duration_ratio = metrics.actual_duration_minutes / metrics.estimated_duration_minutes
            if duration_ratio > 1.2:  # 20% over estimate
                score -= 20
            elif duration_ratio < 0.8:  # 20% under estimate  
                score += 10
        
        # Adjust based on retry count
        if metrics.retry_count > 0:
            score -= metrics.retry_count * 10
        
        # Adjust based on quality score
        if metrics.quality_score:
            if metrics.quality_score >= 8.0:
                score += 10
            elif metrics.quality_score < 6.0:
                score -= 15
        
        return max(0.0, min(100.0, score))
    
    def _update_system_metrics(self, task: ContentTask, metrics: TaskPerformanceMetrics) -> None:
        """Update system-wide performance metrics"""
        self.system_metrics['total_tasks_processed'] += 1
        
        if metrics.actual_duration_minutes:
            self.system_metrics['total_execution_time'] += metrics.actual_duration_minutes
            self.system_metrics['average_task_duration'] = (
                self.system_metrics['total_execution_time'] / 
                self.system_metrics['total_tasks_processed']
            )
        
        # Calculate success rate
        completed_tasks = len([t for t in self.task_metrics.values() if t.actual_completion])
        self.system_metrics['success_rate'] = (completed_tasks / self.system_metrics['total_tasks_processed']) * 100
    
    async def _trigger_callbacks(self, event_type: str, *args) -> None:
        """Trigger registered callbacks for an event"""
        callbacks = self.task_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {str(e)}")
    
    # Public API methods
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for task events"""
        if event_type not in self.task_callbacks:
            self.task_callbacks[event_type] = []
        self.task_callbacks[event_type].append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        queue_status = self.task_queue.get_queue_status()
        resource_utilization = self.resource_manager.get_resource_utilization()
        
        return {
            'is_running': self.is_running,
            'scheduling_strategy': self.scheduling_strategy.value,
            'execution_mode': self.execution_mode.value,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'queue_status': queue_status,
            'resource_utilization': resource_utilization,
            'system_metrics': self.system_metrics
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.task_metrics:
            return None
        
        metrics = self.task_metrics[task_id]
        task = self.task_queue._task_map.get(task_id)
        
        status = {
            'task_id': task_id,
            'status': task.status.value if task else 'unknown',
            'performance_metrics': {
                'estimated_duration_minutes': metrics.estimated_duration_minutes,
                'actual_duration_minutes': metrics.actual_duration_minutes,
                'performance_score': metrics.performance_score,
                'retry_count': metrics.retry_count,
                'quality_score': metrics.quality_score
            }
        }
        
        if metrics.actual_start:
            status['actual_start'] = metrics.actual_start.isoformat()
        if metrics.actual_completion:
            status['actual_completion'] = metrics.actual_completion.isoformat()
        
        return status
    
    async def pause_task(self, task_id: str) -> bool:
        """Pause a specific task (if possible)"""
        # This would require more complex implementation with task interruption
        logger.info(f"Pause request for task {task_id} (not implemented)")
        return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        if task_id in self.task_queue._task_map:
            task = self.task_queue._task_map[task_id]
            task.status = ContentTaskStatus.CANCELLED
            
            # Remove from queue if pending
            # This is simplified - real implementation would need more complex queue manipulation
            
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        # Calculate additional analytics
        task_metrics_list = list(self.task_metrics.values())
        
        completed_tasks = [m for m in task_metrics_list if m.actual_completion]
        performance_scores = [m.performance_score for m in completed_tasks if m.performance_score > 0]
        
        analytics = {
            'system_metrics': self.system_metrics,
            'task_analytics': {
                'total_tasks': len(task_metrics_list),
                'completed_tasks': len(completed_tasks),
                'average_performance_score': sum(performance_scores) / len(performance_scores) if performance_scores else 0,
                'task_distribution_by_type': self._get_task_distribution_by_type(),
                'performance_trends': self._get_performance_trends()
            },
            'resource_analytics': {
                'resource_utilization_history': len(self.resource_manager.resource_usage_history),
                'current_utilization': self.resource_manager.get_resource_utilization(),
                'peak_concurrent_usage': {
                    'ai_agents': max(self.resource_manager.current_ai_agents, self.resource_manager.max_concurrent_ai_agents),
                    'review_slots': max(self.resource_manager.current_review_slots, self.resource_manager.max_concurrent_review_slots)
                }
            }
        }
        
        return analytics
    
    def _get_task_distribution_by_type(self) -> Dict[str, int]:
        """Get distribution of tasks by content type"""
        distribution = defaultdict(int)
        for task in self.task_queue._task_map.values():
            distribution[task.content_type.value] += 1
        return dict(distribution)
    
    def _get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        # This would be more sophisticated in a real implementation
        # returning time-series data for dashboard visualization
        return {
            'completion_rate_trend': [95, 96, 94, 97, 95],  # Last 5 time periods
            'average_duration_trend': [12.5, 11.8, 13.2, 12.0, 11.9],  # Average duration
            'quality_score_trend': [8.2, 8.4, 8.1, 8.5, 8.3]  # Average quality scores
        }

# Create global task management system instance
task_management_system = TaskManagementSystem()