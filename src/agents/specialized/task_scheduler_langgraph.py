"""
LangGraph-enhanced Task Scheduler Workflow for intelligent campaign task scheduling and automation.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None  # PostgreSQL checkpointing not available
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
# Removed broken import: from .task_scheduler import TaskSchedulerAgent, PlatformSchedule, ScheduledPost
# from ...config.database import DatabaseConnection  # Temporarily disabled


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

@dataclass
class PlatformSchedule:
    """Platform posting schedule configuration."""
    platform: str
    optimal_times: List[str]  # List of optimal posting times
    frequency: str  # daily, weekly, etc.
    content_types: List[str] = field(default_factory=list)

@dataclass  
class ScheduledPost:
    """Scheduled social media post."""
    content: str
    platform: str
    scheduled_time: str
    status: str = "pending"
    post_id: str = ""


class SchedulingStrategy(Enum):
    """Scheduling optimization strategies."""
    ENGAGEMENT_OPTIMIZED = "engagement_optimized"
    FREQUENCY_BALANCED = "frequency_balanced"
    RUSH_MODE = "rush_mode"
    CUSTOM_TIMELINE = "custom_timeline"


@dataclass
class TaskDependency:
    """Task dependency configuration."""
    dependent_task_id: str
    dependency_task_id: str
    dependency_type: str = "completion"  # completion, approval, timing
    delay_hours: int = 0


@dataclass
class SchedulingConstraint:
    """Scheduling constraint definition."""
    constraint_type: str  # time_window, platform_limit, resource_limit
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM


@dataclass
class TaskSchedulingMetrics:
    """Task scheduling performance metrics."""
    total_tasks_scheduled: int = 0
    average_scheduling_time: float = 0.0
    constraint_violations: int = 0
    optimization_score: float = 0.0
    platform_distribution: Dict[str, int] = field(default_factory=dict)
    time_slot_utilization: Dict[str, float] = field(default_factory=dict)


class TaskSchedulingState(WorkflowState):
    """Enhanced state for task scheduling workflow."""
    
    # Input parameters
    campaign_id: str = ""
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    scheduling_strategy: SchedulingStrategy = SchedulingStrategy.ENGAGEMENT_OPTIMIZED
    time_window: Dict[str, datetime] = field(default_factory=dict)
    
    # Constraint management
    scheduling_constraints: List[SchedulingConstraint] = field(default_factory=list)
    task_dependencies: List[TaskDependency] = field(default_factory=list)
    platform_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Analysis results
    task_analysis: Dict[str, Any] = field(default_factory=dict)
    platform_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimal_schedule_matrix: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Scheduling results
    scheduled_tasks: List[Dict[str, Any]] = field(default_factory=list)
    scheduling_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    scheduling_metrics: TaskSchedulingMetrics = field(default_factory=TaskSchedulingMetrics)
    optimization_iterations: int = 0
    constraint_satisfaction_score: float = 0.0
    
    # Calendar integration
    content_calendar: List[Dict[str, Any]] = field(default_factory=list)
    calendar_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class TaskSchedulerWorkflow(LangGraphWorkflowBase[TaskSchedulingState]):
    """LangGraph workflow for advanced task scheduling with dependency management and optimization."""
    
    def __init__(
        self, 
        workflow_name: str = "task_scheduler_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = TaskSchedulerAgent()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> TaskSchedulingState:
        """Create initial workflow state from context."""
        return TaskSchedulingState(
            workflow_id=context.get("workflow_id", f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            campaign_id=context.get("campaign_id", ""),
            tasks=context.get("tasks", []),
            scheduling_strategy=SchedulingStrategy(context.get("scheduling_strategy", "engagement_optimized")),
            time_window=context.get("time_window", {
                "start": datetime.now(),
                "end": datetime.now() + timedelta(days=30)
            }),
            scheduling_constraints=context.get("scheduling_constraints", []),
            task_dependencies=context.get("task_dependencies", []),
            platform_limits=context.get("platform_limits", {}),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the task scheduling workflow graph."""
        workflow = StateGraph(TaskSchedulingState)
        
        # Define workflow nodes
        workflow.add_node("validate_inputs", self._validate_inputs_node)
        workflow.add_node("analyze_tasks", self._analyze_tasks_node)
        workflow.add_node("analyze_platforms", self._analyze_platforms_node)
        workflow.add_node("resolve_dependencies", self._resolve_dependencies_node)
        workflow.add_node("generate_schedule_matrix", self._generate_schedule_matrix_node)
        workflow.add_node("optimize_schedule", self._optimize_schedule_node)
        workflow.add_node("validate_constraints", self._validate_constraints_node)
        workflow.add_node("finalize_schedule", self._finalize_schedule_node)
        workflow.add_node("create_calendar", self._create_calendar_node)
        
        # Define workflow edges
        workflow.add_edge("validate_inputs", "analyze_tasks")
        workflow.add_edge("analyze_tasks", "analyze_platforms") 
        workflow.add_edge("analyze_platforms", "resolve_dependencies")
        workflow.add_edge("resolve_dependencies", "generate_schedule_matrix")
        workflow.add_edge("generate_schedule_matrix", "optimize_schedule")
        
        # Conditional routing for optimization iterations
        workflow.add_conditional_edges(
            "optimize_schedule",
            self._should_continue_optimization,
            {
                "continue": "optimize_schedule",
                "validate": "validate_constraints"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_constraints",
            self._check_constraint_satisfaction,
            {
                "regenerate": "generate_schedule_matrix",
                "finalize": "finalize_schedule"
            }
        )
        
        workflow.add_edge("finalize_schedule", "create_calendar")
        workflow.add_edge("create_calendar", END)
        
        # Set entry point
        workflow.set_entry_point("validate_inputs")
        
        return workflow
    
    async def _validate_inputs_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Validate input parameters and scheduling requirements."""
        try:
            self._log_progress("Validating task scheduling inputs and requirements")
            
            validation_errors = []
            
            # Validate campaign ID
            if not state.campaign_id:
                validation_errors.append("Campaign ID is required")
            
            # Validate tasks
            if not state.tasks:
                validation_errors.append("At least one task is required for scheduling")
            
            # Validate time window
            if not state.time_window.get("start") or not state.time_window.get("end"):
                # Set default time window
                state.time_window = {
                    "start": datetime.now(),
                    "end": datetime.now() + timedelta(days=30)
                }
                self._log_progress("Set default 30-day scheduling window")
            
            # Validate time window logic
            if state.time_window["start"] >= state.time_window["end"]:
                validation_errors.append("Start time must be before end time")
            
            # Validate task structure
            required_task_fields = ["id", "task_type", "content"]
            for i, task in enumerate(state.tasks):
                for field in required_task_fields:
                    if field not in task:
                        validation_errors.append(f"Task {i+1} missing required field: {field}")
            
            # Set default constraints if none provided
            if not state.scheduling_constraints:
                state.scheduling_constraints = [
                    SchedulingConstraint(
                        constraint_type="platform_limit",
                        parameters={"max_posts_per_day": 3, "min_hours_between_posts": 2}
                    ),
                    SchedulingConstraint(
                        constraint_type="time_window", 
                        parameters={"business_hours_only": True, "exclude_weekends": False}
                    )
                ]
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 12.0
                
                state.messages.append(HumanMessage(
                    content=f"Input validation completed. Processing {len(state.tasks)} tasks "
                           f"for campaign {state.campaign_id} with {state.scheduling_strategy.value} strategy."
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Input validation failed: {str(e)}"
            return state
    
    async def _analyze_tasks_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Analyze tasks for scheduling complexity and requirements."""
        try:
            self._log_progress("Analyzing task characteristics and requirements")
            
            task_analysis = {
                "total_tasks": len(state.tasks),
                "task_types": {},
                "priority_distribution": {},
                "estimated_durations": {},
                "platform_distribution": {},
                "content_complexity": {},
                "dependency_count": len(state.task_dependencies),
                "urgency_analysis": {}
            }
            
            # Analyze each task
            for task in state.tasks:
                task_id = task["id"]
                task_type = task.get("task_type", "unknown")
                platform = task.get("metadata", {}).get("platform", "unknown")
                priority = task.get("priority", "medium")
                
                # Count task types
                task_analysis["task_types"][task_type] = task_analysis["task_types"].get(task_type, 0) + 1
                
                # Count priority distribution
                task_analysis["priority_distribution"][priority] = \
                    task_analysis["priority_distribution"].get(priority, 0) + 1
                
                # Count platform distribution
                task_analysis["platform_distribution"][platform] = \
                    task_analysis["platform_distribution"].get(platform, 0) + 1
                
                # Estimate task duration based on type and complexity
                duration = self._estimate_task_duration(task)
                task_analysis["estimated_durations"][task_id] = duration
                
                # Analyze content complexity
                content_complexity = self._analyze_content_complexity(task.get("content", ""))
                task_analysis["content_complexity"][task_id] = content_complexity
                
                # Analyze urgency
                urgency_score = self._calculate_urgency_score(task, state.time_window)
                task_analysis["urgency_analysis"][task_id] = urgency_score
            
            # Calculate overall metrics
            total_estimated_time = sum(task_analysis["estimated_durations"].values())
            available_time_days = (state.time_window["end"] - state.time_window["start"]).days
            
            task_analysis["scheduling_feasibility"] = {
                "total_estimated_hours": total_estimated_time,
                "available_days": available_time_days,
                "average_tasks_per_day": len(state.tasks) / max(available_time_days, 1),
                "workload_intensity": "high" if total_estimated_time / max(available_time_days, 1) > 8 else "moderate"
            }
            
            state.task_analysis = task_analysis
            state.progress_percentage = 25.0
            
            state.messages.append(SystemMessage(
                content=f"Task analysis completed. {task_analysis['total_tasks']} tasks analyzed. "
                       f"Workload intensity: {task_analysis['scheduling_feasibility']['workload_intensity']}. "
                       f"Platform distribution: {dict(list(task_analysis['platform_distribution'].items())[:3])}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Task analysis failed: {str(e)}"
            return state
    
    async def _analyze_platforms_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Analyze platform-specific scheduling requirements and optimal times."""
        try:
            self._log_progress("Analyzing platform scheduling requirements and optimal times")
            
            platform_analysis = {}
            platforms = set(task.get("metadata", {}).get("platform", "unknown") for task in state.tasks)
            
            for platform in platforms:
                if platform == "unknown":
                    continue
                
                # Get platform schedule from legacy agent
                platform_schedule = self.legacy_agent.platform_schedules.get(platform)
                
                if platform_schedule:
                    # Analyze platform requirements
                    analysis = {
                        "optimal_posting_times": platform_schedule.best_times,
                        "posting_frequency": platform_schedule.posting_frequency,
                        "supported_content_types": platform_schedule.content_types,
                        "engagement_optimization": platform_schedule.engagement_optimization,
                        "best_days": platform_schedule.engagement_optimization.get("best_days", []),
                        "content_requirements": {
                            "max_length": platform_schedule.engagement_optimization.get("content_length"),
                            "hashtag_count": platform_schedule.engagement_optimization.get("hashtag_count", 0)
                        }
                    }
                    
                    # Calculate platform-specific metrics
                    platform_tasks = [task for task in state.tasks 
                                    if task.get("metadata", {}).get("platform") == platform]
                    
                    analysis["task_count"] = len(platform_tasks)
                    analysis["content_volume"] = sum(
                        len(task.get("content", "")) for task in platform_tasks
                    )
                    
                    # Estimate optimal time slots for this platform
                    time_slots = self._generate_platform_time_slots(
                        platform_schedule, state.time_window
                    )
                    analysis["available_time_slots"] = len(time_slots)
                    analysis["time_slots"] = time_slots[:20]  # First 20 slots for reference
                    
                    # Calculate capacity utilization
                    max_posts_per_week = self._calculate_max_posts_per_week(platform_schedule)
                    weeks_available = max((state.time_window["end"] - state.time_window["start"]).days / 7, 1)
                    total_capacity = max_posts_per_week * weeks_available
                    
                    analysis["capacity_analysis"] = {
                        "max_posts_per_week": max_posts_per_week,
                        "total_capacity": int(total_capacity),
                        "utilization_percentage": min(100, (analysis["task_count"] / total_capacity) * 100)
                    }
                    
                else:
                    # Default analysis for unknown platforms
                    analysis = {
                        "optimal_posting_times": ["09:00", "14:00", "17:00"],
                        "posting_frequency": "3x per week",
                        "task_count": len([task for task in state.tasks 
                                         if task.get("metadata", {}).get("platform") == platform]),
                        "capacity_analysis": {"utilization_percentage": 50.0}
                    }
                
                platform_analysis[platform] = analysis
            
            state.platform_analysis = platform_analysis
            state.progress_percentage = 37.0
            
            # Calculate overall platform utilization
            avg_utilization = sum(
                analysis.get("capacity_analysis", {}).get("utilization_percentage", 0)
                for analysis in platform_analysis.values()
            ) / max(len(platform_analysis), 1)
            
            state.messages.append(SystemMessage(
                content=f"Platform analysis completed for {len(platform_analysis)} platforms. "
                       f"Average capacity utilization: {avg_utilization:.1f}%. "
                       f"Total available time slots across platforms: {sum(analysis.get('available_time_slots', 0) for analysis in platform_analysis.values())}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Platform analysis failed: {str(e)}"
            return state
    
    async def _resolve_dependencies_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Resolve task dependencies and create execution order."""
        try:
            self._log_progress("Resolving task dependencies and creating execution order")
            
            if not state.task_dependencies:
                # No dependencies - create simple priority-based ordering
                execution_order = self._create_priority_based_order(state.tasks)
                state.task_analysis["execution_order"] = execution_order
                state.task_analysis["dependency_resolution"] = {"has_dependencies": False}
                state.progress_percentage = 50.0
                return state
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(state.tasks, state.task_dependencies)
            
            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(dependency_graph)
            if circular_deps:
                state.status = WorkflowStatus.FAILED
                state.error_message = f"Circular dependencies detected: {circular_deps}"
                return state
            
            # Perform topological sort to determine execution order
            execution_order = self._topological_sort(dependency_graph)
            
            # Calculate dependency metrics
            dependency_analysis = {
                "has_dependencies": True,
                "dependency_count": len(state.task_dependencies),
                "dependency_levels": self._calculate_dependency_levels(dependency_graph),
                "critical_path": self._find_critical_path(dependency_graph, state.task_analysis["estimated_durations"]),
                "parallel_execution_groups": self._identify_parallel_groups(dependency_graph)
            }
            
            # Validate dependency timing constraints
            timing_conflicts = self._validate_dependency_timing(
                state.task_dependencies, state.task_analysis["estimated_durations"]
            )
            
            if timing_conflicts:
                dependency_analysis["timing_conflicts"] = timing_conflicts
                self._log_progress(f"Found {len(timing_conflicts)} timing conflicts in dependencies")
            
            state.task_analysis["execution_order"] = execution_order
            state.task_analysis["dependency_resolution"] = dependency_analysis
            state.progress_percentage = 50.0
            
            state.messages.append(SystemMessage(
                content=f"Dependency resolution completed. {len(state.task_dependencies)} dependencies processed. "
                       f"Critical path length: {len(dependency_analysis.get('critical_path', []))} tasks. "
                       f"Parallel execution groups: {len(dependency_analysis.get('parallel_execution_groups', []))}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Dependency resolution failed: {str(e)}"
            return state
    
    async def _generate_schedule_matrix_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Generate optimal schedule matrix considering all constraints."""
        try:
            self._log_progress("Generating optimal schedule matrix with constraint consideration")
            
            schedule_matrix = {}
            
            # Get execution order from dependency resolution
            execution_order = state.task_analysis.get("execution_order", [task["id"] for task in state.tasks])
            
            # Generate schedule for each platform
            for platform, analysis in state.platform_analysis.items():
                if platform == "unknown":
                    continue
                
                platform_tasks = [task for task in state.tasks 
                                if task.get("metadata", {}).get("platform") == platform 
                                and task["id"] in execution_order]
                
                if not platform_tasks:
                    continue
                
                # Generate time slots for this platform
                time_slots = analysis.get("time_slots", [])
                
                # Create schedule for platform tasks
                platform_schedule = []
                slot_index = 0
                
                for task in platform_tasks:
                    if slot_index >= len(time_slots):
                        # Generate more time slots if needed
                        additional_slots = self._generate_additional_time_slots(
                            platform, analysis, state.time_window, 
                            start_date=time_slots[-1]["datetime"] if time_slots else state.time_window["start"]
                        )
                        time_slots.extend(additional_slots)
                    
                    if slot_index < len(time_slots):
                        scheduled_slot = time_slots[slot_index]
                        
                        # Apply dependency delays
                        adjusted_time = self._apply_dependency_delays(
                            task["id"], scheduled_slot["datetime"], state.task_dependencies, schedule_matrix
                        )
                        
                        schedule_entry = {
                            "task_id": task["id"],
                            "task_type": task.get("task_type"),
                            "platform": platform,
                            "scheduled_time": adjusted_time,
                            "estimated_duration": state.task_analysis["estimated_durations"].get(task["id"], 1.0),
                            "priority": task.get("priority", "medium"),
                            "content": task.get("content", ""),
                            "metadata": task.get("metadata", {}),
                            "slot_quality_score": scheduled_slot.get("quality_score", 50.0)
                        }
                        
                        platform_schedule.append(schedule_entry)
                        slot_index += 1
                
                schedule_matrix[platform] = platform_schedule
            
            # Validate schedule against constraints
            constraint_violations = self._validate_schedule_constraints(
                schedule_matrix, state.scheduling_constraints
            )
            
            state.optimal_schedule_matrix = schedule_matrix
            state.scheduling_conflicts = constraint_violations
            state.progress_percentage = 62.0
            
            total_scheduled = sum(len(schedule) for schedule in schedule_matrix.values())
            state.messages.append(SystemMessage(
                content=f"Schedule matrix generated. {total_scheduled} tasks scheduled across {len(schedule_matrix)} platforms. "
                       f"Constraint violations: {len(constraint_violations)}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Schedule matrix generation failed: {str(e)}"
            return state
    
    async def _optimize_schedule_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Optimize schedule for maximum engagement and constraint satisfaction."""
        try:
            self._log_progress(f"Optimizing schedule (iteration {state.optimization_iterations + 1})")
            
            if state.optimization_iterations >= 3:  # Limit optimization iterations
                return state
            
            # Calculate current optimization score
            current_score = self._calculate_optimization_score(
                state.optimal_schedule_matrix, state.platform_analysis, state.scheduling_conflicts
            )
            
            # Apply optimization techniques
            optimized_matrix = {}
            
            for platform, schedule in state.optimal_schedule_matrix.items():
                platform_analysis = state.platform_analysis.get(platform, {})
                
                # Optimization strategies based on scheduling strategy
                if state.scheduling_strategy == SchedulingStrategy.ENGAGEMENT_OPTIMIZED:
                    optimized_schedule = self._optimize_for_engagement(schedule, platform_analysis)
                elif state.scheduling_strategy == SchedulingStrategy.FREQUENCY_BALANCED:
                    optimized_schedule = self._optimize_for_balance(schedule, platform_analysis)
                elif state.scheduling_strategy == SchedulingStrategy.RUSH_MODE:
                    optimized_schedule = self._optimize_for_speed(schedule, platform_analysis)
                else:
                    optimized_schedule = schedule  # Keep original for custom timeline
                
                optimized_matrix[platform] = optimized_schedule
            
            # Calculate new optimization score
            new_score = self._calculate_optimization_score(
                optimized_matrix, state.platform_analysis, state.scheduling_conflicts
            )
            
            # Accept optimization if it improves the score
            if new_score > current_score:
                state.optimal_schedule_matrix = optimized_matrix
                self._log_progress(f"Optimization improved score from {current_score:.2f} to {new_score:.2f}")
            
            # Update optimization metrics
            state.optimization_iterations += 1
            state.scheduling_metrics.optimization_score = max(current_score, new_score)
            state.progress_percentage = min(75.0, 62.0 + (state.optimization_iterations * 4))
            
            state.messages.append(SystemMessage(
                content=f"Schedule optimization iteration {state.optimization_iterations} completed. "
                       f"Current optimization score: {state.scheduling_metrics.optimization_score:.2f}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Schedule optimization failed: {str(e)}"
            return state
    
    async def _validate_constraints_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Validate final schedule against all constraints."""
        try:
            self._log_progress("Validating schedule against all constraints")
            
            # Re-validate constraints after optimization
            constraint_violations = self._validate_schedule_constraints(
                state.optimal_schedule_matrix, state.scheduling_constraints
            )
            
            # Calculate constraint satisfaction score
            total_constraints = len(state.scheduling_constraints) * sum(
                len(schedule) for schedule in state.optimal_schedule_matrix.values()
            )
            violation_count = len(constraint_violations)
            
            satisfaction_score = max(0, (total_constraints - violation_count) / max(total_constraints, 1) * 100)
            
            state.scheduling_conflicts = constraint_violations
            state.constraint_satisfaction_score = satisfaction_score
            state.progress_percentage = 87.0
            
            # Classify violation severity
            critical_violations = [v for v in constraint_violations if v.get("severity") == "critical"]
            warning_violations = [v for v in constraint_violations if v.get("severity") == "warning"]
            
            state.messages.append(SystemMessage(
                content=f"Constraint validation completed. Satisfaction score: {satisfaction_score:.1f}%. "
                       f"Critical violations: {len(critical_violations)}, Warnings: {len(warning_violations)}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Constraint validation failed: {str(e)}"
            return state
    
    async def _finalize_schedule_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Finalize the schedule and prepare for execution."""
        try:
            self._log_progress("Finalizing schedule and preparing execution plan")
            
            # Flatten schedule matrix into final scheduled tasks list
            scheduled_tasks = []
            
            for platform, schedule in state.optimal_schedule_matrix.items():
                for entry in schedule:
                    scheduled_task = {
                        "id": str(uuid.uuid4()),
                        "task_id": entry["task_id"],
                        "campaign_id": state.campaign_id,
                        "platform": platform,
                        "content": entry["content"],
                        "scheduled_at": entry["scheduled_time"],
                        "status": TaskStatus.SCHEDULED.value,
                        "priority": entry["priority"],
                        "estimated_duration": entry["estimated_duration"],
                        "metadata": {
                            **entry["metadata"],
                            "slot_quality_score": entry.get("slot_quality_score", 50.0),
                            "scheduling_strategy": state.scheduling_strategy.value,
                            "optimization_score": state.scheduling_metrics.optimization_score
                        }
                    }
                    scheduled_tasks.append(scheduled_task)
            
            # Sort by scheduled time
            scheduled_tasks.sort(key=lambda x: x["scheduled_at"])
            
            # Calculate final metrics
            metrics = TaskSchedulingMetrics(
                total_tasks_scheduled=len(scheduled_tasks),
                average_scheduling_time=sum(
                    entry["estimated_duration"] 
                    for schedule in state.optimal_schedule_matrix.values() 
                    for entry in schedule
                ) / max(len(scheduled_tasks), 1),
                constraint_violations=len(state.scheduling_conflicts),
                optimization_score=state.scheduling_metrics.optimization_score,
                platform_distribution={
                    platform: len(schedule) 
                    for platform, schedule in state.optimal_schedule_matrix.items()
                }
            )
            
            # Calculate time slot utilization
            time_slots_used = {}
            for task in scheduled_tasks:
                hour = task["scheduled_at"].hour
                time_slot = f"{hour:02d}:00"
                time_slots_used[time_slot] = time_slots_used.get(time_slot, 0) + 1
            
            total_possible_slots = len(scheduled_tasks)
            metrics.time_slot_utilization = {
                slot: count / total_possible_slots 
                for slot, count in time_slots_used.items()
            }
            
            state.scheduled_tasks = scheduled_tasks
            state.scheduling_metrics = metrics
            state.status = WorkflowStatus.COMPLETED
            state.progress_percentage = 95.0
            state.completed_at = datetime.utcnow()
            
            state.messages.append(SystemMessage(
                content=f"Schedule finalized. {len(scheduled_tasks)} tasks scheduled with "
                       f"{metrics.optimization_score:.1f} optimization score. "
                       f"Average task duration: {metrics.average_scheduling_time:.1f} hours."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Schedule finalization failed: {str(e)}"
            return state
    
    async def _create_calendar_node(self, state: TaskSchedulingState) -> TaskSchedulingState:
        """Create content calendar entries for the scheduled tasks."""
        try:
            self._log_progress("Creating content calendar entries")
            
            calendar_entries = []
            calendar_conflicts = []
            
            # Group tasks by date for calendar creation
            tasks_by_date = {}
            for task in state.scheduled_tasks:
                date_key = task["scheduled_at"].date()
                if date_key not in tasks_by_date:
                    tasks_by_date[date_key] = []
                tasks_by_date[date_key].append(task)
            
            # Create calendar entries
            for date, tasks in tasks_by_date.items():
                # Check for potential conflicts (too many tasks on same day/time)
                time_conflicts = self._detect_time_conflicts(tasks)
                if time_conflicts:
                    calendar_conflicts.extend(time_conflicts)
                
                # Create calendar entry for each task
                for task in tasks:
                    calendar_entry = {
                        "id": str(uuid.uuid4()),
                        "campaign_id": state.campaign_id,
                        "date": date,
                        "time": task["scheduled_at"].time(),
                        "title": f"{task['platform'].title()} {task.get('task_type', 'Post').title()}",
                        "description": task["content"][:200] + "..." if len(task["content"]) > 200 else task["content"],
                        "platform": task["platform"],
                        "task_id": task["task_id"],
                        "scheduled_task_id": task["id"],
                        "status": "planned",
                        "priority": task["priority"],
                        "estimated_duration_minutes": int(task["estimated_duration"] * 60),
                        "metadata": {
                            "content_preview": task["content"][:100],
                            "optimization_score": task["metadata"].get("optimization_score", 0)
                        }
                    }
                    calendar_entries.append(calendar_entry)
            
            state.content_calendar = calendar_entries
            state.calendar_conflicts = calendar_conflicts
            state.progress_percentage = 100.0
            
            state.messages.append(SystemMessage(
                content=f"Content calendar created with {len(calendar_entries)} entries. "
                       f"Calendar conflicts detected: {len(calendar_conflicts)}"
            ))
            
            return state
            
        except Exception as e:
            # Calendar creation failure shouldn't fail the entire workflow
            self._log_error(f"Calendar creation failed: {str(e)}")
            state.messages.append(SystemMessage(
                content="Calendar creation failed, but task scheduling completed successfully."
            ))
            state.progress_percentage = 100.0
            return state
    
    def _should_continue_optimization(self, state: TaskSchedulingState) -> str:
        """Determine if optimization should continue."""
        if state.optimization_iterations >= 3:
            return "validate"
        
        # Continue if we have significant constraint violations
        critical_violations = len([v for v in state.scheduling_conflicts if v.get("severity") == "critical"])
        if critical_violations > 0 and state.optimization_iterations < 2:
            return "continue"
        
        return "validate"
    
    def _check_constraint_satisfaction(self, state: TaskSchedulingState) -> str:
        """Check if constraint satisfaction is acceptable."""
        if state.constraint_satisfaction_score < 70 and state.optimization_iterations < 3:
            return "regenerate"
        return "finalize"
    
    # Helper methods for task analysis and optimization
    
    def _estimate_task_duration(self, task: Dict[str, Any]) -> float:
        """Estimate task duration in hours based on type and complexity."""
        task_type = task.get("task_type", "unknown")
        content_length = len(task.get("content", ""))
        
        base_durations = {
            "content_creation": 2.0,
            "image_generation": 1.5,
            "social_media_post": 0.5,
            "email_campaign": 3.0,
            "blog_post": 4.0,
            "video_content": 6.0,
            "unknown": 1.0
        }
        
        base_duration = base_durations.get(task_type, 1.0)
        
        # Adjust for content complexity
        if content_length > 1000:
            base_duration *= 1.5
        elif content_length > 500:
            base_duration *= 1.2
        
        return base_duration
    
    def _analyze_content_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze content complexity for scheduling purposes."""
        return {
            "length": len(content),
            "word_count": len(content.split()),
            "has_links": "http" in content.lower(),
            "has_hashtags": "#" in content,
            "has_mentions": "@" in content,
            "complexity_score": min(100, len(content) / 10 + len(content.split()) / 5)
        }
    
    def _calculate_urgency_score(self, task: Dict[str, Any], time_window: Dict[str, datetime]) -> float:
        """Calculate urgency score for task prioritization."""
        priority = task.get("priority", "medium")
        
        priority_scores = {
            "low": 25.0,
            "medium": 50.0, 
            "high": 75.0,
            "urgent": 95.0
        }
        
        base_score = priority_scores.get(priority, 50.0)
        
        # Adjust based on deadline if present
        deadline = task.get("deadline")
        if deadline and isinstance(deadline, datetime):
            days_until_deadline = (deadline - datetime.now()).days
            if days_until_deadline < 1:
                base_score += 20
            elif days_until_deadline < 3:
                base_score += 10
        
        return min(100.0, base_score)
    
    def _generate_platform_time_slots(self, platform_schedule: PlatformSchedule, time_window: Dict[str, datetime]) -> List[Dict[str, Any]]:
        """Generate optimal time slots for a platform."""
        slots = []
        current_date = time_window["start"]
        end_date = time_window["end"]
        
        while current_date < end_date:
            # Check if this is a good day for the platform
            if self.legacy_agent._is_good_day(current_date, platform_schedule):
                # Add time slots for each optimal posting time
                for time_str in platform_schedule.best_times:
                    hour, minute = map(int, time_str.split(":"))
                    slot_datetime = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    if slot_datetime > datetime.now() and slot_datetime < end_date:
                        slots.append({
                            "datetime": slot_datetime,
                            "time_str": time_str,
                            "day_of_week": current_date.strftime("%A"),
                            "quality_score": self._calculate_time_slot_quality(slot_datetime, platform_schedule)
                        })
            
            current_date += timedelta(days=1)
        
        return sorted(slots, key=lambda x: x["quality_score"], reverse=True)
    
    def _calculate_time_slot_quality(self, slot_datetime: datetime, platform_schedule: PlatformSchedule) -> float:
        """Calculate quality score for a time slot."""
        base_score = 50.0
        
        # Boost for optimal times
        time_str = slot_datetime.strftime("%H:%M")
        if time_str in platform_schedule.best_times:
            base_score += 30
        
        # Boost for optimal days
        day_name = slot_datetime.strftime("%A")
        best_days = platform_schedule.engagement_optimization.get("best_days", [])
        if day_name in best_days:
            base_score += 20
        
        return base_score
    
    def _calculate_max_posts_per_week(self, platform_schedule: PlatformSchedule) -> int:
        """Calculate maximum posts per week for a platform."""
        frequency = platform_schedule.posting_frequency.lower()
        
        if "daily" in frequency or "1x per day" in frequency:
            return 7
        elif "5x per week" in frequency:
            return 5
        elif "3x per week" in frequency:
            return 3
        elif "weekly" in frequency or "1x per week" in frequency:
            return 1
        else:
            return 3  # Default
    
    async def execute_workflow(
        self,
        campaign_id: str,
        tasks: List[Dict[str, Any]],
        scheduling_strategy: SchedulingStrategy = SchedulingStrategy.ENGAGEMENT_OPTIMIZED,
        time_window: Optional[Dict[str, datetime]] = None,
        scheduling_constraints: Optional[List[SchedulingConstraint]] = None,
        task_dependencies: Optional[List[TaskDependency]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the task scheduling workflow."""
        
        context = {
            "campaign_id": campaign_id,
            "tasks": tasks,
            "scheduling_strategy": scheduling_strategy.value,
            "time_window": time_window or {
                "start": datetime.now(),
                "end": datetime.now() + timedelta(days=30)
            },
            "scheduling_constraints": scheduling_constraints or [],
            "task_dependencies": task_dependencies or [],
            "workflow_id": f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "scheduled_tasks": [
                        {
                            "id": task["id"],
                            "task_id": task["task_id"],
                            "platform": task["platform"],
                            "content": task["content"],
                            "scheduled_at": task["scheduled_at"].isoformat(),
                            "status": task["status"],
                            "priority": task["priority"],
                            "estimated_duration_hours": task["estimated_duration"],
                            "metadata": task["metadata"]
                        }
                        for task in final_state.scheduled_tasks
                    ],
                    "content_calendar": final_state.content_calendar,
                    "scheduling_metrics": {
                        "total_tasks_scheduled": final_state.scheduling_metrics.total_tasks_scheduled,
                        "optimization_score": final_state.scheduling_metrics.optimization_score,
                        "constraint_violations": final_state.scheduling_metrics.constraint_violations,
                        "platform_distribution": final_state.scheduling_metrics.platform_distribution,
                        "constraint_satisfaction_score": final_state.constraint_satisfaction_score
                    },
                    "scheduling_conflicts": final_state.scheduling_conflicts,
                    "workflow_summary": {
                        "strategy_used": scheduling_strategy.value,
                        "optimization_iterations": final_state.optimization_iterations,
                        "total_processing_time_seconds": (final_state.completed_at - final_state.created_at).total_seconds()
                    }
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "tasks_scheduled": len(final_state.scheduled_tasks),
                        "optimization_score": final_state.scheduling_metrics.optimization_score
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Workflow failed",
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "final_status": final_state.status.value
                    }
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=f"Workflow execution failed: {str(e)}",
                metadata={"error_type": "workflow_execution_error"}
            )
    
    # Additional helper methods for dependency resolution and optimization
    
    def _create_priority_based_order(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Create execution order based on task priorities."""
        priority_order = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
        
        sorted_tasks = sorted(
            tasks,
            key=lambda x: (
                priority_order.get(x.get("priority", "medium"), 2),
                x.get("created_at", datetime.now())
            ),
            reverse=True
        )
        
        return [task["id"] for task in sorted_tasks]
    
    def _build_dependency_graph(self, tasks: List[Dict[str, Any]], dependencies: List[TaskDependency]) -> Dict[str, List[str]]:
        """Build dependency graph from task dependencies."""
        graph = {task["id"]: [] for task in tasks}
        
        for dep in dependencies:
            if dep.dependency_task_id in graph:
                graph[dep.dependency_task_id].append(dep.dependent_task_id)
        
        return graph
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Detect circular dependencies in the dependency graph."""
        # Implementation of cycle detection algorithm
        visited = set()
        rec_stack = set()
        cycles = []
        
        def has_cycle(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependency_graph.get(node, []):
                if has_cycle(neighbor, path):
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for node in dependency_graph:
            if node not in visited:
                has_cycle(node, [])
        
        return cycles
    
    def _topological_sort(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort to determine execution order."""
        in_degree = {node: 0 for node in dependency_graph}
        
        # Calculate in-degrees
        for node in dependency_graph:
            for neighbor in dependency_graph[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Find nodes with no dependencies
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in dependency_graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _calculate_dependency_levels(self, dependency_graph: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate dependency levels for each task."""
        levels = {}
        
        def calculate_level(node):
            if node in levels:
                return levels[node]
            
            max_level = 0
            for predecessor in dependency_graph:
                if node in dependency_graph[predecessor]:
                    max_level = max(max_level, calculate_level(predecessor) + 1)
            
            levels[node] = max_level
            return max_level
        
        for node in dependency_graph:
            calculate_level(node)
        
        return levels
    
    def _find_critical_path(self, dependency_graph: Dict[str, List[str]], durations: Dict[str, float]) -> List[str]:
        """Find the critical path through the dependency graph."""
        # Simplified critical path calculation
        topological_order = self._topological_sort(dependency_graph)
        
        # Calculate earliest start times
        earliest_start = {node: 0 for node in topological_order}
        
        for node in topological_order:
            for successor in dependency_graph.get(node, []):
                earliest_start[successor] = max(
                    earliest_start[successor],
                    earliest_start[node] + durations.get(node, 1.0)
                )
        
        # Find the critical path by backtracking
        max_time_node = max(earliest_start, key=earliest_start.get)
        critical_path = [max_time_node]
        
        current_node = max_time_node
        while True:
            predecessors = [node for node in dependency_graph if current_node in dependency_graph[node]]
            if not predecessors:
                break
            
            # Find predecessor with maximum earliest start time
            critical_predecessor = max(
                predecessors,
                key=lambda x: earliest_start[x] + durations.get(x, 1.0)
            )
            critical_path.append(critical_predecessor)
            current_node = critical_predecessor
        
        return list(reversed(critical_path))
    
    def _identify_parallel_groups(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Identify groups of tasks that can be executed in parallel."""
        levels = self._calculate_dependency_levels(dependency_graph)
        
        # Group tasks by dependency level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Return groups with more than one task (parallel execution possible)
        return [group for group in level_groups.values() if len(group) > 1]
    
    def _validate_dependency_timing(self, dependencies: List[TaskDependency], durations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Validate timing constraints in dependencies."""
        conflicts = []
        
        for dep in dependencies:
            dependency_duration = durations.get(dep.dependency_task_id, 1.0)
            
            if dep.delay_hours < dependency_duration:
                conflicts.append({
                    "type": "timing_conflict",
                    "description": f"Delay of {dep.delay_hours}h is less than dependency duration of {dependency_duration}h",
                    "dependency": dep.dependency_task_id,
                    "dependent": dep.dependent_task_id,
                    "severity": "warning"
                })
        
        return conflicts
    
    def _apply_dependency_delays(self, task_id: str, base_time: datetime, dependencies: List[TaskDependency], schedule_matrix: Dict[str, List[Dict[str, Any]]]) -> datetime:
        """Apply dependency delays to task scheduling time."""
        adjusted_time = base_time
        
        for dep in dependencies:
            if dep.dependent_task_id == task_id:
                # Find the completion time of the dependency task
                for platform_schedule in schedule_matrix.values():
                    for scheduled_task in platform_schedule:
                        if scheduled_task["task_id"] == dep.dependency_task_id:
                            dependency_end_time = scheduled_task["scheduled_time"] + timedelta(
                                hours=scheduled_task["estimated_duration"]
                            )
                            required_start_time = dependency_end_time + timedelta(hours=dep.delay_hours)
                            adjusted_time = max(adjusted_time, required_start_time)
                            break
        
        return adjusted_time
    
    def _validate_schedule_constraints(self, schedule_matrix: Dict[str, List[Dict[str, Any]]], constraints: List[SchedulingConstraint]) -> List[Dict[str, Any]]:
        """Validate schedule against constraints."""
        violations = []
        
        for constraint in constraints:
            constraint_type = constraint.constraint_type
            parameters = constraint.parameters
            
            if constraint_type == "platform_limit":
                max_posts_per_day = parameters.get("max_posts_per_day", 10)
                min_hours_between = parameters.get("min_hours_between_posts", 1)
                
                for platform, schedule in schedule_matrix.items():
                    # Check daily post limits
                    daily_counts = {}
                    for task in schedule:
                        date_key = task["scheduled_time"].date()
                        daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
                    
                    for date, count in daily_counts.items():
                        if count > max_posts_per_day:
                            violations.append({
                                "type": "platform_limit_violation",
                                "platform": platform,
                                "date": str(date),
                                "violation": f"Exceeded daily limit: {count} > {max_posts_per_day}",
                                "severity": "critical"
                            })
                    
                    # Check minimum time between posts
                    sorted_schedule = sorted(schedule, key=lambda x: x["scheduled_time"])
                    for i in range(1, len(sorted_schedule)):
                        time_diff = (sorted_schedule[i]["scheduled_time"] - sorted_schedule[i-1]["scheduled_time"]).total_seconds() / 3600
                        if time_diff < min_hours_between:
                            violations.append({
                                "type": "timing_violation",
                                "platform": platform,
                                "task1": sorted_schedule[i-1]["task_id"],
                                "task2": sorted_schedule[i]["task_id"],
                                "violation": f"Posts too close: {time_diff:.1f}h < {min_hours_between}h",
                                "severity": "warning"
                            })
            
            elif constraint_type == "time_window":
                business_hours_only = parameters.get("business_hours_only", False)
                exclude_weekends = parameters.get("exclude_weekends", False)
                
                if business_hours_only:
                    for platform, schedule in schedule_matrix.items():
                        for task in schedule:
                            hour = task["scheduled_time"].hour
                            if hour < 9 or hour > 17:
                                violations.append({
                                    "type": "business_hours_violation",
                                    "platform": platform,
                                    "task_id": task["task_id"],
                                    "scheduled_time": task["scheduled_time"].isoformat(),
                                    "violation": "Scheduled outside business hours",
                                    "severity": "warning"
                                })
                
                if exclude_weekends:
                    for platform, schedule in schedule_matrix.items():
                        for task in schedule:
                            if task["scheduled_time"].weekday() >= 5:  # Saturday = 5, Sunday = 6
                                violations.append({
                                    "type": "weekend_violation",
                                    "platform": platform,
                                    "task_id": task["task_id"],
                                    "scheduled_time": task["scheduled_time"].isoformat(),
                                    "violation": "Scheduled on weekend",
                                    "severity": "warning"
                                })
        
        return violations
    
    def _calculate_optimization_score(self, schedule_matrix: Dict[str, List[Dict[str, Any]]], platform_analysis: Dict[str, Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score for the schedule."""
        base_score = 100.0
        
        # Penalty for constraint violations
        critical_violations = len([c for c in conflicts if c.get("severity") == "critical"])
        warning_violations = len([c for c in conflicts if c.get("severity") == "warning"])
        
        base_score -= (critical_violations * 15) + (warning_violations * 5)
        
        # Bonus for optimal time slot usage
        optimal_slots_used = 0
        total_slots = 0
        
        for platform, schedule in schedule_matrix.items():
            platform_info = platform_analysis.get(platform, {})
            optimal_times = platform_info.get("optimal_posting_times", [])
            
            for task in schedule:
                total_slots += 1
                task_time = task["scheduled_time"].strftime("%H:%M")
                if task_time in optimal_times:
                    optimal_slots_used += 1
        
        if total_slots > 0:
            optimal_ratio = optimal_slots_used / total_slots
            base_score += optimal_ratio * 20  # Up to 20 bonus points
        
        # Bonus for balanced distribution across platforms
        if len(schedule_matrix) > 1:
            task_counts = [len(schedule) for schedule in schedule_matrix.values()]
            avg_tasks = sum(task_counts) / len(task_counts)
            variance = sum((count - avg_tasks) ** 2 for count in task_counts) / len(task_counts)
            balance_score = max(0, 10 - variance)  # Lower variance = better balance
            base_score += balance_score
        
        return max(0, min(100, base_score))
    
    def _optimize_for_engagement(self, schedule: List[Dict[str, Any]], platform_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize schedule for maximum engagement."""
        optimal_times = platform_analysis.get("optimal_posting_times", [])
        
        # Sort tasks by priority and try to assign to optimal times
        sorted_schedule = sorted(schedule, key=lambda x: {
            "urgent": 4, "high": 3, "medium": 2, "low": 1
        }.get(x["priority"], 2), reverse=True)
        
        # Reassign times favoring optimal slots for high-priority tasks
        time_index = 0
        for i, task in enumerate(sorted_schedule):
            if time_index < len(optimal_times):
                # Try to use optimal times for high-priority tasks
                if task["priority"] in ["urgent", "high"]:
                    # Find next available optimal time slot
                    optimal_time = optimal_times[time_index % len(optimal_times)]
                    hour, minute = map(int, optimal_time.split(":"))
                    
                    base_date = task["scheduled_time"].date()
                    new_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                    
                    task["scheduled_time"] = new_time
                    task["slot_quality_score"] = task.get("slot_quality_score", 50) + 25  # Boost quality
                    time_index += 1
        
        return sorted_schedule
    
    def _optimize_for_balance(self, schedule: List[Dict[str, Any]], platform_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize schedule for balanced frequency."""
        if not schedule:
            return schedule
        
        # Distribute tasks evenly over available time period
        schedule.sort(key=lambda x: x["scheduled_time"])
        
        start_time = schedule[0]["scheduled_time"]
        end_time = schedule[-1]["scheduled_time"]
        total_duration = (end_time - start_time).total_seconds()
        
        if len(schedule) > 1:
            interval = total_duration / (len(schedule) - 1)
            
            for i, task in enumerate(schedule):
                new_time = start_time + timedelta(seconds=i * interval)
                task["scheduled_time"] = new_time
        
        return schedule
    
    def _optimize_for_speed(self, schedule: List[Dict[str, Any]], platform_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize schedule for fastest completion (rush mode)."""
        # Compress schedule to earliest possible times with minimum delays
        schedule.sort(key=lambda x: x["priority"] == "urgent", reverse=True)
        
        current_time = datetime.now() + timedelta(hours=1)  # Start soon
        min_interval = timedelta(hours=2)  # Minimum 2 hours between tasks
        
        for task in schedule:
            task["scheduled_time"] = current_time
            current_time += min_interval
        
        return schedule
    
    def _generate_additional_time_slots(self, platform: str, analysis: Dict[str, Any], time_window: Dict[str, datetime], start_date: datetime) -> List[Dict[str, Any]]:
        """Generate additional time slots when initial slots are exhausted."""
        slots = []
        optimal_times = analysis.get("optimal_posting_times", ["09:00", "14:00", "17:00"])
        
        current_date = start_date + timedelta(days=1)
        end_date = time_window["end"]
        
        days_needed = min(14, (end_date - current_date).days)  # Generate up to 14 more days
        
        for day in range(days_needed):
            date = current_date + timedelta(days=day)
            
            for time_str in optimal_times:
                hour, minute = map(int, time_str.split(":"))
                slot_datetime = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if slot_datetime < end_date:
                    slots.append({
                        "datetime": slot_datetime,
                        "time_str": time_str,
                        "day_of_week": date.strftime("%A"),
                        "quality_score": 60.0  # Default quality for additional slots
                    })
        
        return slots
    
    def _detect_time_conflicts(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect time conflicts between tasks on the same day."""
        conflicts = []
        
        # Group by hour
        tasks_by_hour = {}
        for task in tasks:
            hour_key = task["scheduled_at"].replace(minute=0, second=0, microsecond=0)
            if hour_key not in tasks_by_hour:
                tasks_by_hour[hour_key] = []
            tasks_by_hour[hour_key].append(task)
        
        # Check for conflicts (more than 2 tasks in same hour)
        for hour, hour_tasks in tasks_by_hour.items():
            if len(hour_tasks) > 2:
                conflicts.append({
                    "type": "time_conflict",
                    "hour": hour.isoformat(),
                    "task_count": len(hour_tasks),
                    "task_ids": [task["task_id"] for task in hour_tasks],
                    "severity": "warning"
                })
        
        return conflicts