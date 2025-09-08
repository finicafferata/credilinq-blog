#!/usr/bin/env python3
"""
Workflow Executor - Real agent execution engine for Master Planner workflows.

This module provides the actual execution engine that runs real agents according to 
execution plans created by the Master Planner Agent, with live status updates and 
result capture.
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

# Agent imports
from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentType
from ..core.agent_factory import AgentFactory
from ..core.database_service import DatabaseService
from .master_planner_agent import MasterPlannerAgent, ExecutionPlan

# Recovery system imports
try:
    from ...core.checkpoint_manager import CheckpointManager, CheckpointType
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CheckpointManager = None
    CheckpointType = None
    CHECKPOINT_MANAGER_AVAILABLE = False
    logger.warning("CheckpointManager not available - recovery features disabled")

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

@dataclass
class AgentExecutionResult:
    """Result of individual agent execution."""
    agent_name: str
    agent_type: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0

@dataclass
class WorkflowExecution:
    """Active workflow execution state."""
    workflow_id: str
    execution_plan: ExecutionPlan
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    current_agents: Set[str] = field(default_factory=set)
    completed_agents: Set[str] = field(default_factory=set)
    failed_agents: Set[str] = field(default_factory=set)
    agent_results: Dict[str, AgentExecutionResult] = field(default_factory=dict)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

class WorkflowExecutor:
    """
    Real workflow execution engine that runs agents according to Master Planner execution plans.
    Provides live status updates, result capture, and complete workflow orchestration.
    """
    
    def __init__(self, enable_recovery_systems: bool = True):
        self.agent_factory = AgentFactory()
        self.db_service = DatabaseService()
        self.master_planner = MasterPlannerAgent()
        
        # Recovery systems integration
        self.enable_recovery_systems = enable_recovery_systems and CHECKPOINT_MANAGER_AVAILABLE
        if self.enable_recovery_systems:
            self.checkpoint_manager = CheckpointManager()
            logger.info("✅ WorkflowExecutor: Recovery systems enabled with CheckpointManager")
        else:
            self.checkpoint_manager = None
            logger.info("⚠️ WorkflowExecutor: Recovery systems disabled")
        
        # Active workflow tracking
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.execution_callbacks: Dict[str, List[Callable]] = {}
        
        # Thread pool for parallel agent execution
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="WorkflowAgent")
        
        # Agent type mapping
        self.agent_type_map = {
            "planner": AgentType.PLANNER,
            "researcher": AgentType.RESEARCHER,
            "writer": AgentType.WRITER,
            "editor": AgentType.EDITOR,
            "seo": AgentType.SEO,
            "image": AgentType.IMAGE_PROMPT,  # Fix: Use correct agent type
            "social_media": AgentType.SOCIAL_MEDIA,
            "campaign_manager": AgentType.CAMPAIGN_MANAGER,
            "content_repurposer": AgentType.CONTENT_REPURPOSER,
            "distribution": AgentType.TASK_SCHEDULER  # Use task scheduler as fallback for distribution
        }
        
        logger.info("WorkflowExecutor initialized with agent factory and database service")
    
    async def execute_workflow(
        self, 
        execution_plan: ExecutionPlan,
        context_data: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Callable] = None
    ) -> WorkflowExecution:
        """
        Execute a complete workflow according to the execution plan.
        
        Args:
            execution_plan: Master Planner execution plan
            context_data: Additional context data for agents
            status_callback: Optional callback for status updates
            
        Returns:
            WorkflowExecution: Final execution state with all results
        """
        workflow_id = execution_plan.workflow_execution_id
        
        logger.info(f"Starting workflow execution {workflow_id} with {len(execution_plan.agent_sequence)} agents")
        
        # Create workflow execution state
        workflow_execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_plan=execution_plan,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.utcnow(),
            execution_metadata={
                "strategy": execution_plan.strategy,
                "total_agents": len(execution_plan.agent_sequence),
                "context_data_keys": list(context_data.keys()) if context_data else []
            }
        )
        
        self.active_workflows[workflow_id] = workflow_execution
        
        if status_callback:
            self.execution_callbacks[workflow_id] = [status_callback]
        
        try:
            # Update Master Planner with execution start
            await self.master_planner.update_agent_status(
                workflow_id, "workflow", "starting", 
                {"total_agents": len(execution_plan.agent_sequence)}
            )
            
            # Execute workflow based on strategy
            if execution_plan.strategy == "parallel":
                await self._execute_parallel_workflow(workflow_execution, context_data)
            elif execution_plan.strategy == "sequential":
                await self._execute_sequential_workflow(workflow_execution, context_data)
            else:  # adaptive
                await self._execute_adaptive_workflow(workflow_execution, context_data)
            
            # Finalize workflow
            workflow_execution.status = ExecutionStatus.COMPLETED
            workflow_execution.end_time = datetime.utcnow()
            
            logger.info(f"Workflow {workflow_id} completed successfully with {len(workflow_execution.completed_agents)} agents")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            workflow_execution.status = ExecutionStatus.FAILED
            workflow_execution.end_time = datetime.utcnow()
            workflow_execution.execution_metadata["error"] = str(e)
        
        finally:
            # Update Master Planner with final status
            await self.master_planner.update_agent_status(
                workflow_id, "workflow", 
                workflow_execution.status.value,
                {
                    "completed_agents": list(workflow_execution.completed_agents),
                    "failed_agents": list(workflow_execution.failed_agents),
                    "execution_time_seconds": (
                        workflow_execution.end_time - workflow_execution.start_time
                    ).total_seconds() if workflow_execution.end_time else None
                }
            )
            
            # Notify callbacks
            await self._notify_status_callbacks(workflow_id, "workflow_completed")
        
        return workflow_execution
    
    async def _execute_sequential_workflow(
        self, 
        workflow_execution: WorkflowExecution,
        context_data: Optional[Dict[str, Any]]
    ):
        """Execute agents sequentially according to dependency order."""
        plan = workflow_execution.execution_plan
        
        for step in plan.agent_sequence:
            agent_name = step["agent_name"]
            
            # Check dependencies are complete
            dependencies = step["dependencies"]
            if not all(dep in workflow_execution.completed_agents for dep in dependencies):
                missing_deps = [dep for dep in dependencies if dep not in workflow_execution.completed_agents]
                raise Exception(f"Agent {agent_name} cannot execute - missing dependencies: {missing_deps}")
            
            # Execute agent
            await self._execute_single_agent(workflow_execution, step, context_data)
    
    async def _execute_parallel_workflow(
        self, 
        workflow_execution: WorkflowExecution,
        context_data: Optional[Dict[str, Any]]
    ):
        """Execute agents in parallel groups according to parallel_groups."""
        plan = workflow_execution.execution_plan
        
        # Group agents by parallel group
        parallel_groups = {}
        sequential_agents = []
        
        for step in plan.agent_sequence:
            parallel_group_id = step.get("parallel_group_id")
            if parallel_group_id is not None:
                if parallel_group_id not in parallel_groups:
                    parallel_groups[parallel_group_id] = []
                parallel_groups[parallel_group_id].append(step)
            else:
                sequential_agents.append(step)
        
        # Execute parallel groups first, then sequential agents
        for group_id in sorted(parallel_groups.keys()):
            group_agents = parallel_groups[group_id]
            logger.info(f"Executing parallel group {group_id} with {len(group_agents)} agents")
            
            # Execute all agents in this group concurrently
            tasks = []
            for step in group_agents:
                task = asyncio.create_task(
                    self._execute_single_agent(workflow_execution, step, context_data)
                )
                tasks.append(task)
            
            # Wait for all agents in group to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Execute sequential agents
        for step in sequential_agents:
            await self._execute_single_agent(workflow_execution, step, context_data)
    
    async def _execute_adaptive_workflow(
        self, 
        workflow_execution: WorkflowExecution,
        context_data: Optional[Dict[str, Any]]
    ):
        """Execute workflow with adaptive scheduling based on dependencies and resources."""
        plan = workflow_execution.execution_plan
        
        # Track agents that can be executed (dependencies met)
        available_agents = []
        remaining_agents = list(plan.agent_sequence)
        
        while remaining_agents or workflow_execution.current_agents:
            # Find agents ready to execute
            newly_available = []
            for step in remaining_agents:
                dependencies = step["dependencies"]
                if all(dep in workflow_execution.completed_agents for dep in dependencies):
                    newly_available.append(step)
            
            # Remove newly available from remaining
            for step in newly_available:
                remaining_agents.remove(step)
                available_agents.append(step)
            
            # Execute available agents (up to max parallel)
            max_parallel = 3  # Could be configurable
            while available_agents and len(workflow_execution.current_agents) < max_parallel:
                step = available_agents.pop(0)
                
                # Start agent execution
                asyncio.create_task(
                    self._execute_single_agent(workflow_execution, step, context_data)
                )
            
            # Wait a bit before checking again
            await asyncio.sleep(1)
            
            # Check if we're stuck
            if not available_agents and not workflow_execution.current_agents and remaining_agents:
                # Find the issue
                unmet_deps = []
                for step in remaining_agents:
                    for dep in step["dependencies"]:
                        if dep not in workflow_execution.completed_agents and dep not in workflow_execution.current_agents:
                            unmet_deps.append(dep)
                
                raise Exception(f"Workflow stuck - unmet dependencies: {set(unmet_deps)}")
    
    async def _execute_single_agent(
        self,
        workflow_execution: WorkflowExecution,
        agent_step: Dict[str, Any],
        context_data: Optional[Dict[str, Any]]
    ):
        """Execute a single agent and capture results."""
        agent_name = agent_step["agent_name"]
        agent_type_str = agent_step["agent_type"]
        
        logger.info(f"Starting execution of {agent_name} ({agent_type_str})")
        
        # Track agent as currently running
        workflow_execution.current_agents.add(agent_name)
        
        # Create execution result tracker
        execution_result = AgentExecutionResult(
            agent_name=agent_name,
            agent_type=agent_type_str,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        workflow_execution.agent_results[agent_name] = execution_result
        
        # Update Master Planner
        await self.master_planner.update_agent_status(
            workflow_execution.workflow_id, agent_name, "starting"
        )
        
        # Notify callbacks
        await self._notify_status_callbacks(workflow_execution.workflow_id, f"agent_starting:{agent_name}")
        
        try:
            # Get agent type enum
            agent_type = self.agent_type_map.get(agent_name)
            if not agent_type:
                raise Exception(f"Unknown agent type for {agent_name}")
            
            # Create agent instance
            agent = self.agent_factory.create_agent(agent_type)
            
            # Prepare input data combining context and intermediate results
            input_data = self._prepare_agent_input(
                agent_name, workflow_execution, context_data, agent_step
            )
            
            # Create execution context
            execution_context = AgentExecutionContext(
                workflow_id=workflow_execution.workflow_id,
                execution_id=str(uuid.uuid4()),
                user_id="workflow_executor",
                session_id=workflow_execution.workflow_id,
                execution_metadata={
                    "agent_step": agent_step,
                    "dependencies_results": {
                        dep: workflow_execution.agent_results.get(dep, {}).output_data 
                        for dep in agent_step["dependencies"]
                        if dep in workflow_execution.agent_results
                    }
                }
            )
            
            # Execute agent in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_agent_sync,
                agent, input_data, execution_context
            )
            
            # Process result
            execution_result.end_time = datetime.utcnow()
            execution_result.execution_time_seconds = (
                execution_result.end_time - execution_result.start_time
            ).total_seconds()
            
            if result.success:
                execution_result.status = ExecutionStatus.COMPLETED
                execution_result.output_data = result.data
                
                # Store intermediate data for dependent agents
                workflow_execution.intermediate_data[agent_name] = result.data
                workflow_execution.completed_agents.add(agent_name)
                
                logger.info(f"Agent {agent_name} completed successfully")
                
                # Update Master Planner
                await self.master_planner.update_agent_status(
                    workflow_execution.workflow_id, agent_name, "completed",
                    result.data, None, execution_result.execution_time_seconds
                )
                
                # Create checkpoint after successful agent completion
                if self.enable_recovery_systems and self.checkpoint_manager:
                    try:
                        checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                            workflow_id=workflow_execution.workflow_id,
                            workflow_type="orchestrated_workflow",
                            state_data={
                                "completed_agents": list(workflow_execution.completed_agents),
                                "failed_agents": list(workflow_execution.failed_agents),
                                "intermediate_data": workflow_execution.intermediate_data,
                                "execution_metadata": workflow_execution.execution_metadata,
                                "current_phase": f"agent_{agent_name}_completed"
                            },
                            phase_name=f"agent_{agent_name}_completed",
                            checkpoint_type=CheckpointType.AUTOMATIC,
                            agent_name=agent_name,
                            description=f"Checkpoint after {agent_name} completion"
                        )
                        logger.info(f"📦 Created checkpoint {checkpoint_id} after {agent_name} completion")
                    except Exception as checkpoint_error:
                        logger.warning(f"Failed to create checkpoint after {agent_name}: {checkpoint_error}")
                
            else:
                execution_result.status = ExecutionStatus.FAILED
                execution_result.error_message = result.error_message
                workflow_execution.failed_agents.add(agent_name)
                
                logger.error(f"Agent {agent_name} failed: {result.error_message}")
                
                # Update Master Planner
                await self.master_planner.update_agent_status(
                    workflow_execution.workflow_id, agent_name, "failed",
                    None, result.error_message, execution_result.execution_time_seconds
                )
        
        except Exception as e:
            execution_result.end_time = datetime.utcnow()
            execution_result.status = ExecutionStatus.FAILED
            execution_result.error_message = str(e)
            workflow_execution.failed_agents.add(agent_name)
            
            logger.error(f"Agent {agent_name} execution failed with exception: {str(e)}")
            
            # Update Master Planner
            await self.master_planner.update_agent_status(
                workflow_execution.workflow_id, agent_name, "failed",
                None, str(e)
            )
        
        finally:
            # Remove from current agents
            workflow_execution.current_agents.discard(agent_name)
            
            # Notify callbacks
            await self._notify_status_callbacks(
                workflow_execution.workflow_id, 
                f"agent_completed:{agent_name}"
            )
    
    def _run_agent_sync(
        self, 
        agent: BaseAgent, 
        input_data: Dict[str, Any],
        context: AgentExecutionContext
    ) -> AgentResult:
        """Synchronously run agent in thread pool."""
        return agent.execute(input_data, context)
    
    def _prepare_agent_input(
        self,
        agent_name: str,
        workflow_execution: WorkflowExecution,
        context_data: Optional[Dict[str, Any]],
        agent_step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input data for agent based on context and dependencies."""
        input_data = {}
        
        # Add context data
        if context_data:
            input_data.update(context_data)
        
        # Add dependency results
        for dep_name in agent_step["dependencies"]:
            if dep_name in workflow_execution.intermediate_data:
                dep_data = workflow_execution.intermediate_data[dep_name]
                input_data[f"{dep_name}_output"] = dep_data
                
                # Also flatten certain common data
                if isinstance(dep_data, dict):
                    if "content" in dep_data:
                        input_data[f"{dep_name}_content"] = dep_data["content"]
                    if "research" in dep_data:
                        input_data["research"] = dep_data["research"]
                    if "outline" in dep_data:
                        input_data["outline"] = dep_data["outline"]
        
        # Add agent-specific configuration
        if "configuration" in agent_step:
            input_data.update(agent_step["configuration"])
        
        # Add workflow metadata
        input_data["workflow_id"] = workflow_execution.workflow_id
        input_data["agent_name"] = agent_name
        input_data["execution_step"] = len(workflow_execution.completed_agents) + 1
        
        return input_data
    
    async def _notify_status_callbacks(self, workflow_id: str, event: str):
        """Notify registered callbacks about workflow events."""
        if workflow_id in self.execution_callbacks:
            for callback in self.execution_callbacks[workflow_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(workflow_id, event, self.active_workflows.get(workflow_id))
                    else:
                        callback(workflow_id, event, self.active_workflows.get(workflow_id))
                except Exception as e:
                    logger.error(f"Callback error for workflow {workflow_id}: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow execution."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_execution = self.active_workflows[workflow_id]
        
        total_agents = len(workflow_execution.execution_plan.agent_sequence)
        completed_count = len(workflow_execution.completed_agents)
        failed_count = len(workflow_execution.failed_agents)
        
        progress = (completed_count + failed_count) / total_agents * 100 if total_agents > 0 else 0
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_execution.status.value,
            "progress_percentage": round(progress, 1),
            "total_agents": total_agents,
            "completed_agents": list(workflow_execution.completed_agents),
            "failed_agents": list(workflow_execution.failed_agents),
            "current_agents": list(workflow_execution.current_agents),
            "start_time": workflow_execution.start_time.isoformat(),
            "end_time": workflow_execution.end_time.isoformat() if workflow_execution.end_time else None,
            "agent_results": {
                name: {
                    "status": result.status.value,
                    "execution_time": result.execution_time_seconds,
                    "output_keys": list(result.output_data.keys()) if result.output_data else [],
                    "error": result.error_message
                }
                for name, result in workflow_execution.agent_results.items()
            }
        }
    
    def register_status_callback(self, workflow_id: str, callback: Callable):
        """Register a callback for workflow status updates."""
        if workflow_id not in self.execution_callbacks:
            self.execution_callbacks[workflow_id] = []
        self.execution_callbacks[workflow_id].append(callback)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow execution."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow_execution = self.active_workflows[workflow_id]
        workflow_execution.status = ExecutionStatus.CANCELLED
        workflow_execution.end_time = datetime.utcnow()
        
        logger.info(f"Workflow {workflow_id} cancelled")
        return True
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows from memory."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for workflow_id, workflow_execution in self.active_workflows.items():
            if (workflow_execution.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]
                and workflow_execution.end_time 
                and workflow_execution.end_time < cutoff_time):
                to_remove.append(workflow_id)
        
        for workflow_id in to_remove:
            del self.active_workflows[workflow_id]
            if workflow_id in self.execution_callbacks:
                del self.execution_callbacks[workflow_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old workflow executions")
    
    def get_agent_intermediate_results(
        self, 
        workflow_id: str, 
        agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get intermediate results from a specific agent in a workflow."""
        if workflow_id not in self.active_workflows:
            return None
            
        workflow_execution = self.active_workflows[workflow_id]
        return workflow_execution.intermediate_data.get(agent_name)
    
    # Recovery System Methods
    
    async def list_workflow_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a specific workflow."""
        if not self.enable_recovery_systems or not self.checkpoint_manager:
            return []
        
        try:
            # Use get_recovery_points instead of list_checkpoints
            recovery_points = await self.checkpoint_manager.get_recovery_points(workflow_id)
            return [
                {
                    "checkpoint_id": rp.checkpoint_id,
                    "phase_name": rp.phase_name,
                    "created_at": rp.created_at.isoformat(),
                    "description": rp.description,
                    "resume_instructions": rp.instructions,
                    "available_actions": rp.available_actions
                }
                for rp in recovery_points
            ]
        except Exception as e:
            logger.error(f"Failed to list checkpoints for workflow {workflow_id}: {e}")
            return []
    
    async def resume_workflow_from_checkpoint(
        self, 
        checkpoint_id: str,
        execution_plan: Optional[ExecutionPlan] = None
    ) -> Optional[str]:
        """Resume a workflow from a specific checkpoint."""
        if not self.enable_recovery_systems or not self.checkpoint_manager:
            logger.warning("Recovery systems not available for workflow resume")
            return None
        
        try:
            # Load checkpoint data
            checkpoint_data = await self.checkpoint_manager.load_checkpoint(checkpoint_id)
            if not checkpoint_data:
                logger.error(f"Checkpoint {checkpoint_id} not found")
                return None
            
            # Extract workflow state
            workflow_id = checkpoint_data["workflow_id"]
            state_data = checkpoint_data["state_data"]
            
            logger.info(f"🔄 Resuming workflow {workflow_id} from checkpoint {checkpoint_id}")
            
            # Create new workflow execution from checkpoint
            workflow_execution = WorkflowExecution(
                workflow_id=workflow_id,
                execution_plan=execution_plan or self._create_default_execution_plan(workflow_id),
                status=ExecutionStatus.RUNNING,
                start_time=datetime.utcnow(),
                completed_agents=set(state_data.get("completed_agents", [])),
                failed_agents=set(state_data.get("failed_agents", [])),
                intermediate_data=state_data.get("intermediate_data", {}),
                execution_metadata=state_data.get("execution_metadata", {})
            )
            
            self.active_workflows[workflow_id] = workflow_execution
            
            # Continue execution from where we left off
            if execution_plan:
                if execution_plan.strategy == "parallel":
                    await self._execute_parallel_workflow(workflow_execution, {})
                elif execution_plan.strategy == "sequential":
                    await self._execute_sequential_workflow(workflow_execution, {})
                else:  # adaptive
                    await self._execute_adaptive_workflow(workflow_execution, {})
            
            logger.info(f"✅ Workflow {workflow_id} resumed successfully from checkpoint")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to resume workflow from checkpoint {checkpoint_id}: {e}")
            return None
    
    async def create_manual_checkpoint(
        self,
        workflow_id: str,
        description: str = "Manual checkpoint"
    ) -> Optional[str]:
        """Create a manual checkpoint for the specified workflow."""
        if not self.enable_recovery_systems or not self.checkpoint_manager:
            return None
        
        if workflow_id not in self.active_workflows:
            logger.warning(f"Workflow {workflow_id} not found for manual checkpoint")
            return None
        
        workflow_execution = self.active_workflows[workflow_id]
        
        try:
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                workflow_id=workflow_id,
                workflow_type="orchestrated_workflow",
                state_data={
                    "completed_agents": list(workflow_execution.completed_agents),
                    "failed_agents": list(workflow_execution.failed_agents),
                    "intermediate_data": workflow_execution.intermediate_data,
                    "execution_metadata": workflow_execution.execution_metadata,
                    "current_phase": "manual_checkpoint"
                },
                phase_name="manual_checkpoint",
                checkpoint_type=CheckpointType.MANUAL,
                description=description,
                manual_trigger=True
            )
            logger.info(f"📦 Created manual checkpoint {checkpoint_id} for workflow {workflow_id}")
            return checkpoint_id
        except Exception as e:
            logger.error(f"Failed to create manual checkpoint: {e}")
            return None
    
    def _create_default_execution_plan(self, workflow_id: str) -> ExecutionPlan:
        """Create a default execution plan for resume operations."""
        # This is a simple fallback - in production you'd want to reconstruct the original plan
        return ExecutionPlan(
            id=f"resume_{workflow_id}",
            workflow_execution_id=workflow_id,
            strategy="adaptive",
            agent_sequence=[
                {"agent_name": "researcher", "agent_type": "researcher", "dependencies": []},
                {"agent_name": "writer", "agent_type": "writer", "dependencies": ["researcher"]},
                {"agent_name": "editor", "agent_type": "editor", "dependencies": ["writer"]}
            ],
            estimated_duration=30
        )
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)