#!/usr/bin/env python3
"""
Workflow Orchestration API Routes
Handles Master Planner Agent workflow monitoring, execution planning, and real-time state tracking.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uuid

from src.agents.orchestration.master_planner_agent import MasterPlannerAgent
from src.agents.orchestration.workflow_executor import WorkflowExecutor
from src.agents.orchestration.advanced_orchestrator import AdvancedOrchestrator, RecoveryStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflow-orchestration", tags=["workflow-orchestration"])

# Initialize Master Planner Agent, Workflow Executor, and Advanced Orchestrator
master_planner = MasterPlannerAgent()
workflow_executor = WorkflowExecutor()
advanced_orchestrator = AdvancedOrchestrator()

# Pydantic models for API requests and responses

class CreateExecutionPlanRequest(BaseModel):
    campaign_id: str = Field(..., description="Campaign ID for execution plan")
    workflow_execution_id: Optional[str] = Field(None, description="Optional custom workflow ID")
    strategy: str = Field(default="adaptive", description="Execution strategy: sequential, parallel, or adaptive")
    required_agents: List[str] = Field(default_factory=lambda: ["planner", "researcher", "writer", "editor"], description="List of required agent names")
    max_parallel_agents: int = Field(default=3, description="Maximum agents to run in parallel", ge=1, le=10)
    estimated_duration_limit: Optional[int] = Field(None, description="Optional duration limit in minutes", gt=0)

class ExecutionPlanResponse(BaseModel):
    success: bool
    plan_id: str
    workflow_execution_id: str
    strategy: str
    total_agents: int
    estimated_duration_minutes: int
    agent_sequence: List[Dict[str, Any]]
    created_at: str
    message: str

class WorkflowStatusResponse(BaseModel):
    workflow_execution_id: str
    status: str
    progress_percentage: float
    current_step: int
    total_steps: int
    agents_status: Dict[str, List[str]]
    start_time: Optional[str]
    estimated_completion_time: Optional[str]
    actual_completion_time: Optional[str]
    execution_metadata: Dict[str, Any]
    intermediate_outputs: Dict[str, Any]
    last_heartbeat: str

class AgentExecutionUpdate(BaseModel):
    workflow_execution_id: str
    agent_name: str
    status: str  # starting, running, completed, failed
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None

class ExecuteWorkflowRequest(BaseModel):
    campaign_id: str = Field(..., description="Campaign ID to execute workflow for")
    execution_strategy: str = Field(default="adaptive", description="Execution strategy")
    required_agents: List[str] = Field(default_factory=lambda: ["planner", "researcher", "writer", "editor"], description="Required agents")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data for agents")
    blog_title: Optional[str] = Field(None, description="Blog title for content generation")
    company_context: Optional[str] = Field(None, description="Company context")
    content_type: str = Field(default="blog", description="Content type to generate")

class WorkflowExecutionResponse(BaseModel):
    success: bool
    workflow_execution_id: str
    execution_plan_id: str
    status: str
    total_agents: int
    message: str
    estimated_duration_minutes: int

class AdvancedExecuteWorkflowRequest(BaseModel):
    campaign_id: str = Field(..., description="Campaign ID to execute workflow for")
    execution_strategy: str = Field(default="adaptive", description="Execution strategy")
    required_agents: List[str] = Field(default_factory=lambda: ["planner", "researcher", "writer", "editor"], description="Required agents")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data for agents")
    blog_title: Optional[str] = Field(None, description="Blog title for content generation")
    company_context: Optional[str] = Field(None, description="Company context")
    content_type: str = Field(default="blog", description="Content type to generate")
    enable_resequencing: bool = Field(default=True, description="Enable dynamic re-sequencing")
    recovery_strategy: str = Field(default="replan_remaining", description="Failure recovery strategy")

class OrchestrationAnalyticsResponse(BaseModel):
    workflow_id: str
    orchestration_summary: Dict[str, Any]
    resequencing_decisions: List[Dict[str, Any]]
    critical_path_analysis: Dict[str, Any] 
    alternative_paths: List[List[str]]

# API Endpoints

@router.post("/execution-plans", response_model=ExecutionPlanResponse)
async def create_execution_plan(request: CreateExecutionPlanRequest):
    """
    Create a new execution plan using the Master Planner Agent.
    This endpoint orchestrates the agent execution sequence based on dependencies and strategy.
    """
    try:
        logger.info(f"Creating execution plan for campaign {request.campaign_id} with strategy {request.strategy}")
        
        # Generate workflow ID if not provided
        workflow_execution_id = request.workflow_execution_id or str(uuid.uuid4())
        
        # Create execution plan using Master Planner Agent
        execution_plan = await master_planner.create_execution_plan(
            campaign_id=request.campaign_id,
            workflow_execution_id=workflow_execution_id,
            strategy=request.strategy,
            required_agents=request.required_agents
        )
        
        logger.info(f"✅ Created execution plan {execution_plan.id} with {len(execution_plan.agent_sequence)} agents")
        
        return ExecutionPlanResponse(
            success=True,
            plan_id=execution_plan.id,
            workflow_execution_id=workflow_execution_id,
            strategy=execution_plan.strategy,
            total_agents=len(execution_plan.agent_sequence),
            estimated_duration_minutes=execution_plan.estimated_duration,
            agent_sequence=execution_plan.agent_sequence,
            created_at=datetime.utcnow().isoformat(),
            message=f"Execution plan created successfully with {len(execution_plan.agent_sequence)} agents"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create execution plan: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create execution plan: {str(e)}"
        )

@router.post("/workflows/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(request: ExecuteWorkflowRequest, background_tasks: BackgroundTasks):
    """
    Execute a complete content generation workflow with real agents.
    This creates an execution plan and immediately starts executing agents.
    """
    try:
        logger.info(f"Starting workflow execution for campaign {request.campaign_id}")
        
        # Generate workflow execution ID
        workflow_execution_id = str(uuid.uuid4())
        
        # Create execution plan
        execution_plan = await master_planner.create_execution_plan(
            campaign_id=request.campaign_id,
            workflow_execution_id=workflow_execution_id,
            strategy=request.execution_strategy,
            required_agents=request.required_agents
        )
        
        # Prepare context data for agents
        context_data = dict(request.context_data)
        if request.blog_title:
            context_data["blog_title"] = request.blog_title
        if request.company_context:
            context_data["company_context"] = request.company_context
        context_data["content_type"] = request.content_type
        context_data["campaign_id"] = request.campaign_id
        
        # Start workflow execution in background
        def status_callback(workflow_id: str, event: str, execution_state):
            logger.info(f"Workflow {workflow_id} event: {event}")
        
        background_tasks.add_task(
            workflow_executor.execute_workflow,
            execution_plan,
            context_data,
            status_callback
        )
        
        logger.info(f"✅ Started workflow execution {workflow_execution_id} with {len(execution_plan.agent_sequence)} agents")
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_execution_id=workflow_execution_id,
            execution_plan_id=execution_plan.id,
            status="running",
            total_agents=len(execution_plan.agent_sequence),
            estimated_duration_minutes=execution_plan.estimated_duration,
            message=f"Workflow execution started with {len(execution_plan.agent_sequence)} agents using {request.execution_strategy} strategy"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute workflow: {str(e)}"
        )

@router.post("/workflows/execute-advanced", response_model=WorkflowExecutionResponse)
async def execute_advanced_workflow(request: AdvancedExecuteWorkflowRequest, background_tasks: BackgroundTasks):
    """
    Execute a workflow with advanced orchestration capabilities including:
    - Dynamic re-sequencing based on agent completion and failures
    - Intelligent failure recovery strategies
    - Critical path analysis and optimization
    - Alternative execution path evaluation
    """
    try:
        logger.info(f"Starting advanced workflow execution for campaign {request.campaign_id}")
        
        # Generate workflow execution ID
        workflow_execution_id = str(uuid.uuid4())
        
        # Create execution plan using Master Planner
        execution_plan = await master_planner.create_execution_plan(
            campaign_id=request.campaign_id,
            workflow_execution_id=workflow_execution_id,
            strategy=request.execution_strategy,
            required_agents=request.required_agents
        )
        
        # Prepare context data
        context_data = dict(request.context_data)
        if request.blog_title:
            context_data["blog_title"] = request.blog_title
        if request.company_context:
            context_data["company_context"] = request.company_context
        context_data["content_type"] = request.content_type
        context_data["campaign_id"] = request.campaign_id
        
        # Parse recovery strategy
        recovery_strategy_map = {
            "skip_and_continue": RecoveryStrategy.SKIP_AND_CONTINUE,
            "retry_with_alternatives": RecoveryStrategy.RETRY_WITH_ALTERNATIVES,
            "replan_remaining": RecoveryStrategy.REPLAN_REMAINING,
            "abort_workflow": RecoveryStrategy.ABORT_WORKFLOW
        }
        recovery_strategy = recovery_strategy_map.get(
            request.recovery_strategy, 
            RecoveryStrategy.REPLAN_REMAINING
        )
        
        # Execute workflow with advanced orchestration
        background_tasks.add_task(
            advanced_orchestrator.execute_adaptive_workflow,
            execution_plan,
            context_data,
            request.enable_resequencing,
            recovery_strategy
        )
        
        logger.info(f"✅ Started advanced workflow execution {workflow_execution_id} with orchestration capabilities")
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_execution_id=workflow_execution_id,
            execution_plan_id=execution_plan.id,
            status="running",
            total_agents=len(execution_plan.agent_sequence),
            estimated_duration_minutes=execution_plan.estimated_duration,
            message=f"Advanced workflow execution started with dynamic orchestration: {len(execution_plan.agent_sequence)} agents, resequencing {'enabled' if request.enable_resequencing else 'disabled'}, recovery strategy: {request.recovery_strategy}"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to execute advanced workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute advanced workflow: {str(e)}"
        )

@router.get("/workflows/{workflow_execution_id}/orchestration-analytics", response_model=OrchestrationAnalyticsResponse)
async def get_orchestration_analytics(workflow_execution_id: str):
    """
    Get advanced orchestration analytics for a workflow including:
    - Resequencing decisions and reasoning
    - Critical path analysis
    - Performance adjustments
    - Alternative execution paths
    """
    try:
        logger.info(f"Retrieving orchestration analytics for workflow {workflow_execution_id}")
        
        analytics = advanced_orchestrator.get_orchestration_analytics(workflow_execution_id)
        
        if not analytics:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration analytics not found for workflow {workflow_execution_id}"
            )
        
        return OrchestrationAnalyticsResponse(
            workflow_id=workflow_execution_id,
            orchestration_summary=analytics["orchestration_summary"],
            resequencing_decisions=analytics["resequencing_decisions"],
            critical_path_analysis=analytics["critical_path_analysis"],
            alternative_paths=analytics["alternative_paths"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get orchestration analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get orchestration analytics: {str(e)}"
        )

@router.get("/workflows/{workflow_execution_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_execution_id: str):
    """
    Get real-time status of a workflow execution including agent progress and intermediate outputs.
    """
    try:
        logger.info(f"Retrieving workflow status for {workflow_execution_id}")
        
        # Try to get status from Workflow Executor first (live executions)
        executor_status = workflow_executor.get_workflow_status(workflow_execution_id)
        
        if executor_status:
            # Use live execution data
            status_data = executor_status
        else:
            # Fall back to Master Planner stored data
            status_data = await master_planner.get_workflow_status(workflow_execution_id)
        
        if not status_data:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_execution_id} not found"
            )
        
        return WorkflowStatusResponse(
            workflow_execution_id=workflow_execution_id,
            status=status_data.get("status", "unknown"),
            progress_percentage=status_data.get("progress_percentage", 0),
            current_step=status_data.get("current_step", 0),
            total_steps=status_data.get("total_steps", 0),
            agents_status=status_data.get("agents_status", {
                "waiting": [],
                "running": [],
                "completed": [],
                "failed": []
            }),
            start_time=status_data.get("start_time"),
            estimated_completion_time=status_data.get("estimated_completion_time"),
            actual_completion_time=status_data.get("actual_completion_time"),
            execution_metadata=status_data.get("execution_metadata", {}),
            intermediate_outputs=status_data.get("intermediate_outputs", {}),
            last_heartbeat=status_data.get("last_heartbeat", datetime.utcnow().isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get workflow status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )

@router.post("/workflows/{workflow_execution_id}/agents/{agent_name}/update")
async def update_agent_execution(
    workflow_execution_id: str, 
    agent_name: str,
    update: AgentExecutionUpdate
):
    """
    Update the execution status of a specific agent within a workflow.
    Used by agents to report their status and output data.
    """
    try:
        logger.info(f"Updating agent {agent_name} status to {update.status} for workflow {workflow_execution_id}")
        
        # Update agent status using Master Planner Agent
        result = await master_planner.update_agent_status(
            workflow_execution_id=workflow_execution_id,
            agent_name=agent_name,
            status=update.status,
            output_data=update.output_data,
            error_message=update.error_message,
            execution_time_seconds=update.execution_time_seconds
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_execution_id} or agent {agent_name} not found"
            )
        
        return {
            "success": True,
            "workflow_execution_id": workflow_execution_id,
            "agent_name": agent_name,
            "status": update.status,
            "updated_at": datetime.utcnow().isoformat(),
            "message": f"Agent {agent_name} status updated to {update.status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update agent status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update agent status: {str(e)}"
        )

@router.get("/workflows/active", response_model=List[Dict[str, Any]])
async def get_active_workflows():
    """
    Get all currently active workflows with their status and progress.
    """
    try:
        logger.info("Retrieving all active workflows")
        
        # Get active workflows from Master Planner Agent
        active_workflows = await master_planner.get_active_workflows()
        
        return [
            {
                "workflow_execution_id": wf["workflow_execution_id"],
                "campaign_id": wf.get("campaign_id"),
                "status": wf.get("status", "unknown"),
                "progress_percentage": wf.get("progress_percentage", 0),
                "total_agents": wf.get("total_agents", 0),
                "completed_agents": len(wf.get("completed_agents", [])),
                "start_time": wf.get("start_time"),
                "estimated_completion_time": wf.get("estimated_completion_time"),
                "last_heartbeat": wf.get("last_heartbeat")
            }
            for wf in active_workflows
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get active workflows: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active workflows: {str(e)}"
        )

@router.get("/execution-plans/{plan_id}")
async def get_execution_plan(plan_id: str):
    """
    Get detailed information about a specific execution plan.
    """
    try:
        logger.info(f"Retrieving execution plan {plan_id}")
        
        # Get plan details from Master Planner Agent
        plan_data = await master_planner.get_execution_plan(plan_id)
        
        if not plan_data:
            raise HTTPException(
                status_code=404,
                detail=f"Execution plan {plan_id} not found"
            )
        
        return {
            "success": True,
            "plan": plan_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get execution plan: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution plan: {str(e)}"
        )

@router.get("/workflows/{workflow_execution_id}/agents/{agent_name}/results")
async def get_agent_intermediate_results(workflow_execution_id: str, agent_name: str):
    """
    Get intermediate results from a specific agent in an active workflow.
    """
    try:
        logger.info(f"Retrieving agent {agent_name} results for workflow {workflow_execution_id}")
        
        # Get intermediate results from workflow executor
        results = workflow_executor.get_agent_intermediate_results(workflow_execution_id, agent_name)
        
        if results is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_name} results not found for workflow {workflow_execution_id}"
            )
        
        return {
            "success": True,
            "workflow_execution_id": workflow_execution_id,
            "agent_name": agent_name,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get agent results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent results: {str(e)}"
        )

@router.post("/workflows/{workflow_execution_id}/cancel")
async def cancel_workflow_execution(workflow_execution_id: str):
    """
    Cancel an active workflow execution.
    """
    try:
        logger.info(f"Cancelling workflow execution {workflow_execution_id}")
        
        success = workflow_executor.cancel_workflow(workflow_execution_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_execution_id} not found or not active"
            )
        
        return {
            "success": True,
            "workflow_execution_id": workflow_execution_id,
            "message": "Workflow execution cancelled",
            "cancelled_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to cancel workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel workflow: {str(e)}"
        )

@router.websocket("/workflows/{workflow_execution_id}/stream")
async def workflow_status_stream(websocket: WebSocket, workflow_execution_id: str):
    """
    WebSocket endpoint for real-time workflow status streaming.
    Provides live updates on agent progress, outputs, and status changes.
    """
    await websocket.accept()
    
    try:
        logger.info(f"Starting real-time stream for workflow {workflow_execution_id}")
        
        last_status = None
        while True:
            try:
                # Get current workflow status from executor (real-time) or master planner (historical)
                executor_status = workflow_executor.get_workflow_status(workflow_execution_id)
                
                if executor_status:
                    status_data = executor_status
                else:
                    status_data = await master_planner.get_workflow_status(workflow_execution_id)
                
                if status_data:
                    # Only send update if status changed
                    if status_data != last_status:
                        await websocket.send_json({
                            "type": "status_update",
                            "workflow_execution_id": workflow_execution_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": status_data
                        })
                        last_status = status_data.copy() if isinstance(status_data, dict) else status_data
                    
                    # If workflow is completed or failed, send final message and close
                    if status_data.get("status") in ["completed", "failed", "cancelled"]:
                        await websocket.send_json({
                            "type": "workflow_finished",
                            "workflow_execution_id": workflow_execution_id,
                            "final_status": status_data.get("status"),
                            "timestamp": datetime.utcnow().isoformat(),
                            "final_results": {
                                "completed_agents": status_data.get("completed_agents", []),
                                "failed_agents": status_data.get("failed_agents", []),
                                "agent_results": status_data.get("agent_results", {})
                            }
                        })
                        break
                else:
                    # Workflow not found, might be completed and cleaned up
                    await websocket.send_json({
                        "type": "workflow_not_found",
                        "workflow_execution_id": workflow_execution_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": "Workflow not found - may have been completed and cleaned up"
                    })
                    break
                
                # Wait before next update
                await asyncio.sleep(1)  # Update every 1 second for real-time feel
                
            except Exception as e:
                logger.error(f"Error in workflow stream: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Stream error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket error for workflow {workflow_execution_id}: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@router.get("/agents/knowledge-base")
async def get_agent_knowledge_base():
    """
    Get the current agent knowledge base with dependencies and capabilities.
    """
    try:
        knowledge_base = master_planner.agent_knowledge_base
        
        return {
            "success": True,
            "total_agents": len(knowledge_base),
            "agents": {
                name: {
                    "type": info.get("type"),
                    "dependencies": info.get("dependencies", []),
                    "execution_time_estimate": info.get("execution_time_estimate"),
                    "priority": info.get("priority"),
                    "description": info.get("description", f"{name.title()} agent for content pipeline")
                }
                for name, info in knowledge_base.items()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get agent knowledge base: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent knowledge base: {str(e)}"
        )

# Recovery System Endpoints

@router.get("/workflows/{workflow_execution_id}/checkpoints", response_model=List[Dict[str, Any]])
async def list_workflow_checkpoints(workflow_execution_id: str):
    """List all checkpoints for a specific workflow."""
    try:
        logger.info(f"Listing checkpoints for workflow: {workflow_execution_id}")
        
        checkpoints = await workflow_executor.list_workflow_checkpoints(workflow_execution_id)
        
        return {
            "success": True,
            "workflow_execution_id": workflow_execution_id,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list workflow checkpoints: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflow checkpoints: {str(e)}"
        )

@router.post("/workflows/{workflow_execution_id}/checkpoints/manual")
async def create_manual_checkpoint(
    workflow_execution_id: str,
    description: str = Query("Manual checkpoint", description="Description for the checkpoint")
):
    """Create a manual checkpoint for a running workflow."""
    try:
        logger.info(f"Creating manual checkpoint for workflow: {workflow_execution_id}")
        
        checkpoint_id = await workflow_executor.create_manual_checkpoint(
            workflow_execution_id, description
        )
        
        if not checkpoint_id:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_execution_id} not found or checkpointing failed"
            )
        
        return {
            "success": True,
            "workflow_execution_id": workflow_execution_id,
            "checkpoint_id": checkpoint_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "message": f"Manual checkpoint created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create manual checkpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create manual checkpoint: {str(e)}"
        )

@router.post("/workflows/resume/{checkpoint_id}")
async def resume_workflow_from_checkpoint(checkpoint_id: str):
    """Resume a workflow from a specific checkpoint."""
    try:
        logger.info(f"Resuming workflow from checkpoint: {checkpoint_id}")
        
        resumed_workflow_id = await workflow_executor.resume_workflow_from_checkpoint(checkpoint_id)
        
        if not resumed_workflow_id:
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {checkpoint_id} not found or resume failed"
            )
        
        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "resumed_workflow_id": resumed_workflow_id,
            "resumed_at": datetime.utcnow().isoformat(),
            "message": f"Workflow resumed successfully from checkpoint"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to resume workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume workflow: {str(e)}"
        )

@router.get("/system/health-status")
async def get_system_health_status():
    """Get comprehensive system health status including degradation information."""
    try:
        logger.info("Retrieving system health status")
        
        # Get health from advanced orchestrator
        health_status = await advanced_orchestrator.get_system_health_status()
        
        # Add workflow executor information
        health_status.update({
            "active_workflows": len(workflow_executor.active_workflows),
            "recovery_systems_enabled": workflow_executor.enable_recovery_systems,
            "workflow_executor_status": "healthy"
        })
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Failed to get system health status: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "recovery_systems_enabled": False,
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/system/degradation-status")  
async def get_degradation_status():
    """Get current system degradation status and recommendations."""
    try:
        logger.info("Retrieving system degradation status")
        
        # Get degradation status from advanced orchestrator
        health_status = await advanced_orchestrator.get_system_health_status()
        
        return {
            "success": True,
            "degradation_level": health_status.get("degradation_level", "unknown"),
            "status": health_status.get("status", "unknown"),
            "service_health": health_status.get("service_health", {}),
            "recommendations": {
                "continue_workflows": health_status.get("status") in ["healthy", "degraded"],
                "create_checkpoint": health_status.get("degradation_level") in ["MODERATE", "SEVERE"],
                "limit_parallel_execution": health_status.get("degradation_level") == "SEVERE"
            },
            "timestamp": health_status.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get degradation status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get degradation status: {str(e)}"
        )

# Health check endpoint
@router.get("/health")
async def orchestration_health():
    """Health check for workflow orchestration service."""
    try:
        # Test Master Planner Agent initialization
        agent_status = "healthy" if master_planner else "unhealthy"
        kb_size = len(master_planner.agent_knowledge_base) if master_planner else 0
        
        return {
            "status": "healthy",
            "master_planner_agent": agent_status,
            "knowledge_base_size": kb_size,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }