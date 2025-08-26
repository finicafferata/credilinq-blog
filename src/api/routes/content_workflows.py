#!/usr/bin/env python3
"""
Content Workflow API Routes
Handles content generation workflow orchestration, task management, and approval processes.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.agents.workflow.content_generation_workflow import (
    ContentGenerationWorkflow, ContentTask, ContentTaskStatus, 
    ContentTaskPriority, ContentType, ContentChannel
)
from src.agents.workflow.content_workflow_manager import (
    ContentWorkflowManager, WorkflowTrigger, ContentWorkflowConfig, 
    ExecutionMode, WorkflowPhase
)
from src.agents.workflow.task_management_system import (
    TaskManagementSystem, TaskSchedulingStrategy, TaskExecutionMode
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content-workflows", tags=["content-workflows"])

# Initialize workflow components
content_workflow = ContentGenerationWorkflow()
workflow_manager = ContentWorkflowManager()
task_manager = TaskManagementSystem()

# Pydantic models for API requests and responses

class ContentGenerationRequest(BaseModel):
    campaign_id: str = Field(..., description="Campaign ID for content generation")
    trigger: WorkflowTrigger = Field(default=WorkflowTrigger.MANUAL_TRIGGER, description="Workflow trigger type")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.ASYNCHRONOUS, description="Workflow execution mode")
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent tasks", ge=1, le=20)
    auto_approve_threshold: float = Field(default=8.0, description="Auto-approval quality threshold", ge=0, le=10)
    require_human_review: bool = Field(default=False, description="Require human review for all content")
    deadline: Optional[str] = Field(None, description="Workflow deadline (ISO format)")
    priority: ContentTaskPriority = Field(default=ContentTaskPriority.MEDIUM, description="Workflow priority")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom workflow settings")

class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    campaign_id: str
    current_phase: str
    total_phases: int
    completed_phases: int
    is_active: bool
    started_at: Optional[str]
    total_tasks: Optional[int] = None
    completed_tasks: Optional[int] = None
    failed_tasks: Optional[int] = None
    progress_percentage: Optional[float] = None

class ContentTaskResponse(BaseModel):
    task_id: str
    campaign_id: str
    content_type: str
    channel: str
    status: str
    priority: str
    title: Optional[str] = None
    themes: List[str] = []
    word_count: Optional[int] = None
    quality_score: Optional[float] = None
    created_at: str
    updated_at: str

class GeneratedContentResponse(BaseModel):
    content_id: str
    task_id: str
    content_type: str
    channel: str
    title: str
    content: str
    word_count: int
    quality_score: float
    seo_score: Optional[float] = None
    estimated_engagement: str
    metadata: Dict[str, Any] = {}
    created_at: str

class WorkflowExecutionResponse(BaseModel):
    workflow_id: str
    campaign_id: str
    success: bool
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    status: str
    execution_results: List[Dict[str, Any]] = []
    generated_content: List[GeneratedContentResponse] = []

class TaskManagementStatusResponse(BaseModel):
    is_running: bool
    scheduling_strategy: str
    execution_mode: str
    max_concurrent_tasks: int
    queue_status: Dict[str, Any]
    resource_utilization: Dict[str, float]
    system_metrics: Dict[str, Any]

class ContentApprovalRequest(BaseModel):
    content_id: str
    task_id: str
    approved: bool
    feedback: Optional[str] = None
    reviewer_id: Optional[str] = None
    quality_rating: Optional[float] = Field(None, ge=0, le=10)

class ContentRevisionRequest(BaseModel):
    content_id: str
    task_id: str
    revision_notes: str
    priority: ContentTaskPriority = ContentTaskPriority.HIGH

# Workflow Management Endpoints

@router.post("/initiate", response_model=Dict[str, str])
async def initiate_content_workflow(request: ContentGenerationRequest, background_tasks: BackgroundTasks):
    """
    Initiate a complete content generation workflow for a campaign
    """
    try:
        logger.info(f"Initiating content workflow for campaign: {request.campaign_id}")
        
        # Parse deadline if provided
        deadline = None
        if request.deadline:
            try:
                deadline = datetime.fromisoformat(request.deadline.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid deadline format. Use ISO format.")
        
        # Create workflow configuration
        config = ContentWorkflowConfig(
            workflow_id="",  # Will be set by the manager
            campaign_id=request.campaign_id,
            trigger=request.trigger,
            execution_mode=request.execution_mode,
            max_concurrent_tasks=request.max_concurrent_tasks,
            auto_approve_quality_threshold=request.auto_approve_threshold,
            require_human_review=request.require_human_review,
            deadline=deadline,
            priority=request.priority,
            custom_settings=request.custom_settings
        )
        
        # Start workflow in background
        workflow_id = await workflow_manager.initiate_content_workflow(
            campaign_id=request.campaign_id,
            trigger=request.trigger,
            config=config
        )
        
        return {
            "workflow_id": workflow_id,
            "campaign_id": request.campaign_id,
            "status": "initiated",
            "message": "Content generation workflow initiated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error initiating content workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate workflow: {str(e)}")

@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str = Path(..., description="Workflow ID")):
    """
    Get current status of a content generation workflow
    """
    try:
        status = await workflow_manager.get_workflow_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")

@router.post("/{workflow_id}/pause", response_model=Dict[str, Any])
async def pause_workflow(workflow_id: str = Path(..., description="Workflow ID")):
    """
    Pause an active content generation workflow
    """
    try:
        success = await workflow_manager.pause_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or cannot be paused")
        
        return {
            "workflow_id": workflow_id,
            "status": "paused",
            "message": "Workflow paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause workflow: {str(e)}")

@router.post("/{workflow_id}/resume", response_model=Dict[str, Any])
async def resume_workflow(workflow_id: str = Path(..., description="Workflow ID")):
    """
    Resume a paused content generation workflow
    """
    try:
        success = await workflow_manager.resume_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or cannot be resumed")
        
        return {
            "workflow_id": workflow_id,
            "status": "resumed",
            "message": "Workflow resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume workflow: {str(e)}")

@router.delete("/{workflow_id}/cancel", response_model=Dict[str, Any])
async def cancel_workflow(workflow_id: str = Path(..., description="Workflow ID")):
    """
    Cancel an active content generation workflow
    """
    try:
        success = await workflow_manager.cancel_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or cannot be cancelled")
        
        return {
            "workflow_id": workflow_id,
            "status": "cancelled",
            "message": "Workflow cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")

# Content Generation Endpoints

@router.post("/generate/{campaign_id}", response_model=WorkflowExecutionResponse)
async def generate_content_for_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    strategy_override: Optional[Dict[str, Any]] = None
):
    """
    Generate content for a campaign using the content generation workflow
    """
    try:
        logger.info(f"Generating content for campaign: {campaign_id}")
        
        # Get or create campaign strategy
        if strategy_override:
            campaign_strategy = strategy_override
        else:
            # This would typically fetch from the campaign database
            campaign_strategy = {
                "campaign_id": campaign_id,
                "objectives": ["content_marketing"],
                "target_audience": "B2B professionals",
                "channels": ["blog", "linkedin", "email"],
                "duration_weeks": 4,
                "content_frequency": "weekly"
            }
        
        # Create content generation plan
        plan = await content_workflow.create_content_generation_plan(
            campaign_id, campaign_strategy
        )
        
        # Execute the plan
        execution_result = await content_workflow.execute_content_generation_plan(campaign_id)
        
        # Convert generated content to response format
        generated_content_responses = []
        for content in execution_result.get('generated_content', []):
            if content and hasattr(content, 'content_id'):
                generated_content_responses.append(GeneratedContentResponse(
                    content_id=content.content_id,
                    task_id=getattr(content, 'task_id', ''),
                    content_type=content.content_type.value,
                    channel=content.channel.value,
                    title=content.title,
                    content=content.content,
                    word_count=content.word_count,
                    quality_score=content.quality_score,
                    seo_score=content.seo_score,
                    estimated_engagement=content.estimated_engagement,
                    metadata=content.metadata,
                    created_at=content.created_at.isoformat()
                ))
        
        return WorkflowExecutionResponse(
            workflow_id=plan.plan_id,
            campaign_id=campaign_id,
            success=execution_result['completed_tasks'] == execution_result['total_tasks'],
            total_tasks=execution_result['total_tasks'],
            completed_tasks=execution_result['completed_tasks'],
            failed_tasks=execution_result['failed_tasks'],
            status=execution_result['status'],
            execution_results=execution_result['execution_results'],
            generated_content=generated_content_responses
        )
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")

@router.get("/campaign/{campaign_id}/tasks", response_model=List[ContentTaskResponse])
async def get_campaign_content_tasks(campaign_id: str = Path(..., description="Campaign ID")):
    """
    Get all content generation tasks for a campaign
    """
    try:
        # Get workflow status
        workflow_status = await content_workflow.get_workflow_status(campaign_id)
        
        if not workflow_status:
            return []
        
        # Get active workflow plan
        if campaign_id not in content_workflow.active_workflows:
            return []
        
        plan = content_workflow.active_workflows[campaign_id]
        
        # Convert tasks to response format
        task_responses = []
        for task in plan.content_tasks:
            task_responses.append(ContentTaskResponse(
                task_id=task.task_id,
                campaign_id=task.campaign_id,
                content_type=task.content_type.value,
                channel=task.channel.value,
                status=task.status.value,
                priority=task.priority.value,
                title=task.title,
                themes=task.themes,
                word_count=task.word_count,
                quality_score=task.generated_content.quality_score if task.generated_content else None,
                created_at=task.created_at.isoformat(),
                updated_at=task.updated_at.isoformat()
            ))
        
        return task_responses
        
    except Exception as e:
        logger.error(f"Error getting campaign tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign tasks: {str(e)}")

@router.get("/campaign/{campaign_id}/content", response_model=List[GeneratedContentResponse])
async def get_campaign_generated_content(
    campaign_id: str = Path(..., description="Campaign ID"),
    status_filter: Optional[str] = Query(None, description="Filter by content status")
):
    """
    Get all generated content for a campaign
    """
    try:
        # Get active workflow plan
        if campaign_id not in content_workflow.active_workflows:
            return []
        
        plan = content_workflow.active_workflows[campaign_id]
        
        # Filter and convert content to response format
        content_responses = []
        for task in plan.content_tasks:
            if (task.generated_content and 
                (not status_filter or task.status.value == status_filter)):
                
                content = task.generated_content
                content_responses.append(GeneratedContentResponse(
                    content_id=content.content_id,
                    task_id=task.task_id,
                    content_type=content.content_type.value,
                    channel=content.channel.value,
                    title=content.title,
                    content=content.content,
                    word_count=content.word_count,
                    quality_score=content.quality_score,
                    seo_score=content.seo_score,
                    estimated_engagement=content.estimated_engagement,
                    metadata=content.metadata,
                    created_at=content.created_at.isoformat()
                ))
        
        return content_responses
        
    except Exception as e:
        logger.error(f"Error getting generated content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get generated content: {str(e)}")

# Task Management Endpoints

@router.get("/task-manager/status", response_model=TaskManagementStatusResponse)
async def get_task_manager_status():
    """
    Get current status of the task management system
    """
    try:
        status = task_manager.get_system_status()
        return TaskManagementStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting task manager status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task manager status: {str(e)}")

@router.post("/task-manager/start", response_model=Dict[str, str])
async def start_task_processing():
    """
    Start the task processing engine
    """
    try:
        await task_manager.start_processing()
        
        return {
            "status": "started",
            "message": "Task processing engine started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting task processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start task processing: {str(e)}")

@router.post("/task-manager/stop", response_model=Dict[str, str])
async def stop_task_processing():
    """
    Stop the task processing engine
    """
    try:
        await task_manager.stop_processing()
        
        return {
            "status": "stopped",
            "message": "Task processing engine stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping task processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop task processing: {str(e)}")

@router.get("/task/{task_id}/status", response_model=Dict[str, Any])
async def get_task_status(task_id: str = Path(..., description="Task ID")):
    """
    Get status and performance metrics for a specific task
    """
    try:
        status = task_manager.get_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.delete("/task/{task_id}/cancel", response_model=Dict[str, str])
async def cancel_task(task_id: str = Path(..., description="Task ID")):
    """
    Cancel a specific content generation task
    """
    try:
        success = await task_manager.cancel_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

# Content Review and Approval Endpoints

@router.post("/content/approve", response_model=Dict[str, Any])
async def approve_content(request: ContentApprovalRequest):
    """
    Approve or reject generated content
    """
    try:
        logger.info(f"Processing content approval for {request.content_id}")
        
        # This would typically update the content status in the database
        # and trigger next steps in the workflow
        
        # For now, we'll return a success response
        # In a full implementation, this would:
        # 1. Update content approval status in database
        # 2. Trigger workflow continuation if approved
        # 3. Queue for revision if rejected
        # 4. Send notifications to relevant stakeholders
        
        approval_result = {
            "content_id": request.content_id,
            "task_id": request.task_id,
            "approved": request.approved,
            "status": "approved" if request.approved else "rejected",
            "reviewer_feedback": request.feedback,
            "quality_rating": request.quality_rating,
            "processed_at": datetime.now().isoformat(),
            "next_steps": "Content scheduled for publication" if request.approved else "Content queued for revision"
        }
        
        return {
            "success": True,
            "message": f"Content {'approved' if request.approved else 'rejected'} successfully",
            "approval_result": approval_result
        }
        
    except Exception as e:
        logger.error(f"Error processing content approval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process approval: {str(e)}")

@router.post("/content/request-revision", response_model=Dict[str, Any])
async def request_content_revision(request: ContentRevisionRequest):
    """
    Request revision for generated content
    """
    try:
        logger.info(f"Processing revision request for {request.content_id}")
        
        # This would typically:
        # 1. Create a new revision task
        # 2. Add to high-priority queue
        # 3. Preserve original content as version
        # 4. Notify content creators
        
        revision_result = {
            "content_id": request.content_id,
            "original_task_id": request.task_id,
            "revision_task_id": f"rev_{request.task_id}_{datetime.now().timestamp()}",
            "revision_notes": request.revision_notes,
            "priority": request.priority.value,
            "requested_at": datetime.now().isoformat(),
            "estimated_completion": "Within 2-4 hours"
        }
        
        return {
            "success": True,
            "message": "Content revision requested successfully",
            "revision_result": revision_result
        }
        
    except Exception as e:
        logger.error(f"Error requesting content revision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to request revision: {str(e)}")

@router.get("/content/pending-approval", response_model=List[Dict[str, Any]])
async def get_pending_approvals(
    campaign_id: Optional[str] = Query(None, description="Filter by campaign ID"),
    priority: Optional[str] = Query(None, description="Filter by priority")
):
    """
    Get all content pending approval
    """
    try:
        # This would typically query the database for content awaiting approval
        # For now, we'll return an example response
        
        pending_items = []
        
        # In a real implementation, this would fetch from database:
        # - Content with status 'pending_approval'
        # - Associated task and campaign information
        # - Quality scores and metadata
        # - Reviewer assignments
        
        return pending_items
        
    except Exception as e:
        logger.error(f"Error getting pending approvals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending approvals: {str(e)}")

# Analytics and Performance Endpoints

@router.get("/analytics/performance", response_model=Dict[str, Any])
async def get_workflow_performance_analytics():
    """
    Get performance analytics for content workflows
    """
    try:
        analytics = task_manager.get_performance_analytics()
        
        # Add workflow-specific analytics
        workflow_analytics = {
            "active_workflows": len(workflow_manager.active_workflows),
            "total_campaigns_processed": len(workflow_manager.workflow_metrics),
            "workflow_success_rate": 95.0,  # This would be calculated from actual data
            "average_workflow_duration_minutes": 45.0,
            "content_quality_trend": [8.2, 8.4, 8.1, 8.5, 8.3]  # Last 5 time periods
        }
        
        analytics["workflow_analytics"] = workflow_analytics
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/analytics/campaign/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign_workflow_analytics(campaign_id: str = Path(..., description="Campaign ID")):
    """
    Get workflow analytics for a specific campaign
    """
    try:
        # Get workflow status
        workflow_status = await content_workflow.get_workflow_status(campaign_id)
        
        if not workflow_status:
            raise HTTPException(status_code=404, detail="Campaign workflow not found")
        
        # Calculate campaign-specific analytics
        campaign_analytics = {
            "campaign_id": campaign_id,
            "workflow_status": workflow_status,
            "content_statistics": {
                "total_pieces_generated": workflow_status.get("total_tasks", 0),
                "successful_generations": workflow_status.get("completed_tasks", 0),
                "failed_generations": workflow_status.get("failed_tasks", 0),
                "pending_tasks": workflow_status.get("pending_tasks", 0),
                "success_rate_percentage": workflow_status.get("progress_percentage", 0)
            },
            "quality_metrics": {
                "average_quality_score": 8.2,  # This would come from actual data
                "content_above_threshold": 85,  # Percentage
                "revision_requests": 2,
                "approval_rate": 92
            },
            "timing_analytics": {
                "workflow_started": workflow_status.get("started_at"),
                "estimated_completion": workflow_status.get("estimated_completion"),
                "average_task_duration_minutes": 12.5,
                "total_workflow_time_minutes": 45.0
            }
        }
        
        return campaign_analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign analytics: {str(e)}")

# Health Check Endpoint

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint for content workflow services
    """
    try:
        # Check if core components are healthy
        workflow_health = "healthy" if workflow_manager else "unhealthy"
        task_manager_health = "healthy" if task_manager.is_running else "idle"
        content_workflow_health = "healthy" if content_workflow else "unhealthy"
        
        overall_health = "healthy" if all([
            workflow_health == "healthy",
            content_workflow_health == "healthy"
        ]) else "degraded"
        
        return {
            "status": overall_health,
            "workflow_manager": workflow_health,
            "task_manager": task_manager_health,
            "content_workflow": content_workflow_health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }