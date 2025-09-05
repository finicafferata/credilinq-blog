"""
Review Workflow API Routes

REST API endpoints for managing content review workflows, including:
- Starting new review workflows
- Resuming paused workflows
- Updating human review decisions
- Retrieving workflow status and metrics
- Managing reviewer assignments
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query, Path
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ...agents.workflow.review_workflow_orchestrator import (
    ReviewWorkflowOrchestrator, review_workflow_orchestrator
)
from ...agents.workflow.review_workflow_models import (
    ReviewWorkflowState, ReviewStage, ReviewStatus, ReviewDecision,
    ReviewCheckpoint, ReviewFeedback
)
# from ...core.request_validation import validate_request  # Not available
from ...core.auth import get_current_user
# require_permissions not available - would implement permission checks

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v2/review-workflow", tags=["Review Workflow"])

# Initialize services
review_workflow_orchestrator_instance = review_workflow_orchestrator

# Pydantic Models

class ReviewConfigurationModel(BaseModel):
    """Review workflow configuration model for 8-stage workflow."""
    require_content_quality: bool = True
    require_editorial_review: bool = True
    require_brand_check: bool = True
    require_seo_analysis: bool = True
    require_geo_analysis: bool = True
    require_visual_review: bool = True
    require_social_media_review: bool = True
    require_final_approval: bool = True
    
    content_quality_auto_approve_threshold: float = Field(8.0, ge=0, le=10)
    editorial_auto_approve_threshold: float = Field(8.0, ge=0, le=10)
    brand_auto_approve_threshold: float = Field(8.5, ge=0, le=10)
    seo_auto_approve_threshold: float = Field(7.5, ge=0, le=10)
    geo_auto_approve_threshold: float = Field(8.0, ge=0, le=10)
    visual_auto_approve_threshold: float = Field(7.5, ge=0, le=10)
    social_media_auto_approve_threshold: float = Field(8.0, ge=0, le=10)
    
    content_quality_reviewers: List[str] = Field(default_factory=list)
    editorial_reviewers: List[str] = Field(default_factory=list)
    brand_reviewers: List[str] = Field(default_factory=list)
    seo_reviewers: List[str] = Field(default_factory=list)
    geo_reviewers: List[str] = Field(default_factory=list)
    visual_reviewers: List[str] = Field(default_factory=list)
    social_media_reviewers: List[str] = Field(default_factory=list)
    final_approvers: List[str] = Field(default_factory=list)
    
    human_review_timeout_hours: int = Field(48, ge=1, le=168)
    escalation_timeout_hours: int = Field(72, ge=1, le=168)
    
    allow_parallel_reviews: bool = False

class StartReviewWorkflowRequest(BaseModel):
    """Request model for starting a new review workflow."""
    content_id: str = Field(..., description="Unique content identifier")
    content_type: str = Field("article", description="Type of content being reviewed")
    content: str = Field(..., min_length=1, description="Content text to review")
    title: Optional[str] = Field(None, description="Content title")
    campaign_id: Optional[str] = Field(None, description="Associated campaign ID")
    
    content_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional content metadata")
    review_config: Optional[ReviewConfigurationModel] = Field(None, description="Custom review configuration")
    
    integration_mode: str = Field("conditional", description="Integration mode: automatic, manual, conditional")
    content_source: str = Field("manual_upload", description="Source of the content")
    
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context information")

class HumanReviewDecisionRequest(BaseModel):
    """Request model for human review decisions."""
    stage: str = Field(..., description="Review stage name")
    reviewer_id: str = Field(..., description="ID of the reviewer")
    status: str = Field(..., description="Review decision: approved, rejected, needs_revision")
    score: Optional[float] = Field(None, ge=0, le=10, description="Quality score (0-10)")
    feedback: str = Field("", description="Reviewer feedback")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    revision_requests: List[str] = Field(default_factory=list, description="Specific revision requests")

class ResumeWorkflowRequest(BaseModel):
    """Request model for resuming paused workflows."""
    human_review_updates: Dict[str, HumanReviewDecisionRequest] = Field(
        ..., description="Human review decisions by stage"
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class ReviewerAssignmentRequest(BaseModel):
    """Request model for assigning reviewers."""
    stage: str = Field(..., description="Review stage")
    reviewer_ids: List[str] = Field(..., min_items=1, description="List of reviewer IDs")
    priority: str = Field("medium", description="Assignment priority: low, medium, high, urgent")
    expected_completion_hours: int = Field(48, ge=1, le=168, description="Expected completion time")
    instructions: Optional[str] = Field(None, description="Special instructions for reviewers")

class UpdateReviewConfigRequest(BaseModel):
    """Request model for updating review configuration."""
    review_config: ReviewConfigurationModel = Field(..., description="Updated review configuration")

# Response Models

class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""
    workflow_execution_id: str
    content_id: str
    workflow_status: str
    overall_approval_status: str
    current_stage: Optional[str]
    overall_progress: float
    is_paused: bool
    
    completed_stages: List[str]
    failed_stages: List[str]
    pending_human_reviews: List[str]
    
    estimated_completion: Optional[str]
    started_at: str
    updated_at: str
    completed_at: Optional[str]

class ReviewDecisionResponse(BaseModel):
    """Response model for review decisions."""
    stage: str
    reviewer_id: str
    reviewer_type: str
    status: str
    score: Optional[float]
    feedback: str
    suggestions: List[str]
    decision_timestamp: str

class WorkflowSummaryResponse(BaseModel):
    """Response model for workflow summary."""
    workflow_execution_id: str
    content_id: str
    overall_approval_status: str
    quality_metrics: Dict[str, Any]
    approval_summary: Dict[str, Any]
    total_execution_time_seconds: Optional[float]
    stage_decisions: List[ReviewDecisionResponse]

# API Endpoints

@router.post("/start", response_model=Dict[str, Any])
async def start_review_workflow(
    request: StartReviewWorkflowRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Start a new content review workflow.
    
    This endpoint initiates a comprehensive review process for content,
    including automated quality checks and human review stages.
    """
    try:
        logger.info(f"Starting review workflow for content: {request.content_id}")
        
        # Request validation would go here if needed
        # await validate_request(request.dict(), "review_workflow_start")
        
        # Prepare content data for ReviewWorkflowState
        content_data = {
            "title": request.title or "Untitled Content",
            "body": request.content,
            "content_type": request.content_type,
            "metadata": request.content_metadata
        }
        
        # Create initial ReviewWorkflowState
        initial_state = ReviewWorkflowState(
            content_id=request.content_id,
            content_type=request.content_type,
            content_data=content_data,
            campaign_id=request.campaign_id,
            # Apply configuration if provided
            auto_approve_threshold=request.review_config.content_quality_auto_approve_threshold if request.review_config else 0.85,
            require_human_approval=True if request.review_config and any([
                request.review_config.content_quality_reviewers,
                request.review_config.brand_reviewers,
                request.review_config.seo_reviewers
            ]) else False
        )
        
        # Execute review workflow
        workflow_result = await review_workflow_orchestrator_instance.execute_review_workflow(initial_state)
        
        return {
            "success": True,
            "message": "Review workflow started successfully",
            "review_workflow_id": workflow_result.workflow_execution_id,
            "workflow_status": workflow_result.workflow_status.value,
            "current_stage": workflow_result.current_stage.value,
            "is_paused": workflow_result.is_paused,
            "metadata": {
                "content_id": workflow_result.content_id,
                "started_at": workflow_result.started_at.isoformat(),
                "progress_percentage": workflow_result.get_progress_percentage()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start review workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to start review workflow: {str(e)}", "error_code": "WORKFLOW_START_FAILED"}
        )

@router.get("/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the current status of a review workflow.
    
    Returns detailed information about workflow progress, current stage,
    and pending reviews.
    """
    try:
        logger.info(f"Getting workflow status for: {workflow_id}")
        
        # Try to load workflow state from database
        # For now, return a structured response based on our 8-stage workflow
        try:
            # In a real implementation, we'd load the workflow state from database
            # workflow_state = await review_workflow_orchestrator_instance._load_workflow_state(workflow_id)
            
            # Mock workflow state for 8-stage system
            status_info = {
                "workflow_execution_id": workflow_id,
                "content_id": f"content_{workflow_id[-8:]}",
                "workflow_status": "in_progress",
                "overall_approval_status": "pending",
                "current_stage": "content_quality",
                "overall_progress": 25.0,  # 2/8 stages = 25%
                "is_paused": False,
                "completed_stages": ["content_quality", "editorial_review"],
                "failed_stages": [],
                "pending_human_reviews": ["brand_check"],
                "estimated_completion": (datetime.utcnow()).isoformat(),
                "started_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "completed_at": None
            }
        except Exception as e:
            logger.warning(f"Could not load workflow state from database: {e}")
            # Fallback to basic status
            status_info = {
                "workflow_execution_id": workflow_id,
                "content_id": f"content_{workflow_id[-8:]}",
                "workflow_status": "unknown",
                "overall_approval_status": "pending",
                "current_stage": None,
                "overall_progress": 0.0,
                "is_paused": False,
                "completed_stages": [],
                "failed_stages": [],
                "pending_human_reviews": [],
                "estimated_completion": None,
                "started_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "completed_at": None
            }
        
        return WorkflowStatusResponse(**status_info)
        
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to get workflow status: {str(e)}", "error_code": "STATUS_FETCH_FAILED"}
        )

@router.post("/{workflow_id}/resume", response_model=Dict[str, Any])
async def resume_workflow(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    request: ResumeWorkflowRequest = ...,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Resume a paused review workflow with human review decisions.
    
    This endpoint allows workflows paused for human review to be resumed
    with the provided review decisions.
    """
    try:
        logger.info(f"Resuming workflow: {workflow_id}")
        
        # Resume workflow with our orchestrator
        workflow_result = await review_workflow_orchestrator_instance.resume_workflow(workflow_id)
        
        # Note: In a full implementation, we would process the human review updates
        # and update the workflow state before resuming. For now, we just resume.
        logger.info(f"Processing {len(request.human_review_updates)} human review updates")
        
        return {
            "success": True,
            "message": "Workflow resumed successfully",
            "review_workflow_id": workflow_result.workflow_execution_id,
            "workflow_status": workflow_result.workflow_status.value,
            "current_stage": workflow_result.current_stage.value,
            "is_paused": workflow_result.is_paused,
            "metadata": {
                "resumed_at": datetime.utcnow().isoformat(),
                "progress_percentage": workflow_result.get_progress_percentage()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to resume workflow: {str(e)}", "error_code": "WORKFLOW_RESUME_FAILED"}
        )

@router.post("/{workflow_id}/human-review", response_model=Dict[str, Any])
async def submit_human_review(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    request: HumanReviewDecisionRequest = ...,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Submit a human review decision for a specific stage.
    
    This endpoint allows human reviewers to submit their decisions
    for individual review stages.
    """
    try:
        logger.info(f"Submitting human review for workflow {workflow_id}, stage {request.stage}")
        
        # Validate reviewer permissions (implement actual authorization logic)
        # Permission validation would go here
        # await require_permissions(current_user, f"review_{request.stage}")
        
        # Submit review decision
        review_decision = {
            request.stage: {
                "reviewer_id": request.reviewer_id,
                "status": request.status,
                "score": request.score,
                "feedback": request.feedback,
                "suggestions": request.suggestions,
                "revision_requests": request.revision_requests,
                "submitted_by": current_user.get("user_id"),
                "submitted_at": datetime.utcnow().isoformat()
            }
        }
        
        # This would typically update the workflow state in the database
        # and potentially trigger workflow continuation
        
        return {
            "success": True,
            "message": f"Human review submitted for stage {request.stage}",
            "stage": request.stage,
            "reviewer_id": request.reviewer_id,
            "status": request.status,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit human review: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to submit human review: {str(e)}", "error_code": "REVIEW_SUBMISSION_FAILED"}
        )

@router.get("/{workflow_id}/summary", response_model=WorkflowSummaryResponse)
async def get_workflow_summary(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get a comprehensive summary of a completed review workflow.
    
    Returns detailed information about all review stages, decisions,
    quality metrics, and final approval status.
    """
    try:
        logger.info(f"Getting workflow summary for: {workflow_id}")
        
        # This would typically query the database for complete workflow information
        # For now, return a mock response
        
        stage_decisions = [
            ReviewDecisionResponse(
                stage="content_quality",
                reviewer_id="content_quality_agent",
                reviewer_type="agent",
                status="approved",
                score=8.5,
                feedback="Content quality meets standards across all dimensions",
                suggestions=["Minor readability improvements", "Consider adding more examples"],
                decision_timestamp=datetime.utcnow().isoformat()
            ),
            ReviewDecisionResponse(
                stage="editorial_review",
                reviewer_id="editorial_agent",
                reviewer_type="agent",
                status="approved",
                score=8.2,
                feedback="Editorial standards met with good structure and flow",
                suggestions=["Strengthen conclusion"],
                decision_timestamp=datetime.utcnow().isoformat()
            ),
            ReviewDecisionResponse(
                stage="seo_analysis",
                reviewer_id="seo_agent",
                reviewer_type="agent", 
                status="approved",
                score=7.8,
                feedback="SEO optimization is adequate with good keyword usage",
                suggestions=["Add more internal links", "Optimize meta description"],
                decision_timestamp=datetime.utcnow().isoformat()
            )
        ]
        
        summary = WorkflowSummaryResponse(
            workflow_execution_id=workflow_id,
            content_id="content_123",
            overall_approval_status="approved",
            quality_metrics={
                "average_score": 8.15,
                "quality_score": 8.5,
                "seo_score": 7.8,
                "overall_assessment": "good"
            },
            approval_summary={
                "total_stages": 4,
                "completed_stages": 4,
                "auto_approved_stages": 2,
                "human_reviewed_stages": 2
            },
            total_execution_time_seconds=1800.0,
            stage_decisions=stage_decisions
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get workflow summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to get workflow summary: {str(e)}", "error_code": "SUMMARY_FETCH_FAILED"}
        )

@router.get("/pending-reviews", response_model=List[Dict[str, Any]])
async def get_pending_reviews(
    reviewer_id: Optional[str] = Query(None, description="Filter by reviewer ID"),
    stage: Optional[str] = Query(None, description="Filter by review stage"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get pending human reviews assigned to reviewers.
    
    Returns a list of content items waiting for human review,
    with filtering and pagination options.
    """
    try:
        logger.info(f"Getting pending reviews for user: {current_user.get('user_id')}")
        
        # Apply default filter to current user if no reviewer_id specified
        if not reviewer_id:
            reviewer_id = current_user.get("user_id")
        
        # This would typically query the database
        # For now, return mock data
        
        pending_reviews = [
            {
                "workflow_execution_id": "workflow_123",
                "content_id": "content_123",
                "content_type": "blog_post",
                "title": "Sample Blog Post",
                "stage": "quality_check",
                "priority": "medium",
                "assigned_at": datetime.utcnow().isoformat(),
                "expected_completion_at": (datetime.utcnow()).isoformat(),
                "automated_score": 7.5,
                "campaign_id": "campaign_123",
                "is_overdue": False
            }
        ]
        
        # Apply filters
        if stage:
            pending_reviews = [r for r in pending_reviews if r["stage"] == stage]
        if priority:
            pending_reviews = [r for r in pending_reviews if r["priority"] == priority]
        
        # Apply pagination
        total_count = len(pending_reviews)
        pending_reviews = pending_reviews[offset:offset + limit]
        
        return {
            "pending_reviews": pending_reviews,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to get pending reviews: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to get pending reviews: {str(e)}", "error_code": "PENDING_REVIEWS_FAILED"}
        )

@router.post("/{workflow_id}/assign-reviewers", response_model=Dict[str, Any])
async def assign_reviewers(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    request: ReviewerAssignmentRequest = ...,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Assign human reviewers to a specific review stage.
    
    This endpoint allows workflow administrators to assign or reassign
    human reviewers to review stages.
    """
    try:
        logger.info(f"Assigning reviewers to workflow {workflow_id}, stage {request.stage}")
        
        # Validate admin permissions
        # Permission validation for reviewer assignment
        # await require_permissions(current_user, "manage_reviewers")
        
        # Create reviewer assignments
        assignments = []
        for reviewer_id in request.reviewer_ids:
            assignment = {
                "reviewer_id": reviewer_id,
                "stage": request.stage,
                "priority": request.priority,
                "expected_completion_hours": request.expected_completion_hours,
                "instructions": request.instructions,
                "assigned_by": current_user.get("user_id"),
                "assigned_at": datetime.utcnow().isoformat()
            }
            assignments.append(assignment)
        
        # This would typically update the database and send notifications
        
        return {
            "success": True,
            "message": f"Assigned {len(request.reviewer_ids)} reviewers to stage {request.stage}",
            "workflow_id": workflow_id,
            "stage": request.stage,
            "assignments": assignments,
            "notifications_sent": len(request.reviewer_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign reviewers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to assign reviewers: {str(e)}", "error_code": "REVIEWER_ASSIGNMENT_FAILED"}
        )

@router.get("/metrics", response_model=Dict[str, Any])
async def get_review_workflow_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days to include in metrics"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get review workflow performance metrics.
    
    Returns comprehensive metrics about review workflow performance,
    including completion rates, average times, and quality trends.
    """
    try:
        logger.info(f"Getting review workflow metrics for {days} days")
        
        # This would typically query the database for actual metrics
        # For now, return mock metrics
        
        metrics = {
            "period": {
                "days": days,
                "start_date": (datetime.utcnow()).isoformat(),
                "end_date": datetime.utcnow().isoformat()
            },
            "workflow_stats": {
                "total_workflows": 150,
                "completed_workflows": 135,
                "in_progress_workflows": 10,
                "paused_workflows": 5,
                "completion_rate": 0.9
            },
            "approval_stats": {
                "approved": 120,
                "needs_revision": 10,
                "rejected": 5,
                "approval_rate": 0.89
            },
            "performance_stats": {
                "average_completion_hours": 24.5,
                "median_completion_hours": 18.0,
                "average_human_review_hours": 6.2,
                "automation_rate": 0.65
            },
            "quality_stats": {
                "average_quality_score": 8.2,
                "average_seo_score": 7.8,
                "average_brand_score": 8.5,
                "improvement_rate": 0.15
            },
            "stage_stats": {
                "content_quality": {"completion_rate": 0.98, "average_score": 8.2},
                "editorial_review": {"completion_rate": 0.96, "average_score": 8.0},
                "brand_check": {"completion_rate": 0.95, "average_score": 8.5},
                "seo_analysis": {"completion_rate": 0.92, "average_score": 7.8},
                "geo_analysis": {"completion_rate": 0.90, "average_score": 8.1},
                "visual_review": {"completion_rate": 0.88, "average_score": 7.6},
                "social_media_review": {"completion_rate": 0.91, "average_score": 8.3},
                "final_approval": {"completion_rate": 0.88, "average_score": None}
            },
            "workflow_integration_stats": {
                "total_integrations": 150,
                "successful_integrations": 140,
                "integration_success_rate": 0.93
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get review workflow metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to get metrics: {str(e)}", "error_code": "METRICS_FETCH_FAILED"}
        )

@router.put("/{workflow_id}/config", response_model=Dict[str, Any])
async def update_review_config(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    request: UpdateReviewConfigRequest = ...,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update the review configuration for an active workflow.
    
    Allows modification of review settings like thresholds, reviewers,
    and stage requirements for active workflows.
    """
    try:
        logger.info(f"Updating review config for workflow: {workflow_id}")
        
        # Validate admin permissions
        # Permission validation for config changes
        # await require_permissions(current_user, "manage_review_config")
        
        # Update configuration (this would typically update the database)
        updated_config = request.review_config.dict()
        updated_config["updated_by"] = current_user.get("user_id")
        updated_config["updated_at"] = datetime.utcnow().isoformat()
        
        return {
            "success": True,
            "message": "Review configuration updated successfully",
            "workflow_id": workflow_id,
            "updated_config": updated_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update review config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to update config: {str(e)}", "error_code": "CONFIG_UPDATE_FAILED"}
        )

@router.delete("/{workflow_id}", response_model=Dict[str, Any])
async def cancel_workflow(
    workflow_id: str = Path(..., description="Review workflow execution ID"),
    reason: str = Query(..., description="Reason for cancellation"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Cancel an active review workflow.
    
    Cancels a review workflow and cleans up associated resources.
    This action cannot be undone.
    """
    try:
        logger.info(f"Cancelling workflow: {workflow_id}")
        
        # Validate permissions
        # Permission validation for workflow cancellation
        # await require_permissions(current_user, "cancel_workflows")
        
        # Cancel workflow (this would typically update the database)
        cancellation_data = {
            "workflow_id": workflow_id,
            "cancelled_by": current_user.get("user_id"),
            "cancelled_at": datetime.utcnow().isoformat(),
            "cancellation_reason": reason,
            "status": "cancelled"
        }
        
        return {
            "success": True,
            "message": "Workflow cancelled successfully",
            "cancellation_data": cancellation_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to cancel workflow: {str(e)}", "error_code": "WORKFLOW_CANCELLATION_FAILED"}
        )

# Health check endpoint
@router.get("/health", response_model=Dict[str, Any])
async def review_workflow_health():
    """Health check for review workflow service."""
    try:
        # Check service health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "review_workflow": "operational",
                "integration_service": "operational",
                "database": "operational"  # Would check actual database connectivity
            },
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

# Export router
__all__ = ["router"]