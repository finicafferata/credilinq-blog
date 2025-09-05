"""
Review Workflow Models and State Management
LangGraph-compatible dataclasses for the 4-stage review workflow system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json

# Enums matching Prisma schema
class ReviewStage(Enum):
    """Review workflow stages - Complete 8-stage review system"""
    CONTENT_QUALITY = "content_quality"
    EDITORIAL_REVIEW = "editorial_review"  
    BRAND_CHECK = "brand_check"
    SEO_ANALYSIS = "seo_analysis"
    GEO_ANALYSIS = "geo_analysis"
    VISUAL_REVIEW = "visual_review"
    SOCIAL_MEDIA_REVIEW = "social_media_review"
    FINAL_APPROVAL = "final_approval"

class ReviewStatus(Enum):
    """Review status for each stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AGENT_APPROVED = "agent_approved"
    REQUIRES_HUMAN_REVIEW = "requires_human_review"
    HUMAN_APPROVED = "human_approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"

class ReviewDecision(Enum):
    """Review decision types"""
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"

@dataclass
class ReviewCheckpoint:
    """Human review checkpoint data matching database schema"""
    stage: ReviewStage
    content_id: str
    workflow_id: str
    
    # Human reviewer assignment
    reviewer_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Review state
    status: ReviewStatus = ReviewStatus.PENDING
    requires_human: bool = False
    notification_sent: bool = False
    
    # AI agent results
    automated_score: Optional[float] = None  # 0.0-1.0
    automated_feedback: List[str] = field(default_factory=list)
    
    # Human review results
    human_decision: Optional[ReviewDecision] = None
    human_comments: Optional[str] = None
    human_score: Optional[float] = None  # 0.0-1.0
    
    # Completion tracking
    completed_at: Optional[datetime] = None
    review_duration_ms: Optional[int] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database/API serialization"""
        return {
            "stage": self.stage.value,
            "content_id": self.content_id,
            "workflow_id": self.workflow_id,
            "reviewer_id": self.reviewer_id,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status.value,
            "requires_human": self.requires_human,
            "notification_sent": self.notification_sent,
            "automated_score": self.automated_score,
            "automated_feedback": self.automated_feedback,
            "human_decision": self.human_decision.value if self.human_decision else None,
            "human_comments": self.human_comments,
            "human_score": self.human_score,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "review_duration_ms": self.review_duration_ms,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ReviewFeedback:
    """Structured feedback from reviewers"""
    workflow_id: str
    content_id: str
    reviewer_id: str
    stage: ReviewStage
    decision: ReviewDecision
    
    # Feedback details
    comments: str = ""
    reviewed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database/API serialization"""
        return {
            "workflow_id": self.workflow_id,
            "content_id": self.content_id,
            "reviewer_id": self.reviewer_id,
            "stage": self.stage.value,
            "decision": self.decision.value,
            "comments": self.comments,
            "reviewed_at": self.reviewed_at.isoformat()
        }

@dataclass
class ReviewWorkflowState:
    """
    LangGraph state for review workflow - matches database schema
    This is the main state object passed between workflow nodes
    """
    content_id: str
    content_type: str  # blog_post, social_post, email
    content_data: Dict[str, Any]  # Snapshot of content being reviewed
    
    # Workflow identification
    workflow_execution_id: str = field(default_factory=lambda: f"review_{uuid.uuid4().hex[:16]}")
    campaign_id: Optional[str] = None
    
    # Review stage tracking
    current_stage: ReviewStage = ReviewStage.CONTENT_QUALITY
    completed_stages: List[ReviewStage] = field(default_factory=list)
    failed_stages: List[ReviewStage] = field(default_factory=list)
    
    # Workflow status
    workflow_status: ReviewStatus = ReviewStatus.PENDING
    is_paused: bool = False
    pause_reason: Optional[str] = None
    
    # Human review management
    active_checkpoints: Dict[str, ReviewCheckpoint] = field(default_factory=dict)
    review_history: List[ReviewFeedback] = field(default_factory=list)
    
    # Configuration
    auto_approve_threshold: float = 0.85
    require_human_approval: bool = True
    parallel_reviews: bool = False
    
    # Performance tracking
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_review_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LangGraph state management and database storage"""
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "content_data": self.content_data,
            "workflow_execution_id": self.workflow_execution_id,
            "campaign_id": self.campaign_id,
            "current_stage": self.current_stage.value,
            "completed_stages": [stage.value for stage in self.completed_stages],
            "failed_stages": [stage.value for stage in self.failed_stages],
            "workflow_status": self.workflow_status.value,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "active_checkpoints": {k: v.to_dict() for k, v in self.active_checkpoints.items()},
            "review_history": [feedback.to_dict() for feedback in self.review_history],
            "auto_approve_threshold": self.auto_approve_threshold,
            "require_human_approval": self.require_human_approval,
            "parallel_reviews": self.parallel_reviews,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_review_time_ms": self.total_review_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewWorkflowState':
        """Create ReviewWorkflowState from dictionary (database/API deserialization)"""
        # Convert string enums back to enum objects
        current_stage = ReviewStage(data["current_stage"])
        completed_stages = [ReviewStage(s) for s in data.get("completed_stages", [])]
        failed_stages = [ReviewStage(s) for s in data.get("failed_stages", [])]
        workflow_status = ReviewStatus(data["workflow_status"])
        
        # Reconstruct active checkpoints
        active_checkpoints = {}
        for key, checkpoint_data in data.get("active_checkpoints", {}).items():
            checkpoint = ReviewCheckpoint(
                stage=ReviewStage(checkpoint_data["stage"]),
                content_id=checkpoint_data["content_id"],
                workflow_id=checkpoint_data["workflow_id"],
                reviewer_id=checkpoint_data.get("reviewer_id"),
                assigned_at=datetime.fromisoformat(checkpoint_data["assigned_at"]) if checkpoint_data.get("assigned_at") else None,
                deadline=datetime.fromisoformat(checkpoint_data["deadline"]) if checkpoint_data.get("deadline") else None,
                status=ReviewStatus(checkpoint_data["status"]),
                requires_human=checkpoint_data.get("requires_human", False),
                notification_sent=checkpoint_data.get("notification_sent", False),
                automated_score=checkpoint_data.get("automated_score"),
                automated_feedback=checkpoint_data.get("automated_feedback", []),
                human_decision=ReviewDecision(checkpoint_data["human_decision"]) if checkpoint_data.get("human_decision") else None,
                human_comments=checkpoint_data.get("human_comments"),
                human_score=checkpoint_data.get("human_score"),
                completed_at=datetime.fromisoformat(checkpoint_data["completed_at"]) if checkpoint_data.get("completed_at") else None,
                review_duration_ms=checkpoint_data.get("review_duration_ms"),
                metadata=checkpoint_data.get("metadata", {}),
                created_at=datetime.fromisoformat(checkpoint_data["created_at"]) if checkpoint_data.get("created_at") else datetime.utcnow()
            )
            active_checkpoints[key] = checkpoint
        
        # Reconstruct review history
        review_history = []
        for feedback_data in data.get("review_history", []):
            feedback = ReviewFeedback(
                workflow_id=feedback_data["workflow_id"],
                content_id=feedback_data["content_id"],
                reviewer_id=feedback_data["reviewer_id"],
                stage=ReviewStage(feedback_data["stage"]),
                decision=ReviewDecision(feedback_data["decision"]),
                comments=feedback_data.get("comments", ""),
                reviewed_at=datetime.fromisoformat(feedback_data["reviewed_at"]) if feedback_data.get("reviewed_at") else datetime.utcnow()
            )
            review_history.append(feedback)
        
        return cls(
            content_id=data["content_id"],
            content_type=data["content_type"],
            content_data=data["content_data"],
            workflow_execution_id=data["workflow_execution_id"],
            campaign_id=data.get("campaign_id"),
            current_stage=current_stage,
            completed_stages=completed_stages,
            failed_stages=failed_stages,
            workflow_status=workflow_status,
            is_paused=data.get("is_paused", False),
            pause_reason=data.get("pause_reason"),
            active_checkpoints=active_checkpoints,
            review_history=review_history,
            auto_approve_threshold=data.get("auto_approve_threshold", 0.85),
            require_human_approval=data.get("require_human_approval", True),
            parallel_reviews=data.get("parallel_reviews", False),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            total_review_time_ms=data.get("total_review_time_ms")
        )

    def get_current_checkpoint(self) -> Optional[ReviewCheckpoint]:
        """Get the active checkpoint for current stage"""
        checkpoint_key = f"{self.current_stage.value}_{self.content_id}"
        return self.active_checkpoints.get(checkpoint_key)

    def create_checkpoint(self, stage: ReviewStage, requires_human: bool = False) -> ReviewCheckpoint:
        """Create a new checkpoint for the given stage"""
        checkpoint = ReviewCheckpoint(
            stage=stage,
            content_id=self.content_id,
            workflow_id=self.workflow_execution_id,
            requires_human=requires_human,
            deadline=datetime.utcnow() + timedelta(hours=24) if requires_human else None  # 24h default deadline
        )
        
        checkpoint_key = f"{stage.value}_{self.content_id}"
        self.active_checkpoints[checkpoint_key] = checkpoint
        
        return checkpoint

    def complete_stage(self, stage: ReviewStage, success: bool = True):
        """Mark a stage as completed or failed"""
        if success:
            if stage not in self.completed_stages:
                self.completed_stages.append(stage)
            # Remove from failed if it was there
            if stage in self.failed_stages:
                self.failed_stages.remove(stage)
        else:
            if stage not in self.failed_stages:
                self.failed_stages.append(stage)
            # Remove from completed if it was there  
            if stage in self.completed_stages:
                self.completed_stages.remove(stage)
        
        # Clean up checkpoint
        checkpoint_key = f"{stage.value}_{self.content_id}"
        if checkpoint_key in self.active_checkpoints:
            self.active_checkpoints[checkpoint_key].completed_at = datetime.utcnow()

    def add_feedback(self, feedback: ReviewFeedback):
        """Add human feedback to review history"""
        self.review_history.append(feedback)

    def get_progress_percentage(self) -> int:
        """Calculate review progress as percentage (0-100)"""
        total_stages = len(ReviewStage)
        completed_count = len(self.completed_stages)
        return int((completed_count / total_stages) * 100)

    def get_current_stage_index(self) -> int:
        """Get the current stage index (1-based) for UI display"""
        stages_order = [
            ReviewStage.CONTENT_QUALITY,
            ReviewStage.EDITORIAL_REVIEW,
            ReviewStage.BRAND_CHECK, 
            ReviewStage.SEO_ANALYSIS,
            ReviewStage.GEO_ANALYSIS,
            ReviewStage.VISUAL_REVIEW,
            ReviewStage.SOCIAL_MEDIA_REVIEW,
            ReviewStage.FINAL_APPROVAL
        ]
        try:
            return stages_order.index(self.current_stage) + 1
        except ValueError:
            return 1

    def should_require_human_review(self, automated_score: float) -> bool:
        """Determine if human review is required based on configuration and score"""
        if self.require_human_approval:
            return True
        
        return automated_score < self.auto_approve_threshold

    def is_workflow_complete(self) -> bool:
        """Check if the entire workflow is complete"""
        return len(self.completed_stages) == len(ReviewStage)

    def is_workflow_failed(self) -> bool:
        """Check if the workflow has failed (any stage failed and not recoverable)"""
        return len(self.failed_stages) > 0

@dataclass
class ReviewAgentResult:
    """Result from an AI review agent (Quality, Brand, SEO, etc.)"""
    stage: ReviewStage
    content_id: str
    
    # Analysis results
    automated_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    
    # Feedback and recommendations
    feedback: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    
    # Metrics (stage-specific)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Decision
    requires_human_review: bool = False
    auto_approved: bool = False
    
    # Execution details
    execution_time_ms: int = 0
    model_used: str = ""
    tokens_used: int = 0
    cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API"""
        return {
            "stage": self.stage.value,
            "content_id": self.content_id,
            "automated_score": self.automated_score,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "recommendations": self.recommendations,
            "issues_found": self.issues_found,
            "metrics": self.metrics,
            "requires_human_review": self.requires_human_review,
            "auto_approved": self.auto_approved,
            "execution_time_ms": self.execution_time_ms,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "cost": self.cost
        }