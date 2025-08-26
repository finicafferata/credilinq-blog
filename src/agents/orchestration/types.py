"""
Shared types for campaign orchestration to avoid circular imports.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from ..core.base_agent import AgentResult

class CampaignType(Enum):
    """Types of campaigns that can be orchestrated."""
    BLOG_CREATION = "blog_creation"
    CONTENT_REPURPOSING = "content_repurposing"
    SOCIAL_MEDIA_CAMPAIGN = "social_media_campaign"
    SEO_OPTIMIZATION = "seo_optimization"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    CONTENT_MARKETING = "content_marketing"
    BLOG_SERIES = "blog_series"
    SEO_CONTENT = "seo_content"

class TaskStatus(Enum):
    """Status of individual campaign tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowStatus(Enum):
    """Status of workflow execution."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CampaignTask:
    """Represents a single task within a campaign workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: str = ""
    task_type: str = ""
    agent_type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AgentResult] = None
    assigned_agent_id: Optional[str] = None
    priority: int = 0
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CampaignWithTasks:
    """Campaign data with associated tasks and metadata."""
    id: str
    name: str
    description: str
    campaign_type: CampaignType
    status: str
    orchestrator_id: Optional[str]
    strategy_id: Optional[str]
    tasks: List[CampaignTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowExecutionCreate:
    """Data for creating a new workflow execution record."""
    campaign_id: str
    orchestrator_id: str
    workflow_type: str
    status: WorkflowStatus = WorkflowStatus.INITIALIZING
    input_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)