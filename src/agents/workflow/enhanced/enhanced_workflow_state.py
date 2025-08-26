"""
Enhanced Workflow State Management with Checkpointing

This module provides advanced state management capabilities for campaign workflows,
including state persistence, checkpointing, and recovery mechanisms.
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Union
from typing_extensions import Annotated
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateTransitionType(Enum):
    """Types of state transitions."""
    NODE_ENTRY = "node_entry"
    NODE_EXIT = "node_exit"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    RECOVERY = "recovery"
    COMPLETION = "completion"


@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a specific point in time."""
    snapshot_id: str
    campaign_id: str
    workflow_id: str
    node_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create snapshot from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""
    checkpoint_id: str
    campaign_id: str
    workflow_id: str
    node_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    execution_path: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_recovery_point: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        """Create checkpoint from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EnhancedWorkflowState(TypedDict):
    """
    Enhanced workflow state with campaign integration and checkpointing.
    
    This extends the basic workflow state with campaign-specific fields
    and advanced state management capabilities.
    """
    # Core workflow identifiers
    workflow_id: str
    campaign_id: str
    execution_id: str
    
    # Workflow status and control
    status: WorkflowStatus
    current_node: str
    execution_path: Annotated[List[str], "Path of executed nodes"]
    
    # Checkpointing and recovery
    checkpoints: Annotated[List[WorkflowCheckpoint], "Workflow checkpoints"]
    last_checkpoint: Optional[str]
    recovery_point: Optional[str]
    
    # State data
    state_data: Annotated[Dict[str, Any], "Dynamic state data"]
    context: Annotated[Dict[str, Any], "Execution context"]
    
    # Error handling
    errors: Annotated[List[Dict[str, Any]], "Execution errors"]
    retry_count: int
    max_retries: int
    
    # Metadata and tracking
    created_at: datetime
    updated_at: datetime
    metadata: Annotated[Dict[str, Any], "Additional metadata"]


class CampaignWorkflowState(TypedDict):
    """
    Campaign-specific workflow state that integrates with blog workflow.
    
    This state extends the blog workflow state while adding campaign orchestration.
    """
    # Campaign identifiers
    campaign_id: str
    workflow_id: str
    execution_id: str
    
    # Campaign configuration
    campaign_type: str
    campaign_config: Annotated[Dict[str, Any], "Campaign configuration"]
    
    # Blog workflow integration (from existing blog_workflow.py)
    blog_title: str
    company_context: str
    content_type: str
    outline: Annotated[List[str], "The blog post's outline"]
    research: Annotated[dict, "Research keyed by section title"]
    geo_metadata: Annotated[dict, "GEO optimization package"]
    draft: Annotated[str, "The current draft of the blog post"]
    review_notes: Annotated[str, "Notes from the editor for revision"]
    final_post: str
    
    # Enhanced workflow management
    status: WorkflowStatus
    current_node: str
    execution_path: Annotated[List[str], "Path of executed nodes"]
    
    # Multi-channel content (for campaign orchestration)
    social_content: Annotated[Dict[str, Any], "Social media content variants"]
    email_content: Annotated[Dict[str, Any], "Email marketing content"]
    landing_page_content: Annotated[Dict[str, Any], "Landing page content"]
    
    # Agent coordination
    agent_results: Annotated[Dict[str, Any], "Results from individual agents"]
    agent_assignments: Annotated[Dict[str, str], "Agent assignments for tasks"]
    
    # Campaign tracking
    performance_metrics: Annotated[Dict[str, Any], "Campaign performance data"]
    optimization_suggestions: Annotated[List[str], "AI-generated optimizations"]
    
    # Checkpointing
    checkpoints: Annotated[List[WorkflowCheckpoint], "Workflow checkpoints"]
    last_checkpoint: Optional[str]
    
    # Error handling
    errors: Annotated[List[Dict[str, Any]], "Execution errors"]
    retry_count: int
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    metadata: Annotated[Dict[str, Any], "Additional metadata"]


class StateManager:
    """
    Advanced state manager for workflow state persistence and recovery.
    """
    
    def __init__(self):
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.checkpoints: Dict[str, WorkflowCheckpoint] = {}
        
    def create_snapshot(
        self,
        campaign_id: str,
        workflow_id: str,
        node_id: str,
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateSnapshot:
        """Create a state snapshot."""
        snapshot_id = f"{campaign_id}_{workflow_id}_{node_id}_{int(datetime.now().timestamp())}"
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            node_id=node_id,
            timestamp=datetime.now(),
            state_data=self._sanitize_state_data(state_data),
            metadata=metadata or {}
        )
        
        self.snapshots[snapshot_id] = snapshot
        logger.debug(f"Created state snapshot: {snapshot_id}")
        
        return snapshot
        
    def create_checkpoint(
        self,
        campaign_id: str,
        workflow_id: str,
        node_id: str,
        state_data: Dict[str, Any],
        execution_path: List[str],
        is_recovery_point: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowCheckpoint:
        """Create a workflow checkpoint."""
        checkpoint_id = f"{campaign_id}_{workflow_id}_checkpoint_{int(datetime.now().timestamp())}"
        
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            node_id=node_id,
            timestamp=datetime.now(),
            state_data=self._sanitize_state_data(state_data),
            execution_path=execution_path.copy(),
            metadata=metadata or {},
            is_recovery_point=is_recovery_point
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(f"Created checkpoint: {checkpoint_id}")
        
        return checkpoint
        
    def get_latest_checkpoint(
        self,
        campaign_id: str,
        workflow_id: str
    ) -> Optional[WorkflowCheckpoint]:
        """Get the latest checkpoint for a workflow."""
        matching_checkpoints = [
            cp for cp in self.checkpoints.values()
            if cp.campaign_id == campaign_id and cp.workflow_id == workflow_id
        ]
        
        if not matching_checkpoints:
            return None
            
        return max(matching_checkpoints, key=lambda cp: cp.timestamp)
        
    def get_recovery_point(
        self,
        campaign_id: str,
        workflow_id: str
    ) -> Optional[WorkflowCheckpoint]:
        """Get the latest recovery point for a workflow."""
        matching_checkpoints = [
            cp for cp in self.checkpoints.values()
            if (cp.campaign_id == campaign_id and 
                cp.workflow_id == workflow_id and 
                cp.is_recovery_point)
        ]
        
        if not matching_checkpoints:
            return None
            
        return max(matching_checkpoints, key=lambda cp: cp.timestamp)
        
    def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """Restore workflow state from a checkpoint."""
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return None
            
        checkpoint = self.checkpoints[checkpoint_id]
        
        # Create restoration state
        restored_state = checkpoint.state_data.copy()
        restored_state.update({
            'workflow_id': checkpoint.workflow_id,
            'campaign_id': checkpoint.campaign_id,
            'current_node': checkpoint.node_id,
            'execution_path': checkpoint.execution_path.copy(),
            'restored_from_checkpoint': checkpoint_id,
            'restored_at': datetime.now()
        })
        
        logger.info(f"Restored state from checkpoint: {checkpoint_id}")
        return restored_state
        
    def get_execution_history(
        self,
        campaign_id: str,
        workflow_id: str
    ) -> List[StateSnapshot]:
        """Get execution history for a workflow."""
        matching_snapshots = [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.campaign_id == campaign_id and snapshot.workflow_id == workflow_id
        ]
        
        return sorted(matching_snapshots, key=lambda s: s.timestamp)
        
    def _sanitize_state_data(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state data for serialization."""
        sanitized = {}
        
        for key, value in state_data.items():
            try:
                # Test if the value is JSON serializable
                json.dumps(value, default=str)
                sanitized[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable objects to string representation
                sanitized[key] = str(value)
                logger.debug(f"Converted non-serializable value for key {key}")
                
        return sanitized
        
    def cleanup_old_data(
        self,
        max_age_days: int = 30,
        keep_recovery_points: bool = True
    ):
        """Clean up old snapshots and checkpoints."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean up snapshots
        old_snapshots = [
            sid for sid, snapshot in self.snapshots.items()
            if snapshot.timestamp < cutoff_date
        ]
        
        for snapshot_id in old_snapshots:
            del self.snapshots[snapshot_id]
            
        # Clean up checkpoints (preserve recovery points if requested)
        old_checkpoints = [
            cid for cid, checkpoint in self.checkpoints.items()
            if (checkpoint.timestamp < cutoff_date and 
                not (keep_recovery_points and checkpoint.is_recovery_point))
        ]
        
        for checkpoint_id in old_checkpoints:
            del self.checkpoints[checkpoint_id]
            
        logger.info(f"Cleaned up {len(old_snapshots)} snapshots and {len(old_checkpoints)} checkpoints")


# Global state manager instance
state_manager = StateManager()


def create_enhanced_state(
    campaign_id: str,
    workflow_id: str,
    initial_data: Optional[Dict[str, Any]] = None
) -> EnhancedWorkflowState:
    """Create an enhanced workflow state."""
    execution_id = f"{workflow_id}_{int(datetime.now().timestamp())}"
    
    state = EnhancedWorkflowState(
        workflow_id=workflow_id,
        campaign_id=campaign_id,
        execution_id=execution_id,
        status=WorkflowStatus.PENDING,
        current_node="",
        execution_path=[],
        checkpoints=[],
        last_checkpoint=None,
        recovery_point=None,
        state_data=initial_data or {},
        context={},
        errors=[],
        retry_count=0,
        max_retries=3,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={}
    )
    
    return state


def create_campaign_state(
    campaign_id: str,
    campaign_type: str,
    blog_title: str,
    company_context: str,
    content_type: str = "blog",
    campaign_config: Optional[Dict[str, Any]] = None
) -> CampaignWorkflowState:
    """Create a campaign workflow state."""
    workflow_id = f"campaign_{campaign_type}_{int(datetime.now().timestamp())}"
    execution_id = f"{workflow_id}_{int(datetime.now().timestamp())}"
    
    state = CampaignWorkflowState(
        campaign_id=campaign_id,
        workflow_id=workflow_id,
        execution_id=execution_id,
        campaign_type=campaign_type,
        campaign_config=campaign_config or {},
        
        # Blog workflow fields
        blog_title=blog_title,
        company_context=company_context,
        content_type=content_type,
        outline=[],
        research={},
        geo_metadata={},
        draft="",
        review_notes="",
        final_post="",
        
        # Enhanced workflow fields
        status=WorkflowStatus.PENDING,
        current_node="",
        execution_path=[],
        
        # Multi-channel content
        social_content={},
        email_content={},
        landing_page_content={},
        
        # Agent coordination
        agent_results={},
        agent_assignments={},
        
        # Campaign tracking
        performance_metrics={},
        optimization_suggestions=[],
        
        # Checkpointing
        checkpoints=[],
        last_checkpoint=None,
        
        # Error handling
        errors=[],
        retry_count=0,
        
        # Metadata
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={}
    )
    
    return state


def update_state_node(
    state: Union[EnhancedWorkflowState, CampaignWorkflowState],
    node_id: str,
    node_data: Optional[Dict[str, Any]] = None
) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
    """Update state when entering a new node."""
    state['current_node'] = node_id
    state['execution_path'].append(node_id)
    state['updated_at'] = datetime.now()
    
    if node_data:
        state['state_data'].update(node_data)
        
    return state


def add_state_error(
    state: Union[EnhancedWorkflowState, CampaignWorkflowState],
    error: str,
    node_id: Optional[str] = None
) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
    """Add an error to the workflow state."""
    error_data = {
        'error': error,
        'node_id': node_id or state.get('current_node', ''),
        'timestamp': datetime.now().isoformat(),
        'retry_count': state.get('retry_count', 0)
    }
    
    state['errors'].append(error_data)
    state['updated_at'] = datetime.now()
    
    return state