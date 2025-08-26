"""
Campaign State Manager for Persistent State Management

This module provides campaign-specific state persistence, recovery, and
management capabilities integrated with the database backend.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

from src.agents.core.database_service import DatabaseService, get_db_service
from .enhanced_workflow_state import (
    WorkflowCheckpoint, StateSnapshot, EnhancedWorkflowState, 
    CampaignWorkflowState, WorkflowStatus
)

logger = logging.getLogger(__name__)


class PersistenceStrategy(Enum):
    """Strategies for state persistence."""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    INTERVAL = "interval"
    ON_CHECKPOINT = "on_checkpoint"


class StateTransitionType(Enum):
    """Types of state transitions."""
    CREATED = "created"
    UPDATED = "updated"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass
class StateTransition:
    """Record of a state transition."""
    transition_id: str
    campaign_id: str
    workflow_id: str
    transition_type: StateTransitionType
    from_state: Optional[str]
    to_state: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return asdict(self)


@dataclass
class PersistenceConfig:
    """Configuration for state persistence."""
    strategy: PersistenceStrategy = PersistenceStrategy.ON_CHECKPOINT
    batch_size: int = 10
    interval_seconds: int = 30
    max_history_days: int = 30
    compress_old_states: bool = True
    encryption_enabled: bool = False
    backup_enabled: bool = True


class CampaignStateManager:
    """
    Advanced state manager for campaign workflows with database persistence.
    
    This manager provides:
    - Persistent state storage in the database
    - State recovery and restoration
    - State transition tracking
    - Automatic cleanup and archival
    - Integration with campaign orchestration
    """
    
    def __init__(
        self,
        db_service: Optional[DatabaseService] = None,
        config: Optional[PersistenceConfig] = None
    ):
        self.db_service = db_service or get_db_service()
        self.config = config or PersistenceConfig()
        self.active_states: Dict[str, Union[EnhancedWorkflowState, CampaignWorkflowState]] = {}
        self.transition_history: Dict[str, List[StateTransition]] = {}
        
    async def save_state(
        self,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        transition_type: StateTransitionType = StateTransitionType.UPDATED
    ) -> bool:
        """
        Save workflow state to persistent storage.
        
        Args:
            state: Workflow state to save
            transition_type: Type of state transition
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            campaign_id = state['campaign_id']
            workflow_id = state['workflow_id']
            
            # Prepare state data for storage
            state_data = self._prepare_state_for_storage(state)
            
            # Save to database
            async with self.db_service.get_connection() as conn:
                # Insert or update workflow state
                query = """
                INSERT INTO campaign_workflow_states 
                (campaign_id, workflow_id, execution_id, state_data, status, 
                 current_node, execution_path, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (campaign_id, workflow_id, execution_id)
                DO UPDATE SET
                    state_data = EXCLUDED.state_data,
                    status = EXCLUDED.status,
                    current_node = EXCLUDED.current_node,
                    execution_path = EXCLUDED.execution_path,
                    updated_at = EXCLUDED.updated_at
                """
                
                await conn.execute(
                    query,
                    campaign_id,
                    workflow_id,
                    state.get('execution_id', ''),
                    json.dumps(state_data),
                    state.get('status', WorkflowStatus.PENDING).value,
                    state.get('current_node', ''),
                    json.dumps(state.get('execution_path', [])),
                    state.get('created_at', datetime.now()),
                    datetime.now()
                )
                
            # Record state transition
            await self._record_state_transition(state, transition_type)
            
            # Update active states cache
            state_key = f"{campaign_id}_{workflow_id}"
            self.active_states[state_key] = state
            
            logger.debug(f"Saved state for campaign {campaign_id}, workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            return False
            
    async def load_state(
        self,
        campaign_id: str,
        workflow_id: str,
        execution_id: Optional[str] = None
    ) -> Optional[Union[EnhancedWorkflowState, CampaignWorkflowState]]:
        """
        Load workflow state from persistent storage.
        
        Args:
            campaign_id: ID of the campaign
            workflow_id: ID of the workflow
            execution_id: Optional specific execution ID
            
        Returns:
            Loaded workflow state or None if not found
        """
        try:
            async with self.db_service.get_connection() as conn:
                if execution_id:
                    query = """
                    SELECT state_data, status, current_node, execution_path, 
                           created_at, updated_at
                    FROM campaign_workflow_states 
                    WHERE campaign_id = $1 AND workflow_id = $2 AND execution_id = $3
                    """
                    params = [campaign_id, workflow_id, execution_id]
                else:
                    query = """
                    SELECT state_data, status, current_node, execution_path, 
                           created_at, updated_at
                    FROM campaign_workflow_states 
                    WHERE campaign_id = $1 AND workflow_id = $2
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                    params = [campaign_id, workflow_id]
                    
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    logger.warning(f"No state found for campaign {campaign_id}, workflow {workflow_id}")
                    return None
                    
                # Reconstruct state from database data
                state_data = json.loads(row['state_data'])
                state = self._reconstruct_state_from_storage(
                    state_data,
                    campaign_id,
                    workflow_id,
                    row['status'],
                    row['current_node'],
                    json.loads(row['execution_path']),
                    row['created_at'],
                    row['updated_at']
                )
                
                # Cache the loaded state
                state_key = f"{campaign_id}_{workflow_id}"
                self.active_states[state_key] = state
                
                logger.debug(f"Loaded state for campaign {campaign_id}, workflow {workflow_id}")
                return state
                
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            return None
            
    async def save_checkpoint(
        self,
        campaign_id: str,
        workflow_id: str,
        checkpoint_id: str,
        state_data: Dict[str, Any],
        node_id: str = "",
        is_recovery_point: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a workflow checkpoint.
        
        Args:
            campaign_id: ID of the campaign
            workflow_id: ID of the workflow
            checkpoint_id: Unique checkpoint identifier
            state_data: State data to checkpoint
            node_id: Current node ID
            is_recovery_point: Whether this is a recovery point
            metadata: Additional metadata
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            checkpoint_data = self._prepare_state_for_storage(state_data)
            
            async with self.db_service.get_connection() as conn:
                query = """
                INSERT INTO campaign_workflow_checkpoints 
                (checkpoint_id, campaign_id, workflow_id, node_id, state_data, 
                 execution_path, is_recovery_point, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """
                
                await conn.execute(
                    query,
                    checkpoint_id,
                    campaign_id,
                    workflow_id,
                    node_id,
                    json.dumps(checkpoint_data),
                    json.dumps(state_data.get('execution_path', [])),
                    is_recovery_point,
                    json.dumps(metadata or {}),
                    datetime.now()
                )
                
            logger.info(f"Saved checkpoint {checkpoint_id} for campaign {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            return False
            
    async def load_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[WorkflowCheckpoint]:
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            
        Returns:
            Loaded checkpoint or None if not found
        """
        try:
            async with self.db_service.get_connection() as conn:
                query = """
                SELECT campaign_id, workflow_id, node_id, state_data, execution_path,
                       is_recovery_point, metadata, created_at
                FROM campaign_workflow_checkpoints 
                WHERE checkpoint_id = $1
                """
                
                row = await conn.fetchrow(query, checkpoint_id)
                
                if not row:
                    logger.warning(f"Checkpoint not found: {checkpoint_id}")
                    return None
                    
                checkpoint = WorkflowCheckpoint(
                    checkpoint_id=checkpoint_id,
                    campaign_id=row['campaign_id'],
                    workflow_id=row['workflow_id'],
                    node_id=row['node_id'],
                    timestamp=row['created_at'],
                    state_data=json.loads(row['state_data']),
                    execution_path=json.loads(row['execution_path']),
                    metadata=json.loads(row['metadata']),
                    is_recovery_point=row['is_recovery_point']
                )
                
                logger.debug(f"Loaded checkpoint: {checkpoint_id}")
                return checkpoint
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
            
    async def get_latest_checkpoint(
        self,
        campaign_id: str,
        workflow_id: str,
        recovery_point_only: bool = False
    ) -> Optional[WorkflowCheckpoint]:
        """
        Get the latest checkpoint for a workflow.
        
        Args:
            campaign_id: ID of the campaign
            workflow_id: ID of the workflow
            recovery_point_only: Only return recovery points
            
        Returns:
            Latest checkpoint or None if not found
        """
        try:
            async with self.db_service.get_connection() as conn:
                if recovery_point_only:
                    query = """
                    SELECT checkpoint_id, node_id, state_data, execution_path,
                           is_recovery_point, metadata, created_at
                    FROM campaign_workflow_checkpoints 
                    WHERE campaign_id = $1 AND workflow_id = $2 AND is_recovery_point = true
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                else:
                    query = """
                    SELECT checkpoint_id, node_id, state_data, execution_path,
                           is_recovery_point, metadata, created_at
                    FROM campaign_workflow_checkpoints 
                    WHERE campaign_id = $1 AND workflow_id = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                    
                row = await conn.fetchrow(query, campaign_id, workflow_id)
                
                if not row:
                    return None
                    
                checkpoint = WorkflowCheckpoint(
                    checkpoint_id=row['checkpoint_id'],
                    campaign_id=campaign_id,
                    workflow_id=workflow_id,
                    node_id=row['node_id'],
                    timestamp=row['created_at'],
                    state_data=json.loads(row['state_data']),
                    execution_path=json.loads(row['execution_path']),
                    metadata=json.loads(row['metadata']),
                    is_recovery_point=row['is_recovery_point']
                )
                
                return checkpoint
                
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {str(e)}")
            return None
            
    async def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Union[EnhancedWorkflowState, CampaignWorkflowState]]:
        """
        Restore workflow state from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore from
            
        Returns:
            Restored workflow state or None if restoration failed
        """
        try:
            checkpoint = await self.load_checkpoint(checkpoint_id)
            if not checkpoint:
                return None
                
            # Reconstruct state from checkpoint
            restored_state = self._reconstruct_state_from_checkpoint(checkpoint)
            
            # Record restoration as a state transition
            await self._record_state_transition(
                restored_state, 
                StateTransitionType.RECOVERED,
                metadata={'restored_from_checkpoint': checkpoint_id}
            )
            
            logger.info(f"Restored state from checkpoint: {checkpoint_id}")
            return restored_state
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {str(e)}")
            return None
            
    async def get_state_history(
        self,
        campaign_id: str,
        workflow_id: str,
        limit: int = 50
    ) -> List[StateTransition]:
        """
        Get state transition history for a workflow.
        
        Args:
            campaign_id: ID of the campaign
            workflow_id: ID of the workflow
            limit: Maximum number of transitions to return
            
        Returns:
            List of state transitions
        """
        try:
            async with self.db_service.get_connection() as conn:
                query = """
                SELECT transition_id, transition_type, from_state, to_state,
                       timestamp, metadata
                FROM campaign_state_transitions 
                WHERE campaign_id = $1 AND workflow_id = $2
                ORDER BY timestamp DESC
                LIMIT $3
                """
                
                rows = await conn.fetch(query, campaign_id, workflow_id, limit)
                
                transitions = []
                for row in rows:
                    transition = StateTransition(
                        transition_id=row['transition_id'],
                        campaign_id=campaign_id,
                        workflow_id=workflow_id,
                        transition_type=StateTransitionType(row['transition_type']),
                        from_state=row['from_state'],
                        to_state=row['to_state'],
                        timestamp=row['timestamp'],
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    transitions.append(transition)
                    
                return transitions
                
        except Exception as e:
            logger.error(f"Failed to get state history: {str(e)}")
            return []
            
    async def cleanup_old_data(
        self,
        max_age_days: int = 30,
        keep_recovery_points: bool = True
    ) -> Dict[str, int]:
        """
        Clean up old state data and checkpoints.
        
        Args:
            max_age_days: Maximum age of data to keep
            keep_recovery_points: Whether to preserve recovery points
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleanup_stats = {'states_cleaned': 0, 'checkpoints_cleaned': 0, 'transitions_cleaned': 0}
            
            async with self.db_service.get_connection() as conn:
                # Clean up old workflow states
                state_query = """
                DELETE FROM campaign_workflow_states 
                WHERE updated_at < $1
                """
                result = await conn.execute(state_query, cutoff_date)
                cleanup_stats['states_cleaned'] = int(result.split()[-1])
                
                # Clean up old checkpoints (preserve recovery points if requested)
                if keep_recovery_points:
                    checkpoint_query = """
                    DELETE FROM campaign_workflow_checkpoints 
                    WHERE created_at < $1 AND is_recovery_point = false
                    """
                else:
                    checkpoint_query = """
                    DELETE FROM campaign_workflow_checkpoints 
                    WHERE created_at < $1
                    """
                result = await conn.execute(checkpoint_query, cutoff_date)
                cleanup_stats['checkpoints_cleaned'] = int(result.split()[-1])
                
                # Clean up old state transitions
                transition_query = """
                DELETE FROM campaign_state_transitions 
                WHERE timestamp < $1
                """
                result = await conn.execute(transition_query, cutoff_date)
                cleanup_stats['transitions_cleaned'] = int(result.split()[-1])
                
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return {'states_cleaned': 0, 'checkpoints_cleaned': 0, 'transitions_cleaned': 0}
            
    def _prepare_state_for_storage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state data for database storage."""
        storage_state = {}
        
        for key, value in state.items():
            try:
                # Test JSON serialization
                json.dumps(value, default=str)
                storage_state[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable objects to string
                storage_state[key] = str(value)
                
        return storage_state
        
    def _reconstruct_state_from_storage(
        self,
        state_data: Dict[str, Any],
        campaign_id: str,
        workflow_id: str,
        status: str,
        current_node: str,
        execution_path: List[str],
        created_at: datetime,
        updated_at: datetime
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Reconstruct workflow state from stored data."""
        
        # Determine state type based on available fields
        if 'blog_title' in state_data:
            # This is a campaign workflow state
            state = CampaignWorkflowState(
                campaign_id=campaign_id,
                workflow_id=workflow_id,
                execution_id=state_data.get('execution_id', ''),
                campaign_type=state_data.get('campaign_type', ''),
                campaign_config=state_data.get('campaign_config', {}),
                
                # Blog workflow fields
                blog_title=state_data.get('blog_title', ''),
                company_context=state_data.get('company_context', ''),
                content_type=state_data.get('content_type', 'blog'),
                outline=state_data.get('outline', []),
                research=state_data.get('research', {}),
                geo_metadata=state_data.get('geo_metadata', {}),
                draft=state_data.get('draft', ''),
                review_notes=state_data.get('review_notes', ''),
                final_post=state_data.get('final_post', ''),
                
                # Enhanced workflow fields
                status=WorkflowStatus(status),
                current_node=current_node,
                execution_path=execution_path,
                
                # Multi-channel content
                social_content=state_data.get('social_content', {}),
                email_content=state_data.get('email_content', {}),
                landing_page_content=state_data.get('landing_page_content', {}),
                
                # Agent coordination
                agent_results=state_data.get('agent_results', {}),
                agent_assignments=state_data.get('agent_assignments', {}),
                
                # Campaign tracking
                performance_metrics=state_data.get('performance_metrics', {}),
                optimization_suggestions=state_data.get('optimization_suggestions', []),
                
                # Checkpointing
                checkpoints=state_data.get('checkpoints', []),
                last_checkpoint=state_data.get('last_checkpoint'),
                
                # Error handling
                errors=state_data.get('errors', []),
                retry_count=state_data.get('retry_count', 0),
                
                # Metadata
                created_at=created_at,
                updated_at=updated_at,
                metadata=state_data.get('metadata', {})
            )
        else:
            # This is an enhanced workflow state
            state = EnhancedWorkflowState(
                workflow_id=workflow_id,
                campaign_id=campaign_id,
                execution_id=state_data.get('execution_id', ''),
                status=WorkflowStatus(status),
                current_node=current_node,
                execution_path=execution_path,
                checkpoints=state_data.get('checkpoints', []),
                last_checkpoint=state_data.get('last_checkpoint'),
                recovery_point=state_data.get('recovery_point'),
                state_data=state_data.get('state_data', {}),
                context=state_data.get('context', {}),
                errors=state_data.get('errors', []),
                retry_count=state_data.get('retry_count', 0),
                max_retries=state_data.get('max_retries', 3),
                created_at=created_at,
                updated_at=updated_at,
                metadata=state_data.get('metadata', {})
            )
            
        return state
        
    def _reconstruct_state_from_checkpoint(
        self,
        checkpoint: WorkflowCheckpoint
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Reconstruct workflow state from a checkpoint."""
        return self._reconstruct_state_from_storage(
            checkpoint.state_data,
            checkpoint.campaign_id,
            checkpoint.workflow_id,
            WorkflowStatus.RUNNING.value,  # Set to running for restoration
            checkpoint.node_id,
            checkpoint.execution_path,
            checkpoint.timestamp,
            datetime.now()
        )
        
    async def _record_state_transition(
        self,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        transition_type: StateTransitionType,
        from_state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a state transition in the database."""
        try:
            transition_id = f"{state['campaign_id']}_{state['workflow_id']}_{transition_type.value}_{int(datetime.now().timestamp())}"
            
            async with self.db_service.get_connection() as conn:
                query = """
                INSERT INTO campaign_state_transitions 
                (transition_id, campaign_id, workflow_id, transition_type, 
                 from_state, to_state, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """
                
                await conn.execute(
                    query,
                    transition_id,
                    state['campaign_id'],
                    state['workflow_id'],
                    transition_type.value,
                    from_state,
                    state.get('current_node', ''),
                    datetime.now(),
                    json.dumps(metadata or {})
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to record state transition: {str(e)}")
            return False