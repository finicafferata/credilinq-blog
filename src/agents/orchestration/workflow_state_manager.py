"""
Workflow State Manager - Manages state persistence and recovery for campaign workflows.

This module provides comprehensive state management for campaign workflows,
including persistence, recovery, and cleanup operations.
"""

import json
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CampaignWorkflowState:
    """
    Complete state representation for a campaign workflow execution.
    
    This class encapsulates all the information needed to persist and recover
    a workflow execution, including task progress, agent assignments, and metadata.
    """
    campaign_id: str
    workflow_execution_id: str
    current_task_id: Optional[str] = None
    completed_tasks: List[str] = None
    failed_tasks: List[str] = None
    agent_results: Dict[str, Any] = None
    workflow_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.completed_tasks is None:
            self.completed_tasks = []
        if self.failed_tasks is None:
            self.failed_tasks = []
        if self.agent_results is None:
            self.agent_results = {}
        if self.workflow_metadata is None:
            self.workflow_metadata = {}

class WorkflowStateManager:
    """
    Manages state persistence and recovery for campaign workflows.
    
    Provides functionality to save workflow states to the database for recovery,
    load states for workflow resumption, and cleanup old workflow states.
    """
    
    def __init__(self):
        """Initialize the workflow state manager."""
        # Import here to avoid circular dependencies
        from .campaign_database_service import CampaignDatabaseService
        self.db_service = CampaignDatabaseService()
        
        # State caching for performance
        self._state_cache: Dict[str, CampaignWorkflowState] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=30)
        
        logger.info("Initialized WorkflowStateManager with database persistence")
    
    async def save_workflow_state(self, state: CampaignWorkflowState) -> bool:
        """
        Persist workflow state to database for recovery.
        
        Args:
            state: Workflow state to persist
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Convert state to dictionary for JSON serialization
            state_data = asdict(state)
            
            # Ensure datetime objects are serialized properly
            state_data["last_updated"] = datetime.utcnow().isoformat()
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Check if state record already exists
                cur.execute("""
                    SELECT id FROM campaign_workflow_states 
                    WHERE workflow_execution_id = %s
                """, (state.workflow_execution_id,))
                
                existing_record = cur.fetchone()
                
                if existing_record:
                    # Update existing state
                    cur.execute("""
                        UPDATE campaign_workflow_states 
                        SET state_data = %s, updated_at = %s
                        WHERE workflow_execution_id = %s
                    """, (
                        json.dumps(state_data),
                        datetime.utcnow(),
                        state.workflow_execution_id
                    ))
                    logger.info(f"Updated workflow state for execution {state.workflow_execution_id}")
                else:
                    # Create new state record
                    state_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO campaign_workflow_states (
                            id, workflow_execution_id, campaign_id, 
                            state_data, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        state_id,
                        state.workflow_execution_id,
                        state.campaign_id,
                        json.dumps(state_data),
                        datetime.utcnow(),
                        datetime.utcnow()
                    ))
                    logger.info(f"Created new workflow state for execution {state.workflow_execution_id}")
                
                # Update cache
                self._state_cache[state.workflow_execution_id] = state
                self._cache_ttl[state.workflow_execution_id] = datetime.utcnow() + self._cache_duration
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save workflow state for {state.workflow_execution_id}: {str(e)}")
            
            # Fallback: try to save to temporary file
            try:
                import tempfile
                import os
                
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f"workflow_state_{state.workflow_execution_id}.json")
                
                with open(temp_file, 'w') as f:
                    json.dump(asdict(state), f, default=str, indent=2)
                
                logger.warning(f"Saved workflow state to temporary file: {temp_file}")
                return True
                
            except Exception as temp_error:
                logger.error(f"Failed to save to temporary file: {str(temp_error)}")
                return False
    
    async def load_workflow_state(self, workflow_execution_id: str) -> CampaignWorkflowState:
        """
        Load workflow state for resumption.
        
        Args:
            workflow_execution_id: ID of the workflow execution to load state for
            
        Returns:
            CampaignWorkflowState: Loaded workflow state
            
        Raises:
            ValueError: If workflow state not found
        """
        try:
            # Check cache first
            if workflow_execution_id in self._state_cache:
                cache_time = self._cache_ttl.get(workflow_execution_id, datetime.min)
                if datetime.utcnow() < cache_time:
                    logger.info(f"Loaded workflow state from cache for {workflow_execution_id}")
                    return self._state_cache[workflow_execution_id]
                else:
                    # Remove expired cache entry
                    del self._state_cache[workflow_execution_id]
                    del self._cache_ttl[workflow_execution_id]
            
            # Load from database
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT state_data, campaign_id, updated_at
                    FROM campaign_workflow_states 
                    WHERE workflow_execution_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (workflow_execution_id,))
                
                record = cur.fetchone()
                
                if not record:
                    # Try to load from temporary file as fallback
                    state = await self._load_from_temp_file(workflow_execution_id)
                    if state:
                        return state
                    
                    raise ValueError(f"Workflow state not found for execution {workflow_execution_id}")
                
                # Parse state data
                state_data = json.loads(record[0]) if isinstance(record[0], str) else record[0]
                
                # Reconstruct CampaignWorkflowState object
                state = CampaignWorkflowState(
                    campaign_id=state_data.get("campaign_id", record[1]),
                    workflow_execution_id=workflow_execution_id,
                    current_task_id=state_data.get("current_task_id"),
                    completed_tasks=state_data.get("completed_tasks", []),
                    failed_tasks=state_data.get("failed_tasks", []),
                    agent_results=state_data.get("agent_results", {}),
                    workflow_metadata=state_data.get("workflow_metadata", {})
                )
                
                # Update cache
                self._state_cache[workflow_execution_id] = state
                self._cache_ttl[workflow_execution_id] = datetime.utcnow() + self._cache_duration
                
                logger.info(f"Loaded workflow state from database for {workflow_execution_id}")
                return state
                
        except Exception as e:
            logger.error(f"Failed to load workflow state for {workflow_execution_id}: {str(e)}")
            raise
    
    async def cleanup_completed_workflows(self, retention_days: int = 30) -> int:
        """
        Clean up old workflow states to free up storage.
        
        Args:
            retention_days: Number of days to retain completed workflow states
            
        Returns:
            int: Number of workflow states cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Find workflow states to clean up
                cur.execute("""
                    SELECT workflow_execution_id 
                    FROM campaign_workflow_states 
                    WHERE updated_at < %s
                """, (cutoff_date,))
                
                workflow_ids = [row[0] for row in cur.fetchall()]
                
                if not workflow_ids:
                    logger.info("No old workflow states found for cleanup")
                    return 0
                
                # Delete old workflow states
                cur.execute("""
                    DELETE FROM campaign_workflow_states 
                    WHERE updated_at < %s
                """, (cutoff_date,))
                
                deleted_count = cur.rowcount
                
                # Clean up cache entries
                for workflow_id in workflow_ids:
                    if workflow_id in self._state_cache:
                        del self._state_cache[workflow_id]
                    if workflow_id in self._cache_ttl:
                        del self._cache_ttl[workflow_id]
                
                # Clean up temporary files
                await self._cleanup_temp_files(retention_days)
                
                logger.info(f"Cleaned up {deleted_count} old workflow states")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup workflow states: {str(e)}")
            return 0
    
    async def get_workflow_state_summary(self, workflow_execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of workflow state without loading full state.
        
        Args:
            workflow_execution_id: ID of the workflow execution
            
        Returns:
            Optional[Dict]: Summary of workflow state, or None if not found
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT campaign_id, created_at, updated_at,
                           jsonb_array_length(state_data->'completed_tasks') as completed_count,
                           jsonb_array_length(state_data->'failed_tasks') as failed_count
                    FROM campaign_workflow_states 
                    WHERE workflow_execution_id = %s
                """, (workflow_execution_id,))
                
                record = cur.fetchone()
                
                if not record:
                    return None
                
                return {
                    "workflow_execution_id": workflow_execution_id,
                    "campaign_id": record[0],
                    "created_at": record[1].isoformat() if record[1] else None,
                    "updated_at": record[2].isoformat() if record[2] else None,
                    "completed_tasks_count": record[3] or 0,
                    "failed_tasks_count": record[4] or 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get workflow state summary for {workflow_execution_id}: {str(e)}")
            return None
    
    async def list_active_workflows(self) -> List[Dict[str, Any]]:
        """
        List all currently active workflow executions.
        
        Returns:
            List[Dict]: Summary of active workflows
        """
        try:
            # Define active cutoff (workflows updated in last 24 hours)
            active_cutoff = datetime.utcnow() - timedelta(hours=24)
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT workflow_execution_id, campaign_id, created_at, updated_at
                    FROM campaign_workflow_states 
                    WHERE updated_at > %s
                    ORDER BY updated_at DESC
                """, (active_cutoff,))
                
                active_workflows = []
                for record in cur.fetchall():
                    workflow_summary = await self.get_workflow_state_summary(record[0])
                    if workflow_summary:
                        active_workflows.append(workflow_summary)
                
                logger.info(f"Found {len(active_workflows)} active workflows")
                return active_workflows
                
        except Exception as e:
            logger.error(f"Failed to list active workflows: {str(e)}")
            return []
    
    async def _load_from_temp_file(self, workflow_execution_id: str) -> Optional[CampaignWorkflowState]:
        """Load workflow state from temporary file as fallback."""
        try:
            import tempfile
            import os
            
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"workflow_state_{workflow_execution_id}.json")
            
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    state_data = json.load(f)
                
                state = CampaignWorkflowState(
                    campaign_id=state_data.get("campaign_id"),
                    workflow_execution_id=workflow_execution_id,
                    current_task_id=state_data.get("current_task_id"),
                    completed_tasks=state_data.get("completed_tasks", []),
                    failed_tasks=state_data.get("failed_tasks", []),
                    agent_results=state_data.get("agent_results", {}),
                    workflow_metadata=state_data.get("workflow_metadata", {})
                )
                
                logger.info(f"Loaded workflow state from temporary file for {workflow_execution_id}")
                return state
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load from temporary file: {str(e)}")
            return None
    
    async def _cleanup_temp_files(self, retention_days: int):
        """Clean up old temporary workflow state files."""
        try:
            import tempfile
            import os
            import glob
            
            temp_dir = tempfile.gettempdir()
            pattern = os.path.join(temp_dir, "workflow_state_*.json")
            
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            for temp_file in glob.glob(pattern):
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(temp_file))
                    if file_time < cutoff_time:
                        os.remove(temp_file)
                        logger.debug(f"Removed old temporary file: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {str(cleanup_error)}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup temporary files: {str(e)}")
    
    def clear_cache(self):
        """Clear the workflow state cache."""
        self._state_cache.clear()
        self._cache_ttl.clear()
        logger.info("Cleared workflow state cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the workflow state cache."""
        current_time = datetime.utcnow()
        valid_entries = sum(1 for ttl in self._cache_ttl.values() if ttl > current_time)
        
        return {
            "total_cached_states": len(self._state_cache),
            "valid_cached_states": valid_entries,
            "expired_cached_states": len(self._state_cache) - valid_entries,
            "cache_duration_minutes": self._cache_duration.total_seconds() / 60
        }