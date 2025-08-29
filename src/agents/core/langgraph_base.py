"""
LangGraph foundation infrastructure for CrediLinq Content Agent.

This module provides the core LangGraph utilities, state management classes, 
workflow checkpointing, and recovery mechanisms for the migration from LangChain to LangGraph.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
import logging
import uuid
import asyncio
from contextlib import asynccontextmanager

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

# Try to import PostgreSQLSaver, but make it optional
try:
    from langgraph.checkpoint.postgres import PostgreSQLSaver
    POSTGRES_CHECKPOINT_AVAILABLE = True
except ImportError:
    POSTGRES_CHECKPOINT_AVAILABLE = False
    PostgreSQLSaver = None

# Import existing base types
from .base_agent import (
    BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, 
    AgentType, AgentStatus, AgentState
)

logger = logging.getLogger(__name__)

# Type definitions for LangGraph
T = TypeVar('T')
StateT = TypeVar('StateT', bound=Dict[str, Any])

class WorkflowStatus(Enum):
    """Workflow execution status for LangGraph workflows."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RECOVERED = "recovered"

class CheckpointStrategy(Enum):
    """Strategy for workflow checkpointing."""
    MEMORY_ONLY = "memory_only"
    DATABASE_PERSISTENT = "database_persistent"
    HYBRID = "hybrid"

@dataclass
class WorkflowState:
    """Base workflow state that all LangGraph workflows should extend."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow state to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create workflow state from dictionary."""
        # Convert datetime strings back to datetime objects
        for field_name in ['started_at', 'updated_at', 'completed_at']:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)

@dataclass
class LangGraphExecutionContext:
    """Enhanced execution context for LangGraph workflows."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    checkpoint_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_workflow_id: Optional[str] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT
    
    def to_agent_context(self) -> AgentExecutionContext:
        """Convert to legacy AgentExecutionContext for backward compatibility."""
        return AgentExecutionContext(
            request_id=self.workflow_id,
            user_id=self.user_id,
            session_id=self.session_id,
            workflow_id=self.workflow_id,
            execution_metadata=self.execution_metadata
        )

class StateManagerProtocol(ABC):
    """Protocol for managing workflow state persistence."""
    
    @abstractmethod
    async def save_state(self, workflow_id: str, state: Dict[str, Any]) -> bool:
        """Save workflow state to persistent storage."""
        pass
    
    @abstractmethod
    async def load_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state from persistent storage."""
        pass
    
    @abstractmethod
    async def delete_state(self, workflow_id: str) -> bool:
        """Delete workflow state from persistent storage."""
        pass
    
    @abstractmethod
    async def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[str]:
        """List workflow IDs by status."""
        pass

class DatabaseStateManager(StateManagerProtocol):
    """Production-ready database-backed state manager for workflow persistence."""
    
    def __init__(self, database_url: Optional[str] = None):
        import os
        self.database_url = database_url or os.getenv("DATABASE_URL_DIRECT", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
        self._connection_pool = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for better performance."""
        try:
            import psycopg2.pool
            from psycopg2.extras import RealDictCursor
            
            self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.database_url,
                cursor_factory=RealDictCursor
            )
            self.logger.info("Database connection pool initialized for LangGraph state management")
        except Exception as e:
            self.logger.warning(f"Failed to initialize connection pool: {e}. Will use direct connections.")
            self._connection_pool = None
    
    @asynccontextmanager
    async def _get_db_connection(self):
        """Get database connection with proper async handling."""
        conn = None
        try:
            if self._connection_pool:
                conn = self._connection_pool.getconn()
            else:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
            
            conn.autocommit = True
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                if self._connection_pool:
                    self._connection_pool.putconn(conn)
                else:
                    conn.close()
    
    async def save_state(self, workflow_id: str, state: Dict[str, Any]) -> bool:
        """Save workflow state to PostgreSQL database."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # First, try to update existing workflow
                update_query = """
                    UPDATE langgraph_workflows 
                    SET workflow_state = %s,
                        status = %s,
                        current_step = %s,
                        updated_at = NOW()
                    WHERE workflow_id = %s
                    RETURNING id;
                """
                
                workflow_status = state.get('status', 'running')
                current_step = state.get('current_step')
                state_json = json.dumps(state, default=str)
                
                cursor.execute(update_query, (state_json, workflow_status, current_step, workflow_id))
                result = cursor.fetchone()
                
                if result:
                    self.logger.debug(f"Updated existing workflow state for {workflow_id}")
                else:
                    # Insert new workflow if it doesn't exist
                    insert_query = """
                        INSERT INTO langgraph_workflows (
                            workflow_id, workflow_name, status, current_step,
                            workflow_state, started_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (workflow_id) DO UPDATE SET
                            workflow_state = EXCLUDED.workflow_state,
                            status = EXCLUDED.status,
                            current_step = EXCLUDED.current_step,
                            updated_at = NOW();
                    """
                    
                    workflow_name = state.get('workflow_name', 'unknown')
                    cursor.execute(insert_query, (workflow_id, workflow_name, workflow_status, current_step, state_json))
                    self.logger.debug(f"Created new workflow state for {workflow_id}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save state for {workflow_id}: {e}")
            return False
    
    async def load_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state from PostgreSQL database."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT workflow_state, status, current_step, started_at, updated_at,
                           completed_at, error_message, retry_count, max_retries
                    FROM langgraph_workflows 
                    WHERE workflow_id = %s;
                """
                
                cursor.execute(query, (workflow_id,))
                result = cursor.fetchone()
                
                if result:
                    workflow_state = result['workflow_state']
                    if isinstance(workflow_state, str):
                        workflow_state = json.loads(workflow_state)
                    
                    # Merge with workflow metadata
                    full_state = {
                        **workflow_state,
                        'workflow_id': workflow_id,
                        'status': result['status'],
                        'current_step': result['current_step'],
                        'started_at': result['started_at'].isoformat() if result['started_at'] else None,
                        'updated_at': result['updated_at'].isoformat() if result['updated_at'] else None,
                        'completed_at': result['completed_at'].isoformat() if result['completed_at'] else None,
                        'error_message': result['error_message'],
                        'retry_count': result['retry_count'],
                        'max_retries': result['max_retries']
                    }
                    
                    self.logger.debug(f"Loaded workflow state for {workflow_id}")
                    return full_state
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load state for {workflow_id}: {e}")
            return None
    
    async def delete_state(self, workflow_id: str) -> bool:
        """Delete workflow state from PostgreSQL database."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Delete related records first (checkpoints, agent executions)
                delete_checkpoints = "DELETE FROM langgraph_checkpoints WHERE workflow_id = %s;"
                delete_executions = "DELETE FROM langgraph_agent_executions WHERE workflow_id = %s;"
                delete_workflow = "DELETE FROM langgraph_workflows WHERE workflow_id = %s;"
                
                cursor.execute(delete_checkpoints, (workflow_id,))
                cursor.execute(delete_executions, (workflow_id,))
                cursor.execute(delete_workflow, (workflow_id,))
                
                self.logger.info(f"Deleted workflow state and related data for {workflow_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete state for {workflow_id}: {e}")
            return False
    
    async def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[str]:
        """List workflow IDs by status from PostgreSQL database."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                if status:
                    query = "SELECT workflow_id FROM langgraph_workflows WHERE status = %s ORDER BY updated_at DESC;"
                    cursor.execute(query, (status.value,))
                else:
                    query = "SELECT workflow_id FROM langgraph_workflows ORDER BY updated_at DESC;"
                    cursor.execute(query)
                
                results = cursor.fetchall()
                workflow_ids = [row['workflow_id'] for row in results]
                
                self.logger.debug(f"Listed {len(workflow_ids)} workflows with status {status}")
                return workflow_ids
                
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
            return []
    
    async def create_checkpoint(self, workflow_id: str, step_name: str, step_index: int, 
                               state: Dict[str, Any], checkpoint_id: Optional[str] = None) -> str:
        """Create a checkpoint for a workflow step."""
        if not checkpoint_id:
            checkpoint_id = f"{workflow_id}_{step_name}_{step_index}_{int(time.time())}"
        
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                insert_query = """
                    INSERT INTO langgraph_checkpoints (
                        workflow_id, checkpoint_id, step_name, step_index, state, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (workflow_id, checkpoint_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW();
                """
                
                state_json = json.dumps(state, default=str)
                metadata = {'created_at': datetime.utcnow().isoformat(), 'step_name': step_name}
                metadata_json = json.dumps(metadata)
                
                cursor.execute(insert_query, (workflow_id, checkpoint_id, step_name, step_index, state_json, metadata_json))
                
                self.logger.debug(f"Created checkpoint {checkpoint_id} for workflow {workflow_id} at step {step_name}")
                return checkpoint_id
                
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    async def load_checkpoint(self, workflow_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT state, step_name, step_index, metadata, created_at
                    FROM langgraph_checkpoints 
                    WHERE workflow_id = %s AND checkpoint_id = %s;
                """
                
                cursor.execute(query, (workflow_id, checkpoint_id))
                result = cursor.fetchone()
                
                if result:
                    state = json.loads(result['state']) if isinstance(result['state'], str) else result['state']
                    return {
                        'state': state,
                        'step_name': result['step_name'],
                        'step_index': result['step_index'],
                        'checkpoint_id': checkpoint_id,
                        'metadata': json.loads(result['metadata']) if result['metadata'] else {},
                        'created_at': result['created_at'].isoformat() if result['created_at'] else None
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    async def list_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a workflow."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT checkpoint_id, step_name, step_index, created_at, is_recoverable
                    FROM langgraph_checkpoints 
                    WHERE workflow_id = %s 
                    ORDER BY step_index ASC, created_at ASC;
                """
                
                cursor.execute(query, (workflow_id,))
                results = cursor.fetchall()
                
                checkpoints = []
                for row in results:
                    checkpoints.append({
                        'checkpoint_id': row['checkpoint_id'],
                        'step_name': row['step_name'],
                        'step_index': row['step_index'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'is_recoverable': row['is_recoverable']
                    })
                
                return checkpoints
                
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints for {workflow_id}: {e}")
            return []
    
    async def create_state_snapshot(self, workflow_id: str, snapshot_type: str = "manual", 
                                   description: Optional[str] = None, 
                                   tags: Optional[List[str]] = None) -> str:
        """Create a state snapshot for backup and versioning."""
        snapshot_id = f"snapshot_{workflow_id}_{int(time.time())}"
        
        try:
            # Load current workflow state
            current_state = await self.load_state(workflow_id)
            if not current_state:
                raise ValueError(f"No state found for workflow {workflow_id}")
            
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get individual step and agent states
                step_states_query = "SELECT step_name, agent_state FROM langgraph_agent_executions WHERE workflow_id = %s;"
                cursor.execute(step_states_query, (workflow_id,))
                step_results = cursor.fetchall()
                
                step_states = {}
                agent_states = {}
                for row in step_results:
                    step_states[row['step_name']] = row.get('agent_state', {})
                    if row.get('agent_state'):
                        agent_states[row['step_name']] = row['agent_state']
                
                # Create snapshot record
                insert_query = """
                    INSERT INTO workflow_state_snapshots (
                        workflow_id, snapshot_id, snapshot_type, full_state,
                        step_states, agent_states, description, tags, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                full_state_json = json.dumps(current_state, default=str)
                step_states_json = json.dumps(step_states, default=str) if step_states else None
                agent_states_json = json.dumps(agent_states, default=str) if agent_states else None
                expires_at = datetime.utcnow() + timedelta(days=30)  # 30-day retention
                
                cursor.execute(insert_query, (
                    workflow_id, snapshot_id, snapshot_type, full_state_json,
                    step_states_json, agent_states_json, description, 
                    tags or [], expires_at
                ))
                
                self.logger.info(f"Created state snapshot {snapshot_id} for workflow {workflow_id}")
                return snapshot_id
                
        except Exception as e:
            self.logger.error(f"Failed to create state snapshot: {e}")
            raise
    
    async def restore_from_snapshot(self, workflow_id: str, snapshot_id: str) -> bool:
        """Restore workflow state from a snapshot."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Load snapshot
                query = "SELECT full_state FROM workflow_state_snapshots WHERE workflow_id = %s AND snapshot_id = %s;"
                cursor.execute(query, (workflow_id, snapshot_id))
                result = cursor.fetchone()
                
                if not result:
                    self.logger.error(f"Snapshot {snapshot_id} not found")
                    return False
                
                full_state = json.loads(result['full_state'])
                
                # Restore state
                success = await self.save_state(workflow_id, full_state)
                if success:
                    self.logger.info(f"Restored workflow {workflow_id} from snapshot {snapshot_id}")
                    
                    # Mark workflow as recovered
                    update_query = """
                        UPDATE langgraph_workflows 
                        SET status = 'recovered' 
                        WHERE workflow_id = %s;
                    """
                    cursor.execute(update_query, (workflow_id,))
                
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to restore from snapshot {snapshot_id}: {e}")
            return False
    
    async def cleanup_expired_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up expired workflows, checkpoints, and snapshots."""
        cleanup_stats = {'workflows': 0, 'checkpoints': 0, 'snapshots': 0, 'agent_executions': 0}
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up completed workflows older than cutoff
                cursor.execute(
                    "DELETE FROM langgraph_workflows WHERE status IN ('completed', 'failed', 'cancelled') AND updated_at < %s;",
                    (cutoff_date,)
                )
                cleanup_stats['workflows'] = cursor.rowcount
                
                # Clean up old checkpoints for non-recoverable states
                cursor.execute(
                    "DELETE FROM langgraph_checkpoints WHERE created_at < %s AND is_recoverable = false;",
                    (cutoff_date,)
                )
                cleanup_stats['checkpoints'] = cursor.rowcount
                
                # Clean up expired snapshots
                cursor.execute(
                    "DELETE FROM workflow_state_snapshots WHERE expires_at < NOW();"
                )
                cleanup_stats['snapshots'] = cursor.rowcount
                
                # Clean up old agent execution records
                cursor.execute(
                    "DELETE FROM langgraph_agent_executions WHERE created_at < %s AND is_recoverable = false;",
                    (cutoff_date,)
                )
                cleanup_stats['agent_executions'] = cursor.rowcount
                
                self.logger.info(f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired data: {e}")
            return cleanup_stats
    
    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a workflow."""
        try:
            async with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get workflow info
                workflow_query = """
                    SELECT workflow_name, status, started_at, updated_at, completed_at,
                           retry_count, max_retries, error_message
                    FROM langgraph_workflows WHERE workflow_id = %s;
                """
                cursor.execute(workflow_query, (workflow_id,))
                workflow_info = cursor.fetchone()
                
                if not workflow_info:
                    return {}
                
                # Get agent execution metrics
                execution_query = """
                    SELECT agent_name, status, duration, input_tokens, output_tokens, 
                           estimated_cost_usd, output_quality_score
                    FROM langgraph_agent_executions WHERE workflow_id = %s;
                """
                cursor.execute(execution_query, (workflow_id,))
                executions = cursor.fetchall()
                
                # Calculate metrics
                total_duration = sum(ex.get('duration', 0) or 0 for ex in executions)
                total_cost = sum(ex.get('estimated_cost_usd', 0) or 0 for ex in executions)
                total_tokens = sum((ex.get('input_tokens', 0) or 0) + (ex.get('output_tokens', 0) or 0) for ex in executions)
                avg_quality = sum(ex.get('output_quality_score', 0) or 0 for ex in executions) / max(len(executions), 1)
                
                success_count = sum(1 for ex in executions if ex.get('status') == 'success')
                success_rate = success_count / max(len(executions), 1)
                
                return {
                    'workflow_id': workflow_id,
                    'workflow_name': workflow_info['workflow_name'],
                    'status': workflow_info['status'],
                    'started_at': workflow_info['started_at'].isoformat() if workflow_info['started_at'] else None,
                    'updated_at': workflow_info['updated_at'].isoformat() if workflow_info['updated_at'] else None,
                    'completed_at': workflow_info['completed_at'].isoformat() if workflow_info['completed_at'] else None,
                    'total_duration_ms': total_duration,
                    'total_cost_usd': total_cost,
                    'total_tokens': total_tokens,
                    'average_quality_score': round(avg_quality, 3),
                    'success_rate': round(success_rate, 3),
                    'agent_executions': len(executions),
                    'retry_count': workflow_info['retry_count'],
                    'error_message': workflow_info['error_message']
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow metrics: {e}")
            return {}
    
    def close(self):
        """Clean up resources."""
        if self._connection_pool:
            self._connection_pool.closeall()
            self.logger.info("Closed database connection pool")

class PostgreSQLStateCheckpointer:
    """
    Custom PostgreSQL checkpointer that integrates with DatabaseStateManager.
    This provides more control over checkpointing behavior than the default PostgreSQLSaver.
    """
    
    def __init__(self, state_manager: DatabaseStateManager):
        self.state_manager = state_manager
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    async def put(self, config: dict, checkpoint: dict, metadata: dict = None) -> dict:
        """Save a checkpoint to the database."""
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id", str(uuid.uuid4()))
            
            if not thread_id:
                raise ValueError("thread_id is required in config")
            
            # Extract step information from checkpoint metadata
            step_name = metadata.get("step", "unknown") if metadata else "unknown"
            step_index = metadata.get("step_index", 0) if metadata else 0
            
            # Save checkpoint using our DatabaseStateManager
            await self.state_manager.create_checkpoint(
                workflow_id=thread_id,
                step_name=step_name,
                step_index=step_index,
                state=checkpoint,
                checkpoint_id=checkpoint_id
            )
            
            # Return the config with the checkpoint_id
            return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": checkpoint_id}}
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def get(self, config: dict) -> Optional[dict]:
        """Retrieve a checkpoint from the database."""
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            
            if not thread_id:
                return None
            
            if checkpoint_id:
                # Load specific checkpoint
                checkpoint_data = await self.state_manager.load_checkpoint(thread_id, checkpoint_id)
                if checkpoint_data:
                    return checkpoint_data.get("state")
            else:
                # Load latest workflow state
                state = await self.state_manager.load_state(thread_id)
                return state
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve checkpoint: {e}")
            return None
    
    async def list(self, config: dict) -> List[dict]:
        """List available checkpoints for a workflow."""
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return []
            
            checkpoints = await self.state_manager.list_checkpoints(thread_id)
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []

class LangGraphWorkflowBase(ABC, Generic[StateT]):
    """
    Base class for LangGraph workflows with state management and recovery.
    """
    
    def __init__(
        self,
        workflow_name: str,
        state_manager: Optional[StateManagerProtocol] = None,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        max_retries: int = 3
    ):
        self.workflow_name = workflow_name
        self.state_manager = state_manager or DatabaseStateManager()
        self.checkpoint_strategy = checkpoint_strategy
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize workflow graph
        self._graph = None
        self._checkpointer = None
        self._initialize_workflow()
    
    def _initialize_workflow(self):
        """Initialize the LangGraph workflow structure."""
        try:
            # Initialize checkpointer based on strategy
            if self.checkpoint_strategy == CheckpointStrategy.MEMORY_ONLY:
                self._checkpointer = MemorySaver()
            elif self.checkpoint_strategy == CheckpointStrategy.DATABASE_PERSISTENT:
                # Use our custom PostgreSQL checkpointer that integrates with DatabaseStateManager
                try:
                    self._checkpointer = PostgreSQLStateCheckpointer(self.state_manager)
                    self.logger.info("Using custom PostgreSQL checkpointer for state persistence")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize custom PostgreSQL checkpointer: {e}. Using MemorySaver as fallback.")
                    self._checkpointer = MemorySaver()
            elif self.checkpoint_strategy == CheckpointStrategy.HYBRID:
                # For hybrid approach, use database state manager with memory checkpointer for fast access
                self._checkpointer = MemorySaver()
            
            # Create the workflow graph
            self._graph = self._create_workflow_graph()
            
            self.logger.info(f"Initialized LangGraph workflow: {self.workflow_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow {self.workflow_name}: {e}")
            raise
    
    @abstractmethod
    def _create_workflow_graph(self) -> StateGraph:
        """Create and configure the LangGraph workflow structure."""
        pass
    
    @abstractmethod
    def _create_initial_state(self, input_data: Dict[str, Any]) -> StateT:
        """Create the initial state for the workflow."""
        pass
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """
        Execute the LangGraph workflow with full state management.
        
        Args:
            input_data: Input data for the workflow
            context: Execution context for tracking and recovery
            
        Returns:
            AgentResult: Workflow execution result
        """
        if context is None:
            context = LangGraphExecutionContext()
        
        workflow_id = context.workflow_id
        
        try:
            self.logger.info(f"Starting workflow {self.workflow_name} with ID {workflow_id}")
            
            # Check if this is a recovery from previous execution
            existing_state = await self._try_recover_workflow(workflow_id)
            
            if existing_state:
                self.logger.info(f"Recovering workflow {workflow_id} from checkpoint")
                initial_state = existing_state
            else:
                # Create new initial state
                initial_state = self._create_initial_state(input_data)
                initial_state['workflow_context'] = context.to_dict() if hasattr(context, 'to_dict') else context.__dict__
            
            # Execute the workflow
            result = await self._execute_with_recovery(initial_state, context)
            
            # Save final state
            await self.state_manager.save_state(workflow_id, result)
            
            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "workflow_id": workflow_id,
                    "workflow_name": self.workflow_name,
                    "execution_type": "langgraph_workflow"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            
            # Save error state
            error_state = {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.FAILED.value,
                "error_message": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
            await self.state_manager.save_state(workflow_id, error_state)
            
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="LANGGRAPH_WORKFLOW_FAILED",
                metadata={
                    "workflow_id": workflow_id,
                    "workflow_name": self.workflow_name
                }
            )
    
    async def _execute_with_recovery(
        self,
        initial_state: StateT,
        context: LangGraphExecutionContext
    ) -> Dict[str, Any]:
        """Execute workflow with recovery capabilities."""
        app = self._graph.compile(checkpointer=self._checkpointer)
        
        # Configure execution
        config = {
            "configurable": {
                "thread_id": context.workflow_id,
                "checkpoint_id": context.checkpoint_id
            }
        }
        
        # Execute the workflow
        final_state = None
        async for state in app.astream(initial_state, config=config):
            # Save intermediate state periodically
            await self.state_manager.save_state(context.workflow_id, state)
            final_state = state
        
        return final_state
    
    async def _try_recover_workflow(self, workflow_id: str) -> Optional[StateT]:
        """Try to recover workflow from saved state."""
        try:
            saved_state = await self.state_manager.load_state(workflow_id)
            if saved_state and saved_state.get('status') != WorkflowStatus.COMPLETED.value:
                self.logger.info(f"Found recoverable state for workflow {workflow_id}")
                return saved_state
        except Exception as e:
            self.logger.warning(f"Failed to recover workflow {workflow_id}: {e}")
        
        return None
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        try:
            state = await self.state_manager.load_state(workflow_id)
            if state:
                state['status'] = WorkflowStatus.PAUSED.value
                state['paused_at'] = datetime.utcnow().isoformat()
                return await self.state_manager.save_state(workflow_id, state)
        except Exception as e:
            self.logger.error(f"Failed to pause workflow {workflow_id}: {e}")
        
        return False
    
    async def resume_workflow(self, workflow_id: str) -> AgentResult:
        """Resume a paused workflow."""
        try:
            state = await self.state_manager.load_state(workflow_id)
            if state and state.get('status') == WorkflowStatus.PAUSED.value:
                state['status'] = WorkflowStatus.RUNNING.value
                state['resumed_at'] = datetime.utcnow().isoformat()
                
                # Create context from saved state
                context = LangGraphExecutionContext(workflow_id=workflow_id)
                
                # Resume execution
                return await self._execute_with_recovery(state, context)
        except Exception as e:
            self.logger.error(f"Failed to resume workflow {workflow_id}: {e}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="WORKFLOW_RESUME_FAILED"
            )
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about this workflow."""
        graph_nodes = []
        graph_edges = []
        
        if self._graph:
            graph_nodes = list(self._graph.nodes.keys()) if hasattr(self._graph, 'nodes') else []
            # Handle edges - they might be a set or dict depending on LangGraph version
            if hasattr(self._graph, 'edges'):
                edges = self._graph.edges
                if isinstance(edges, dict):
                    graph_edges = list(edges.keys())
                elif hasattr(edges, '__iter__'):
                    graph_edges = list(edges)
        
        return {
            "workflow_name": self.workflow_name,
            "checkpoint_strategy": self.checkpoint_strategy.value,
            "max_retries": self.max_retries,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges
        }

class LangGraphAgentMixin:
    """
    Mixin class to add LangGraph capabilities to existing BaseAgent implementations.
    This enables gradual migration from LangChain to LangGraph.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._langgraph_enabled = False
        self._workflow: Optional[LangGraphWorkflowBase] = None
    
    def enable_langgraph(
        self,
        workflow: LangGraphWorkflowBase,
        fallback_to_langchain: bool = True
    ):
        """
        Enable LangGraph workflow for this agent.
        
        Args:
            workflow: LangGraph workflow to use
            fallback_to_langchain: Whether to fallback to LangChain on LangGraph failure
        """
        self._workflow = workflow
        self._langgraph_enabled = True
        self._fallback_to_langchain = fallback_to_langchain
        self.logger.info(f"LangGraph enabled for {self.__class__.__name__}")
    
    async def execute_langgraph(
        self,
        input_data: Any,
        context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute using LangGraph workflow if enabled."""
        if not self._langgraph_enabled or not self._workflow:
            raise RuntimeError("LangGraph not enabled for this agent")
        
        try:
            return await self._workflow.execute(input_data, context)
        except Exception as e:
            self.logger.error(f"LangGraph execution failed: {e}")
            
            # Fallback to LangChain if enabled
            if getattr(self, '_fallback_to_langchain', False):
                self.logger.info("Falling back to LangChain execution")
                # Convert LangGraphExecutionContext to AgentExecutionContext
                agent_context = context.to_agent_context() if context else None
                return self.execute_safe(input_data, agent_context)
            
            raise

def create_hybrid_agent(
    agent_class: type,
    workflow_class: Optional[type] = None,
    enable_langgraph: bool = True,
    **agent_kwargs
) -> BaseAgent:
    """
    Factory function to create agents with hybrid LangChain/LangGraph support.
    
    Args:
        agent_class: The base agent class to create
        workflow_class: Optional LangGraph workflow class
        enable_langgraph: Whether to enable LangGraph capabilities
        **agent_kwargs: Additional arguments for agent creation
        
    Returns:
        BaseAgent: Agent with hybrid capabilities
    """
    
    # Create a hybrid class that includes the LangGraph mixin
    class HybridAgent(LangGraphAgentMixin, agent_class):
        pass
    
    # Create the agent instance
    agent = HybridAgent(**agent_kwargs)
    
    # Enable LangGraph if requested and workflow provided
    if enable_langgraph and workflow_class:
        workflow = workflow_class(workflow_name=f"{agent_class.__name__}_workflow")
        agent.enable_langgraph(workflow)
    
    return agent

# Utility functions for workflow state management
async def cleanup_old_workflows(
    state_manager: StateManagerProtocol,
    max_age_days: int = 7,
    statuses: Optional[List[WorkflowStatus]] = None
) -> int:
    """
    Clean up old workflow states from storage.
    
    Args:
        state_manager: State manager instance
        max_age_days: Maximum age in days for workflow states
        statuses: List of statuses to clean up (default: completed and failed)
        
    Returns:
        int: Number of workflows cleaned up
    """
    if statuses is None:
        statuses = [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
    
    cleaned_count = 0
    
    for status in statuses:
        workflow_ids = await state_manager.list_workflows(status)
        
        for workflow_id in workflow_ids:
            try:
                state = await state_manager.load_state(workflow_id)
                if state and 'completed_at' in state:
                    completed_at = datetime.fromisoformat(state['completed_at'])
                    age = datetime.utcnow() - completed_at
                    
                    if age.days > max_age_days:
                        if await state_manager.delete_state(workflow_id):
                            cleaned_count += 1
            except Exception as e:
                logger.warning(f"Error cleaning up workflow {workflow_id}: {e}")
    
    return cleaned_count

# Export key classes and functions
__all__ = [
    'WorkflowStatus',
    'CheckpointStrategy', 
    'WorkflowState',
    'LangGraphExecutionContext',
    'StateManagerProtocol',
    'DatabaseStateManager',
    'PostgreSQLStateCheckpointer',
    'LangGraphWorkflowBase',
    'LangGraphAgentMixin',
    'create_hybrid_agent',
    'cleanup_old_workflows'
]