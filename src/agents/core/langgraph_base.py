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
    """Database-backed state manager for workflow persistence."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url
        self._connection_pool = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    async def _get_db_connection(self):
        """Get database connection (implement based on your DB setup)."""
        # This will be implemented with the actual database connection
        # For now, return None to indicate not implemented
        return None
    
    async def save_state(self, workflow_id: str, state: Dict[str, Any]) -> bool:
        """Save workflow state to database."""
        try:
            # Implement database save logic here
            # This is a placeholder for the actual implementation
            self.logger.info(f"Saving state for workflow {workflow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save state for {workflow_id}: {e}")
            return False
    
    async def load_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state from database."""
        try:
            # Implement database load logic here
            # This is a placeholder for the actual implementation
            self.logger.info(f"Loading state for workflow {workflow_id}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load state for {workflow_id}: {e}")
            return None
    
    async def delete_state(self, workflow_id: str) -> bool:
        """Delete workflow state from database."""
        try:
            # Implement database delete logic here
            self.logger.info(f"Deleting state for workflow {workflow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete state for {workflow_id}: {e}")
            return False
    
    async def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[str]:
        """List workflow IDs by status."""
        try:
            # Implement database list logic here
            self.logger.info(f"Listing workflows with status {status}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
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
                # TODO: Implement PostgreSQLSaver when database connection is ready
                self._checkpointer = MemorySaver()  # Fallback for now
                self.logger.warning("Using MemorySaver as fallback for database checkpointing")
            
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
    'LangGraphWorkflowBase',
    'LangGraphAgentMixin',
    'create_hybrid_agent',
    'cleanup_old_workflows'
]