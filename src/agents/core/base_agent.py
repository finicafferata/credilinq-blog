"""
Enhanced base agent implementation with proper interfaces and communication protocols.
Now includes LangGraph state management capabilities for hybrid workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import uuid
from datetime import datetime
import traceback
import asyncio

# Import performance tracking
try:
    from ...core.agent_performance import (
        AgentPerformanceTracker, 
        cached_agent_execution, 
        global_performance_tracker,
        AgentCacheConfig
    )
    PERFORMANCE_TRACKING_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKING_AVAILABLE = False

# Type definitions
T = TypeVar('T')
AgentInput = Union[str, Dict[str, Any], List[Any]]
AgentOutput = Union[str, Dict[str, Any], List[Any]]

class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(Enum):
    """Agent type classification."""
    PLANNER = "planner"
    RESEARCHER = "researcher" 
    WRITER = "writer"
    EDITOR = "editor"
    CAMPAIGN_MANAGER = "campaign_manager"
    CONTENT_REPURPOSER = "content_repurposer"
    CONTENT_BRIEF = "content_brief"
    IMAGE_PROMPT = "image_prompt"
    VIDEO_PROMPT = "video_prompt"
    SEO = "seo"
    SOCIAL_MEDIA = "social_media"
    DOCUMENT_PROCESSOR = "document_processor"
    SEARCH = "search"
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"
    CONTENT_GENERATOR = "content_generator"
    AI_CONTENT_GENERATOR = "ai_content_generator"
    CONTENT_OPTIMIZER = "content_optimizer"
    CONTENT_AGENT = "content_agent"
    TASK_SCHEDULER = "task_scheduler"
    DISTRIBUTION_AGENT = "distribution_agent"

@dataclass
class AgentMetadata:
    """Agent metadata and configuration."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.PLANNER
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class AgentExecutionContext:
    """Context for agent execution."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    execution_started_at: datetime = field(default_factory=datetime.utcnow)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    """Standardized agent execution result."""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        return self.success
    
    @property
    def is_failure(self) -> bool:
        return not self.success

@dataclass 
class AgentState:
    """Agent execution state with LangGraph workflow support."""
    status: AgentStatus = AgentStatus.IDLE
    current_operation: Optional[str] = None
    progress_percentage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    execution_context: Optional[AgentExecutionContext] = None
    result: Optional[AgentResult] = None
    
    # LangGraph state management additions
    workflow_id: Optional[str] = None
    workflow_state: Optional[Dict[str, Any]] = None
    checkpoint_id: Optional[str] = None
    is_langgraph_workflow: bool = False
    recoverable: bool = False
    
    def to_workflow_state(self) -> Dict[str, Any]:
        """Convert to LangGraph workflow state format."""
        return {
            'agent_status': self.status.value,
            'current_operation': self.current_operation,
            'progress_percentage': self.progress_percentage,
            'last_updated': self.last_updated.isoformat(),
            'workflow_id': self.workflow_id,
            'checkpoint_id': self.checkpoint_id,
            'workflow_state': self.workflow_state or {},
            'recoverable': self.recoverable
        }
    
    @classmethod
    def from_workflow_state(cls, workflow_state: Dict[str, Any]) -> 'AgentState':
        """Create AgentState from LangGraph workflow state."""
        return cls(
            status=AgentStatus(workflow_state.get('agent_status', 'idle')),
            current_operation=workflow_state.get('current_operation'),
            progress_percentage=workflow_state.get('progress_percentage', 0.0),
            last_updated=datetime.fromisoformat(workflow_state.get('last_updated', datetime.utcnow().isoformat())),
            workflow_id=workflow_state.get('workflow_id'),
            workflow_state=workflow_state.get('workflow_state'),
            checkpoint_id=workflow_state.get('checkpoint_id'),
            is_langgraph_workflow=True,
            recoverable=workflow_state.get('recoverable', False)
        )

class AgentCommunicationProtocol:
    """Standard communication protocol for agent interactions."""
    
    @staticmethod
    def create_message(
        sender_id: str,
        recipient_id: str,
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized message between agents."""
        return {
            "message_id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
    
    @staticmethod
    def create_request(
        sender_id: str,
        recipient_id: str,
        operation: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized request message."""
        return AgentCommunicationProtocol.create_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type="request",
            content={
                "operation": operation,
                "parameters": parameters
            },
            metadata=metadata
        )
    
    @staticmethod
    def create_response(
        sender_id: str,
        recipient_id: str,
        request_id: str,
        result: AgentResult,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized response message."""
        return AgentCommunicationProtocol.create_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type="response",
            content={
                "request_id": request_id,
                "result": result.__dict__
            },
            metadata=metadata
        )

class BaseAgent(ABC, Generic[T]):
    """
    Enhanced base class for all agents with proper interfaces and communication protocols.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize base agent."""
        self.metadata = metadata or AgentMetadata()
        self.state = AgentState()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._setup_logging()
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = 0.0
        
        # Enhanced performance tracking
        if PERFORMANCE_TRACKING_AVAILABLE:
            self.performance_tracker = global_performance_tracker
            self.agent_name = self.metadata.name or self.__class__.__name__.lower()
        else:
            self.performance_tracker = None
            self.agent_name = self.__class__.__name__.lower()
        
        # LangGraph workflow capabilities
        self._langgraph_enabled = False
        self._workflow_manager = None
        self._state_persistence = {}  # In-memory state for workflow recovery
        
        # Initialize agent-specific setup
        self._initialize()
    
    def _setup_logging(self):
        """Set up agent-specific logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.metadata.name or self.__class__.__name__} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize(self):
        """Override this method for agent-specific initialization."""
        pass
    
    @abstractmethod
    def execute(
        self, 
        input_data: AgentInput, 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute the agent's main functionality.
        
        Args:
            input_data: Input data for the agent
            context: Execution context
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Standardized result object
        """
        pass
    
    def execute_safe(
        self, 
        input_data: AgentInput, 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute agent with comprehensive error handling and state management.
        
        Args:
            input_data: Input data for the agent
            context: Execution context
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Standardized result object
        """
        if context is None:
            context = AgentExecutionContext()
        
        # Update state
        self.state.status = AgentStatus.RUNNING
        self.state.execution_context = context
        self.state.last_updated = datetime.utcnow()
        
        start_time = time.time()
        retries = 0
        
        while retries <= self.metadata.max_retries:
            try:
                self.logger.info(f"Executing agent {self.metadata.name} (attempt {retries + 1}/{self.metadata.max_retries + 1})")
                
                # Validate input
                self._validate_input(input_data)
                
                # Execute main logic
                result = self.execute(input_data, context, **kwargs)
                
                # Update metrics
                execution_time = (time.time() - start_time) * 1000
                self._execution_count += 1
                self._total_execution_time += execution_time
                self._last_execution_time = execution_time
                
                # Update result with execution time
                result.execution_time_ms = execution_time
                
                # Update state
                self.state.status = AgentStatus.COMPLETED
                self.state.result = result
                self.state.progress_percentage = 100.0
                self.state.last_updated = datetime.utcnow()
                
                self.logger.info(f"Agent execution completed successfully in {execution_time:.2f}ms")
                return result
                
            except Exception as e:
                retries += 1
                error_message = str(e)
                error_traceback = traceback.format_exc()
                
                self.logger.error(f"Agent execution failed (attempt {retries}): {error_message}")
                self.logger.debug(f"Error traceback: {error_traceback}")
                
                if retries > self.metadata.max_retries:
                    # Final failure
                    execution_time = (time.time() - start_time) * 1000
                    result = AgentResult(
                        success=False,
                        error_message=error_message,
                        error_code=self._get_error_code(e),
                        execution_time_ms=execution_time,
                        metadata={"traceback": error_traceback, "attempts": retries}
                    )
                    
                    # Update state
                    self.state.status = AgentStatus.FAILED
                    self.state.result = result
                    self.state.last_updated = datetime.utcnow()
                    
                    return result
                else:
                    # Wait before retry
                    time.sleep(2 ** (retries - 1))  # Exponential backoff
        
        # Should never reach here, but just in case
        return AgentResult(
            success=False,
            error_message="Unknown error occurred",
            error_code="UNKNOWN_ERROR"
        )
    
    def _validate_input(self, input_data: AgentInput) -> None:
        """
        Validate input data. Override in subclasses for specific validation.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if input_data is None:
            raise ValueError("Input data cannot be None")
    
    def _get_error_code(self, exception: Exception) -> str:
        """
        Get standardized error code for exception.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            str: Error code
        """
        error_type = type(exception).__name__
        return f"{self.metadata.agent_type.value.upper()}_{error_type.upper()}"
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return self.metadata.capabilities.copy()
    
    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        return self.state.status
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        avg_execution_time = (
            self._total_execution_time / self._execution_count 
            if self._execution_count > 0 else 0.0
        )
        
        return {
            "execution_count": self._execution_count,
            "total_execution_time_ms": self._total_execution_time,
            "average_execution_time_ms": avg_execution_time,
            "last_execution_time_ms": self._last_execution_time,
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate agent success rate."""
        # This would need to be implemented with proper tracking
        # For now, return 100% if no failures recorded
        return 100.0
    
    def send_message(
        self, 
        recipient_agent: 'BaseAgent', 
        message_type: str, 
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to another agent.
        
        Args:
            recipient_agent: Target agent
            message_type: Type of message
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Dict: Message sent
        """
        message = AgentCommunicationProtocol.create_message(
            sender_id=self.metadata.agent_id,
            recipient_id=recipient_agent.metadata.agent_id,
            message_type=message_type,
            content=content,
            metadata=metadata
        )
        
        self.logger.info(f"Sent {message_type} message to {recipient_agent.metadata.name}")
        return message
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming message from another agent.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional[Dict]: Response message if needed
        """
        self.logger.info(f"Received {message.get('message_type')} message from {message.get('sender_id')}")
        
        # Override in subclasses for specific message handling
        return None
    
    def __str__(self) -> str:
        """String representation of agent."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.metadata.agent_id[:8]}, "
            f"name={self.metadata.name}, "
            f"status={self.state.status.value})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of agent."""
        return (
            f"{self.__class__.__name__}("
            f"metadata={self.metadata}, "
            f"state={self.state})"
        )
    
    # Performance tracking helper methods
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics for this agent."""
        if self.performance_tracker:
            return await self.performance_tracker.get_agent_analytics(self.agent_name)
        else:
            return {
                "agent_name": self.agent_name,
                "execution_count": self._execution_count,
                "total_execution_time_ms": self._total_execution_time,
                "avg_execution_time_ms": self._total_execution_time / max(self._execution_count, 1),
                "last_execution_time_ms": self._last_execution_time
            }
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries for this agent."""
        if self.performance_tracker:
            return await self.performance_tracker.invalidate_agent_cache(self.agent_name, pattern)
        return 0
    
    def configure_caching(
        self, 
        ttl: int = 3600, 
        enable_content_based_caching: bool = True,
        cache_key_fields: Optional[List[str]] = None
    ):
        """Configure caching for this agent."""
        if self.performance_tracker:
            cache_config = AgentCacheConfig(
                ttl=ttl,
                enable_content_based_caching=enable_content_based_caching,
                cache_key_fields=cache_key_fields
            )
            self.performance_tracker.update_cache_config(self.agent_name, cache_config)
            self.logger.info(f"Updated cache configuration for {self.agent_name}: TTL={ttl}s")
    
    # LangGraph workflow methods
    def enable_langgraph_workflow(self, workflow_manager=None):
        """
        Enable LangGraph workflow capabilities for this agent.
        
        Args:
            workflow_manager: Optional workflow manager for state persistence
        """
        self._langgraph_enabled = True
        self._workflow_manager = workflow_manager
        self.logger.info(f"LangGraph workflow enabled for {self.agent_name}")
    
    def disable_langgraph_workflow(self):
        """Disable LangGraph workflow capabilities."""
        self._langgraph_enabled = False
        self._workflow_manager = None
        self.logger.info(f"LangGraph workflow disabled for {self.agent_name}")
    
    def is_langgraph_enabled(self) -> bool:
        """Check if LangGraph workflow is enabled."""
        return self._langgraph_enabled
    
    def save_workflow_state(self, workflow_id: str, state_data: Dict[str, Any]) -> bool:
        """
        Save workflow state for recovery.
        
        Args:
            workflow_id: Unique workflow identifier
            state_data: State data to save
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Update agent state with workflow info
            self.state.workflow_id = workflow_id
            self.state.workflow_state = state_data
            self.state.is_langgraph_workflow = True
            self.state.recoverable = True
            self.state.last_updated = datetime.utcnow()
            
            # Store in in-memory persistence (can be enhanced with database later)
            self._state_persistence[workflow_id] = {
                'agent_state': self.state.to_workflow_state(),
                'workflow_data': state_data,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Saved workflow state for {workflow_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save workflow state for {workflow_id}: {e}")
            return False
    
    def load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load workflow state for recovery.
        
        Args:
            workflow_id: Workflow identifier to load
            
        Returns:
            Optional state data if found
        """
        try:
            if workflow_id in self._state_persistence:
                saved_data = self._state_persistence[workflow_id]
                
                # Restore agent state
                self.state = AgentState.from_workflow_state(saved_data['agent_state'])
                
                self.logger.info(f"Loaded workflow state for {workflow_id}")
                return saved_data['workflow_data']
        except Exception as e:
            self.logger.error(f"Failed to load workflow state for {workflow_id}: {e}")
        
        return None
    
    def clear_workflow_state(self, workflow_id: str) -> bool:
        """
        Clear saved workflow state.
        
        Args:
            workflow_id: Workflow identifier to clear
            
        Returns:
            bool: True if cleared successfully
        """
        try:
            if workflow_id in self._state_persistence:
                del self._state_persistence[workflow_id]
                
                # Reset agent state if it matches this workflow
                if self.state.workflow_id == workflow_id:
                    self.state.workflow_id = None
                    self.state.workflow_state = None
                    self.state.checkpoint_id = None
                    self.state.is_langgraph_workflow = False
                    self.state.recoverable = False
                
                self.logger.info(f"Cleared workflow state for {workflow_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to clear workflow state for {workflow_id}: {e}")
        
        return False
    
    def get_workflow_states(self) -> Dict[str, Any]:
        """Get all saved workflow states for this agent."""
        return self._state_persistence.copy()
    
    async def execute_with_workflow_recovery(
        self,
        input_data: AgentInput,
        workflow_id: Optional[str] = None,
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute agent with LangGraph workflow recovery capabilities.
        
        Args:
            input_data: Input data for the agent
            workflow_id: Optional workflow ID for recovery
            context: Execution context
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Execution result with workflow state
        """
        if not self._langgraph_enabled:
            # Fall back to regular execution
            return self.execute_safe(input_data, context, **kwargs)
        
        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())
        
        # Try to recover previous state
        recovered_state = self.load_workflow_state(workflow_id)
        if recovered_state:
            self.logger.info(f"Recovered workflow {workflow_id}, resuming execution")
            # Merge recovered state with current input
            if isinstance(input_data, dict) and isinstance(recovered_state, dict):
                input_data = {**recovered_state, **input_data}
        
        # Set workflow ID in context
        if context is None:
            context = AgentExecutionContext()
        context.workflow_id = workflow_id
        
        try:
            # Execute with state saving
            result = self.execute_safe(input_data, context, **kwargs)
            
            # Save successful state
            if result.is_success:
                self.save_workflow_state(workflow_id, {
                    'input_data': input_data,
                    'result': result.data,
                    'completed': True
                })
            
            # Add workflow info to result metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'workflow_id': workflow_id,
                'langgraph_enabled': True,
                'recoverable': True
            })
            
            return result
            
        except Exception as e:
            # Save error state for recovery
            self.save_workflow_state(workflow_id, {
                'input_data': input_data,
                'error': str(e),
                'failed_at': datetime.utcnow().isoformat(),
                'completed': False
            })
            
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code=self._get_error_code(e),
                metadata={
                    'workflow_id': workflow_id,
                    'langgraph_enabled': True,
                    'recoverable': True
                }
            )

class WorkflowAgent(BaseAgent[Dict[str, Any]]):
    """
    Base class for workflow orchestration agents.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
                name="WorkflowAgent"
            )
        super().__init__(metadata)
        self.child_agents: List[BaseAgent] = []
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add a child agent to the workflow."""
        self.child_agents.append(agent)
        self.logger.info(f"Added agent {agent.metadata.name} to workflow")
    
    def remove_agent(self, agent: BaseAgent) -> bool:
        """Remove a child agent from the workflow."""
        try:
            self.child_agents.remove(agent)
            self.logger.info(f"Removed agent {agent.metadata.name} from workflow")
            return True
        except ValueError:
            return False
    
    def get_child_agents(self) -> List[BaseAgent]:
        """Get all child agents."""
        return self.child_agents.copy()
    
    @abstractmethod
    def execute_workflow(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """Execute the workflow with child agents."""
        pass
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """Execute workflow."""
        return self.execute_workflow(input_data, context)


# Backward compatibility - keep the old BaseAgent interface
class LegacyBaseAgent:
    """
    Legacy base class for backward compatibility.
    """
    def execute(self, *args, **kwargs):
        raise NotImplementedError("The execute method must be implemented by subclasses.")