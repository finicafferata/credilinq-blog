"""
Agent factory pattern implementation for creating and managing agents.
"""

from typing import Type, Dict, Any, Optional, List, Callable
from enum import Enum
import logging
from .base_agent import BaseAgent, AgentType, AgentMetadata, AgentResult, AgentExecutionContext

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for managing agent types and their implementations."""
    
    def __init__(self):
        self._agents: Dict[AgentType, Type[BaseAgent]] = {}
        self._factories: Dict[AgentType, Callable[..., BaseAgent]] = {}
        self._metadata_templates: Dict[AgentType, AgentMetadata] = {}
    
    def register_agent(
        self, 
        agent_type: AgentType, 
        agent_class: Type[BaseAgent],
        metadata_template: Optional[AgentMetadata] = None,
        factory_function: Optional[Callable[..., BaseAgent]] = None
    ) -> None:
        """
        Register an agent type with its implementation.
        
        Args:
            agent_type: Type of agent
            agent_class: Agent implementation class
            metadata_template: Default metadata template
            factory_function: Custom factory function (optional)
        """
        self._agents[agent_type] = agent_class
        
        if factory_function:
            self._factories[agent_type] = factory_function
        
        if metadata_template:
            self._metadata_templates[agent_type] = metadata_template
        else:
            # Create default metadata
            self._metadata_templates[agent_type] = AgentMetadata(
                agent_type=agent_type,
                name=agent_class.__name__,
                description=agent_class.__doc__ or ""
            )
        
        logger.info(f"Registered agent type {agent_type.value} with class {agent_class.__name__}")
    
    def get_agent_class(self, agent_type: AgentType) -> Optional[Type[BaseAgent]]:
        """Get agent class for a given type."""
        return self._agents.get(agent_type)
    
    def get_metadata_template(self, agent_type: AgentType) -> Optional[AgentMetadata]:
        """Get metadata template for a given agent type."""
        return self._metadata_templates.get(agent_type)
    
    def get_registered_types(self) -> List[AgentType]:
        """Get all registered agent types."""
        return list(self._agents.keys())
    
    def is_registered(self, agent_type: AgentType) -> bool:
        """Check if agent type is registered."""
        return agent_type in self._agents
    
    def create_agent(
        self, 
        agent_type: AgentType, 
        metadata: Optional[AgentMetadata] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent instance using the factory.
        
        Args:
            agent_type: Type of agent to create
            metadata: Agent metadata (uses template if not provided)
            **kwargs: Additional arguments for agent construction
            
        Returns:
            BaseAgent: Created agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in self._agents:
            raise ValueError(f"Agent type {agent_type.value} is not registered")
        
        # Use custom factory if available
        if agent_type in self._factories:
            return self._factories[agent_type](metadata=metadata, **kwargs)
        
        # Use default factory
        agent_class = self._agents[agent_type]
        if metadata is None:
            metadata = self._metadata_templates[agent_type]
        
        return agent_class(metadata=metadata, **kwargs)

# Global agent registry
_global_registry = AgentRegistry()

class AgentFactory:
    """
    Main factory class for creating agents with various configurations.
    """
    
    def __init__(self, registry: Optional[AgentRegistry] = None):
        """
        Initialize agent factory.
        
        Args:
            registry: Agent registry to use (uses global if not provided)
        """
        self.registry = registry or _global_registry
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def create_agent(
        self, 
        agent_type: AgentType, 
        metadata: Optional[AgentMetadata] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            metadata: Custom metadata (optional)
            **kwargs: Additional arguments
            
        Returns:
            BaseAgent: Created agent instance
        """
        try:
            agent = self.registry.create_agent(agent_type, metadata, **kwargs)
            self.logger.info(f"Created agent {agent.metadata.name} of type {agent_type.value}")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to create agent of type {agent_type.value}: {str(e)}")
            raise
    
    def create_workflow_agents(
        self, 
        agent_types: List[AgentType],
        workflow_metadata: Optional[Dict[AgentType, AgentMetadata]] = None
    ) -> List[BaseAgent]:
        """
        Create multiple agents for a workflow.
        
        Args:
            agent_types: List of agent types to create
            workflow_metadata: Metadata for each agent type
            
        Returns:
            List[BaseAgent]: Created agents
        """
        agents = []
        workflow_metadata = workflow_metadata or {}
        
        for agent_type in agent_types:
            metadata = workflow_metadata.get(agent_type)
            agent = self.create_agent(agent_type, metadata)
            agents.append(agent)
        
        self.logger.info(f"Created workflow with {len(agents)} agents")
        return agents
    
    def create_specialized_agent(
        self,
        agent_type: AgentType,
        name: str,
        capabilities: List[str],
        description: str = "",
        version: str = "1.0.0",
        max_retries: int = 3,
        **kwargs
    ) -> BaseAgent:
        """
        Create a specialized agent with custom configuration.
        
        Args:
            agent_type: Type of agent
            name: Agent name
            capabilities: List of capabilities
            description: Agent description
            version: Agent version
            max_retries: Maximum retry attempts
            **kwargs: Additional arguments
            
        Returns:
            BaseAgent: Configured agent
        """
        metadata = AgentMetadata(
            agent_type=agent_type,
            name=name,
            description=description,
            version=version,
            capabilities=capabilities,
            max_retries=max_retries
        )
        
        return self.create_agent(agent_type, metadata, **kwargs)
    
    def get_available_types(self) -> List[AgentType]:
        """Get all available agent types."""
        return self.registry.get_registered_types()
    
    def clone_agent(self, source_agent: BaseAgent, new_name: Optional[str] = None) -> BaseAgent:
        """
        Clone an existing agent with the same configuration.
        
        Args:
            source_agent: Agent to clone
            new_name: New name for cloned agent (optional)
            
        Returns:
            BaseAgent: Cloned agent
        """
        metadata = AgentMetadata(
            agent_type=source_agent.metadata.agent_type,
            name=new_name or f"{source_agent.metadata.name}_clone",
            description=source_agent.metadata.description,
            version=source_agent.metadata.version,
            capabilities=source_agent.metadata.capabilities.copy(),
            dependencies=source_agent.metadata.dependencies.copy(),
            max_retries=source_agent.metadata.max_retries,
            timeout_seconds=source_agent.metadata.timeout_seconds,
            tags=source_agent.metadata.tags.copy()
        )
        
        return self.create_agent(source_agent.metadata.agent_type, metadata)

class AgentPool:
    """
    Pool for managing and reusing agent instances.
    """
    
    def __init__(self, factory: Optional[AgentFactory] = None):
        """
        Initialize agent pool.
        
        Args:
            factory: Agent factory to use
        """
        self.factory = factory or AgentFactory()
        self._pool: Dict[AgentType, List[BaseAgent]] = {}
        self._active: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def get_agent(
        self, 
        agent_type: AgentType, 
        create_if_needed: bool = True
    ) -> Optional[BaseAgent]:
        """
        Get an agent from the pool.
        
        Args:
            agent_type: Type of agent needed
            create_if_needed: Create new agent if pool is empty
            
        Returns:
            Optional[BaseAgent]: Available agent or None
        """
        if agent_type not in self._pool:
            self._pool[agent_type] = []
        
        # Get available agent from pool
        if self._pool[agent_type]:
            agent = self._pool[agent_type].pop()
            self._active[agent.metadata.agent_id] = agent
            self.logger.info(f"Retrieved agent {agent.metadata.name} from pool")
            return agent
        
        # Create new agent if needed
        if create_if_needed:
            agent = self.factory.create_agent(agent_type)
            self._active[agent.metadata.agent_id] = agent
            self.logger.info(f"Created new agent {agent.metadata.name} for pool")
            return agent
        
        return None
    
    def return_agent(self, agent: BaseAgent) -> None:
        """
        Return an agent to the pool.
        
        Args:
            agent: Agent to return
        """
        agent_id = agent.metadata.agent_id
        
        if agent_id in self._active:
            del self._active[agent_id]
            
            # Reset agent state
            agent.state.status = agent.state.status.IDLE
            agent.state.current_operation = None
            agent.state.progress_percentage = 0.0
            
            # Add to pool
            agent_type = agent.metadata.agent_type
            if agent_type not in self._pool:
                self._pool[agent_type] = []
            
            self._pool[agent_type].append(agent)
            self.logger.info(f"Returned agent {agent.metadata.name} to pool")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        stats = {
            "total_pooled": sum(len(agents) for agents in self._pool.values()),
            "active_agents": len(self._active),
            "pool_by_type": {
                agent_type.value: len(agents) 
                for agent_type, agents in self._pool.items()
            }
        }
        return stats
    
    def clear_pool(self) -> None:
        """Clear all agents from the pool."""
        self._pool.clear()
        self._active.clear()
        self.logger.info("Cleared agent pool")

# Convenience functions for global registry
def register_agent(
    agent_type: AgentType, 
    agent_class: Type[BaseAgent],
    metadata_template: Optional[AgentMetadata] = None,
    factory_function: Optional[Callable[..., BaseAgent]] = None
) -> None:
    """Register an agent type globally."""
    _global_registry.register_agent(
        agent_type, agent_class, metadata_template, factory_function
    )

def create_agent(
    agent_type: AgentType, 
    metadata: Optional[AgentMetadata] = None,
    **kwargs
) -> BaseAgent:
    """Create an agent using the global factory."""
    factory = AgentFactory()
    return factory.create_agent(agent_type, metadata, **kwargs)

def get_available_agent_types() -> List[AgentType]:
    """Get all available agent types from global registry."""
    return _global_registry.get_registered_types()

# Pre-configured factories for common use cases
class BlogWorkflowAgentFactory(AgentFactory):
    """Specialized factory for blog workflow agents."""
    
    def create_blog_workflow_agents(self) -> List[BaseAgent]:
        """Create all agents needed for blog workflow."""
        agent_types = [
            AgentType.PLANNER,
            AgentType.RESEARCHER,
            AgentType.WRITER,
            AgentType.EDITOR
        ]
        
        return self.create_workflow_agents(agent_types)
    
    def create_campaign_workflow_agents(self) -> List[BaseAgent]:
        """Create all agents needed for campaign workflow."""
        agent_types = [
            AgentType.CAMPAIGN_MANAGER,
            AgentType.CONTENT_REPURPOSER,
            AgentType.IMAGE_PROMPT_GENERATOR
        ]
        
        return self.create_workflow_agents(agent_types)

# Global instances
global_factory = AgentFactory()
global_pool = AgentPool()
blog_workflow_factory = BlogWorkflowAgentFactory()