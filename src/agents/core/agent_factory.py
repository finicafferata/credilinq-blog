"""
Agent factory pattern implementation for creating and managing agents.
Now supports hybrid LangChain/LangGraph agent creation with workflow capabilities.
"""

from typing import Type, Dict, Any, Optional, List, Callable, Union
from enum import Enum
import logging
from .base_agent import BaseAgent, AgentType, AgentMetadata, AgentResult, AgentExecutionContext
from .langgraph_base import (
    LangGraphWorkflowBase, LangGraphAgentMixin, create_hybrid_agent, 
    LangGraphExecutionContext, CheckpointStrategy
)

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for managing agent types and their implementations."""
    
    def __init__(self):
        self._agents: Dict[AgentType, Type[BaseAgent]] = {}
        self._factories: Dict[AgentType, Callable[..., BaseAgent]] = {}
        self._metadata_templates: Dict[AgentType, AgentMetadata] = {}
        
        # LangGraph workflow support
        self._workflows: Dict[AgentType, Type[LangGraphWorkflowBase]] = {}
        self._hybrid_factories: Dict[AgentType, Callable[..., BaseAgent]] = {}
        self._langgraph_enabled: Dict[AgentType, bool] = {}
    
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
    
    def register_workflow(
        self,
        agent_type: AgentType,
        workflow_class: Type[LangGraphWorkflowBase],
        enable_by_default: bool = False
    ) -> None:
        """
        Register a LangGraph workflow for an agent type.
        
        Args:
            agent_type: Agent type to register workflow for
            workflow_class: LangGraph workflow class
            enable_by_default: Whether to enable LangGraph by default for this agent
        """
        self._workflows[agent_type] = workflow_class
        self._langgraph_enabled[agent_type] = enable_by_default
        
        # Create hybrid factory that combines agent and workflow
        def hybrid_factory(metadata: Optional[AgentMetadata] = None, enable_langgraph: bool = enable_by_default, **kwargs) -> BaseAgent:
            agent_class = self._agents[agent_type]
            if enable_langgraph:
                return create_hybrid_agent(
                    agent_class=agent_class,
                    workflow_class=workflow_class,
                    enable_langgraph=True,
                    metadata=metadata,
                    **kwargs
                )
            else:
                return agent_class(metadata=metadata, **kwargs)
        
        self._hybrid_factories[agent_type] = hybrid_factory
        
        logger.info(f"Registered LangGraph workflow for {agent_type.value}: {workflow_class.__name__}")
    
    def unregister_workflow(self, agent_type: AgentType) -> None:
        """Remove LangGraph workflow registration for an agent type."""
        if agent_type in self._workflows:
            del self._workflows[agent_type]
        if agent_type in self._hybrid_factories:
            del self._hybrid_factories[agent_type]
        if agent_type in self._langgraph_enabled:
            del self._langgraph_enabled[agent_type]
        
        logger.info(f"Unregistered LangGraph workflow for {agent_type.value}")
    
    def has_workflow(self, agent_type: AgentType) -> bool:
        """Check if agent type has a registered LangGraph workflow."""
        return agent_type in self._workflows
    
    def is_langgraph_enabled_by_default(self, agent_type: AgentType) -> bool:
        """Check if LangGraph is enabled by default for an agent type."""
        return self._langgraph_enabled.get(agent_type, False)
    
    def get_workflow_class(self, agent_type: AgentType) -> Optional[Type[LangGraphWorkflowBase]]:
        """Get the registered workflow class for an agent type."""
        return self._workflows.get(agent_type)
    
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
        enable_langgraph: Optional[bool] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent instance using the factory with optional LangGraph workflow support.
        
        Args:
            agent_type: Type of agent to create
            metadata: Agent metadata (uses template if not provided)
            enable_langgraph: Whether to enable LangGraph workflow (None = use default)
            **kwargs: Additional arguments for agent construction
            
        Returns:
            BaseAgent: Created agent instance with optional LangGraph capabilities
            
        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in self._agents:
            raise ValueError(f"Agent type {agent_type.value} is not registered")
        
        # Determine if LangGraph should be enabled
        if enable_langgraph is None:
            enable_langgraph = self.is_langgraph_enabled_by_default(agent_type)
        
        # Use hybrid factory if LangGraph is enabled and workflow is available
        if enable_langgraph and agent_type in self._hybrid_factories:
            return self._hybrid_factories[agent_type](
                metadata=metadata, 
                enable_langgraph=True, 
                **kwargs
            )
        
        # Use custom factory if available
        if agent_type in self._factories:
            return self._factories[agent_type](metadata=metadata, **kwargs)
        
        # Use default factory
        agent_class = self._agents[agent_type]
        if metadata is None:
            metadata = self._metadata_templates[agent_type]
        
        agent = agent_class(metadata=metadata, **kwargs)
        
        # Enable LangGraph workflow if requested and available
        if enable_langgraph and agent_type in self._workflows:
            workflow_class = self._workflows[agent_type]
            workflow_name = f"{agent_class.__name__}_workflow"
            workflow = workflow_class(workflow_name=workflow_name)
            
            # Add LangGraph mixin capabilities if not already present
            if not hasattr(agent, 'enable_langgraph'):
                # Dynamically add mixin capabilities
                agent.__class__ = type(
                    f"Hybrid{agent.__class__.__name__}",
                    (LangGraphAgentMixin, agent.__class__),
                    {}
                )
                agent._langgraph_enabled = False
                agent._workflow = None
            
            agent.enable_langgraph(workflow)
        
        return agent

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
        enable_langgraph: Optional[bool] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent of the specified type with optional LangGraph support.
        
        Args:
            agent_type: Type of agent to create
            metadata: Custom metadata (optional)
            enable_langgraph: Whether to enable LangGraph workflow (None = use default)
            **kwargs: Additional arguments
            
        Returns:
            BaseAgent: Created agent instance
        """
        try:
            agent = self.registry.create_agent(
                agent_type, metadata, enable_langgraph=enable_langgraph, **kwargs
            )
            langgraph_status = "with LangGraph" if getattr(agent, '_langgraph_enabled', False) else "LangChain only"
            self.logger.info(f"Created agent {agent.metadata.name} of type {agent_type.value} ({langgraph_status})")
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

def register_workflow(
    agent_type: AgentType,
    workflow_class: Type[LangGraphWorkflowBase],
    enable_by_default: bool = False
) -> None:
    """Register a LangGraph workflow globally for an agent type."""
    _global_registry.register_workflow(agent_type, workflow_class, enable_by_default)

def create_langgraph_agent(
    agent_type: AgentType,
    metadata: Optional[AgentMetadata] = None,
    **kwargs
) -> BaseAgent:
    """Create an agent with LangGraph workflow enabled."""
    factory = AgentFactory()
    return factory.create_agent(agent_type, metadata, enable_langgraph=True, **kwargs)

def create_langchain_agent(
    agent_type: AgentType,
    metadata: Optional[AgentMetadata] = None,
    **kwargs
) -> BaseAgent:
    """Create an agent with only LangChain capabilities (no LangGraph)."""
    factory = AgentFactory()
    return factory.create_agent(agent_type, metadata, enable_langgraph=False, **kwargs)

def get_workflow_info(agent_type: AgentType) -> Optional[Dict[str, Any]]:
    """Get information about registered workflow for an agent type."""
    if not _global_registry.has_workflow(agent_type):
        return None
    
    workflow_class = _global_registry.get_workflow_class(agent_type)
    return {
        "agent_type": agent_type.value,
        "workflow_class": workflow_class.__name__ if workflow_class else None,
        "langgraph_enabled_by_default": _global_registry.is_langgraph_enabled_by_default(agent_type),
        "has_workflow": True
    }

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

# Initialize specialized agents registration
def _initialize_default_agents():
    """Initialize registration of all specialized agents with LangGraph workflows."""
    try:
        # TEMPORARILY DISABLED during Railway debugging - agents causing startup failure
        logger.warning("⚠️ Agent registration temporarily disabled during Railway debugging")
        return
        
        # Import only LangGraph-based agents (deleted non-LangGraph versions)
        
        # Import all LangGraph workflows
        from ..specialized.planner_agent_langgraph import PlannerAgentWorkflow
        from ..specialized.researcher_agent_langgraph import ResearcherAgentWorkflow
        from ..specialized.writer_agent_langgraph import WriterAgentLangGraph
        from ..specialized.editor_agent_langgraph import EditorAgentWorkflow
        from ..specialized.seo_agent_langgraph import SEOAgentWorkflow
        from ..specialized.social_media_agent_langgraph import SocialMediaAgentWorkflow
        from ..specialized.geo_analysis_agent_langgraph import GEOAnalysisAgentWorkflow
        from ..specialized.content_repurposer_langgraph import ContentRepurposerWorkflow
        from ..specialized.distribution_agent_langgraph import DistributionAgentWorkflow
        from ..specialized.task_scheduler_langgraph import TaskSchedulerWorkflow
        from ..specialized.document_processor_langgraph import DocumentProcessorWorkflow
        from ..specialized.content_agent_langgraph import ContentAgentWorkflow
        from ..specialized.ai_content_generator_langgraph import AIContentGeneratorWorkflow
        from ..specialized.content_brief_agent_langgraph import ContentBriefAgentWorkflow
        from ..specialized.campaign_manager_langgraph import CampaignManagerWorkflow
        from ..specialized.search_agent_langgraph import SearchAgentWorkflow
        from ..specialized.image_prompt_agent_langgraph import ImagePromptAgentLangGraph
        from ..specialized.video_prompt_agent_langgraph import VideoPromptAgentLangGraph
        
        # Register Core Blog Workflow Agents
        register_agent(
            AgentType.PLANNER,
            PlannerAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.PLANNER,
                name="PlannerAgent",
                description="Creates strategic content plans with SEO research and competitive analysis",
                capabilities=["content_planning", "strategy_development", "seo_research", "competitive_analysis"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.RESEARCHER,
            ResearcherAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.RESEARCHER,
                name="ResearcherAgent",
                description="Conducts comprehensive research for content creation",
                capabilities=["web_research", "fact_checking", "source_verification", "data_analysis"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.WRITER,
            WriterAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.WRITER,
                name="WriterAgent",
                description="Creates high-quality content with advanced writing capabilities",
                capabilities=["content_writing", "tone_adaptation", "structure_optimization", "engagement_optimization"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.EDITOR,
            EditorAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.EDITOR,
                name="EditorAgent",
                description="Reviews and enhances content for quality, clarity, and engagement",
                capabilities=["content_editing", "quality_assurance", "grammar_checking", "style_optimization"],
                dependencies=[]
            )
        )
        
        # Register Image Prompt Agent
        register_agent(
            AgentType.IMAGE_PROMPT,
            ImagePromptAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.IMAGE_PROMPT,
                name="ImagePromptAgent",
                description="Generates creative prompts for image generation services",
                capabilities=["image_prompt_generation", "creative_prompt_optimization", "platform_specific_prompts"],
                dependencies=[]
            )
        )
        
        # Register Video Prompt Agent
        register_agent(
            AgentType.VIDEO_PROMPT,
            VideoPromptAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.VIDEO_PROMPT,
                name="VideoPromptAgent",
                description="Generates comprehensive prompts for video generation services",
                capabilities=["video_prompt_generation", "scene_planning", "narrative_structure", "transition_design"],
                dependencies=[]
            )
        )
        
        # Register Campaign Manager Agent
        register_agent(
            AgentType.CAMPAIGN_MANAGER,
            CampaignManagerAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.CAMPAIGN_MANAGER,
                name="CampaignManagerAgent",
                description="Manages campaign strategy and coordination",
                capabilities=["campaign_planning", "strategy_development", "workflow_coordination"],
                dependencies=[]
            )
        )
        
        # Register Content Repurposer Agent
        register_agent(
            AgentType.CONTENT_REPURPOSER,
            ContentRepurposer,
            metadata_template=AgentMetadata(
                agent_type=AgentType.CONTENT_REPURPOSER,
                name="ContentRepurposer", 
                description="Repurposes content across different formats and platforms",
                capabilities=["content_adaptation", "format_conversion", "platform_optimization"],
                dependencies=[]
            )
        )
        
        # Register Content Brief Agent
        register_agent(
            AgentType.CONTENT_BRIEF,
            ContentBriefAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.CONTENT_BRIEF,
                name="ContentBriefAgent",
                description="Creates strategic content briefs with SEO research and competitive analysis",
                capabilities=["seo_keyword_research", "competitor_analysis", "content_strategy", "audience_analysis"],
                dependencies=[]
            )
        )
        
        # Register SEO Agent
        register_agent(
            AgentType.SEO,
            SEOAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.SEO,
                name="SEOAgent", 
                description="Optimizes content for search engines with advanced SEO analysis and recommendations",
                capabilities=["keyword_optimization", "meta_tag_generation", "technical_seo", "competitive_analysis", "schema_markup"],
                dependencies=[]
            )
        )
        
        # Register Social Media Agent
        register_agent(
            AgentType.SOCIAL_MEDIA,
            SocialMediaAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.SOCIAL_MEDIA,
                name="SocialMediaAgent",
                description="Adapts content for different social media platforms with engagement optimization",
                capabilities=["platform_adaptation", "engagement_optimization", "hashtag_generation", "social_strategy"],
                dependencies=[]
            )
        )
        
        # Register GEO Analysis Agent (Content Optimizer)
        register_agent(
            AgentType.CONTENT_OPTIMIZER,
            GEOAnalysisAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.CONTENT_OPTIMIZER,
                name="GEOAnalysisAgent",
                description="Optimizes content for AI search engines like ChatGPT and Gemini with E-E-A-T analysis",
                capabilities=["generative_engine_optimization", "eeat_analysis", "factual_density", "ai_citability"],
                dependencies=[]
            )
        )
        
        # Register additional specialized agents
        register_agent(
            AgentType.DOCUMENT_PROCESSOR,
            DocumentProcessorAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.DOCUMENT_PROCESSOR,
                name="DocumentProcessorAgent",
                description="Processes and analyzes documents for knowledge base management",
                capabilities=["document_processing", "text_extraction", "content_analysis", "knowledge_extraction"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.SEARCH,
            WebSearchAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.SEARCH,
                name="SearchAgent",
                description="Performs web search and information gathering",
                capabilities=["web_search", "information_retrieval", "source_analysis", "competitive_intelligence"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.AI_CONTENT_GENERATOR,
            AIContentGeneratorAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.AI_CONTENT_GENERATOR,
                name="AIContentGenerator",
                description="Generates content using advanced AI templates and optimization",
                capabilities=["ai_content_generation", "template_processing", "content_optimization", "multi_format_output"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.CONTENT_AGENT,
            ContentGenerationAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.CONTENT_AGENT,
                name="ContentGenerationAgent",
                description="General content operations and management",
                capabilities=["content_management", "content_operations", "workflow_coordination"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.TASK_SCHEDULER,
            TaskSchedulerAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.TASK_SCHEDULER,
                name="TaskSchedulerAgent",
                description="Schedules and manages workflow tasks and execution timing",
                capabilities=["task_scheduling", "workflow_management", "timing_optimization", "resource_allocation"],
                dependencies=[]
            )
        )
        
        register_agent(
            AgentType.DISTRIBUTION_AGENT,
            DistributionAgent,
            metadata_template=AgentMetadata(
                agent_type=AgentType.DISTRIBUTION_AGENT,
                name="DistributionAgent",
                description="Distributes content across multiple channels and platforms",
                capabilities=["multi_channel_distribution", "platform_publishing", "content_syndication", "scheduling"],
                dependencies=[]
            )
        )
        
        # Register LangGraph workflows for all agents
        # Core workflow agents - enable by default for improved performance
        register_workflow(
            AgentType.PLANNER,
            PlannerAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced planning
        )
        
        register_workflow(
            AgentType.RESEARCHER,
            ResearcherAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced research
        )
        
        register_workflow(
            AgentType.WRITER,
            WriterAgentLangGraph,
            enable_by_default=True  # Enable by default for enhanced writing
        )
        
        register_workflow(
            AgentType.EDITOR,
            EditorAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced editing
        )
        
        # Optimization agents - enable by default for enhanced capabilities
        register_workflow(
            AgentType.SEO,
            SEOAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced optimization
        )
        
        register_workflow(
            AgentType.SOCIAL_MEDIA,
            SocialMediaAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced platform adaptation
        )
        
        register_workflow(
            AgentType.CONTENT_OPTIMIZER,
            GEOAnalysisAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced AI optimization
        )
        
        # Additional specialized workflows
        register_workflow(
            AgentType.CONTENT_REPURPOSER,
            ContentRepurposerWorkflow,
            enable_by_default=True  # Enable by default for enhanced repurposing
        )
        
        register_workflow(
            AgentType.DOCUMENT_PROCESSOR,
            DocumentProcessorWorkflow,
            enable_by_default=True  # Enable by default for enhanced document processing
        )
        
        register_workflow(
            AgentType.AI_CONTENT_GENERATOR,
            AIContentGeneratorWorkflow,
            enable_by_default=True  # Enable by default for enhanced content generation
        )
        
        register_workflow(
            AgentType.CONTENT_BRIEF,
            ContentBriefAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced briefing
        )
        
        register_workflow(
            AgentType.CAMPAIGN_MANAGER,
            CampaignManagerWorkflow,
            enable_by_default=True  # Enable by default for enhanced campaign management
        )
        
        register_workflow(
            AgentType.SEARCH,
            SearchAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced search
        )
        
        register_workflow(
            AgentType.CONTENT_AGENT,
            ContentAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced content operations
        )
        
        register_workflow(
            AgentType.TASK_SCHEDULER,
            TaskSchedulerWorkflow,
            enable_by_default=True  # Enable by default for enhanced task scheduling
        )
        
        register_workflow(
            AgentType.DISTRIBUTION_AGENT,
            DistributionAgentWorkflow,
            enable_by_default=True  # Enable by default for enhanced distribution
        )
        
        # New prompt generation agents
        register_workflow(
            AgentType.IMAGE_PROMPT,
            ImagePromptAgentLangGraph,
            enable_by_default=True  # Enable by default for enhanced image prompt generation
        )
        
        register_workflow(
            AgentType.VIDEO_PROMPT,
            VideoPromptAgentLangGraph,
            enable_by_default=True  # Enable by default for enhanced video prompt generation
        )
        
        logger.info("Successfully registered all specialized agents and LangGraph workflows with enhanced capabilities")
        
    except ImportError as e:
        logger.warning(f"Failed to register some specialized agents: {e}")
    except Exception as e:
        logger.error(f"Error during agent registration: {e}")

# Initialize agents on module load
_initialize_default_agents()

# Global instances
global_factory = AgentFactory()
global_pool = AgentPool()
blog_workflow_factory = BlogWorkflowAgentFactory()