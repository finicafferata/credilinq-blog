"""
Campaign Workflow Builder for Dynamic LangGraph Construction

This module provides tools for dynamically constructing LangGraph workflows
based on campaign types, agent requirements, and execution contexts.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.agents.core.base_agent import AgentType, BaseAgent
from src.agents.orchestration.types import CampaignType

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in campaign workflows."""
    AGENT = "agent"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    MERGE = "merge"
    CHECKPOINT = "checkpoint"
    ERROR_HANDLER = "error_handler"


class EdgeType(Enum):
    """Types of edges in campaign workflows."""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    ERROR = "error"
    RETRY = "retry"


@dataclass
class NodeDefinition:
    """Definition of a workflow node."""
    node_id: str
    node_type: NodeType
    agent_type: Optional[AgentType] = None
    function: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    parallel_group: Optional[str] = None
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    error_handler: Optional[str] = None


@dataclass
class EdgeDefinition:
    """Definition of a workflow edge."""
    from_node: str
    to_node: str
    edge_type: EdgeType
    condition: Optional[Callable] = None
    condition_map: Optional[Dict[str, str]] = None
    weight: float = 1.0


@dataclass
class WorkflowTemplate:
    """Template for creating campaign workflows."""
    template_id: str
    name: str
    description: str
    campaign_types: List[CampaignType]
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    entry_point: str
    end_points: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CampaignWorkflowBuilder:
    """
    Builds dynamic LangGraph workflows for campaign orchestration.
    
    This builder creates workflows that integrate with the existing blog_workflow.py
    architecture while adding campaign-specific orchestration capabilities.
    """
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.node_registry: Dict[str, Callable] = {}
        self.agent_registry: Dict[AgentType, Type[BaseAgent]] = {}
        self._initialize_agent_registry()
        self._initialize_builtin_templates()
    
    def _initialize_agent_registry(self):
        """Initialize agent registry with specialized agents."""
        try:
            # Import specialized agents
            from src.agents.specialized.image_agent import ImageAgent
            from src.agents.specialized.campaign_manager import CampaignManagerAgent
            from src.agents.specialized.content_repurposer import ContentRepurposer
            
            # Register agents in local registry
            self.agent_registry[AgentType.IMAGE] = ImageAgent
            self.agent_registry[AgentType.CAMPAIGN_MANAGER] = CampaignManagerAgent
            self.agent_registry[AgentType.CONTENT_REPURPOSER] = ContentRepurposer
            
            logger.info("Initialized workflow builder agent registry")
            
        except ImportError as e:
            logger.warning(f"Failed to import specialized agents for workflow builder: {e}")
        
    def _initialize_builtin_templates(self):
        """Initialize built-in workflow templates."""
        # Content Marketing Campaign Template
        content_marketing_template = WorkflowTemplate(
            template_id="content_marketing",
            name="Content Marketing Campaign",
            description="Multi-channel content creation and distribution",
            campaign_types=[CampaignType.CONTENT_MARKETING, CampaignType.BLOG_SERIES],
            nodes=[
                NodeDefinition(
                    node_id="campaign_planner",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.PLANNER,
                    config={"analysis_depth": "comprehensive"}
                ),
                NodeDefinition(
                    node_id="content_outliner", 
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.PLANNER,
                    dependencies=["campaign_planner"]
                ),
                NodeDefinition(
                    node_id="parallel_content_creation",
                    node_type=NodeType.PARALLEL,
                    parallel_group="content_creation"
                ),
                NodeDefinition(
                    node_id="blog_creator",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.WRITER,
                    parallel_group="content_creation",
                    dependencies=["content_outliner"]
                ),
                NodeDefinition(
                    node_id="social_creator",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.SOCIAL_MEDIA,
                    parallel_group="content_creation", 
                    dependencies=["content_outliner"]
                ),
                NodeDefinition(
                    node_id="email_creator",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.CONTENT_GENERATOR,
                    parallel_group="content_creation",
                    dependencies=["content_outliner"]
                ),
                NodeDefinition(
                    node_id="image_creator",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.IMAGE,
                    parallel_group="content_creation",
                    dependencies=["content_outliner"],
                    config={"style": "professional", "count": 3}
                ),
                NodeDefinition(
                    node_id="content_merger",
                    node_type=NodeType.MERGE,
                    dependencies=["blog_creator", "social_creator", "email_creator", "image_creator"]
                ),
                NodeDefinition(
                    node_id="campaign_optimizer",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.CONTENT_OPTIMIZER,
                    dependencies=["content_merger"]
                )
            ],
            edges=[
                EdgeDefinition("campaign_planner", "content_outliner", EdgeType.SEQUENTIAL),
                EdgeDefinition("content_outliner", "parallel_content_creation", EdgeType.PARALLEL),
                EdgeDefinition("parallel_content_creation", "blog_creator", EdgeType.PARALLEL),
                EdgeDefinition("parallel_content_creation", "social_creator", EdgeType.PARALLEL),
                EdgeDefinition("parallel_content_creation", "email_creator", EdgeType.PARALLEL),
                EdgeDefinition("parallel_content_creation", "image_creator", EdgeType.PARALLEL),
                EdgeDefinition("blog_creator", "content_merger", EdgeType.SEQUENTIAL),
                EdgeDefinition("social_creator", "content_merger", EdgeType.SEQUENTIAL),
                EdgeDefinition("email_creator", "content_merger", EdgeType.SEQUENTIAL),
                EdgeDefinition("image_creator", "content_merger", EdgeType.SEQUENTIAL),
                EdgeDefinition("content_merger", "campaign_optimizer", EdgeType.SEQUENTIAL)
            ],
            entry_point="campaign_planner",
            end_points=["campaign_optimizer"]
        )
        
        # Blog Series Template (Enhanced from existing blog_workflow.py)
        blog_series_template = WorkflowTemplate(
            template_id="enhanced_blog_series",
            name="Enhanced Blog Series Creation",
            description="Enhanced version of blog_workflow.py with campaign integration",
            campaign_types=[CampaignType.BLOG_SERIES, CampaignType.SEO_CONTENT],
            nodes=[
                NodeDefinition(
                    node_id="campaign_analyzer",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.PLANNER,
                    config={"integration_mode": "blog_workflow"}
                ),
                NodeDefinition(
                    node_id="outliner",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.PLANNER,
                    function="outliner_node",  # Reference to existing function
                    dependencies=["campaign_analyzer"]
                ),
                NodeDefinition(
                    node_id="geo_optimizer",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.CONTENT_OPTIMIZER,
                    function="geo_optimizer_node",  # Reference to existing function
                    dependencies=["outliner"]
                ),
                NodeDefinition(
                    node_id="researcher",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.RESEARCHER,
                    function="researcher_node",  # Reference to existing function
                    dependencies=["geo_optimizer"]
                ),
                NodeDefinition(
                    node_id="writer",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.WRITER,
                    function="writer_node",  # Reference to existing function
                    dependencies=["researcher"],
                    retry_count=2
                ),
                NodeDefinition(
                    node_id="editor",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.EDITOR,
                    function="editor_node",  # Reference to existing function
                    dependencies=["writer"]
                ),
                NodeDefinition(
                    node_id="image_generator",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.IMAGE,
                    dependencies=["writer"],
                    config={"style": "professional", "count": 2}
                ),
                NodeDefinition(
                    node_id="content_finalizer",
                    node_type=NodeType.MERGE,
                    dependencies=["editor", "image_generator"]
                ),
                NodeDefinition(
                    node_id="campaign_integrator",
                    node_type=NodeType.AGENT,
                    agent_type=AgentType.CAMPAIGN_MANAGER,
                    dependencies=["content_finalizer"]
                )
            ],
            edges=[
                EdgeDefinition("campaign_analyzer", "outliner", EdgeType.SEQUENTIAL),
                EdgeDefinition("outliner", "geo_optimizer", EdgeType.SEQUENTIAL),
                EdgeDefinition("geo_optimizer", "researcher", EdgeType.SEQUENTIAL),
                EdgeDefinition("researcher", "writer", EdgeType.SEQUENTIAL),
                EdgeDefinition("writer", "editor", EdgeType.SEQUENTIAL),
                EdgeDefinition("writer", "image_generator", EdgeType.SEQUENTIAL),
                EdgeDefinition("editor", "content_finalizer", EdgeType.SEQUENTIAL),
                EdgeDefinition("image_generator", "content_finalizer", EdgeType.SEQUENTIAL),
                EdgeDefinition("content_finalizer", "campaign_integrator", EdgeType.SEQUENTIAL),
                EdgeDefinition("editor", "writer", EdgeType.CONDITIONAL, 
                             condition_map={"writer": "writer", "content_finalizer": "content_finalizer"})
            ],
            entry_point="campaign_analyzer",
            end_points=["campaign_integrator"]
        )
        
        self.templates["content_marketing"] = content_marketing_template
        self.templates["enhanced_blog_series"] = blog_series_template
        
    def register_template(self, template: WorkflowTemplate):
        """Register a new workflow template."""
        self.templates[template.template_id] = template
        logger.info(f"Registered workflow template: {template.template_id}")
        
    def register_node_function(self, node_id: str, function: Callable):
        """Register a node function for workflow execution."""
        self.node_registry[node_id] = function
        logger.info(f"Registered node function: {node_id}")
        
    def register_agent_type(self, agent_type: AgentType, agent_class: Type[BaseAgent]):
        """Register an agent class for workflow node creation."""
        self.agent_registry[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
        
    def build_workflow(
        self,
        template_id: str,
        campaign_id: str,
        config: Optional[Dict[str, Any]] = None,
        state_class: Optional[Type] = None
    ) -> CompiledStateGraph:
        """
        Build a LangGraph workflow from a template.
        
        Args:
            template_id: ID of the workflow template to use
            campaign_id: ID of the campaign this workflow serves
            config: Additional configuration for the workflow
            state_class: Custom state class (defaults to template's state class)
            
        Returns:
            Compiled LangGraph workflow ready for execution
        """
        if template_id not in self.templates:
            raise ValueError(f"Unknown template: {template_id}")
            
        template = self.templates[template_id]
        config = config or {}
        
        # Import state class from enhanced workflow state
        from .enhanced_workflow_state import CampaignWorkflowState
        if state_class is None:
            state_class = CampaignWorkflowState
            
        # Create the workflow graph
        workflow = StateGraph(state_class)
        
        # Add nodes
        for node_def in template.nodes:
            node_function = self._create_node_function(node_def, campaign_id, config)
            workflow.add_node(node_def.node_id, node_function)
            logger.debug(f"Added node: {node_def.node_id}")
            
        # Set entry point
        workflow.set_entry_point(template.entry_point)
        
        # Add edges
        for edge_def in template.edges:
            if edge_def.edge_type == EdgeType.SEQUENTIAL:
                workflow.add_edge(edge_def.from_node, edge_def.to_node)
            elif edge_def.edge_type == EdgeType.CONDITIONAL:
                if edge_def.condition:
                    workflow.add_conditional_edges(
                        edge_def.from_node,
                        edge_def.condition,
                        edge_def.condition_map or {}
                    )
                elif edge_def.condition_map:
                    # Use default condition function that checks for continuation
                    condition_func = self._create_conditional_function(edge_def)
                    workflow.add_conditional_edges(
                        edge_def.from_node,
                        condition_func,
                        edge_def.condition_map
                    )
            elif edge_def.edge_type == EdgeType.PARALLEL:
                # Parallel edges are handled during node creation
                workflow.add_edge(edge_def.from_node, edge_def.to_node)
                
        # Compile and return the workflow
        compiled_workflow = workflow.compile()
        logger.info(f"Built workflow for template: {template_id}, campaign: {campaign_id}")
        
        return compiled_workflow
        
    def _create_node_function(
        self,
        node_def: NodeDefinition,
        campaign_id: str,
        config: Dict[str, Any]
    ) -> Callable:
        """Create a function for a workflow node."""
        
        if node_def.function:
            # Use existing function (e.g., from blog_workflow.py)
            if isinstance(node_def.function, str):
                if node_def.function in self.node_registry:
                    return self.node_registry[node_def.function]
                else:
                    # Try to import from blog_workflow
                    try:
                        from src.agents.workflow.blog_workflow import (
                            outliner_node, geo_optimizer_node, researcher_node,
                            writer_node, editor_node, should_continue
                        )
                        function_map = {
                            "outliner_node": outliner_node,
                            "geo_optimizer_node": geo_optimizer_node,
                            "researcher_node": researcher_node,
                            "writer_node": writer_node,
                            "editor_node": editor_node,
                            "should_continue": should_continue
                        }
                        if node_def.function in function_map:
                            return function_map[node_def.function]
                    except ImportError:
                        logger.warning(f"Could not import function: {node_def.function}")
            else:
                return node_def.function
                
        # Create agent-based node function
        def agent_node_function(state):
            """Dynamic agent node function."""
            try:
                # Add campaign context to state
                enhanced_state = dict(state)
                enhanced_state['campaign_id'] = campaign_id
                enhanced_state['node_config'] = node_def.config
                enhanced_state['node_id'] = node_def.node_id
                
                if node_def.agent_type:
                    # Create agent instance
                    if node_def.agent_type in self.agent_registry:
                        agent_class = self.agent_registry[node_def.agent_type]
                        agent = agent_class()
                        result = agent.execute(enhanced_state)
                        
                        # Update state with result
                        if hasattr(result, 'data') and result.data:
                            enhanced_state.update(result.data)
                            
                        return enhanced_state
                    else:
                        logger.warning(f"No agent registered for type: {node_def.agent_type}")
                        return state
                else:
                    # Handle special node types
                    if node_def.node_type == NodeType.CHECKPOINT:
                        return self._handle_checkpoint_node(enhanced_state)
                    elif node_def.node_type == NodeType.MERGE:
                        return self._handle_merge_node(enhanced_state)
                    elif node_def.node_type == NodeType.PARALLEL:
                        return enhanced_state  # Parallel handling is done by LangGraph
                        
                return enhanced_state
                
            except Exception as e:
                logger.error(f"Error in node {node_def.node_id}: {str(e)}")
                error_state = dict(state)
                error_state['error'] = str(e)
                error_state['failed_node'] = node_def.node_id
                return error_state
                
        return agent_node_function
        
    def _create_conditional_function(self, edge_def: EdgeDefinition) -> Callable:
        """Create a conditional function for workflow edges."""
        
        def conditional_function(state):
            """Dynamic conditional function."""
            try:
                # Default condition logic - check for errors or completion signals
                if state.get('error'):
                    return 'error' if 'error' in edge_def.condition_map else END
                    
                if state.get('review_notes'):
                    return 'writer' if 'writer' in edge_def.condition_map else END
                    
                if state.get('final_post') or state.get('campaign_complete'):
                    return END
                    
                # Check for custom continuation signals
                continue_to = state.get('continue_to')
                if continue_to and continue_to in edge_def.condition_map:
                    return continue_to
                    
                return END
                
            except Exception as e:
                logger.error(f"Error in conditional function: {str(e)}")
                return END
                
        return conditional_function
        
    def _handle_checkpoint_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle checkpoint node for state persistence."""
        from .campaign_state_manager import CampaignStateManager
        
        state_manager = CampaignStateManager()
        state_manager.save_checkpoint(
            campaign_id=state['campaign_id'],
            checkpoint_id=state['node_id'],
            state_data=state
        )
        
        state['checkpoint_saved'] = True
        state['last_checkpoint'] = state['node_id']
        return state
        
    def _handle_merge_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle merge node for combining parallel execution results."""
        # Merge logic for combining results from parallel nodes
        merged_content = []
        
        if 'blog_content' in state:
            merged_content.append(state['blog_content'])
        if 'social_content' in state:
            merged_content.append(state['social_content'])
        if 'email_content' in state:
            merged_content.append(state['email_content'])
        if 'generated_images' in state:
            merged_content.append({
                'type': 'images',
                'content': state['generated_images'],
                'prompts': state.get('image_prompts', [])
            })
            
        state['merged_content'] = merged_content
        state['merge_complete'] = True
        
        # Create comprehensive final content structure
        state['final_deliverable'] = {
            'text_content': {
                'blog': state.get('blog_content', ''),
                'social': state.get('social_content', ''),
                'email': state.get('email_content', '')
            },
            'visual_content': {
                'images': state.get('generated_images', []),
                'prompts': state.get('image_prompts', [])
            },
            'metadata': {
                'merge_timestamp': state.get('merge_complete'),
                'content_types': [content.get('type', 'text') if isinstance(content, dict) else 'text' for content in merged_content]
            }
        }
        
        return state
        
    def get_available_templates(self) -> List[WorkflowTemplate]:
        """Get list of available workflow templates."""
        return list(self.templates.values())
        
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific workflow template."""
        return self.templates.get(template_id)
        
    def create_custom_template(
        self,
        template_id: str,
        name: str,
        description: str,
        campaign_types: List[CampaignType],
        nodes: List[NodeDefinition],
        edges: List[EdgeDefinition],
        entry_point: str
    ) -> WorkflowTemplate:
        """Create a custom workflow template."""
        template = WorkflowTemplate(
            template_id=template_id,
            name=name,
            description=description,
            campaign_types=campaign_types,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point
        )
        
        self.register_template(template)
        return template