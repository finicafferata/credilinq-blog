"""
LangGraph-enhanced Campaign Manager Workflow for intelligent strategic campaign orchestration and management.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
import uuid

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None  # PostgreSQL checkpointing not available
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
from .campaign_manager import CampaignManagerAgent, CampaignStrategy, CampaignTask
# from ...config.database import DatabaseConnection  # Temporarily disabled


class CampaignPhase(Enum):
    """Campaign execution phases."""
    STRATEGY_DEVELOPMENT = "strategy_development"
    CONTENT_PLANNING = "content_planning" 
    RESOURCE_ALLOCATION = "resource_allocation"
    EXECUTION_PLANNING = "execution_planning"
    QUALITY_ASSURANCE = "quality_assurance"
    LAUNCH_PREPARATION = "launch_preparation"


class CampaignObjective(Enum):
    """Campaign objectives and goals."""
    BRAND_AWARENESS = "brand_awareness"
    LEAD_GENERATION = "lead_generation"
    THOUGHT_LEADERSHIP = "thought_leadership"
    CUSTOMER_ENGAGEMENT = "customer_engagement"
    PRODUCT_LAUNCH = "product_launch"
    CUSTOMER_RETENTION = "customer_retention"


class CampaignComplexity(Enum):
    """Campaign complexity levels."""
    SIMPLE = "simple"          # Single channel, basic content
    MODERATE = "moderate"      # Multi-channel, standard content mix
    COMPLEX = "complex"        # Advanced multi-channel, diverse content
    ENTERPRISE = "enterprise"  # Full-scale enterprise campaign


@dataclass
class CampaignIntelligence:
    """AI-powered campaign intelligence data."""
    market_analysis: Dict[str, Any] = field(default_factory=dict)
    competitor_insights: Dict[str, Any] = field(default_factory=dict)
    audience_personas: List[Dict[str, Any]] = field(default_factory=list)
    content_opportunities: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)
    performance_predictions: Dict[str, float] = field(default_factory=dict)


@dataclass
class CampaignMetrics:
    """Campaign performance metrics and KPIs."""
    target_reach: int = 0
    target_engagement_rate: float = 0.0
    target_conversions: int = 0
    predicted_roi: float = 0.0
    quality_score: float = 0.0
    complexity_score: float = 0.0


@dataclass
class ResourceAllocation:
    """Campaign resource allocation tracking."""
    content_creators: int = 0
    estimated_budget: float = 0.0
    time_allocation_hours: Dict[str, int] = field(default_factory=dict)
    agent_assignments: Dict[str, List[str]] = field(default_factory=dict)
    priority_distribution: Dict[str, int] = field(default_factory=dict)


class CampaignManagerState(WorkflowState):
    """Enhanced state for campaign management workflow."""
    
    # Input parameters
    campaign_name: str = ""
    campaign_type: str = "blog_based"  # blog_based, orchestration, custom
    campaign_objective: CampaignObjective = CampaignObjective.BRAND_AWARENESS
    campaign_complexity: CampaignComplexity = CampaignComplexity.MODERATE
    
    # Source data
    blog_id: Optional[str] = None
    company_context: str = ""
    template_id: Optional[str] = None
    template_config: Dict[str, Any] = field(default_factory=dict)
    
    # Campaign intelligence
    campaign_intelligence: CampaignIntelligence = field(default_factory=CampaignIntelligence)
    strategy_analysis: Dict[str, Any] = field(default_factory=dict)
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Strategy development
    campaign_strategy: Optional[CampaignStrategy] = None
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    budget_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Task management
    campaign_tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    resource_allocation: ResourceAllocation = field(default_factory=ResourceAllocation)
    
    # Quality and optimization
    quality_checks: Dict[str, Any] = field(default_factory=dict)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    campaign_metrics: CampaignMetrics = field(default_factory=CampaignMetrics)
    success_predictions: Dict[str, float] = field(default_factory=dict)
    
    # Execution planning
    launch_readiness: Dict[str, bool] = field(default_factory=dict)
    execution_plan: Dict[str, Any] = field(default_factory=dict)
    contingency_plans: List[Dict[str, Any]] = field(default_factory=list)
    
    # Database integration
    campaign_id: Optional[str] = None
    saved_to_database: bool = False
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class CampaignManagerWorkflow(LangGraphWorkflowBase[CampaignManagerState]):
    """LangGraph workflow for advanced campaign management with AI-powered orchestration."""
    
    def __init__(
        self, 
        workflow_name: str = "campaign_manager_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = CampaignManagerAgent()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> CampaignManagerState:
        """Create initial workflow state from context."""
        campaign_type = context.get("campaign_type", "blog_based")
        
        return CampaignManagerState(
            workflow_id=context.get("workflow_id", f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            campaign_name=context.get("campaign_name", ""),
            campaign_type=campaign_type,
            campaign_objective=CampaignObjective(context.get("campaign_objective", "brand_awareness")),
            campaign_complexity=CampaignComplexity(context.get("campaign_complexity", "moderate")),
            blog_id=context.get("blog_id"),
            company_context=context.get("company_context", ""),
            template_id=context.get("template_id"),
            template_config=context.get("template_config", {}),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the campaign management workflow graph."""
        workflow = StateGraph(CampaignManagerState)
        
        # Define workflow nodes
        workflow.add_node("validate_inputs", self._validate_inputs_node)
        workflow.add_node("analyze_requirements", self._analyze_requirements_node)
        workflow.add_node("gather_intelligence", self._gather_intelligence_node)
        workflow.add_node("develop_strategy", self._develop_strategy_node)
        workflow.add_node("plan_content", self._plan_content_node)
        workflow.add_node("allocate_resources", self._allocate_resources_node)
        workflow.add_node("create_timeline", self._create_timeline_node)
        workflow.add_node("generate_tasks", self._generate_tasks_node)
        workflow.add_node("quality_assurance", self._quality_assurance_node)
        workflow.add_node("prepare_launch", self._prepare_launch_node)
        workflow.add_node("finalize_campaign", self._finalize_campaign_node)
        
        # Define workflow edges
        workflow.add_edge("validate_inputs", "analyze_requirements")
        workflow.add_edge("analyze_requirements", "gather_intelligence")
        workflow.add_edge("gather_intelligence", "develop_strategy")
        workflow.add_edge("develop_strategy", "plan_content")
        workflow.add_edge("plan_content", "allocate_resources")
        workflow.add_edge("allocate_resources", "create_timeline")
        workflow.add_edge("create_timeline", "generate_tasks")
        workflow.add_edge("generate_tasks", "quality_assurance")
        
        # Conditional routing for launch readiness
        workflow.add_conditional_edges(
            "quality_assurance",
            self._check_quality_readiness,
            {
                "revise": "develop_strategy",
                "prepare": "prepare_launch"
            }
        )
        
        workflow.add_edge("prepare_launch", "finalize_campaign")
        workflow.add_edge("finalize_campaign", END)
        
        # Set entry point
        workflow.set_entry_point("validate_inputs")
        
        return workflow
    
    async def _validate_inputs_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Validate input parameters and campaign requirements."""
        try:
            self._log_progress("Validating campaign inputs and requirements")
            
            validation_errors = []
            
            # Validate campaign name
            if not state.campaign_name or len(state.campaign_name.strip()) < 3:
                validation_errors.append("Campaign name must be at least 3 characters long")
            
            # Validate campaign type specific requirements
            if state.campaign_type == "blog_based":
                if not state.blog_id:
                    validation_errors.append("Blog ID is required for blog-based campaigns")
            elif state.campaign_type == "orchestration":
                if not state.template_config or not state.template_config.get("campaign_data"):
                    validation_errors.append("Campaign data is required for orchestration campaigns")
            
            # Validate company context
            if not state.company_context:
                state.company_context = "Professional services company"
                self._log_progress("Set default company context")
            
            # Set default complexity based on campaign type
            if state.campaign_type == "orchestration":
                content_mix = state.template_config.get("campaign_data", {}).get("success_metrics", {})
                total_content_pieces = sum(content_mix.values()) if content_mix else 0
                
                if total_content_pieces <= 5:
                    state.campaign_complexity = CampaignComplexity.SIMPLE
                elif total_content_pieces <= 15:
                    state.campaign_complexity = CampaignComplexity.MODERATE
                elif total_content_pieces <= 30:
                    state.campaign_complexity = CampaignComplexity.COMPLEX
                else:
                    state.campaign_complexity = CampaignComplexity.ENTERPRISE
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 10.0
                
                state.messages.append(HumanMessage(
                    content=f"Input validation completed. Creating {state.campaign_type} campaign '{state.campaign_name}' "
                           f"with {state.campaign_complexity.value} complexity level."
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Input validation failed: {str(e)}"
            return state
    
    async def _analyze_requirements_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Analyze campaign requirements and context."""
        try:
            self._log_progress("Analyzing campaign requirements and business context")
            
            requirements_analysis = {
                "campaign_scope": self._analyze_campaign_scope(state),
                "business_context": self._analyze_business_context(state),
                "resource_requirements": self._estimate_resource_requirements(state),
                "timeline_constraints": self._analyze_timeline_constraints(state),
                "success_criteria": self._define_success_criteria(state),
                "risk_factors": self._identify_risk_factors(state)
            }
            
            # Analyze content source if blog-based
            if state.campaign_type == "blog_based" and state.blog_id:
                blog_analysis = await self.legacy_agent._analyze_blog_content_enhanced(
                    state.blog_id, state.company_context
                )
                requirements_analysis["content_source"] = blog_analysis
                state.content_analysis = blog_analysis
            
            # Analyze campaign data if orchestration
            if state.campaign_type == "orchestration":
                campaign_data = state.template_config.get("campaign_data", {})
                requirements_analysis["orchestration_data"] = campaign_data
                state.content_analysis = {
                    "campaign_objective": campaign_data.get("campaign_objective", ""),
                    "target_market": campaign_data.get("target_market", ""),
                    "success_metrics": campaign_data.get("success_metrics", {}),
                    "channels": campaign_data.get("channels", [])
                }
            
            state.strategy_analysis = requirements_analysis
            state.progress_percentage = 20.0
            
            state.messages.append(SystemMessage(
                content=f"Requirements analysis completed. Campaign scope: {requirements_analysis['campaign_scope']['complexity']}. "
                       f"Estimated resource requirement: {requirements_analysis['resource_requirements']['complexity_level']}. "
                       f"Primary risk factors: {len(requirements_analysis['risk_factors'])}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Requirements analysis failed: {str(e)}"
            return state
    
    async def _gather_intelligence_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Gather AI-powered campaign intelligence and market insights."""
        try:
            self._log_progress("Gathering AI-powered campaign intelligence and market insights")
            
            # Gather competitive intelligence
            competitive_insights = {}
            market_opportunities = {}
            audience_personas = []
            
            try:
                if state.campaign_type == "blog_based":
                    # Use blog analysis for intelligence gathering
                    competitive_insights = await self.legacy_agent._analyze_competitive_landscape(
                        state.content_analysis
                    )
                    market_opportunities = await self.legacy_agent._analyze_market_opportunities(
                        state.content_analysis, competitive_insights
                    )
                elif state.campaign_type == "orchestration":
                    # Use campaign data for intelligence gathering
                    campaign_data = state.template_config.get("campaign_data", {})
                    competitive_insights = await self.legacy_agent._analyze_competitive_landscape_for_campaign(
                        campaign_data, state.company_context
                    )
                    market_opportunities = await self.legacy_agent._analyze_market_opportunities_for_campaign(
                        campaign_data, competitive_insights
                    )
                    # Generate enhanced personas
                    target_personas = campaign_data.get("target_personas", [])
                    if target_personas:
                        audience_personas = await self.legacy_agent._generate_ai_audience_personas(
                            target_personas, competitive_insights
                        )
            except Exception as intelligence_error:
                self._log_error(f"Intelligence gathering partially failed: {str(intelligence_error)}")
                # Use fallback intelligence data
                competitive_insights = {"opportunities": [], "threats": [], "insights": []}
                market_opportunities = {"opportunities": [], "market_size": "Unknown"}
                audience_personas = []
            
            # Enhanced content opportunity analysis
            content_opportunities = self._analyze_content_opportunities(
                state.content_analysis, competitive_insights, market_opportunities
            )
            
            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                state, competitive_insights, market_opportunities
            )
            
            # Performance predictions using AI analysis
            performance_predictions = self._predict_campaign_performance(
                state, competitive_insights, market_opportunities
            )
            
            # Compile campaign intelligence
            campaign_intelligence = CampaignIntelligence(
                market_analysis=market_opportunities,
                competitor_insights=competitive_insights,
                audience_personas=audience_personas,
                content_opportunities=content_opportunities,
                optimization_recommendations=optimization_recommendations,
                performance_predictions=performance_predictions
            )
            
            state.campaign_intelligence = campaign_intelligence
            state.progress_percentage = 35.0
            
            state.messages.append(SystemMessage(
                content=f"Campaign intelligence gathered. Market opportunities: {len(content_opportunities)}, "
                       f"Competitive insights: {len(competitive_insights.get('insights', []))}, "
                       f"Audience personas: {len(audience_personas)}, "
                       f"Optimization recommendations: {len(optimization_recommendations)}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Intelligence gathering failed: {str(e)}"
            return state
    
    async def _develop_strategy_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Develop comprehensive campaign strategy using AI intelligence."""
        try:
            self._log_progress("Developing AI-enhanced campaign strategy")
            
            # Generate campaign strategy using legacy agent with enhancements
            if state.campaign_type == "blog_based":
                if state.template_id and state.template_config:
                    strategy = await self.legacy_agent._generate_intelligent_template_strategy(
                        state.content_analysis, state.template_id, state.template_config,
                        state.campaign_intelligence.competitor_insights, 
                        state.campaign_intelligence.market_analysis
                    )
                else:
                    strategy = await self.legacy_agent._generate_ai_enhanced_strategy(
                        state.content_analysis, "blog",
                        state.campaign_intelligence.competitor_insights,
                        state.campaign_intelligence.market_analysis
                    )
            elif state.campaign_type == "orchestration":
                # Create orchestration strategy
                campaign_data = state.template_config.get("campaign_data", {})
                strategy = CampaignStrategy(
                    target_audience=campaign_data.get('target_market', 'B2B professionals'),
                    key_messages=campaign_data.get('key_messages', []),
                    distribution_channels=campaign_data.get('channels', []),
                    timeline_weeks=campaign_data.get('timeline_weeks', 4),
                    budget_allocation=campaign_data.get('budget_allocation', {
                        "content_creation": 0.5,
                        "distribution": 0.3,
                        "promotion": 0.15,
                        "analytics": 0.05
                    }),
                    success_metrics=campaign_data.get('success_metrics', {}),
                    # AI enhancements
                    market_analysis=state.campaign_intelligence.market_analysis,
                    competitor_insights=state.campaign_intelligence.competitor_insights,
                    audience_personas=state.campaign_intelligence.audience_personas,
                    content_themes=state.campaign_intelligence.content_opportunities[:5],
                    optimization_recommendations=state.campaign_intelligence.optimization_recommendations
                )
            else:
                # Custom strategy development
                strategy = await self._develop_custom_strategy(state)
            
            # Calculate budget allocation with AI optimization
            optimized_budget = self._optimize_budget_allocation(
                strategy, state.campaign_complexity, state.campaign_intelligence
            )
            strategy.budget_allocation = optimized_budget
            
            # Calculate campaign metrics
            campaign_metrics = self._calculate_campaign_metrics(
                strategy, state.campaign_intelligence, state.campaign_complexity
            )
            
            state.campaign_strategy = strategy
            state.campaign_metrics = campaign_metrics
            state.budget_allocation = optimized_budget
            state.progress_percentage = 50.0
            
            state.messages.append(SystemMessage(
                content=f"Campaign strategy developed. Target audience: {strategy.target_audience[:50]}... "
                       f"Channels: {len(strategy.distribution_channels)}, "
                       f"Timeline: {strategy.timeline_weeks} weeks, "
                       f"Predicted ROI: {campaign_metrics.predicted_roi:.1f}%"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Strategy development failed: {str(e)}"
            return state
    
    async def _plan_content_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Plan comprehensive content strategy and content mix."""
        try:
            self._log_progress("Planning content strategy and content mix")
            
            content_plan = {
                "content_themes": [],
                "content_types": {},
                "channel_strategy": {},
                "content_calendar": [],
                "quality_requirements": {},
                "content_dependencies": {}
            }
            
            # Generate content themes from intelligence
            content_plan["content_themes"] = state.campaign_intelligence.content_opportunities[:8]
            
            # Plan content types based on campaign strategy
            if state.campaign_type == "orchestration":
                # Use exact content mix from campaign data
                campaign_data = state.template_config.get("campaign_data", {})
                success_metrics = campaign_data.get("success_metrics", {})
                
                for content_type, count in success_metrics.items():
                    if count > 0:
                        content_plan["content_types"][content_type] = {
                            "count": count,
                            "priority": self._get_content_type_priority(content_type),
                            "estimated_hours": self._get_content_type_hours(content_type) * count,
                            "complexity": self._get_content_type_complexity(content_type)
                        }
            else:
                # Generate content mix for blog-based campaigns
                content_plan["content_types"] = self._generate_blog_content_mix(
                    state.campaign_strategy, state.campaign_intelligence
                )
            
            # Develop channel-specific strategies
            for channel in state.campaign_strategy.distribution_channels:
                content_plan["channel_strategy"][channel] = self._develop_channel_strategy(
                    channel, state.campaign_intelligence, state.campaign_metrics
                )
            
            # Generate quality requirements
            content_plan["quality_requirements"] = {
                "brand_consistency_score": ">90%",
                "readability_score": ">8.0",
                "engagement_prediction": ">75%",
                "seo_optimization_score": ">80%",
                "factual_accuracy": "100%"
            }
            
            # Identify content dependencies
            content_plan["content_dependencies"] = self._identify_content_dependencies(
                content_plan["content_types"]
            )
            
            # Store content plan in strategy analysis
            state.strategy_analysis["content_plan"] = content_plan
            state.progress_percentage = 60.0
            
            total_content_pieces = sum(
                item["count"] for item in content_plan["content_types"].values() 
                if isinstance(item, dict)
            )
            
            state.messages.append(SystemMessage(
                content=f"Content planning completed. Total content pieces: {total_content_pieces}, "
                       f"Content themes: {len(content_plan['content_themes'])}, "
                       f"Channels: {len(content_plan['channel_strategy'])}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content planning failed: {str(e)}"
            return state
    
    async def _allocate_resources_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Allocate resources optimally across campaign tasks and phases."""
        try:
            self._log_progress("Allocating resources optimally across campaign tasks")
            
            content_plan = state.strategy_analysis.get("content_plan", {})
            content_types = content_plan.get("content_types", {})
            
            # Calculate total resource requirements
            total_hours = sum(
                item.get("estimated_hours", 0) for item in content_types.values()
                if isinstance(item, dict)
            )
            
            # Estimate team size needed
            working_weeks = state.campaign_strategy.timeline_weeks
            hours_per_week_per_person = 30  # Assuming 30 productive hours per week
            total_available_hours = working_weeks * hours_per_week_per_person
            
            content_creators_needed = max(1, int(total_hours / total_available_hours) + 1)
            
            # Distribute time allocation by content type
            time_allocation = {}
            for content_type, details in content_types.items():
                if isinstance(details, dict):
                    time_allocation[content_type] = details.get("estimated_hours", 0)
            
            # Agent assignment optimization
            agent_assignments = {}
            for content_type in content_types.keys():
                optimal_agent = self._get_optimal_agent_for_content(
                    content_type, state.campaign_complexity
                )
                if optimal_agent not in agent_assignments:
                    agent_assignments[optimal_agent] = []
                agent_assignments[optimal_agent].append(content_type)
            
            # Priority distribution
            priority_distribution = {"high": 0, "medium": 0, "low": 0}
            for details in content_types.values():
                if isinstance(details, dict):
                    priority = details.get("priority", "medium")
                    priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            # Estimate budget based on complexity and content volume
            estimated_budget = self._estimate_campaign_budget(
                state.campaign_complexity, len(content_types), total_hours
            )
            
            # Create resource allocation
            resource_allocation = ResourceAllocation(
                content_creators=content_creators_needed,
                estimated_budget=estimated_budget,
                time_allocation_hours=time_allocation,
                agent_assignments=agent_assignments,
                priority_distribution=priority_distribution
            )
            
            state.resource_allocation = resource_allocation
            state.progress_percentage = 70.0
            
            state.messages.append(SystemMessage(
                content=f"Resource allocation completed. Content creators needed: {content_creators_needed}, "
                       f"Estimated budget: ${estimated_budget:,.2f}, "
                       f"Total hours: {total_hours}, Agent assignments: {len(agent_assignments)}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Resource allocation failed: {str(e)}"
            return state
    
    async def _create_timeline_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Create optimized campaign timeline with phase management."""
        try:
            self._log_progress("Creating optimized campaign timeline with phase management")
            
            if state.campaign_type == "orchestration":
                # Use orchestration timeline creation
                content_tasks_preview = []
                timeline = await self.legacy_agent._create_orchestration_timeline(
                    state.campaign_strategy, content_tasks_preview
                )
            else:
                # Use optimized timeline creation
                timeline = await self.legacy_agent._create_optimized_timeline(
                    state.campaign_strategy, state.campaign_intelligence.competitor_insights
                )
            
            # Enhance timeline with resource allocation insights
            for phase in timeline:
                phase_week = phase.get("week", 1)
                
                # Add resource requirements for each phase
                phase["resource_requirements"] = self._calculate_phase_resources(
                    phase_week, state.resource_allocation, state.campaign_strategy.timeline_weeks
                )
                
                # Add quality gates
                phase["quality_gates"] = self._define_phase_quality_gates(
                    phase.get("phase", "unknown"), state.campaign_complexity
                )
                
                # Add risk mitigation
                phase["risk_mitigation"] = self._identify_phase_risks(
                    phase.get("phase", "unknown"), state.strategy_analysis.get("risk_factors", [])
                )
            
            state.timeline = timeline
            state.progress_percentage = 75.0
            
            state.messages.append(SystemMessage(
                content=f"Campaign timeline created with {len(timeline)} phases over "
                       f"{state.campaign_strategy.timeline_weeks} weeks. "
                       f"Each phase includes resource requirements and quality gates."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Timeline creation failed: {str(e)}"
            return state
    
    async def _generate_tasks_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Generate detailed campaign tasks with dependencies and assignments."""
        try:
            self._log_progress("Generating detailed campaign tasks with AI optimization")
            
            campaign_tasks = []
            task_dependencies = {}
            
            if state.campaign_type == "orchestration":
                # Generate orchestration tasks with exact content mix
                campaign_data = state.template_config.get("campaign_data", {})
                content_strategy = state.strategy_analysis.get("content_plan", {})
                
                orchestration_tasks = await self.legacy_agent._generate_orchestration_content_tasks(
                    campaign_data, content_strategy
                )
                
                # Convert to campaign task format
                for task in orchestration_tasks:
                    campaign_task = {
                        "id": task.get("id", str(uuid.uuid4())),
                        "task_type": task.get("type", "content_creation"),
                        "content_type": task.get("content_type", "content"),
                        "channel": task.get("channel", "multi"),
                        "title": task.get("title", "Content Task"),
                        "description": task.get("description", ""),
                        "priority": task.get("priority", "medium"),
                        "estimated_hours": task.get("estimated_hours", 2),
                        "assigned_agent": task.get("assigned_agent", "ContentAgent"),
                        "status": task.get("status", "pending"),
                        "themes": task.get("themes", []),
                        "success_metrics": task.get("success_metrics", {}),
                        "dependencies": task.get("dependencies", []),
                        "phase": self._determine_task_phase(task, state.timeline),
                        "ai_enhanced": True
                    }
                    campaign_tasks.append(campaign_task)
                    
                    # Build dependency mapping
                    if campaign_task["dependencies"]:
                        task_dependencies[campaign_task["id"]] = campaign_task["dependencies"]
            else:
                # Generate blog-based campaign tasks
                legacy_tasks = await self.legacy_agent._generate_intelligent_tasks(
                    state.campaign_strategy, state.timeline, 
                    state.campaign_intelligence.market_analysis
                )
                
                # Convert legacy tasks to enhanced format
                for i, task in enumerate(legacy_tasks):
                    campaign_task = {
                        "id": f"task_{i+1}",
                        "task_type": task.task_type,
                        "content_type": getattr(task, 'content_type', 'content'),
                        "channel": getattr(task, 'platform', 'multi'),
                        "title": f"{task.task_type.replace('_', ' ').title()} for {getattr(task, 'platform', 'platform')}",
                        "description": f"Execute {task.task_type} for campaign",
                        "priority": task.priority,
                        "estimated_hours": task.estimated_duration_hours,
                        "assigned_agent": task.assigned_agent,
                        "status": "pending",
                        "dependencies": task.dependencies,
                        "phase": self._determine_task_phase_from_type(task.task_type),
                        "ai_enhanced": True
                    }
                    campaign_tasks.append(campaign_task)
                    
                    if campaign_task["dependencies"]:
                        task_dependencies[campaign_task["id"]] = campaign_task["dependencies"]
            
            # Add campaign management tasks
            management_tasks = self._generate_management_tasks(
                state.campaign_complexity, state.campaign_strategy.timeline_weeks
            )
            campaign_tasks.extend(management_tasks)
            
            state.campaign_tasks = campaign_tasks
            state.task_dependencies = task_dependencies
            state.progress_percentage = 85.0
            
            total_tasks = len(campaign_tasks)
            content_tasks = len([t for t in campaign_tasks if t["task_type"] == "content_creation"])
            
            state.messages.append(SystemMessage(
                content=f"Task generation completed. Total tasks: {total_tasks}, "
                       f"Content creation tasks: {content_tasks}, "
                       f"Task dependencies: {len(task_dependencies)}, "
                       f"AI-enhanced tasks: {len([t for t in campaign_tasks if t.get('ai_enhanced')])}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Task generation failed: {str(e)}"
            return state
    
    async def _quality_assurance_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Perform comprehensive quality assurance and optimization checks."""
        try:
            self._log_progress("Performing comprehensive quality assurance and optimization")
            
            quality_checks = {
                "strategy_coherence": self._check_strategy_coherence(state),
                "resource_feasibility": self._check_resource_feasibility(state),
                "timeline_realistic": self._check_timeline_realism(state),
                "task_dependencies": self._validate_task_dependencies(state),
                "budget_alignment": self._check_budget_alignment(state),
                "channel_optimization": self._check_channel_optimization(state),
                "content_diversity": self._check_content_diversity(state),
                "risk_mitigation": self._assess_risk_mitigation(state)
            }
            
            # Calculate overall quality score
            quality_scores = [check["score"] for check in quality_checks.values() if "score" in check]
            overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Identify optimization opportunities
            optimization_opportunities = []
            for check_name, check_result in quality_checks.items():
                if check_result.get("score", 100) < 80:
                    optimization_opportunities.append({
                        "area": check_name,
                        "issue": check_result.get("issue", "Quality threshold not met"),
                        "recommendation": check_result.get("recommendation", "Review and optimize"),
                        "priority": "high" if check_result.get("score", 100) < 60 else "medium"
                    })
            
            # Risk assessment
            risk_assessment = {
                "overall_risk_level": self._calculate_overall_risk(state, quality_checks),
                "critical_risks": [opp for opp in optimization_opportunities if opp["priority"] == "high"],
                "mitigation_strategies": self._generate_mitigation_strategies(optimization_opportunities),
                "contingency_required": len([opp for opp in optimization_opportunities if opp["priority"] == "high"]) > 0
            }
            
            # Update campaign metrics with quality score
            state.campaign_metrics.quality_score = overall_quality
            state.quality_checks = quality_checks
            state.optimization_opportunities = optimization_opportunities
            state.risk_assessment = risk_assessment
            
            state.progress_percentage = 95.0
            
            state.messages.append(SystemMessage(
                content=f"Quality assurance completed. Overall quality score: {overall_quality:.1f}%, "
                       f"Optimization opportunities: {len(optimization_opportunities)}, "
                       f"Critical risks: {len(risk_assessment['critical_risks'])}, "
                       f"Risk level: {risk_assessment['overall_risk_level']}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Quality assurance failed: {str(e)}"
            return state
    
    async def _prepare_launch_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Prepare campaign for launch with final checks and execution planning."""
        try:
            self._log_progress("Preparing campaign for launch with execution planning")
            
            # Launch readiness checklist
            launch_readiness = {
                "strategy_approved": state.campaign_metrics.quality_score >= 70,
                "resources_allocated": state.resource_allocation.content_creators > 0,
                "timeline_finalized": len(state.timeline) > 0,
                "tasks_defined": len(state.campaign_tasks) > 0,
                "quality_passed": len([opp for opp in state.optimization_opportunities if opp["priority"] == "high"]) == 0,
                "budget_approved": state.resource_allocation.estimated_budget > 0,
                "team_assigned": len(state.resource_allocation.agent_assignments) > 0,
                "contingency_planned": len(state.risk_assessment.get("mitigation_strategies", [])) > 0
            }
            
            # Create execution plan
            execution_plan = {
                "launch_sequence": self._create_launch_sequence(state),
                "communication_plan": self._create_communication_plan(state),
                "monitoring_strategy": self._create_monitoring_strategy(state),
                "escalation_procedures": self._create_escalation_procedures(state),
                "success_tracking": self._create_success_tracking_plan(state),
                "review_schedule": self._create_review_schedule(state)
            }
            
            # Generate contingency plans
            contingency_plans = []
            for risk in state.risk_assessment.get("critical_risks", []):
                contingency_plan = {
                    "trigger": risk["issue"],
                    "response": risk["recommendation"],
                    "owner": "Campaign Manager",
                    "timeline": "Immediate",
                    "backup_resources": self._identify_backup_resources(risk["area"])
                }
                contingency_plans.append(contingency_plan)
            
            state.launch_readiness = launch_readiness
            state.execution_plan = execution_plan
            state.contingency_plans = contingency_plans
            
            # Check overall launch readiness
            readiness_percentage = (sum(launch_readiness.values()) / len(launch_readiness)) * 100
            
            state.messages.append(SystemMessage(
                content=f"Launch preparation completed. Launch readiness: {readiness_percentage:.1f}%, "
                       f"Execution plan components: {len(execution_plan)}, "
                       f"Contingency plans: {len(contingency_plans)}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Launch preparation failed: {str(e)}"
            return state
    
    async def _finalize_campaign_node(self, state: CampaignManagerState) -> CampaignManagerState:
        """Finalize campaign and save to database."""
        try:
            self._log_progress("Finalizing campaign and saving to database")
            
            # Save campaign to database using legacy agent
            if state.campaign_type == "orchestration":
                campaign_id = await self.legacy_agent._save_orchestration_campaign_to_db(
                    state.campaign_name,
                    state.campaign_strategy,
                    state.template_config.get("campaign_data", {}),
                    state.strategy_analysis.get("content_plan", {})
                )
                
                # Save tasks for orchestration campaign
                await self.legacy_agent._save_orchestration_tasks_to_db(campaign_id, state.campaign_tasks)
                
            elif state.campaign_type == "blog_based" and state.blog_id:
                campaign_id = await self.legacy_agent._save_enhanced_campaign_to_db(
                    state.blog_id, state.campaign_name, state.campaign_strategy
                )
                
                # Convert tasks to legacy format for saving
                legacy_tasks = []
                for task in state.campaign_tasks:
                    if task["task_type"] == "content_creation":
                        legacy_task = CampaignTask(
                            task_type=task["task_type"],
                            platform=task["channel"],
                            content_type=task["content_type"],
                            priority=task["priority"],
                            estimated_duration_hours=task["estimated_hours"],
                            dependencies=task["dependencies"],
                            assigned_agent=task["assigned_agent"]
                        )
                        legacy_tasks.append(legacy_task)
                
                await self.legacy_agent._save_enhanced_tasks_to_db(campaign_id, legacy_tasks)
            else:
                # Create a custom campaign entry
                campaign_id = str(uuid.uuid4())
                self._log_progress(f"Created campaign ID for custom campaign: {campaign_id}")
            
            # Calculate final metrics
            total_tasks = len(state.campaign_tasks)
            content_tasks = len([t for t in state.campaign_tasks if t["task_type"] == "content_creation"])
            estimated_duration = sum(task["estimated_hours"] for task in state.campaign_tasks)
            
            # Update final state
            state.campaign_id = campaign_id
            state.saved_to_database = True
            state.status = WorkflowStatus.COMPLETED
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            # Calculate processing time
            processing_time = (state.completed_at - state.created_at).total_seconds()
            
            state.messages.append(SystemMessage(
                content=f"Campaign finalization completed successfully. Campaign ID: {campaign_id}. "
                       f"Total tasks: {total_tasks}, Content tasks: {content_tasks}, "
                       f"Estimated duration: {estimated_duration} hours. "
                       f"Processing time: {processing_time:.1f} seconds."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Campaign finalization failed: {str(e)}"
            return state
    
    def _check_quality_readiness(self, state: CampaignManagerState) -> str:
        """Check if campaign quality meets launch readiness criteria."""
        critical_issues = len([opp for opp in state.optimization_opportunities if opp["priority"] == "high"])
        overall_quality = state.campaign_metrics.quality_score
        
        if critical_issues > 0 or overall_quality < 70:
            return "revise"
        return "prepare"
    
    # Helper methods for campaign analysis and optimization
    
    def _analyze_campaign_scope(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Analyze campaign scope and complexity."""
        scope_indicators = {
            "channels": len(state.campaign_strategy.distribution_channels) if state.campaign_strategy else 1,
            "timeline_weeks": state.campaign_strategy.timeline_weeks if state.campaign_strategy else 4,
            "complexity_level": state.campaign_complexity.value
        }
        
        if state.campaign_type == "orchestration":
            campaign_data = state.template_config.get("campaign_data", {})
            content_count = sum(campaign_data.get("success_metrics", {}).values())
            scope_indicators["content_pieces"] = content_count
            
            if content_count > 20:
                scope_indicators["complexity"] = "high"
            elif content_count > 10:
                scope_indicators["complexity"] = "medium"
            else:
                scope_indicators["complexity"] = "low"
        else:
            scope_indicators["complexity"] = "medium"  # Default for blog-based
        
        return scope_indicators
    
    def _analyze_business_context(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Analyze business context and industry factors."""
        return {
            "industry": self._extract_industry_from_context(state.company_context),
            "company_size": self._estimate_company_size(state.company_context),
            "market_maturity": "established",  # Default assumption
            "competitive_landscape": "moderate",  # Default assumption
            "business_objectives": [state.campaign_objective.value]
        }
    
    def _estimate_resource_requirements(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Estimate resource requirements based on campaign scope."""
        base_hours = {
            CampaignComplexity.SIMPLE: 20,
            CampaignComplexity.MODERATE: 50,
            CampaignComplexity.COMPLEX: 100,
            CampaignComplexity.ENTERPRISE: 200
        }
        
        estimated_hours = base_hours.get(state.campaign_complexity, 50)
        
        if state.campaign_type == "orchestration":
            campaign_data = state.template_config.get("campaign_data", {})
            content_count = sum(campaign_data.get("success_metrics", {}).values())
            estimated_hours = content_count * 3  # 3 hours per content piece average
        
        return {
            "estimated_total_hours": estimated_hours,
            "team_size_needed": max(1, estimated_hours // 40),  # Assuming 40 hours per person
            "complexity_level": state.campaign_complexity.value,
            "specialized_skills": self._identify_specialized_skills(state)
        }
    
    def _analyze_timeline_constraints(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Analyze timeline constraints and scheduling factors."""
        weeks = state.campaign_strategy.timeline_weeks if state.campaign_strategy else 4
        
        return {
            "total_weeks": weeks,
            "urgency_level": "high" if weeks < 2 else "medium" if weeks < 4 else "normal",
            "holiday_considerations": [],  # Could be enhanced with calendar integration
            "resource_availability": "standard",  # Default assumption
            "external_dependencies": []  # Could be enhanced with dependency analysis
        }
    
    def _define_success_criteria(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Define success criteria based on campaign objectives."""
        base_criteria = {
            "content_quality_score": ">80%",
            "deadline_adherence": "100%",
            "resource_efficiency": ">90%"
        }
        
        # Add objective-specific criteria
        if state.campaign_objective == CampaignObjective.BRAND_AWARENESS:
            base_criteria.update({
                "reach_target": ">10,000 impressions",
                "brand_mention_increase": ">25%"
            })
        elif state.campaign_objective == CampaignObjective.LEAD_GENERATION:
            base_criteria.update({
                "lead_conversion_rate": ">3%",
                "cost_per_lead": "<$50"
            })
        elif state.campaign_objective == CampaignObjective.THOUGHT_LEADERSHIP:
            base_criteria.update({
                "engagement_rate": ">5%",
                "share_rate": ">2%"
            })
        
        return base_criteria
    
    def _identify_risk_factors(self, state: CampaignManagerState) -> List[Dict[str, Any]]:
        """Identify potential risk factors for the campaign."""
        risks = []
        
        # Timeline risks
        if state.campaign_strategy and state.campaign_strategy.timeline_weeks < 3:
            risks.append({
                "category": "timeline",
                "risk": "Compressed timeline may impact content quality",
                "probability": "medium",
                "impact": "high"
            })
        
        # Resource risks
        if state.campaign_complexity == CampaignComplexity.ENTERPRISE:
            risks.append({
                "category": "resources", 
                "risk": "High complexity may require additional specialized resources",
                "probability": "high",
                "impact": "medium"
            })
        
        # Content risks
        if state.campaign_type == "orchestration":
            campaign_data = state.template_config.get("campaign_data", {})
            content_count = sum(campaign_data.get("success_metrics", {}).values())
            if content_count > 25:
                risks.append({
                    "category": "content",
                    "risk": "High content volume may impact consistency and quality",
                    "probability": "medium",
                    "impact": "medium"
                })
        
        return risks
    
    def _analyze_content_opportunities(self, content_analysis: Dict[str, Any], 
                                     competitive_insights: Dict[str, Any],
                                     market_opportunities: Dict[str, Any]) -> List[str]:
        """Analyze content opportunities from various intelligence sources."""
        opportunities = []
        
        # From content analysis
        if "key_themes" in content_analysis.get("analysis", {}):
            themes = content_analysis["analysis"]["key_themes"]
            opportunities.extend([f"Explore {theme} in depth" for theme in themes[:3]])
        
        # From competitive insights
        if "opportunities" in competitive_insights:
            opportunities.extend(competitive_insights["opportunities"][:3])
        
        # From market opportunities
        if "opportunities" in market_opportunities:
            opportunities.extend(market_opportunities["opportunities"][:3])
        
        # Default opportunities if none found
        if not opportunities:
            opportunities = [
                "Industry trend analysis",
                "Customer success stories", 
                "Expert insights and commentary",
                "Educational content series",
                "Behind-the-scenes content"
            ]
        
        return opportunities[:8]  # Limit to top 8 opportunities
    
    def _generate_optimization_recommendations(self, state: CampaignManagerState,
                                             competitive_insights: Dict[str, Any],
                                             market_opportunities: Dict[str, Any]) -> List[str]:
        """Generate AI-powered optimization recommendations."""
        recommendations = []
        
        # Complexity-based recommendations
        if state.campaign_complexity == CampaignComplexity.ENTERPRISE:
            recommendations.append("Implement phased rollout to manage complexity")
            recommendations.append("Establish dedicated quality review process")
        
        # Timeline-based recommendations
        if state.campaign_strategy and state.campaign_strategy.timeline_weeks < 3:
            recommendations.append("Prioritize high-impact content for immediate creation")
            recommendations.append("Consider parallel content development streams")
        
        # Channel-based recommendations
        if state.campaign_strategy and len(state.campaign_strategy.distribution_channels) > 3:
            recommendations.append("Implement cross-channel content repurposing strategy")
            recommendations.append("Optimize posting schedules for each channel")
        
        # Intelligence-based recommendations
        if competitive_insights.get("opportunities"):
            recommendations.append("Leverage identified competitive gaps for differentiation")
        
        if market_opportunities.get("growth_potential", "").lower() == "high":
            recommendations.append("Focus on high-growth market segments")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Implement data-driven content optimization",
                "Focus on audience engagement metrics",
                "Maintain consistent brand messaging across channels"
            ]
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def _predict_campaign_performance(self, state: CampaignManagerState,
                                    competitive_insights: Dict[str, Any],
                                    market_opportunities: Dict[str, Any]) -> Dict[str, float]:
        """Predict campaign performance using AI analysis."""
        # Base predictions
        predictions = {
            "reach_prediction": 5000.0,
            "engagement_rate_prediction": 3.5,
            "conversion_rate_prediction": 2.0,
            "roi_prediction": 150.0,
            "quality_score_prediction": 75.0
        }
        
        # Adjust based on complexity
        complexity_multiplier = {
            CampaignComplexity.SIMPLE: 0.8,
            CampaignComplexity.MODERATE: 1.0,
            CampaignComplexity.COMPLEX: 1.3,
            CampaignComplexity.ENTERPRISE: 1.8
        }
        
        multiplier = complexity_multiplier.get(state.campaign_complexity, 1.0)
        predictions["reach_prediction"] *= multiplier
        
        # Adjust based on timeline
        if state.campaign_strategy and state.campaign_strategy.timeline_weeks >= 4:
            predictions["quality_score_prediction"] *= 1.1  # More time = better quality
        
        # Adjust based on channel count
        if state.campaign_strategy and len(state.campaign_strategy.distribution_channels) > 2:
            predictions["reach_prediction"] *= 1.2
            predictions["engagement_rate_prediction"] *= 0.9  # Spread across channels
        
        return predictions
    
    async def _develop_custom_strategy(self, state: CampaignManagerState) -> CampaignStrategy:
        """Develop custom campaign strategy for non-standard campaigns."""
        return CampaignStrategy(
            target_audience="Business professionals and decision makers",
            key_messages=["Industry expertise", "Value-driven solutions", "Innovation leadership"],
            distribution_channels=["linkedin", "email", "blog"],
            timeline_weeks=4,
            budget_allocation={
                "content_creation": 0.5,
                "distribution": 0.3,
                "promotion": 0.15,
                "analytics": 0.05
            },
            success_metrics={
                "reach": 10000,
                "engagement_rate": 0.05,
                "conversions": 100
            },
            # AI enhancements
            market_analysis=state.campaign_intelligence.market_analysis,
            competitor_insights=state.campaign_intelligence.competitor_insights,
            audience_personas=state.campaign_intelligence.audience_personas,
            content_themes=state.campaign_intelligence.content_opportunities[:5],
            optimization_recommendations=state.campaign_intelligence.optimization_recommendations
        )
    
    def _optimize_budget_allocation(self, strategy: CampaignStrategy, 
                                  complexity: CampaignComplexity,
                                  intelligence: CampaignIntelligence) -> Dict[str, float]:
        """Optimize budget allocation based on campaign characteristics."""
        base_allocation = strategy.budget_allocation.copy()
        
        # Adjust based on complexity
        if complexity == CampaignComplexity.ENTERPRISE:
            # More budget for coordination and quality assurance
            base_allocation["content_creation"] *= 0.9
            base_allocation["analytics"] *= 1.5
        elif complexity == CampaignComplexity.SIMPLE:
            # More budget for content, less for complex analytics
            base_allocation["content_creation"] *= 1.1
            base_allocation["analytics"] *= 0.7
        
        # Adjust based on channel count
        channel_count = len(strategy.distribution_channels)
        if channel_count > 3:
            # More budget for distribution
            base_allocation["distribution"] *= 1.2
            base_allocation["content_creation"] *= 0.9
        
        # Normalize to ensure total equals 1.0
        total = sum(base_allocation.values())
        return {k: v / total for k, v in base_allocation.items()}
    
    def _calculate_campaign_metrics(self, strategy: CampaignStrategy,
                                   intelligence: CampaignIntelligence,
                                   complexity: CampaignComplexity) -> CampaignMetrics:
        """Calculate comprehensive campaign metrics."""
        # Base metrics from performance predictions
        performance_predictions = intelligence.performance_predictions
        
        return CampaignMetrics(
            target_reach=int(performance_predictions.get("reach_prediction", 5000)),
            target_engagement_rate=performance_predictions.get("engagement_rate_prediction", 3.5),
            target_conversions=int(performance_predictions.get("conversion_rate_prediction", 50)),
            predicted_roi=performance_predictions.get("roi_prediction", 150.0),
            quality_score=performance_predictions.get("quality_score_prediction", 75.0),
            complexity_score=self._calculate_complexity_score(complexity, strategy)
        )
    
    def _calculate_complexity_score(self, complexity: CampaignComplexity, 
                                  strategy: CampaignStrategy) -> float:
        """Calculate campaign complexity score."""
        base_scores = {
            CampaignComplexity.SIMPLE: 25.0,
            CampaignComplexity.MODERATE: 50.0,
            CampaignComplexity.COMPLEX: 75.0,
            CampaignComplexity.ENTERPRISE: 95.0
        }
        
        base_score = base_scores.get(complexity, 50.0)
        
        # Adjust based on strategy elements
        channel_adjustment = len(strategy.distribution_channels) * 5
        timeline_adjustment = max(0, (8 - strategy.timeline_weeks) * 3)  # Shorter timeline = more complex
        
        return min(100.0, base_score + channel_adjustment + timeline_adjustment)
    
    async def execute_workflow(
        self,
        campaign_name: str,
        campaign_type: str = "blog_based",
        campaign_objective: CampaignObjective = CampaignObjective.BRAND_AWARENESS,
        campaign_complexity: CampaignComplexity = CampaignComplexity.MODERATE,
        blog_id: Optional[str] = None,
        company_context: str = "",
        template_id: Optional[str] = None,
        template_config: Optional[Dict[str, Any]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the campaign management workflow."""
        
        context = {
            "campaign_name": campaign_name,
            "campaign_type": campaign_type,
            "campaign_objective": campaign_objective.value if isinstance(campaign_objective, CampaignObjective) else campaign_objective,
            "campaign_complexity": campaign_complexity.value if isinstance(campaign_complexity, CampaignComplexity) else campaign_complexity,
            "blog_id": blog_id,
            "company_context": company_context,
            "template_id": template_id,
            "template_config": template_config or {},
            "workflow_id": f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "campaign_id": final_state.campaign_id,
                    "campaign_strategy": {
                        "target_audience": final_state.campaign_strategy.target_audience,
                        "key_messages": final_state.campaign_strategy.key_messages,
                        "distribution_channels": final_state.campaign_strategy.distribution_channels,
                        "timeline_weeks": final_state.campaign_strategy.timeline_weeks,
                        "budget_allocation": final_state.campaign_strategy.budget_allocation,
                        "success_metrics": final_state.campaign_strategy.success_metrics,
                        # AI enhancements
                        "market_analysis": final_state.campaign_strategy.market_analysis,
                        "competitor_insights": final_state.campaign_strategy.competitor_insights,
                        "audience_personas": final_state.campaign_strategy.audience_personas,
                        "content_themes": final_state.campaign_strategy.content_themes,
                        "optimization_recommendations": final_state.campaign_strategy.optimization_recommendations
                    },
                    "campaign_tasks": [
                        {
                            "id": task["id"],
                            "task_type": task["task_type"],
                            "content_type": task["content_type"],
                            "channel": task["channel"],
                            "title": task["title"],
                            "description": task["description"],
                            "priority": task["priority"],
                            "estimated_hours": task["estimated_hours"],
                            "assigned_agent": task["assigned_agent"],
                            "status": task["status"],
                            "dependencies": task["dependencies"],
                            "phase": task.get("phase"),
                            "ai_enhanced": task.get("ai_enhanced", False)
                        }
                        for task in final_state.campaign_tasks
                    ],
                    "timeline": final_state.timeline,
                    "resource_allocation": {
                        "content_creators": final_state.resource_allocation.content_creators,
                        "estimated_budget": final_state.resource_allocation.estimated_budget,
                        "time_allocation_hours": final_state.resource_allocation.time_allocation_hours,
                        "agent_assignments": final_state.resource_allocation.agent_assignments,
                        "priority_distribution": final_state.resource_allocation.priority_distribution
                    },
                    "campaign_metrics": {
                        "target_reach": final_state.campaign_metrics.target_reach,
                        "target_engagement_rate": final_state.campaign_metrics.target_engagement_rate,
                        "target_conversions": final_state.campaign_metrics.target_conversions,
                        "predicted_roi": final_state.campaign_metrics.predicted_roi,
                        "quality_score": final_state.campaign_metrics.quality_score,
                        "complexity_score": final_state.campaign_metrics.complexity_score
                    },
                    "quality_assessment": {
                        "overall_quality_score": final_state.campaign_metrics.quality_score,
                        "optimization_opportunities": final_state.optimization_opportunities,
                        "risk_assessment": final_state.risk_assessment,
                        "launch_readiness": final_state.launch_readiness
                    },
                    "execution_plan": final_state.execution_plan,
                    "workflow_summary": {
                        "campaign_type": final_state.campaign_type,
                        "complexity": final_state.campaign_complexity.value,
                        "total_tasks": len(final_state.campaign_tasks),
                        "processing_time_seconds": (final_state.completed_at - final_state.created_at).total_seconds(),
                        "saved_to_database": final_state.saved_to_database
                    }
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "campaign_id": final_state.campaign_id,
                        "campaign_complexity": final_state.campaign_complexity.value,
                        "quality_score": final_state.campaign_metrics.quality_score
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Workflow failed",
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "final_status": final_state.status.value
                    }
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=f"Workflow execution failed: {str(e)}",
                metadata={"error_type": "workflow_execution_error"}
            )
    
    # Additional helper methods for campaign management
    
    def _get_content_type_priority(self, content_type: str) -> str:
        """Get priority level for content type."""
        high_priority = ["blog_posts", "case_studies", "whitepapers", "webinars"]
        medium_priority = ["social_posts", "infographics", "video_content"]
        
        if content_type in high_priority:
            return "high"
        elif content_type in medium_priority:
            return "medium"
        else:
            return "low"
    
    def _get_content_type_hours(self, content_type: str) -> int:
        """Get estimated hours for content type."""
        time_estimates = {
            'social_posts': 1,
            'blog_posts': 4,
            'email_content': 2,
            'video_content': 6,
            'infographics': 3,
            'case_studies': 6,
            'whitepapers': 8,
            'webinars': 10,
            'ebooks': 12,
            'seo_optimization': 2,
            'competitor_analysis': 3,
            'image_generation': 1,
            'repurposed_content': 1,
            'performance_analytics': 2
        }
        return time_estimates.get(content_type, 2)
    
    def _get_content_type_complexity(self, content_type: str) -> str:
        """Get complexity level for content type."""
        high_complexity = ["whitepapers", "webinars", "ebooks", "case_studies"]
        medium_complexity = ["blog_posts", "video_content", "infographics"]
        
        if content_type in high_complexity:
            return "high"
        elif content_type in medium_complexity:
            return "medium"
        else:
            return "low"
    
    def _generate_blog_content_mix(self, strategy: CampaignStrategy, 
                                 intelligence: CampaignIntelligence) -> Dict[str, Dict[str, Any]]:
        """Generate content mix for blog-based campaigns."""
        content_mix = {}
        
        # Base content types for blog campaigns
        for channel in strategy.distribution_channels:
            if channel == "linkedin":
                content_mix["linkedin_posts"] = {
                    "count": 3,
                    "priority": "high",
                    "estimated_hours": 3,
                    "complexity": "medium"
                }
            elif channel == "twitter":
                content_mix["twitter_threads"] = {
                    "count": 2, 
                    "priority": "medium",
                    "estimated_hours": 2,
                    "complexity": "low"
                }
            elif channel == "email":
                content_mix["email_campaigns"] = {
                    "count": 1,
                    "priority": "high", 
                    "estimated_hours": 2,
                    "complexity": "medium"
                }
        
        # Add repurposed content
        content_mix["repurposed_content"] = {
            "count": len(strategy.distribution_channels),
            "priority": "medium",
            "estimated_hours": len(strategy.distribution_channels),
            "complexity": "low"
        }
        
        return content_mix
    
    def _develop_channel_strategy(self, channel: str, intelligence: CampaignIntelligence,
                                metrics: CampaignMetrics) -> Dict[str, Any]:
        """Develop channel-specific strategy."""
        base_strategies = {
            "linkedin": {
                "content_focus": "Professional insights and thought leadership",
                "posting_frequency": "Daily",
                "optimal_times": ["8-9 AM", "12-1 PM", "5-6 PM"],
                "engagement_tactics": ["Industry discussions", "Professional networking", "Expert commentary"]
            },
            "twitter": {
                "content_focus": "Quick insights and industry commentary",
                "posting_frequency": "Multiple times daily",
                "optimal_times": ["9-10 AM", "1-3 PM", "5-6 PM"],
                "engagement_tactics": ["Hashtag utilization", "Thread conversations", "Real-time commentary"]
            },
            "email": {
                "content_focus": "Comprehensive insights and value delivery",
                "posting_frequency": "Weekly",
                "optimal_times": ["Tuesday-Thursday 10 AM"],
                "engagement_tactics": ["Personalization", "Value-packed content", "Clear CTAs"]
            },
            "blog": {
                "content_focus": "In-depth analysis and comprehensive coverage",
                "posting_frequency": "Bi-weekly",
                "optimal_times": ["Tuesday-Thursday"],
                "engagement_tactics": ["SEO optimization", "Comprehensive coverage", "Expert insights"]
            }
        }
        
        strategy = base_strategies.get(channel, {
            "content_focus": "Valuable insights and engagement",
            "posting_frequency": "Regular",
            "optimal_times": ["Business hours"],
            "engagement_tactics": ["Quality content", "Audience interaction"]
        })
        
        # Add intelligence-based enhancements
        if intelligence.content_opportunities:
            strategy["content_opportunities"] = intelligence.content_opportunities[:3]
        
        if intelligence.optimization_recommendations:
            strategy["optimization_focus"] = intelligence.optimization_recommendations[:2]
        
        return strategy
    
    def _identify_content_dependencies(self, content_types: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify dependencies between content types."""
        dependencies = {}
        
        # Blog posts often serve as source for other content
        if "blog_posts" in content_types and "social_posts" in content_types:
            dependencies["social_posts"] = ["blog_posts"]
        
        if "blog_posts" in content_types and "email_content" in content_types:
            dependencies["email_content"] = ["blog_posts"]
        
        # Infographics depend on research content
        if "infographics" in content_types and "competitor_analysis" in content_types:
            dependencies["infographics"] = ["competitor_analysis"]
        
        # Video content may depend on written content
        if "video_content" in content_types and "blog_posts" in content_types:
            dependencies["video_content"] = ["blog_posts"]
        
        return dependencies
    
    def _get_optimal_agent_for_content(self, content_type: str, complexity: CampaignComplexity) -> str:
        """Get optimal agent assignment for content type."""
        agent_mapping = {
            'blog_posts': 'WriterAgent',
            'social_posts': 'SocialMediaAgent',
            'email_content': 'ContentAgent', 
            'video_content': 'ContentAgent',
            'infographics': 'ImageAgent',
            'case_studies': 'ContentAgent',
            'whitepapers': 'ResearchAgent',
            'seo_optimization': 'SEOAgent',
            'competitor_analysis': 'SearchAgent',
            'image_generation': 'ImageAgent',
            'repurposed_content': 'ContentRepurposerAgent',
            'performance_analytics': 'AnalyticsAgent'
        }
        
        base_agent = agent_mapping.get(content_type, 'ContentAgent')
        
        # For enterprise complexity, consider senior agents
        if complexity == CampaignComplexity.ENTERPRISE and base_agent == 'ContentAgent':
            return 'SeniorContentAgent'
        
        return base_agent
    
    def _estimate_campaign_budget(self, complexity: CampaignComplexity, 
                                content_count: int, total_hours: int) -> float:
        """Estimate campaign budget based on parameters."""
        base_hourly_rate = 75  # Base rate per hour
        
        complexity_multipliers = {
            CampaignComplexity.SIMPLE: 0.8,
            CampaignComplexity.MODERATE: 1.0,
            CampaignComplexity.COMPLEX: 1.3,
            CampaignComplexity.ENTERPRISE: 1.6
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        estimated_budget = total_hours * base_hourly_rate * multiplier
        
        # Add overhead for tools, software, and coordination
        overhead = estimated_budget * 0.2
        
        return estimated_budget + overhead
    
    def _determine_task_phase(self, task: Dict[str, Any], timeline: List[Dict[str, Any]]) -> str:
        """Determine which phase a task belongs to."""
        task_type = task.get("type", "content_creation")
        
        if task_type == "content_creation":
            return "content_creation"
        elif task_type == "content_editing":
            return "content_optimization" 
        elif task_type == "distribution":
            return "campaign_execution"
        elif task_type == "analytics":
            return "performance_analysis"
        else:
            return "launch_preparation"
    
    def _determine_task_phase_from_type(self, task_type: str) -> str:
        """Determine phase from task type string."""
        phase_mapping = {
            "content_creation": "content_creation",
            "content_editing": "content_optimization",
            "distribution": "campaign_execution", 
            "analytics": "performance_analysis",
            "review": "quality_assurance"
        }
        return phase_mapping.get(task_type, "execution_planning")
    
    def _generate_management_tasks(self, complexity: CampaignComplexity, timeline_weeks: int) -> List[Dict[str, Any]]:
        """Generate campaign management and coordination tasks."""
        management_tasks = []
        
        # Always include these management tasks
        base_tasks = [
            {
                "id": "mgmt_review_1",
                "task_type": "campaign_management",
                "content_type": "review",
                "channel": "internal",
                "title": "Mid-Campaign Quality Review",
                "description": "Comprehensive quality review of all content and campaign progress",
                "priority": "high",
                "estimated_hours": 3,
                "assigned_agent": "CampaignManager",
                "status": "pending",
                "dependencies": [],
                "phase": "quality_assurance"
            },
            {
                "id": "mgmt_analytics_1", 
                "task_type": "performance_analysis",
                "content_type": "analytics",
                "channel": "all",
                "title": "Campaign Performance Analysis",
                "description": "Analyze campaign performance metrics and optimization opportunities",
                "priority": "medium",
                "estimated_hours": 4,
                "assigned_agent": "AnalyticsAgent",
                "status": "pending",
                "dependencies": [],
                "phase": "performance_analysis"
            }
        ]
        
        # Add complexity-based tasks
        if complexity in [CampaignComplexity.COMPLEX, CampaignComplexity.ENTERPRISE]:
            base_tasks.append({
                "id": "mgmt_coordination_1",
                "task_type": "coordination",
                "content_type": "management",
                "channel": "internal",
                "title": "Cross-Team Coordination",
                "description": "Coordinate activities across multiple content creation teams",
                "priority": "high",
                "estimated_hours": 2 * timeline_weeks,  # Weekly coordination
                "assigned_agent": "CampaignManager", 
                "status": "pending",
                "dependencies": [],
                "phase": "execution_planning"
            })
        
        return base_tasks
    
    # Quality assurance helper methods
    
    def _check_strategy_coherence(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Check if campaign strategy is coherent and well-aligned."""
        if not state.campaign_strategy:
            return {"score": 0, "issue": "No strategy defined", "recommendation": "Develop campaign strategy"}
        
        strategy = state.campaign_strategy
        score = 100
        issues = []
        
        # Check if key messages align with objectives
        if len(strategy.key_messages) < 2:
            score -= 20
            issues.append("Insufficient key messages")
        
        # Check channel alignment
        if len(strategy.distribution_channels) == 0:
            score -= 30
            issues.append("No distribution channels defined")
        
        # Check budget allocation
        if sum(strategy.budget_allocation.values()) < 0.95 or sum(strategy.budget_allocation.values()) > 1.05:
            score -= 15
            issues.append("Budget allocation doesn't sum to 100%")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Strategy is coherent",
            "recommendation": "Review and align strategy components" if issues else "Strategy approved"
        }
    
    def _check_resource_feasibility(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Check if resource allocation is feasible."""
        if not state.resource_allocation:
            return {"score": 0, "issue": "No resource allocation", "recommendation": "Define resource requirements"}
        
        allocation = state.resource_allocation
        score = 100
        issues = []
        
        # Check if we have enough content creators
        if allocation.content_creators == 0:
            score -= 40
            issues.append("No content creators allocated")
        
        # Check budget reasonableness
        if allocation.estimated_budget == 0:
            score -= 30
            issues.append("No budget allocated")
        
        # Check time allocation
        if not allocation.time_allocation_hours:
            score -= 20
            issues.append("No time allocation defined")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Resource allocation is feasible", 
            "recommendation": "Adjust resource allocation" if issues else "Resources approved"
        }
    
    def _check_timeline_realism(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Check if timeline is realistic."""
        if not state.timeline:
            return {"score": 0, "issue": "No timeline defined", "recommendation": "Create campaign timeline"}
        
        score = 100
        issues = []
        
        # Check if timeline matches strategy
        if state.campaign_strategy and len(state.timeline) != state.campaign_strategy.timeline_weeks:
            score -= 15
            issues.append("Timeline phases don't match strategy duration")
        
        # Check for complexity vs timeline
        if state.campaign_complexity == CampaignComplexity.ENTERPRISE and len(state.timeline) < 4:
            score -= 25
            issues.append("Timeline too short for enterprise complexity")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Timeline is realistic",
            "recommendation": "Extend timeline or reduce scope" if issues else "Timeline approved"
        }
    
    def _validate_task_dependencies(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Validate task dependencies for logical consistency."""
        if not state.task_dependencies:
            return {"score": 100, "issue": "No complex dependencies", "recommendation": "Dependencies validated"}
        
        score = 100
        issues = []
        
        # Check for circular dependencies (simplified check)
        for task_id, deps in state.task_dependencies.items():
            if task_id in deps:
                score -= 30
                issues.append(f"Circular dependency detected in task {task_id}")
        
        # Check if dependency tasks exist
        all_task_ids = {task["id"] for task in state.campaign_tasks}
        for task_id, deps in state.task_dependencies.items():
            for dep in deps:
                if dep not in all_task_ids:
                    score -= 15
                    issues.append(f"Dependency {dep} not found in tasks")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Dependencies are valid",
            "recommendation": "Fix dependency issues" if issues else "Dependencies approved"
        }
    
    def _check_budget_alignment(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Check if budget allocation aligns with strategy."""
        if not state.budget_allocation or not state.resource_allocation:
            return {"score": 50, "issue": "Incomplete budget data", "recommendation": "Complete budget planning"}
        
        score = 100
        issues = []
        
        # Check if budget categories are reasonable
        content_budget = state.budget_allocation.get("content_creation", 0)
        if content_budget < 0.3:  # Less than 30% for content creation seems low
            score -= 20
            issues.append("Content creation budget may be too low")
        
        # Check if estimated budget is reasonable for scope
        if state.resource_allocation.estimated_budget < 1000 and len(state.campaign_tasks) > 10:
            score -= 15
            issues.append("Budget may be insufficient for task scope")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Budget alignment is good",
            "recommendation": "Review budget allocation" if issues else "Budget approved"
        }
    
    def _check_channel_optimization(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Check if channel selection and strategy is optimized."""
        if not state.campaign_strategy:
            return {"score": 0, "issue": "No channel strategy", "recommendation": "Define channel strategy"}
        
        channels = state.campaign_strategy.distribution_channels
        score = 100
        issues = []
        
        # Check channel diversity
        if len(channels) < 2:
            score -= 20
            issues.append("Limited channel diversity may reduce reach")
        elif len(channels) > 5:
            score -= 10
            issues.append("Too many channels may dilute focus")
        
        # Check for logical channel combinations
        if "email" in channels and "linkedin" not in channels:
            score -= 5
            issues.append("Consider adding LinkedIn for B2B email campaigns")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Channel optimization is good",
            "recommendation": "Optimize channel mix" if issues else "Channels approved"
        }
    
    def _check_content_diversity(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Check content diversity and mix."""
        if not state.campaign_tasks:
            return {"score": 0, "issue": "No content tasks defined", "recommendation": "Define content tasks"}
        
        content_tasks = [t for t in state.campaign_tasks if t["task_type"] == "content_creation"]
        score = 100
        issues = []
        
        # Check content type diversity
        content_types = set(task["content_type"] for task in content_tasks)
        if len(content_types) < 2:
            score -= 15
            issues.append("Limited content type diversity")
        
        # Check for content balance
        if len(content_tasks) < 3:
            score -= 20
            issues.append("Insufficient content volume for effective campaign")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Content diversity is good",
            "recommendation": "Increase content variety" if issues else "Content mix approved"
        }
    
    def _assess_risk_mitigation(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Assess risk mitigation preparedness."""
        score = 100
        issues = []
        
        # Check if high-risk scenarios are addressed
        if state.campaign_complexity == CampaignComplexity.ENTERPRISE:
            if not state.timeline or len(state.timeline) < 4:
                score -= 20
                issues.append("Enterprise campaigns need longer timelines for risk management")
        
        # Check resource backup plans
        if state.resource_allocation.content_creators == 1:
            score -= 15
            issues.append("Single point of failure in content creation team")
        
        # Check timeline buffers
        if state.campaign_strategy and state.campaign_strategy.timeline_weeks < 3:
            score -= 10
            issues.append("Short timeline increases execution risk")
        
        return {
            "score": max(0, score),
            "issue": "; ".join(issues) if issues else "Risk mitigation is adequate",
            "recommendation": "Strengthen risk mitigation plans" if issues else "Risk management approved"
        }
    
    def _calculate_overall_risk(self, state: CampaignManagerState, quality_checks: Dict[str, Any]) -> str:
        """Calculate overall campaign risk level."""
        # Count critical quality issues
        critical_issues = sum(1 for check in quality_checks.values() if check.get("score", 100) < 60)
        moderate_issues = sum(1 for check in quality_checks.values() if 60 <= check.get("score", 100) < 80)
        
        if critical_issues >= 3:
            return "high"
        elif critical_issues >= 1 or moderate_issues >= 4:
            return "medium"
        else:
            return "low"
    
    def _generate_mitigation_strategies(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate mitigation strategies for identified risks."""
        strategies = []
        
        for opp in opportunities:
            if opp["priority"] == "high":
                if "timeline" in opp["area"]:
                    strategies.append("Implement parallel development streams to accelerate delivery")
                elif "resource" in opp["area"]:
                    strategies.append("Secure backup resources and cross-training")
                elif "quality" in opp["area"]:
                    strategies.append("Implement enhanced quality review checkpoints")
                elif "budget" in opp["area"]:
                    strategies.append("Establish contingency budget reserves")
                else:
                    strategies.append(f"Address {opp['area']} through enhanced monitoring and controls")
        
        return strategies[:5]  # Limit to top 5 strategies
    
    # Launch preparation helper methods
    
    def _create_launch_sequence(self, state: CampaignManagerState) -> List[Dict[str, Any]]:
        """Create detailed launch sequence."""
        sequence = [
            {
                "step": 1,
                "action": "Final quality review of all content assets",
                "owner": "Quality Assurance Team",
                "duration": "2 hours"
            },
            {
                "step": 2,
                "action": "Content distribution setup and scheduling",
                "owner": "Distribution Team", 
                "duration": "3 hours"
            },
            {
                "step": 3,
                "action": "Analytics and tracking implementation",
                "owner": "Analytics Team",
                "duration": "1 hour"
            },
            {
                "step": 4,
                "action": "Campaign launch execution",
                "owner": "Campaign Manager",
                "duration": "1 hour"
            },
            {
                "step": 5,
                "action": "Initial performance monitoring and adjustment",
                "owner": "Campaign Manager",
                "duration": "2 hours"
            }
        ]
        return sequence
    
    def _create_communication_plan(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Create campaign communication plan."""
        return {
            "stakeholder_updates": "Weekly progress reports",
            "team_synchronization": "Daily standups during active phases",
            "issue_escalation": "Immediate notification for critical issues",
            "success_celebrations": "Campaign milestone achievements",
            "feedback_collection": "Post-campaign retrospective"
        }
    
    def _create_monitoring_strategy(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Create comprehensive monitoring strategy."""
        return {
            "performance_metrics": ["reach", "engagement", "conversions", "quality_scores"],
            "monitoring_frequency": "Daily during active campaign",
            "alert_thresholds": {
                "engagement_rate_drop": "Below 2%",
                "quality_score_drop": "Below 70%",
                "budget_overrun": "Above 110% of allocation"
            },
            "dashboard_access": ["Campaign Manager", "Analytics Team", "Stakeholders"],
            "reporting_schedule": "Weekly comprehensive reports"
        }
    
    def _create_escalation_procedures(self, state: CampaignManagerState) -> List[Dict[str, Any]]:
        """Create escalation procedures for issues."""
        return [
            {
                "issue_type": "Quality concerns",
                "escalation_path": "Content Team Lead → Campaign Manager → Creative Director",
                "response_time": "2 hours"
            },
            {
                "issue_type": "Budget overrun",
                "escalation_path": "Campaign Manager → Finance Team → Executive Sponsor",
                "response_time": "4 hours"
            },
            {
                "issue_type": "Timeline delays",
                "escalation_path": "Project Coordinator → Campaign Manager → Operations Lead",
                "response_time": "1 hour"
            },
            {
                "issue_type": "Resource shortages",
                "escalation_path": "Team Lead → Resource Manager → Campaign Manager",
                "response_time": "2 hours"
            }
        ]
    
    def _create_success_tracking_plan(self, state: CampaignManagerState) -> Dict[str, Any]:
        """Create success tracking and measurement plan."""
        return {
            "primary_kpis": [
                "Content pieces delivered on time",
                "Quality scores achieved",
                "Engagement rates per channel",
                "Campaign ROI"
            ],
            "measurement_tools": ["Analytics dashboard", "Social media insights", "Email metrics"],
            "tracking_frequency": "Real-time for engagement, weekly for ROI",
            "success_criteria": state.strategy_analysis.get("success_criteria", {}),
            "benchmark_comparisons": "Previous campaigns and industry standards"
        }
    
    def _create_review_schedule(self, state: CampaignManagerState) -> List[Dict[str, Any]]:
        """Create campaign review and optimization schedule."""
        timeline_weeks = state.campaign_strategy.timeline_weeks if state.campaign_strategy else 4
        
        reviews = []
        for week in range(1, timeline_weeks + 1):
            if week == 1:
                reviews.append({
                    "week": week,
                    "review_type": "Launch Review",
                    "focus": "Initial performance and immediate optimizations",
                    "participants": ["Campaign Manager", "Analytics Team"]
                })
            elif week == timeline_weeks // 2:
                reviews.append({
                    "week": week,
                    "review_type": "Mid-Campaign Review",
                    "focus": "Performance assessment and strategy adjustments",
                    "participants": ["Campaign Manager", "Creative Team", "Analytics Team"]
                })
            elif week == timeline_weeks:
                reviews.append({
                    "week": week,
                    "review_type": "Final Campaign Review",
                    "focus": "Results analysis and lessons learned",
                    "participants": ["All Team Members", "Stakeholders"]
                })
        
        return reviews
    
    def _identify_backup_resources(self, risk_area: str) -> List[str]:
        """Identify backup resources for different risk areas."""
        backup_resources = {
            "timeline": ["Freelance content creators", "Overtime scheduling", "Content templates"],
            "resources": ["Backup team members", "External contractors", "Automated tools"],
            "quality": ["Additional review cycles", "Expert consultants", "Quality templates"],
            "budget": ["Contingency funds", "Scope reduction options", "Alternative solutions"],
            "content": ["Content libraries", "Template variations", "Repurposing strategies"]
        }
        return backup_resources.get(risk_area, ["General backup plans", "Emergency procedures"])
    
    def _calculate_phase_resources(self, phase_week: int, allocation: ResourceAllocation, total_weeks: int) -> Dict[str, Any]:
        """Calculate resource requirements for a specific phase."""
        # Distribute resources across weeks
        weekly_hours = sum(allocation.time_allocation_hours.values()) / total_weeks
        weekly_budget = allocation.estimated_budget / total_weeks
        
        return {
            "estimated_hours": weekly_hours,
            "budget_allocation": weekly_budget,
            "team_members_needed": max(1, int(weekly_hours / 40)),  # 40 hours per person per week
            "key_resources": list(allocation.agent_assignments.keys())[:3]
        }
    
    def _define_phase_quality_gates(self, phase: str, complexity: CampaignComplexity) -> List[str]:
        """Define quality gates for each phase."""
        base_gates = {
            "strategy_development": [
                "Strategy coherence check",
                "Stakeholder approval",
                "Resource feasibility validation"
            ],
            "content_planning": [
                "Content strategy alignment",
                "Channel optimization review",
                "Timeline feasibility check"
            ],
            "content_creation": [
                "Content quality review",
                "Brand consistency check",
                "SEO optimization validation"
            ],
            "content_optimization": [
                "Performance metrics review",
                "User feedback integration",
                "Final quality assurance"
            ],
            "launch_preparation": [
                "Launch readiness checklist",
                "Risk mitigation verification",
                "Team preparedness confirmation"
            ]
        }
        
        gates = base_gates.get(phase, ["Standard quality check"])
        
        # Add complexity-specific gates
        if complexity == CampaignComplexity.ENTERPRISE:
            gates.append("Executive review and approval")
            gates.append("Legal and compliance check")
        
        return gates
    
    def _identify_phase_risks(self, phase: str, overall_risks: List[Dict[str, Any]]) -> List[str]:
        """Identify specific risks for each phase."""
        phase_specific_risks = {
            "strategy_development": [
                "Misaligned objectives",
                "Insufficient market research",
                "Unrealistic expectations"
            ],
            "content_planning": [
                "Content gaps in strategy",
                "Channel mismatch",
                "Resource overallocation"
            ],
            "content_creation": [
                "Quality inconsistencies",
                "Timeline delays",
                "Resource bottlenecks"
            ],
            "content_optimization": [
                "Performance below expectations",
                "Last-minute changes",
                "Quality degradation"
            ],
            "launch_preparation": [
                "Technical issues",
                "Coordination failures",
                "Missing approvals"
            ]
        }
        
        base_risks = phase_specific_risks.get(phase, ["General execution risks"])
        
        # Add relevant overall risks
        relevant_risks = [risk["risk"] for risk in overall_risks if phase.lower() in risk["risk"].lower()]
        
        return list(set(base_risks + relevant_risks[:2]))  # Combine and deduplicate
    
    # Utility helper methods
    
    def _extract_industry_from_context(self, company_context: str) -> str:
        """Extract industry information from company context."""
        # Simple keyword matching - could be enhanced with NLP
        context_lower = company_context.lower()
        
        if any(keyword in context_lower for keyword in ["finance", "financial", "bank"]):
            return "Financial Services"
        elif any(keyword in context_lower for keyword in ["tech", "software", "saas"]):
            return "Technology"
        elif any(keyword in context_lower for keyword in ["health", "medical", "pharma"]):
            return "Healthcare"
        elif any(keyword in context_lower for keyword in ["retail", "commerce", "shop"]):
            return "Retail"
        elif any(keyword in context_lower for keyword in ["education", "learning", "school"]):
            return "Education"
        else:
            return "Professional Services"
    
    def _estimate_company_size(self, company_context: str) -> str:
        """Estimate company size from context."""
        context_lower = company_context.lower()
        
        if any(keyword in context_lower for keyword in ["startup", "small", "local"]):
            return "Small (1-50 employees)"
        elif any(keyword in context_lower for keyword in ["medium", "growing", "regional"]):
            return "Medium (51-500 employees)"
        elif any(keyword in context_lower for keyword in ["large", "enterprise", "global", "multinational"]):
            return "Large (500+ employees)"
        else:
            return "Medium (51-500 employees)"  # Default assumption
    
    def _identify_specialized_skills(self, state: CampaignManagerState) -> List[str]:
        """Identify specialized skills needed for the campaign."""
        skills = ["Content Creation", "Digital Marketing"]
        
        if state.campaign_strategy:
            # Add channel-specific skills
            if "linkedin" in state.campaign_strategy.distribution_channels:
                skills.append("LinkedIn Marketing")
            if "email" in state.campaign_strategy.distribution_channels:
                skills.append("Email Marketing")
            if "blog" in state.campaign_strategy.distribution_channels:
                skills.append("SEO Writing")
        
        # Add complexity-specific skills
        if state.campaign_complexity in [CampaignComplexity.COMPLEX, CampaignComplexity.ENTERPRISE]:
            skills.extend(["Project Management", "Analytics", "Campaign Strategy"])
        
        if state.campaign_type == "orchestration":
            skills.append("Campaign Orchestration")
            skills.append("Multi-Channel Coordination")
        
        return list(set(skills))  # Remove duplicates