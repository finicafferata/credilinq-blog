"""
LangGraph-enhanced Content Brief Agent Workflow for strategic content brief creation with comprehensive research.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum

# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
# Removed broken import: from .content_brief_agent import ...
# from ...config.database import DatabaseConnection  # Temporarily disabled


class ContentType(str, Enum):
    """Types of content to create."""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"

class ContentPurpose(str, Enum):
    """Purpose of the content."""
    EDUCATE = "educate"
    INFORM = "inform"
    PERSUADE = "persuade"
    ENTERTAIN = "entertain"
    CONVERT = "convert"

class BriefComplexity(str, Enum):
    """Content brief complexity levels."""
    BASIC = "basic"           # Simple topic, clear market
    STANDARD = "standard"     # Moderate research required
    ADVANCED = "advanced"     # Deep research and analysis
    STRATEGIC = "strategic"   # Executive-level strategic brief


class ResearchDepth(str, Enum):
    """Depth of research to conduct."""
    SURFACE = "surface"       # Basic keyword and competitor research
    MODERATE = "moderate"     # Standard SEO and competitive analysis
    COMPREHENSIVE = "comprehensive"  # Full market and strategic analysis
    EXPERT = "expert"         # Deep industry expertise and analysis


@dataclass
class CompetitorInsight:
    """Competitor analysis insight."""
    competitor_name: str
    content_type: str
    key_strength: str
    gap_opportunity: str
    confidence_score: float = 0.8


@dataclass
class ContentStructure:
    """Content structure outline."""
    outline: List[str]
    sections: List[str] 
    word_count_target: int = 1000
    reading_time_minutes: int = 5


@dataclass
class ContentBrief:
    """Final content brief document."""
    title: str
    executive_summary: str
    target_audience: str
    key_messages: List[str]
    content_structure: ContentStructure
    seo_keywords: List[str]
    competitor_insights: List[CompetitorInsight]


@dataclass
class BriefRequirements:
    """Detailed requirements for content brief creation."""
    topic: str
    content_type: ContentType
    primary_purpose: ContentPurpose
    target_audience: str
    company_context: str
    complexity_level: BriefComplexity = BriefComplexity.STANDARD
    research_depth: ResearchDepth = ResearchDepth.MODERATE
    include_competitive_analysis: bool = True
    include_seo_strategy: bool = True
    include_content_calendar: bool = False
    custom_requirements: List[str] = field(default_factory=list)


@dataclass
class MarketIntelligence:
    """Market intelligence gathered during research."""
    market_size_indicators: List[str]
    trend_analysis: List[str]
    opportunity_assessment: str
    threat_analysis: List[str]
    market_maturity: str
    regulatory_considerations: List[str]
    technology_trends: List[str]


@dataclass
class StrategicInsights:
    """Strategic insights for content positioning."""
    market_positioning: str
    value_proposition_angles: List[str]
    competitive_advantages: List[str]
    content_opportunities: List[str]
    strategic_recommendations: List[str]
    thought_leadership_angles: List[str]
    partnership_opportunities: List[str]


class ContentBriefState(WorkflowState):
    """Enhanced state for content brief workflow."""
    
    # Input requirements
    brief_requirements: Optional[BriefRequirements] = None
    brand_guidelines: Dict[str, Any] = field(default_factory=dict)
    existing_content_context: List[str] = field(default_factory=list)
    
    # Research phase results
    keyword_research: Dict[str, Any] = field(default_factory=dict)
    competitor_intelligence: List[CompetitorInsight] = field(default_factory=list)
    market_intelligence: Optional[MarketIntelligence] = None
    strategic_insights: Optional[StrategicInsights] = None
    
    # Content strategy results
    content_structure: Optional[ContentStructure] = None
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    distribution_strategy: Dict[str, Any] = field(default_factory=dict)
    
    # Final brief
    content_brief: Optional[ContentBrief] = None
    executive_summary: str = ""
    implementation_plan: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assurance
    brief_quality_score: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)
    stakeholder_review_notes: List[str] = field(default_factory=list)
    
    # Performance metrics
    research_depth_achieved: ResearchDepth = ResearchDepth.SURFACE
    time_to_completion: float = 0.0
    resource_utilization: Dict[str, Any] = field(default_factory=dict)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class ContentBriefAgentWorkflow(LangGraphWorkflowBase[ContentBriefState]):
    """LangGraph workflow for strategic content brief creation with comprehensive research."""
    
    def __init__(
        self, 
        workflow_name: str = "content_brief_agent_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        # self.legacy_agent = ContentBriefAgent()  # Temporarily disabled for pipeline testing
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> ContentBriefState:
        """Create initial workflow state from context."""
        # Parse brief requirements
        requirements = BriefRequirements(
            topic=context.get("topic", ""),
            content_type=ContentType(context.get("content_type", "blog_post")),
            primary_purpose=ContentPurpose(context.get("primary_purpose", "lead_generation")),
            target_audience=context.get("target_audience", "B2B finance professionals"),
            company_context=context.get("company_context", ""),
            complexity_level=BriefComplexity(context.get("complexity_level", "standard")),
            research_depth=ResearchDepth(context.get("research_depth", "moderate")),
            include_competitive_analysis=context.get("include_competitive_analysis", True),
            include_seo_strategy=context.get("include_seo_strategy", True),
            include_content_calendar=context.get("include_content_calendar", False),
            custom_requirements=context.get("custom_requirements", [])
        )
        
        return ContentBriefState(
            workflow_id=context.get("workflow_id", f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            brief_requirements=requirements,
            brand_guidelines=context.get("brand_guidelines", {}),
            existing_content_context=context.get("existing_content_context", []),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the content brief workflow graph."""
        workflow = StateGraph(ContentBriefState)
        
        # Define workflow nodes
        workflow.add_node("validate_requirements", self._validate_requirements_node)
        workflow.add_node("conduct_keyword_research", self._conduct_keyword_research_node)
        workflow.add_node("analyze_competitive_landscape", self._analyze_competitive_landscape_node)
        workflow.add_node("gather_market_intelligence", self._gather_market_intelligence_node)
        workflow.add_node("develop_strategic_insights", self._develop_strategic_insights_node)
        workflow.add_node("create_content_structure", self._create_content_structure_node)
        workflow.add_node("define_success_metrics", self._define_success_metrics_node)
        workflow.add_node("generate_content_brief", self._generate_content_brief_node)
        workflow.add_node("validate_brief_quality", self._validate_brief_quality_node)
        workflow.add_node("finalize_deliverables", self._finalize_deliverables_node)
        
        # Define workflow edges
        workflow.add_edge("validate_requirements", "conduct_keyword_research")
        
        # Parallel research phase
        workflow.add_conditional_edges(
            "conduct_keyword_research",
            self._should_conduct_competitive_analysis,
            {
                "competitive_analysis": "analyze_competitive_landscape",
                "market_intelligence": "gather_market_intelligence"
            }
        )
        workflow.add_edge("analyze_competitive_landscape", "gather_market_intelligence")
        
        # Strategic development phase
        workflow.add_conditional_edges(
            "gather_market_intelligence",
            self._determine_research_depth,
            {
                "strategic_insights": "develop_strategic_insights",
                "content_structure": "create_content_structure"
            }
        )
        workflow.add_edge("develop_strategic_insights", "create_content_structure")
        workflow.add_edge("create_content_structure", "define_success_metrics")
        workflow.add_edge("define_success_metrics", "generate_content_brief")
        workflow.add_edge("generate_content_brief", "validate_brief_quality")
        workflow.add_edge("validate_brief_quality", "finalize_deliverables")
        workflow.add_edge("finalize_deliverables", END)
        
        # Set entry point
        workflow.set_entry_point("validate_requirements")
        
        return workflow
    
    async def _validate_requirements_node(self, state: ContentBriefState) -> ContentBriefState:
        """Validate brief requirements and prepare for research."""
        try:
            self._log_progress("Validating content brief requirements")
            
            requirements = state.brief_requirements
            validation_errors = []
            
            # Validate core requirements
            if not requirements.topic or len(requirements.topic.strip()) < 3:
                validation_errors.append("Topic must be at least 3 characters long")
            
            if not requirements.target_audience:
                validation_errors.append("Target audience must be specified")
            
            if not requirements.company_context:
                self._log_progress("No company context provided, will use generic context")
            
            # Validate content type and purpose alignment
            incompatible_combinations = [
                (ContentType.WHITE_PAPER, ContentPurpose.SEO_RANKING),
                (ContentType.CASE_STUDY, ContentPurpose.BRAND_AWARENESS)
            ]
            
            if (requirements.content_type, requirements.primary_purpose) in incompatible_combinations:
                self._log_progress(f"Content type and purpose may not be optimally aligned: {requirements.content_type} + {requirements.primary_purpose}")
            
            # Adjust research depth based on complexity
            if requirements.complexity_level == BriefComplexity.STRATEGIC and requirements.research_depth == ResearchDepth.SURFACE:
                requirements.research_depth = ResearchDepth.COMPREHENSIVE
                self._log_progress("Upgraded research depth to comprehensive for strategic brief")
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 10.0
                
                state.messages.append(HumanMessage(
                    content=f"Brief requirements validated for '{requirements.topic}' "
                           f"({requirements.content_type.value}, {requirements.complexity_level.value} complexity)"
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Requirements validation failed: {str(e)}"
            return state
    
    async def _conduct_keyword_research_node(self, state: ContentBriefState) -> ContentBriefState:
        """Conduct comprehensive SEO keyword research."""
        try:
            self._log_progress("Conducting comprehensive SEO keyword research")
            
            requirements = state.brief_requirements
            
            # Prepare research request for legacy agent
            research_request = {
                "topic": requirements.topic,
                "content_type": requirements.content_type.value,
                "primary_purpose": requirements.primary_purpose.value,
                "target_audience": requirements.target_audience,
                "research_depth": requirements.research_depth.value
            }
            
            # Use legacy agent's keyword research
            keyword_research = await self.legacy_agent._conduct_keyword_research(
                requirements.topic,
                requirements.content_type,
                requirements.primary_purpose
            )
            
            # Enhanced keyword analysis based on research depth
            if requirements.research_depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.EXPERT]:
                keyword_research = await self._enhance_keyword_research(keyword_research, requirements)
            
            state.keyword_research = keyword_research
            state.progress_percentage = 25.0
            
            primary_keyword = keyword_research.get("primary_keyword")
            if primary_keyword:
                keyword_text = primary_keyword.keyword if hasattr(primary_keyword, 'keyword') else str(primary_keyword)
                state.messages.append(SystemMessage(
                    content=f"Keyword research completed. Primary keyword: '{keyword_text}' "
                           f"with {len(keyword_research.get('secondary_keywords', []))} secondary keywords identified."
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Keyword research failed: {str(e)}"
            return state
    
    async def _analyze_competitive_landscape_node(self, state: ContentBriefState) -> ContentBriefState:
        """Analyze competitive content landscape."""
        try:
            self._log_progress("Analyzing competitive content landscape")
            
            requirements = state.brief_requirements
            
            # Use legacy agent's competitive analysis
            competitor_insights = await self.legacy_agent._analyze_competitive_landscape(
                requirements.topic,
                requirements.target_audience
            )
            
            # Enhanced competitive analysis for higher research depths
            if requirements.research_depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.EXPERT]:
                competitor_insights = await self._enhance_competitive_analysis(
                    competitor_insights, requirements, state.keyword_research
                )
            
            state.competitor_intelligence = competitor_insights
            state.progress_percentage = 40.0
            
            state.messages.append(SystemMessage(
                content=f"Competitive landscape analysis completed. {len(competitor_insights)} competitors analyzed "
                       f"with detailed intelligence profiles."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Competitive analysis failed: {str(e)}"
            return state
    
    async def _gather_market_intelligence_node(self, state: ContentBriefState) -> ContentBriefState:
        """Gather comprehensive market intelligence."""
        try:
            self._log_progress("Gathering market intelligence and trend analysis")
            
            requirements = state.brief_requirements
            
            # Gather market intelligence based on research depth
            if requirements.research_depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.EXPERT]:
                market_intelligence = await self._conduct_market_research(requirements)
                state.market_intelligence = market_intelligence
            else:
                # Basic market intelligence
                state.market_intelligence = MarketIntelligence(
                    market_size_indicators=["Growing B2B fintech market"],
                    trend_analysis=["Digital transformation acceleration", "API-first solutions"],
                    opportunity_assessment="Significant growth potential in embedded finance",
                    threat_analysis=["Increasing competition", "Regulatory changes"],
                    market_maturity="Early growth stage",
                    regulatory_considerations=["Financial services regulations", "Data privacy"],
                    technology_trends=["API integration", "Real-time processing"]
                )
            
            state.progress_percentage = 55.0
            
            intelligence_depth = "comprehensive" if requirements.research_depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.EXPERT] else "basic"
            state.messages.append(SystemMessage(
                content=f"Market intelligence gathered with {intelligence_depth} analysis. "
                       f"Identified {len(state.market_intelligence.trend_analysis)} key trends."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Market intelligence gathering failed: {str(e)}"
            return state
    
    async def _develop_strategic_insights_node(self, state: ContentBriefState) -> ContentBriefState:
        """Develop strategic insights for content positioning."""
        try:
            self._log_progress("Developing strategic insights and positioning")
            
            requirements = state.brief_requirements
            
            # Generate strategic insights
            strategic_insights = await self._generate_strategic_insights(
                requirements, 
                state.keyword_research, 
                state.competitor_intelligence,
                state.market_intelligence
            )
            
            state.strategic_insights = strategic_insights
            state.progress_percentage = 70.0
            
            state.messages.append(SystemMessage(
                content=f"Strategic insights developed. {len(strategic_insights.value_proposition_angles)} "
                       f"value proposition angles and {len(strategic_insights.strategic_recommendations)} recommendations identified."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Strategic insights development failed: {str(e)}"
            return state
    
    async def _create_content_structure_node(self, state: ContentBriefState) -> ContentBriefState:
        """Create detailed content structure and outline."""
        try:
            self._log_progress("Creating detailed content structure and outline")
            
            requirements = state.brief_requirements
            
            # Use legacy agent for base structure
            content_structure = await self.legacy_agent._generate_content_structure(
                requirements.topic,
                requirements.content_type,
                state.keyword_research,
                state.competitor_intelligence
            )
            
            # Enhance structure with strategic insights
            if state.strategic_insights:
                content_structure = await self._enhance_content_structure(
                    content_structure, state.strategic_insights, requirements
                )
            
            state.content_structure = content_structure
            state.progress_percentage = 80.0
            
            state.messages.append(SystemMessage(
                content=f"Content structure created. {content_structure.estimated_word_count} words, "
                       f"{len(content_structure.content_outline)} sections, "
                       f"{len(content_structure.call_to_actions)} CTAs."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content structure creation failed: {str(e)}"
            return state
    
    async def _define_success_metrics_node(self, state: ContentBriefState) -> ContentBriefState:
        """Define comprehensive success metrics and KPIs."""
        try:
            self._log_progress("Defining success metrics and KPIs")
            
            requirements = state.brief_requirements
            
            # Use legacy agent for base metrics
            success_metrics = await self.legacy_agent._define_success_metrics(
                requirements.primary_purpose,
                requirements.content_type
            )
            
            # Enhanced metrics for strategic briefs
            if requirements.complexity_level == BriefComplexity.STRATEGIC:
                success_metrics = await self._enhance_success_metrics(
                    success_metrics, requirements, state.strategic_insights
                )
            
            # Add distribution strategy
            distribution_strategy = await self._develop_distribution_strategy(requirements)
            
            state.success_metrics = success_metrics
            state.distribution_strategy = distribution_strategy
            state.progress_percentage = 90.0
            
            state.messages.append(SystemMessage(
                content=f"Success metrics defined. {len(success_metrics['kpis'])} KPIs identified "
                       f"with {len(distribution_strategy.get('channels', []))} distribution channels."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Success metrics definition failed: {str(e)}"
            return state
    
    async def _generate_content_brief_node(self, state: ContentBriefState) -> ContentBriefState:
        """Generate comprehensive content brief."""
        try:
            self._log_progress("Generating comprehensive content brief")
            
            requirements = state.brief_requirements
            
            # Prepare brief request for legacy agent
            brief_request = {
                "topic": requirements.topic,
                "content_type": requirements.content_type.value,
                "primary_purpose": requirements.primary_purpose.value,
                "target_audience": requirements.target_audience,
                "company_context": requirements.company_context,
                "brand_guidelines": state.brand_guidelines,
                "distribution_channels": list(state.distribution_strategy.get("channels", []))
            }
            
            # Generate strategic brief using legacy agent
            strategic_brief = await self.legacy_agent._generate_strategic_brief(
                brief_request,
                state.keyword_research,
                state.competitor_intelligence,
                state.content_structure
            )
            
            # Create comprehensive content brief
            content_brief = ContentBrief(
                brief_id=f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=strategic_brief.get("title", f"{requirements.topic} Strategic Brief"),
                content_type=requirements.content_type,
                primary_purpose=requirements.primary_purpose,
                target_audience=requirements.target_audience,
                marketing_objective=strategic_brief.get("marketing_objective", ""),
                business_context=requirements.company_context,
                unique_value_proposition=strategic_brief.get("unique_value_proposition", ""),
                key_messages=strategic_brief.get("key_messages", []),
                primary_keyword=state.keyword_research["primary_keyword"],
                secondary_keywords=state.keyword_research["secondary_keywords"],
                semantic_keywords=state.keyword_research["semantic_keywords"],
                target_search_intent=state.keyword_research["search_intent"],
                seo_goals=state.keyword_research["seo_goals"],
                competitor_insights=state.competitor_intelligence,
                market_gap_identified=strategic_brief.get("market_gap", ""),
                differentiation_strategy=strategic_brief.get("differentiation_strategy", ""),
                content_structure=state.content_structure,
                tone_and_voice=strategic_brief.get("tone_and_voice", "Professional and authoritative"),
                writing_style_notes=strategic_brief.get("writing_style_notes", ""),
                brand_guidelines=state.brand_guidelines,
                success_kpis=state.success_metrics["kpis"],
                target_metrics=state.success_metrics["targets"],
                estimated_creation_time=self.legacy_agent._estimate_creation_time(
                    requirements.content_type, 
                    state.content_structure.estimated_word_count
                ),
                content_calendar_notes=strategic_brief.get("content_calendar_notes", ""),
                distribution_channels=list(state.distribution_strategy.get("channels", []))
            )
            
            state.content_brief = content_brief
            state.progress_percentage = 95.0
            
            state.messages.append(SystemMessage(
                content=f"Content brief generated: '{content_brief.title}' "
                       f"({content_brief.content_type.value}, {content_brief.estimated_creation_time})"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content brief generation failed: {str(e)}"
            return state
    
    async def _validate_brief_quality_node(self, state: ContentBriefState) -> ContentBriefState:
        """Validate content brief quality and completeness."""
        try:
            self._log_progress("Validating content brief quality")
            
            validation_results = await self._validate_brief_completeness(state.content_brief, state.brief_requirements)
            quality_score = await self._calculate_brief_quality_score(state.content_brief, state.brief_requirements)
            
            state.validation_results = validation_results
            state.brief_quality_score = quality_score
            
            # Determine if brief meets quality standards
            quality_threshold = 0.8 if state.brief_requirements.complexity_level == BriefComplexity.STRATEGIC else 0.7
            
            if quality_score < quality_threshold:
                state.stakeholder_review_notes.append(
                    f"Brief quality score ({quality_score:.2f}) below threshold ({quality_threshold}). Review recommended."
                )
            
            state.progress_percentage = 98.0
            
            state.messages.append(SystemMessage(
                content=f"Brief quality validation completed. Quality score: {quality_score:.2f}/1.0. "
                       f"Validation passed: {quality_score >= quality_threshold}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Brief quality validation failed: {str(e)}"
            return state
    
    async def _finalize_deliverables_node(self, state: ContentBriefState) -> ContentBriefState:
        """Finalize deliverables and create implementation plan."""
        try:
            self._log_progress("Finalizing deliverables and implementation plan")
            
            # Generate executive summary
            executive_summary = await self.legacy_agent.generate_brief_summary(state.content_brief)
            state.executive_summary = executive_summary
            
            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(state)
            state.implementation_plan = implementation_plan
            
            # Calculate final metrics
            state.time_to_completion = (datetime.utcnow() - state.created_at).total_seconds()
            state.research_depth_achieved = state.brief_requirements.research_depth
            
            # Resource utilization tracking
            state.resource_utilization = {
                "research_phases_completed": 5 if state.strategic_insights else 3,
                "ai_model_calls": self._estimate_ai_calls(state.brief_requirements),
                "data_sources_analyzed": len(state.competitor_intelligence) + len(state.keyword_research.get("secondary_keywords", [])),
                "insights_generated": len(state.strategic_insights.strategic_recommendations) if state.strategic_insights else 0
            }
            
            state.status = WorkflowStatus.COMPLETED
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            state.messages.append(SystemMessage(
                content=f"Content brief workflow completed successfully. "
                       f"Quality score: {state.brief_quality_score:.2f}, "
                       f"Processing time: {state.time_to_completion:.1f}s"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Deliverables finalization failed: {str(e)}"
            return state
    
    def _should_conduct_competitive_analysis(self, state: ContentBriefState) -> str:
        """Determine if competitive analysis should be conducted."""
        return "competitive_analysis" if state.brief_requirements.include_competitive_analysis else "market_intelligence"
    
    def _determine_research_depth(self, state: ContentBriefState) -> str:
        """Determine if strategic insights development is needed."""
        return "strategic_insights" if state.brief_requirements.research_depth in [ResearchDepth.COMPREHENSIVE, ResearchDepth.EXPERT] else "content_structure"
    
    # Helper methods for enhanced brief creation
    
    async def _enhance_keyword_research(
        self, 
        base_research: Dict[str, Any], 
        requirements: BriefRequirements
    ) -> Dict[str, Any]:
        """Enhance keyword research with deeper analysis."""
        enhanced_research = base_research.copy()
        
        # Add advanced keyword analysis
        if requirements.research_depth == ResearchDepth.EXPERT:
            # Add competitive keyword gaps
            enhanced_research["competitive_keyword_gaps"] = [
                f"{requirements.topic} implementation",
                f"{requirements.topic} ROI analysis",
                f"{requirements.topic} case studies"
            ]
            
            # Add long-tail opportunities
            enhanced_research["long_tail_opportunities"] = [
                f"how to implement {requirements.topic}",
                f"{requirements.topic} best practices for {requirements.target_audience.lower()}",
                f"{requirements.topic} vs alternatives comparison"
            ]
        
        return enhanced_research
    
    async def _enhance_competitive_analysis(
        self,
        base_insights: List[CompetitorInsight],
        requirements: BriefRequirements,
        keyword_research: Dict[str, Any]
    ) -> List[CompetitorInsight]:
        """Enhance competitive analysis with deeper intelligence."""
        enhanced_insights = base_insights.copy()
        
        # Add competitive keyword analysis
        for insight in enhanced_insights:
            if requirements.research_depth == ResearchDepth.EXPERT:
                insight.differentiation_opportunities.extend([
                    "Target underserved keyword clusters",
                    "Focus on implementation-specific content",
                    "Develop thought leadership positioning"
                ])
        
        return enhanced_insights
    
    async def _conduct_market_research(self, requirements: BriefRequirements) -> MarketIntelligence:
        """Conduct comprehensive market research."""
        try:
            # Market intelligence based on topic and industry
            market_intelligence = MarketIntelligence(
                market_size_indicators=[
                    f"{requirements.topic} market showing 25-35% annual growth",
                    "Increasing enterprise adoption in financial services",
                    "SMB segment emerging as high-growth opportunity"
                ],
                trend_analysis=[
                    "API-first architecture adoption accelerating",
                    "Embedded finance solutions gaining traction",
                    "Regulatory compliance becoming differentiator",
                    "Real-time processing becoming table stakes"
                ],
                opportunity_assessment=f"Significant opportunity in {requirements.topic} education and thought leadership",
                threat_analysis=[
                    "Increasing market saturation",
                    "Rising customer acquisition costs",
                    "Regulatory complexity increasing",
                    "Technology commoditization risk"
                ],
                market_maturity="Growth stage with early consolidation",
                regulatory_considerations=[
                    "Financial services regulations (PCI, SOX)",
                    "Data privacy requirements (GDPR, CCPA)",
                    "Anti-money laundering compliance",
                    "Open banking regulations"
                ],
                technology_trends=[
                    "AI/ML integration for decision making",
                    "Blockchain for transparency and trust",
                    "Cloud-native architectures",
                    "API-first platform strategies"
                ]
            )
            
            return market_intelligence
            
        except Exception as e:
            self._log_error(f"Market research failed: {str(e)}")
            return MarketIntelligence(
                market_size_indicators=["Market data unavailable"],
                trend_analysis=["Trend analysis pending"],
                opportunity_assessment="Opportunity assessment required",
                threat_analysis=["Threat analysis needed"],
                market_maturity="Analysis pending",
                regulatory_considerations=["Regulatory review required"],
                technology_trends=["Technology trend analysis needed"]
            )
    
    async def _generate_strategic_insights(
        self,
        requirements: BriefRequirements,
        keyword_research: Dict[str, Any],
        competitor_intelligence: List[CompetitorInsight],
        market_intelligence: MarketIntelligence
    ) -> StrategicInsights:
        """Generate comprehensive strategic insights."""
        try:
            strategic_insights = StrategicInsights(
                market_positioning=f"Position as the definitive {requirements.topic} thought leader for {requirements.target_audience}",
                value_proposition_angles=[
                    f"Practical {requirements.topic} implementation guidance",
                    "ROI-focused strategic approach",
                    "Industry-specific best practices",
                    "Proven methodology and frameworks"
                ],
                competitive_advantages=[
                    "Deep technical expertise combined with business acumen",
                    "Proven track record with enterprise clients",
                    "Comprehensive implementation methodology",
                    "Strong regulatory compliance focus"
                ],
                content_opportunities=[
                    f"Technical implementation guides for {requirements.topic}",
                    "ROI calculators and business case templates",
                    "Regulatory compliance checklists",
                    "Industry-specific case studies"
                ],
                strategic_recommendations=[
                    "Develop comprehensive content series around implementation",
                    "Create interactive tools and calculators",
                    "Build partnerships with complementary solution providers",
                    "Establish thought leadership through industry speaking"
                ],
                thought_leadership_angles=[
                    f"Future of {requirements.topic} in financial services",
                    "Regulatory implications and compliance strategies",
                    "Integration with emerging technologies (AI, blockchain)",
                    "Market consolidation predictions and implications"
                ],
                partnership_opportunities=[
                    "System integrators and consultants",
                    "Technology platforms and marketplaces",
                    "Industry associations and conferences",
                    "Regulatory bodies and compliance organizations"
                ]
            )
            
            return strategic_insights
            
        except Exception as e:
            self._log_error(f"Strategic insights generation failed: {str(e)}")
            return StrategicInsights(
                market_positioning="Market positioning analysis required",
                value_proposition_angles=["Value proposition development needed"],
                competitive_advantages=["Competitive advantage analysis pending"],
                content_opportunities=["Content opportunity identification required"],
                strategic_recommendations=["Strategic recommendation development needed"],
                thought_leadership_angles=["Thought leadership positioning required"],
                partnership_opportunities=["Partnership analysis needed"]
            )
    
    async def _enhance_content_structure(
        self,
        base_structure: ContentStructure,
        strategic_insights: StrategicInsights,
        requirements: BriefRequirements
    ) -> ContentStructure:
        """Enhance content structure with strategic insights."""
        enhanced_structure = base_structure
        
        # Add strategic sections based on insights
        strategic_sections = []
        for opportunity in strategic_insights.content_opportunities[:3]:
            strategic_sections.append({
                "section": f"Strategic Focus: {opportunity.split(':')[0] if ':' in opportunity else opportunity}",
                "description": f"Detailed analysis and implementation guidance for {opportunity.lower()}"
            })
        
        enhanced_structure.content_outline.extend(strategic_sections)
        
        # Add thought leadership CTAs
        enhanced_structure.call_to_actions.extend([
            "Join executive roundtable discussion",
            "Access exclusive industry research",
            "Schedule strategic consultation"
        ])
        
        return enhanced_structure
    
    async def _enhance_success_metrics(
        self,
        base_metrics: Dict[str, Any],
        requirements: BriefRequirements,
        strategic_insights: Optional[StrategicInsights]
    ) -> Dict[str, Any]:
        """Enhance success metrics for strategic briefs."""
        enhanced_metrics = base_metrics.copy()
        
        # Add strategic KPIs
        strategic_kpis = [
            "Thought leadership mentions",
            "Executive engagement rate",
            "Strategic partnership inquiries",
            "Speaking opportunity requests"
        ]
        
        enhanced_metrics["kpis"].extend(strategic_kpis)
        
        # Add strategic targets
        strategic_targets = {
            "thought_leadership_mentions": "5+ per quarter",
            "executive_engagement_rate": "15%+ C-level interaction",
            "strategic_partnerships": "2+ qualified partnership discussions",
            "industry_recognition": "Top 3 in industry content ranking"
        }
        
        enhanced_metrics["targets"].update(strategic_targets)
        
        return enhanced_metrics
    
    async def _develop_distribution_strategy(self, requirements: BriefRequirements) -> Dict[str, Any]:
        """Develop comprehensive distribution strategy."""
        strategy = {
            "primary_channels": ["company_blog", "linkedin", "email_newsletter"],
            "secondary_channels": ["industry_publications", "partner_networks", "social_media"],
            "timing_strategy": "Stagger release across 2-week period",
            "amplification_tactics": [
                "Executive social media promotion",
                "Industry influencer outreach",
                "Partner cross-promotion",
                "Conference presentation opportunities"
            ],
            "measurement_approach": "Multi-touch attribution with engagement tracking"
        }
        
        # Adjust based on content type
        if requirements.content_type == ContentType.WHITE_PAPER:
            strategy["primary_channels"] = ["gated_download", "email_nurture", "sales_enablement"]
            strategy["lead_capture"] = "Progressive profiling with value-based gating"
        elif requirements.content_type == ContentType.LINKEDIN_ARTICLE:
            strategy["primary_channels"] = ["linkedin_native", "executive_networks", "industry_groups"]
            strategy["engagement_optimization"] = "LinkedIn algorithm optimization"
        
        return strategy
    
    async def _validate_brief_completeness(
        self, 
        content_brief: ContentBrief, 
        requirements: BriefRequirements
    ) -> Dict[str, Any]:
        """Validate content brief completeness."""
        validation_results = {
            "required_fields_complete": True,
            "strategic_elements_present": True,
            "seo_strategy_complete": True,
            "competitive_analysis_adequate": True,
            "success_metrics_defined": True,
            "missing_elements": [],
            "recommendations": []
        }
        
        # Check required fields
        required_checks = [
            ("title", content_brief.title),
            ("target_audience", content_brief.target_audience),
            ("primary_keyword", content_brief.primary_keyword),
            ("content_structure", content_brief.content_structure)
        ]
        
        for field_name, field_value in required_checks:
            if not field_value:
                validation_results["required_fields_complete"] = False
                validation_results["missing_elements"].append(field_name)
        
        # Check strategic completeness
        if requirements.complexity_level == BriefComplexity.STRATEGIC:
            if len(content_brief.key_messages) < 3:
                validation_results["strategic_elements_present"] = False
                validation_results["recommendations"].append("Add more strategic key messages")
        
        return validation_results
    
    async def _calculate_brief_quality_score(
        self, 
        content_brief: ContentBrief, 
        requirements: BriefRequirements
    ) -> float:
        """Calculate comprehensive brief quality score."""
        quality_score = 0.0
        max_score = 1.0
        
        # Completeness score (40%)
        completeness_score = 0.4
        if content_brief.title and content_brief.target_audience:
            completeness_score *= 1.0
        else:
            completeness_score *= 0.5
        
        # Strategic depth score (30%)
        strategic_score = 0.3
        if len(content_brief.key_messages) >= 3 and content_brief.differentiation_strategy:
            strategic_score *= 1.0
        elif len(content_brief.key_messages) >= 2:
            strategic_score *= 0.7
        else:
            strategic_score *= 0.4
        
        # SEO strategy score (20%)
        seo_score = 0.2
        if (content_brief.primary_keyword and 
            len(content_brief.secondary_keywords) >= 3 and 
            content_brief.target_search_intent):
            seo_score *= 1.0
        else:
            seo_score *= 0.6
        
        # Competitive analysis score (10%)
        competitive_score = 0.1
        if len(content_brief.competitor_insights) >= 2:
            competitive_score *= 1.0
        else:
            competitive_score *= 0.5
        
        quality_score = completeness_score + strategic_score + seo_score + competitive_score
        
        return min(quality_score, max_score)
    
    async def _create_implementation_plan(self, state: ContentBriefState) -> Dict[str, Any]:
        """Create comprehensive implementation plan."""
        implementation_plan = {
            "content_creation_phases": [
                {
                    "phase": "Research and outline",
                    "duration": "2-3 days",
                    "deliverables": ["Detailed outline", "Research compilation", "Source verification"]
                },
                {
                    "phase": "Content development",
                    "duration": state.content_brief.estimated_creation_time,
                    "deliverables": ["First draft", "Internal review", "Stakeholder feedback"]
                },
                {
                    "phase": "Review and optimization",
                    "duration": "1-2 days",
                    "deliverables": ["SEO optimization", "Final review", "Approval"]
                },
                {
                    "phase": "Distribution preparation",
                    "duration": "1 day",
                    "deliverables": ["Asset creation", "Distribution setup", "Measurement tracking"]
                }
            ],
            "resource_requirements": {
                "content_creator": "Senior level with industry expertise",
                "review_stakeholders": "Subject matter expert, Marketing lead",
                "design_support": "Visual assets and formatting",
                "distribution_support": "Channel management and amplification"
            },
            "success_tracking": {
                "measurement_setup": "Analytics and tracking implementation",
                "reporting_schedule": "Weekly for first month, monthly thereafter",
                "optimization_points": "30, 60, 90 day performance reviews"
            },
            "risk_mitigation": {
                "content_quality": "Multiple review stages and expert validation",
                "timeline_risk": "Buffer time built into each phase",
                "market_changes": "Quarterly brief refresh and update process"
            }
        }
        
        return implementation_plan
    
    def _estimate_ai_calls(self, requirements: BriefRequirements) -> int:
        """Estimate number of AI model calls based on complexity."""
        base_calls = 5  # Basic research and structure
        
        if requirements.research_depth == ResearchDepth.COMPREHENSIVE:
            base_calls += 3
        elif requirements.research_depth == ResearchDepth.EXPERT:
            base_calls += 5
        
        if requirements.complexity_level == BriefComplexity.STRATEGIC:
            base_calls += 3
        
        if requirements.include_competitive_analysis:
            base_calls += 2
        
        return base_calls
    
    async def execute_workflow(
        self,
        topic: str,
        content_type: str = "blog_post",
        primary_purpose: str = "lead_generation",
        target_audience: str = "B2B finance professionals",
        company_context: str = "",
        complexity_level: str = "standard",
        research_depth: str = "moderate",
        include_competitive_analysis: bool = True,
        include_seo_strategy: bool = True,
        brand_guidelines: Optional[Dict[str, Any]] = None,
        custom_requirements: Optional[List[str]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the content brief agent workflow."""
        
        context = {
            "topic": topic,
            "content_type": content_type,
            "primary_purpose": primary_purpose,
            "target_audience": target_audience,
            "company_context": company_context,
            "complexity_level": complexity_level,
            "research_depth": research_depth,
            "include_competitive_analysis": include_competitive_analysis,
            "include_seo_strategy": include_seo_strategy,
            "brand_guidelines": brand_guidelines or {},
            "custom_requirements": custom_requirements or [],
            "workflow_id": f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "content_brief": final_state.content_brief.model_dump() if final_state.content_brief else {},
                    "executive_summary": final_state.executive_summary,
                    "implementation_plan": final_state.implementation_plan,
                    "keyword_research": final_state.keyword_research,
                    "competitive_intelligence": [
                        {
                            "competitor_name": comp.competitor_name,
                            "content_title": comp.content_title,
                            "key_angles": comp.key_angles,
                            "content_gaps": comp.content_gaps,
                            "differentiation_opportunities": comp.differentiation_opportunities
                        }
                        for comp in final_state.competitor_intelligence
                    ],
                    "market_intelligence": final_state.market_intelligence.__dict__ if final_state.market_intelligence else {},
                    "strategic_insights": final_state.strategic_insights.__dict__ if final_state.strategic_insights else {},
                    "success_metrics": final_state.success_metrics,
                    "distribution_strategy": final_state.distribution_strategy,
                    "quality_assessment": {
                        "brief_quality_score": final_state.brief_quality_score,
                        "validation_results": final_state.validation_results,
                        "stakeholder_review_notes": final_state.stakeholder_review_notes
                    },
                    "workflow_metrics": {
                        "time_to_completion": final_state.time_to_completion,
                        "research_depth_achieved": final_state.research_depth_achieved.value,
                        "resource_utilization": final_state.resource_utilization
                    }
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "brief_quality_score": final_state.brief_quality_score,
                        "research_depth": final_state.research_depth_achieved.value,
                        "complexity_level": final_state.brief_requirements.complexity_level.value,
                        "competitor_insights_count": len(final_state.competitor_intelligence),
                        "strategic_recommendations_count": len(final_state.strategic_insights.strategic_recommendations) if final_state.strategic_insights else 0
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Content brief workflow failed",
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