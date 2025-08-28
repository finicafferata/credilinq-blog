"""
LangGraph-enhanced Search Agent Workflow for advanced web research and competitive intelligence.
"""

import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
from .search_agent import WebSearchAgent
from ...config.database import DatabaseConnection


class ResearchType(str, Enum):
    """Types of research the agent can perform."""
    MARKET_RESEARCH = "market_research"
    TECHNICAL_RESEARCH = "technical_research"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    TREND_ANALYSIS = "trend_analysis"
    GENERAL_RESEARCH = "general_research"
    ACADEMIC_RESEARCH = "academic_research"
    NEWS_ANALYSIS = "news_analysis"


class SourceCredibility(str, Enum):
    """Source credibility levels."""
    VERY_HIGH = "very_high"  # .edu, .gov, major publications
    HIGH = "high"           # Industry publications, established sources
    MEDIUM = "medium"       # General websites with good reputation
    LOW = "low"            # Questionable or unverified sources
    UNKNOWN = "unknown"    # Cannot determine credibility


@dataclass
class ResearchSource:
    """Enhanced research source with detailed metadata."""
    url: str
    title: str
    content: str
    credibility: SourceCredibility
    credibility_score: float
    publication_date: Optional[datetime] = None
    source_type: str = "web"  # web, academic, news, industry
    bias_indicators: List[str] = field(default_factory=list)
    key_topics: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    citation_count: Optional[int] = None


@dataclass
class ResearchInsight:
    """Structured research insight extracted from sources."""
    insight_text: str
    confidence_level: float
    supporting_sources: List[str]
    related_topics: List[str]
    insight_type: str  # finding, trend, recommendation, warning
    importance_score: float = 0.0


@dataclass
class CompetitiveIntelligence:
    """Competitive intelligence findings."""
    competitor_name: str
    market_position: str
    key_strengths: List[str]
    key_weaknesses: List[str]
    recent_developments: List[str]
    market_share_indicators: List[str]
    strategic_moves: List[str]
    threat_level: str  # high, medium, low


class SearchAgentState(WorkflowState):
    """Enhanced state for search agent workflow."""
    
    # Input configuration
    research_query: str = ""
    research_type: ResearchType = ResearchType.GENERAL_RESEARCH
    research_scope: str = "comprehensive"  # focused, comprehensive, deep_dive
    max_sources: int = 10
    time_range: Optional[str] = None  # "1y", "6m", "3m", "1m", "1w"
    
    # Search strategy
    search_strategy: Dict[str, Any] = field(default_factory=dict)
    search_refinements: List[str] = field(default_factory=list)
    targeted_domains: List[str] = field(default_factory=list)
    excluded_domains: List[str] = field(default_factory=list)
    
    # Research results
    research_sources: List[ResearchSource] = field(default_factory=list)
    raw_search_results: List[Dict[str, Any]] = field(default_factory=list)
    verified_sources: List[ResearchSource] = field(default_factory=list)
    
    # Intelligence analysis
    research_insights: List[ResearchInsight] = field(default_factory=list)
    competitive_intelligence: List[CompetitiveIntelligence] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    gap_analysis: List[str] = field(default_factory=list)
    
    # Quality metrics
    source_credibility_distribution: Dict[str, int] = field(default_factory=dict)
    research_confidence_score: float = 0.0
    information_completeness: float = 0.0
    bias_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Output generation
    structured_report: str = ""
    executive_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    follow_up_research_suggestions: List[str] = field(default_factory=list)
    
    # Performance tracking
    search_metrics: Dict[str, Any] = field(default_factory=dict)
    processing_times: Dict[str, float] = field(default_factory=dict)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class SearchAgentWorkflow(LangGraphWorkflowBase[SearchAgentState]):
    """LangGraph workflow for advanced research and competitive intelligence."""
    
    def __init__(
        self, 
        workflow_name: str = "search_agent_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = WebSearchAgent()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> SearchAgentState:
        """Create initial workflow state from context."""
        return SearchAgentState(
            workflow_id=context.get("workflow_id", f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            research_query=context.get("research_query", ""),
            research_type=ResearchType(context.get("research_type", "general_research")),
            research_scope=context.get("research_scope", "comprehensive"),
            max_sources=context.get("max_sources", 10),
            time_range=context.get("time_range"),
            targeted_domains=context.get("targeted_domains", []),
            excluded_domains=context.get("excluded_domains", []),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the search agent workflow graph."""
        workflow = StateGraph(SearchAgentState)
        
        # Define workflow nodes
        workflow.add_node("validate_research_request", self._validate_research_request_node)
        workflow.add_node("develop_search_strategy", self._develop_search_strategy_node)
        workflow.add_node("execute_multi_source_search", self._execute_multi_source_search_node)
        workflow.add_node("verify_and_filter_sources", self._verify_and_filter_sources_node)
        workflow.add_node("extract_intelligence", self._extract_intelligence_node)
        workflow.add_node("analyze_competitive_landscape", self._analyze_competitive_landscape_node)
        workflow.add_node("synthesize_insights", self._synthesize_insights_node)
        workflow.add_node("generate_research_report", self._generate_research_report_node)
        
        # Define workflow edges
        workflow.add_edge("validate_research_request", "develop_search_strategy")
        workflow.add_edge("develop_search_strategy", "execute_multi_source_search")
        workflow.add_edge("execute_multi_source_search", "verify_and_filter_sources")
        workflow.add_edge("verify_and_filter_sources", "extract_intelligence")
        
        # Conditional routing for competitive intelligence
        workflow.add_conditional_edges(
            "extract_intelligence",
            self._should_analyze_competitive_landscape,
            {
                "analyze_competitive": "analyze_competitive_landscape",
                "synthesize_insights": "synthesize_insights"
            }
        )
        workflow.add_edge("analyze_competitive_landscape", "synthesize_insights")
        workflow.add_edge("synthesize_insights", "generate_research_report")
        workflow.add_edge("generate_research_report", END)
        
        # Set entry point
        workflow.set_entry_point("validate_research_request")
        
        return workflow
    
    async def _validate_research_request_node(self, state: SearchAgentState) -> SearchAgentState:
        """Validate research request and parameters."""
        try:
            self._log_progress("Validating research request parameters")
            
            validation_errors = []
            
            # Validate research query
            if not state.research_query or len(state.research_query.strip()) < 3:
                validation_errors.append("Research query must be at least 3 characters long")
            
            # Validate research type
            if state.research_type not in ResearchType:
                validation_errors.append(f"Invalid research type: {state.research_type}")
            
            # Validate scope and parameters
            if state.max_sources < 1 or state.max_sources > 50:
                state.max_sources = max(1, min(50, state.max_sources))
                self._log_progress(f"Adjusted max_sources to {state.max_sources}")
            
            # Validate time range
            valid_time_ranges = ["1y", "6m", "3m", "1m", "1w", "1d", None]
            if state.time_range and state.time_range not in valid_time_ranges:
                validation_errors.append(f"Invalid time range: {state.time_range}")
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 15.0
                
                state.messages.append(HumanMessage(
                    content=f"Research request validated: '{state.research_query}' ({state.research_type.value}). "
                           f"Scope: {state.research_scope}, Max sources: {state.max_sources}"
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Research validation failed: {str(e)}"
            return state
    
    async def _develop_search_strategy_node(self, state: SearchAgentState) -> SearchAgentState:
        """Develop comprehensive search strategy based on research type and scope."""
        try:
            self._log_progress("Developing multi-source search strategy")
            
            search_strategy = await self._create_research_strategy(state)
            state.search_strategy = search_strategy
            
            # Generate search refinements
            search_refinements = await self._generate_search_refinements(state)
            state.search_refinements = search_refinements
            
            # Determine targeted domains based on research type
            targeted_domains = self._get_targeted_domains(state.research_type)
            state.targeted_domains.extend(targeted_domains)
            
            # Set excluded domains for noise reduction
            excluded_domains = self._get_excluded_domains(state.research_type)
            state.excluded_domains.extend(excluded_domains)
            
            state.progress_percentage = 25.0
            
            state.messages.append(SystemMessage(
                content=f"Search strategy developed. {len(search_refinements)} refined queries, "
                       f"{len(targeted_domains)} prioritized domains."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Search strategy development failed: {str(e)}"
            return state
    
    async def _execute_multi_source_search_node(self, state: SearchAgentState) -> SearchAgentState:
        """Execute multi-source search across various information channels."""
        try:
            self._log_progress("Executing multi-source search operations")
            
            start_time = datetime.utcnow()
            all_results = []
            search_metrics = {}
            
            # Execute searches for each refinement
            for i, query_refinement in enumerate(state.search_refinements):
                try:
                    search_start = datetime.utcnow()
                    
                    # Use legacy agent for actual search
                    search_input = {
                        "query": query_refinement,
                        "research_type": state.research_type.value,
                        "max_results": state.max_sources // len(state.search_refinements),
                        "include_analysis": True
                    }
                    
                    result = self.legacy_agent.execute(search_input)
                    
                    if result.success:
                        search_results = result.data.get("search_results", [])
                        all_results.extend(search_results)
                        
                        # Track search metrics
                        search_time = (datetime.utcnow() - search_start).total_seconds()
                        search_metrics[f"query_{i+1}"] = {
                            "query": query_refinement,
                            "results_count": len(search_results),
                            "search_time_seconds": search_time,
                            "success": True
                        }
                    else:
                        search_metrics[f"query_{i+1}"] = {
                            "query": query_refinement,
                            "results_count": 0,
                            "success": False,
                            "error": result.error_message
                        }
                        
                except Exception as search_error:
                    self._log_error(f"Search failed for query '{query_refinement}': {str(search_error)}")
                    search_metrics[f"query_{i+1}"] = {
                        "query": query_refinement,
                        "results_count": 0,
                        "success": False,
                        "error": str(search_error)
                    }
            
            # Deduplicate and process raw results
            unique_results = self._deduplicate_search_results(all_results)
            state.raw_search_results = unique_results[:state.max_sources]
            
            # Calculate processing metrics
            total_time = (datetime.utcnow() - start_time).total_seconds()
            search_metrics["total_search_time"] = total_time
            search_metrics["total_unique_results"] = len(unique_results)
            search_metrics["queries_executed"] = len(state.search_refinements)
            
            state.search_metrics = search_metrics
            state.progress_percentage = 45.0
            
            successful_searches = sum(1 for m in search_metrics.values() 
                                    if isinstance(m, dict) and m.get("success", False))
            
            state.messages.append(SystemMessage(
                content=f"Multi-source search completed. {successful_searches}/{len(state.search_refinements)} "
                       f"queries successful. {len(unique_results)} unique sources found."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Multi-source search failed: {str(e)}"
            return state
    
    async def _verify_and_filter_sources_node(self, state: SearchAgentState) -> SearchAgentState:
        """Verify source credibility and filter for quality."""
        try:
            self._log_progress("Verifying source credibility and filtering for quality")
            
            research_sources = []
            credibility_distribution = {level.value: 0 for level in SourceCredibility}
            
            for raw_result in state.raw_search_results:
                try:
                    # Enhanced source analysis
                    source = await self._analyze_source_credibility(raw_result, state.research_type)
                    research_sources.append(source)
                    
                    # Track credibility distribution
                    credibility_distribution[source.credibility.value] += 1
                    
                except Exception as source_error:
                    self._log_error(f"Source analysis failed: {str(source_error)}")
                    continue
            
            # Filter sources based on credibility and relevance
            verified_sources = []
            for source in research_sources:
                if self._meets_quality_threshold(source, state.research_type):
                    verified_sources.append(source)
            
            # Sort by combined credibility and relevance score
            verified_sources.sort(
                key=lambda s: (s.credibility_score * 0.7 + s.relevance_score * 0.3), 
                reverse=True
            )
            
            state.research_sources = research_sources
            state.verified_sources = verified_sources[:state.max_sources]
            state.source_credibility_distribution = credibility_distribution
            
            # Calculate information completeness
            completeness_score = self._calculate_information_completeness(verified_sources, state.research_type)
            state.information_completeness = completeness_score
            
            state.progress_percentage = 60.0
            
            state.messages.append(SystemMessage(
                content=f"Source verification completed. {len(verified_sources)}/{len(research_sources)} "
                       f"sources meet quality standards. Info completeness: {completeness_score:.1%}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Source verification failed: {str(e)}"
            return state
    
    async def _extract_intelligence_node(self, state: SearchAgentState) -> SearchAgentState:
        """Extract structured intelligence and insights from verified sources."""
        try:
            self._log_progress("Extracting intelligence and insights from sources")
            
            research_insights = []
            gap_analysis = []
            
            # Group sources by topic/theme for better analysis
            source_groups = self._group_sources_by_theme(state.verified_sources)
            
            for theme, sources in source_groups.items():
                try:
                    # Extract insights for each theme
                    theme_insights = await self._extract_theme_insights(theme, sources, state.research_query)
                    research_insights.extend(theme_insights)
                    
                except Exception as theme_error:
                    self._log_error(f"Theme insight extraction failed for {theme}: {str(theme_error)}")
                    continue
            
            # Identify information gaps
            gap_analysis = await self._identify_information_gaps(
                state.research_query, state.research_type, research_insights
            )
            
            # Calculate research confidence
            confidence_score = self._calculate_research_confidence(research_insights, state.verified_sources)
            
            state.research_insights = research_insights
            state.gap_analysis = gap_analysis
            state.research_confidence_score = confidence_score
            state.progress_percentage = 75.0
            
            state.messages.append(SystemMessage(
                content=f"Intelligence extraction completed. {len(research_insights)} insights identified "
                       f"across {len(source_groups)} themes. Confidence: {confidence_score:.1%}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Intelligence extraction failed: {str(e)}"
            return state
    
    async def _analyze_competitive_landscape_node(self, state: SearchAgentState) -> SearchAgentState:
        """Analyze competitive landscape for competitive intelligence research."""
        try:
            self._log_progress("Analyzing competitive landscape and market positioning")
            
            competitive_intelligence = []
            
            # Extract competitor mentions and analysis
            competitors = await self._identify_competitors(state.verified_sources, state.research_query)
            
            for competitor in competitors:
                try:
                    # Analyze each competitor
                    intel = await self._analyze_competitor_intelligence(
                        competitor, state.verified_sources, state.research_query
                    )
                    competitive_intelligence.append(intel)
                    
                except Exception as competitor_error:
                    self._log_error(f"Competitor analysis failed for {competitor}: {str(competitor_error)}")
                    continue
            
            # Perform trend analysis if applicable
            if state.research_type in [ResearchType.TREND_ANALYSIS, ResearchType.MARKET_RESEARCH]:
                trend_analysis = await self._perform_trend_analysis(state.verified_sources, state.research_query)
                state.trend_analysis = trend_analysis
            
            state.competitive_intelligence = competitive_intelligence
            state.progress_percentage = 85.0
            
            state.messages.append(SystemMessage(
                content=f"Competitive landscape analysis completed. {len(competitive_intelligence)} "
                       f"competitors analyzed with detailed intelligence profiles."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Competitive landscape analysis failed: {str(e)}"
            return state
    
    async def _synthesize_insights_node(self, state: SearchAgentState) -> SearchAgentState:
        """Synthesize all research findings into actionable insights."""
        try:
            self._log_progress("Synthesizing research findings into actionable insights")
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(state)
            state.executive_summary = executive_summary
            
            # Generate strategic recommendations
            recommendations = await self._generate_strategic_recommendations(state)
            state.recommendations = recommendations
            
            # Suggest follow-up research
            follow_up_suggestions = await self._suggest_follow_up_research(state)
            state.follow_up_research_suggestions = follow_up_suggestions
            
            # Perform bias assessment
            bias_assessment = await self._assess_research_bias(state.verified_sources)
            state.bias_assessment = bias_assessment
            
            state.progress_percentage = 95.0
            
            state.messages.append(SystemMessage(
                content=f"Research synthesis completed. Executive summary generated with "
                       f"{len(recommendations)} strategic recommendations and "
                       f"{len(follow_up_suggestions)} follow-up research suggestions."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Insight synthesis failed: {str(e)}"
            return state
    
    async def _generate_research_report_node(self, state: SearchAgentState) -> SearchAgentState:
        """Generate comprehensive research report with all findings."""
        try:
            self._log_progress("Generating comprehensive research report")
            
            # Generate structured report
            structured_report = await self._create_structured_report(state)
            state.structured_report = structured_report
            
            # Calculate final performance metrics
            processing_times = {
                "total_workflow_time": (datetime.utcnow() - state.created_at).total_seconds(),
                "search_time": state.search_metrics.get("total_search_time", 0),
                "analysis_time": 0,  # Would be calculated from actual processing
                "report_generation_time": 0
            }
            state.processing_times = processing_times
            
            state.status = WorkflowStatus.COMPLETED
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            # Generate final summary
            final_summary = {
                "research_query": state.research_query,
                "research_type": state.research_type.value,
                "sources_analyzed": len(state.verified_sources),
                "insights_extracted": len(state.research_insights),
                "competitors_analyzed": len(state.competitive_intelligence),
                "confidence_score": state.research_confidence_score,
                "information_completeness": state.information_completeness,
                "processing_time": processing_times["total_workflow_time"]
            }
            
            state.messages.append(SystemMessage(
                content=f"Research report completed successfully. Analyzed {len(state.verified_sources)} sources, "
                       f"extracted {len(state.research_insights)} insights with {state.research_confidence_score:.1%} confidence."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Research report generation failed: {str(e)}"
            return state
    
    def _should_analyze_competitive_landscape(self, state: SearchAgentState) -> str:
        """Determine if competitive landscape analysis should be performed."""
        competitive_types = [
            ResearchType.COMPETITIVE_INTELLIGENCE, 
            ResearchType.MARKET_RESEARCH,
            ResearchType.TREND_ANALYSIS
        ]
        return "analyze_competitive" if state.research_type in competitive_types else "synthesize_insights"
    
    # Helper methods for enhanced research capabilities
    
    async def _create_research_strategy(self, state: SearchAgentState) -> Dict[str, Any]:
        """Create comprehensive research strategy based on type and scope."""
        strategy = {
            "research_focus": self._get_research_focus(state.research_type),
            "information_priorities": self._get_information_priorities(state.research_type),
            "source_preferences": self._get_source_preferences(state.research_type),
            "analysis_depth": state.research_scope,
            "quality_thresholds": self._get_quality_thresholds(state.research_type)
        }
        return strategy
    
    def _get_research_focus(self, research_type: ResearchType) -> List[str]:
        """Get research focus areas based on type."""
        focus_mapping = {
            ResearchType.MARKET_RESEARCH: ["market size", "growth trends", "key players", "market dynamics"],
            ResearchType.TECHNICAL_RESEARCH: ["specifications", "implementation", "best practices", "case studies"],
            ResearchType.COMPETITIVE_INTELLIGENCE: ["competitor analysis", "market positioning", "strategic moves"],
            ResearchType.TREND_ANALYSIS: ["emerging trends", "future predictions", "industry shifts"],
            ResearchType.ACADEMIC_RESEARCH: ["peer-reviewed studies", "research methodology", "citations"],
            ResearchType.NEWS_ANALYSIS: ["recent developments", "breaking news", "expert opinions"]
        }
        return focus_mapping.get(research_type, ["comprehensive overview", "expert insights"])
    
    def _get_information_priorities(self, research_type: ResearchType) -> List[str]:
        """Get information collection priorities."""
        priority_mapping = {
            ResearchType.MARKET_RESEARCH: ["quantitative data", "market reports", "industry analysis"],
            ResearchType.TECHNICAL_RESEARCH: ["documentation", "tutorials", "technical specifications"],
            ResearchType.COMPETITIVE_INTELLIGENCE: ["competitor profiles", "product comparisons", "market share"],
            ResearchType.TREND_ANALYSIS: ["trend reports", "forecasts", "expert predictions"],
            ResearchType.ACADEMIC_RESEARCH: ["peer-reviewed papers", "research studies", "academic sources"],
            ResearchType.NEWS_ANALYSIS: ["recent news", "press releases", "expert commentary"]
        }
        return priority_mapping.get(research_type, ["authoritative sources", "comprehensive information"])
    
    def _get_source_preferences(self, research_type: ResearchType) -> List[str]:
        """Get preferred source types for research."""
        preferences = {
            ResearchType.MARKET_RESEARCH: ["industry reports", "market research firms", "business publications"],
            ResearchType.TECHNICAL_RESEARCH: ["technical documentation", "developer blogs", "official guides"],
            ResearchType.COMPETITIVE_INTELLIGENCE: ["company websites", "industry analysis", "product reviews"],
            ResearchType.TREND_ANALYSIS: ["trend analysis sites", "forecasting reports", "expert blogs"],
            ResearchType.ACADEMIC_RESEARCH: ["academic journals", "research institutions", "scholarly databases"],
            ResearchType.NEWS_ANALYSIS: ["news outlets", "press releases", "industry publications"]
        }
        return preferences.get(research_type, ["authoritative websites", "expert sources"])
    
    def _get_quality_thresholds(self, research_type: ResearchType) -> Dict[str, float]:
        """Get quality thresholds for different research types."""
        return {
            "minimum_credibility": 0.6 if research_type == ResearchType.ACADEMIC_RESEARCH else 0.4,
            "minimum_relevance": 0.7,
            "minimum_confidence": 0.6,
            "bias_tolerance": 0.3 if research_type == ResearchType.NEWS_ANALYSIS else 0.2
        }
    
    async def _generate_search_refinements(self, state: SearchAgentState) -> List[str]:
        """Generate refined search queries for comprehensive coverage."""
        base_query = state.research_query
        refinements = [base_query]  # Always include original query
        
        # Add research type specific refinements
        if state.research_type == ResearchType.MARKET_RESEARCH:
            refinements.extend([
                f"{base_query} market analysis 2024",
                f"{base_query} industry trends report",
                f"{base_query} market size statistics",
                f"{base_query} competitive landscape"
            ])
        elif state.research_type == ResearchType.COMPETITIVE_INTELLIGENCE:
            refinements.extend([
                f"{base_query} competitors analysis",
                f"{base_query} market leaders comparison",
                f"{base_query} competitive positioning",
                f"{base_query} vs alternatives"
            ])
        elif state.research_type == ResearchType.TECHNICAL_RESEARCH:
            refinements.extend([
                f"{base_query} implementation guide",
                f"{base_query} best practices",
                f"{base_query} technical documentation",
                f"{base_query} tutorial"
            ])
        elif state.research_type == ResearchType.TREND_ANALYSIS:
            refinements.extend([
                f"{base_query} trends 2024",
                f"{base_query} future predictions",
                f"{base_query} emerging developments",
                f"{base_query} forecast analysis"
            ])
        
        # Add time-range specific queries if specified
        if state.time_range:
            time_specific = f"{base_query} {self._convert_time_range_to_text(state.time_range)}"
            refinements.append(time_specific)
        
        return list(set(refinements))  # Remove duplicates
    
    def _convert_time_range_to_text(self, time_range: str) -> str:
        """Convert time range code to search text."""
        mapping = {
            "1y": "past year",
            "6m": "past 6 months", 
            "3m": "past quarter",
            "1m": "past month",
            "1w": "past week",
            "1d": "today"
        }
        return mapping.get(time_range, "recent")
    
    def _get_targeted_domains(self, research_type: ResearchType) -> List[str]:
        """Get prioritized domains for specific research types."""
        domain_mapping = {
            ResearchType.ACADEMIC_RESEARCH: [".edu", ".org", "scholar.google.com", "researchgate.net"],
            ResearchType.NEWS_ANALYSIS: ["reuters.com", "bloomberg.com", "wsj.com", "ft.com"],
            ResearchType.TECHNICAL_RESEARCH: ["github.com", "stackoverflow.com", "medium.com"],
            ResearchType.MARKET_RESEARCH: ["mckinsey.com", "bcg.com", "deloitte.com", "pwc.com"]
        }
        return domain_mapping.get(research_type, [])
    
    def _get_excluded_domains(self, research_type: ResearchType) -> List[str]:
        """Get domains to exclude for noise reduction."""
        general_exclusions = ["pinterest.com", "instagram.com", "tiktok.com"]
        
        type_specific = {
            ResearchType.ACADEMIC_RESEARCH: ["wikipedia.org", "quora.com"],
            ResearchType.TECHNICAL_RESEARCH: ["pinterest.com", "facebook.com"],
            ResearchType.NEWS_ANALYSIS: ["opinion blogs", "personal websites"]
        }
        
        return general_exclusions + type_specific.get(research_type, [])
    
    def _deduplicate_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate search results based on URL and content similarity."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url not in seen_urls and url:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    async def _analyze_source_credibility(
        self, 
        raw_result: Dict[str, Any], 
        research_type: ResearchType
    ) -> ResearchSource:
        """Analyze source credibility with enhanced metadata."""
        try:
            url = raw_result.get("url", "")
            title = raw_result.get("title", "")
            content = raw_result.get("content", "")
            
            # Determine credibility level and score
            credibility, credibility_score = self._assess_detailed_credibility(url, content, research_type)
            
            # Extract key topics
            key_topics = self._extract_key_topics(content, title)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(content, title, research_type)
            
            # Identify bias indicators
            bias_indicators = self._identify_bias_indicators(content, url)
            
            return ResearchSource(
                url=url,
                title=title,
                content=content,
                credibility=credibility,
                credibility_score=credibility_score,
                source_type=self._determine_source_type(url),
                bias_indicators=bias_indicators,
                key_topics=key_topics,
                relevance_score=relevance_score
            )
            
        except Exception as e:
            self._log_error(f"Source analysis failed: {str(e)}")
            return ResearchSource(
                url=raw_result.get("url", ""),
                title=raw_result.get("title", ""),
                content=raw_result.get("content", ""),
                credibility=SourceCredibility.UNKNOWN,
                credibility_score=0.5
            )
    
    def _assess_detailed_credibility(
        self, 
        url: str, 
        content: str, 
        research_type: ResearchType
    ) -> tuple[SourceCredibility, float]:
        """Assess detailed source credibility."""
        score = 0.5  # Base score
        
        # Domain credibility
        if any(domain in url.lower() for domain in [".edu", ".gov"]):
            score += 0.4
            level = SourceCredibility.VERY_HIGH
        elif any(domain in url.lower() for domain in ["reuters.com", "bloomberg.com", "wsj.com"]):
            score += 0.3
            level = SourceCredibility.HIGH
        elif any(domain in url.lower() for domain in ["mckinsey.com", "bcg.com", "deloitte.com"]):
            score += 0.25
            level = SourceCredibility.HIGH
        else:
            level = SourceCredibility.MEDIUM
        
        # Content quality indicators
        if len(content) > 500:  # Substantial content
            score += 0.1
        
        quality_indicators = ["research", "study", "analysis", "data", "report", "methodology"]
        if any(indicator in content.lower() for indicator in quality_indicators):
            score += 0.15
        
        # Research type specific adjustments
        if research_type == ResearchType.ACADEMIC_RESEARCH:
            academic_indicators = ["peer-reviewed", "journal", "doi:", "citation"]
            if any(indicator in content.lower() for indicator in academic_indicators):
                score += 0.2
        
        return level, min(score, 1.0)
    
    def _extract_key_topics(self, content: str, title: str) -> List[str]:
        """Extract key topics from content."""
        # Simple topic extraction (could be enhanced with NLP)
        text = f"{title} {content}".lower()
        
        # Common business and technical topics
        topics = []
        topic_keywords = {
            "artificial intelligence": ["ai", "artificial intelligence", "machine learning", "neural network"],
            "market analysis": ["market", "analysis", "trends", "growth", "revenue"],
            "technology": ["technology", "software", "platform", "system", "digital"],
            "business strategy": ["strategy", "business", "competitive", "planning", "strategic"],
            "finance": ["finance", "financial", "investment", "funding", "revenue", "profit"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics[:5]  # Limit to top 5 topics
    
    def _calculate_relevance_score(
        self, 
        content: str, 
        title: str, 
        research_type: ResearchType
    ) -> float:
        """Calculate relevance score based on content and research type."""
        # Simple relevance calculation based on keyword presence
        score = 0.5  # Base relevance
        
        # Research type specific keywords
        type_keywords = self._get_research_focus(research_type)
        text = f"{title} {content}".lower()
        
        keyword_matches = sum(1 for keyword in type_keywords if keyword.lower() in text)
        score += (keyword_matches / len(type_keywords)) * 0.5
        
        return min(score, 1.0)
    
    def _identify_bias_indicators(self, content: str, url: str) -> List[str]:
        """Identify potential bias indicators in content."""
        bias_indicators = []
        content_lower = content.lower()
        
        # Opinion indicators
        opinion_words = ["i think", "in my opinion", "i believe", "clearly", "obviously"]
        if any(word in content_lower for word in opinion_words):
            bias_indicators.append("opinion_language")
        
        # Emotional language
        emotional_words = ["amazing", "terrible", "shocking", "unbelievable", "revolutionary"]
        if any(word in content_lower for word in emotional_words):
            bias_indicators.append("emotional_language")
        
        # Commercial bias
        if any(word in content_lower for word in ["buy now", "special offer", "limited time"]):
            bias_indicators.append("commercial_bias")
        
        return bias_indicators
    
    def _determine_source_type(self, url: str) -> str:
        """Determine source type based on URL."""
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in [".edu", "scholar.google"]):
            return "academic"
        elif any(domain in url_lower for domain in ["reuters.com", "bloomberg.com", "wsj.com", "cnn.com", "bbc.com"]):
            return "news"
        elif any(domain in url_lower for domain in ["github.com", "stackoverflow.com"]):
            return "technical"
        elif any(domain in url_lower for domain in [".gov"]):
            return "government"
        else:
            return "web"
    
    def _meets_quality_threshold(self, source: ResearchSource, research_type: ResearchType) -> bool:
        """Determine if source meets quality threshold for research type."""
        thresholds = self._get_quality_thresholds(research_type)
        
        return (source.credibility_score >= thresholds["minimum_credibility"] and
                source.relevance_score >= thresholds["minimum_relevance"])
    
    def _calculate_information_completeness(
        self, 
        sources: List[ResearchSource], 
        research_type: ResearchType
    ) -> float:
        """Calculate information completeness based on sources and research focus."""
        if not sources:
            return 0.0
        
        focus_areas = self._get_research_focus(research_type)
        covered_areas = 0
        
        for focus_area in focus_areas:
            if any(focus_area.lower() in " ".join(source.key_topics).lower() or
                   focus_area.lower() in source.content.lower() for source in sources):
                covered_areas += 1
        
        return covered_areas / len(focus_areas) if focus_areas else 0.8
    
    def _group_sources_by_theme(self, sources: List[ResearchSource]) -> Dict[str, List[ResearchSource]]:
        """Group sources by common themes."""
        theme_groups = {}
        
        for source in sources:
            # Simple theme grouping based on key topics
            primary_theme = source.key_topics[0] if source.key_topics else "general"
            
            if primary_theme not in theme_groups:
                theme_groups[primary_theme] = []
            theme_groups[primary_theme].append(source)
        
        return theme_groups
    
    async def _extract_theme_insights(
        self, 
        theme: str, 
        sources: List[ResearchSource], 
        research_query: str
    ) -> List[ResearchInsight]:
        """Extract insights for a specific theme from grouped sources."""
        insights = []
        
        # Simple insight extraction (could be enhanced with AI)
        combined_content = " ".join([source.content for source in sources])
        
        # Create insights based on source analysis
        if len(sources) > 1:
            insights.append(ResearchInsight(
                insight_text=f"Multiple sources confirm {theme} relevance to {research_query}",
                confidence_level=0.8,
                supporting_sources=[source.url for source in sources],
                related_topics=[theme],
                insight_type="finding",
                importance_score=0.7
            ))
        
        return insights
    
    async def _identify_information_gaps(
        self, 
        research_query: str, 
        research_type: ResearchType, 
        insights: List[ResearchInsight]
    ) -> List[str]:
        """Identify gaps in research coverage."""
        gaps = []
        focus_areas = self._get_research_focus(research_type)
        
        covered_areas = set()
        for insight in insights:
            covered_areas.update(insight.related_topics)
        
        for focus_area in focus_areas:
            if not any(focus_area.lower() in topic.lower() for topic in covered_areas):
                gaps.append(f"Limited information on {focus_area}")
        
        return gaps
    
    def _calculate_research_confidence(
        self, 
        insights: List[ResearchInsight], 
        sources: List[ResearchSource]
    ) -> float:
        """Calculate overall research confidence score."""
        if not insights or not sources:
            return 0.0
        
        # Average insight confidence
        avg_insight_confidence = sum(insight.confidence_level for insight in insights) / len(insights)
        
        # Average source credibility
        avg_source_credibility = sum(source.credibility_score for source in sources) / len(sources)
        
        # Source diversity bonus
        source_types = set(source.source_type for source in sources)
        diversity_bonus = min(len(source_types) * 0.1, 0.3)
        
        return min(avg_insight_confidence * 0.4 + avg_source_credibility * 0.4 + diversity_bonus, 1.0)
    
    async def _identify_competitors(
        self, 
        sources: List[ResearchSource], 
        research_query: str
    ) -> List[str]:
        """Identify competitors mentioned in sources."""
        competitors = []
        
        # Simple competitor identification
        competitor_indicators = ["competitor", "rival", "alternative", "versus", "vs", "compared to"]
        
        for source in sources:
            content_lower = source.content.lower()
            if any(indicator in content_lower for indicator in competitor_indicators):
                # Extract potential competitor names (simplified)
                words = source.content.split()
                for i, word in enumerate(words):
                    if word.lower() in competitor_indicators and i + 1 < len(words):
                        potential_competitor = words[i + 1].strip(".,!?")
                        if potential_competitor not in competitors:
                            competitors.append(potential_competitor)
        
        return competitors[:5]  # Limit to top 5 competitors
    
    async def _analyze_competitor_intelligence(
        self, 
        competitor: str, 
        sources: List[ResearchSource], 
        research_query: str
    ) -> CompetitiveIntelligence:
        """Analyze competitive intelligence for a specific competitor."""
        # Simple competitive analysis (could be enhanced with AI)
        relevant_sources = [s for s in sources if competitor.lower() in s.content.lower()]
        
        return CompetitiveIntelligence(
            competitor_name=competitor,
            market_position="Identified competitor",
            key_strengths=["Market presence"],
            key_weaknesses=["Analysis pending"],
            recent_developments=["Mentioned in research"],
            market_share_indicators=["Data not available"],
            strategic_moves=["Under investigation"],
            threat_level="medium"
        )
    
    async def _perform_trend_analysis(
        self, 
        sources: List[ResearchSource], 
        research_query: str
    ) -> Dict[str, Any]:
        """Perform trend analysis on sources."""
        trend_indicators = ["trend", "growing", "increasing", "declining", "emerging", "future"]
        
        trends_mentioned = 0
        for source in sources:
            if any(indicator in source.content.lower() for indicator in trend_indicators):
                trends_mentioned += 1
        
        return {
            "trend_coverage": trends_mentioned / len(sources) if sources else 0,
            "trend_indicators_found": trends_mentioned,
            "trend_confidence": "medium" if trends_mentioned > len(sources) * 0.3 else "low"
        }
    
    async def _generate_executive_summary(self, state: SearchAgentState) -> str:
        """Generate executive summary of research findings."""
        summary_parts = []
        
        summary_parts.append(f"Research Query: {state.research_query}")
        summary_parts.append(f"Research Type: {state.research_type.value}")
        summary_parts.append(f"Sources Analyzed: {len(state.verified_sources)}")
        summary_parts.append(f"Insights Generated: {len(state.research_insights)}")
        summary_parts.append(f"Research Confidence: {state.research_confidence_score:.1%}")
        summary_parts.append(f"Information Completeness: {state.information_completeness:.1%}")
        
        if state.competitive_intelligence:
            summary_parts.append(f"Competitors Identified: {len(state.competitive_intelligence)}")
        
        return ". ".join(summary_parts)
    
    async def _generate_strategic_recommendations(self, state: SearchAgentState) -> List[str]:
        """Generate strategic recommendations based on research findings."""
        recommendations = []
        
        # Base recommendations on research confidence and gaps
        if state.research_confidence_score < 0.7:
            recommendations.append("Conduct additional research to increase confidence in findings")
        
        if state.information_completeness < 0.8:
            recommendations.append("Investigate identified information gaps for comprehensive understanding")
        
        if state.gap_analysis:
            recommendations.append(f"Focus on {len(state.gap_analysis)} identified knowledge gaps")
        
        # Research type specific recommendations
        if state.research_type == ResearchType.COMPETITIVE_INTELLIGENCE:
            recommendations.append("Monitor competitive developments and update intelligence regularly")
        elif state.research_type == ResearchType.MARKET_RESEARCH:
            recommendations.append("Track market trends and validate findings with primary research")
        
        return recommendations
    
    async def _suggest_follow_up_research(self, state: SearchAgentState) -> List[str]:
        """Suggest follow-up research topics."""
        suggestions = []
        
        # Based on information gaps
        for gap in state.gap_analysis:
            suggestions.append(f"Research {gap.lower()}")
        
        # Based on research type
        if state.research_type == ResearchType.MARKET_RESEARCH:
            suggestions.extend([
                "Conduct primary market research surveys",
                "Interview industry experts",
                "Analyze competitor financial reports"
            ])
        elif state.research_type == ResearchType.TECHNICAL_RESEARCH:
            suggestions.extend([
                "Review technical documentation",
                "Conduct proof-of-concept testing",
                "Analyze implementation case studies"
            ])
        
        return suggestions[:5]
    
    async def _assess_research_bias(self, sources: List[ResearchSource]) -> Dict[str, Any]:
        """Assess potential bias in research sources."""
        total_sources = len(sources)
        if total_sources == 0:
            return {"bias_assessment": "No sources to assess"}
        
        biased_sources = sum(1 for source in sources if source.bias_indicators)
        bias_percentage = biased_sources / total_sources
        
        return {
            "bias_percentage": bias_percentage,
            "bias_level": "high" if bias_percentage > 0.5 else "medium" if bias_percentage > 0.2 else "low",
            "biased_sources_count": biased_sources,
            "total_sources": total_sources
        }
    
    async def _create_structured_report(self, state: SearchAgentState) -> str:
        """Create comprehensive structured research report."""
        report_sections = []
        
        # Header
        report_sections.append("# Research Intelligence Report")
        report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"**Workflow ID:** {state.workflow_id}")
        report_sections.append("")
        
        # Executive Summary
        report_sections.append("## Executive Summary")
        report_sections.append(state.executive_summary)
        report_sections.append("")
        
        # Research Methodology
        report_sections.append("## Research Methodology")
        report_sections.append(f"- **Research Type:** {state.research_type.value}")
        report_sections.append(f"- **Search Scope:** {state.research_scope}")
        report_sections.append(f"- **Sources Analyzed:** {len(state.verified_sources)}")
        report_sections.append(f"- **Search Queries:** {len(state.search_refinements)}")
        report_sections.append("")
        
        # Key Insights
        if state.research_insights:
            report_sections.append("## Key Research Insights")
            for i, insight in enumerate(state.research_insights, 1):
                report_sections.append(f"{i}. **{insight.insight_type.title()}:** {insight.insight_text}")
                report_sections.append(f"   - Confidence: {insight.confidence_level:.1%}")
                report_sections.append(f"   - Supporting Sources: {len(insight.supporting_sources)}")
            report_sections.append("")
        
        # Competitive Intelligence
        if state.competitive_intelligence:
            report_sections.append("## Competitive Landscape")
            for competitor in state.competitive_intelligence:
                report_sections.append(f"### {competitor.competitor_name}")
                report_sections.append(f"- **Market Position:** {competitor.market_position}")
                report_sections.append(f"- **Threat Level:** {competitor.threat_level}")
                report_sections.append(f"- **Key Strengths:** {', '.join(competitor.key_strengths)}")
            report_sections.append("")
        
        # Strategic Recommendations
        if state.recommendations:
            report_sections.append("## Strategic Recommendations")
            for i, rec in enumerate(state.recommendations, 1):
                report_sections.append(f"{i}. {rec}")
            report_sections.append("")
        
        # Quality Assessment
        report_sections.append("## Research Quality Assessment")
        report_sections.append(f"- **Research Confidence:** {state.research_confidence_score:.1%}")
        report_sections.append(f"- **Information Completeness:** {state.information_completeness:.1%}")
        report_sections.append(f"- **Source Credibility Distribution:**")
        for credibility, count in state.source_credibility_distribution.items():
            report_sections.append(f"  - {credibility}: {count} sources")
        report_sections.append("")
        
        # Information Gaps
        if state.gap_analysis:
            report_sections.append("## Information Gaps")
            for gap in state.gap_analysis:
                report_sections.append(f"- {gap}")
            report_sections.append("")
        
        # Follow-up Research
        if state.follow_up_research_suggestions:
            report_sections.append("## Recommended Follow-up Research")
            for suggestion in state.follow_up_research_suggestions:
                report_sections.append(f"- {suggestion}")
            report_sections.append("")
        
        # Source References
        report_sections.append("## Source References")
        for i, source in enumerate(state.verified_sources, 1):
            report_sections.append(f"{i}. **{source.title}**")
            report_sections.append(f"   - URL: {source.url}")
            report_sections.append(f"   - Credibility: {source.credibility.value} ({source.credibility_score:.2f})")
            report_sections.append(f"   - Source Type: {source.source_type}")
            if source.key_topics:
                report_sections.append(f"   - Key Topics: {', '.join(source.key_topics)}")
            report_sections.append("")
        
        return "\n".join(report_sections)
    
    async def execute_workflow(
        self,
        research_query: str,
        research_type: str = "general_research",
        research_scope: str = "comprehensive",
        max_sources: int = 10,
        time_range: Optional[str] = None,
        targeted_domains: Optional[List[str]] = None,
        excluded_domains: Optional[List[str]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the search agent workflow."""
        
        context = {
            "research_query": research_query,
            "research_type": research_type,
            "research_scope": research_scope,
            "max_sources": max_sources,
            "time_range": time_range,
            "targeted_domains": targeted_domains or [],
            "excluded_domains": excluded_domains or [],
            "workflow_id": f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "research_query": final_state.research_query,
                    "research_type": final_state.research_type.value,
                    "executive_summary": final_state.executive_summary,
                    "structured_report": final_state.structured_report,
                    "research_insights": [
                        {
                            "insight_text": insight.insight_text,
                            "confidence_level": insight.confidence_level,
                            "insight_type": insight.insight_type,
                            "importance_score": insight.importance_score,
                            "supporting_sources_count": len(insight.supporting_sources),
                            "related_topics": insight.related_topics
                        }
                        for insight in final_state.research_insights
                    ],
                    "competitive_intelligence": [
                        {
                            "competitor_name": comp.competitor_name,
                            "market_position": comp.market_position,
                            "threat_level": comp.threat_level,
                            "key_strengths": comp.key_strengths,
                            "key_weaknesses": comp.key_weaknesses,
                            "recent_developments": comp.recent_developments
                        }
                        for comp in final_state.competitive_intelligence
                    ],
                    "verified_sources": [
                        {
                            "title": source.title,
                            "url": source.url,
                            "credibility": source.credibility.value,
                            "credibility_score": source.credibility_score,
                            "source_type": source.source_type,
                            "relevance_score": source.relevance_score,
                            "key_topics": source.key_topics,
                            "bias_indicators": source.bias_indicators
                        }
                        for source in final_state.verified_sources
                    ],
                    "quality_metrics": {
                        "research_confidence": final_state.research_confidence_score,
                        "information_completeness": final_state.information_completeness,
                        "source_credibility_distribution": final_state.source_credibility_distribution,
                        "bias_assessment": final_state.bias_assessment
                    },
                    "recommendations": final_state.recommendations,
                    "gap_analysis": final_state.gap_analysis,
                    "follow_up_suggestions": final_state.follow_up_research_suggestions,
                    "search_metrics": final_state.search_metrics,
                    "processing_times": final_state.processing_times
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "sources_analyzed": len(final_state.verified_sources),
                        "insights_extracted": len(final_state.research_insights),
                        "competitors_identified": len(final_state.competitive_intelligence),
                        "research_confidence": final_state.research_confidence_score,
                        "information_completeness": final_state.information_completeness
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Search workflow failed",
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