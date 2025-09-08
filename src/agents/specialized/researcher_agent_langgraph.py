"""
LangGraph-based Researcher Agent with advanced multi-phase research workflow.

This agent conducts comprehensive research using sophisticated workflows with
query optimization, source validation, and iterative refinement.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class ResearcherState(WorkflowState):
    """State for Researcher LangGraph workflow."""
    # Input requirements
    research_topic: str = ""
    research_depth: str = "comprehensive"  # basic, comprehensive, expert
    target_audience: str = "general"
    specific_questions: List[str] = field(default_factory=list)
    
    # Research planning
    research_plan: Dict[str, Any] = field(default_factory=dict)
    search_queries: List[str] = field(default_factory=list)
    research_angles: List[str] = field(default_factory=list)
    
    # Research execution
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    validated_sources: List[Dict[str, Any]] = field(default_factory=list)
    research_findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis and synthesis
    key_insights: List[str] = field(default_factory=list)
    conflicting_information: List[Dict[str, Any]] = field(default_factory=list)
    research_gaps: List[str] = field(default_factory=list)
    
    # Quality control
    source_quality_scores: Dict[str, float] = field(default_factory=dict)
    research_completeness: float = 0.0
    
    # Workflow control
    requires_additional_research: bool = False
    research_iterations: int = 0
    max_iterations: int = 3

class ResearcherAgentLangGraph(LangGraphWorkflowBase[ResearcherState]):
    """
    LangGraph-based Researcher with sophisticated multi-phase research workflow.
    """
    
    def __init__(self, workflow_name: str = "Researcher_workflow"):
        super().__init__(workflow_name=workflow_name)
        logger.info("ResearcherAgentLangGraph initialized with advanced research capabilities")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(ResearcherState)
        
        # Define workflow nodes
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("generate_queries", self._generate_queries)
        workflow.add_node("conduct_search", self._conduct_search)
        workflow.add_node("validate_sources", self._validate_sources)
        workflow.add_node("extract_insights", self._extract_insights)
        workflow.add_node("identify_gaps", self._identify_gaps)
        workflow.add_node("synthesize_findings", self._synthesize_findings)
        workflow.add_node("assess_completeness", self._assess_completeness)
        workflow.add_node("conduct_additional_research", self._conduct_additional_research)
        workflow.add_node("finalize_research", self._finalize_research)
        
        # Define workflow edges
        workflow.set_entry_point("plan_research")
        
        workflow.add_edge("plan_research", "generate_queries")
        workflow.add_edge("generate_queries", "conduct_search")
        workflow.add_edge("conduct_search", "validate_sources")
        workflow.add_edge("validate_sources", "extract_insights")
        workflow.add_edge("extract_insights", "identify_gaps")
        workflow.add_edge("identify_gaps", "synthesize_findings")
        workflow.add_edge("synthesize_findings", "assess_completeness")
        
        # Conditional routing based on research completeness
        workflow.add_conditional_edges(
            "assess_completeness",
            self._should_continue_research,
            {
                "continue": "conduct_additional_research",
                "finalize": "finalize_research"
            }
        )
        
        workflow.add_edge("conduct_additional_research", "conduct_search")
        workflow.set_finish_point("finalize_research")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> ResearcherState:
        """Create initial workflow state from input."""
        return ResearcherState(
            research_topic=input_data.get("research_topic", input_data.get("topic", "")),
            research_depth=input_data.get("research_depth", "comprehensive"),
            target_audience=input_data.get("target_audience", "general"),
            specific_questions=input_data.get("specific_questions", []),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="plan_research"
        )
    
    def _plan_research(self, state: ResearcherState) -> ResearcherState:
        """Plan the research approach and methodology."""
        logger.info(f"Planning research for topic: {state.research_topic}")
        
        # Create research plan based on topic and depth
        research_plan = {
            "primary_focus": state.research_topic,
            "research_methodology": self._determine_methodology(state.research_depth),
            "expected_sources": self._identify_source_types(state.research_topic),
            "research_timeline": "multi-phase",
            "quality_criteria": self._define_quality_criteria(state.target_audience)
        }
        
        # Identify research angles
        research_angles = self._identify_research_angles(state.research_topic)
        
        state.research_plan = research_plan
        state.research_angles = research_angles
        state.current_step = "generate_queries"
        
        return state
    
    def _generate_queries(self, state: ResearcherState) -> ResearcherState:
        """Generate optimized search queries for research."""
        logger.info("Generating search queries for research")
        
        queries = []
        
        # Primary queries based on main topic
        primary_queries = [
            state.research_topic,
            f"{state.research_topic} overview",
            f"{state.research_topic} latest developments",
            f"{state.research_topic} best practices"
        ]
        queries.extend(primary_queries)
        
        # Angle-specific queries
        for angle in state.research_angles:
            queries.append(f"{state.research_topic} {angle}")
        
        # Question-specific queries
        for question in state.specific_questions:
            queries.append(question)
            queries.append(f"{state.research_topic} {question}")
        
        state.search_queries = queries
        state.current_step = "conduct_search"
        
        return state
    
    def _conduct_search(self, state: ResearcherState) -> ResearcherState:
        """Conduct search for each query."""
        logger.info(f"Conducting search for {len(state.search_queries)} queries")
        
        # Simulate search results (in real implementation, would use actual search)
        search_results = []
        
        for query in state.search_queries[:10]:  # Limit for demo
            # Simulate search result
            results = [
                {
                    "query": query,
                    "title": f"Research Result for {query}",
                    "url": f"https://example.com/research/{query.replace(' ', '-')}",
                    "snippet": f"Comprehensive information about {query} including key insights and data.",
                    "source_type": "academic" if "overview" in query else "industry",
                    "credibility_score": 0.8,
                    "relevance_score": 0.9,
                    "publication_date": "2024-01-01"
                }
            ]
            search_results.extend(results)
        
        state.search_results = search_results
        state.current_step = "validate_sources"
        
        return state
    
    def _validate_sources(self, state: ResearcherState) -> ResearcherState:
        """Validate and score source quality."""
        logger.info("Validating source quality and credibility")
        
        validated_sources = []
        quality_scores = {}
        
        for result in state.search_results:
            # Validate source quality
            quality_score = self._assess_source_quality(result)
            
            if quality_score >= 0.6:  # Quality threshold
                validated_source = {
                    **result,
                    "validation_score": quality_score,
                    "validation_criteria": {
                        "credibility": result.get("credibility_score", 0.8),
                        "relevance": result.get("relevance_score", 0.8),
                        "recency": 0.9,  # Simulated recency score
                        "authority": 0.8   # Simulated authority score
                    }
                }
                validated_sources.append(validated_source)
                quality_scores[result["url"]] = quality_score
        
        state.validated_sources = validated_sources
        state.source_quality_scores = quality_scores
        state.current_step = "extract_insights"
        
        return state
    
    def _extract_insights(self, state: ResearcherState) -> ResearcherState:
        """Extract key insights from validated sources."""
        logger.info("Extracting insights from research sources")
        
        research_findings = []
        key_insights = []
        
        for source in state.validated_sources:
            # Extract insights from each source
            finding = {
                "source_url": source["url"],
                "source_title": source["title"],
                "key_points": [
                    f"Key insight from {source['title']}",
                    f"Important data point related to {state.research_topic}",
                    f"Trend identified in {source['source_type']} research"
                ],
                "data_points": [
                    {"metric": "growth_rate", "value": "15%", "context": "year-over-year"},
                    {"metric": "adoption_rate", "value": "68%", "context": "enterprise usage"}
                ],
                "source_type": source["source_type"],
                "confidence_level": source["validation_score"]
            }
            research_findings.append(finding)
        
        # Synthesize key insights across sources
        key_insights = [
            f"Primary insight about {state.research_topic}",
            f"Secondary trend in {state.research_topic} domain",
            f"Emerging pattern in {state.research_topic} applications",
            f"Critical factor for {state.research_topic} success"
        ]
        
        state.research_findings = research_findings
        state.key_insights = key_insights
        state.current_step = "identify_gaps"
        
        return state
    
    def _identify_gaps(self, state: ResearcherState) -> ResearcherState:
        """Identify gaps in research coverage."""
        logger.info("Identifying research gaps and areas for additional investigation")
        
        research_gaps = []
        
        # Check coverage of research angles
        covered_angles = set()
        for finding in state.research_findings:
            # Analyze which angles are covered
            for angle in state.research_angles:
                if angle.lower() in finding["source_title"].lower():
                    covered_angles.add(angle)
        
        uncovered_angles = set(state.research_angles) - covered_angles
        for angle in uncovered_angles:
            research_gaps.append(f"Limited coverage of {angle} aspect")
        
        # Check question coverage
        answered_questions = 0
        for question in state.specific_questions:
            question_covered = any(
                question.lower() in finding["source_title"].lower() 
                for finding in state.research_findings
            )
            if question_covered:
                answered_questions += 1
            else:
                research_gaps.append(f"Incomplete answer to: {question}")
        
        # Add general gaps
        if len(state.research_findings) < 5:
            research_gaps.append("Insufficient source diversity")
        
        if not any(f["source_type"] == "academic" for f in state.research_findings):
            research_gaps.append("Lack of academic sources")
        
        state.research_gaps = research_gaps
        state.current_step = "synthesize_findings"
        
        return state
    
    def _synthesize_findings(self, state: ResearcherState) -> ResearcherState:
        """Synthesize findings into coherent research summary."""
        logger.info("Synthesizing research findings")
        
        # Identify conflicting information
        conflicting_info = []
        
        # Simple conflict detection (in real implementation, would be more sophisticated)
        if len(state.research_findings) > 1:
            conflicting_info.append({
                "topic": "Market size estimates",
                "conflict": "Different sources report varying market size figures",
                "sources": [f["source_url"] for f in state.research_findings[:2]]
            })
        
        state.conflicting_information = conflicting_info
        state.current_step = "assess_completeness"
        
        return state
    
    def _assess_completeness(self, state: ResearcherState) -> ResearcherState:
        """Assess research completeness and quality."""
        logger.info("Assessing research completeness")
        
        # Calculate completeness score
        completeness_factors = []
        
        # Source coverage
        source_coverage = min(len(state.validated_sources) / 10, 1.0)  # Target 10 sources
        completeness_factors.append(source_coverage)
        
        # Question coverage
        if state.specific_questions:
            answered = sum(1 for q in state.specific_questions 
                          if not any(q in gap for gap in state.research_gaps))
            question_coverage = answered / len(state.specific_questions)
            completeness_factors.append(question_coverage)
        else:
            completeness_factors.append(0.8)  # Default if no specific questions
        
        # Angle coverage
        covered_angles = len(state.research_angles) - sum(1 for gap in state.research_gaps if "aspect" in gap)
        angle_coverage = covered_angles / len(state.research_angles) if state.research_angles else 1.0
        completeness_factors.append(angle_coverage)
        
        # Source quality
        avg_quality = sum(state.source_quality_scores.values()) / len(state.source_quality_scores) if state.source_quality_scores else 0.8
        completeness_factors.append(avg_quality)
        
        research_completeness = sum(completeness_factors) / len(completeness_factors)
        state.research_completeness = research_completeness
        
        state.current_step = "finalize_research"
        
        return state
    
    def _should_continue_research(self, state: ResearcherState) -> str:
        """Determine if additional research is needed."""
        needs_more_research = (
            state.research_completeness < 0.7 or
            len(state.research_gaps) > 3
        ) and state.research_iterations < state.max_iterations
        
        if needs_more_research:
            state.requires_additional_research = True
            state.research_iterations += 1
            return "continue"
        else:
            return "finalize"
    
    def _conduct_additional_research(self, state: ResearcherState) -> ResearcherState:
        """Conduct additional research to fill gaps."""
        logger.info(f"Conducting additional research (iteration {state.research_iterations})")
        
        # Generate additional queries based on gaps
        additional_queries = []
        for gap in state.research_gaps:
            if "aspect" in gap:
                aspect = gap.replace("Limited coverage of ", "").replace(" aspect", "")
                additional_queries.append(f"{state.research_topic} {aspect}")
        
        # Add to existing queries
        state.search_queries.extend(additional_queries)
        state.current_step = "conduct_search"
        
        return state
    
    def _finalize_research(self, state: ResearcherState) -> ResearcherState:
        """Finalize research with comprehensive summary."""
        logger.info("Finalizing research results")
        
        # Add final metadata
        state.metadata.update({
            "total_sources_found": len(state.search_results),
            "validated_sources": len(state.validated_sources),
            "key_insights_count": len(state.key_insights),
            "research_completeness": state.research_completeness,
            "research_gaps_identified": len(state.research_gaps),
            "research_iterations": state.research_iterations,
            "average_source_quality": sum(state.source_quality_scores.values()) / len(state.source_quality_scores) if state.source_quality_scores else 0,
            "research_complete": True
        })
        
        state.current_step = "completed"
        
        return state
    
    def _determine_methodology(self, depth: str) -> Dict[str, Any]:
        """Determine research methodology based on depth."""
        methodologies = {
            "basic": {
                "source_count_target": 5,
                "source_types": ["web", "industry"],
                "validation_level": "standard"
            },
            "comprehensive": {
                "source_count_target": 10,
                "source_types": ["web", "industry", "academic"],
                "validation_level": "thorough"
            },
            "expert": {
                "source_count_target": 20,
                "source_types": ["web", "industry", "academic", "expert"],
                "validation_level": "rigorous"
            }
        }
        return methodologies.get(depth, methodologies["comprehensive"])
    
    def _identify_source_types(self, topic: str) -> List[str]:
        """Identify appropriate source types for topic."""
        return ["academic_papers", "industry_reports", "expert_opinions", "case_studies", "news_articles"]
    
    def _define_quality_criteria(self, audience: str) -> Dict[str, float]:
        """Define quality criteria based on target audience."""
        criteria_by_audience = {
            "general": {"credibility": 0.7, "readability": 0.8, "recency": 0.6},
            "expert": {"credibility": 0.9, "accuracy": 0.9, "depth": 0.8},
            "business": {"credibility": 0.8, "practicality": 0.9, "recency": 0.8}
        }
        return criteria_by_audience.get(audience, criteria_by_audience["general"])
    
    def _identify_research_angles(self, topic: str) -> List[str]:
        """Identify different angles to research the topic."""
        # Generic angles that apply to most topics
        angles = [
            "current trends",
            "best practices", 
            "challenges and solutions",
            "case studies",
            "future outlook",
            "implementation strategies"
        ]
        
        # Add topic-specific angles based on keywords
        if "technology" in topic.lower():
            angles.extend(["technical specifications", "adoption rates", "competitive landscape"])
        if "business" in topic.lower():
            angles.extend(["market analysis", "ROI considerations", "industry impact"])
        
        return angles[:6]  # Limit to manageable number
    
    def _assess_source_quality(self, source: Dict[str, Any]) -> float:
        """Assess the quality of a research source."""
        quality_factors = []
        
        # Credibility score
        quality_factors.append(source.get("credibility_score", 0.5))
        
        # Relevance score  
        quality_factors.append(source.get("relevance_score", 0.5))
        
        # Source type quality
        source_type_scores = {
            "academic": 0.9,
            "industry": 0.8,
            "expert": 0.85,
            "news": 0.6,
            "blog": 0.4
        }
        source_type_score = source_type_scores.get(source.get("source_type", "unknown"), 0.5)
        quality_factors.append(source_type_score)
        
        # Recency (simulated)
        quality_factors.append(0.8)
        
        return sum(quality_factors) / len(quality_factors)