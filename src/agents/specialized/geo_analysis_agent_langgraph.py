"""
GEOAnalysisAgent LangGraph Implementation - Generative Engine Optimization for AI search engines.

Optimizes content for AI-powered search engines like ChatGPT, Gemini, Claude, and other LLMs.
Focuses on making content discoverable and citable by generative AI models.
"""

from typing import Dict, Any, Optional, List, TypedDict, Tuple
from enum import Enum
import re
import json
import asyncio
from dataclasses import dataclass
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.langgraph_base import (
    LangGraphWorkflowBase,
    WorkflowState,
    LangGraphExecutionContext,
    CheckpointStrategy,
    WorkflowStatus
)
from ..core.base_agent import AgentResult, AgentType, AgentMetadata
from ...core.security import SecurityValidator


class GEOPhase(str, Enum):
    """Phases of the Generative Engine Optimization workflow."""
    INITIALIZATION = "initialization"
    TRUSTWORTHINESS_ANALYSIS = "trustworthiness_analysis"
    PARSABILITY_ANALYSIS = "parsability_analysis"
    FACTUAL_DENSITY_ANALYSIS = "factual_density_analysis"
    AI_CITABILITY_ANALYSIS = "ai_citability_analysis"
    OPTIMIZATION_RECOMMENDATIONS = "optimization_recommendations"
    FINAL_SCORING = "final_scoring"


class TrustworthinessLevel(str, Enum):
    """E-E-A-T trustworthiness levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class EEATMetrics:
    """Experience, Expertise, Authoritativeness, Trust metrics."""
    experience_score: float
    expertise_score: float
    authoritativeness_score: float
    trust_score: float
    overall_score: float


class GEOState(TypedDict):
    """State for the Generative Engine Optimization workflow."""
    # Input data
    content: str
    blog_title: str
    author_info: Dict[str, str]
    publication_date: str
    target_keywords: List[str]
    content_type: str
    
    # Trustworthiness analysis (E-E-A-T)
    eeat_metrics: Dict[str, Any]
    author_credibility: Dict[str, Any]
    content_authority: Dict[str, Any]
    trust_signals: Dict[str, Any]
    
    # Machine parsability
    parsability_score: float
    structured_data: Dict[str, Any]
    clarity_metrics: Dict[str, Any]
    schema_markup: Dict[str, Any]
    
    # Factual density
    factual_density: Dict[str, Any]
    data_points: List[str]
    statistics: List[str]
    citations: List[str]
    unique_insights: List[str]
    
    # AI citability
    citability_score: float
    ai_model_feedback: Dict[str, Any]
    citation_potential: Dict[str, Any]
    answer_relevance: Dict[str, Any]
    
    # Optimization results
    geo_score: float
    optimization_priority: str
    improvement_areas: List[str]
    specific_recommendations: List[str]
    implementation_plan: List[str]
    
    # Performance predictions
    ai_visibility_prediction: Dict[str, float]
    citation_likelihood: float
    search_performance: Dict[str, Any]
    
    # Workflow metadata
    current_phase: str
    analysis_depth: str
    errors: List[str]
    warnings: List[str]


class GEOAnalysisAgentWorkflow(LangGraphWorkflowBase):
    """
    LangGraph-based GEOAnalysisAgent for Generative Engine Optimization.
    Optimizes content for AI-powered search engines and language models.
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        test_llm: Optional[ChatOpenAI] = None,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        analysis_depth: str = "comprehensive"
    ):
        """
        Initialize the GEOAnalysisAgent workflow.
        
        Args:
            llm: Primary language model for analysis
            test_llm: Secondary LLM for testing citability
            checkpoint_strategy: When to save checkpoints
            analysis_depth: shallow, standard, comprehensive
        """
        super().__init__(
            name="GEOAnalysisAgentWorkflow",
            checkpoint_strategy=checkpoint_strategy
        )
        
        self.llm = llm
        self.test_llm = test_llm or llm  # Use same LLM if no test LLM provided
        self.analysis_depth = analysis_depth
        self.security_validator = SecurityValidator()
        
        # GEO optimization configuration
        self.geo_config = {
            "eeat_weights": {
                "experience": 0.25,
                "expertise": 0.30,
                "authoritativeness": 0.25,
                "trust": 0.20
            },
            "scoring_weights": {
                "trustworthiness": 0.35,
                "parsability": 0.25,
                "factual_density": 0.25,
                "ai_citability": 0.15
            },
            "thresholds": {
                "excellent": 85,
                "good": 70,
                "fair": 55,
                "poor": 40
            }
        }
        
        # AI model characteristics for optimization
        self.ai_models = {
            "chatgpt": {
                "prefers": ["factual", "structured", "recent"],
                "citation_style": "conversational",
                "max_context": 8000
            },
            "gemini": {
                "prefers": ["comprehensive", "authoritative", "cited"],
                "citation_style": "academic",
                "max_context": 32000
            },
            "claude": {
                "prefers": ["nuanced", "balanced", "well_sourced"],
                "citation_style": "analytical",
                "max_context": 200000
            }
        }
        
        # Build the workflow graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(GEOState)
        
        # Add nodes for each phase
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("trustworthiness_analysis", self.trustworthiness_analysis_node)
        workflow.add_node("parsability_analysis", self.parsability_analysis_node)
        workflow.add_node("factual_density_analysis", self.factual_density_analysis_node)
        workflow.add_node("ai_citability_analysis", self.ai_citability_analysis_node)
        workflow.add_node("optimization_recommendations", self.optimization_recommendations_node)
        workflow.add_node("final_scoring", self.final_scoring_node)
        
        # Define edges
        workflow.set_entry_point("initialization")
        
        workflow.add_edge("initialization", "trustworthiness_analysis")
        workflow.add_edge("trustworthiness_analysis", "parsability_analysis")
        workflow.add_edge("parsability_analysis", "factual_density_analysis")
        workflow.add_edge("factual_density_analysis", "ai_citability_analysis")
        workflow.add_edge("ai_citability_analysis", "optimization_recommendations")
        workflow.add_edge("optimization_recommendations", "final_scoring")
        workflow.add_edge("final_scoring", END)
        
        # Compile with memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
    
    def initialization_node(self, state: GEOState) -> GEOState:
        """Initialize GEO analysis workflow."""
        try:
            state["current_phase"] = GEOPhase.INITIALIZATION
            
            # Security validation
            self.security_validator.validate_content(state["content"], "content")
            self.security_validator.validate_content(state["blog_title"], "title")
            
            # Initialize state fields
            state["content_type"] = state.get("content_type", "article")
            state["author_info"] = state.get("author_info", {})
            state["publication_date"] = state.get("publication_date", "")
            state["target_keywords"] = state.get("target_keywords", [])
            state["analysis_depth"] = self.analysis_depth
            state["errors"] = []
            state["warnings"] = []
            
            # Extract basic content metrics
            word_count = len(state["content"].split())
            char_count = len(state["content"])
            
            if word_count < 300:
                state["warnings"].append("Content may be too short for optimal AI visibility")
            if not state["target_keywords"]:
                state["warnings"].append("No target keywords provided - may limit optimization effectiveness")
            
            self.logger.info(
                f"GEO analysis initialized - {word_count} words, "
                f"targeting {len(state['target_keywords'])} keywords"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            state["errors"].append(f"Initialization error: {str(e)}")
        
        return state
    
    def trustworthiness_analysis_node(self, state: GEOState) -> GEOState:
        """Analyze content trustworthiness using E-E-A-T framework."""
        try:
            state["current_phase"] = GEOPhase.TRUSTWORTHINESS_ANALYSIS
            
            content = state["content"]
            author_info = state["author_info"]
            
            # Experience analysis
            experience_score = self._analyze_experience(content, author_info)
            
            # Expertise analysis
            expertise_score = self._analyze_expertise(content, author_info, state["target_keywords"])
            
            # Authoritativeness analysis
            authoritativeness_score = self._analyze_authoritativeness(content, author_info)
            
            # Trust analysis
            trust_score = self._analyze_trust(content, state["publication_date"])
            
            # Calculate overall E-E-A-T score
            weights = self.geo_config["eeat_weights"]
            overall_eeat = (
                experience_score * weights["experience"] +
                expertise_score * weights["expertise"] +
                authoritativeness_score * weights["authoritativeness"] +
                trust_score * weights["trust"]
            )
            
            state["eeat_metrics"] = {
                "experience": experience_score,
                "expertise": expertise_score,
                "authoritativeness": authoritativeness_score,
                "trust": trust_score,
                "overall": overall_eeat
            }
            
            # Detailed breakdowns
            state["author_credibility"] = self._assess_author_credibility(author_info)
            state["content_authority"] = self._assess_content_authority(content)
            state["trust_signals"] = self._identify_trust_signals(content, state["publication_date"])
            
            self.logger.info(f"Trustworthiness analysis completed - E-E-A-T score: {overall_eeat:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"Trustworthiness analysis failed: {str(e)}")
            state["errors"].append(f"Trustworthiness analysis error: {str(e)}")
            # Fallback
            state["eeat_metrics"] = {"overall": 50.0}
            state["author_credibility"] = {}
            state["content_authority"] = {}
            state["trust_signals"] = {}
        
        return state
    
    def parsability_analysis_node(self, state: GEOState) -> GEOState:
        """Analyze machine parsability for AI models."""
        try:
            state["current_phase"] = GEOPhase.PARSABILITY_ANALYSIS
            
            content = state["content"]
            title = state["blog_title"]
            
            # Structure analysis
            structure_score = self._analyze_content_structure(content)
            
            # Clarity analysis
            clarity_metrics = self._analyze_content_clarity(content)
            
            # Schema markup potential
            schema_analysis = self._analyze_schema_potential(content, title, state["content_type"])
            
            # Language complexity
            complexity_score = self._analyze_language_complexity(content)
            
            # Calculate overall parsability score
            state["parsability_score"] = (
                structure_score * 0.3 +
                clarity_metrics.get("clarity_score", 70) * 0.3 +
                schema_analysis.get("completeness_score", 50) * 0.2 +
                complexity_score * 0.2
            )
            
            state["structured_data"] = schema_analysis
            state["clarity_metrics"] = clarity_metrics
            state["schema_markup"] = self._generate_geo_schema(content, title, state["author_info"])
            
            self.logger.info(f"Parsability analysis completed - Score: {state['parsability_score']:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"Parsability analysis failed: {str(e)}")
            state["errors"].append(f"Parsability analysis error: {str(e)}")
            # Fallback
            state["parsability_score"] = 50.0
            state["structured_data"] = {}
            state["clarity_metrics"] = {}
            state["schema_markup"] = {}
        
        return state
    
    def factual_density_analysis_node(self, state: GEOState) -> GEOState:
        """Analyze factual density and unique insights."""
        try:
            state["current_phase"] = GEOPhase.FACTUAL_DENSITY_ANALYSIS
            
            content = state["content"]
            
            # Extract data points and statistics
            state["data_points"] = self._extract_data_points(content)
            state["statistics"] = self._extract_statistics(content)
            state["citations"] = self._extract_citations(content)
            state["unique_insights"] = self._identify_unique_insights(content, state["target_keywords"])
            
            # Calculate factual density metrics
            word_count = len(content.split())
            
            factual_density_ratio = (
                len(state["data_points"]) + 
                len(state["statistics"]) + 
                len(state["citations"])
            ) / word_count if word_count > 0 else 0
            
            # Scoring components
            density_score = min(factual_density_ratio * 1000, 60)  # Cap at 60
            uniqueness_score = min(len(state["unique_insights"]) * 5, 25)  # Cap at 25
            citation_score = min(len(state["citations"]) * 2, 15)  # Cap at 15
            
            total_factual_score = density_score + uniqueness_score + citation_score
            
            state["factual_density"] = {
                "ratio": factual_density_ratio,
                "density_score": density_score,
                "uniqueness_score": uniqueness_score,
                "citation_score": citation_score,
                "total_score": total_factual_score,
                "data_points_count": len(state["data_points"]),
                "statistics_count": len(state["statistics"]),
                "citations_count": len(state["citations"]),
                "unique_insights_count": len(state["unique_insights"])
            }
            
            self.logger.info(
                f"Factual density analysis completed - Score: {total_factual_score:.1f}/100, "
                f"Ratio: {factual_density_ratio:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Factual density analysis failed: {str(e)}")
            state["errors"].append(f"Factual density analysis error: {str(e)}")
            # Fallback
            state["factual_density"] = {"total_score": 30.0}
            state["data_points"] = []
            state["statistics"] = []
            state["citations"] = []
            state["unique_insights"] = []
        
        return state
    
    async def ai_citability_analysis_node(self, state: GEOState) -> GEOState:
        """Test content citability by AI models."""
        try:
            state["current_phase"] = GEOPhase.AI_CITABILITY_ANALYSIS
            
            if self.test_llm:
                # Test with multiple query types
                citability_tests = await self._test_ai_citability(
                    state["content"],
                    state["blog_title"],
                    state["target_keywords"]
                )
                
                state["ai_model_feedback"] = citability_tests
                state["citability_score"] = citability_tests.get("overall_score", 50.0)
            else:
                # Fallback analysis without AI testing
                citability_analysis = self._analyze_citability_potential(
                    state["content"],
                    state["eeat_metrics"],
                    state["factual_density"]
                )
                
                state["citability_score"] = citability_analysis["score"]
                state["ai_model_feedback"] = citability_analysis
            
            # Analyze answer relevance for common queries
            state["answer_relevance"] = self._analyze_answer_relevance(
                state["content"],
                state["target_keywords"]
            )
            
            # Citation potential assessment
            state["citation_potential"] = self._assess_citation_potential(
                state["content"],
                state["eeat_metrics"],
                state["factual_density"]
            )
            
            self.logger.info(f"AI citability analysis completed - Score: {state['citability_score']:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"AI citability analysis failed: {str(e)}")
            state["errors"].append(f"AI citability analysis error: {str(e)}")
            # Fallback
            state["citability_score"] = 40.0
            state["ai_model_feedback"] = {}
            state["answer_relevance"] = {}
            state["citation_potential"] = {}
        
        return state
    
    def optimization_recommendations_node(self, state: GEOState) -> GEOState:
        """Generate specific optimization recommendations."""
        try:
            state["current_phase"] = GEOPhase.OPTIMIZATION_RECOMMENDATIONS
            
            # Identify improvement areas
            state["improvement_areas"] = self._identify_improvement_areas(
                state["eeat_metrics"],
                state["parsability_score"],
                state["factual_density"],
                state["citability_score"]
            )
            
            # Generate specific recommendations
            state["specific_recommendations"] = self._generate_specific_recommendations(
                state["content"],
                state["eeat_metrics"],
                state["factual_density"],
                state["improvement_areas"]
            )
            
            # Create implementation plan
            state["implementation_plan"] = self._create_implementation_plan(
                state["specific_recommendations"],
                state["improvement_areas"]
            )
            
            self.logger.info(
                f"Optimization recommendations completed - {len(state['specific_recommendations'])} recommendations"
            )
            
        except Exception as e:
            self.logger.error(f"Optimization recommendations failed: {str(e)}")
            state["errors"].append(f"Optimization recommendations error: {str(e)}")
            # Fallback
            state["improvement_areas"] = []
            state["specific_recommendations"] = []
            state["implementation_plan"] = []
        
        return state
    
    def final_scoring_node(self, state: GEOState) -> GEOState:
        """Calculate final GEO score and predictions."""
        try:
            state["current_phase"] = GEOPhase.FINAL_SCORING
            
            # Calculate overall GEO score
            weights = self.geo_config["scoring_weights"]
            
            state["geo_score"] = (
                state["eeat_metrics"].get("overall", 50) * weights["trustworthiness"] +
                state["parsability_score"] * weights["parsability"] +
                state["factual_density"].get("total_score", 30) * weights["factual_density"] +
                state["citability_score"] * weights["ai_citability"]
            )
            
            # Determine optimization priority
            state["optimization_priority"] = self._determine_optimization_priority(
                state["geo_score"],
                state["improvement_areas"]
            )
            
            # Predict AI visibility and performance
            state["ai_visibility_prediction"] = self._predict_ai_visibility(
                state["geo_score"],
                state["target_keywords"],
                state["content_type"]
            )
            
            state["citation_likelihood"] = self._calculate_citation_likelihood(
                state["citability_score"],
                state["eeat_metrics"],
                state["factual_density"]
            )
            
            state["search_performance"] = self._predict_search_performance(
                state["geo_score"],
                state["target_keywords"]
            )
            
            self.logger.info(
                f"GEO analysis completed - Final score: {state['geo_score']:.1f}/100, "
                f"Priority: {state['optimization_priority']}"
            )
            
        except Exception as e:
            self.logger.error(f"Final scoring failed: {str(e)}")
            state["errors"].append(f"Final scoring error: {str(e)}")
            # Fallback
            state["geo_score"] = 50.0
            state["optimization_priority"] = "medium"
            state["ai_visibility_prediction"] = {}
            state["citation_likelihood"] = 0.3
            state["search_performance"] = {}
        
        return state
    
    # Helper methods for GEO analysis
    def _analyze_experience(self, content: str, author_info: Dict[str, str]) -> float:
        """Analyze experience indicators in content and author info."""
        score = 0.0
        
        # Author experience indicators
        if author_info.get("bio"):
            bio = author_info["bio"].lower()
            experience_keywords = ["years", "experience", "worked", "led", "managed", "founded"]
            score += sum(10 for keyword in experience_keywords if keyword in bio)
        
        # Content experience indicators
        content_lower = content.lower()
        experience_phrases = [
            "in my experience", "i have found", "from my work", "having worked",
            "as someone who", "in practice", "real-world", "hands-on"
        ]
        score += sum(5 for phrase in experience_phrases if phrase in content_lower)
        
        # Case studies and examples
        if "case study" in content_lower or "example" in content_lower:
            score += 15
        
        # Personal anecdotes
        personal_indicators = ["i", "my", "our company", "we implemented"]
        personal_count = sum(content_lower.count(indicator) for indicator in personal_indicators)
        score += min(personal_count * 2, 20)
        
        return min(score, 100.0)
    
    def _analyze_expertise(self, content: str, author_info: Dict[str, str], keywords: List[str]) -> float:
        """Analyze expertise indicators."""
        score = 0.0
        
        # Author credentials
        if author_info.get("credentials"):
            credentials = author_info["credentials"].lower()
            expertise_indicators = ["phd", "master", "certified", "expert", "specialist"]
            score += sum(15 for indicator in expertise_indicators if indicator in credentials)
        
        # Technical depth
        content_lower = content.lower()
        technical_terms = 0
        if keywords:
            for keyword in keywords:
                technical_terms += content_lower.count(keyword.lower())
        score += min(technical_terms * 3, 30)
        
        # Industry-specific language
        business_terms = ["roi", "kpi", "metrics", "analytics", "strategy", "implementation"]
        score += sum(3 for term in business_terms if term in content_lower)
        
        # Depth indicators
        depth_indicators = ["methodology", "framework", "approach", "analysis", "research"]
        score += sum(5 for indicator in depth_indicators if indicator in content_lower)
        
        return min(score, 100.0)
    
    def _analyze_authoritativeness(self, content: str, author_info: Dict[str, str]) -> float:
        """Analyze authoritativeness signals."""
        score = 0.0
        
        # Author authority signals
        if author_info.get("title"):
            title = author_info["title"].lower()
            authority_titles = ["ceo", "cto", "director", "head", "lead", "principal", "senior"]
            score += sum(15 for title_word in authority_titles if title_word in title)
        
        # Content authority indicators
        content_lower = content.lower()
        authority_phrases = [
            "according to research", "studies show", "data indicates",
            "industry report", "survey results", "peer-reviewed"
        ]
        score += sum(10 for phrase in authority_phrases if phrase in content_lower)
        
        # Citations and references
        citation_patterns = [
            r'\[[\d,\s-]+\]',  # [1], [1-3], [1,2,3]
            r'\(\d{4}\)',      # (2024)
            r'et al\.',        # et al.
        ]
        for pattern in citation_patterns:
            matches = len(re.findall(pattern, content))
            score += min(matches * 5, 20)
        
        return min(score, 100.0)
    
    def _analyze_trust(self, content: str, publication_date: str) -> float:
        """Analyze trust signals."""
        score = 0.0
        
        # Recency bonus
        if publication_date:
            try:
                from datetime import datetime, timedelta
                pub_date = datetime.fromisoformat(publication_date.replace('Z', '+00:00'))
                days_old = (datetime.now() - pub_date).days
                if days_old <= 30:
                    score += 20
                elif days_old <= 90:
                    score += 15
                elif days_old <= 365:
                    score += 10
            except:
                pass  # Invalid date format
        
        # Trust indicators
        content_lower = content.lower()
        trust_phrases = [
            "peer-reviewed", "verified", "fact-checked", "confirmed",
            "validated", "tested", "proven", "evidence-based"
        ]
        score += sum(8 for phrase in trust_phrases if phrase in content_lower)
        
        # Transparency indicators
        transparency_phrases = [
            "methodology", "limitations", "disclosure", "conflict of interest",
            "data source", "sample size"
        ]
        score += sum(5 for phrase in transparency_phrases if phrase in content_lower)
        
        # External validation
        if "award" in content_lower or "recognition" in content_lower:
            score += 10
        
        return min(score, 100.0)
    
    def _assess_author_credibility(self, author_info: Dict[str, str]) -> Dict[str, Any]:
        """Assess author credibility in detail."""
        return {
            "has_bio": bool(author_info.get("bio")),
            "has_credentials": bool(author_info.get("credentials")),
            "has_title": bool(author_info.get("title")),
            "bio_length": len(author_info.get("bio", "")),
            "credibility_score": sum([
                20 if author_info.get("bio") else 0,
                15 if author_info.get("credentials") else 0,
                10 if author_info.get("title") else 0
            ])
        }
    
    def _assess_content_authority(self, content: str) -> Dict[str, Any]:
        """Assess content authority signals."""
        citations = len(re.findall(r'\[[\d,\s-]+\]|\(\d{4}\)', content))
        references = content.lower().count("source:")
        studies = content.lower().count("study")
        research = content.lower().count("research")
        
        return {
            "citation_count": citations,
            "reference_count": references,
            "study_mentions": studies,
            "research_mentions": research,
            "authority_score": min(citations * 5 + references * 3 + studies * 2 + research * 2, 100)
        }
    
    def _identify_trust_signals(self, content: str, publication_date: str) -> Dict[str, Any]:
        """Identify trust signals in content."""
        content_lower = content.lower()
        
        return {
            "has_recent_date": bool(publication_date),
            "has_fact_checking": "fact-checked" in content_lower,
            "has_peer_review": "peer-reviewed" in content_lower,
            "has_transparency": any(phrase in content_lower for phrase in [
                "methodology", "limitations", "disclosure"
            ]),
            "trust_score": sum([
                10 if publication_date else 0,
                15 if "fact-checked" in content_lower else 0,
                20 if "peer-reviewed" in content_lower else 0,
                10 if "methodology" in content_lower else 0
            ])
        }
    
    def _analyze_content_structure(self, content: str) -> float:
        """Analyze content structure for AI parsability."""
        score = 0.0
        
        # Heading structure
        headings = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
        score += min(len(headings) * 10, 30)
        
        # Lists and bullet points
        lists = re.findall(r'^[-*•]\s+.+$', content, re.MULTILINE)
        numbered_lists = re.findall(r'^\d+\.\s+.+$', content, re.MULTILINE)
        score += min((len(lists) + len(numbered_lists)) * 5, 25)
        
        # Paragraph structure
        paragraphs = content.split('\n\n')
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        if 50 <= avg_paragraph_length <= 150:
            score += 20
        elif 30 <= avg_paragraph_length <= 200:
            score += 15
        
        # Logical flow indicators
        flow_indicators = ["first", "second", "next", "then", "finally", "in conclusion"]
        score += sum(3 for indicator in flow_indicators if indicator in content.lower())
        
        return min(score, 100.0)
    
    def _analyze_content_clarity(self, content: str) -> Dict[str, Any]:
        """Analyze content clarity metrics."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Clarity scoring
        clarity_score = 100.0
        
        # Sentence length penalty
        if avg_sentence_length > 25:
            clarity_score -= (avg_sentence_length - 25) * 2
        
        # Complex word penalty
        words = content.split()
        complex_words = [w for w in words if len(w) > 12]
        complexity_ratio = len(complex_words) / len(words) if words else 0
        clarity_score -= complexity_ratio * 100
        
        return {
            "clarity_score": max(clarity_score, 0),
            "avg_sentence_length": avg_sentence_length,
            "complexity_ratio": complexity_ratio,
            "total_sentences": len(sentences),
            "readability_grade": "good" if clarity_score > 70 else "fair" if clarity_score > 50 else "poor"
        }
    
    def _analyze_schema_potential(self, content: str, title: str, content_type: str) -> Dict[str, Any]:
        """Analyze potential for structured data markup."""
        schema_elements = 0
        
        # Article schema elements
        if title:
            schema_elements += 1
        if content:
            schema_elements += 1
        
        # FAQ schema potential
        questions = len(re.findall(r'\?', content))
        if questions >= 3:
            schema_elements += 1
        
        # How-to schema potential
        if "step" in content.lower() and any(indicator in content.lower() for indicator in ["1.", "2.", "first", "next"]):
            schema_elements += 1
        
        # Rating/review schema potential
        if any(word in content.lower() for word in ["rating", "review", "stars", "score"]):
            schema_elements += 1
        
        completeness_score = min(schema_elements * 20, 100)
        
        return {
            "potential_schemas": ["Article", "WebPage"],
            "faq_potential": questions >= 3,
            "howto_potential": "step" in content.lower(),
            "review_potential": "review" in content.lower(),
            "completeness_score": completeness_score,
            "schema_elements_count": schema_elements
        }
    
    def _analyze_language_complexity(self, content: str) -> float:
        """Analyze language complexity for AI understanding."""
        score = 100.0
        
        words = content.split()
        
        # Vocabulary complexity
        complex_words = [w for w in words if len(w) > 15]
        complexity_ratio = len(complex_words) / len(words) if words else 0
        score -= complexity_ratio * 200
        
        # Jargon detection
        jargon_indicators = content.count("i.e.") + content.count("e.g.") + content.count("etc.")
        score -= jargon_indicators * 5
        
        # Passive voice (simplified detection)
        passive_indicators = ["was", "were", "been", "being"]
        passive_count = sum(content.lower().count(indicator) for indicator in passive_indicators)
        passive_ratio = passive_count / len(words) if words else 0
        score -= passive_ratio * 300
        
        return max(score, 0.0)
    
    def _extract_data_points(self, content: str) -> List[str]:
        """Extract specific data points from content."""
        data_points = []
        
        # Numbers with units
        number_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+(?:,\d+)*(?:\.\d+)?[KMB]?',  # Money
            r'\d+(?:,\d+)*(?:\.\d+)?\s+(?:users|customers|companies|employees)',  # Counts
            r'\d+(?:\.\d+)?\s*(?:hours|days|weeks|months|years)',  # Time
            r'\d+(?:\.\d+)?x\s+(?:increase|improvement|growth)',  # Multipliers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            data_points.extend(matches)
        
        return data_points[:20]  # Limit to prevent spam
    
    def _extract_statistics(self, content: str) -> List[str]:
        """Extract statistical claims from content."""
        statistics = []
        
        # Statistical phrases
        stat_patterns = [
            r'according to [^,.]+ \d+%[^.]*',
            r'\d+% of [^.]*',
            r'studies? show[^.]*\d+[^.]*',
            r'research indicates[^.]*\d+[^.]*',
            r'survey found[^.]*\d+[^.]*'
        ]
        
        for pattern in stat_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            statistics.extend(matches)
        
        return statistics[:15]  # Limit to prevent spam
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation references."""
        citations = []
        
        # Citation patterns
        citation_patterns = [
            r'\[[\d,\s-]+\]',  # [1], [1-3], [1,2,3]
            r'\(\w+\s+et\s+al\.,?\s+\d{4}\)',  # (Smith et al., 2024)
            r'\(\w+,?\s+\d{4}\)',  # (Smith, 2024)
            r'Source:\s+[^\n]+',  # Source: ...
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        return citations[:10]  # Limit to prevent spam
    
    def _identify_unique_insights(self, content: str, keywords: List[str]) -> List[str]:
        """Identify unique insights in content."""
        insights = []
        
        # Look for insight indicators
        insight_phrases = [
            r'surprisingly[^.]*',
            r'interestingly[^.]*',
            r'contrary to [^,]*,[^.]*',
            r'unlike [^,]*,[^.]*',
            r'however[^.]*',
            r'what[^?]*\?[^.]*'
        ]
        
        for phrase in insight_phrases:
            matches = re.findall(phrase, content, re.IGNORECASE)
            insights.extend(matches)
        
        # Look for contrarian statements
        contrarian_indicators = ["but", "however", "actually", "in fact", "contrary"]
        for indicator in contrarian_indicators:
            pattern = f'{indicator}[^.]*'
            matches = re.findall(pattern, content, re.IGNORECASE)
            insights.extend(matches[:2])  # Limit per indicator
        
        return insights[:10]  # Limit total insights
    
    def _generate_geo_schema(self, content: str, title: str, author_info: Dict[str, str]) -> Dict[str, Any]:
        """Generate GEO-optimized schema markup."""
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": content[:200] if content else "",
            "wordCount": len(content.split()),
            "datePublished": "2024-01-01",  # Would be actual date
            "dateModified": "2024-01-01",   # Would be actual date
            "author": {
                "@type": "Person",
                "name": author_info.get("name", "Author"),
                "jobTitle": author_info.get("title", ""),
                "description": author_info.get("bio", "")
            },
            "publisher": {
                "@type": "Organization",
                "name": "CrediLinq"
            },
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": "https://example.com"  # Would be actual URL
            }
        }
        
        # Add FAQ schema if questions detected
        questions = re.findall(r'^#{2,3}\s*([^#\n]*\?[^#\n]*)', content, re.MULTILINE)
        if questions:
            faq_items = []
            for question in questions[:5]:  # Limit to 5 FAQ items
                faq_items.append({
                    "@type": "Question",
                    "name": question.strip(),
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": "Answer content would be extracted from following content"
                    }
                })
            
            schema["mainEntity"] = {
                "@type": "FAQPage",
                "mainEntity": faq_items
            }
        
        return schema
    
    async def _test_ai_citability(self, content: str, title: str, keywords: List[str]) -> Dict[str, Any]:
        """Test how well AI models can cite and use the content."""
        if not self.test_llm:
            return {"overall_score": 50.0, "tests": []}
        
        test_results = []
        
        # Test different query types
        test_queries = [
            f"What is {keywords[0] if keywords else 'the main topic'} and how does it work?",
            f"Can you explain the benefits of {title}?",
            f"What are the key insights about {keywords[0] if keywords else 'this topic'}?",
            "What does the research say about this topic?",
            "What are the practical applications mentioned?"
        ]
        
        for query in test_queries[:3]:  # Limit to 3 tests
            try:
                prompt = f"""Based on this content, answer the question: {query}

Content:
{content[:3000]}

Provide a comprehensive answer and indicate if you would cite this source."""
                
                response = await self.test_llm.ainvoke([SystemMessage(content=prompt)])
                
                # Analyze response quality
                response_text = response.content
                citation_likelihood = self._assess_citation_likelihood(response_text, content)
                
                test_results.append({
                    "query": query,
                    "response_length": len(response_text),
                    "citation_likelihood": citation_likelihood,
                    "mentions_source": "source" in response_text.lower() or "according to" in response_text.lower()
                })
                
                # Small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"AI citability test failed for query '{query}': {str(e)}")
                continue
        
        # Calculate overall score
        if test_results:
            avg_citation_likelihood = sum(r["citation_likelihood"] for r in test_results) / len(test_results)
            overall_score = avg_citation_likelihood * 100
        else:
            overall_score = 40.0  # Fallback if all tests fail
        
        return {
            "overall_score": overall_score,
            "tests": test_results,
            "test_count": len(test_results)
        }
    
    def _assess_citation_likelihood(self, ai_response: str, original_content: str) -> float:
        """Assess likelihood that AI would cite this content."""
        score = 0.0
        
        # Response quality indicators
        if len(ai_response) > 200:
            score += 0.2
        
        # Specific information usage
        content_words = set(original_content.lower().split())
        response_words = set(ai_response.lower().split())
        overlap = len(content_words.intersection(response_words))
        overlap_ratio = overlap / len(content_words) if content_words else 0
        score += overlap_ratio * 0.5
        
        # Citation indicators
        if any(phrase in ai_response.lower() for phrase in ["according to", "the source states", "as mentioned"]):
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_citability_potential(self, content: str, eeat_metrics: Dict[str, Any], factual_density: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze citability potential without AI testing."""
        score = 0.0
        
        # Base score from E-E-A-T
        score += eeat_metrics.get("overall", 50) * 0.4
        
        # Factual density contribution
        score += factual_density.get("total_score", 30) * 0.4
        
        # Content characteristics
        content_lower = content.lower()
        
        # Authoritative phrases
        authoritative_phrases = ["research shows", "studies indicate", "according to", "data reveals"]
        auth_count = sum(1 for phrase in authoritative_phrases if phrase in content_lower)
        score += min(auth_count * 5, 20)
        
        return {
            "score": min(score, 100),
            "authority_phrases": auth_count,
            "analysis": "Citability assessment based on content analysis"
        }
    
    def _analyze_answer_relevance(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze how well content answers potential queries."""
        relevance_score = 0.0
        
        # Question answering indicators
        content_lower = content.lower()
        qa_indicators = ["what", "how", "why", "when", "where", "who"]
        qa_count = sum(content_lower.count(indicator) for indicator in qa_indicators)
        relevance_score += min(qa_count * 3, 30)
        
        # Comprehensive coverage
        if keywords:
            keyword_coverage = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            coverage_ratio = keyword_coverage / len(keywords)
            relevance_score += coverage_ratio * 40
        
        # Answer structure
        if "conclusion" in content_lower or "summary" in content_lower:
            relevance_score += 15
        
        # Step-by-step content
        if any(indicator in content_lower for indicator in ["step 1", "first", "1.", "next"]):
            relevance_score += 15
        
        return {
            "relevance_score": min(relevance_score, 100),
            "qa_indicators": qa_count,
            "keyword_coverage": keyword_coverage / len(keywords) if keywords else 0,
            "has_conclusion": "conclusion" in content_lower
        }
    
    def _assess_citation_potential(self, content: str, eeat_metrics: Dict[str, Any], factual_density: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall citation potential."""
        base_score = (
            eeat_metrics.get("overall", 50) * 0.4 +
            factual_density.get("total_score", 30) * 0.6
        )
        
        # Citation-worthy indicators
        content_lower = content.lower()
        citation_indicators = [
            "study", "research", "survey", "analysis", "report",
            "data", "statistics", "findings", "results"
        ]
        
        indicator_count = sum(content_lower.count(indicator) for indicator in citation_indicators)
        citation_bonus = min(indicator_count * 2, 20)
        
        total_score = min(base_score + citation_bonus, 100)
        
        return {
            "citation_score": total_score,
            "citation_indicators": indicator_count,
            "likelihood": "high" if total_score > 80 else "medium" if total_score > 60 else "low"
        }
    
    def _identify_improvement_areas(self, eeat: Dict[str, Any], parsability: float, factual: Dict[str, Any], citability: float) -> List[str]:
        """Identify specific areas for improvement."""
        areas = []
        
        if eeat.get("overall", 50) < 70:
            areas.append("trustworthiness")
        if parsability < 70:
            areas.append("parsability")
        if factual.get("total_score", 30) < 60:
            areas.append("factual_density")
        if citability < 60:
            areas.append("ai_citability")
        
        return areas
    
    def _generate_specific_recommendations(self, content: str, eeat: Dict[str, Any], factual: Dict[str, Any], areas: List[str]) -> List[str]:
        """Generate specific, actionable recommendations."""
        recommendations = []
        
        if "trustworthiness" in areas:
            if eeat.get("experience", 0) < 50:
                recommendations.append("Add personal experience examples and case studies")
            if eeat.get("expertise", 0) < 50:
                recommendations.append("Include more technical details and industry-specific terminology")
            if eeat.get("authoritativeness", 0) < 50:
                recommendations.append("Add citations from authoritative sources")
            if eeat.get("trust", 0) < 50:
                recommendations.append("Include publication date and author bio")
        
        if "factual_density" in areas:
            if factual.get("data_points_count", 0) < 5:
                recommendations.append("Include more specific data points and statistics")
            if factual.get("citations_count", 0) < 3:
                recommendations.append("Add references to credible sources")
            if factual.get("unique_insights_count", 0) < 3:
                recommendations.append("Provide unique insights and contrarian viewpoints")
        
        if "parsability" in areas:
            recommendations.append("Improve content structure with clear headings")
            recommendations.append("Add bullet points and numbered lists")
            recommendations.append("Simplify complex sentences")
        
        if "ai_citability" in areas:
            recommendations.append("Include direct answers to common questions")
            recommendations.append("Add FAQ section")
            recommendations.append("Use more authoritative language")
        
        return recommendations
    
    def _create_implementation_plan(self, recommendations: List[str], areas: List[str]) -> List[str]:
        """Create prioritized implementation plan."""
        plan = []
        
        # High priority items
        if "trustworthiness" in areas:
            plan.append("PRIORITY 1: Enhance author credibility and add citations")
        
        if "factual_density" in areas:
            plan.append("PRIORITY 2: Add more data points and statistics")
        
        # Medium priority
        if "parsability" in areas:
            plan.append("PRIORITY 3: Improve content structure and clarity")
        
        if "ai_citability" in areas:
            plan.append("PRIORITY 4: Optimize for AI model understanding")
        
        # Add specific action items
        plan.extend([f"Action: {rec}" for rec in recommendations[:8]])
        
        return plan
    
    def _determine_optimization_priority(self, score: float, areas: List[str]) -> str:
        """Determine optimization priority level."""
        if score < 40 or len(areas) >= 4:
            return "critical"
        elif score < 60 or len(areas) >= 3:
            return "high"
        elif score < 75 or len(areas) >= 2:
            return "medium"
        else:
            return "low"
    
    def _predict_ai_visibility(self, score: float, keywords: List[str], content_type: str) -> Dict[str, float]:
        """Predict AI model visibility and citation likelihood."""
        base_visibility = score / 100.0
        
        # Content type modifiers
        type_modifiers = {
            "article": 1.0,
            "blog": 0.9,
            "guide": 1.1,
            "research": 1.2,
            "opinion": 0.8
        }
        
        modifier = type_modifiers.get(content_type, 1.0)
        adjusted_visibility = base_visibility * modifier
        
        return {
            "chatgpt": adjusted_visibility * 0.9,  # Slightly lower for conversational AI
            "gemini": adjusted_visibility * 1.1,   # Higher for comprehensive AI
            "claude": adjusted_visibility * 1.0,   # Baseline
            "overall": adjusted_visibility
        }
    
    def _calculate_citation_likelihood(self, citability_score: float, eeat: Dict[str, Any], factual: Dict[str, Any]) -> float:
        """Calculate likelihood of being cited by AI models."""
        base_likelihood = citability_score / 100.0
        
        # Authority bonus
        authority_bonus = eeat.get("authoritativeness", 50) / 100.0 * 0.2
        
        # Factual density bonus
        factual_bonus = factual.get("total_score", 30) / 100.0 * 0.3
        
        total_likelihood = min(base_likelihood + authority_bonus + factual_bonus, 1.0)
        
        return total_likelihood
    
    def _predict_search_performance(self, geo_score: float, keywords: List[str]) -> Dict[str, Any]:
        """Predict search performance across AI platforms."""
        performance_base = geo_score / 100.0
        
        return {
            "ai_search_visibility": performance_base,
            "citation_probability": performance_base * 0.7,
            "answer_relevance": performance_base * 0.9,
            "recommendation": "excellent" if performance_base > 0.8 else 
                           "good" if performance_base > 0.65 else 
                           "fair" if performance_base > 0.5 else "needs_improvement"
        }
    
    async def execute_workflow(
        self,
        initial_state: Dict[str, Any],
        context: Optional[LangGraphExecutionContext] = None
    ) -> WorkflowState:
        """Execute the GEO analysis workflow."""
        try:
            # Convert input to GEOState
            geo_state = GEOState(
                content=initial_state["content"],
                blog_title=initial_state["blog_title"],
                author_info=initial_state.get("author_info", {}),
                publication_date=initial_state.get("publication_date", ""),
                target_keywords=initial_state.get("target_keywords", []),
                content_type=initial_state.get("content_type", "article"),
                eeat_metrics={},
                author_credibility={},
                content_authority={},
                trust_signals={},
                parsability_score=0.0,
                structured_data={},
                clarity_metrics={},
                schema_markup={},
                factual_density={},
                data_points=[],
                statistics=[],
                citations=[],
                unique_insights=[],
                citability_score=0.0,
                ai_model_feedback={},
                citation_potential={},
                answer_relevance={},
                geo_score=0.0,
                optimization_priority="medium",
                improvement_areas=[],
                specific_recommendations=[],
                implementation_plan=[],
                ai_visibility_prediction={},
                citation_likelihood=0.0,
                search_performance={},
                current_phase=GEOPhase.INITIALIZATION,
                analysis_depth=self.analysis_depth,
                errors=[],
                warnings=[]
            )
            
            # Execute the graph
            config = {"configurable": {"thread_id": context.session_id if context else "default"}}
            final_state = await self.graph.ainvoke(geo_state, config)
            
            # Convert to WorkflowState
            return WorkflowState(
                status=WorkflowStatus.COMPLETED if final_state["geo_score"] > 0 else WorkflowStatus.FAILED,
                phase=final_state["current_phase"],
                data={
                    "geo_score": final_state["geo_score"],
                    "optimization_priority": final_state["optimization_priority"],
                    "eeat_metrics": final_state["eeat_metrics"],
                    "parsability_score": final_state["parsability_score"],
                    "factual_density": final_state["factual_density"],
                    "citability_score": final_state["citability_score"],
                    "improvement_areas": final_state["improvement_areas"],
                    "specific_recommendations": final_state["specific_recommendations"],
                    "implementation_plan": final_state["implementation_plan"],
                    "ai_visibility_prediction": final_state["ai_visibility_prediction"],
                    "citation_likelihood": final_state["citation_likelihood"],
                    "schema_markup": final_state["schema_markup"]
                },
                errors=final_state.get("errors", []),
                metadata={
                    "warnings": final_state.get("warnings", []),
                    "analysis_depth": final_state.get("analysis_depth", ""),
                    "data_points_found": len(final_state.get("data_points", [])),
                    "citations_found": len(final_state.get("citations", []))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return WorkflowState(
                status=WorkflowStatus.FAILED,
                phase=GEOPhase.INITIALIZATION,
                data={},
                errors=[str(e)],
                metadata={"error_type": type(e).__name__}
            )


# Adapter for backward compatibility
class GEOAnalysisAgentLangGraph:
    """Adapter to make LangGraph workflow compatible with existing GEOAnalysisAgent interface."""
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize the adapter."""
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_OPTIMIZER,
                name="GEOAnalysisAgentLangGraph",
                description="LangGraph-powered Generative Engine Optimization analyzer for AI search engines",
                capabilities=[
                    "eeat_analysis",
                    "ai_citability_testing", 
                    "factual_density_analysis",
                    "parsability_optimization",
                    "ai_visibility_prediction",
                    "schema_optimization"
                ],
                version="3.0.0"
            )
        
        self.metadata = metadata
        self.workflow = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the workflow."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Use same LLM for testing (in production might use different models)
            self.workflow = GEOAnalysisAgentWorkflow(llm=llm, test_llm=llm)
            
        except Exception as e:
            # Fallback without LLM
            self.workflow = GEOAnalysisAgentWorkflow(llm=None, test_llm=None)
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Any] = None
    ) -> AgentResult:
        """Execute the GEO analysis workflow."""
        try:
            # Execute workflow
            result = await self.workflow.execute_workflow(
                input_data,
                LangGraphExecutionContext(
                    session_id=context.session_id if context else "default",
                    user_id=context.user_id if context else None
                )
            )
            
            # Convert to AgentResult
            return AgentResult(
                success=result.status == WorkflowStatus.COMPLETED,
                data=result.data,
                metadata={
                    "agent_type": "geo_analysis_langgraph",
                    "workflow_status": result.status,
                    "final_phase": result.phase,
                    **result.metadata
                },
                error_message="; ".join(result.errors) if result.errors else None
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="GEO_ANALYSIS_WORKFLOW_FAILED"
            )