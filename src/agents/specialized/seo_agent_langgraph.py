"""
SEOAgent LangGraph Implementation - Advanced SEO optimization with intelligent workflows.
"""

from typing import Dict, Any, Optional, List, TypedDict, Tuple
from enum import Enum
import re
import json
import asyncio
from collections import Counter
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    import textstat
except ImportError:
    textstat = None

from ..core.langgraph_base import (
    LangGraphWorkflowBase,
    WorkflowState,
    LangGraphExecutionContext,
    CheckpointStrategy,
    WorkflowStatus
)
from ..core.base_agent import AgentResult, AgentType, AgentMetadata
from ...core.security import SecurityValidator


class SEOPhase(str, Enum):
    """Phases of the SEO optimization workflow."""
    INITIALIZATION = "initialization"
    TECHNICAL_ANALYSIS = "technical_analysis"
    KEYWORD_RESEARCH = "keyword_research"
    CONTENT_OPTIMIZATION = "content_optimization"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    SCHEMA_GENERATION = "schema_generation"
    FINAL_SCORING = "final_scoring"


class SEOStrategy(str, Enum):
    """SEO optimization strategies."""
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"


class SEOState(TypedDict):
    """State for the SEO workflow."""
    # Input data
    content: str
    blog_title: str
    outline: List[str]
    target_keywords: List[str]
    content_type: str
    seo_strategy: str
    
    # Technical analysis
    technical_metrics: Dict[str, Any]
    readability_analysis: Dict[str, Any]
    heading_structure: Dict[str, Any]
    url_analysis: Dict[str, Any]
    
    # Keyword analysis
    keyword_research: Dict[str, Any]
    semantic_keywords: List[str]
    keyword_density: Dict[str, float]
    search_intent: str
    
    # Content optimization
    optimization_suggestions: List[str]
    content_gaps: List[str]
    title_variations: List[str]
    meta_suggestions: Dict[str, str]
    
    # Competitive insights
    competitive_analysis: Dict[str, Any]
    ranking_opportunities: List[str]
    
    # Schema and structured data
    schema_markup: Dict[str, Any]
    structured_data: Dict[str, Any]
    
    # Final results
    seo_score: float
    optimization_priority: str
    implementation_plan: List[str]
    
    # Workflow metadata
    current_phase: str
    analysis_depth: str
    errors: List[str]
    warnings: List[str]


class SEOAgentWorkflow(LangGraphWorkflowBase):
    """
    LangGraph-based SEOAgent with comprehensive optimization workflows.
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        analysis_depth: str = "comprehensive"
    ):
        """
        Initialize the SEOAgent workflow.
        
        Args:
            llm: Language model for SEO analysis
            checkpoint_strategy: When to save checkpoints
            analysis_depth: shallow, standard, comprehensive
        """
        super().__init__(
            name="SEOAgentWorkflow",
            checkpoint_strategy=checkpoint_strategy
        )
        
        self.llm = llm
        self.analysis_depth = analysis_depth
        self.security_validator = SecurityValidator()
        
        # SEO configuration
        self.seo_config = {
            "target_word_count": {"blog": (1500, 3000), "article": (2000, 4000)},
            "keyword_density_range": (1.0, 3.0),  # percentage
            "title_length_range": (30, 60),  # characters
            "meta_description_range": (140, 160),
            "heading_ratios": {"h1": 1, "h2": (3, 6), "h3": (0, 12)}
        }
        
        # Build the workflow graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(SEOState)
        
        # Add nodes for each phase
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("technical_analysis", self.technical_analysis_node)
        workflow.add_node("keyword_research", self.keyword_research_node)
        workflow.add_node("content_optimization", self.content_optimization_node)
        workflow.add_node("competitive_analysis", self.competitive_analysis_node)
        workflow.add_node("schema_generation", self.schema_generation_node)
        workflow.add_node("final_scoring", self.final_scoring_node)
        
        # Define edges
        workflow.set_entry_point("initialization")
        
        workflow.add_edge("initialization", "technical_analysis")
        workflow.add_edge("technical_analysis", "keyword_research")
        workflow.add_edge("keyword_research", "content_optimization")
        workflow.add_edge("content_optimization", "competitive_analysis")
        workflow.add_edge("competitive_analysis", "schema_generation")
        workflow.add_edge("schema_generation", "final_scoring")
        workflow.add_edge("final_scoring", END)
        
        # Compile with memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
    
    def initialization_node(self, state: SEOState) -> SEOState:
        """Initialize SEO workflow and determine strategy."""
        try:
            state["current_phase"] = SEOPhase.INITIALIZATION
            
            # Security validation
            self.security_validator.validate_content(state["content"], "content")
            self.security_validator.validate_content(state["blog_title"], "title")
            
            # Initialize state fields
            state["content_type"] = state.get("content_type", "blog")
            state["analysis_depth"] = self.analysis_depth
            state["target_keywords"] = state.get("target_keywords", [])
            state["outline"] = state.get("outline", [])
            state["errors"] = []
            state["warnings"] = []
            
            # Determine SEO strategy based on content analysis
            state["seo_strategy"] = self._determine_seo_strategy(
                state["content"],
                state["blog_title"],
                state["target_keywords"]
            )
            
            # Detect search intent
            state["search_intent"] = self._detect_search_intent(
                state["blog_title"],
                state["content"]
            )
            
            self.logger.info(
                f"SEO analysis initialized - Strategy: {state['seo_strategy']}, "
                f"Intent: {state['search_intent']}"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            state["errors"].append(f"Initialization error: {str(e)}")
        
        return state
    
    def technical_analysis_node(self, state: SEOState) -> SEOState:
        """Perform technical SEO analysis."""
        try:
            state["current_phase"] = SEOPhase.TECHNICAL_ANALYSIS
            
            # Content metrics
            content = state["content"]
            title = state["blog_title"]
            
            state["technical_metrics"] = {
                "word_count": len(content.split()),
                "character_count": len(content),
                "sentence_count": len([s for s in content.split('.') if s.strip()]),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
                "title_length": len(title),
                "title_word_count": len(title.split())
            }
            
            # Readability analysis
            state["readability_analysis"] = self._analyze_readability(content)
            
            # Heading structure analysis
            state["heading_structure"] = self._analyze_heading_structure(content)
            
            # URL structure analysis
            state["url_analysis"] = self._analyze_url_structure(title)
            
            self.logger.info(f"Technical analysis completed - {state['technical_metrics']['word_count']} words")
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {str(e)}")
            state["errors"].append(f"Technical analysis error: {str(e)}")
            # Fallback metrics
            state["technical_metrics"] = {"word_count": len(state["content"].split())}
            state["readability_analysis"] = {"score": 50}
            state["heading_structure"] = {"h1_count": 0, "h2_count": 0}
        
        return state
    
    def keyword_research_node(self, state: SEOState) -> SEOState:
        """Perform comprehensive keyword research and analysis."""
        try:
            state["current_phase"] = SEOPhase.KEYWORD_RESEARCH
            
            content = state["content"]
            title = state["blog_title"]
            target_keywords = state["target_keywords"]
            
            # Extract and analyze keywords
            extracted_keywords = self._extract_keywords(content, title)
            
            # Keyword density analysis
            state["keyword_density"] = self._calculate_keyword_density(
                content, target_keywords + extracted_keywords
            )
            
            # Semantic keyword analysis
            if self.llm:
                semantic_keywords = self._generate_semantic_keywords(
                    content, title, target_keywords
                )
                state["semantic_keywords"] = semantic_keywords
            else:
                state["semantic_keywords"] = extracted_keywords[:20]
            
            # Keyword research data
            state["keyword_research"] = {
                "primary_keywords": target_keywords[:3] if target_keywords else extracted_keywords[:3],
                "secondary_keywords": extracted_keywords[:10],
                "long_tail_keywords": self._extract_long_tail_keywords(content),
                "keyword_variations": self._generate_keyword_variations(target_keywords),
                "keyword_gaps": self._identify_keyword_gaps(content, state["semantic_keywords"])
            }
            
            self.logger.info(
                f"Keyword research completed - {len(state['semantic_keywords'])} semantic keywords identified"
            )
            
        except Exception as e:
            self.logger.error(f"Keyword research failed: {str(e)}")
            state["errors"].append(f"Keyword research error: {str(e)}")
            # Fallback
            state["keyword_density"] = {}
            state["semantic_keywords"] = []
            state["keyword_research"] = {"primary_keywords": []}
        
        return state
    
    def content_optimization_node(self, state: SEOState) -> SEOState:
        """Generate content optimization recommendations."""
        try:
            state["current_phase"] = SEOPhase.CONTENT_OPTIMIZATION
            
            # Analyze content gaps
            state["content_gaps"] = self._identify_content_gaps(
                state["content"],
                state["semantic_keywords"],
                state["search_intent"]
            )
            
            # Generate optimization suggestions
            state["optimization_suggestions"] = self._generate_optimization_suggestions(
                state["technical_metrics"],
                state["heading_structure"],
                state["keyword_density"],
                state["content_gaps"]
            )
            
            # Title optimization
            state["title_variations"] = self._generate_title_variations(
                state["blog_title"],
                state["keyword_research"]["primary_keywords"]
            )
            
            # Meta data suggestions
            state["meta_suggestions"] = {
                "title": self._optimize_meta_title(
                    state["blog_title"],
                    state["keyword_research"]["primary_keywords"]
                ),
                "description": self._optimize_meta_description(
                    state["content"],
                    state["keyword_research"]["primary_keywords"]
                ),
                "keywords": ", ".join(state["semantic_keywords"][:10])
            }
            
            self.logger.info(
                f"Content optimization completed - {len(state['optimization_suggestions'])} suggestions generated"
            )
            
        except Exception as e:
            self.logger.error(f"Content optimization failed: {str(e)}")
            state["errors"].append(f"Content optimization error: {str(e)}")
            # Fallback
            state["content_gaps"] = []
            state["optimization_suggestions"] = []
            state["title_variations"] = [state["blog_title"]]
            state["meta_suggestions"] = {"title": state["blog_title"], "description": ""}
        
        return state
    
    def competitive_analysis_node(self, state: SEOState) -> SEOState:
        """Perform competitive analysis and identify opportunities."""
        try:
            state["current_phase"] = SEOPhase.COMPETITIVE_ANALYSIS
            
            if self.llm and state["keyword_research"]["primary_keywords"]:
                # AI-powered competitive insights
                competitive_insights = self._analyze_competitive_landscape(
                    state["keyword_research"]["primary_keywords"],
                    state["content"],
                    state["search_intent"]
                )
                state["competitive_analysis"] = competitive_insights
            else:
                # Fallback competitive analysis
                state["competitive_analysis"] = self._basic_competitive_analysis(
                    state["keyword_research"]["primary_keywords"]
                )
            
            # Ranking opportunities
            state["ranking_opportunities"] = self._identify_ranking_opportunities(
                state["keyword_research"],
                state["technical_metrics"],
                state["content_gaps"]
            )
            
            self.logger.info("Competitive analysis completed")
            
        except Exception as e:
            self.logger.error(f"Competitive analysis failed: {str(e)}")
            state["errors"].append(f"Competitive analysis error: {str(e)}")
            # Fallback
            state["competitive_analysis"] = {"opportunities": []}
            state["ranking_opportunities"] = []
        
        return state
    
    def schema_generation_node(self, state: SEOState) -> SEOState:
        """Generate schema markup and structured data."""
        try:
            state["current_phase"] = SEOPhase.SCHEMA_GENERATION
            
            # Generate schema markup
            state["schema_markup"] = self._generate_schema_markup(
                state["blog_title"],
                state["content"],
                state["content_type"],
                state["keyword_research"]["primary_keywords"]
            )
            
            # Additional structured data
            state["structured_data"] = {
                "breadcrumbs": self._generate_breadcrumbs(state["outline"]),
                "faq_schema": self._extract_faq_schema(state["content"]),
                "article_schema": self._generate_article_schema(
                    state["blog_title"],
                    state["content"]
                )
            }
            
            self.logger.info("Schema generation completed")
            
        except Exception as e:
            self.logger.error(f"Schema generation failed: {str(e)}")
            state["errors"].append(f"Schema generation error: {str(e)}")
            # Fallback
            state["schema_markup"] = {}
            state["structured_data"] = {}
        
        return state
    
    def final_scoring_node(self, state: SEOState) -> SEOState:
        """Calculate final SEO score and create implementation plan."""
        try:
            state["current_phase"] = SEOPhase.FINAL_SCORING
            
            # Calculate comprehensive SEO score
            state["seo_score"] = self._calculate_seo_score(
                state["technical_metrics"],
                state["readability_analysis"],
                state["heading_structure"],
                state["keyword_density"],
                state["keyword_research"]
            )
            
            # Determine optimization priority
            state["optimization_priority"] = self._determine_optimization_priority(
                state["seo_score"],
                state["content_gaps"],
                state["ranking_opportunities"]
            )
            
            # Create implementation plan
            state["implementation_plan"] = self._create_implementation_plan(
                state["optimization_suggestions"],
                state["content_gaps"],
                state["ranking_opportunities"],
                state["optimization_priority"]
            )
            
            self.logger.info(f"SEO analysis completed - Score: {state['seo_score']:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"Final scoring failed: {str(e)}")
            state["errors"].append(f"Final scoring error: {str(e)}")
            # Fallback
            state["seo_score"] = 50.0
            state["optimization_priority"] = "medium"
            state["implementation_plan"] = []
        
        return state
    
    # Helper methods for SEO analysis
    def _determine_seo_strategy(self, content: str, title: str, keywords: List[str]) -> str:
        """Determine the best SEO strategy based on content analysis."""
        # Simple heuristic-based strategy determination
        commercial_indicators = ["buy", "price", "cost", "deal", "discount", "purchase"]
        transactional_indicators = ["how to", "tutorial", "guide", "step", "download"]
        
        text = (title + " " + content[:500]).lower()
        
        if any(indicator in text for indicator in commercial_indicators):
            return SEOStrategy.COMMERCIAL
        elif any(indicator in text for indicator in transactional_indicators):
            return SEOStrategy.TRANSACTIONAL
        else:
            return SEOStrategy.INFORMATIONAL
    
    def _detect_search_intent(self, title: str, content: str) -> str:
        """Detect the search intent of the content."""
        text = (title + " " + content[:300]).lower()
        
        if any(word in text for word in ["how", "what", "why", "guide", "tutorial"]):
            return "informational"
        elif any(word in text for word in ["buy", "best", "top", "review", "compare"]):
            return "commercial"
        elif any(word in text for word in ["download", "login", "sign up", "register"]):
            return "transactional"
        else:
            return "navigational"
    
    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability."""
        if textstat:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(content),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(content),
                "coleman_liau_index": textstat.coleman_liau_index(content),
                "automated_readability_index": textstat.automated_readability_index(content),
                "average_sentence_length": textstat.avg_sentence_length(content)
            }
        else:
            # Fallback readability analysis
            sentences = len([s for s in content.split('.') if s.strip()])
            words = len(content.split())
            avg_sentence_length = words / sentences if sentences > 0 else 0
            
            return {
                "flesch_reading_ease": 60.0,  # Default middle score
                "average_sentence_length": avg_sentence_length,
                "word_count": words,
                "sentence_count": sentences
            }
    
    def _analyze_heading_structure(self, content: str) -> Dict[str, Any]:
        """Analyze heading structure and hierarchy."""
        headings = {
            "h1": len(re.findall(r'^# ', content, re.MULTILINE)),
            "h2": len(re.findall(r'^## ', content, re.MULTILINE)),
            "h3": len(re.findall(r'^### ', content, re.MULTILINE)),
            "h4": len(re.findall(r'^#### ', content, re.MULTILINE)),
            "h5": len(re.findall(r'^##### ', content, re.MULTILINE)),
            "h6": len(re.findall(r'^###### ', content, re.MULTILINE))
        }
        
        # Extract heading texts
        heading_texts = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            heading_texts.append({"level": level, "text": text})
        
        return {
            "heading_counts": headings,
            "total_headings": sum(headings.values()),
            "heading_texts": heading_texts,
            "has_h1": headings["h1"] > 0,
            "h1_count": headings["h1"],
            "hierarchy_issues": self._check_heading_hierarchy(headings)
        }
    
    def _analyze_url_structure(self, title: str) -> Dict[str, Any]:
        """Analyze and suggest URL structure."""
        # Generate SEO-friendly URL slug
        slug = re.sub(r'[^\w\s-]', '', title.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        slug = slug.strip('-')
        
        return {
            "suggested_slug": slug,
            "slug_length": len(slug),
            "word_count": len(slug.split('-')),
            "contains_keywords": True,  # Simplified assumption
            "readability_score": min(100, max(0, 100 - len(slug)))
        }
    
    def _extract_keywords(self, content: str, title: str) -> List[str]:
        """Extract potential keywords from content."""
        # Combine title and content
        text = (title + " " + content).lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        words = [word for word in words if word not in stop_words]
        
        # Count frequency and return top keywords
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(50) if count > 1]
    
    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword density for given keywords."""
        if not keywords:
            return {}
        
        word_count = len(content.split())
        content_lower = content.lower()
        
        densities = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            densities[keyword] = density
        
        return densities
    
    def _generate_semantic_keywords(self, content: str, title: str, target_keywords: List[str]) -> List[str]:
        """Generate semantic keywords using LLM."""
        if not self.llm:
            return self._extract_keywords(content, title)
        
        try:
            prompt = f"""Analyze this content and generate semantic keywords:

Title: {title}
Content: {content[:2000]}
Target Keywords: {', '.join(target_keywords)}

Generate 20 semantic keywords that are:
1. Semantically related to the main topic
2. Relevant for search optimization
3. Include both short and long-tail keywords
4. Consider search intent and user queries

Return only the keywords, one per line."""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            semantic_keywords = [
                kw.strip() for kw in response.content.strip().split('\n')
                if kw.strip()
            ]
            return semantic_keywords[:20]
            
        except Exception as e:
            self.logger.warning(f"Semantic keyword generation failed: {str(e)}")
            return self._extract_keywords(content, title)
    
    def _extract_long_tail_keywords(self, content: str) -> List[str]:
        """Extract potential long-tail keywords from content."""
        # Find phrases of 3-5 words
        words = content.lower().split()
        long_tail = []
        
        for i in range(len(words) - 2):
            for length in [3, 4, 5]:
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i + length])
                    # Simple quality check
                    if len(phrase) > 10 and phrase.count(' ') == length - 1:
                        long_tail.append(phrase)
        
        # Return most frequent phrases
        phrase_counts = Counter(long_tail)
        return [phrase for phrase, count in phrase_counts.most_common(15) if count > 1]
    
    def _generate_keyword_variations(self, keywords: List[str]) -> List[str]:
        """Generate keyword variations."""
        if not keywords:
            return []
        
        variations = []
        for keyword in keywords:
            # Add plural forms
            if not keyword.endswith('s'):
                variations.append(keyword + 's')
            
            # Add question forms
            variations.append(f"what is {keyword}")
            variations.append(f"how to {keyword}")
            variations.append(f"{keyword} guide")
            variations.append(f"best {keyword}")
        
        return variations
    
    def _identify_keyword_gaps(self, content: str, semantic_keywords: List[str]) -> List[str]:
        """Identify missing keywords that should be included."""
        content_lower = content.lower()
        gaps = []
        
        for keyword in semantic_keywords:
            if keyword.lower() not in content_lower:
                gaps.append(keyword)
        
        return gaps[:10]
    
    def _identify_content_gaps(self, content: str, semantic_keywords: List[str], search_intent: str) -> List[str]:
        """Identify gaps in content coverage."""
        gaps = []
        content_lower = content.lower()
        
        # Intent-based content gaps
        if search_intent == "informational":
            if "what" not in content_lower:
                gaps.append("Add definition or explanation section")
            if "why" not in content_lower:
                gaps.append("Include reasons or benefits")
            if "how" not in content_lower:
                gaps.append("Add step-by-step guidance")
        
        elif search_intent == "commercial":
            if "benefit" not in content_lower and "advantage" not in content_lower:
                gaps.append("Include benefits or advantages")
            if "comparison" not in content_lower and "vs" not in content_lower:
                gaps.append("Add comparison with alternatives")
            if "review" not in content_lower:
                gaps.append("Include reviews or testimonials")
        
        # Missing semantic keywords
        missing_keywords = self._identify_keyword_gaps(content, semantic_keywords)
        if missing_keywords:
            gaps.append(f"Include missing keywords: {', '.join(missing_keywords[:5])}")
        
        return gaps
    
    def _generate_optimization_suggestions(
        self,
        technical_metrics: Dict[str, Any],
        heading_structure: Dict[str, Any],
        keyword_density: Dict[str, float],
        content_gaps: List[str]
    ) -> List[str]:
        """Generate specific optimization suggestions."""
        suggestions = []
        
        # Word count optimization
        word_count = technical_metrics.get("word_count", 0)
        if word_count < 1500:
            suggestions.append(f"Increase content length to at least 1500 words (current: {word_count})")
        elif word_count > 4000:
            suggestions.append(f"Consider breaking into multiple articles (current: {word_count} words)")
        
        # Title optimization
        title_length = technical_metrics.get("title_length", 0)
        if title_length < 30:
            suggestions.append("Lengthen title to 30-60 characters for better SEO")
        elif title_length > 60:
            suggestions.append("Shorten title to under 60 characters")
        
        # Heading structure
        h1_count = heading_structure.get("heading_counts", {}).get("h1", 0)
        if h1_count == 0:
            suggestions.append("Add H1 heading for better structure")
        elif h1_count > 1:
            suggestions.append("Use only one H1 heading per page")
        
        h2_count = heading_structure.get("heading_counts", {}).get("h2", 0)
        if h2_count < 3:
            suggestions.append("Add more H2 headings to improve content structure")
        
        # Keyword density
        high_density_keywords = [
            kw for kw, density in keyword_density.items() if density > 3.5
        ]
        if high_density_keywords:
            suggestions.append(f"Reduce keyword density for: {', '.join(high_density_keywords[:3])}")
        
        low_density_keywords = [
            kw for kw, density in keyword_density.items() if density < 0.5
        ]
        if low_density_keywords:
            suggestions.append(f"Increase usage of: {', '.join(low_density_keywords[:3])}")
        
        # Content gaps
        suggestions.extend(content_gaps[:5])
        
        return suggestions
    
    def _generate_title_variations(self, title: str, primary_keywords: List[str]) -> List[str]:
        """Generate optimized title variations."""
        variations = [title]  # Include original
        
        if primary_keywords:
            main_keyword = primary_keywords[0]
            
            # Add keyword-optimized variations
            if main_keyword.lower() not in title.lower():
                variations.append(f"{main_keyword}: {title}")
                variations.append(f"{title} - {main_keyword} Guide")
                variations.append(f"Complete {main_keyword} Guide: {title}")
            
            # Add action-oriented variations
            variations.append(f"How to {title}")
            variations.append(f"Ultimate Guide to {title}")
            variations.append(f"{title}: Everything You Need to Know")
        
        return variations[:5]
    
    def _optimize_meta_title(self, title: str, primary_keywords: List[str]) -> str:
        """Optimize meta title for SEO."""
        if len(title) <= 60 and primary_keywords and primary_keywords[0].lower() in title.lower():
            return title
        
        if primary_keywords:
            main_keyword = primary_keywords[0]
            optimized = f"{main_keyword} - {title}"
            if len(optimized) <= 60:
                return optimized
        
        # Truncate if too long
        if len(title) > 60:
            return title[:57] + "..."
        
        return title
    
    def _optimize_meta_description(self, content: str, primary_keywords: List[str]) -> str:
        """Generate optimized meta description."""
        # Extract first meaningful paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        if not paragraphs:
            return ""
        
        description = paragraphs[0]
        
        # Include primary keyword if not present
        if primary_keywords and primary_keywords[0].lower() not in description.lower():
            description = f"{primary_keywords[0]} - {description}"
        
        # Truncate to optimal length
        if len(description) > 160:
            description = description[:157] + "..."
        
        return description
    
    def _check_heading_hierarchy(self, headings: Dict[str, int]) -> List[str]:
        """Check for heading hierarchy issues."""
        issues = []
        
        if headings["h1"] == 0:
            issues.append("Missing H1 heading")
        elif headings["h1"] > 1:
            issues.append("Multiple H1 headings found")
        
        if headings["h3"] > 0 and headings["h2"] == 0:
            issues.append("H3 used without H2")
        
        if headings["h4"] > 0 and headings["h3"] == 0:
            issues.append("H4 used without H3")
        
        return issues
    
    def _analyze_competitive_landscape(
        self,
        keywords: List[str],
        content: str,
        search_intent: str
    ) -> Dict[str, Any]:
        """Analyze competitive landscape using LLM."""
        if not self.llm or not keywords:
            return self._basic_competitive_analysis(keywords)
        
        try:
            prompt = f"""Analyze the competitive landscape for these keywords:

Keywords: {', '.join(keywords[:5])}
Search Intent: {search_intent}
Content Preview: {content[:1000]}

Provide competitive analysis including:
1. Content differentiation opportunities
2. Gaps in current market content
3. Unique angles to pursue
4. Competition difficulty assessment

Return as structured analysis."""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            
            # Parse response for structured data
            analysis_text = response.content
            
            return {
                "analysis": analysis_text,
                "difficulty": "medium",  # Default
                "opportunities": self._extract_opportunities(analysis_text),
                "differentiation": self._extract_differentiation_points(analysis_text)
            }
            
        except Exception as e:
            self.logger.warning(f"Competitive analysis failed: {str(e)}")
            return self._basic_competitive_analysis(keywords)
    
    def _basic_competitive_analysis(self, keywords: List[str]) -> Dict[str, Any]:
        """Basic competitive analysis fallback."""
        return {
            "analysis": "Competitive analysis based on keyword research",
            "difficulty": "medium",
            "opportunities": [
                "Focus on long-tail keywords",
                "Create comprehensive content",
                "Optimize for user experience"
            ],
            "differentiation": [
                "Add unique insights",
                "Include practical examples",
                "Provide actionable advice"
            ]
        }
    
    def _extract_opportunities(self, analysis_text: str) -> List[str]:
        """Extract opportunities from analysis text."""
        # Simple extraction based on common patterns
        opportunities = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['opportunity', 'gap', 'missing', 'lack']):
                opportunities.append(line.strip())
        
        return opportunities[:5] if opportunities else ["Create comprehensive content"]
    
    def _extract_differentiation_points(self, analysis_text: str) -> List[str]:
        """Extract differentiation points from analysis."""
        differentiation = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['unique', 'different', 'angle', 'approach']):
                differentiation.append(line.strip())
        
        return differentiation[:5] if differentiation else ["Add unique perspective"]
    
    def _identify_ranking_opportunities(
        self,
        keyword_research: Dict[str, Any],
        technical_metrics: Dict[str, Any],
        content_gaps: List[str]
    ) -> List[str]:
        """Identify specific ranking opportunities."""
        opportunities = []
        
        # Keyword-based opportunities
        primary_keywords = keyword_research.get("primary_keywords", [])
        if primary_keywords:
            opportunities.append(f"Target primary keyword: {primary_keywords[0]}")
        
        long_tail_keywords = keyword_research.get("long_tail_keywords", [])
        if long_tail_keywords:
            opportunities.append(f"Optimize for long-tail: {long_tail_keywords[0]}")
        
        # Content length opportunities
        word_count = technical_metrics.get("word_count", 0)
        if 1500 <= word_count <= 3000:
            opportunities.append("Content length in optimal range for ranking")
        
        # Gap-based opportunities
        if content_gaps:
            opportunities.append(f"Address content gap: {content_gaps[0]}")
        
        return opportunities
    
    def _generate_schema_markup(
        self,
        title: str,
        content: str,
        content_type: str,
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Generate appropriate schema markup."""
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": content[:160] if content else "",
            "keywords": ", ".join(keywords[:10]) if keywords else "",
            "articleSection": content_type.title(),
            "wordCount": len(content.split()),
            "author": {
                "@type": "Organization",
                "name": "CrediLinq"
            }
        }
        
        return schema
    
    def _generate_breadcrumbs(self, outline: List[str]) -> Dict[str, Any]:
        """Generate breadcrumb schema from outline."""
        if not outline:
            return {}
        
        breadcrumbs = {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": []
        }
        
        for i, section in enumerate(outline[:5]):
            item = {
                "@type": "ListItem",
                "position": i + 1,
                "name": section
            }
            breadcrumbs["itemListElement"].append(item)
        
        return breadcrumbs
    
    def _extract_faq_schema(self, content: str) -> Dict[str, Any]:
        """Extract FAQ schema from content if questions are present."""
        # Look for question patterns
        questions = re.findall(r'^(?:##?\s+)?(What|How|Why|When|Where|Who)[^?]*\?', content, re.MULTILINE | re.IGNORECASE)
        
        if not questions:
            return {}
        
        faq_schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": []
        }
        
        for question in questions[:5]:  # Limit to 5 questions
            faq_item = {
                "@type": "Question",
                "name": question,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": "Answer would be extracted from content following the question"
                }
            }
            faq_schema["mainEntity"].append(faq_item)
        
        return faq_schema
    
    def _generate_article_schema(self, title: str, content: str) -> Dict[str, Any]:
        """Generate Article schema markup."""
        return {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": content[:200] if content else "",
            "wordCount": len(content.split()),
            "datePublished": "2024-01-01",  # Would be dynamic
            "dateModified": "2024-01-01",   # Would be dynamic
            "author": {
                "@type": "Organization",
                "name": "CrediLinq"
            },
            "publisher": {
                "@type": "Organization",
                "name": "CrediLinq"
            }
        }
    
    def _calculate_seo_score(
        self,
        technical_metrics: Dict[str, Any],
        readability_analysis: Dict[str, Any],
        heading_structure: Dict[str, Any],
        keyword_density: Dict[str, float],
        keyword_research: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive SEO score."""
        score = 0.0
        max_score = 100.0
        
        # Technical metrics (30 points)
        word_count = technical_metrics.get("word_count", 0)
        if 1500 <= word_count <= 3000:
            score += 15
        elif 1000 <= word_count < 1500 or 3000 < word_count <= 4000:
            score += 10
        elif word_count >= 500:
            score += 5
        
        title_length = technical_metrics.get("title_length", 0)
        if 30 <= title_length <= 60:
            score += 15
        elif 20 <= title_length < 30 or 60 < title_length <= 80:
            score += 10
        
        # Readability (20 points)
        flesch_score = readability_analysis.get("flesch_reading_ease", 60)
        if 60 <= flesch_score <= 80:
            score += 20
        elif 40 <= flesch_score < 60 or 80 < flesch_score <= 90:
            score += 15
        elif flesch_score >= 30:
            score += 10
        
        # Heading structure (20 points)
        h1_count = heading_structure.get("heading_counts", {}).get("h1", 0)
        h2_count = heading_structure.get("heading_counts", {}).get("h2", 0)
        
        if h1_count == 1:
            score += 10
        elif h1_count > 1:
            score += 5
        
        if h2_count >= 3:
            score += 10
        elif h2_count >= 1:
            score += 5
        
        # Keyword optimization (20 points)
        primary_keywords = keyword_research.get("primary_keywords", [])
        if primary_keywords:
            score += 10
        
        optimal_density_count = sum(
            1 for density in keyword_density.values()
            if 1.0 <= density <= 3.0
        )
        if optimal_density_count >= 3:
            score += 10
        elif optimal_density_count >= 1:
            score += 5
        
        # Content quality (10 points)
        if len(keyword_research.get("secondary_keywords", [])) >= 5:
            score += 5
        
        if len(keyword_research.get("long_tail_keywords", [])) >= 3:
            score += 5
        
        return min(score, max_score)
    
    def _determine_optimization_priority(
        self,
        seo_score: float,
        content_gaps: List[str],
        ranking_opportunities: List[str]
    ) -> str:
        """Determine optimization priority level."""
        if seo_score >= 80:
            return "low"
        elif seo_score >= 60:
            return "medium"
        elif seo_score >= 40:
            return "high"
        else:
            return "critical"
    
    def _create_implementation_plan(
        self,
        optimization_suggestions: List[str],
        content_gaps: List[str],
        ranking_opportunities: List[str],
        priority: str
    ) -> List[str]:
        """Create prioritized implementation plan."""
        plan = []
        
        if priority in ["critical", "high"]:
            plan.append("IMMEDIATE ACTIONS:")
            plan.extend(optimization_suggestions[:3])
            plan.append("")
            plan.append("ADDRESS CONTENT GAPS:")
            plan.extend(content_gaps[:2])
        else:
            plan.append("OPTIMIZATION ACTIONS:")
            plan.extend(optimization_suggestions[:5])
        
        plan.append("")
        plan.append("RANKING OPPORTUNITIES:")
        plan.extend(ranking_opportunities[:3])
        
        return plan
    
    async def execute_workflow(
        self,
        initial_state: Dict[str, Any],
        context: Optional[LangGraphExecutionContext] = None
    ) -> WorkflowState:
        """Execute the SEO workflow."""
        try:
            # Convert input to SEOState
            seo_state = SEOState(
                content=initial_state["content"],
                blog_title=initial_state["blog_title"],
                outline=initial_state.get("outline", []),
                target_keywords=initial_state.get("target_keywords", []),
                content_type=initial_state.get("content_type", "blog"),
                seo_strategy="",
                technical_metrics={},
                readability_analysis={},
                heading_structure={},
                url_analysis={},
                keyword_research={},
                semantic_keywords=[],
                keyword_density={},
                search_intent="",
                optimization_suggestions=[],
                content_gaps=[],
                title_variations=[],
                meta_suggestions={},
                competitive_analysis={},
                ranking_opportunities=[],
                schema_markup={},
                structured_data={},
                seo_score=0.0,
                optimization_priority="medium",
                implementation_plan=[],
                current_phase=SEOPhase.INITIALIZATION,
                analysis_depth=self.analysis_depth,
                errors=[],
                warnings=[]
            )
            
            # Execute the graph
            config = {"configurable": {"thread_id": context.session_id if context else "default"}}
            final_state = await self.graph.ainvoke(seo_state, config)
            
            # Convert to WorkflowState
            return WorkflowState(
                status=WorkflowStatus.COMPLETED if final_state["seo_score"] > 0 else WorkflowStatus.FAILED,
                phase=final_state["current_phase"],
                data={
                    "seo_score": final_state["seo_score"],
                    "optimization_priority": final_state["optimization_priority"],
                    "technical_metrics": final_state["technical_metrics"],
                    "keyword_research": final_state["keyword_research"],
                    "optimization_suggestions": final_state["optimization_suggestions"],
                    "content_gaps": final_state["content_gaps"],
                    "meta_suggestions": final_state["meta_suggestions"],
                    "schema_markup": final_state["schema_markup"],
                    "competitive_analysis": final_state["competitive_analysis"],
                    "implementation_plan": final_state["implementation_plan"]
                },
                errors=final_state.get("errors", []),
                metadata={
                    "warnings": final_state.get("warnings", []),
                    "analysis_depth": final_state.get("analysis_depth", ""),
                    "seo_strategy": final_state.get("seo_strategy", "")
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return WorkflowState(
                status=WorkflowStatus.FAILED,
                phase=SEOPhase.INITIALIZATION,
                data={},
                errors=[str(e)],
                metadata={"error_type": type(e).__name__}
            )


# Adapter for backward compatibility
class SEOAgentLangGraph:
    """Adapter to make LangGraph workflow compatible with existing SEOAgent interface."""
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize the adapter."""
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.SEO,
                name="SEOAgentLangGraph",
                description="LangGraph-powered SEO optimizer with comprehensive analysis and optimization",
                capabilities=[
                    "advanced_keyword_research",
                    "technical_seo_analysis",
                    "competitive_analysis",
                    "schema_generation",
                    "content_optimization",
                    "readability_analysis"
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
            
            self.workflow = SEOAgentWorkflow(llm=llm)
            
        except Exception as e:
            # Fallback without LLM
            self.workflow = SEOAgentWorkflow(llm=None)
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Any] = None
    ) -> AgentResult:
        """Execute the SEO workflow."""
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
                    "agent_type": "seo_langgraph",
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
                error_code="SEO_WORKFLOW_FAILED"
            )