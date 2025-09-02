"""
GEO Analysis Agent - Generative Engine Optimization Content Analyzer

This agent analyzes digital content to determine its optimization level for 
generative AI models (GPT-4, Gemini, etc.) and tests performance through 
AI model feedback loops.

The analysis is based on content that is:
- Trustworthy (E-E-A-T): Experience, Expertise, Authoritativeness, Trust
- Machine-Parsable: Clear language and structured data (JSON-LD Schema)
- Factually Dense: Unique data, statistics, verifiable facts
- Proven: Feedback loop to check if content is cited by AI models

This agent complements the existing GEOAgent by providing analysis and scoring
of existing content, while GEOAgent focuses on generating new GEO-optimized content.

Author: Claude Code Assistant
Version: 1.0.0
"""

import json
import re
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import openai

from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GEOAnalysisConfig:
    """Configuration class for GEO Analysis Agent settings."""
    
    # OpenAI API Configuration
    model_name: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 1000
    api_delay_seconds: float = 1.0
    
    # E-E-A-T Scoring Weights (should sum to 100)
    eeat_author_name_points: int = 10
    eeat_author_bio_points: int = 8
    eeat_author_credentials_points: int = 7
    eeat_research_keyword_multiplier: int = 2
    eeat_authority_signal_multiplier: int = 2
    eeat_external_citations_max_points: int = 20
    eeat_publication_date_points: int = 5
    eeat_trust_indicator_multiplier: int = 3
    eeat_max_research_keywords: int = 15
    eeat_max_authority_signals: int = 10
    eeat_max_trust_indicators: int = 8
    eeat_max_citations: int = 20
    
    # Schema Scoring Weights
    schema_weights: Dict[str, int] = None
    schema_properties_weight: int = 20
    
    # Factual Density Configuration
    factual_density_multiplier: int = 1000  # Convert ratio to score
    factual_base_score_max: int = 60
    factual_keyword_score_max: int = 25
    factual_diversity_score_max: int = 15
    factual_keyword_multiplier: int = 2
    factual_diversity_multiplier: int = 2
    
    # Overall Scoring Weights (should sum to 1.0)
    overall_eeat_weight: float = 0.4
    overall_schema_weight: float = 0.3
    overall_factual_weight: float = 0.3
    
    # Performance Grading Thresholds
    grade_a_threshold: float = 80.0
    grade_b_threshold: float = 70.0
    grade_c_threshold: float = 60.0
    grade_d_threshold: float = 50.0
    
    # Performance Assessment Thresholds
    excellent_citation_rate: float = 50.0
    good_citation_rate: float = 30.0
    fair_citation_rate: float = 15.0
    
    # Citation weighting for combined score
    citation_weight: float = 0.7
    mention_weight: float = 0.3
    
    # Negative sentiment threshold for recommendations
    negative_sentiment_threshold: float = 0.3
    
    def __post_init__(self):
        """Initialize default schema weights if not provided."""
        if self.schema_weights is None:
            self.schema_weights = {
                "Article": 20,
                "Person": 15,
                "Organization": 15,
                "Dataset": 20,
                "FAQPage": 10,
                "WebPage": 10,
                "BreadcrumbList": 10
            }


class GEOAnalysisAgent(BaseAgent):
    """
    Generative Engine Optimization Analysis Agent.
    
    Analyzes digital content for optimization targeting generative AI models
    by evaluating E-E-A-T signals, structured data, factual density, and
    testing performance through AI model feedback loops.
    """
    
    def __init__(
        self, 
        openai_api_key: str, 
        metadata: Optional[AgentMetadata] = None,
        config: Optional[GEOAnalysisConfig] = None
    ):
        """
        Initialize the GEO Analysis Agent.
        
        Args:
            openai_api_key (str): OpenAI API key for generative model testing
            metadata (AgentMetadata, optional): Agent metadata
            config (GEOAnalysisConfig, optional): Configuration settings
        """
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_OPTIMIZER,
                name="GEOAnalysisAgent",
                description="Advanced AI-powered GEO analysis and content scoring",
                capabilities=[
                    "eeat_analysis",
                    "schema_validation",
                    "factual_density_analysis",
                    "ai_feedback_testing",
                    "citation_tracking",
                    "performance_grading"
                ],
                version="1.0.0"
            )
        super().__init__(metadata)
        self.config = config or GEOAnalysisConfig()
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        logger.info(f"Initialized {self.metadata.name} with OpenAI client and config")
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[Any] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute the GEO Analysis Agent with provided input data.
        
        Args:
            input_data (dict): Input data containing:
                - content (str): Content to analyze
                - schema (dict, optional): JSON-LD schema data
                - author_info (dict, optional): Author information
                - mode (str, optional): 'analysis' or 'feedback_loop'
                - For feedback_loop mode:
                  - target_url (str): URL to test
                  - brand_name (str): Brand name to track
                  - prompts (list): Test prompts
            context: Execution context (optional)
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Analysis results or feedback loop results
        """
        try:
            # Validate input data
            if not isinstance(input_data, dict):
                return AgentResult(
                    success=False,
                    error_message="Input data must be a dictionary",
                    error_code="INVALID_INPUT_FORMAT"
                )
            
            mode = input_data.get('mode', 'analysis')
            
            if mode == 'analysis':
                # Standard GEO content analysis
                content = input_data.get('content', '')
                schema = input_data.get('schema', {})
                author_info = input_data.get('author_info', {})
                
                if not content:
                    return AgentResult(
                        success=False,
                        error_message="Content is required for analysis mode",
                        error_code="MISSING_CONTENT"
                    )
                
                results = self.analyze_content_for_geo(content, schema, author_info)
                
                return AgentResult(
                    success=True,
                    data=results,
                    metadata={"mode": "analysis", "agent_type": "geo_analysis"}
                )
                
            elif mode == 'feedback_loop':
                # AI feedback loop testing
                target_url = input_data.get('target_url', '')
                brand_name = input_data.get('brand_name', '')
                prompts = input_data.get('prompts', [])
                
                if not all([target_url, brand_name, prompts]):
                    return AgentResult(
                        success=False,
                        error_message="target_url, brand_name, and prompts are required for feedback_loop mode",
                        error_code="MISSING_FEEDBACK_PARAMS"
                    )
                
                # Note: This method is async, so we need to handle it properly
                import asyncio
                
                try:
                    results = asyncio.run(self.execute_geo_feedback_loop(
                        target_url, brand_name, prompts
                    ))
                    
                    return AgentResult(
                        success=True,
                        data=results,
                        metadata={"mode": "feedback_loop", "agent_type": "geo_analysis"}
                    )
                except Exception as async_error:
                    return AgentResult(
                        success=False,
                        error_message=f"Feedback loop execution failed: {str(async_error)}",
                        error_code="FEEDBACK_LOOP_ERROR"
                    )
            
            else:
                return AgentResult(
                    success=False,
                    error_message=f"Unknown mode: {mode}. Supported modes: 'analysis', 'feedback_loop'",
                    error_code="INVALID_MODE"
                )
                
        except Exception as e:
            logger.error(f"Error in GEO Analysis Agent execution: {e}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    def analyze_content_for_geo(
        self, 
        content: str, 
        schema: Dict[str, Any], 
        author_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Primary method to analyze content for Generative Engine Optimization.
        
        Orchestrates comprehensive content analysis including E-E-A-T signals,
        structured data validation, and factual density assessment.
        
        Args:
            content (str): The main text content to analyze
            schema (dict): JSON-LD schema dictionary for structured data analysis
            author_info (dict): Author information including name, bio, credentials
            
        Returns:
            dict: Consolidated analysis results with scores and breakdowns
        """
        try:
            logger.info("Starting GEO content analysis")
            
            # Part 1: E-E-A-T Analysis
            eeat_results = self._analyze_eeat_for_geo(content, author_info)
            
            # Part 2: Structured Data Analysis
            schema_results = self._analyze_structured_data_for_geo(schema)
            
            # Part 3: Factual Density Analysis
            factual_results = self._analyze_factual_density(content)
            
            # Calculate overall GEO score
            overall_score = self._calculate_overall_geo_score(
                eeat_results['score'],
                schema_results['score'], 
                factual_results['score']
            )
            
            results = {
                "overall_geo_score": overall_score,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "content_length": len(content),
                "word_count": len(content.split()),
                "eeat_analysis": eeat_results,
                "structured_data_analysis": schema_results,
                "factual_density_analysis": factual_results,
                "recommendations": self._generate_geo_recommendations(
                    eeat_results, schema_results, factual_results
                )
            }
            
            logger.info(f"GEO analysis completed. Overall score: {overall_score}/100")
            return results
            
        except Exception as e:
            logger.error(f"Error in GEO content analysis: {e}")
            return {
                "error": str(e),
                "overall_geo_score": 0,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def _analyze_eeat_for_geo(self, content: str, author_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze content for E-E-A-T (Experience, Expertise, Authoritativeness, Trust) signals.
        
        E-E-A-T is crucial for GEO as generative models prioritize trustworthy,
        authoritative content with clear provenance.
        
        Args:
            content (str): Content text to analyze
            author_info (dict): Author information dictionary
            
        Returns:
            dict: E-E-A-T analysis with score (0-100) and signal breakdown
        """
        try:
            signals = {
                "has_named_author": bool(author_info.get("name")),
                "has_author_bio": bool(author_info.get("bio")),
                "has_author_credentials": bool(author_info.get("credentials")),
                "has_publication_date": bool(author_info.get("publication_date")),
                "external_citations": self._count_external_citations(content),
                "research_keywords": self._count_research_keywords(content),
                "authority_signals": self._detect_authority_signals(content),
                "trust_indicators": self._detect_trust_indicators(content)
            }
            
            # Calculate E-E-A-T score
            score = 0
            
            # Experience signals
            if signals["has_named_author"]:
                score += self.config.eeat_author_name_points
            if signals["has_author_bio"]:
                score += self.config.eeat_author_bio_points
            if signals["has_author_credentials"]:
                score += self.config.eeat_author_credentials_points
            
            # Expertise signals
            score += min(signals["research_keywords"] * self.config.eeat_research_keyword_multiplier, 15)
            score += min(signals["authority_signals"] * self.config.eeat_authority_signal_multiplier, 10)
            
            # Authoritativeness signals
            score += min(signals["external_citations"], self.config.eeat_external_citations_max_points) 
            if signals["has_publication_date"]:
                score += self.config.eeat_publication_date_points
            
            # Trust signals
            score += min(signals["trust_indicators"] * self.config.eeat_trust_indicator_multiplier, 25)
            
            return {
                "score": min(score, 100),
                "signals": signals,
                "breakdown": {
                    "experience": min((10 if signals["has_named_author"] else 0) + 
                                   (8 if signals["has_author_bio"] else 0) + 
                                   (7 if signals["has_author_credentials"] else 0), 25),
                    "expertise": min(signals["research_keywords"] * 2 + 
                                   signals["authority_signals"] * 2, 25),
                    "authoritativeness": min(signals["external_citations"] + 
                                           (5 if signals["has_publication_date"] else 0), 25),
                    "trust": min(signals["trust_indicators"] * 3, 25)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in E-E-A-T analysis: {e}")
            return {"score": 0, "error": str(e)}
    
    def _analyze_structured_data_for_geo(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze JSON-LD schema for GEO-critical structured data elements.
        
        Structured data helps generative models understand and cite content
        by providing machine-parsable context and metadata.
        
        Args:
            schema (dict): JSON-LD schema dictionary
            
        Returns:
            dict: Structured data analysis with score (0-100) and schema breakdown
        """
        try:
            critical_schemas = {
                "Article": False,
                "Person": False,
                "Organization": False,
                "FAQPage": False,
                "Dataset": False,
                "WebPage": False,
                "BreadcrumbList": False
            }
            
            schema_found = {}
            score = 0
            
            # Check for schema types in the provided schema
            schema_type = schema.get("@type", "")
            if isinstance(schema_type, str):
                schema_types = [schema_type]
            elif isinstance(schema_type, list):
                schema_types = schema_type
            else:
                schema_types = []
            
            # Check main schema types
            for schema_name in critical_schemas:
                if schema_name in schema_types:
                    critical_schemas[schema_name] = True
                    schema_found[schema_name] = True
                    
            # Check nested schemas (author, publisher, etc.)
            if "author" in schema and isinstance(schema["author"], dict):
                if schema["author"].get("@type") == "Person":
                    critical_schemas["Person"] = True
                    schema_found["Person"] = True
                    
            if "publisher" in schema and isinstance(schema["publisher"], dict):
                if schema["publisher"].get("@type") == "Organization":
                    critical_schemas["Organization"] = True
                    schema_found["Organization"] = True
            
            # Calculate score based on critical schema presence
            # Use config schema weights
            
            for schema_name, present in critical_schemas.items():
                if present:
                    score += self.config.schema_weights.get(schema_name, 5)
            
            # Additional scoring for schema completeness
            required_properties = ["headline", "datePublished", "author", "description"]
            properties_found = sum(1 for prop in required_properties if prop in schema)
            score += (properties_found / len(required_properties)) * self.config.schema_properties_weight
            
            return {
                "score": min(score, 100),
                "schemas_found": schema_found,
                "critical_schemas": critical_schemas,
                "total_schemas": len(schema_found),
                "properties_completeness": (properties_found / len(required_properties)) * 100,
                "recommendations": self._generate_schema_recommendations(critical_schemas)
            }
            
        except Exception as e:
            logger.error(f"Error in structured data analysis: {e}")
            return {"score": 0, "error": str(e)}
    
    def _analyze_factual_density(self, content: str) -> Dict[str, Any]:
        """
        Analyze the density of verifiable facts and data in the content.
        
        Factually dense content is prioritized by generative models as it
        provides unique, citable information for synthesis.
        
        Args:
            content (str): Content text to analyze
            
        Returns:
            dict: Factual density analysis with score (0-100) and statistics
        """
        try:
            # Pattern matching for factual elements
            patterns = {
                "percentages": r'\b\d+(?:\.\d+)?%\b',
                "currency": r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
                "numbers": r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
                "dates": r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\w+ \d{1,2}, \d{4})\b',
                "statistics": r'\b(?:according to|study shows?|research indicates?|data shows?|statistics show)\b',
                "citations": r'\[[^\]]+\]|\(\w+,?\s*\d{4}\)',
                "measurements": r'\b\d+(?:\.\d+)?\s*(?:kg|lb|km|mi|ft|m|cm|mm|inches?|feet)\b'
            }
            
            word_count = len(content.split())
            factual_elements = {}
            total_facts = 0
            
            for category, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                factual_elements[category] = len(matches)
                total_facts += len(matches)
            
            # Data-related keywords
            data_keywords = [
                "research", "study", "data", "analysis", "survey", "poll",
                "experiment", "findings", "results", "evidence", "proof",
                "statistics", "metrics", "measurement", "benchmark"
            ]
            
            keyword_count = sum(
                len(re.findall(rf'\b{keyword}\b', content, re.IGNORECASE))
                for keyword in data_keywords
            )
            
            # Calculate factual density score
            if word_count == 0:
                density_ratio = 0
            else:
                density_ratio = total_facts / word_count
            
            # Score calculation (0-100)
            base_score = min(
                density_ratio * self.config.factual_density_multiplier, 
                self.config.factual_base_score_max
            )
            keyword_score = min(
                keyword_count * self.config.factual_keyword_multiplier, 
                self.config.factual_keyword_score_max
            )
            diversity_score = min(
                len([k for k, v in factual_elements.items() if v > 0]) * self.config.factual_diversity_multiplier, 
                self.config.factual_diversity_score_max
            )
            
            total_score = base_score + keyword_score + diversity_score
            
            return {
                "score": min(int(total_score), 100),
                "total_factual_elements": total_facts,
                "factual_density_ratio": round(density_ratio * 100, 2),  # Facts per 100 words
                "word_count": word_count,
                "factual_breakdown": factual_elements,
                "data_keywords_count": keyword_count,
                "factual_categories_used": len([k for k, v in factual_elements.items() if v > 0])
            }
            
        except Exception as e:
            logger.error(f"Error in factual density analysis: {e}")
            return {"score": 0, "error": str(e)}
    
    async def execute_geo_feedback_loop(
        self, 
        target_url: str, 
        brand_name: str, 
        prompt_suite: List[str]
    ) -> Dict[str, Any]:
        """
        Execute generative output feedback loop to test content performance.
        
        Tests if the content is actually being cited or mentioned by generative
        AI models when responding to relevant prompts.
        
        Args:
            target_url (str): URL of the content being tested
            brand_name (str): Brand/organization name to track mentions
            prompt_suite (list): List of test prompts to query AI models
            
        Returns:
            dict: Feedback loop results with citation and mention rates
        """
        try:
            logger.info(f"Starting GEO feedback loop for {target_url}")
            
            results = {
                "target_url": target_url,
                "brand_name": brand_name,
                "test_timestamp": datetime.utcnow().isoformat(),
                "prompts_tested": len(prompt_suite),
                "individual_results": [],
                "errors": []
            }
            
            for i, prompt in enumerate(prompt_suite):
                try:
                    logger.info(f"Testing prompt {i+1}/{len(prompt_suite)}")
                    
                    # Query OpenAI API
                    response = self.openai_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    generated_text = response.choices[0].message.content
                    
                    # Parse response for citations and mentions
                    parsed_result = self._parse_geo_response(
                        generated_text, target_url, brand_name
                    )
                    
                    individual_result = {
                        "prompt_index": i,
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        "response_length": len(generated_text),
                        "was_cited": parsed_result["was_cited"],
                        "was_mentioned": parsed_result["was_mentioned"],
                        "sentiment": parsed_result["sentiment"],
                        "citation_context": parsed_result.get("citation_context", ""),
                        "mention_context": parsed_result.get("mention_context", "")
                    }
                    
                    results["individual_results"].append(individual_result)
                    
                except Exception as e:
                    error_info = f"Error with prompt {i+1}: {str(e)}"
                    logger.error(error_info)
                    results["errors"].append(error_info)
                    
                # Add delay to respect API rate limits
                await asyncio.sleep(self.config.api_delay_seconds)
            
            # Summarize performance
            performance_summary = self._summarize_geo_performance(results)
            results.update(performance_summary)
            
            logger.info(f"Feedback loop completed. Citation rate: {performance_summary.get('citation_rate', 0)}%")
            return results
            
        except Exception as e:
            logger.error(f"Error in GEO feedback loop: {e}")
            return {
                "error": str(e),
                "target_url": target_url,
                "brand_name": brand_name,
                "test_timestamp": datetime.utcnow().isoformat()
            }
    
    def _parse_geo_response(
        self, 
        generated_text: str, 
        target_url: str, 
        brand_name: str
    ) -> Dict[str, Any]:
        """
        Parse AI-generated response for citations and brand mentions.
        
        Args:
            generated_text (str): Text generated by AI model
            target_url (str): Target URL to search for citations
            brand_name (str): Brand name to search for mentions
            
        Returns:
            dict: Parsing results with citation/mention flags and sentiment
        """
        try:
            # Check for URL citation (exact or domain match)
            was_cited = target_url.lower() in generated_text.lower()
            
            # If exact URL not found, check for domain
            if not was_cited and target_url.startswith("http"):
                from urllib.parse import urlparse
                domain = urlparse(target_url).netloc
                was_cited = domain.lower() in generated_text.lower()
            
            # Check for brand mention
            was_mentioned = brand_name.lower() in generated_text.lower()
            
            # Extract context around citations and mentions
            citation_context = ""
            mention_context = ""
            
            if was_cited:
                # Find sentence containing the citation
                sentences = re.split(r'[.!?]+', generated_text)
                for sentence in sentences:
                    if target_url.lower() in sentence.lower() or (
                        target_url.startswith("http") and 
                        urlparse(target_url).netloc.lower() in sentence.lower()
                    ):
                        citation_context = sentence.strip()
                        break
            
            if was_mentioned:
                # Find sentence containing the brand mention
                sentences = re.split(r'[.!?]+', generated_text)
                for sentence in sentences:
                    if brand_name.lower() in sentence.lower():
                        mention_context = sentence.strip()
                        break
            
            # Simple sentiment analysis
            sentiment = self._analyze_sentiment(mention_context if was_mentioned else citation_context)
            
            return {
                "was_cited": was_cited,
                "was_mentioned": was_mentioned,
                "sentiment": sentiment,
                "citation_context": citation_context,
                "mention_context": mention_context,
                "response_word_count": len(generated_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error parsing GEO response: {e}")
            return {
                "was_cited": False,
                "was_mentioned": False,
                "sentiment": "neutral",
                "error": str(e)
            }
    
    def _summarize_geo_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize overall GEO performance from feedback loop results.
        
        Args:
            results (dict): Individual test results from feedback loop
            
        Returns:
            dict: Performance summary with rates and metrics
        """
        try:
            individual_results = results.get("individual_results", [])
            total_tests = len(individual_results)
            
            if total_tests == 0:
                return {
                    "citation_rate": 0,
                    "mention_rate": 0,
                    "overall_performance": "No valid tests completed"
                }
            
            citations = sum(1 for r in individual_results if r.get("was_cited", False))
            mentions = sum(1 for r in individual_results if r.get("was_mentioned", False))
            
            # Sentiment analysis
            sentiments = [r.get("sentiment", "neutral") for r in individual_results]
            positive_sentiment = sentiments.count("positive")
            negative_sentiment = sentiments.count("negative")
            neutral_sentiment = sentiments.count("neutral")
            
            citation_rate = (citations / total_tests) * 100
            mention_rate = (mentions / total_tests) * 100
            
            # Overall performance assessment
            if citation_rate >= 50:
                performance = "Excellent"
            elif citation_rate >= 30:
                performance = "Good"
            elif citation_rate >= 15:
                performance = "Fair"
            else:
                performance = "Poor"
            
            return {
                "citation_rate": round(citation_rate, 1),
                "mention_rate": round(mention_rate, 1),
                "total_tests": total_tests,
                "total_citations": citations,
                "total_mentions": mentions,
                "sentiment_breakdown": {
                    "positive": positive_sentiment,
                    "negative": negative_sentiment,
                    "neutral": neutral_sentiment
                },
                "overall_performance": performance,
                "performance_grade": self._assign_performance_grade(citation_rate, mention_rate),
                "recommendations": self._generate_performance_recommendations(
                    citation_rate, mention_rate, sentiments
                )
            }
            
        except Exception as e:
            logger.error(f"Error summarizing GEO performance: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    def _count_external_citations(self, content: str) -> int:
        """Count external citations and references in content."""
        # Look for URLs, citation patterns, and reference markers
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        citation_pattern = r'\[[^\]]+\]|\(\w+,?\s*\d{4}\)'
        
        urls = len(re.findall(url_pattern, content))
        citations = len(re.findall(citation_pattern, content))
        
        return min(urls + citations, 20)  # Cap at 20 for scoring
    
    def _count_research_keywords(self, content: str) -> int:
        """Count research and expertise-related keywords."""
        keywords = [
            "research", "study", "analysis", "investigation", "methodology",
            "peer-reviewed", "journal", "publication", "findings", "conclusion",
            "hypothesis", "experiment", "survey", "sample", "correlation"
        ]
        
        count = 0
        for keyword in keywords:
            count += len(re.findall(rf'\b{keyword}\b', content, re.IGNORECASE))
        
        return min(count, 15)  # Cap for scoring
    
    def _detect_authority_signals(self, content: str) -> int:
        """Detect authority and expertise signals in content."""
        authority_phrases = [
            "according to experts", "leading researcher", "published in",
            "professor", "PhD", "doctor", "specialist", "expert",
            "certified", "licensed", "board-certified", "authority"
        ]
        
        count = 0
        for phrase in authority_phrases:
            count += len(re.findall(rf'\b{phrase}\b', content, re.IGNORECASE))
        
        return min(count, 10)  # Cap for scoring
    
    def _detect_trust_indicators(self, content: str) -> int:
        """Detect trust and credibility indicators."""
        trust_indicators = [
            "verified", "fact-checked", "reviewed", "updated",
            "disclaimer", "privacy policy", "terms", "contact",
            "about us", "credentials", "certification", "accredited"
        ]
        
        count = 0
        for indicator in trust_indicators:
            count += len(re.findall(rf'\b{indicator}\b', content, re.IGNORECASE))
        
        return min(count, 8)  # Cap for scoring
    
    def _calculate_overall_geo_score(
        self, 
        eeat_score: int, 
        schema_score: int, 
        factual_score: int
    ) -> int:
        """Calculate weighted overall GEO score."""
        # Weighted scoring: E-E-A-T (40%), Schema (30%), Factual (30%)
        overall = (
            eeat_score * self.config.overall_eeat_weight + 
            schema_score * self.config.overall_schema_weight + 
            factual_score * self.config.overall_factual_weight
        )
        return min(int(overall), 100)
    
    def _generate_geo_recommendations(
        self, 
        eeat_results: Dict[str, Any], 
        schema_results: Dict[str, Any], 
        factual_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable GEO improvement recommendations."""
        recommendations = []
        
        # E-E-A-T recommendations
        if eeat_results.get("score", 0) < 70:
            if not eeat_results.get("signals", {}).get("has_named_author"):
                recommendations.append("Add clear author attribution with name and credentials")
            if eeat_results.get("signals", {}).get("external_citations", 0) < 3:
                recommendations.append("Include more citations to authoritative external sources")
            if eeat_results.get("signals", {}).get("research_keywords", 0) < 5:
                recommendations.append("Incorporate more research and data-focused language")
        
        # Schema recommendations
        if schema_results.get("score", 0) < 70:
            missing_schemas = [
                k for k, v in schema_results.get("critical_schemas", {}).items() 
                if not v and k in ["Article", "Person", "Dataset"]
            ]
            if missing_schemas:
                recommendations.append(f"Add missing critical schema types: {', '.join(missing_schemas)}")
        
        # Factual density recommendations
        if factual_results.get("score", 0) < 70:
            if factual_results.get("total_factual_elements", 0) < 10:
                recommendations.append("Include more quantitative data, statistics, and measurable facts")
            if factual_results.get("data_keywords_count", 0) < 5:
                recommendations.append("Use more data-oriented language and research terminology")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_schema_recommendations(self, critical_schemas: Dict[str, bool]) -> List[str]:
        """Generate schema-specific recommendations."""
        recommendations = []
        
        if not critical_schemas.get("Article"):
            recommendations.append("Add Article schema for content recognition")
        if not critical_schemas.get("Person"):
            recommendations.append("Add Person schema for author attribution")
        if not critical_schemas.get("Dataset"):
            recommendations.append("Add Dataset schema if content contains data/research")
        if not critical_schemas.get("Organization"):
            recommendations.append("Add Organization schema for publisher credibility")
        
        return recommendations
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis of text."""
        if not text:
            return "neutral"
        
        positive_words = [
            "excellent", "great", "good", "positive", "beneficial", "helpful",
            "effective", "successful", "valuable", "important", "significant",
            "impressive", "outstanding", "reliable", "trustworthy"
        ]
        
        negative_words = [
            "bad", "poor", "negative", "harmful", "ineffective", "unreliable",
            "questionable", "problematic", "concerning", "disappointing",
            "inadequate", "insufficient", "misleading", "inaccurate"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _assign_performance_grade(self, citation_rate: float, mention_rate: float) -> str:
        """Assign letter grade based on performance metrics."""
        combined_score = (
            citation_rate * self.config.citation_weight + 
            mention_rate * self.config.mention_weight
        )
        
        if combined_score >= self.config.grade_a_threshold:
            return "A"
        elif combined_score >= self.config.grade_b_threshold:
            return "B"
        elif combined_score >= self.config.grade_c_threshold:
            return "C"
        elif combined_score >= self.config.grade_d_threshold:
            return "D"
        else:
            return "F"
    
    def _generate_performance_recommendations(
        self, 
        citation_rate: float, 
        mention_rate: float, 
        sentiments: List[str]
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if citation_rate < 20:
            recommendations.append("Content rarely cited - improve authority and uniqueness")
        elif citation_rate < 40:
            recommendations.append("Moderate citation rate - enhance factual density and expertise signals")
        
        if mention_rate < 30:
            recommendations.append("Low brand mention rate - strengthen brand authority and thought leadership")
        
        negative_sentiment_rate = sentiments.count("negative") / len(sentiments) if sentiments else 0
        if negative_sentiment_rate > 0.3:
            recommendations.append("High negative sentiment - review content quality and accuracy")
        
        return recommendations


# Example usage and demonstration
if __name__ == "__main__":
    # Sample data for demonstration
    sample_content = """
    According to a recent study published in the Journal of Digital Marketing (2024), 
    companies that implement structured data see a 73% increase in search visibility. 
    Our research team at TechCorp analyzed over 10,000 websites and found that only 
    23% properly implement JSON-LD schema markup. The data shows that websites with 
    comprehensive schema markup receive 2.5x more citations from AI models.
    
    Dr. Sarah Johnson, PhD in Computer Science and lead researcher, states: 
    "The correlation between structured data and AI citation rates is significant at p<0.001."
    This finding aligns with previous research from Stanford University showing similar patterns.
    
    Our methodology involved testing 500 different prompts across multiple AI models 
    including GPT-4, Claude, and Gemini. The results consistently demonstrated that 
    content with proper E-E-A-T signals and factual density performed 4x better 
    in generative engine results.
    """
    
    sample_schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": "The Impact of Structured Data on AI Citation Rates",
        "author": {
            "@type": "Person",
            "name": "Dr. Sarah Johnson",
            "jobTitle": "Lead Researcher",
            "affiliation": {
                "@type": "Organization",
                "name": "TechCorp Research Division"
            }
        },
        "publisher": {
            "@type": "Organization",
            "name": "TechCorp",
            "url": "https://techcorp.com"
        },
        "datePublished": "2024-01-15",
        "dateModified": "2024-01-15",
        "description": "Research study on structured data impact on AI model citations",
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": "https://techcorp.com/structured-data-research"
        }
    }
    
    sample_author_info = {
        "name": "Dr. Sarah Johnson",
        "bio": "Lead researcher with 15 years experience in digital marketing and AI",
        "credentials": "PhD Computer Science, Stanford University",
        "publication_date": "2024-01-15"
    }
    
    sample_prompts = [
        "What are the latest findings on structured data and SEO performance?",
        "How does schema markup affect AI model citations?",
        "What research exists on generative engine optimization?",
        "Which companies are leading in structured data implementation?",
        "What are the best practices for AI-friendly content optimization?"
    ]
    
    # Initialize agent (use placeholder API key for demo)
    try:
        # Create custom configuration if needed
        custom_config = GEOAnalysisConfig(
            model_name="gpt-4",  # or "gpt-3.5-turbo" for cost savings
            temperature=0.2,     # Lower temperature for more consistent results
            max_tokens=800,      # Reduce for shorter responses
            api_delay_seconds=0.5,  # Faster API calls if rate limits allow
            # Adjust scoring weights as needed
            overall_eeat_weight=0.5,  # Increase E-E-A-T importance
            overall_schema_weight=0.3,
            overall_factual_weight=0.2
        )
        
        # Initialize with custom config (or use default by omitting config parameter)
        agent = GEOAnalysisAgent("YOUR_GEMINI_API_KEY_OR_GOOGLE_API_KEY", config=custom_config)        
        print("=== GEO Analysis Agent Demo ===\n")
        
        # Part 1: Content Analysis
        print("1. Analyzing content for GEO optimization...")
        content_results = agent.analyze_content_for_geo(
            sample_content, 
            sample_schema, 
            sample_author_info
        )
        
        print("Content Analysis Results:")
        print(json.dumps(content_results, indent=2))
        print("\n" + "="*50 + "\n")
        
        # Part 2: Feedback Loop (commented out to avoid API calls in demo)
        print("2. Feedback Loop Analysis (Demo Mode - Skipped)")
        print("In production, this would test the content against AI models:")
        print(f"Target URL: https://techcorp.com/structured-data-research")
        print(f"Brand Name: TechCorp")
        print(f"Test Prompts: {len(sample_prompts)} prompts")
        
        # Uncomment the following lines to run actual feedback loop:
        # print("2. Executing feedback loop...")
        # feedback_results = asyncio.run(agent.execute_geo_feedback_loop(
        #     "https://techcorp.com/structured-data-research",
        #     "TechCorp",
        #     sample_prompts
        # ))
        # print("Feedback Loop Results:")
        # print(json.dumps(feedback_results, indent=2))
        
    except Exception as e:
        print(f"Demo Error: {e}")
        print("Note: This demo requires a valid Gemini or Google API key for full functionality.")
        print("Replace 'YOUR_GEMINI_API_KEY_OR_GOOGLE_API_KEY' with your actual API key to test the feedback loop.")