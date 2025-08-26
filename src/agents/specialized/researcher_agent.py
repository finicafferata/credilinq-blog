"""
Researcher Agent - Performs knowledge base research and information gathering.
"""

from typing import List, Dict, Any, Optional
import os
import time
from langchain_openai import OpenAIEmbeddings

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator
from ..core.database_service import get_db_service


class ResearcherAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for researching information from knowledge base and external sources.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            # Initialize database URL from environment
            self.db_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
            
            # Initialize security validator
            self.security_validator = SecurityValidator()
            
            # Initialize embeddings model immediately
            from ...config.settings import settings
            self.embeddings_model = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key
            )
            # Initialize database service immediately
            self.db_service = get_db_service()
            
            self.logger.info("ResearcherAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ResearcherAgent: {str(e)}")
            raise
    
    def _initialize(self):
        """Initialize method - everything is already initialized in __init__."""
        self.logger.info("ResearcherAgent already fully initialized")
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for research."""
        super()._validate_input(input_data)
        
        required_fields = ["outline", "blog_title"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate outline is a list
        if not isinstance(input_data["outline"], list):
            raise ValueError("Outline must be a list of sections")
        
        # Security validation (relaxed for NL fields)
        self.security_validator.validate_content(str(input_data["blog_title"]), "blog_title")
        for section in input_data["outline"]:
            self.security_validator.validate_content(str(section), "outline_section")
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Research information for each section in the outline.
        
        Args:
            input_data: Dictionary containing:
                - outline: List of sections to research
                - blog_title: Main topic for context
                - company_context: Company context (optional)
            context: Execution context
            
        Returns:
            AgentResult: Result containing research for each section
        """
        try:
            # Initialize if not already done
            self._initialize()
            
            outline = input_data["outline"]
            blog_title = input_data["blog_title"]
            company_context = input_data.get("company_context", "")
            
            self.logger.info(f"Researching {len(outline)} sections for: {blog_title}")
            
            # Perform research for each section
            research_results = {}
            research_metadata = {
                "total_sections": len(outline),
                "successful_searches": 0,
                "fallback_sections": 0,
                "total_chunks_found": 0
            }
            
            for section in outline:
                section_research = self._research_section(
                    section, blog_title, company_context
                )
                research_results[section] = section_research
                
                # Update metadata
                if section_research["source"] == "vector_search":
                    research_metadata["successful_searches"] += 1
                    research_metadata["total_chunks_found"] += section_research.get("chunks_found", 0)
                else:
                    research_metadata["fallback_sections"] += 1
            
            result_data = {
                "research": research_results,
                "research_quality": self._assess_research_quality(research_results),
                "sections_researched": len(research_results)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata=research_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Research failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="RESEARCH_FAILED"
            )
    
    def _research_section(
        self, 
        section: str, 
        blog_title: str, 
        company_context: str
    ) -> Dict[str, Any]:
        """Research information for a specific section."""
        try:
            # Create search query
            query = f"Find relevant information for a blog section titled '{section}' on the main topic of '{blog_title}'."
            
            self.logger.info(f"Researching section: {section}")
            
            # Try vector search first
            try:
                research_content, chunks_found = self._perform_vector_search(query)
                
                if research_content:
                    return {
                        "content": research_content,
                        "source": "vector_search",
                        "chunks_found": chunks_found,
                        "query_used": query,
                        "confidence": "high"
                    }
            except Exception as e:
                self.logger.warning(f"Vector search failed for section '{section}': {str(e)}")
            
            # Fallback to local research
            fallback_content = self._perform_fallback_research(section, blog_title, company_context)
            
            return {
                "content": fallback_content,
                "source": "fallback",
                "chunks_found": 0,
                "query_used": query,
                "confidence": "medium"
            }
            
        except Exception as e:
            self.logger.error(f"Section research failed for '{section}': {str(e)}")
            return {
                "content": f"General information about {section} based on {company_context}",
                "source": "error_fallback",
                "chunks_found": 0,
                "query_used": "",
                "confidence": "low",
                "error": str(e)
            }
    
    def _perform_vector_search(self, query: str) -> tuple[str, int]:
        """Perform vector search using the database service."""
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Validate embedding for security
            self.security_validator.validate_vector_embedding(query_embedding)
            
            # Perform vector search using database service
            search_results = self.db_service.vector_search(query, limit=3)
            
            if search_results:
                # Combine results
                research_content = "\n\n".join([
                    result.get("content", "") for result in search_results
                ])
                return research_content, len(search_results)
            
            return "", 0
            
        except Exception as e:
            self.logger.error(f"Vector search error: {str(e)}")
            raise
    
    def _perform_fallback_research(
        self, 
        section: str, 
        blog_title: str, 
        company_context: str
    ) -> str:
        """Perform fallback research using local files or general knowledge."""
        self.logger.info("Using fallback research method")
        
        # Try to find local knowledge base files
        knowledge_base_path = "knowledge_base"
        
        if os.path.exists(knowledge_base_path):
            return self._search_local_files(section, knowledge_base_path)
        
        # Final fallback: create contextual information
        return self._generate_contextual_content(section, blog_title, company_context)
    
    def _search_local_files(self, section: str, knowledge_base_path: str) -> str:
        """Search for relevant content in local files."""
        section_research = []
        section_keywords = section.lower().split()
        
        try:
            for filename in os.listdir(knowledge_base_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(knowledge_base_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Simple keyword matching
                        if any(keyword in content.lower() for keyword in section_keywords):
                            # Take relevant excerpt
                            excerpt = content[:500] + "..." if len(content) > 500 else content
                            section_research.append(excerpt)
                            
                    except Exception as e:
                        self.logger.warning(f"Error reading {filename}: {str(e)}")
                        continue
            
            if section_research:
                return "\n\n".join(section_research)
                
        except Exception as e:
            self.logger.warning(f"Local file search failed: {str(e)}")
        
        return ""
    
    def _generate_contextual_content(
        self, 
        section: str, 
        blog_title: str, 
        company_context: str
    ) -> str:
        """Generate contextual content focusing on supporting elements for compelling content."""
        return f"""
        Research Framework for: {section} in the context of {blog_title}
        
        STATISTICS & DATA POINTS NEEDED:
        - Industry benchmarks relevant to {section}
        - Market statistics that support the problem/challenge
        - Performance metrics demonstrating the impact
        - Percentage-based claims that add credibility
        
        REAL EXAMPLES & CASE STUDIES:
        - Specific company examples facing {section} challenges
        - Customer scenarios that illustrate the pain points
        - Success stories showing problem resolution
        - Industry-specific examples (e.g., "Amazon sellers", "TikTok merchants")
        
        EXPERT QUOTES & AUTHORITY SOURCES:
        - Industry leader quotes about {section}
        - Expert insights on the challenges and solutions
        - Authority figures supporting the approach (like Jack Welch style quotes)
        - Credible third-party validation
        
        CUSTOMER TESTIMONIALS & SOCIAL PROOF:
        - User testimonials about {section} challenges
        - Forum quotes or social media evidence
        - Customer success metrics and outcomes
        - Before/after scenarios from real users
        
        COMPARISON DATA:
        - Traditional approach vs. modern solutions
        - Competitor analysis for {section}
        - Feature/benefit comparisons
        - "Old way vs. new way" structures
        
        VISUAL ELEMENTS TO SUPPORT:
        - Chart concepts for illustrating {section} data
        - Table structures for comparisons
        - Process diagrams or flowcharts
        - Timeline or step-by-step visual concepts
        
        COMPANY-SPECIFIC APPLICATION:
        - How {company_context} specifically addresses {section}
        - Unique value propositions related to this challenge
        - Competitive advantages in solving {section} problems
        - Customer success metrics specific to the company solution
        
        ENGAGEMENT HOOKS:
        - Surprising statistics or counterintuitive facts about {section}
        - Common misconceptions to address
        - Urgent problems that need immediate attention
        - Future trends and implications
        
        Note: This research framework should be populated with specific, credible data points.
        """
    
    def _assess_research_quality(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of research results."""
        total_sections = len(research_results)
        vector_search_sections = sum(
            1 for result in research_results.values() 
            if result.get("source") == "vector_search"
        )
        
        quality_score = (vector_search_sections / total_sections) * 100 if total_sections > 0 else 0
        
        quality_assessment = {
            "overall_score": quality_score,
            "quality_level": self._get_quality_level(quality_score),
            "vector_search_coverage": (vector_search_sections / total_sections) * 100,
            "sections_with_high_confidence": sum(
                1 for result in research_results.values()
                if result.get("confidence") == "high"
            ),
            "total_content_length": sum(
                len(result.get("content", "")) for result in research_results.values()
            )
        }
        
        return quality_assessment
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"
    
    def research_specific_topic(
        self,
        topic: str,
        max_results: int = 5,
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Research a specific topic independently.
        
        Args:
            topic: Topic to research
            max_results: Maximum number of results
            context: Execution context
            
        Returns:
            AgentResult: Research results for the topic
        """
        try:
            research_content, chunks_found = self._perform_vector_search(topic)
            
            if not research_content:
                research_content = self._generate_contextual_content(
                    topic, topic, "general research"
                )
            
            result_data = {
                "topic": topic,
                "content": research_content,
                "chunks_found": chunks_found,
                "research_method": "vector_search" if chunks_found > 0 else "fallback"
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "research_type": "specific_topic",
                    "chunks_found": chunks_found
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="TOPIC_RESEARCH_FAILED"
            )