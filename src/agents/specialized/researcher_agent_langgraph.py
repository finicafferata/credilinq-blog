"""
ResearcherAgent LangGraph Implementation - Advanced research with parallel data gathering.
"""

from typing import Dict, Any, Optional, List, TypedDict, Tuple
from enum import Enum
import asyncio
import os
# from langchain_openai import OpenAIEmbeddings  # Removed - using create_embeddings
from langchain_core.messages import SystemMessage
from src.core.llm_client import create_llm, create_embeddings
# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.langgraph_base import (
    LangGraphWorkflowBase,
    WorkflowState,
    LangGraphExecutionContext,
    CheckpointStrategy,
    WorkflowStatus
)
from ..core.base_agent import AgentResult, AgentType, AgentMetadata
from ..core.database_service import get_db_service
from ...core.security import SecurityValidator


class ResearchPhase(str, Enum):
    """Phases of the research workflow."""
    INITIALIZATION = "initialization"
    QUERY_PLANNING = "query_planning"
    PARALLEL_SEARCH = "parallel_search"
    SOURCE_VERIFICATION = "source_verification"
    CONTENT_SYNTHESIS = "content_synthesis"
    QUALITY_ASSESSMENT = "quality_assessment"


class ResearchStrategy(str, Enum):
    """Research strategies."""
    VECTOR_SEARCH = "vector_search"
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


class ResearcherState(TypedDict):
    """State for the researcher workflow."""
    # Input data
    outline: List[str]
    blog_title: str
    company_context: str
    research_depth: str  # shallow, medium, deep
    
    # Research planning
    research_queries: Dict[str, List[str]]  # section -> queries
    research_strategy: str
    parallel_batch_size: int
    
    # Research results
    raw_research: Dict[str, List[Dict[str, Any]]]  # section -> research chunks
    verified_sources: Dict[str, List[Dict[str, Any]]]
    synthesized_content: Dict[str, str]
    research_metadata: Dict[str, Any]
    
    # Quality assessment
    research_quality: Dict[str, float]
    confidence_scores: Dict[str, float]
    coverage_analysis: Dict[str, Any]
    
    # Workflow metadata
    current_phase: str
    sections_processed: int
    total_sections: int
    errors: List[str]
    warnings: List[str]


class ResearcherAgentWorkflow(LangGraphWorkflowBase[ResearcherState]):
    """
    LangGraph-based ResearcherAgent with parallel search and intelligent synthesis.
    """
    
    def __init__(
        self,
        embeddings_model: Optional[OpenAIEmbeddings] = None,
        llm: Optional[ChatOpenAI] = None,
        workflow_name: str = "researcher_agent_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        parallel_batch_size: int = 3,
        max_retries: int = 2
    ):
        """
        Initialize the ResearcherAgent workflow.
        
        Args:
            embeddings_model: Embeddings model for vector search
            llm: Language model for query planning and synthesis
            workflow_name: Name of the workflow
            checkpoint_strategy: When to save checkpoints
            parallel_batch_size: Number of parallel searches
            max_retries: Maximum retries for failed searches
        """
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.parallel_batch_size = parallel_batch_size
        self.security_validator = SecurityValidator()
        self.db_service = None
        
        # Research configuration
        self.research_config = {
            "shallow": {"queries_per_section": 1, "search_limit": 3},
            "medium": {"queries_per_section": 2, "search_limit": 5},
            "deep": {"queries_per_section": 3, "search_limit": 10}
        }
        
        # Initialize base class
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            max_retries=max_retries
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create and configure the LangGraph workflow structure."""
        workflow = StateGraph(ResearcherState)
        
        # Add nodes for each phase
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("query_planning", self.query_planning_node)
        workflow.add_node("parallel_search", self.parallel_search_node)
        workflow.add_node("source_verification", self.source_verification_node)
        workflow.add_node("content_synthesis", self.content_synthesis_node)
        workflow.add_node("quality_assessment", self.quality_assessment_node)
        
        # Define edges
        workflow.set_entry_point("initialization")
        
        workflow.add_edge("initialization", "query_planning")
        workflow.add_edge("query_planning", "parallel_search")
        workflow.add_edge("parallel_search", "source_verification")
        workflow.add_edge("source_verification", "content_synthesis")
        workflow.add_edge("content_synthesis", "quality_assessment")
        
        # Conditional routing based on quality
        workflow.add_conditional_edges(
            "quality_assessment",
            self.should_retry_research,
            {
                "retry": "query_planning",
                "complete": END
            }
        )
        
        return workflow
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> ResearcherState:
        """Create the initial state for the workflow."""
        return ResearcherState(
            # Input data
            outline=input_data.get("outline", []),
            blog_title=input_data.get("blog_title", ""),
            company_context=input_data.get("company_context", ""),
            research_depth=input_data.get("research_depth", "medium"),
            
            # Research planning - will be filled during workflow
            research_queries={},
            research_strategy="",
            parallel_batch_size=self.parallel_batch_size,
            
            # Research results - will be filled during workflow
            raw_research={},
            verified_sources={},
            synthesized_content={},
            research_metadata={},
            
            # Quality assessment - will be filled during workflow
            research_quality={},
            confidence_scores={},
            coverage_analysis={},
            
            # Workflow metadata
            current_phase=ResearchPhase.INITIALIZATION,
            sections_processed=0,
            total_sections=len(input_data.get("outline", [])),
            errors=[],
            warnings=[]
        )
    
    def initialization_node(self, state: ResearcherState) -> ResearcherState:
        """Initialize research workflow and validate inputs."""
        try:
            state["current_phase"] = ResearchPhase.INITIALIZATION
            
            # Initialize database service
            if not self.db_service:
                self.db_service = get_db_service()
            
            # Security validation
            self.security_validator.validate_content(state["blog_title"], "blog_title")
            for section in state["outline"]:
                self.security_validator.validate_content(section, "outline_section")
            
            # Initialize state fields
            state["research_depth"] = state.get("research_depth", "medium")
            state["parallel_batch_size"] = min(
                self.parallel_batch_size,
                len(state["outline"])
            )
            state["total_sections"] = len(state["outline"])
            state["sections_processed"] = 0
            state["errors"] = []
            state["warnings"] = []
            state["raw_research"] = {}
            state["verified_sources"] = {}
            state["synthesized_content"] = {}
            state["research_metadata"] = {
                "total_queries": 0,
                "successful_searches": 0,
                "failed_searches": 0,
                "total_chunks_found": 0
            }
            
            # Determine research strategy
            state["research_strategy"] = self._determine_research_strategy(state)
            
            self.logger.info(
                f"Research initialized for {state['total_sections']} sections "
                f"with {state['research_depth']} depth"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            state["errors"].append(f"Initialization error: {str(e)}")
        
        return state
    
    def query_planning_node(self, state: ResearcherState) -> ResearcherState:
        """Plan research queries for each section."""
        try:
            state["current_phase"] = ResearchPhase.QUERY_PLANNING
            
            config = self.research_config[state["research_depth"]]
            queries_per_section = config["queries_per_section"]
            
            state["research_queries"] = {}
            
            for section in state["outline"]:
                # Generate multiple queries for better coverage
                queries = self._generate_research_queries(
                    section,
                    state["blog_title"],
                    state["company_context"],
                    queries_per_section
                )
                state["research_queries"][section] = queries
                state["research_metadata"]["total_queries"] += len(queries)
            
            self.logger.info(
                f"Planned {state['research_metadata']['total_queries']} queries "
                f"for {len(state['outline'])} sections"
            )
            
        except Exception as e:
            self.logger.error(f"Query planning failed: {str(e)}")
            state["errors"].append(f"Query planning error: {str(e)}")
            # Fallback to simple queries
            state["research_queries"] = {
                section: [f"{section} {state['blog_title']}"]
                for section in state["outline"]
            }
        
        return state
    
    async def parallel_search_node(self, state: ResearcherState) -> ResearcherState:
        """Perform parallel searches for all sections."""
        try:
            state["current_phase"] = ResearchPhase.PARALLEL_SEARCH
            
            config = self.research_config[state["research_depth"]]
            search_limit = config["search_limit"]
            
            # Process sections in batches for parallel execution
            sections = list(state["research_queries"].keys())
            
            for i in range(0, len(sections), state["parallel_batch_size"]):
                batch = sections[i:i + state["parallel_batch_size"]]
                
                # Create parallel search tasks
                tasks = []
                for section in batch:
                    queries = state["research_queries"][section]
                    task = self._research_section_async(
                        section,
                        queries,
                        search_limit
                    )
                    tasks.append(task)
                
                # Execute searches in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for section, result in zip(batch, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Search failed for {section}: {str(result)}")
                        state["warnings"].append(f"Search failed for {section}")
                        state["raw_research"][section] = []
                        state["research_metadata"]["failed_searches"] += 1
                    else:
                        state["raw_research"][section] = result
                        state["research_metadata"]["successful_searches"] += 1
                        state["research_metadata"]["total_chunks_found"] += len(result)
                
                state["sections_processed"] += len(batch)
                
                self.logger.info(
                    f"Processed batch {i//state['parallel_batch_size'] + 1}: "
                    f"{state['sections_processed']}/{state['total_sections']} sections"
                )
            
        except Exception as e:
            self.logger.error(f"Parallel search failed: {str(e)}")
            state["errors"].append(f"Parallel search error: {str(e)}")
            # Fallback to empty research
            for section in state["outline"]:
                if section not in state["raw_research"]:
                    state["raw_research"][section] = []
        
        return state
    
    def source_verification_node(self, state: ResearcherState) -> ResearcherState:
        """Verify and rank research sources."""
        try:
            state["current_phase"] = ResearchPhase.SOURCE_VERIFICATION
            
            for section, research_chunks in state["raw_research"].items():
                # Verify and rank sources
                verified = self._verify_sources(
                    research_chunks,
                    section,
                    state["blog_title"]
                )
                state["verified_sources"][section] = verified
            
            self.logger.info(f"Source verification completed for {len(state['verified_sources'])} sections")
            
        except Exception as e:
            self.logger.error(f"Source verification failed: {str(e)}")
            state["errors"].append(f"Source verification error: {str(e)}")
            # Use raw research as verified
            state["verified_sources"] = state["raw_research"]
        
        return state
    
    def content_synthesis_node(self, state: ResearcherState) -> ResearcherState:
        """Synthesize research into coherent content."""
        try:
            state["current_phase"] = ResearchPhase.CONTENT_SYNTHESIS
            
            for section, verified_chunks in state["verified_sources"].items():
                # Synthesize content from verified sources
                synthesized = self._synthesize_content(
                    section,
                    verified_chunks,
                    state["blog_title"],
                    state["company_context"]
                )
                state["synthesized_content"][section] = synthesized
            
            self.logger.info(f"Content synthesis completed for {len(state['synthesized_content'])} sections")
            
        except Exception as e:
            self.logger.error(f"Content synthesis failed: {str(e)}")
            state["errors"].append(f"Content synthesis error: {str(e)}")
            # Fallback to combining chunks
            for section, chunks in state["verified_sources"].items():
                state["synthesized_content"][section] = self._fallback_synthesis(chunks)
        
        return state
    
    def quality_assessment_node(self, state: ResearcherState) -> ResearcherState:
        """Assess research quality and coverage."""
        try:
            state["current_phase"] = ResearchPhase.QUALITY_ASSESSMENT
            
            # Calculate quality metrics
            state["research_quality"] = {}
            state["confidence_scores"] = {}
            
            for section in state["outline"]:
                content = state["synthesized_content"].get(section, "")
                chunks = state["verified_sources"].get(section, [])
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(content, chunks)
                state["research_quality"][section] = quality_score
                
                # Calculate confidence
                confidence = self._calculate_confidence(content, chunks, section)
                state["confidence_scores"][section] = confidence
            
            # Overall coverage analysis
            state["coverage_analysis"] = {
                "sections_covered": len(state["synthesized_content"]),
                "total_sections": state["total_sections"],
                "coverage_percentage": (len(state["synthesized_content"]) / state["total_sections"]) * 100,
                "average_quality": sum(state["research_quality"].values()) / len(state["research_quality"]) if state["research_quality"] else 0,
                "average_confidence": sum(state["confidence_scores"].values()) / len(state["confidence_scores"]) if state["confidence_scores"] else 0,
                "high_confidence_sections": sum(1 for c in state["confidence_scores"].values() if c >= 0.8),
                "low_confidence_sections": sum(1 for c in state["confidence_scores"].values() if c < 0.5)
            }
            
            self.logger.info(
                f"Quality assessment completed. Average quality: "
                f"{state['coverage_analysis']['average_quality']:.2f}, "
                f"Average confidence: {state['coverage_analysis']['average_confidence']:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            state["errors"].append(f"Quality assessment error: {str(e)}")
            # Default quality scores
            state["research_quality"] = {section: 0.5 for section in state["outline"]}
            state["confidence_scores"] = {section: 0.5 for section in state["outline"]}
        
        return state
    
    def should_retry_research(self, state: ResearcherState) -> str:
        """Determine whether to retry research based on quality."""
        if state.get("errors"):
            return "complete"
        
        # Check if quality is too low
        avg_quality = state.get("coverage_analysis", {}).get("average_quality", 0)
        avg_confidence = state.get("coverage_analysis", {}).get("average_confidence", 0)
        
        # Don't retry if we've already tried multiple times
        retry_count = state.get("retry_count", 0)
        if retry_count >= self.max_retries:
            return "complete"
        
        # Retry if quality is very low
        if avg_quality < 0.3 or avg_confidence < 0.3:
            state["retry_count"] = retry_count + 1
            self.logger.info(f"Retrying research (attempt {state['retry_count']})")
            return "retry"
        
        return "complete"
    
    def _determine_research_strategy(self, state: ResearcherState) -> str:
        """Determine the best research strategy."""
        # Check available resources
        has_embeddings = self.embeddings_model is not None
        has_db = self.db_service is not None
        has_llm = self.llm is not None
        
        if has_embeddings and has_db:
            return ResearchStrategy.VECTOR_SEARCH
        elif has_llm:
            return ResearchStrategy.WEB_SEARCH
        else:
            return ResearchStrategy.FALLBACK
    
    def _generate_research_queries(
        self,
        section: str,
        blog_title: str,
        company_context: str,
        num_queries: int
    ) -> List[str]:
        """Generate multiple research queries for better coverage."""
        queries = []
        
        # Base query
        base_query = f"{section} in the context of {blog_title}"
        queries.append(base_query)
        
        if num_queries > 1 and self.llm:
            try:
                # Use LLM to generate additional queries
                prompt = f"""Generate {num_queries - 1} different search queries for researching:
                Section: {section}
                Blog Title: {blog_title}
                Context: {company_context}
                
                Return only the queries, one per line."""
                
                response = self.llm.invoke([SystemMessage(content=prompt)])
                additional_queries = response.content.strip().split('\n')
                queries.extend(additional_queries[:num_queries - 1])
            except:
                # Fallback to variations
                queries.append(f"statistics and data about {section}")
                if num_queries > 2:
                    queries.append(f"examples and case studies for {section}")
        else:
            # Manual variations
            if num_queries > 1:
                queries.append(f"data and statistics for {section}")
            if num_queries > 2:
                queries.append(f"{company_context} approach to {section}")
        
        return queries[:num_queries]
    
    async def _research_section_async(
        self,
        section: str,
        queries: List[str],
        search_limit: int
    ) -> List[Dict[str, Any]]:
        """Asynchronously research a section with multiple queries."""
        all_results = []
        
        for query in queries:
            try:
                # Perform vector search
                if self.embeddings_model and self.db_service:
                    results = await self._async_vector_search(query, search_limit)
                    all_results.extend(results)
                else:
                    # Fallback research
                    fallback = self._generate_fallback_content(section, query)
                    all_results.append({
                        "content": fallback,
                        "source": "fallback",
                        "query": query,
                        "score": 0.5
                    })
            except Exception as e:
                self.logger.warning(f"Query failed for '{query}': {str(e)}")
                continue
        
        return all_results
    
    async def _async_vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform asynchronous vector search."""
        try:
            # Generate embedding
            embedding = self.embeddings_model.embed_query(query)
            
            # Security validation
            self.security_validator.validate_vector_embedding(embedding)
            
            # Perform search
            results = self.db_service.vector_search(query, limit=limit)
            
            return [
                {
                    "content": r.get("content", ""),
                    "source": "vector_search",
                    "query": query,
                    "score": r.get("similarity", 0.5),
                    "metadata": r.get("metadata", {})
                }
                for r in results
            ]
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def _verify_sources(
        self,
        chunks: List[Dict[str, Any]],
        section: str,
        blog_title: str
    ) -> List[Dict[str, Any]]:
        """Verify and rank sources by relevance."""
        if not chunks:
            return []
        
        # Sort by score and relevance
        verified = sorted(
            chunks,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        # Filter out low-quality sources
        verified = [
            chunk for chunk in verified
            if chunk.get("score", 0) > 0.3
        ]
        
        # Limit to top sources
        return verified[:10]
    
    def _synthesize_content(
        self,
        section: str,
        chunks: List[Dict[str, Any]],
        blog_title: str,
        company_context: str
    ) -> str:
        """Synthesize research chunks into coherent content."""
        if not chunks:
            return self._generate_fallback_content(section, blog_title)
        
        if self.llm:
            try:
                # Combine chunk contents
                combined = "\n\n".join([
                    chunk.get("content", "") for chunk in chunks[:5]
                ])
                
                prompt = f"""Synthesize the following research into a coherent summary for the section "{section}" 
                in a blog about "{blog_title}":
                
                Research:
                {combined[:3000]}  # Truncate for token limits
                
                Company Context: {company_context}
                
                Create a comprehensive summary that includes key points, statistics, and examples."""
                
                response = self.llm.invoke([SystemMessage(content=prompt)])
                return response.content.strip()
                
            except Exception as e:
                self.logger.warning(f"LLM synthesis failed: {str(e)}")
        
        # Fallback to combining chunks
        return self._fallback_synthesis(chunks)
    
    def _fallback_synthesis(self, chunks: List[Dict[str, Any]]) -> str:
        """Fallback method to combine research chunks."""
        if not chunks:
            return ""
        
        # Combine top chunks
        contents = [chunk.get("content", "") for chunk in chunks[:3]]
        return "\n\n".join(contents)
    
    def _generate_fallback_content(self, section: str, query: str) -> str:
        """Generate fallback content when research fails."""
        return f"""Research Framework for: {section}
        
        KEY AREAS TO EXPLORE:
        - Industry statistics and benchmarks
        - Real-world examples and case studies
        - Expert insights and quotes
        - Customer testimonials and outcomes
        - Comparison data and alternatives
        - Visual elements and data representations
        - Company-specific applications
        - Future trends and implications
        
        Note: Specific data points should be researched and validated."""
    
    def _calculate_quality_score(self, content: str, chunks: List[Dict[str, Any]]) -> float:
        """Calculate quality score for researched content."""
        if not content:
            return 0.0
        
        # Content length score
        word_count = len(content.split())
        length_score = min(word_count / 200, 1.0)  # Ideal: 200+ words
        
        # Source diversity score
        sources = len(set(chunk.get("source", "") for chunk in chunks))
        diversity_score = min(sources / 3, 1.0)  # Ideal: 3+ sources
        
        # Average chunk score
        avg_score = sum(chunk.get("score", 0) for chunk in chunks) / len(chunks) if chunks else 0
        
        # Combined quality score
        quality_score = (length_score * 0.3 + diversity_score * 0.3 + avg_score * 0.4)
        
        return quality_score
    
    def _calculate_confidence(self, content: str, chunks: List[Dict[str, Any]], section: str) -> float:
        """Calculate confidence score for research results."""
        if not content:
            return 0.0
        
        # Base confidence on content availability
        has_content = 1.0 if len(content) > 100 else 0.5
        
        # Source quality
        has_vector_search = any(c.get("source") == "vector_search" for c in chunks)
        source_confidence = 1.0 if has_vector_search else 0.5
        
        # Chunk count confidence
        chunk_confidence = min(len(chunks) / 5, 1.0)
        
        # Combined confidence
        confidence = (has_content * 0.4 + source_confidence * 0.4 + chunk_confidence * 0.2)
        
        return confidence
    


# Adapter for backward compatibility
class ResearcherAgentLangGraph:
    """Adapter to make LangGraph workflow compatible with existing ResearcherAgent interface."""
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize the adapter."""
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.RESEARCHER,
                name="ResearcherAgentLangGraph",
                description="LangGraph-powered researcher with parallel search and intelligent synthesis",
                capabilities=[
                    "parallel_search",
                    "source_verification",
                    "content_synthesis",
                    "quality_assessment",
                    "multi_query_research"
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
            
            embeddings = create_embeddings(
                api_key=settings.primary_api_key
            )
            
            llm = create_llm(
                model="gemini-1.5-flash",
                temperature=0.5,
                api_key=settings.primary_api_key
            )
            
            self.workflow = ResearcherAgentWorkflow(
                embeddings_model=embeddings,
                llm=llm
            )
            
        except Exception as e:
            # Fallback without embeddings/LLM
            self.workflow = ResearcherAgentWorkflow(
                embeddings_model=None,
                llm=None
            )
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Any] = None
    ) -> AgentResult:
        """Execute the researcher workflow."""
        try:
            # Execute workflow using the base class method
            result = await self.workflow.execute(
                input_data,
                LangGraphExecutionContext(
                    session_id=context.session_id if context else "default",
                    user_id=context.user_id if context else None
                )
            )
            
            return result
            
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="RESEARCHER_WORKFLOW_FAILED"
            )