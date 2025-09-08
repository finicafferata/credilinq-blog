"""
Parallel Research Orchestrator for LangGraph Agent System

This module implements User Story 2.1: Parallel Research Phase
- Executes ResearcherAgent and SearchAgent in parallel using asyncio.gather()
- Provides result merging logic for parallel outputs
- Implements conflict resolution for overlapping information
- Maintains research quality metrics
- Targets 40% reduction in research phase execution time

Key Features:
- Parallel execution with fault tolerance
- Intelligent result aggregation and deduplication
- Quality-aware conflict resolution
- Performance monitoring and validation
- Full LangGraph integration with state management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Agent imports
from ..specialized.researcher_agent_langgraph import ResearcherAgentLangGraph, ResearcherState
from ..specialized.search_agent_langgraph import SearchAgentWorkflow, SearchAgentState
from ..core.base_agent import AgentResult, AgentExecutionContext

# Performance tracking
try:
    from ...core.langgraph_performance_tracker import global_performance_tracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResearchPriority(Enum):
    """Research priority levels for conflict resolution."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicting research information."""
    SOURCE_CREDIBILITY = "source_credibility"  # Prioritize higher credibility sources
    RECENCY = "recency"  # Prioritize more recent information
    CONSENSUS = "consensus"  # Use majority consensus
    EXPERT_WEIGHTED = "expert_weighted"  # Weight by source expertise
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class ResearchSource:
    """Enhanced research source with conflict resolution metadata."""
    source_id: str
    title: str
    url: str
    content: str
    credibility_score: float
    relevance_score: float
    publication_date: Optional[datetime] = None
    source_type: str = "web"
    agent_origin: str = ""  # Which agent found this source
    key_topics: List[str] = field(default_factory=list)
    content_hash: str = ""
    
    def __post_init__(self):
        """Generate content hash for deduplication."""
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                f"{self.title}{self.url}{self.content[:500]}".encode()
            ).hexdigest()


@dataclass 
class ResearchInsight:
    """Research insight with conflict resolution metadata."""
    insight_id: str
    text: str
    confidence_level: float
    supporting_sources: List[str]
    conflicting_sources: List[str] = field(default_factory=list)
    topic_category: str = ""
    agent_origin: str = ""
    priority: ResearchPriority = ResearchPriority.MEDIUM
    validation_score: float = 0.0


@dataclass
class ConflictingInformation:
    """Represents conflicting research information."""
    topic: str
    primary_claim: str
    alternative_claims: List[str]
    source_distribution: Dict[str, int]  # claim -> num_sources
    resolution_strategy: ConflictResolutionStrategy
    resolved_claim: str = ""
    confidence_level: float = 0.0


class ParallelResearchState(TypedDict):
    """Enhanced state for parallel research workflow."""
    # Input parameters
    workflow_id: str
    research_topics: List[str]
    target_audience: str
    research_depth: str
    max_sources_per_agent: int
    time_range: Optional[str]
    
    # Parallel execution tracking
    researcher_task_id: Optional[str]
    search_agent_task_id: Optional[str]
    parallel_start_time: Optional[datetime]
    parallel_completion_time: Optional[datetime]
    
    # Individual agent results
    researcher_results: Dict[str, Any]
    search_agent_results: Dict[str, Any]
    researcher_execution_time: float
    search_agent_execution_time: float
    
    # Aggregated results
    merged_sources: List[ResearchSource]
    deduplicated_sources: List[ResearchSource]
    aggregated_insights: List[ResearchInsight]
    conflicting_information: List[ConflictingInformation]
    
    # Quality metrics
    source_quality_distribution: Dict[str, int]
    insight_confidence_avg: float
    research_completeness_score: float
    conflict_resolution_rate: float
    
    # Performance metrics
    total_execution_time: float
    parallel_efficiency_gain: float
    time_reduction_percentage: float
    quality_preservation_score: float
    
    # Error handling
    researcher_errors: List[str]
    search_agent_errors: List[str]
    aggregation_errors: List[str]
    
    # Workflow status
    status: str
    created_at: datetime
    updated_at: datetime
    completion_percentage: float
    
    # Performance tracking
    workflow_execution_id: Optional[str]


class ParallelResearchOrchestrator:
    """Orchestrator for parallel research execution with advanced conflict resolution."""
    
    def __init__(self, checkpoint_strategy: str = "memory"):
        """Initialize the parallel research orchestrator."""
        self.researcher_agent = ResearcherAgentLangGraph()
        self.search_agent = SearchAgentWorkflow()
        
        # Checkpoint configuration
        if checkpoint_strategy == "memory":
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None
            
        # Conflict resolution configuration
        self.default_resolution_strategy = ConflictResolutionStrategy.HYBRID
        self.quality_threshold = 0.7
        self.confidence_threshold = 0.6
        
        # Performance targets
        self.target_time_reduction = 0.40  # 40% reduction target
        self.quality_preservation_target = 0.95  # Maintain 95% quality
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for parallel research execution."""
        workflow = StateGraph(ParallelResearchState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self.initialize_parallel_research)
        workflow.add_node("execute_parallel_research", self.execute_parallel_research)
        workflow.add_node("aggregate_results", self.aggregate_research_results)
        workflow.add_node("resolve_conflicts", self.resolve_conflicting_information)
        workflow.add_node("validate_quality", self.validate_research_quality)
        workflow.add_node("finalize_results", self.finalize_parallel_research)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "execute_parallel_research")
        workflow.add_edge("execute_parallel_research", "aggregate_results")
        workflow.add_edge("aggregate_results", "resolve_conflicts")
        
        # Add conditional edge for quality validation
        workflow.add_conditional_edges(
            "resolve_conflicts",
            self.check_quality_gate,
            {
                "quality_sufficient": "validate_quality",
                "needs_additional_research": "execute_parallel_research",
                "quality_failed": END
            }
        )
        
        workflow.add_edge("validate_quality", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        # Compile with checkpointer if available
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()
    
    async def initialize_parallel_research(self, state: ParallelResearchState) -> ParallelResearchState:
        """Initialize the parallel research workflow."""
        logger.info("Initializing parallel research workflow")
        
        # Ensure state consistency at the start
        state = self.ensure_state_consistency(state, "initialize_parallel_research")
        
        # Set workflow metadata
        state['workflow_id'] = state.get('workflow_id', f"parallel-research-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        state['status'] = 'initializing'
        state['created_at'] = datetime.now()
        state['updated_at'] = datetime.now()
        state['completion_percentage'] = 10.0
        
        # Initialize collections
        state['researcher_results'] = {}
        state['search_agent_results'] = {}
        state['merged_sources'] = []
        state['deduplicated_sources'] = []
        state['aggregated_insights'] = []
        state['conflicting_information'] = []
        state['researcher_errors'] = []
        state['search_agent_errors'] = []
        state['aggregation_errors'] = []
        
        # Initialize performance metrics
        state['researcher_execution_time'] = 0.0
        state['search_agent_execution_time'] = 0.0
        state['total_execution_time'] = 0.0
        state['parallel_efficiency_gain'] = 0.0
        state['time_reduction_percentage'] = 0.0
        state['quality_preservation_score'] = 0.0
        
        # Initialize quality metrics (previously missing)
        state['source_quality_distribution'] = {
            'very_high': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'unknown': 0
        }
        state['insight_confidence_avg'] = 0.0
        state['research_completeness_score'] = 0.0
        state['conflict_resolution_rate'] = 1.0  # Start at 100%, will be calculated later
        
        # Initialize parallel execution tracking (previously missing)
        state['researcher_task_id'] = None
        state['search_agent_task_id'] = None
        state['parallel_start_time'] = None
        state['parallel_completion_time'] = None
        
        # Track workflow start (using execution tracking)
        if TRACKING_AVAILABLE:
            try:
                workflow_execution_id = await global_performance_tracker.track_execution_start(
                    agent_name="ParallelResearchOrchestrator",
                    agent_type="workflow",
                    metadata={
                        "workflow_id": state['workflow_id'],
                        "workflow_type": "parallel_research",
                        "topics": state['research_topics'],
                        "target_reduction": self.target_time_reduction,
                        "topic_count": len(state['research_topics']),
                        "target_audience": state['target_audience'],
                        "research_depth": state['research_depth']
                    }
                )
                # Store execution ID in state for later use
                state['workflow_execution_id'] = workflow_execution_id
                logger.info(f"Started performance tracking for workflow: {workflow_execution_id}")
            except Exception as e:
                logger.error(f"Failed to start workflow performance tracking: {e}")
                # Continue without breaking the workflow
                state['workflow_execution_id'] = None
        
        # Validate state after initialization
        validation_issues = self.validate_state_variables(state, "initialize_parallel_research")
        if validation_issues:
            logger.error(f"State validation failed after initialization: {validation_issues}")
        
        logger.info(f"Parallel research workflow initialized: {state['workflow_id']}")
        return state
    
    async def execute_parallel_research(self, state: ParallelResearchState) -> ParallelResearchState:
        """Execute ResearcherAgent and SearchAgent in parallel."""
        logger.info("Starting parallel research execution")
        
        state['status'] = 'executing_research'
        state['parallel_start_time'] = datetime.now()
        state['completion_percentage'] = 25.0
        
        try:
            # Prepare inputs for both agents
            researcher_input = self._prepare_researcher_input(state)
            search_agent_input = self._prepare_search_agent_input(state)
            
            # Execute both agents in parallel using asyncio.gather()
            logger.info("Launching parallel research tasks")
            
            results = await asyncio.gather(
                self._execute_researcher_with_timeout(researcher_input, state),
                self._execute_search_agent_with_timeout(search_agent_input, state),
                return_exceptions=True
            )
            
            state['parallel_completion_time'] = datetime.now()
            execution_time = (state['parallel_completion_time'] - state['parallel_start_time']).total_seconds()
            
            # Process results
            researcher_result, search_agent_result = results
            
            # Handle ResearcherAgent result
            if isinstance(researcher_result, Exception):
                error_msg = f"ResearcherAgent failed: {str(researcher_result)}"
                state['researcher_errors'].append(error_msg)
                logger.error(error_msg)
            elif researcher_result and hasattr(researcher_result, 'success') and researcher_result.success:
                state['researcher_results'] = researcher_result.data
                state['researcher_execution_time'] = researcher_result.execution_time_ms / 1000.0
                logger.info(f"ResearcherAgent completed in {state['researcher_execution_time']:.2f}s")
            else:
                error_msg = f"ResearcherAgent returned invalid result: {researcher_result}"
                state['researcher_errors'].append(error_msg)
                logger.error(error_msg)
            
            # Handle SearchAgent result
            if isinstance(search_agent_result, Exception):
                error_msg = f"SearchAgent failed: {str(search_agent_result)}"
                state['search_agent_errors'].append(error_msg)
                logger.error(error_msg)
            elif search_agent_result and hasattr(search_agent_result, 'success') and search_agent_result.success:
                state['search_agent_results'] = search_agent_result.data
                state['search_agent_execution_time'] = search_agent_result.execution_time_ms / 1000.0
                logger.info(f"SearchAgent completed in {state['search_agent_execution_time']:.2f}s")
            else:
                error_msg = f"SearchAgent returned invalid result: {search_agent_result}"
                state['search_agent_errors'].append(error_msg)
                logger.error(error_msg)
            
            # Calculate performance metrics
            state['total_execution_time'] = max(state['researcher_execution_time'], state['search_agent_execution_time'])
            
            # Estimate sequential time (sum of individual execution times)
            sequential_time = state['researcher_execution_time'] + state['search_agent_execution_time']
            if sequential_time > 0:
                state['time_reduction_percentage'] = ((sequential_time - state['total_execution_time']) / sequential_time) * 100
                state['parallel_efficiency_gain'] = sequential_time / state['total_execution_time'] if state['total_execution_time'] > 0 else 1.0
            
            logger.info(f"Parallel execution completed in {state['total_execution_time']:.2f}s")
            logger.info(f"Time reduction achieved: {state['time_reduction_percentage']:.1f}%")
            logger.info(f"Parallel efficiency gain: {state['parallel_efficiency_gain']:.2f}x")
            
            state['completion_percentage'] = 50.0
            
        except Exception as e:
            error_msg = f"Parallel research execution failed: {str(e)}"
            state['aggregation_errors'].append(error_msg)
            logger.error(error_msg)
            state['status'] = 'failed'
        
        state['updated_at'] = datetime.now()
        return state
    
    async def _execute_researcher_with_timeout(
        self, 
        input_data: Dict[str, Any], 
        state: ParallelResearchState,
        timeout: int = 300
    ) -> AgentResult:
        """Execute ResearcherAgent with timeout protection."""
        try:
            # Create execution context (using only supported parameters)
            context = AgentExecutionContext(
                workflow_id=state['workflow_id'],
                request_id=f"researcher-{state['workflow_id']}",
                execution_metadata={
                    "task_type": "research",
                    "priority": "high", 
                    "agent_role": "researcher",
                    "parallel_execution": True
                }
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.researcher_agent.run_workflow(input_data),
                timeout=timeout
            )
            
            # Convert workflow state to AgentResult format
            if hasattr(result, 'research_completeness') and result.research_completeness > 0:
                agent_result = AgentResult(
                    success=True,
                    data={
                        'research_findings': result.research_findings,
                        'key_insights': result.key_insights,
                        'validated_sources': result.validated_sources,
                        'source_quality_scores': result.source_quality_scores,
                        'research_completeness': result.research_completeness
                    },
                    execution_time_ms=state.get('researcher_execution_time', 0) * 1000,
                    metadata={
                        'agent_type': 'researcher',
                        'workflow_id': state['workflow_id']
                    }
                )
            else:
                agent_result = AgentResult(
                    success=False,
                    error_message="ResearcherAgent did not complete successfully",
                    metadata={'agent_type': 'researcher'}
                )
            
            return agent_result
            
        except asyncio.TimeoutError:
            logger.error(f"ResearcherAgent timed out after {timeout} seconds")
            return AgentResult(
                success=False,
                error_message=f"ResearcherAgent execution timed out ({timeout}s)",
                metadata={'agent_type': 'researcher', 'timeout': timeout}
            )
        except Exception as e:
            logger.error(f"ResearcherAgent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=f"ResearcherAgent execution failed: {str(e)}",
                metadata={'agent_type': 'researcher', 'error': str(e)}
            )
    
    async def _execute_search_agent_with_timeout(
        self, 
        input_data: Dict[str, Any], 
        state: ParallelResearchState,
        timeout: int = 300
    ) -> AgentResult:
        """Execute SearchAgent with timeout protection."""
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.search_agent.execute_workflow(**input_data),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"SearchAgent timed out after {timeout} seconds")
            return AgentResult(
                success=False,
                error_message=f"SearchAgent execution timed out ({timeout}s)",
                metadata={'agent_type': 'search_agent', 'timeout': timeout}
            )
        except Exception as e:
            logger.error(f"SearchAgent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=f"SearchAgent execution failed: {str(e)}",
                metadata={'agent_type': 'search_agent', 'error': str(e)}
            )
    
    def _prepare_researcher_input(self, state: ParallelResearchState) -> Dict[str, Any]:
        """Prepare input for ResearcherAgent."""
        return {
            'research_topic': ' '.join(state['research_topics']),
            'research_depth': state.get('research_depth', 'comprehensive'),
            'target_audience': state.get('target_audience', 'business_professionals'),
            'specific_questions': []  # Could be enhanced with specific questions
        }
    
    def _prepare_search_agent_input(self, state: ParallelResearchState) -> Dict[str, Any]:
        """Prepare input for SearchAgent."""
        return {
            'research_query': ' '.join(state['research_topics']),
            'research_type': 'general_research',
            'research_scope': state.get('research_depth', 'comprehensive'),
            'max_sources': state.get('max_sources_per_agent', 10),
            'time_range': state.get('time_range'),
            'targeted_domains': [],
            'excluded_domains': []
        }
    
    async def aggregate_research_results(self, state: ParallelResearchState) -> ParallelResearchState:
        """Aggregate results from parallel research execution."""
        logger.info("Aggregating research results from parallel execution")
        
        state['status'] = 'aggregating_results'
        state['completion_percentage'] = 60.0
        
        try:
            # Extract and merge sources from both agents
            researcher_sources = self._extract_researcher_sources(state['researcher_results'])
            search_agent_sources = self._extract_search_agent_sources(state['search_agent_results'])
            
            # Merge all sources
            all_sources = researcher_sources + search_agent_sources
            state['merged_sources'] = all_sources
            
            logger.info(f"Merged {len(all_sources)} sources from parallel execution")
            
            # Deduplicate sources using advanced algorithms
            deduplicated_sources = await self._deduplicate_sources_advanced(all_sources)
            state['deduplicated_sources'] = deduplicated_sources
            
            logger.info(f"After deduplication: {len(deduplicated_sources)} unique sources")
            
            # Extract and merge insights
            researcher_insights = self._extract_researcher_insights(state['researcher_results'])
            search_agent_insights = self._extract_search_agent_insights(state['search_agent_results'])
            
            all_insights = researcher_insights + search_agent_insights
            state['aggregated_insights'] = all_insights
            
            logger.info(f"Aggregated {len(all_insights)} insights from parallel execution")
            
            # Calculate quality distribution
            state['source_quality_distribution'] = self._calculate_quality_distribution(deduplicated_sources)
            
            # Calculate average insight confidence
            if all_insights:
                state['insight_confidence_avg'] = sum(insight.confidence_level for insight in all_insights) / len(all_insights)
            else:
                state['insight_confidence_avg'] = 0.0
            
        except Exception as e:
            error_msg = f"Result aggregation failed: {str(e)}"
            state['aggregation_errors'].append(error_msg)
            logger.error(error_msg)
        
        state['updated_at'] = datetime.now()
        return state
    
    def _extract_researcher_sources(self, researcher_data: Dict[str, Any]) -> List[ResearchSource]:
        """Extract sources from ResearcherAgent results."""
        sources = []
        validated_sources = researcher_data.get('validated_sources', [])
        
        for i, source in enumerate(validated_sources):
            research_source = ResearchSource(
                source_id=f"researcher_{i}",
                title=source.get('title', ''),
                url=source.get('url', ''),
                content=source.get('snippet', ''),
                credibility_score=source.get('validation_score', 0.8),
                relevance_score=source.get('relevance_score', 0.8),
                source_type=source.get('source_type', 'academic'),
                agent_origin='researcher'
            )
            sources.append(research_source)
        
        return sources
    
    def _extract_search_agent_sources(self, search_data: Dict[str, Any]) -> List[ResearchSource]:
        """Extract sources from SearchAgent results."""
        sources = []
        verified_sources = search_data.get('verified_sources', [])
        
        for i, source in enumerate(verified_sources):
            research_source = ResearchSource(
                source_id=f"search_agent_{i}",
                title=source.get('title', ''),
                url=source.get('url', ''),
                content=source.get('content', ''),
                credibility_score=source.get('credibility_score', 0.7),
                relevance_score=source.get('relevance_score', 0.7),
                source_type=source.get('source_type', 'web'),
                agent_origin='search_agent',
                key_topics=source.get('key_topics', [])
            )
            sources.append(research_source)
        
        return sources
    
    def _extract_researcher_insights(self, researcher_data: Dict[str, Any]) -> List[ResearchInsight]:
        """Extract insights from ResearcherAgent results."""
        insights = []
        key_insights = researcher_data.get('key_insights', [])
        
        for i, insight_text in enumerate(key_insights):
            insight = ResearchInsight(
                insight_id=f"researcher_insight_{i}",
                text=insight_text,
                confidence_level=0.8,  # Default confidence for researcher insights
                supporting_sources=[],  # Could be enhanced with source mapping
                agent_origin='researcher',
                priority=ResearchPriority.HIGH
            )
            insights.append(insight)
        
        return insights
    
    def _extract_search_agent_insights(self, search_data: Dict[str, Any]) -> List[ResearchInsight]:
        """Extract insights from SearchAgent results."""
        insights = []
        research_insights = search_data.get('research_insights', [])
        
        for i, insight_data in enumerate(research_insights):
            insight = ResearchInsight(
                insight_id=f"search_agent_insight_{i}",
                text=insight_data.get('insight_text', ''),
                confidence_level=insight_data.get('confidence_level', 0.7),
                supporting_sources=insight_data.get('supporting_sources', []),
                agent_origin='search_agent',
                topic_category=insight_data.get('insight_type', 'finding')
            )
            insights.append(insight)
        
        return insights
    
    async def _deduplicate_sources_advanced(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """Advanced source deduplication using multiple techniques."""
        if not sources:
            return []
        
        # Group sources by URL (exact matches)
        url_groups = defaultdict(list)
        for source in sources:
            url_groups[source.url].append(source)
        
        deduplicated = []
        
        for url, source_group in url_groups.items():
            if len(source_group) == 1:
                # No duplicates for this URL
                deduplicated.append(source_group[0])
            else:
                # Multiple sources with same URL - merge them
                merged_source = await self._merge_duplicate_sources(source_group)
                deduplicated.append(merged_source)
        
        # Additional deduplication based on content similarity
        final_deduplicated = await self._deduplicate_by_content_similarity(deduplicated)
        
        return final_deduplicated
    
    async def _merge_duplicate_sources(self, duplicate_sources: List[ResearchSource]) -> ResearchSource:
        """Merge duplicate sources intelligently."""
        # Sort by credibility score (highest first)
        sorted_sources = sorted(duplicate_sources, key=lambda s: s.credibility_score, reverse=True)
        primary_source = sorted_sources[0]
        
        # Combine content from all sources
        combined_content = " ".join([s.content for s in sorted_sources if s.content])
        
        # Take the highest credibility and relevance scores
        max_credibility = max(s.credibility_score for s in sorted_sources)
        max_relevance = max(s.relevance_score for s in sorted_sources)
        
        # Combine key topics
        all_topics = set()
        for source in sorted_sources:
            all_topics.update(source.key_topics)
        
        # Create merged source
        merged_source = ResearchSource(
            source_id=f"merged_{primary_source.source_id}",
            title=primary_source.title,
            url=primary_source.url,
            content=combined_content[:2000],  # Limit content length
            credibility_score=max_credibility,
            relevance_score=max_relevance,
            source_type=primary_source.source_type,
            agent_origin=f"merged_{'+'.join(set(s.agent_origin for s in sorted_sources))}",
            key_topics=list(all_topics)
        )
        
        return merged_source
    
    async def _deduplicate_by_content_similarity(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """Deduplicate sources based on content similarity."""
        if len(sources) <= 1:
            return sources
        
        # Simple similarity check using content hashes
        unique_sources = []
        seen_hashes = set()
        
        for source in sources:
            if source.content_hash not in seen_hashes:
                unique_sources.append(source)
                seen_hashes.add(source.content_hash)
        
        return unique_sources
    
    def _calculate_quality_distribution(self, sources: List[ResearchSource]) -> Dict[str, int]:
        """Calculate distribution of source quality levels."""
        distribution = {
            'very_high': 0,    # >= 0.9
            'high': 0,         # >= 0.8
            'medium': 0,       # >= 0.7
            'low': 0           # < 0.7
        }
        
        for source in sources:
            score = source.credibility_score
            if score >= 0.9:
                distribution['very_high'] += 1
            elif score >= 0.8:
                distribution['high'] += 1
            elif score >= 0.7:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    async def resolve_conflicting_information(self, state: ParallelResearchState) -> ParallelResearchState:
        """Resolve conflicts in research information using multiple strategies."""
        logger.info("Resolving conflicting information from parallel research")
        
        state['status'] = 'resolving_conflicts'
        state['completion_percentage'] = 75.0
        
        try:
            # Identify conflicts
            conflicts = await self._identify_conflicts(state['aggregated_insights'])
            
            # Resolve each conflict using appropriate strategy
            resolved_conflicts = []
            for conflict in conflicts:
                resolved_conflict = await self._resolve_single_conflict(
                    conflict, 
                    state['deduplicated_sources'],
                    self.default_resolution_strategy
                )
                resolved_conflicts.append(resolved_conflict)
            
            state['conflicting_information'] = resolved_conflicts
            
            # Calculate conflict resolution rate
            total_conflicts = len(conflicts)
            resolved_count = sum(1 for c in resolved_conflicts if c.resolved_claim)
            state['conflict_resolution_rate'] = resolved_count / total_conflicts if total_conflicts > 0 else 1.0
            
            logger.info(f"Resolved {resolved_count}/{total_conflicts} conflicts ({state['conflict_resolution_rate']:.1%})")
            
        except Exception as e:
            error_msg = f"Conflict resolution failed: {str(e)}"
            state['aggregation_errors'].append(error_msg)
            logger.error(error_msg)
        
        state['updated_at'] = datetime.now()
        return state
    
    async def _identify_conflicts(self, insights: List[ResearchInsight]) -> List[ConflictingInformation]:
        """Identify conflicting information in research insights."""
        conflicts = []
        
        # Group insights by topic category
        topic_groups = defaultdict(list)
        for insight in insights:
            topic = insight.topic_category or "general"
            topic_groups[topic].append(insight)
        
        # Look for conflicts within each topic group
        for topic, topic_insights in topic_groups.items():
            if len(topic_insights) > 1:
                # Simple conflict detection based on contradictory keywords
                conflict_keywords = [
                    ("increase", "decrease"),
                    ("improve", "worsen"),
                    ("positive", "negative"),
                    ("success", "failure"),
                    ("effective", "ineffective")
                ]
                
                for keyword_pair in conflict_keywords:
                    positive_claims = [i for i in topic_insights if keyword_pair[0] in i.text.lower()]
                    negative_claims = [i for i in topic_insights if keyword_pair[1] in i.text.lower()]
                    
                    if positive_claims and negative_claims:
                        conflict = ConflictingInformation(
                            topic=topic,
                            primary_claim=positive_claims[0].text,
                            alternative_claims=[c.text for c in negative_claims],
                            source_distribution={
                                "positive": len(positive_claims),
                                "negative": len(negative_claims)
                            },
                            resolution_strategy=self.default_resolution_strategy
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    async def _resolve_single_conflict(
        self, 
        conflict: ConflictingInformation,
        sources: List[ResearchSource],
        strategy: ConflictResolutionStrategy
    ) -> ConflictingInformation:
        """Resolve a single conflict using the specified strategy."""
        
        if strategy == ConflictResolutionStrategy.SOURCE_CREDIBILITY:
            resolved_claim, confidence = self._resolve_by_credibility(conflict, sources)
        elif strategy == ConflictResolutionStrategy.RECENCY:
            resolved_claim, confidence = self._resolve_by_recency(conflict, sources)
        elif strategy == ConflictResolutionStrategy.CONSENSUS:
            resolved_claim, confidence = self._resolve_by_consensus(conflict)
        elif strategy == ConflictResolutionStrategy.HYBRID:
            resolved_claim, confidence = self._resolve_by_hybrid_approach(conflict, sources)
        else:
            resolved_claim, confidence = self._resolve_by_consensus(conflict)
        
        conflict.resolved_claim = resolved_claim
        conflict.confidence_level = confidence
        
        return conflict
    
    def _resolve_by_credibility(self, conflict: ConflictingInformation, sources: List[ResearchSource]) -> Tuple[str, float]:
        """Resolve conflict by prioritizing higher credibility sources."""
        # Simple implementation - would be enhanced with actual source mapping
        primary_credibility = 0.8  # Simulated
        alternative_credibility = 0.6  # Simulated
        
        if primary_credibility > alternative_credibility:
            return conflict.primary_claim, primary_credibility
        else:
            return conflict.alternative_claims[0] if conflict.alternative_claims else conflict.primary_claim, alternative_credibility
    
    def _resolve_by_recency(self, conflict: ConflictingInformation, sources: List[ResearchSource]) -> Tuple[str, float]:
        """Resolve conflict by prioritizing more recent information."""
        # Simplified recency resolution
        return conflict.primary_claim, 0.75
    
    def _resolve_by_consensus(self, conflict: ConflictingInformation) -> Tuple[str, float]:
        """Resolve conflict by majority consensus."""
        positive_count = conflict.source_distribution.get("positive", 0)
        negative_count = conflict.source_distribution.get("negative", 0)
        
        if positive_count > negative_count:
            confidence = positive_count / (positive_count + negative_count)
            return conflict.primary_claim, confidence
        else:
            confidence = negative_count / (positive_count + negative_count)
            return conflict.alternative_claims[0] if conflict.alternative_claims else conflict.primary_claim, confidence
    
    def _resolve_by_hybrid_approach(self, conflict: ConflictingInformation, sources: List[ResearchSource]) -> Tuple[str, float]:
        """Resolve conflict using a hybrid of multiple strategies."""
        # Combine credibility and consensus
        credibility_result, credibility_confidence = self._resolve_by_credibility(conflict, sources)
        consensus_result, consensus_confidence = self._resolve_by_consensus(conflict)
        
        # Weight the results (70% credibility, 30% consensus)
        if credibility_result == consensus_result:
            final_confidence = (credibility_confidence * 0.7) + (consensus_confidence * 0.3)
            return credibility_result, final_confidence
        else:
            # Credibility wins in case of disagreement
            return credibility_result, credibility_confidence * 0.8
    
    async def validate_research_quality(self, state: ParallelResearchState) -> ParallelResearchState:
        """Validate the quality of parallel research results."""
        logger.info("Validating research quality")
        
        state['status'] = 'validating_quality'
        state['completion_percentage'] = 85.0
        
        try:
            # Calculate research completeness score
            completeness_factors = []
            
            # Source diversity factor
            if state['deduplicated_sources']:
                source_types = set(source.source_type for source in state['deduplicated_sources'])
                source_diversity = len(source_types) / 4.0  # Assuming 4 max types
                completeness_factors.append(min(source_diversity, 1.0))
            
            # Insight confidence factor
            completeness_factors.append(state['insight_confidence_avg'])
            
            # Conflict resolution factor
            completeness_factors.append(state['conflict_resolution_rate'])
            
            # Quality distribution factor
            high_quality_sources = (
                state['source_quality_distribution'].get('very_high', 0) + 
                state['source_quality_distribution'].get('high', 0)
            )
            total_sources = sum(state['source_quality_distribution'].values())
            quality_ratio = high_quality_sources / total_sources if total_sources > 0 else 0
            completeness_factors.append(quality_ratio)
            
            # Calculate final completeness score
            state['research_completeness_score'] = sum(completeness_factors) / len(completeness_factors)
            
            # Calculate quality preservation score (comparing parallel vs sequential quality)
            state['quality_preservation_score'] = min(state['research_completeness_score'] / 0.85, 1.0)  # Assuming 0.85 as sequential baseline
            
            logger.info(f"Research completeness: {state['research_completeness_score']:.2f}")
            logger.info(f"Quality preservation: {state['quality_preservation_score']:.2f}")
            
        except Exception as e:
            error_msg = f"Quality validation failed: {str(e)}"
            state['aggregation_errors'].append(error_msg)
            logger.error(error_msg)
        
        state['updated_at'] = datetime.now()
        return state
    
    def check_quality_gate(self, state: ParallelResearchState) -> str:
        """Check if research quality meets acceptance criteria."""
        completeness_threshold = 0.75
        quality_preservation_threshold = self.quality_preservation_target
        
        if (state['research_completeness_score'] >= completeness_threshold and 
            state['quality_preservation_score'] >= quality_preservation_threshold):
            return "quality_sufficient"
        elif state['research_completeness_score'] < 0.5:
            return "quality_failed"
        else:
            return "needs_additional_research"
    
    async def finalize_parallel_research(self, state: ParallelResearchState) -> ParallelResearchState:
        """Finalize the parallel research workflow."""
        logger.info("Finalizing parallel research workflow")
        
        state['status'] = 'completed'
        state['completion_percentage'] = 100.0
        state['updated_at'] = datetime.now()
        
        # Log final performance metrics
        logger.info(f"=== Parallel Research Performance Summary ===")
        logger.info(f"Time reduction achieved: {state['time_reduction_percentage']:.1f}% (Target: {self.target_time_reduction * 100}%)")
        logger.info(f"Parallel efficiency gain: {state['parallel_efficiency_gain']:.2f}x")
        logger.info(f"Quality preservation: {state['quality_preservation_score']:.1%}")
        logger.info(f"Research completeness: {state['research_completeness_score']:.1%}")
        logger.info(f"Sources processed: {len(state['deduplicated_sources'])}")
        logger.info(f"Insights generated: {len(state['aggregated_insights'])}")
        logger.info(f"Conflicts resolved: {len(state['conflicting_information'])}")
        
        # Evaluate success criteria
        time_target_met = state['time_reduction_percentage'] >= (self.target_time_reduction * 100)
        quality_target_met = state['quality_preservation_score'] >= self.quality_preservation_target
        
        if time_target_met and quality_target_met:
            logger.info("✅ All performance targets achieved!")
        elif time_target_met:
            logger.info("✅ Time reduction target achieved")
            logger.warning("⚠️ Quality preservation target not met")
        elif quality_target_met:
            logger.info("✅ Quality preservation target achieved")
            logger.warning("⚠️ Time reduction target not met")
        else:
            logger.warning("⚠️ Performance targets not met")
        
        # Track workflow completion (using execution tracking)
        if TRACKING_AVAILABLE and state.get('workflow_execution_id'):
            try:
                await global_performance_tracker.track_execution_end(
                    execution_id=state['workflow_execution_id'],
                    status="success" if (time_target_met and quality_target_met) else "partial_success"
                )
                
                # Track performance decision
                await global_performance_tracker.track_decision(
                    execution_id=state['workflow_execution_id'],
                    decision_point="workflow_completion_assessment",
                    input_data={
                        'time_reduction_target': self.target_time_reduction * 100,
                        'quality_target': self.quality_preservation_target
                    },
                    output_data={
                        'time_reduction_achieved': state['time_reduction_percentage'],
                        'quality_preservation_achieved': state['quality_preservation_score'],
                        'sources_processed': len(state['deduplicated_sources']),
                        'insights_generated': len(state['aggregated_insights'])
                    },
                    reasoning=f"Time target: {'✅' if time_target_met else '❌'}, Quality target: {'✅' if quality_target_met else '❌'}",
                    confidence_score=1.0,
                    execution_time_ms=int(state['total_execution_time'] * 1000) if state['total_execution_time'] else 0
                )
                
                logger.info(f"Completed performance tracking for workflow: {state['workflow_execution_id']}")
            except Exception as e:
                logger.error(f"Failed to complete workflow performance tracking: {e}")
        
        return state
    
    async def execute_parallel_research_workflow(
        self,
        research_topics: List[str],
        target_audience: str = "business_professionals",
        research_depth: str = "comprehensive",
        max_sources_per_agent: int = 15,
        time_range: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the complete parallel research workflow."""
        
        # Create initial state
        initial_state = ParallelResearchState(
            workflow_id=f"parallel-research-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            research_topics=research_topics,
            target_audience=target_audience,
            research_depth=research_depth,
            max_sources_per_agent=max_sources_per_agent,
            time_range=time_range
        )
        
        # Execute workflow
        # Configure checkpointer with thread_id
        config = {"configurable": {"thread_id": f"research_{initial_state['workflow_id']}"}}
        final_state = await self.workflow.ainvoke(initial_state, config=config)
        
        # Return structured results
        return {
            'workflow_id': final_state['workflow_id'],
            'status': final_state['status'],
            'performance_metrics': {
                'total_execution_time': final_state['total_execution_time'],
                'time_reduction_percentage': final_state['time_reduction_percentage'],
                'parallel_efficiency_gain': final_state['parallel_efficiency_gain'],
                'quality_preservation_score': final_state['quality_preservation_score'],
                'research_completeness_score': final_state['research_completeness_score']
            },
            'research_results': {
                'deduplicated_sources': final_state['deduplicated_sources'],
                'aggregated_insights': final_state['aggregated_insights'],
                'conflicting_information': final_state['conflicting_information'],
                'source_quality_distribution': final_state['source_quality_distribution']
            },
            'quality_metrics': {
                'insight_confidence_avg': final_state['insight_confidence_avg'],
                'conflict_resolution_rate': final_state['conflict_resolution_rate']
            },
            'errors': final_state['researcher_errors'] + final_state['search_agent_errors'] + final_state['aggregation_errors']
        }

    def validate_state_variables(self, state: ParallelResearchState, step_name: str) -> List[str]:
        """
        Validate that all required state variables are properly initialized.
        
        Returns:
            List of missing or invalid state variables
        """
        issues = []
        
        # Required string fields
        required_strings = ['workflow_id', 'status', 'research_depth', 'target_audience']
        for field in required_strings:
            if field not in state or not state[field]:
                issues.append(f"Missing or empty required field: {field}")
        
        # Required list fields
        required_lists = [
            'research_topics', 'merged_sources', 'deduplicated_sources',
            'aggregated_insights', 'conflicting_information', 'researcher_errors',
            'search_agent_errors', 'aggregation_errors'
        ]
        for field in required_lists:
            if field not in state:
                issues.append(f"Missing required list field: {field}")
            elif not isinstance(state[field], list):
                issues.append(f"Field {field} should be a list, got {type(state[field])}")
        
        # Required dict fields
        required_dicts = ['researcher_results', 'search_agent_results', 'source_quality_distribution']
        for field in required_dicts:
            if field not in state:
                issues.append(f"Missing required dict field: {field}")
            elif not isinstance(state[field], dict):
                issues.append(f"Field {field} should be a dict, got {type(state[field])}")
        
        # Required numeric fields
        required_numerics = [
            'researcher_execution_time', 'search_agent_execution_time',
            'total_execution_time', 'parallel_efficiency_gain',
            'time_reduction_percentage', 'quality_preservation_score',
            'research_completeness_score', 'insight_confidence_avg',
            'conflict_resolution_rate', 'completion_percentage', 'max_sources_per_agent'
        ]
        for field in required_numerics:
            if field not in state:
                issues.append(f"Missing required numeric field: {field}")
            elif not isinstance(state[field], (int, float)):
                issues.append(f"Field {field} should be numeric, got {type(state[field])}")
        
        # Required datetime fields
        required_datetimes = ['created_at', 'updated_at']
        for field in required_datetimes:
            if field not in state:
                issues.append(f"Missing required datetime field: {field}")
        
        # Validate quality distribution structure
        if 'source_quality_distribution' in state:
            required_quality_keys = ['very_high', 'high', 'medium', 'low', 'unknown']
            for key in required_quality_keys:
                if key not in state['source_quality_distribution']:
                    issues.append(f"Missing quality distribution key: {key}")
        
        if issues:
            logger.warning(f"State validation issues in {step_name}: {issues}")
        
        return issues
    
    def ensure_state_consistency(self, state: ParallelResearchState, step_name: str) -> ParallelResearchState:
        """
        Ensure state consistency by gracefully handling missing variables.
        
        This method provides fallback values for any missing state variables
        to prevent workflow crashes.
        """
        # Ensure all required fields exist with default values
        defaults = {
            # Basic workflow fields
            'workflow_id': f"parallel-research-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'status': 'in_progress',
            'completion_percentage': 0.0,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            
            # Input parameters
            'research_topics': [],
            'target_audience': 'general',
            'research_depth': 'standard',
            'max_sources_per_agent': 10,
            'time_range': None,
            
            # Collections
            'merged_sources': [],
            'deduplicated_sources': [],
            'aggregated_insights': [],
            'conflicting_information': [],
            'researcher_errors': [],
            'search_agent_errors': [],
            'aggregation_errors': [],
            
            # Agent results
            'researcher_results': {},
            'search_agent_results': {},
            
            # Performance metrics
            'researcher_execution_time': 0.0,
            'search_agent_execution_time': 0.0,
            'total_execution_time': 0.0,
            'parallel_efficiency_gain': 0.0,
            'time_reduction_percentage': 0.0,
            'quality_preservation_score': 0.0,
            
            # Quality metrics
            'source_quality_distribution': {
                'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'unknown': 0
            },
            'insight_confidence_avg': 0.0,
            'research_completeness_score': 0.0,
            'conflict_resolution_rate': 1.0,
            
            # Parallel execution tracking
            'researcher_task_id': None,
            'search_agent_task_id': None,
            'parallel_start_time': None,
            'parallel_completion_time': None,
            
            # Performance tracking
            'workflow_execution_id': None,
        }
        
        # Apply defaults for missing fields
        for key, default_value in defaults.items():
            if key not in state or state[key] is None:
                state[key] = default_value
                logger.debug(f"Applied default value for {key} in {step_name}")
        
        # Update timestamp
        state['updated_at'] = datetime.now()
        
        return state


# Create global instance for easy access
parallel_research_orchestrator = ParallelResearchOrchestrator()

# Export the workflow graph for LangGraph Studio integration
parallel_research_workflow = parallel_research_orchestrator.workflow

logger.info("🚀 Parallel Research Orchestrator loaded successfully!")
logger.info("Features: Parallel Execution, Advanced Deduplication, Conflict Resolution, Performance Tracking")