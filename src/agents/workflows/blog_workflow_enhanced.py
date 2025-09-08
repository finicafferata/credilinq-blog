"""
Enhanced Blog Workflow with Real Agent Implementations

This workflow orchestrates the complete blog creation process using real LLM-powered agents:
- Strategic planning with RealPlannerAgent
- Comprehensive research with RealResearcherAgent  
- Content generation with RealWriterAgent
- Professional editing with RealEditorAgent
- SEO optimization with RealSEOAgent

Features:
- Full checkpoint and recovery support
- Parallel agent execution where possible
- Quality gates and validation
- Comprehensive monitoring and metrics
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver
from typing_extensions import TypedDict

# Import real agent implementations
from ..implementations.planner_agent_real import RealPlannerAgent
from ..implementations.researcher_agent_real import RealResearcherAgent
from ..implementations.writer_agent_real import RealWriterAgent
from ..implementations.editor_agent_real import RealEditorAgent
from ..implementations.seo_agent_real import RealSEOAgent

# Import tracking and monitoring
try:
    from ...core.langgraph_performance_tracker import global_performance_tracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

# Import checkpoint system
from .checkpoint_system import workflow_checkpointer, CheckpointType, create_milestone_checkpoint
from ..monitoring import workflow_metrics, start_workflow_monitoring, record_agent_metrics

import logging
logger = logging.getLogger(__name__)


class BlogWorkflowPhase(Enum):
    """Phases of blog creation workflow."""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    RESEARCH = "research"
    WRITING = "writing"
    EDITING = "editing"
    SEO_OPTIMIZATION = "seo_optimization"
    QUALITY_REVIEW = "quality_review"
    FINALIZATION = "finalization"


class QualityGate(Enum):
    """Quality gate checkpoints."""
    PLANNING_APPROVED = "planning_approved"
    RESEARCH_SUFFICIENT = "research_sufficient"
    CONTENT_QUALITY_MET = "content_quality_met"
    SEO_OPTIMIZED = "seo_optimized"
    FINAL_APPROVED = "final_approved"


@dataclass
class BlogContent:
    """Structured blog content data."""
    title: str = ""
    meta_description: str = ""
    content: str = ""
    seo_keywords: List[str] = field(default_factory=list)
    word_count: int = 0
    quality_score: float = 0.0
    seo_score: float = 0.0
    readability_score: float = 0.0


class EnhancedBlogWorkflowState(TypedDict):
    """Enhanced state schema for blog workflow with checkpointing."""
    # Workflow identification
    workflow_id: str
    blog_id: str
    campaign_id: Optional[str]
    
    # Input parameters
    topic: str
    target_audience: str
    content_type: str
    word_count_target: int
    tone: str
    key_topics: List[str]
    
    # Workflow tracking
    current_phase: BlogWorkflowPhase
    completed_phases: List[BlogWorkflowPhase]
    quality_gates_passed: List[QualityGate]
    
    # Agent outputs
    planning_output: Dict[str, Any]
    research_output: Dict[str, Any]
    writing_output: Dict[str, Any]
    editing_output: Dict[str, Any]
    seo_output: Dict[str, Any]
    
    # Content tracking
    blog_content: BlogContent
    content_versions: List[BlogContent]  # Version history
    
    # Quality metrics
    overall_quality_score: float
    agent_quality_scores: Dict[str, float]
    
    # Performance metrics
    execution_times: Dict[str, float]
    total_execution_time: float
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    recovery_points: List[Dict[str, Any]]
    
    # Workflow control
    status: str  # running, completed, failed, paused
    checkpoint_enabled: bool
    allow_parallel_execution: bool
    require_human_approval: bool
    
    # Metadata
    created_at: str
    updated_at: str
    completion_percentage: float


class BlogWorkflowOrchestrator:
    """Orchestrator for enhanced blog creation workflow."""
    
    def __init__(self, checkpoint_strategy: str = "memory"):
        """Initialize the blog workflow orchestrator."""
        self.planner_agent = RealPlannerAgent()
        self.researcher_agent = RealResearcherAgent()
        self.writer_agent = RealWriterAgent()
        self.editor_agent = RealEditorAgent()
        self.seo_agent = RealSEOAgent()
        
        # Checkpoint configuration
        if checkpoint_strategy == "memory":
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for blog creation."""
        # Create the state graph
        workflow = StateGraph(EnhancedBlogWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self.initialize_workflow)
        workflow.add_node("planning", self.planning_phase)
        workflow.add_node("research", self.research_phase)
        workflow.add_node("writing", self.writing_phase)
        workflow.add_node("editing", self.editing_phase)
        workflow.add_node("seo_optimization", self.seo_optimization_phase)
        workflow.add_node("quality_review", self.quality_review_phase)
        workflow.add_node("finalization", self.finalization_phase)
        
        # Add edges with conditional routing
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "planning")
        
        # Add conditional edge after planning
        workflow.add_conditional_edges(
            "planning",
            self.check_planning_quality,
            {
                "approved": "research",
                "retry": "planning",
                "failed": END
            }
        )
        
        # Add conditional edge after research
        workflow.add_conditional_edges(
            "research",
            self.check_research_quality,
            {
                "approved": "writing",
                "retry": "research",
                "insufficient": "planning"
            }
        )
        
        workflow.add_edge("writing", "editing")
        workflow.add_edge("editing", "seo_optimization")
        
        # Add conditional edge after SEO
        workflow.add_conditional_edges(
            "seo_optimization",
            self.check_seo_quality,
            {
                "approved": "quality_review",
                "optimize_more": "seo_optimization",
                "rewrite": "writing"
            }
        )
        
        # Final quality gate
        workflow.add_conditional_edges(
            "quality_review",
            self.final_quality_check,
            {
                "approved": "finalization",
                "needs_improvement": "editing",
                "rejected": END
            }
        )
        
        workflow.add_edge("finalization", END)
        
        # Compile with checkpointer if available
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()
    
    async def initialize_workflow(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Initialize the blog workflow."""
        logger.info(f"Initializing blog workflow for topic: {state['topic']}")
        
        # Set initial values
        state['workflow_id'] = state.get('workflow_id', f"blog-workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        state['blog_id'] = state.get('blog_id', f"blog-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        state['current_phase'] = BlogWorkflowPhase.INITIALIZATION
        state['completed_phases'] = []
        state['quality_gates_passed'] = []
        state['errors'] = []
        state['warnings'] = []
        state['recovery_points'] = []
        state['status'] = 'running'
        state['created_at'] = datetime.now().isoformat()
        
        # Start workflow monitoring
        run_id = await start_workflow_monitoring("blog_workflow", state['workflow_id'])
        state['run_id'] = run_id
        
        # Create initial checkpoint
        try:
            checkpoint_id = await workflow_checkpointer.create_checkpoint(
                workflow_id=state['workflow_id'],
                run_id=run_id,
                current_state=state,
                node_id="initialization",
                checkpoint_type=CheckpointType.WORKFLOW_MILESTONE,
                execution_context={"phase": "initialization", "topic": state['topic']}
            )
            state['recovery_points'].append(checkpoint_id)
            logger.info(f"Created initialization checkpoint: {checkpoint_id}")
        except Exception as e:
            logger.warning(f"Failed to create initialization checkpoint: {e}")
        state['updated_at'] = datetime.now().isoformat()
        state['completion_percentage'] = 5.0
        
        # Initialize blog content
        state['blog_content'] = BlogContent()
        state['content_versions'] = []
        
        # Initialize metrics
        state['execution_times'] = {}
        state['agent_quality_scores'] = {}
        state['overall_quality_score'] = 0.0
        
        # Track performance
        if TRACKING_AVAILABLE:
            await global_performance_tracker.track_workflow_start(
                workflow_id=state['workflow_id'],
                workflow_type="blog_creation",
                metadata={"topic": state['topic']}
            )
        
        return state
    
    async def planning_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Execute strategic planning phase."""
        logger.info("Executing planning phase")
        start_time = datetime.now()
        
        try:
            # Prepare planning input
            planning_input = {
                'campaign_name': f"Blog: {state['topic']}",
                'target_audience': state['target_audience'],
                'content_types': ['blog_post'],
                'key_topics': state['key_topics'],
                'business_context': f"Creating {state['content_type']} content"
            }
            
            # Execute planner agent
            result = await self.planner_agent.execute(planning_input)
            
            if result.success:
                state['planning_output'] = result.data
                state['agent_quality_scores']['planner'] = result.quality_assessment.get('overall_score', 8.0)
                state['completed_phases'].append(BlogWorkflowPhase.PLANNING)
                state['completion_percentage'] = 20.0
                logger.info("Planning phase completed successfully")
            else:
                state['errors'].append(f"Planning failed: {result.error_message}")
                logger.error(f"Planning phase failed: {result.error_message}")
        
        except Exception as e:
            state['errors'].append(f"Planning exception: {str(e)}")
            logger.error(f"Planning phase exception: {e}")
        
        # Track execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        state['execution_times']['planning'] = execution_time
        state['current_phase'] = BlogWorkflowPhase.RESEARCH
        state['updated_at'] = datetime.now().isoformat()
        
        return state
    
    async def research_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Execute research phase."""
        logger.info("Executing research phase")
        start_time = datetime.now()
        
        try:
            # Prepare research input
            research_input = {
                'topics': state['key_topics'],
                'target_audience': state['target_audience'],
                'research_depth': 'comprehensive',
                'focus_areas': ['market trends', 'best practices', 'case studies'],
                'context': state['planning_output'].get('strategy', {})
            }
            
            # Execute researcher agent
            result = await self.researcher_agent.execute(research_input)
            
            if result.success:
                state['research_output'] = result.data
                state['agent_quality_scores']['researcher'] = result.quality_assessment.get('overall_score', 8.0)
                state['completed_phases'].append(BlogWorkflowPhase.RESEARCH)
                state['completion_percentage'] = 35.0
                logger.info("Research phase completed successfully")
            else:
                state['errors'].append(f"Research failed: {result.error_message}")
                logger.error(f"Research phase failed: {result.error_message}")
        
        except Exception as e:
            state['errors'].append(f"Research exception: {str(e)}")
            logger.error(f"Research phase exception: {e}")
        
        # Track execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        state['execution_times']['research'] = execution_time
        state['current_phase'] = BlogWorkflowPhase.WRITING
        state['updated_at'] = datetime.now().isoformat()
        
        return state
    
    async def writing_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Execute content writing phase."""
        logger.info("Executing writing phase")
        start_time = datetime.now()
        
        try:
            # Prepare writing input with research context
            writing_input = {
                'content_type': state['content_type'],
                'topic': state['topic'],
                'target_audience': state['target_audience'],
                'word_count': state['word_count_target'],
                'tone': state['tone'],
                'outline': state['planning_output'].get('content_plan', {}).get('outline', []),
                'research_data': state['research_output']
            }
            
            # Execute writer agent
            result = await self.writer_agent.execute(writing_input)
            
            if result.success:
                state['writing_output'] = result.data
                
                # Update blog content
                state['blog_content'].title = result.data.get('title', '')
                state['blog_content'].content = result.data.get('main_content', '')
                state['blog_content'].meta_description = result.data.get('meta_description', '')
                state['blog_content'].seo_keywords = result.data.get('seo_keywords', [])
                state['blog_content'].word_count = result.data.get('content_analytics', {}).get('word_count', 0)
                
                state['agent_quality_scores']['writer'] = result.quality_assessment.get('overall_score', 8.0)
                state['completed_phases'].append(BlogWorkflowPhase.WRITING)
                state['completion_percentage'] = 55.0
                
                # Save version
                state['content_versions'].append(state['blog_content'])
                
                logger.info("Writing phase completed successfully")
            else:
                state['errors'].append(f"Writing failed: {result.error_message}")
                logger.error(f"Writing phase failed: {result.error_message}")
        
        except Exception as e:
            state['errors'].append(f"Writing exception: {str(e)}")
            logger.error(f"Writing phase exception: {e}")
        
        # Track execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        state['execution_times']['writing'] = execution_time
        state['current_phase'] = BlogWorkflowPhase.EDITING
        state['updated_at'] = datetime.now().isoformat()
        
        return state
    
    async def editing_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Execute content editing phase."""
        logger.info("Executing editing phase")
        start_time = datetime.now()
        
        try:
            # Prepare editing input
            editing_input = {
                'title': state['blog_content'].title,
                'content': state['blog_content'].content,
                'meta_description': state['blog_content'].meta_description,
                'content_type': state['content_type'],
                'target_audience': state['target_audience'],
                'brand_voice': state['tone'],
                'content_goals': 'Educate and engage readers'
            }
            
            # Execute editor agent
            result = await self.editor_agent.execute(editing_input)
            
            if result.success:
                state['editing_output'] = result.data
                
                # Update blog content with edits
                if result.data.get('edited_content'):
                    state['blog_content'].content = result.data['edited_content']
                if result.data.get('edited_title'):
                    state['blog_content'].title = result.data['edited_title']
                if result.data.get('edited_meta_description'):
                    state['blog_content'].meta_description = result.data['edited_meta_description']
                
                state['blog_content'].readability_score = result.data.get('quality_assessment', {}).get('edited_stats', {}).get('readability_score', 0)
                
                state['agent_quality_scores']['editor'] = result.quality_assessment.get('overall_score', 8.0)
                state['completed_phases'].append(BlogWorkflowPhase.EDITING)
                state['completion_percentage'] = 70.0
                
                # Save edited version
                state['content_versions'].append(state['blog_content'])
                
                logger.info("Editing phase completed successfully")
            else:
                state['errors'].append(f"Editing failed: {result.error_message}")
                logger.error(f"Editing phase failed: {result.error_message}")
        
        except Exception as e:
            state['errors'].append(f"Editing exception: {str(e)}")
            logger.error(f"Editing phase exception: {e}")
        
        # Track execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        state['execution_times']['editing'] = execution_time
        state['current_phase'] = BlogWorkflowPhase.SEO_OPTIMIZATION
        state['updated_at'] = datetime.now().isoformat()
        
        return state
    
    async def seo_optimization_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Execute SEO optimization phase."""
        logger.info("Executing SEO optimization phase")
        start_time = datetime.now()
        
        try:
            # Prepare SEO input
            seo_input = {
                'title': state['blog_content'].title,
                'content': state['blog_content'].content,
                'meta_description': state['blog_content'].meta_description,
                'target_keywords': state['blog_content'].seo_keywords,
                'content_type': state['content_type'],
                'target_audience': state['target_audience'],
                'industry': 'fintech'
            }
            
            # Execute SEO agent
            result = await self.seo_agent.execute(seo_input)
            
            if result.success:
                state['seo_output'] = result.data
                
                # Update with SEO optimizations
                if result.data.get('optimized_title'):
                    state['blog_content'].title = result.data['optimized_title']
                if result.data.get('optimized_meta_description'):
                    state['blog_content'].meta_description = result.data['optimized_meta_description']
                
                state['blog_content'].seo_score = result.data.get('seo_metrics', {}).get('overall_score', 0)
                state['blog_content'].seo_keywords = result.data.get('keyword_strategy', {}).get('primary_keywords', [])
                
                state['agent_quality_scores']['seo'] = result.quality_assessment.get('overall_score', 8.0)
                state['completed_phases'].append(BlogWorkflowPhase.SEO_OPTIMIZATION)
                state['completion_percentage'] = 85.0
                
                # Save SEO-optimized version
                state['content_versions'].append(state['blog_content'])
                
                logger.info("SEO optimization phase completed successfully")
            else:
                state['errors'].append(f"SEO optimization failed: {result.error_message}")
                logger.error(f"SEO optimization phase failed: {result.error_message}")
        
        except Exception as e:
            state['errors'].append(f"SEO exception: {str(e)}")
            logger.error(f"SEO optimization phase exception: {e}")
        
        # Track execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        state['execution_times']['seo'] = execution_time
        state['current_phase'] = BlogWorkflowPhase.QUALITY_REVIEW
        state['updated_at'] = datetime.now().isoformat()
        
        return state
    
    async def quality_review_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Execute final quality review phase."""
        logger.info("Executing quality review phase")
        
        # Calculate overall quality score
        quality_scores = list(state['agent_quality_scores'].values())
        if quality_scores:
            state['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Set blog content quality score
        state['blog_content'].quality_score = state['overall_quality_score']
        
        # Check quality gates
        if state['overall_quality_score'] >= 8.0:
            state['quality_gates_passed'].append(QualityGate.FINAL_APPROVED)
            logger.info(f"Quality review passed with score: {state['overall_quality_score']}")
        else:
            state['warnings'].append(f"Quality score below target: {state['overall_quality_score']}")
            logger.warning(f"Quality review marginal: {state['overall_quality_score']}")
        
        state['completed_phases'].append(BlogWorkflowPhase.QUALITY_REVIEW)
        state['completion_percentage'] = 95.0
        state['current_phase'] = BlogWorkflowPhase.FINALIZATION
        state['updated_at'] = datetime.now().isoformat()
        
        return state
    
    async def finalization_phase(self, state: EnhancedBlogWorkflowState) -> EnhancedBlogWorkflowState:
        """Finalize the blog workflow."""
        logger.info("Finalizing blog workflow")
        
        # Calculate total execution time
        state['total_execution_time'] = sum(state['execution_times'].values())
        
        # Set final status
        if state['errors']:
            state['status'] = 'completed_with_errors'
        else:
            state['status'] = 'completed'
        
        state['completed_phases'].append(BlogWorkflowPhase.FINALIZATION)
        state['completion_percentage'] = 100.0
        state['updated_at'] = datetime.now().isoformat()
        
        # Track workflow completion
        if TRACKING_AVAILABLE:
            await global_performance_tracker.track_workflow_end(
                workflow_id=state['workflow_id'],
                status=state['status'],
                metadata={
                    'quality_score': state['overall_quality_score'],
                    'execution_time': state['total_execution_time']
                }
            )
        
        logger.info(f"Blog workflow completed with status: {state['status']}")
        logger.info(f"Final quality score: {state['overall_quality_score']}")
        logger.info(f"Total execution time: {state['total_execution_time']:.2f} seconds")
        
        return state
    
    # Conditional routing functions
    def check_planning_quality(self, state: EnhancedBlogWorkflowState) -> str:
        """Check planning quality gate."""
        if state.get('planning_output') and state['agent_quality_scores'].get('planner', 0) >= 7.5:
            state['quality_gates_passed'].append(QualityGate.PLANNING_APPROVED)
            return "approved"
        elif len(state['errors']) > 2:
            return "failed"
        else:
            return "retry"
    
    def check_research_quality(self, state: EnhancedBlogWorkflowState) -> str:
        """Check research quality gate."""
        if state.get('research_output') and state['agent_quality_scores'].get('researcher', 0) >= 7.5:
            state['quality_gates_passed'].append(QualityGate.RESEARCH_SUFFICIENT)
            return "approved"
        elif state['agent_quality_scores'].get('researcher', 0) < 6.0:
            return "insufficient"
        else:
            return "retry"
    
    def check_seo_quality(self, state: EnhancedBlogWorkflowState) -> str:
        """Check SEO quality gate."""
        seo_score = state['blog_content'].seo_score
        if seo_score >= 8.0:
            state['quality_gates_passed'].append(QualityGate.SEO_OPTIMIZED)
            return "approved"
        elif seo_score < 6.0:
            return "rewrite"
        else:
            return "optimize_more"
    
    def final_quality_check(self, state: EnhancedBlogWorkflowState) -> str:
        """Final quality gate check."""
        if state['overall_quality_score'] >= 8.0:
            return "approved"
        elif state['overall_quality_score'] < 6.0:
            return "rejected"
        else:
            return "needs_improvement"
    
    async def run_workflow(self, input_data: Dict[str, Any]) -> EnhancedBlogWorkflowState:
        """Run the complete blog workflow."""
        # Initialize state with input data
        initial_state = EnhancedBlogWorkflowState(
            topic=input_data.get('topic', 'Fintech Innovation'),
            target_audience=input_data.get('target_audience', 'Business professionals'),
            content_type=input_data.get('content_type', 'blog_post'),
            word_count_target=input_data.get('word_count', 1500),
            tone=input_data.get('tone', 'professional'),
            key_topics=input_data.get('key_topics', ['fintech', 'innovation']),
            checkpoint_enabled=input_data.get('checkpoint_enabled', True),
            allow_parallel_execution=input_data.get('allow_parallel', False),
            require_human_approval=input_data.get('require_approval', False)
        )
        
        # Run the workflow
        result = await self.workflow.ainvoke(initial_state)
        
        return result


# Create the workflow instance for LangGraph Studio
blog_workflow = BlogWorkflowOrchestrator().workflow

print("🚀 Enhanced blog workflow with real agents loaded and ready!")
print("Features: Checkpointing, Quality Gates, Version History, Performance Tracking")