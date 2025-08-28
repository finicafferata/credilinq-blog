"""
Enhanced Blog Workflow with LangGraph State Management - Phase 2 Implementation.

This workflow integrates multiple agents (Planner, Researcher, Writer, Editor) using LangGraph
for better coordination, state persistence, and recovery capabilities. It builds on the existing
blog_workflow.py with enhanced state management and checkpointing.
"""

import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage

# Internal imports
from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, LangGraphExecutionContext,
    WorkflowStatus, CheckpointStrategy
)
from ..specialized.writer_agent_langgraph import WriterAgentLangGraphAdapter
from ..specialized.planner_agent import PlannerAgent
from ..specialized.researcher_agent import ResearcherAgent
from ..specialized.editor_agent import EditorAgent
from ..specialized.geo_agent import GEOAgent
from ..specialized.image_agent import ImageAgent
from ...config.settings import get_settings

import logging
logger = logging.getLogger(__name__)

class BlogWorkflowPhase(Enum):
    """Phases of the blog generation workflow."""
    PLANNING = "planning"
    RESEARCH = "research"
    WRITING = "writing"
    EDITING = "editing"
    IMAGE_GENERATION = "image_generation"
    FINAL_ASSEMBLY = "final_assembly"
    QUALITY_ASSURANCE = "quality_assurance"
    COMPLETED = "completed"

@dataclass
class AgentExecutionResult:
    """Result from individual agent execution within workflow."""
    agent_name: str
    success: bool
    execution_time_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)

class BlogWorkflowState(TypedDict):
    """Comprehensive state schema for the blog generation workflow."""
    # Input data
    blog_title: str
    company_context: str
    content_type: str  # "blog", "linkedin", "article"
    target_audience: Optional[str]
    unique_angle: Optional[str]
    
    # Workflow orchestration
    workflow_id: str
    current_phase: str
    completed_phases: List[str]
    failed_phases: List[str]
    checkpoint_data: Dict[str, Any]
    
    # Agent results tracking
    agent_results: Dict[str, AgentExecutionResult]
    
    # Content generation pipeline state
    outline: Annotated[List[str], "Blog post outline sections"]
    research: Annotated[Dict[str, Any], "Research data by section"]
    geo_metadata: Annotated[Dict[str, Any], "SEO optimization data"]
    draft_content: str
    editor_feedback: Optional[str]
    revised_content: str
    final_content: str
    
    # Media and enhancement
    generated_images: List[Dict[str, Any]]
    image_prompts: List[str]
    
    # Quality tracking
    content_quality_score: float
    seo_score: float
    readability_score: float
    revision_count: int
    max_revisions: int
    
    # Metadata and timing
    started_at: str
    updated_at: str
    completed_at: Optional[str]
    total_execution_time_ms: float
    
    # Error handling
    error_state: Optional[str]
    recovery_attempts: int
    
    # Final output
    publication_ready: bool
    content_metadata: Dict[str, Any]

class BlogWorkflowLangGraph(LangGraphWorkflowBase[BlogWorkflowState]):
    """
    Enhanced blog generation workflow using LangGraph for superior coordination,
    state management, and recovery capabilities.
    """
    
    def __init__(
        self,
        workflow_name: str = "enhanced_blog_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        max_retries: int = 3
    ):
        super().__init__(workflow_name, checkpoint_strategy=checkpoint_strategy, max_retries=max_retries)
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Workflow configuration
        self.parallel_execution_enabled = True
        self.quality_thresholds = {
            "min_content_quality_score": 7.0,
            "min_seo_score": 6.0,
            "min_readability_score": 0.7
        }
        
        self.logger.info("Enhanced Blog Workflow initialized with LangGraph")
    
    def _initialize_agents(self):
        """Initialize all specialized agents for the workflow."""
        try:
            self.planner_agent = PlannerAgent()
            self.researcher_agent = ResearcherAgent()
            self.writer_agent = WriterAgentLangGraphAdapter()
            self.editor_agent = EditorAgent()
            self.geo_agent = GEOAgent()
            self.image_agent = ImageAgent()
            
            self.logger.info("All specialized agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the enhanced LangGraph workflow with parallel processing and checkpointing."""
        workflow = StateGraph(BlogWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize_workflow", self._initialize_workflow)
        workflow.add_node("planning_phase", self._execute_planning_phase)
        workflow.add_node("research_phase", self._execute_research_phase)
        workflow.add_node("parallel_content_generation", self._execute_parallel_content_generation)
        workflow.add_node("writing_phase", self._execute_writing_phase)
        workflow.add_node("image_generation_phase", self._execute_image_generation_phase)
        workflow.add_node("content_assembly", self._execute_content_assembly)
        workflow.add_node("editing_phase", self._execute_editing_phase)
        workflow.add_node("quality_assurance", self._execute_quality_assurance)
        workflow.add_node("revision_phase", self._execute_revision_phase)
        workflow.add_node("finalize_workflow", self._finalize_workflow)
        
        # Set entry point
        workflow.set_entry_point("initialize_workflow")
        
        # Sequential workflow edges
        workflow.add_edge("initialize_workflow", "planning_phase")
        workflow.add_edge("planning_phase", "research_phase")
        workflow.add_edge("research_phase", "parallel_content_generation")
        workflow.add_edge("parallel_content_generation", "content_assembly")
        workflow.add_edge("content_assembly", "editing_phase")
        workflow.add_edge("editing_phase", "quality_assurance")
        
        # Conditional edges for revision loop
        workflow.add_conditional_edges(
            "quality_assurance",
            self._should_revise_content,
            {
                "revise": "revision_phase",
                "finalize": "finalize_workflow"
            }
        )
        
        workflow.add_edge("revision_phase", "editing_phase")
        workflow.add_edge("finalize_workflow", END)
        
        return workflow
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> BlogWorkflowState:
        """Create comprehensive initial state for the blog workflow."""
        workflow_id = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()
        
        return BlogWorkflowState(
            # Input data
            blog_title=input_data.get("blog_title", ""),
            company_context=input_data.get("company_context", ""),
            content_type=input_data.get("content_type", "blog").lower(),
            target_audience=input_data.get("target_audience", "Business professionals"),
            unique_angle=input_data.get("unique_angle"),
            
            # Workflow orchestration
            workflow_id=workflow_id,
            current_phase=BlogWorkflowPhase.PLANNING.value,
            completed_phases=[],
            failed_phases=[],
            checkpoint_data={},
            
            # Agent results
            agent_results={},
            
            # Content pipeline
            outline=[],
            research={},
            geo_metadata={},
            draft_content="",
            editor_feedback=None,
            revised_content="",
            final_content="",
            
            # Media
            generated_images=[],
            image_prompts=[],
            
            # Quality tracking
            content_quality_score=0.0,
            seo_score=0.0,
            readability_score=0.0,
            revision_count=0,
            max_revisions=input_data.get("max_revisions", 3),
            
            # Timing
            started_at=current_time,
            updated_at=current_time,
            completed_at=None,
            total_execution_time_ms=0.0,
            
            # Error handling
            error_state=None,
            recovery_attempts=0,
            
            # Output
            publication_ready=False,
            content_metadata={}
        )
    
    async def _initialize_workflow(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Initialize the blog generation workflow with enhanced state tracking."""
        self.logger.info(f"Initializing blog workflow for: {state['blog_title']}")
        
        start_time = datetime.utcnow()
        
        # Set initial workflow state
        state["current_phase"] = BlogWorkflowPhase.PLANNING.value
        state["checkpoint_data"]["initialization"] = {
            "timestamp": start_time.isoformat(),
            "input_validation": "passed"
        }
        
        # Validate required inputs
        required_fields = ["blog_title", "company_context"]
        missing_fields = [field for field in required_fields if not state.get(field)]
        
        if missing_fields:
            state["error_state"] = f"Missing required fields: {', '.join(missing_fields)}"
            state["failed_phases"].append("initialization")
            return state
        
        # Initialize agent results tracking
        state["agent_results"] = {
            "planner": AgentExecutionResult("planner", False, 0.0),
            "researcher": AgentExecutionResult("researcher", False, 0.0),
            "writer": AgentExecutionResult("writer", False, 0.0),
            "editor": AgentExecutionResult("editor", False, 0.0),
            "geo": AgentExecutionResult("geo", False, 0.0),
            "image": AgentExecutionResult("image", False, 0.0)
        }
        
        state["completed_phases"].append("initialization")
        state["updated_at"] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Workflow {state['workflow_id']} initialized successfully")
        return state
    
    async def _execute_planning_phase(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute content planning with the PlannerAgent."""
        self.logger.info("Executing planning phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = BlogWorkflowPhase.PLANNING.value
        
        try:
            # Prepare input for planner agent
            planner_input = {
                "blog_title": state["blog_title"],
                "company_context": state["company_context"],
                "content_type": state["content_type"],
                "target_audience": state["target_audience"],
                "unique_angle": state["unique_angle"]
            }
            
            # Execute planner agent
            planner_result = self.planner_agent.execute(planner_input)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if planner_result.success:
                # Extract planning results
                state["outline"] = planner_result.data.get("outline", [])
                state["geo_metadata"] = planner_result.data.get("seo_strategy", {})
                
                # Update agent results
                state["agent_results"]["planner"] = AgentExecutionResult(
                    "planner", True, execution_time, planner_result.data
                )
                
                state["completed_phases"].append("planning")
                self.logger.info(f"Planning completed: {len(state['outline'])} sections planned")
                
            else:
                state["failed_phases"].append("planning")
                state["error_state"] = f"Planning failed: {planner_result.error_message}"
                state["agent_results"]["planner"] = AgentExecutionResult(
                    "planner", False, execution_time, {}, planner_result.error_message
                )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.error(f"Planning phase failed: {e}")
            state["failed_phases"].append("planning")
            state["error_state"] = f"Planning exception: {str(e)}"
            state["agent_results"]["planner"] = AgentExecutionResult(
                "planner", False, execution_time, {}, str(e)
            )
        
        # Checkpoint
        state["checkpoint_data"]["planning"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "outline_sections": len(state["outline"]),
            "success": "planning" in state["completed_phases"]
        }
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_research_phase(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute research phase with enhanced data collection."""
        self.logger.info("Executing research phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = BlogWorkflowPhase.RESEARCH.value
        
        try:
            # Prepare research input
            research_input = {
                "blog_title": state["blog_title"],
                "outline": state["outline"],
                "company_context": state["company_context"],
                "research_depth": "comprehensive"
            }
            
            # Execute researcher agent
            research_result = self.researcher_agent.execute(research_input)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if research_result.success:
                state["research"] = research_result.data.get("research_results", {})
                
                state["agent_results"]["researcher"] = AgentExecutionResult(
                    "researcher", True, execution_time, research_result.data
                )
                
                state["completed_phases"].append("research")
                self.logger.info(f"Research completed for {len(state['research'])} sections")
                
            else:
                state["failed_phases"].append("research")
                state["error_state"] = f"Research failed: {research_result.error_message}"
                state["agent_results"]["researcher"] = AgentExecutionResult(
                    "researcher", False, execution_time, {}, research_result.error_message
                )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.error(f"Research phase failed: {e}")
            state["failed_phases"].append("research")
            state["error_state"] = f"Research exception: {str(e)}"
            state["agent_results"]["researcher"] = AgentExecutionResult(
                "researcher", False, execution_time, {}, str(e)
            )
        
        # Checkpoint
        state["checkpoint_data"]["research"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "research_sections": len(state["research"]),
            "success": "research" in state["completed_phases"]
        }
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_parallel_content_generation(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute writing and image generation in parallel for efficiency."""
        self.logger.info("Executing parallel content generation (writing + images)")
        
        start_time = datetime.utcnow()
        
        # Create tasks for parallel execution
        tasks = []
        
        # Writing task
        async def writing_task():
            return await self._execute_writing_phase(state.copy())
        
        # Image generation task
        async def image_task():
            return await self._execute_image_generation_phase(state.copy())
        
        try:
            # Execute tasks in parallel
            if self.parallel_execution_enabled:
                writing_result, image_result = await asyncio.gather(
                    writing_task(),
                    image_task(),
                    return_exceptions=True
                )
                
                # Merge results
                if isinstance(writing_result, dict) and not isinstance(writing_result, Exception):
                    state.update({
                        k: v for k, v in writing_result.items() 
                        if k in ["draft_content", "agent_results"]
                    })
                
                if isinstance(image_result, dict) and not isinstance(image_result, Exception):
                    state.update({
                        k: v for k, v in image_result.items() 
                        if k in ["generated_images", "image_prompts", "agent_results"]
                    })
                    # Merge agent results
                    if "agent_results" in image_result:
                        state["agent_results"].update(image_result["agent_results"])
                
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.logger.info(f"Parallel execution completed in {execution_time:.2f}ms")
                
            else:
                # Sequential execution fallback
                state = await writing_task()
                state = await image_task()
        
        except Exception as e:
            self.logger.error(f"Parallel content generation failed: {e}")
            state["error_state"] = f"Parallel generation failed: {str(e)}"
        
        state["current_phase"] = BlogWorkflowPhase.FINAL_ASSEMBLY.value
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_writing_phase(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute content writing with the enhanced WriterAgent."""
        self.logger.info("Executing writing phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = BlogWorkflowPhase.WRITING.value
        
        try:
            # Prepare writer input
            writer_input = {
                "blog_title": state["blog_title"],
                "company_context": state["company_context"],
                "content_type": state["content_type"],
                "outline": state["outline"],
                "research": state["research"],
                "max_revisions": state["max_revisions"]
            }
            
            # Execute enhanced writer agent
            writer_result = self.writer_agent.execute(writer_input)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if writer_result.success:
                state["draft_content"] = writer_result.data.get("content", "")
                state["content_quality_score"] = writer_result.data.get("quality_score", 0.0)
                
                state["agent_results"]["writer"] = AgentExecutionResult(
                    "writer", True, execution_time, writer_result.data
                )
                
                state["completed_phases"].append("writing")
                self.logger.info(f"Writing completed: {len(state['draft_content'])} characters")
                
            else:
                state["failed_phases"].append("writing")
                state["error_state"] = f"Writing failed: {writer_result.error_message}"
                state["agent_results"]["writer"] = AgentExecutionResult(
                    "writer", False, execution_time, {}, writer_result.error_message
                )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.error(f"Writing phase failed: {e}")
            state["failed_phases"].append("writing")
            state["error_state"] = f"Writing exception: {str(e)}"
            state["agent_results"]["writer"] = AgentExecutionResult(
                "writer", False, execution_time, {}, str(e)
            )
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_image_generation_phase(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute image generation for visual content."""
        self.logger.info("Executing image generation phase")
        
        start_time = datetime.utcnow()
        
        try:
            # Prepare image generation input
            image_input = {
                "content": state.get("draft_content", ""),
                "blog_title": state["blog_title"],
                "outline": state["outline"],
                "style": "professional",
                "count": 3
            }
            
            # Execute image agent
            image_result = self.image_agent.execute(image_input)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if image_result.success:
                state["generated_images"] = image_result.data.get("images", [])
                state["image_prompts"] = image_result.data.get("prompts", [])
                
                state["agent_results"]["image"] = AgentExecutionResult(
                    "image", True, execution_time, image_result.data
                )
                
                state["completed_phases"].append("image_generation")
                self.logger.info(f"Image generation completed: {len(state['generated_images'])} images")
                
            else:
                # Image generation failure is not critical
                self.logger.warning(f"Image generation failed: {image_result.error_message}")
                state["agent_results"]["image"] = AgentExecutionResult(
                    "image", False, execution_time, {}, image_result.error_message
                )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.warning(f"Image generation exception: {e}")
            state["agent_results"]["image"] = AgentExecutionResult(
                "image", False, execution_time, {}, str(e)
            )
        
        return state
    
    async def _execute_content_assembly(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Assemble content with images and metadata."""
        self.logger.info("Executing content assembly")
        
        # Assemble final content structure
        assembled_content = state["draft_content"]
        
        # Add image references if available
        if state["generated_images"]:
            image_references = []
            for i, image in enumerate(state["generated_images"]):
                image_ref = f"![Image {i+1}: {image.get('description', 'Generated image')}]({image.get('url', '')})"
                image_references.append(image_ref)
            
            # Insert images at appropriate positions
            if image_references:
                sections = assembled_content.split('\n## ')
                if len(sections) > 1:
                    # Insert images between sections
                    for i in range(min(len(image_references), len(sections) - 1)):
                        sections[i+1] = f"{sections[i+1]}\n\n{image_references[i]}\n"
                    assembled_content = '\n## '.join(sections)
        
        state["draft_content"] = assembled_content
        state["completed_phases"].append("content_assembly")
        state["current_phase"] = BlogWorkflowPhase.EDITING.value
        state["updated_at"] = datetime.utcnow().isoformat()
        
        return state
    
    async def _execute_editing_phase(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute content editing and review."""
        self.logger.info("Executing editing phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = BlogWorkflowPhase.EDITING.value
        
        try:
            # Prepare editor input
            editor_input = {
                "content": state["draft_content"],
                "blog_title": state["blog_title"],
                "company_context": state["company_context"],
                "content_type": state["content_type"]
            }
            
            # Execute editor agent
            editor_result = self.editor_agent.execute(editor_input)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if editor_result.success:
                state["editor_feedback"] = editor_result.data.get("feedback")
                state["revised_content"] = editor_result.data.get("revised_content", state["draft_content"])
                state["readability_score"] = editor_result.data.get("readability_score", 0.0)
                
                state["agent_results"]["editor"] = AgentExecutionResult(
                    "editor", True, execution_time, editor_result.data
                )
                
                state["completed_phases"].append("editing")
                self.logger.info("Editing phase completed")
                
            else:
                state["failed_phases"].append("editing")
                state["error_state"] = f"Editing failed: {editor_result.error_message}"
                state["agent_results"]["editor"] = AgentExecutionResult(
                    "editor", False, execution_time, {}, editor_result.error_message
                )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.error(f"Editing phase failed: {e}")
            state["failed_phases"].append("editing")
            state["error_state"] = f"Editing exception: {str(e)}"
            state["agent_results"]["editor"] = AgentExecutionResult(
                "editor", False, execution_time, {}, str(e)
            )
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_quality_assurance(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute comprehensive quality assurance checks."""
        self.logger.info("Executing quality assurance")
        
        state["current_phase"] = BlogWorkflowPhase.QUALITY_ASSURANCE.value
        
        # Calculate overall quality metrics
        content = state.get("revised_content") or state.get("draft_content", "")
        
        # Word count and basic metrics
        word_count = len(content.split())
        char_count = len(content)
        
        # Quality scoring
        quality_score = state.get("content_quality_score", 0.0)
        seo_score = self._calculate_seo_score(content, state)
        readability_score = state.get("readability_score", 0.0)
        
        state["content_quality_score"] = quality_score
        state["seo_score"] = seo_score
        state["readability_score"] = readability_score
        
        # Quality assessment
        quality_passed = (
            quality_score >= self.quality_thresholds["min_content_quality_score"] and
            seo_score >= self.quality_thresholds["min_seo_score"] and
            readability_score >= self.quality_thresholds["min_readability_score"]
        )
        
        # Update content metadata
        state["content_metadata"] = {
            "word_count": word_count,
            "character_count": char_count,
            "quality_score": quality_score,
            "seo_score": seo_score,
            "readability_score": readability_score,
            "quality_passed": quality_passed,
            "revision_count": state["revision_count"],
            "sections_count": len(state["outline"]),
            "images_count": len(state["generated_images"]),
            "qa_timestamp": datetime.utcnow().isoformat()
        }
        
        state["completed_phases"].append("quality_assurance")
        state["updated_at"] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Quality assurance completed: Quality={quality_score:.2f}, SEO={seo_score:.2f}, Readability={readability_score:.2f}")
        return state
    
    async def _execute_revision_phase(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Execute content revision based on quality assessment."""
        self.logger.info(f"Executing revision phase (attempt {state['revision_count'] + 1})")
        
        state["revision_count"] += 1
        
        # Create revision prompt based on quality issues
        revision_notes = self._generate_revision_notes(state)
        
        # Re-execute writing with revision notes
        writer_input = {
            "blog_title": state["blog_title"],
            "company_context": state["company_context"],
            "content_type": state["content_type"],
            "outline": state["outline"],
            "research": state["research"],
            "review_notes": revision_notes
        }
        
        try:
            writer_result = self.writer_agent.execute(writer_input)
            
            if writer_result.success:
                state["draft_content"] = writer_result.data.get("content", "")
                state["content_quality_score"] = writer_result.data.get("quality_score", 0.0)
                self.logger.info("Revision completed successfully")
            else:
                state["error_state"] = f"Revision failed: {writer_result.error_message}"
        
        except Exception as e:
            state["error_state"] = f"Revision exception: {str(e)}"
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _finalize_workflow(self, state: BlogWorkflowState) -> BlogWorkflowState:
        """Finalize the blog generation workflow."""
        self.logger.info("Finalizing blog generation workflow")
        
        # Set final content
        state["final_content"] = state.get("revised_content") or state.get("draft_content", "")
        
        # Mark as publication ready if quality passes
        quality_passed = state["content_metadata"].get("quality_passed", False)
        state["publication_ready"] = quality_passed and not state.get("error_state")
        
        # Calculate total execution time
        started_at = datetime.fromisoformat(state["started_at"])
        completed_at = datetime.utcnow()
        state["total_execution_time_ms"] = (completed_at - started_at).total_seconds() * 1000
        
        state["completed_at"] = completed_at.isoformat()
        state["current_phase"] = BlogWorkflowPhase.COMPLETED.value
        state["completed_phases"].append("finalization")
        
        # Final checkpoint
        state["checkpoint_data"]["finalization"] = {
            "timestamp": completed_at.isoformat(),
            "publication_ready": state["publication_ready"],
            "total_execution_time_ms": state["total_execution_time_ms"]
        }
        
        success_rate = len(state["completed_phases"]) / (len(state["completed_phases"]) + len(state["failed_phases"]))
        
        self.logger.info(f"Blog workflow completed: Publication ready={state['publication_ready']}, "
                        f"Success rate={success_rate:.2%}, "
                        f"Execution time={state['total_execution_time_ms']:.2f}ms")
        
        return state
    
    # Helper methods
    
    def _calculate_seo_score(self, content: str, state: BlogWorkflowState) -> float:
        """Calculate basic SEO score for the content."""
        score = 5.0  # Base score
        
        # Title presence and optimization
        title = state["blog_title"]
        if title and len(title.split()) >= 5:
            score += 1.0
        
        # Content length
        word_count = len(content.split())
        if word_count >= 1500:
            score += 1.0
        elif word_count >= 1000:
            score += 0.5
        
        # Header structure
        if "##" in content:
            score += 0.5
        if "###" in content:
            score += 0.5
        
        # Internal structure
        if "bullet" in content.lower() or "-" in content:
            score += 0.5
        
        # Call to action
        if any(cta in content.lower() for cta in ["learn more", "contact", "get started", "try"]):
            score += 0.5
        
        return min(10.0, score)
    
    def _generate_revision_notes(self, state: BlogWorkflowState) -> str:
        """Generate revision notes based on quality assessment."""
        notes = []
        
        if state["content_quality_score"] < self.quality_thresholds["min_content_quality_score"]:
            notes.append(f"Content quality score {state['content_quality_score']:.2f} is below threshold {self.quality_thresholds['min_content_quality_score']}")
        
        if state["seo_score"] < self.quality_thresholds["min_seo_score"]:
            notes.append(f"SEO score {state['seo_score']:.2f} is below threshold - improve keywords and structure")
        
        if state["readability_score"] < self.quality_thresholds["min_readability_score"]:
            notes.append(f"Readability score {state['readability_score']:.2f} is below threshold - simplify language and structure")
        
        word_count = state["content_metadata"].get("word_count", 0)
        if word_count < 1500 and state["content_type"] == "blog":
            notes.append(f"Content too short: {word_count} words, need at least 1500 for blog")
        
        return "; ".join(notes) if notes else "General quality improvement needed"
    
    # Conditional logic
    
    def _should_revise_content(self, state: BlogWorkflowState) -> str:
        """Determine if content needs revision or can be finalized."""
        quality_passed = state["content_metadata"].get("quality_passed", False)
        max_revisions_reached = state["revision_count"] >= state["max_revisions"]
        
        if not quality_passed and not max_revisions_reached:
            return "revise"
        return "finalize"


class BlogWorkflowOrchestrator:
    """
    Orchestrator class to manage the enhanced blog workflow with backward compatibility.
    """
    
    def __init__(self):
        self.langgraph_workflow = BlogWorkflowLangGraph()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_blog(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a blog using the enhanced LangGraph workflow.
        
        Args:
            input_data: Blog generation parameters
            
        Returns:
            Dict: Blog generation results
        """
        try:
            # Execute the LangGraph workflow
            result = await self.langgraph_workflow.execute(input_data)
            
            if result.success:
                final_state = result.data
                
                return {
                    "success": True,
                    "final_post": final_state.get("final_content", ""),
                    "blog_title": final_state.get("blog_title", ""),
                    "content_type": final_state.get("content_type", "blog"),
                    "publication_ready": final_state.get("publication_ready", False),
                    "workflow_metadata": {
                        "workflow_id": final_state.get("workflow_id"),
                        "execution_time_ms": final_state.get("total_execution_time_ms", 0),
                        "revision_count": final_state.get("revision_count", 0),
                        "quality_scores": {
                            "content": final_state.get("content_quality_score", 0),
                            "seo": final_state.get("seo_score", 0),
                            "readability": final_state.get("readability_score", 0)
                        },
                        "agent_results": final_state.get("agent_results", {}),
                        "completed_phases": final_state.get("completed_phases", []),
                        "failed_phases": final_state.get("failed_phases", [])
                    },
                    "content_metadata": final_state.get("content_metadata", {}),
                    "generated_images": final_state.get("generated_images", [])
                }
            else:
                return {
                    "success": False,
                    "error_message": result.error_message,
                    "error_code": result.error_code
                }
                
        except Exception as e:
            self.logger.error(f"Blog generation failed: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "error_code": "WORKFLOW_EXECUTION_FAILED"
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the LangGraph workflow."""
        return self.langgraph_workflow.get_workflow_info()


# Backward compatibility wrapper
async def generate_blog_langgraph(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible function for blog generation using LangGraph.
    
    Args:
        input_data: Blog generation input data
        
    Returns:
        Dict: Blog generation results
    """
    orchestrator = BlogWorkflowOrchestrator()
    return await orchestrator.generate_blog(input_data)


# Export key components
__all__ = [
    'BlogWorkflowLangGraph',
    'BlogWorkflowOrchestrator', 
    'BlogWorkflowState',
    'BlogWorkflowPhase',
    'generate_blog_langgraph'
]