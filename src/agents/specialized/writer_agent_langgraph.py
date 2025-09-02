"""
WriterAgent LangGraph Implementation - Enhanced content generation with state management.

This is the LangGraph-migrated version of WriterAgent with enhanced capabilities:
- Multi-stage content generation with revision loops
- State management for drafts and quality tracking
- Parallel content generation for multiple sections
- Intelligent quality validation and improvement
- Full workflow checkpointing and recovery
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# LangGraph imports
from ..core.langgraph_compat import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.llm_client import create_llm

# Internal imports
from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, LangGraphExecutionContext, 
    WorkflowStatus, CheckpointStrategy
)
from ...core.security import SecurityValidator
from ...config.settings import get_settings

import logging
logger = logging.getLogger(__name__)

class ContentQualityLevel(Enum):
    """Content quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

@dataclass
class ContentSection:
    """Individual content section with quality tracking."""
    title: str
    content: str
    word_count: int
    quality_score: float = 0.0
    requires_revision: bool = False
    revision_notes: List[str] = field(default_factory=list)
    research_used: List[str] = field(default_factory=list)

class WriterWorkflowState(TypedDict):
    """State schema for the Writer LangGraph workflow."""
    # Input data
    blog_title: str
    company_context: str
    content_type: str
    outline: Annotated[List[str], "Content outline sections"]
    research: Annotated[Dict[str, Any], "Research data by section"]
    
    # Generation state
    current_section: int
    sections: Annotated[List[ContentSection], "Generated content sections"]
    draft_content: str
    
    # Quality tracking
    quality_assessment: Dict[str, Any]
    revision_count: int
    max_revisions: int
    
    # Review and editing state
    editor_feedback: Optional[str]
    requires_revision: bool
    
    # Final output
    final_content: str
    content_metadata: Dict[str, Any]
    
    # Workflow management
    workflow_id: str
    current_step: str
    step_history: List[str]
    error_state: Optional[str]

class WriterAgentLangGraph(LangGraphWorkflowBase[WriterWorkflowState]):
    """
    Enhanced WriterAgent implementation using LangGraph for better state management,
    content quality tracking, and multi-stage generation with revision loops.
    """
    
    def __init__(
        self,
        workflow_name: str = "writer_content_generation",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        max_retries: int = 3
    ):
        super().__init__(workflow_name, checkpoint_strategy=checkpoint_strategy, max_retries=max_retries)
        
        # Initialize content generation components
        self.security_validator = SecurityValidator()
        self.llm = None
        self._quality_thresholds = {
            "min_words_per_section": 200,
            "max_words_per_section": 800,
            "quality_score_threshold": 7.0,
            "readability_threshold": 0.7
        }
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            settings = get_settings()
            self.llm = create_llm(
                model="gemini-1.5-pro",  # Use GPT-4 for better content quality
                temperature=0.7,
                api_key=settings.primary_api_key
            )
            self.logger.info("WriterAgent LangGraph initialized with GPT-4")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow for enhanced content generation."""
        workflow = StateGraph(WriterWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_generation)
        workflow.add_node("plan_sections", self._plan_section_generation)
        workflow.add_node("generate_section", self._generate_section_content)
        workflow.add_node("assess_quality", self._assess_section_quality)
        workflow.add_node("revise_section", self._revise_section_content)
        workflow.add_node("compile_draft", self._compile_draft_content)
        workflow.add_node("review_draft", self._review_draft_quality)
        workflow.add_node("perform_revision", self._perform_full_revision)
        workflow.add_node("finalize_content", self._finalize_content)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add workflow edges
        workflow.add_edge("initialize", "plan_sections")
        workflow.add_edge("plan_sections", "generate_section")
        workflow.add_edge("generate_section", "assess_quality")
        
        # Conditional edge for section revision
        workflow.add_conditional_edges(
            "assess_quality",
            self._should_revise_section,
            {
                "revise": "revise_section",
                "continue": "compile_draft",
                "next_section": "generate_section"
            }
        )
        
        workflow.add_edge("revise_section", "assess_quality")
        workflow.add_edge("compile_draft", "review_draft")
        
        # Conditional edge for full draft revision
        workflow.add_conditional_edges(
            "review_draft",
            self._should_revise_draft,
            {
                "revise": "perform_revision",
                "finalize": "finalize_content"
            }
        )
        
        workflow.add_edge("perform_revision", "review_draft")
        workflow.add_edge("finalize_content", END)
        
        return workflow
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> WriterWorkflowState:
        """Create initial state for the writer workflow."""
        return WriterWorkflowState(
            # Input data
            blog_title=input_data.get("blog_title", ""),
            company_context=input_data.get("company_context", ""),
            content_type=input_data.get("content_type", "blog").lower(),
            outline=input_data.get("outline", []),
            research=input_data.get("research", {}),
            
            # Generation state
            current_section=0,
            sections=[],
            draft_content="",
            
            # Quality tracking
            quality_assessment={},
            revision_count=0,
            max_revisions=input_data.get("max_revisions", 3),
            
            # Review state
            editor_feedback=input_data.get("review_notes"),
            requires_revision=False,
            
            # Output
            final_content="",
            content_metadata={},
            
            # Workflow management
            workflow_id=str(uuid.uuid4()),
            current_step="initialize",
            step_history=[],
            error_state=None
        )
    
    async def _initialize_generation(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Initialize the content generation process."""
        self.logger.info(f"Initializing content generation for: {state['blog_title']}")
        
        # Validate input data
        try:
            self.security_validator.validate_content(str(state["blog_title"]), "blog_title")
            self.security_validator.validate_content(str(state["company_context"]), "company_context")
        except Exception as e:
            state["error_state"] = f"Security validation failed: {e}"
            return state
        
        state["current_step"] = "plan_sections"
        state["step_history"].append("initialize")
        
        # Initialize sections based on outline
        sections = []
        for i, section_title in enumerate(state["outline"]):
            sections.append(ContentSection(
                title=section_title,
                content="",
                word_count=0
            ))
        
        state["sections"] = sections
        state["quality_assessment"] = {
            "total_sections": len(sections),
            "completed_sections": 0,
            "avg_quality_score": 0.0,
            "revision_history": []
        }
        
        return state
    
    async def _plan_section_generation(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Plan the generation strategy for each section."""
        self.logger.info(f"Planning generation for {len(state['sections'])} sections")
        
        state["current_step"] = "generate_section"
        state["step_history"].append("plan_sections")
        state["current_section"] = 0
        
        # Add generation metadata for each section
        for i, section in enumerate(state["sections"]):
            section.research_used = list(state["research"].keys()) if state["research"] else []
        
        return state
    
    async def _generate_section_content(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Generate content for the current section."""
        current_idx = state["current_section"]
        
        if current_idx >= len(state["sections"]):
            # All sections complete, move to compilation
            state["current_step"] = "compile_draft"
            return state
        
        section = state["sections"][current_idx]
        self.logger.info(f"Generating content for section: {section.title}")
        
        # Prepare research context for this section
        section_research = self._get_section_research(section.title, state["research"])
        
        # Generate section content based on content type
        try:
            if state["content_type"] == "linkedin":
                content = await self._generate_linkedin_section(
                    section.title, section_research, state
                )
            elif state["content_type"] == "article":
                content = await self._generate_article_section(
                    section.title, section_research, state
                )
            else:  # blog
                content = await self._generate_blog_section(
                    section.title, section_research, state
                )
            
            # Update section with generated content
            section.content = content
            section.word_count = len(content.split())
            
            state["current_step"] = "assess_quality"
            state["step_history"].append("generate_section")
            
        except Exception as e:
            self.logger.error(f"Section generation failed: {e}")
            state["error_state"] = f"Section generation failed: {e}"
        
        return state
    
    async def _assess_section_quality(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Assess the quality of the currently generated section."""
        current_idx = state["current_section"]
        section = state["sections"][current_idx]
        
        self.logger.info(f"Assessing quality for section: {section.title}")
        
        # Perform quality assessment
        quality_score = await self._calculate_quality_score(section, state)
        section.quality_score = quality_score
        
        # Check if revision is needed
        if quality_score < self._quality_thresholds["quality_score_threshold"]:
            section.requires_revision = True
            # Generate revision notes
            revision_notes = await self._generate_revision_notes(section, state)
            section.revision_notes.extend(revision_notes)
        
        state["current_step"] = "decision"  # This will be handled by conditional logic
        state["step_history"].append("assess_quality")
        
        return state
    
    async def _revise_section_content(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Revise the current section based on quality assessment."""
        current_idx = state["current_section"]
        section = state["sections"][current_idx]
        
        self.logger.info(f"Revising section: {section.title}")
        
        # Create revision prompt
        revision_prompt = self._create_revision_prompt(section, state)
        
        try:
            # Generate revised content
            messages = [SystemMessage(content=revision_prompt)]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            
            # Update section with revised content
            section.content = response.content.strip()
            section.word_count = len(section.content.split())
            section.requires_revision = False
            
            state["revision_count"] += 1
            state["current_step"] = "assess_quality"
            state["step_history"].append("revise_section")
            
        except Exception as e:
            self.logger.error(f"Section revision failed: {e}")
            state["error_state"] = f"Section revision failed: {e}"
        
        return state
    
    async def _compile_draft_content(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Compile all sections into a complete draft."""
        self.logger.info("Compiling sections into draft content")
        
        # Combine all sections into complete content
        content_parts = []
        
        # Add title
        if state["content_type"] == "blog":
            content_parts.append(f"# {state['blog_title']}\n")
        
        # Add sections
        for section in state["sections"]:
            if state["content_type"] == "blog":
                content_parts.append(f"## {section.title}\n")
            content_parts.append(section.content)
            content_parts.append("\n")
        
        state["draft_content"] = "\n".join(content_parts).strip()
        
        # Update quality assessment
        state["quality_assessment"]["completed_sections"] = len(state["sections"])
        state["quality_assessment"]["avg_quality_score"] = sum(
            s.quality_score for s in state["sections"]
        ) / len(state["sections"])
        
        state["current_step"] = "review_draft"
        state["step_history"].append("compile_draft")
        
        return state
    
    async def _review_draft_quality(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Review the overall draft quality and determine if revision is needed."""
        self.logger.info("Reviewing draft quality")
        
        # Calculate overall quality metrics
        total_words = sum(s.word_count for s in state["sections"])
        avg_quality = state["quality_assessment"]["avg_quality_score"]
        
        # Determine if revision is needed
        revision_needed = False
        revision_reasons = []
        
        if avg_quality < self._quality_thresholds["quality_score_threshold"]:
            revision_needed = True
            revision_reasons.append(f"Average quality score {avg_quality:.2f} below threshold")
        
        if state["content_type"] == "blog" and total_words < 1500:
            revision_needed = True
            revision_reasons.append(f"Content too short: {total_words} words")
        elif state["content_type"] == "linkedin" and total_words < 800:
            revision_needed = True
            revision_reasons.append(f"Content too short: {total_words} words")
        
        if state["revision_count"] >= state["max_revisions"]:
            revision_needed = False  # Stop revising after max attempts
        
        state["requires_revision"] = revision_needed
        if revision_needed:
            state["editor_feedback"] = "; ".join(revision_reasons)
        
        state["current_step"] = "decision"  # Will be handled by conditional logic
        state["step_history"].append("review_draft")
        
        return state
    
    async def _perform_full_revision(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Perform a full revision of the draft content."""
        self.logger.info(f"Performing full revision (attempt {state['revision_count'] + 1})")
        
        # Create comprehensive revision prompt
        revision_prompt = f"""You are an expert content editor. Revise the following content to address these issues:

ISSUES TO ADDRESS:
{state['editor_feedback']}

ORIGINAL CONTENT:
{state['draft_content']}

CONTENT TYPE: {state['content_type']}
COMPANY CONTEXT: {state['company_context']}

REVISION REQUIREMENTS:
- Address all feedback points specifically
- Maintain the original structure and tone
- Ensure content flows naturally and is engaging
- Meet word count requirements for {state['content_type']} format
- Keep all important information and research insights
- Improve clarity and readability

Provide the complete revised content:"""

        try:
            messages = [SystemMessage(content=revision_prompt)]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            
            # Update draft with revised content
            revised_content = response.content.strip()
            state["draft_content"] = revised_content
            
            # Parse revised content back into sections for tracking
            self._update_sections_from_revised_content(state, revised_content)
            
            state["revision_count"] += 1
            state["requires_revision"] = False
            state["current_step"] = "review_draft"
            state["step_history"].append("perform_revision")
            
            # Add to revision history
            state["quality_assessment"]["revision_history"].append({
                "revision_count": state["revision_count"],
                "feedback": state["editor_feedback"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Full revision failed: {e}")
            state["error_state"] = f"Full revision failed: {e}"
        
        return state
    
    async def _finalize_content(self, state: WriterWorkflowState) -> WriterWorkflowState:
        """Finalize the content and prepare metadata."""
        self.logger.info("Finalizing content generation")
        
        state["final_content"] = state["draft_content"]
        
        # Generate content metadata
        total_words = sum(s.word_count for s in state["sections"])
        reading_time = max(1, total_words // 200)  # 200 words per minute
        
        state["content_metadata"] = {
            "word_count": total_words,
            "section_count": len(state["sections"]),
            "reading_time_minutes": reading_time,
            "content_type": state["content_type"],
            "quality_score": state["quality_assessment"]["avg_quality_score"],
            "revision_count": state["revision_count"],
            "generation_completed_at": datetime.utcnow().isoformat(),
            "sections_quality": [
                {
                    "title": s.title,
                    "word_count": s.word_count,
                    "quality_score": s.quality_score
                }
                for s in state["sections"]
            ]
        }
        
        state["current_step"] = "completed"
        state["step_history"].append("finalize_content")
        
        return state
    
    # Helper methods for content generation
    
    async def _generate_blog_section(self, section_title: str, research: str, state: WriterWorkflowState) -> str:
        """Generate blog section content."""
        prompt = f"""You are 'ContextMark', an expert blog writer. Write a comprehensive section for a blog post.

BLOG TITLE: {state['blog_title']}
SECTION TITLE: {section_title}
COMPANY CONTEXT: {state['company_context']}

RESEARCH DATA:
{research}

REQUIREMENTS:
- Write 300-500 words for this section
- Use engaging, professional tone
- Include specific examples and insights from research
- Use proper markdown formatting
- Make content scannable with bullet points where appropriate
- Ensure content flows naturally with the overall blog structure

Write the section content now (without the section header - that will be added separately):"""

        messages = [SystemMessage(content=prompt)]
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content.strip()
    
    async def _generate_linkedin_section(self, section_title: str, research: str, state: WriterWorkflowState) -> str:
        """Generate LinkedIn section content."""
        prompt = f"""You are 'ContextMark', an expert LinkedIn content creator. Write a section for a LinkedIn post.

POST TITLE: {state['blog_title']}
SECTION: {section_title}
COMPANY CONTEXT: {state['company_context']}

RESEARCH DATA:
{research}

REQUIREMENTS:
- Write 150-200 words for this section
- Use professional yet personable tone
- Include actionable insights from research
- Make it engaging for LinkedIn audience
- Use minimal formatting - focus on readability

Write the section content now:"""

        messages = [SystemMessage(content=prompt)]
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content.strip()
    
    async def _generate_article_section(self, section_title: str, research: str, state: WriterWorkflowState) -> str:
        """Generate article section content."""
        prompt = f"""You are 'ContextMark', an expert article writer. Write an analytical section for an article.

ARTICLE TITLE: {state['blog_title']}
SECTION TITLE: {section_title}
COMPANY CONTEXT: {state['company_context']}

RESEARCH DATA:
{research}

REQUIREMENTS:
- Write 400-600 words for this section
- Use analytical, informative tone
- Include data-driven insights from research
- Use proper markdown formatting with headers
- Provide objective analysis and multiple perspectives
- Include specific examples and case studies

Write the section content now:"""

        messages = [SystemMessage(content=prompt)]
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content.strip()
    
    def _get_section_research(self, section_title: str, research_data: Dict[str, Any]) -> str:
        """Extract relevant research for a specific section."""
        if not research_data:
            return "No specific research available."
        
        # Look for exact match first
        if section_title in research_data:
            research = research_data[section_title]
            if isinstance(research, dict):
                return research.get("content", str(research))
            return str(research)
        
        # Look for partial matches
        for key, value in research_data.items():
            if any(word.lower() in key.lower() for word in section_title.split()):
                if isinstance(value, dict):
                    return value.get("content", str(value))
                return str(value)
        
        # Return combined research if no specific match
        combined = []
        for key, value in research_data.items():
            if isinstance(value, dict):
                content = value.get("content", str(value))
            else:
                content = str(value)
            combined.append(f"Research for {key}: {content}")
        
        return "\n\n".join(combined[:3])  # Limit to first 3 research items
    
    async def _calculate_quality_score(self, section: ContentSection, state: WriterWorkflowState) -> float:
        """Calculate quality score for a content section."""
        score = 10.0  # Start with perfect score
        
        # Word count scoring
        word_count = section.word_count
        min_words = self._quality_thresholds["min_words_per_section"]
        max_words = self._quality_thresholds["max_words_per_section"]
        
        if word_count < min_words:
            score -= (min_words - word_count) / min_words * 3.0
        elif word_count > max_words * 1.5:
            score -= 2.0
        
        # Content quality checks
        content_lower = section.content.lower()
        
        # Check for engagement elements
        if not any(marker in content_lower for marker in ["example", "for instance", "such as", "consider"]):
            score -= 1.0
        
        # Check for actionable insights
        if not any(word in content_lower for word in ["should", "can", "will", "how to", "steps", "process"]):
            score -= 1.0
        
        # Check for research integration
        if not section.research_used:
            score -= 1.5
        
        # Ensure score is within bounds
        return max(0.0, min(10.0, score))
    
    async def _generate_revision_notes(self, section: ContentSection, state: WriterWorkflowState) -> List[str]:
        """Generate specific revision notes for a section."""
        notes = []
        
        if section.word_count < self._quality_thresholds["min_words_per_section"]:
            notes.append(f"Expand content - only {section.word_count} words, need at least {self._quality_thresholds['min_words_per_section']}")
        
        if section.quality_score < 5.0:
            notes.append("Add more specific examples and actionable insights")
        
        if not section.research_used:
            notes.append("Better integrate research data and specific examples")
        
        if "example" not in section.content.lower():
            notes.append("Include concrete examples to illustrate key points")
        
        return notes
    
    def _create_revision_prompt(self, section: ContentSection, state: WriterWorkflowState) -> str:
        """Create a revision prompt for a specific section."""
        return f"""You are an expert content editor. Revise this section to address the specific issues identified.

SECTION TITLE: {section.title}
CURRENT CONTENT:
{section.content}

ISSUES TO ADDRESS:
{'; '.join(section.revision_notes)}

CONTENT TYPE: {state['content_type']}
COMPANY CONTEXT: {state['company_context']}

REVISION REQUIREMENTS:
- Address all specific issues mentioned above
- Maintain the professional tone and style
- Ensure content is engaging and actionable
- Keep the core message but improve quality
- Use research insights more effectively

Provide the complete revised section content:"""
    
    def _update_sections_from_revised_content(self, state: WriterWorkflowState, revised_content: str):
        """Update section objects from revised full content."""
        # This is a simplified approach - in production, you might want more sophisticated parsing
        lines = revised_content.split('\n')
        current_section_idx = -1
        current_content = []
        
        for line in lines:
            if line.startswith('## ') and current_section_idx < len(state["sections"]) - 1:
                # Save previous section if exists
                if current_section_idx >= 0:
                    section_content = '\n'.join(current_content).strip()
                    state["sections"][current_section_idx].content = section_content
                    state["sections"][current_section_idx].word_count = len(section_content.split())
                
                # Move to next section
                current_section_idx += 1
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section_idx >= 0 and current_section_idx < len(state["sections"]):
            section_content = '\n'.join(current_content).strip()
            state["sections"][current_section_idx].content = section_content
            state["sections"][current_section_idx].word_count = len(section_content.split())
    
    # Conditional logic functions
    
    def _should_revise_section(self, state: WriterWorkflowState) -> str:
        """Determine if current section needs revision or if workflow should continue."""
        current_idx = state["current_section"]
        section = state["sections"][current_idx]
        
        if section.requires_revision and state["revision_count"] < state["max_revisions"]:
            return "revise"
        
        # Move to next section
        state["current_section"] += 1
        if state["current_section"] < len(state["sections"]):
            return "next_section"
        
        # All sections complete
        return "continue"
    
    def _should_revise_draft(self, state: WriterWorkflowState) -> str:
        """Determine if the draft needs revision or should be finalized."""
        if state["requires_revision"] and state["revision_count"] < state["max_revisions"]:
            return "revise"
        return "finalize"


class WriterAgentLangGraphAdapter(BaseAgent):
    """
    Adapter class to integrate WriterAgentLangGraph with existing BaseAgent interface.
    This provides backward compatibility while enabling LangGraph capabilities.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WRITER,
                name="WriterAgentLangGraph",
                description="Enhanced content generation with LangGraph workflows",
                capabilities=[
                    "multi_stage_content_generation",
                    "quality_assessment",
                    "revision_loops",
                    "parallel_section_generation",
                    "workflow_checkpointing",
                    "state_recovery"
                ],
                version="3.0.0-langgraph"
            )
        
        super().__init__(metadata)
        
        # Initialize LangGraph workflow
        self.langgraph_workflow = WriterAgentLangGraph()
        
        # Backward compatibility
        self.security_validator = SecurityValidator()
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute content generation using LangGraph workflow.
        
        Args:
            input_data: Dictionary containing:
                - blog_title: Title of the content
                - company_context: Company context
                - content_type: Type of content (blog/linkedin/article)
                - outline: List of sections
                - research: Research data by section
                - review_notes: Editor feedback (optional)
                - max_revisions: Maximum revision attempts (optional)
            context: Execution context
            
        Returns:
            AgentResult: Generated content with metadata
        """
        try:
            # Convert to LangGraph execution context
            langgraph_context = LangGraphExecutionContext(
                workflow_id=context.workflow_id if context else None,
                user_id=context.user_id if context else None,
                session_id=context.session_id if context else None,
                execution_metadata=context.execution_metadata if context else {}
            )
            
            # Run the LangGraph workflow
            import asyncio
            
            # Handle different event loop scenarios
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new thread for async execution
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self.langgraph_workflow.execute(input_data, langgraph_context)
                        )
                        result = future.result()
                else:
                    result = loop.run_until_complete(
                        self.langgraph_workflow.execute(input_data, langgraph_context)
                    )
            except RuntimeError:
                # No event loop exists, create one
                result = asyncio.run(self.langgraph_workflow.execute(input_data, langgraph_context))
            
            if result.success:
                # Extract final state data
                final_state = result.data
                
                # Format result for backward compatibility
                formatted_result = AgentResult(
                    success=True,
                    data={
                        "content": final_state.get("final_content", ""),
                        "content_type": final_state.get("content_type", "blog"),
                        "word_count": final_state.get("content_metadata", {}).get("word_count", 0),
                        "reading_time": final_state.get("content_metadata", {}).get("reading_time_minutes", 0),
                        "quality_score": final_state.get("content_metadata", {}).get("quality_score", 0),
                        "revision_count": final_state.get("revision_count", 0),
                        "sections_covered": len(final_state.get("sections", [])),
                        "content_analysis": final_state.get("content_metadata", {}),
                        "workflow_metadata": {
                            "workflow_id": final_state.get("workflow_id"),
                            "step_history": final_state.get("step_history", []),
                            "generation_method": "langgraph_enhanced"
                        }
                    },
                    metadata={
                        "agent_type": "writer_langgraph",
                        "content_format": final_state.get("content_type", "blog"),
                        "quality_score": final_state.get("content_metadata", {}).get("quality_score", 0),
                        "enhanced_generation": True
                    }
                )
            else:
                # Handle workflow failure
                formatted_result = AgentResult(
                    success=False,
                    error_message=result.error_message,
                    error_code=result.error_code,
                    metadata=result.metadata
                )
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"LangGraph content generation failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="LANGGRAPH_GENERATION_FAILED"
            )
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the LangGraph workflow."""
        return self.langgraph_workflow.get_workflow_info()
    
    async def pause_generation(self, workflow_id: str) -> bool:
        """Pause an active content generation workflow."""
        return await self.langgraph_workflow.pause_workflow(workflow_id)
    
    async def resume_generation(self, workflow_id: str) -> AgentResult:
        """Resume a paused content generation workflow."""
        return await self.langgraph_workflow.resume_workflow(workflow_id)