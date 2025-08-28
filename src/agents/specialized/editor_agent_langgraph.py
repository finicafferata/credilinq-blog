"""
EditorAgent LangGraph Implementation - Advanced content editing with quality assurance.
"""

from typing import Dict, Any, Optional, List, TypedDict
from enum import Enum
import json
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


class EditingPhase(str, Enum):
    """Phases of the editing workflow."""
    INITIAL_REVIEW = "initial_review"
    CONTENT_ANALYSIS = "content_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    REVISION_PLANNING = "revision_planning"
    FINAL_APPROVAL = "final_approval"


class EditorState(TypedDict):
    """State for the editor workflow."""
    # Input data
    content: str
    blog_title: str
    company_context: str
    content_type: str
    quality_standards: Dict[str, Any]
    
    # Review results
    automated_score: Dict[str, float]
    llm_review: Dict[str, Any]
    quality_dimensions: Dict[str, float]
    improvement_suggestions: List[str]
    
    # Decision data
    overall_score: float
    approved: bool
    revision_notes: str
    confidence_level: str
    
    # Workflow metadata
    current_phase: str
    revision_count: int
    max_revisions: int
    phase_results: Dict[str, Any]
    errors: List[str]


class EditorAgentWorkflow(LangGraphWorkflowBase):
    """
    LangGraph-based EditorAgent with multi-stage quality assurance.
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        max_revisions: int = 3
    ):
        """
        Initialize the EditorAgent workflow.
        
        Args:
            llm: Language model for content review
            checkpoint_strategy: When to save checkpoints
            max_revisions: Maximum number of revision cycles
        """
        super().__init__(
            name="EditorAgentWorkflow",
            checkpoint_strategy=checkpoint_strategy
        )
        
        self.llm = llm
        self.security_validator = SecurityValidator()
        self.max_revisions = max_revisions
        self.quality_thresholds = {
            "excellent": 90,
            "good": 80,
            "acceptable": 70,
            "needs_improvement": 60
        }
        
        # Build the workflow graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(EditorState)
        
        # Add nodes for each phase
        workflow.add_node("validate_input", self.validate_input_node)
        workflow.add_node("initial_review", self.initial_review_node)
        workflow.add_node("content_analysis", self.content_analysis_node)
        workflow.add_node("quality_assessment", self.quality_assessment_node)
        workflow.add_node("revision_planning", self.revision_planning_node)
        workflow.add_node("final_approval", self.final_approval_node)
        
        # Define edges
        workflow.set_entry_point("validate_input")
        
        workflow.add_edge("validate_input", "initial_review")
        workflow.add_edge("initial_review", "content_analysis")
        workflow.add_edge("content_analysis", "quality_assessment")
        
        # Conditional routing based on quality
        workflow.add_conditional_edges(
            "quality_assessment",
            self.should_revise,
            {
                "revise": "revision_planning",
                "approve": "final_approval",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "revision_planning",
            self.check_revision_limit,
            {
                "continue": "initial_review",
                "stop": "final_approval"
            }
        )
        
        workflow.add_edge("final_approval", END)
        
        # Compile with memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
    
    def validate_input_node(self, state: EditorState) -> EditorState:
        """Validate input data and initialize state."""
        try:
            # Security validation
            self.security_validator.validate_input(state["content"])
            self.security_validator.validate_input(state["blog_title"])
            self.security_validator.validate_input(state["company_context"])
            
            # Initialize state fields
            state["current_phase"] = EditingPhase.INITIAL_REVIEW
            state["revision_count"] = state.get("revision_count", 0)
            state["max_revisions"] = self.max_revisions
            state["phase_results"] = {}
            state["errors"] = []
            state["content_type"] = state.get("content_type", "blog")
            state["quality_standards"] = state.get("quality_standards", {})
            
            self.logger.info(f"Input validation successful for: {state['blog_title']}")
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            state["errors"].append(f"Validation error: {str(e)}")
        
        return state
    
    def initial_review_node(self, state: EditorState) -> EditorState:
        """Perform initial automated content review."""
        try:
            state["current_phase"] = EditingPhase.INITIAL_REVIEW
            
            content = state["content"]
            content_type = state["content_type"]
            
            # Automated quality checks
            automated_score = {
                "length_score": self._check_content_length(content, content_type),
                "format_score": self._check_formatting(content),
                "readability_score": self._check_readability(content),
                "structure_score": self._check_content_structure(content, content_type),
                "consistency_score": self._check_consistency(content)
            }
            
            # Calculate total score
            automated_score["total_score"] = (
                automated_score["length_score"] * 0.2 +
                automated_score["format_score"] * 0.25 +
                automated_score["readability_score"] * 0.25 +
                automated_score["structure_score"] * 0.2 +
                automated_score["consistency_score"] * 0.1
            )
            
            state["automated_score"] = automated_score
            state["phase_results"]["initial_review"] = {
                "status": "completed",
                "score": automated_score["total_score"]
            }
            
            self.logger.info(f"Initial review completed. Score: {automated_score['total_score']:.1f}")
            
        except Exception as e:
            self.logger.error(f"Initial review failed: {str(e)}")
            state["errors"].append(f"Initial review error: {str(e)}")
            state["automated_score"] = {"total_score": 0}
        
        return state
    
    def content_analysis_node(self, state: EditorState) -> EditorState:
        """Perform deep content analysis using LLM."""
        try:
            state["current_phase"] = EditingPhase.CONTENT_ANALYSIS
            
            if not self.llm:
                # Fallback if no LLM available
                state["llm_review"] = self._get_fallback_review()
            else:
                # LLM-based content review
                llm_review = self._perform_llm_review(
                    state["content"],
                    state["blog_title"],
                    state["company_context"],
                    state["content_type"]
                )
                state["llm_review"] = llm_review
            
            state["phase_results"]["content_analysis"] = {
                "status": "completed",
                "llm_score": state["llm_review"].get("score", 70)
            }
            
            self.logger.info("Content analysis completed")
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            state["errors"].append(f"Content analysis error: {str(e)}")
            state["llm_review"] = self._get_fallback_review()
        
        return state
    
    def quality_assessment_node(self, state: EditorState) -> EditorState:
        """Assess overall content quality and generate suggestions."""
        try:
            state["current_phase"] = EditingPhase.QUALITY_ASSESSMENT
            
            # Analyze quality dimensions
            quality_dimensions = {
                "clarity": self._assess_clarity(state["content"]),
                "completeness": self._assess_completeness(state["content"], state["content_type"]),
                "engagement": self._assess_engagement(state["content"]),
                "professionalism": self._assess_professionalism(state["content"]),
                "actionability": self._assess_actionability(state["content"])
            }
            state["quality_dimensions"] = quality_dimensions
            
            # Calculate overall score
            automated_score = state.get("automated_score", {}).get("total_score", 70)
            llm_score = state.get("llm_review", {}).get("score", 70)
            
            state["overall_score"] = (automated_score + llm_score) / 2
            
            # Generate improvement suggestions
            suggestions = self._generate_suggestions(
                state["automated_score"],
                state["llm_review"],
                quality_dimensions
            )
            state["improvement_suggestions"] = suggestions
            
            state["phase_results"]["quality_assessment"] = {
                "status": "completed",
                "overall_score": state["overall_score"],
                "dimension_scores": quality_dimensions
            }
            
            self.logger.info(f"Quality assessment completed. Overall score: {state['overall_score']:.1f}")
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            state["errors"].append(f"Quality assessment error: {str(e)}")
            state["overall_score"] = 0
        
        return state
    
    def revision_planning_node(self, state: EditorState) -> EditorState:
        """Plan and document required revisions."""
        try:
            state["current_phase"] = EditingPhase.REVISION_PLANNING
            
            # Compile revision notes
            revision_notes = self._compile_revision_notes(
                state["overall_score"],
                state["llm_review"],
                state["improvement_suggestions"],
                state["quality_dimensions"]
            )
            
            state["revision_notes"] = revision_notes
            state["revision_count"] = state.get("revision_count", 0) + 1
            
            state["phase_results"]["revision_planning"] = {
                "status": "completed",
                "revision_number": state["revision_count"],
                "notes_length": len(revision_notes)
            }
            
            self.logger.info(f"Revision planning completed. Revision #{state['revision_count']}")
            
        except Exception as e:
            self.logger.error(f"Revision planning failed: {str(e)}")
            state["errors"].append(f"Revision planning error: {str(e)}")
            state["revision_notes"] = "Unable to generate revision notes"
        
        return state
    
    def final_approval_node(self, state: EditorState) -> EditorState:
        """Make final approval decision."""
        try:
            state["current_phase"] = EditingPhase.FINAL_APPROVAL
            
            overall_score = state["overall_score"]
            llm_recommendation = state.get("llm_review", {}).get("approval_recommendation", "revise")
            
            # Determine approval based on score and recommendation
            if overall_score >= self.quality_thresholds["excellent"] and llm_recommendation == "approve":
                state["approved"] = True
                state["confidence_level"] = "high"
            elif overall_score >= self.quality_thresholds["good"]:
                state["approved"] = True
                state["confidence_level"] = "medium"
            elif overall_score >= self.quality_thresholds["acceptable"]:
                state["approved"] = True
                state["confidence_level"] = "low"
            else:
                state["approved"] = False
                state["confidence_level"] = "insufficient"
            
            state["phase_results"]["final_approval"] = {
                "status": "completed",
                "approved": state["approved"],
                "confidence": state["confidence_level"],
                "final_score": overall_score
            }
            
            self.logger.info(
                f"Final approval: {'Approved' if state['approved'] else 'Rejected'} "
                f"with {state['confidence_level']} confidence"
            )
            
        except Exception as e:
            self.logger.error(f"Final approval failed: {str(e)}")
            state["errors"].append(f"Final approval error: {str(e)}")
            state["approved"] = False
            state["confidence_level"] = "error"
        
        return state
    
    def should_revise(self, state: EditorState) -> str:
        """Determine whether content needs revision."""
        if state.get("errors"):
            return "end"
        
        overall_score = state.get("overall_score", 0)
        
        if overall_score >= self.quality_thresholds["acceptable"]:
            return "approve"
        else:
            return "revise"
    
    def check_revision_limit(self, state: EditorState) -> str:
        """Check if revision limit has been reached."""
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", self.max_revisions)
        
        if revision_count >= max_revisions:
            self.logger.warning(f"Revision limit reached ({revision_count}/{max_revisions})")
            return "stop"
        else:
            return "continue"
    
    def _perform_llm_review(
        self,
        content: str,
        blog_title: str,
        company_context: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Perform LLM-based content review."""
        prompt = f"""You are a senior editor reviewing content for publication. Evaluate this {content_type} comprehensively.

COMPANY CONTEXT:
{company_context}

CONTENT TITLE: {blog_title}

CONTENT TO REVIEW:
---
{content[:3000]}  # Truncate for token limits
---

EVALUATION CRITERIA:
- Content quality and accuracy
- Alignment with company voice
- Structure and flow
- Engagement and readability
- Actionable value
- Professional presentation

Provide your review in this JSON format:
{{
  "score": <number 0-100>,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "specific_issues": ["issue1", "issue2"],
  "recommendations": ["rec1", "rec2"],
  "approval_recommendation": "approve" or "revise",
  "revision_priority": "high", "medium", or "low"
}}"""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            
            # Parse JSON response
            try:
                review_data = json.loads(response.content.strip())
                return review_data
            except json.JSONDecodeError:
                return self._parse_review_fallback(response.content)
                
        except Exception as e:
            self.logger.warning(f"LLM review failed: {str(e)}")
            return self._get_fallback_review()
    
    def _get_fallback_review(self) -> Dict[str, Any]:
        """Get fallback review when LLM is unavailable."""
        return {
            "score": 70,
            "strengths": ["Content structure", "Topic coverage"],
            "weaknesses": ["Requires manual review"],
            "specific_issues": ["LLM review unavailable"],
            "recommendations": ["Manual editorial review recommended"],
            "approval_recommendation": "revise",
            "revision_priority": "medium"
        }
    
    def _parse_review_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses."""
        score = 70
        
        if "excellent" in response_text.lower():
            score = 90
        elif "good" in response_text.lower():
            score = 80
        elif "needs improvement" in response_text.lower():
            score = 60
        
        return {
            "score": score,
            "strengths": ["Content coverage"],
            "weaknesses": ["Needs review"],
            "specific_issues": [],
            "recommendations": ["Manual review"],
            "approval_recommendation": "revise" if score < 70 else "approve",
            "revision_priority": "medium"
        }
    
    # Content quality check methods
    def _check_content_length(self, content: str, content_type: str) -> float:
        """Check if content length is appropriate."""
        word_count = len(content.split())
        target_ranges = {
            "blog": (1500, 2500),
            "linkedin": (800, 1200),
            "article": (2000, 3000)
        }
        
        min_words, max_words = target_ranges.get(content_type, (1000, 2000))
        
        if min_words <= word_count <= max_words:
            return 100.0
        elif word_count < min_words:
            return max(50.0, (word_count / min_words) * 100)
        else:
            excess_ratio = (word_count - max_words) / max_words
            penalty = min(30, excess_ratio * 50)
            return max(70.0, 100 - penalty)
    
    def _check_formatting(self, content: str) -> float:
        """Check formatting quality."""
        score = 0.0
        
        if "#" in content:  # Headers
            score += 25
        if any(marker in content for marker in ["-", "•", "1.", "2."]):  # Lists
            score += 20
        if any(marker in content for marker in ["**", "*", "_"]):  # Emphasis
            score += 15
        
        paragraph_breaks = content.count("\n\n")
        score += min(20, paragraph_breaks * 2)
        
        if "[" in content and "]" in content:  # Links
            score += 10
        if "```" in content or ">" in content:  # Code blocks or quotes
            score += 10
        
        return min(100.0, score)
    
    def _check_readability(self, content: str) -> float:
        """Check content readability."""
        sentences = content.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 50.0
        
        words = content.split()
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal sentence length is 15-20 words
        if 15 <= avg_sentence_length <= 20:
            sentence_score = 100
        elif 10 <= avg_sentence_length <= 25:
            sentence_score = 80
        else:
            sentence_score = 60
        
        # Check paragraph length
        paragraphs = content.split('\n\n')
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # Optimal paragraph length is 50-100 words
        if 50 <= avg_paragraph_length <= 100:
            paragraph_score = 100
        elif 30 <= avg_paragraph_length <= 120:
            paragraph_score = 80
        else:
            paragraph_score = 60
        
        return (sentence_score + paragraph_score) / 2
    
    def _check_content_structure(self, content: str, content_type: str) -> float:
        """Check content structure."""
        score = 0.0
        
        # Check for introduction
        first_100_words = ' '.join(content.split()[:100]).lower()
        if any(kw in first_100_words for kw in ['introduction', 'begin', 'overview']):
            score += 25
        
        # Check for conclusion
        last_100_words = ' '.join(content.split()[-100:]).lower()
        if any(kw in last_100_words for kw in ['conclusion', 'summary', 'takeaway']):
            score += 25
        
        # Check for headers
        header_count = content.count('#')
        expected_headers = {"blog": 3, "linkedin": 1, "article": 4}.get(content_type, 2)
        if header_count >= expected_headers:
            score += 30
        else:
            score += header_count * 5
        
        # Check for logical flow
        if '\n\n' in content:
            score += 20
        
        return min(100.0, score)
    
    def _check_consistency(self, content: str) -> float:
        """Check content consistency."""
        score = 100.0
        
        # Check header formatting consistency
        headers = [line for line in content.split('\n') if line.startswith('#')]
        if headers:
            header_levels = [len(h) - len(h.lstrip('#')) for h in headers]
            if len(set(header_levels)) > 3:
                score -= 10
        
        return max(60.0, score)
    
    def _assess_clarity(self, content: str) -> float:
        """Assess content clarity."""
        words = content.split()
        complex_words = sum(1 for word in words if len(word) > 12)
        complexity_ratio = complex_words / len(words) if words else 0
        
        return max(60.0, 100 - (complexity_ratio * 200))
    
    def _assess_completeness(self, content: str, content_type: str) -> float:
        """Assess content completeness."""
        word_count = len(content.split())
        target_ranges = {
            "blog": (1500, 2500),
            "linkedin": (800, 1200),
            "article": (2000, 3000)
        }
        
        min_words, max_words = target_ranges.get(content_type, (1000, 2000))
        
        if min_words <= word_count <= max_words:
            return 100.0
        elif word_count < min_words:
            return (word_count / min_words) * 100
        else:
            return max(80.0, 100 - ((word_count - max_words) / max_words * 20))
    
    def _assess_engagement(self, content: str) -> float:
        """Assess content engagement level."""
        engagement_indicators = [
            content.count('?'),  # Questions
            content.count('!'),  # Exclamations
            len([line for line in content.split('\n') if line.strip().startswith('-')]),  # Lists
            content.lower().count('you'),  # Direct address
            content.lower().count('example'),  # Examples
        ]
        
        engagement_score = min(100.0, sum(engagement_indicators) * 10)
        return max(50.0, engagement_score)
    
    def _assess_professionalism(self, content: str) -> float:
        """Assess professional tone."""
        professional_score = 80.0
        
        informal_words = ['gonna', 'wanna', 'kinda', 'sorta', 'yeah', 'nah']
        informal_count = sum(content.lower().count(word) for word in informal_words)
        professional_score -= informal_count * 5
        
        if '#' in content:  # Headers indicate structure
            professional_score += 10
        
        return max(50.0, min(100.0, professional_score))
    
    def _assess_actionability(self, content: str) -> float:
        """Assess how actionable the content is."""
        action_words = ['how to', 'step', 'guide', 'implement', 'apply', 'use', 'start']
        action_score = sum(content.lower().count(word) for word in action_words) * 10
        
        return min(100.0, max(40.0, action_score))
    
    def _generate_suggestions(
        self,
        automated_score: Dict[str, float],
        llm_review: Dict[str, Any],
        quality_dimensions: Dict[str, float]
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Based on automated checks
        if automated_score.get("length_score", 100) < 70:
            suggestions.append("Adjust content length for better coverage")
        if automated_score.get("format_score", 100) < 70:
            suggestions.append("Improve formatting with headers and lists")
        if automated_score.get("readability_score", 100) < 70:
            suggestions.append("Enhance readability with shorter sentences")
        
        # Based on quality dimensions
        if quality_dimensions.get("clarity", 100) < 70:
            suggestions.append("Simplify language for better clarity")
        if quality_dimensions.get("engagement", 100) < 70:
            suggestions.append("Add questions and examples to increase engagement")
        
        # Based on LLM review
        if llm_review.get("recommendations"):
            suggestions.extend(llm_review["recommendations"][:3])
        
        return suggestions
    
    def _compile_revision_notes(
        self,
        overall_score: float,
        llm_review: Dict[str, Any],
        suggestions: List[str],
        dimensions: Dict[str, float]
    ) -> str:
        """Compile detailed revision notes."""
        notes = []
        
        notes.append(f"Overall Quality Score: {overall_score:.1f}/100\n")
        
        if llm_review.get("specific_issues"):
            notes.append("Specific Issues:")
            for issue in llm_review["specific_issues"]:
                notes.append(f"• {issue}")
            notes.append("")
        
        if suggestions:
            notes.append("Improvement Suggestions:")
            for suggestion in suggestions:
                notes.append(f"• {suggestion}")
            notes.append("")
        
        # Add dimension scores
        notes.append("Quality Dimensions:")
        for dimension, score in dimensions.items():
            notes.append(f"• {dimension.title()}: {score:.1f}/100")
        
        return "\n".join(notes)
    
    async def execute_workflow(
        self,
        initial_state: Dict[str, Any],
        context: Optional[LangGraphExecutionContext] = None
    ) -> WorkflowState:
        """Execute the editor workflow."""
        try:
            # Convert input to EditorState
            editor_state = EditorState(
                content=initial_state["content"],
                blog_title=initial_state["blog_title"],
                company_context=initial_state["company_context"],
                content_type=initial_state.get("content_type", "blog"),
                quality_standards=initial_state.get("quality_standards", {}),
                automated_score={},
                llm_review={},
                quality_dimensions={},
                improvement_suggestions=[],
                overall_score=0,
                approved=False,
                revision_notes="",
                confidence_level="",
                current_phase=EditingPhase.INITIAL_REVIEW,
                revision_count=0,
                max_revisions=self.max_revisions,
                phase_results={},
                errors=[]
            )
            
            # Execute the graph
            config = {"configurable": {"thread_id": context.session_id if context else "default"}}
            final_state = await self.graph.ainvoke(editor_state, config)
            
            # Convert to WorkflowState
            return WorkflowState(
                status=WorkflowStatus.COMPLETED if final_state["approved"] else WorkflowStatus.FAILED,
                phase=final_state["current_phase"],
                data={
                    "approved": final_state["approved"],
                    "overall_score": final_state["overall_score"],
                    "confidence_level": final_state["confidence_level"],
                    "revision_notes": final_state.get("revision_notes", ""),
                    "quality_dimensions": final_state.get("quality_dimensions", {}),
                    "improvement_suggestions": final_state.get("improvement_suggestions", []),
                    "phase_results": final_state.get("phase_results", {})
                },
                errors=final_state.get("errors", []),
                metadata={
                    "revision_count": final_state.get("revision_count", 0),
                    "final_phase": final_state["current_phase"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return WorkflowState(
                status=WorkflowStatus.FAILED,
                phase=EditingPhase.INITIAL_REVIEW,
                data={},
                errors=[str(e)],
                metadata={"error_type": type(e).__name__}
            )


# Adapter for backward compatibility
class EditorAgentLangGraph:
    """Adapter to make LangGraph workflow compatible with existing EditorAgent interface."""
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize the adapter."""
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.EDITOR,
                name="EditorAgentLangGraph",
                description="LangGraph-powered content editor with advanced quality assurance",
                capabilities=[
                    "multi-stage_review",
                    "quality_assessment",
                    "revision_planning",
                    "automated_scoring",
                    "llm_review"
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
            
            self.workflow = EditorAgentWorkflow(llm=llm)
            
        except Exception as e:
            # Fallback without LLM
            self.workflow = EditorAgentWorkflow(llm=None)
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Any] = None
    ) -> AgentResult:
        """Execute the editor workflow."""
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
                    "agent_type": "editor_langgraph",
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
                error_code="EDITOR_WORKFLOW_FAILED"
            )