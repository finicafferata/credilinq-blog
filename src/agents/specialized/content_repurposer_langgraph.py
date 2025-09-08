"""
LangGraph-based Content Repurposer Agent with advanced multi-format workflow.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class ContentRepurposerState(WorkflowState):
    """State for Content Repurposer LangGraph workflow."""
    original_content: str = ""
    target_formats: List[str] = field(default_factory=list)
    target_audience: str = "general audience"
    brand_voice: str = "professional and engaging"
    
    # Processing results
    repurposed_content: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

class ContentRepurposerAgentLangGraph(LangGraphWorkflowBase[ContentRepurposerState]):
    """LangGraph-based Content Repurposer with sophisticated workflow."""
    
    def __init__(self, workflow_name: str = "ContentRepurposer_workflow"):
        super().__init__(workflow_name=workflow_name)
        logger.info("ContentRepurposerAgentLangGraph initialized")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(ContentRepurposerState)
        
        # Simple workflow for now
        workflow.add_node("repurpose_content", self._repurpose_content)
        workflow.add_node("assess_quality", self._assess_quality)
        workflow.add_node("finalize_output", self._finalize_output)
        
        workflow.set_entry_point("repurpose_content")
        workflow.add_edge("repurpose_content", "assess_quality")
        workflow.add_edge("assess_quality", "finalize_output")
        workflow.set_finish_point("finalize_output")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> ContentRepurposerState:
        """Create initial workflow state."""
        return ContentRepurposerState(
            original_content=input_data.get("original_content", ""),
            target_formats=input_data.get("target_formats", []),
            target_audience=input_data.get("target_audience", "general audience"),
            brand_voice=input_data.get("brand_voice", "professional and engaging"),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="repurpose_content"
        )
    
    def _repurpose_content(self, state: ContentRepurposerState) -> ContentRepurposerState:
        """Repurpose content for target formats."""
        logger.info(f"Repurposing content for {len(state.target_formats)} formats")
        
        repurposed = {}
        for format_type in state.target_formats:
            if format_type == "social_media_post":
                repurposed[format_type] = {
                    "content": f"🚀 {state.original_content[:200]}... #innovation",
                    "hashtags": ["#innovation", "#growth"],
                    "character_count": 250
                }
            elif format_type == "email_newsletter":
                repurposed[format_type] = {
                    "subject_line": "Important Update",
                    "content": f"Dear Reader,\n\n{state.original_content[:400]}...\n\nBest regards",
                    "character_count": len(state.original_content[:400])
                }
            else:
                repurposed[format_type] = {
                    "content": state.original_content[:500],
                    "format_type": format_type,
                    "character_count": len(state.original_content[:500])
                }
        
        state.repurposed_content = repurposed
        state.current_step = "assess_quality"
        return state
    
    def _assess_quality(self, state: ContentRepurposerState) -> ContentRepurposerState:
        """Assess quality of repurposed content."""
        logger.info("Assessing content quality")
        
        quality_scores = {}
        for format_type in state.repurposed_content.keys():
            quality_scores[format_type] = 0.85  # Simulated quality score
        
        state.quality_scores = quality_scores
        state.current_step = "finalize_output"
        return state
    
    def _finalize_output(self, state: ContentRepurposerState) -> ContentRepurposerState:
        """Finalize repurposed content."""
        logger.info("Finalizing repurposed content")
        
        state.metadata.update({
            "total_formats": len(state.target_formats),
            "successful_adaptations": len(state.repurposed_content),
            "average_quality_score": sum(state.quality_scores.values()) / len(state.quality_scores) if state.quality_scores else 0,
            "processing_complete": True
        })
        
        state.current_step = "completed"
        return state