"""
Review Workflow Orchestrator - LangGraph-based 8-stage review system
Orchestrates Content Quality → Editorial → Brand → SEO → GEO → Visual → Social Media → Final Approval
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .review_workflow_models import (
    ReviewWorkflowState, ReviewStage, ReviewStatus, ReviewDecision, 
    ReviewCheckpoint, ReviewFeedback, ReviewAgentResult
)
from .review_agent_base import ReviewAgentBase
# Legacy agents replaced with LangGraph adapters
from ..adapters.langgraph_legacy_adapter import AdapterFactory
from ..specialized.editor_agent_langgraph import EditorAgentLangGraph
from ..specialized.seo_agent_langgraph import SEOAgentLangGraph
from ..specialized.geo_analysis_agent_langgraph import GeoAnalysisAgentLangGraph
# Additional LangGraph agents for migration
from ..specialized.social_media_agent_langgraph import SocialMediaAgentLangGraph
from ...config.database import db_config

logger = logging.getLogger(__name__)

class ReviewWorkflowOrchestrator:
    """
    LangGraph-based orchestrator for the 8-stage review workflow.
    Manages the complete flow: Content Quality → Editorial → Brand → SEO → GEO → Visual → Social Media → Final Approval
    """
    
    def __init__(self, checkpointer=None):
        self.workflow_id = "content_review_workflow"
        self.checkpointer = checkpointer or MemorySaver()
        
        # Initialize all specialized review agents with LangGraph adapters
        # Using modern LangGraph agents via adapters for backward compatibility
        self.review_agents = {
            "content_quality": AdapterFactory.create_editor_adapter(),  # ContentQualityAgent → EditorAgent
            "editorial_review": EditorAgentLangGraph(),
            "brand": AdapterFactory.create_brand_review_adapter(),  # BrandReviewAgent → EditorAgent w/ brand focus
            "seo_analysis": SEOAgentLangGraph(),
            "geo_analysis": GeoAnalysisAgentLangGraph(),
            "visual_review": self._create_placeholder_agent("visual_review"), 
            "social_media_review": SocialMediaAgentLangGraph(),
            "final_approval": AdapterFactory.create_editor_adapter()  # FinalApprovalAgent → EditorAgent
        }
        
        # Build the LangGraph workflow
        self.graph = self._build_workflow_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow for 8-stage comprehensive review process"""
        graph = StateGraph(dict)  # Use dict for state (will contain ReviewWorkflowState)
        
        # Add nodes for each review stage - Complete 8-stage workflow
        graph.add_node("content_quality", self._content_quality_node)
        graph.add_node("content_quality_human_review", self._human_review_checkpoint_node)
        graph.add_node("editorial_review", self._editorial_review_node)
        graph.add_node("editorial_human_review", self._human_review_checkpoint_node)
        graph.add_node("brand_check", self._brand_check_node)
        graph.add_node("brand_human_review", self._human_review_checkpoint_node)
        graph.add_node("seo_analysis", self._seo_analysis_node)
        graph.add_node("seo_human_review", self._human_review_checkpoint_node)
        graph.add_node("geo_analysis", self._geo_analysis_node)
        graph.add_node("geo_human_review", self._human_review_checkpoint_node)
        graph.add_node("visual_review", self._visual_review_node)
        graph.add_node("visual_human_review", self._human_review_checkpoint_node)
        graph.add_node("social_media_review", self._social_media_review_node)
        graph.add_node("social_human_review", self._human_review_checkpoint_node)
        graph.add_node("final_approval", self._final_approval_node)
        graph.add_node("approval_human_review", self._human_review_checkpoint_node)
        graph.add_node("workflow_complete", self._completion_node)
        graph.add_node("handle_rejection", self._rejection_handler_node)
        
        # Define workflow edges with conditional routing
        graph.set_entry_point("content_quality")
        
        # Content Quality Flow
        graph.add_conditional_edges(
            "content_quality",
            self._should_require_human_review,
            {
                "human_review": "content_quality_human_review",
                "auto_approved": "editorial_review", 
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "content_quality_human_review",
            self._process_human_decision,
            {
                "approved": "editorial_review",
                "rejected": "handle_rejection",
                "wait": END  # Pause workflow for human input
            }
        )
        
        # Editorial Review Flow
        graph.add_conditional_edges(
            "editorial_review",
            self._should_require_human_review,
            {
                "human_review": "editorial_human_review",
                "auto_approved": "brand_check",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "editorial_human_review",
            self._process_human_decision,
            {
                "approved": "brand_check",
                "rejected": "handle_rejection",
                "wait": END
            }
        )
        
        # Brand Check Flow
        graph.add_conditional_edges(
            "brand_check",
            self._should_require_human_review,
            {
                "human_review": "brand_human_review",
                "auto_approved": "seo_analysis",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "brand_human_review", 
            self._process_human_decision,
            {
                "approved": "seo_analysis",
                "rejected": "handle_rejection",
                "wait": END
            }
        )
        
        # SEO Analysis Flow
        graph.add_conditional_edges(
            "seo_analysis",
            self._should_require_human_review,
            {
                "human_review": "seo_human_review",
                "auto_approved": "geo_analysis",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "seo_human_review",
            self._process_human_decision,
            {
                "approved": "geo_analysis",
                "rejected": "handle_rejection", 
                "wait": END
            }
        )
        
        # GEO Analysis Flow
        graph.add_conditional_edges(
            "geo_analysis",
            self._should_require_human_review,
            {
                "human_review": "geo_human_review",
                "auto_approved": "visual_review",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "geo_human_review",
            self._process_human_decision,
            {
                "approved": "visual_review",
                "rejected": "handle_rejection",
                "wait": END
            }
        )
        
        # Visual Review Flow
        graph.add_conditional_edges(
            "visual_review",
            self._should_require_human_review,
            {
                "human_review": "visual_human_review",
                "auto_approved": "social_media_review",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "visual_human_review",
            self._process_human_decision,
            {
                "approved": "social_media_review",
                "rejected": "handle_rejection",
                "wait": END
            }
        )
        
        # Social Media Review Flow
        graph.add_conditional_edges(
            "social_media_review",
            self._should_require_human_review,
            {
                "human_review": "social_human_review",
                "auto_approved": "final_approval",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "social_human_review",
            self._process_human_decision,
            {
                "approved": "final_approval",
                "rejected": "handle_rejection",
                "wait": END
            }
        )
        
        # Final Approval Flow
        graph.add_conditional_edges(
            "final_approval",
            self._should_require_human_review,
            {
                "human_review": "approval_human_review",
                "auto_approved": "workflow_complete",
                "rejected": "handle_rejection"
            }
        )
        
        graph.add_conditional_edges(
            "approval_human_review",
            self._process_human_decision,
            {
                "approved": "workflow_complete", 
                "rejected": "handle_rejection",
                "wait": END
            }
        )
        
        # Terminal nodes
        graph.add_edge("workflow_complete", END)
        graph.add_edge("handle_rejection", END)
        
        return graph
    
    async def execute_review_workflow(self, initial_state: ReviewWorkflowState) -> ReviewWorkflowState:
        """
        Execute the complete review workflow.
        
        Args:
            initial_state: Initial workflow state with content to review
            
        Returns:
            Final workflow state after completion or pause
        """
        try:
            logger.info(f"Starting review workflow for content {initial_state.content_id}")
            
            # Save initial state to database
            await self._save_workflow_state(initial_state)
            
            # Convert to dict format for LangGraph
            state_dict = initial_state.to_dict()
            
            # Execute the workflow
            result = await self.compiled_graph.ainvoke(
                state_dict,
                config={"configurable": {"thread_id": initial_state.workflow_execution_id}}
            )
            
            # Convert back to ReviewWorkflowState
            final_state = ReviewWorkflowState.from_dict(result)
            
            # Update database with final state
            await self._save_workflow_state(final_state)
            
            logger.info(f"Review workflow completed for {initial_state.content_id}: {final_state.workflow_status.value}")
            return final_state
            
        except Exception as e:
            logger.error(f"Review workflow failed for {initial_state.content_id}: {e}")
            initial_state.workflow_status = ReviewStatus.FAILED
            initial_state.pause_reason = str(e)
            await self._save_workflow_state(initial_state)
            raise
    
    async def resume_workflow(self, workflow_execution_id: str) -> ReviewWorkflowState:
        """
        Resume a paused workflow after human input.
        
        Args:
            workflow_execution_id: Workflow ID to resume
            
        Returns:
            Updated workflow state
        """
        try:
            # Load current state from database
            current_state = await self._load_workflow_state(workflow_execution_id)
            
            if not current_state:
                raise ValueError(f"Workflow {workflow_execution_id} not found")
            
            if not current_state.is_paused:
                logger.info(f"Workflow {workflow_execution_id} is not paused")
                return current_state
            
            # Check for new human feedback
            await self._refresh_human_feedback(current_state)
            
            if current_state.is_paused and not self._has_new_feedback(current_state):
                logger.info(f"Workflow {workflow_execution_id} still waiting for feedback")
                return current_state
            
            # Resume workflow execution
            current_state.is_paused = False
            current_state.pause_reason = None
            
            state_dict = current_state.to_dict()
            
            result = await self.compiled_graph.ainvoke(
                state_dict,
                config={"configurable": {"thread_id": workflow_execution_id}}
            )
            
            final_state = ReviewWorkflowState.from_dict(result)
            await self._save_workflow_state(final_state)
            
            logger.info(f"Resumed workflow {workflow_execution_id}: {final_state.workflow_status.value}")
            return final_state
            
        except Exception as e:
            logger.error(f"Failed to resume workflow {workflow_execution_id}: {e}")
            raise
    
    # ============================================================================
    # WORKFLOW NODE IMPLEMENTATIONS
    # ============================================================================
    
    async def _content_quality_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Content quality node with automated analysis"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.CONTENT_QUALITY
        
        logger.info(f"Starting quality check for content {workflow_state.content_id}")
        
        try:
            # Get content quality agent
            quality_agent = await self._get_review_agent('content_quality')
            
            # Execute quality analysis
            agent_result = await quality_agent.execute_safe(
                workflow_state.content_data,
                workflow_id=workflow_state.workflow_execution_id,
                content_id=workflow_state.content_id,
                stage=ReviewStage.CONTENT_QUALITY.value,
                campaign_id=workflow_state.campaign_id
            )
            
            # agent_result is already a ReviewAgentResult from our review agents
            review_result = agent_result
            
            # Create checkpoint with results
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.CONTENT_QUALITY,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Quality check completed: score={review_result.automated_score}, requires_human={review_result.requires_human_review}")
                
        except Exception as e:
            logger.error(f"Content quality check failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.CONTENT_QUALITY, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _brand_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Brand check node with brand consistency analysis"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.BRAND_CHECK
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.QUALITY_CHECK, success=True)
        
        logger.info(f"Starting brand check for content {workflow_state.content_id}")
        
        try:
            brand_agent = await self._get_review_agent('brand')
            
            agent_result = await brand_agent.execute_safe(
                workflow_state.content_data,
                workflow_id=workflow_state.workflow_execution_id,
                content_id=workflow_state.content_id,
                stage=ReviewStage.BRAND_CHECK.value,
                campaign_id=workflow_state.campaign_id
            )
            
            review_result = agent_result
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.BRAND_CHECK,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Brand check completed: score={review_result.automated_score}")
                
        except Exception as e:
            logger.error(f"Brand check failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.BRAND_CHECK, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _editorial_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Editorial review node with content editing analysis"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.EDITORIAL_REVIEW
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.CONTENT_QUALITY, success=True)
        
        logger.info(f"Starting editorial review for content {workflow_state.content_id}")
        
        try:
            editorial_agent = await self._get_review_agent('editorial_review')
            
            agent_result = await editorial_agent.execute_safe(
                workflow_state.content_data,
                workflow_id=workflow_state.workflow_execution_id,
                content_id=workflow_state.content_id,
                stage=ReviewStage.EDITORIAL_REVIEW.value,
                campaign_id=workflow_state.campaign_id
            )
            
            review_result = agent_result
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.EDITORIAL_REVIEW,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Editorial review completed: score={review_result.automated_score}")
                
        except Exception as e:
            logger.error(f"Editorial review failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.EDITORIAL_REVIEW, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _seo_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """SEO analysis node with optimization analysis"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.SEO_ANALYSIS
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.BRAND_CHECK, success=True)
        
        logger.info(f"Starting SEO analysis for content {workflow_state.content_id}")
        
        try:
            # For now, create a simple SEO review result until we integrate the full LangGraph SEO agent
            from src.agents.workflow.review_workflow_models import ReviewAgentResult
            
            content_text = workflow_state.content_data.get("body", "")
            title = workflow_state.content_data.get("title", "")
            
            # Simple SEO analysis
            word_count = len(content_text.split()) if content_text else 0
            title_length = len(title) if title else 0
            
            # Basic scoring
            seo_score = 0.8  # Default good score
            if word_count < 300:
                seo_score -= 0.2
            if title_length < 30 or title_length > 60:
                seo_score -= 0.1
            if not title:
                seo_score -= 0.3
            
            seo_score = max(0.0, seo_score)
            
            # Create review result
            review_result = ReviewAgentResult(
                stage=ReviewStage.SEO_ANALYSIS,
                content_id=workflow_state.content_id,
                automated_score=seo_score,
                confidence=0.8,
                feedback=[f"Content has {word_count} words", f"Title length: {title_length} characters"],
                recommendations=["Consider SEO optimization" if seo_score < 0.7 else "Good SEO foundations"],
                issues_found=[] if seo_score >= 0.7 else ["Content may need SEO improvements"],
                metrics={"word_count": word_count, "title_length": title_length},
                requires_human_review=seo_score < 0.8,
                auto_approved=seo_score >= 0.8,
                execution_time_ms=50,
                model_used="rule-based",
                tokens_used=0,
                cost=0.0
            )
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.SEO_ANALYSIS,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"SEO analysis completed: score={review_result.automated_score}")
                
        except Exception as e:
            logger.error(f"SEO analysis failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.SEO_ANALYSIS, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _geo_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """GEO analysis node with geographic targeting analysis"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.GEO_ANALYSIS
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.SEO_ANALYSIS, success=True)
        
        logger.info(f"Starting GEO analysis for content {workflow_state.content_id}")
        
        try:
            geo_agent = await self._get_review_agent('geo_analysis')
            
            agent_result = await geo_agent.execute_safe(
                workflow_state.content_data,
                workflow_id=workflow_state.workflow_execution_id,
                content_id=workflow_state.content_id,
                stage=ReviewStage.GEO_ANALYSIS.value,
                campaign_id=workflow_state.campaign_id
            )
            
            review_result = agent_result
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.GEO_ANALYSIS,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"GEO analysis completed: score={review_result.automated_score}")
                
        except Exception as e:
            logger.error(f"GEO analysis failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.GEO_ANALYSIS, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _visual_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Visual review node with image and visual content analysis"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.VISUAL_REVIEW
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.GEO_ANALYSIS, success=True)
        
        logger.info(f"Starting visual review for content {workflow_state.content_id}")
        
        try:
            visual_agent = await self._get_review_agent('visual_review')
            
            agent_result = await visual_agent.execute_safe(
                workflow_state.content_data,
                workflow_id=workflow_state.workflow_execution_id,
                content_id=workflow_state.content_id,
                stage=ReviewStage.VISUAL_REVIEW.value,
                campaign_id=workflow_state.campaign_id
            )
            
            review_result = agent_result
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.VISUAL_REVIEW,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Visual review completed: score={review_result.automated_score}")
                
        except Exception as e:
            logger.error(f"Visual review failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.VISUAL_REVIEW, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _social_media_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Social media review node with social platform optimization"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.SOCIAL_MEDIA_REVIEW
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.VISUAL_REVIEW, success=True)
        
        logger.info(f"Starting social media review for content {workflow_state.content_id}")
        
        try:
            social_agent = await self._get_review_agent('social_media_review')
            
            agent_result = await social_agent.execute_safe(
                workflow_state.content_data,
                workflow_id=workflow_state.workflow_execution_id,
                content_id=workflow_state.content_id,
                stage=ReviewStage.SOCIAL_MEDIA_REVIEW.value,
                campaign_id=workflow_state.campaign_id
            )
            
            review_result = agent_result
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.SOCIAL_MEDIA_REVIEW,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Social media review completed: score={review_result.automated_score}")
                
        except Exception as e:
            logger.error(f"Social media review failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.SOCIAL_MEDIA_REVIEW, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _final_approval_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Final approval node for publication readiness"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        workflow_state.current_stage = ReviewStage.FINAL_APPROVAL
        
        # Complete previous stage
        workflow_state.complete_stage(ReviewStage.SOCIAL_MEDIA_REVIEW, success=True)
        
        logger.info(f"Starting final approval for content {workflow_state.content_id}")
        
        try:
            approval_agent = await self._get_review_agent('final_approval')
            
            # Gather previous review results for final approval analysis
            previous_reviews = {}
            for checkpoint_key, checkpoint in workflow_state.active_checkpoints.items():
                stage_name = checkpoint.stage.value
                previous_reviews[stage_name] = {
                    "automated_score": checkpoint.automated_score,
                    "automated_feedback": checkpoint.automated_feedback,
                    "requires_human_review": checkpoint.requires_human,
                    "issues_found": [],  # Could extract from feedback
                    "recommendations": []  # Could extract from feedback
                }
            
            agent_result = await approval_agent.execute_safe(
                workflow_state.content_data,
                previous_reviews=previous_reviews,
                campaign_context={"campaign_id": workflow_state.campaign_id}
            )
            
            review_result = agent_result
            
            checkpoint = workflow_state.create_checkpoint(
                ReviewStage.FINAL_APPROVAL,
                requires_human=review_result.requires_human_review
            )
            checkpoint.automated_score = review_result.automated_score
            checkpoint.automated_feedback = review_result.feedback
            
            workflow_state.workflow_status = ReviewStatus.AGENT_APPROVED if review_result.auto_approved else ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Final approval completed: score={review_result.automated_score}, requires_human={review_result.requires_human_review}")
                
        except Exception as e:
            logger.error(f"Final approval failed for {workflow_state.content_id}: {e}")
            workflow_state.complete_stage(ReviewStage.FINAL_APPROVAL, success=False)
            workflow_state.workflow_status = ReviewStatus.FAILED
        
        return workflow_state.to_dict()
    
    async def _human_review_checkpoint_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Human review checkpoint with pause/resume functionality"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        
        current_stage = workflow_state.current_stage
        checkpoint_key = f"{current_stage.value}_{workflow_state.content_id}"
        
        logger.info(f"Human review checkpoint for {current_stage.value}")
        
        if checkpoint_key not in workflow_state.active_checkpoints:
            logger.error(f"No checkpoint found for {checkpoint_key}")
            workflow_state.workflow_status = ReviewStatus.FAILED
            return workflow_state.to_dict()
        
        checkpoint = workflow_state.active_checkpoints[checkpoint_key]
        
        # Check for human feedback
        pending_feedback = await self._get_pending_human_feedback(workflow_state.content_id, current_stage)
        
        if pending_feedback:
            # Process human feedback
            workflow_state.add_feedback(pending_feedback)
            
            if pending_feedback.decision == ReviewDecision.APPROVE:
                workflow_state.complete_stage(current_stage, success=True)
                workflow_state.workflow_status = ReviewStatus.HUMAN_APPROVED
                # Remove checkpoint
                del workflow_state.active_checkpoints[checkpoint_key]
                
            elif pending_feedback.decision == ReviewDecision.REJECT:
                workflow_state.complete_stage(current_stage, success=False)
                workflow_state.workflow_status = ReviewStatus.REJECTED
                
            else:  # REQUEST_CHANGES
                workflow_state.workflow_status = ReviewStatus.REQUIRES_HUMAN_REVIEW
                # Keep checkpoint active for follow-up
            
            logger.info(f"Human feedback processed: {pending_feedback.decision.value}")
        else:
            # No human feedback yet - ensure notification sent and pause workflow
            if not checkpoint.notification_sent:
                await self._send_review_notification(checkpoint, workflow_state)
                checkpoint.notification_sent = True
            
            # Pause workflow
            workflow_state.is_paused = True
            workflow_state.pause_reason = f"Waiting for human review: {current_stage.value}"
            workflow_state.workflow_status = ReviewStatus.REQUIRES_HUMAN_REVIEW
            
            logger.info(f"Workflow paused waiting for human review: {current_stage.value}")
        
        return workflow_state.to_dict()
    
    async def _completion_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Workflow completion node"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        
        # Complete final stage
        workflow_state.complete_stage(ReviewStage.FINAL_APPROVAL, success=True)
        
        workflow_state.workflow_status = ReviewStatus.COMPLETED
        workflow_state.completed_at = datetime.utcnow()
        workflow_state.total_review_time_ms = int((workflow_state.completed_at - workflow_state.started_at).total_seconds() * 1000)
        
        logger.info(f"Review workflow completed for {workflow_state.content_id} in {workflow_state.total_review_time_ms}ms")
        
        return workflow_state.to_dict()
    
    async def _rejection_handler_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow rejection"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        
        workflow_state.workflow_status = ReviewStatus.REJECTED
        workflow_state.completed_at = datetime.utcnow()
        workflow_state.total_review_time_ms = int((workflow_state.completed_at - workflow_state.started_at).total_seconds() * 1000)
        
        logger.info(f"Review workflow rejected for {workflow_state.content_id}")
        
        return workflow_state.to_dict()
    
    # ============================================================================
    # CONDITIONAL ROUTING FUNCTIONS
    # ============================================================================
    
    def _should_require_human_review(self, state: Dict[str, Any]) -> str:
        """Determine if human review is required"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        
        if workflow_state.workflow_status == ReviewStatus.FAILED:
            return "rejected"
        elif workflow_state.workflow_status == ReviewStatus.REQUIRES_HUMAN_REVIEW:
            return "human_review"
        elif workflow_state.workflow_status == ReviewStatus.AGENT_APPROVED:
            return "auto_approved"
        else:
            return "human_review"  # Default to human review
    
    def _process_human_decision(self, state: Dict[str, Any]) -> str:
        """Process human review decision"""
        workflow_state = ReviewWorkflowState.from_dict(state)
        
        if workflow_state.workflow_status == ReviewStatus.HUMAN_APPROVED:
            return "approved"
        elif workflow_state.workflow_status == ReviewStatus.REJECTED:
            return "rejected"
        else:
            return "wait"  # Still waiting for human input
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _create_placeholder_agent(self, agent_name: str):
        """Create a placeholder agent for agents with incomplete implementations"""
        class PlaceholderAgent:
            def __init__(self, name):
                self.name = name
            
            async def execute_safe(self, content_data, **kwargs):
                """Mock agent execution with reasonable defaults"""
                from .review_workflow_models import ReviewAgentResult
                return ReviewAgentResult(
                    stage=kwargs.get('stage', self.name),
                    content_id=kwargs.get('content_id', 'unknown'),
                    automated_score=0.8,  # Default good score
                    confidence=0.7,
                    feedback=[f"{self.name.replace('_', ' ').title()} analysis completed"],
                    recommendations=[f"Consider {self.name.replace('_', ' ')} optimization"],
                    issues_found=[],
                    metrics={"agent_type": self.name},
                    requires_human_review=False,
                    auto_approved=True,
                    execution_time_ms=100,
                    model_used="placeholder",
                    tokens_used=0,
                    cost=0.0
                )
        
        return PlaceholderAgent(agent_name)
    
    async def _get_review_agent(self, agent_type: str) -> ReviewAgentBase:
        """Get review agent of specified type"""
        if agent_type not in self.review_agents:
            raise ValueError(f"Unknown review agent type: {agent_type}")
        return self.review_agents[agent_type]
    
    def _parse_agent_result(self, agent_result, stage: ReviewStage) -> ReviewAgentResult:
        """Parse agent result into structured review result"""
        data = agent_result.data or {}
        
        # Handle different agent result formats
        if stage == ReviewStage.SEO_ANALYSIS and "seo_score" in data:
            # SEO agent has different data structure
            automated_score = data.get("seo_score", 0.8) / 100.0  # Convert from 0-100 to 0-1
            feedback = data.get("implementation_plan", [])
            recommendations = data.get("optimization_suggestions", [])
            issues_found = data.get("content_gaps", [])
            metrics = data.get("technical_metrics", {})
        else:
            # Standard review agent format
            automated_score = data.get("automated_score", 0.8)
            feedback = data.get("feedback", [])
            recommendations = data.get("recommendations", [])
            issues_found = data.get("issues_found", [])
            metrics = data.get("metrics", {})
        
        confidence = data.get("confidence", 0.8)
        
        return ReviewAgentResult(
            stage=stage,
            content_id="",  # Will be set by caller
            automated_score=automated_score,
            confidence=confidence,
            feedback=feedback,
            recommendations=recommendations,
            issues_found=issues_found,
            metrics=metrics,
            requires_human_review=data.get("requires_human_review", automated_score < 0.85),
            auto_approved=data.get("auto_approved", automated_score >= 0.85),
            execution_time_ms=int(agent_result.metadata.get("execution_time_ms", 0)) if agent_result.metadata else 0,
            model_used=agent_result.metadata.get("model_used", "gemini-1.5-flash") if agent_result.metadata else "gemini-1.5-flash",
            tokens_used=agent_result.metadata.get("tokens_used", 0) if agent_result.metadata else 0,
            cost=agent_result.metadata.get("cost", 0.0) if agent_result.metadata else 0.0
        )
    
    async def _save_workflow_state(self, workflow_state: ReviewWorkflowState):
        """Save workflow state to database"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Upsert workflow record
                cur.execute("""
                    INSERT INTO content_review_workflows (
                        id, content_id, content_type, campaign_id,
                        workflow_execution_id, current_stage, workflow_status,
                        content_data, is_paused, pause_reason,
                        auto_approve_threshold, require_human_approval, parallel_reviews,
                        started_at, completed_at, total_review_time_ms
                    ) VALUES (
                        gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (workflow_execution_id) 
                    DO UPDATE SET
                        current_stage = EXCLUDED.current_stage,
                        workflow_status = EXCLUDED.workflow_status,
                        is_paused = EXCLUDED.is_paused,
                        pause_reason = EXCLUDED.pause_reason,
                        completed_at = EXCLUDED.completed_at,
                        total_review_time_ms = EXCLUDED.total_review_time_ms
                """, (
                    workflow_state.content_id,
                    workflow_state.content_type,
                    workflow_state.campaign_id,
                    workflow_state.workflow_execution_id,
                    workflow_state.current_stage.value,
                    workflow_state.workflow_status.value,
                    json.dumps(workflow_state.content_data),
                    workflow_state.is_paused,
                    workflow_state.pause_reason,
                    workflow_state.auto_approve_threshold,
                    workflow_state.require_human_approval,
                    workflow_state.parallel_reviews,
                    workflow_state.started_at,
                    workflow_state.completed_at,
                    workflow_state.total_review_time_ms
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving workflow state: {e}")
    
    async def _load_workflow_state(self, workflow_execution_id: str) -> Optional[ReviewWorkflowState]:
        """Load workflow state from database"""
        # Implementation would query database and reconstruct ReviewWorkflowState
        # For now, return None (will implement when adding database integration)
        return None
    
    async def _get_pending_human_feedback(self, content_id: str, stage: ReviewStage) -> Optional[ReviewFeedback]:
        """Check for pending human feedback"""
        # Implementation would query database for recent feedback
        # For now, return None (will implement when adding human review interface)
        return None
    
    async def _refresh_human_feedback(self, workflow_state: ReviewWorkflowState):
        """Refresh human feedback from database"""
        # Implementation would update workflow_state with latest feedback
        pass
    
    def _has_new_feedback(self, workflow_state: ReviewWorkflowState) -> bool:
        """Check if workflow has received new feedback"""
        # Implementation would check if new feedback exists since last check
        return False
    
    async def _send_review_notification(self, checkpoint: ReviewCheckpoint, workflow_state: ReviewWorkflowState):
        """Send notification to assigned reviewer"""
        # Implementation would send email/slack/dashboard notification
        logger.info(f"Notification sent for {checkpoint.stage.value} review")
        pass


# Global instance
review_workflow_orchestrator = ReviewWorkflowOrchestrator()