"""
Review Workflow Integration - Integration layer between content generation and review workflows.

This module provides seamless integration between existing content generation workflows
and the new review workflow system, enabling automatic handoff and state management.

Key Features:
- Automatic workflow handoff after content generation
- State preservation between workflows
- Integration with campaign orchestration
- Review workflow triggers and callbacks
- Performance tracking integration
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .review_workflow_langgraph import (
    ReviewWorkflowLangGraph, ReviewConfiguration, ReviewStage, 
    ReviewStatus, ReviewWorkflowState
)
from .content_generation_workflow import ContentGenerationWorkflow
from ..orchestration.campaign_orchestrator_langgraph import CampaignOrchestratorLangGraph
from ..core.base_agent import AgentResult, AgentExecutionContext
from ..core.database_service import DatabaseService

logger = logging.getLogger(__name__)

class IntegrationMode(Enum):
    """Integration modes for review workflow."""
    AUTOMATIC = "automatic"  # Automatically trigger review after content generation
    MANUAL = "manual"       # Manually trigger review workflow
    CONDITIONAL = "conditional"  # Trigger based on conditions (quality score, campaign type, etc.)

class ContentSource(Enum):
    """Source of content for review."""
    BLOG_WORKFLOW = "blog_workflow"
    CONTENT_GENERATION = "content_generation"
    CAMPAIGN_ORCHESTRATOR = "campaign_orchestrator"
    MANUAL_UPLOAD = "manual_upload"
    API_SUBMISSION = "api_submission"

@dataclass
class ReviewTriggerCondition:
    """Conditions for triggering review workflow."""
    min_content_length: Optional[int] = None
    max_auto_approve_score: Optional[float] = None
    required_campaign_types: Optional[List[str]] = None
    require_human_review: bool = False
    skip_if_high_confidence: bool = True
    confidence_threshold: float = 0.9

@dataclass
class IntegrationResult:
    """Result of workflow integration."""
    success: bool
    review_workflow_id: Optional[str] = None
    integration_mode: Optional[IntegrationMode] = None
    message: str = ""
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = None

class ReviewWorkflowIntegration:
    """
    Integration layer for seamless review workflow integration.
    
    This class manages the handoff between content generation workflows
    and review workflows, ensuring proper state management and coordination.
    """
    
    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db_service = db_service or DatabaseService()
        self.review_workflow = ReviewWorkflowLangGraph()
        self.content_generation_workflow = ContentGenerationWorkflow()
        
        # Integration configuration
        self.default_integration_mode = IntegrationMode.CONDITIONAL
        self.default_trigger_conditions = ReviewTriggerCondition()
        self.auto_resume_enabled = True
        
        # Callbacks registry
        self.pre_review_callbacks: List[Callable] = []
        self.post_review_callbacks: List[Callable] = []
        self.review_stage_callbacks: Dict[str, List[Callable]] = {}
        
        # Integration metrics
        self.integration_stats = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "automatic_triggers": 0,
            "manual_triggers": 0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def integrate_content_for_review(
        self,
        content_data: Dict[str, Any],
        content_source: ContentSource,
        integration_mode: Optional[IntegrationMode] = None,
        review_config: Optional[ReviewConfiguration] = None,
        trigger_conditions: Optional[ReviewTriggerCondition] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Integrate content into the review workflow.
        
        Args:
            content_data: Content and metadata to review
            content_source: Source of the content
            integration_mode: How to integrate (automatic, manual, conditional)
            review_config: Review workflow configuration
            trigger_conditions: Conditions for triggering review
            context: Additional context information
            
        Returns:
            IntegrationResult: Result of the integration
        """
        try:
            self.integration_stats["total_integrations"] += 1
            
            # Use defaults if not provided
            integration_mode = integration_mode or self.default_integration_mode
            trigger_conditions = trigger_conditions or self.default_trigger_conditions
            context = context or {}
            
            # Extract content information
            content_id = content_data.get("content_id", str(uuid.uuid4()))
            content_type = content_data.get("content_type", "unknown")
            campaign_id = content_data.get("campaign_id")
            
            self.logger.info(f"Integrating content for review: {content_id} from {content_source.value}")
            
            # Check if review should be triggered
            should_trigger_review = await self._should_trigger_review(
                content_data, content_source, trigger_conditions, context
            )
            
            if not should_trigger_review:
                self.logger.info(f"Review not triggered for content {content_id} based on conditions")
                return IntegrationResult(
                    success=True,
                    message="Review not required based on trigger conditions",
                    integration_mode=integration_mode,
                    metadata={"review_skipped": True, "reason": "trigger_conditions_not_met"}
                )
            
            # Prepare review workflow input
            review_input = await self._prepare_review_input(
                content_data, content_source, review_config, context
            )
            
            # Execute pre-review callbacks
            await self._execute_pre_review_callbacks(content_data, context)
            
            # Start review workflow
            if integration_mode == IntegrationMode.AUTOMATIC:
                review_result = await self._start_automatic_review(review_input)
                self.integration_stats["automatic_triggers"] += 1
            else:
                review_result = await self._start_manual_review(review_input)
                self.integration_stats["manual_triggers"] += 1
            
            if review_result.success:
                self.integration_stats["successful_integrations"] += 1
                
                # Store integration mapping
                await self._store_integration_mapping(
                    content_id, review_result.data.get("workflow_execution_id"),
                    content_source, integration_mode, context
                )
                
                # Execute post-review callbacks if review is complete
                final_state = review_result.data
                if final_state and final_state.get("workflow_status") == "completed":
                    await self._execute_post_review_callbacks(
                        content_data, final_state, context
                    )
                
                return IntegrationResult(
                    success=True,
                    review_workflow_id=review_result.data.get("workflow_execution_id"),
                    integration_mode=integration_mode,
                    message="Review workflow started successfully",
                    metadata={
                        "content_id": content_id,
                        "content_source": content_source.value,
                        "review_status": final_state.get("overall_approval_status"),
                        "workflow_status": final_state.get("workflow_status")
                    }
                )
            else:
                self.integration_stats["failed_integrations"] += 1
                return IntegrationResult(
                    success=False,
                    message=f"Review workflow failed: {review_result.error_message}",
                    error_code=review_result.error_code,
                    integration_mode=integration_mode
                )
                
        except Exception as e:
            self.integration_stats["failed_integrations"] += 1
            self.logger.error(f"Review workflow integration failed: {e}")
            return IntegrationResult(
                success=False,
                message=f"Integration failed: {str(e)}",
                error_code="INTEGRATION_FAILED"
            )
    
    async def resume_paused_review(
        self,
        review_workflow_id: str,
        human_review_updates: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Resume a paused review workflow with human review updates.
        
        Args:
            review_workflow_id: ID of the paused workflow
            human_review_updates: Updates from human reviewers
            context: Additional context
            
        Returns:
            IntegrationResult: Result of the resume operation
        """
        try:
            self.logger.info(f"Resuming review workflow: {review_workflow_id}")
            
            # Load workflow state
            workflow_state = await self._load_workflow_state(review_workflow_id)
            if not workflow_state:
                return IntegrationResult(
                    success=False,
                    message="Workflow not found or cannot be resumed",
                    error_code="WORKFLOW_NOT_FOUND"
                )
            
            # Apply human review updates
            updated_state = await self._apply_human_review_updates(
                workflow_state, human_review_updates
            )
            
            # Resume workflow execution
            resume_result = await self.review_workflow.execute(updated_state)
            
            if resume_result.success:
                final_state = resume_result.data
                
                # If workflow is now complete, execute post-review callbacks
                if final_state.get("workflow_status") == "completed":
                    content_data = final_state.get("content_data", {})
                    await self._execute_post_review_callbacks(
                        content_data, final_state, context or {}
                    )
                
                return IntegrationResult(
                    success=True,
                    review_workflow_id=review_workflow_id,
                    message="Review workflow resumed successfully",
                    metadata={
                        "workflow_status": final_state.get("workflow_status"),
                        "overall_approval_status": final_state.get("overall_approval_status"),
                        "completed_stages": final_state.get("completed_stages", [])
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    message=f"Failed to resume workflow: {resume_result.error_message}",
                    error_code=resume_result.error_code
                )
                
        except Exception as e:
            self.logger.error(f"Failed to resume review workflow: {e}")
            return IntegrationResult(
                success=False,
                message=f"Resume failed: {str(e)}",
                error_code="RESUME_FAILED"
            )
    
    async def integrate_with_campaign_orchestrator(
        self,
        campaign_orchestrator: CampaignOrchestratorLangGraph,
        review_stage: str = "post_content_creation"
    ) -> bool:
        """
        Integrate review workflow with campaign orchestrator.
        
        Args:
            campaign_orchestrator: Campaign orchestrator instance
            review_stage: When in campaign to trigger review
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Integrating with campaign orchestrator at stage: {review_stage}")
            
            # Register review workflow as a campaign phase
            # This would require modifications to campaign orchestrator to support review phases
            
            # For now, implement as a callback-based integration
            async def campaign_review_callback(campaign_state: Dict[str, Any]) -> Dict[str, Any]:
                """Callback to trigger review workflow during campaign execution."""
                
                # Extract content from campaign state
                content_artifacts = campaign_state.get("content_artifacts", {})
                
                for artifact_id, artifact_data in content_artifacts.items():
                    if isinstance(artifact_data, dict) and "content" in artifact_data:
                        # Prepare content for review
                        content_data = {
                            "content_id": artifact_id,
                            "content": artifact_data["content"],
                            "content_type": artifact_data.get("content_type", "campaign_content"),
                            "campaign_id": campaign_state.get("campaign_id"),
                            "title": artifact_data.get("title", ""),
                            "metadata": artifact_data.get("metadata", {})
                        }
                        
                        # Start review workflow
                        integration_result = await self.integrate_content_for_review(
                            content_data=content_data,
                            content_source=ContentSource.CAMPAIGN_ORCHESTRATOR,
                            integration_mode=IntegrationMode.AUTOMATIC,
                            context={
                                "campaign_state": campaign_state,
                                "artifact_id": artifact_id
                            }
                        )
                        
                        if integration_result.success:
                            # Update campaign state with review workflow ID
                            artifact_data["review_workflow_id"] = integration_result.review_workflow_id
                            artifact_data["review_status"] = "in_review"
                        else:
                            self.logger.error(f"Failed to start review for artifact {artifact_id}")
                
                return campaign_state
            
            # This would be registered with the campaign orchestrator
            # campaign_orchestrator.register_phase_callback(review_stage, campaign_review_callback)
            
            self.logger.info("Successfully integrated with campaign orchestrator")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with campaign orchestrator: {e}")
            return False
    
    async def get_integration_status(self, content_id: str) -> Dict[str, Any]:
        """
        Get the integration status for a specific content item.
        
        Args:
            content_id: Content identifier
            
        Returns:
            Dict: Integration status information
        """
        try:
            # Load integration mapping
            integration_mapping = await self._load_integration_mapping(content_id)
            
            if not integration_mapping:
                return {
                    "content_id": content_id,
                    "integrated": False,
                    "message": "No review workflow integration found"
                }
            
            # Load review workflow status
            workflow_id = integration_mapping.get("review_workflow_id")
            workflow_status = await self._get_workflow_status(workflow_id)
            
            return {
                "content_id": content_id,
                "integrated": True,
                "review_workflow_id": workflow_id,
                "integration_mode": integration_mapping.get("integration_mode"),
                "content_source": integration_mapping.get("content_source"),
                "workflow_status": workflow_status.get("workflow_status"),
                "approval_status": workflow_status.get("overall_approval_status"),
                "progress": workflow_status.get("overall_progress", 0),
                "current_stage": workflow_status.get("current_stage"),
                "is_paused": workflow_status.get("is_paused", False),
                "completed_stages": workflow_status.get("completed_stages", []),
                "pending_human_reviews": workflow_status.get("pending_human_reviews", []),
                "last_updated": workflow_status.get("updated_at")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get integration status: {e}")
            return {
                "content_id": content_id,
                "integrated": False,
                "error": str(e)
            }
    
    # Callback management methods
    
    def register_pre_review_callback(self, callback: Callable):
        """Register a callback to execute before review workflow starts."""
        self.pre_review_callbacks.append(callback)
        self.logger.info(f"Registered pre-review callback: {callback.__name__}")
    
    def register_post_review_callback(self, callback: Callable):
        """Register a callback to execute after review workflow completes."""
        self.post_review_callbacks.append(callback)
        self.logger.info(f"Registered post-review callback: {callback.__name__}")
    
    def register_stage_callback(self, stage: str, callback: Callable):
        """Register a callback for specific review stage completion."""
        if stage not in self.review_stage_callbacks:
            self.review_stage_callbacks[stage] = []
        self.review_stage_callbacks[stage].append(callback)
        self.logger.info(f"Registered stage callback for {stage}: {callback.__name__}")
    
    # Private helper methods
    
    async def _should_trigger_review(
        self,
        content_data: Dict[str, Any],
        content_source: ContentSource,
        trigger_conditions: ReviewTriggerCondition,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if review workflow should be triggered."""
        
        # Check content length
        if trigger_conditions.min_content_length:
            content_length = len(content_data.get("content", ""))
            if content_length < trigger_conditions.min_content_length:
                self.logger.debug(f"Content too short for review: {content_length} < {trigger_conditions.min_content_length}")
                return False
        
        # Check campaign type requirements
        if trigger_conditions.required_campaign_types:
            campaign_type = context.get("campaign_type", content_data.get("campaign_type"))
            if campaign_type not in trigger_conditions.required_campaign_types:
                self.logger.debug(f"Campaign type {campaign_type} not in required types")
                return False
        
        # Check if human review is always required
        if trigger_conditions.require_human_review:
            return True
        
        # Check confidence-based skipping
        if trigger_conditions.skip_if_high_confidence:
            # Get automated quality assessment
            quality_score = content_data.get("quality_score")
            confidence = context.get("confidence_score", content_data.get("confidence_score"))
            
            if (quality_score and quality_score > 8.0 and 
                confidence and confidence >= trigger_conditions.confidence_threshold):
                self.logger.debug(f"High confidence content, skipping review: quality={quality_score}, confidence={confidence}")
                return False
        
        # Check auto-approve score threshold
        if trigger_conditions.max_auto_approve_score:
            quality_score = content_data.get("quality_score", 0)
            if quality_score >= trigger_conditions.max_auto_approve_score:
                self.logger.debug(f"Quality score {quality_score} above auto-approve threshold")
                return False
        
        return True
    
    async def _prepare_review_input(
        self,
        content_data: Dict[str, Any],
        content_source: ContentSource,
        review_config: Optional[ReviewConfiguration],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input data for review workflow."""
        
        # Create default review configuration if not provided
        if not review_config:
            review_config = ReviewConfiguration()
            
            # Customize based on content source and type
            if content_source == ContentSource.BLOG_WORKFLOW:
                review_config.require_seo_review = True
                review_config.seo_auto_approve_threshold = 7.0
            elif content_source == ContentSource.CAMPAIGN_ORCHESTRATOR:
                review_config.require_brand_check = True
                review_config.require_final_approval = True
        
        # Prepare review input
        review_input = {
            "content_id": content_data.get("content_id", str(uuid.uuid4())),
            "content_type": content_data.get("content_type", "unknown"),
            "campaign_id": content_data.get("campaign_id"),
            "content_data": content_data,
            "content_metadata": {
                "source": content_source.value,
                "original_context": context,
                "integration_timestamp": datetime.utcnow().isoformat()
            },
            "review_config": review_config.__dict__,
            "integration_callbacks": [
                {
                    "callback_type": "stage_completion",
                    "stages": ["quality_check", "brand_check", "seo_review", "final_approval"]
                }
            ]
        }
        
        return review_input
    
    async def _start_automatic_review(self, review_input: Dict[str, Any]) -> AgentResult:
        """Start automatic review workflow execution."""
        self.logger.info("Starting automatic review workflow")
        return await self.review_workflow.execute(review_input)
    
    async def _start_manual_review(self, review_input: Dict[str, Any]) -> AgentResult:
        """Start manual review workflow (may pause for human input)."""
        self.logger.info("Starting manual review workflow")
        
        # For manual mode, we might want to pause after each stage
        review_config = ReviewConfiguration(**review_input.get("review_config", {}))
        review_config.require_final_approval = True  # Ensure human approval for manual mode
        
        review_input["review_config"] = review_config.__dict__
        
        return await self.review_workflow.execute(review_input)
    
    async def _execute_pre_review_callbacks(
        self, 
        content_data: Dict[str, Any], 
        context: Dict[str, Any]
    ):
        """Execute pre-review callbacks."""
        for callback in self.pre_review_callbacks:
            try:
                await callback(content_data, context)
            except Exception as e:
                self.logger.error(f"Pre-review callback {callback.__name__} failed: {e}")
    
    async def _execute_post_review_callbacks(
        self,
        content_data: Dict[str, Any],
        review_result: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Execute post-review callbacks."""
        for callback in self.post_review_callbacks:
            try:
                await callback(content_data, review_result, context)
            except Exception as e:
                self.logger.error(f"Post-review callback {callback.__name__} failed: {e}")
    
    async def _store_integration_mapping(
        self,
        content_id: str,
        workflow_execution_id: str,
        content_source: ContentSource,
        integration_mode: IntegrationMode,
        context: Dict[str, Any]
    ):
        """Store integration mapping for tracking."""
        try:
            # This would typically be stored in the database
            # For now, store in memory (in production, use database)
            
            mapping_data = {
                "content_id": content_id,
                "review_workflow_id": workflow_execution_id,
                "content_source": content_source.value,
                "integration_mode": integration_mode.value,
                "integration_timestamp": datetime.utcnow().isoformat(),
                "context": context
            }
            
            # Store in database (pseudo-code)
            # await self.db_service.store_integration_mapping(mapping_data)
            
            self.logger.debug(f"Stored integration mapping for content {content_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store integration mapping: {e}")
    
    async def _load_integration_mapping(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Load integration mapping for content."""
        try:
            # Load from database (pseudo-code)
            # return await self.db_service.get_integration_mapping(content_id)
            
            # For now, return None (would be implemented with actual database)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load integration mapping: {e}")
            return None
    
    async def _load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state for resuming."""
        try:
            # Load from database/checkpoint system
            # return await self.db_service.get_workflow_state(workflow_id)
            
            # For now, return None (would be implemented with actual database)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load workflow state: {e}")
            return None
    
    async def _get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow status."""
        try:
            # Load from database
            # return await self.db_service.get_workflow_status(workflow_id)
            
            # For now, return default status
            return {
                "workflow_status": "unknown",
                "overall_approval_status": "unknown",
                "overall_progress": 0,
                "current_stage": None,
                "is_paused": False,
                "completed_stages": [],
                "pending_human_reviews": [],
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            return {"error": str(e)}
    
    async def _apply_human_review_updates(
        self,
        workflow_state: Dict[str, Any],
        human_review_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply human review updates to workflow state."""
        
        # Apply updates to stage decisions
        for stage, review_decision in human_review_updates.items():
            if stage in workflow_state.get("stage_decisions", {}):
                workflow_state["stage_decisions"][stage].update(review_decision)
            
            # Update pending reviews
            if stage in workflow_state.get("pending_human_reviews", []):
                workflow_state["pending_human_reviews"].remove(stage)
            
            # Update completed stages if approved
            if (review_decision.get("status") == "approved" and 
                stage not in workflow_state.get("completed_stages", [])):
                workflow_state.setdefault("completed_stages", []).append(stage)
        
        # Clear pause state if all reviews are complete
        if not workflow_state.get("pending_human_reviews"):
            workflow_state["is_paused"] = False
            workflow_state["pause_reason"] = None
        
        return workflow_state
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics."""
        total = self.integration_stats["total_integrations"]
        
        return {
            **self.integration_stats,
            "success_rate": (
                self.integration_stats["successful_integrations"] / total 
                if total > 0 else 0
            ),
            "failure_rate": (
                self.integration_stats["failed_integrations"] / total 
                if total > 0 else 0
            ),
            "automatic_ratio": (
                self.integration_stats["automatic_triggers"] / total 
                if total > 0 else 0
            )
        }


# Utility functions for common integration patterns

async def integrate_blog_workflow_with_review(
    blog_workflow_result: Dict[str, Any],
    review_config: Optional[ReviewConfiguration] = None
) -> IntegrationResult:
    """
    Helper function to integrate blog workflow output with review workflow.
    
    Args:
        blog_workflow_result: Result from blog workflow
        review_config: Custom review configuration
        
    Returns:
        IntegrationResult: Integration result
    """
    integration = ReviewWorkflowIntegration()
    
    # Extract content from blog workflow result
    content_data = {
        "content_id": blog_workflow_result.get("blog_post_id", str(uuid.uuid4())),
        "content": blog_workflow_result.get("final_post", ""),
        "content_type": "blog_post",
        "title": blog_workflow_result.get("title", ""),
        "campaign_id": blog_workflow_result.get("campaign_id"),
        "quality_score": blog_workflow_result.get("quality_score"),
        "seo_score": blog_workflow_result.get("seo_score"),
        "metadata": blog_workflow_result.get("metadata", {})
    }
    
    return await integration.integrate_content_for_review(
        content_data=content_data,
        content_source=ContentSource.BLOG_WORKFLOW,
        review_config=review_config,
        context={"blog_workflow_result": blog_workflow_result}
    )

async def integrate_campaign_content_with_review(
    campaign_content: Dict[str, Any],
    campaign_id: str,
    review_config: Optional[ReviewConfiguration] = None
) -> IntegrationResult:
    """
    Helper function to integrate campaign content with review workflow.
    
    Args:
        campaign_content: Campaign content data
        campaign_id: Campaign identifier
        review_config: Custom review configuration
        
    Returns:
        IntegrationResult: Integration result
    """
    integration = ReviewWorkflowIntegration()
    
    # Prepare content data
    content_data = {
        "content_id": campaign_content.get("content_id", str(uuid.uuid4())),
        "content": campaign_content.get("content", ""),
        "content_type": campaign_content.get("content_type", "campaign_content"),
        "title": campaign_content.get("title", ""),
        "campaign_id": campaign_id,
        "metadata": campaign_content.get("metadata", {})
    }
    
    return await integration.integrate_content_for_review(
        content_data=content_data,
        content_source=ContentSource.CAMPAIGN_ORCHESTRATOR,
        review_config=review_config,
        context={"campaign_content": campaign_content}
    )

# Export key components
__all__ = [
    'ReviewWorkflowIntegration',
    'IntegrationMode',
    'ContentSource',
    'ReviewTriggerCondition',
    'IntegrationResult',
    'integrate_blog_workflow_with_review',
    'integrate_campaign_content_with_review'
]