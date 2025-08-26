#!/usr/bin/env python3
"""
Content-First Campaign API Routes

This module provides API endpoints for managing content-first campaigns that generate
cohesive narrative content instead of fragmented tasks. Users get complete content
deliverables (12 blog posts, 3 LinkedIn posts) that flow as a unified story.

Key Features:
- Create content-first campaigns
- Execute narrative-driven workflows  
- Track content deliverables (not tasks)
- Get complete content portfolio
- Narrative flow insights
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.agents.workflow.content_first_workflow import (
    ContentFirstWorkflowOrchestrator,
    ContentDeliverableType, ContentStatus, NarrativePosition
)
from src.agents.core.content_deliverable_service import content_deliverable_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/content-first-campaigns", tags=["Content-First Campaigns"])

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ContentFirstCampaignRequest(BaseModel):
    """Request for creating a content-first campaign"""
    campaign_name: str = Field(..., description="Name of the content campaign")
    campaign_brief: str = Field(..., description="Detailed campaign brief and objectives")
    objectives: List[str] = Field(default=[], description="Campaign objectives")
    target_audience: str = Field(default="B2B professionals", description="Target audience description")
    brand_context: str = Field(..., description="Brand context and voice guidelines")
    content_requirements: Optional[Dict[str, Any]] = Field(default=None, description="Specific content requirements")
    timeline_weeks: int = Field(default=4, description="Campaign timeline in weeks")
    priority: str = Field(default="medium", description="Campaign priority level")

class ContentDeliverableResponse(BaseModel):
    """Response model for content deliverables"""
    content_id: str
    deliverable_type: str
    title: str
    summary: str
    word_count: int
    status: str
    narrative_position: str
    key_message: str
    quality_score: Optional[float]
    completed_at: Optional[datetime]

class ContentPortfolioResponse(BaseModel):
    """Response model for content portfolio"""
    campaign_id: str
    workflow_status: str
    total_deliverables: int
    completed_deliverables: int
    content_pieces: List[ContentDeliverableResponse]
    narrative_summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    execution_time: float

class CampaignStatusResponse(BaseModel):
    """Response model for campaign status"""
    campaign_id: str
    status: str
    current_phase: str
    progress_percentage: float
    deliverables_completed: int
    deliverables_total: int
    estimated_completion: Optional[datetime]

# =============================================================================
# WORKFLOW ORCHESTRATOR INSTANCE
# =============================================================================

workflow_orchestrator = ContentFirstWorkflowOrchestrator()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/create", response_model=Dict[str, Any])
async def create_content_first_campaign(
    request: ContentFirstCampaignRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Create a new content-first campaign that generates cohesive content deliverables
    
    This endpoint:
    1. Creates a campaign focused on content deliverables
    2. Plans a narrative-driven content portfolio 
    3. Executes content generation workflow in background
    4. Returns campaign ID and initial portfolio plan
    """
    try:
        logger.info(f"Creating content-first campaign: {request.campaign_name}")
        
        # Generate campaign ID
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare campaign input
        campaign_input = {
            'campaign_id': campaign_id,
            'campaign_brief': request.campaign_brief,
            'objectives': request.objectives,
            'target_audience': request.target_audience,
            'brand_context': request.brand_context,
            'content_requirements': request.content_requirements or {},
            'timeline_weeks': request.timeline_weeks,
            'priority': request.priority
        }
        
        # Start workflow execution in background
        background_tasks.add_task(
            execute_content_workflow_task, 
            campaign_id, 
            campaign_input
        )
        
        # Return immediate response with campaign details
        return {
            'campaign_id': campaign_id,
            'status': 'initializing',
            'message': f'Content-first campaign "{request.campaign_name}" created successfully',
            'workflow_started': True,
            'estimated_deliverables': '12-15 content pieces',
            'expected_formats': [
                'Blog posts (3-5 pieces)',
                'LinkedIn articles (2-3 pieces)', 
                'Social media posts (4-6 pieces)',
                'Email sequences (2-3 pieces)',
                'Case studies (1-2 pieces)'
            ],
            'narrative_approach': 'Cohesive story-driven content portfolio',
            'created_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create content-first campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Campaign creation failed: {str(e)}")

@router.get("/{campaign_id}/status", response_model=CampaignStatusResponse)
async def get_campaign_status(campaign_id: str) -> CampaignStatusResponse:
    """
    Get the current status of a content-first campaign
    
    Returns workflow progress, deliverable completion status, and timeline estimates
    """
    try:
        logger.info(f"Getting status for campaign: {campaign_id}")
        
        # Get workflow progress
        workflow_progress = await content_deliverable_service.get_workflow_progress(campaign_id)
        
        if not workflow_progress:
            raise HTTPException(status_code=404, detail="Campaign not found or not started")
        
        # Calculate progress percentage
        total_deliverables = workflow_progress.get('total_deliverables', 0)
        completed_deliverables = workflow_progress.get('completed_deliverables', 0)
        progress_percentage = (completed_deliverables / total_deliverables * 100) if total_deliverables > 0 else 0
        
        return CampaignStatusResponse(
            campaign_id=campaign_id,
            status=workflow_progress.get('workflow_status', 'unknown'),
            current_phase=workflow_progress.get('current_phase', ''),
            progress_percentage=progress_percentage,
            deliverables_completed=completed_deliverables,
            deliverables_total=total_deliverables,
            estimated_completion=workflow_progress.get('estimated_completion')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get campaign status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/{campaign_id}/portfolio", response_model=ContentPortfolioResponse)
async def get_content_portfolio(campaign_id: str) -> ContentPortfolioResponse:
    """
    Get the complete content portfolio for a campaign
    
    Returns all content deliverables with their narrative context and relationships
    """
    try:
        logger.info(f"Getting content portfolio for campaign: {campaign_id}")
        
        # Get all deliverables
        deliverables = await content_deliverable_service.get_campaign_deliverables(campaign_id)
        
        if not deliverables:
            raise HTTPException(status_code=404, detail="No content portfolio found for campaign")
        
        # Get narrative context
        narrative_context = await content_deliverable_service.get_narrative_context(campaign_id)
        
        # Get workflow progress for additional metrics
        workflow_progress = await content_deliverable_service.get_workflow_progress(campaign_id)
        
        # Convert deliverables to response format
        content_pieces = []
        for deliverable in deliverables:
            content_pieces.append(ContentDeliverableResponse(
                content_id=deliverable.content_id,
                deliverable_type=deliverable.deliverable_type.value,
                title=deliverable.title,
                summary=deliverable.summary,
                word_count=deliverable.word_count,
                status=deliverable.status.value,
                narrative_position=deliverable.narrative_position.value,
                key_message=deliverable.key_message,
                quality_score=deliverable.quality_score,
                completed_at=deliverable.completed_at
            ))
        
        # Prepare narrative summary
        narrative_summary = {
            'central_theme': narrative_context.central_theme if narrative_context else 'Not defined',
            'key_messages': narrative_context.key_messages if narrative_context else [],
            'supporting_themes': narrative_context.supporting_themes if narrative_context else [],
            'content_relationships': len([d for d in deliverables if d.references_content])
        }
        
        # Calculate quality metrics
        completed_deliverables = [d for d in deliverables if d.status == ContentStatus.PUBLISHED]
        quality_scores = [d.quality_score for d in completed_deliverables if d.quality_score]
        
        quality_metrics = {
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'total_word_count': sum(d.word_count for d in completed_deliverables),
            'completion_rate': len(completed_deliverables) / len(deliverables) if deliverables else 0,
            'content_with_high_quality': len([s for s in quality_scores if s >= 0.8])
        }
        
        return ContentPortfolioResponse(
            campaign_id=campaign_id,
            workflow_status=workflow_progress.get('workflow_status', 'unknown') if workflow_progress else 'unknown',
            total_deliverables=len(deliverables),
            completed_deliverables=len(completed_deliverables),
            content_pieces=content_pieces,
            narrative_summary=narrative_summary,
            quality_metrics=quality_metrics,
            execution_time=0  # Would calculate from workflow timestamps
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio retrieval failed: {str(e)}")

@router.get("/{campaign_id}/deliverables/{content_id}", response_model=Dict[str, Any])
async def get_content_deliverable(campaign_id: str, content_id: str) -> Dict[str, Any]:
    """
    Get a specific content deliverable with full content body and metadata
    
    Returns complete content piece with narrative context and relationships
    """
    try:
        logger.info(f"Getting content deliverable: {content_id}")
        
        deliverable = await content_deliverable_service.get_content_deliverable(content_id)
        
        if not deliverable:
            raise HTTPException(status_code=404, detail="Content deliverable not found")
        
        # Get related content
        related_content = await content_deliverable_service.get_related_content(content_id)
        
        # Get quality reviews
        quality_reviews = await content_deliverable_service.get_quality_reviews(content_id)
        
        return {
            'content_id': deliverable.content_id,
            'deliverable_type': deliverable.deliverable_type.value,
            'title': deliverable.title,
            'content_body': deliverable.content_body,
            'summary': deliverable.summary,
            'word_count': deliverable.word_count,
            'status': deliverable.status.value,
            'narrative_position': deliverable.narrative_position.value,
            'key_message': deliverable.key_message,
            'supporting_points': deliverable.supporting_points,
            'target_audience': deliverable.target_audience,
            'tone': deliverable.tone,
            'channel': deliverable.channel,
            'seo_keywords': deliverable.seo_keywords,
            'call_to_action': deliverable.call_to_action,
            'quality_score': deliverable.quality_score,
            'readability_score': deliverable.readability_score,
            'engagement_prediction': deliverable.engagement_prediction,
            'related_content': related_content,
            'quality_reviews': quality_reviews,
            'created_at': deliverable.created_at.isoformat() if deliverable.created_at else None,
            'updated_at': deliverable.updated_at.isoformat() if deliverable.updated_at else None,
            'completed_at': deliverable.completed_at.isoformat() if deliverable.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content deliverable: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deliverable retrieval failed: {str(e)}")

@router.get("/{campaign_id}/insights", response_model=Dict[str, Any])
async def get_campaign_insights(campaign_id: str) -> Dict[str, Any]:
    """
    Get comprehensive insights about the campaign's content portfolio
    
    Returns analytics on content performance, narrative flow, and quality metrics
    """
    try:
        logger.info(f"Getting campaign insights for: {campaign_id}")
        
        # Get portfolio insights
        insights = await content_deliverable_service.get_content_portfolio_insights(campaign_id)
        
        if 'error' in insights:
            raise HTTPException(status_code=404, detail=insights['error'])
        
        # Get content summary
        content_summary = await content_deliverable_service.get_campaign_content_summary(campaign_id)
        
        # Combine insights
        return {
            **insights,
            'content_summary': content_summary,
            'recommendations': [
                'Consider creating more foundation content for better narrative setup',
                'Add cross-references between related content pieces',
                'Develop more case studies to support transformation messaging'
            ],
            'next_steps': [
                'Review content with low quality scores',
                'Strengthen narrative connections between pieces',
                'Plan distribution schedule for completed content'
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get campaign insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights retrieval failed: {str(e)}")

@router.get("/{campaign_id}/narrative", response_model=Dict[str, Any])
async def get_narrative_context(campaign_id: str) -> Dict[str, Any]:
    """
    Get the narrative context and thematic flow for a campaign
    
    Returns the complete narrative framework that guides content creation
    """
    try:
        logger.info(f"Getting narrative context for campaign: {campaign_id}")
        
        narrative_context = await content_deliverable_service.get_narrative_context(campaign_id)
        
        if not narrative_context:
            raise HTTPException(status_code=404, detail="Narrative context not found for campaign")
        
        return {
            'campaign_id': campaign_id,
            'central_theme': narrative_context.central_theme,
            'supporting_themes': narrative_context.supporting_themes,
            'key_messages': narrative_context.key_messages,
            'target_transformation': narrative_context.target_transformation,
            'brand_voice_guidelines': narrative_context.brand_voice_guidelines,
            'content_journey_map': narrative_context.content_journey_map,
            'terminology_glossary': narrative_context.terminology_glossary,
            'recurring_concepts': narrative_context.recurring_concepts,
            'brand_examples': narrative_context.brand_examples,
            'thematic_connections': narrative_context.thematic_connections
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get narrative context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Narrative context retrieval failed: {str(e)}")

@router.post("/{campaign_id}/regenerate-content/{content_id}")
async def regenerate_content_deliverable(
    campaign_id: str, 
    content_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Regenerate a specific content deliverable with improved quality
    
    Useful for improving content that didn't meet quality standards
    """
    try:
        logger.info(f"Regenerating content deliverable: {content_id}")
        
        # Get existing deliverable
        deliverable = await content_deliverable_service.get_content_deliverable(content_id)
        
        if not deliverable:
            raise HTTPException(status_code=404, detail="Content deliverable not found")
        
        # Start regeneration in background
        background_tasks.add_task(
            regenerate_content_task,
            campaign_id,
            content_id
        )
        
        return {
            'message': f'Content regeneration started for: {deliverable.title}',
            'content_id': content_id,
            'status': 'regenerating',
            'estimated_completion_minutes': 5
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start content regeneration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content regeneration failed: {str(e)}")

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def execute_content_workflow_task(campaign_id: str, campaign_input: Dict[str, Any]):
    """Background task to execute content workflow"""
    try:
        logger.info(f"Starting background workflow execution for campaign: {campaign_id}")
        
        result = await workflow_orchestrator.execute_content_workflow(campaign_input)
        
        if result.get('workflow_status') == 'completed':
            logger.info(f"Content workflow completed for campaign: {campaign_id}")
        else:
            logger.warning(f"Content workflow did not complete successfully: {result}")
            
    except Exception as e:
        logger.error(f"Background workflow execution failed: {str(e)}")

async def regenerate_content_task(campaign_id: str, content_id: str):
    """Background task to regenerate specific content"""
    try:
        logger.info(f"Regenerating content: {content_id}")
        
        # Get deliverable
        deliverable = await content_deliverable_service.get_content_deliverable(content_id)
        if deliverable:
            # Reset status for regeneration
            deliverable.status = ContentStatus.IN_PROGRESS
            deliverable.updated_at = datetime.now()
            
            # Save updated status
            await content_deliverable_service.save_content_deliverable(deliverable)
            
            # Here you would trigger the content generation agent again
            # For now, just simulate regeneration completion
            await asyncio.sleep(10)  # Simulate work
            
            deliverable.status = ContentStatus.DRAFT_COMPLETE
            deliverable.quality_score = 0.9  # Assume improved quality
            deliverable.updated_at = datetime.now()
            
            await content_deliverable_service.save_content_deliverable(deliverable)
            
            logger.info(f"Content regeneration completed: {content_id}")
        
    except Exception as e:
        logger.error(f"Content regeneration failed: {str(e)}")

# =============================================================================
# EXAMPLE USAGE DOCUMENTATION
# =============================================================================

@router.get("/example-usage", response_model=Dict[str, Any])
async def get_example_usage() -> Dict[str, Any]:
    """
    Get example usage patterns for content-first campaigns
    
    Shows how to use the API to create campaigns that produce cohesive content portfolios
    """
    return {
        'content_first_approach': {
            'description': 'Generate cohesive content portfolios instead of fragmented tasks',
            'benefits': [
                'Content pieces flow as unified narrative',
                'Clear relationships between content',
                'Consistent brand voice across all pieces',
                'Complete deliverables, not task fragments'
            ]
        },
        'example_request': {
            'endpoint': 'POST /api/v2/content-first-campaigns/create',
            'payload': {
                'campaign_name': 'Q1 Thought Leadership Campaign',
                'campaign_brief': 'Establish our company as the leading voice in AI-powered business transformation. Target enterprise decision-makers with content that demonstrates our expertise and drives consideration for our solutions.',
                'objectives': [
                    'thought_leadership',
                    'lead_generation', 
                    'brand_awareness'
                ],
                'target_audience': 'Enterprise CTOs and IT decision-makers',
                'brand_context': 'Professional, innovative, results-driven technology leader with proven enterprise experience',
                'timeline_weeks': 6
            }
        },
        'expected_deliverables': {
            'blog_posts': '3-5 comprehensive pieces (1500+ words each)',
            'linkedin_articles': '2-3 professional articles (800+ words each)',
            'social_media_posts': '6-8 engaging posts (150-200 words each)',
            'email_sequences': '2-3 nurture sequences (300+ words each)',
            'case_studies': '1-2 detailed success stories (2000+ words each)',
            'total_content': '12-15 complete, narrative-connected pieces'
        },
        'narrative_flow_example': {
            'foundation_content': 'Blog post introducing core AI transformation concepts',
            'exploration_content': 'LinkedIn articles diving into specific use cases',
            'application_content': 'Case studies showing real implementations',
            'transformation_content': 'Success stories demonstrating outcomes',
            'reinforcement_content': 'Social posts and emails reinforcing key messages'
        },
        'key_differences_from_tasks': {
            'old_approach': 'Generate 21 separate tasks (repurposing, images, distribution)',
            'new_approach': 'Generate 12-15 complete content pieces with narrative flow',
            'user_experience': 'Receive finished content ready for use, not task fragments',
            'content_quality': 'Cohesive story across all pieces, not isolated content',
            'relationships': 'Content references and builds on other pieces naturally'
        }
    }