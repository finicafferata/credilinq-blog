#!/usr/bin/env python3
"""
Simple Campaign API Routes for testing
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])

# Simple Pydantic models
class CampaignCreateRequest(BaseModel):
    blog_id: str
    campaign_name: str
    company_context: str
    content_type: str = "blog"

class CampaignSummary(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    created_at: str

@router.get("/", response_model=List[CampaignSummary])
async def list_campaigns():
    """
    List all campaigns (simplified)
    """
    try:
        # Return mock data for now
        return [
            CampaignSummary(
                id="test-campaign-1",
                name="Test Campaign 1",
                status="draft",
                progress=0.0,
                total_tasks=5,
                completed_tasks=0,
                created_at="2025-01-27T10:00:00Z"
            ),
            CampaignSummary(
                id="test-campaign-2",
                name="Test Campaign 2",
                status="active",
                progress=60.0,
                total_tasks=3,
                completed_tasks=2,
                created_at="2025-01-27T11:00:00Z"
            )
        ]
    except Exception as e:
        logger.error(f"Error listing campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list campaigns: {str(e)}")

@router.post("/", response_model=Dict[str, Any])
async def create_campaign(request: CampaignCreateRequest):
    """
    Create a new campaign (simplified)
    """
    try:
        logger.info(f"Creating campaign for blog {request.blog_id}")
        
        return {
            "success": True,
            "campaign_id": "new-campaign-id",
            "message": "Campaign created successfully (simplified)",
            "strategy": {
                "target_audience": "Professional audience",
                "key_messages": ["Message 1", "Message 2"],
                "distribution_channels": ["linkedin", "twitter"]
            },
            "timeline": [],
            "tasks": 5
        }
        
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")

@router.get("/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign(campaign_id: str):
    """
    Get campaign details (simplified)
    """
    try:
        return {
            "id": campaign_id,
            "name": f"Campaign {campaign_id}",
            "status": "draft",
            "strategy": {
                "target_audience": "Professional audience",
                "key_messages": ["Message 1", "Message 2"],
                "distribution_channels": ["linkedin", "twitter"]
            },
            "timeline": [],
            "tasks": [],
            "scheduled_posts": [],
            "performance": {
                "total_posts": 0,
                "published_posts": 0,
                "success_rate": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign: {str(e)}")

@router.post("/{campaign_id}/schedule", response_model=Dict[str, Any])
async def schedule_campaign(campaign_id: str):
    """
    Schedule campaign tasks (simplified)
    """
    try:
        return {
            "success": True,
            "message": "Campaign scheduled successfully (simplified)",
            "scheduled_posts": 3,
            "schedule": []
        }
        
    except Exception as e:
        logger.error(f"Error scheduling campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule campaign: {str(e)}")

@router.post("/{campaign_id}/distribute", response_model=Dict[str, Any])
async def distribute_campaign(campaign_id: str):
    """
    Distribute campaign posts (simplified)
    """
    try:
        return {
            "success": True,
            "message": "Campaign distribution completed (simplified)",
            "published": 2,
            "failed": 0,
            "posts": []
        }
        
    except Exception as e:
        logger.error(f"Error distributing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to distribute campaign: {str(e)}") 