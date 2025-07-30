#!/usr/bin/env python3
"""
Simple test server for campaign endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Campaign Test Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CampaignCreateRequest(BaseModel):
    blog_id: str
    campaign_name: str
    company_context: str
    content_type: str = "blog"

@app.get("/")
async def root():
    return {"message": "Campaign Test Server"}

@app.get("/api/campaigns")
async def list_campaigns():
    """List all campaigns"""
    return [
        {
            "id": "test-campaign-1",
            "name": "Test Campaign 1",
            "status": "draft",
            "progress": 0.0,
            "total_tasks": 5,
            "completed_tasks": 0,
            "created_at": "2025-01-27T10:00:00Z"
        },
        {
            "id": "test-campaign-2",
            "name": "Test Campaign 2",
            "status": "active",
            "progress": 60.0,
            "total_tasks": 3,
            "completed_tasks": 2,
            "created_at": "2025-01-27T11:00:00Z"
        }
    ]

@app.post("/api/campaigns")
async def create_campaign(request: CampaignCreateRequest):
    """Create a new campaign"""
    return {
        "success": True,
        "campaign_id": "new-campaign-id",
        "message": "Campaign created successfully",
        "strategy": {
            "target_audience": "Professional audience",
            "key_messages": ["Message 1", "Message 2"],
            "distribution_channels": ["linkedin", "twitter"]
        },
        "timeline": [],
        "tasks": 5
    }

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get campaign details"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 