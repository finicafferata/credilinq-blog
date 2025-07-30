"""
Simple Image Agent API Routes for debugging
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify router is working."""
    return {"message": "Images router is working!"}

@router.get("/blogs")
async def get_available_blogs():
    """
    Get list of available blogs for image generation.
    """
    return {
        "blogs": [
            {
                "id": "test-id-1",
                "title": "Test Blog 1",
                "status": "draft",
                "created_at": "2025-07-29 20:00:00"
            },
            {
                "id": "test-id-2", 
                "title": "Test Blog 2",
                "status": "published",
                "created_at": "2025-07-29 19:00:00"
            }
        ]
    } 