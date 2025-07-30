"""
Simple Workflow API Routes for testing
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/workflow/start")
async def start_workflow():
    """
    Simple test endpoint
    """
    try:
        logger.info("Starting workflow test")
        return {"message": "Workflow started successfully", "status": "ok"}
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@router.get("/workflow/test")
async def test_workflow():
    """
    Simple test endpoint
    """
    return {"message": "Workflow test endpoint working", "status": "ok"} 