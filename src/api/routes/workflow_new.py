"""
New Simple Workflow API Routes
"""

from fastapi import APIRouter, HTTPException
import uuid
import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/workflow/start")
async def start_workflow(request: dict):
    """
    Start a new workflow with the given title and company context.
    """
    try:
        logger.info(f"Starting workflow with request: {request}")
        
        workflow_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()
        
        # Create simple response
        response = {
            "workflow_id": workflow_id,
            "current_step": "planner",
            "progress": 0,
            "status": "pending",
            "blog_title": request.get("title", "Unknown"),
            "company_context": request.get("company_context", "Unknown"),
            "content_type": request.get("content_type", "blog"),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        logger.info(f"Started workflow {workflow_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@router.get("/workflow/test")
async def test_workflow():
    """
    Simple test endpoint
    """
    return {"message": "Workflow test endpoint working", "status": "ok"} 