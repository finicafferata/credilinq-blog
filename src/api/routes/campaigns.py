"""Campaign management endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import uuid
import datetime

from ...config.database import db_config
from ...core.exceptions import (
    AgentExecutionError, DatabaseQueryError, RecordNotFoundError,
    convert_to_http_exception
)
from ..models.campaign import (
    CampaignCreateRequest, CampaignTaskExecuteRequest, 
    CampaignTaskUpdateRequest, CampaignTaskResponse, CampaignResponse
)

router = APIRouter()


@router.post("/campaigns", response_model=CampaignResponse)
def create_campaign(request: CampaignCreateRequest):
    """Create a campaign for a given blog post."""
    # Check if campaign already exists for this blog_id
    existing = db_config.supabase.table("campaign").select("*").eq("blog_id", request.blog_id).single().execute()
    if getattr(existing, 'data', None):
        # Fetch tasks
        campaign = existing.data
        tasks_resp = db_config.supabase.table("campaign_task").select("*").eq("campaign_id", campaign["id"]).execute()
        tasks = getattr(tasks_resp, 'data', [])
        return CampaignResponse(
            id=campaign["id"],
            blog_id=campaign["blog_id"],
            created_at=campaign["created_at"],
            tasks=[CampaignTaskResponse(
                id=task["id"],
                task_type=task["task_type"],
                target_format=task.get("target_format"),
                target_asset=task.get("target_asset"),
                status=task["status"],
                result=task.get("result"),
                image_url=task.get("image_url"),
                error=task.get("error"),
                created_at=task["created_at"],
                updated_at=task["updated_at"]
            ) for task in tasks]
        )
    
    # Call CampaignManagerAgent to generate plan
    try:
        from ...agents.specialized.campaign_manager import CampaignManagerAgent
        agent = CampaignManagerAgent()
        plan = agent.execute(request.blog_id)
    except Exception as e:
        raise convert_to_http_exception(AgentExecutionError("CampaignManager", "campaign_generation", f"Campaign generation failed: {str(e)}"))
    
    # Create campaign
    campaign_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    campaign_data = {
        "id": campaign_id,
        "blog_id": request.blog_id,
        "created_at": created_at
    }
    resp = db_config.supabase.table("campaign").insert(campaign_data).execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to create campaign: {resp}"))
    
    # Create tasks
    tasks = []
    for task in plan:
        task_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        task_data = {
            "id": task_id,
            "campaign_id": campaign_id,
            "task_type": task["task_type"],
            "target_format": task.get("target_format"),
            "target_asset": task.get("target_asset"),
            "status": task["status"],
            "result": task.get("result"),
            "image_url": task.get("image_url"),
            "error": None,
            "created_at": now,
            "updated_at": now
        }
        db_config.supabase.table("campaign_task").insert(task_data).execute()
        tasks.append(CampaignTaskResponse(
            id=task_id,
            task_type=task["task_type"],
            target_format=task.get("target_format"),
            target_asset=task.get("target_asset"),
            status=task["status"],
            result=task.get("result"),
            image_url=task.get("image_url"),
            error=None,
            created_at=now,
            updated_at=now
        ))
    
    return CampaignResponse(
        id=campaign_id,
        blog_id=request.blog_id,
        created_at=created_at,
        tasks=tasks
    )


@router.get("/campaigns/{blog_id}", response_model=CampaignResponse)
def get_campaign(blog_id: str):
    """Fetch the current state of the campaign plan for a given blog."""
    # Fetch campaign by blog_id
    campaign_resp = db_config.supabase.table("campaign").select("*").eq("blog_id", blog_id).single().execute()
    campaign = getattr(campaign_resp, 'data', None)
    if not campaign:
        raise convert_to_http_exception(RecordNotFoundError(f"Campaign not found for blog_id {blog_id}"))
    
    # Fetch tasks for this campaign
    tasks_resp = db_config.supabase.table("campaign_task").select("*").eq("campaign_id", campaign["id"]).execute()
    tasks = getattr(tasks_resp, 'data', [])
    
    # Build response
    return CampaignResponse(
        id=campaign["id"],
        blog_id=campaign["blog_id"],
        created_at=campaign["created_at"],
        tasks=[CampaignTaskResponse(
            id=task["id"],
            task_type=task["task_type"],
            target_format=task.get("target_format"),
            target_asset=task.get("target_asset"),
            status=task["status"],
            result=task.get("result"),
            image_url=task.get("image_url"),
            error=task.get("error"),
            created_at=task["created_at"],
            updated_at=task["updated_at"]
        ) for task in tasks]
    )


@router.post("/campaigns/tasks/execute", status_code=202)
def execute_campaign_task(request: CampaignTaskExecuteRequest, background_tasks: BackgroundTasks):
    """Execute a specific campaign task asynchronously."""
    # Mark task as In Progress
    task_resp = db_config.supabase.table("campaign_task").select("*").eq("id", request.task_id).single().execute()
    task = getattr(task_resp, 'data', None)
    if not task:
        raise convert_to_http_exception(RecordNotFoundError(f"Campaign task {request.task_id} not found"))
    
    db_config.supabase.table("campaign_task").update({"status": "in_progress"}).eq("id", request.task_id).execute()

    def run_agent_and_update():
        try:
            # Fetch campaign and blog context
            campaign_resp = db_config.supabase.table("campaign").select("*").eq("id", task["campaign_id"]).single().execute()
            campaign = getattr(campaign_resp, 'data', None)
            blog_id = campaign["blog_id"] if campaign else None
            
            # Import agents
            from ...agents.specialized.repurpose_agent import ContentRepurposingAgent
            from ...agents.specialized.image_agent import ImagePromptAgent
            from ...agents.specialized.content_agent import ContentGenerationAgent
            
            # Select agent based on taskType
            result = None
            if task["task_type"] == "repurpose":
                # Fetch original blog content
                blog_resp = db_config.supabase.table("blog_posts").select("content_markdown").eq("id", blog_id).single().execute()
                blog = getattr(blog_resp, 'data', None)
                original_content = blog["content_markdown"] if blog else ""
                agent = ContentRepurposingAgent()
                result = agent.execute(original_content, task.get("target_format", ""))
            elif task["task_type"] == "create_image_prompt":
                # Fetch blog content for context
                blog_resp = db_config.supabase.table("blog_posts").select("title, content_markdown").eq("id", blog_id).single().execute()
                blog = getattr(blog_resp, 'data', None)
                content_topic = blog["title"] if blog else ""
                content_body = blog["content_markdown"] if blog else ""
                agent = ImagePromptAgent()
                result = agent.execute(content_topic, content_body)
            elif task["task_type"] == "generate_content":
                agent = ContentGenerationAgent()
                result = agent.execute("", "")
            else:
                raise Exception(f"Unknown task_type: {task['task_type']}")
                
            # Update task with result
            db_config.supabase.table("campaign_task").update({
                "result": result,
                "status": "completed",
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "error": None
            }).eq("id", request.task_id).execute()
            
        except Exception as e:
            # Update task with error
            db_config.supabase.table("campaign_task").update({
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.datetime.utcnow().isoformat()
            }).eq("id", request.task_id).execute()

    background_tasks.add_task(run_agent_and_update)
    return {"message": "Task execution started", "task_id": request.task_id}


@router.put("/campaigns/tasks/{task_id}", response_model=CampaignTaskResponse)
def update_campaign_task(task_id: str, request: CampaignTaskUpdateRequest):
    """Update the content and status of a campaign task."""
    # Update task result and status
    update_data = {
        "status": request.status,
        "updated_at": datetime.datetime.utcnow().isoformat()
    }
    if request.content is not None:
        update_data["result"] = request.content
        
    resp = db_config.supabase.table("campaign_task").update(update_data).eq("id", task_id).execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update campaign task: {resp}"))
    
    # Fetch updated task
    task_resp = db_config.supabase.table("campaign_task").select("*").eq("id", task_id).single().execute()
    task = getattr(task_resp, 'data', None)
    if not task:
        raise convert_to_http_exception(RecordNotFoundError(f"Campaign task {task_id} not found after update"))
        
    return CampaignTaskResponse(
        id=task["id"],
        task_type=task["task_type"],
        target_format=task.get("target_format"),
        target_asset=task.get("target_asset"),
        status=task["status"],
        result=task.get("result"),
        image_url=task.get("image_url"),
        error=task.get("error"),
        created_at=task["created_at"],
        updated_at=task["updated_at"]
    )