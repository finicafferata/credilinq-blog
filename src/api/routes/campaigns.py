#!/usr/bin/env python3
"""
Campaign API Routes
Handles campaign creation, management, scheduling, and distribution.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.specialized.campaign_manager import CampaignManagerAgent
from src.agents.specialized.task_scheduler import TaskSchedulerAgent
from src.agents.specialized.distribution_agent import DistributionAgent
from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/campaigns", tags=["campaigns"])

# Pydantic models
class CampaignCreateRequest(BaseModel):
    blog_id: str
    campaign_name: str
    company_context: str
    content_type: str = "blog"
    template_id: Optional[str] = None
    template_config: Optional[Dict[str, Any]] = None

class CampaignSummary(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    created_at: str

class CampaignDetail(BaseModel):
    id: str
    name: str
    status: str
    strategy: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    scheduled_posts: List[Dict[str, Any]]
    performance: Dict[str, Any]

class ScheduledPostRequest(BaseModel):
    campaign_id: str

class DistributionRequest(BaseModel):
    campaign_id: str

# Initialize agents
campaign_manager = CampaignManagerAgent()
task_scheduler = TaskSchedulerAgent()
distribution_agent = DistributionAgent()

@router.post("/", response_model=Dict[str, Any])
async def create_campaign(request: CampaignCreateRequest):
    """
    Create a new campaign for a blog post with optional template configuration
    """
    try:
        logger.info(f"Creating campaign for blog {request.blog_id} with template {request.template_id}")
        
        # Create campaign plan with template support
        campaign_plan = await campaign_manager.create_campaign_plan(
            blog_id=request.blog_id,
            campaign_name=request.campaign_name,
            company_context=request.company_context,
            content_type=request.content_type,
            template_id=request.template_id,
            template_config=request.template_config
        )
        
        return {
            "success": True,
            "campaign_id": campaign_plan["campaign_id"],
            "message": "Campaign created successfully",
            "strategy": campaign_plan["strategy"],
            "timeline": campaign_plan["timeline"],
            "tasks": len(campaign_plan["tasks"]),
            "template_id": request.template_id,
            "auto_execute": request.template_id in ["social-blast", "professional-share", "email-campaign"] if request.template_id else False
        }
        
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")

class QuickCampaignRequest(BaseModel):
    blog_id: str
    campaign_name: str

@router.post("/quick/{template_id}", response_model=Dict[str, Any])
async def create_quick_campaign(template_id: str, request: QuickCampaignRequest):
    """
    Create a quick campaign using a predefined template
    """
    try:
        logger.info(f"Creating quick campaign with template {template_id} for blog {request.blog_id}")
        print(f"DEBUG: Quick campaign endpoint called with template {template_id}")  # Debug print
        
        # Define template configurations
        template_configs = {
            "social-blast": {
                "channels": ["linkedin", "twitter", "facebook"],
                "auto_adapt": True,
                "schedule_immediately": True
            },
            "professional-share": {
                "channels": ["linkedin"],
                "format": "professional_article",
                "auto_adapt": True,
                "schedule_immediately": True
            },
            "email-campaign": {
                "channels": ["email"],
                "format": "newsletter",
                "auto_adapt": True,
                "schedule_immediately": False
            }
        }
        
        if template_id not in template_configs:
            raise HTTPException(status_code=400, detail=f"Unknown template: {template_id}")
        
        # Fetch blog info to get company context
        from src.config.database import db_config
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT title, "initialPrompt"
                    FROM "BlogPost" 
                    WHERE id = %s
                """, (request.blog_id,))
                
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                blog_title, initial_prompt = row
                # Extract company context from initial prompt if available
                company_context = ""
                if initial_prompt and isinstance(initial_prompt, dict):
                    company_context = initial_prompt.get('company_context', '')
        except Exception as e:
            logger.warning(f"Could not fetch blog context: {str(e)}")
            company_context = ""

        # Create campaign plan using the campaign manager
        campaign_plan = await campaign_manager.create_campaign_plan(
            blog_id=request.blog_id,
            campaign_name=request.campaign_name,
            company_context=company_context,
            content_type="blog",
            template_id=template_id,
            template_config=template_configs[template_id]
        )
        
        # Auto-execute for simple templates
        auto_executed = False
        if template_configs[template_id].get("schedule_immediately"):
            try:
                await task_scheduler.schedule_campaign_tasks(
                    campaign_plan["campaign_id"], 
                    campaign_plan["strategy"]
                )
                auto_executed = True
                logger.info(f"Auto-scheduled campaign {campaign_plan['campaign_id']}")
            except Exception as e:
                logger.warning(f"Failed to auto-schedule campaign: {str(e)}")
        
        return JSONResponse(content={
            "success": True,
            "campaign_id": campaign_plan["campaign_id"],
            "message": f"Quick campaign '{template_id}' created successfully",
            "template_id": template_id,
            "auto_executed": auto_executed,
            "strategy": campaign_plan["strategy"],
            "tasks": len(campaign_plan["tasks"])
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating quick campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create quick campaign: {str(e)}")

@router.get("/simple-test")
async def simple_test():
    """
    Simple test endpoint
    """
    return {"message": "Hello World"}

@router.get("/test/{template_id}", response_model=Dict[str, Any])
async def test_quick_campaign(template_id: str, blog_id: str = Query(...), campaign_name: str = Query(...)):
    """
    Test endpoint for debugging quick campaign creation
    """
    try:
        return {
            "template_id": template_id,
            "blog_id": blog_id,
            "campaign_name": campaign_name,
            "message": "Test successful"
        }
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/", response_model=List[CampaignSummary])
async def list_campaigns():
    """
    List all campaigns
    """
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT c.id, 
                       COALESCE(b."campaignName", 'Unnamed Campaign') as name,
                       CASE 
                           WHEN COUNT(ct.id) = 0 THEN 'draft'
                           WHEN COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) = COUNT(ct.id) THEN 'completed'
                           ELSE 'active'
                       END as status,
                       c."createdAt",
                       COUNT(ct.id) as total_tasks,
                       COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                FROM "Campaign" c
                LEFT JOIN "Briefing" b ON c.id = b."campaignId"
                LEFT JOIN "CampaignTask" ct ON c.id = ct."campaignId"
                GROUP BY c.id, c."createdAt", b."campaignName"
                ORDER BY c."createdAt" DESC
            """)
            
            rows = cur.fetchall()
            campaigns = []
            
            for row in rows:
                campaign_id, name, status, created_at, total_tasks, completed_tasks = row
                
                progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                
                campaigns.append(CampaignSummary(
                    id=str(campaign_id),
                    name=name or "Untitled Campaign",
                    status=status or "draft",
                    progress=progress,
                    total_tasks=total_tasks or 0,
                    completed_tasks=completed_tasks or 0,
                    created_at=created_at.isoformat() if created_at else datetime.now().isoformat()
                ))
            
            return campaigns
            
    except Exception as e:
        logger.error(f"Error listing campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list campaigns: {str(e)}")

@router.get("/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(campaign_id: str):
    """
    Get detailed information about a campaign
    """
    try:
        # Get campaign status
        campaign_status = await campaign_manager.get_campaign_status(campaign_id)
        
        # Get scheduled posts
        scheduled_posts = await task_scheduler.get_scheduled_posts(campaign_id)
        
        # Get performance metrics
        performance = await distribution_agent.get_campaign_performance(campaign_id)
        
        # Get campaign details from database
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COALESCE(b."campaignName", 'Unnamed Campaign') as name,
                       c."createdAt"
                FROM "Campaign" c
                LEFT JOIN "Briefing" b ON c.id = b."campaignId"
                WHERE c.id = %s
            """, (campaign_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            name, created_at = row
            
            # Get strategy from ContentStrategy table
            cur.execute("""
                SELECT "narrativeApproach", hooks, themes, "toneByChannel", "keyPhrases", notes
                FROM "ContentStrategy"
                WHERE "campaignId" = %s
            """, (campaign_id,))
            
            strategy_row = cur.fetchone()
            if strategy_row:
                strategy = {
                    "narrative_approach": strategy_row[0],
                    "hooks": strategy_row[1],
                    "themes": strategy_row[2],
                    "tone_by_channel": strategy_row[3],
                    "key_phrases": strategy_row[4],
                    "notes": strategy_row[5]
                }
            else:
                strategy = {}
            
            # Get tasks
            cur.execute("""
                SELECT id, "taskType", status, result, error
                FROM "CampaignTask"
                WHERE "campaignId" = %s
                ORDER BY "taskType", "createdAt"
            """, (campaign_id,))
            
            task_rows = cur.fetchall()
            tasks = []
            
            for task_row in task_rows:
                task_id, task_type, status, result, error = task_row
                
                tasks.append({
                    "id": task_id,
                    "task_type": task_type,
                    "status": status,
                    "result": result,
                    "error": error
                })
            
            return CampaignDetail(
                id=campaign_id,
                name=name or "Untitled Campaign",
                status=campaign_status["status"],
                strategy=strategy,
                timeline=[],  # TODO: Add timeline from strategy
                tasks=tasks,
                scheduled_posts=scheduled_posts,
                performance=performance
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign: {str(e)}")

@router.post("/{campaign_id}/schedule", response_model=Dict[str, Any])
async def schedule_campaign(campaign_id: str, request: ScheduledPostRequest):
    """
    Schedule all tasks for a campaign
    """
    try:
        logger.info(f"Scheduling campaign {campaign_id}")
        
        # Get campaign strategy
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT "narrativeApproach", hooks, themes, "toneByChannel", "keyPhrases", notes
                FROM "ContentStrategy"
                WHERE "campaignId" = %s
            """, (campaign_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            if row:
                strategy = {
                    "narrative_approach": row[0],
                    "hooks": row[1],
                    "themes": row[2],
                    "tone_by_channel": row[3],
                    "key_phrases": row[4],
                    "notes": row[5]
                }
            else:
                strategy = {}
        
        # Schedule tasks
        schedule_result = await task_scheduler.schedule_campaign_tasks(campaign_id, strategy)
        
        return {
            "success": True,
            "message": "Campaign scheduled successfully",
            "scheduled_posts": schedule_result["scheduled_posts"],
            "schedule": schedule_result["schedule"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule campaign: {str(e)}")

@router.post("/{campaign_id}/distribute", response_model=Dict[str, Any])
async def distribute_campaign(campaign_id: str, request: DistributionRequest):
    """
    Publish scheduled posts for a campaign
    """
    try:
        logger.info(f"Distributing campaign {campaign_id}")
        
        # Publish scheduled posts
        distribution_result = await distribution_agent.publish_scheduled_posts()
        
        return {
            "success": True,
            "message": "Campaign distribution completed",
            "published": distribution_result["published"],
            "failed": distribution_result["failed"],
            "posts": distribution_result["posts"]
        }
        
    except Exception as e:
        logger.error(f"Error distributing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to distribute campaign: {str(e)}")

@router.get("/{campaign_id}/scheduled-posts", response_model=List[Dict[str, Any]])
async def get_scheduled_posts(campaign_id: str):
    """
    Get all scheduled posts for a campaign
    """
    try:
        scheduled_posts = await task_scheduler.get_scheduled_posts(campaign_id)
        return scheduled_posts
        
    except Exception as e:
        logger.error(f"Error getting scheduled posts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled posts: {str(e)}")

@router.get("/{campaign_id}/performance", response_model=Dict[str, Any])
async def get_campaign_performance(campaign_id: str):
    """
    Get performance metrics for a campaign
    """
    try:
        performance = await distribution_agent.get_campaign_performance(campaign_id)
        return performance
        
    except Exception as e:
        logger.error(f"Error getting campaign performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign performance: {str(e)}")

@router.post("/{campaign_id}/status", response_model=Dict[str, Any])
async def update_campaign_status(campaign_id: str, status: str):
    """
    Update campaign status
    """
    try:
        success = await campaign_manager.update_campaign_status(campaign_id, status)
        
        if success:
            return {
                "success": True,
                "message": f"Campaign status updated to {status}",
                "campaign_id": campaign_id,
                "status": status
            }
        else:
            raise HTTPException(status_code=404, detail="Campaign not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating campaign status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update campaign status: {str(e)}")

@router.post("/publish-due-posts", response_model=Dict[str, Any])
async def publish_due_posts(background_tasks: BackgroundTasks):
    """
    Publish all posts that are due (background task)
    """
    try:
        # Add to background tasks
        background_tasks.add_task(distribution_agent.publish_scheduled_posts)
        
        return {
            "success": True,
            "message": "Background task started to publish due posts"
        }
        
    except Exception as e:
        logger.error(f"Error starting background publish task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start publish task: {str(e)}")

@router.get("/upcoming-posts", response_model=List[Dict[str, Any]])
async def get_upcoming_posts(hours_ahead: int = 24):
    """
    Get posts scheduled for the next N hours
    """
    try:
        upcoming_posts = await task_scheduler.get_upcoming_posts(hours_ahead)
        return upcoming_posts
        
    except Exception as e:
        logger.error(f"Error getting upcoming posts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get upcoming posts: {str(e)}")

@router.post("/{post_id}/track-engagement", response_model=Dict[str, Any])
async def track_post_engagement(post_id: str):
    """
    Track engagement for a specific post
    """
    try:
        engagement_data = await distribution_agent.track_engagement(post_id)
        return engagement_data
        
    except Exception as e:
        logger.error(f"Error tracking engagement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track engagement: {str(e)}")

class TaskStatusUpdate(BaseModel):
    task_id: str
    status: str

@router.put("/{campaign_id}/tasks/{task_id}/status", response_model=Dict[str, Any])
async def update_task_status(campaign_id: str, task_id: str, status_update: TaskStatusUpdate):
    """
    Update the status of a specific task in a campaign
    """
    try:
        logger.info(f"Updating task {task_id} in campaign {campaign_id} to status {status_update.status}")
        
        # Validate status
        valid_statuses = ["pending", "in_progress", "completed"]
        if status_update.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        # Update task status in database
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # First check if task exists and belongs to campaign
            cur.execute("""
                SELECT id FROM "CampaignTask" 
                WHERE id = %s AND "campaignId" = %s
            """, (task_id, campaign_id))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Task not found or doesn't belong to this campaign")
            
            # Update task status
            cur.execute("""
                UPDATE "CampaignTask" 
                SET status = %s 
                WHERE id = %s AND "campaignId" = %s
            """, (status_update.status, task_id, campaign_id))
            
            conn.commit()
            
            # Get updated task
            cur.execute("""
                SELECT id, "taskType", status, result, error
                FROM "CampaignTask"
                WHERE id = %s
            """, (task_id,))
            
            row = cur.fetchone()
            if row:
                task_id_db, task_type, status, content, metadata_json = row
                
                # Handle metadata JSON parsing
                if metadata_json:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    elif isinstance(metadata_json, dict):
                        metadata = metadata_json
                    else:
                        metadata = {}
                else:
                    metadata = {}
                
                return {
                    "success": True,
                    "message": f"Task status updated to {status_update.status}",
                    "task": {
                        "id": task_id_db,
                        "task_type": task_type,
                        "status": status,
                        "content": content,
                        "metadata": metadata
                    }
                }
        
        raise HTTPException(status_code=500, detail="Failed to retrieve updated task")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update task status: {str(e)}")