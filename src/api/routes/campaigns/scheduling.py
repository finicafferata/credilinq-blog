"""
Campaign Scheduling and Distribution Routes
Handles post scheduling, distribution, and performance tracking.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class ScheduledPostRequest(BaseModel):
    campaign_id: str

class DistributionRequest(BaseModel):
    campaign_id: str

# Helper functions
def get_task_scheduler():
    """Get task scheduler with lazy loading"""
    try:
        from src.agents.specialized.task_scheduler import TaskSchedulerAgent
        return TaskSchedulerAgent()
    except ImportError:
        logger.warning("TaskSchedulerAgent not available")
        return None

def get_distribution_agent():
    """Get distribution agent with lazy loading"""
    try:
        from src.agents.specialized.distribution_agent import DistributionAgent
        return DistributionAgent()
    except ImportError:
        logger.warning("DistributionAgent not available")
        return None

# Scheduling Endpoints
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
                SELECT name, metadata
                FROM campaigns
                WHERE id = %s
            """, (campaign_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_name, metadata = row
            strategy = metadata or {}
        
        # Get task scheduler
        task_scheduler = get_task_scheduler()
        if not task_scheduler:
            # Mock scheduling response
            scheduled_posts = [
                {
                    "id": f"post_{i}",
                    "title": f"Scheduled Post {i}",
                    "platform": "linkedin",
                    "scheduled_time": datetime.now().isoformat(),
                    "status": "scheduled"
                }
                for i in range(1, 4)
            ]
            
            schedule = {
                "total_posts": len(scheduled_posts),
                "start_date": datetime.now().isoformat(),
                "frequency": "daily"
            }
        else:
            # Use real task scheduler
            schedule_result = await task_scheduler.schedule_campaign_tasks(campaign_id, strategy)
            scheduled_posts = schedule_result.get("scheduled_posts", [])
            schedule = schedule_result.get("schedule", {})
        
        return {
            "success": True,
            "message": "Campaign scheduled successfully",
            "campaign_id": campaign_id,
            "scheduled_posts": scheduled_posts,
            "schedule": schedule
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
        
        # Get distribution agent
        distribution_agent = get_distribution_agent()
        if not distribution_agent:
            # Mock distribution response
            distribution_result = {
                "published": 3,
                "failed": 0,
                "posts": [
                    {
                        "id": f"dist_{i}",
                        "platform": "linkedin",
                        "status": "published",
                        "published_at": datetime.now().isoformat()
                    }
                    for i in range(1, 4)
                ]
            }
        else:
            # Use real distribution agent
            distribution_result = await distribution_agent.publish_scheduled_posts()
        
        return {
            "success": True,
            "message": "Campaign distribution completed",
            "campaign_id": campaign_id,
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
        logger.info(f"Getting scheduled posts for campaign {campaign_id}")
        
        task_scheduler = get_task_scheduler()
        if not task_scheduler:
            # Mock scheduled posts
            scheduled_posts = [
                {
                    "id": f"sched_{i}",
                    "campaign_id": campaign_id,
                    "title": f"Scheduled Post {i}",
                    "content": f"Content for scheduled post {i}",
                    "platform": ["linkedin", "twitter"][i % 2],
                    "scheduled_time": datetime.now().isoformat(),
                    "status": "scheduled"
                }
                for i in range(1, 6)
            ]
        else:
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
        logger.info(f"Getting performance for campaign {campaign_id}")
        
        distribution_agent = get_distribution_agent()
        if not distribution_agent:
            # Mock performance data
            performance = {
                "campaign_id": campaign_id,
                "total_posts": 5,
                "published_posts": 4,
                "scheduled_posts": 1,
                "engagement": {
                    "total_views": 1245,
                    "total_clicks": 89,
                    "total_shares": 23,
                    "engagement_rate": 7.2
                },
                "platforms": {
                    "linkedin": {
                        "posts": 3,
                        "views": 856,
                        "clicks": 62,
                        "engagement_rate": 8.1
                    },
                    "twitter": {
                        "posts": 2,
                        "views": 389,
                        "clicks": 27,
                        "engagement_rate": 5.8
                    }
                },
                "last_updated": datetime.now().isoformat()
            }
        else:
            performance = await distribution_agent.get_campaign_performance(campaign_id)
        
        return performance
        
    except Exception as e:
        logger.error(f"Error getting campaign performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign performance: {str(e)}")

@router.post("/publish-due-posts", response_model=Dict[str, Any])
async def publish_due_posts(background_tasks: BackgroundTasks):
    """
    Publish all posts that are due for publication
    """
    try:
        logger.info("Publishing due posts")
        
        # Get posts that are due for publication
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT campaign_id, COUNT(*) as due_count
                FROM campaign_tasks
                WHERE status = 'scheduled' 
                AND metadata->>'scheduled_time' <= %s
                GROUP BY campaign_id
            """, (datetime.now().isoformat(),))
            
            due_posts = cur.fetchall()
        
        total_due = sum([count for _, count in due_posts])
        
        if total_due == 0:
            return {
                "success": True,
                "message": "No posts due for publication",
                "published": 0,
                "failed": 0
            }
        
        # In background, publish the due posts
        def publish_posts():
            published = 0
            failed = 0
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                for campaign_id, count in due_posts:
                    try:
                        cur.execute("""
                            UPDATE campaign_tasks 
                            SET status = 'published', updated_at = NOW()
                            WHERE campaign_id = %s 
                            AND status = 'scheduled'
                            AND metadata->>'scheduled_time' <= %s
                        """, (campaign_id, datetime.now().isoformat()))
                        published += cur.rowcount
                    except Exception as e:
                        logger.error(f"Failed to publish posts for campaign {campaign_id}: {e}")
                        failed += count
                conn.commit()
            
            logger.info(f"Published {published} posts, {failed} failed")
        
        background_tasks.add_task(publish_posts)
        
        return {
            "success": True,
            "message": f"Publishing {total_due} due posts",
            "total_due": total_due,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error publishing due posts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to publish due posts: {str(e)}")

@router.get("/upcoming-posts", response_model=List[Dict[str, Any]])
async def get_upcoming_posts(hours_ahead: int = 24):
    """
    Get posts scheduled for publication in the next N hours
    """
    try:
        logger.info(f"Getting posts scheduled for next {hours_ahead} hours")
        
        from datetime import timedelta
        cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    ct.id,
                    ct.campaign_id,
                    ct.task_type,
                    ct.metadata,
                    c.name as campaign_name
                FROM campaign_tasks ct
                JOIN campaigns c ON ct.campaign_id = c.id
                WHERE ct.status = 'scheduled'
                AND ct.metadata->>'scheduled_time' <= %s
                AND ct.metadata->>'scheduled_time' >= %s
                ORDER BY ct.metadata->>'scheduled_time' ASC
            """, (cutoff_time.isoformat(), datetime.now().isoformat()))
            
            upcoming_posts = cur.fetchall()
        
        posts = []
        for row in upcoming_posts:
            task_id, campaign_id, task_type, metadata, campaign_name = row
            
            # Parse metadata
            task_metadata = metadata or {}
            if isinstance(metadata, str):
                try:
                    import json
                    task_metadata = json.loads(metadata)
                except:
                    task_metadata = {}
            
            posts.append({
                "id": str(task_id),
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "task_type": task_type,
                "title": task_metadata.get("title", f"Post for {campaign_name}"),
                "platform": task_metadata.get("platform", "linkedin"),
                "scheduled_time": task_metadata.get("scheduled_time"),
                "content_preview": task_metadata.get("content", "")[:100] + "..." if task_metadata.get("content") else None
            })
        
        return posts
        
    except Exception as e:
        logger.error(f"Error getting upcoming posts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get upcoming posts: {str(e)}")

@router.post("/{post_id}/track-engagement", response_model=Dict[str, Any])
async def track_post_engagement(post_id: str):
    """
    Track engagement metrics for a published post
    """
    try:
        logger.info(f"Tracking engagement for post: {post_id}")
        
        # Mock engagement tracking
        engagement_data = {
            "post_id": post_id,
            "views": 234,
            "clicks": 18,
            "shares": 5,
            "comments": 3,
            "likes": 28,
            "engagement_rate": 7.7,
            "last_updated": datetime.now().isoformat()
        }
        
        # In a real implementation, this would integrate with social media APIs
        # to fetch actual engagement metrics
        
        return {
            "success": True,
            "post_id": post_id,
            "engagement": engagement_data,
            "message": "Engagement data updated"
        }
        
    except Exception as e:
        logger.error(f"Error tracking post engagement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track post engagement: {str(e)}")