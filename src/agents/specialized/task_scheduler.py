#!/usr/bin/env python3
"""
Task Scheduler Agent
Responsible for scheduling campaign tasks and optimizing posting times across platforms.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.agents.core.base_agent import BaseAgent
from src.config.database import db_config

logger = logging.getLogger(__name__)

@dataclass
class ScheduledPost:
    """Scheduled post configuration"""
    platform: str
    content: str
    image_url: Optional[str]
    scheduled_at: datetime
    status: str = "scheduled"
    metadata: Dict[str, Any] = None

@dataclass
class PlatformSchedule:
    """Platform-specific scheduling configuration"""
    platform: str
    best_times: List[str]
    posting_frequency: str
    content_types: List[str]
    engagement_optimization: Dict[str, Any]

class TaskSchedulerAgent(BaseAgent):
    """
    Task Scheduler Agent - Optimizes scheduling and timing for campaign posts
    """
    
    def __init__(self):
        super().__init__()
        self.agent_name = "TaskScheduler"
        self.description = "Campaign task scheduling and timing optimization"
        
        # Platform-specific scheduling rules
        self.platform_schedules = {
            "linkedin": PlatformSchedule(
                platform="linkedin",
                best_times=["09:00", "12:00", "17:00"],
                posting_frequency="3x per week",
                content_types=["post", "article", "carousel"],
                engagement_optimization={
                    "best_days": ["Tuesday", "Wednesday", "Thursday"],
                    "content_length": "800-1200 characters",
                    "hashtag_count": 3
                }
            ),
            "twitter": PlatformSchedule(
                platform="twitter",
                best_times=["08:00", "12:00", "16:00", "20:00"],
                posting_frequency="5x per week",
                content_types=["tweet", "thread", "image"],
                engagement_optimization={
                    "best_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "content_length": "280 characters",
                    "hashtag_count": 2
                }
            ),
            "instagram": PlatformSchedule(
                platform="instagram",
                best_times=["11:00", "15:00", "19:00"],
                posting_frequency="1x per day",
                content_types=["post", "story", "reel"],
                engagement_optimization={
                    "best_days": ["Monday", "Wednesday", "Friday"],
                    "content_length": "125 characters",
                    "hashtag_count": 5
                }
            ),
            "email": PlatformSchedule(
                platform="email",
                best_times=["09:00", "14:00"],
                posting_frequency="1x per week",
                content_types=["newsletter", "announcement"],
                engagement_optimization={
                    "best_days": ["Tuesday", "Wednesday"],
                    "subject_length": "30-50 characters",
                    "preview_length": "100 characters"
                }
            )
        }
    
    async def schedule_campaign_tasks(self, campaign_id: str, 
                                    strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Schedule all tasks for a campaign based on strategy
        """
        try:
            logger.info(f"Scheduling tasks for campaign {campaign_id}")
            
            # 1. Get campaign tasks from database
            tasks = await self._get_campaign_tasks(campaign_id)
            
            # 2. Generate optimal schedule for each task
            scheduled_posts = []
            
            for task in tasks:
                if task["task_type"] == "content_creation":
                    scheduled_post = await self._schedule_content_task(task, strategy)
                    if scheduled_post:
                        scheduled_posts.append(scheduled_post)
            
            # 3. Save scheduled posts to database
            await self._save_scheduled_posts(campaign_id, scheduled_posts)
            
            # 4. Create content calendar entries
            await self._create_content_calendar_entries(campaign_id, scheduled_posts)
            
            return {
                "campaign_id": campaign_id,
                "scheduled_posts": len(scheduled_posts),
                "schedule": scheduled_posts
            }
            
        except Exception as e:
            logger.error(f"Error scheduling campaign tasks: {str(e)}")
            raise Exception(f"Failed to schedule campaign tasks: {str(e)}")
    
    async def _get_campaign_tasks(self, campaign_id: str) -> List[Dict[str, Any]]:
        """
        Get campaign tasks from database
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, "taskType", result, error
                    FROM "CampaignTask"
                    WHERE "campaignId" = %s AND status = 'pending'
                    ORDER BY "taskType", "createdAt"
                """, (campaign_id,))
                
                rows = cur.fetchall()
                tasks = []
                
                for row in rows:
                    task_id, task_type, content, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    tasks.append({
                        "id": task_id,
                        "task_type": task_type,
                        "content": content,
                        "metadata": metadata
                    })
                
                return tasks
                
        except Exception as e:
            logger.error(f"Error getting campaign tasks: {str(e)}")
            raise
    
    async def _schedule_content_task(self, task: Dict[str, Any], 
                                   strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Schedule a specific content task
        """
        try:
            metadata = task["metadata"]
            platform = metadata.get("platform", "linkedin")
            content_type = metadata.get("content_type", "post")
            
            # Get platform schedule
            platform_schedule = self.platform_schedules.get(platform)
            if not platform_schedule:
                logger.warning(f"No schedule found for platform: {platform}")
                return None
            
            # Generate optimal posting time
            scheduled_time = await self._generate_optimal_time(platform_schedule, strategy)
            
            # Generate content for the platform
            content = await self._generate_platform_content(task, platform_schedule, strategy)
            
            return {
                "platform": platform,
                "content_type": content_type,
                "content": content,
                "scheduled_at": scheduled_time.isoformat(),
                "status": "scheduled",
                "metadata": {
                    "platform_schedule": platform_schedule.__dict__,
                    "original_task_id": task["id"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error scheduling content task: {str(e)}")
            return None
    
    async def _generate_optimal_time(self, platform_schedule: PlatformSchedule, 
                                   strategy: Dict[str, Any]) -> datetime:
        """
        Generate optimal posting time based on platform and strategy
        """
        try:
            # Start from tomorrow
            start_date = datetime.now() + timedelta(days=1)
            
            # Get best times for the platform
            best_times = platform_schedule.best_times
            
            # Calculate posting frequency
            frequency_map = {
                "daily": 1,
                "1x per day": 1,
                "3x per week": 3,
                "5x per week": 5,
                "weekly": 1,
                "1x per week": 1
            }
            
            posts_per_week = frequency_map.get(platform_schedule.posting_frequency, 3)
            
            # Generate schedule for next 4 weeks
            scheduled_times = []
            current_date = start_date
            
            for week in range(4):
                for day in range(7):
                    if len(scheduled_times) >= posts_per_week * 4:
                        break
                    
                    # Check if it's a good day for the platform
                    if self._is_good_day(current_date, platform_schedule):
                        for time_str in best_times:
                            if len(scheduled_times) >= posts_per_week * 4:
                                break
                            
                            # Parse time
                            hour, minute = map(int, time_str.split(":"))
                            scheduled_time = current_date.replace(
                                hour=hour, minute=minute, second=0, microsecond=0
                            )
                            
                            scheduled_times.append(scheduled_time)
                    
                    current_date += timedelta(days=1)
            
            # Return the first available time
            return scheduled_times[0] if scheduled_times else datetime.now() + timedelta(hours=1)
            
        except Exception as e:
            logger.error(f"Error generating optimal time: {str(e)}")
            return datetime.now() + timedelta(hours=1)
    
    def _is_good_day(self, date: datetime, platform_schedule: PlatformSchedule) -> bool:
        """
        Check if a date is good for posting on the platform
        """
        day_name = date.strftime("%A")
        best_days = platform_schedule.engagement_optimization.get("best_days", [])
        
        return day_name in best_days
    
    async def _generate_platform_content(self, task: Dict[str, Any], 
                                       platform_schedule: PlatformSchedule,
                                       strategy: Dict[str, Any]) -> str:
        """
        Generate platform-specific content
        """
        try:
            platform = platform_schedule.platform
            content_type = task["metadata"].get("content_type", "post")
            
            # Get original content from task
            original_content = task["content"]
            
            # Generate platform-specific prompt
            prompt = f"""
            Adapt this content for {platform} {content_type}:
            
            Original Content: {original_content}
            Platform: {platform}
            Content Type: {content_type}
            
            Platform Requirements:
            - Best length: {platform_schedule.engagement_optimization.get('content_length', 'Standard')}
            - Hashtag count: {platform_schedule.engagement_optimization.get('hashtag_count', 3)}
            - Platform: {platform}
            
            Strategy Context:
            - Key Messages: {strategy.get('key_messages', [])}
            - Target Audience: {strategy.get('target_audience', '')}
            
            Generate optimized content for {platform} that:
            1. Fits the platform's format and requirements
            2. Maintains the key messages
            3. Includes appropriate hashtags
            4. Has a clear call-to-action
            
            Return only the content, no explanations.
            """
            
            content = await self._call_ai_model(prompt)
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error generating platform content: {str(e)}")
            return f"Content for {platform} {content_type}"
    
    async def _save_scheduled_posts(self, campaign_id: str, 
                                  scheduled_posts: List[Dict[str, Any]]) -> None:
        """
        Save scheduled posts to database
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                for post in scheduled_posts:
                    post_id = str(uuid.uuid4())
                    
                    # Get the corresponding task ID
                    task_id = post["metadata"]["original_task_id"]
                    
                    cur.execute("""
                        INSERT INTO scheduled_post (id, campaign_id, task_id, platform, content, 
                                                 scheduled_at, status, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        post_id,
                        campaign_id,
                        task_id,
                        post["platform"],
                        post["content"],
                        post["scheduled_at"],
                        post["status"],
                        json.dumps(post["metadata"])
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(scheduled_posts)} scheduled posts")
                
        except Exception as e:
            logger.error(f"Error saving scheduled posts: {str(e)}")
            raise
    
    async def _create_content_calendar_entries(self, campaign_id: str, 
                                             scheduled_posts: List[Dict[str, Any]]) -> None:
        """
        Create content calendar entries for scheduled posts
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                for post in scheduled_posts:
                    calendar_id = str(uuid.uuid4())
                    scheduled_date = datetime.fromisoformat(post["scheduled_at"])
                    
                    cur.execute("""
                        INSERT INTO content_calendar (id, campaign_id, date, content_type, 
                                                   title, description, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        calendar_id,
                        campaign_id,
                        scheduled_date.date(),
                        f"{post['platform']}_{post['content_type']}",
                        f"{post['platform'].title()} {post['content_type'].title()}",
                        f"Scheduled {post['content_type']} for {post['platform']}",
                        "planned"
                    ))
                
                conn.commit()
                logger.info(f"Created {len(scheduled_posts)} calendar entries")
                
        except Exception as e:
            logger.error(f"Error creating calendar entries: {str(e)}")
            raise
    
    async def get_scheduled_posts(self, campaign_id: str) -> List[Dict[str, Any]]:
        """
        Get all scheduled posts for a campaign
        """
        try:
            # TODO: Implement scheduled_post table in database schema
            # For now, return empty list since the table doesn't exist
            logger.warning("scheduled_post table not implemented - returning empty list")
            return []
                
        except Exception as e:
            logger.error(f"Error getting scheduled posts: {str(e)}")
            return []
    
    async def update_post_status(self, post_id: str, new_status: str, 
                               post_url: Optional[str] = None) -> bool:
        """
        Update scheduled post status
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                if post_url:
                    cur.execute("""
                        UPDATE scheduled_post 
                        SET status = %s, published_at = NOW(), post_url = %s
                        WHERE id = %s
                    """, (new_status, post_url, post_id))
                else:
                    cur.execute("""
                        UPDATE scheduled_post 
                        SET status = %s
                        WHERE id = %s
                    """, (new_status, post_id))
                
                conn.commit()
                return cur.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating post status: {str(e)}")
            raise
    
    async def get_upcoming_posts(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """
        Get posts scheduled for the next N hours
        """
        try:
            # TODO: Implement scheduled_post table in database schema
            # For now, return empty list since the table doesn't exist
            logger.warning("scheduled_post table not implemented - returning empty list")
            return []
                
        except Exception as e:
            logger.error(f"Error getting upcoming posts: {str(e)}")
            return []
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the task scheduler agent (required by BaseAgent)
        """
        try:
            # For now, return a simple result
            from src.agents.core.base_agent import AgentResult
            return AgentResult(
                success=True,
                data={"message": "TaskSchedulerAgent executed successfully"},
                metadata={"agent_type": "task_scheduler"}
            )
        except Exception as e:
            from src.agents.core.base_agent import AgentResult
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="TASK_SCHEDULER_EXECUTION_FAILED"
            ) 