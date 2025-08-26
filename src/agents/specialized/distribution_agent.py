#!/usr/bin/env python3
"""
Distribution Agent
Responsible for automatically publishing content to different platforms and tracking engagement.
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
class PlatformConfig:
    """Platform-specific configuration"""
    platform: str
    api_enabled: bool
    api_credentials: Dict[str, str]
    posting_rules: Dict[str, Any]
    engagement_tracking: bool

@dataclass
class PublishedPost:
    """Published post information"""
    platform: str
    post_id: str
    post_url: str
    published_at: datetime
    engagement_metrics: Dict[str, Any]

class DistributionAgent(BaseAgent):
    """
    Distribution Agent - Handles automatic publishing and engagement tracking
    """
    
    def __init__(self):
        super().__init__()
        self.agent_name = "DistributionAgent"
        self.description = "Automatic content distribution and engagement tracking"
        
        # Platform configurations (simulated for now)
        self.platform_configs = {
            "linkedin": PlatformConfig(
                platform="linkedin",
                api_enabled=True,
                api_credentials={"api_key": "simulated", "access_token": "simulated"},
                posting_rules={
                    "max_length": 3000,
                    "hashtag_limit": 5,
                    "image_supported": True,
                    "video_supported": True
                },
                engagement_tracking=True
            ),
            "twitter": PlatformConfig(
                platform="twitter",
                api_enabled=True,
                api_credentials={"api_key": "simulated", "access_token": "simulated"},
                posting_rules={
                    "max_length": 280,
                    "hashtag_limit": 2,
                    "image_supported": True,
                    "video_supported": True
                },
                engagement_tracking=True
            ),
            "instagram": PlatformConfig(
                platform="instagram",
                api_enabled=False,  # Instagram API is limited
                api_credentials={},
                posting_rules={
                    "max_length": 125,
                    "hashtag_limit": 30,
                    "image_supported": True,
                    "video_supported": True
                },
                engagement_tracking=False
            ),
            "email": PlatformConfig(
                platform="email",
                api_enabled=True,
                api_credentials={"smtp_server": "simulated", "api_key": "simulated"},
                posting_rules={
                    "subject_max_length": 50,
                    "body_max_length": 50000,
                    "image_supported": True,
                    "video_supported": False
                },
                engagement_tracking=True
            )
        }
    
    async def publish_scheduled_posts(self) -> Dict[str, Any]:
        """
        Publish all scheduled posts that are due
        """
        try:
            logger.info("Starting scheduled post publication")
            
            # 1. Get posts due for publication
            due_posts = await self._get_due_posts()
            
            if not due_posts:
                logger.info("No posts due for publication")
                return {"published": 0, "failed": 0, "posts": []}
            
            # 2. Publish each post
            published_posts = []
            failed_posts = []
            
            for post in due_posts:
                try:
                    published_post = await self._publish_single_post(post)
                    if published_post:
                        published_posts.append(published_post)
                    else:
                        failed_posts.append(post)
                except Exception as e:
                    logger.error(f"Failed to publish post {post['id']}: {str(e)}")
                    failed_posts.append(post)
            
            # 3. Update post statuses in database
            await self._update_post_statuses(published_posts, failed_posts)
            
            # 4. Schedule engagement tracking for published posts
            await self._schedule_engagement_tracking(published_posts)
            
            return {
                "published": len(published_posts),
                "failed": len(failed_posts),
                "posts": published_posts
            }
            
        except Exception as e:
            logger.error(f"Error publishing scheduled posts: {str(e)}")
            raise Exception(f"Failed to publish scheduled posts: {str(e)}")
    
    async def _get_due_posts(self) -> List[Dict[str, Any]]:
        """
        Get posts that are due for publication
        """
        try:
            # TODO: scheduled_post table not implemented yet - return empty list
            logger.warning("scheduled_post table not implemented - returning empty list")
            return []
            
        except Exception as e:
            logger.error(f"Error getting due posts: {str(e)}")
            return []
    
    async def _publish_single_post(self, post: Dict[str, Any]) -> Optional[PublishedPost]:
        """
        Publish a single post to its platform
        """
        try:
            platform = post["platform"]
            content = post["content"]
            
            # Get platform configuration
            platform_config = self.platform_configs.get(platform)
            if not platform_config:
                logger.warning(f"No configuration found for platform: {platform}")
                return None
            
            # Check if platform API is enabled
            if not platform_config.api_enabled:
                logger.info(f"Platform {platform} API not enabled, simulating publication")
                return await self._simulate_publication(post, platform_config)
            
            # Validate content for platform
            if not self._validate_content_for_platform(content, platform_config):
                logger.warning(f"Content validation failed for {platform}")
                return None
            
            # Publish to platform
            published_post = await self._publish_to_platform(post, platform_config)
            
            if published_post:
                logger.info(f"Successfully published to {platform}")
                return published_post
            else:
                logger.error(f"Failed to publish to {platform}")
                return None
                
        except Exception as e:
            logger.error(f"Error publishing single post: {str(e)}")
            return None
    
    def _validate_content_for_platform(self, content: str, 
                                     platform_config: PlatformConfig) -> bool:
        """
        Validate content meets platform requirements
        """
        try:
            rules = platform_config.posting_rules
            max_length = rules.get("max_length", 1000)
            
            if len(content) > max_length:
                logger.warning(f"Content too long for {platform_config.platform}: {len(content)} > {max_length}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating content: {str(e)}")
            return False
    
    async def _simulate_publication(self, post: Dict[str, Any], 
                                  platform_config: PlatformConfig) -> PublishedPost:
        """
        Simulate publication for platforms without API access
        """
        try:
            platform = platform_config.platform
            post_id = str(uuid.uuid4())
            
            # Generate simulated post URL
            post_url = f"https://{platform}.com/simulated/{post_id}"
            
            # Generate simulated engagement metrics
            engagement_metrics = {
                "views": 0,
                "likes": 0,
                "shares": 0,
                "comments": 0,
                "clicks": 0
            }
            
            return PublishedPost(
                platform=platform,
                post_id=post_id,
                post_url=post_url,
                published_at=datetime.now(),
                engagement_metrics=engagement_metrics
            )
            
        except Exception as e:
            logger.error(f"Error simulating publication: {str(e)}")
            return None
    
    async def _publish_to_platform(self, post: Dict[str, Any], 
                                  platform_config: PlatformConfig) -> Optional[PublishedPost]:
        """
        Publish to platform using API (simulated for now)
        """
        try:
            platform = platform_config.platform
            content = post["content"]
            
            # Simulate API call
            logger.info(f"Publishing to {platform}: {content[:100]}...")
            
            # Simulate processing time
            await self._simulate_api_delay()
            
            # Generate simulated response
            post_id = str(uuid.uuid4())
            post_url = f"https://{platform}.com/posts/{post_id}"
            
            # Simulate engagement metrics
            engagement_metrics = {
                "views": 0,
                "likes": 0,
                "shares": 0,
                "comments": 0,
                "clicks": 0
            }
            
            return PublishedPost(
                platform=platform,
                post_id=post_id,
                post_url=post_url,
                published_at=datetime.now(),
                engagement_metrics=engagement_metrics
            )
            
        except Exception as e:
            logger.error(f"Error publishing to platform: {str(e)}")
            return None
    
    async def _simulate_api_delay(self):
        """
        Simulate API processing delay
        """
        import asyncio
        await asyncio.sleep(0.1)  # 100ms delay
    
    async def _update_post_statuses(self, published_posts: List[PublishedPost], 
                                  failed_posts: List[Dict[str, Any]]) -> None:
        """
        Update post statuses in database
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Update successful publications
                for post in published_posts:
                    cur.execute("""
                        UPDATE scheduled_post 
                        SET status = 'published', 
                            published_at = %s,
                            post_url = %s,
                            metadata = jsonb_set(metadata, '{engagement_metrics}', %s)
                        WHERE platform = %s AND content = %s AND status = 'scheduled'
                    """, (
                        post.published_at,
                        post.post_url,
                        json.dumps(post.engagement_metrics),
                        post.platform,
                        post.content[:100]  # Match by content preview
                    ))
                
                # Update failed publications
                for post in failed_posts:
                    cur.execute("""
                        UPDATE scheduled_post 
                        SET status = 'failed'
                        WHERE id = %s
                    """, (post["id"],))
                
                conn.commit()
                logger.info(f"Updated {len(published_posts)} successful and {len(failed_posts)} failed posts")
                
        except Exception as e:
            logger.error(f"Error updating post statuses: {str(e)}")
            raise
    
    async def _schedule_engagement_tracking(self, published_posts: List[PublishedPost]) -> None:
        """
        Schedule engagement tracking for published posts
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                for post in published_posts:
                    # Schedule tracking for 1 hour, 24 hours, and 7 days
                    tracking_times = [
                        datetime.now() + timedelta(hours=1),
                        datetime.now() + timedelta(hours=24),
                        datetime.now() + timedelta(days=7)
                    ]
                    
                    for tracking_time in tracking_times:
                        tracking_id = str(uuid.uuid4())
                        
                        cur.execute("""
                            INSERT INTO engagement_tracking (id, scheduled_post_id, metric_type, 
                                                          metric_value, recorded_at, source)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            tracking_id,
                            post.post_id,  # Using post_id as scheduled_post_id for now
                            "scheduled_tracking",
                            0,
                            tracking_time,
                            f"{post.platform}_api"
                        ))
                
                conn.commit()
                logger.info(f"Scheduled engagement tracking for {len(published_posts)} posts")
                
        except Exception as e:
            logger.error(f"Error scheduling engagement tracking: {str(e)}")
            raise
    
    async def track_engagement(self, post_id: str) -> Dict[str, Any]:
        """
        Track engagement for a specific post
        """
        try:
            # Get post information
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT platform, post_url, metadata
                    FROM scheduled_post
                    WHERE id = %s
                """, (post_id,))
                
                row = cur.fetchone()
                if not row:
                    raise Exception("Post not found")
                
                platform, post_url, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Simulate engagement tracking
                engagement_metrics = await self._simulate_engagement_tracking(platform, post_url)
                
                # Update engagement metrics
                metadata["engagement_metrics"] = engagement_metrics
                
                cur.execute("""
                    UPDATE scheduled_post 
                    SET metadata = %s
                    WHERE id = %s
                """, (json.dumps(metadata), post_id))
                
                conn.commit()
                
                return {
                    "post_id": post_id,
                    "platform": platform,
                    "engagement_metrics": engagement_metrics,
                    "tracked_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error tracking engagement: {str(e)}")
            raise
    
    async def _simulate_engagement_tracking(self, platform: str, post_url: str) -> Dict[str, Any]:
        """
        Simulate engagement tracking for a post
        """
        try:
            import random
            
            # Simulate realistic engagement metrics based on platform
            if platform == "linkedin":
                metrics = {
                    "views": random.randint(50, 500),
                    "likes": random.randint(5, 50),
                    "shares": random.randint(1, 10),
                    "comments": random.randint(1, 15),
                    "clicks": random.randint(10, 100)
                }
            elif platform == "twitter":
                metrics = {
                    "views": random.randint(100, 1000),
                    "likes": random.randint(10, 100),
                    "retweets": random.randint(1, 20),
                    "replies": random.randint(1, 10),
                    "clicks": random.randint(20, 200)
                }
            elif platform == "instagram":
                metrics = {
                    "views": random.randint(200, 2000),
                    "likes": random.randint(20, 200),
                    "comments": random.randint(1, 15),
                    "saves": random.randint(1, 10),
                    "shares": random.randint(1, 5)
                }
            else:  # email or other
                metrics = {
                    "opens": random.randint(100, 1000),
                    "clicks": random.randint(10, 100),
                    "unsubscribes": random.randint(0, 5),
                    "forwards": random.randint(1, 10)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error simulating engagement tracking: {str(e)}")
            return {"error": "Failed to track engagement"}
    
    async def get_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get overall performance metrics for a campaign
        """
        try:
            # TODO: Implement scheduled_post table in database schema
            # For now, return mock performance data
            logger.warning("scheduled_post table not implemented - returning mock performance data")
            
            total_posts = 0
            published_posts = 0
            failed_posts = 0
            
            # Calculate total engagement
            total_engagement = {
                "views": 0,
                "likes": 0,
                "shares": 0,
                "comments": 0,
                "clicks": 0
            }
            
            # Calculate averages
            avg_engagement = {}
            if published_posts > 0:
                for metric, total in total_engagement.items():
                    avg_engagement[metric] = total / published_posts
            
            return {
                "campaign_id": campaign_id,
                "total_posts": total_posts,
                "published_posts": published_posts,
                "failed_posts": failed_posts,
                "success_rate": (published_posts / total_posts * 100) if total_posts > 0 else 0,
                "total_engagement": total_engagement,
                "average_engagement": avg_engagement
            }
                
        except Exception as e:
            logger.error(f"Error getting campaign performance: {str(e)}")
            raise
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the distribution agent (required by BaseAgent)
        """
        try:
            # For now, return a simple result
            from src.agents.core.base_agent import AgentResult
            return AgentResult(
                success=True,
                data={"message": "DistributionAgent executed successfully"},
                metadata={"agent_type": "distribution_agent"}
            )
        except Exception as e:
            from src.agents.core.base_agent import AgentResult
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="DISTRIBUTION_AGENT_EXECUTION_FAILED"
            ) 