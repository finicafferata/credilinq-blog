"""
Social media service for competitor intelligence.
Coordinates social media monitoring and data storage.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from .social_media_monitor import SocialMediaMonitor, SocialMediaPost
from .competitor_intelligence_db import ci_db
from .alert_service import alert_service, AlertType, AlertPriority
import psycopg2.extras
from ..config.database import db_config
import json

logger = logging.getLogger(__name__)

class SocialMediaService:
    """Service for managing social media monitoring and analysis."""
    
    def __init__(self):
        self.monitor = SocialMediaMonitor()
        
    async def discover_social_handles(self, competitor_id: str) -> Dict[str, str]:
        """Discover social media handles for a competitor."""
        try:
            # Get competitor details
            competitor = await ci_db.get_competitor(competitor_id)
            if not competitor:
                raise ValueError(f"Competitor {competitor_id} not found")
                
            async with self.monitor:
                # Extract social handles from website
                handles = await self.monitor.extract_social_handles_from_website(
                    competitor['domain']
                )
                
                # Update competitor with discovered handles
                if handles:
                    await ci_db.update_competitor(competitor_id, {
                        'social_handles': handles
                    })
                    logger.info(f"Discovered {len(handles)} social handles for {competitor['name']}")
                    
                return handles
                
        except Exception as e:
            logger.error(f"Error discovering social handles for {competitor_id}: {str(e)}")
            return {}
            
    async def monitor_competitor_social_media(self, competitor_id: str) -> Dict[str, Any]:
        """Monitor social media for a specific competitor."""
        try:
            # Get competitor details
            competitor = await ci_db.get_competitor(competitor_id)
            if not competitor:
                raise ValueError(f"Competitor {competitor_id} not found")
                
            results = {
                'competitor_id': competitor_id,
                'competitor_name': competitor['name'],
                'new_posts_count': 0,
                'updated_posts_count': 0,
                'platforms_monitored': [],
                'errors': [],
                'monitoring_time': datetime.utcnow().isoformat()
            }
            
            # Get social handles
            social_handles = competitor.get('socialHandles', {}) or {}
            
            if not social_handles:
                # Try to discover handles if not set
                logger.info(f"No social handles found for {competitor['name']}, attempting discovery...")
                social_handles = await self.discover_social_handles(competitor_id)
                
            if not social_handles:
                results['errors'].append("No social media handles found")
                return results
                
            async with self.monitor:
                # Monitor all platforms
                all_posts = await self.monitor.monitor_competitor_social_media(
                    competitor['name'], 
                    social_handles
                )
                
                # Store posts in database
                for post in all_posts:
                    try:
                        stored = await self._store_social_post(competitor_id, post)
                        if stored == 'new':
                            results['new_posts_count'] += 1
                        elif stored == 'updated':
                            results['updated_posts_count'] += 1
                    except Exception as e:
                        error_msg = f"Error storing post from {post.platform}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                        
                results['platforms_monitored'] = list(social_handles.keys())
                
                # Generate alerts for social media activity
                if all_posts:
                    try:
                        alert_ids = await self._generate_social_alerts(competitor_id, all_posts)
                        results['alerts_generated'] = len(alert_ids)
                    except Exception as e:
                        error_msg = f"Error generating social alerts: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                # Update last social monitored time
                await ci_db.update_competitor(competitor_id, {
                    'last_social_monitored': datetime.utcnow().isoformat()
                })
                
            logger.info(f"Social monitoring complete for {competitor['name']}: {results['new_posts_count']} new posts")
            return results
            
        except Exception as e:
            error_msg = f"Error monitoring social media for {competitor_id}: {str(e)}"
            logger.error(error_msg)
            return {
                'competitor_id': competitor_id,
                'error': error_msg,
                'monitoring_time': datetime.utcnow().isoformat()
            }
            
    async def _store_social_post(self, competitor_id: str, post: SocialMediaPost) -> str:
        """Store social media post in database."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Check if post already exists
                    cur.execute("""
                        SELECT id, content, "engagementMetrics"
                        FROM ci_social_posts 
                        WHERE "competitorId" = %s AND platform = %s AND "postId" = %s
                        ORDER BY "discoveredAt" DESC
                        LIMIT 1
                    """, (competitor_id, post.platform, post.post_id))
                    
                    existing = cur.fetchone()
                    
                    if existing:
                        # Check if content or engagement has changed
                        needs_update = False
                        if existing['content'] != post.content:
                            needs_update = True
                        
                        # Check if engagement metrics have improved
                        existing_engagement = existing.get('engagementMetrics') or {}
                        if post.engagement_metrics and post.engagement_metrics != existing_engagement:
                            needs_update = True
                            
                        if needs_update:
                            # Update existing post
                            cur.execute("""
                                UPDATE ci_social_posts 
                                SET 
                                    content = %s,
                                    "engagementMetrics" = %s,
                                    "viralityScore" = %s,
                                    metadata = %s,
                                    "discoveredAt" = NOW()
                                WHERE id = %s
                                RETURNING id
                            """, (
                                post.content,
                                json.dumps(post.engagement_metrics),
                                self._calculate_virality_score(post),
                                json.dumps({
                                    **post.metadata,
                                    'last_updated': datetime.utcnow().isoformat(),
                                    'update_reason': 'content_or_engagement_changed'
                                }),
                                existing['id']
                            ))
                            return 'updated'
                        else:
                            return 'exists'
                    else:
                        # Insert new post
                        cur.execute("""
                            INSERT INTO ci_social_posts (
                                id, "competitorId", platform, "postId", url, content,
                                author, "authorHandle", "publishedAt", "discoveredAt",
                                "engagementMetrics", hashtags, mentions, "mediaUrls",
                                "postType", "sentimentScore", "viralityScore", metadata
                            ) VALUES (
                                uuid_generate_v4(), %s, %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s
                            )
                            RETURNING id
                        """, (
                            competitor_id,
                            post.platform,
                            post.post_id,
                            post.url,
                            post.content,
                            post.author,
                            post.author_handle,
                            post.published_at,
                            post.published_at,  # discoveredAt
                            json.dumps(post.engagement_metrics),
                            post.hashtags,
                            post.mentions,
                            post.media_urls,
                            post.post_type,
                            self._calculate_sentiment_score(post.content),
                            self._calculate_virality_score(post),
                            json.dumps({
                                **post.metadata,
                                'discovery_method': 'social_monitoring',
                                'discovered_at': datetime.utcnow().isoformat()
                            })
                        ))
                        return 'new'
                        
        except Exception as e:
            logger.error(f"Database error storing social post: {str(e)}")
            raise
            
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate basic sentiment score for content."""
        # Simple sentiment analysis based on keywords
        positive_words = ['great', 'excellent', 'amazing', 'love', 'fantastic', 'awesome', 'best', 'happy', 'excited', 'thrilled']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'frustrated', 'angry', 'sad']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        total_words = len(content.split())
        if total_words == 0:
            return 0.0
            
        # Calculate sentiment score (-1 to 1)
        sentiment = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment))
        
    def _calculate_virality_score(self, post: SocialMediaPost) -> float:
        """Calculate virality score based on engagement metrics."""
        score = 0.0
        
        engagement = post.engagement_metrics
        if not engagement:
            return score
            
        # Platform-specific scoring
        if post.platform == 'twitter':
            score += engagement.get('retweets', 0) * 2.0
            score += engagement.get('likes', 0) * 1.0
            score += engagement.get('replies', 0) * 1.5
        elif post.platform == 'linkedin':
            score += engagement.get('reactions', 0) * 1.5
            score += engagement.get('comments', 0) * 2.0
            score += engagement.get('shares', 0) * 3.0
        elif post.platform == 'facebook':
            score += engagement.get('likes', 0) * 1.0
            score += engagement.get('comments', 0) * 1.5
            score += engagement.get('shares', 0) * 2.5
        elif post.platform == 'instagram':
            score += engagement.get('likes', 0) * 1.0
            score += engagement.get('comments', 0) * 2.0
        elif post.platform == 'youtube':
            score += engagement.get('views', 0) * 0.01
            score += engagement.get('likes', 0) * 1.0
            score += engagement.get('comments', 0) * 2.0
            
        # Content type multipliers
        if post.post_type == 'video':
            score *= 1.5
        elif post.post_type == 'image':
            score *= 1.2
            
        # Hashtag bonus
        if len(post.hashtags) > 0:
            score *= (1 + len(post.hashtags) * 0.1)
            
        # Normalize to 0-10 scale
        return min(score / 100, 10.0)
        
    async def _generate_social_alerts(self, competitor_id: str, posts: List[SocialMediaPost]) -> List[str]:
        """Generate alerts based on social media activity."""
        alerts_created = []
        
        try:
            competitor = await ci_db.get_competitor(competitor_id)
            if not competitor:
                return alerts_created
                
            # Check for viral posts
            for post in posts:
                virality_score = self._calculate_virality_score(post)
                if virality_score > 5.0:  # High virality threshold
                    alert_id = await alert_service.create_alert(
                        title=f"Viral content from {competitor['name']}",
                        message=f"High engagement on {post.platform}: {post.content[:100]}...",
                        alert_type=AlertType.ENGAGEMENT_SPIKE,
                        priority=AlertPriority.HIGH,
                        competitor_id=competitor_id,
                        metadata={
                            'platform': post.platform,
                            'post_url': post.url,
                            'virality_score': virality_score,
                            'engagement_metrics': post.engagement_metrics
                        }
                    )
                    alerts_created.append(alert_id)
                    
            # Check for keyword mentions in social posts
            monitoring_keywords = competitor.get('monitoringKeywords', [])
            important_keywords = ['launch', 'funding', 'acquisition', 'partnership', 'beta', 'hiring']
            all_keywords = monitoring_keywords + important_keywords
            
            for post in posts:
                content_lower = post.content.lower()
                found_keywords = [kw for kw in all_keywords if kw.lower() in content_lower]
                
                if found_keywords:
                    alert_id = await alert_service.create_alert(
                        title=f"Key topics mentioned by {competitor['name']}",
                        message=f"Mentioned: {', '.join(found_keywords)} on {post.platform}",
                        alert_type=AlertType.KEYWORD_MENTION,
                        priority=AlertPriority.MEDIUM,
                        competitor_id=competitor_id,
                        metadata={
                            'platform': post.platform,
                            'post_url': post.url,
                            'keywords_found': found_keywords,
                            'post_content': post.content[:200]
                        }
                    )
                    alerts_created.append(alert_id)
                    
            # Check for high posting frequency
            if len(posts) > 5:  # More than 5 posts in one monitoring session
                alert_id = await alert_service.create_alert(
                    title=f"High social media activity: {competitor['name']}",
                    message=f"Posted {len(posts)} times across platforms recently",
                    alert_type=AlertType.CONTENT_UPDATE,
                    priority=AlertPriority.MEDIUM,
                    competitor_id=competitor_id,
                    metadata={
                        'post_count': len(posts),
                        'platforms': list(set(post.platform for post in posts))
                    }
                )
                alerts_created.append(alert_id)
                
        except Exception as e:
            logger.error(f"Error generating social alerts: {str(e)}")
            
        return alerts_created
        
    async def get_competitor_social_posts(
        self,
        competitor_id: str,
        platform: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get social media posts for a competitor."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Build query
                    query = """
                        SELECT * FROM ci_social_posts 
                        WHERE "competitorId" = %s 
                        AND "publishedAt" >= NOW() - INTERVAL '%s days'
                    """
                    params = [competitor_id, days_back]
                    
                    if platform:
                        query += " AND platform = %s"
                        params.append(platform)
                        
                    query += " ORDER BY \"publishedAt\" DESC LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    posts = []
                    for result in results:
                        post = dict(result)
                        # Format datetime objects
                        for field in ['publishedAt', 'discoveredAt']:
                            if post.get(field):
                                post[field] = post[field].isoformat()
                        posts.append(post)
                        
                    return posts
                    
        except Exception as e:
            logger.error(f"Error retrieving social posts for competitor {competitor_id}: {str(e)}")
            return []
            
    async def get_social_analytics(self, competitor_id: str) -> Dict[str, Any]:
        """Get social media analytics for a competitor."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get platform breakdown
                    cur.execute("""
                        SELECT 
                            platform,
                            COUNT(*) as post_count,
                            AVG("viralityScore") as avg_virality,
                            AVG("sentimentScore") as avg_sentiment,
                            MAX("publishedAt") as latest_post
                        FROM ci_social_posts 
                        WHERE "competitorId" = %s 
                        AND "publishedAt" >= NOW() - INTERVAL '30 days'
                        GROUP BY platform
                        ORDER BY post_count DESC
                    """, (competitor_id,))
                    
                    platform_stats = [dict(row) for row in cur.fetchall()]
                    
                    # Format datetime objects
                    for stat in platform_stats:
                        if stat.get('latest_post'):
                            stat['latest_post'] = stat['latest_post'].isoformat()
                            
                    # Get overall stats
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_posts,
                            AVG("viralityScore") as avg_virality,
                            AVG("sentimentScore") as avg_sentiment,
                            COUNT(DISTINCT platform) as platforms_active
                        FROM ci_social_posts 
                        WHERE "competitorId" = %s 
                        AND "publishedAt" >= NOW() - INTERVAL '30 days'
                    """, (competitor_id,))
                    
                    overall_stats = dict(cur.fetchone())
                    
                    # Get top performing posts
                    cur.execute("""
                        SELECT platform, content, "viralityScore", "engagementMetrics", url
                        FROM ci_social_posts 
                        WHERE "competitorId" = %s 
                        AND "viralityScore" > 0
                        ORDER BY "viralityScore" DESC
                        LIMIT 5
                    """, (competitor_id,))
                    
                    top_posts = [dict(row) for row in cur.fetchall()]
                    
                    return {
                        'overall_stats': overall_stats,
                        'platform_breakdown': platform_stats,
                        'top_performing_posts': top_posts,
                        'analysis_period': '30 days'
                    }
                    
        except Exception as e:
            logger.error(f"Error getting social analytics for {competitor_id}: {str(e)}")
            return {}

# Global service instance
social_media_service = SocialMediaService()