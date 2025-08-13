"""
Alert service for competitor intelligence notifications.
Manages alerts based on competitor activity and content changes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import psycopg2.extras
from ..config.database import db_config

logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    NEW_CONTENT = "new_content"
    CONTENT_UPDATE = "content_update"
    TRENDING_TOPIC = "trending_topic"
    COMPETITIVE_MOVE = "competitive_move"
    KEYWORD_MENTION = "keyword_mention"
    ENGAGEMENT_SPIKE = "engagement_spike"
    NEW_COMPETITOR = "new_competitor"

@dataclass
class Alert:
    id: Optional[str]
    title: str
    message: str
    alert_type: AlertType
    priority: AlertPriority
    competitor_id: Optional[str]
    content_id: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    is_read: bool = False
    is_dismissed: bool = False

class AlertService:
    """Service for managing competitor intelligence alerts."""
    
    def __init__(self):
        self.alert_rules = self._load_default_alert_rules()
        
    def _load_default_alert_rules(self) -> Dict[str, Dict]:
        """Load default alert rules configuration."""
        return {
            'new_content_threshold': {
                'enabled': True,
                'min_content_length': 100,
                'priority': AlertPriority.MEDIUM
            },
            'keyword_monitoring': {
                'enabled': True,
                'priority': AlertPriority.HIGH,
                'keywords': ['acquisition', 'funding', 'partnership', 'product launch', 'expansion']
            },
            'content_volume_spike': {
                'enabled': True,
                'threshold_multiplier': 2.0,  # 2x normal volume
                'priority': AlertPriority.HIGH
            },
            'competitor_activity_frequency': {
                'enabled': True,
                'daily_threshold': 5,  # More than 5 pieces of content per day
                'priority': AlertPriority.MEDIUM
            }
        }
        
    async def create_alert(
        self,
        title: str,
        message: str,
        alert_type: AlertType,
        priority: AlertPriority,
        competitor_id: Optional[str] = None,
        content_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO ci_alerts (
                            id, title, message, alert_type, priority,
                            competitor_id, content_id, metadata, created_at, is_read, is_dismissed
                        ) VALUES (
                            gen_random_uuid(), %s, %s, %s, %s,
                            %s, %s, %s::jsonb, %s, %s, %s
                        ) RETURNING id
                    """, (
                        title,
                        message,
                        alert_type.value,
                        priority.value,
                        competitor_id,
                        content_id,
                        psycopg2.extras.Json(metadata or {}),
                        datetime.utcnow(),
                        False,
                        False
                    ))
                    
                    alert_id = cur.fetchone()['id']
                    logger.info(f"Created alert {alert_id}: {title}")
                    return alert_id
                    
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            raise
            
    async def analyze_content_for_alerts(self, competitor_id: str, content_items: List[Dict]) -> List[str]:
        """Analyze content and generate relevant alerts."""
        alerts_created = []
        
        try:
            # Get competitor details
            competitor = await self._get_competitor_details(competitor_id)
            if not competitor:
                return alerts_created
                
            for content in content_items:
                # Normaliza a dict si viene un dataclass/objeto
                if not isinstance(content, dict):
                    try:
                        content = {
                            'id': getattr(content, 'id', None),
                            'title': getattr(content, 'title', ''),
                            'content': getattr(content, 'content', ''),
                            'contentType': getattr(content, 'content_type', 'content'),
                            'url': getattr(content, 'url', None),
                            'keywords': getattr(content, 'metadata', {}).get('keywords', []) if hasattr(content, 'metadata') else []
                        }
                    except Exception:
                        continue
                
                # Check for new content alerts
                if self.alert_rules['new_content_threshold']['enabled']:
                    alert_id = await self._check_new_content_alert(competitor, content)
                    if alert_id:
                        alerts_created.append(alert_id)
                        
                # Check for keyword mentions
                if self.alert_rules['keyword_monitoring']['enabled']:
                    alert_id = await self._check_keyword_alerts(competitor, content)
                    if alert_id:
                        alerts_created.append(alert_id)
                        
            # Check for content volume spikes
            if self.alert_rules['content_volume_spike']['enabled']:
                alert_id = await self._check_volume_spike_alert(competitor_id, len(content_items))
                if alert_id:
                    alerts_created.append(alert_id)
                    
        except Exception as e:
            logger.error(f"Error analyzing content for alerts: {str(e)}")
            
        return alerts_created
        
    async def _check_new_content_alert(self, competitor: Dict, content: Dict) -> Optional[str]:
        """Check if new content should trigger an alert."""
        try:
            content_length = len(content.get('content', ''))
            min_length = self.alert_rules['new_content_threshold']['min_content_length']
            
            if content_length >= min_length:
                title = f"New content from {competitor['name']}"
                message = f"New {content.get('contentType', 'content')} published: {content.get('title', 'Untitled')[:100]}..."
                
                alert_id = await self.create_alert(
                    title=title,
                    message=message,
                    alert_type=AlertType.NEW_CONTENT,
                    priority=self.alert_rules['new_content_threshold']['priority'],
                    competitor_id=competitor['id'],
                    content_id=content.get('id'),
                    metadata={
                        'content_url': content.get('url'),
                        'content_type': content.get('contentType'),
                        'word_count': content_length
                    }
                )
                return alert_id
                
        except Exception as e:
            logger.error(f"Error checking new content alert: {str(e)}")
            
        return None
        
    async def _check_keyword_alerts(self, competitor: Dict, content: Dict) -> Optional[str]:
        """Check for important keyword mentions."""
        try:
            content_text = (content.get('content', '') + ' ' + content.get('title', '')).lower()
            keywords = self.alert_rules['keyword_monitoring']['keywords']
            
            # Also check competitor's monitoring keywords
            if competitor.get('monitoringKeywords'):
                keywords.extend([kw.lower() for kw in competitor['monitoringKeywords']])
                
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in content_text:
                    found_keywords.append(keyword)
                    
            if found_keywords:
                title = f"Key topics mentioned by {competitor['name']}"
                message = f"Mentioned: {', '.join(found_keywords)} in '{content.get('title', 'Untitled')[:50]}...'"
                
                alert_id = await self.create_alert(
                    title=title,
                    message=message,
                    alert_type=AlertType.KEYWORD_MENTION,
                    priority=self.alert_rules['keyword_monitoring']['priority'],
                    competitor_id=competitor['id'],
                    content_id=content.get('id'),
                    metadata={
                        'keywords_found': found_keywords,
                        'content_url': content.get('url'),
                        'content_title': content.get('title')
                    }
                )
                return alert_id
                
        except Exception as e:
            logger.error(f"Error checking keyword alerts: {str(e)}")
            
        return None
        
    async def _check_volume_spike_alert(self, competitor_id: str, current_count: int) -> Optional[str]:
        """Check for unusual content volume spikes."""
        try:
            # Get average content count for this competitor over last 7 days
            with db_config.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) as daily_count
                        FROM ci_content_items 
                        WHERE competitor_id = %s 
                        AND discovered_at >= NOW() - INTERVAL '7 days'
                        GROUP BY DATE(discovered_at)
                        ORDER BY DATE(discovered_at) DESC
                        LIMIT 7
                    """, (competitor_id,))
                    
                    daily_counts = [row[0] for row in cur.fetchall()]
                    
                    if daily_counts:
                        avg_daily = sum(daily_counts) / len(daily_counts)
                        threshold = avg_daily * self.alert_rules['content_volume_spike']['threshold_multiplier']
                        
                        if current_count > threshold and current_count > 3:  # Minimum 3 items to trigger
                            competitor = await self._get_competitor_details(competitor_id)
                            if competitor:
                                title = f"Content volume spike: {competitor['name']}"
                                message = f"Published {current_count} items today (avg: {avg_daily:.1f})"
                                
                                alert_id = await self.create_alert(
                                    title=title,
                                    message=message,
                                    alert_type=AlertType.CONTENT_UPDATE,
                                    priority=self.alert_rules['content_volume_spike']['priority'],
                                    competitor_id=competitor_id,
                                    metadata={
                                        'current_count': current_count,
                                        'average_count': avg_daily,
                                        'spike_ratio': current_count / avg_daily if avg_daily > 0 else 0
                                    }
                                )
                                return alert_id
                                
        except Exception as e:
            logger.error(f"Error checking volume spike alert: {str(e)}")
            
        return None
        
    async def _get_competitor_details(self, competitor_id: str) -> Optional[Dict]:
        """Get competitor details from database."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM ci_competitors WHERE id = %s", (competitor_id,))
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting competitor details: {str(e)}")
            return None
            
    async def get_alerts(
        self,
        limit: int = 50,
        priority: Optional[AlertPriority] = None,
        alert_type: Optional[AlertType] = None,
        competitor_id: Optional[str] = None,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get alerts with filtering options."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Build query
                    query = """
                        SELECT a.*, c.name as competitor_name
                        FROM ci_alerts a
                        LEFT JOIN ci_competitors c ON a.competitor_id = c.id
                        WHERE 1=1
                    """
                    params = []
                    
                    if priority:
                        query += " AND a.priority = %s"
                        params.append(priority.value)
                        
                    if alert_type:
                        query += " AND a.alert_type = %s"
                        params.append(alert_type.value)
                        
                    if competitor_id:
                        query += " AND a.competitor_id = %s"
                        params.append(competitor_id)
                        
                    if unread_only:
                        query += " AND a.is_read = FALSE AND a.is_dismissed = FALSE"
                        
                    query += " ORDER BY a.created_at DESC LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    alerts = []
                    for result in results:
                        alert = dict(result)
                        alert['created_at'] = alert['created_at'].isoformat()
                        alerts.append(alert)
                        
                    return alerts
                    
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return []
            
    async def get_alerts_grouped_by_competitor(
        self,
        days_back: int = 7,
        limit_per_competitor: int = 10,
        priority: Optional[AlertPriority] = None,
        alert_type: Optional[AlertType] = None,
        competitor_ids: Optional[List[str]] = None,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get recent alerts grouped by competitor with optional filters.

        Returns a list of groups like:
        [{ 'competitorId': str | null, 'competitorName': str | null, 'alerts': [...] }]
        """
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    query = (
                        """
                        SELECT a.*, c.name as competitor_name
                        FROM ci_alerts a
                        LEFT JOIN ci_competitors c ON a.competitor_id = c.id
                        WHERE a.created_at >= NOW() - make_interval(days => %s)
                        """
                    )
                    params: List[Any] = [days_back]

                    if priority:
                        query += " AND a.priority = %s"
                        params.append(priority.value)

                    if alert_type:
                        query += " AND a.alert_type = %s"
                        params.append(alert_type.value)

                    if competitor_ids:
                        # Build IN clause safely
                        placeholders = ",".join(["%s"] * len(competitor_ids))
                        query += f" AND a.competitor_id IN ({placeholders})"
                        params.extend(competitor_ids)

                    if unread_only:
                        query += " AND a.is_read = FALSE AND a.is_dismissed = FALSE"

                    query += " ORDER BY a.created_at DESC"

                    cur.execute(query, params)
                    rows = cur.fetchall()

                    # Group in Python and limit per competitor
                    groups_map: Dict[str, Dict[str, Any]] = {}
                    order_of_keys: List[str] = []

                    for r in rows:
                        comp_id = r.get("competitor_id") or "__none__"
                        if comp_id not in groups_map:
                            groups_map[comp_id] = {
                                "competitorId": None if comp_id == "__none__" else comp_id,
                                "competitorName": r.get("competitor_name") if comp_id != "__none__" else None,
                                "alerts": []
                            }
                            order_of_keys.append(comp_id)

                        alert = dict(r)
                        if isinstance(alert.get("created_at"), datetime):
                            alert["created_at"] = alert["created_at"].isoformat()
                        groups_map[comp_id]["alerts"].append(alert)

                    # Apply per-competitor limit and build ordered list
                    grouped_list: List[Dict[str, Any]] = []
                    for key in order_of_keys:
                        g = groups_map[key]
                        g["alerts"] = g["alerts"][: max(0, int(limit_per_competitor))]
                        grouped_list.append(g)

                    return grouped_list

        except Exception as e:
            logger.error(f"Error getting grouped alerts: {str(e)}")
            return []

    async def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get a single alert by ID, including competitor name."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT a.*, c.name as competitor_name
                        FROM ci_alerts a
                        LEFT JOIN ci_competitors c ON a.competitor_id = c.id
                        WHERE a.id = %s
                        """,
                        (alert_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    alert = dict(row)
                    if isinstance(alert.get("created_at"), datetime):
                        alert["created_at"] = alert["created_at"].isoformat()
                    if isinstance(alert.get("read_at"), datetime):
                        alert["read_at"] = alert["read_at"].isoformat()
                    if isinstance(alert.get("dismissed_at"), datetime):
                        alert["dismissed_at"] = alert["dismissed_at"].isoformat()
                    return alert
        except Exception as e:
            logger.error(f"Error getting alert by id: {str(e)}")
            return None

    async def mark_alert_read(self, alert_id: str) -> bool:
        """Mark an alert as read."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ci_alerts 
                        SET is_read = TRUE, read_at = NOW()
                        WHERE id = %s
                    """, (alert_id,))
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking alert read: {str(e)}")
            return False
            
    async def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ci_alerts 
                        SET is_dismissed = TRUE, dismissed_at = NOW()
                        WHERE id = %s
                    """, (alert_id,))
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error dismissing alert: {str(e)}")
            return False
            
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert statistics."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get counts by priority and type
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_alerts,
                            COUNT(CASE WHEN is_read = FALSE AND is_dismissed = FALSE THEN 1 END) as unread_alerts,
                            COUNT(CASE WHEN priority = 'critical' AND is_read = FALSE THEN 1 END) as critical_unread,
                            COUNT(CASE WHEN priority = 'high' AND is_read = FALSE THEN 1 END) as high_unread,
                            COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as last_24h,
                            COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as last_7d
                        FROM ci_alerts
                    """)
                    
                    summary = dict(cur.fetchone())
                    
                    # Get top alert types
                    cur.execute("""
                        SELECT alert_type, COUNT(*) as count
                        FROM ci_alerts
                        WHERE created_at >= NOW() - INTERVAL '7 days'
                        GROUP BY alert_type
                        ORDER BY count DESC
                        LIMIT 5
                    """)
                    
                    summary['top_alert_types'] = [dict(row) for row in cur.fetchall()]
                    
                    return summary
                    
        except Exception as e:
            logger.error(f"Error getting alert summary: {str(e)}")
            return {}

# Global service instance
alert_service = AlertService()