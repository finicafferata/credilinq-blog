"""Analytics and metrics endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime, timedelta

from ...config.database import db_config
from ...core.exceptions import (
    DatabaseQueryError, AgentExecutionError, 
    convert_to_http_exception
)
from ..models.analytics import BlogAnalyticsRequest, MarketingMetricRequest, AgentFeedbackRequest

router = APIRouter()


@router.post("/blogs/{post_id}/analytics")
def update_blog_analytics(post_id: str, analytics: BlogAnalyticsRequest):
    """Update analytics data for a blog post."""
    try:
        from ...agents.core.database_service import get_db_service, BlogAnalyticsData
        db_service = get_db_service()
        
        analytics_data = BlogAnalyticsData(
            blog_id=post_id,
            **analytics.dict()
        )
        
        record_id = db_service.update_blog_analytics(analytics_data)
        
        return {
            "message": "Analytics updated successfully",
            "record_id": record_id,
            "blog_id": post_id
        }
        
    except Exception as e:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update blog analytics: {str(e)}"))


@router.get("/blogs/{post_id}/analytics")
def get_blog_analytics(post_id: str):
    """Get analytics data for a blog post."""
    try:
        resp = db_config.supabase.table("blog_analytics").select("*").eq("blog_id", post_id).single().execute()
        if resp.data:
            return resp.data
        else:
            return {
                "blog_id": post_id,
                "views": 0,
                "unique_visitors": 0,
                "engagement_rate": 0.0,
                "social_shares": 0,
                "message": "No analytics data found"
            }
    except Exception as e:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update blog analytics: {str(e)}"))


@router.post("/blogs/{post_id}/metrics")
def record_marketing_metric(post_id: str, metric: MarketingMetricRequest):
    """Record a marketing metric for a blog post."""
    try:
        from ...agents.core.database_service import get_db_service, MarketingMetric
        db_service = get_db_service()
        
        marketing_metric = MarketingMetric(
            blog_id=post_id,
            **metric.dict()
        )
        
        record_id = db_service.record_marketing_metric(marketing_metric)
        
        return {
            "message": "Marketing metric recorded successfully",
            "record_id": record_id,
            "blog_id": post_id
        }
        
    except Exception as e:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update blog analytics: {str(e)}"))


@router.post("/agents/feedback")
def record_agent_feedback(feedback: AgentFeedbackRequest):
    """Record feedback for agent learning and improvement."""
    try:
        from ...agents.core.database_service import get_db_service
        db_service = get_db_service()
        
        record_id = db_service.record_agent_feedback(
            agent_type=feedback.agent_type,
            feedback_type=feedback.feedback_type,
            feedback_value=feedback.feedback_value,
            feedback_text=feedback.feedback_text,
            user_id=feedback.user_id
        )
        
        return {
            "message": "Agent feedback recorded successfully",
            "record_id": record_id
        }
        
    except Exception as e:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update blog analytics: {str(e)}"))


@router.get("/analytics/dashboard")
def get_dashboard_analytics(days: int = 30):
    """Get comprehensive analytics for dashboard."""
    try:
        from ...agents.core.database_service import get_db_service
        db_service = get_db_service()
        analytics = db_service.get_dashboard_analytics(days=days)
        return analytics
    except Exception as e:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update blog analytics: {str(e)}"))


@router.get("/analytics/agents")
def get_agent_analytics(agent_type: Optional[str] = None, days: int = 30):
    """Get agent performance analytics."""
    try:
        from ...agents.core.database_service import get_db_service
        db_service = get_db_service()
        analytics = db_service.get_agent_performance_analytics(agent_type=agent_type, days=days)
        return {
            "agent_type": agent_type,
            "days": days,
            "performance_data": analytics
        }
    except Exception as e:
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to get agent analytics: {str(e)}"))


@router.get("/analytics/competitor-intelligence")
def get_competitor_intelligence_analytics(days: int = 30):
    """Get competitor intelligence analytics using real data."""
    try:
        from ...agents.core.database_service import get_db_service
        db_service = get_db_service()
        
        with db_service.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get real competitor count
            cur.execute("SELECT COUNT(*) FROM ci_competitors WHERE is_active = true")
            total_competitors = cur.fetchone()[0] or 0
            
            # Get active monitoring count (competitors with recent activity)
            cutoff_date = datetime.now() - timedelta(days=7)  # Active in last 7 days
            cur.execute("""
                SELECT COUNT(DISTINCT competitor_id) 
                FROM ci_content_items 
                WHERE discovered_at >= %s
            """, (cutoff_date,))
            active_monitoring = cur.fetchone()[0] or 0
            
            # Get content analyzed count
            content_cutoff = datetime.now() - timedelta(days=days)
            cur.execute("SELECT COUNT(*) FROM ci_content_items WHERE discovered_at >= %s", (content_cutoff,))
            content_analyzed = cur.fetchone()[0] or 0
            
            # Get trends count
            cur.execute("SELECT COUNT(*) FROM ci_trends WHERE first_detected >= %s", (content_cutoff,))
            trends_identified = cur.fetchone()[0] or 0
            
            # Get alerts count
            cur.execute("SELECT COUNT(*) FROM ci_alerts WHERE created_at >= %s", (content_cutoff,))
            alerts_generated = cur.fetchone()[0] or 0
            
            # Get content type distribution (real data if available)
            cur.execute("""
                SELECT content_type, COUNT(*) as count
                FROM ci_content_items 
                WHERE discovered_at >= %s
                GROUP BY content_type
                ORDER BY count DESC
            """, (content_cutoff,))
            
            content_types = []
            total_content = content_analyzed
            for row in cur.fetchall():
                count = row[1]
                percentage = (count / total_content * 100) if total_content > 0 else 0
                content_types.append({
                    "type": row[0].replace('_', ' ').title(),
                    "count": count,
                    "percentage": round(percentage, 1)
                })
            
            # Get platform activity (real data if available)
            cur.execute("""
                SELECT platform, COUNT(*) as posts
                FROM ci_content_items 
                WHERE discovered_at >= %s
                GROUP BY platform
                ORDER BY posts DESC
            """, (content_cutoff,))
            
            platform_activity = []
            for row in cur.fetchall():
                platform_activity.append({
                    "platform": row[0].title(),
                    "posts": row[1]
                })
            
            # Get top competitors (real data)
            cur.execute("""
                SELECT c.id, c.name, c.domain, COUNT(ci.id) as content_count
                FROM ci_competitors c
                LEFT JOIN ci_content_items ci ON c.id = ci.competitor_id 
                    AND ci.discovered_at >= %s
                WHERE c.is_active = true
                GROUP BY c.id, c.name, c.domain
                ORDER BY content_count DESC
                LIMIT 5
            """, (content_cutoff,))
            
            top_competitors = []
            for row in cur.fetchall():
                top_competitors.append({
                    "id": row[0],
                    "name": row[1],
                    "domain": row[2],
                    "content_count": row[3]
                })
            
            # Get trending topics (real data if available)
            cur.execute("""
                SELECT topic, COUNT(*) as mentions, AVG(growth_rate) as avg_growth
                FROM ci_trends 
                WHERE first_detected >= %s
                GROUP BY topic
                ORDER BY mentions DESC
                LIMIT 5
            """, (content_cutoff,))
            
            trending_topics = []
            for row in cur.fetchall():
                trending_topics.append({
                    "topic": row[0],
                    "mentions": row[1],
                    "growth_rate": round(row[2], 1) if row[2] else 0.0
                })
        
        analytics = {
            "total_competitors": total_competitors,
            "active_monitoring": active_monitoring,
            "content_analyzed": content_analyzed,
            "trends_identified": trends_identified,
            "alerts_generated": alerts_generated,
            "content_types_distribution": content_types,
            "platform_activity": platform_activity,
            "top_competitors": top_competitors,
            "trending_topics": trending_topics,
            "data_notes": {
                "status": "Real data from CI database tables",
                "note": "Some metrics may show 0 if no competitor intelligence data has been collected yet"
            }
        }
        
        return analytics
    except Exception as e:
        # Return empty state with explanation
        return {
            "total_competitors": 0,
            "active_monitoring": 0,
            "content_analyzed": 0,
            "trends_identified": 0,
            "alerts_generated": 0,
            "content_types_distribution": [],
            "platform_activity": [],
            "top_competitors": [],
            "trending_topics": [],
            "error": str(e),
            "data_notes": {
                "status": "No data available - CI system not yet populated or database connection issue"
            }
        }