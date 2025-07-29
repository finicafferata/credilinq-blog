"""Analytics and metrics endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Optional

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
        raise convert_to_http_exception(DatabaseQueryError(f"Failed to update blog analytics: {str(e)}"))