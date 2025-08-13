"""
API routes for competitor intelligence and market analysis.
Provides endpoints for competitor monitoring and strategic insights.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

from ...agents.competitor_intelligence.competitor_intelligence_orchestrator import CompetitorIntelligenceOrchestrator
from ...agents.competitor_intelligence.models import (
    Competitor, CompetitorCreate, CompetitorTier, Industry, Platform, ContentType,
    MonitoringConfig, TrendQuery, GapAnalysisRequest, AlertSubscription,
    CompetitorSummary, TrendSummary, InsightSummary, CompetitorIntelligenceReport
)
from ...config.database import db_config
from ...services.competitor_intelligence_db import ci_db
from ...services.content_monitoring_service import content_monitoring_service
from ...services.alert_service import alert_service, AlertPriority, AlertType
from ...services.social_media_service import social_media_service
from ...services.trend_analysis import trend_analysis_engine
from ...services.ai_content_analyzer import ai_content_analyzer, ContentType as AIContentType
from ...services.advanced_reporting import advanced_reporting_service, ReportConfig, ReportType, ReportFormat
from ...services.external_integrations import external_integrations_service, IntegrationConfig, IntegrationType, NotificationMessage, EventType, MessagePriority
from ...services.webhook_service import webhook_service
from ...services.digest_service import digest_service
from ...services.change_detection_service import change_detection_service
from ...services.news_ingestion_service import news_ingestion_service

router = APIRouter(prefix="/competitor-intelligence", tags=["Competitor Intelligence"])

# Initialize orchestrator lazily to avoid instantiation issues
orchestrator = None

def get_orchestrator():
    """Get or create the orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        orchestrator = CompetitorIntelligenceOrchestrator()
    return orchestrator

@router.post("/competitors", response_model=Dict[str, Any])
async def create_competitor(competitor_data: CompetitorCreate):
    """Create a new competitor for monitoring."""
    try:
        # Convert Pydantic model to dictionary for database service
        competitor_dict = {
            "name": competitor_data.name,
            "domain": competitor_data.domain,
            "tier": competitor_data.tier.value,
            "industry": competitor_data.industry.value,
            "description": competitor_data.description,
            "platforms": [platform.value for platform in competitor_data.platforms] if competitor_data.platforms else [],
            "monitoring_keywords": competitor_data.monitoring_keywords or []
        }
        
        # Save to database
        competitor = await ci_db.create_competitor(competitor_dict)
        
        return {
            "success": True,
            "message": "Competitor created successfully",
            "competitor": competitor
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create competitor: {str(e)}")

@router.get("/competitors")
async def list_competitors(
    industry: Optional[Industry] = None,
    tier: Optional[CompetitorTier] = None,
    active_only: bool = True
):
    """List all competitors with optional filtering."""
    try:
        # Get competitors from database
        competitors = await ci_db.list_competitors(
            industry=industry.value if industry else None,
            tier=tier.value if tier else None,
            active_only=active_only
        )
        
        # Convert to CompetitorSummary format
        summaries = []
        for comp in competitors:
            # Convert to frontend-expected format (camelCase)
            summary = {
                "id": comp["id"],
                "name": comp["name"],
                "tier": comp["tier"],
                "industry": comp["industry"],
                "contentCount": 0,  # Mock data - would be real data in full implementation
                "lastActivity": comp.get("lastMonitored"),
                "avgEngagement": 0.0,  # Mock data - would be calculated from content items
                "trendingScore": 0.0   # Mock data - would be calculated from trends
            }
            summaries.append(summary)
        
        return summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list competitors: {str(e)}")

@router.get("/competitors/{competitor_id}")
async def get_competitor(competitor_id: str):
    """Get a specific competitor by ID."""
    try:
        competitor = await ci_db.get_competitor(competitor_id)
        if not competitor:
            raise HTTPException(status_code=404, detail="Competitor not found")
        
        # Convert to frontend format (camelCase)
        formatted_competitor = {
            "id": competitor["id"],
            "name": competitor["name"],
            "domain": competitor["domain"],
            "tier": competitor["tier"],
            "industry": competitor["industry"],
            "description": competitor["description"],
            "platforms": competitor["platforms"],
            "monitoringKeywords": competitor["monitoringKeywords"],
            "isActive": competitor["isActive"],
            "createdAt": competitor["createdAt"],
            "updatedAt": competitor["updatedAt"],
            "lastMonitored": competitor.get("lastMonitored")
        }
        
        return formatted_competitor
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get competitor: {str(e)}")

@router.put("/competitors/{competitor_id}")
async def update_competitor(competitor_id: str, updates: CompetitorCreate):
    """Update a competitor."""
    try:
        # Convert Pydantic model to dictionary for database service
        update_dict = {
            "name": updates.name,
            "domain": updates.domain,
            "tier": updates.tier.value,
            "industry": updates.industry.value,
            "description": updates.description,
            "platforms": [platform.value for platform in updates.platforms] if updates.platforms else [],
            "monitoring_keywords": updates.monitoring_keywords or []
        }
        
        competitor = await ci_db.update_competitor(competitor_id, update_dict)
        if not competitor:
            raise HTTPException(status_code=404, detail="Competitor not found")
        
        # Convert to frontend format
        formatted_competitor = {
            "id": competitor["id"],
            "name": competitor["name"],
            "domain": competitor["domain"],
            "tier": competitor["tier"],
            "industry": competitor["industry"],
            "description": competitor["description"],
            "platforms": competitor["platforms"],
            "monitoringKeywords": competitor["monitoringKeywords"],
            "isActive": competitor["isActive"],
            "createdAt": competitor["createdAt"],
            "updatedAt": competitor["updatedAt"],
            "lastMonitored": competitor.get("lastMonitored")
        }
        
        return {
            "success": True,
            "message": "Competitor updated successfully",
            "competitor": formatted_competitor
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update competitor: {str(e)}")

@router.delete("/competitors/{competitor_id}")
async def delete_competitor(competitor_id: str):
    """Delete a competitor."""
    try:
        success = await ci_db.delete_competitor(competitor_id)
        if not success:
            raise HTTPException(status_code=404, detail="Competitor not found")
        
        return {
            "success": True,
            "message": "Competitor deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete competitor: {str(e)}")

@router.post("/competitors/{competitor_id}/monitor")
async def monitor_competitor_content(competitor_id: str):
    """Trigger content monitoring for a specific competitor."""
    try:
        result = await content_monitoring_service.monitor_competitor(competitor_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to monitor competitor: {str(e)}")

@router.get("/competitors/{competitor_id}/content")
async def get_competitor_content(
    competitor_id: str,
    content_type: Optional[str] = None,
    days_back: int = 30,
    limit: int = 50
):
    """Get stored content for a competitor."""
    try:
        content = await content_monitoring_service.get_competitor_content(
            competitor_id, content_type, days_back, limit
        )
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get competitor content: {str(e)}")

@router.post("/monitoring/run-all")
async def run_monitoring_for_all_competitors():
    """Run content monitoring for all active competitors."""
    try:
        result = await content_monitoring_service.monitor_all_active_competitors()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run monitoring: {str(e)}")

@router.post("/competitors/{competitor_id}/social/discover")
async def discover_social_handles(competitor_id: str):
    """Discover social media handles for a competitor."""
    try:
        handles = await social_media_service.discover_social_handles(competitor_id)
        return {"success": True, "handles": handles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover social handles: {str(e)}")

@router.post("/competitors/{competitor_id}/social/monitor")
async def monitor_competitor_social_media(competitor_id: str):
    """Monitor social media for a specific competitor."""
    try:
        result = await social_media_service.monitor_competitor_social_media(competitor_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to monitor social media: {str(e)}")

@router.get("/competitors/{competitor_id}/social/posts")
async def get_competitor_social_posts(
    competitor_id: str,
    platform: Optional[str] = None,
    days_back: int = 30,
    limit: int = 50
):
    """Get social media posts for a competitor."""
    try:
        posts = await social_media_service.get_competitor_social_posts(
            competitor_id, platform, days_back, limit
        )
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get social posts: {str(e)}")

@router.get("/competitors/{competitor_id}/social/analytics")
async def get_competitor_social_analytics(competitor_id: str):
    """Get social media analytics for a competitor."""
    try:
        analytics = await social_media_service.get_social_analytics(competitor_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get social analytics: {str(e)}")

@router.post("/analyze/comprehensive")
async def run_comprehensive_analysis(
    background_tasks: BackgroundTasks,
    competitor_ids: List[str],
    industry: Industry,
    your_content_topics: List[str] = [],
    analysis_depth: str = "standard"
):
    """Run comprehensive competitive intelligence analysis."""
    try:
        # In a real implementation, you'd:
        # 1. Fetch competitors from database
        # 2. Run the analysis in background
        # 3. Return job ID for status tracking
        
        return {
            "success": True,
            "message": "Comprehensive analysis started",
            "job_id": "analysis-job-123",
            "estimated_completion": "15-30 minutes",
            "competitors_count": len(competitor_ids),
            "analysis_depth": analysis_depth
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@router.post("/analyze/incremental")
async def run_incremental_monitoring(
    competitor_ids: List[str],
    hours_since_last_check: int = 4
):
    """Run incremental monitoring for recent competitor activity."""
    try:
        return {
            "success": True,
            "message": "Incremental monitoring completed",
            "new_content_found": 0,
            "alerts_generated": 0,
            "next_check_in": f"{hours_since_last_check} hours"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run monitoring: {str(e)}")

@router.get("/trends")
async def get_trends(
    industry: Optional[Industry] = None,
    time_range_days: int = 30,
    trend_type: Optional[str] = None
):
    """Get current market trends."""
    try:
        # Analyze content trends
        content_trends = await trend_analysis_engine.analyze_content_trends(
            industry=industry.value if industry else None,
            days_back=time_range_days
        )
        
        # Analyze social media trends
        social_trends = await trend_analysis_engine.analyze_social_media_trends(
            industry=industry.value if industry else None,
            days_back=time_range_days
        )
        
        all_trends = content_trends + social_trends
        
        # Filter by trend type if specified
        if trend_type:
            all_trends = [t for t in all_trends if t.trend_type == trend_type]
            
        # Convert to serializable format
        trends_data = []
        for trend in all_trends:
            trends_data.append({
                'id': trend.id,
                'title': trend.title,
                'description': trend.description,
                'trend_type': trend.trend_type,
                'strength': trend.strength,
                'confidence': trend.confidence,
                'data_points': trend.data_points,
                'timeframe': trend.timeframe,
                'industries': trend.industries,
                'competitors_involved': trend.competitors_involved,
                'metadata': trend.metadata,
                'created_at': trend.created_at.isoformat()
            })
            
        return trends_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")

@router.get("/insights")
async def get_market_insights(
    industry: Optional[Industry] = None,
    days_back: int = 30
):
    """Get strategic market insights."""
    try:
        # Get trends first
        content_trends = await trend_analysis_engine.analyze_content_trends(
            industry=industry.value if industry else None,
            days_back=days_back
        )
        
        social_trends = await trend_analysis_engine.analyze_social_media_trends(
            industry=industry.value if industry else None,
            days_back=days_back
        )
        
        all_trends = content_trends + social_trends
        
        # Generate insights
        insights = await trend_analysis_engine.generate_market_insights(
            trends=all_trends,
            industry=industry.value if industry else None
        )
        
        # Convert to serializable format
        insights_data = []
        for insight in insights:
            insights_data.append({
                'id': insight.id,
                'title': insight.title,
                'insight_type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'impact_level': insight.impact_level,
                'supporting_data': insight.supporting_data,
                'recommendations': insight.recommendations,
                'metadata': insight.metadata,
                'created_at': insight.created_at.isoformat()
            })
            
        return insights_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@router.get("/gaps", response_model=List[Dict[str, Any]])
async def get_content_gaps(
    industry: Optional[Industry] = None,
    min_opportunity_score: float = 50.0
):
    """Get identified content gaps and opportunities."""
    try:
        # In a real implementation, query gaps from database
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get content gaps: {str(e)}")

@router.get("/insights", response_model=List[InsightSummary])
async def get_insights(
    competitor_id: Optional[str] = None,
    insight_type: Optional[str] = None,
    days_back: int = 30
):
    """Get strategic insights about competitors."""
    try:
        # In a real implementation, query insights from database
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@router.get("/dashboard")
async def get_dashboard_data(
    industry: Industry,
    competitor_ids: List[str] = [],
    time_range_days: int = 30
):
    """Get dashboard data for competitive intelligence overview."""
    try:
        # In a real implementation, you'd call the orchestrator
        # dashboard_data = await get_orchestrator().get_competitor_dashboard_data(
        #     competitors, industry, time_range_days
        # )
        
        # Count stored competitors for the given industry using database service
        competitors_count = await ci_db.get_competitors_count_by_industry(industry.value)
        
        return {
            "overview": {
                "competitorsMonitored": competitors_count,
                "industry": industry.value,
                "lastUpdated": datetime.utcnow().isoformat(),
                "analysisStatus": "ready"
            },
            "keyMetrics": {
                "totalContentAnalyzed": 0,
                "trendsIdentified": 0,
                "opportunitiesFound": 0,
                "insightsGenerated": 0,
                "highPriorityOpportunities": 0
            },
            "message": "Connect database to see real data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

# ===== Webhooks & Digests (MVP) =====

@router.post("/webhooks/subscribe")
async def subscribe_webhook(name: str, target_url: str, event_types: Optional[List[str]] = None, secret_hmac: Optional[str] = None):
    try:
        sub = await webhook_service.create_subscription(name, target_url, event_types or [], secret_hmac)
        return {"success": True, "subscription": sub}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create webhook subscription: {str(e)}")

@router.get("/webhooks/subscriptions")
async def list_webhook_subscriptions():
    try:
        subs = await webhook_service.list_subscriptions()
        return subs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list webhook subscriptions: {str(e)}")

@router.delete("/webhooks/subscriptions/{subscription_id}")
async def delete_webhook_subscription(subscription_id: str):
    try:
        ok = await webhook_service.remove_subscription(subscription_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete webhook subscription: {str(e)}")

@router.post("/digests/subscribe")
async def subscribe_digest(channel: str, address_or_webhook: str, frequency: str = "daily", timezone: str = "UTC"):
    try:
        sub = await digest_service.subscribe(channel, address_or_webhook, frequency, timezone)
        return {"success": True, "subscription": sub}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create digest subscription: {str(e)}")

@router.get("/digests/subscriptions")
async def list_digest_subscriptions():
    try:
        subs = await digest_service.list_subscriptions()
        return subs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list digest subscriptions: {str(e)}")

@router.post("/digests/test")
async def send_test_digest(channel: str, address_or_webhook: str, hours_back: int = 24):
    try:
        await digest_service.send_test_digest(channel, address_or_webhook, hours_back)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test digest: {str(e)}")

# ===== Recent Change Events =====

@router.get("/changes/recent")
async def get_recent_changes(
    competitor_id: Optional[str] = None,
    change_type: Optional[str] = None,
    limit: int = 20
):
    """List recent change events (joins competitor name)."""
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                query = (
                    """
                    SELECT e.*, c.name as competitor_name
                    FROM ci_change_events e
                    JOIN ci_competitors c ON e.competitor_id = c.id
                    WHERE 1=1
                    """
                )
                params: list = []
                if competitor_id:
                    query += " AND e.competitor_id = %s"
                    params.append(competitor_id)
                if change_type:
                    query += " AND e.change_type = %s"
                    params.append(change_type)
                query += " ORDER BY e.detected_at DESC LIMIT %s"
                params.append(limit)
                cur.execute(query, params)
                rows = [dict(r) for r in cur.fetchall()]
                # Ensure datetime serialization and json output shape
                for r in rows:
                    if r.get("detected_at"):
                        r["detected_at"] = r["detected_at"].isoformat()
                return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent changes: {str(e)}")

# ===== Pricing Change Detection =====

@router.post("/detect/pricing/{competitor_id}")
async def detect_pricing_for_competitor(competitor_id: str):
    try:
        result = await change_detection_service.detect_pricing_for_competitor(competitor_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect pricing: {str(e)}")

@router.post("/detect/pricing/run-all")
async def detect_pricing_for_all():
    try:
        result = await change_detection_service.detect_pricing_for_all()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run pricing detection: {str(e)}")

@router.post("/detect/copy/{competitor_id}")
async def detect_copy_for_competitor(competitor_id: str):
    try:
        result = await change_detection_service.detect_copy_for_competitor(competitor_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect copy changes: {str(e)}")

@router.post("/detect/copy/run-all")
async def detect_copy_for_all():
    try:
        result = await change_detection_service.detect_copy_for_all()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run copy detection: {str(e)}")

# ===== News Ingestion =====

@router.post("/ingest/news/{competitor_id}")
async def ingest_news_for_competitor(competitor_id: str, days_back: int = 7):
    try:
        result = await news_ingestion_service.ingest_for_competitor(competitor_id, days_back)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest news: {str(e)}")

@router.post("/ingest/news/run-all")
async def ingest_news_for_all(days_back: int = 7):
    try:
        result = await news_ingestion_service.ingest_for_all(days_back)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest news for all: {str(e)}")

@router.post("/alerts/subscribe")
async def create_alert_subscription(subscription: AlertSubscription):
    """Create a new alert subscription."""
    try:
        # In a real implementation, save to database
        return {
            "success": True,
            "message": "Alert subscription created",
            "subscription_id": "sub-123"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create subscription: {str(e)}")

@router.get("/alerts")
async def get_alerts(
    limit: int = 50,
    priority: Optional[str] = None,
    alert_type: Optional[str] = None,
    competitor_id: Optional[str] = None,
    unread_only: bool = False
):
    """Get alerts with filtering options."""
    try:
        # Convert string enums to enum objects
        priority_enum = AlertPriority(priority) if priority else None
        alert_type_enum = AlertType(alert_type) if alert_type else None
        
        alerts = await alert_service.get_alerts(
            limit=limit,
            priority=priority_enum,
            alert_type=alert_type_enum,
            competitor_id=competitor_id,
            unread_only=unread_only
        )
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.get("/alerts/grouped")
async def get_alerts_grouped(
    days_back: int = 7,
    limit_per_competitor: int = 10,
    priority: Optional[str] = None,
    alert_type: Optional[str] = None,
    competitor_ids: Optional[List[str]] = None,
    unread_only: bool = False
):
    """Get recent alerts grouped by competitor."""
    try:
        priority_enum = AlertPriority(priority) if priority else None
        alert_type_enum = AlertType(alert_type) if alert_type else None

        grouped = await alert_service.get_alerts_grouped_by_competitor(
            days_back=days_back,
            limit_per_competitor=limit_per_competitor,
            priority=priority_enum,
            alert_type=alert_type_enum,
            competitor_ids=competitor_ids,
            unread_only=unread_only
        )
        return grouped
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get grouped alerts: {str(e)}")

@router.post("/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: str):
    """Mark an alert as read."""
    try:
        success = await alert_service.mark_alert_read(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"success": True, "message": "Alert marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark alert as read: {str(e)}")

@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(alert_id: str):
    """Dismiss an alert."""
    try:
        success = await alert_service.dismiss_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"success": True, "message": "Alert dismissed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to dismiss alert: {str(e)}")

@router.get("/alerts/summary")
async def get_alert_summary():
    """Get alert summary statistics."""
    try:
        summary = await alert_service.get_alert_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert summary: {str(e)}")

@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get alert details by ID."""
    try:
        alert = await alert_service.get_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        return alert
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert: {str(e)}")

@router.get("/status")
async def get_system_status():
    """Get competitor intelligence system status."""
    try:
        status = await get_orchestrator().get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "competitor-intelligence",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# ========== PHASE 4 ENDPOINTS: AI ANALYSIS & INTEGRATIONS ==========

@router.post("/analyze/content")
async def analyze_content_with_ai(
    content: str,
    content_url: Optional[str] = None,
    competitor_name: Optional[str] = None,
    content_type: str = "article"
):
    """Analyze content using AI for quality scoring, topics, and insights."""
    try:
        # Convert content type string to enum
        ai_content_type = AIContentType.ARTICLE
        try:
            ai_content_type = AIContentType(content_type.lower())
        except ValueError:
            pass  # Use default
        
        result = await ai_content_analyzer.analyze_content(
            content=content,
            content_url=content_url,
            competitor_name=competitor_name,
            content_type=ai_content_type
        )
        
        # Convert dataclass to dict for JSON serialization
        return {
            "success": True,
            "analysis": {
                "content_id": result.content_id,
                "content_type": result.content_type.value,
                "processing_time_ms": result.processing_time_ms,
                "analysis_timestamp": result.analysis_timestamp.isoformat(),
                "topics": {
                    "primary_topics": result.topics.primary_topics,
                    "secondary_topics": result.topics.secondary_topics,
                    "entities": result.topics.entities,
                    "keywords": result.topics.keywords,
                    "themes": result.topics.themes,
                    "confidence": result.topics.confidence
                },
                "quality": {
                    "overall_score": result.quality.overall_score,
                    "readability_score": result.quality.readability_score,
                    "engagement_potential": result.quality.engagement_potential,
                    "seo_score": result.quality.seo_score,
                    "information_density": result.quality.information_density,
                    "originality_score": result.quality.originality_score,
                    "structure_score": result.quality.structure_score,
                    "quality_rating": result.quality.quality_rating.value
                },
                "sentiment": {
                    "overall_sentiment": result.sentiment.overall_sentiment,
                    "sentiment_score": result.sentiment.sentiment_score,
                    "emotional_tone": result.sentiment.emotional_tone,
                    "confidence": result.sentiment.confidence,
                    "key_phrases": result.sentiment.key_phrases
                },
                "competitive_insights": {
                    "content_strategy": result.competitive_insights.content_strategy,
                    "target_audience": result.competitive_insights.target_audience,
                    "positioning": result.competitive_insights.positioning,
                    "strengths": result.competitive_insights.strengths,
                    "weaknesses": result.competitive_insights.weaknesses,
                    "differentiation_opportunities": result.competitive_insights.differentiation_opportunities,
                    "threat_level": result.competitive_insights.threat_level
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze content: {str(e)}")

@router.post("/reports/generate")
async def generate_report(
    report_type: str,
    format: str,
    title: str,
    description: str = "",
    include_charts: bool = True,
    include_raw_data: bool = False,
    date_range_days: int = 30,
    competitor_ids: List[str] = [],
    industry: Optional[str] = None,
    custom_sections: List[str] = []
):
    """Generate comprehensive reports in multiple formats."""
    try:
        # Convert string enums to enum objects
        try:
            report_type_enum = ReportType(report_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid report type: {report_type}")
        
        try:
            format_enum = ReportFormat(format.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid format: {format}")
        
        # Create report configuration
        config = ReportConfig(
            report_type=report_type_enum,
            format=format_enum,
            title=title,
            description=description,
            include_charts=include_charts,
            include_raw_data=include_raw_data,
            date_range_days=date_range_days,
            competitor_ids=competitor_ids if competitor_ids else None,
            industry=industry,
            custom_sections=custom_sections if custom_sections else None
        )
        
        # Generate report
        report = await advanced_reporting_service.generate_report(config)
        
        # Notify integrations about report generation
        await external_integrations_service.notify_report_generated({
            "title": report.config.title,
            "type": report.config.report_type.value,
            "format": report.config.format.value,
            "file_path": report.file_path,
            "file_size_bytes": report.file_size_bytes,
            "generation_time_ms": report.generation_time_ms
        })
        
        return {
            "success": True,
            "report": {
                "report_id": report.report_id,
                "file_path": report.file_path,
                "file_size_bytes": report.file_size_bytes,
                "generation_time_ms": report.generation_time_ms,
                "created_at": report.created_at.isoformat(),
                "sections_count": len(report.sections),
                "metadata": report.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/reports/download/{report_id}")
async def download_report(report_id: str):
    """Download generated report file."""
    try:
        # In a real implementation, you would:
        # 1. Look up the report by ID in database
        # 2. Verify user permissions
        # 3. Return the file
        
        from fastapi.responses import FileResponse
        import os
        
        # This is a placeholder - in production you'd store report metadata in DB
        # and retrieve the actual file path
        raise HTTPException(status_code=404, detail="Report not found or expired")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")

@router.post("/integrations/register")
async def register_integration(
    integration_type: str,
    name: str,
    webhook_url: Optional[str] = None,
    api_token: Optional[str] = None,
    channel: Optional[str] = None,
    email_settings: Optional[Dict[str, str]] = None,
    event_filters: List[str] = [],
    priority_threshold: str = "normal",
    enabled: bool = True
):
    """Register a new external integration."""
    try:
        # Convert string enums
        try:
            integration_type_enum = IntegrationType(integration_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid integration type: {integration_type}")
        
        try:
            priority_enum = MessagePriority(priority_threshold.lower())
        except ValueError:
            priority_enum = MessagePriority.NORMAL
        
        # Convert event filters
        event_filter_enums = []
        for event in event_filters:
            try:
                event_filter_enums.append(EventType(event.lower()))
            except ValueError:
                continue  # Skip invalid event types
        
        # Create integration config
        config = IntegrationConfig(
            integration_type=integration_type_enum,
            name=name,
            enabled=enabled,
            webhook_url=webhook_url,
            api_token=api_token,
            channel=channel,
            email_settings=email_settings,
            event_filters=event_filter_enums if event_filter_enums else None,
            priority_threshold=priority_enum
        )
        
        # Register integration
        success = external_integrations_service.register_integration(config)
        
        if success:
            return {
                "success": True,
                "message": f"Integration '{name}' registered successfully",
                "integration_name": name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register integration")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register integration: {str(e)}")

@router.get("/integrations/status")
async def get_integrations_status():
    """Get status of all registered integrations."""
    try:
        status = external_integrations_service.get_integration_status()
        return {
            "success": True,
            "integrations": status,
            "total_integrations": len(status),
            "enabled_integrations": sum(1 for config in status.values() if config["enabled"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get integration status: {str(e)}")

@router.post("/integrations/{integration_name}/test")
async def test_integration(integration_name: str):
    """Test a specific integration."""
    try:
        result = await external_integrations_service.test_integration(integration_name)
        
        return {
            "success": result.success,
            "integration_name": result.integration_name,
            "message_id": result.message_id,
            "error": result.error,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test integration: {str(e)}")

@router.delete("/integrations/{integration_name}")
async def remove_integration(integration_name: str):
    """Remove an integration."""
    try:
        success = external_integrations_service.remove_integration(integration_name)
        
        if success:
            return {
                "success": True,
                "message": f"Integration '{integration_name}' removed successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Integration not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove integration: {str(e)}")

@router.post("/integrations/notify")
async def send_custom_notification(
    title: str,
    content: str,
    event_type: str = "custom",
    priority: str = "normal",
    integration_names: List[str] = [],
    data: Dict[str, Any] = {}
):
    """Send custom notification through integrations."""
    try:
        # Convert enums
        try:
            event_type_enum = EventType(event_type.lower())
        except ValueError:
            event_type_enum = EventType.CUSTOM
        
        try:
            priority_enum = MessagePriority(priority.lower())
        except ValueError:
            priority_enum = MessagePriority.NORMAL
        
        # Create notification message
        message = NotificationMessage(
            title=title,
            content=content,
            event_type=event_type_enum,
            priority=priority_enum,
            data=data
        )
        
        # Send notification
        results = await external_integrations_service.send_notification(
            message, 
            integration_names if integration_names else None
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "success": result.success,
                "integration_name": result.integration_name,
                "message_id": result.message_id,
                "error": result.error,
                "timestamp": result.timestamp.isoformat() if result.timestamp else None
            })
        
        return {
            "success": True,
            "message": "Notification sent",
            "results": formatted_results,
            "total_sent": len([r for r in results if r.success]),
            "total_failed": len([r for r in results if not r.success])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")

@router.post("/integrations/broadcast/health")
async def broadcast_system_health():
    """Broadcast system health status to all integrations."""
    try:
        # Get system health data
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "competitor-intelligence",
            "version": "1.0.0"
        }
        
        results = await external_integrations_service.broadcast_system_health(health_data)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "success": result.success,
                "integration_name": result.integration_name,
                "error": result.error
            })
        
        return {
            "success": True,
            "message": "Health status broadcasted",
            "results": formatted_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast health status: {str(e)}")

# Export functions for content quality analysis
@router.post("/analyze/batch-content")
async def batch_analyze_content(
    content_items: List[Dict[str, Any]]
):
    """Analyze multiple content items in batch."""
    try:
        results = []
        
        # Process in parallel (limit concurrency to avoid overwhelming the AI service)
        semaphore = asyncio.Semaphore(5)
        
        async def analyze_single_item(item):
            async with semaphore:
                return await ai_content_analyzer.analyze_content(
                    content=item.get("content", ""),
                    content_url=item.get("content_url"),
                    competitor_name=item.get("competitor_name"),
                    content_type=AIContentType(item.get("content_type", "article"))
                )
        
        tasks = [analyze_single_item(item) for item in content_items]
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                results.append({
                    "success": False,
                    "error": str(result),
                    "content_index": i
                })
            else:
                results.append({
                    "success": True,
                    "content_index": i,
                    "content_id": result.content_id,
                    "overall_quality_score": result.quality.overall_score,
                    "sentiment_score": result.sentiment.sentiment_score,
                    "threat_level": result.competitive_insights.threat_level,
                    "primary_topics": result.topics.primary_topics[:3]  # Top 3 topics
                })
        
        return {
            "success": True,
            "total_analyzed": len(content_items),
            "successful_analyses": len([r for r in results if r.get("success")]),
            "failed_analyses": len([r for r in results if not r.get("success")]),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to batch analyze content: {str(e)}")