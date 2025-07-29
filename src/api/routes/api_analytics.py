"""
API analytics dashboard endpoints.
Provides comprehensive insights into API usage, performance, and security metrics.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ...core.api_analytics import analytics_collector, APIAnalyticsEvent
from ...core.auth import get_current_user, require_admin_access
from ...core.monitoring import metrics

router = APIRouter()

class AnalyticsSummary(BaseModel):
    """API analytics summary response model."""
    timeframe: str
    total_requests: int
    successful_requests: int
    error_requests: int
    error_rate: float
    avg_response_time_ms: float
    total_data_transferred_mb: float
    unique_users: int
    top_endpoints: Dict[str, int]
    status_code_distribution: Dict[int, int]

class RealTimeMetrics(BaseModel):
    """Real-time API metrics response model."""
    current_rps: float  # Requests per second
    current_error_rate: float
    avg_response_time_ms: float
    active_connections: int
    cache_hit_rate: float
    system_health_score: float

class UserAnalytics(BaseModel):
    """User-specific analytics response model."""
    user_id: str
    total_requests: int
    success_rate: float
    avg_response_time_ms: float
    most_used_endpoints: Dict[str, int]
    last_activity: str
    geographic_locations: Dict[str, int]

class EndpointPerformance(BaseModel):
    """Endpoint performance metrics."""
    endpoint: str
    method: str
    total_requests: int
    success_rate: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    error_distribution: Dict[str, int]

class SecurityMetrics(BaseModel):
    """Security-related metrics."""
    auth_failures_24h: int
    rate_limit_violations_24h: int
    suspicious_ips: List[str]
    blocked_requests_24h: int
    geographic_anomalies: List[Dict[str, Any]]

@router.get("/analytics/summary", response_model=Dict[str, Any])
async def get_analytics_summary(
    period: str = Query("day", description="Period: hour, day, week, month"),
    limit: int = Query(24, description="Number of periods to include"),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive API analytics summary."""
    
    # Validate period
    if period not in ["hour", "day", "week", "month"]:
        raise HTTPException(status_code=400, detail="Invalid period. Use: hour, day, week, month")
    
    try:
        # Get usage analytics
        usage_data = analytics_collector.get_usage_analytics(period=period, limit=limit)
        
        # Get real-time metrics for current status
        real_time_data = analytics_collector.get_real_time_metrics(minutes=60)
        
        # Calculate trends
        timeline = usage_data.get("timeline", {})
        if len(timeline) >= 2:
            sorted_keys = sorted(timeline.keys(), reverse=True)
            current_period = timeline[sorted_keys[0]]
            previous_period = timeline[sorted_keys[1]]
            
            request_trend = (
                (current_period.total_requests - previous_period.total_requests) / 
                previous_period.total_requests * 100 
                if previous_period.total_requests > 0 else 0
            )
            
            error_trend = (
                (current_period.error_requests - previous_period.error_requests) / 
                max(previous_period.error_requests, 1) * 100
            )
            
            response_time_trend = (
                (current_period.avg_response_time_ms - previous_period.avg_response_time_ms) / 
                previous_period.avg_response_time_ms * 100
                if previous_period.avg_response_time_ms > 0 else 0
            )
        else:
            request_trend = error_trend = response_time_trend = 0
        
        return {
            "summary": usage_data["summary"],
            "real_time": {
                "requests_per_minute": real_time_data["requests_per_minute"],
                "current_error_rate": real_time_data["error_rate"],
                "avg_response_time_ms": real_time_data["avg_response_time_ms"]
            },
            "trends": {
                "request_growth_percent": round(request_trend, 2),
                "error_trend_percent": round(error_trend, 2),
                "response_time_trend_percent": round(response_time_trend, 2)
            },
            "timeline": timeline,
            "period": period,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics summary: {str(e)}")

@router.get("/analytics/real-time", response_model=Dict[str, Any])
async def get_real_time_metrics(
    minutes: int = Query(60, description="Minutes of real-time data to include"),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get real-time API metrics and system status."""
    
    try:
        # Get real-time metrics
        real_time_data = analytics_collector.get_real_time_metrics(minutes=minutes)
        
        # Get system metrics from monitoring
        system_metrics = metrics.get_all_metrics()
        
        # Calculate health score based on various factors
        error_rate = real_time_data["error_rate"]
        avg_response_time = real_time_data["avg_response_time_ms"]
        
        health_score = 100
        if error_rate > 0.05:  # More than 5% errors
            health_score -= min(50, error_rate * 1000)
        if avg_response_time > 1000:  # Slower than 1 second
            health_score -= min(30, (avg_response_time - 1000) / 100)
        
        health_score = max(0, health_score)
        
        return {
            "current_metrics": {
                "requests_per_second": real_time_data["requests_per_minute"] / 60,
                "error_rate": real_time_data["error_rate"],
                "avg_response_time_ms": real_time_data["avg_response_time_ms"],
                "total_requests": real_time_data["total_requests"],
                "data_transferred_mb": real_time_data["total_data_transferred_mb"]
            },
            "system_health": {
                "health_score": round(health_score, 1),
                "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy"
            },
            "timeline": real_time_data["timeline"],
            "time_range_minutes": minutes,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metrics: {str(e)}")

@router.get("/analytics/endpoints", response_model=Dict[str, Any])
async def get_endpoint_analytics(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get performance analytics for all API endpoints."""
    
    try:
        endpoint_data = analytics_collector.get_endpoint_analytics()
        
        return {
            "endpoint_performance": endpoint_data["endpoints"],
            "performance_insights": {
                "slowest_endpoints": endpoint_data["performance_summary"]["slowest_endpoints"],
                "highest_error_endpoints": endpoint_data["performance_summary"]["highest_error_rate_endpoints"],
                "most_popular_endpoints": endpoint_data["performance_summary"]["most_popular_endpoints"]
            },
            "recommendations": generate_performance_recommendations(endpoint_data),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get endpoint analytics: {str(e)}")

@router.get("/analytics/users/{user_id}", response_model=Dict[str, Any])
async def get_user_analytics(
    user_id: str,
    current_user: Dict = Depends(get_current_user),
    admin_access: bool = Depends(require_admin_access)
) -> Dict[str, Any]:
    """Get analytics for a specific user."""
    
    try:
        user_data = analytics_collector.get_user_analytics(user_id)
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User analytics not found")
        
        return {
            "user_analytics": user_data,
            "usage_insights": generate_user_insights(user_data),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user analytics: {str(e)}")

@router.get("/analytics/security", response_model=Dict[str, Any])
async def get_security_analytics(
    current_user: Dict = Depends(get_current_user),
    admin_access: bool = Depends(require_admin_access)
) -> Dict[str, Any]:
    """Get security-related analytics and threat intelligence."""
    
    try:
        security_data = analytics_collector.get_security_analytics()
        
        # Generate security insights
        threat_level = calculate_threat_level(security_data)
        security_recommendations = generate_security_recommendations(security_data)
        
        return {
            "security_metrics": security_data,
            "threat_assessment": {
                "level": threat_level,
                "score": calculate_security_score(security_data),
                "active_threats": identify_active_threats(security_data)
            },
            "recommendations": security_recommendations,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security analytics: {str(e)}")

@router.get("/analytics/export")
async def export_analytics_data(
    format: str = Query("json", description="Export format: json, csv"),
    period: str = Query("day", description="Period: hour, day, week"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: Dict = Depends(get_current_user),
    admin_access: bool = Depends(require_admin_access)
):
    """Export analytics data in various formats."""
    
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use: json, csv")
    
    try:
        # Get comprehensive analytics data
        usage_data = analytics_collector.get_usage_analytics(period=period, limit=100)
        endpoint_data = analytics_collector.get_endpoint_analytics()
        security_data = analytics_collector.get_security_analytics()
        
        export_data = {
            "export_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "period": period,
                "format": format,
                "start_date": start_date,
                "end_date": end_date
            },
            "usage_analytics": usage_data,
            "endpoint_analytics": endpoint_data,
            "security_analytics": security_data
        }
        
        if format == "csv":
            # Convert to CSV format
            import pandas as pd
            from io import StringIO
            
            # Create CSV data for different sections
            csv_buffer = StringIO()
            
            # Export usage timeline
            if usage_data.get("timeline"):
                timeline_df = pd.DataFrame([
                    {"timestamp": k, **v} 
                    for k, v in usage_data["timeline"].items()
                ])
                timeline_df.to_csv(csv_buffer, index=False)
            
            return {
                "content_type": "text/csv",
                "content": csv_buffer.getvalue(),
                "filename": f"api_analytics_{period}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        else:
            return export_data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export analytics data: {str(e)}")

@router.post("/analytics/alerts")
async def create_analytics_alert(
    alert_config: Dict[str, Any],
    current_user: Dict = Depends(get_current_user),
    admin_access: bool = Depends(require_admin_access)
) -> Dict[str, Any]:
    """Create a new analytics alert rule."""
    
    # Validate alert configuration
    required_fields = ["name", "metric", "condition", "threshold", "webhook_url"]
    for field in required_fields:
        if field not in alert_config:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    try:
        from ...core.monitoring import alert_manager
        
        # Add alert rule to monitoring system
        alert_manager.add_alert_rule(
            name=alert_config["name"],
            metric_name=alert_config["metric"],
            condition=alert_config["condition"],
            threshold=float(alert_config["threshold"])
        )
        
        return {
            "message": "Alert rule created successfully",
            "alert_id": alert_config["name"],
            "status": "active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")

# Helper functions for generating insights and recommendations

def generate_performance_recommendations(endpoint_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate performance optimization recommendations."""
    recommendations = []
    
    slowest_endpoints = endpoint_data["performance_summary"]["slowest_endpoints"]
    error_endpoints = endpoint_data["performance_summary"]["highest_error_rate_endpoints"]
    
    # Recommendations for slow endpoints
    for endpoint in slowest_endpoints[:3]:
        if endpoint["avg_response_time_ms"] > 2000:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "endpoint": endpoint["endpoint"],
                "issue": "High response time",
                "recommendation": f"Optimize {endpoint['endpoint']} - current avg response time is {endpoint['avg_response_time_ms']:.0f}ms. Consider adding caching, database optimization, or async processing."
            })
    
    # Recommendations for high error rate endpoints
    for endpoint in error_endpoints[:3]:
        if endpoint["error_rate"] > 0.1:  # More than 10% error rate
            recommendations.append({
                "type": "reliability",
                "priority": "critical",
                "endpoint": endpoint["endpoint"],
                "issue": "High error rate",
                "recommendation": f"Fix {endpoint['endpoint']} - error rate is {endpoint['error_rate']*100:.1f}%. Review error logs and improve error handling."
            })
    
    return recommendations

def generate_user_insights(user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate insights about user behavior."""
    insights = []
    
    metrics = user_data["metrics"]
    
    # Usage patterns
    if metrics["total_requests"] > 1000:
        insights.append({
            "type": "usage",
            "insight": "High-volume user",
            "details": f"User has made {metrics['total_requests']} requests with {metrics['successful_requests']/metrics['total_requests']*100:.1f}% success rate"
        })
    
    # Error patterns
    if metrics["error_requests"] > 100:
        insights.append({
            "type": "errors",
            "insight": "User experiencing frequent errors",
            "details": f"User has encountered {metrics['error_requests']} errors. Consider reaching out for support."
        })
    
    return insights

def calculate_threat_level(security_data: Dict[str, Any]) -> str:
    """Calculate overall threat level based on security metrics."""
    score = 0
    
    # Factor in various security indicators
    if security_data["auth_failures"] > 100:
        score += 30
    if security_data["rate_limit_violations"] > 50:
        score += 20
    if len(security_data["suspicious_ips"]) > 10:
        score += 25
    if len(security_data["high_volume_ips"]) > 5:
        score += 15
    
    if score >= 70:
        return "high"
    elif score >= 40:
        return "medium"
    else:
        return "low"

def calculate_security_score(security_data: Dict[str, Any]) -> float:
    """Calculate a security score (0-100, higher is better)."""
    base_score = 100
    
    # Deduct points for security issues
    base_score -= min(30, security_data["auth_failures"] / 10)
    base_score -= min(20, security_data["rate_limit_violations"] / 5)
    base_score -= min(25, len(security_data["suspicious_ips"]) * 2)
    base_score -= min(15, len(security_data["high_volume_ips"]))
    
    return max(0, base_score)

def identify_active_threats(security_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify active security threats."""
    threats = []
    
    if security_data["auth_failures"] > 50:
        threats.append({
            "type": "brute_force",
            "severity": "high" if security_data["auth_failures"] > 200 else "medium",
            "description": f"{security_data['auth_failures']} authentication failures in last 24h"
        })
    
    if len(security_data["suspicious_ips"]) > 0:
        threats.append({
            "type": "suspicious_traffic",
            "severity": "medium",
            "description": f"{len(security_data['suspicious_ips'])} suspicious IP addresses detected"
        })
    
    return threats

def generate_security_recommendations(security_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate security improvement recommendations."""
    recommendations = []
    
    if security_data["auth_failures"] > 100:
        recommendations.append({
            "type": "authentication",
            "priority": "high",
            "recommendation": "Consider implementing stricter rate limiting on authentication endpoints and adding CAPTCHA verification."
        })
    
    if len(security_data["suspicious_ips"]) > 5:
        recommendations.append({
            "type": "ip_blocking",
            "priority": "medium", 
            "recommendation": "Review and consider blocking suspicious IP addresses. Implement geographic IP filtering if appropriate."
        })
    
    return recommendations