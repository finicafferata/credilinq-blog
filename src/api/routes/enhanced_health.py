"""
Enhanced health check and monitoring endpoints with comprehensive system metrics.
Provides detailed health information for monitoring and alerting systems.
"""

import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.database import db_config
from ...config.settings import settings
from ...core.cache import cache, cache_health_check
from ...core.rate_limiting import get_rate_limit_status

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    version: str
    environment: str
    uptime_seconds: float
    checks: Dict[str, Any]


class DetailedHealthStatus(BaseModel):
    """Detailed health check response model."""
    status: str
    timestamp: str
    version: str
    environment: str
    uptime_seconds: float
    system_metrics: Dict[str, Any]
    service_checks: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    rate_limiting: Optional[Dict[str, Any]] = None


# Track application start time
app_start_time = time.time()


async def check_database_health() -> Dict[str, Any]:
    """Comprehensive database health check."""
    try:
        from ...agents.core.database_service import get_db_service
        
        start_time = time.time()
        db_service = get_db_service()
        
        # Basic health check
        health_data = db_service.health_check()
        
        # Performance metrics
        query_time = time.time() - start_time
        
        # Additional checks
        checks = {
            "basic_connectivity": health_data.get("status") == "healthy",
            "query_performance": query_time < 1.0,  # Should complete within 1 second
            "response_time_ms": round(query_time * 1000, 2)
        }
        
        # Try a simple query to test read performance
        try:
            start_time = time.time()
            # This should be a lightweight query
            db_service.execute_query("SELECT 1 as test_query", {})
            read_time = time.time() - start_time
            checks["read_performance"] = read_time < 0.5
            checks["read_time_ms"] = round(read_time * 1000, 2)
        except Exception as e:
            checks["read_performance"] = False
            checks["read_error"] = str(e)
        
        overall_status = "healthy" if all(
            checks.get(k, False) for k in ["basic_connectivity", "query_performance"]
        ) else "degraded"
        
        return {
            "status": overall_status,
            "checks": checks,
            "details": health_data
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "checks": {"basic_connectivity": False},
            "fallback": db_config.health_check()
        }


async def check_cache_health() -> Dict[str, Any]:
    """Check Redis cache health."""
    try:
        return await cache_health_check()
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False
        }


async def check_ai_services_health() -> Dict[str, Any]:
    """Check AI services connectivity."""
    checks = {}
    
    # OpenAI API check
    try:
        import openai
        openai.api_key = settings.openai_api_key
        
        start_time = time.time()
        # Simple test request
        response = await asyncio.wait_for(
            openai.Model.alist(),
            timeout=5.0
        )
        response_time = time.time() - start_time
        
        checks["openai"] = {
            "status": "healthy",
            "response_time_ms": round(response_time * 1000, 2),
            "models_available": len(response.data) if hasattr(response, 'data') else 0
        }
    except asyncio.TimeoutError:
        checks["openai"] = {
            "status": "unhealthy",
            "error": "Request timeout"
        }
    except Exception as e:
        checks["openai"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Additional AI service checks can be added here
    overall_status = "healthy" if checks.get("openai", {}).get("status") == "healthy" else "degraded"
    
    return {
        "status": overall_status,
        "services": checks
    }


def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "usage_percent": round(cpu_percent, 2),
                "count": psutil.cpu_count(),
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": round(memory.percent, 2),
                "free_gb": round(memory.free / (1024**3), 2)
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_percent": round((disk.used / disk.total) * 100, 2)
            },
            "network": {
                "connections": len(psutil.net_connections()),
            }
        }
    except Exception as e:
        return {
            "error": f"Failed to get system metrics: {str(e)}"
        }


def get_performance_metrics() -> Dict[str, Any]:
    """Get application performance metrics."""
    uptime = time.time() - app_start_time
    
    return {
        "uptime": {
            "seconds": round(uptime, 2),
            "human_readable": str(timedelta(seconds=int(uptime)))
        },
        "application": {
            "version": settings.api_version,
            "environment": settings.environment,
            "debug_mode": settings.debug
        },
        "configuration": {
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "cache_enabled": settings.enable_cache,
            "api_timeout": settings.api_timeout,
            "max_concurrent_agents": settings.max_concurrent_agents
        }
    }


@router.get("/health", response_model=HealthStatus)
async def basic_health_check():
    """Basic health check endpoint for load balancers."""
    start_time = time.time()
    
    # Quick checks only
    db_health = await check_database_health()
    cache_health = await check_cache_health()
    
    # Determine overall status
    db_healthy = db_health.get("status") in ["healthy", "degraded"]
    cache_healthy = cache_health.get("status") in ["healthy", "degraded"]
    
    if db_healthy and cache_healthy:
        overall_status = "healthy"
    elif db_healthy or cache_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    response_time = time.time() - start_time
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=settings.api_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - app_start_time, 2),
        checks={
            "database": db_health.get("status"),
            "cache": cache_health.get("status"),
            "response_time_ms": round(response_time * 1000, 2)
        }
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(request: Request):
    """Comprehensive health check with detailed metrics."""
    start_time = time.time()
    
    # Run all health checks concurrently
    db_check_task = asyncio.create_task(check_database_health())
    cache_check_task = asyncio.create_task(check_cache_health())
    ai_check_task = asyncio.create_task(check_ai_services_health())
    
    # Get system metrics
    system_metrics = get_system_metrics()
    performance_metrics = get_performance_metrics()
    
    # Wait for async checks
    db_health = await db_check_task
    cache_health = await cache_check_task
    ai_health = await ai_check_task
    
    # Get rate limiting status if available
    rate_limiting_status = None
    try:
        rate_limiting_status = await get_rate_limit_status(request)
    except Exception as e:
        rate_limiting_status = {"error": str(e)}
    
    # Determine overall status
    service_statuses = [
        db_health.get("status"),
        cache_health.get("status"),
        ai_health.get("status")
    ]
    
    healthy_count = sum(1 for status in service_statuses if status == "healthy")
    degraded_count = sum(1 for status in service_statuses if status == "degraded")
    
    if healthy_count == len(service_statuses):
        overall_status = "healthy"
    elif healthy_count + degraded_count == len(service_statuses):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    response_time = time.time() - start_time
    performance_metrics["health_check_time_ms"] = round(response_time * 1000, 2)
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=settings.api_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - app_start_time, 2),
        system_metrics=system_metrics,
        service_checks={
            "database": db_health,
            "cache": cache_health,
            "ai_services": ai_health
        },
        performance_metrics=performance_metrics,
        rate_limiting=rate_limiting_status
    )


@router.get("/health/database")
async def database_health_check():
    """Specific database health check."""
    return await check_database_health()


@router.get("/health/cache")
async def cache_health_check_endpoint():
    """Specific cache health check."""
    return await check_cache_health()


@router.get("/health/ai-services")
async def ai_services_health_check():
    """Specific AI services health check."""
    return await check_ai_services_health()


@router.get("/health/system")
async def system_health_check():
    """System metrics and resource usage."""
    return {
        "system_metrics": get_system_metrics(),
        "performance_metrics": get_performance_metrics(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/rate-limits")
async def rate_limits_status(request: Request):
    """Current rate limiting status."""
    try:
        return await get_rate_limit_status(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rate limit status: {str(e)}")


@router.get("/readiness")
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    # Check if critical services are ready
    db_health = await check_database_health()
    
    if db_health.get("status") in ["healthy", "degraded"]:
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": "Database not available",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    # Simple check that the application is running
    uptime = time.time() - app_start_time
    
    return {
        "status": "alive",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat()
    }