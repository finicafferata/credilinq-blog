"""Health check and monitoring endpoints with comprehensive system status."""

import os
import psutil
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Response
from ...config.database import db_config, secure_db
from ...config.settings import settings
from ...core.metrics import get_metrics_response, metrics

router = APIRouter()

@router.get("/health/railway")
async def railway_health():
    """Railway-optimized health check endpoint."""
    try:
        # Quick database check
        db_status = "unknown"
        try:
            db_health = db_config.health_check()
            db_status = db_health.get("status", "unknown")
        except Exception:
            db_status = "error"
        
        # Railway environment info
        railway_info = {
            "service": os.getenv('RAILWAY_SERVICE_NAME', 'unknown'),
            "environment": os.getenv('RAILWAY_ENVIRONMENT', 'unknown'),
            "replica_id": os.getenv('RAILWAY_REPLICA_ID', 'unknown'),
            "deployment_id": os.getenv('RAILWAY_DEPLOYMENT_ID', 'unknown')
        }
        
        # Quick memory check
        try:
            memory = psutil.virtual_memory()
            memory_status = "healthy" if memory.percent < 85 else "warning"
        except Exception:
            memory_status = "unknown"
        
        overall_status = "healthy"
        if db_status == "error" or memory_status == "warning":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "credilinq-ai-platform",
            "version": "2.0.0-railway",
            "database": {"status": db_status},
            "memory": {"status": memory_status},
            "railway": railway_info
        }
        
    except Exception as e:
        return Response(
            content=f'{{"status": "error", "error": "{str(e)}"}}',
            status_code=500,
            media_type="application/json"
        )

# Track application start time
APP_START_TIME = datetime.utcnow()

def get_system_metrics() -> Dict[str, Any]:
    """Get current system resource metrics."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "percent": memory.percent
        }
        
        # Disk usage
        disk = psutil.disk_usage("/")
        disk_usage = {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "used_gb": round(disk.used / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "percent": disk.percent
        }
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network = {
            "bytes_sent_mb": round(net_io.bytes_sent / (1024 ** 2), 2),
            "bytes_recv_mb": round(net_io.bytes_recv / (1024 ** 2), 2),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": cpu_count,
                "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 90 else "critical"
            },
            "memory": memory_usage,
            "disk": disk_usage,
            "network": network
        }
    except Exception as e:
        return {
            "error": f"Failed to get system metrics: {str(e)}"
        }

def get_application_metrics() -> Dict[str, Any]:
    """Get application-specific metrics."""
    uptime = datetime.utcnow() - APP_START_TIME
    
    return {
        "version": "2.0.0",
        "environment": settings.environment,
        "uptime": {
            "days": uptime.days,
            "hours": uptime.seconds // 3600,
            "minutes": (uptime.seconds % 3600) // 60,
            "total_seconds": int(uptime.total_seconds())
        },
        "start_time": APP_START_TIME.isoformat(),
        "current_time": datetime.utcnow().isoformat()
    }

@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns system status including:
    - Database connectivity
    - System resources (CPU, memory, disk)
    - Application metrics
    - Service dependencies
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check database connectivity
    try:
        db_health = secure_db.health_check()
        health_status["checks"]["database"] = db_health
        if db_health.get("status") != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Get system metrics
    system_metrics = get_system_metrics()
    health_status["checks"]["system"] = system_metrics
    
    # Get application metrics
    app_metrics = get_application_metrics()
    health_status["checks"]["application"] = app_metrics
    
    # Check critical services
    services_status = {
        "ai_agents": "healthy",  # Simplified - would check agent pool in production
        "cache": "healthy" if settings.environment == "development" else "not_configured",
        "webhooks": "healthy",
        "monitoring": "healthy"
    }
    health_status["checks"]["services"] = services_status
    
    # Determine overall status
    if any(check.get("status") == "unhealthy" for check in health_status["checks"].values() if isinstance(check, dict)):
        health_status["status"] = "unhealthy"
    elif any(check.get("status") == "warning" for check in health_status["checks"].values() if isinstance(check, dict)):
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status

@router.get("/health/live")
async def liveness_check():
    """
    Simple liveness check for container orchestration.
    Returns 200 if the application is running.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check for load balancers.
    Returns 200 if the application is ready to serve traffic.
    """
    try:
        # Quick database connectivity check
        db_health = secure_db.health_check()
        
        if db_health.get("status") != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "reason": "Database not healthy",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "services": "operational"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/health/startup")
async def startup_check():
    """
    Startup probe for Kubernetes.
    Indicates if the application has finished its initialization.
    """
    uptime = datetime.utcnow() - APP_START_TIME
    
    # Consider app started after 10 seconds
    if uptime.total_seconds() < 10:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "starting",
                "uptime_seconds": int(uptime.total_seconds()),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return {
        "status": "started",
        "uptime_seconds": int(uptime.total_seconds()),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health/dependencies")
async def check_dependencies():
    """
    Check status of all external dependencies.
    """
    dependencies = {}
    
    # Database
    try:
        db_health = secure_db.health_check()
        dependencies["postgresql"] = {
            "status": db_health.get("status", "unknown"),
            "response_time_ms": db_health.get("checks", {}).get("response_time_ms", None)
        }
    except Exception as e:
        dependencies["postgresql"] = {
            "status": "unreachable",
            "error": str(e)
        }
    
    # OpenAI API (for AI agents)
    dependencies["openai"] = {
        "status": "configured" if settings.openai_api_key else "not_configured",
        "model": "gpt-4"
    }
    
    # Redis (if configured)
    dependencies["redis"] = {
        "status": "not_configured",
        "note": "Using in-memory cache"
    }
    
    # External APIs
    dependencies["external_apis"] = {
        "langchain": "configured" if getattr(settings, 'langchain_api_key', None) else "not_configured"
    }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies
    }

@router.get("/health/metrics")
async def get_metrics():
    """
    Get detailed application metrics for monitoring.
    """
    # Get system metrics
    system_metrics = get_system_metrics()
    
    # Get application metrics  
    app_metrics = get_application_metrics()
    
    # Database pool metrics
    db_metrics = {}
    try:
        if secure_db._pool:
            with secure_db._pool_lock:
                db_metrics = {
                    "pool_size": secure_db._pool.maxconn,
                    "connections_in_use": len(secure_db._pool._used) if hasattr(secure_db._pool, '_used') else 0
                }
    except:
        db_metrics = {"status": "unavailable"}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "application": app_metrics,
        "system": system_metrics,
        "database": db_metrics
    }


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update system metrics before exposing
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        metrics.update_system_metrics(
            memory_usage=memory.used,
            cpu_usage=cpu_percent
        )
        
        # Generate metrics response
        content, content_type = get_metrics_response()
        return Response(content=content, media_type=content_type)
    except Exception as e:
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )