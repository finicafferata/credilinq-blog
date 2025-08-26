"""
Simplified health check endpoints for Railway deployment.
Minimal dependencies to avoid startup crashes.
"""

import os
import time
from datetime import datetime
from fastapi import APIRouter, Response

router = APIRouter()

# Track application start time
START_TIME = time.time()

@router.get("/health/live")
async def liveness_check():
    """
    Minimal liveness check for Railway health checks.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@router.get("/ping")
async def ping():
    """Simple ping endpoint for Railway health checks."""
    return {"message": "pong", "timestamp": datetime.utcnow().isoformat()}

@router.get("/health/ready")
async def readiness_check():
    """
    Simple readiness check.
    Checks basic environment variables and startup time.
    """
    uptime = time.time() - START_TIME
    
    # Check critical environment variables
    has_db_url = bool(os.getenv("DATABASE_URL"))
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    
    # Consider ready after 30 seconds if env vars are present
    is_ready = uptime > 30 and has_db_url and has_openai_key
    
    if not is_ready:
        return Response(
            content='{"status":"not_ready","uptime":' + str(int(uptime)) + '}',
            status_code=503,
            media_type="application/json"
        )
    
    return {
        "status": "ready",
        "uptime_seconds": int(uptime),
        "timestamp": datetime.utcnow().isoformat(),
        "environment_check": {
            "database_url": has_db_url,
            "openai_api_key": has_openai_key
        }
    }

@router.get("/health")
async def basic_health_check():
    """Basic health check without heavy dependencies."""
    uptime = time.time() - START_TIME
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(uptime),
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT"),
        "port": os.getenv("PORT", "8000")
    }