"""Health check and monitoring endpoints."""

from fastapi import APIRouter
from ...config.database import db_config

router = APIRouter()


@router.get("/health")
def health_check():
    """Enhanced health check with database performance metrics."""
    try:
        # Import here to avoid circular imports during startup
        from ...agents.core.database_service import get_db_service
        db_service = get_db_service()
        health_data = db_service.health_check()
        return health_data
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "fallback": db_config.health_check()
        }