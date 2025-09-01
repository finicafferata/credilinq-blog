"""
Railway-optimized FastAPI application for CrediLinq AI Content Platform.
Simplified version for Railway deployment with reduced memory footprint.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from .config import settings, db_config
from .api.routes import health
from .api.routes import blogs, campaigns, analytics, documents
from .core.enhanced_logging import enhanced_logger
from .core.enhanced_exceptions import ErrorHandlingMiddleware

# Configure minimal logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def railway_lifespan(app: FastAPI):
    """Railway-optimized lifespan manager with minimal initialization."""
    # Startup
    enhanced_logger.info("🚂 Starting Railway-optimized CrediLinq Platform")
    enhanced_logger.info(f"Environment: {settings.environment}")
    
    # Essential database health check only
    try:
        db_health = db_config.health_check()
        if db_health["status"] == "healthy":
            enhanced_logger.info("✅ Database connection verified")
        else:
            enhanced_logger.warning(f"⚠️ Database health check: {db_health}")
    except Exception as e:
        enhanced_logger.error(f"❌ Database connection failed: {e}")
        # Don't fail startup - let health endpoint handle this
    
    enhanced_logger.info("🚀 Railway deployment started successfully")
    
    yield
    
    # Shutdown
    enhanced_logger.info("🔄 Railway deployment shutting down")
    try:
        db_config.close()
        enhanced_logger.info("✅ Database connections closed")
    except Exception as e:
        enhanced_logger.warning(f"⚠️ Database shutdown warning: {e}")
    
    enhanced_logger.info("✅ Railway deployment shutdown completed")

# Create Railway-optimized FastAPI application
app = FastAPI(
    title="CrediLinq AI Content Platform API",
    description="Railway-optimized AI-powered content management platform",
    version="2.0.0-railway",
    debug=False,  # Force debug off in Railway
    lifespan=railway_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add essential middleware only
app.add_middleware(ErrorHandlingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include essential routes only
app.include_router(health.router, tags=["health"])
app.include_router(blogs.router, prefix="/api/v2", tags=["blogs"])
app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns"])
app.include_router(analytics.router, prefix="/api/v2", tags=["analytics"])
app.include_router(documents.router, prefix="/api/v2", tags=["documents"])

# Simple root endpoint
@app.get("/")
async def railway_root():
    """Railway-optimized root endpoint."""
    return {
        "message": "CrediLinq AI Platform (Railway Optimized)",
        "version": "2.0.0-railway",
        "status": "operational",
        "environment": settings.environment,
        "optimization": "railway-minimal"
    }

# Simple ping endpoint for Railway health checks
@app.get("/ping")
async def ping():
    """Simple ping endpoint for Railway."""
    return {"status": "ok", "service": "credilinq-railway"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.main_railway:app",
        host="0.0.0.0",
        port=port,
        access_log=True
    )