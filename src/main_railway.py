"""
Railway-optimized FastAPI application for CrediLinq AI Content Platform.
Ultra-minimal version for Railway deployment to isolate startup issues.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Configure minimal logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def railway_lifespan(app: FastAPI):
    """Ultra-minimal lifespan manager for Railway debugging."""
    # Startup
    logger.info("🚂 Starting minimal Railway CrediLinq Platform")
    logger.info("🚀 Railway deployment started successfully")
    
    yield
    
    # Shutdown
    logger.info("🔄 Railway deployment shutting down")
    logger.info("✅ Railway deployment shutdown completed")

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

# Add minimal CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Simple root endpoint
@app.get("/")
async def railway_root():
    """Ultra-minimal Railway root endpoint."""
    return {
        "message": "CrediLinq AI Platform (Railway Minimal)",
        "version": "2.0.0-railway-debug",
        "status": "operational",
        "optimization": "railway-ultra-minimal"
    }

# Simple ping endpoint for Railway health checks
@app.get("/ping")
async def ping():
    """Simple ping endpoint for Railway."""
    return {"status": "ok", "service": "credilinq-railway"}

# Railway health endpoint
@app.get("/health/railway")
async def health_railway():
    """Railway-specific health check."""
    return {"status": "healthy", "service": "credilinq-minimal"}

@app.get("/health/live")
async def health_live():
    """Railway liveness probe."""
    return {"status": "alive"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.main_railway:app",
        host="0.0.0.0",
        port=port,
        access_log=True
    )