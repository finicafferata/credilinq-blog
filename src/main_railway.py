"""
Railway-optimized FastAPI application for CrediLinq AI Content Platform.
Simplified startup with minimal dependencies to avoid crashes.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import simplified health check
from .api.routes.health_simple import router as health_router

@asynccontextmanager
async def railway_lifespan(app: FastAPI):
    """Simplified lifespan manager for Railway deployment."""
    # Startup
    logger.info("🚀 Starting CrediLinq AI Platform on Railway")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'unknown')}")
    
    # Basic environment validation
    required_vars = ['DATABASE_URL', 'OPENAI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing)}")
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    else:
        logger.info("✅ Required environment variables present")
    
    # Import and validate config
    try:
        from .config import settings
        logger.info(f"✅ Configuration loaded - Environment: {settings.environment}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        raise
    
    logger.info("🎉 Railway startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Railway shutdown initiated")

# Create minimal FastAPI application
app = FastAPI(
    title="CrediLinq AI Content Platform",
    description="AI-powered content management platform - Railway deployment",
    version="2.0.0",
    lifespan=railway_lifespan,
    docs_url="/docs",  # Enable docs
    redoc_url="/redoc",  # Enable redoc
)

# Add CORS middleware with Railway-friendly settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for Railway
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Include simplified health check routes
app.include_router(health_router, tags=["health"])

@app.get("/")
async def railway_root():
    """Root endpoint optimized for Railway."""
    return {
        "message": "CrediLinq AI Content Platform",
        "status": "operational",
        "platform": "Railway",
        "version": "2.0.0",
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "unknown"),
        "health_check": "/health/live",
        "api_docs": "/docs"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for Railway debugging."""
    return {
        "status": "ok",
        "message": "Railway deployment test successful",
        "environment_vars": {
            "PORT": os.getenv("PORT"),
            "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT"),
            "DATABASE_URL": "***MASKED***" if os.getenv("DATABASE_URL") else None,
            "OPENAI_API_KEY": "***MASKED***" if os.getenv("OPENAI_API_KEY") else None,
        }
    }

# Conditional import of full application features
def load_full_application():
    """Load full application features if environment is stable."""
    try:
        logger.info("🔄 Loading full application features...")
        
        # Import and register routes
        from .api.routes import blogs, campaigns, analytics
        
        app.include_router(blogs.router, prefix="/api/v2", tags=["blogs"])
        app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns"])
        app.include_router(analytics.router, prefix="/api/v2", tags=["analytics"])
        
        logger.info("✅ Full application features loaded")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load full features: {e}")
        return False

# Try to load full features after basic startup
@app.on_event("startup")
async def load_features_after_startup():
    """Load additional features after basic startup is complete."""
    # Add delay to let Railway health checks pass first
    import asyncio
    await asyncio.sleep(5)
    
    # Only load full features if we're not in a critical state
    if os.getenv("MINIMAL_MODE", "").lower() != "true":
        load_full_application()

# Export for railway startup script
handler = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.main_railway:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Never reload in Railway
        access_log=True,
        log_level="info"
    )