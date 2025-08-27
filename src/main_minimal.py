"""
Minimal FastAPI application for Railway deployment.
Optimized for Railway's resource constraints with essential features only.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from .config import settings, db_config, secure_db
from .api.routes import blogs, campaigns, health
from .core.enhanced_exceptions import ErrorHandlingMiddleware
from .core.security_headers import SecurityHeadersMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events - minimal version."""
    logger.info("🚀 Starting CrediLinq AI Platform (Minimal Mode)")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Railway Mode: {'Yes' if os.getenv('RAILWAY_ENVIRONMENT') else 'No'}")
    
    try:
        # Test database connection
        logger.info("📊 Testing database connection...")
        await db_config.database.connect()
        logger.info("✅ Database connected successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔄 Shutting down...")
        try:
            if hasattr(db_config.database, 'disconnect'):
                await db_config.database.disconnect()
                logger.info("✅ Database disconnected")
        except Exception as e:
            logger.error(f"⚠️ Cleanup error: {e}")

def create_minimal_app() -> FastAPI:
    """Create minimal FastAPI app optimized for Railway."""
    
    app = FastAPI(
        title="CrediLinq Content Agent API",
        description="AI-powered content management platform (Railway Minimal Mode)",
        version="4.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    # Essential middleware only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # Essential routes only
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(blogs.router, prefix="/api/blogs", tags=["Blogs"])
    app.include_router(campaigns.router, prefix="/api/campaigns", tags=["Campaigns"])
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "CrediLinq Content Agent API",
            "version": "4.0.0",
            "mode": "Railway Minimal",
            "status": "healthy"
        }

    @app.get("/health")
    async def health_check():
        """Simple health check for Railway."""
        return {"status": "healthy", "mode": "minimal"}

    logger.info("✅ Minimal FastAPI app created successfully")
    return app

# Create the app instance
app = create_minimal_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main_minimal:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )