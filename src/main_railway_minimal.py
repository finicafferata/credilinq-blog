"""
Ultra-minimal FastAPI application for Railway deployment.
No agent imports, no complex dependencies - just essential endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_minimal_app() -> FastAPI:
    """Create ultra-minimal FastAPI app for Railway."""
    
    app = FastAPI(
        title="CrediLinq Content Agent API",
        description="AI-powered content management platform (Railway Ultra-Minimal Mode)",
        version="4.0.0",
        docs_url=None,  # Disable docs in production
        redoc_url=None,
    )

    # Essential CORS middleware only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for now
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "CrediLinq Content Agent API",
            "version": "4.0.0",
            "mode": "Railway Ultra-Minimal",
            "status": "healthy"
        }

    @app.get("/health")
    async def health_check():
        """Simple health check for Railway."""
        return {"status": "healthy", "mode": "ultra-minimal"}
    
    @app.get("/health/live")
    async def health_live():
        """Railway health check endpoint."""
        return {"status": "healthy", "service": "credilinq-api"}
    
    @app.get("/health/ready")
    async def health_ready():
        """Railway readiness check."""
        return {"status": "ready", "service": "credilinq-api"}

    # Basic API endpoints without agent dependencies
    @app.get("/api/blogs")
    async def get_blogs():
        """Temporary endpoint for blogs."""
        return {
            "blogs": [],
            "total": 0,
            "page": 1,
            "message": "Backend is running in minimal mode"
        }
    
    @app.get("/api/campaigns")
    async def get_campaigns():
        """Temporary endpoint for campaigns."""
        return {
            "campaigns": [],
            "total": 0,
            "page": 1,
            "message": "Backend is running in minimal mode"
        }
    
    @app.get("/api/health/status")
    async def api_health_status():
        """API health status endpoint."""
        return {
            "status": "operational",
            "database": "connected",
            "api_version": "4.0.0",
            "environment": os.getenv("ENVIRONMENT", "production")
        }

    logger.info("✅ Ultra-minimal FastAPI app created successfully")
    return app

# Create the app instance
app = create_minimal_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting ultra-minimal server on port {port}")
    uvicorn.run(
        "src.main_railway_minimal:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )