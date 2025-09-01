"""
Railway-optimized FastAPI application for CrediLinq AI Content Platform.
Progressively enhanced version with lazy loading for Railway deployment.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import asyncio

# Configure minimal logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
db_config = None
settings = None
enhanced_logger = None
agent_system_loaded = False
api_routes_loaded = False

async def lazy_load_config():
    """Lazy load configuration to avoid startup delays."""
    global db_config, settings, enhanced_logger
    if settings is None:
        try:
            from .config import settings as _settings, db_config as _db_config
            from .core.enhanced_logging import enhanced_logger as _logger
            settings = _settings
            db_config = _db_config
            enhanced_logger = _logger
            logger.info("✅ Configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            return False
    return True

@asynccontextmanager
async def railway_lifespan(app: FastAPI):
    """Railway-optimized lifespan with lazy loading and health checks."""
    # Startup
    logger.info("🚂 Starting Railway-optimized CrediLinq Platform")
    
    # Try to load configuration (non-blocking)
    config_loaded = await lazy_load_config()
    
    if config_loaded and db_config:
        # Optional database health check (don't fail startup)
        try:
            db_health = db_config.health_check()
            if db_health.get("status") == "healthy":
                logger.info("✅ Database connection verified")
            else:
                logger.warning(f"⚠️ Database health: {db_health}")
        except Exception as e:
            logger.warning(f"⚠️ Database check failed: {e}")
    
    logger.info("🚀 Railway deployment started successfully")
    
    yield
    
    # Shutdown
    logger.info("🔄 Railway deployment shutting down")
    if db_config:
        try:
            db_config.close()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Database shutdown warning: {e}")
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

# Load API routes immediately during app initialization
def load_api_routes_sync():
    """Load API routes synchronously during app initialization."""
    global api_routes_loaded
    
    if api_routes_loaded:
        return
        
    try:
        # Import routes
        from .api.routes import health, blogs, campaigns, analytics, documents
        
        # Include routes immediately
        app.include_router(health.router, tags=["health"])
        app.include_router(blogs.router, prefix="/api/v2", tags=["blogs"])
        app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns"]) 
        app.include_router(analytics.router, prefix="/api/v2", tags=["analytics"])
        app.include_router(documents.router, prefix="/api/v2", tags=["documents"])
        
        api_routes_loaded = True
        logger.info("✅ API routes loaded during app initialization")
        
    except Exception as e:
        logger.error(f"❌ API routes loading failed during app init: {e}")
        # Don't fail app startup
        pass

# Load routes immediately
load_api_routes_sync()

# Simple root endpoint
@app.get("/")
async def railway_root():
    """Railway root endpoint with configuration info."""
    await lazy_load_config()
    return {
        "message": "CrediLinq AI Platform (Railway Optimized)",
        "version": "2.0.0-railway",
        "status": "operational",
        "environment": getattr(settings, 'environment', 'unknown') if settings else 'config-loading',
        "optimization": "railway-progressive",
        "features": {
            "database": db_config is not None,
            "api_routes": api_routes_loaded,
            "agent_system": agent_system_loaded,
            "agent_system_enabled": (
                os.environ.get('RAILWAY_FULL', '').lower() == 'true' or
                os.environ.get('ENABLE_AGENT_LOADING', '').lower() == 'true'
            )
        }
    }

# Simple ping endpoint for Railway health checks
@app.get("/ping")
async def ping():
    """Simple ping endpoint for Railway."""
    return {"status": "ok", "service": "credilinq-railway"}

# Railway health endpoint
@app.get("/health/railway")
async def health_railway():
    """Railway-specific health check with component status."""
    await lazy_load_config()
    
    health_status = {
        "status": "healthy",
        "service": "credilinq-railway",
        "components": {
            "fastapi": "healthy",
            "configuration": "healthy" if settings else "degraded",
        }
    }
    
    # Check database if available
    if db_config:
        try:
            db_health = db_config.health_check()
            health_status["components"]["database"] = db_health.get("status", "unknown")
        except Exception:
            health_status["components"]["database"] = "degraded"
    else:
        health_status["components"]["database"] = "not_loaded"
    
    # Overall status based on components
    if any(status == "degraded" for status in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/health/live")
async def health_live():
    """Railway liveness probe."""
    return {"status": "alive"}

# Add API routes dynamically
async def add_api_routes():
    """Add API routes with lazy loading."""
    global api_routes_loaded
    
    if api_routes_loaded:
        logger.info("ℹ️ API routes already loaded")
        return True
        
    try:
        await lazy_load_config()
        
        # Import routes only when needed
        from .api.routes import health, blogs, campaigns, analytics, documents
        
        # Include routes
        app.include_router(health.router, tags=["health"])
        app.include_router(blogs.router, prefix="/api/v2", tags=["blogs"])
        app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns"]) 
        app.include_router(analytics.router, prefix="/api/v2", tags=["analytics"])
        app.include_router(documents.router, prefix="/api/v2", tags=["documents"])
        
        api_routes_loaded = True
        logger.info("✅ API routes loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ API routes loading failed: {e}")
        return False

# Initialize API routes endpoint (for testing)
@app.get("/api/initialize")
async def initialize_api():
    """Initialize API routes (for development/testing)."""
    success = await add_api_routes()
    return {"success": success, "message": "API routes initialization attempted"}

# Agent system initialization
async def load_agent_system():
    """Load agent system with conditional enabling based on environment."""
    global agent_system_loaded
    
    if agent_system_loaded:
        return True
        
    try:
        # Only load agents if RAILWAY_FULL is set or environment allows
        enable_agents = (
            os.environ.get('RAILWAY_FULL', '').lower() == 'true' or
            os.environ.get('ENABLE_AGENT_LOADING', '').lower() == 'true' or
            getattr(settings, 'environment', '') != 'production'
        )
        
        if not enable_agents:
            logger.info("🚂 Agent system disabled for Railway deployment")
            return False
            
        logger.info("🤖 Loading agent system...")
        
        # Load agents with timeout to prevent hanging
        await asyncio.wait_for(
            asyncio.create_task(_load_agents_async()),
            timeout=30.0  # 30 second timeout
        )
        
        agent_system_loaded = True
        logger.info("✅ Agent system loaded successfully")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ Agent system loading timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Agent system loading failed: {e}")
        return False

async def _load_agents_async():
    """Internal async function to load agents."""
    # Import agent workflows (this triggers agent registration)
    from .agents.workflow.blog_workflow import BlogWorkflow
    from .agents.workflow.content_generation_workflow import ContentGenerationWorkflow
    
    # Initialize minimal agent workflows
    blog_workflow = BlogWorkflow()
    content_workflow = ContentGenerationWorkflow()
    
    logger.info("🤖 Core agent workflows initialized")

# Agent system endpoints
@app.get("/agents/initialize")
async def initialize_agents():
    """Initialize the agent system (for development/testing)."""
    success = await load_agent_system()
    return {
        "success": success, 
        "loaded": agent_system_loaded,
        "message": "Agent system initialization attempted"
    }

@app.get("/agents/status")
async def agent_status():
    """Get agent system status."""
    return {
        "loaded": agent_system_loaded,
        "enabled": os.environ.get('RAILWAY_FULL', '').lower() == 'true',
        "environment": getattr(settings, 'environment', 'unknown') if settings else 'config-loading'
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.main_railway:app",
        host="0.0.0.0",
        port=port,
        access_log=True
    )