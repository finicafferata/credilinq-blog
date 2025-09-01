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
    
    print("🔍 [RAILWAY DEBUG] Starting API routes loading...")
    logger.info("🔍 [RAILWAY DEBUG] Starting API routes loading...")
    
    if api_routes_loaded:
        print("ℹ️ [RAILWAY DEBUG] API routes already loaded - skipping")
        logger.info("ℹ️ [RAILWAY DEBUG] API routes already loaded - skipping")
        return
        
    # Initialize route modules as None
    health = None
    blogs = None
    campaigns = None
    analytics = None
    documents = None
    
    try:
        print("🔍 [RAILWAY DEBUG] Attempting to import route modules...")
        logger.info("🔍 [RAILWAY DEBUG] Attempting to import route modules...")
        
        # Import routes with individual error handling
        try:
            from .api.routes import health
            print("✅ [RAILWAY DEBUG] Health routes imported successfully")
            logger.info("✅ [RAILWAY DEBUG] Health routes imported successfully")
        except Exception as e:
            print(f"❌ [RAILWAY DEBUG] Health routes import failed: {e}")
            logger.error(f"❌ [RAILWAY DEBUG] Health routes import failed: {e}")
            health = None
            
        try:
            from .api.routes import blogs
            print("✅ [RAILWAY DEBUG] Blog routes imported successfully")
            logger.info("✅ [RAILWAY DEBUG] Blog routes imported successfully")
        except Exception as e:
            print(f"❌ [RAILWAY DEBUG] Blog routes import failed: {e}")
            logger.error(f"❌ [RAILWAY DEBUG] Blog routes import failed: {e}")
            blogs = None
            
        try:
            from .api.routes import campaigns
            print("✅ [RAILWAY DEBUG] Campaign routes imported successfully")
            logger.info("✅ [RAILWAY DEBUG] Campaign routes imported successfully")
        except Exception as e:
            print(f"❌ [RAILWAY DEBUG] Campaign routes import failed: {e}")
            logger.error(f"❌ [RAILWAY DEBUG] Campaign routes import failed: {e}")
            campaigns = None
            
        try:
            from .api.routes import analytics
            print("✅ [RAILWAY DEBUG] Analytics routes imported successfully")
            logger.info("✅ [RAILWAY DEBUG] Analytics routes imported successfully")
        except Exception as e:
            print(f"❌ [RAILWAY DEBUG] Analytics routes import failed: {e}")
            logger.error(f"❌ [RAILWAY DEBUG] Analytics routes import failed: {e}")
            analytics = None
            
        try:
            from .api.routes import documents
            print("✅ [RAILWAY DEBUG] Document routes imported successfully")
            logger.info("✅ [RAILWAY DEBUG] Document routes imported successfully")
        except Exception as e:
            print(f"❌ [RAILWAY DEBUG] Document routes import failed: {e}")
            logger.error(f"❌ [RAILWAY DEBUG] Document routes import failed: {e}")
            documents = None
        
        print("🔍 [RAILWAY DEBUG] Starting to include routers in FastAPI app...")
        logger.info("🔍 [RAILWAY DEBUG] Starting to include routers in FastAPI app...")
        
        # Include routes only if successfully imported
        if health is not None:
            try:
                app.include_router(health.router, tags=["health"])
                print("✅ [RAILWAY DEBUG] Health router included")
                logger.info("✅ [RAILWAY DEBUG] Health router included")
            except Exception as e:
                print(f"❌ [RAILWAY DEBUG] Health router include failed: {e}")
                logger.error(f"❌ [RAILWAY DEBUG] Health router include failed: {e}")
        else:
            print("⏭️ [RAILWAY DEBUG] Skipping health router (import failed)")
            logger.info("⏭️ [RAILWAY DEBUG] Skipping health router (import failed)")
            
        if blogs is not None:
            try:
                app.include_router(blogs.router, prefix="/api/v2", tags=["blogs"])
                print("✅ [RAILWAY DEBUG] Blog router included with prefix /api/v2")
                logger.info("✅ [RAILWAY DEBUG] Blog router included with prefix /api/v2")
            except Exception as e:
                print(f"❌ [RAILWAY DEBUG] Blog router include failed: {e}")
                logger.error(f"❌ [RAILWAY DEBUG] Blog router include failed: {e}")
        else:
            print("⏭️ [RAILWAY DEBUG] Skipping blog router (import failed)")
            logger.info("⏭️ [RAILWAY DEBUG] Skipping blog router (import failed)")
            
        if campaigns is not None:
            try:
                app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns"])
                print("✅ [RAILWAY DEBUG] Campaign router included with prefix /api/v2")
                logger.info("✅ [RAILWAY DEBUG] Campaign router included with prefix /api/v2")
            except Exception as e:
                print(f"❌ [RAILWAY DEBUG] Campaign router include failed: {e}")
                logger.error(f"❌ [RAILWAY DEBUG] Campaign router include failed: {e}")
        else:
            print("⏭️ [RAILWAY DEBUG] Skipping campaign router (import failed)")
            logger.info("⏭️ [RAILWAY DEBUG] Skipping campaign router (import failed)")
            
        if analytics is not None:
            try:
                app.include_router(analytics.router, prefix="/api/v2", tags=["analytics"])
                print("✅ [RAILWAY DEBUG] Analytics router included with prefix /api/v2")
                logger.info("✅ [RAILWAY DEBUG] Analytics router included with prefix /api/v2")
            except Exception as e:
                print(f"❌ [RAILWAY DEBUG] Analytics router include failed: {e}")
                logger.error(f"❌ [RAILWAY DEBUG] Analytics router include failed: {e}")
        else:
            print("⏭️ [RAILWAY DEBUG] Skipping analytics router (import failed)")
            logger.info("⏭️ [RAILWAY DEBUG] Skipping analytics router (import failed)")
            
        if documents is not None:
            try:
                app.include_router(documents.router, prefix="/api/v2", tags=["documents"])
                print("✅ [RAILWAY DEBUG] Document router included with prefix /api/v2")
                logger.info("✅ [RAILWAY DEBUG] Document router included with prefix /api/v2")
            except Exception as e:
                print(f"❌ [RAILWAY DEBUG] Document router include failed: {e}")
                logger.error(f"❌ [RAILWAY DEBUG] Document router include failed: {e}")
        else:
            print("⏭️ [RAILWAY DEBUG] Skipping document router (import failed)")
            logger.info("⏭️ [RAILWAY DEBUG] Skipping document router (import failed)")
        
        # Check final route count
        total_routes = len(app.routes)
        api_routes = [route for route in app.routes if hasattr(route, 'path') and '/api/v2' in route.path]
        
        print(f"🔍 [RAILWAY DEBUG] Final route count: {total_routes} total, {len(api_routes)} API routes")
        logger.info(f"🔍 [RAILWAY DEBUG] Final route count: {total_routes} total, {len(api_routes)} API routes")
        
        # List first 10 API routes for verification
        for i, route in enumerate(api_routes[:10]):
            route_path = route.path if hasattr(route, 'path') else 'unknown'
            print(f"🔍 [RAILWAY DEBUG] API Route {i+1}: {route_path}")
            logger.info(f"🔍 [RAILWAY DEBUG] API Route {i+1}: {route_path}")
        
        api_routes_loaded = True
        print("✅ [RAILWAY DEBUG] API routes loading completed successfully")
        logger.info("✅ [RAILWAY DEBUG] API routes loading completed successfully")
        
    except Exception as e:
        print(f"❌ [RAILWAY DEBUG] CRITICAL: API routes loading failed: {e}")
        logger.error(f"❌ [RAILWAY DEBUG] CRITICAL: API routes loading failed: {e}")
        import traceback
        tb = traceback.format_exc()
        print(f"❌ [RAILWAY DEBUG] Full traceback: {tb}")
        logger.error(f"❌ [RAILWAY DEBUG] Full traceback: {tb}")
        # Don't fail app startup
        pass

# Load routes immediately
print("🚀 [RAILWAY DEBUG] Calling load_api_routes_sync()...")
logger.info("🚀 [RAILWAY DEBUG] Calling load_api_routes_sync()...")
load_api_routes_sync()
print("🏁 [RAILWAY DEBUG] load_api_routes_sync() completed")
logger.info("🏁 [RAILWAY DEBUG] load_api_routes_sync() completed")

# Simple root endpoint
@app.get("/")
async def railway_root():
    """Railway root endpoint with configuration info."""
    print("🔍 [RAILWAY DEBUG] Root endpoint called")
    logger.info("🔍 [RAILWAY DEBUG] Root endpoint called")
    
    await lazy_load_config()
    
    # Count current routes for debugging
    total_routes = len(app.routes)
    api_routes = [route for route in app.routes if hasattr(route, 'path') and '/api/v2' in route.path]
    
    print(f"🔍 [RAILWAY DEBUG] Current route status: {total_routes} total, {len(api_routes)} API routes")
    logger.info(f"🔍 [RAILWAY DEBUG] Current route status: {total_routes} total, {len(api_routes)} API routes")
    
    response_data = {
        "message": "CrediLinq AI Platform (Railway Optimized)",
        "version": "2.0.0-railway",
        "status": "operational",
        "environment": getattr(settings, 'environment', 'unknown') if settings else 'config-loading',
        "optimization": "railway-progressive",
        "debug_info": {
            "total_routes": total_routes,
            "api_routes_count": len(api_routes),
            "api_routes_loaded_flag": api_routes_loaded
        },
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
    
    print(f"🔍 [RAILWAY DEBUG] Root response: {response_data}")
    logger.info(f"🔍 [RAILWAY DEBUG] Root response: {response_data}")
    
    return response_data

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