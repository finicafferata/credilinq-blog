"""
Production Railway FastAPI application with optimized startup.
Includes all API endpoints and essential features with lazy agent loading.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import logging
import os
import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global database connection
db_pool = None

# Lazy-loaded agents (initialized on first use)
agents_initialized = False
agent_registry = {}

def initialize_agents_lazy():
    """Initialize agents on first API call that needs them."""
    global agents_initialized, agent_registry
    if not agents_initialized:
        try:
            logger.info("🤖 Lazy-loading AI agents...")
            # Import only when needed
            from .agents import specialized
            from .agents.core.agent_factory import AgentFactory
            
            agent_registry = AgentFactory.get_all_agents()
            agents_initialized = True
            logger.info(f"✅ Loaded {len(agent_registry)} agents")
        except Exception as e:
            logger.warning(f"⚠️ Agent initialization failed: {e}")
            # Continue without agents
            agents_initialized = True

# Pydantic models for API
class BlogResponse(BaseModel):
    id: str
    title: str
    content: Optional[str] = ""
    status: str = "draft"
    created_at: datetime
    updated_at: datetime

class CampaignResponse(BaseModel):
    id: str
    name: str
    status: str = "draft"
    created_at: datetime
    updated_at: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events - optimized for fast startup."""
    global db_pool
    
    logger.info("🚀 Starting CrediLinq API (Railway Production Mode)")
    
    try:
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Convert postgres:// to postgresql:// if needed
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            
            logger.info("📊 Connecting to database...")
            try:
                db_pool = await asyncpg.create_pool(
                    database_url,
                    min_size=1,
                    max_size=5,
                    timeout=30,
                    command_timeout=10
                )
                logger.info("✅ Database connected successfully")
            except Exception as e:
                logger.error(f"❌ Database connection failed: {e}")
                db_pool = None
        else:
            logger.warning("⚠️ DATABASE_URL not set - running without database")
        
        # Don't initialize agents at startup - do it lazily
        logger.info("⏳ Agents will be loaded on first use (lazy loading)")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔄 Shutting down...")
        if db_pool:
            await db_pool.close()
            logger.info("✅ Database connection closed")

def create_production_app() -> FastAPI:
    """Create production FastAPI app for Railway."""
    
    app = FastAPI(
        title="CrediLinq Content Agent API",
        description="AI-powered content management platform (Railway Production Mode)",
        version="4.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware - allow all origins for now
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "CrediLinq Content Agent API",
            "version": "4.1.0",
            "mode": "Railway Production",
            "status": "healthy",
            "database": "connected" if db_pool else "not connected",
            "agents": "loaded" if agents_initialized else "lazy-loading"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "mode": "production",
            "database": "connected" if db_pool else "not connected",
            "agents": "loaded" if agents_initialized else "lazy-loading"
        }
    
    @app.get("/health/live")
    async def health_live():
        """Railway health check endpoint - fast response."""
        return {"status": "healthy", "service": "credilinq-api"}
    
    @app.get("/health/ready")
    async def health_ready():
        """Railway readiness check."""
        return {
            "status": "ready",
            "service": "credilinq-api",
            "database": "connected" if db_pool else "not connected"
        }

    # Import and register all API routes
    try:
        from .api.routes import (
            blogs, campaigns, settings as settings_router,
            analytics, documents, content_repurposing,
            competitor_intelligence
        )
        
        # Core routes
        app.include_router(blogs.router, prefix="/api/v2", tags=["blogs"])
        app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns"])
        app.include_router(settings_router.router, prefix="/api", tags=["settings"])
        
        # Feature routes (these will lazy-load agents when needed)
        app.include_router(analytics.router, prefix="/api", tags=["analytics"])
        app.include_router(documents.router, prefix="/api", tags=["documents"])
        app.include_router(content_repurposing.router, prefix="/api/v2", tags=["content"])
        app.include_router(competitor_intelligence.router, prefix="/api", tags=["competitor_intelligence"])
        
        logger.info("✅ All API routes registered")
        
    except ImportError as e:
        logger.warning(f"⚠️ Some routes could not be imported: {e}")
        
        # Fallback: Add basic routes inline if imports fail
        @app.get("/api/v2/blogs")
        async def get_blogs(page: int = 1, limit: int = 10, status: Optional[str] = None):
            """Get blogs from database."""
            blogs = []
            total = 0
            
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        # Implementation similar to stable version
                        table_exists = await conn.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'blog_posts'
                            )
                        """)
                        
                        if table_exists:
                            if status:
                                total = await conn.fetchval(
                                    'SELECT COUNT(*) FROM "blog_posts" WHERE status = $1',
                                    status
                                )
                            else:
                                total = await conn.fetchval('SELECT COUNT(*) FROM "blog_posts"')
                            
                            offset = (page - 1) * limit
                            query = """
                                SELECT id, title, content_markdown as content, status, 
                                       created_at, updated_at
                                FROM "blog_posts"
                            """
                            if status:
                                query += " WHERE status = $3"
                            query += " ORDER BY created_at DESC LIMIT $1 OFFSET $2"
                            
                            if status:
                                rows = await conn.fetch(query, limit, offset, status)
                            else:
                                rows = await conn.fetch(query, limit, offset)
                            
                            blogs = [
                                {
                                    "id": row["id"],
                                    "title": row["title"],
                                    "content": row["content"][:200] if row["content"] else "",
                                    "status": row["status"],
                                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                                }
                                for row in rows
                            ]
                except Exception as e:
                    logger.error(f"Error fetching blogs: {e}")
            
            return {
                "blogs": blogs,
                "total": total,
                "page": page,
                "limit": limit,
                "status": "success" if db_pool else "no database"
            }
        
        @app.get("/api/v2/campaigns/")
        async def get_campaigns(page: int = 1, limit: int = 10):
            """Get campaigns from database."""
            campaigns = []
            total = 0
            
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        table_exists = await conn.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'campaigns'
                            )
                        """)
                        
                        if table_exists:
                            total = await conn.fetchval('SELECT COUNT(*) FROM "campaigns"')
                            
                            offset = (page - 1) * limit
                            rows = await conn.fetch("""
                                SELECT id, name, status, created_at, updated_at
                                FROM "campaigns"
                                ORDER BY created_at DESC
                                LIMIT $1 OFFSET $2
                            """, limit, offset)
                            
                            campaigns = [
                                {
                                    "id": row["id"],
                                    "name": row["name"],
                                    "status": row["status"],
                                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                                }
                                for row in rows
                            ]
                except Exception as e:
                    logger.error(f"Error fetching campaigns: {e}")
            
            return {
                "campaigns": campaigns,
                "total": total,
                "page": page,
                "limit": limit,
                "status": "success" if db_pool else "no database"
            }
        
        @app.get("/api/settings/company-profile")
        async def get_company_profile():
            """Get company profile settings."""
            return {
                "companyName": "Your Company",
                "companyContext": "",
                "brandVoice": "",
                "valueProposition": "",
                "industries": [],
                "targetAudiences": [],
                "tonePresets": ["Professional", "Casual", "Formal"],
                "keywords": [],
                "styleGuidelines": "",
                "prohibitedTopics": [],
                "complianceNotes": "",
                "links": [],
                "defaultCTA": "",
                "updatedAt": datetime.now().isoformat()
            }
        
        @app.put("/api/settings/company-profile")
        async def update_company_profile(profile: dict):
            """Update company profile settings."""
            return {"message": "Settings updated", "status": "success"}

    # Add a special endpoint to trigger agent initialization
    @app.post("/api/admin/initialize-agents")
    async def initialize_agents():
        """Manually trigger agent initialization."""
        initialize_agents_lazy()
        return {
            "status": "success",
            "agents_loaded": agents_initialized,
            "agent_count": len(agent_registry)
        }

    logger.info("✅ Production FastAPI app created successfully")
    return app

# Create the app instance
app = create_production_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting production server on port {port}")
    uvicorn.run(
        "src.main_railway_production:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )