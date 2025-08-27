"""
Stable Railway FastAPI application with database and API endpoints.
No agents, but includes real functionality for blogs and campaigns.
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
    """Application lifespan events."""
    global db_pool
    
    logger.info("🚀 Starting CrediLinq API (Railway Stable Mode)")
    
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
                    timeout=30
                )
                logger.info("✅ Database connected successfully")
                
                # Test the connection
                async with db_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    logger.info(f"✅ Database test query successful: {result}")
                    
            except Exception as e:
                logger.error(f"❌ Database connection failed: {e}")
                logger.warning("⚠️ Running without database - API will return empty data")
                db_pool = None
        else:
            logger.warning("⚠️ DATABASE_URL not set - running without database")
        
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

def create_stable_app() -> FastAPI:
    """Create stable FastAPI app for Railway."""
    
    app = FastAPI(
        title="CrediLinq Content Agent API",
        description="AI-powered content management platform (Railway Stable Mode)",
        version="4.0.0",
        lifespan=lifespan,
        docs_url="/docs",  # Enable docs
        redoc_url="/redoc",
    )

    # CORS middleware - allow all origins for now to fix the issue
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins temporarily
        allow_credentials=False,  # Must be False when using wildcard
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    logger.info("CORS configured: allowing all origins (*)")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "CrediLinq Content Agent API",
            "version": "4.0.0",
            "mode": "Railway Stable",
            "status": "healthy",
            "database": "connected" if db_pool else "not connected"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "mode": "stable",
            "database": "connected" if db_pool else "not connected"
        }
    
    @app.get("/health/live")
    async def health_live():
        """Railway health check endpoint."""
        return {"status": "healthy", "service": "credilinq-api"}
    
    @app.get("/health/ready")
    async def health_ready():
        """Railway readiness check."""
        return {
            "status": "ready" if db_pool else "not ready",
            "service": "credilinq-api",
            "database": "connected" if db_pool else "not connected"
        }

    @app.get("/api/v2/blogs")
    async def get_blogs(
        page: int = 1,
        limit: int = 10,
        status: Optional[str] = None
    ):
        """Get blogs from database."""
        blogs = []
        total = 0
        
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    # Check if blog_posts table exists
                    table_exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'blog_posts'
                        )
                    """)
                    
                    if table_exists:
                        # Get total count
                        if status:
                            total = await conn.fetchval(
                                'SELECT COUNT(*) FROM "blog_posts" WHERE status = $1',
                                status
                            )
                        else:
                            total = await conn.fetchval('SELECT COUNT(*) FROM "blog_posts"')
                        
                        # Get blogs with pagination
                        offset = (page - 1) * limit
                        query = """
                            SELECT id, title, content_markdown as content, status, 
                                   created_at, updated_at
                            FROM "blog_posts"
                        """
                        if status:
                            query += " WHERE status = $3"
                            query += " ORDER BY created_at DESC LIMIT $1 OFFSET $2"
                            rows = await conn.fetch(query, limit, offset, status)
                        else:
                            query += " ORDER BY created_at DESC LIMIT $1 OFFSET $2"
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
                    else:
                        logger.warning("BlogPost table does not exist")
                        
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
    async def get_campaigns(
        page: int = 1,
        limit: int = 10
    ):
        """Get campaigns from database."""
        campaigns = []
        total = 0
        
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    # Check if campaigns table exists
                    table_exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'campaigns'
                        )
                    """)
                    
                    if table_exists:
                        # Get total count
                        total = await conn.fetchval('SELECT COUNT(*) FROM "campaigns"')
                        
                        # Get campaigns with pagination
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
                    else:
                        logger.warning("Campaign table does not exist")
                        
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
        # Return mock/default settings for stable mode
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
        # In stable mode, just return success without persisting
        return {"message": "Settings updated (not persisted in stable mode)", "status": "success"}

    @app.post("/api/v2/blogs")
    async def create_blog(title: str, content: str = ""):
        """Create a new blog post."""
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        try:
            async with db_pool.acquire() as conn:
                # Check if table exists first
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'blog_posts'
                    )
                """)
                
                if not table_exists:
                    raise HTTPException(status_code=503, detail="blog_posts table does not exist")
                
                # Create blog post
                result = await conn.fetchrow("""
                    INSERT INTO "blog_posts" (title, content_markdown, status, created_at, updated_at)
                    VALUES ($1, $2, 'draft', NOW(), NOW())
                    RETURNING id, title, content_markdown as content, status, created_at, updated_at
                """, title, content)
                
                return {
                    "id": result["id"],
                    "title": result["title"],
                    "content": result["content"],
                    "status": result["status"],
                    "created_at": result["created_at"].isoformat(),
                    "updated_at": result["updated_at"].isoformat(),
                }
                
        except Exception as e:
            logger.error(f"Error creating blog: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    logger.info("✅ Stable FastAPI app created successfully")
    return app

# Create the app instance
app = create_stable_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting stable server on port {port}")
    uvicorn.run(
        "src.main_railway_stable:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )