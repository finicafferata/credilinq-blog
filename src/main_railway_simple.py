"""
Simple Railway FastAPI app with essential API routes.
Connects to real database but bypasses complex agent imports.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CrediLinq AI Platform (Railway Simple)",
    description="Simple Railway deployment with hardcoded API routes",
    version="2.0.0-railway-simple",
    debug=False
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Try to connect to database
db_config = None
try:
    from .config import db_config as _db_config
    db_config = _db_config
    print("✅ [RAILWAY DEBUG] Database connection loaded")
    logger.info("✅ [RAILWAY DEBUG] Database connection loaded")
except Exception as e:
    print(f"⚠️ [RAILWAY DEBUG] Database connection failed: {e}")
    logger.warning(f"⚠️ [RAILWAY DEBUG] Database connection failed: {e}")

print("🚀 [RAILWAY DEBUG] Creating simple Railway app with essential routes")
logger.info("🚀 [RAILWAY DEBUG] Creating simple Railway app with essential routes")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "CrediLinq AI Platform (Railway Simple)",
        "version": "2.0.0-railway-simple",
        "status": "operational",
        "routes_loaded": "hardcoded",
        "test_endpoints": [
            "/api/v2/campaigns/",
            "/api/v2/campaigns/orchestration/dashboard",
            "/api/v2/blogs/",
            "/ping"
        ]
    }

# Health endpoints
@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "credilinq-railway-simple"}

@app.get("/health/railway")
async def health_railway():
    return {"status": "healthy", "service": "credilinq-simple", "routes": "hardcoded"}

@app.get("/health/live") 
async def health_live():
    return {"status": "alive"}

# API routes with database connection
@app.get("/api/v2/campaigns/")
async def list_campaigns():
    """Get campaigns from database."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, name, status, created_at, updated_at
                    FROM campaigns 
                    ORDER BY created_at DESC
                """)
                campaigns = []
                for row in cur.fetchall():
                    campaigns.append({
                        "id": row[0],
                        "name": row[1],
                        "title": row[1],  # Use name as title for frontend compatibility
                        "status": row[2],
                        "created_at": row[3].isoformat() if row[3] else None,
                        "updated_at": row[4].isoformat() if row[4] else None,
                        "description": row[1] or "No description"  # Use name as fallback
                    })
                
                return {
                    "campaigns": campaigns,
                    "total": len(campaigns),
                    "service": "railway-simple"
                }
        except Exception as e:
            logger.error(f"Database error in campaigns: {e}")
            return {"campaigns": [], "total": 0, "error": str(e), "service": "railway-simple"}
    
    # Fallback if no database
    return {
        "campaigns": [],
        "total": 0,
        "message": "No database connection",
        "service": "railway-simple"
    }

@app.get("/api/v2/campaigns/orchestration/dashboard")
async def orchestration_dashboard():
    """Hardcoded orchestration dashboard endpoint."""
    return {
        "dashboard": {
            "campaigns": [],
            "tasks": [],
            "agents": [],
            "systemMetrics": {"status": "operational"},
            "lastUpdated": "2025-09-01T16:45:00Z"
        },
        "message": "Orchestration dashboard working (hardcoded)",
        "service": "railway-simple"
    }

@app.get("/api/v2/blogs/")
async def list_blogs():
    """Hardcoded blogs list endpoint."""
    return {
        "blogs": [],
        "total": 0,
        "message": "Blogs endpoint working (hardcoded)",
        "service": "railway-simple"
    }

@app.get("/api/v2/analytics/summary")
async def analytics_summary():
    """Analytics summary endpoint."""
    return {
        "summary": {
            "total_blogs": 0,
            "total_campaigns": 0,
            "status": "operational"
        },
        "message": "Analytics endpoint working",
        "service": "railway-simple"
    }

# Missing endpoints that frontend is requesting
@app.get("/api/documents")
async def list_documents():
    """Documents endpoint."""
    return {
        "documents": [],
        "total": 0,
        "message": "Documents endpoint working",
        "service": "railway-simple"
    }

@app.get("/api/settings/company-profile")
async def company_profile():
    """Company profile settings."""
    return {
        "profile": {
            "name": "CrediLinq",
            "industry": "Financial Services",
            "website": "https://credilinq.com"
        },
        "message": "Company profile endpoint working",
        "service": "railway-simple"
    }

# Additional common endpoints
@app.get("/api/v2/blogs/{blog_id}")
async def get_blog(blog_id: str):
    """Get specific blog post."""
    return {
        "id": blog_id,
        "title": f"Blog {blog_id}",
        "status": "draft",
        "message": "Blog detail endpoint working",
        "service": "railway-simple"
    }

@app.get("/api/v2/campaigns/{campaign_id}")  
async def get_campaign(campaign_id: str):
    """Get specific campaign."""
    return {
        "id": campaign_id,
        "title": f"Campaign {campaign_id}",
        "status": "active", 
        "message": "Campaign detail endpoint working",
        "service": "railway-simple"
    }

# Campaign orchestration endpoints
@app.get("/api/v2/campaigns/orchestration/campaigns/{campaign_id}/scheduled-content")
async def get_scheduled_content(campaign_id: str):
    """Get scheduled content for campaign."""
    return {
        "scheduled_content": [],
        "calendar": {
            "events": [],
            "timeline": []
        },
        "campaign_id": campaign_id,
        "message": "Scheduled content endpoint working",
        "service": "railway-simple"
    }

@app.get("/api/v2/campaigns/orchestration/campaigns/{campaign_id}/feedback-analytics")
async def get_feedback_analytics(campaign_id: str):
    """Get feedback analytics for campaign."""
    return {
        "analytics": {
            "engagement": {"likes": 0, "shares": 0, "comments": 0},
            "reach": {"impressions": 0, "unique_users": 0},
            "conversion": {"clicks": 0, "conversions": 0, "rate": 0}
        },
        "feedback": [],
        "campaign_id": campaign_id,
        "message": "Feedback analytics endpoint working",
        "service": "railway-simple"
    }

# Settings endpoints
@app.get("/api/settings")
async def get_settings():
    """Get application settings."""
    return {
        "settings": {
            "notifications": {"email": True, "push": False},
            "theme": "light",
            "language": "en",
            "timezone": "UTC"
        },
        "message": "Settings endpoint working",
        "service": "railway-simple"
    }

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Get knowledge base documents."""
    return {
        "documents": [],
        "categories": ["General", "Marketing", "Finance", "Technology"],
        "total": 0,
        "message": "Knowledge base endpoint working",
        "service": "railway-simple"
    }

@app.get("/api/knowledge-base/documents")
async def get_knowledge_documents():
    """Get knowledge base documents list."""
    return {
        "documents": [],
        "total": 0,
        "message": "Knowledge base documents endpoint working",
        "service": "railway-simple"
    }

# Additional orchestration endpoints
@app.get("/api/v2/campaigns/orchestration/campaigns/{campaign_id}/tasks")
async def get_campaign_tasks(campaign_id: str):
    """Get tasks for campaign."""
    return {
        "tasks": [],
        "campaign_id": campaign_id,
        "message": "Campaign tasks endpoint working",
        "service": "railway-simple"
    }

@app.get("/api/v2/campaigns/orchestration/campaigns/{campaign_id}/content")
async def get_campaign_content(campaign_id: str):
    """Get content for campaign."""
    return {
        "content": [],
        "campaign_id": campaign_id,
        "message": "Campaign content endpoint working",
        "service": "railway-simple"
    }

# User and profile endpoints
@app.get("/api/user/profile")
async def get_user_profile():
    """Get user profile."""
    return {
        "user": {
            "id": "user-123",
            "name": "Demo User",
            "email": "demo@credilinq.com",
            "role": "admin"
        },
        "message": "User profile endpoint working",
        "service": "railway-simple"
    }

print("✅ [RAILWAY DEBUG] Simple Railway app created with essential API routes")
logger.info("✅ [RAILWAY DEBUG] Simple Railway app created with essential API routes")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.main_railway_simple:app", host="0.0.0.0", port=port)