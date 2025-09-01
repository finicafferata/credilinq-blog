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
                    SELECT id, title, status, created_at, updated_at, priority, description
                    FROM campaigns 
                    ORDER BY created_at DESC
                """)
                campaigns = []
                for row in cur.fetchall():
                    campaigns.append({
                        "id": row[0],
                        "title": row[1],
                        "status": row[2],
                        "created_at": row[3].isoformat() if row[3] else None,
                        "updated_at": row[4].isoformat() if row[4] else None,
                        "priority": row[5],
                        "description": row[6]
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

print("✅ [RAILWAY DEBUG] Simple Railway app created with essential API routes")
logger.info("✅ [RAILWAY DEBUG] Simple Railway app created with essential API routes")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.main_railway_simple:app", host="0.0.0.0", port=port)