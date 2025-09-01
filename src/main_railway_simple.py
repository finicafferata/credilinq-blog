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
    """Get specific campaign from database."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get campaign details with metadata
                cur.execute("""
                    SELECT id, name, status, created_at, updated_at, blog_post_id, metadata
                    FROM campaigns 
                    WHERE id = %s
                """, (campaign_id,))
                
                campaign_row = cur.fetchone()
                if not campaign_row:
                    raise HTTPException(status_code=404, detail="Campaign not found")
                
                # Extract metadata JSON
                metadata = campaign_row[6] if campaign_row[6] else {}
                
                # Get campaign tasks count
                cur.execute("""
                    SELECT COUNT(*) as total_tasks,
                           COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks
                    FROM campaign_tasks 
                    WHERE campaign_id = %s
                """, (campaign_id,))
                
                task_stats = cur.fetchone()
                total_tasks = task_stats[0] if task_stats else 0
                completed_tasks = task_stats[1] if task_stats else 0
                
                # Debug: Log the raw metadata for troubleshooting
                logger.info(f"🔍 Campaign {campaign_id} metadata: {metadata}")
                logger.info(f"🔍 Task stats: total={total_tasks}, completed={completed_tasks}")
                
                response_data = {
                    "id": campaign_row[0],
                    "name": campaign_row[1],
                    "title": campaign_row[1],  # For frontend compatibility
                    "status": campaign_row[2],
                    "created_at": campaign_row[3].isoformat() if campaign_row[3] else None,
                    "updated_at": campaign_row[4].isoformat() if campaign_row[4] else None,
                    "blog_post_id": campaign_row[5],
                    # Extract data from JSON metadata using the exact keys from debug output
                    "target_market": metadata.get("target_audience", "Companies interested in embedded finance"),
                    "campaign_type": metadata.get("strategy_type", "product_launch").replace("_", " ").title(),
                    "focus": metadata.get("company_context", "Agentic AI marketing and lead analysis"),
                    "description": metadata.get("description", "product launch campaign targeting companies interested in embedded finance solutions with focus on partnership acquisition"),
                    "priority": metadata.get("priority", "high"),
                    "timeline_weeks": metadata.get("timeline_weeks", 2),
                    "distribution_channels": metadata.get("distribution_channels", []),
                    "success_metrics": metadata.get("success_metrics", {}),
                    "progress": metadata.get("progress", 0.0),
                    "rerun_count": metadata.get("rerun_count", 0),
                    "content_pieces": metadata.get("success_metrics", {}).get("content_pieces", 0),
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "scheduled_count": 0,  # Will add this later
                    "service": "railway-simple"
                }
                
                logger.info(f"🔍 Returning campaign data: {response_data}")
                return response_data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error getting campaign {campaign_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # Fallback if no database
    raise HTTPException(status_code=503, detail="Database not available")

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
    """Get tasks for campaign from database."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, task_type, target_format, target_asset, status, 
                           result, image_url, error, priority, created_at, 
                           updated_at, started_at, completed_at
                    FROM campaign_tasks 
                    WHERE campaign_id = %s
                    ORDER BY priority DESC, created_at DESC
                """, (campaign_id,))
                
                tasks = []
                for row in cur.fetchall():
                    tasks.append({
                        "id": row[0],
                        "task_type": row[1],
                        "target_format": row[2],
                        "target_asset": row[3],
                        "status": row[4],
                        "result": row[5],
                        "image_url": row[6],
                        "error": row[7],
                        "priority": row[8],
                        "created_at": row[9].isoformat() if row[9] else None,
                        "updated_at": row[10].isoformat() if row[10] else None,
                        "started_at": row[11].isoformat() if row[11] else None,
                        "completed_at": row[12].isoformat() if row[12] else None,
                    })
                
                return {
                    "tasks": tasks,
                    "total": len(tasks),
                    "campaign_id": campaign_id,
                    "service": "railway-simple"
                }
        except Exception as e:
            logger.error(f"Database error getting campaign tasks: {e}")
            return {
                "tasks": [],
                "total": 0,
                "campaign_id": campaign_id,
                "error": str(e),
                "service": "railway-simple"
            }
    
    return {
        "tasks": [],
        "total": 0,
        "campaign_id": campaign_id,
        "message": "No database connection",
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

# Debug endpoint to check campaign data
@app.get("/api/debug/campaign/{campaign_id}")
async def debug_campaign(campaign_id: str):
    """Debug campaign data to see what exists in database."""
    if not db_config:
        return {"error": "No database connection"}
    
    debug_info = {"campaign_id": campaign_id, "queries": []}
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check campaign exists
            cur.execute("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            debug_info["queries"].append({
                "query": "campaigns table",
                "found": campaign is not None,
                "data": campaign if campaign else None
            })
            
            # Check briefings in main schema
            cur.execute("SELECT * FROM briefings WHERE campaign_id = %s", (campaign_id,))
            briefing = cur.fetchone()
            debug_info["queries"].append({
                "query": "briefings table (main schema)",
                "found": briefing is not None,
                "data": briefing if briefing else None
            })
            
            # Check campaign tasks
            cur.execute("SELECT COUNT(*), status FROM campaign_tasks WHERE campaign_id = %s GROUP BY status", (campaign_id,))
            tasks = cur.fetchall()
            debug_info["queries"].append({
                "query": "campaign_tasks counts by status", 
                "found": len(tasks) > 0,
                "data": tasks
            })
            
            # Check blog posts linked to campaign
            cur.execute("SELECT * FROM blog_posts WHERE id = (SELECT blog_post_id FROM campaigns WHERE id = %s)", (campaign_id,))
            blog_post = cur.fetchone()
            debug_info["queries"].append({
                "query": "linked blog_post",
                "found": blog_post is not None,
                "data": blog_post[:5] if blog_post else None  # First 5 columns only
            })
            
            return debug_info
            
    except Exception as e:
        return {"error": str(e), "campaign_id": campaign_id}

print("✅ [RAILWAY DEBUG] Simple Railway app created with essential API routes")
logger.info("✅ [RAILWAY DEBUG] Simple Railway app created with essential API routes")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.main_railway_simple:app", host="0.0.0.0", port=port)