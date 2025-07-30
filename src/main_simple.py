"""
Simplified FastAPI application for debugging.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
import datetime
from pydantic import BaseModel
from typing import List

# Import campaign routes (simplified) - commented out for now
# from src.api.routes.campaigns_simple import router as campaigns_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class BlogCreateRequest(BaseModel):
    title: str
    company_context: str
    content_type: str = "blog"

class BlogSummary(BaseModel):
    id: str
    title: str
    status: str
    created_at: str

def get_db_connection():
    """Get database connection."""
    import psycopg2
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
    return psycopg2.connect(database_url)

# In-memory storage for blogs
blogs_db = [
    {"id": "test-1", "title": "Test Blog 1", "status": "draft", "created_at": "2025-07-30T15:30:00Z"},
    {"id": "test-2", "title": "Test Blog 2", "status": "published", "created_at": "2025-07-30T15:31:00Z"}
]

# Create simplified FastAPI application
app = FastAPI(
    title="CrediLinQ AI Content Platform API (Debug)",
    description="Simplified version for debugging",
    version="2.0.0",
    debug=True
)

# Add CORS middleware only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include campaign routes - commented out for now
# try:
#     app.include_router(campaigns_router)
#     print("✅ Campaign routes included successfully")
# except Exception as e:
#     print(f"❌ Error including campaign routes: {e}")

# Simple test endpoints
@app.get("/")
async def root():
    return {"message": "CrediLinQ API (Debug Mode)"}

@app.get("/api/test")
async def test():
    return {"message": "Test endpoint working"}

@app.get("/api/blogs", response_model=List[BlogSummary])
async def get_blogs():
    """Get all blogs from database."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, status, "createdAt"
                FROM "BlogPost" 
                WHERE status != 'deleted' 
                ORDER BY "createdAt" DESC
            """)
            rows = cur.fetchall()
            
            logger.info(f"Found {len(rows)} rows from database")
            
            blogs = []
            for i, row in enumerate(rows):
                logger.info(f"Processing row {i}: {row}")
                
                # Access by index: id, title, status, createdAt
                blog_id = str(row[0]) if row[0] else ''
                title = str(row[1]) if row[1] else 'Untitled'
                status = str(row[2]) if row[2] else 'draft'
                
                # Handle date
                created_at = row[3]
                if created_at:
                    if hasattr(created_at, 'isoformat'):
                        created_at_str = created_at.isoformat()
                    else:
                        created_at_str = str(created_at)
                else:
                    created_at_str = datetime.datetime.utcnow().isoformat()
                
                blog_summary = BlogSummary(
                    id=blog_id,
                    title=title,
                    status=status,
                    created_at=created_at_str
                )
                
                logger.info(f"Created blog summary: {blog_summary}")
                blogs.append(blog_summary)
            
            logger.info(f"Returning {len(blogs)} blogs")
            return blogs
    except Exception as e:
        logger.error(f"Error listing blogs: {str(e)}")
        return []

@app.post("/api/blogs", response_model=BlogSummary)
async def create_blog(request: BlogCreateRequest):
    """Create a new blog."""
    blog_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    
    new_blog = {
        "id": blog_id,
        "title": request.title,
        "status": "draft",
        "created_at": created_at
    }
    
    blogs_db.append(new_blog)
    logger.info(f"Created blog: {new_blog}")
    
    return new_blog

@app.get("/api/blogs/{blog_id}", response_model=BlogSummary)
async def get_blog(blog_id: str):
    """Get a specific blog."""
    for blog in blogs_db:
        if blog["id"] == blog_id:
            return blog
    return {"error": "Blog not found"}

@app.put("/api/blogs/{blog_id}")
async def update_blog(blog_id: str, request: dict):
    """Update a blog."""
    for blog in blogs_db:
        if blog["id"] == blog_id:
            blog["title"] = request.get("title", blog["title"])
            blog["status"] = request.get("status", blog["status"])
            return blog
    return {"error": "Blog not found"}

@app.post("/api/blogs/{blog_id}/publish")
async def publish_blog(blog_id: str):
    """Publish a blog."""
    for blog in blogs_db:
        if blog["id"] == blog_id:
            blog["status"] = "published"
            return blog
    return {"error": "Blog not found"}


@app.post("/api/blogs/{blog_id}/create-campaign")
async def create_campaign_from_blog(blog_id: str, request: dict):
    """Create a campaign from a blog post."""
    try:
        campaign_name = request.get("campaign_name", f"Campaign for {blog_id}")
        
        # Check if blog exists in our mock data
        blog_exists = any(blog["id"] == blog_id for blog in blogs_db)
        if not blog_exists:
            return {"error": "Blog post not found"}
        
        # Create campaign ID
        campaign_id = str(uuid.uuid4())
        
        logger.info(f"Created campaign {campaign_id} for blog {blog_id}")
        
        return {
            "message": "Campaign created successfully",
            "campaign_id": campaign_id,
            "blog_id": blog_id,
            "campaign_name": campaign_name
        }
        
    except Exception as e:
        logger.error(f"Error creating campaign for blog {blog_id}: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "debug": True, "blogs_count": len(blogs_db)}

# Campaign endpoints (direct implementation)
@app.get("/api/campaigns")
async def list_campaigns():
    """List all campaigns (direct implementation)"""
    return [
        {
            "id": "test-campaign-1",
            "name": "Test Campaign 1",
            "status": "draft",
            "progress": 0.0,
            "total_tasks": 5,
            "completed_tasks": 0,
            "created_at": "2025-01-27T10:00:00Z"
        },
        {
            "id": "test-campaign-2",
            "name": "Test Campaign 2",
            "status": "active",
            "progress": 60.0,
            "total_tasks": 3,
            "completed_tasks": 2,
            "created_at": "2025-01-27T11:00:00Z"
        }
    ]

@app.post("/api/campaigns")
async def create_campaign(request: dict):
    """Create a new campaign (direct implementation)"""
    return {
        "success": True,
        "campaign_id": "new-campaign-id",
        "message": "Campaign created successfully",
        "strategy": {
            "target_audience": "Professional audience",
            "key_messages": ["Message 1", "Message 2"],
            "distribution_channels": ["linkedin", "twitter"]
        },
        "timeline": [],
        "tasks": 5
    }

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get campaign details (direct implementation)"""
    return {
        "id": campaign_id,
        "name": f"Campaign {campaign_id}",
        "status": "draft",
        "strategy": {
            "target_audience": "Professional audience",
            "key_messages": ["Message 1", "Message 2"],
            "distribution_channels": ["linkedin", "twitter"]
        },
        "timeline": [],
        "tasks": [],
        "scheduled_posts": [],
        "performance": {
            "total_posts": 0,
            "published_posts": 0,
            "success_rate": 0
        }
    }


@app.post("/api/campaigns/{campaign_id}/schedule")
async def schedule_campaign(campaign_id: str):
    """Schedule a campaign (direct implementation)"""
    return {
        "success": True,
        "message": f"Campaign {campaign_id} scheduled successfully",
        "scheduled_posts": 5,
        "next_post": "2025-01-28T10:00:00Z"
    }


@app.post("/api/campaigns/{campaign_id}/distribute")
async def distribute_campaign(campaign_id: str):
    """Distribute a campaign (direct implementation)"""
    return {
        "success": True,
        "message": f"Campaign {campaign_id} distributed successfully",
        "published_posts": 3,
        "remaining_posts": 2
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 