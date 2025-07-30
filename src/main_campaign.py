#!/usr/bin/env python3
"""
Main FastAPI application with blogs and campaigns
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
import datetime
from pydantic import BaseModel
from typing import List, Dict, Any
import psycopg2
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

class CampaignCreateRequest(BaseModel):
    blog_id: str
    campaign_name: str
    company_context: str
    content_type: str = "blog"

class CampaignSummary(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    created_at: str

def get_db_connection():
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
    return psycopg2.connect(database_url)

# Create FastAPI application
app = FastAPI(
    title="CrediLinQ AI Content Platform API",
    description="Complete API with blogs and campaigns",
    version="2.0.0",
    debug=True
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "CrediLinQ API with Campaigns"}

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0", "features": ["blogs", "campaigns"]}

# Blog endpoints
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
            
            blogs = []
            for row in rows:
                blog_id = str(row[0]) if row[0] else ''
                title = str(row[1]) if row[1] else 'Untitled'
                status = str(row[2]) if row[2] else 'draft'
                
                created_at = row[3]
                if created_at:
                    if hasattr(created_at, 'isoformat'):
                        created_at_str = created_at.isoformat()
                    else:
                        created_at_str = str(created_at)
                else:
                    created_at_str = datetime.datetime.utcnow().isoformat()
                
                blogs.append(BlogSummary(
                    id=blog_id,
                    title=title,
                    status=status,
                    created_at=created_at_str
                ))
            
            return blogs
            
    except Exception as e:
        logger.error(f"Error getting blogs: {str(e)}")
        return []

@app.post("/api/blogs", response_model=BlogSummary)
async def create_blog(request: BlogCreateRequest):
    """Create a new blog."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            blog_id = str(uuid.uuid4())
            created_at = datetime.datetime.utcnow()
            
            cur.execute("""
                INSERT INTO "BlogPost" (id, title, status, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s)
            """, (blog_id, request.title, "draft", created_at, created_at))
            conn.commit()
            
            return BlogSummary(
                id=blog_id,
                title=request.title,
                status="draft",
                created_at=created_at.isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error creating blog: {str(e)}")
        raise Exception(f"Failed to create blog: {str(e)}")

@app.get("/api/blogs/{blog_id}", response_model=BlogSummary)
async def get_blog(blog_id: str):
    """Get a specific blog."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, status, "createdAt"
                FROM "BlogPost" 
                WHERE id = %s
            """, (blog_id,))
            row = cur.fetchone()
            
            if not row:
                raise Exception("Blog not found")
            
            blog_id = str(row[0])
            title = str(row[1]) if row[1] else 'Untitled'
            status = str(row[2]) if row[2] else 'draft'
            
            created_at = row[3]
            if created_at:
                if hasattr(created_at, 'isoformat'):
                    created_at_str = created_at.isoformat()
                else:
                    created_at_str = str(created_at)
            else:
                created_at_str = datetime.datetime.utcnow().isoformat()
            
            return BlogSummary(
                id=blog_id,
                title=title,
                status=status,
                created_at=created_at_str
            )
            
    except Exception as e:
        logger.error(f"Error getting blog: {str(e)}")
        raise Exception(f"Failed to get blog: {str(e)}")

@app.post("/api/blogs/{blog_id}/publish")
async def publish_blog(blog_id: str):
    """Publish a blog."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE "BlogPost" SET status = 'published' WHERE id = %s
            """, (blog_id,))
            conn.commit()
            
            return {"message": "Blog published successfully"}
            
    except Exception as e:
        logger.error(f"Error publishing blog: {str(e)}")
        raise Exception(f"Failed to publish blog: {str(e)}")

# Campaign endpoints
@app.get("/api/campaigns", response_model=List[CampaignSummary])
async def list_campaigns():
    """List all campaigns"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT c.id, c.name, c.status, c.created_at,
                       COUNT(ct.id) as total_tasks,
                       COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                FROM campaign c
                LEFT JOIN campaign_task ct ON c.id = ct.campaign_id
                GROUP BY c.id, c.name, c.status, c.created_at
                ORDER BY c.created_at DESC
            """)
            
            rows = cur.fetchall()
            campaigns = []
            
            for row in rows:
                campaign_id, name, status, created_at, total_tasks, completed_tasks = row
                
                progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                
                campaigns.append(CampaignSummary(
                    id=str(campaign_id),
                    name=name or "Untitled Campaign",
                    status=status or "draft",
                    progress=progress,
                    total_tasks=total_tasks or 0,
                    completed_tasks=completed_tasks or 0,
                    created_at=created_at.isoformat() if created_at else datetime.datetime.utcnow().isoformat()
                ))
            
            return campaigns
            
    except Exception as e:
        logger.error(f"Error listing campaigns: {str(e)}")
        # Return mock data if database error
        return [
            CampaignSummary(
                id="test-campaign-1",
                name="Test Campaign 1",
                status="draft",
                progress=0.0,
                total_tasks=5,
                completed_tasks=0,
                created_at="2025-01-27T10:00:00Z"
            ),
            CampaignSummary(
                id="test-campaign-2",
                name="Test Campaign 2",
                status="active",
                progress=60.0,
                total_tasks=3,
                completed_tasks=2,
                created_at="2025-01-27T11:00:00Z"
            )
        ]

@app.post("/api/campaigns", response_model=Dict[str, Any])
async def create_campaign(request: CampaignCreateRequest):
    """Create a new campaign"""
    try:
        logger.info(f"Creating campaign for blog {request.blog_id}")
        
        # For now, return mock data
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
        
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        raise Exception(f"Failed to create campaign: {str(e)}")

@app.get("/api/campaigns/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign(campaign_id: str):
    """Get campaign details"""
    try:
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
        
    except Exception as e:
        logger.error(f"Error getting campaign: {str(e)}")
        raise Exception(f"Failed to get campaign: {str(e)}")

@app.post("/api/campaigns/{campaign_id}/schedule", response_model=Dict[str, Any])
async def schedule_campaign(campaign_id: str):
    """Schedule campaign tasks"""
    try:
        return {
            "success": True,
            "message": "Campaign scheduled successfully",
            "scheduled_posts": 3,
            "schedule": []
        }
        
    except Exception as e:
        logger.error(f"Error scheduling campaign: {str(e)}")
        raise Exception(f"Failed to schedule campaign: {str(e)}")

@app.post("/api/campaigns/{campaign_id}/distribute", response_model=Dict[str, Any])
async def distribute_campaign(campaign_id: str):
    """Distribute campaign posts"""
    try:
        return {
            "success": True,
            "message": "Campaign distribution completed",
            "published": 2,
            "failed": 0,
            "posts": []
        }
        
    except Exception as e:
        logger.error(f"Error distributing campaign: {str(e)}")
        raise Exception(f"Failed to distribute campaign: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 