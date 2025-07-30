#!/usr/bin/env python3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import psycopg2
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import uuid
import datetime

# Load environment variables
load_dotenv()

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

# Create FastAPI application
app = FastAPI(
    title="CrediLinQ AI Content Platform API (Final)",
    description="Simple and working version",
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

def get_db_connection():
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
    return psycopg2.connect(database_url)

@app.get("/")
async def root():
    return {"message": "CrediLinQ API (Final Version)"}

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
            
            new_blog = BlogSummary(
                id=blog_id,
                title=request.title,
                status="draft",
                created_at=created_at.isoformat()
            )
            
            logger.info(f"Created blog: {new_blog}")
            return new_blog
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
                UPDATE "BlogPost" 
                SET status = 'published'
                WHERE id = %s
            """, (blog_id,))
            
            if cur.rowcount == 0:
                raise Exception("Blog not found")
            
            conn.commit()
            return {"message": "Blog published successfully"}
    except Exception as e:
        logger.error(f"Error publishing blog: {str(e)}")
        raise Exception(f"Failed to publish blog: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "final"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 