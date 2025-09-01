"""
Simple Railway FastAPI app with essential API routes.
Connects to real database but bypasses complex agent imports.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import json
import time
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
    """Documents endpoint - returns same as knowledge base."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Check for documents table
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'documents'
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                documents = []
                
                if 'documents' in tables:
                    cur.execute("""
                        SELECT id, title, content, metadata, created_at, updated_at
                        FROM documents 
                        ORDER BY created_at DESC
                        LIMIT 100
                    """)
                    
                    for row in cur.fetchall():
                        doc_id, title, content, metadata, created_at, updated_at = row
                        metadata = metadata or {}
                        
                        documents.append({
                            "id": doc_id,
                            "title": title,
                            "content": content[:300] + "..." if content and len(content) > 300 else content or "",
                            "file_name": title,
                            "file_size": len(content) if content else 0,
                            "file_type": metadata.get("file_type", "text"),
                            "category": metadata.get("category", "General"),
                            "created_at": created_at.isoformat() if created_at else None,
                            "updated_at": updated_at.isoformat() if updated_at else None,
                            "status": "processed"
                        })
                
                logger.info(f"🔍 Documents list: {len(documents)} documents found")
                
                return {
                    "documents": documents,
                    "total": len(documents),
                    "service": "railway-simple"
                }
                
        except Exception as e:
            logger.error(f"Database error getting documents: {e}")
            return {
                "documents": [],
                "total": 0,
                "error": str(e),
                "service": "railway-simple"
            }
    
    return {
        "documents": [],
        "total": 0,
        "message": "No database connection",
        "service": "railway-simple"
    }

@app.post("/api/documents/upload")
async def upload_documents():
    """Upload documents endpoint - simplified for Railway."""
    # For now, return a success response for frontend compatibility
    # In a full implementation, this would handle file uploads and store in database
    
    logger.info("🔍 Document upload attempted (simplified response)")
    
    return {
        "id": f"doc-{int(time.time())}",
        "title": "Uploaded Document",
        "status": "processed",
        "file_name": "document.txt",
        "file_size": 1024,
        "created_at": datetime.now().isoformat(),
        "message": "Upload functionality not fully implemented in Railway simple version",
        "service": "railway-simple"
    }

# Company Profile Models and Functions
class LinkItem(BaseModel):
    label: str = Field(..., max_length=200)
    url: str = Field(..., max_length=1000)

class CompanyProfile(BaseModel):
    companyName: Optional[str] = Field(None, max_length=200)
    companyContext: str = Field(..., max_length=10000)
    brandVoice: Optional[str] = Field(None, max_length=5000)
    valueProposition: Optional[str] = Field(None, max_length=5000)
    industries: List[str] = Field(default_factory=list, max_items=10)
    targetAudiences: List[str] = Field(default_factory=list, max_items=10)
    tonePresets: List[str] = Field(default_factory=list, max_items=10)
    keywords: List[str] = Field(default_factory=list, max_items=50)
    styleGuidelines: Optional[str] = Field(None, max_length=10000)
    prohibitedTopics: List[str] = Field(default_factory=list, max_items=50)
    complianceNotes: Optional[str] = Field(None, max_length=10000)
    links: List[LinkItem] = Field(default_factory=list, max_items=20)
    defaultCTA: Optional[str] = Field(None, max_length=1000)
    updatedAt: Optional[str] = None

def _ensure_settings_table():
    """Create the app_settings table if it doesn't exist."""
    if not db_config:
        return
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error ensuring settings table: {e}")

def _get_setting(key: str) -> Optional[dict]:
    """Get a setting from the database."""
    if not db_config:
        return None
    _ensure_settings_table()
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
            row = cur.fetchone()
            if not row:
                return None
            raw = row[0] if isinstance(row, tuple) else row.get("value")
            try:
                return raw if isinstance(raw, dict) else json.loads(raw)
            except Exception:
                return {}
    except Exception as e:
        logger.error(f"Error getting setting {key}: {e}")
        return None

@app.get("/api/settings/company-profile")
async def get_company_profile():
    """Get company profile from database."""
    try:
        data = _get_setting("company_profile") or {}
        # Provide minimal defaults so frontend can prefill
        if "companyContext" not in data:
            data["companyContext"] = ""
        return CompanyProfile(**data)
    except Exception as e:
        logger.error(f"Error getting company profile: {e}")
        # Fallback to basic profile if database fails
        return CompanyProfile(
            companyName="CrediLinq.ai",
            companyContext="AI-powered content management platform",
            industries=[],
            targetAudiences=[],
            tonePresets=[],
            keywords=[],
            prohibitedTopics=[],
            links=[]
        )

def _upsert_setting(key: str, value: dict):
    """Update or insert a setting in the database."""
    if not db_config:
        return
    _ensure_settings_table()
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO app_settings(key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (key, json.dumps(value)),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error upserting setting {key}: {e}")
        raise

def _normalize_profile_dict(data: dict) -> dict:
    """Normalize user-provided data."""
    def _clean_list(values):
        if not isinstance(values, list):
            return []
        cleaned = []
        for v in values:
            if isinstance(v, str) and v.strip():
                cleaned.append(v.strip())
        return cleaned

    # Normalize arrays
    for key in ["industries", "targetAudiences", "tonePresets", "keywords", "prohibitedTopics"]:
        data[key] = _clean_list(data.get(key, []))

    # Links: filter invalid and add https:// if missing
    links = []
    for link in data.get("links", []) or []:
        try:
            label = (link.get("label") or "").strip()
            url = (link.get("url") or "").strip()
            if label and url:
                if not url.startswith("http://") and not url.startswith("https://"):
                    url = "https://" + url
                links.append({"label": label, "url": url})
        except Exception:
            continue
    data["links"] = links

    # Strings: trim
    for key in ["companyName", "companyContext", "brandVoice", "valueProposition", "styleGuidelines", "complianceNotes", "defaultCTA"]:
        if key in data and isinstance(data[key], str):
            data[key] = data[key].strip()
    return data

@app.put("/api/settings/company-profile")
async def update_company_profile(profile: CompanyProfile):
    """Update company profile in database."""
    try:
        payload = _normalize_profile_dict(profile.model_dump())
        payload["updatedAt"] = datetime.utcnow().isoformat()
        _upsert_setting("company_profile", payload)
        return {"message": "Company profile updated", "updatedAt": payload["updatedAt"]}
    except Exception as e:
        logger.error(f"Error updating company profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

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
                
                # Get campaign tasks with full details
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
                
                total_tasks = len(tasks)
                completed_tasks = len([t for t in tasks if t["status"] == "completed"])
                
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
                    "tasks": tasks,  # Include full tasks array for frontend
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
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get actual tasks as scheduled content
                cur.execute("""
                    SELECT id, task_type, target_format, target_asset, status, 
                           result, created_at, completed_at, priority
                    FROM campaign_tasks 
                    WHERE campaign_id = %s
                    ORDER BY priority DESC, created_at DESC
                """, (campaign_id,))
                
                scheduled_content = []
                calendar_events = []
                
                for row in cur.fetchall():
                    task_id, task_type, target_format, target_asset, status, result, created_at, completed_at, priority = row
                    
                    # Create scheduled content item
                    content_item = {
                        "id": task_id,
                        "title": f"{task_type} - {target_format}" if target_format else task_type,
                        "type": task_type,
                        "format": target_format,
                        "asset": target_asset,
                        "status": status,
                        "result": result,
                        "priority": priority,
                        "created_at": created_at.isoformat() if created_at else None,
                        "completed_at": completed_at.isoformat() if completed_at else None,
                        "scheduled_date": created_at.isoformat() if created_at else None
                    }
                    scheduled_content.append(content_item)
                    
                    # Create calendar event
                    if created_at:
                        calendar_events.append({
                            "id": task_id,
                            "title": content_item["title"],
                            "date": created_at.isoformat(),
                            "status": status,
                            "type": "task"
                        })
                
                logger.info(f"🔍 Scheduled content for {campaign_id}: {len(scheduled_content)} items")
                
                return {
                    "scheduled_content": scheduled_content,
                    "calendar": {
                        "events": calendar_events,
                        "timeline": calendar_events
                    },
                    "campaign_id": campaign_id,
                    "total": len(scheduled_content),
                    "service": "railway-simple"
                }
                
        except Exception as e:
            logger.error(f"Database error getting scheduled content: {e}")
            return {
                "scheduled_content": [],
                "calendar": {"events": [], "timeline": []},
                "campaign_id": campaign_id,
                "error": str(e),
                "service": "railway-simple"
            }
    
    return {
        "scheduled_content": [],
        "calendar": {
            "events": [],
            "timeline": []
        },
        "campaign_id": campaign_id,
        "message": "No database connection",
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
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Check for documents table
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('documents', 'document_chunks')
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                documents = []
                categories = set()
                
                # If we have documents table, query it
                if 'documents' in tables:
                    cur.execute("""
                        SELECT id, title, content, metadata, created_at, updated_at
                        FROM documents 
                        ORDER BY created_at DESC
                        LIMIT 100
                    """)
                    
                    for row in cur.fetchall():
                        doc_id, title, content, metadata, created_at, updated_at = row
                        metadata = metadata or {}
                        category = metadata.get("category", "General")
                        categories.add(category)
                        
                        documents.append({
                            "id": doc_id,
                            "title": title,
                            "content": content[:500] + "..." if content and len(content) > 500 else content or "",
                            "category": category,
                            "created_at": created_at.isoformat() if created_at else None,
                            "updated_at": updated_at.isoformat() if updated_at else None,
                            "metadata": metadata
                        })
                
                # If we have document_chunks table, also query that
                if 'document_chunks' in tables:
                    cur.execute("""
                        SELECT document_id, chunk_text, metadata
                        FROM document_chunks 
                        ORDER BY document_id, chunk_index
                        LIMIT 50
                    """)
                    
                    chunk_docs = {}
                    for row in cur.fetchall():
                        doc_id, chunk_text, metadata = row
                        metadata = metadata or {}
                        
                        if doc_id not in chunk_docs:
                            chunk_docs[doc_id] = {
                                "id": f"chunk-doc-{doc_id}",
                                "title": f"Document {doc_id} (Chunked)",
                                "content": "",
                                "category": metadata.get("category", "Knowledge Base"),
                                "created_at": None,
                                "updated_at": None,
                                "metadata": {"type": "chunked_document", "source_id": doc_id}
                            }
                            categories.add("Knowledge Base")
                        
                        chunk_docs[doc_id]["content"] += chunk_text + "\n\n"
                    
                    # Add chunk documents to main documents list
                    for doc in chunk_docs.values():
                        if len(doc["content"]) > 500:
                            doc["content"] = doc["content"][:500] + "..."
                        documents.append(doc)
                
                logger.info(f"🔍 Knowledge base: {len(documents)} documents found")
                
                return {
                    "documents": documents,
                    "categories": list(categories) if categories else ["General", "Marketing", "Finance", "Technology"],
                    "total": len(documents),
                    "service": "railway-simple"
                }
                
        except Exception as e:
            logger.error(f"Database error getting knowledge base: {e}")
            return {
                "documents": [],
                "categories": ["General", "Marketing", "Finance", "Technology"],
                "total": 0,
                "error": str(e),
                "service": "railway-simple"
            }
    
    return {
        "documents": [],
        "categories": ["General", "Marketing", "Finance", "Technology"],
        "total": 0,
        "message": "No database connection",
        "service": "railway-simple"
    }

@app.get("/api/knowledge-base/documents")
async def get_knowledge_documents():
    """Get knowledge base documents list."""
    # Reuse the same logic from get_knowledge_base
    knowledge_base_result = await get_knowledge_base()
    
    return {
        "documents": knowledge_base_result.get("documents", []),
        "total": knowledge_base_result.get("total", 0),
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

# Content deliverables endpoints (for Content Narrative tab)
@app.get("/api/v2/deliverables/campaign/{campaign_id}")
async def get_campaign_deliverables(campaign_id: str):
    """Get all content deliverables for a campaign."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Convert campaign tasks to deliverables format
                cur.execute("""
                    SELECT id, task_type, target_format, target_asset, status, 
                           result, created_at, completed_at, priority
                    FROM campaign_tasks 
                    WHERE campaign_id = %s
                    ORDER BY priority DESC, created_at DESC
                """, (campaign_id,))
                
                deliverables = []
                for row in cur.fetchall():
                    task_id, task_type, target_format, target_asset, status, result, created_at, completed_at, priority = row
                    
                    # Map task to deliverable format
                    deliverables.append({
                        "id": task_id,
                        "campaign_id": campaign_id,
                        "title": f"{task_type.replace('_', ' ').title()}" + (f" - {target_format}" if target_format else ""),
                        "content": result if isinstance(result, str) else str(result) if result else "",
                        "summary": (result[:200] + "..." if isinstance(result, str) and len(result) > 200 else result) if result else "",
                        "content_type": "blog_post" if "content" in task_type else "social_media_post" if "social" in task_type else "email_campaign" if "email" in task_type else "blog_post",
                        "format": target_format or "text",
                        "status": "published" if status == "completed" else "draft" if status == "pending" else "in_review",
                        "narrative_order": priority or 1,
                        "target_audience": "Financial services professionals",
                        "tone": "professional",
                        "platform": target_format or "general",
                        "word_count": len(result.split()) if isinstance(result, str) else 0,
                        "reading_time": max(1, len(result.split()) // 200) if isinstance(result, str) else 1,
                        "created_at": created_at.isoformat() if created_at else None,
                        "updated_at": completed_at.isoformat() if completed_at else created_at.isoformat() if created_at else None,
                        "created_by": "AI Agent",
                        "metadata": {
                            "task_type": task_type,
                            "original_status": status,
                            "priority": priority
                        }
                    })
                
                logger.info(f"🔍 Campaign deliverables for {campaign_id}: {len(deliverables)} items")
                return deliverables
                
        except Exception as e:
            logger.error(f"Database error getting campaign deliverables: {e}")
            return []
    
    return []

@app.get("/api/v2/deliverables/campaign/{campaign_id}/narrative")
async def get_campaign_narrative(campaign_id: str):
    """Get the content narrative for a campaign."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get campaign info
                cur.execute("""
                    SELECT name, metadata, created_at, updated_at
                    FROM campaigns 
                    WHERE id = %s
                """, (campaign_id,))
                
                campaign_row = cur.fetchone()
                if not campaign_row:
                    raise HTTPException(status_code=404, detail="Campaign not found")
                
                name, metadata, created_at, updated_at = campaign_row
                metadata = metadata or {}
                
                # Get task count
                cur.execute("""
                    SELECT COUNT(*) as total, COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
                    FROM campaign_tasks 
                    WHERE campaign_id = %s
                """, (campaign_id,))
                
                task_stats = cur.fetchone()
                total_pieces = task_stats[0] if task_stats else 0
                completed_pieces = task_stats[1] if task_stats else 0
                
                # Build narrative based on campaign metadata
                narrative = {
                    "id": f"narrative-{campaign_id}",
                    "campaign_id": campaign_id,
                    "title": f"{name} - Content Strategy Narrative",
                    "description": metadata.get("description", "AI-powered content campaign focused on lead generation and partnership acquisition"),
                    "narrative_theme": metadata.get("strategy_type", "lead_generation").replace("_", " ").title(),
                    "key_story_arc": [
                        "Market Analysis & Opportunity Identification",
                        "Audience Engagement & Thought Leadership",
                        "Solution Positioning & Value Proposition",
                        "Partnership Development & Lead Generation",
                        "Conversion Optimization & Relationship Building"
                    ],
                    "content_flow": {
                        "phase_1": {
                            "title": "Foundation Building",
                            "description": "Establish thought leadership and market presence",
                            "content_types": ["blog_post", "email_campaign"],
                            "pieces_count": total_pieces // 3 if total_pieces > 0 else 0
                        },
                        "phase_2": {
                            "title": "Engagement Expansion", 
                            "description": "Amplify reach through social channels and partnerships",
                            "content_types": ["social_media_post", "email_campaign"],
                            "pieces_count": total_pieces // 3 if total_pieces > 0 else 0
                        },
                        "phase_3": {
                            "title": "Conversion Focus",
                            "description": "Drive qualified leads and partnership opportunities",
                            "content_types": ["blog_post", "social_media_post", "email_campaign"],
                            "pieces_count": total_pieces - (2 * (total_pieces // 3)) if total_pieces > 0 else 0
                        }
                    },
                    "total_pieces": total_pieces,
                    "completed_pieces": completed_pieces,
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None
                }
                
                logger.info(f"🔍 Campaign narrative for {campaign_id}: {narrative['title']}")
                return narrative
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error getting campaign narrative: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    raise HTTPException(status_code=503, detail="Database not available")

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