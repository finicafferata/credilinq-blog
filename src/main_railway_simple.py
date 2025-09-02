"""
Simple Railway FastAPI app with essential API routes.
Connects to real database but bypasses complex agent imports.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
import sys
import json
import time
from datetime import datetime
import asyncio

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

# Try to connect to database - with extensive debugging
db_config = None
print(f"🔍 [RAILWAY DEBUG] Current working directory: {os.getcwd()}")
print(f"🔍 [RAILWAY DEBUG] Python path: {sys.path[:3]}...")  # Show first 3 paths
print(f"🔍 [RAILWAY DEBUG] __file__: {__file__ if '__file__' in globals() else 'not defined'}")

import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
    print("🔍 [RAILWAY DEBUG] Added '.' to Python path")

if '/app' not in sys.path and os.path.exists('/app'):
    sys.path.insert(0, '/app')
    print("🔍 [RAILWAY DEBUG] Added '/app' to Python path")

# Try multiple import strategies
import_attempts = [
    ("src.config.database", "db_config"),
    ("config.database", "db_config"), 
    (".config.database", "db_config"),
]

for module_path, attr_name in import_attempts:
    try:
        print(f"🔍 [RAILWAY DEBUG] Trying to import {attr_name} from {module_path}")
        module = __import__(module_path, fromlist=[attr_name])
        db_config = getattr(module, attr_name)
        print(f"✅ [RAILWAY DEBUG] Successfully imported db_config from {module_path}")
        print(f"✅ [RAILWAY DEBUG] db_config type: {type(db_config)}")
        logger.info(f"✅ [RAILWAY DEBUG] Database connection loaded from {module_path}")
        break
    except Exception as e:
        print(f"⚠️ [RAILWAY DEBUG] Failed to import from {module_path}: {e}")
        logger.warning(f"⚠️ [RAILWAY DEBUG] Failed to import from {module_path}: {e}")

if db_config is None:
    print("❌ [RAILWAY DEBUG] All database import attempts failed!")
    logger.error("❌ [RAILWAY DEBUG] All database import attempts failed!")
    
    # List available modules for debugging
    try:
        import os
        if os.path.exists('/app/src'):
            print("🔍 [RAILWAY DEBUG] Contents of /app/src:")
            for item in os.listdir('/app/src'):
                print(f"    {item}")
        if os.path.exists('/app/src/config'):
            print("🔍 [RAILWAY DEBUG] Contents of /app/src/config:")
            for item in os.listdir('/app/src/config'):
                print(f"    {item}")
    except Exception as dir_e:
        print(f"⚠️ [RAILWAY DEBUG] Could not list directories: {dir_e}")

# Initialize AI content service
ai_content_service = None
try:
    from src.services.ai_content_service import ai_content_service as _ai_service
    ai_content_service = _ai_service
    if ai_content_service.is_available():
        print("✅ [RAILWAY DEBUG] AI content service loaded with Google Gemini")
        logger.info("✅ [RAILWAY DEBUG] AI content service loaded with Google Gemini")
    else:
        print("⚠️ [RAILWAY DEBUG] AI content service loaded but Gemini not available (missing GEMINI_API_KEY or GOOGLE_API_KEY)")
        logger.warning("⚠️ [RAILWAY DEBUG] AI content service loaded but Gemini not available (missing GEMINI_API_KEY or GOOGLE_API_KEY)")
except Exception as e:
    print(f"⚠️ [RAILWAY DEBUG] AI content service failed to load: {e}")
    logger.warning(f"⚠️ [RAILWAY DEBUG] AI content service failed to load: {e}")

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

@app.get("/health/database")
async def health_database():
    """Check database connection health."""
    if not db_config:
        raise HTTPException(status_code=503, detail="Database configuration not available")
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version(), current_database(), current_user, now()")
            result = cur.fetchone()
            
            return {
                "status": "healthy",
                "database": result[1] if result else "unknown",
                "user": result[2] if result else "unknown",
                "timestamp": result[3].isoformat() if result and result[3] else None,
                "version": result[0][:50] if result and result[0] else "unknown",
                "service": "railway-simple"
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

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

# Campaign creation endpoint with full wizard data support
@app.post("/api/v2/campaigns/")
async def create_campaign(wizard_data: CampaignWizardData):
    """Create a new campaign with complete wizard data persistence."""
    try:
        if db_config:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Generate campaign ID
                import uuid
                campaign_id = str(uuid.uuid4())
                
                # Prepare metadata with ALL wizard data
                metadata = {
                    # Campaign Foundation
                    "campaign_name": wizard_data.campaign_name,
                    "primary_objective": wizard_data.primary_objective,
                    "campaign_purpose": wizard_data.campaign_purpose,
                    "target_market": wizard_data.target_market,
                    "company_context": wizard_data.company_context,
                    
                    # Strategy & Audience
                    "campaign_unique_angle": wizard_data.campaign_unique_angle,
                    "campaign_focus_message": wizard_data.campaign_focus_message,
                    "personas": [{"name": p.name, "description": p.description} for p in wizard_data.personas],
                    "key_messages": wizard_data.key_messages,
                    
                    # AI Content Planning - THE CRITICAL DATA!
                    "content_mix": {
                        "blog_posts": wizard_data.content_mix.blog_posts,
                        "social_media_posts": wizard_data.content_mix.social_media_posts,
                        "email_campaigns": wizard_data.content_mix.email_campaigns,
                        "total_pieces": wizard_data.content_mix.blog_posts + wizard_data.content_mix.social_media_posts + wizard_data.content_mix.email_campaigns
                    },
                    "content_themes": wizard_data.content_themes,
                    "content_tone": wizard_data.content_tone,
                    
                    # Distribution & Timeline
                    "distribution_channels": wizard_data.distribution_channels,
                    "timeline_weeks": wizard_data.timeline_weeks,
                    "publishing_frequency": wizard_data.publishing_frequency,
                    "budget_range": wizard_data.budget_range,
                    
                    # Automation Settings
                    "auto_generate_content": wizard_data.auto_generate_content,
                    "auto_schedule_publishing": wizard_data.auto_schedule_publishing,
                    "require_approval": wizard_data.require_approval,
                    
                    # System metadata
                    "created_via": "campaign_wizard",
                    "wizard_version": "1.0",
                    "created_at": datetime.now().isoformat()
                }
                
                logger.info(f"🎯 Creating campaign '{wizard_data.campaign_name}' with {metadata['content_mix']['total_pieces']} planned content pieces")
                
                # Insert campaign with full metadata
                cur.execute("""
                    INSERT INTO campaigns (id, name, status, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                    RETURNING id, name, status, created_at
                """, (campaign_id, wizard_data.campaign_name, "active", json.dumps(metadata)))
                
                new_campaign = cur.fetchone()
                conn.commit()
                
                # Auto-generate tasks if enabled
                total_tasks_generated = 0
                if wizard_data.auto_generate_content:
                    total_tasks_generated = await generate_tasks_from_wizard_data(
                        conn, campaign_id, wizard_data
                    )
                
                logger.info(f"✅ Campaign created: {wizard_data.campaign_name} with {total_tasks_generated} tasks generated")
                
                return {
                    "id": new_campaign[0],
                    "name": new_campaign[1],
                    "status": new_campaign[2],
                    "created_at": new_campaign[3].isoformat() if new_campaign[3] else None,
                    "content_pieces_planned": metadata["content_mix"]["total_pieces"],
                    "tasks_generated": total_tasks_generated,
                    "auto_generation_enabled": wizard_data.auto_generate_content,
                    "themes": wizard_data.content_themes,
                    "message": f"Campaign '{wizard_data.campaign_name}' created successfully with complete wizard data",
                    "service": "railway-simple"
                }
        
        # Fallback without database
        import uuid
        return {
            "id": str(uuid.uuid4()),
            "name": wizard_data.campaign_name,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "message": "Campaign created (no database) but wizard data received",
            "service": "railway-simple"
        }
        
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")

async def generate_tasks_from_wizard_data(conn, campaign_id: str, wizard_data: CampaignWizardData) -> int:
    """Generate tasks based on user's exact wizard specifications."""
    try:
        cur = conn.cursor()
        import uuid
        
        tasks_created = 0
        
        # Generate blog posts using user's themes
        for i in range(wizard_data.content_mix.blog_posts):
            theme = wizard_data.content_themes[i % len(wizard_data.content_themes)] if wizard_data.content_themes else f"Blog Topic {i+1}"
            task_id = str(uuid.uuid4())
            
            cur.execute("""
                INSERT INTO campaign_tasks (
                    id, campaign_id, task_type, target_format, 
                    target_asset, status, priority, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (task_id, campaign_id, "content_repurposing", "blog_post", theme, "pending", 3))
            tasks_created += 1
        
        # Generate social media posts
        for i in range(wizard_data.content_mix.social_media_posts):
            theme = wizard_data.content_themes[i % len(wizard_data.content_themes)] if wizard_data.content_themes else f"Social Topic {i+1}"
            platform = wizard_data.distribution_channels[i % len(wizard_data.distribution_channels)] if wizard_data.distribution_channels else "linkedin"
            if platform.lower() == "linkedin":
                platform = "linkedin_post"
            elif platform.lower() == "twitter":
                platform = "twitter_post"
            else:
                platform = "social_media_post"
                
            task_id = str(uuid.uuid4())
            
            cur.execute("""
                INSERT INTO campaign_tasks (
                    id, campaign_id, task_type, target_format, 
                    target_asset, status, priority, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (task_id, campaign_id, "content_repurposing", platform, theme, "pending", 2))
            tasks_created += 1
        
        # Generate email campaigns
        for i in range(wizard_data.content_mix.email_campaigns):
            theme = wizard_data.content_themes[i % len(wizard_data.content_themes)] if wizard_data.content_themes else f"Email Topic {i+1}"
            task_id = str(uuid.uuid4())
            
            cur.execute("""
                INSERT INTO campaign_tasks (
                    id, campaign_id, task_type, target_format, 
                    target_asset, status, priority, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (task_id, campaign_id, "content_repurposing", "email_sequence", theme, "pending", 2))
            tasks_created += 1
        
        logger.info(f"🎯 Generated {tasks_created} tasks from wizard data: {wizard_data.content_mix.blog_posts} blogs + {wizard_data.content_mix.social_media_posts} social + {wizard_data.content_mix.email_campaigns} emails")
        
        return tasks_created
        
    except Exception as e:
        logger.error(f"Error generating tasks from wizard data: {e}")
        return 0

# Add missing POST endpoint for AI recommendations
@app.post("/api/v2/campaigns/ai-recommendations")
async def get_ai_recommendations():
    """Get AI recommendations - simplified for Railway."""
    return {
        "recommendations": [
            {
                "id": "rec-1",
                "title": "Content Strategy Optimization",
                "description": "Optimize your content strategy for better engagement",
                "priority": "high",
                "type": "strategy"
            },
            {
                "id": "rec-2", 
                "title": "Social Media Amplification",
                "description": "Expand reach through targeted social media campaigns",
                "priority": "medium",
                "type": "distribution"
            }
        ],
        # Add the expected structure for the frontend
        "recommended_content_mix": {
            "blog_post": 3,
            "social_media_post": 5,
            "email_campaign": 2
        },
        "optimal_channels": ["linkedin", "email", "blog"],
        "recommended_posting_frequency": "weekly",
        "ai_reasoning": "Based on your campaign objectives, this content mix provides optimal engagement across professional channels.",
        "suggested_themes": [
            "Financial Technology Innovation",
            "Digital Transformation in Finance",
            "Partnership Opportunities"
        ],
        "generated_by": "railway-simple",
        "total": 2,
        "message": "AI recommendations (hardcoded)",
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

# Campaign Creation Models
class CampaignPersona(BaseModel):
    name: str = Field(..., max_length=200)
    description: str = Field(..., max_length=1000)

class CampaignContentMix(BaseModel):
    blog_posts: int = Field(default=3, ge=0, le=20)
    social_media_posts: int = Field(default=5, ge=0, le=50)
    email_campaigns: int = Field(default=2, ge=0, le=20)

class CampaignWizardData(BaseModel):
    # Campaign Foundation
    campaign_name: str = Field(..., max_length=200)
    primary_objective: str = Field(..., max_length=100)  
    campaign_purpose: str = Field(..., max_length=100)
    target_market: str = Field(..., max_length=200)
    company_context: str = Field(..., max_length=10000)
    
    # Strategy & Audience  
    campaign_unique_angle: str = Field(default="", max_length=5000)
    campaign_focus_message: str = Field(default="", max_length=5000)
    personas: List[CampaignPersona] = Field(default_factory=list)
    key_messages: List[str] = Field(default_factory=list)
    
    # AI Content Planning
    content_mix: CampaignContentMix = Field(default_factory=CampaignContentMix)
    content_themes: List[str] = Field(default_factory=list)
    content_tone: str = Field(default="professional", max_length=100)
    
    # Distribution & Timeline
    distribution_channels: List[str] = Field(default_factory=list)
    timeline_weeks: int = Field(default=4, ge=1, le=52)
    publishing_frequency: str = Field(default="weekly", max_length=50)
    budget_range: str = Field(default="$1,000-$5,000", max_length=100)
    
    # Automation Settings
    auto_generate_content: bool = Field(default=True)
    auto_schedule_publishing: bool = Field(default=False)
    require_approval: bool = Field(default=True)

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
# Add endpoint to generate campaign tasks (simplified)
@app.post("/api/v2/campaigns/{campaign_id}/generate-tasks")
async def generate_campaign_tasks(campaign_id: str):
    """Generate tasks for a campaign - simplified version."""
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Verify campaign exists
                cur.execute("SELECT name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
                campaign = cur.fetchone()
                if not campaign:
                    raise HTTPException(status_code=404, detail="Campaign not found")
                
                # Generate realistic AI-ready tasks for the campaign
                import uuid
                
                campaign_metadata = campaign[1] or {}
                
                tasks = [
                    {
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_id,
                        "task_type": "content_repurposing",  # Valid enum value
                        "target_format": "blog_post",
                        "target_asset": "Embedded Finance Solutions for Modern Businesses",
                        "status": "pending",
                        "priority": 3,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_id,
                        "task_type": "content_repurposing",  # Valid enum value
                        "target_format": "linkedin_post",
                        "target_asset": "Why Financial Partnerships Drive Innovation",
                        "status": "pending",
                        "priority": 2,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_id,
                        "task_type": "content_repurposing",  # Valid enum value
                        "target_format": "email_sequence",
                        "target_asset": "Strategic Partnership Opportunity in Embedded Finance",
                        "status": "pending",
                        "priority": 2,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_id,
                        "task_type": "content_repurposing",  # Valid enum value
                        "target_format": "twitter_post",
                        "target_asset": "The Future of Fintech Integration",
                        "status": "pending",
                        "priority": 1,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_id,
                        "task_type": "content_repurposing",  # Valid enum value
                        "target_format": "blog_post",
                        "target_asset": "Building Trust Through Financial Technology",
                        "status": "pending",
                        "priority": 1,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    }
                ]
                
                # Insert tasks into database
                for task in tasks:
                    cur.execute("""
                        INSERT INTO campaign_tasks (
                            id, campaign_id, task_type, target_format, 
                            target_asset, status, priority, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        task["id"], task["campaign_id"], task["task_type"],
                        task["target_format"], task["target_asset"], task["status"],
                        task["priority"], task["created_at"], task["updated_at"]
                    ))
                
                conn.commit()
                
                logger.info(f"🎯 Generated {len(tasks)} tasks for campaign {campaign_id}")
                
                return {
                    "campaign_id": campaign_id,
                    "tasks_created": len(tasks),
                    "tasks": tasks,
                    "message": "Tasks generated successfully (simple mode)",
                    "service": "railway-simple"
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating tasks: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate tasks: {str(e)}")
    
    raise HTTPException(status_code=503, detail="Database not available")

# Add agents endpoints for compatibility
@app.get("/api/v2/agents/")
async def list_agents():
    """List available agents - simplified version."""
    return [
        {
            "id": "agent_planner_001",
            "name": "Planning Agent", 
            "type": "planner",
            "status": "online",
            "capabilities": ["Content Strategy", "Campaign Planning"]
        },
        {
            "id": "agent_writer_002",
            "name": "Writer Agent",
            "type": "writer", 
            "status": "online",
            "capabilities": ["Blog Writing", "Content Creation"]
        },
        {
            "id": "agent_seo_003",
            "name": "SEO Agent",
            "type": "seo",
            "status": "online",
            "capabilities": ["Keyword Research", "Content Optimization"]
        }
    ]

@app.post("/api/v2/campaigns/{campaign_id}/rerun-agents")
async def rerun_campaign_agents(campaign_id: str):
    """Rerun agents for a campaign - REAL AI content generation."""
    logger.info(f"🔄 Rerun agents request for campaign {campaign_id}")
    
    if not db_config:
        logger.error("❌ Database config not available")
        raise HTTPException(status_code=503, detail="Database configuration not available")
    
    try:
        # Test database connection first
        with db_config.get_db_connection() as test_conn:
            test_cur = test_conn.cursor()
            test_cur.execute("SELECT 1")
            test_result = test_cur.fetchone()
            logger.info(f"✅ Database connection test successful: {test_result}")
    except Exception as db_test_error:
        logger.error(f"❌ Database connection test failed: {db_test_error}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(db_test_error)}")
    
    if db_config:
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Verify campaign exists and get context
                cur.execute("SELECT name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
                campaign = cur.fetchone()
                if not campaign:
                    raise HTTPException(status_code=404, detail="Campaign not found")
                
                campaign_name, metadata = campaign
                metadata = metadata or {}
                
                # Get pending tasks and debug existing tasks
                cur.execute("""
                    SELECT id, task_type, target_format, target_asset, status
                    FROM campaign_tasks 
                    WHERE campaign_id = %s
                """, (campaign_id,))
                
                all_tasks = cur.fetchall()
                logger.info(f"🔍 DEBUG: Found {len(all_tasks)} total tasks for campaign {campaign_id}")
                
                # Log task statuses
                for task in all_tasks:
                    logger.info(f"   Task {task[0]}: {task[1]} | {task[2]} | {task[3]} | Status: {task[4]}")
                
                # Get only pending tasks
                cur.execute("""
                    SELECT id, task_type, target_format, target_asset
                    FROM campaign_tasks 
                    WHERE campaign_id = %s AND status = 'pending'
                """, (campaign_id,))
                
                pending_tasks = cur.fetchall()
                updated_count = 0
                
                logger.info(f"🤖 Processing {len(pending_tasks)} PENDING tasks with REAL AI for campaign {campaign_id}")
                
                # If no pending tasks, reset all tasks to pending so they can be reprocessed
                if len(pending_tasks) == 0 and len(all_tasks) > 0:
                    logger.info("🔄 No pending tasks found, resetting all tasks to pending for reprocessing")
                    cur.execute("""
                        UPDATE campaign_tasks 
                        SET status = 'pending', 
                            result = NULL, 
                            error = NULL,
                            completed_at = NULL,
                            updated_at = NOW()
                        WHERE campaign_id = %s
                    """, (campaign_id,))
                    
                    # Get the newly reset tasks
                    cur.execute("""
                        SELECT id, task_type, target_format, target_asset
                        FROM campaign_tasks 
                        WHERE campaign_id = %s AND status = 'pending'
                    """, (campaign_id,))
                    
                    pending_tasks = cur.fetchall()
                    logger.info(f"🔄 Reset {len(pending_tasks)} tasks to pending status")
                
                # If still no tasks at all, create new ones
                elif len(all_tasks) == 0:
                    logger.info("🚀 No tasks exist for campaign, creating new tasks")
                    import uuid
                    
                    # Generate the FULL 10-piece content plan as originally requested
                    new_tasks = [
                        # Blog Posts (3 total as originally requested)
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "blog_post",
                            "target_asset": "Financial Technology Innovation in Embedded Finance",
                            "status": "pending",
                            "priority": 3,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "blog_post",
                            "target_asset": "Digital Transformation in Finance: A Complete Guide",
                            "status": "pending",
                            "priority": 3,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "blog_post",
                            "target_asset": "Partnership Opportunities in Modern Finance",
                            "status": "pending",
                            "priority": 3,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        # Social Media Posts (5 total as originally requested)
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "linkedin_post",
                            "target_asset": "Why Financial Partnerships Drive Innovation",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "linkedin_post",
                            "target_asset": "The Future of Embedded Finance Technology",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "linkedin_post",
                            "target_asset": "Digital Transformation Success Stories",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "linkedin_post",
                            "target_asset": "Building Strategic Partnerships in FinTech",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "linkedin_post",
                            "target_asset": "Embedded Finance: Transforming Customer Experience",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        # Email Campaigns (2 total as originally requested)
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "email_sequence",
                            "target_asset": "Strategic Partnership Opportunity in Embedded Finance",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        },
                        {
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_id,
                            "task_type": "content_repurposing",
                            "target_format": "email_sequence",
                            "target_asset": "Digital Finance Innovation: Join Our Partner Network",
                            "status": "pending",
                            "priority": 2,
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        }
                    ]
                    
                    # Insert new tasks
                    for task in new_tasks:
                        cur.execute("""
                            INSERT INTO campaign_tasks (
                                id, campaign_id, task_type, target_format, 
                                target_asset, status, priority, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            task["id"], task["campaign_id"], task["task_type"],
                            task["target_format"], task["target_asset"], task["status"],
                            task["priority"], task["created_at"], task["updated_at"]
                        ))
                    
                    # Get the new tasks
                    cur.execute("""
                        SELECT id, task_type, target_format, target_asset
                        FROM campaign_tasks 
                        WHERE campaign_id = %s AND status = 'pending'
                    """, (campaign_id,))
                    
                    pending_tasks = cur.fetchall()
                    logger.info(f"🆕 Created {len(pending_tasks)} new tasks for processing")
                
                # Generate real content for each task
                for task_id, task_type, target_format, target_asset in pending_tasks:
                    try:
                        # Generate real AI content
                        if ai_content_service and ai_content_service.is_available():
                            # Prepare campaign context for AI
                            campaign_context = {
                                "target_audience": metadata.get("target_audience", "financial services professionals"),
                                "company_context": metadata.get("company_context", "CrediLinq AI platform for embedded finance"),
                                "campaign_name": campaign_name
                            }
                            
                            # Generate content based on task type
                            content_result = await ai_content_service.generate_content_for_task(
                                task_type=task_type,
                                target_format=target_format,
                                target_asset=target_asset,
                                campaign_context=campaign_context
                            )
                            
                            # Format the result for database storage
                            if content_result.get("status") == "ai_generated":
                                # Store the full AI-generated content
                                if "title" in content_result:
                                    # Blog post format
                                    result_content = f"Title: {content_result['title']}\n\nSummary: {content_result.get('summary', '')}\n\nContent:\n{content_result['content']}"
                                elif "subject" in content_result:
                                    # Email format
                                    result_content = f"Subject: {content_result['subject']}\n\nContent:\n{content_result['content']}"
                                else:
                                    # General content
                                    result_content = content_result['content']
                                
                                result_content += f"\n\n[Generated by AI on {content_result.get('generated_at', datetime.now().isoformat())}]"
                            else:
                                # Fallback or error case
                                result_content = content_result.get("content", f"Error generating content for {target_asset}")
                        else:
                            # Fallback to enhanced mock if AI not available
                            result_content = f"Enhanced Mock Content for {target_asset}\n\nThis content would be generated by AI agents for {target_format} format.\nTarget: {target_asset}\nType: {task_type}\n\nKey Benefits:\n• Advanced embedded finance solutions\n• Seamless API integration\n• Partnership opportunities\n• Lead generation focus\n\n[Note: Real AI generation requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable]"
                        
                        # Update task with generated content
                        cur.execute("""
                            UPDATE campaign_tasks 
                            SET result = %s,
                                status = 'completed',
                                completed_at = NOW(),
                                updated_at = NOW()
                            WHERE id = %s
                        """, (result_content, task_id))
                        
                        updated_count += 1
                        logger.info(f"✅ Generated content for task {task_id}: {target_asset}")
                        
                    except Exception as task_error:
                        logger.error(f"Error generating content for task {task_id}: {task_error}")
                        # Mark task as failed with error info (using 'pending' since 'failed' is not a valid enum)
                        cur.execute("""
                            UPDATE campaign_tasks 
                            SET result = %s,
                                status = 'pending',
                                error = %s,
                                updated_at = NOW()
                            WHERE id = %s
                        """, (f"Content generation failed: {str(task_error)}", str(task_error), task_id))
                
                conn.commit()
                
                ai_status = "Real AI" if (ai_content_service and ai_content_service.is_available()) else "Enhanced Mock"
                logger.info(f"🎯 {ai_status} content generated for campaign {campaign_id}, updated {updated_count} tasks")
                
                return {
                    "campaign_id": campaign_id,
                    "tasks_updated": updated_count,
                    "status": "success",
                    "message": f"{ai_status} agents completed {updated_count} tasks successfully",
                    "ai_enabled": ai_content_service.is_available() if ai_content_service else False,
                    "service": "railway-simple-ai"
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error rerunning agents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to rerun agents: {str(e)}")
    
    raise HTTPException(status_code=503, detail="Database not available")

@app.post("/api/v2/campaigns/{campaign_id}/execute")
async def execute_campaign(campaign_id: str):
    """Execute campaign tasks - simplified version."""
    # Reuse the rerun-agents logic
    return await rerun_campaign_agents(campaign_id)

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