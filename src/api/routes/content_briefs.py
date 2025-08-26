"""
Content Brief API Routes
Strategic content brief generation with SEO research and competitive analysis.
"""

import logging
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.agents.specialized.content_brief_agent import (
    ContentBriefAgent, 
    ContentBrief, 
    ContentType, 
    ContentPurpose,
    SEOKeyword,
    CompetitorInsight,
    ContentStructure
)
from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content-briefs", tags=["content-briefs"])

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Pydantic models for API requests/responses
def sanitize_text_input(text: str) -> str:
    """Sanitize text input by removing potentially harmful characters and excess whitespace."""
    if not text:
        return text
    
    # Remove potentially harmful HTML/XML tags and script content
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    # Remove SQL injection patterns (basic protection)
    text = re.sub(r'(union|select|insert|update|delete|drop|create|alter)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\'";]', '', text)
    
    # Clean up whitespace and normalize
    text = ' '.join(text.split())
    
    return text.strip()

class ContentBriefRequest(BaseModel):
    """Request model for creating a content brief with enhanced validation and security."""
    topic: str = Field(..., min_length=5, max_length=200, description="Content topic/subject", example="Embedded Finance for B2B Marketplaces")
    content_type: str = Field(default="blog_post", description="Type of content", example="blog_post")
    primary_purpose: str = Field(default="lead_generation", description="Primary content purpose", example="lead_generation")
    target_audience: str = Field(default="B2B finance professionals", max_length=500, description="Target audience", example="CFOs and Finance Directors")
    company_context: Optional[str] = Field(None, max_length=2000, description="Company context and positioning", example="Leading fintech API platform")
    competitive_focus: Optional[str] = Field(None, max_length=1000, description="Specific competitive focus", example="Traditional banking solutions")
    distribution_channels: List[str] = Field(default_factory=lambda: ["website", "linkedin", "email"], max_items=20, description="Content distribution channels")
    brand_guidelines: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Brand guidelines and requirements")
    success_metrics: Optional[List[str]] = Field(default_factory=list, max_items=10, description="Custom success metrics")
    
    @validator('topic', 'target_audience', 'company_context', 'competitive_focus', pre=True)
    def sanitize_text_fields(cls, v):
        """Sanitize text fields for security."""
        if isinstance(v, str):
            return sanitize_text_input(v)
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type is from allowed set."""
        allowed_types = {"blog_post", "linkedin_article", "whitepaper", "case_study", "newsletter"}
        if v not in allowed_types:
            raise ValueError(f"content_type must be one of: {', '.join(allowed_types)}")
        return v
    
    @validator('primary_purpose')
    def validate_primary_purpose(cls, v):
        """Validate primary purpose is from allowed set."""
        allowed_purposes = {"lead_generation", "thought_leadership", "brand_awareness", "seo", "conversion"}
        if v not in allowed_purposes:
            raise ValueError(f"primary_purpose must be one of: {', '.join(allowed_purposes)}")
        return v
    
    @validator('distribution_channels')
    def validate_distribution_channels(cls, v):
        """Validate and sanitize distribution channels."""
        if not v:
            return ["website", "linkedin", "email"]
        
        allowed_channels = {"website", "linkedin", "twitter", "facebook", "email", "newsletter", "blog", "press_release"}
        validated_channels = []
        
        for channel in v[:20]:  # Limit to 20 channels
            channel = sanitize_text_input(str(channel).lower())
            if channel in allowed_channels:
                validated_channels.append(channel)
        
        return validated_channels or ["website", "linkedin", "email"]
    
    @validator('success_metrics')
    def validate_success_metrics(cls, v):
        """Validate and sanitize success metrics."""
        if not v:
            return []
        
        sanitized_metrics = []
        for metric in v[:10]:  # Limit to 10 metrics
            if isinstance(metric, str):
                clean_metric = sanitize_text_input(metric)
                if clean_metric and len(clean_metric.strip()) >= 3:
                    sanitized_metrics.append(clean_metric)
        
        return sanitized_metrics

class ContentBriefResponse(BaseModel):
    """Response model for content brief creation."""
    brief_id: str
    title: str
    content_type: str
    primary_purpose: str
    marketing_objective: str
    target_audience: str
    primary_keyword: Dict[str, Any]
    secondary_keywords: List[Dict[str, Any]]
    content_structure: Dict[str, Any]
    success_kpis: List[str]
    estimated_creation_time: str
    created_at: datetime
    summary: str

class BriefSummaryResponse(BaseModel):
    """Response model for brief summary."""
    brief_id: str
    title: str
    summary: str
    key_metrics: Dict[str, Any]
    next_steps: List[str]

# Initialize Content Brief Agent
try:
    content_brief_agent = ContentBriefAgent()
    logger.info("Content Brief Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Content Brief Agent: {str(e)}")
    content_brief_agent = None

@router.post("/create", response_model=ContentBriefResponse)
async def create_content_brief(
    brief_request: ContentBriefRequest,
    background_tasks: BackgroundTasks,
    request: Request
) -> ContentBriefResponse:
    """
    Create a comprehensive strategic content brief with SEO research and competitive analysis.
    
    This endpoint generates a detailed content brief that includes:
    - SEO keyword research with search volume and difficulty analysis
    - Competitive landscape analysis and differentiation opportunities
    - Strategic content structure and outline recommendations
    - Success metrics and KPIs aligned with business objectives
    """
    
    if not content_brief_agent:
        raise HTTPException(
            status_code=503,
            detail="Content Brief Agent is not available. Please try again later."
        )
    
    try:
        # Security validation: Check request size and rate limits
        client_ip = request.client.host if request.client else "unknown"
        
        # Basic request size validation
        request_size = len(str(brief_request.model_dump()))
        if request_size > 10000:  # 10KB limit
            logger.warning(f"Request size too large from {client_ip}: {request_size} bytes")
            raise HTTPException(
                status_code=413,
                detail="Request payload too large. Maximum size is 10KB."
            )
        
        # Additional validation for potentially suspicious patterns  
        topic_lower = brief_request.topic.lower()
        suspicious_keywords = ["script", "eval", "exec", "system", "shell", "cmd", "powershell", "rm -rf", "sudo", "chmod", "kill"]
        if any(keyword in topic_lower for keyword in suspicious_keywords):
            logger.warning(f"Suspicious topic detected from {client_ip}: {brief_request.topic}")
            raise HTTPException(
                status_code=400,
                detail="Invalid topic content detected. Please use business-appropriate topics."
            )
        
        logger.info(f"Creating content brief for topic: {brief_request.topic} (client: {client_ip})")
        
        # Convert request to agent format
        brief_request_dict = {
            "topic": brief_request.topic,
            "content_type": brief_request.content_type,
            "primary_purpose": brief_request.primary_purpose,
            "target_audience": brief_request.target_audience,
            "company_context": brief_request.company_context or "",
            "competitive_focus": brief_request.competitive_focus or "",
            "distribution_channels": brief_request.distribution_channels,
            "brand_guidelines": brief_request.brand_guidelines,
            "success_metrics": brief_request.success_metrics
        }
        
        # Generate comprehensive content brief
        content_brief = await content_brief_agent.create_content_brief(brief_request_dict)
        
        # Generate executive summary
        brief_summary = await content_brief_agent.generate_brief_summary(content_brief)
        
        # Store the brief in the database (background task)
        background_tasks.add_task(
            _store_content_brief,
            content_brief.brief_id,
            content_brief.model_dump(),
            brief_summary
        )
        
        # Convert to response format
        response = ContentBriefResponse(
            brief_id=content_brief.brief_id,
            title=content_brief.title,
            content_type=content_brief.content_type.value,
            primary_purpose=content_brief.primary_purpose.value,
            marketing_objective=content_brief.marketing_objective,
            target_audience=content_brief.target_audience,
            primary_keyword={
                "keyword": content_brief.primary_keyword.keyword,
                "search_volume": content_brief.primary_keyword.search_volume,
                "difficulty": content_brief.primary_keyword.difficulty.value,
                "intent": content_brief.primary_keyword.intent
            },
            secondary_keywords=[
                {
                    "keyword": kw.keyword,
                    "search_volume": kw.search_volume,
                    "difficulty": kw.difficulty.value,
                    "intent": kw.intent
                }
                for kw in content_brief.secondary_keywords
            ],
            content_structure={
                "estimated_word_count": content_brief.content_structure.estimated_word_count,
                "suggested_headlines": content_brief.content_structure.suggested_headlines,
                "content_outline": content_brief.content_structure.content_outline,
                "call_to_actions": content_brief.content_structure.call_to_actions
            },
            success_kpis=content_brief.success_kpis,
            estimated_creation_time=content_brief.estimated_creation_time,
            created_at=content_brief.created_at,
            summary=brief_summary
        )
        
        logger.info(f"Content brief created successfully: {content_brief.brief_id}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error creating content brief: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation Error",
                "message": "Invalid input data provided",
                "details": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating content brief: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while creating the content brief",
                "request_id": str(uuid.uuid4())[:8]
            }
        )

@router.get("/", response_model=List[Dict[str, Any]])
async def list_content_briefs(
    limit: int = Query(default=20, le=100, description="Number of briefs to return"),
    offset: int = Query(default=0, description="Number of briefs to skip"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    purpose: Optional[str] = Query(None, description="Filter by primary purpose")
) -> List[Dict[str, Any]]:
    """
    List all content briefs with optional filtering.
    
    Returns a paginated list of content briefs with summary information.
    """
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Build query with filters
            query = """
                SELECT id, title, content_type, primary_purpose, target_audience,
                       marketing_objective, estimated_creation_time, created_at
                FROM content_briefs 
                WHERE 1=1
            """
            params = []
            
            if content_type:
                query += " AND content_type = %s"
                params.append(content_type)
            
            if purpose:
                query += " AND primary_purpose = %s"
                params.append(purpose)
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            briefs = cur.fetchall()
            
            # Convert to response format
            brief_list = []
            for brief in briefs:
                brief_list.append({
                    "brief_id": brief[0],
                    "id": brief[0],
                    "title": brief[1],
                    "content_type": brief[2],
                    "primary_purpose": brief[3],
                    "target_audience": brief[4],
                    "marketing_objective": brief[5],
                    "estimated_creation_time": brief[6],
                    "created_at": brief[7].isoformat() if brief[7] else None
                })
            
            return brief_list
            
    except Exception as e:
        logger.error(f"Failed to list content briefs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve content briefs: {str(e)}"
        )

@router.get("/{brief_id}", response_model=Dict[str, Any])
async def get_content_brief(
    brief_id: str = Path(..., description="Content brief ID")
) -> Dict[str, Any]:
    """
    Get a specific content brief by ID.
    
    Returns the complete content brief including all strategic details,
    SEO research, competitive analysis, and content structure.
    """
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute(
                "SELECT brief_data, summary FROM content_briefs WHERE id = %s",
                (brief_id,)
            )
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Content brief {brief_id} not found"
                )
            
            brief_data, summary = result
            
            # Parse the stored JSON data
            if isinstance(brief_data, str):
                brief_data = json.loads(brief_data)
            
            # Add summary to response
            brief_data["summary"] = summary
            
            return brief_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content brief {brief_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve content brief: {str(e)}"
        )

@router.get("/{brief_id}/summary", response_model=BriefSummaryResponse)
async def get_brief_summary(
    brief_id: str = Path(..., description="Content brief ID")
) -> BriefSummaryResponse:
    """
    Get an executive summary of a content brief.
    
    Returns a concise summary suitable for executive review and approval.
    """
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT title, summary, brief_data 
                FROM content_briefs 
                WHERE id = %s
            """, (brief_id,))
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Content brief {brief_id} not found"
                )
            
            title, summary, brief_data = result
            
            # Parse brief data for key metrics
            if isinstance(brief_data, str):
                brief_data = json.loads(brief_data)
            
            key_metrics = {
                "primary_keyword": brief_data.get("primary_keyword", {}).get("keyword", ""),
                "estimated_word_count": brief_data.get("content_structure", {}).get("estimated_word_count", 0),
                "estimated_creation_time": brief_data.get("estimated_creation_time", ""),
                "target_audience": brief_data.get("target_audience", "")
            }
            
            next_steps = [
                "Review and approve content brief",
                "Assign to content creator",
                "Begin content development according to structure",
                "Schedule content for publication",
                "Set up success metric tracking"
            ]
            
            return BriefSummaryResponse(
                brief_id=brief_id,
                title=title,
                summary=summary,
                key_metrics=key_metrics,
                next_steps=next_steps
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get brief summary {brief_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve brief summary: {str(e)}"
        )

@router.put("/{brief_id}/approve")
async def approve_content_brief(
    brief_id: str = Path(..., description="Content brief ID"),
    approval_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Approve a content brief and mark it ready for content creation.
    
    This moves the brief from "draft" to "approved" status and can trigger
    automatic assignment to content creators.
    """
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Update brief status to approved
            cur.execute("""
                UPDATE content_briefs 
                SET status = 'approved', 
                    approval_notes = %s, 
                    approved_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                RETURNING title, content_type
            """, (approval_notes, brief_id))
            
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Content brief {brief_id} not found"
                )
            
            title, content_type = result
            conn.commit()
            
            return {
                "brief_id": brief_id,
                "title": title,
                "content_type": content_type,
                "status": "approved",
                "approved_at": datetime.now().isoformat(),
                "approval_notes": approval_notes,
                "message": "Content brief approved successfully",
                "next_steps": [
                    "Brief is ready for content creation",
                    "Content creator will be notified",
                    "Development can begin according to structure",
                    "Success metrics tracking is active"
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve content brief {brief_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to approve content brief: {str(e)}"
        )

@router.delete("/{brief_id}")
async def delete_content_brief(
    brief_id: str = Path(..., description="Content brief ID")
) -> Dict[str, str]:
    """
    Delete a content brief.
    
    This permanently removes the content brief from the system.
    Only draft briefs can be deleted.
    """
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if brief exists and is in draft status
            cur.execute(
                "SELECT status FROM content_briefs WHERE id = %s",
                (brief_id,)
            )
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Content brief {brief_id} not found"
                )
            
            status = result[0]
            if status != "draft":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot delete {status} content brief. Only draft briefs can be deleted."
                )
            
            # Delete the brief
            cur.execute(
                "DELETE FROM content_briefs WHERE id = %s",
                (brief_id,)
            )
            conn.commit()
            
            return {
                "brief_id": brief_id,
                "message": "Content brief deleted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete content brief {brief_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete content brief: {str(e)}"
        )

@router.get("/types/available")
async def get_available_content_types() -> Dict[str, Any]:
    """
    Get available content types and purposes for brief creation.
    
    Returns the available options for content type and primary purpose
    to help with brief creation forms.
    """
    
    content_types = [
        {"value": "blog_post", "label": "Blog Post", "description": "Long-form educational content"},
        {"value": "linkedin_article", "label": "LinkedIn Article", "description": "Professional platform article"},
        {"value": "white_paper", "label": "White Paper", "description": "In-depth research document"},
        {"value": "case_study", "label": "Case Study", "description": "Customer success story"},
        {"value": "guide", "label": "Guide", "description": "How-to and implementation guide"},
        {"value": "tutorial", "label": "Tutorial", "description": "Step-by-step instructional content"}
    ]
    
    content_purposes = [
        {"value": "lead_generation", "label": "Lead Generation", "description": "Drive qualified leads"},
        {"value": "brand_awareness", "label": "Brand Awareness", "description": "Increase market visibility"},
        {"value": "thought_leadership", "label": "Thought Leadership", "description": "Establish industry expertise"},
        {"value": "customer_education", "label": "Customer Education", "description": "Educate existing customers"},
        {"value": "seo_ranking", "label": "SEO Ranking", "description": "Improve search visibility"},
        {"value": "competitor_response", "label": "Competitor Response", "description": "Respond to competitive threats"}
    ]
    
    return {
        "content_types": content_types,
        "content_purposes": content_purposes,
        "default_audience_segments": [
            "B2B finance professionals",
            "CFOs and Finance Directors", 
            "Fintech entrepreneurs",
            "Platform business owners",
            "Financial service providers"
        ]
    }

# Background task for storing content briefs
async def _store_content_brief(brief_id: str, brief_data: Dict[str, Any], summary: str) -> None:
    """Store content brief in database (background task)."""
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Insert the content brief
            cur.execute("""
                INSERT INTO content_briefs (
                    id, title, content_type, primary_purpose, target_audience,
                    company_context, brief_data, summary
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    brief_data = EXCLUDED.brief_data,
                    summary = EXCLUDED.summary,
                    updated_at = NOW()
            """, (
                brief_id,
                brief_data.get("title", ""),
                brief_data.get("content_type", ""),
                brief_data.get("primary_purpose", ""),
                brief_data.get("target_audience", ""),
                brief_data.get("business_context", ""),
                json.dumps(brief_data, cls=DateTimeEncoder),
                summary
            ))
            
            conn.commit()
            logger.info(f"Content brief {brief_id} stored successfully")
            
    except Exception as e:
        logger.error(f"Failed to store content brief {brief_id}: {str(e)}")