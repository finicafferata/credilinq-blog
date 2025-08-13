"""Blog management endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List
import json
import uuid
import datetime
import time
import logging

logger = logging.getLogger(__name__)

from ...config.database import db_config
import psycopg2.extras
from ...core.security import validate_api_input, InputValidator
from ...core.exceptions import (
    convert_to_http_exception, InputValidationError, SecurityException
)
from ..models.blog import (
    BlogCreateRequest, BlogEditRequest, BlogReviseRequest, 
    BlogSearchRequest, BlogSummary, BlogDetail
)
from ...services.ai_content_analyzer import generate_review_suggestions
from .comments import add_comment as add_comment_api
from .suggestions import add_suggestion as add_suggestion_api

router = APIRouter()


@router.get("/test")
def test_endpoint():
    """Basic test endpoint."""
    return {"message": "Test endpoint working"}


@router.post("/blogs", response_model=BlogSummary)
def create_blog(request: BlogCreateRequest):
    """Generate a new blog post using the multi-agent workflow with comprehensive input validation."""
    
    logger.info(f"Received blog creation request: {request}")
    
    try:
        # Validate and sanitize input data
        validated_title = request.title.strip()
        if len(validated_title) < 1 or len(validated_title) > 200:
            raise HTTPException(status_code=400, detail="Title must be between 1 and 200 characters")
        
        validated_context = InputValidator.validate_content_text(
            request.company_context, "company_context", max_length=1000
        )
        validated_content_type = InputValidator.validate_string_input(
            request.content_type, "content_type", max_length=50
        )
        
        # Additional validation for content_type
        allowed_content_types = ['blog', 'linkedin', 'article', 'social']
        if validated_content_type.lower() not in allowed_content_types:
            raise HTTPException(
                status_code=400,
                detail=f"content_type must be one of: {', '.join(allowed_content_types)}"
            )
        
    except SecurityException as e:
        logger.error(f"Validation error: {str(e)}")
        raise convert_to_http_exception(e)
    except Exception as e:
        logger.error(f"Unexpected error in create_blog: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")
    
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from ...agents.workflow.structured_blog_workflow import BlogWorkflowCompatibility
        from ...agents.core.database_service import get_db_service, AgentDecision, AgentPerformanceMetrics, BlogAnalyticsData
        
        # Get database service instance
        db_service = get_db_service()
        
        agent_input = {
            "title": validated_title,
            "company_context": validated_context,
            "content_type": validated_content_type
        }
        
        # Create execution ID for tracking
        execution_id = str(uuid.uuid4())
        
        # Use individual agents directly to avoid database operations
        from ...agents.specialized.planner_agent import PlannerAgent
        from ...agents.specialized.researcher_agent import ResearcherAgent
        from ...agents.specialized.writer_agent import WriterAgent
        from ...agents.core.base_agent import AgentExecutionContext
        
        context = AgentExecutionContext(workflow_id=str(uuid.uuid4()))
        
        try:
            # Step 1: Create outline
            planner_input = {
                "blog_title": agent_input["title"],
                "company_context": agent_input["company_context"],
                "content_type": agent_input["content_type"]
            }
            planner = PlannerAgent()
            outline_result = planner.execute_safe(planner_input, context)
            
            if not outline_result.success:
                raise Exception(f"Planning failed: {outline_result.error_message}")
            
            # Step 2: Research
            researcher = ResearcherAgent()
            research_input = {
                "outline": outline_result.data["outline"],
                "blog_title": agent_input["title"],
                "company_context": agent_input["company_context"]
            }
            research_result = researcher.execute_safe(research_input, context)
            
            if not research_result.success:
                raise Exception(f"Research failed: {research_result.error_message}")
            
            # Step 3: Write content
            writer = WriterAgent()
            writing_input = {
                "outline": outline_result.data["outline"],
                "research": research_result.data["research"],
                "blog_title": agent_input["title"],
                "company_context": agent_input["company_context"],
                "content_type": agent_input["content_type"]
            }
            writing_result = writer.execute_safe(writing_input, context)
            
            if not writing_result.success:
                raise Exception(f"Writing failed: {writing_result.error_message}")
            
            final_post = writing_result.data["content"]
            result = {"final_post": final_post, "success": True}
            
        except Exception as e:
            result = {"final_post": f"Content generation error: {str(e)}", "success": False}
        final_post = result.get("final_post", "")
        initial_prompt = json.dumps(agent_input)
        blog_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        updated_at = datetime.datetime.utcnow().isoformat()
        
        # Usar conexión directa de PostgreSQL en lugar de Supabase API
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Calculate additional metadata
                    word_count = len(final_post.split()) if final_post else 0
                    reading_time = max(1, word_count // 200)  # Average reading speed
                    
                    # Use snake_case schema (optimized)
                    cur.execute(
                        """
                        INSERT INTO blog_posts (
                            id, title, content_markdown, initial_prompt, status, 
                            word_count, reading_time, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            blog_id,
                            validated_title,
                            final_post,
                            initial_prompt,
                            "draft",
                            word_count,
                            reading_time,
                            created_at,
                            updated_at,
                        ),
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Error creating blog in database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # Log successful creation
        execution_time = int((time.time() - start_time) * 1000)
        
        try:
            # TODO: Re-implement performance logging with new schema
            # For now, just log basic analytics
            analytics = BlogAnalyticsData(
                blog_id=blog_id,
                word_count=len(final_post.split()),
                reading_time=max(1, len(final_post.split()) // 200),
                content_type=validated_content_type,
                creation_time_ms=execution_time
            )
            # db_service.log_blog_analytics(analytics)  # Temporarily disabled
            logger.info(f"Blog created successfully: {blog_id}")
        except Exception as e:
            logger.warning(f"Failed to log analytics data: {str(e)}")
            # Don't fail blog creation if logging fails
        
        return BlogSummary(
            id=blog_id,
            title=validated_title,
            status="draft",
            created_at=created_at,
            word_count=word_count,
            reading_time=reading_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in blog creation workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blog creation failed: {str(e)}")


@router.get("/blogs/test")
def test_blogs():
    """Test endpoint to debug blog listing."""
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, title, status, created_at FROM blog_posts LIMIT 3")
            rows = cur.fetchall()
            
            result = []
            for i, row in enumerate(rows):
                result.append({
                    "row_index": i,
                    "raw_row": str(row),
                    "id": str(row[0]) if row[0] else None,
                    "title": str(row[1]) if row[1] else None,
                    "status": str(row[2]) if row[2] else None,
                    "created_at": str(row[3]) if row[3] else None
                })
            
            return {"rows": result, "total": len(rows)}
    except Exception as e:
        return {"error": str(e)}


@router.get("/blogs/simple")
def simple_blogs():
    """Very simple test endpoint."""
    try:
        return {
            "message": "Simple endpoint working",
            "test_data": [
                {"id": "test-1", "title": "Test Blog 1", "status": "draft", "created_at": "2025-07-30T15:30:00Z"},
                {"id": "test-2", "title": "Test Blog 2", "status": "published", "created_at": "2025-07-30T15:31:00Z"}
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/blogs/{post_id}/review/ai")
def run_ai_review(post_id: str):
    """Run a simple AI review on a blog and store comments/suggestions."""
    try:
        # Fetch blog content
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, content_markdown FROM blog_posts WHERE id = %s
                    """,
                    (post_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                content = row["content_markdown"] or ""

        results = generate_review_suggestions(content)

        # Persist comments
        created_comments = []
        for c in results.get("comments", []):
            created_comments.append(
                add_comment_api(post_id, type("obj", (), {
                    "author": c.get("author") or "AI Assistant",
                    "content": c["content"],
                    "position": c.get("position")
                }))
            )

        # Persist suggestions
        created_suggestions = []
        for s in results.get("suggestions", []):
            created_suggestions.append(
                add_suggestion_api(post_id, type("obj", (), {
                    "author": s.get("author") or "AI Assistant",
                    "originalText": s["originalText"],
                    "suggestedText": s["suggestedText"],
                    "reason": s.get("reason", ""),
                    "position": type("pos", (), {"start": s["position"]["start"], "end": s["position"]["end"]})
                }))
            )

        return {
            "created_comments": created_comments,
            "created_suggestions": created_suggestions,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI review error for blog {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blogs", response_model=List[BlogSummary])
def list_blogs():
    """List all blog posts stored in database (excluding deleted posts)."""
    try:
        # Use direct connection without db_config
        import psycopg2
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
        
        with psycopg2.connect(database_url) as conn:
            cur = conn.cursor()
            # Use snake_case schema (optimized)
            cur.execute("""
                SELECT 
                    id, 
                    title, 
                    status, 
                    created_at,
                    word_count,
                    reading_time,
                    seo_score,
                    published_at
                FROM blog_posts 
                WHERE status != 'deleted' 
                ORDER BY created_at DESC
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
                
                # Handle date - convert to ISO string or use current time if null
                created_at = row[3]
                if created_at:
                    if hasattr(created_at, 'isoformat'):
                        created_at_str = created_at.isoformat()
                    else:
                        created_at_str = str(created_at)
                else:
                    # If no date, use current time
                    import datetime
                    created_at_str = datetime.datetime.utcnow().isoformat()
                
                # Extract additional fields from database
                word_count = row[4] if len(row) > 4 else None
                reading_time = row[5] if len(row) > 5 else None
                seo_score = row[6] if len(row) > 6 else None
                published_at = row[7] if len(row) > 7 else None
                
                # Format published_at if present
                published_at_str = None
                if published_at:
                    if hasattr(published_at, 'isoformat'):
                        published_at_str = published_at.isoformat()
                    else:
                        published_at_str = str(published_at)
                
                blog_summary = BlogSummary(
                    id=blog_id,
                    title=title,
                    status=status,
                    created_at=created_at_str,
                    word_count=word_count,
                    reading_time=reading_time,
                    seo_score=seo_score,
                    published_at=published_at_str
                )
                
                logger.info(f"Created blog summary: {blog_summary}")
                blogs.append(blog_summary)
            
            logger.info(f"Returning {len(blogs)} blogs")
            return blogs
    except Exception as e:
        logger.error(f"Error listing blogs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/blogs/{post_id}", response_model=BlogDetail)
def get_blog(post_id: str):
    """Retrieve a single blog post by ID with input validation."""
    try:
        # Validate UUID format
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Use snake_case schema (optimized)
                cur.execute(
                    """
                    SELECT 
                        id, 
                        title, 
                        status, 
                        created_at, 
                        updated_at,
                        published_at,
                        content_markdown, 
                        initial_prompt,
                        word_count,
                        reading_time,
                        seo_score,
                        geo_optimized,
                        geo_score,
                        geo_metadata
                    FROM blog_posts 
                    WHERE id = %s
                    """,
                    (validated_id,),
                )
                row = cur.fetchone()
                
                if not row:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                # Dict access
                blog_id = str(row["id"]) if row["id"] else ''
                title = str(row["title"]) if row["title"] else 'Untitled'
                status = str(row["status"]) if row["status"] else 'draft'
                created_at = row["created_at"]
                content_markdown = str(row["content_markdown"]) if row["content_markdown"] else ''
                initial_prompt_raw = row.get("initial_prompt")
            
            # Handle initial_prompt - it might already be a dict or a JSON string
            initial_prompt = initial_prompt_raw
            if initial_prompt:
                if isinstance(initial_prompt, str):
                    initial_prompt = initial_prompt.strip()
                    if initial_prompt:
                        try:
                            initial_prompt = json.loads(initial_prompt)
                        except Exception:
                            initial_prompt = {}
                    else:
                        initial_prompt = {}
                elif not isinstance(initial_prompt, dict):
                    initial_prompt = {}
            else:
                initial_prompt = {}
            
            # Extract additional fields
            updated_at = row.get("updated_at")
            published_at = row.get("published_at")
            word_count = row.get("word_count")
            reading_time = row.get("reading_time")
            seo_score = row.get("seo_score")
            geo_optimized = row.get("geo_optimized", False)
            geo_score = row.get("geo_score")
            geo_metadata = row.get("geo_metadata")
            
            # Handle geo_metadata - it might be JSON string or dict
            if geo_metadata and isinstance(geo_metadata, str):
                try:
                    geo_metadata = json.loads(geo_metadata)
                except Exception:
                    geo_metadata = None
            
            return BlogDetail(
                id=blog_id,
                title=title,
                status=status,
                created_at=str(created_at) if created_at else datetime.datetime.utcnow().isoformat(),
                updated_at=str(updated_at) if updated_at else None,
                published_at=str(published_at) if published_at else None,
                content_markdown=content_markdown,
                initial_prompt=initial_prompt,
                word_count=word_count,
                reading_time=reading_time,
                seo_score=seo_score,
                geo_optimized=geo_optimized,
                geo_score=geo_score,
                geo_metadata=geo_metadata
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.put("/blogs/{post_id}", response_model=BlogDetail)
def edit_blog(post_id: str, request: BlogEditRequest):
    """Manually edit the content of a blog post with input validation."""
    try:
        # Validate inputs
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
        # Use relaxed validator for natural-language markdown
        validated_content = InputValidator.validate_content_text(
            request.content_markdown, "content_markdown", max_length=50000
        )
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            # Calculate updated metadata
            word_count = len(validated_content.split()) if validated_content else 0
            reading_time = max(1, word_count // 200)
            
            # Use snake_case schema (optimized)
            cur.execute("""
                UPDATE blog_posts 
                SET content_markdown = %s, status = 'edited', 
                    word_count = %s, reading_time = %s, updated_at = NOW()
                WHERE id = %s
            """, (validated_content, word_count, reading_time, validated_id))
            
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Blog post not found")
            conn.commit()
        
        return get_blog(post_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.delete("/blogs/{post_id}")
def delete_blog(post_id: str):
    """Delete a blog post by ID (soft delete) with input validation."""
    try:
        # Validate UUID format
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if blog exists
            cur.execute("SELECT id, title FROM blog_posts WHERE id = %s", (validated_id,))
            existing_blog = cur.fetchone()
            if not existing_blog:
                raise HTTPException(status_code=404, detail="Blog post not found")
            
            # Soft delete
            cur.execute("""
                UPDATE blog_posts 
                SET status = 'deleted'
                WHERE id = %s
            """, (validated_id,))
            conn.commit()
        
        return {"message": "Blog post deleted successfully", "id": validated_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/blogs/{post_id}/publish", response_model=BlogDetail)
def publish_blog(post_id: str):
    """Publish a blog post by changing its status to 'published' with input validation."""
    try:
        # Validate UUID format
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Check if blog exists and get current status
                cur.execute("SELECT id, title, status FROM blog_posts WHERE id = %s", (validated_id,))
                existing_blog = cur.fetchone()
                if not existing_blog:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                current_status = existing_blog["status"].lower()
                if current_status not in ["draft", "edited"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot publish blog with status '{current_status}'. Only 'draft' or 'edited' posts can be published."
                    )
                
                # Update status to published with published_at timestamp
                cur.execute(
                    """
                    UPDATE blog_posts 
                    SET status = 'published', published_at = NOW(), updated_at = NOW()
                    WHERE id = %s
                    """,
                    (validated_id,),
                )
                conn.commit()
        
        return get_blog(validated_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/blogs/{post_id}/metadata", response_model=dict)
def get_blog_metadata(post_id: str):
    """Get enhanced metadata for a blog post including SEO and content metrics."""
    try:
        # Validate UUID format
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT 
                        id, title, word_count, reading_time, seo_score,
                        geo_optimized, geo_score, geo_metadata,
                        created_at, updated_at, published_at,
                        content_markdown
                    FROM blog_posts 
                    WHERE id = %s
                    """,
                    (validated_id,),
                )
                row = cur.fetchone()
                
                if not row:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                # Calculate additional metrics
                content = row["content_markdown"] or ""
                word_count = len(content.split()) if content else 0
                char_count = len(content)
                reading_time = max(1, word_count // 200)
                
                # Basic readability score (simplified)
                sentences = content.count('.') + content.count('!') + content.count('?')
                avg_words_per_sentence = word_count / max(1, sentences)
                readability_score = max(0, min(100, 100 - (avg_words_per_sentence - 15) * 2))
                
                metadata = {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "content_metrics": {
                        "word_count": word_count,
                        "character_count": char_count,
                        "reading_time_minutes": reading_time,
                        "sentence_count": sentences,
                        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
                        "readability_score": round(readability_score, 1)
                    },
                    "seo_metrics": {
                        "seo_score": row["seo_score"],
                        "geo_optimized": row["geo_optimized"],
                        "geo_score": row["geo_score"],
                        "geo_metadata": row["geo_metadata"]
                    },
                    "timestamps": {
                        "created_at": str(row["created_at"]) if row["created_at"] else None,
                        "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                        "published_at": str(row["published_at"]) if row["published_at"] else None
                    }
                }
                
                return metadata
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blog metadata {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/blogs/{post_id}/analyze-seo")
def analyze_blog_seo(post_id: str):
    """Run SEO analysis on a blog post and update SEO score."""
    try:
        # Validate UUID format
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get blog content
                cur.execute(
                    "SELECT title, content_markdown FROM blog_posts WHERE id = %s",
                    (validated_id,)
                )
                row = cur.fetchone()
                
                if not row:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                title = row["title"]
                content = row["content_markdown"] or ""
                
                # Simple SEO analysis
                seo_score = 0
                seo_suggestions = []
                
                # Title length check
                if 30 <= len(title) <= 60:
                    seo_score += 20
                else:
                    seo_suggestions.append("Title should be 30-60 characters")
                
                # Content length check
                word_count = len(content.split())
                if word_count >= 300:
                    seo_score += 20
                else:
                    seo_suggestions.append("Content should have at least 300 words")
                
                # Header structure check
                h1_count = content.count('# ')
                h2_count = content.count('## ')
                if h1_count >= 1 and h2_count >= 2:
                    seo_score += 20
                else:
                    seo_suggestions.append("Add proper header structure (H1, H2)")
                
                # Keyword density (simplified)
                title_words = title.lower().split()
                content_lower = content.lower()
                keyword_mentions = sum(content_lower.count(word) for word in title_words[:3])
                if keyword_mentions >= 3:
                    seo_score += 20
                else:
                    seo_suggestions.append("Include title keywords more frequently in content")
                
                # Readability
                sentences = content.count('.') + content.count('!') + content.count('?')
                avg_words_per_sentence = word_count / max(1, sentences)
                if avg_words_per_sentence <= 20:
                    seo_score += 20
                else:
                    seo_suggestions.append("Shorter sentences improve readability")
                
                # Update SEO score in database
                cur.execute(
                    "UPDATE blog_posts SET seo_score = %s, updated_at = NOW() WHERE id = %s",
                    (seo_score, validated_id)
                )
                conn.commit()
                
                return {
                    "seo_score": seo_score,
                    "analysis": {
                        "title_length": len(title),
                        "word_count": word_count,
                        "header_count": {"h1": h1_count, "h2": h2_count},
                        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
                        "keyword_mentions": keyword_mentions
                    },
                    "suggestions": seo_suggestions,
                    "updated_at": datetime.datetime.utcnow().isoformat()
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing SEO for blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SEO analysis failed: {str(e)}")


@router.get("/blogs/analytics/summary")
def get_blog_analytics_summary():
    """Get summary analytics for all blog posts."""
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_blogs,
                        COUNT(CASE WHEN status = 'published' THEN 1 END) as published_count,
                        COUNT(CASE WHEN status = 'draft' THEN 1 END) as draft_count,
                        AVG(word_count) as avg_word_count,
                        AVG(reading_time) as avg_reading_time,
                        AVG(seo_score) as avg_seo_score,
                        COUNT(CASE WHEN geo_optimized = true THEN 1 END) as geo_optimized_count
                    FROM blog_posts 
                    WHERE status != 'deleted'
                """)
                
                row = cur.fetchone()
                
                return {
                    "total_blogs": row["total_blogs"] or 0,
                    "published_count": row["published_count"] or 0,
                    "draft_count": row["draft_count"] or 0,
                    "average_metrics": {
                        "word_count": round(row["avg_word_count"] or 0, 1),
                        "reading_time": round(row["avg_reading_time"] or 0, 1),
                        "seo_score": round(row["avg_seo_score"] or 0, 1)
                    },
                    "geo_optimized_count": row["geo_optimized_count"] or 0,
                    "generated_at": datetime.datetime.utcnow().isoformat()
                }
                
    except Exception as e:
        logger.error(f"Error getting analytics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@router.post("/blogs/{post_id}/create-campaign")
def create_campaign_from_blog(post_id: str, request: dict):
    """Create a campaign from a blog post."""
    try:
        # Validate UUID format
        validated_id = InputValidator.validate_uuid(post_id, "post_id")
        campaign_name = request.get("campaign_name", f"Campaign for {post_id}")
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if blog exists
            cur.execute("SELECT id, title, status FROM \"blog_posts\" WHERE id = %s", (validated_id,))
            existing_blog = cur.fetchone()
            if not existing_blog:
                raise HTTPException(status_code=404, detail="Blog post not found")
            
            # Check if blog status allows campaign creation
            current_status = existing_blog["status"].lower()
            allowed_statuses = ["edited", "completed", "published"]
            if current_status not in allowed_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot create campaign for blog with status '{current_status}'. Only 'edited', 'completed', or 'published' posts can have campaigns."
                )
            
            # Check if campaign already exists for this blog
            cur.execute('SELECT id FROM campaigns WHERE blog_post_id = %s', (validated_id,))
            existing_campaign = cur.fetchone()
            if existing_campaign:
                raise HTTPException(
                    status_code=400,
                    detail="Campaign already exists for this blog post"
                )
            
            # Create campaign
            campaign_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO campaigns (id, blog_post_id, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
            """, (campaign_id, validated_id))
            
            # Create briefing record
            briefing_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO briefings (id, "campaignName", "marketingObjective", "targetAudience", 
                                      channels, "desiredTone", language, "createdAt", "updatedAt", "campaignId")
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s)
            """, (briefing_id, campaign_name, "Brand awareness", 
                  '["B2B professionals"]', '["LinkedIn", "Email"]', "Professional", "English", campaign_id))
            
            # Create initial campaign task
            task_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO campaign_tasks (id, "campaignId", "taskType", status, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            """, (task_id, campaign_id, "content_repurposing", "pending"))
            
            conn.commit()
            
            logger.info(f"Created campaign {campaign_id} for blog {validated_id}")
            
            return {
                "message": "Campaign created successfully",
                "campaign_id": campaign_id,
                "blog_id": validated_id,
                "campaign_name": campaign_name
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating campaign for blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")