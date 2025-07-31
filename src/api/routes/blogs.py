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
from ...core.security import validate_api_input, InputValidator
from ...core.exceptions import (
    convert_to_http_exception, InputValidationError, SecurityException
)
from ..models.blog import (
    BlogCreateRequest, BlogEditRequest, BlogReviseRequest, 
    BlogSearchRequest, BlogSummary, BlogDetail
)

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
        
        validated_context = InputValidator.validate_string_input(
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
        
        # Log agent decision
        decision = AgentDecision(
            agent_type="blog_workflow",
            decision_context=agent_input,
            reasoning=f"Creating {request.content_type} about '{request.title}'",
            confidence_score=0.9
        )
        
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
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO "BlogPost" (id, title, "contentMarkdown", "initialPrompt", status, "createdAt", "updatedAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    blog_id, 
                    validated_title, 
                    final_post, 
                    initial_prompt, 
                    "draft", 
                    created_at,
                    updated_at
                ))
        except Exception as e:
            logger.error(f"Error creating blog in database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # Log successful creation
        execution_time = int((time.time() - start_time) * 1000)
        decision.blog_id = blog_id
        decision.outcome = "success"
        decision.execution_time_ms = execution_time
        
        try:
            db_service.log_agent_decision(decision)
            
            # Log performance metrics
            metrics = AgentPerformanceMetrics(
                agent_type="blog_workflow",
                execution_time_ms=execution_time,
                success=True,
                input_size=len(str(agent_input)),
                output_size=len(final_post)
            )
            db_service.log_performance_metrics(metrics)
            
            # Log analytics data
            analytics = BlogAnalyticsData(
                blog_id=blog_id,
                content_type=validated_content_type,
                word_count=len(final_post.split()),
                creation_time_ms=execution_time
            )
            db_service.log_blog_analytics(analytics)
        except Exception as e:
            logger.warning(f"Failed to log analytics data: {str(e)}")
            # No fallar la creación del blog si falla el logging
        
        return BlogSummary(
            id=blog_id,
            title=validated_title,
            status="draft",
            created_at=created_at
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
            cur.execute("SELECT id, title, status, \"createdAt\" FROM \"BlogPost\" LIMIT 3")
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
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, status, "createdAt", "contentMarkdown", "initialPrompt"
                FROM "BlogPost" 
                WHERE id = %s
            """, (validated_id,))
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Blog post not found")
            
            # Access by index: id, title, status, createdAt, contentMarkdown, initialPrompt
            blog_id = str(row[0]) if row[0] else ''
            title = str(row[1]) if row[1] else 'Untitled'
            status = str(row[2]) if row[2] else 'draft'
            created_at = row[3]
            content_markdown = str(row[4]) if row[4] else ''
            initial_prompt_raw = row[5]
            
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
            
            return BlogDetail(
                id=blog_id,
                title=title,
                status=status,
                created_at=str(created_at) if created_at else datetime.datetime.utcnow().isoformat(),
                content_markdown=content_markdown,
                initial_prompt=initial_prompt
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
        validated_content = InputValidator.validate_string_input(
            request.content_markdown, "content_markdown", max_length=50000
        )
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE "BlogPost" 
                SET "contentMarkdown" = %s, status = 'edited'
                WHERE id = %s
            """, (validated_content, validated_id))
            
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Blog post not found")
        
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
            cur.execute("SELECT id, title FROM \"BlogPost\" WHERE id = %s", (validated_id,))
            existing_blog = cur.fetchone()
            if not existing_blog:
                raise HTTPException(status_code=404, detail="Blog post not found")
            
            # Soft delete
            cur.execute("""
                UPDATE "BlogPost" 
                SET status = 'deleted'
                WHERE id = %s
            """, (validated_id,))
        
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
            cur = conn.cursor()
            
            # Check if blog exists and get current status
            cur.execute("SELECT id, title, status FROM \"BlogPost\" WHERE id = %s", (validated_id,))
            existing_blog = cur.fetchone()
            if not existing_blog:
                raise HTTPException(status_code=404, detail="Blog post not found")
            
            current_status = existing_blog["status"].lower()
            if current_status not in ["draft", "edited"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot publish blog with status '{current_status}'. Only 'draft' or 'edited' posts can be published."
                )
            
            # Update status to published
            cur.execute("""
                UPDATE "BlogPost" 
                SET status = 'published'
                WHERE id = %s
            """, (validated_id,))
        
        return get_blog(validated_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing blog {post_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


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
            cur.execute("SELECT id, title, status FROM \"BlogPost\" WHERE id = %s", (validated_id,))
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
            cur.execute('SELECT id FROM "Campaign" WHERE "blogPostId" = %s', (validated_id,))
            existing_campaign = cur.fetchone()
            if existing_campaign:
                raise HTTPException(
                    status_code=400,
                    detail="Campaign already exists for this blog post"
                )
            
            # Create campaign
            campaign_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO "Campaign" (id, "blogPostId", "createdAt")
                VALUES (%s, %s, NOW())
            """, (campaign_id, validated_id))
            
            # Create briefing record
            briefing_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO "Briefing" (id, "campaignName", "marketingObjective", "targetAudience", 
                                      channels, "desiredTone", language, "createdAt", "updatedAt", "campaignId")
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s)
            """, (briefing_id, campaign_name, "Brand awareness", 
                  '["B2B professionals"]', '["LinkedIn", "Email"]', "Professional", "English", campaign_id))
            
            # Create initial campaign task
            task_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO "CampaignTask" (id, "campaignId", "taskType", status, "createdAt", "updatedAt")
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