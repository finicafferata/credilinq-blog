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


@router.get("/blogs", response_model=List[BlogSummary])
def list_blogs():
    """List all blog posts stored in database (excluding deleted posts)."""
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, status, "createdAt"
                FROM "BlogPost" 
                WHERE status != 'deleted' 
                ORDER BY "createdAt" DESC
            """)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            blogs = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert datetime to string
                row_dict['created_at'] = str(row_dict['createdAt'])
                del row_dict['createdAt']  # Remove the original key
                blogs.append(BlogSummary(**row_dict))
            
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
            
            columns = [desc[0] for desc in cur.description]
            row_dict = dict(zip(columns, row))
            
            # Handle initial_prompt - it might already be a dict or a JSON string
            initial_prompt = row_dict["initialPrompt"]
            if initial_prompt:
                if isinstance(initial_prompt, str):
                    initial_prompt = json.loads(initial_prompt)
                elif not isinstance(initial_prompt, dict):
                    initial_prompt = {}
            else:
                initial_prompt = {}
            
            return BlogDetail(
                id=row_dict["id"],
                title=row_dict["title"],
                status=row_dict["status"],
                created_at=str(row_dict["createdAt"]),
                content_markdown=row_dict["contentMarkdown"] or "",
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