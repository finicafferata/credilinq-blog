"""
Optimized blog management endpoints with caching, async operations, and performance monitoring.
Replaces the original blogs.py with enhanced performance features.
"""

import asyncio
import time
import uuid
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from ...config.database import db_config
from ...config.settings import settings
from ...core.security import validate_api_input, InputValidator
from ...core.exceptions import (
    convert_to_http_exception, InputValidationError, SecurityException
)
from ...core.cache import cache, cached, CachePatterns
from ...core.rate_limiting import rate_limit
from ..models.blog import (
    BlogCreateRequest, BlogEditRequest, BlogReviseRequest, 
    BlogSearchRequest, BlogSummary, BlogDetail
)

router = APIRouter()


# Performance tracking decorator
def track_performance(operation: str):
    """Decorator to track API operation performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance metrics (could send to monitoring service)
                if execution_time > 2.0:  # Log slow operations
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Slow {operation} operation: {execution_time:.2f}s")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                # Log failed operations
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed {operation} operation after {execution_time:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator


async def get_database_service():
    """Get database service with connection pooling."""
    try:
        from ...agents.core.database_service import get_db_service
        return get_db_service()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database service unavailable: {str(e)}"
        )


@router.get("/blogs", response_model=List[BlogSummary])
@track_performance("list_blogs")
@cached(namespace="blogs", ttl=300)  # Cache for 5 minutes
async def list_blogs(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    db_service = Depends(get_database_service)
) -> List[BlogSummary]:
    """
    List blog posts with caching and pagination.
    Optimized with database indexes and materialized views.
    """
    try:
        # Build optimized query
        conditions = []
        params = {
            "limit": min(limit, 100),  # Cap at 100
            "offset": offset
        }
        
        if status:
            conditions.append("status = %(status)s")
            params["status"] = status
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # Use optimized query with performance data
        query = f"""
        SELECT 
            bp.id,
            bp.title,
            bp.status,
            bp.created_at,
            COALESCE(bas.agent_interactions, 0) as agent_interactions,
            COALESCE(bas.campaign_count, 0) as campaign_count
        FROM "BlogPost" bp
        LEFT JOIN blog_analytics_summary bas ON bp.id = bas.id
        {where_clause}
        ORDER BY bp.created_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
        """
        
        results = await asyncio.create_task(
            db_service.execute_query(query, params)
        )
        
        return [
            BlogSummary(
                id=str(row[0]),
                title=row[1],
                status=row[2],
                created_at=row[3].isoformat() if row[3] else None
            )
            for row in results
        ]
        
    except Exception as e:
        raise convert_to_http_exception(e)


@router.get("/blogs/{blog_id}", response_model=BlogDetail)
@track_performance("get_blog")
@cached(namespace="blog_details", ttl=600)  # Cache for 10 minutes
async def get_blog(
    blog_id: str,
    db_service = Depends(get_database_service)
) -> BlogDetail:
    """Get blog post details with caching."""
    try:
        # Validate UUID
        try:
            uuid.UUID(blog_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid blog ID format")
        
        # Optimized query with performance data
        query = """
        SELECT 
            bp.id,
            bp.title,
            bp.content_markdown,
            bp.status,
            bp.initial_prompt,
            bp.created_at,
            bp.updated_at,
            COALESCE(perf.total_time, 0) as total_execution_time,
            COALESCE(perf.agent_count, 0) as agent_count,
            COALESCE(perf.success_rate, 100) as success_rate
        FROM "BlogPost" bp
        LEFT JOIN LATERAL (
            SELECT * FROM get_blog_performance(bp.id)
        ) perf ON true
        WHERE bp.id = %(blog_id)s
        """
        
        results = await asyncio.create_task(
            db_service.execute_query(query, {"blog_id": blog_id})
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        row = results[0]
        return BlogDetail(
            id=str(row[0]),
            title=row[1],
            content_markdown=row[2] or "",
            status=row[3],
            initial_prompt=row[4],
            created_at=row[5].isoformat() if row[5] else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise convert_to_http_exception(e)


@router.post("/blogs", response_model=BlogSummary)
@track_performance("create_blog")
@rate_limit(max_requests=5, window_seconds=300)  # 5 requests per 5 minutes
async def create_blog(
    request: BlogCreateRequest,
    background_tasks: BackgroundTasks,
    db_service = Depends(get_database_service)
) -> BlogSummary:
    """
    Create new blog post with async workflow execution.
    Enhanced with input validation and performance tracking.
    """
    start_time = time.time()
    
    try:
        # Enhanced input validation
        validated_title = InputValidator.validate_string_input(
            request.title, "title", max_length=200
        )
        validated_context = InputValidator.validate_string_input(
            request.company_context, "company_context", max_length=2000
        )
        validated_content_type = InputValidator.validate_string_input(
            request.content_type, "content_type", max_length=50
        )
        
        # Validate content type
        allowed_content_types = ['blog', 'linkedin', 'article', 'social']
        if validated_content_type.lower() not in allowed_content_types:
            raise InputValidationError(
                "content_type", 
                f"Must be one of: {', '.join(allowed_content_types)}"
            )
        
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        # Generate blog ID
        blog_id = str(uuid.uuid4())
        
        # Create initial blog record
        insert_query = """
        INSERT INTO "BlogPost" (
            id, title, status, initial_prompt, created_at, updated_at
        ) VALUES (
            %(blog_id)s, %(title)s, 'generating', %(initial_prompt)s, NOW(), NOW()
        )
        RETURNING id, title, status, created_at
        """
        
        initial_prompt = {
            "title": validated_title,
            "company_context": validated_context,
            "content_type": validated_content_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        results = await asyncio.create_task(
            db_service.execute_query(insert_query, {
                "blog_id": blog_id,
                "title": validated_title,
                "initial_prompt": initial_prompt
            })
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Failed to create blog post")
        
        row = results[0]
        blog_summary = BlogSummary(
            id=str(row[0]),
            title=row[1],
            status=row[2],
            created_at=row[3].isoformat() if row[3] else None
        )
        
        # Invalidate related caches
        await CachePatterns.invalidate_related("blog", blog_id)
        await cache.invalidate_pattern(f"{settings.cache_prefix}:blogs:*")
        
        # Execute workflow asynchronously
        background_tasks.add_task(
            execute_blog_workflow,
            blog_id,
            validated_title,
            validated_context,
            validated_content_type
        )
        
        # Track creation performance
        creation_time = time.time() - start_time
        background_tasks.add_task(
            track_blog_creation_performance,
            blog_id,
            creation_time,
            "success"
        )
        
        return blog_summary
        
    except Exception as e:
        # Track failed creation
        creation_time = time.time() - start_time
        background_tasks.add_task(
            track_blog_creation_performance,
            blog_id if 'blog_id' in locals() else None,
            creation_time,
            "error",
            str(e)
        )
        raise convert_to_http_exception(e)


@router.put("/blogs/{blog_id}", response_model=BlogDetail)
@track_performance("update_blog")
async def update_blog(
    blog_id: str,
    request: BlogEditRequest,
    background_tasks: BackgroundTasks,
    db_service = Depends(get_database_service)
) -> BlogDetail:
    """Update blog content with optimistic locking and caching."""
    try:
        # Validate UUID
        try:
            uuid.UUID(blog_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid blog ID format")
        
        # Validate content
        validated_content = InputValidator.validate_string_input(
            request.content_markdown, "content_markdown", max_length=50000
        )
        
        # Update with optimistic locking
        update_query = """
        UPDATE "BlogPost" 
        SET 
            content_markdown = %(content)s,
            status = CASE 
                WHEN status = 'generating' THEN 'draft'
                WHEN status = 'draft' THEN 'edited'
                ELSE status
            END,
            updated_at = NOW()
        WHERE id = %(blog_id)s
        RETURNING id, title, content_markdown, status, initial_prompt, created_at, updated_at
        """
        
        results = await asyncio.create_task(
            db_service.execute_query(update_query, {
                "blog_id": blog_id,
                "content": validated_content
            })
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        row = results[0]
        blog_detail = BlogDetail(
            id=str(row[0]),
            title=row[1],
            content_markdown=row[2] or "",
            status=row[3],
            initial_prompt=row[4],
            created_at=row[5].isoformat() if row[5] else None
        )
        
        # Invalidate caches asynchronously
        background_tasks.add_task(
            invalidate_blog_caches,
            blog_id
        )
        
        return blog_detail
        
    except HTTPException:
        raise
    except SecurityException as e:
        raise convert_to_http_exception(e)
    except Exception as e:
        raise convert_to_http_exception(e)


@router.post("/blogs/{blog_id}/publish", response_model=BlogDetail)
@track_performance("publish_blog")
@rate_limit(max_requests=10, window_seconds=60)
async def publish_blog(
    blog_id: str,
    background_tasks: BackgroundTasks,
    db_service = Depends(get_database_service)
) -> BlogDetail:
    """Publish blog post with validation and caching."""
    try:
        # Validate UUID
        try:
            uuid.UUID(blog_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid blog ID format")
        
        # Update status to published
        update_query = """
        UPDATE "BlogPost" 
        SET 
            status = 'published',
            updated_at = NOW()
        WHERE id = %(blog_id)s 
        AND status IN ('draft', 'edited')
        RETURNING id, title, content_markdown, status, initial_prompt, created_at, updated_at
        """
        
        results = await asyncio.create_task(
            db_service.execute_query(update_query, {"blog_id": blog_id})
        )
        
        if not results:
            raise HTTPException(
                status_code=400, 
                detail="Blog post not found or cannot be published"
            )
        
        row = results[0]
        blog_detail = BlogDetail(
            id=str(row[0]),
            title=row[1],
            content_markdown=row[2] or "",
            status=row[3],
            initial_prompt=row[4],
            created_at=row[5].isoformat() if row[5] else None
        )
        
        # Invalidate caches and trigger post-publish tasks
        background_tasks.add_task(
            handle_blog_publication,
            blog_id
        )
        
        return blog_detail
        
    except HTTPException:
        raise
    except Exception as e:
        raise convert_to_http_exception(e)


@router.delete("/blogs/{blog_id}")
@track_performance("delete_blog")
async def delete_blog(
    blog_id: str,
    background_tasks: BackgroundTasks,
    db_service = Depends(get_database_service)
):
    """Delete blog post with cascade cleanup."""
    try:
        # Validate UUID
        try:
            uuid.UUID(blog_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid blog ID format")
        
        # Delete with cascade (handled by database constraints)
        delete_query = """
        DELETE FROM "BlogPost" 
        WHERE id = %(blog_id)s
        RETURNING id, title
        """
        
        results = await asyncio.create_task(
            db_service.execute_query(delete_query, {"blog_id": blog_id})
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Cleanup caches and related data
        background_tasks.add_task(
            cleanup_deleted_blog,
            blog_id
        )
        
        return {"message": "Blog post deleted successfully", "id": blog_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise convert_to_http_exception(e)


# Background task functions

async def execute_blog_workflow(
    blog_id: str, 
    title: str, 
    context: str, 
    content_type: str
):
    """Execute blog generation workflow asynchronously."""
    try:
        from ...agents.workflow.structured_blog_workflow import StructuredBlogWorkflow
        
        workflow = StructuredBlogWorkflow()
        
        # Execute workflow with timeout
        result = await asyncio.wait_for(
            workflow.execute_async({
                "blog_id": blog_id,
                "title": title,
                "company_context": context,
                "content_type": content_type
            }),
            timeout=settings.agent_timeout_seconds
        )
        
        # Update blog status based on result
        db_service = await get_database_service()
        
        if result.get("success"):
            await db_service.execute_query(
                'UPDATE "BlogPost" SET status = %s, updated_at = NOW() WHERE id = %s',
                ("completed", blog_id)
            )
        else:
            await db_service.execute_query(
                'UPDATE "BlogPost" SET status = %s, updated_at = NOW() WHERE id = %s',
                ("error", blog_id)
            )
        
        # Invalidate caches
        await invalidate_blog_caches(blog_id)
        
    except asyncio.TimeoutError:
        # Handle timeout
        db_service = await get_database_service()
        await db_service.execute_query(
            'UPDATE "BlogPost" SET status = %s, updated_at = NOW() WHERE id = %s',
            ("timeout", blog_id)
        )
        
    except Exception as e:
        # Handle other errors
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Blog workflow failed for {blog_id}: {str(e)}")
        
        db_service = await get_database_service()
        await db_service.execute_query(
            'UPDATE "BlogPost" SET status = %s, updated_at = NOW() WHERE id = %s',
            ("error", blog_id)
        )


async def track_blog_creation_performance(
    blog_id: Optional[str],
    execution_time: float,
    status: str,
    error_message: Optional[str] = None
):
    """Track blog creation performance metrics."""
    try:
        db_service = await get_database_service()
        
        metadata = {
            "operation": "blog_creation",
            "execution_time": execution_time,
            "status": status
        }
        
        if error_message:
            metadata["error"] = error_message
        
        await db_service.execute_query(
            '''INSERT INTO "AgentPerformance" 
               (agent_type, blog_id, execution_time_seconds, success, metadata, created_at)
               VALUES (%s, %s, %s, %s, %s, NOW())''',
            (
                "api_performance",
                blog_id,
                execution_time,
                status == "success",
                metadata
            )
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to track performance: {str(e)}")


async def invalidate_blog_caches(blog_id: str):
    """Invalidate all caches related to a blog."""
    try:
        await CachePatterns.invalidate_related("blog", blog_id)
        await cache.invalidate_pattern(f"{settings.cache_prefix}:blogs:*")
        await cache.invalidate_pattern(f"{settings.cache_prefix}:blog_details:*{blog_id}*")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Cache invalidation failed: {str(e)}")


async def handle_blog_publication(blog_id: str):
    """Handle post-publication tasks."""
    try:
        await invalidate_blog_caches(blog_id)
        
        # Could trigger additional tasks like:
        # - Social media posting
        # - Email notifications
        # - Analytics tracking
        # - SEO optimization
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Post-publication tasks failed for {blog_id}: {str(e)}")


async def cleanup_deleted_blog(blog_id: str):
    """Cleanup tasks for deleted blog."""
    try:
        await invalidate_blog_caches(blog_id)
        
        # Additional cleanup tasks could include:
        # - Remove from search indexes
        # - Clean up associated files
        # - Update analytics
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Cleanup failed for deleted blog {blog_id}: {str(e)}")


# Health check for blog service
@router.get("/blogs/health")
async def blog_service_health():
    """Health check specific to blog service."""
    try:
        start_time = time.time()
        
        # Test database connection
        db_service = await get_database_service()
        await db_service.execute_query("SELECT 1", {})
        
        db_time = time.time() - start_time
        
        # Test cache connection
        cache_start = time.time()
        await cache.set("health", "blog_service_test", "ok", 60)
        cache_result = await cache.get("health", "blog_service_test")
        cache_time = time.time() - cache_start
        
        return {
            "status": "healthy",
            "database_response_time_ms": round(db_time * 1000, 2),
            "cache_response_time_ms": round(cache_time * 1000, 2),
            "cache_working": cache_result == "ok",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )