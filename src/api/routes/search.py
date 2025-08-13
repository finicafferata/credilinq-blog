"""Vector search and document retrieval endpoints."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...config.database import db_config
from ...core.security import InputValidator
from ...core.exceptions import convert_to_http_exception, SecurityException
import psycopg2.extras

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.7

class SearchResult(BaseModel):
    id: str
    content: str
    similarity: float
    document_id: Optional[str] = None
    document_title: Optional[str] = None

class VectorSearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time_ms: int

@router.post("/search/documents", response_model=VectorSearchResponse)
def search_documents(request: SearchRequest):
    """
    Perform vector similarity search across document chunks.
    Note: This endpoint requires embeddings to be generated first.
    """
    try:
        # Validate input
        validated_query = InputValidator.validate_content_text(
            request.query, "query", max_length=1000
        )
        validated_limit = min(max(1, request.limit or 10), 100)
        validated_threshold = max(0.0, min(1.0, request.threshold or 0.7))
        
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    import time
    start_time = time.time()
    
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # For now, implement basic text search until embeddings are generated
                # TODO: Replace with actual vector similarity search when embeddings are available
                cur.execute("""
                    SELECT 
                        dc.id,
                        dc.content,
                        d.id as document_id,
                        d.title as document_title,
                        ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', %s)) as similarity
                    FROM document_chunks dc
                    LEFT JOIN documents d ON dc.document_id = d.id
                    WHERE to_tsvector('english', dc.content) @@ plainto_tsquery('english', %s)
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (validated_query, validated_query, validated_limit))
                
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    if row["similarity"] >= validated_threshold:
                        results.append(SearchResult(
                            id=str(row["id"]),
                            content=row["content"][:500] + "..." if len(row["content"]) > 500 else row["content"],
                            similarity=float(row["similarity"]),
                            document_id=str(row["document_id"]) if row["document_id"] else None,
                            document_title=row["document_title"]
                        ))
                
                processing_time = int((time.time() - start_time) * 1000)
                
                return VectorSearchResponse(
                    results=results,
                    query=validated_query,
                    total_results=len(results),
                    processing_time_ms=processing_time
                )
                
    except Exception as e:
        logger.error(f"Error in document search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/documents/stats")
def get_search_stats():
    """Get statistics about the document corpus for search."""
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get document statistics
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT d.id) as document_count,
                        COUNT(dc.id) as chunk_count,
                        COUNT(CASE WHEN dc.embedding IS NOT NULL THEN 1 END) as embedded_chunks,
                        AVG(LENGTH(dc.content)) as avg_chunk_length
                    FROM documents d
                    LEFT JOIN document_chunks dc ON d.id = dc.document_id
                """)
                
                stats = cur.fetchone()
                
                # Get recent documents
                cur.execute("""
                    SELECT id, title, uploaded_at, chunk_count
                    FROM documents
                    ORDER BY uploaded_at DESC
                    LIMIT 5
                """)
                
                recent_docs = cur.fetchall()
                
                return {
                    "corpus_stats": {
                        "total_documents": stats["document_count"] or 0,
                        "total_chunks": stats["chunk_count"] or 0,
                        "embedded_chunks": stats["embedded_chunks"] or 0,
                        "avg_chunk_length": round(stats["avg_chunk_length"] or 0, 1),
                        "embedding_coverage": round((stats["embedded_chunks"] or 0) / max(1, stats["chunk_count"] or 1) * 100, 1)
                    },
                    "recent_documents": [
                        {
                            "id": str(doc["id"]),
                            "title": doc["title"],
                            "uploaded_at": str(doc["uploaded_at"]) if doc["uploaded_at"] else None,
                            "chunk_count": doc["chunk_count"]
                        }
                        for doc in recent_docs
                    ],
                    "search_capabilities": {
                        "vector_search": stats["embedded_chunks"] > 0,
                        "text_search": True,
                        "hybrid_search": stats["embedded_chunks"] > 0
                    }
                }
                
    except Exception as e:
        logger.error(f"Error getting search stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@router.post("/search/blogs", response_model=dict)
def search_blogs(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    include_content: bool = Query(False, description="Include full content in results")
):
    """Search blog posts by title and content."""
    try:
        # Validate input
        validated_query = InputValidator.validate_content_text(
            query, "query", max_length=500
        )
        
    except SecurityException as e:
        raise convert_to_http_exception(e)
    
    try:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Full-text search across blog posts
                if include_content:
                    cur.execute("""
                        SELECT 
                            id, title, status, created_at, updated_at, published_at,
                            word_count, reading_time, seo_score,
                            content_markdown,
                            ts_rank(
                                to_tsvector('english', title || ' ' || content_markdown), 
                                plainto_tsquery('english', %s)
                            ) as relevance
                        FROM blog_posts
                        WHERE status != 'deleted'
                        AND (
                            to_tsvector('english', title || ' ' || content_markdown) @@ plainto_tsquery('english', %s)
                        )
                        ORDER BY relevance DESC
                        LIMIT %s
                    """, (validated_query, validated_query, limit))
                else:
                    cur.execute("""
                        SELECT 
                            id, title, status, created_at, updated_at, published_at,
                            word_count, reading_time, seo_score,
                            ts_rank(
                                to_tsvector('english', title || ' ' || content_markdown), 
                                plainto_tsquery('english', %s)
                            ) as relevance
                        FROM blog_posts
                        WHERE status != 'deleted'
                        AND (
                            to_tsvector('english', title || ' ' || content_markdown) @@ plainto_tsquery('english', %s)
                        )
                        ORDER BY relevance DESC
                        LIMIT %s
                    """, (validated_query, validated_query, limit))
                
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    blog_result = {
                        "id": str(row["id"]),
                        "title": row["title"],
                        "status": row["status"],
                        "relevance": float(row["relevance"]),
                        "metadata": {
                            "word_count": row["word_count"],
                            "reading_time": row["reading_time"],
                            "seo_score": row["seo_score"],
                            "created_at": str(row["created_at"]) if row["created_at"] else None,
                            "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                            "published_at": str(row["published_at"]) if row["published_at"] else None
                        }
                    }
                    
                    if include_content:
                        content = row["content_markdown"] or ""
                        # Include a snippet of the content
                        blog_result["content_snippet"] = content[:300] + "..." if len(content) > 300 else content
                    
                    results.append(blog_result)
                
                return {
                    "results": results,
                    "query": validated_query,
                    "total_results": len(results),
                    "search_type": "full_text"
                }
                
    except Exception as e:
        logger.error(f"Error in blog search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blog search failed: {str(e)}")