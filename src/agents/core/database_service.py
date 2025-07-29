"""
Enhanced Database Service Layer for AI Agents
Provides centralized database operations with performance tracking,
agent decision logging, and marketing analytics.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from supabase import create_client, Client
import logging

# Import our custom exceptions
from ...core.exceptions import (
    DatabaseConnectionError, DatabaseQueryError, SQLInjectionAttempt,
    InputValidationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentPerformanceMetrics:
    agent_type: str
    task_type: str
    execution_time_ms: int
    success_rate: float
    quality_score: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error_count: int = 0

@dataclass
class AgentDecision:
    agent_type: str
    blog_id: Optional[str] = None
    campaign_id: Optional[str] = None
    decision_context: Optional[Dict] = None
    reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    outcome: Optional[str] = None
    execution_time_ms: Optional[int] = None

@dataclass
class BlogAnalyticsData:
    blog_id: str
    views: int = 0
    unique_visitors: int = 0
    engagement_rate: float = 0.0
    avg_time_on_page: Optional[int] = None
    bounce_rate: float = 0.0
    social_shares: int = 0
    comments_count: int = 0
    conversion_rate: float = 0.0
    seo_score: float = 0.0
    readability_score: float = 0.0

@dataclass
class MarketingMetric:
    metric_type: str
    metric_value: float
    blog_id: Optional[str] = None
    campaign_id: Optional[str] = None
    source: Optional[str] = None
    medium: Optional[str] = None
    campaign_name: Optional[str] = None

class DatabaseService:
    """
    Enhanced database service with agent performance tracking,
    decision logging, and marketing analytics.
    """
    
    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        self.database_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
        self.database_url_direct = os.getenv("DATABASE_URL_DIRECT", "postgresql://postgres@localhost:5432/credilinq_dev_postgres")
        
        # For local PostgreSQL, we don't use Supabase client
        self.supabase = None
        self.use_supabase = False
        
        # Check if we have Supabase configuration
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            try:
                self.supabase: Client = create_client(supabase_url, supabase_key)
                self.use_supabase = True
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {str(e)}")
                self.supabase = None
                self.use_supabase = False
        self._connection_pool = None
        
    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper error handling and cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(
                self.database_url_direct,
                cursor_factory=RealDictCursor,
                connect_timeout=30
            )
            conn.autocommit = True
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    # ===== AGENT PERFORMANCE TRACKING =====
    
    def log_agent_performance(self, metrics: AgentPerformanceMetrics) -> str:
        """Log agent performance metrics for analysis and optimization."""
        try:
            data = {
                "agent_type": metrics.agent_type,
                "task_type": metrics.task_type,
                "execution_time_ms": metrics.execution_time_ms,
                "success_rate": metrics.success_rate,
                "quality_score": metrics.quality_score,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "cost_usd": metrics.cost_usd,
                "error_count": metrics.error_count
            }
            
            if self.use_supabase and self.supabase:
                response = self.supabase.table("agent_performance").insert(data).execute()
                logger.info(f"Performance logged for {metrics.agent_type}:{metrics.task_type}")
                return response.data[0]["id"]
            else:
                # For local PostgreSQL, just log to console
                logger.info(f"Performance logged for {metrics.agent_type}:{metrics.task_type} - {data}")
                return "local_log"
            
        except Exception as e:
            logger.error(f"Failed to log agent performance: {str(e)}")
            raise

    def log_agent_decision(self, decision: AgentDecision) -> str:
        """Log agent decision for learning and audit purposes."""
        try:
            data = {
                "agent_type": decision.agent_type,
                "blog_id": decision.blog_id,
                "campaign_id": decision.campaign_id,
                "decision_context": decision.decision_context,
                "reasoning": decision.reasoning,
                "confidence_score": decision.confidence_score,
                "outcome": decision.outcome,
                "execution_time_ms": decision.execution_time_ms
            }
            
            if self.use_supabase and self.supabase:
                response = self.supabase.table("agent_decisions").insert(data).execute()
                logger.info(f"Decision logged for {decision.agent_type}")
                return response.data[0]["id"]
            else:
                # For local PostgreSQL, just log to console
                logger.info(f"Decision logged for {decision.agent_type} - {data}")
                return "local_log"
            
        except Exception as e:
            logger.error(f"Failed to log agent decision: {str(e)}")
            raise

    def get_agent_performance_analytics(self, agent_type: Optional[str] = None, 
                                       days: int = 30) -> List[Dict]:
        """Get agent performance analytics for optimization."""
        try:
            if self.use_supabase and self.supabase:
                query = self.supabase.table("agent_performance").select("*")
                
                if agent_type:
                    query = query.eq("agent_type", agent_type)
                
                # Filter by date range
                cutoff_date = (datetime.now(timezone.utc) - 
                              timezone.timedelta(days=days)).isoformat()
                query = query.gte("created_at", cutoff_date)
                
                response = query.order("created_at", desc=True).execute()
                return response.data
            else:
                # For local PostgreSQL, return empty list
                logger.info("Agent performance analytics not available in local mode")
                return []
            
        except Exception as e:
            logger.error(f"Failed to get agent performance analytics: {str(e)}")
            return []

    # ===== BLOG ANALYTICS =====
    
    def update_blog_analytics(self, analytics: BlogAnalyticsData) -> str:
        """Update or create blog analytics record."""
        try:
            # Check if analytics record exists
            existing = self.supabase.table("blog_analytics")\
                .select("id")\
                .eq("blog_id", analytics.blog_id)\
                .single()\
                .execute()
            
            data = {
                "blog_id": analytics.blog_id,
                "views": analytics.views,
                "unique_visitors": analytics.unique_visitors,
                "engagement_rate": analytics.engagement_rate,
                "avg_time_on_page": analytics.avg_time_on_page,
                "bounce_rate": analytics.bounce_rate,
                "social_shares": analytics.social_shares,
                "comments_count": analytics.comments_count,
                "conversion_rate": analytics.conversion_rate,
                "seo_score": analytics.seo_score,
                "readability_score": analytics.readability_score
            }
            
            if existing.data:
                # Update existing record
                response = self.supabase.table("blog_analytics")\
                    .update(data)\
                    .eq("blog_id", analytics.blog_id)\
                    .execute()
                record_id = existing.data["id"]
            else:
                # Create new record
                response = self.supabase.table("blog_analytics")\
                    .insert(data)\
                    .execute()
                record_id = response.data[0]["id"]
            
            logger.info(f"Blog analytics updated for blog {analytics.blog_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to update blog analytics: {str(e)}")
            raise

    def record_marketing_metric(self, metric: MarketingMetric) -> str:
        """Record a marketing metric for campaign tracking."""
        try:
            data = {
                "blog_id": metric.blog_id,
                "campaign_id": metric.campaign_id,
                "metric_type": metric.metric_type,
                "metric_value": metric.metric_value,
                "source": metric.source,
                "medium": metric.medium,
                "campaign_name": metric.campaign_name
            }
            
            response = self.supabase.table("marketing_metrics").insert(data).execute()
            logger.info(f"Marketing metric recorded: {metric.metric_type}")
            return response.data[0]["id"]
            
        except Exception as e:
            logger.error(f"Failed to record marketing metric: {str(e)}")
            raise

    # ===== CONTENT OPTIMIZATION =====
    
    def log_content_optimization(self, blog_id: str, optimization_type: str,
                               before_score: float, after_score: float,
                               agent_type: str, changes_made: Dict,
                               optimization_prompt: Optional[str] = None) -> str:
        """Log content optimization for tracking improvements."""
        try:
            improvement_percentage = ((after_score - before_score) / before_score * 100) \
                if before_score > 0 else 0
            
            data = {
                "blog_id": blog_id,
                "optimization_type": optimization_type,
                "before_score": before_score,
                "after_score": after_score,
                "optimization_prompt": optimization_prompt,
                "agent_type": agent_type,
                "changes_made": changes_made,
                "improvement_percentage": improvement_percentage
            }
            
            response = self.supabase.table("content_optimization").insert(data).execute()
            logger.info(f"Content optimization logged for blog {blog_id}")
            return response.data[0]["id"]
            
        except Exception as e:
            logger.error(f"Failed to log content optimization: {str(e)}")
            raise

    def update_seo_metadata(self, blog_id: str, seo_data: Dict) -> str:
        """Update or create SEO metadata for a blog post."""
        try:
            # Check if SEO metadata exists
            existing = self.supabase.table("seo_metadata")\
                .select("id")\
                .eq("blog_id", blog_id)\
                .single()\
                .execute()
            
            data = {
                "blog_id": blog_id,
                **seo_data
            }
            
            if existing.data:
                # Update existing record
                response = self.supabase.table("seo_metadata")\
                    .update(data)\
                    .eq("blog_id", blog_id)\
                    .execute()
                record_id = existing.data["id"]
            else:
                # Create new record
                response = self.supabase.table("seo_metadata")\
                    .insert(data)\
                    .execute()
                record_id = response.data[0]["id"]
            
            logger.info(f"SEO metadata updated for blog {blog_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to update SEO metadata: {str(e)}")
            raise

    # ===== VECTOR SEARCH WITH PERFORMANCE TRACKING =====
    
    def search_similar_content(self, query_embedding: List[float], 
                              limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """Enhanced vector search with performance tracking."""
        start_time = time.time()
        
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Convert embedding to pgvector format with strict validation
                if not query_embedding or not isinstance(query_embedding, (list, tuple)):
                    raise InputValidationError("query_embedding", "Must be a non-empty list or tuple")
                
                # Validate that all elements are numeric to prevent injection
                if not all(isinstance(x, (int, float, complex)) and not isinstance(x, bool) for x in query_embedding):
                    raise InputValidationError("query_embedding", "All elements must be numeric (int/float)")
                
                if len(query_embedding) != 1536:  # OpenAI embedding dimension
                    raise InputValidationError("query_embedding", f"Expected 1536 dimensions, got {len(query_embedding)}")
                
                # Validate other parameters
                if not isinstance(limit, int) or limit <= 0 or limit > 100:
                    raise InputValidationError("limit", "Must be an integer between 1 and 100")
                
                if not isinstance(similarity_threshold, (int, float)) or not 0.0 <= similarity_threshold <= 1.0:
                    raise InputValidationError("similarity_threshold", "Must be a number between 0.0 and 1.0")
                
                # Convert to safe list of floats
                try:
                    embedding_list = [float(x) for x in query_embedding]
                except (ValueError, TypeError) as e:
                    raise InputValidationError("query_embedding", f"Failed to convert to float: {str(e)}")
                
                # Perform similarity search with safe parameterization
                cur.execute("""
                    SELECT 
                        dc.content,
                        d.title as document_title,
                        dc.document_id,
                        (1 - (dc.embedding <=> %s::vector)) as similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE (1 - (dc.embedding <=> %s::vector)) > %s
                    ORDER BY dc.embedding <=> %s::vector ASC
                    LIMIT %s;
                """, (embedding_list, embedding_list, similarity_threshold, embedding_list, limit))
                
                results = cur.fetchall()
                
                # Log search performance
                execution_time = int((time.time() - start_time) * 1000)
                self.log_agent_performance(AgentPerformanceMetrics(
                    agent_type="vector_search",
                    task_type="similarity_search",
                    execution_time_ms=execution_time,
                    success_rate=1.0 if results else 0.0,
                    quality_score=len(results) / limit * 10
                ))
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            # Log failed search
            self.log_agent_performance(AgentPerformanceMetrics(
                agent_type="vector_search",
                task_type="similarity_search",
                execution_time_ms=int((time.time() - start_time) * 1000),
                success_rate=0.0,
                quality_score=0.0,
                error_count=1
            ))
            raise
    
    def vector_search(self, query: str, limit: int = 3) -> List[Dict]:
        """Simple vector search for compatibility with ResearcherAgent."""
        try:
            # For now, return empty results since we don't have embeddings set up
            # This will allow the ResearcherAgent to fall back to local research
            logger.info(f"Vector search called for query: {query}, returning empty results")
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    # ===== CAMPAIGN MANAGEMENT WITH ANALYTICS =====
    
    def get_campaign_performance(self, campaign_id: str) -> Dict:
        """Get comprehensive campaign performance data."""
        try:
            # Get campaign details with tasks
            campaign_response = self.supabase.table("campaign")\
                .select("*, tasks:campaign_task(*), blog_post:blog_posts(*)")\
                .eq("id", campaign_id)\
                .single()\
                .execute()
            
            if not campaign_response.data:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            campaign = campaign_response.data
            
            # Get marketing metrics
            metrics_response = self.supabase.table("marketing_metrics")\
                .select("*")\
                .eq("campaign_id", campaign_id)\
                .execute()
            
            # Get blog analytics
            blog_analytics = None
            if campaign.get("blog_id"):
                analytics_response = self.supabase.table("blog_analytics")\
                    .select("*")\
                    .eq("blog_id", campaign["blog_id"])\
                    .single()\
                    .execute()
                blog_analytics = analytics_response.data
            
            # Calculate performance metrics
            tasks = campaign.get("tasks", [])
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t["status"] == "completed"])
            failed_tasks = len([t for t in tasks if t["status"] == "failed"])
            
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Aggregate marketing metrics
            metrics_by_type = {}
            for metric in metrics_response.data:
                metric_type = metric["metric_type"]
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = 0
                metrics_by_type[metric_type] += metric["metric_value"]
            
            return {
                "campaign": campaign,
                "performance": {
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "completion_rate": completion_rate
                },
                "marketing_metrics": metrics_by_type,
                "blog_analytics": blog_analytics
            }
            
        except Exception as e:
            logger.error(f"Failed to get campaign performance: {str(e)}")
            raise

    # ===== FEEDBACK AND LEARNING SYSTEM =====
    
    def record_agent_feedback(self, agent_type: str, feedback_type: str,
                            feedback_value: Optional[float] = None,
                            feedback_text: Optional[str] = None,
                            blog_id: Optional[str] = None,
                            campaign_id: Optional[str] = None,
                            task_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            learning_context: Optional[Dict] = None) -> str:
        """Record feedback for agent learning and improvement."""
        try:
            data = {
                "agent_type": agent_type,
                "feedback_type": feedback_type,
                "feedback_value": feedback_value,
                "feedback_text": feedback_text,
                "blog_id": blog_id,
                "campaign_id": campaign_id,
                "task_id": task_id,
                "user_id": user_id,
                "is_positive": feedback_value > 5.0 if feedback_value is not None else None,
                "learning_context": learning_context
            }
            
            response = self.supabase.table("agent_feedback").insert(data).execute()
            logger.info(f"Feedback recorded for agent {agent_type}")
            return response.data[0]["id"]
            
        except Exception as e:
            logger.error(f"Failed to record agent feedback: {str(e)}")
            raise

    # ===== ANALYTICS AND REPORTING =====
    
    def get_dashboard_analytics(self, days: int = 30) -> Dict:
        """Get comprehensive analytics for dashboard."""
        try:
            # Validate days parameter to prevent injection
            if not isinstance(days, int) or days <= 0 or days > 365:
                raise InputValidationError("days", "Must be an integer between 1 and 365")
            
            cutoff_date = (datetime.now(timezone.utc) - 
                          timezone.timedelta(days=days)).isoformat()
            
            # Blog performance - Use safe Supabase queries instead of raw SQL
            blog_posts = self.supabase.table("blog_posts")\
                .select("id, created_at")\
                .gte("created_at", cutoff_date)\
                .execute()
            
            blog_analytics = self.supabase.table("blog_analytics")\
                .select("blog_id, views, engagement_rate, social_shares")\
                .execute()
            
            # Process results safely
            total_blogs = len(blog_posts.data) if blog_posts.data else 0
            blog_ids = [post['id'] for post in blog_posts.data or []]
            
            relevant_analytics = [a for a in blog_analytics.data or [] if a['blog_id'] in blog_ids]
            
            total_views = sum(a.get('views', 0) for a in relevant_analytics)
            total_engagement = sum(a.get('engagement_rate', 0) for a in relevant_analytics)
            total_shares = sum(a.get('social_shares', 0) for a in relevant_analytics)
            
            avg_views = total_views / total_blogs if total_blogs > 0 else 0
            avg_engagement = total_engagement / len(relevant_analytics) if relevant_analytics else 0
            
            blog_performance = {
                "total_blogs": total_blogs,
                "avg_views": avg_views,
                "avg_engagement": avg_engagement,
                "total_shares": total_shares
            }
            
            # Agent performance
            agent_performance = self.supabase.table("agent_performance")\
                .select("agent_type, avg_execution_time:execution_time_ms.avg(), avg_quality:quality_score.avg()")\
                .gte("created_at", cutoff_date)\
                .execute()
            
            # Campaign metrics - Use safe Supabase queries
            campaigns = self.supabase.table("campaign")\
                .select("id, created_at")\
                .gte("created_at", cutoff_date)\
                .execute()
            
            campaign_tasks = self.supabase.table("campaign_task")\
                .select("campaign_id, status")\
                .execute()
            
            # Process campaign metrics safely
            total_campaigns = len(campaigns.data) if campaigns.data else 0
            campaign_ids = [c['id'] for c in campaigns.data or []]
            
            relevant_tasks = [t for t in campaign_tasks.data or [] if t['campaign_id'] in campaign_ids]
            completed_tasks = sum(1 for t in relevant_tasks if t.get('status') == 'completed')
            total_tasks = len(relevant_tasks)
            
            avg_completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            campaign_metrics = {
                "total_campaigns": total_campaigns,
                "avg_completion_rate": avg_completion_rate
            }
            
            return {
                "blog_performance": blog_performance,
                "agent_performance": agent_performance.data,
                "campaign_metrics": campaign_metrics,
                "date_range": f"Last {days} days"
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard analytics: {str(e)}")
            return {}

    # ===== UTILITY METHODS =====
    
    def health_check(self) -> Dict[str, Any]:
        """Check database connectivity and performance with fallback for permission issues."""
        health_data = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "supabase_connection": {"status": "unknown"},
            "direct_db_connection": {"status": "unknown"},
            "tables_accessible": {},
            "permissions_issue": False
        }
        
        # Test Supabase connection
        try:
            start_time = time.time()
            # Try a simple select that should work with basic permissions
            supabase_test = self.supabase.table("blog_posts").select("id").limit(1).execute()
            supabase_time = time.time() - start_time
            
            health_data["supabase_connection"] = {
                "status": "ok",
                "response_time_ms": int(supabase_time * 1000)
            }
            
        except Exception as e:
            health_data["supabase_connection"] = {
                "status": "error",
                "error": str(e)
            }
            if "permission denied" in str(e).lower():
                health_data["permissions_issue"] = True
        
        # Test direct DB connection
        try:
            start_time = time.time()
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            db_time = time.time() - start_time
            
            health_data["direct_db_connection"] = {
                "status": "ok", 
                "response_time_ms": int(db_time * 1000)
            }
            
        except Exception as e:
            health_data["direct_db_connection"] = {
                "status": "error",
                "error": str(e)
            }
            if "permission denied" in str(e).lower():
                health_data["permissions_issue"] = True
        
        # Test access to new analytics tables (if they exist)
        analytics_tables = [
            "agent_performance", "agent_decisions", "blog_analytics",
            "marketing_metrics", "content_optimization", "seo_metadata",
            "content_variants", "agent_feedback"
        ]
        
        for table in analytics_tables:
            try:
                # Try to count rows in each table
                response = self.supabase.table(table).select("id", count="exact").limit(0).execute()
                health_data["tables_accessible"][table] = {
                    "status": "accessible",
                    "count": response.count if hasattr(response, 'count') else 0
                }
            except Exception as e:
                error_msg = str(e).lower()
                if "does not exist" in error_msg or "relation" in error_msg:
                    health_data["tables_accessible"][table] = {"status": "not_created"}
                elif "permission denied" in error_msg:
                    health_data["tables_accessible"][table] = {"status": "permission_denied"}
                    health_data["permissions_issue"] = True
                else:
                    health_data["tables_accessible"][table] = {"status": "error", "error": str(e)}
        
        # Determine overall health status
        if health_data["permissions_issue"]:
            health_data["status"] = "permissions_error"
            health_data["recommendation"] = "Run fix_database_permissions.sql in Supabase SQL Editor as superuser"
        elif health_data["supabase_connection"]["status"] == "ok" or health_data["direct_db_connection"]["status"] == "ok":
            accessible_tables = sum(1 for t in health_data["tables_accessible"].values() if t.get("status") == "accessible")
            if accessible_tables >= 4:  # At least half of analytics tables working
                health_data["status"] = "healthy"
            elif accessible_tables > 0:
                health_data["status"] = "partial"
                health_data["recommendation"] = "Some analytics tables missing - run database_improvements_fixed.sql"
            else:
                health_data["status"] = "basic"
                health_data["recommendation"] = "Basic connectivity OK, but analytics tables not found"
        else:
            health_data["status"] = "unhealthy"
        
        return health_data

# Global database service instance (lazy initialization)
def get_db_service():
    """Get the global database service instance with lazy initialization."""
    return DatabaseService()