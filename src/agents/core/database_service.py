"""
Enhanced Database Service Layer for AI Agents
Provides centralized database operations with performance tracking,
agent decision logging, and marketing analytics.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
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
    agent_name: str
    agent_type: str
    execution_id: str
    blog_post_id: Optional[str] = None
    campaign_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None  # milliseconds
    status: str = "running"  # pending, running, success, error, timeout, cancelled
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict] = None

@dataclass
class AgentDecision:
    performance_id: str
    decision_point: str
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    alternatives_considered: Optional[Dict] = None
    execution_time: Optional[int] = None  # milliseconds
    tokens_used: Optional[int] = None
    decision_latency: Optional[float] = None  # seconds
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict] = None

@dataclass
class BlogAnalyticsData:
    blog_id: str
    word_count: int = 0
    reading_time: int = 0
    seo_score: Optional[float] = None
    geo_optimized: bool = False
    geo_score: Optional[int] = None
    content_type: Optional[str] = None
    creation_time_ms: Optional[int] = None
    views: int = 0
    unique_visitors: int = 0
    engagement_rate: float = 0.0
    social_shares: int = 0
    comments_count: int = 0
    conversion_rate: float = 0.0

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
                logger.info(f"Performance logged for {metrics.agent_name}:{metrics.agent_type}")
                return response.data[0]["id"]
            else:
                # For local PostgreSQL, just log to console
                logger.info(f"Performance logged for {metrics.agent_name}:{metrics.agent_type} - {data}")
                return "local_log"
            
        except Exception as e:
            logger.error(f"Failed to log agent performance: {str(e)}")
            raise

    def log_agent_decision(self, decision: AgentDecision) -> str:
        """Log agent decision for learning and audit purposes."""
        try:
            data = {
                "performance_id": decision.performance_id,
                "decision_point": decision.decision_point,
                "input_data": decision.input_data,
                "output_data": decision.output_data,
                "reasoning": decision.reasoning,
                "confidence_score": decision.confidence_score,
                "alternatives_considered": decision.alternatives_considered,
                "execution_time": decision.execution_time,
                "tokens_used": decision.tokens_used,
                "decision_latency": decision.decision_latency,
                "timestamp": decision.timestamp.isoformat() if decision.timestamp else datetime.now(timezone.utc).isoformat(),
                "metadata": decision.metadata
            }
            
            if self.use_supabase and self.supabase:
                response = self.supabase.table("agent_decisions").insert(data).execute()
                logger.info(f"Decision logged for {decision.decision_point}")
                return response.data[0]["id"]
            else:
                # For local PostgreSQL, just log to console
                logger.info(f"Decision logged for {decision.decision_point} - {data}")
                return "local_log"
            
        except Exception as e:
            logger.error(f"Failed to log agent decision: {str(e)}")
            raise

    def get_agent_performance_analytics(self, agent_type: Optional[str] = None, 
                                       days: int = 30) -> List[Dict]:
        """Get agent performance analytics using real data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Build query with optional agent type filter
                base_query = """
                    SELECT id, agent_name, agent_type, execution_id, start_time, end_time,
                           duration, status, input_tokens, output_tokens, total_tokens, 
                           cost, error_message, retry_count, created_at
                    FROM agent_performance 
                    WHERE start_time >= %s
                """
                params = [cutoff_date]
                
                if agent_type:
                    base_query += " AND agent_type = %s"
                    params.append(agent_type)
                
                base_query += " ORDER BY start_time DESC LIMIT 50"
                
                cur.execute(base_query, params)
                
                performance_data = []
                for row in cur.fetchall():
                    # Map database columns to expected format
                    performance_data.append({
                        'id': row[0],
                        'agent_name': row[1],
                        'agent_type': row[2],
                        'task_type': f"{row[2]}_task",  # Infer task type from agent type
                        'execution_time_ms': row[6] or 0,
                        'success_rate': 1.0 if row[7] == 'success' else 0.0,
                        'quality_score': 0.85,  # Default quality score since not tracked yet
                        'input_tokens': row[8] or 0,
                        'output_tokens': row[9] or 0,
                        'cost_usd': float(row[11]) if row[11] else 0.0,
                        'created_at': row[14].isoformat() if row[14] else None,
                        'status': row[7]
                    })
                
                return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get agent performance analytics: {str(e)}")
            # Return empty list instead of mock data
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
        """Get comprehensive analytics for dashboard using real data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get real blog post counts
                cur.execute("SELECT COUNT(*) FROM blog_posts")
                total_blogs = cur.fetchone()[0] or 0
                
                # Get real campaign counts  
                cur.execute("SELECT COUNT(*) FROM campaigns")
                total_campaigns = cur.fetchone()[0] or 0
                
                # Get real agent performance data
                cur.execute("SELECT COUNT(*) FROM agent_performance")
                total_agent_executions = cur.fetchone()[0] or 0
                
                # Calculate real success rate
                if total_agent_executions > 0:
                    cur.execute("SELECT COUNT(*) FROM agent_performance WHERE status = 'success'")
                    successful_executions = cur.fetchone()[0] or 0
                    success_rate = successful_executions / total_agent_executions
                else:
                    success_rate = 0.0
                
                # Get agent performance by type
                cur.execute("""
                    SELECT agent_type, COUNT(*) as execution_count,
                           AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) as avg_success_rate
                    FROM agent_performance 
                    WHERE start_time >= %s
                    GROUP BY agent_type
                    ORDER BY avg_success_rate DESC, execution_count DESC
                """, (cutoff_date,))
                
                top_performing_agents = []
                for row in cur.fetchall():
                    top_performing_agents.append({
                        'agent_type': row[0],
                        'execution_count': int(row[1]),
                        'avg_success_rate': float(row[2])
                    })
                
                # Get recent performance (daily aggregates)
                cur.execute("""
                    SELECT DATE(start_time) as date,
                           COUNT(*) as executions,
                           AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM agent_performance 
                    WHERE start_time >= %s
                    GROUP BY DATE(start_time)
                    ORDER BY date DESC
                    LIMIT 7
                """, (cutoff_date,))
                
                recent_performance = []
                for row in cur.fetchall():
                    recent_performance.append({
                        'date': row[0].strftime('%Y-%m-%d'),
                        'executions': int(row[1]),
                        'success_rate': float(row[2])
                    })
                recent_performance.reverse()  # Show chronologically
                
                # Get blog performance data
                cur.execute("""
                    SELECT bp.id, bp.title, bp.word_count, bp.reading_time
                    FROM blog_posts bp 
                    WHERE bp.status = 'published'
                    ORDER BY bp.created_at DESC
                    LIMIT 5
                """)
                
                blog_performance = []
                for row in cur.fetchall():
                    # Since we don't have real view/engagement data yet, we'll note this
                    blog_performance.append({
                        'blog_id': row[0],
                        'title': row[1],
                        'views': 0,  # No real view data available
                        'engagement_rate': 0.0  # No real engagement data available
                    })
                
                return {
                    "total_blogs": total_blogs,
                    "total_campaigns": total_campaigns,
                    "total_agent_executions": total_agent_executions,
                    "success_rate": round(success_rate, 3),
                    "top_performing_agents": top_performing_agents,
                    "recent_performance": recent_performance,
                    "blog_performance": blog_performance,
                    "data_notes": {
                        "blog_views": "View tracking not yet implemented",
                        "engagement_metrics": "Engagement tracking not yet implemented"
                    }
                }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard analytics: {str(e)}")
            # Return empty state instead of mock data
            return {
                "total_blogs": 0,
                "total_campaigns": 0,
                "total_agent_executions": 0,
                "success_rate": 0.0,
                "top_performing_agents": [],
                "recent_performance": [],
                "blog_performance": [],
                "error": "Database connection failed",
                "data_notes": {
                    "status": "No data available - database connection issue"
                }
            }
    
    def _get_mock_dashboard_analytics(self, days: int) -> Dict:
        """Return consistent analytics data for demo purposes."""
        
        # Consistent agent performance data
        top_performing_agents = [
            {'agent_type': 'planner', 'execution_count': 23, 'avg_success_rate': 0.913},
            {'agent_type': 'researcher', 'execution_count': 25, 'avg_success_rate': 0.880},
            {'agent_type': 'writer', 'execution_count': 6, 'avg_success_rate': 0.833},
            {'agent_type': 'editor', 'execution_count': 24, 'avg_success_rate': 0.875},
            {'agent_type': 'seo', 'execution_count': 7, 'avg_success_rate': 0.857},
            {'agent_type': 'image_prompt_generator', 'execution_count': 19, 'avg_success_rate': 0.895}
        ]
        
        # Consistent recent performance (last 7 days)
        recent_performance = [
            {'date': '2025-08-07', 'executions': 12, 'success_rate': 0.833},
            {'date': '2025-08-08', 'executions': 18, 'success_rate': 0.889},
            {'date': '2025-08-09', 'executions': 15, 'success_rate': 0.867},
            {'date': '2025-08-10', 'executions': 21, 'success_rate': 0.905},
            {'date': '2025-08-11', 'executions': 16, 'success_rate': 0.875},
            {'date': '2025-08-12', 'executions': 14, 'success_rate': 0.857},
            {'date': '2025-08-13', 'executions': 20, 'success_rate': 0.900}
        ]
        
        # Consistent blog performance data
        blog_performance = [
            {'blog_id': 'blog-1', 'title': 'AI Revolution in Content Marketing', 'views': 2847, 'engagement_rate': 0.052},
            {'blog_id': 'blog-2', 'title': 'The Future of Marketing Automation', 'views': 2156, 'engagement_rate': 0.048},
            {'blog_id': 'blog-3', 'title': 'Building Better Customer Experiences', 'views': 3921, 'engagement_rate': 0.067},
            {'blog_id': 'blog-4', 'title': 'Data-Driven Marketing Strategies', 'views': 1834, 'engagement_rate': 0.043},
            {'blog_id': 'blog-5', 'title': 'Content Creation with AI Agents', 'views': 2673, 'engagement_rate': 0.055}
        ]
        
        # Calculate totals from real data
        total_executions = sum(agent['execution_count'] for agent in top_performing_agents)
        total_successful = sum(int(agent['execution_count'] * agent['avg_success_rate']) for agent in top_performing_agents)
        overall_success_rate = total_successful / total_executions if total_executions > 0 else 0
        
        return {
            "total_blogs": 12,
            "total_campaigns": 8, 
            "total_agent_executions": total_executions,
            "success_rate": round(overall_success_rate, 3),
            "top_performing_agents": top_performing_agents,
            "recent_performance": recent_performance,
            "blog_performance": blog_performance
        }
    
    def _get_mock_agent_performance(self, agent_type: Optional[str] = None, days: int = 30) -> List[Dict]:
        """Return consistent agent performance data for demo purposes."""
        
        # Predefined consistent performance records
        all_performance_data = [
            {'id': 'perf-1001', 'agent_type': 'planner', 'task_type': 'content_planning', 'execution_time_ms': 3420, 'success_rate': 0.91, 'quality_score': 0.88, 'input_tokens': 245, 'output_tokens': 189, 'cost_usd': 0.0009, 'created_at': '2025-08-13T10:15:00'},
            {'id': 'perf-1002', 'agent_type': 'researcher', 'task_type': 'topic_research', 'execution_time_ms': 12760, 'success_rate': 0.87, 'quality_score': 0.85, 'input_tokens': 567, 'output_tokens': 834, 'cost_usd': 0.0024, 'created_at': '2025-08-13T09:30:00'},
            {'id': 'perf-1003', 'agent_type': 'writer', 'task_type': 'blog_writing', 'execution_time_ms': 18950, 'success_rate': 0.82, 'quality_score': 0.79, 'input_tokens': 1234, 'output_tokens': 2156, 'cost_usd': 0.0058, 'created_at': '2025-08-13T08:45:00'},
            {'id': 'perf-1004', 'agent_type': 'editor', 'task_type': 'content_editing', 'execution_time_ms': 8340, 'success_rate': 0.89, 'quality_score': 0.92, 'input_tokens': 1890, 'output_tokens': 1456, 'cost_usd': 0.0057, 'created_at': '2025-08-13T08:00:00'},
            {'id': 'perf-1005', 'agent_type': 'seo', 'task_type': 'seo_optimization', 'execution_time_ms': 5670, 'success_rate': 0.85, 'quality_score': 0.81, 'input_tokens': 445, 'output_tokens': 267, 'cost_usd': 0.0012, 'created_at': '2025-08-12T16:20:00'},
            {'id': 'perf-1006', 'agent_type': 'image_prompt_generator', 'task_type': 'image_prompts', 'execution_time_ms': 2890, 'success_rate': 0.93, 'quality_score': 0.87, 'input_tokens': 123, 'output_tokens': 98, 'cost_usd': 0.0004, 'created_at': '2025-08-12T15:45:00'},
            {'id': 'perf-1007', 'agent_type': 'planner', 'task_type': 'campaign_strategy', 'execution_time_ms': 6780, 'success_rate': 0.88, 'quality_score': 0.84, 'input_tokens': 356, 'output_tokens': 445, 'cost_usd': 0.0014, 'created_at': '2025-08-12T14:30:00'},
            {'id': 'perf-1008', 'agent_type': 'researcher', 'task_type': 'competitor_analysis', 'execution_time_ms': 15230, 'success_rate': 0.91, 'quality_score': 0.89, 'input_tokens': 789, 'output_tokens': 1123, 'cost_usd': 0.0033, 'created_at': '2025-08-12T13:15:00'},
            {'id': 'perf-1009', 'agent_type': 'writer', 'task_type': 'social_content', 'execution_time_ms': 4560, 'success_rate': 0.86, 'quality_score': 0.83, 'input_tokens': 234, 'output_tokens': 345, 'cost_usd': 0.0010, 'created_at': '2025-08-11T17:00:00'},
            {'id': 'perf-1010', 'agent_type': 'editor', 'task_type': 'proofreading', 'execution_time_ms': 3450, 'success_rate': 0.94, 'quality_score': 0.96, 'input_tokens': 567, 'output_tokens': 423, 'cost_usd': 0.0017, 'created_at': '2025-08-11T16:30:00'},
            {'id': 'perf-1011', 'agent_type': 'seo', 'task_type': 'keyword_research', 'execution_time_ms': 7890, 'success_rate': 0.83, 'quality_score': 0.78, 'input_tokens': 345, 'output_tokens': 289, 'cost_usd': 0.0011, 'created_at': '2025-08-11T15:45:00'},
            {'id': 'perf-1012', 'agent_type': 'image_prompt_generator', 'task_type': 'visual_concepts', 'execution_time_ms': 1980, 'success_rate': 0.89, 'quality_score': 0.85, 'input_tokens': 89, 'output_tokens': 156, 'cost_usd': 0.0004, 'created_at': '2025-08-10T14:20:00'},
            {'id': 'perf-1013', 'agent_type': 'planner', 'task_type': 'content_calendar', 'execution_time_ms': 9870, 'success_rate': 0.92, 'quality_score': 0.90, 'input_tokens': 456, 'output_tokens': 678, 'cost_usd': 0.0019, 'created_at': '2025-08-10T11:30:00'},
            {'id': 'perf-1014', 'agent_type': 'researcher', 'task_type': 'trend_analysis', 'execution_time_ms': 11230, 'success_rate': 0.88, 'quality_score': 0.86, 'input_tokens': 678, 'output_tokens': 934, 'cost_usd': 0.0028, 'created_at': '2025-08-09T16:45:00'},
            {'id': 'perf-1015', 'agent_type': 'writer', 'task_type': 'email_campaign', 'execution_time_ms': 6540, 'success_rate': 0.84, 'quality_score': 0.81, 'input_tokens': 345, 'output_tokens': 567, 'cost_usd': 0.0016, 'created_at': '2025-08-09T10:15:00'}
        ]
        
        # Filter by agent type if specified
        if agent_type:
            performance_data = [p for p in all_performance_data if p['agent_type'] == agent_type]
        else:
            performance_data = all_performance_data
        
        return performance_data[:min(20, len(performance_data))]

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