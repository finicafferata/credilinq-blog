"""
Campaign Database Service - Extended database operations for campaign orchestration.

This service extends the base DatabaseService with campaign-specific operations,
integrating with the new campaign-centric database schema from Week 1.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging

from ..core.database_service import DatabaseService, AgentPerformanceMetrics, AgentDecision
from .types import (
    CampaignWithTasks, CampaignTask, TaskStatus, WorkflowExecutionCreate, WorkflowStatus
)

logger = logging.getLogger(__name__)

@dataclass
class AgentPerformanceMetrics:
    """Enhanced agent performance metrics for campaign orchestration."""
    agent_name: str
    agent_type: str
    execution_id: str
    campaign_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_execution_id: Optional[str] = None
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
    task_type: Optional[str] = None
    quality_score: Optional[float] = None
    metadata: Optional[Dict] = None

class CampaignDatabaseService(DatabaseService):
    """
    Extended database service for campaign-specific operations.
    
    Integrates with the new campaign orchestration schema including:
    - campaigns, campaign_orchestrators, campaign_strategies
    - campaign_workflows, campaign_workflow_steps, campaign_content
    - campaign_calendar, campaign_analytics, agent_orchestration_performance
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initialized CampaignDatabaseService with campaign schema support")
    
    async def get_campaign_with_tasks(self, campaign_id: str) -> Optional[CampaignWithTasks]:
        """
        Fetch campaign with all associated tasks and dependencies.
        
        Args:
            campaign_id: ID of the campaign to fetch
            
        Returns:
            CampaignWithTasks: Campaign data with tasks, or None if not found
        """
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get campaign details
                cur.execute("""
                    SELECT id, name, description, status, priority, campaign_type,
                           orchestrator_id, strategy_id, campaign_data, 
                           created_at, updated_at
                    FROM campaigns 
                    WHERE id = %s
                """, (campaign_id,))
                
                campaign_row = cur.fetchone()
                if not campaign_row:
                    logger.warning(f"Campaign {campaign_id} not found")
                    return None
                
                # Get campaign tasks (simulated for now since schema is new)
                cur.execute("""
                    SELECT cc.id, cc.title, cc.content_type, cc.platform, 
                           cc.status, cc.created_at, cc.metadata
                    FROM campaign_content cc
                    WHERE cc.campaign_id = %s
                    ORDER BY cc.created_at ASC
                """, (campaign_id,))
                
                content_rows = cur.fetchall()
                
                # Convert content to tasks
                tasks = []
                for content_row in content_rows:
                    task = CampaignTask(
                        id=str(content_row[0]),
                        campaign_id=campaign_id,
                        task_type=content_row[2] or "content_creation",
                        agent_type=self._infer_agent_type(content_row[2]),
                        input_data={
                            "title": content_row[1],
                            "platform": content_row[3],
                            "metadata": content_row[6] or {}
                        },
                        status=self._map_content_status_to_task_status(content_row[4]),
                        created_at=content_row[5] or datetime.utcnow(),
                        metadata=content_row[6] or {}
                    )
                    tasks.append(task)
                
                # Create campaign object
                from .campaign_orchestrator import CampaignType
                campaign_type = CampaignType.BLOG_CREATION  # Default for now
                
                campaign = CampaignWithTasks(
                    id=str(campaign_row[0]),
                    name=campaign_row[1] or f"Campaign {campaign_id}",
                    description=campaign_row[2] or "Campaign description",
                    campaign_type=campaign_type,
                    status=campaign_row[3] or "draft",
                    orchestrator_id=str(campaign_row[6]) if campaign_row[6] else None,
                    strategy_id=str(campaign_row[7]) if campaign_row[7] else None,
                    tasks=tasks,
                    metadata=campaign_row[8] or {},
                    created_at=campaign_row[9] or datetime.utcnow(),
                    updated_at=campaign_row[10] or datetime.utcnow()
                )
                
                logger.info(f"Retrieved campaign {campaign_id} with {len(tasks)} tasks")
                return campaign
                
        except Exception as e:
            logger.error(f"Failed to get campaign {campaign_id}: {str(e)}")
            return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, result: Optional[Any]) -> bool:
        """
        Update individual task status and results.
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            result: Task execution result (optional)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Update campaign content status (mapping task to content)
                status_mapping = {
                    TaskStatus.PENDING: "draft",
                    TaskStatus.IN_PROGRESS: "in_progress", 
                    TaskStatus.COMPLETED: "published",
                    TaskStatus.FAILED: "failed",
                    TaskStatus.CANCELLED: "cancelled"
                }
                
                content_status = status_mapping.get(status, "draft")
                
                cur.execute("""
                    UPDATE campaign_content 
                    SET status = %s, updated_at = %s
                    WHERE id = %s
                """, (content_status, datetime.utcnow(), task_id))
                
                updated_rows = cur.rowcount
                
                if updated_rows > 0:
                    logger.info(f"Updated task {task_id} status to {status.value}")
                    return True
                else:
                    logger.warning(f"Task {task_id} not found for status update")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update task {task_id} status: {str(e)}")
            return False
    
    async def log_workflow_execution(self, workflow_execution: WorkflowExecutionCreate) -> str:
        """
        Log complete workflow execution to database.
        
        Args:
            workflow_execution: Workflow execution data to log
            
        Returns:
            str: ID of the created workflow execution record
        """
        try:
            execution_id = str(uuid.uuid4())
            
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Insert into campaign_workflows table
                cur.execute("""
                    INSERT INTO campaign_workflows (
                        id, campaign_id, workflow_type, status, 
                        input_data, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    execution_id,
                    workflow_execution.campaign_id,
                    workflow_execution.workflow_type,
                    workflow_execution.status.value,
                    workflow_execution.input_data,
                    workflow_execution.metadata,
                    datetime.utcnow()
                ))
                
                logger.info(f"Logged workflow execution {execution_id} for campaign {workflow_execution.campaign_id}")
                return execution_id
                
        except Exception as e:
            logger.error(f"Failed to log workflow execution: {str(e)}")
            # Return a fallback ID to avoid breaking the workflow
            return str(uuid.uuid4())
    
    async def get_agent_performance_metrics(self, agent_type: str, timeframe: datetime) -> Dict:
        """
        Fetch performance analytics from AgentPerformance table.
        
        Args:
            agent_type: Type of agent to get metrics for
            timeframe: Start time for metrics collection
            
        Returns:
            Dict: Performance metrics and analytics
        """
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get performance data for agent type
                cur.execute("""
                    SELECT COUNT(*) as total_executions,
                           AVG(duration) as avg_duration_ms,
                           AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) as success_rate,
                           SUM(input_tokens) as total_input_tokens,
                           SUM(output_tokens) as total_output_tokens,
                           AVG(cost) as avg_cost
                    FROM agent_performance 
                    WHERE agent_type = %s AND start_time >= %s
                """, (agent_type, timeframe))
                
                metrics_row = cur.fetchone()
                
                if metrics_row:
                    return {
                        "agent_type": agent_type,
                        "total_executions": int(metrics_row[0] or 0),
                        "avg_duration_ms": float(metrics_row[1] or 0),
                        "success_rate": float(metrics_row[2] or 0),
                        "total_input_tokens": int(metrics_row[3] or 0),
                        "total_output_tokens": int(metrics_row[4] or 0),
                        "avg_cost": float(metrics_row[5] or 0),
                        "timeframe_start": timeframe.isoformat()
                    }
                else:
                    return {
                        "agent_type": agent_type,
                        "total_executions": 0,
                        "message": "No performance data found"
                    }
                
        except Exception as e:
            logger.error(f"Failed to get agent performance metrics: {str(e)}")
            return {"agent_type": agent_type, "error": str(e)}
    
    async def log_campaign_performance(self, metrics: AgentPerformanceMetrics) -> str:
        """
        Log agent performance metrics with campaign context.
        
        Args:
            metrics: Enhanced performance metrics with campaign data
            
        Returns:
            str: ID of the logged performance record
        """
        try:
            performance_id = str(uuid.uuid4())
            
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Insert into agent_performance table with campaign context
                cur.execute("""
                    INSERT INTO agent_performance (
                        id, agent_name, agent_type, execution_id, campaign_id,
                        start_time, end_time, duration, status, input_tokens,
                        output_tokens, total_tokens, cost, error_message,
                        retry_count, metadata, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    performance_id, metrics.agent_name, metrics.agent_type,
                    metrics.execution_id, metrics.campaign_id, metrics.start_time,
                    metrics.end_time, metrics.duration, metrics.status,
                    metrics.input_tokens, metrics.output_tokens, metrics.total_tokens,
                    metrics.cost, metrics.error_message, metrics.retry_count,
                    metrics.metadata, datetime.utcnow()
                ))
                
                logger.info(f"Logged campaign performance for agent {metrics.agent_name}")
                return performance_id
                
        except Exception as e:
            logger.error(f"Failed to log campaign performance: {str(e)}")
            return "failed_log"
    
    async def get_campaign_orchestrators(self, orchestrator_type: Optional[str] = None) -> List[Dict]:
        """
        Get available campaign orchestrators from the database.
        
        Args:
            orchestrator_type: Filter by orchestrator type (optional)
            
        Returns:
            List[Dict]: Available orchestrators
        """
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                query = """
                    SELECT id, name, orchestrator_type, capabilities, 
                           configuration, status, created_at
                    FROM campaign_orchestrators 
                    WHERE status = 'active'
                """
                params = []
                
                if orchestrator_type:
                    query += " AND orchestrator_type = %s"
                    params.append(orchestrator_type)
                
                query += " ORDER BY created_at DESC"
                
                cur.execute(query, params)
                
                orchestrators = []
                for row in cur.fetchall():
                    orchestrators.append({
                        "id": str(row[0]),
                        "name": row[1],
                        "orchestrator_type": row[2],
                        "capabilities": row[3] or [],
                        "configuration": row[4] or {},
                        "status": row[5],
                        "created_at": row[6]
                    })
                
                logger.info(f"Retrieved {len(orchestrators)} campaign orchestrators")
                return orchestrators
                
        except Exception as e:
            logger.error(f"Failed to get campaign orchestrators: {str(e)}")
            return []
    
    async def get_campaign_strategies(self, strategy_type: Optional[str] = None) -> List[Dict]:
        """
        Get available campaign strategies from the database.
        
        Args:
            strategy_type: Filter by strategy type (optional)
            
        Returns:
            List[Dict]: Available strategies
        """
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                query = """
                    SELECT id, name, strategy_type, description, parameters,
                           is_template, created_at
                    FROM campaign_strategies
                    WHERE is_active = true
                """
                params = []
                
                if strategy_type:
                    query += " AND strategy_type = %s"
                    params.append(strategy_type)
                
                query += " ORDER BY is_template DESC, created_at DESC"
                
                cur.execute(query, params)
                
                strategies = []
                for row in cur.fetchall():
                    strategies.append({
                        "id": str(row[0]),
                        "name": row[1],
                        "strategy_type": row[2],
                        "description": row[3],
                        "parameters": row[4] or {},
                        "is_template": row[5],
                        "created_at": row[6]
                    })
                
                logger.info(f"Retrieved {len(strategies)} campaign strategies")
                return strategies
                
        except Exception as e:
            logger.error(f"Failed to get campaign strategies: {str(e)}")
            return []
    
    def _infer_agent_type(self, content_type: str) -> str:
        """Infer appropriate agent type from content type."""
        content_to_agent_mapping = {
            "blog_post": "writer",
            "social_media": "social_media",
            "email": "writer", 
            "image": "image_prompt_generator",
            "seo": "seo",
            "video": "content_generator"
        }
        
        return content_to_agent_mapping.get(content_type, "content_generator")
    
    def _map_content_status_to_task_status(self, content_status: str) -> TaskStatus:
        """Map content status to task status."""
        status_mapping = {
            "draft": TaskStatus.PENDING,
            "in_progress": TaskStatus.IN_PROGRESS,
            "published": TaskStatus.COMPLETED,
            "failed": TaskStatus.FAILED,
            "cancelled": TaskStatus.CANCELLED
        }
        
        return status_mapping.get(content_status, TaskStatus.PENDING)
    
    async def health_check_campaign_schema(self) -> Dict[str, Any]:
        """
        Check health of campaign-specific database schema.
        
        Returns:
            Dict: Health status of campaign tables and functions
        """
        health_data = {
            "campaign_schema_status": "unknown",
            "tables_status": {},
            "functions_status": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check campaign-specific tables
        campaign_tables = [
            "campaigns", "campaign_orchestrators", "campaign_strategies",
            "campaign_workflows", "campaign_workflow_steps", "campaign_content",
            "campaign_calendar", "campaign_analytics", "agent_orchestration_performance"
        ]
        
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                
                for table in campaign_tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        health_data["tables_status"][table] = {
                            "status": "accessible",
                            "record_count": count
                        }
                    except Exception as e:
                        health_data["tables_status"][table] = {
                            "status": "error",
                            "error": str(e)
                        }
                
                # Check campaign-specific functions
                campaign_functions = [
                    "calculate_campaign_progress",
                    "update_campaign_status"
                ]
                
                for function in campaign_functions:
                    try:
                        cur.execute(f"SELECT proname FROM pg_proc WHERE proname = '{function}'")
                        exists = cur.fetchone() is not None
                        health_data["functions_status"][function] = {
                            "status": "exists" if exists else "missing"
                        }
                    except Exception as e:
                        health_data["functions_status"][function] = {
                            "status": "error",
                            "error": str(e)
                        }
                
                # Determine overall status
                accessible_tables = sum(1 for t in health_data["tables_status"].values() 
                                     if t.get("status") == "accessible")
                existing_functions = sum(1 for f in health_data["functions_status"].values()
                                       if f.get("status") == "exists")
                
                if accessible_tables >= 6 and existing_functions >= 1:
                    health_data["campaign_schema_status"] = "healthy"
                elif accessible_tables >= 3:
                    health_data["campaign_schema_status"] = "partial"
                else:
                    health_data["campaign_schema_status"] = "unhealthy"
                
        except Exception as e:
            health_data["campaign_schema_status"] = "error"
            health_data["error"] = str(e)
        
        return health_data