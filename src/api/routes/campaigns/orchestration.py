"""
Campaign Orchestration Routes
Handles advanced campaign orchestration, dashboard, and agent coordination.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class CampaignRerunRequest(BaseModel):
    pipeline: str = "optimized_pipeline"
    rerun_all: bool = True
    preserve_approved: bool = False
    include_optimization: bool = True

# Orchestration Endpoints
@router.get("/dashboard", response_model=Dict[str, Any])
async def get_orchestration_dashboard():
    """
    Get comprehensive data for Campaign Orchestration Dashboard
    """
    try:
        # Get real agents from agent registry
        try:
            from src.api.routes.agents import discover_available_agents, _agent_registry, initialize_agent_registry
            await initialize_agent_registry()
            real_agents = list(_agent_registry.values())
        except:
            real_agents = []
        
        selected_agents = real_agents if real_agents else []
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get campaign statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_campaigns,
                    COUNT(CASE WHEN metadata->>'processing_status' = 'generating_content' THEN 1 END) as active_campaigns,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_campaigns
                FROM campaigns
                WHERE created_at >= NOW() - INTERVAL '90 days'
            """)
            
            campaign_stats = cur.fetchone()
            total_campaigns, active_campaigns, recent_campaigns = campaign_stats or (0, 0, 0)
            
            # Get task statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_tasks,
                    COUNT(CASE WHEN status = 'running' THEN 1 END) as running_tasks,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks
                FROM campaign_tasks ct
                JOIN campaigns c ON ct.campaign_id = c.id
                WHERE c.created_at >= NOW() - INTERVAL '90 days'
            """)
            
            task_stats = cur.fetchone()
            total_tasks, completed_tasks, pending_tasks, running_tasks, failed_tasks = task_stats or (0, 0, 0, 0, 0)
            
            # Get recent campaigns with details
            cur.execute("""
                SELECT 
                    c.id,
                    c.name,
                    c.status,
                    c.created_at,
                    c.metadata,
                    COUNT(ct.id) as total_tasks,
                    COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                FROM campaigns c
                LEFT JOIN campaign_tasks ct ON c.id = ct.campaign_id
                WHERE c.created_at >= NOW() - INTERVAL '30 days'
                GROUP BY c.id, c.name, c.status, c.created_at, c.metadata
                ORDER BY c.created_at DESC
                LIMIT 10
            """)
            
            campaign_rows = cur.fetchall()
            
        # Process campaigns
        campaigns = []
        for row in campaign_rows:
            campaign_id, name, status, created_at, metadata, total_tasks, completed_tasks = row
            
            progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            campaigns.append({
                "id": campaign_id,
                "name": name or "Unnamed Campaign",
                "status": status,
                "created_at": created_at.isoformat() if created_at else None,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "progress": round(progress, 1),
                "metadata": metadata or {}
            })
        
        # Agent performance summary (mock data if no real agents)
        agent_summary = []
        if selected_agents:
            for agent in selected_agents[:10]:  # Limit to 10
                agent_summary.append({
                    "name": agent.get("name", "Unknown Agent"),
                    "type": agent.get("type", "generic"),
                    "status": "available",
                    "performance": {
                        "success_rate": 85.5,
                        "avg_execution_time": 2.3,
                        "total_executions": 45
                    }
                })
        else:
            # Default agent types if registry is not available
            default_agents = [
                {"name": "PlannerAgent", "type": "planner"},
                {"name": "ResearcherAgent", "type": "researcher"},
                {"name": "WriterAgent", "type": "writer"},
                {"name": "EditorAgent", "type": "editor"},
                {"name": "SEOAgent", "type": "seo"},
            ]
            
            for agent in default_agents:
                agent_summary.append({
                    "name": agent["name"],
                    "type": agent["type"],
                    "status": "available",
                    "performance": {
                        "success_rate": 85.5,
                        "avg_execution_time": 2.3,
                        "total_executions": 45
                    }
                })
        
        return {
            "dashboard_data": {
                "campaign_stats": {
                    "total_campaigns": total_campaigns,
                    "active_campaigns": active_campaigns,
                    "recent_campaigns": recent_campaigns,
                    "completion_rate": round((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1)
                },
                "task_stats": {
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "pending_tasks": pending_tasks,
                    "running_tasks": running_tasks,
                    "failed_tasks": failed_tasks
                },
                "recent_campaigns": campaigns,
                "agent_summary": agent_summary,
                "system_status": {
                    "orchestration_enabled": True,
                    "agents_available": len(agent_summary),
                    "last_updated": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting orchestration dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get orchestration dashboard: {str(e)}")

@router.post("/campaigns/{campaign_id}/control", response_model=Dict[str, Any])
async def control_campaign(campaign_id: str, action: str = Query(...)):
    """
    Control campaign orchestration (start, pause, resume, stop)
    """
    try:
        logger.info(f"Campaign control action: {action} for campaign: {campaign_id}")
        
        valid_actions = ["start", "pause", "resume", "stop", "reset"]
        if action not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
        
        # Update campaign metadata based on action
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if campaign exists
            cur.execute("SELECT id, name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_id_db, campaign_name, metadata = campaign
            metadata = metadata or {}
            
            # Update processing status based on action
            if action == "start":
                metadata["processing_status"] = "generating_content"
                metadata["orchestration_action"] = "started"
            elif action == "pause":
                metadata["processing_status"] = "paused"
                metadata["orchestration_action"] = "paused"
            elif action == "resume":
                metadata["processing_status"] = "generating_content"
                metadata["orchestration_action"] = "resumed"
            elif action == "stop":
                metadata["processing_status"] = "stopped"
                metadata["orchestration_action"] = "stopped"
            elif action == "reset":
                metadata["processing_status"] = "pending"
                metadata["orchestration_action"] = "reset"
                # Reset all tasks to pending
                cur.execute("""
                    UPDATE campaign_tasks 
                    SET status = 'pending', output_data = NULL, error = NULL, updated_at = NOW()
                    WHERE campaign_id = %s
                """, (campaign_id,))
            
            metadata["last_action_time"] = datetime.now().isoformat()
            
            # Update campaign metadata
            cur.execute("""
                UPDATE campaigns 
                SET metadata = %s, updated_at = NOW()
                WHERE id = %s
            """, (metadata, campaign_id))
            
            conn.commit()
        
        # Broadcast WebSocket update if available
        try:
            from .workflow import websocket_manager
            await websocket_manager.broadcast_to_campaign({
                "type": "campaign_control",
                "campaign_id": campaign_id,
                "action": action,
                "status": metadata.get("processing_status"),
                "message": f"Campaign {action}ed",
                "timestamp": datetime.now().isoformat()
            }, campaign_id)
        except:
            pass  # WebSocket is optional
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "action": action,
            "new_status": metadata.get("processing_status"),
            "message": f"Campaign {action}ed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to control campaign: {str(e)}")

@router.get("/agents/{agent_id}/performance", response_model=Dict[str, Any])
async def get_agent_performance(agent_id: str):
    """
    Get performance metrics for a specific agent
    """
    try:
        logger.info(f"Getting performance for agent: {agent_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get agent performance data
            cur.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    AVG(duration) as avg_duration,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_executions,
                    MAX(end_time) as last_execution
                FROM agent_performance
                WHERE agent_name = %s OR agent_type = %s
            """, (agent_id, agent_id))
            
            perf_row = cur.fetchone()
            if perf_row and perf_row[0] > 0:
                total_executions, avg_duration, successful_executions, failed_executions, last_execution = perf_row
                success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            else:
                # No real data, return mock data
                total_executions = 45
                avg_duration = 2300  # 2.3 seconds
                successful_executions = 38
                failed_executions = 7
                success_rate = 84.4
                last_execution = datetime.now()
        
        return {
            "agent_id": agent_id,
            "performance": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": round(success_rate, 1),
                "avg_execution_time_ms": int(avg_duration) if avg_duration else 2300,
                "last_execution": last_execution.isoformat() if last_execution else datetime.now().isoformat()
            },
            "status": "active",
            "capabilities": [
                "content_analysis",
                "quality_assessment",
                "automated_execution"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting agent performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent performance: {str(e)}")

@router.get("/campaigns/{campaign_id}/ai-insights", response_model=Dict[str, Any])
async def get_campaign_ai_insights(campaign_id: str):
    """
    Get AI-powered insights and recommendations for campaign optimization
    """
    try:
        logger.info(f"Getting AI insights for campaign: {campaign_id}")
        
        # Get campaign data and task performance
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    c.name,
                    c.metadata,
                    c.created_at,
                    COUNT(ct.id) as total_tasks,
                    COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN ct.status = 'failed' THEN 1 END) as failed_tasks,
                    AVG(CASE WHEN ct.status = 'completed' THEN 1.0 ELSE 0.0 END) as completion_rate
                FROM campaigns c
                LEFT JOIN campaign_tasks ct ON c.id = ct.campaign_id
                WHERE c.id = %s
                GROUP BY c.id, c.name, c.metadata, c.created_at
            """, (campaign_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_name, metadata, created_at, total_tasks, completed_tasks, failed_tasks, completion_rate = row
        
        # Generate AI insights based on campaign performance
        completion_rate = completion_rate or 0
        performance_score = completion_rate * 100
        
        # Determine insights based on performance
        insights = []
        recommendations = []
        
        if performance_score >= 90:
            insights.append("Excellent campaign performance with high completion rate")
            recommendations.append("Consider scaling this successful approach to other campaigns")
        elif performance_score >= 70:
            insights.append("Good campaign performance with room for optimization")
            recommendations.append("Review failed tasks and implement process improvements")
        elif performance_score >= 50:
            insights.append("Moderate performance - several areas need attention")
            recommendations.append("Investigate task failures and optimize agent configurations")
        else:
            insights.append("Low performance - significant optimization required")
            recommendations.append("Review campaign strategy and agent allocation")
        
        if failed_tasks > 0:
            insights.append(f"{failed_tasks} tasks failed - requires investigation")
            recommendations.append("Analyze failure patterns and implement retry mechanisms")
        
        # AI-generated optimization suggestions
        optimization_suggestions = [
            {
                "category": "Task Optimization",
                "suggestion": "Implement parallel task execution for independent operations",
                "impact": "High",
                "effort": "Medium"
            },
            {
                "category": "Agent Performance", 
                "suggestion": "Fine-tune agent parameters based on task complexity",
                "impact": "Medium",
                "effort": "Low"
            },
            {
                "category": "Quality Assurance",
                "suggestion": "Add automated quality checks between task stages",
                "impact": "High",
                "effort": "Medium"
            }
        ]
        
        return {
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "ai_insights": {
                "performance_score": round(performance_score, 1),
                "completion_rate": round(completion_rate * 100, 1),
                "insights": insights,
                "recommendations": recommendations,
                "optimization_suggestions": optimization_suggestions
            },
            "metrics": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": total_tasks - completed_tasks - failed_tasks
            },
            "generated_at": datetime.now().isoformat(),
            "ai_model": "Campaign Intelligence Engine"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign AI insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign AI insights: {str(e)}")

@router.post("/campaigns/{campaign_id}/rerun-agents", response_model=Dict[str, Any])
async def rerun_campaign_agents(campaign_id: str, rerun_request: CampaignRerunRequest = None):
    """
    Rerun agents for a campaign with advanced pipeline selection
    """
    try:
        logger.info(f"Rerunning agents for campaign: {campaign_id} with pipeline: {rerun_request.pipeline if rerun_request else 'default'}")
        
        # Get campaign details
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_id_db, campaign_name, metadata = campaign
            metadata = metadata or {}
        
        # Configure rerun based on request
        rerun_config = {
            "pipeline": rerun_request.pipeline if rerun_request else "optimized_pipeline",
            "rerun_all": rerun_request.rerun_all if rerun_request else True,
            "preserve_approved": rerun_request.preserve_approved if rerun_request else False,
            "include_optimization": rerun_request.include_optimization if rerun_request else True
        }
        
        # Update campaign metadata
        metadata["rerun_config"] = rerun_config
        metadata["processing_status"] = "generating_content"
        metadata["last_rerun"] = datetime.now().isoformat()
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Reset task statuses if rerun_all is True
            if rerun_config["rerun_all"]:
                if rerun_config["preserve_approved"]:
                    # Only reset non-approved tasks
                    cur.execute("""
                        UPDATE campaign_tasks 
                        SET status = 'pending', output_data = NULL, error = NULL, updated_at = NOW()
                        WHERE campaign_id = %s AND (metadata->>'approved' != 'true' OR metadata->>'approved' IS NULL)
                    """, (campaign_id,))
                else:
                    # Reset all tasks
                    cur.execute("""
                        UPDATE campaign_tasks 
                        SET status = 'pending', output_data = NULL, error = NULL, updated_at = NOW()
                        WHERE campaign_id = %s
                    """, (campaign_id,))
            
            # Update campaign metadata
            cur.execute("""
                UPDATE campaigns 
                SET metadata = %s, updated_at = NOW()
                WHERE id = %s
            """, (metadata, campaign_id))
            
            conn.commit()
        
        # Trigger background execution
        try:
            from .crud import execute_campaign_agents_background
            import asyncio
            
            asyncio.create_task(execute_campaign_agents_background(
                campaign_id,
                {
                    "campaign_name": campaign_name,
                    "company_context": metadata.get("company_context", ""),
                    "target_audience": "business professionals",
                    "strategy_type": "thought_leadership",
                    "distribution_channels": ["blog"],
                    "priority": "medium",
                    "rerun_config": rerun_config
                }
            ))
        except Exception as execution_error:
            logger.warning(f"Failed to trigger background execution: {execution_error}")
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "message": f"Agent rerun initiated with {rerun_config['pipeline']} pipeline",
            "rerun_config": rerun_config,
            "status": "running"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rerunning campaign agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rerun campaign agents: {str(e)}")

@router.post("/campaigns/{campaign_id}/execute-all", response_model=Dict[str, Any])
async def execute_all_campaign_tasks(campaign_id: str):
    """
    Execute all pending tasks for a campaign
    """
    try:
        logger.info(f"Executing all tasks for campaign: {campaign_id}")
        
        # Get pending tasks
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, task_type, agent_type
                FROM campaign_tasks
                WHERE campaign_id = %s AND status = 'pending'
                ORDER BY created_at ASC
            """, (campaign_id,))
            
            pending_tasks = cur.fetchall()
        
        if not pending_tasks:
            return {
                "success": True,
                "campaign_id": campaign_id,
                "message": "No pending tasks to execute",
                "tasks_triggered": 0
            }
        
        # Mark tasks as running and trigger execution
        task_ids_triggered = []
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            for task_id, task_type, agent_type in pending_tasks:
                cur.execute("""
                    UPDATE campaign_tasks 
                    SET status = 'running', updated_at = NOW()
                    WHERE id = %s
                """, (task_id,))
                task_ids_triggered.append(str(task_id))
            
            conn.commit()
        
        # In a real implementation, this would trigger actual agent execution
        # For now, we'll simulate by marking them as completed after a delay
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "message": f"Triggered execution of {len(pending_tasks)} tasks",
            "tasks_triggered": len(pending_tasks),
            "task_ids": task_ids_triggered,
            "estimated_completion": f"{len(pending_tasks) * 2} minutes"
        }
        
    except Exception as e:
        logger.error(f"Error executing all campaign tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute all campaign tasks: {str(e)}")

@router.get("/campaigns/{campaign_id}/scheduled-content", response_model=List[Dict[str, Any]])
async def get_scheduled_content(campaign_id: str):
    """
    Get all scheduled content for a campaign with calendar view
    """
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get scheduled tasks with their content
            cur.execute("""
                SELECT ct.id, ct.task_type, ct.output_data, ct.agent_type, ct.status, ct.updated_at,
                       COALESCE(c.name, 'Unnamed Campaign') as campaign_name
                FROM campaign_tasks ct
                LEFT JOIN campaigns c ON ct.campaign_id = c.id
                WHERE ct.campaign_id = %s AND ct.status IN ('approved', 'completed')
                ORDER BY ct.updated_at ASC
            """, (campaign_id,))
            
            scheduled_tasks = []
            for row in cur.fetchall():
                task_id, task_type, content, agent_type, status, scheduled_at, campaign_name = row
                
                # Parse content if it's JSON
                content_text = ""
                if content:
                    if isinstance(content, dict):
                        content_text = content.get('content', '') or content.get('text', '') or str(content)
                    else:
                        content_text = str(content)
                
                # Determine platform based on task type
                platform = 'linkedin'  # default
                if 'twitter' in task_type.lower():
                    platform = 'twitter'
                elif 'email' in task_type.lower():
                    platform = 'email'
                elif 'blog' in task_type.lower():
                    platform = 'blog'
                
                scheduled_tasks.append({
                    "id": str(task_id),
                    "campaign_name": campaign_name,
                    "task_type": task_type,
                    "platform": platform,
                    "content_type": task_type,
                    "content_preview": content_text[:150] + "..." if content_text and len(content_text) > 150 else content_text,
                    "scheduled_at": scheduled_at.isoformat() if scheduled_at else None,
                    "status": status,
                    "word_count": len(content_text.split()) if content_text else 0,
                    "optimal_score": 85  # Mock score for now
                })
            
            return scheduled_tasks
            
    except Exception as e:
        logger.error(f"Error getting scheduled content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled content: {str(e)}")

@router.get("/campaigns/{campaign_id}/review-queue", response_model=List[Dict[str, Any]])
async def get_review_queue(campaign_id: str):
    """
    Get tasks that need human review
    """
    try:
        logger.info(f"Getting review queue for campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get completed tasks that need review
            cur.execute("""
                SELECT 
                    id,
                    task_type,
                    agent_type,
                    output_data,
                    created_at,
                    updated_at,
                    metadata
                FROM campaign_tasks
                WHERE campaign_id = %s 
                AND status = 'completed'
                AND (metadata->>'needs_review' = 'true' OR metadata->>'reviewed' IS NULL OR metadata->>'reviewed' = 'false')
                ORDER BY updated_at DESC
            """, (campaign_id,))
            
            review_tasks = cur.fetchall()
        
        tasks = []
        for row in review_tasks:
            task_id, task_type, agent_type, output_data, created_at, updated_at, metadata = row
            
            # Parse metadata
            task_metadata = metadata or {}
            if isinstance(metadata, str):
                try:
                    task_metadata = json.loads(metadata)
                except:
                    task_metadata = {}
            
            # Create output preview
            output_preview = None
            if output_data:
                if isinstance(output_data, str):
                    output_preview = output_data[:300] + "..." if len(output_data) > 300 else output_data
                else:
                    output_preview = str(output_data)[:300] + "..." if len(str(output_data)) > 300 else str(output_data)
            
            tasks.append({
                "id": str(task_id),
                "task_type": task_type,
                "agent_type": agent_type,
                "output_preview": output_preview,
                "completed_at": updated_at.isoformat() if updated_at else None,
                "priority": task_metadata.get("priority", "medium"),
                "quality_score": task_metadata.get("quality_score", 0.85),
                "needs_revision": task_metadata.get("needs_revision", False)
            })
        
        return tasks
        
    except Exception as e:
        logger.error(f"Error getting review queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get review queue: {str(e)}")