"""
Autonomous Workflow Routes
Handles autonomous campaign workflows and status tracking.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Autonomous Workflow Endpoints
@router.post("/{campaign_id}/start", response_model=Dict[str, Any])
async def start_autonomous_workflow(campaign_id: str):
    """
    Start autonomous workflow for a campaign
    """
    try:
        logger.info(f"Starting autonomous workflow for campaign: {campaign_id}")
        
        # Verify campaign exists
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_id_db, campaign_name, metadata = campaign
            metadata = metadata or {}
        
        # Try to import autonomous orchestrator
        try:
            from src.agents.workflow.autonomous_workflow_orchestrator import autonomous_orchestrator
            
            # Start autonomous workflow
            workflow_result = await autonomous_orchestrator.start_campaign_workflow(
                campaign_id=campaign_id,
                campaign_data={
                    "campaign_name": campaign_name,
                    "metadata": metadata
                }
            )
            
            workflow_id = workflow_result.get("workflow_id", f"autonomous_{campaign_id}")
            
        except ImportError:
            logger.warning("Autonomous orchestrator not available, using mock workflow")
            workflow_id = f"mock_autonomous_{campaign_id}"
            workflow_result = {
                "workflow_id": workflow_id,
                "status": "started",
                "estimated_completion": "30 minutes"
            }
        
        # Update campaign metadata
        metadata["autonomous_workflow"] = {
            "workflow_id": workflow_id,
            "status": "running",
            "started_at": datetime.now().isoformat()
        }
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE campaigns 
                SET metadata = %s, updated_at = NOW()
                WHERE id = %s
            """, (metadata, campaign_id))
            conn.commit()
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Autonomous workflow started successfully",
            "estimated_completion": workflow_result.get("estimated_completion", "30 minutes")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting autonomous workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start autonomous workflow: {str(e)}")

@router.get("/{campaign_id}/status", response_model=Dict[str, Any])
async def get_autonomous_workflow_status(campaign_id: str):
    """
    Get status of autonomous workflow for a campaign
    """
    try:
        logger.info(f"Getting autonomous workflow status for campaign: {campaign_id}")
        
        # Get campaign and workflow data
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    c.id,
                    c.name,
                    c.metadata,
                    COUNT(ct.id) as total_tasks,
                    COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN ct.status = 'running' THEN 1 END) as running_tasks,
                    COUNT(CASE WHEN ct.status = 'failed' THEN 1 END) as failed_tasks
                FROM campaigns c
                LEFT JOIN campaign_tasks ct ON c.id = ct.campaign_id
                WHERE c.id = %s
                GROUP BY c.id, c.name, c.metadata
            """, (campaign_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            (campaign_id_db, campaign_name, metadata, total_tasks, 
             completed_tasks, running_tasks, failed_tasks) = row
        
        metadata = metadata or {}
        autonomous_workflow = metadata.get("autonomous_workflow", {})
        
        # Calculate progress
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Determine workflow status
        if autonomous_workflow.get("status") == "running":
            if running_tasks > 0:
                status = "running"
            elif completed_tasks == total_tasks:
                status = "completed"
                # Update metadata
                autonomous_workflow["status"] = "completed"
                autonomous_workflow["completed_at"] = datetime.now().isoformat()
            elif failed_tasks > 0:
                status = "failed"
            else:
                status = "running"
        else:
            status = autonomous_workflow.get("status", "not_started")
        
        # Get current workflow step
        current_step = None
        if running_tasks > 0:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT task_type, agent_type
                    FROM campaign_tasks
                    WHERE campaign_id = %s AND status = 'running'
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (campaign_id,))
                
                task_row = cur.fetchone()
                if task_row:
                    current_step = {
                        "task_type": task_row[0],
                        "agent_type": task_row[1]
                    }
        
        # Estimate remaining time
        remaining_tasks = total_tasks - completed_tasks - failed_tasks
        estimated_remaining_minutes = remaining_tasks * 3  # 3 minutes per task estimate
        
        return {
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "workflow_id": autonomous_workflow.get("workflow_id"),
            "status": status,
            "progress": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "running_tasks": running_tasks,
                "failed_tasks": failed_tasks,
                "progress_percentage": round(progress_percentage, 1)
            },
            "current_step": current_step,
            "started_at": autonomous_workflow.get("started_at"),
            "completed_at": autonomous_workflow.get("completed_at"),
            "estimated_remaining_minutes": estimated_remaining_minutes if status == "running" else 0,
            "last_updated": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting autonomous workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get autonomous workflow status: {str(e)}")

@router.post("/{campaign_id}/pause", response_model=Dict[str, Any])
async def pause_autonomous_workflow(campaign_id: str):
    """
    Pause autonomous workflow for a campaign
    """
    try:
        logger.info(f"Pausing autonomous workflow for campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_id_db, campaign_name, metadata = campaign
            metadata = metadata or {}
        
        # Update autonomous workflow status
        autonomous_workflow = metadata.get("autonomous_workflow", {})
        autonomous_workflow["status"] = "paused"
        autonomous_workflow["paused_at"] = datetime.now().isoformat()
        metadata["autonomous_workflow"] = autonomous_workflow
        
        # Pause running tasks
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE campaign_tasks 
                SET status = 'paused', updated_at = NOW()
                WHERE campaign_id = %s AND status = 'running'
            """, (campaign_id,))
            
            paused_tasks = cur.rowcount
            
            # Update campaign metadata
            cur.execute("""
                UPDATE campaigns 
                SET metadata = %s, updated_at = NOW()
                WHERE id = %s
            """, (metadata, campaign_id))
            
            conn.commit()
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "workflow_id": autonomous_workflow.get("workflow_id"),
            "status": "paused",
            "paused_tasks": paused_tasks,
            "message": "Autonomous workflow paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing autonomous workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause autonomous workflow: {str(e)}")

@router.post("/{campaign_id}/resume", response_model=Dict[str, Any])
async def resume_autonomous_workflow(campaign_id: str):
    """
    Resume paused autonomous workflow for a campaign
    """
    try:
        logger.info(f"Resuming autonomous workflow for campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_id_db, campaign_name, metadata = campaign
            metadata = metadata or {}
        
        autonomous_workflow = metadata.get("autonomous_workflow", {})
        
        if autonomous_workflow.get("status") != "paused":
            raise HTTPException(status_code=400, detail="Workflow is not paused")
        
        # Update autonomous workflow status
        autonomous_workflow["status"] = "running"
        autonomous_workflow["resumed_at"] = datetime.now().isoformat()
        metadata["autonomous_workflow"] = autonomous_workflow
        
        # Resume paused tasks
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE campaign_tasks 
                SET status = 'pending', updated_at = NOW()
                WHERE campaign_id = %s AND status = 'paused'
            """, (campaign_id,))
            
            resumed_tasks = cur.rowcount
            
            # Update campaign metadata
            cur.execute("""
                UPDATE campaigns 
                SET metadata = %s, updated_at = NOW()
                WHERE id = %s
            """, (metadata, campaign_id))
            
            conn.commit()
        
        # Trigger workflow resumption
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
                    "priority": "medium"
                }
            ))
        except Exception as execution_error:
            logger.warning(f"Failed to trigger workflow resumption: {execution_error}")
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "workflow_id": autonomous_workflow.get("workflow_id"),
            "status": "running",
            "resumed_tasks": resumed_tasks,
            "message": "Autonomous workflow resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming autonomous workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume autonomous workflow: {str(e)}")

@router.post("/{campaign_id}/stop", response_model=Dict[str, Any])
async def stop_autonomous_workflow(campaign_id: str):
    """
    Stop autonomous workflow for a campaign
    """
    try:
        logger.info(f"Stopping autonomous workflow for campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, metadata FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            campaign_id_db, campaign_name, metadata = campaign
            metadata = metadata or {}
        
        # Update autonomous workflow status
        autonomous_workflow = metadata.get("autonomous_workflow", {})
        autonomous_workflow["status"] = "stopped"
        autonomous_workflow["stopped_at"] = datetime.now().isoformat()
        metadata["autonomous_workflow"] = autonomous_workflow
        
        # Stop all running/pending tasks
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE campaign_tasks 
                SET status = 'stopped', updated_at = NOW()
                WHERE campaign_id = %s AND status IN ('running', 'pending', 'paused')
            """, (campaign_id,))
            
            stopped_tasks = cur.rowcount
            
            # Update campaign metadata
            cur.execute("""
                UPDATE campaigns 
                SET metadata = %s, updated_at = NOW()
                WHERE id = %s
            """, (metadata, campaign_id))
            
            conn.commit()
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "workflow_id": autonomous_workflow.get("workflow_id"),
            "status": "stopped",
            "stopped_tasks": stopped_tasks,
            "message": "Autonomous workflow stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping autonomous workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop autonomous workflow: {str(e)}")