"""
Campaign Task Management Routes
Handles task creation, status updates, execution, and review.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class TaskStatusUpdate(BaseModel):
    task_id: str
    status: str

# Task Management Endpoints
@router.put("/{campaign_id}/tasks/{task_id}/status", response_model=Dict[str, Any])
async def update_task_status(campaign_id: str, task_id: str, status_update: TaskStatusUpdate):
    """
    Update the status of a specific task in a campaign
    """
    try:
        logger.info(f"Updating task {task_id} in campaign {campaign_id} to status {status_update.status}")
        
        # Validate status
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        if status_update.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        # Update task status in database
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # First check if task exists and belongs to campaign
            cur.execute("""
                SELECT id FROM campaign_tasks 
                WHERE id = %s AND campaign_id = %s
            """, (task_id, campaign_id))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Task not found or doesn't belong to this campaign")
            
            # Update task status
            cur.execute("""
                UPDATE campaign_tasks 
                SET status = %s, updated_at = NOW()
                WHERE id = %s AND campaign_id = %s
            """, (status_update.status, task_id, campaign_id))
            
            conn.commit()
            
            # Get updated task
            cur.execute("""
                SELECT id, task_type, status, output_data, agent_type
                FROM campaign_tasks
                WHERE id = %s
            """, (task_id,))
            
            row = cur.fetchone()
            if row:
                task_id_db, task_type, status, content, agent_type = row
                
                # Handle content JSON parsing
                metadata = {}
                if content:
                    if isinstance(content, str):
                        try:
                            metadata = json.loads(content)
                        except json.JSONDecodeError:
                            metadata = {"raw_content": content}
                    elif isinstance(content, dict):
                        metadata = content
                
                return {
                    "success": True,
                    "message": f"Task status updated to {status_update.status}",
                    "task": {
                        "id": task_id_db,
                        "task_type": task_type,
                        "agent_type": agent_type,
                        "status": status,
                        "content": content,
                        "metadata": metadata
                    }
                }
        
        raise HTTPException(status_code=500, detail="Failed to retrieve updated task")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update task status: {str(e)}")

@router.post("/{campaign_id}/create-tasks", response_model=Dict[str, Any])
async def create_campaign_tasks(campaign_id: str, request: Dict[str, Any] = None):
    """
    Create tasks for a campaign based on campaign requirements
    """
    try:
        logger.info(f"Creating tasks for campaign: {campaign_id}")
        
        # Verify campaign exists
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Define standard task types for content campaigns
        standard_tasks = [
            {
                "type": "content_planning",
                "agent": "PlannerAgent",
                "description": "Create content strategy and planning",
                "priority": 1
            },
            {
                "type": "research",
                "agent": "ResearcherAgent", 
                "description": "Conduct research for content topics",
                "priority": 2
            },
            {
                "type": "content_creation",
                "agent": "WriterAgent",
                "description": "Write primary content",
                "priority": 3
            },
            {
                "type": "content_editing",
                "agent": "EditorAgent",
                "description": "Edit and refine content",
                "priority": 4
            },
            {
                "type": "seo_optimization",
                "agent": "SEOAgent",
                "description": "Optimize content for search engines",
                "priority": 5
            },
            {
                "type": "image_generation",
                "agent": "ImageAgent",
                "description": "Generate images and visual content",
                "priority": 6
            },
            {
                "type": "social_adaptation",
                "agent": "SocialMediaAgent",
                "description": "Adapt content for social media platforms",
                "priority": 7
            },
            {
                "type": "quality_review",
                "agent": "QualityAgent",
                "description": "Final quality assurance review",
                "priority": 8
            }
        ]
        
        # Customize tasks based on request
        tasks_to_create = standard_tasks
        if request and "task_types" in request:
            # Filter tasks based on requested types
            requested_types = request["task_types"]
            tasks_to_create = [task for task in standard_tasks if task["type"] in requested_types]
        
        # Insert tasks into database
        created_tasks = []
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            for task in tasks_to_create:
                try:
                    cur.execute("""
                        INSERT INTO campaign_tasks 
                        (campaign_id, task_type, agent_type, status, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id
                    """, (
                        campaign_id,
                        task["type"],
                        task["agent"],
                        "pending",
                        json.dumps({
                            "description": task["description"],
                            "priority": task["priority"]
                        })
                    ))
                    
                    task_id = cur.fetchone()[0]
                    created_tasks.append({
                        "id": task_id,
                        "type": task["type"],
                        "agent": task["agent"],
                        "status": "pending",
                        "description": task["description"],
                        "priority": task["priority"]
                    })
                    
                except Exception as task_error:
                    logger.warning(f"Failed to create task {task['type']}: {task_error}")
            
            conn.commit()
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "message": f"Created {len(created_tasks)} tasks for campaign",
            "tasks_created": len(created_tasks),
            "tasks": created_tasks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating campaign tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create campaign tasks: {str(e)}")

@router.get("/{campaign_id}/tasks", response_model=Dict[str, Any])
async def get_campaign_tasks(campaign_id: str):
    """
    Get all tasks for a campaign with their current status
    """
    try:
        logger.info(f"Getting tasks for campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Verify campaign exists
            cur.execute("SELECT id, name FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            # Get all tasks for the campaign
            cur.execute("""
                SELECT 
                    id, 
                    task_type, 
                    agent_type, 
                    status, 
                    output_data, 
                    error,
                    created_at, 
                    updated_at,
                    metadata
                FROM campaign_tasks
                WHERE campaign_id = %s
                ORDER BY created_at ASC
            """, (campaign_id,))
            
            task_rows = cur.fetchall()
        
        # Process tasks
        tasks = []
        for row in task_rows:
            task_id, task_type, agent_type, status, output_data, error, created_at, updated_at, metadata = row
            
            # Parse metadata
            task_metadata = {}
            if metadata:
                if isinstance(metadata, str):
                    try:
                        task_metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        task_metadata = {}
                elif isinstance(metadata, dict):
                    task_metadata = metadata
            
            # Parse output data
            output_preview = None
            if output_data:
                if isinstance(output_data, str):
                    output_preview = output_data[:200] + "..." if len(output_data) > 200 else output_data
                elif isinstance(output_data, dict):
                    output_preview = str(output_data)[:200] + "..." if len(str(output_data)) > 200 else str(output_data)
            
            tasks.append({
                "id": str(task_id),
                "task_type": task_type,
                "agent_type": agent_type,
                "status": status,
                "description": task_metadata.get("description", task_type.replace("_", " ").title()),
                "priority": task_metadata.get("priority", 0),
                "output_preview": output_preview,
                "error": error,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "has_output": output_data is not None
            })
        
        # Calculate summary statistics
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t["status"] == "completed"])
        failed_tasks = len([t for t in tasks if t["status"] == "failed"])
        pending_tasks = len([t for t in tasks if t["status"] == "pending"])
        running_tasks = len([t for t in tasks if t["status"] in ["running", "in_progress"]])
        
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "campaign_id": campaign_id,
            "campaign_name": campaign[1],
            "tasks": tasks,
            "summary": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": pending_tasks,
                "running_tasks": running_tasks,
                "progress_percentage": round(progress_percentage, 1)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign tasks: {str(e)}")

@router.get("/{campaign_id}/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_detail(campaign_id: str, task_id: str):
    """
    Get detailed information about a specific task
    """
    try:
        logger.info(f"Getting task detail for task: {task_id} in campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    ct.id,
                    ct.task_type,
                    ct.agent_type,
                    ct.status,
                    ct.output_data,
                    ct.error,
                    ct.created_at,
                    ct.updated_at,
                    ct.metadata,
                    c.name as campaign_name
                FROM campaign_tasks ct
                JOIN campaigns c ON ct.campaign_id = c.id
                WHERE ct.id = %s AND ct.campaign_id = %s
            """, (task_id, campaign_id))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Task not found")
            
            (task_id_db, task_type, agent_type, status, output_data, error, 
             created_at, updated_at, metadata, campaign_name) = row
        
        # Parse metadata
        task_metadata = {}
        if metadata:
            if isinstance(metadata, str):
                try:
                    task_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    task_metadata = {}
            elif isinstance(metadata, dict):
                task_metadata = metadata
        
        # Parse output data
        parsed_output = None
        if output_data:
            if isinstance(output_data, str):
                try:
                    parsed_output = json.loads(output_data)
                except json.JSONDecodeError:
                    parsed_output = {"raw_output": output_data}
            elif isinstance(output_data, dict):
                parsed_output = output_data
        
        return {
            "task_id": str(task_id_db),
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "task_type": task_type,
            "agent_type": agent_type,
            "status": status,
            "description": task_metadata.get("description", task_type.replace("_", " ").title()),
            "priority": task_metadata.get("priority", 0),
            "output_data": parsed_output,
            "error": error,
            "created_at": created_at.isoformat() if created_at else None,
            "updated_at": updated_at.isoformat() if updated_at else None,
            "metadata": task_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task detail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task detail: {str(e)}")

@router.delete("/{campaign_id}/tasks/{task_id}", response_model=Dict[str, Any])
async def delete_task(campaign_id: str, task_id: str):
    """
    Delete a specific task from a campaign
    """
    try:
        logger.info(f"Deleting task: {task_id} from campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if task exists and belongs to campaign
            cur.execute("""
                SELECT id FROM campaign_tasks 
                WHERE id = %s AND campaign_id = %s
            """, (task_id, campaign_id))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Task not found or doesn't belong to this campaign")
            
            # Delete the task
            cur.execute("""
                DELETE FROM campaign_tasks 
                WHERE id = %s AND campaign_id = %s
            """, (task_id, campaign_id))
            
            conn.commit()
        
        return {
            "success": True,
            "message": "Task deleted successfully",
            "task_id": task_id,
            "campaign_id": campaign_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

@router.post("/{campaign_id}/tasks/{task_id}/retry", response_model=Dict[str, Any])
async def retry_task(campaign_id: str, task_id: str):
    """
    Retry a failed task
    """
    try:
        logger.info(f"Retrying task: {task_id} in campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if task exists and is in a retryable state
            cur.execute("""
                SELECT id, status, task_type, agent_type FROM campaign_tasks 
                WHERE id = %s AND campaign_id = %s
            """, (task_id, campaign_id))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Task not found or doesn't belong to this campaign")
            
            _, current_status, task_type, agent_type = row
            
            if current_status not in ["failed", "completed"]:
                raise HTTPException(status_code=400, detail=f"Task cannot be retried in current status: {current_status}")
            
            # Reset task to pending status
            cur.execute("""
                UPDATE campaign_tasks 
                SET status = 'pending', error = NULL, updated_at = NOW()
                WHERE id = %s AND campaign_id = %s
            """, (task_id, campaign_id))
            
            conn.commit()
        
        # Optionally trigger task execution here
        # This would integrate with your agent execution system
        
        return {
            "success": True,
            "message": "Task reset for retry",
            "task_id": task_id,
            "campaign_id": campaign_id,
            "new_status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retry task: {str(e)}")

@router.get("/{campaign_id}/tasks/summary", response_model=Dict[str, Any])
async def get_tasks_summary(campaign_id: str):
    """
    Get a summary of tasks for a campaign (lightweight version)
    """
    try:
        logger.info(f"Getting task summary for campaign: {campaign_id}")
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get task counts by status
            cur.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM campaign_tasks
                WHERE campaign_id = %s
                GROUP BY status
            """, (campaign_id,))
            
            status_counts = dict(cur.fetchall())
            
            # Get total count
            cur.execute("""
                SELECT COUNT(*) FROM campaign_tasks WHERE campaign_id = %s
            """, (campaign_id,))
            
            total_tasks = cur.fetchone()[0]
        
        # Calculate progress
        completed = status_counts.get("completed", 0)
        progress_percentage = (completed / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "campaign_id": campaign_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": status_counts.get("failed", 0),
            "pending_tasks": status_counts.get("pending", 0),
            "running_tasks": status_counts.get("running", 0) + status_counts.get("in_progress", 0),
            "progress_percentage": round(progress_percentage, 1),
            "status_breakdown": status_counts
        }
        
    except Exception as e:
        logger.error(f"Error getting task summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task summary: {str(e)}")