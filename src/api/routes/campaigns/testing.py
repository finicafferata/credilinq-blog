"""
Testing and Debug Utilities Routes
Handles test endpoints and debugging utilities for campaigns.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Testing Endpoints
@router.get("/simple-test")
async def simple_test():
    """
    Simple test endpoint
    """
    return {
        "message": "Campaign testing endpoint is working",
        "timestamp": datetime.now().isoformat(),
        "status": "ok"
    }

@router.get("/test-campaign/{campaign_id}")
async def test_campaign_minimal(campaign_id: str):
    """
    Minimal campaign test endpoint
    """
    try:
        from src.config.database import secure_db
        
        # Test basic campaign query
        campaign = secure_db.execute_query(
            'SELECT id, status FROM campaigns WHERE id = %s', 
            [campaign_id], 
            fetch='one'
        )
        
        if not campaign:
            return {"error": "Campaign not found"}
        
        # Test task query
        tasks = secure_db.execute_query(
            'SELECT id, task_type, status FROM campaign_tasks WHERE campaign_id = %s', 
            [campaign_id], 
            fetch='all'
        )
        
        return {
            "campaign": dict(campaign) if campaign else None,
            "tasks": [dict(task) for task in (tasks or [])],
            "task_count": len(tasks) if tasks else 0,
            "test_status": "passed"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "test_status": "failed"
        }

@router.get("/test/{template_id}", response_model=Dict[str, Any])
async def test_quick_campaign(template_id: str, blog_id: str = Query(...), campaign_name: str = Query(...)):
    """
    Test endpoint for debugging quick campaign creation
    """
    try:
        return {
            "template_id": template_id,
            "blog_id": blog_id,
            "campaign_name": campaign_name,
            "message": "Test successful",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/debug/database")
async def debug_database_connection():
    """
    Test database connectivity and basic queries
    """
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Test basic query
            cur.execute("SELECT 1 as test_value")
            test_result = cur.fetchone()
            
            # Test campaigns table
            cur.execute("SELECT COUNT(*) FROM campaigns")
            campaign_count = cur.fetchone()[0]
            
            # Test campaign_tasks table
            cur.execute("SELECT COUNT(*) FROM campaign_tasks")
            task_count = cur.fetchone()[0]
            
        return {
            "database_status": "connected",
            "test_query": test_result[0] if test_result else None,
            "campaigns_count": campaign_count,
            "tasks_count": task_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database debug error: {str(e)}")
        return {
            "database_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/debug/system-info")
async def debug_system_info():
    """
    Get system information for debugging
    """
    import os
    import sys
    import platform
    
    try:
        return {
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture(),
                "processor": platform.processor()
            },
            "environment": {
                "working_directory": os.getcwd(),
                "python_path_count": len(sys.path),
                "environment_vars": {
                    "DATABASE_URL": "***" if os.getenv("DATABASE_URL") else None,
                    "GEMINI_API_KEY": "***" if os.getenv("GEMINI_API_KEY") else None,
                    "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT"),
                    "ENABLE_AGENT_LOADING": os.getenv("ENABLE_AGENT_LOADING")
                }
            },
            "application": {
                "campaigns_module_loaded": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"System info debug error: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/debug/agent-registry")
async def debug_agent_registry():
    """
    Debug agent registry status
    """
    try:
        # Try to get agent registry
        try:
            from src.api.routes.agents import _agent_registry, initialize_agent_registry
            await initialize_agent_registry()
            
            registry_info = {
                "registry_available": True,
                "registered_agents": len(_agent_registry),
                "agent_types": list(_agent_registry.keys()) if _agent_registry else []
            }
        except ImportError as e:
            registry_info = {
                "registry_available": False,
                "import_error": str(e),
                "agent_types": []
            }
        except Exception as e:
            registry_info = {
                "registry_available": False,
                "error": str(e),
                "agent_types": []
            }
        
        return {
            "agent_registry": registry_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent registry debug error: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/debug/test-websocket")
async def test_websocket_broadcast():
    """
    Test WebSocket broadcasting functionality
    """
    try:
        from .workflow import websocket_manager
        
        # Get connection stats
        stats = websocket_manager.get_connection_stats()
        
        # Try to broadcast a test message
        test_campaign_id = "test_campaign_123"
        test_message = {
            "type": "debug_test",
            "message": "WebSocket test broadcast",
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_to_campaign(test_message, test_campaign_id)
        
        return {
            "websocket_status": "available",
            "connection_stats": stats,
            "test_broadcast": "sent",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"WebSocket test error: {str(e)}")
        return {
            "websocket_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/debug/performance-stats")
async def debug_performance_stats():
    """
    Get performance statistics for debugging
    """
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get campaign statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_campaigns,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_campaigns,
                    COUNT(CASE WHEN metadata->>'processing_status' = 'generating_content' THEN 1 END) as active_campaigns
                FROM campaigns
            """)
            
            campaign_stats = cur.fetchone()
            
            # Get task statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_tasks,
                    COUNT(CASE WHEN status = 'running' THEN 1 END) as running_tasks
                FROM campaign_tasks
            """)
            
            task_stats = cur.fetchone()
            
            # Get agent performance statistics if available
            try:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_executions,
                        AVG(duration) as avg_duration,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions
                    FROM agent_performance
                """)
                agent_stats = cur.fetchone()
            except:
                agent_stats = (0, 0, 0)
        
        return {
            "performance_stats": {
                "campaigns": {
                    "total": campaign_stats[0] if campaign_stats else 0,
                    "recent": campaign_stats[1] if campaign_stats else 0,
                    "active": campaign_stats[2] if campaign_stats else 0
                },
                "tasks": {
                    "total": task_stats[0] if task_stats else 0,
                    "completed": task_stats[1] if task_stats else 0,
                    "failed": task_stats[2] if task_stats else 0,
                    "pending": task_stats[3] if task_stats else 0,
                    "running": task_stats[4] if task_stats else 0
                },
                "agents": {
                    "total_executions": agent_stats[0] if agent_stats else 0,
                    "avg_duration_ms": int(agent_stats[1]) if agent_stats and agent_stats[1] else 0,
                    "successful_executions": agent_stats[2] if agent_stats else 0
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance stats debug error: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/debug/cleanup-test-data")
async def cleanup_test_data():
    """
    Clean up test campaigns and data (use with caution)
    """
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Delete test campaigns (those with names containing 'test')
            cur.execute("""
                DELETE FROM campaign_tasks 
                WHERE campaign_id IN (
                    SELECT id FROM campaigns 
                    WHERE LOWER(name) LIKE '%test%' OR id LIKE 'test_%'
                )
            """)
            deleted_tasks = cur.rowcount
            
            cur.execute("""
                DELETE FROM campaigns 
                WHERE LOWER(name) LIKE '%test%' OR id LIKE 'test_%'
            """)
            deleted_campaigns = cur.rowcount
            
            conn.commit()
        
        return {
            "cleanup_status": "completed",
            "deleted_campaigns": deleted_campaigns,
            "deleted_tasks": deleted_tasks,
            "message": "Test data cleaned up successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return {
            "cleanup_status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }