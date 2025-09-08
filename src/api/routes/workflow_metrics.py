"""
Workflow Metrics API Routes

Provides REST endpoints for accessing workflow monitoring data,
performance metrics, and system health information.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.agents.monitoring import workflow_metrics, get_workflow_health
from src.core.monitoring import get_performance_tracker

router = APIRouter(prefix="/workflow-metrics", tags=["workflow-metrics"])


@router.get("/health")
async def get_system_health() -> Dict[str, Any]:
    """Get overall workflow system health and performance summary"""
    try:
        return await get_workflow_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/workflows/{workflow_id}")
async def get_workflow_metrics(
    workflow_id: str,
    hours: Optional[int] = Query(24, description="Hours of history to include")
) -> Dict[str, Any]:
    """Get metrics for a specific workflow"""
    try:
        time_range = timedelta(hours=hours) if hours else None
        metrics = await workflow_metrics.get_workflow_metrics(
            workflow_id=workflow_id,
            time_range=time_range
        )
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get workflow metrics: {str(e)}"
        )


@router.get("/workflows")
async def get_all_workflow_metrics(
    hours: Optional[int] = Query(24, description="Hours of history to include"),
    limit: Optional[int] = Query(50, description="Maximum number of workflows to return")
) -> Dict[str, Any]:
    """Get metrics for all workflows"""
    try:
        time_range = timedelta(hours=hours) if hours else None
        metrics = await workflow_metrics.get_workflow_metrics(time_range=time_range)
        
        # Limit the number of workflows returned
        if "workflows" in metrics and limit:
            metrics["workflows"] = metrics["workflows"][:limit]
        
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow metrics: {str(e)}"
        )


@router.get("/active")
async def get_active_workflows() -> Dict[str, Any]:
    """Get currently active workflow executions"""
    try:
        active_workflows = []
        for run_id, execution in workflow_metrics.active_workflows.items():
            active_workflows.append({
                "run_id": run_id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "start_time": execution.start_time.isoformat(),
                "duration_so_far": (datetime.utcnow() - execution.start_time).total_seconds(),
                "agent_executions": len(execution.agent_executions),
                "errors": len(execution.errors)
            })
        
        return {
            "active_count": len(active_workflows),
            "workflows": active_workflows
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active workflows: {str(e)}"
        )


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get detailed performance metrics and analytics"""
    try:
        # Get workflow metrics
        workflow_health = await get_workflow_health()
        
        # Get system performance metrics if available
        try:
            performance_tracker = get_performance_tracker()
            system_metrics = await performance_tracker.get_current_metrics()
        except Exception:
            system_metrics = {"message": "System metrics unavailable"}
        
        # Calculate additional analytics
        recent_workflows = list(workflow_metrics.completed_workflows)[-100:]
        
        # Duration distribution
        durations = [w.duration for w in recent_workflows if w.duration is not None]
        duration_stats = {}
        if durations:
            duration_stats = {
                "min": min(durations),
                "max": max(durations), 
                "avg": sum(durations) / len(durations),
                "p95": sorted(durations)[int(0.95 * len(durations))] if len(durations) > 20 else max(durations)
            }
        
        # Agent execution stats
        agent_stats = {}
        for workflow in recent_workflows:
            for agent_exec in workflow.agent_executions:
                agent_name = agent_exec["agent_name"]
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {"executions": 0, "total_duration": 0, "avg_quality": 0}
                
                agent_stats[agent_name]["executions"] += 1
                agent_stats[agent_name]["total_duration"] += agent_exec.get("duration", 0)
                if agent_exec.get("quality_score"):
                    agent_stats[agent_name]["avg_quality"] += agent_exec["quality_score"]
        
        # Calculate averages for agents
        for agent_name in agent_stats:
            stats = agent_stats[agent_name]
            if stats["executions"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["executions"]
                stats["avg_quality"] = stats["avg_quality"] / stats["executions"]
        
        return {
            "workflow_health": workflow_health,
            "system_metrics": system_metrics,
            "duration_statistics": duration_stats,
            "agent_performance": agent_stats,
            "total_workflows_analyzed": len(recent_workflows)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/errors")
async def get_error_analysis(
    hours: Optional[int] = Query(24, description="Hours of history to analyze")
) -> Dict[str, Any]:
    """Get error analysis and troubleshooting information"""
    try:
        time_range = timedelta(hours=hours) if hours else None
        cutoff_time = datetime.utcnow() - time_range if time_range else None
        
        all_errors = []
        error_types = {}
        agent_errors = {}
        
        # Analyze completed workflows
        for workflow in workflow_metrics.completed_workflows:
            if cutoff_time and workflow.start_time < cutoff_time:
                continue
                
            for error in workflow.errors:
                all_errors.append({
                    "workflow_id": workflow.workflow_id,
                    "run_id": workflow.run_id,
                    "error_type": error["error_type"],
                    "message": error["message"],
                    "agent": error.get("agent", "unknown"),
                    "timestamp": error["timestamp"]
                })
                
                # Count error types
                error_type = error["error_type"]
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
                
                # Count agent errors
                agent = error.get("agent", "unknown")
                if agent not in agent_errors:
                    agent_errors[agent] = 0
                agent_errors[agent] += 1
        
        # Sort by frequency
        error_types = dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True))
        agent_errors = dict(sorted(agent_errors.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "total_errors": len(all_errors),
            "error_types": error_types,
            "errors_by_agent": agent_errors,
            "recent_errors": all_errors[-20:],  # Last 20 errors
            "analysis_period_hours": hours or "all_time"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze errors: {str(e)}"
        )


@router.get("/trends")
async def get_workflow_trends(
    days: Optional[int] = Query(7, description="Number of days for trend analysis")
) -> Dict[str, Any]:
    """Get workflow execution trends and patterns"""
    try:
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Group workflows by day
        daily_stats = {}
        for workflow in workflow_metrics.completed_workflows:
            if workflow.start_time < cutoff_time:
                continue
                
            day_key = workflow.start_time.strftime("%Y-%m-%d")
            if day_key not in daily_stats:
                daily_stats[day_key] = {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "avg_duration": 0,
                    "durations": []
                }
            
            stats = daily_stats[day_key]
            stats["total"] += 1
            
            if workflow.status.value == "completed":
                stats["completed"] += 1
            elif workflow.status.value == "failed":
                stats["failed"] += 1
                
            if workflow.duration:
                stats["durations"].append(workflow.duration)
        
        # Calculate averages
        for day_key in daily_stats:
            stats = daily_stats[day_key]
            if stats["durations"]:
                stats["avg_duration"] = sum(stats["durations"]) / len(stats["durations"])
            del stats["durations"]  # Remove raw data
        
        # Sort by date
        sorted_stats = dict(sorted(daily_stats.items()))
        
        return {
            "analysis_days": days,
            "daily_statistics": sorted_stats,
            "trend_summary": {
                "total_days": len(sorted_stats),
                "avg_daily_workflows": sum(stats["total"] for stats in sorted_stats.values()) / len(sorted_stats) if sorted_stats else 0,
                "overall_success_rate": sum(stats["completed"] for stats in sorted_stats.values()) / sum(stats["total"] for stats in sorted_stats.values()) if sorted_stats else 0
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze trends: {str(e)}"
        )