#!/usr/bin/env python3
"""
Campaign Progress Service
Provides unified progress calculation and synchronization between legacy and new orchestration schemas.
Ensures consistent progress reporting across dashboard and detail views.
"""

import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from src.config.database import db_config

logger = logging.getLogger(__name__)

@dataclass
class CampaignProgress:
    """Unified campaign progress structure"""
    campaign_id: str
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    pending_tasks: int
    failed_tasks: int
    progress_percentage: float
    current_phase: str
    workflow_state: Dict[str, Any]
    last_updated: datetime

@dataclass
class TaskProgress:
    """Individual task progress structure"""
    task_id: str
    task_type: str
    status: str
    progress: float
    result: Optional[str]
    error: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    workflow_step_id: Optional[str]

class CampaignProgressService:
    """
    Unified service for campaign progress calculation and synchronization.
    Handles both legacy campaign_tasks and new orchestration workflow schemas.
    """
    
    def __init__(self):
        self.service_name = "CampaignProgressService"
        self.version = "1.0.0"
    
    async def get_campaign_progress(self, campaign_id: str) -> CampaignProgress:
        """
        Get unified campaign progress from both legacy and orchestration sources.
        This method ensures consistent progress calculation across all views.
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get campaign basic info
                cur.execute("""
                    SELECT c.id, c.status, c.created_at, c.updated_at,
                           COALESCE(b.campaign_name, 'Unnamed Campaign') as name
                    FROM campaigns c
                    LEFT JOIN briefings b ON c.id = b.campaign_id
                    WHERE c.id = %s
                """, (campaign_id,))
                
                campaign_row = cur.fetchone()
                if not campaign_row:
                    raise ValueError(f"Campaign {campaign_id} not found")
                
                campaign_id_db, status, created_at, updated_at, name = campaign_row
                
                # Get tasks from legacy schema
                legacy_tasks = await self._get_legacy_task_progress(campaign_id, cur)
                
                # Get orchestration workflow progress (if exists)
                orchestration_progress = await self._get_orchestration_progress(campaign_id, cur)
                
                # Combine both sources for unified progress
                unified_progress = await self._calculate_unified_progress(
                    campaign_id, legacy_tasks, orchestration_progress, cur
                )
                
                # Update database with calculated progress
                await self._persist_progress_state(campaign_id, unified_progress, cur)
                
                conn.commit()
                return unified_progress
                
        except Exception as e:
            logger.error(f"Error getting campaign progress for {campaign_id}: {str(e)}")
            # Return fallback progress
            return CampaignProgress(
                campaign_id=campaign_id,
                total_tasks=0,
                completed_tasks=0,
                in_progress_tasks=0,
                pending_tasks=0,
                failed_tasks=0,
                progress_percentage=0.0,
                current_phase="unknown",
                workflow_state={},
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _get_legacy_task_progress(self, campaign_id: str, cur) -> List[TaskProgress]:
        """Get progress from legacy campaign_tasks table"""
        try:
            cur.execute("""
                SELECT id, task_type, status, result, error, 
                       started_at, completed_at, created_at
                FROM campaign_tasks
                WHERE campaign_id = %s
                ORDER BY created_at
            """, (campaign_id,))
            
            tasks = []
            for row in cur.fetchall():
                task_id, task_type, status, result, error, started_at, completed_at, created_at = row
                
                # Calculate task progress based on status
                progress = self._calculate_task_progress(status)
                
                tasks.append(TaskProgress(
                    task_id=task_id,
                    task_type=task_type,
                    status=status,
                    progress=progress,
                    result=result,
                    error=error,
                    started_at=started_at,
                    completed_at=completed_at,
                    workflow_step_id=None
                ))
            
            return tasks
        except Exception as e:
            logger.warning(f"Error getting legacy task progress: {e}")
            return []
    
    async def _get_orchestration_progress(self, campaign_id: str, cur) -> Dict[str, Any]:
        """Get progress from new orchestration workflow schema"""
        try:
            # Check if orchestration tables exist and have data for this campaign
            cur.execute("""
                SELECT cw.id, cw.status, cw.current_step, cw.steps_total, 
                       cw.steps_completed, cw.steps_failed, cw.execution_context,
                       cw.started_at, cw.completed_at
                FROM campaign_workflows cw
                WHERE cw.campaign_id = %s
                ORDER BY cw.created_at DESC
                LIMIT 1
            """, (campaign_id,))
            
            workflow_row = cur.fetchone()
            if not workflow_row:
                return {}
            
            (workflow_id, status, current_step, steps_total, steps_completed, 
             steps_failed, execution_context, started_at, completed_at) = workflow_row
            
            # Get workflow steps
            cur.execute("""
                SELECT id, step_name, step_type, status, started_at, completed_at,
                       execution_time_ms, error_message
                FROM campaign_workflow_steps
                WHERE workflow_id = %s
                ORDER BY step_order
            """, (workflow_id,))
            
            steps = []
            for step_row in cur.fetchall():
                (step_id, step_name, step_type, step_status, step_started_at, 
                 step_completed_at, execution_time_ms, error_message) = step_row
                
                steps.append({
                    'id': step_id,
                    'name': step_name,
                    'type': step_type,
                    'status': step_status,
                    'progress': self._calculate_task_progress(step_status),
                    'started_at': step_started_at,
                    'completed_at': step_completed_at,
                    'execution_time_ms': execution_time_ms,
                    'error_message': error_message
                })
            
            return {
                'workflow_id': workflow_id,
                'status': status,
                'current_step': current_step,
                'steps_total': steps_total,
                'steps_completed': steps_completed,
                'steps_failed': steps_failed,
                'execution_context': execution_context,
                'steps': steps,
                'started_at': started_at,
                'completed_at': completed_at
            }
            
        except Exception as e:
            logger.debug(f"No orchestration data found or error: {e}")
            return {}
    
    async def _calculate_unified_progress(self, campaign_id: str, legacy_tasks: List[TaskProgress], 
                                        orchestration_progress: Dict[str, Any], cur) -> CampaignProgress:
        """Calculate unified progress from both legacy and orchestration sources"""
        try:
            # Prioritize orchestration progress if available and has data
            if orchestration_progress and orchestration_progress.get('steps'):
                return await self._calculate_orchestration_progress(campaign_id, orchestration_progress)
            
            # Fall back to legacy progress calculation
            return await self._calculate_legacy_progress(campaign_id, legacy_tasks)
            
        except Exception as e:
            logger.warning(f"Error calculating unified progress: {e}")
            # Return minimal progress structure
            return CampaignProgress(
                campaign_id=campaign_id,
                total_tasks=len(legacy_tasks),
                completed_tasks=0,
                in_progress_tasks=0,
                pending_tasks=len(legacy_tasks),
                failed_tasks=0,
                progress_percentage=0.0,
                current_phase="planning",
                workflow_state={},
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_orchestration_progress(self, campaign_id: str, 
                                              orchestration_data: Dict[str, Any]) -> CampaignProgress:
        """Calculate progress from orchestration workflow data"""
        steps = orchestration_data.get('steps', [])
        total_tasks = len(steps)
        
        # Count tasks by status
        completed_tasks = len([s for s in steps if s['status'] in ['completed']])
        in_progress_tasks = len([s for s in steps if s['status'] in ['running', 'in_progress']])
        failed_tasks = len([s for s in steps if s['status'] in ['failed', 'error']])
        pending_tasks = len([s for s in steps if s['status'] in ['pending', 'retry']])
        
        # Calculate progress percentage
        if total_tasks > 0:
            progress_percentage = (completed_tasks / total_tasks) * 100
        else:
            progress_percentage = 0.0
        
        # Determine current phase based on workflow state
        current_step = orchestration_data.get('current_step', 'planning')
        current_phase = await self._determine_current_phase(current_step, steps)
        
        return CampaignProgress(
            campaign_id=campaign_id,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            in_progress_tasks=in_progress_tasks,
            pending_tasks=pending_tasks,
            failed_tasks=failed_tasks,
            progress_percentage=round(progress_percentage, 2),
            current_phase=current_phase,
            workflow_state=orchestration_data.get('execution_context', {}),
            last_updated=datetime.now(timezone.utc)
        )
    
    async def _calculate_legacy_progress(self, campaign_id: str, legacy_tasks: List[TaskProgress]) -> CampaignProgress:
        """Calculate progress from legacy campaign_tasks data"""
        total_tasks = len(legacy_tasks)
        
        # Count tasks by status with proper mapping
        completed_statuses = ['completed', 'approved', 'scheduled']
        in_progress_statuses = ['in_progress', 'generated', 'under_review']
        failed_statuses = ['failed', 'error', 'cancelled', 'timeout']
        
        completed_tasks = len([t for t in legacy_tasks if t.status in completed_statuses])
        in_progress_tasks = len([t for t in legacy_tasks if t.status in in_progress_statuses])
        failed_tasks = len([t for t in legacy_tasks if t.status in failed_statuses])
        pending_tasks = total_tasks - completed_tasks - in_progress_tasks - failed_tasks
        
        # Calculate progress percentage
        if total_tasks > 0:
            progress_percentage = (completed_tasks / total_tasks) * 100
        else:
            progress_percentage = 0.0
        
        # Determine current phase based on task states
        current_phase = await self._determine_current_phase_legacy(legacy_tasks)
        
        return CampaignProgress(
            campaign_id=campaign_id,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            in_progress_tasks=in_progress_tasks,
            pending_tasks=pending_tasks,
            failed_tasks=failed_tasks,
            progress_percentage=round(progress_percentage, 2),
            current_phase=current_phase,
            workflow_state={'source': 'legacy_tasks'},
            last_updated=datetime.now(timezone.utc)
        )
    
    def _calculate_task_progress(self, status: str) -> float:
        """Calculate individual task progress based on status"""
        status_progress_map = {
            'pending': 0.0,
            'in_progress': 0.5,
            'generated': 0.8,
            'under_review': 0.85,
            'revision_needed': 0.7,
            'approved': 0.95,
            'completed': 1.0,
            'scheduled': 1.0,
            'failed': 0.0,
            'error': 0.0,
            'cancelled': 0.0,
            'timeout': 0.0
        }
        return status_progress_map.get(status.lower(), 0.0)
    
    async def _determine_current_phase(self, current_step: str, steps: List[Dict[str, Any]]) -> str:
        """Determine current campaign phase from orchestration workflow"""
        if not steps:
            return "planning"
        
        completed_count = len([s for s in steps if s['status'] == 'completed'])
        total_count = len(steps)
        
        if completed_count == 0:
            return "planning"
        elif completed_count < total_count * 0.3:
            return "content_creation"
        elif completed_count < total_count * 0.7:
            return "content_review"
        elif completed_count < total_count:
            return "distribution_prep"
        else:
            return "campaign_execution"
    
    async def _determine_current_phase_legacy(self, tasks: List[TaskProgress]) -> str:
        """Determine current campaign phase from legacy tasks"""
        if not tasks:
            return "planning"
        
        completed_count = len([t for t in tasks if t.status in ['completed', 'approved', 'scheduled']])
        in_progress_count = len([t for t in tasks if t.status in ['in_progress', 'generated', 'under_review']])
        
        if completed_count == 0 and in_progress_count == 0:
            return "planning"
        elif in_progress_count > 0:
            return "content_creation"
        elif completed_count < len(tasks):
            return "content_review"
        else:
            return "campaign_execution"
    
    async def _persist_progress_state(self, campaign_id: str, progress: CampaignProgress, cur) -> None:
        """Persist calculated progress state to database for consistency"""
        try:
            # Update campaigns table with progress information
            cur.execute("""
                UPDATE campaigns 
                SET status = CASE 
                    WHEN %s = 100.0 THEN 'completed'
                    WHEN %s > 0 THEN 'active' 
                    ELSE status 
                END,
                updated_at = NOW()
                WHERE id = %s
            """, (progress.progress_percentage, progress.progress_percentage, campaign_id))
            
            # Try to update orchestration progress (if tables exist)
            try:
                cur.execute("""
                    UPDATE campaign_workflows 
                    SET steps_total = %s,
                        steps_completed = %s,
                        updated_at = NOW()
                    WHERE campaign_id = %s
                """, (progress.total_tasks, progress.completed_tasks, campaign_id))
            except Exception:
                # Orchestration tables may not exist
                pass
            
            logger.debug(f"Persisted progress state for campaign {campaign_id}: {progress.progress_percentage}%")
            
        except Exception as e:
            logger.warning(f"Error persisting progress state: {e}")
    
    async def sync_workflow_to_database(self, campaign_id: str, workflow_state: Dict[str, Any]) -> None:
        """
        Synchronize in-memory workflow state to database.
        This ensures workflow progress is persisted and consistent across restarts.
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get or create workflow record
                workflow_id = await self._get_or_create_workflow(campaign_id, cur)
                
                # Update workflow state
                cur.execute("""
                    UPDATE campaign_workflows
                    SET current_step = %s,
                        execution_context = %s,
                        status = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (
                    workflow_state.get('current_step'),
                    json.dumps(workflow_state.get('context', {})),
                    workflow_state.get('status', 'running'),
                    workflow_id
                ))
                
                # Sync workflow steps
                await self._sync_workflow_steps(workflow_id, workflow_state.get('steps', []), cur)
                
                conn.commit()
                logger.debug(f"Synced workflow state for campaign {campaign_id}")
                
        except Exception as e:
            logger.error(f"Error syncing workflow to database: {e}")
    
    async def _get_or_create_workflow(self, campaign_id: str, cur) -> str:
        """Get existing workflow or create new one"""
        try:
            # Check if workflow exists
            cur.execute("""
                SELECT id FROM campaign_workflows 
                WHERE campaign_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (campaign_id,))
            
            row = cur.fetchone()
            if row:
                return row[0]
            
            # Create new workflow record
            workflow_id = str(uuid.uuid4())
            orchestrator_id = str(uuid.uuid4())  # Default orchestrator
            
            cur.execute("""
                INSERT INTO campaign_workflows 
                (id, campaign_id, orchestrator_id, workflow_instance_id, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'running', NOW(), NOW())
            """, (workflow_id, campaign_id, orchestrator_id, workflow_id))
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error getting or creating workflow: {e}")
            return str(uuid.uuid4())
    
    async def _sync_workflow_steps(self, workflow_id: str, steps: List[Dict[str, Any]], cur) -> None:
        """Sync workflow steps to database"""
        try:
            for i, step in enumerate(steps):
                step_id = step.get('id', str(uuid.uuid4()))
                
                # Insert or update step
                cur.execute("""
                    INSERT INTO campaign_workflow_steps 
                    (id, workflow_id, step_name, step_type, step_order, status, 
                     input_data, output_data, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        output_data = EXCLUDED.output_data,
                        updated_at = NOW()
                """, (
                    step_id, workflow_id, step.get('name', f'Step {i+1}'),
                    step.get('type', 'content_creation'), i,
                    step.get('status', 'pending'),
                    json.dumps(step.get('input', {})),
                    json.dumps(step.get('output', {}))
                ))
                
        except Exception as e:
            logger.warning(f"Error syncing workflow steps: {e}")
    
    async def update_task_progress(self, campaign_id: str, task_id: str, 
                                 new_status: str, result: Optional[str] = None,
                                 error: Optional[str] = None) -> None:
        """
        Update individual task progress.
        Fixed to avoid recursive calls and transaction conflicts.
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Update legacy task only (remove orchestration update to avoid schema conflicts)
                cur.execute("""
                    UPDATE campaign_tasks
                    SET status = %s,
                        result = COALESCE(%s, result),
                        error = %s,
                        completed_at = CASE WHEN %s IN ('completed', 'approved', 'scheduled', 'generated') 
                                          THEN NOW() ELSE completed_at END,
                        updated_at = NOW()
                    WHERE id = %s AND campaign_id = %s
                """, (new_status, result, error, new_status, task_id, campaign_id))
                
                # Commit immediately to avoid transaction conflicts
                conn.commit()
                logger.debug(f"Updated task {task_id} progress: {new_status}")
                
        except Exception as e:
            logger.error(f"Error updating task progress: {e}")
            # Don't re-raise to avoid breaking the execution flow

# Global service instance
campaign_progress_service = CampaignProgressService()