#!/usr/bin/env python3
"""
Content Workflow Manager - High-level orchestration for content generation workflows

This module provides high-level management for content generation workflows,
integrating with the campaign orchestration system and managing the lifecycle
of content creation processes.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.agents.core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentType
from src.agents.core.agent_factory import create_agent
from src.agents.workflow.content_generation_workflow import (
    ContentGenerationWorkflow, ContentGenerationPlan, ContentTask, 
    ContentTaskStatus, ContentTaskPriority, content_generation_workflow
)
# Temporarily disabled due to circular imports
# from src.agents.workflow.enhanced.workflow_execution_engine import (
#     WorkflowExecutionEngine, ExecutionContext, ExecutionMode, 
#     ErrorRecoveryStrategy, ExecutionResult
# )
# Removed circular import - using local enums instead  
# from src.agents.workflow.enhanced.enhanced_workflow_state import CampaignWorkflowState, WorkflowStatus
from src.agents.specialized.campaign_manager import CampaignManagerAgent

logger = logging.getLogger(__name__)

class WorkflowPhase(Enum):
    """Phases of content workflow execution"""
    PLANNING = "planning"
    STRATEGY_ANALYSIS = "strategy_analysis"
    TASK_CREATION = "task_creation"
    CONTENT_GENERATION = "content_generation"
    QUALITY_REVIEW = "quality_review"
    APPROVAL = "approval"
    DELIVERY = "delivery"
    COMPLETION = "completion"

class WorkflowTrigger(Enum):
    """Triggers for workflow initiation"""
    CAMPAIGN_CREATED = "campaign_created"
    CONTENT_REQUESTED = "content_requested"
    SCHEDULED_EXECUTION = "scheduled_execution"
    MANUAL_TRIGGER = "manual_trigger"
    STRATEGY_UPDATED = "strategy_updated"

class ExecutionMode(Enum):
    """Execution modes (local definition to avoid circular imports)"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STEP_BY_STEP = "step_by_step"
    DEBUG = "debug"

@dataclass
class ContentWorkflowConfig:
    """Configuration for content workflow execution"""
    workflow_id: str
    campaign_id: str
    trigger: WorkflowTrigger
    execution_mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS
    max_concurrent_tasks: int = 5
    auto_approve_quality_threshold: float = 8.0
    require_human_review: bool = False
    deadline: Optional[datetime] = None
    priority: ContentTaskPriority = ContentTaskPriority.MEDIUM
    notification_settings: Dict[str, bool] = field(default_factory=lambda: {
        'on_start': True,
        'on_completion': True,
        'on_error': True,
        'on_milestone': False
    })
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowMetrics:
    """Metrics for content workflow performance"""
    workflow_id: str
    campaign_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_quality_score: float = 0.0
    average_task_duration_seconds: float = 0.0
    throughput_tasks_per_hour: float = 0.0
    success_rate: float = 0.0
    phases_completed: List[WorkflowPhase] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0

class ContentWorkflowManager:
    """
    High-level manager for content generation workflows
    
    This manager orchestrates the entire content creation process from campaign
    strategy analysis through content delivery and tracking. It integrates with
    the campaign orchestration system and provides comprehensive workflow management.
    """
    
    def __init__(self):
        self.manager_id = "content_workflow_manager"
        self.description = "High-level orchestration for content generation workflows"
        self.content_workflow = content_generation_workflow
        # self.execution_engine = WorkflowExecutionEngine()  # Disabled due to circular imports
        self.campaign_manager = CampaignManagerAgent()
        
        # Active workflow tracking
        self.active_workflows: Dict[str, ContentWorkflowConfig] = {}
        self.workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self.workflow_callbacks: Dict[str, List[Callable]] = {}
        
    async def initiate_content_workflow(self, campaign_id: str, 
                                      trigger: WorkflowTrigger = WorkflowTrigger.CAMPAIGN_CREATED,
                                      config: Optional[ContentWorkflowConfig] = None) -> str:
        """
        Initiate a complete content generation workflow for a campaign
        """
        try:
            workflow_id = str(uuid.uuid4())
            logger.info(f"Initiating content workflow {workflow_id} for campaign {campaign_id}")
            
            # Create workflow configuration
            if config is None:
                config = ContentWorkflowConfig(
                    workflow_id=workflow_id,
                    campaign_id=campaign_id,
                    trigger=trigger
                )
            else:
                config.workflow_id = workflow_id
                config.campaign_id = campaign_id
            
            # Initialize workflow metrics
            metrics = WorkflowMetrics(
                workflow_id=workflow_id,
                campaign_id=campaign_id,
                start_time=datetime.now()
            )
            
            # Store active workflow
            self.active_workflows[workflow_id] = config
            self.workflow_metrics[workflow_id] = metrics
            
            # Send start notification
            if config.notification_settings.get('on_start', True):
                await self._send_workflow_notification(workflow_id, "Workflow Started", 
                                                     f"Content generation workflow initiated for campaign {campaign_id}")
            
            # Execute workflow phases
            result = await self._execute_workflow_phases(config, metrics)
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error initiating content workflow: {str(e)}")
            raise
    
    async def _execute_workflow_phases(self, config: ContentWorkflowConfig, 
                                     metrics: WorkflowMetrics) -> Dict[str, Any]:
        """
        Execute all phases of the content workflow
        """
        try:
            workflow_id = config.workflow_id
            campaign_id = config.campaign_id
            
            # Phase 1: Strategy Analysis
            await self._execute_phase(WorkflowPhase.PLANNING, config, metrics)
            strategy_analysis = await self._analyze_campaign_strategy(campaign_id)
            
            # Phase 2: Task Creation and Planning
            await self._execute_phase(WorkflowPhase.TASK_CREATION, config, metrics)
            generation_plan = await self.content_workflow.create_content_generation_plan(
                campaign_id, strategy_analysis
            )
            
            # Update metrics with task information
            metrics.total_tasks = generation_plan.total_tasks
            
            # Phase 3: Content Generation
            await self._execute_phase(WorkflowPhase.CONTENT_GENERATION, config, metrics)
            generation_result = await self.content_workflow.execute_content_generation_plan(campaign_id)
            
            # Phase 4: Quality Review
            await self._execute_phase(WorkflowPhase.QUALITY_REVIEW, config, metrics)
            review_result = await self._execute_quality_review(generation_plan, config)
            
            # Phase 5: Approval (if required)
            if config.require_human_review or review_result.get('requires_human_review', False):
                await self._execute_phase(WorkflowPhase.APPROVAL, config, metrics)
                approval_result = await self._execute_approval_process(generation_plan, config)
            else:
                approval_result = {'auto_approved': True}
            
            # Phase 6: Delivery
            await self._execute_phase(WorkflowPhase.DELIVERY, config, metrics)
            delivery_result = await self._execute_content_delivery(generation_plan, config)
            
            # Phase 7: Completion
            await self._execute_phase(WorkflowPhase.COMPLETION, config, metrics)
            completion_result = await self._finalize_workflow(config, metrics)
            
            # Update final metrics
            metrics.end_time = datetime.now()
            metrics.total_duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.completed_tasks = generation_result.get('completed_tasks', 0)
            metrics.failed_tasks = generation_result.get('failed_tasks', 0)
            metrics.success_rate = (metrics.completed_tasks / metrics.total_tasks * 100) if metrics.total_tasks > 0 else 0
            
            # Calculate quality metrics
            generated_content = generation_result.get('generated_content', [])
            if generated_content:
                quality_scores = [content.quality_score for content in generated_content if content and hasattr(content, 'quality_score')]
                if quality_scores:
                    metrics.average_quality_score = sum(quality_scores) / len(quality_scores)
            
            # Calculate throughput
            if metrics.total_duration_seconds > 0:
                metrics.throughput_tasks_per_hour = (metrics.completed_tasks * 3600) / metrics.total_duration_seconds
            
            # Send completion notification
            if config.notification_settings.get('on_completion', True):
                await self._send_workflow_notification(
                    workflow_id, 
                    "Workflow Completed", 
                    f"Content generation completed: {metrics.completed_tasks}/{metrics.total_tasks} tasks successful"
                )
            
            # Clean up active workflow
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            return {
                'workflow_id': workflow_id,
                'campaign_id': campaign_id,
                'success': True,
                'generation_result': generation_result,
                'review_result': review_result,
                'approval_result': approval_result,
                'delivery_result': delivery_result,
                'metrics': metrics,
                'completion_result': completion_result
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow phases: {str(e)}")
            
            # Update metrics with error
            metrics.error_count += 1
            metrics.end_time = datetime.now()
            
            # Send error notification
            if config.notification_settings.get('on_error', True):
                await self._send_workflow_notification(
                    config.workflow_id,
                    "Workflow Error",
                    f"Content generation workflow failed: {str(e)}"
                )
            
            raise
    
    async def _execute_phase(self, phase: WorkflowPhase, config: ContentWorkflowConfig, 
                           metrics: WorkflowMetrics) -> None:
        """Execute a specific workflow phase"""
        try:
            logger.info(f"Executing phase: {phase.value} for workflow {config.workflow_id}")
            
            # Record phase start
            phase_start = datetime.now()
            
            # Add phase to completed phases
            if phase not in metrics.phases_completed:
                metrics.phases_completed.append(phase)
            
            # Send milestone notification if enabled
            if config.notification_settings.get('on_milestone', False):
                await self._send_workflow_notification(
                    config.workflow_id,
                    f"Phase {phase.value.title()} Started",
                    f"Executing {phase.value} phase for campaign {config.campaign_id}"
                )
            
            # Phase-specific processing time simulation
            await asyncio.sleep(0.1)  # Minimal processing time
            
            logger.debug(f"Completed phase: {phase.value} in {(datetime.now() - phase_start).total_seconds():.2f}s")
            
        except Exception as e:
            logger.error(f"Error executing phase {phase.value}: {str(e)}")
            raise
    
    async def _analyze_campaign_strategy(self, campaign_id: str) -> Dict[str, Any]:
        """
        Analyze campaign strategy to determine content requirements
        """
        try:
            logger.info(f"Analyzing campaign strategy for {campaign_id}")
            
            # Get campaign briefing and strategy from Campaign Manager Agent
            campaign_context = AgentExecutionContext()
            
            # Use Campaign Manager Agent to analyze the campaign
            strategy_result = self.campaign_manager.execute({
                'action': 'analyze_campaign_strategy',
                'campaign_id': campaign_id,
                'analysis_depth': 'comprehensive'
            }, campaign_context)
            
            if strategy_result.success and strategy_result.data:
                campaign_strategy = strategy_result.data
            else:
                # Fallback: get basic campaign information
                campaign_strategy = await self._get_basic_campaign_info(campaign_id)
            
            # Enhance strategy with content-specific analysis
            enhanced_strategy = await self._enhance_strategy_for_content(campaign_strategy, campaign_id)
            
            logger.info(f"Campaign strategy analysis completed for {campaign_id}")
            return enhanced_strategy
            
        except Exception as e:
            logger.error(f"Error analyzing campaign strategy: {str(e)}")
            raise
    
    async def _enhance_strategy_for_content(self, base_strategy: Dict[str, Any], 
                                          campaign_id: str) -> Dict[str, Any]:
        """
        Enhance campaign strategy with content-specific insights
        """
        try:
            # Add content-specific recommendations
            enhanced_strategy = base_strategy.copy()
            
            # Determine optimal content mix
            content_mix = self._determine_optimal_content_mix(base_strategy)
            enhanced_strategy['recommended_content_mix'] = content_mix
            
            # Add SEO keywords and content pillars
            seo_analysis = await self._perform_seo_analysis(base_strategy)
            enhanced_strategy.update(seo_analysis)
            
            # Add timing and scheduling recommendations
            scheduling_analysis = self._analyze_content_scheduling(base_strategy)
            enhanced_strategy['content_scheduling'] = scheduling_analysis
            
            # Add competitive insights if available
            competitive_insights = await self._get_competitive_insights(campaign_id)
            if competitive_insights:
                enhanced_strategy['competitive_insights'] = competitive_insights
            
            return enhanced_strategy
            
        except Exception as e:
            logger.warning(f"Error enhancing strategy for content: {str(e)}")
            return base_strategy  # Return base strategy if enhancement fails
    
    async def _execute_quality_review(self, generation_plan: ContentGenerationPlan, 
                                    config: ContentWorkflowConfig) -> Dict[str, Any]:
        """
        Execute automated quality review of generated content
        """
        try:
            logger.info(f"Executing quality review for plan {generation_plan.plan_id}")
            
            review_results = []
            requires_human_review = False
            
            for task in generation_plan.content_tasks:
                if task.generated_content:
                    content = task.generated_content
                    
                    # Perform quality checks
                    quality_check = {
                        'task_id': task.task_id,
                        'content_id': content.content_id,
                        'quality_score': content.quality_score,
                        'word_count_check': self._check_word_count(content, task),
                        'seo_score': getattr(content, 'seo_score', None),
                        'brand_consistency': self._check_brand_consistency(content),
                        'readability_score': self._calculate_readability_score(content.content),
                        'automated_review_passed': content.quality_score >= config.auto_approve_quality_threshold
                    }
                    
                    # Check if human review is needed
                    if content.quality_score < config.auto_approve_quality_threshold:
                        requires_human_review = True
                        quality_check['review_required'] = True
                        quality_check['review_reason'] = f"Quality score {content.quality_score} below threshold {config.auto_approve_quality_threshold}"
                    
                    review_results.append(quality_check)
            
            # Calculate overall quality metrics
            quality_scores = [r['quality_score'] for r in review_results if r['quality_score']]
            average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            passed_count = len([r for r in review_results if r.get('automated_review_passed', False)])
            pass_rate = (passed_count / len(review_results)) * 100 if review_results else 0
            
            return {
                'plan_id': generation_plan.plan_id,
                'total_content_pieces': len(review_results),
                'average_quality_score': average_quality,
                'automated_pass_rate': pass_rate,
                'requires_human_review': requires_human_review,
                'review_results': review_results,
                'review_completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing quality review: {str(e)}")
            raise
    
    async def _execute_approval_process(self, generation_plan: ContentGenerationPlan, 
                                      config: ContentWorkflowConfig) -> Dict[str, Any]:
        """
        Execute approval process for content requiring human review
        """
        try:
            logger.info(f"Executing approval process for plan {generation_plan.plan_id}")
            
            # For now, simulate approval process
            # In a real implementation, this would integrate with approval workflows
            
            pending_approvals = []
            for task in generation_plan.content_tasks:
                if task.generated_content and task.generated_content.quality_score < config.auto_approve_quality_threshold:
                    pending_approvals.append({
                        'task_id': task.task_id,
                        'content_id': task.generated_content.content_id,
                        'content_type': task.content_type.value,
                        'quality_score': task.generated_content.quality_score,
                        'status': 'pending_approval',
                        'submitted_at': datetime.now().isoformat()
                    })
            
            return {
                'plan_id': generation_plan.plan_id,
                'pending_approvals': pending_approvals,
                'approval_process_initiated': True,
                'estimated_approval_time': '24-48 hours',
                'approval_initiated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing approval process: {str(e)}")
            raise
    
    async def _execute_content_delivery(self, generation_plan: ContentGenerationPlan, 
                                      config: ContentWorkflowConfig) -> Dict[str, Any]:
        """
        Execute content delivery process
        """
        try:
            logger.info(f"Executing content delivery for plan {generation_plan.plan_id}")
            
            delivered_content = []
            
            for task in generation_plan.content_tasks:
                if task.status == ContentTaskStatus.COMPLETED and task.generated_content:
                    # Simulate content delivery
                    delivery_info = {
                        'task_id': task.task_id,
                        'content_id': task.generated_content.content_id,
                        'content_type': task.content_type.value,
                        'channel': task.channel.value,
                        'title': task.generated_content.title,
                        'word_count': task.generated_content.word_count,
                        'quality_score': task.generated_content.quality_score,
                        'delivered_at': datetime.now().isoformat(),
                        'delivery_status': 'delivered'
                    }
                    delivered_content.append(delivery_info)
            
            return {
                'plan_id': generation_plan.plan_id,
                'delivered_content_count': len(delivered_content),
                'delivered_content': delivered_content,
                'delivery_completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing content delivery: {str(e)}")
            raise
    
    async def _finalize_workflow(self, config: ContentWorkflowConfig, 
                               metrics: WorkflowMetrics) -> Dict[str, Any]:
        """
        Finalize workflow and perform cleanup
        """
        try:
            logger.info(f"Finalizing workflow {config.workflow_id}")
            
            # Create final report
            final_report = {
                'workflow_id': config.workflow_id,
                'campaign_id': config.campaign_id,
                'execution_summary': {
                    'total_tasks': metrics.total_tasks,
                    'completed_tasks': metrics.completed_tasks,
                    'failed_tasks': metrics.failed_tasks,
                    'success_rate': metrics.success_rate,
                    'average_quality_score': metrics.average_quality_score,
                    'total_duration_seconds': metrics.total_duration_seconds,
                    'throughput_tasks_per_hour': metrics.throughput_tasks_per_hour
                },
                'phases_completed': [phase.value for phase in metrics.phases_completed],
                'completion_timestamp': datetime.now().isoformat()
            }
            
            # Archive workflow data
            await self._archive_workflow_data(config, metrics, final_report)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error finalizing workflow: {str(e)}")
            raise
    
    # Helper methods
    
    async def _get_basic_campaign_info(self, campaign_id: str) -> Dict[str, Any]:
        """Get basic campaign information as fallback"""
        # Implementation would fetch from database
        return {
            'campaign_id': campaign_id,
            'campaign_name': f'Campaign {campaign_id}',
            'objectives': ['content_marketing'],
            'target_audience': 'B2B professionals',
            'channels': ['blog', 'linkedin', 'email'],
            'duration_weeks': 4,
            'content_frequency': 'weekly'
        }
    
    def _determine_optimal_content_mix(self, strategy: Dict[str, Any]) -> Dict[str, int]:
        """Determine optimal content mix based on strategy"""
        objectives = strategy.get('objectives', [])
        channels = strategy.get('channels', [])
        
        content_mix = {
            'blog_posts': 2,
            'social_posts': 4,
            'email_content': 2,
            'case_studies': 1,
            'newsletters': 1
        }
        
        # Adjust based on objectives
        if 'lead_generation' in objectives:
            content_mix['case_studies'] += 1
            content_mix['email_content'] += 1
        
        if 'thought_leadership' in objectives:
            content_mix['blog_posts'] += 1
        
        return content_mix
    
    async def _perform_seo_analysis(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SEO analysis for content strategy"""
        # Simplified SEO analysis
        return {
            'seo_keywords': ['B2B', 'business strategy', 'digital transformation'],
            'content_pillars': ['Industry Insights', 'Best Practices', 'Strategic Planning'],
            'seo_focus': True,
            'target_search_volume': 1000
        }
    
    def _analyze_content_scheduling(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content scheduling requirements"""
        duration_weeks = strategy.get('duration_weeks', 4)
        frequency = strategy.get('content_frequency', 'weekly')
        
        return {
            'recommended_frequency': frequency,
            'optimal_posting_times': {
                'blog': '9:00 AM',
                'linkedin': '8:00 AM',
                'email': '10:00 AM'
            },
            'content_calendar_duration': duration_weeks,
            'total_pieces_recommended': duration_weeks * 2  # 2 pieces per week
        }
    
    async def _get_competitive_insights(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get competitive insights for content strategy"""
        # This would integrate with competitive intelligence agents
        return None
    
    def _check_word_count(self, content: Any, task: ContentTask) -> bool:
        """Check if content meets word count requirements"""
        if not task.word_count:
            return True
        
        actual_count = getattr(content, 'word_count', 0)
        target_count = task.word_count
        
        # Allow 20% variance
        return abs(actual_count - target_count) <= (target_count * 0.2)
    
    def _check_brand_consistency(self, content: Any) -> float:
        """Check brand consistency of content"""
        # Simplified brand consistency check
        return 85.0  # Placeholder score
    
    def _calculate_readability_score(self, content_text: str) -> float:
        """Calculate readability score of content"""
        # Simplified readability calculation
        words = len(content_text.split())
        sentences = content_text.count('.') + content_text.count('!') + content_text.count('?')
        
        if sentences == 0:
            return 50.0
        
        avg_sentence_length = words / sentences
        
        # Higher score for moderate sentence length
        if 15 <= avg_sentence_length <= 20:
            return 90.0
        elif 10 <= avg_sentence_length <= 25:
            return 75.0
        else:
            return 60.0
    
    async def _send_workflow_notification(self, workflow_id: str, title: str, message: str) -> None:
        """Send workflow notification"""
        logger.info(f"[NOTIFICATION] {title}: {message}")
        # In a real implementation, this would send notifications via email, Slack, etc.
    
    async def _archive_workflow_data(self, config: ContentWorkflowConfig, 
                                   metrics: WorkflowMetrics, final_report: Dict[str, Any]) -> None:
        """Archive workflow data for historical analysis"""
        try:
            # Store workflow data in database or archive system
            archive_data = {
                'config': config.__dict__,
                'metrics': metrics.__dict__,
                'final_report': final_report
            }
            
            # In a real implementation, this would save to a database
            logger.info(f"Archived workflow data for {config.workflow_id}")
            
        except Exception as e:
            logger.warning(f"Error archiving workflow data: {str(e)}")
    
    # Public API methods
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id not in self.active_workflows:
            return None
        
        config = self.active_workflows[workflow_id]
        metrics = self.workflow_metrics.get(workflow_id)
        
        status = {
            'workflow_id': workflow_id,
            'campaign_id': config.campaign_id,
            'current_phase': metrics.phases_completed[-1].value if metrics and metrics.phases_completed else 'not_started',
            'total_phases': len(WorkflowPhase),
            'completed_phases': len(metrics.phases_completed) if metrics else 0,
            'is_active': True,
            'started_at': metrics.start_time.isoformat() if metrics else None
        }
        
        if metrics:
            status.update({
                'total_tasks': metrics.total_tasks,
                'completed_tasks': metrics.completed_tasks,
                'failed_tasks': metrics.failed_tasks,
                'progress_percentage': (len(metrics.phases_completed) / len(WorkflowPhase)) * 100
            })
        
        return status
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause an active workflow"""
        if workflow_id not in self.active_workflows:
            return False
        
        config = self.active_workflows[workflow_id]
        
        # Pause the underlying content workflow
        paused = await self.content_workflow.pause_workflow(config.campaign_id)
        
        if paused:
            logger.info(f"Paused workflow: {workflow_id}")
            await self._send_workflow_notification(workflow_id, "Workflow Paused", 
                                                 "Content generation workflow has been paused")
        
        return paused
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow"""
        if workflow_id not in self.active_workflows:
            return False
        
        config = self.active_workflows[workflow_id]
        
        # Resume the underlying content workflow
        resumed = await self.content_workflow.resume_workflow(config.campaign_id)
        
        if resumed:
            logger.info(f"Resumed workflow: {workflow_id}")
            await self._send_workflow_notification(workflow_id, "Workflow Resumed", 
                                                 "Content generation workflow has been resumed")
        
        return resumed
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id not in self.active_workflows:
            return False
        
        config = self.active_workflows[workflow_id]
        
        # Cancel the underlying content workflow
        cancelled = await self.content_workflow.cancel_workflow(config.campaign_id)
        
        if cancelled:
            # Clean up
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            logger.info(f"Cancelled workflow: {workflow_id}")
            await self._send_workflow_notification(workflow_id, "Workflow Cancelled", 
                                                 "Content generation workflow has been cancelled")
        
        return cancelled
    
    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a workflow"""
        return self.workflow_metrics.get(workflow_id)
    
    def list_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs"""
        return list(self.active_workflows.keys())

# Create global workflow manager instance
content_workflow_manager = ContentWorkflowManager()