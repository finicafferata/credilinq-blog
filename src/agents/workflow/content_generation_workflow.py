#!/usr/bin/env python3
"""
Content Generation Workflow - Campaign-Centric Content Pipeline

This module implements a comprehensive content generation workflow that orchestrates
campaign-driven content creation using AI agents. It connects Campaign Manager Agents
with AI Content Generator Agents to create multi-format content based on campaign strategies.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.agents.core.base_agent import BaseAgent, AgentResult, AgentExecutionContext
from src.agents.core.agent_factory import create_agent, AgentType
from src.agents.specialized.ai_content_generator import (
    AIContentGeneratorAgent, ContentGenerationRequest, 
    ContentType, ContentChannel, GeneratedContent
)
# Removed circular import - using local enums instead
# from src.agents.workflow.enhanced.enhanced_workflow_state import CampaignWorkflowState, WorkflowStatus
from src.config.database import db_config

logger = logging.getLogger(__name__)

class ContentTaskStatus(Enum):
    """Status of content generation tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_REVIEW = "requires_review"

class ContentTaskPriority(Enum):
    """Priority levels for content tasks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class ContentTask:
    """Individual content generation task"""
    task_id: str
    campaign_id: str
    content_type: ContentType
    channel: ContentChannel
    title: Optional[str] = None
    themes: List[str] = field(default_factory=list)
    key_messages: List[str] = field(default_factory=list)
    target_audience: str = "B2B professionals"
    tone: str = "Professional"
    word_count: Optional[int] = None
    call_to_action: Optional[str] = None
    seo_keywords: List[str] = field(default_factory=list)
    status: ContentTaskStatus = ContentTaskStatus.PENDING
    priority: ContentTaskPriority = ContentTaskPriority.MEDIUM
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    generated_content: Optional[GeneratedContent] = None
    review_notes: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ContentGenerationPlan:
    """Plan for content generation workflow"""
    plan_id: str
    campaign_id: str
    content_tasks: List[ContentTask]
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    estimated_completion: Optional[datetime] = None
    workflow_status: ContentTaskStatus = ContentTaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContentGenerationWorkflow:
    """
    Content Generation Workflow - Orchestrates AI content creation
    
    This workflow manages the end-to-end process of:
    1. Analyzing campaign requirements
    2. Creating content generation tasks
    3. Executing content creation with AI agents
    4. Managing task dependencies and scheduling
    5. Quality review and approval process
    6. Content delivery and tracking
    """
    
    def __init__(self):
        self.workflow_id = "content_generation_workflow"
        self.description = "Campaign-centric AI content generation pipeline"
        self.ai_content_generator = AIContentGeneratorAgent()
        self.active_workflows: Dict[str, ContentGenerationPlan] = {}
        self.task_queue: List[ContentTask] = []
        self.max_concurrent_tasks = 5
        
    async def create_content_generation_plan(self, campaign_id: str, 
                                           campaign_strategy: Dict[str, Any]) -> ContentGenerationPlan:
        """
        Create a comprehensive content generation plan based on campaign strategy
        """
        try:
            logger.info(f"Creating content generation plan for campaign: {campaign_id}")
            
            # Analyze campaign requirements
            content_requirements = await self._analyze_campaign_requirements(campaign_id, campaign_strategy)
            
            # Create content tasks based on requirements
            content_tasks = await self._create_content_tasks(campaign_id, content_requirements)
            
            # Optimize task scheduling and dependencies
            optimized_tasks = await self._optimize_task_scheduling(content_tasks, campaign_strategy)
            
            # Create generation plan
            plan = ContentGenerationPlan(
                plan_id=str(uuid.uuid4()),
                campaign_id=campaign_id,
                content_tasks=optimized_tasks,
                total_tasks=len(optimized_tasks),
                estimated_completion=self._calculate_estimated_completion(optimized_tasks),
                metadata={
                    'campaign_strategy': campaign_strategy,
                    'content_types': list(set(task.content_type for task in optimized_tasks)),
                    'channels': list(set(task.channel for task in optimized_tasks)),
                    'created_by': 'content_generation_workflow'
                }
            )
            
            # Save plan to database
            await self._save_generation_plan(plan)
            
            # Add to active workflows
            self.active_workflows[campaign_id] = plan
            
            logger.info(f"Created content generation plan with {len(optimized_tasks)} tasks")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating content generation plan: {str(e)}")
            raise

    async def execute_content_generation_plan(self, campaign_id: str) -> Dict[str, Any]:
        """
        Execute the content generation plan for a campaign
        """
        try:
            if campaign_id not in self.active_workflows:
                raise ValueError(f"No active content generation plan for campaign: {campaign_id}")
            
            plan = self.active_workflows[campaign_id]
            logger.info(f"Executing content generation plan: {plan.plan_id}")
            
            # Update workflow status
            plan.workflow_status = ContentTaskStatus.IN_PROGRESS
            
            # Execute tasks based on priority and dependencies
            execution_results = await self._execute_content_tasks(plan)
            
            # Update plan status
            plan.completed_tasks = len([t for t in plan.content_tasks if t.status == ContentTaskStatus.COMPLETED])
            plan.failed_tasks = len([t for t in plan.content_tasks if t.status == ContentTaskStatus.FAILED])
            
            if plan.completed_tasks == plan.total_tasks:
                plan.workflow_status = ContentTaskStatus.COMPLETED
            elif plan.failed_tasks > 0:
                plan.workflow_status = ContentTaskStatus.FAILED
            
            # Save updated plan
            await self._save_generation_plan(plan)
            
            return {
                'plan_id': plan.plan_id,
                'campaign_id': campaign_id,
                'total_tasks': plan.total_tasks,
                'completed_tasks': plan.completed_tasks,
                'failed_tasks': plan.failed_tasks,
                'status': plan.workflow_status.value,
                'execution_results': execution_results,
                'generated_content': [task.generated_content for task in plan.content_tasks if task.generated_content]
            }
            
        except Exception as e:
            logger.error(f"Error executing content generation plan: {str(e)}")
            raise

    async def _analyze_campaign_requirements(self, campaign_id: str, 
                                           campaign_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze campaign strategy to determine content requirements
        """
        try:
            # Extract key information from campaign strategy
            content_pillars = campaign_strategy.get('content_pillars', [])
            target_channels = campaign_strategy.get('channels', [])
            campaign_objectives = campaign_strategy.get('objectives', [])
            target_audience = campaign_strategy.get('target_audience', 'B2B professionals')
            tone = campaign_strategy.get('tone', 'Professional')
            
            # Get campaign details from database
            campaign_details = await self._get_campaign_details(campaign_id)
            
            # Map strategy to content requirements
            requirements = {
                'primary_content_types': self._determine_primary_content_types(campaign_objectives, target_channels),
                'content_themes': content_pillars[:5],  # Limit to top 5 themes
                'distribution_channels': self._map_channels_to_content_channels(target_channels),
                'content_volume': self._calculate_content_volume(campaign_strategy),
                'quality_requirements': {
                    'target_audience': target_audience,
                    'tone': tone,
                    'seo_optimization': campaign_strategy.get('seo_focus', True),
                    'brand_consistency': True
                },
                'timeline_requirements': self._extract_timeline_requirements(campaign_strategy),
                'campaign_context': campaign_details
            }
            
            logger.info(f"Analyzed campaign requirements for {len(requirements['primary_content_types'])} content types")
            return requirements
            
        except Exception as e:
            logger.error(f"Error analyzing campaign requirements: {str(e)}")
            raise

    async def _create_content_tasks(self, campaign_id: str, 
                                  content_requirements: Dict[str, Any]) -> List[ContentTask]:
        """
        Create specific content generation tasks based on requirements
        """
        try:
            tasks = []
            
            content_types = content_requirements['primary_content_types']
            themes = content_requirements['content_themes']
            channels = content_requirements['distribution_channels']
            quality_reqs = content_requirements['quality_requirements']
            
            # Create tasks for each content type and channel combination
            for content_type in content_types:
                for channel in channels:
                    if self._is_valid_content_channel_combination(content_type, channel):
                        # Create multiple variants for important content types
                        variant_count = self._get_variant_count(content_type, channel)
                        
                        for variant in range(variant_count):
                            task = ContentTask(
                                task_id=str(uuid.uuid4()),
                                campaign_id=campaign_id,
                                content_type=content_type,
                                channel=channel,
                                themes=themes[:3],  # Use top 3 themes per task
                                target_audience=quality_reqs['target_audience'],
                                tone=quality_reqs['tone'],
                                word_count=self._determine_word_count(content_type, channel),
                                priority=self._determine_task_priority(content_type, channel),
                                seo_keywords=content_requirements.get('seo_keywords', []),
                                metadata={
                                    'variant_number': variant + 1,
                                    'total_variants': variant_count,
                                    'content_pillar': themes[0] if themes else 'general',
                                    'campaign_objective': content_requirements.get('primary_objective', '')
                                }
                            )
                            tasks.append(task)
            
            # Add special content tasks based on campaign type
            special_tasks = await self._create_special_content_tasks(campaign_id, content_requirements)
            tasks.extend(special_tasks)
            
            logger.info(f"Created {len(tasks)} content generation tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Error creating content tasks: {str(e)}")
            raise

    async def _execute_content_tasks(self, plan: ContentGenerationPlan) -> List[Dict[str, Any]]:
        """
        Execute content generation tasks with concurrency control
        """
        try:
            execution_results = []
            
            # Group tasks by priority and dependencies
            task_groups = self._organize_tasks_by_execution_order(plan.content_tasks)
            
            # Execute task groups in order
            for group_idx, task_group in enumerate(task_groups):
                logger.info(f"Executing task group {group_idx + 1}/{len(task_groups)} with {len(task_group)} tasks")
                
                # Execute tasks in parallel within each group
                group_results = await self._execute_task_group_concurrent(task_group)
                execution_results.extend(group_results)
                
                # Update plan after each group
                await self._save_generation_plan(plan)
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error executing content tasks: {str(e)}")
            raise

    async def _execute_task_group_concurrent(self, task_group: List[ContentTask]) -> List[Dict[str, Any]]:
        """
        Execute a group of tasks concurrently with proper error handling
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        async def execute_single_task(task: ContentTask) -> Dict[str, Any]:
            async with semaphore:
                try:
                    logger.info(f"Executing content task: {task.task_id} ({task.content_type.value})")
                    
                    # Update task status
                    task.status = ContentTaskStatus.IN_PROGRESS
                    task.updated_at = datetime.now()
                    
                    # Create content generation request
                    generation_request = ContentGenerationRequest(
                        campaign_id=task.campaign_id,
                        content_type=task.content_type,
                        channel=task.channel,
                        title=task.title,
                        themes=task.themes,
                        target_audience=task.target_audience,
                        tone=task.tone,
                        word_count=task.word_count,
                        key_messages=task.key_messages,
                        call_to_action=task.call_to_action,
                        seo_keywords=task.seo_keywords
                    )
                    
                    # Generate content
                    generated_content = await self.ai_content_generator.generate_content(generation_request)
                    
                    # Update task with results
                    task.generated_content = generated_content
                    task.status = ContentTaskStatus.COMPLETED
                    task.updated_at = datetime.now()
                    
                    # Save task to database
                    await self._save_content_task_result(task)
                    
                    return {
                        'task_id': task.task_id,
                        'status': 'success',
                        'content_id': generated_content.content_id,
                        'quality_score': generated_content.quality_score,
                        'word_count': generated_content.word_count
                    }
                    
                except Exception as e:
                    logger.error(f"Error executing task {task.task_id}: {str(e)}")
                    
                    # Update task status
                    task.status = ContentTaskStatus.FAILED
                    task.metadata['error'] = str(e)
                    task.updated_at = datetime.now()
                    
                    return {
                        'task_id': task.task_id,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        # Execute all tasks in the group concurrently
        results = await asyncio.gather(
            *[execute_single_task(task) for task in task_group],
            return_exceptions=True
        )
        
        # Process results and handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task execution exception: {str(result)}")
                processed_results.append({
                    'status': 'exception',
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results

    # Helper methods for workflow execution
    
    def _determine_primary_content_types(self, objectives: List[str], channels: List[str]) -> List[ContentType]:
        """Determine primary content types based on objectives and channels"""
        content_types = []
        
        # Map objectives to content types
        if 'thought_leadership' in str(objectives).lower():
            content_types.extend([ContentType.BLOG_POST, ContentType.LINKEDIN_ARTICLE])
        if 'lead_generation' in str(objectives).lower():
            content_types.extend([ContentType.EMAIL_CONTENT, ContentType.CASE_STUDY])
        if 'engagement' in str(objectives).lower():
            content_types.extend([ContentType.SOCIAL_POST, ContentType.TWITTER_THREAD])
        if 'education' in str(objectives).lower():
            content_types.extend([ContentType.BLOG_POST, ContentType.NEWSLETTER])
        
        # Map channels to content types
        for channel in channels:
            if channel.lower() in ['linkedin']:
                content_types.extend([ContentType.LINKEDIN_ARTICLE, ContentType.SOCIAL_POST])
            elif channel.lower() in ['twitter']:
                content_types.extend([ContentType.TWITTER_THREAD, ContentType.SOCIAL_POST])
            elif channel.lower() in ['email']:
                content_types.extend([ContentType.EMAIL_CONTENT, ContentType.NEWSLETTER])
            elif channel.lower() in ['blog', 'website']:
                content_types.extend([ContentType.BLOG_POST, ContentType.CASE_STUDY])
        
        # Default content types if none specified
        if not content_types:
            content_types = [ContentType.BLOG_POST, ContentType.SOCIAL_POST, ContentType.EMAIL_CONTENT]
        
        return list(set(content_types))  # Remove duplicates
    
    def _map_channels_to_content_channels(self, channels: List[str]) -> List[ContentChannel]:
        """Map campaign channels to content channels"""
        channel_mapping = {
            'linkedin': ContentChannel.LINKEDIN,
            'twitter': ContentChannel.TWITTER,
            'email': ContentChannel.EMAIL,
            'facebook': ContentChannel.FACEBOOK,
            'youtube': ContentChannel.YOUTUBE,
            'blog': ContentChannel.BLOG,
            'website': ContentChannel.BLOG
        }
        
        content_channels = []
        for channel in channels:
            if channel.lower() in channel_mapping:
                content_channels.append(channel_mapping[channel.lower()])
        
        # Default channels if none mapped
        if not content_channels:
            content_channels = [ContentChannel.BLOG, ContentChannel.LINKEDIN, ContentChannel.EMAIL]
        
        return list(set(content_channels))
    
    def _calculate_content_volume(self, campaign_strategy: Dict[str, Any]) -> Dict[str, int]:
        """Calculate content volume based on campaign strategy"""
        duration_weeks = campaign_strategy.get('duration_weeks', 4)
        content_frequency = campaign_strategy.get('content_frequency', 'weekly')
        
        # Base multipliers
        frequency_multipliers = {
            'daily': 7,
            'weekly': 1,
            'bi-weekly': 0.5,
            'monthly': 0.25
        }
        
        base_multiplier = frequency_multipliers.get(content_frequency, 1)
        total_content_pieces = int(duration_weeks * base_multiplier)
        
        return {
            'total_pieces': max(total_content_pieces, 3),  # Minimum 3 pieces
            'duration_weeks': duration_weeks,
            'frequency': content_frequency
        }
    
    def _extract_timeline_requirements(self, campaign_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timeline requirements from campaign strategy"""
        start_date = campaign_strategy.get('start_date')
        end_date = campaign_strategy.get('end_date')
        
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        return {
            'start_date': start_date or datetime.now(),
            'end_date': end_date or (datetime.now() + timedelta(weeks=4)),
            'content_deadlines': campaign_strategy.get('content_deadlines', {}),
            'review_timeline': campaign_strategy.get('review_timeline', 2)  # days
        }
    
    def _is_valid_content_channel_combination(self, content_type: ContentType, channel: ContentChannel) -> bool:
        """Check if content type and channel combination is valid"""
        valid_combinations = {
            ContentType.BLOG_POST: [ContentChannel.BLOG],
            ContentType.SOCIAL_POST: [ContentChannel.LINKEDIN, ContentChannel.TWITTER, ContentChannel.FACEBOOK],
            ContentType.EMAIL_CONTENT: [ContentChannel.EMAIL],
            ContentType.LINKEDIN_ARTICLE: [ContentChannel.LINKEDIN],
            ContentType.TWITTER_THREAD: [ContentChannel.TWITTER],
            ContentType.CASE_STUDY: [ContentChannel.BLOG, ContentChannel.LINKEDIN],
            ContentType.NEWSLETTER: [ContentChannel.EMAIL],
            ContentType.VIDEO_SCRIPT: [ContentChannel.YOUTUBE]
        }
        
        return channel in valid_combinations.get(content_type, [])
    
    def _get_variant_count(self, content_type: ContentType, channel: ContentChannel) -> int:
        """Determine number of variants to create for content type/channel combination"""
        # High-value content gets more variants for A/B testing
        high_value_types = [ContentType.BLOG_POST, ContentType.EMAIL_CONTENT, ContentType.LINKEDIN_ARTICLE]
        
        if content_type in high_value_types:
            return 2  # Create 2 variants for A/B testing
        else:
            return 1  # Single variant for other types
    
    def _determine_word_count(self, content_type: ContentType, channel: ContentChannel) -> Optional[int]:
        """Determine appropriate word count for content type"""
        word_count_map = {
            ContentType.BLOG_POST: 1200,
            ContentType.LINKEDIN_ARTICLE: 800,
            ContentType.SOCIAL_POST: 150,
            ContentType.TWITTER_THREAD: 200,
            ContentType.EMAIL_CONTENT: 300,
            ContentType.CASE_STUDY: 1500,
            ContentType.NEWSLETTER: 600
        }
        
        return word_count_map.get(content_type)
    
    def _determine_task_priority(self, content_type: ContentType, channel: ContentChannel) -> ContentTaskPriority:
        """Determine task priority based on content type and channel"""
        high_priority_types = [ContentType.BLOG_POST, ContentType.EMAIL_CONTENT]
        high_priority_channels = [ContentChannel.BLOG, ContentChannel.EMAIL]
        
        if content_type in high_priority_types or channel in high_priority_channels:
            return ContentTaskPriority.HIGH
        else:
            return ContentTaskPriority.MEDIUM
    
    async def _create_special_content_tasks(self, campaign_id: str, 
                                          content_requirements: Dict[str, Any]) -> List[ContentTask]:
        """Create special content tasks based on campaign needs"""
        special_tasks = []
        
        # Add case study if lead generation is an objective
        objectives = content_requirements.get('objectives', [])
        if 'lead_generation' in str(objectives).lower():
            case_study_task = ContentTask(
                task_id=str(uuid.uuid4()),
                campaign_id=campaign_id,
                content_type=ContentType.CASE_STUDY,
                channel=ContentChannel.BLOG,
                themes=content_requirements['content_themes'][:2],
                priority=ContentTaskPriority.HIGH,
                word_count=1500,
                metadata={'special_task': True, 'task_type': 'lead_generation_case_study'}
            )
            special_tasks.append(case_study_task)
        
        return special_tasks
    
    def _organize_tasks_by_execution_order(self, tasks: List[ContentTask]) -> List[List[ContentTask]]:
        """Organize tasks into execution groups based on priority and dependencies"""
        # Group by priority
        high_priority = [t for t in tasks if t.priority == ContentTaskPriority.HIGH]
        medium_priority = [t for t in tasks if t.priority == ContentTaskPriority.MEDIUM]
        low_priority = [t for t in tasks if t.priority == ContentTaskPriority.LOW]
        
        # Return groups in priority order
        return [group for group in [high_priority, medium_priority, low_priority] if group]
    
    def _calculate_estimated_completion(self, tasks: List[ContentTask]) -> datetime:
        """Calculate estimated completion time for all tasks"""
        # Estimate based on task count and complexity
        total_tasks = len(tasks)
        avg_task_time_minutes = 15  # Average time per content generation task
        
        # Account for concurrency
        estimated_minutes = (total_tasks * avg_task_time_minutes) / self.max_concurrent_tasks
        
        return datetime.now() + timedelta(minutes=estimated_minutes)
    
    # Database operations
    
    async def _get_campaign_details(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign details from database"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT b.campaign_name, b.marketing_objective, b.target_audience, 
                           b.channels, b.company_context, b.desired_tone, b.duration_weeks
                    FROM briefings b
                    WHERE b.campaign_id = %s
                """, (campaign_id,))
                
                row = cur.fetchone()
                if not row:
                    return {}
                
                campaign_name, objective, audience, channels, company_context, tone, duration = row
                
                return {
                    'campaign_name': campaign_name,
                    'marketing_objective': objective,
                    'target_audience': json.loads(audience) if isinstance(audience, str) else audience,
                    'channels': json.loads(channels) if isinstance(channels, str) else channels,
                    'company_context': company_context,
                    'desired_tone': tone,
                    'duration_weeks': duration or 4
                }
                
        except Exception as e:
            logger.warning(f"Error getting campaign details: {str(e)}")
            return {}
    
    async def _save_generation_plan(self, plan: ContentGenerationPlan) -> None:
        """Save content generation plan to database"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Save plan to campaign_tasks table
                cur.execute("""
                    INSERT INTO campaign_tasks (id, campaign_id, task_type, status, result, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        updated_at = NOW()
                """, (
                    plan.plan_id,
                    plan.campaign_id,
                    "content_generation_plan",
                    plan.workflow_status.value,
                    json.dumps({
                        'total_tasks': plan.total_tasks,
                        'completed_tasks': plan.completed_tasks,
                        'failed_tasks': plan.failed_tasks,
                        'estimated_completion': plan.estimated_completion.isoformat() if plan.estimated_completion else None,
                        'metadata': plan.metadata,
                        'content_tasks': [
                            {
                                'task_id': task.task_id,
                                'content_type': task.content_type.value,
                                'channel': task.channel.value,
                                'status': task.status.value,
                                'priority': task.priority.value
                            }
                            for task in plan.content_tasks
                        ]
                    })
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving generation plan: {str(e)}")
    
    async def _save_content_task_result(self, task: ContentTask) -> None:
        """Save content task result to database"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Prepare result data
                result_data = {
                    'task_details': {
                        'content_type': task.content_type.value,
                        'channel': task.channel.value,
                        'themes': task.themes,
                        'target_audience': task.target_audience,
                        'tone': task.tone,
                        'priority': task.priority.value
                    },
                    'execution_result': {
                        'status': task.status.value,
                        'generated_content_id': task.generated_content.content_id if task.generated_content else None,
                        'quality_score': task.generated_content.quality_score if task.generated_content else None,
                        'word_count': task.generated_content.word_count if task.generated_content else None,
                        'error': task.metadata.get('error')
                    },
                    'metadata': task.metadata
                }
                
                cur.execute("""
                    INSERT INTO campaign_tasks (id, campaign_id, task_type, status, result, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        updated_at = NOW()
                """, (
                    task.task_id,
                    task.campaign_id,
                    f"content_task_{task.content_type.value}",
                    task.status.value,
                    json.dumps(result_data)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving content task result: {str(e)}")

    # Public API methods for workflow management
    
    async def get_workflow_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of content generation workflow"""
        if campaign_id not in self.active_workflows:
            return None
        
        plan = self.active_workflows[campaign_id]
        
        return {
            'plan_id': plan.plan_id,
            'campaign_id': campaign_id,
            'total_tasks': plan.total_tasks,
            'completed_tasks': plan.completed_tasks,
            'failed_tasks': plan.failed_tasks,
            'pending_tasks': len([t for t in plan.content_tasks if t.status == ContentTaskStatus.PENDING]),
            'in_progress_tasks': len([t for t in plan.content_tasks if t.status == ContentTaskStatus.IN_PROGRESS]),
            'workflow_status': plan.workflow_status.value,
            'estimated_completion': plan.estimated_completion.isoformat() if plan.estimated_completion else None,
            'progress_percentage': (plan.completed_tasks / plan.total_tasks * 100) if plan.total_tasks > 0 else 0
        }
    
    async def pause_workflow(self, campaign_id: str) -> bool:
        """Pause content generation workflow"""
        if campaign_id not in self.active_workflows:
            return False
        
        plan = self.active_workflows[campaign_id]
        plan.workflow_status = ContentTaskStatus.PENDING
        await self._save_generation_plan(plan)
        
        logger.info(f"Paused content generation workflow for campaign: {campaign_id}")
        return True
    
    async def resume_workflow(self, campaign_id: str) -> bool:
        """Resume paused content generation workflow"""
        if campaign_id not in self.active_workflows:
            return False
        
        plan = self.active_workflows[campaign_id]
        plan.workflow_status = ContentTaskStatus.IN_PROGRESS
        
        # Resume execution of pending tasks
        execution_results = await self.execute_content_generation_plan(campaign_id)
        
        logger.info(f"Resumed content generation workflow for campaign: {campaign_id}")
        return True
    
    async def cancel_workflow(self, campaign_id: str) -> bool:
        """Cancel content generation workflow"""
        if campaign_id not in self.active_workflows:
            return False
        
        plan = self.active_workflows[campaign_id]
        plan.workflow_status = ContentTaskStatus.CANCELLED
        
        # Cancel all pending tasks
        for task in plan.content_tasks:
            if task.status == ContentTaskStatus.PENDING or task.status == ContentTaskStatus.IN_PROGRESS:
                task.status = ContentTaskStatus.CANCELLED
                task.updated_at = datetime.now()
        
        await self._save_generation_plan(plan)
        del self.active_workflows[campaign_id]
        
        logger.info(f"Cancelled content generation workflow for campaign: {campaign_id}")
        return True

# Create global workflow instance
content_generation_workflow = ContentGenerationWorkflow()