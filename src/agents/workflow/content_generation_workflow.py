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
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END, CompiledStateGraph

from src.agents.core.base_agent import BaseAgent, AgentResult, AgentExecutionContext
from src.agents.core.agent_factory import create_agent, AgentType
from src.agents.specialized.ai_content_generator_langgraph import (
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
    
    def store_agent_performance(self, agent_type: str, task_type: str, campaign_id: str, 
                               task_id: str, quality_score: float, duration_ms: int,
                               success: bool = True, error_message: str = None,
                               input_tokens: int = None, output_tokens: int = None):
        """Store agent performance metrics directly in database"""
        try:
            conn = db_config.get_db_connection()
            cur = conn.cursor()
            
            # Insert into agent_performance table
            cur.execute("""
                INSERT INTO agent_performance 
                (agent_type, task_type, execution_time_ms, success_rate, quality_score, 
                 input_tokens, output_tokens, campaign_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                agent_type, task_type, duration_ms, 1.0 if success else 0.0, 
                quality_score, input_tokens, output_tokens, campaign_id
            ))
            
            conn.commit()
            logger.info(f"📊 Stored performance metrics: {agent_type} - Quality: {quality_score:.1f}/1.0")
            
        except Exception as e:
            logger.error(f"Failed to store agent performance metrics: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
        
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
                    
                    # Generate content with performance tracking
                    start_time = time.time()
                    generated_content = await self.ai_content_generator.generate_content(generation_request)
                    generation_duration = int((time.time() - start_time) * 1000)
                    
                    # Store content generator performance
                    self.store_agent_performance(
                        agent_type="ai_content_generator",
                        task_type=f"content_generation_{task.content_type.value}",
                        campaign_id=task.campaign_id,
                        task_id=task.task_id,
                        quality_score=generated_content.quality_score,  # Uses built-in quality score (0.0-1.0)
                        duration_ms=generation_duration,
                        success=True,
                        output_tokens=generated_content.word_count // 4  # Rough token estimation
                    )
                    
                    # Run Quality Review Agent for enhanced scoring
                    if generated_content.quality_score > 0:  # Only if content was successfully generated
                        try:
                            from src.agents.specialized.quality_review_agent import QualityReviewAgent
                            start_time = time.time()
                            quality_agent = QualityReviewAgent()
                            
                            # Prepare content data for quality review
                            content_data = {
                                'content': generated_content.content,
                                'title': task.title,
                                'content_type': task.content_type.value,
                                'word_count': generated_content.word_count,
                                'content_id': generated_content.content_id
                            }
                            
                            # Execute quality review
                            quality_result = await quality_agent.execute_safe(content_data)
                            quality_duration = int((time.time() - start_time) * 1000)
                            
                            # Update content with real quality score from Quality Review Agent
                            if quality_result.success and quality_result.automated_score is not None:
                                generated_content.quality_score = quality_result.automated_score  # 0.0-1.0 range
                                
                                # Store Quality Review Agent performance
                                self.store_agent_performance(
                                    agent_type="quality_review_agent",
                                    task_type="quality_analysis",
                                    campaign_id=task.campaign_id,
                                    task_id=task.task_id,
                                    quality_score=quality_result.automated_score,
                                    duration_ms=quality_duration,
                                    success=True
                                )
                                
                                logger.info(f"✅ Quality Review completed: {quality_result.automated_score:.2f}/1.0 (auto-approved: {quality_result.auto_approved})")
                                
                                # Run Brand Review Agent if quality review passed
                                try:
                                    from src.agents.specialized.brand_review_agent import BrandReviewAgent
                                    brand_start_time = time.time()
                                    brand_agent = BrandReviewAgent()
                                    
                                    # Execute brand review using same content data
                                    brand_result = await brand_agent.execute_safe(content_data)
                                    brand_duration = int((time.time() - brand_start_time) * 1000)
                                    
                                    if brand_result.success and brand_result.automated_score is not None:
                                        # Store Brand Review Agent performance
                                        self.store_agent_performance(
                                            agent_type="brand_review_agent",
                                            task_type="brand_analysis",
                                            campaign_id=task.campaign_id,
                                            task_id=task.task_id,
                                            quality_score=brand_result.automated_score,
                                            duration_ms=brand_duration,
                                            success=True
                                        )
                                        
                                        logger.info(f"✅ Brand Review completed: {brand_result.automated_score:.2f}/1.0 (auto-approved: {brand_result.auto_approved})")
                                        
                                        # Phase 4.3: Add SEO Analysis for content-based optimization scoring
                                        seo_approved = True  # Default to approved if SEO analysis fails
                                        try:
                                            from src.agents.specialized.seo_agent_langgraph import SEOAgentLangGraph
                                            seo_start_time = time.time()
                                            seo_agent = SEOAgentLangGraph()
                                            
                                            # Prepare SEO-specific content data with keywords
                                            seo_content_data = {
                                                'content': generated_content.content,
                                                'blog_title': task.title,
                                                'keywords': task.seo_keywords if task.seo_keywords else [],
                                                'content_type': task.content_type.value,
                                                'word_count': generated_content.word_count
                                            }
                                            
                                            # Execute SEO analysis
                                            seo_result = await seo_agent.execute(seo_content_data)
                                            seo_duration = int((time.time() - seo_start_time) * 1000)
                                            
                                            if seo_result.success and hasattr(seo_result, 'score') and seo_result.score is not None:
                                                # Store SEO Agent performance
                                                self.store_agent_performance(
                                                    agent_type="seo_agent",
                                                    task_type="seo_analysis",
                                                    campaign_id=task.campaign_id,
                                                    task_id=task.task_id,
                                                    quality_score=seo_result.score,
                                                    duration_ms=seo_duration,
                                                    success=True
                                                )
                                                
                                                # SEO approval threshold (0.7 for SEO optimization)
                                                seo_approved = seo_result.score >= 0.7
                                                logger.info(f"✅ SEO Analysis completed: {seo_result.score:.2f}/1.0 (auto-approved: {seo_approved})")
                                            else:
                                                logger.warning(f"SEO Analysis returned invalid score for task {task.task_id}")
                                                
                                        except Exception as seo_error:
                                            logger.warning(f"SEO Analysis error for task {task.task_id}: {seo_error}")
                                            # Continue with default approval
                                        
                                        # Phase 4.3: Add Content Agent readability analysis
                                        content_approved = True  # Default to approved if content analysis fails
                                        try:
                                            readability_start_time = time.time()
                                            
                                            # Calculate readability metrics using textstat if available
                                            readability_score = 0.8  # Default score
                                            try:
                                                import textstat
                                                
                                                # Calculate multiple readability metrics
                                                flesch_reading_ease = textstat.flesch_reading_ease(generated_content.content)
                                                flesch_kincaid_grade = textstat.flesch_kincaid_grade(generated_content.content)
                                                avg_sentence_length = textstat.avg_sentence_length(generated_content.content)
                                                
                                                # Normalize scores to 0-1 range
                                                # Flesch Reading Ease: 0-100 (higher is better, 60-70 is good)
                                                flesch_normalized = min(1.0, max(0.0, flesch_reading_ease / 100))
                                                
                                                # Flesch-Kincaid Grade: lower is better (target: 8-12 grade level)
                                                grade_normalized = max(0.0, min(1.0, (20 - flesch_kincaid_grade) / 20))
                                                
                                                # Average sentence length: target 15-20 words
                                                sentence_normalized = max(0.0, min(1.0, 1.0 - abs(avg_sentence_length - 17.5) / 17.5))
                                                
                                                # Combined readability score (weighted average)
                                                readability_score = (flesch_normalized * 0.4 + grade_normalized * 0.4 + sentence_normalized * 0.2)
                                                
                                                logger.info(f"📊 Readability metrics - Flesch: {flesch_reading_ease:.1f}, Grade: {flesch_kincaid_grade:.1f}, Sentence length: {avg_sentence_length:.1f}")
                                                
                                            except ImportError:
                                                logger.warning("textstat not available, using estimated readability score")
                                                # Fallback: estimate based on word count and content length
                                                sentences = generated_content.content.count('.') + generated_content.content.count('!') + generated_content.content.count('?')
                                                if sentences > 0:
                                                    avg_words_per_sentence = generated_content.word_count / sentences
                                                    # Target 15-20 words per sentence for good readability
                                                    readability_score = max(0.0, min(1.0, 1.0 - abs(avg_words_per_sentence - 17.5) / 17.5))
                                            
                                            readability_duration = int((time.time() - readability_start_time) * 1000)
                                            
                                            # Store Content Agent readability performance
                                            self.store_agent_performance(
                                                agent_type="content_agent_readability",
                                                task_type="readability_analysis", 
                                                campaign_id=task.campaign_id,
                                                task_id=task.task_id,
                                                quality_score=readability_score,
                                                duration_ms=readability_duration,
                                                success=True
                                            )
                                            
                                            # Content readability approval threshold (0.65 for readability)
                                            content_approved = readability_score >= 0.65
                                            logger.info(f"✅ Content Readability completed: {readability_score:.2f}/1.0 (auto-approved: {content_approved})")
                                            
                                        except Exception as content_error:
                                            logger.warning(f"Content Readability Analysis error for task {task.task_id}: {content_error}")
                                            # Continue with default approval
                                        
                                        # Phase 4.3: Add GEO Analysis for market-specific optimization
                                        geo_approved = True  # Default to approved if GEO analysis fails
                                        try:
                                            from src.agents.specialized.geo_analysis_agent_langgraph import GEOAnalysisAgentLangGraph
                                            geo_start_time = time.time()
                                            geo_agent = GEOAnalysisAgentLangGraph()
                                            
                                            # Prepare GEO-specific content data
                                            geo_content_data = {
                                                'content': generated_content.content,
                                                'title': task.title,
                                                'target_audience': task.target_audience if task.target_audience else "general",
                                                'content_type': task.content_type.value,
                                                'word_count': generated_content.word_count
                                            }
                                            
                                            # Execute GEO analysis
                                            geo_result = await geo_agent.execute(geo_content_data)
                                            geo_duration = int((time.time() - geo_start_time) * 1000)
                                            
                                            if geo_result.success and hasattr(geo_result, 'score') and geo_result.score is not None:
                                                # Store GEO Agent performance
                                                self.store_agent_performance(
                                                    agent_type="geo_analysis_agent",
                                                    task_type="geo_analysis",
                                                    campaign_id=task.campaign_id,
                                                    task_id=task.task_id,
                                                    quality_score=geo_result.score,
                                                    duration_ms=geo_duration,
                                                    success=True
                                                )
                                                
                                                # GEO approval threshold (0.75 for generative engine optimization)
                                                geo_approved = geo_result.score >= 0.75
                                                logger.info(f"✅ GEO Analysis completed: {geo_result.score:.2f}/1.0 (auto-approved: {geo_approved})")
                                            else:
                                                logger.warning(f"GEO Analysis returned invalid score for task {task.task_id}")
                                                
                                        except Exception as geo_error:
                                            logger.warning(f"GEO Analysis error for task {task.task_id}: {geo_error}")
                                            # Continue with default approval
                                        
                                        # Combined approval logic: quality, brand, SEO, content readability, and GEO all need to pass
                                        if quality_result.auto_approved and brand_result.auto_approved and seo_approved and content_approved and geo_approved:
                                            task.status = ContentTaskStatus.COMPLETED
                                        else:
                                            task.status = ContentTaskStatus.REQUIRES_REVIEW
                                    else:
                                        logger.warning(f"Brand Review Agent failed for task {task.task_id}")
                                        task.status = ContentTaskStatus.REQUIRES_REVIEW
                                        
                                except Exception as brand_error:
                                    logger.error(f"Brand Review Agent error for task {task.task_id}: {brand_error}")
                                    # If brand review fails, fall back to quality review decision
                                    if quality_result.auto_approved:
                                        task.status = ContentTaskStatus.COMPLETED
                                    else:
                                        task.status = ContentTaskStatus.REQUIRES_REVIEW
                            else:
                                logger.warning(f"Quality Review Agent failed for task {task.task_id}")
                                
                        except Exception as quality_error:
                            logger.error(f"Quality Review Agent error for task {task.task_id}: {quality_error}")
                            # Don't fail the entire task if quality review fails
                            task.status = ContentTaskStatus.COMPLETED
                    else:
                        task.status = ContentTaskStatus.COMPLETED
                        
                    # Update task with results  
                    task.generated_content = generated_content
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