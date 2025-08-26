#!/usr/bin/env python3
"""
Migration Strategy: Task-Centric to Content-First Workflow

This module provides utilities and strategies for migrating from the current
task-based system to the new content-first workflow approach.

Key Migration Components:
1. Task-to-Content mapping utilities
2. Existing workflow adaptation patterns
3. Backward compatibility layers
4. Progressive migration strategies
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .content_first_workflow import (
    ContentDeliverable, ContentDeliverableType, ContentStatus, NarrativePosition,
    NarrativeContext, ContentFirstWorkflowOrchestrator
)
from ..core.content_deliverable_service import content_deliverable_service
from .blog_workflow import BlogWriterState
from .content_generation_workflow import ContentTask, ContentTaskStatus

logger = logging.getLogger(__name__)

# =============================================================================
# MIGRATION STRATEGY CLASSES
# =============================================================================

@dataclass
class TaskToContentMapping:
    """Mapping strategy from tasks to content deliverables"""
    task_id: str
    task_type: str
    task_result: str
    mapped_content_type: ContentDeliverableType
    narrative_position: NarrativePosition
    consolidation_group: str  # Multiple tasks can map to one content piece
    content_priority: int     # Priority within the consolidation group

class MigrationStrategy:
    """
    Core migration strategy from task-centric to content-first workflows
    """
    
    def __init__(self):
        self.mapping_rules = self._initialize_mapping_rules()
        self.consolidation_strategies = self._initialize_consolidation_strategies()
        
    def _initialize_mapping_rules(self) -> Dict[str, TaskToContentMapping]:
        """Initialize task-to-content mapping rules"""
        return {
            # Blog creation tasks -> Blog post deliverable
            'content_creation': TaskToContentMapping(
                task_id='*',
                task_type='content_creation',
                task_result='*',
                mapped_content_type=ContentDeliverableType.BLOG_POST,
                narrative_position=NarrativePosition.FOUNDATION,
                consolidation_group='primary_content',
                content_priority=1
            ),
            
            # Social media tasks -> Social posts
            'blog_to_linkedin': TaskToContentMapping(
                task_id='*',
                task_type='blog_to_linkedin',
                task_result='*',
                mapped_content_type=ContentDeliverableType.LINKEDIN_POST,
                narrative_position=NarrativePosition.EXPLORATION,
                consolidation_group='linkedin_content',
                content_priority=2
            ),
            
            'blog_to_twitter': TaskToContentMapping(
                task_id='*',
                task_type='blog_to_twitter',
                task_result='*',
                mapped_content_type=ContentDeliverableType.TWITTER_THREAD,
                narrative_position=NarrativePosition.REINFORCEMENT,
                consolidation_group='twitter_content',
                content_priority=3
            ),
            
            # Email tasks -> Email sequence
            'email_formatting': TaskToContentMapping(
                task_id='*',
                task_type='email_formatting',
                task_result='*',
                mapped_content_type=ContentDeliverableType.EMAIL_SEQUENCE,
                narrative_position=NarrativePosition.APPLICATION,
                consolidation_group='email_content',
                content_priority=2
            ),
            
            # Repurposing tasks -> Multiple content types
            'content_repurposing': TaskToContentMapping(
                task_id='*',
                task_type='content_repurposing',
                task_result='*',
                mapped_content_type=ContentDeliverableType.CASE_STUDY,  # Default
                narrative_position=NarrativePosition.TRANSFORMATION,
                consolidation_group='repurposed_content',
                content_priority=3
            )
        }
    
    def _initialize_consolidation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategies for consolidating multiple tasks into single deliverables"""
        return {
            'primary_content': {
                'target_deliverable': ContentDeliverableType.BLOG_POST,
                'consolidate_tasks': ['content_creation', 'content_editing', 'seo_optimization'],
                'narrative_position': NarrativePosition.FOUNDATION,
                'word_count_target': 1500
            },
            
            'linkedin_content': {
                'target_deliverable': ContentDeliverableType.LINKEDIN_POST,
                'consolidate_tasks': ['blog_to_linkedin', 'social_media_adaptation'],
                'narrative_position': NarrativePosition.EXPLORATION,
                'word_count_target': 800
            },
            
            'email_content': {
                'target_deliverable': ContentDeliverableType.EMAIL_SEQUENCE,
                'consolidate_tasks': ['email_formatting', 'content_repurposing'],
                'narrative_position': NarrativePosition.APPLICATION,
                'word_count_target': 400
            },
            
            'social_content': {
                'target_deliverable': ContentDeliverableType.TWITTER_THREAD,
                'consolidate_tasks': ['blog_to_twitter', 'social_media_adaptation'],
                'narrative_position': NarrativePosition.REINFORCEMENT,
                'word_count_target': 280
            }
        }

# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

class WorkflowMigrator:
    """
    Utility class to migrate existing task-based workflows to content-first approach
    """
    
    def __init__(self):
        self.strategy = MigrationStrategy()
        self.content_orchestrator = ContentFirstWorkflowOrchestrator()
        
    async def migrate_campaign_tasks_to_deliverables(self, campaign_id: str, 
                                                   existing_tasks: List[Dict[str, Any]]) -> List[ContentDeliverable]:
        """
        Migrate existing campaign tasks to content deliverables
        
        Args:
            campaign_id: ID of the campaign to migrate
            existing_tasks: List of existing task data
            
        Returns:
            List of content deliverables created from tasks
        """
        try:
            logger.info(f"Migrating {len(existing_tasks)} tasks to content deliverables for campaign: {campaign_id}")
            
            # Group tasks by consolidation strategy
            task_groups = self._group_tasks_for_consolidation(existing_tasks)
            
            # Create content deliverables from task groups
            deliverables = []
            for group_name, tasks in task_groups.items():
                deliverable = await self._create_deliverable_from_task_group(
                    campaign_id, 
                    group_name, 
                    tasks
                )
                if deliverable:
                    deliverables.append(deliverable)
            
            # Create narrative context from existing tasks
            narrative_context = await self._generate_narrative_from_tasks(campaign_id, existing_tasks)
            
            # Save narrative context
            if narrative_context:
                await content_deliverable_service.save_narrative_context(campaign_id, narrative_context)
            
            # Save deliverables
            for deliverable in deliverables:
                await content_deliverable_service.save_content_deliverable(deliverable)
            
            logger.info(f"Migration completed: {len(deliverables)} deliverables created")
            return deliverables
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return []
    
    def _group_tasks_for_consolidation(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tasks by their consolidation strategy"""
        groups = {}
        
        for task in tasks:
            task_type = task.get('task_type', 'unknown')
            
            # Find consolidation group for this task type
            consolidation_group = None
            for strategy_name, strategy in self.strategy.consolidation_strategies.items():
                if task_type in strategy['consolidate_tasks']:
                    consolidation_group = strategy_name
                    break
            
            # Default group if no specific strategy found
            if not consolidation_group:
                if 'blog' in task_type or 'content' in task_type:
                    consolidation_group = 'primary_content'
                elif 'linkedin' in task_type:
                    consolidation_group = 'linkedin_content'
                elif 'twitter' in task_type:
                    consolidation_group = 'social_content'
                elif 'email' in task_type:
                    consolidation_group = 'email_content'
                else:
                    consolidation_group = 'miscellaneous'
            
            if consolidation_group not in groups:
                groups[consolidation_group] = []
            groups[consolidation_group].append(task)
        
        return groups
    
    async def _create_deliverable_from_task_group(self, campaign_id: str, 
                                                group_name: str, 
                                                tasks: List[Dict[str, Any]]) -> Optional[ContentDeliverable]:
        """Create a content deliverable from a group of related tasks"""
        try:
            strategy = self.strategy.consolidation_strategies.get(group_name)
            if not strategy:
                logger.warning(f"No consolidation strategy for group: {group_name}")
                return None
            
            # Extract content from tasks
            combined_content = self._extract_and_combine_content(tasks)
            
            # Generate title from task results
            title = self._generate_title_from_tasks(tasks, strategy['target_deliverable'])
            
            # Create deliverable
            deliverable = ContentDeliverable(
                content_id=str(uuid.uuid4()),
                deliverable_type=strategy['target_deliverable'],
                title=title,
                content_body=combined_content['body'],
                summary=combined_content['summary'],
                word_count=len(combined_content['body'].split()),
                status=ContentStatus.PUBLISHED,  # Tasks are already completed
                narrative_position=strategy['narrative_position'],
                narrative_thread_id=f"{campaign_id}_main_thread",
                key_message=combined_content['key_message'],
                supporting_points=combined_content['supporting_points'],
                target_audience='B2B professionals',
                tone='Professional',
                channel=self._determine_channel_from_type(strategy['target_deliverable']),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                completed_at=datetime.now()
            )
            
            return deliverable
            
        except Exception as e:
            logger.error(f"Failed to create deliverable from task group: {str(e)}")
            return None
    
    def _extract_and_combine_content(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and combine content from multiple tasks"""
        combined_body = []
        key_messages = []
        supporting_points = []
        
        for task in tasks:
            result = task.get('result', '')
            if result and isinstance(result, str):
                # Try to parse as JSON first
                try:
                    result_data = json.loads(result)
                    if isinstance(result_data, dict):
                        if 'content' in result_data:
                            combined_body.append(result_data['content'])
                        if 'key_message' in result_data:
                            key_messages.append(result_data['key_message'])
                        if 'supporting_points' in result_data:
                            supporting_points.extend(result_data['supporting_points'])
                    else:
                        combined_body.append(str(result_data))
                except json.JSONDecodeError:
                    # Treat as plain text
                    combined_body.append(result)
        
        # Create summary from first 200 characters of combined content
        full_content = '\n\n'.join(combined_body)
        summary = full_content[:200] + '...' if len(full_content) > 200 else full_content
        
        return {
            'body': full_content,
            'summary': summary,
            'key_message': key_messages[0] if key_messages else 'Content migrated from tasks',
            'supporting_points': list(set(supporting_points))  # Remove duplicates
        }
    
    def _generate_title_from_tasks(self, tasks: List[Dict[str, Any]], 
                                 content_type: ContentDeliverableType) -> str:
        """Generate an appropriate title from task data"""
        # Extract titles or subjects from task results
        titles = []
        for task in tasks:
            result = task.get('result', '')
            if result:
                try:
                    result_data = json.loads(result)
                    if isinstance(result_data, dict) and 'title' in result_data:
                        titles.append(result_data['title'])
                except json.JSONDecodeError:
                    pass
        
        # Use first title found, or generate generic one
        if titles:
            return titles[0]
        else:
            type_titles = {
                ContentDeliverableType.BLOG_POST: "Professional Blog Post",
                ContentDeliverableType.LINKEDIN_POST: "LinkedIn Professional Update",
                ContentDeliverableType.TWITTER_THREAD: "Twitter Thread Series",
                ContentDeliverableType.EMAIL_SEQUENCE: "Email Marketing Sequence",
                ContentDeliverableType.CASE_STUDY: "Customer Success Story"
            }
            return type_titles.get(content_type, "Content Piece")
    
    def _determine_channel_from_type(self, content_type: ContentDeliverableType) -> str:
        """Determine distribution channel from content type"""
        channel_mapping = {
            ContentDeliverableType.BLOG_POST: "blog",
            ContentDeliverableType.LINKEDIN_POST: "linkedin",
            ContentDeliverableType.TWITTER_THREAD: "twitter",
            ContentDeliverableType.EMAIL_SEQUENCE: "email",
            ContentDeliverableType.CASE_STUDY: "website"
        }
        return channel_mapping.get(content_type, "general")
    
    async def _generate_narrative_from_tasks(self, campaign_id: str, 
                                           tasks: List[Dict[str, Any]]) -> Optional[NarrativeContext]:
        """Generate narrative context from existing task data"""
        try:
            # Extract themes and messages from task results
            themes = set()
            messages = set()
            examples = []
            
            for task in tasks:
                result = task.get('result', '')
                if result:
                    # Extract keywords and themes (simple approach)
                    words = result.lower().split()
                    
                    # Look for business/industry keywords
                    business_keywords = ['business', 'enterprise', 'solution', 'innovation', 
                                       'growth', 'efficiency', 'transformation', 'success']
                    for keyword in business_keywords:
                        if keyword in words:
                            themes.add(keyword.title())
                    
                    # Extract potential key messages (sentences with impact words)
                    impact_words = ['achieve', 'deliver', 'improve', 'increase', 'optimize', 'transform']
                    sentences = result.split('.')
                    for sentence in sentences[:3]:  # Only check first 3 sentences
                        if any(word in sentence.lower() for word in impact_words):
                            messages.add(sentence.strip())
            
            # Create narrative context
            narrative = NarrativeContext(
                central_theme="Business transformation through innovative solutions",
                supporting_themes=list(themes)[:5],  # Limit to 5 themes
                key_messages=list(messages)[:3],     # Limit to 3 messages
                target_transformation="Move prospects from awareness to consideration",
                brand_voice_guidelines="Professional, results-focused, authoritative",
                content_journey_map={},
                cross_references={},
                thematic_connections={},
                terminology_glossary={
                    "Solution": "Comprehensive business solution addressing specific needs",
                    "Transformation": "Positive change in business operations and outcomes"
                },
                recurring_concepts=["innovation", "efficiency", "growth", "success"],
                brand_examples=examples[:3]
            )
            
            return narrative
            
        except Exception as e:
            logger.error(f"Failed to generate narrative from tasks: {str(e)}")
            return None

# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

class BackwardCompatibilityAdapter:
    """
    Provides backward compatibility for existing task-based API calls
    while transitioning to content-first approach
    """
    
    def __init__(self):
        self.migrator = WorkflowMigrator()
        self.content_orchestrator = ContentFirstWorkflowOrchestrator()
    
    async def adapt_blog_workflow_to_content_first(self, blog_state: BlogWriterState) -> Dict[str, Any]:
        """
        Adapt existing blog workflow state to content-first workflow
        
        This allows existing blog workflow calls to work with the new system
        """
        try:
            # Convert blog state to campaign input
            campaign_input = {
                'campaign_id': f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'campaign_brief': f"Create blog post: {blog_state['blog_title']}",
                'objectives': ['content_creation', 'thought_leadership'],
                'target_audience': 'B2B professionals',
                'brand_context': blog_state['company_context'],
                'blog_content': {
                    'title': blog_state['blog_title'],
                    'outline': blog_state.get('outline', []),
                    'research': blog_state.get('research', {}),
                    'draft': blog_state.get('draft', ''),
                    'final_post': blog_state.get('final_post', '')
                }
            }
            
            # Execute content-first workflow
            result = await self.content_orchestrator.execute_content_workflow(campaign_input)
            
            # Convert result back to blog workflow format for compatibility
            return {
                'blog_title': blog_state['blog_title'],
                'final_post': result.get('content_pieces', [{}])[0].get('content_body', ''),
                'status': 'completed',
                'content_first_campaign_id': result.get('campaign_id'),
                'deliverables_created': result.get('total_deliverables', 0)
            }
            
        except Exception as e:
            logger.error(f"Blog workflow adaptation failed: {str(e)}")
            return {'error': str(e)}
    
    async def adapt_task_results_to_deliverables(self, campaign_id: str, 
                                               task_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adapt task results to deliverable format for API compatibility
        """
        try:
            # Migrate tasks to deliverables
            deliverables = await self.migrator.migrate_campaign_tasks_to_deliverables(
                campaign_id, 
                task_results
            )
            
            # Convert to API-compatible format
            adapted_results = []
            for deliverable in deliverables:
                adapted_results.append({
                    'id': deliverable.content_id,
                    'type': deliverable.deliverable_type.value,
                    'title': deliverable.title,
                    'content': deliverable.content_body,
                    'status': deliverable.status.value,
                    'word_count': deliverable.word_count,
                    'quality_score': deliverable.quality_score,
                    'narrative_position': deliverable.narrative_position.value,
                    'created_at': deliverable.created_at.isoformat() if deliverable.created_at else None
                })
            
            return adapted_results
            
        except Exception as e:
            logger.error(f"Task result adaptation failed: {str(e)}")
            return []

# =============================================================================
# PROGRESSIVE MIGRATION UTILITIES
# =============================================================================

class ProgressiveMigrationManager:
    """
    Manages progressive migration from task-centric to content-first workflows
    """
    
    def __init__(self):
        self.adapter = BackwardCompatibilityAdapter()
        self.feature_flags = {
            'enable_content_first': True,
            'maintain_task_compatibility': True,
            'migrate_existing_campaigns': True
        }
    
    async def should_use_content_first_workflow(self, campaign_data: Dict[str, Any]) -> bool:
        """
        Determine whether to use content-first workflow for a campaign
        """
        if not self.feature_flags['enable_content_first']:
            return False
        
        # Use content-first for new campaigns with narrative requirements
        narrative_indicators = [
            'narrative', 'story', 'cohesive', 'journey', 
            'content portfolio', 'content series'
        ]
        
        campaign_brief = campaign_data.get('campaign_brief', '').lower()
        return any(indicator in campaign_brief for indicator in narrative_indicators)
    
    async def execute_hybrid_workflow(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow using appropriate approach based on campaign requirements
        """
        try:
            use_content_first = await self.should_use_content_first_workflow(campaign_data)
            
            if use_content_first:
                logger.info("Using content-first workflow for campaign")
                orchestrator = ContentFirstWorkflowOrchestrator()
                return await orchestrator.execute_content_workflow(campaign_data)
            else:
                logger.info("Using traditional task-based workflow with adaptation")
                # Use traditional workflow but adapt results to content-first format
                return await self._execute_traditional_with_adaptation(campaign_data)
                
        except Exception as e:
            logger.error(f"Hybrid workflow execution failed: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_traditional_with_adaptation(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute traditional workflow but present results as content deliverables
        """
        # This would call the existing workflow systems
        # For demonstration, create a placeholder response
        return {
            'campaign_id': campaign_data.get('campaign_id', 'traditional_campaign'),
            'workflow_approach': 'traditional_adapted',
            'message': 'Executed traditional workflow with content-first result formatting',
            'deliverables_created': 1,
            'adaptation_applied': True
        }

# =============================================================================
# EXAMPLE MIGRATION USAGE
# =============================================================================

async def example_migration_workflow():
    """
    Example of how to use migration utilities
    """
    logger.info("Starting example migration workflow")
    
    # Example existing task data
    existing_tasks = [
        {
            'id': 'task_1',
            'task_type': 'content_creation',
            'result': '{"title": "AI in Business", "content": "Artificial intelligence is transforming modern businesses..."}',
            'status': 'completed'
        },
        {
            'id': 'task_2', 
            'task_type': 'blog_to_linkedin',
            'result': '{"content": "Key insights on AI transformation for LinkedIn professionals..."}',
            'status': 'completed'
        },
        {
            'id': 'task_3',
            'task_type': 'email_formatting',
            'result': '{"subject": "AI Transformation Guide", "content": "Dear professionals, discover how AI can transform your business..."}',
            'status': 'completed'
        }
    ]
    
    # Migrate tasks to content deliverables
    migrator = WorkflowMigrator()
    deliverables = await migrator.migrate_campaign_tasks_to_deliverables(
        'example_campaign_123',
        existing_tasks
    )
    
    logger.info(f"Migration example completed: {len(deliverables)} deliverables created")
    
    # Example of hybrid workflow usage
    migration_manager = ProgressiveMigrationManager()
    
    campaign_data = {
        'campaign_id': 'hybrid_test_campaign',
        'campaign_brief': 'Create a cohesive content narrative showcasing our AI expertise',
        'objectives': ['thought_leadership', 'lead_generation']
    }
    
    result = await migration_manager.execute_hybrid_workflow(campaign_data)
    logger.info(f"Hybrid workflow result: {result}")

# Create global instances
workflow_migrator = WorkflowMigrator()
compatibility_adapter = BackwardCompatibilityAdapter()
migration_manager = ProgressiveMigrationManager()