"""
Task to Content Deliverable Migration Service
Transforms existing campaign task results into content deliverables
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..config.database import secure_db
from ..agents.core.content_deliverable_service import content_service, ContentDeliverable

logger = logging.getLogger(__name__)

@dataclass
class MigrationResult:
    """Result of migration operation"""
    campaign_id: str
    tasks_processed: int
    deliverables_created: int
    errors: List[str]
    success_rate: float
    created_deliverables: List[str]

class TaskToContentMigrationService:
    """Service for migrating campaign tasks to content deliverables"""
    
    def __init__(self):
        self.content_type_mapping = {
            'content_creation': 'blog_post',
            'blog_creation': 'blog_post',
            'blog_post_creation': 'blog_post',
            'social_media_adaptation': 'social_media_post',
            'social_media_creation': 'social_media_post',
            'linkedin_post': 'social_media_post',
            'twitter_post': 'social_media_post',
            'email_formatting': 'email_campaign',
            'email_creation': 'email_campaign',
            'newsletter_creation': 'newsletter',
            'image_generation': 'infographic_concept',
            'content_editing': 'editorial',
            'content_repurposing': 'repurposed_content',
            'seo_optimization': 'seo_content',
            'press_release': 'press_release',
            'case_study': 'case_study',
            'whitepaper': 'whitepaper',
            'video_script': 'video_script',
            'podcast_script': 'podcast_script',
            'ad_copy': 'ad_copy',
            'landing_page': 'landing_page',
            'product_description': 'product_description'
        }
    
    def migrate_campaign_tasks(self, campaign_id: str, dry_run: bool = False) -> MigrationResult:
        """
        Migrate all tasks for a specific campaign to content deliverables
        
        Args:
            campaign_id: Campaign to migrate
            dry_run: If True, don't actually create deliverables, just report what would be done
        """
        logger.info(f"🚀 Starting task migration for campaign {campaign_id} (dry_run={dry_run})")
        
        try:
            # Get all tasks with results for the campaign
            tasks = self._get_campaign_tasks_with_results(campaign_id)
            
            if not tasks:
                logger.info(f"📋 No tasks with results found for campaign {campaign_id}")
                return MigrationResult(
                    campaign_id=campaign_id,
                    tasks_processed=0,
                    deliverables_created=0,
                    errors=[],
                    success_rate=100.0,
                    created_deliverables=[]
                )
            
            logger.info(f"📋 Found {len(tasks)} tasks with results to migrate")
            
            errors = []
            created_deliverables = []
            
            for task in tasks:
                try:
                    if dry_run:
                        # Just validate the task can be migrated
                        self._validate_task_for_migration(task)
                        logger.info(f"✅ Task {task['id']} would be migrated successfully")
                    else:
                        # Actually migrate the task
                        deliverable = self._migrate_single_task(task, campaign_id)
                        if deliverable:
                            created_deliverables.append(deliverable.id)
                            logger.info(f"✅ Migrated task {task['id']} to deliverable {deliverable.id}")
                        else:
                            errors.append(f"Failed to create deliverable for task {task['id']}")
                            
                except Exception as e:
                    error_msg = f"Error migrating task {task.get('id', 'unknown')}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            deliverables_created = len(created_deliverables)
            success_rate = ((len(tasks) - len(errors)) / len(tasks) * 100) if tasks else 100.0
            
            result = MigrationResult(
                campaign_id=campaign_id,
                tasks_processed=len(tasks),
                deliverables_created=deliverables_created,
                errors=errors,
                success_rate=success_rate,
                created_deliverables=created_deliverables
            )
            
            logger.info(f"✅ Migration completed: {deliverables_created}/{len(tasks)} tasks migrated ({success_rate:.1f}% success)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Migration failed for campaign {campaign_id}: {e}")
            return MigrationResult(
                campaign_id=campaign_id,
                tasks_processed=0,
                deliverables_created=0,
                errors=[f"Migration failed: {str(e)}"],
                success_rate=0.0,
                created_deliverables=[]
            )
    
    def migrate_all_campaigns(self, dry_run: bool = False) -> Dict[str, MigrationResult]:
        """
        Migrate tasks to content deliverables for all campaigns that have tasks with results
        
        Args:
            dry_run: If True, don't actually create deliverables, just report what would be done
        """
        logger.info(f"🚀 Starting migration for all campaigns (dry_run={dry_run})")
        
        try:
            # Get all campaigns that have tasks with results
            campaigns_with_tasks = self._get_campaigns_with_tasks()
            
            if not campaigns_with_tasks:
                logger.info("📋 No campaigns with tasks found")
                return {}
            
            logger.info(f"📋 Found {len(campaigns_with_tasks)} campaigns with tasks to migrate")
            
            results = {}
            
            for campaign_id in campaigns_with_tasks:
                logger.info(f"🔄 Migrating campaign {campaign_id}")
                
                try:
                    result = self.migrate_campaign_tasks(campaign_id, dry_run)
                    results[campaign_id] = result
                    
                except Exception as e:
                    logger.error(f"❌ Failed to migrate campaign {campaign_id}: {e}")
                    results[campaign_id] = MigrationResult(
                        campaign_id=campaign_id,
                        tasks_processed=0,
                        deliverables_created=0,
                        errors=[f"Campaign migration failed: {str(e)}"],
                        success_rate=0.0,
                        created_deliverables=[]
                    )
            
            # Log summary
            total_campaigns = len(results)
            successful_campaigns = len([r for r in results.values() if r.success_rate > 0])
            total_deliverables = sum(r.deliverables_created for r in results.values())
            
            logger.info(f"✅ Migration completed: {successful_campaigns}/{total_campaigns} campaigns, {total_deliverables} total deliverables created")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Global migration failed: {e}")
            return {}
    
    def _get_campaign_tasks_with_results(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Get all tasks with results for a specific campaign"""
        query = """
            SELECT id, task_type, result, created_at, updated_at, status, metadata
            FROM campaign_tasks 
            WHERE campaign_id = %s 
              AND result IS NOT NULL 
              AND result != ''
              AND content_deliverable_id IS NULL
            ORDER BY created_at ASC
        """
        
        try:
            results = secure_db.execute_query(query, (campaign_id,), fetch="all")
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"❌ Failed to get tasks for campaign {campaign_id}: {e}")
            return []
    
    def _get_campaigns_with_tasks(self) -> List[str]:
        """Get all campaign IDs that have tasks with results"""
        query = """
            SELECT DISTINCT campaign_id 
            FROM campaign_tasks 
            WHERE result IS NOT NULL 
              AND result != ''
              AND content_deliverable_id IS NULL
        """
        
        try:
            results = secure_db.execute_query(query, fetch="all")
            return [row['campaign_id'] for row in results] if results else []
        except Exception as e:
            logger.error(f"❌ Failed to get campaigns with tasks: {e}")
            return []
    
    def _validate_task_for_migration(self, task: Dict[str, Any]) -> bool:
        """Validate that a task can be migrated"""
        required_fields = ['id', 'task_type', 'result']
        
        for field in required_fields:
            if not task.get(field):
                raise ValueError(f"Missing required field: {field}")
        
        if len(task['result'].strip()) < 10:
            raise ValueError("Task result is too short to be meaningful content")
        
        return True
    
    def _migrate_single_task(self, task: Dict[str, Any], campaign_id: str) -> Optional[ContentDeliverable]:
        """Migrate a single task to a content deliverable"""
        try:
            # Validate task
            self._validate_task_for_migration(task)
            
            # Extract content information
            task_type = task.get('task_type', 'content_creation')
            content = task.get('result', '')
            task_id = task.get('id')
            
            # Map task type to content type
            content_type = self.content_type_mapping.get(task_type.lower(), 'blog_post')
            
            # Generate title from content or task type
            title = self._generate_title_from_content(content, task_type)
            
            # Generate summary
            summary = self._generate_summary_from_content(content, task_type)
            
            # Extract metadata
            metadata = task.get('metadata', {}) or {}
            metadata['migrated_from_task'] = task_id
            metadata['original_task_type'] = task_type
            metadata['migration_date'] = datetime.now().isoformat()
            
            # Create content deliverable
            deliverable = content_service.create_content(
                title=title,
                content=content,
                campaign_id=campaign_id,
                content_type=content_type,
                summary=summary,
                metadata=metadata
            )
            
            if deliverable:
                # Link the task to the content deliverable
                content_service.link_task_to_content(task_id, deliverable.id)
                return deliverable
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate task {task.get('id', 'unknown')}: {e}")
            return None
    
    def _generate_title_from_content(self, content: str, task_type: str) -> str:
        """Generate a title from content"""
        try:
            # Try to extract title from first line of content
            lines = content.strip().split('\n')
            first_line = lines[0].strip()
            
            # If first line looks like a title (starts with #, is short, etc.)
            if first_line.startswith('#'):
                return first_line.strip('# ').strip()
            elif len(first_line) < 100 and not first_line.endswith('.'):
                return first_line
            
            # Otherwise, generate based on task type
            content_type = self.content_type_mapping.get(task_type.lower(), 'blog_post')
            return f"{content_type.replace('_', ' ').title()} Content"
            
        except Exception:
            return f"Generated {task_type.replace('_', ' ').title()}"
    
    def _generate_summary_from_content(self, content: str, task_type: str) -> str:
        """Generate a summary from content"""
        try:
            # Take first 200 characters as summary
            summary = content.strip()[:200]
            if len(content) > 200:
                summary += "..."
            return summary
        except Exception:
            return f"Generated content from {task_type} task"

# Global service instance
migration_service = TaskToContentMigrationService()