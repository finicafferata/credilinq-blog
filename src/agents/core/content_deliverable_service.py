"""
Content Deliverable Service
Handles content-first operations using the existing database schema
"""

import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from src.config.database import secure_db

logger = logging.getLogger(__name__)

@dataclass
class ContentDeliverable:
    """Content deliverable model matching the existing database schema"""
    id: str
    title: str
    content: str
    summary: Optional[str] = None
    content_type: str = 'blog_post'  # blog_post, social_media, email, etc.
    format: str = 'markdown'
    status: str = 'draft'  # draft, review, approved, published
    parent_id: Optional[str] = None
    campaign_id: str = None
    narrative_order: Optional[int] = None
    key_messages: Optional[List[str]] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    platform: Optional[str] = None
    word_count: Optional[int] = None
    reading_time: Optional[int] = None
    seo_score: Optional[float] = None
    engagement_score: Optional[float] = None
    created_by: Optional[str] = 'ContentAgent'
    last_edited_by: Optional[str] = None
    version: int = 1
    is_published: bool = False
    published_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'content_type': self.content_type,
            'format': self.format,
            'status': self.status,
            'parent_id': self.parent_id,
            'campaign_id': self.campaign_id,
            'narrative_order': self.narrative_order,
            'key_messages': self.key_messages,
            'target_audience': self.target_audience,
            'tone': self.tone,
            'platform': self.platform,
            'word_count': self.word_count,
            'reading_time': self.reading_time,
            'seo_score': self.seo_score,
            'engagement_score': self.engagement_score,
            'created_by': self.created_by,
            'version': self.version,
            'is_published': self.is_published,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }

class ContentDeliverableService:
    """Service for managing content deliverables"""
    
    def __init__(self):
        self.table_name = 'content_deliverables'
    
    def create_content(
        self,
        title: str,
        content: str,
        campaign_id: str,
        content_type: str = 'blog_post',
        summary: Optional[str] = None,
        platform: Optional[str] = None,
        **kwargs
    ) -> ContentDeliverable:
        """Create a new content deliverable"""
        
        content_id = str(uuid.uuid4())
        word_count = len(content.split()) if content else 0
        reading_time = max(1, word_count // 200)  # Approximate reading time
        
        # Prepare data matching the existing schema
        content_data = {
            'id': content_id,
            'title': title,
            'content': content,
            'summary': summary or f"Generated content for {content_type}",
            'content_type': content_type,
            'format': 'markdown',
            'status': 'draft',
            'campaign_id': campaign_id,
            'narrative_order': None,
            'key_messages': [],
            'target_audience': None,
            'tone': None,
            'platform': platform,
            'word_count': word_count,
            'reading_time': reading_time,
            'seo_score': None,
            'engagement_score': None,
            'created_by': 'ContentAgent',
            'version': 1,
            'is_published': False,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'metadata': {}
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key in content_data:
                content_data[key] = value
        
        # Convert complex types for database storage
        if content_data['metadata']:
            content_data['metadata'] = json.dumps(content_data['metadata'])
        else:
            content_data['metadata'] = None
            
        # Insert into database
        insert_query = """
            INSERT INTO content_deliverables (
                id, title, content, summary, content_type, format, status,
                campaign_id, narrative_order, key_messages, target_audience, tone, platform, 
                word_count, reading_time, seo_score, engagement_score, created_by,
                version, is_published, created_at, updated_at, metadata
            ) VALUES (
                %(id)s, %(title)s, %(content)s, %(summary)s, %(content_type)s, %(format)s, %(status)s,
                %(campaign_id)s, %(narrative_order)s, %(key_messages)s, %(target_audience)s, %(tone)s, %(platform)s,
                %(word_count)s, %(reading_time)s, %(seo_score)s, %(engagement_score)s, %(created_by)s,
                %(version)s, %(is_published)s, %(created_at)s, %(updated_at)s, %(metadata)s
            )
        """
        
        try:
            with secure_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, content_data)
                    conn.commit()
            
            logger.info(f"✅ Created content deliverable: {title} ({content_type})")
            
            # Return the created content
            return self.get_content_by_id(content_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to create content deliverable: {e}")
            raise
    
    def get_content_by_id(self, content_id: str) -> Optional[ContentDeliverable]:
        """Get content deliverable by ID"""
        query = "SELECT * FROM content_deliverables WHERE id = %s"
        
        try:
            result = secure_db.execute_query(query, (content_id,), fetch="one")
            if result:
                return self._row_to_deliverable(result)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get content by ID {content_id}: {e}")
            return None
    
    def get_campaign_content(self, campaign_id: str) -> List[ContentDeliverable]:
        """Get all content deliverables for a campaign"""
        query = """
            SELECT * FROM content_deliverables 
            WHERE campaign_id = %s 
            ORDER BY narrative_order ASC, created_at ASC
        """
        
        try:
            results = secure_db.execute_query(query, (campaign_id,), fetch="all")
            return [self._row_to_deliverable(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get campaign content for {campaign_id}: {e}")
            return []
    
    def get_content_by_type(self, campaign_id: str, content_type: str) -> List[ContentDeliverable]:
        """Get content deliverables by type for a campaign"""
        query = """
            SELECT * FROM content_deliverables 
            WHERE campaign_id = %s AND content_type = %s
            ORDER BY narrative_order ASC, created_at ASC
        """
        
        try:
            results = secure_db.execute_query(query, (campaign_id, content_type), fetch="all")
            return [self._row_to_deliverable(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get content by type {content_type}: {e}")
            return []
    
    def update_content(self, content_id: str, **updates) -> bool:
        """Update a content deliverable"""
        if not updates:
            return True
        
        # Add updated_at timestamp
        updates['updated_at'] = datetime.now()
        
        # Build dynamic update query
        set_clauses = [f"{key} = %({key})s" for key in updates.keys()]
        query = f"""
            UPDATE content_deliverables 
            SET {', '.join(set_clauses)}
            WHERE id = %(content_id)s
        """
        
        updates['content_id'] = content_id
        
        try:
            secure_db.execute_query(query, updates, fetch="none")
            logger.info(f"✅ Updated content deliverable {content_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update content {content_id}: {e}")
            return False
    
    def update_content_status(self, content_id: str, status: str) -> bool:
        """Update content status"""
        return self.update_content(content_id, status=status)
    
    def get_campaign_content_grouped(self, campaign_id: str) -> Dict[str, List[ContentDeliverable]]:
        """Get campaign content grouped by type"""
        content_items = self.get_campaign_content(campaign_id)
        
        grouped = {}
        for item in content_items:
            content_type = item.content_type
            if content_type not in grouped:
                grouped[content_type] = []
            grouped[content_type].append(item)
        
        return grouped
    
    def get_campaign_content_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get summary statistics for campaign content"""
        content_items = self.get_campaign_content(campaign_id)
        
        total_items = len(content_items)
        completed_items = len([item for item in content_items if item.status in ['approved', 'published']])
        
        # Group by type for counts
        type_counts = {}
        for item in content_items:
            content_type = item.content_type
            if content_type not in type_counts:
                type_counts[content_type] = {'total': 0, 'completed': 0}
            type_counts[content_type]['total'] += 1
            if item.status in ['approved', 'published']:
                type_counts[content_type]['completed'] += 1
        
        return {
            'total_content': total_items,
            'completed_content': completed_items,
            'completion_rate': round((completed_items / total_items * 100) if total_items > 0 else 0, 1),
            'content_by_type': type_counts,
            'total_word_count': sum(item.word_count or 0 for item in content_items),
            'estimated_reading_time': sum(item.reading_time or 0 for item in content_items)
        }
    
    def link_task_to_content(self, task_id: str, content_id: str) -> bool:
        """Link a campaign task to a content deliverable"""
        query = "UPDATE campaign_tasks SET content_deliverable_id = %s WHERE id = %s"
        
        try:
            secure_db.execute_query(query, (content_id, task_id), fetch="none")
            logger.info(f"✅ Linked task {task_id} to content {content_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to link task to content: {e}")
            return False
    
    def migrate_task_to_content(self, task_id: str, campaign_id: str, task_data: Dict[str, Any]) -> Optional[ContentDeliverable]:
        """Migrate a campaign task result to a content deliverable"""
        
        # Extract content from task result
        result = task_data.get('result', '')
        task_type = task_data.get('task_type', 'content_creation')
        
        # Map task types to content types
        content_type_mapping = {
            'content_creation': 'blog_post',
            'social_media_adaptation': 'social_media',
            'email_formatting': 'email',
            'image_generation': 'visual_content',
            'content_editing': 'editorial',
            'content_repurposing': 'repurposed_content',
            'seo_optimization': 'seo_content'
        }
        
        content_type = content_type_mapping.get(task_type, 'blog_post')
        
        # Generate title from result or task type
        title = f"{content_type.replace('_', ' ').title()} - {task_data.get('title', task_type)}"
        
        try:
            # Create content deliverable
            content = self.create_content(
                title=title,
                content=result,
                campaign_id=campaign_id,
                content_type=content_type,
                summary=f"Generated content from {task_type} task",
                metadata={'migrated_from_task': task_id}
            )
            
            # Link task to content
            if content:
                self.link_task_to_content(task_id, content.id)
                logger.info(f"✅ Migrated task {task_id} to content deliverable {content.id}")
                return content
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate task {task_id} to content: {e}")
            
        return None
    
    def _row_to_deliverable(self, row: Dict[str, Any]) -> ContentDeliverable:
        """Convert database row to ContentDeliverable object"""
        return ContentDeliverable(
            id=row['id'],
            title=row['title'],
            content=row['content'],
            summary=row.get('summary'),
            content_type=row.get('content_type', 'blog_post'),
            format=row.get('format', 'markdown'),
            status=row.get('status', 'draft'),
            parent_id=row.get('parent_id'),
            campaign_id=row.get('campaign_id'),
            narrative_order=row.get('narrative_order'),
            key_messages=row.get('key_messages', []),
            target_audience=row.get('target_audience'),
            tone=row.get('tone'),
            platform=row.get('platform'),
            word_count=row.get('word_count'),
            reading_time=row.get('reading_time'),
            seo_score=row.get('seo_score'),
            engagement_score=row.get('engagement_score'),
            created_by=row.get('created_by', 'ContentAgent'),
            last_edited_by=row.get('last_edited_by'),
            version=row.get('version', 1),
            is_published=row.get('is_published', False),
            published_at=row.get('published_at'),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
            metadata=row.get('metadata', {}) if isinstance(row.get('metadata'), dict) else json.loads(row.get('metadata', '{}')) if row.get('metadata') else {}
        )

# Global service instance
content_service = ContentDeliverableService()