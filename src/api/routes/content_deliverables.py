"""
Content Deliverables API Routes
Handles content-first deliverable operations, replacing task-centric endpoints
with deliverable-centric ones.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from enum import Enum

# Import database service and content service
from ...agents.core.database_service import DatabaseService, get_db_service
from ...agents.core.content_deliverable_service import content_service, ContentDeliverable
from ...agents.workflow.content_deliverable_workflow import content_deliverable_app
from ...services.task_to_content_migration import migration_service, MigrationResult

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v2/deliverables", tags=["Content Deliverables"])

# --- Pydantic Models ---

class ContentType(str, Enum):
    blog_post = "blog_post"
    social_media_post = "social_media_post" 
    email_campaign = "email_campaign"
    newsletter = "newsletter"
    whitepaper = "whitepaper"
    case_study = "case_study"
    video_script = "video_script"
    podcast_script = "podcast_script"
    press_release = "press_release"
    product_description = "product_description"
    landing_page = "landing_page"
    ad_copy = "ad_copy"
    infographic_concept = "infographic_concept"
    webinar_outline = "webinar_outline"

class ContentFormat(str, Enum):
    markdown = "markdown"
    html = "html"
    plain_text = "plain_text"
    json = "json"
    structured_data = "structured_data"

class DeliverableStatus(str, Enum):
    draft = "draft"
    in_review = "in_review"
    approved = "approved"
    published = "published"
    archived = "archived"
    needs_revision = "needs_revision"

class ContentDeliverableCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    summary: Optional[str] = Field(None, max_length=1000)
    content_type: ContentType
    format: ContentFormat = ContentFormat.markdown
    status: DeliverableStatus = DeliverableStatus.draft
    campaign_id: str = Field(..., description="Campaign ID this deliverable belongs to")
    narrative_order: Optional[int] = Field(None, description="Order in the campaign narrative")
    key_messages: List[str] = Field(default_factory=list)
    target_audience: Optional[str] = Field(None, max_length=500)
    tone: Optional[str] = Field(None, max_length=100)
    platform: Optional[str] = Field(None, max_length=100)
    word_count: Optional[int] = Field(None, ge=0)
    reading_time: Optional[int] = Field(None, ge=0)
    created_by: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None

class ContentDeliverableUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    summary: Optional[str] = Field(None, max_length=1000)
    status: Optional[DeliverableStatus] = None
    key_messages: Optional[List[str]] = None
    target_audience: Optional[str] = Field(None, max_length=500)
    tone: Optional[str] = Field(None, max_length=100)
    platform: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None

class ContentDeliverableResponse(BaseModel):
    id: str
    title: str
    content: str
    summary: Optional[str]
    content_type: ContentType
    format: ContentFormat
    status: DeliverableStatus
    campaign_id: str
    narrative_order: Optional[int]
    key_messages: List[str]
    target_audience: Optional[str]
    tone: Optional[str]
    platform: Optional[str]
    word_count: Optional[int]
    reading_time: Optional[int]
    seo_score: Optional[float]
    engagement_score: Optional[float]
    created_by: Optional[str]
    last_edited_by: Optional[str]
    version: int
    is_published: bool
    published_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]

class CampaignGenerationRequest(BaseModel):
    """Request model for generating content deliverables for a campaign"""
    campaign_id: str
    briefing: Dict[str, Any] = Field(..., description="Campaign briefing information")
    content_strategy: Optional[Dict[str, Any]] = Field(None, description="Optional content strategy")
    deliverable_count: int = Field(3, ge=1, le=10, description="Number of deliverables to generate")
    content_types: Optional[List[ContentType]] = Field(None, description="Specific content types to generate")

class ContentGenerationResponse(BaseModel):
    """Response model for content generation"""
    campaign_id: str
    narrative_theme: str
    story_arc: List[str]
    deliverables_created: int
    deliverable_ids: List[str]
    content_relationships: Dict[str, List[str]]
    generation_summary: str

class ContentNarrativeResponse(BaseModel):
    """Response model for content narrative"""
    id: str
    campaign_id: str
    title: str
    description: Optional[str]
    narrative_theme: str
    key_story_arc: List[str]
    content_flow: Dict[str, Any]
    total_pieces: int
    completed_pieces: int
    created_at: datetime
    updated_at: datetime

# --- API Endpoints ---

@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_campaign_deliverables(
    request: CampaignGenerationRequest,
    db: DatabaseService = Depends(get_db_service)
):
    """
    Generate content deliverables for a campaign using the content-first workflow
    This replaces the task-based generation with deliverable-focused creation
    """
    try:
        logger.info(f"🚀 Starting content generation for campaign: {request.campaign_id}")
        
        # Prepare workflow input
        workflow_input = {
            "campaign_id": request.campaign_id,
            "briefing": request.briefing,
            "content_strategy": request.content_strategy or {},
            "target_deliverable_count": request.deliverable_count,
            "requested_content_types": [ct.value for ct in (request.content_types or [])]
        }
        
        # Execute content-first workflow
        result = content_deliverable_app.invoke(workflow_input)
        
        # Extract results
        deliverables = result.get("deliverables", [])
        narrative_theme = result.get("narrative_theme", "")
        story_arc = result.get("story_arc", [])
        relationships = result.get("content_relationships", {})
        deliverable_ids = result.get("stored_deliverable_ids", [])
        
        logger.info(f"✅ Generated {len(deliverables)} deliverables for campaign {request.campaign_id}")
        
        # Create response
        return ContentGenerationResponse(
            campaign_id=request.campaign_id,
            narrative_theme=narrative_theme,
            story_arc=story_arc,
            deliverables_created=len(deliverables),
            deliverable_ids=deliverable_ids,
            content_relationships=relationships,
            generation_summary=f"Successfully generated {len(deliverables)} content deliverables with theme: {narrative_theme}"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate deliverables for campaign {request.campaign_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate content deliverables: {str(e)}"
        )

@router.get("/campaign/{campaign_id}", response_model=List[ContentDeliverableResponse])
async def get_campaign_deliverables(
    campaign_id: str,
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    status: Optional[DeliverableStatus] = Query(None, description="Filter by status"),
    platform: Optional[str] = Query(None, description="Filter by platform")
):
    """
    Get all content deliverables for a campaign
    This replaces task listing with actual deliverable listing
    """
    try:
        logger.info(f"📋 Retrieving deliverables for campaign: {campaign_id}")
        
        # Get deliverables using content service
        if content_type:
            deliverables = content_service.get_content_by_type(campaign_id, content_type.value)
        else:
            deliverables = content_service.get_campaign_content(campaign_id)
        
        # Apply additional filters
        if status:
            deliverables = [d for d in deliverables if d.status == status.value]
        if platform:
            deliverables = [d for d in deliverables if d.platform == platform]
        
        logger.info(f"📄 Found {len(deliverables)} deliverables for campaign {campaign_id}")
        
        # Convert to response models
        response_deliverables = []
        for d in deliverables:
            response_deliverables.append(ContentDeliverableResponse(
                id=d.id,
                title=d.title,
                content=d.content,
                summary=d.summary,
                content_type=ContentType(d.content_type),
                format=ContentFormat(d.format),
                status=DeliverableStatus(d.status),
                campaign_id=d.campaign_id,
                narrative_order=d.narrative_order,
                key_messages=d.key_messages or [],
                target_audience=d.target_audience,
                tone=d.tone,
                platform=d.platform,
                word_count=d.word_count,
                reading_time=d.reading_time,
                seo_score=d.seo_score,
                engagement_score=d.engagement_score,
                created_by=d.created_by,
                last_edited_by=d.last_edited_by,
                version=d.version,
                is_published=d.is_published,
                published_at=d.published_at,
                created_at=d.created_at,
                updated_at=d.updated_at,
                metadata=d.metadata
            ))
        
        return response_deliverables
        
    except Exception as e:
        logger.error(f"❌ Failed to get deliverables for campaign {campaign_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve deliverables: {str(e)}"
        )

# --- Health Check (must come before parameterized routes) ---
@router.get("/health")
async def deliverables_health_check():
    """Health check for content deliverables system"""
    try:
        db = get_db_service()
        # Basic database connectivity test
        health = db.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": health.get("status", "unknown"),
            "features": [
                "content_deliverable_generation",
                "narrative_coordination", 
                "content_relationships",
                "deliverable_management"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/{deliverable_id}", response_model=ContentDeliverableResponse)
async def get_deliverable(
    deliverable_id: str
):
    """Get a specific content deliverable by ID"""
    try:
        deliverable = content_service.get_content_by_id(deliverable_id)
        
        if not deliverable:
            raise HTTPException(status_code=404, detail="Content deliverable not found")
        
        return ContentDeliverableResponse(
            id=deliverable.id,
            title=deliverable.title,
            content=deliverable.content,
            summary=deliverable.summary,
            content_type=ContentType(deliverable.content_type),
            format=ContentFormat(deliverable.format),
            status=DeliverableStatus(deliverable.status),
            campaign_id=deliverable.campaign_id,
            narrative_order=deliverable.narrative_order,
            key_messages=deliverable.key_messages or [],
            target_audience=deliverable.target_audience,
            tone=deliverable.tone,
            platform=deliverable.platform,
            word_count=deliverable.word_count,
            reading_time=deliverable.reading_time,
            seo_score=deliverable.seo_score,
            engagement_score=deliverable.engagement_score,
            created_by=deliverable.created_by,
            last_edited_by=deliverable.last_edited_by,
            version=deliverable.version,
            is_published=deliverable.is_published,
            published_at=deliverable.published_at,
            created_at=deliverable.created_at,
            updated_at=deliverable.updated_at,
            metadata=deliverable.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get deliverable {deliverable_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve deliverable: {str(e)}"
        )

@router.post("/", response_model=ContentDeliverableResponse)
async def create_deliverable(
    deliverable: ContentDeliverableCreate
):
    """Create a new content deliverable"""
    try:
        logger.info(f"📝 Creating new deliverable: {deliverable.title}")
        
        # Create using content service
        created_deliverable = content_service.create_content(
            title=deliverable.title,
            content=deliverable.content,
            campaign_id=deliverable.campaign_id,
            content_type=deliverable.content_type.value,
            summary=deliverable.summary,
            platform=deliverable.platform,
            key_messages=deliverable.key_messages,
            target_audience=deliverable.target_audience,
            tone=deliverable.tone,
            narrative_order=deliverable.narrative_order,
            word_count=deliverable.word_count,
            reading_time=deliverable.reading_time,
            created_by=deliverable.created_by,
            metadata=deliverable.metadata
        )
        
        if not created_deliverable:
            raise HTTPException(status_code=500, detail="Failed to create content deliverable")
        
        logger.info(f"✅ Created deliverable: {created_deliverable.id}")
        
        return ContentDeliverableResponse(
            id=created_deliverable.id,
            title=created_deliverable.title,
            content=created_deliverable.content,
            summary=created_deliverable.summary,
            content_type=ContentType(created_deliverable.content_type),
            format=ContentFormat(created_deliverable.format),
            status=DeliverableStatus(created_deliverable.status),
            campaign_id=created_deliverable.campaign_id,
            narrative_order=created_deliverable.narrative_order,
            key_messages=created_deliverable.key_messages or [],
            target_audience=created_deliverable.target_audience,
            tone=created_deliverable.tone,
            platform=created_deliverable.platform,
            word_count=created_deliverable.word_count,
            reading_time=created_deliverable.reading_time,
            seo_score=created_deliverable.seo_score,
            engagement_score=created_deliverable.engagement_score,
            created_by=created_deliverable.created_by,
            last_edited_by=created_deliverable.last_edited_by,
            version=created_deliverable.version,
            is_published=created_deliverable.is_published,
            published_at=created_deliverable.published_at,
            created_at=created_deliverable.created_at,
            updated_at=created_deliverable.updated_at,
            metadata=created_deliverable.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create deliverable: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create deliverable: {str(e)}"
        )

@router.patch("/{deliverable_id}", response_model=ContentDeliverableResponse)
async def update_deliverable(
    deliverable_id: str,
    updates: ContentDeliverableUpdate
):
    """Update a content deliverable"""
    try:
        logger.info(f"📝 Updating deliverable: {deliverable_id}")
        
        # Update using content service
        update_data = updates.dict(exclude_unset=True)
        
        # Convert enum values to strings
        if 'status' in update_data and update_data['status']:
            update_data['status'] = update_data['status'].value
        
        success = content_service.update_content(deliverable_id, **update_data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Content deliverable not found")
        
        # Retrieve the updated deliverable
        updated_deliverable = content_service.get_content_by_id(deliverable_id)
        
        if not updated_deliverable:
            raise HTTPException(status_code=404, detail="Content deliverable not found after update")
        
        logger.info(f"✅ Updated deliverable: {deliverable_id}")
        
        return ContentDeliverableResponse(
            id=updated_deliverable.id,
            title=updated_deliverable.title,
            content=updated_deliverable.content,
            summary=updated_deliverable.summary,
            content_type=ContentType(updated_deliverable.content_type),
            format=ContentFormat(updated_deliverable.format),
            status=DeliverableStatus(updated_deliverable.status),
            campaign_id=updated_deliverable.campaign_id,
            narrative_order=updated_deliverable.narrative_order,
            key_messages=updated_deliverable.key_messages or [],
            target_audience=updated_deliverable.target_audience,
            tone=updated_deliverable.tone,
            platform=updated_deliverable.platform,
            word_count=updated_deliverable.word_count,
            reading_time=updated_deliverable.reading_time,
            seo_score=updated_deliverable.seo_score,
            engagement_score=updated_deliverable.engagement_score,
            created_by=updated_deliverable.created_by,
            last_edited_by=updated_deliverable.last_edited_by,
            version=updated_deliverable.version,
            is_published=updated_deliverable.is_published,
            published_at=updated_deliverable.published_at,
            created_at=updated_deliverable.created_at,
            updated_at=updated_deliverable.updated_at,
            metadata=updated_deliverable.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update deliverable {deliverable_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update deliverable: {str(e)}"
        )

@router.patch("/{deliverable_id}/status", response_model=Dict[str, Any])
async def update_deliverable_status(
    deliverable_id: str,
    status: DeliverableStatus,
    notes: Optional[str] = None
):
    """Update the status of a content deliverable"""
    try:
        logger.info(f"🔄 Updating status for deliverable {deliverable_id} to {status}")
        
        update_data = {"status": status.value}
        if status.value == "published":
            update_data["published_at"] = datetime.utcnow()
            update_data["is_published"] = True
        
        if notes:
            # Add notes to metadata if provided
            existing = content_service.get_content_by_id(deliverable_id)
            if existing:
                metadata = existing.metadata or {}
                metadata["status_notes"] = notes
                update_data["metadata"] = metadata
        
        success = content_service.update_content(deliverable_id, **update_data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Content deliverable not found")
        
        return {
            "deliverable_id": deliverable_id,
            "new_status": status.value,
            "notes": notes,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update deliverable status {deliverable_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update deliverable status: {str(e)}"
        )

@router.get("/campaign/{campaign_id}/summary")
async def get_campaign_content_summary(campaign_id: str) -> Dict[str, Any]:
    """
    Get campaign content summary with statistics
    Shows actual deliverable counts instead of task counts
    """
    try:
        logger.info(f"📊 Retrieving content summary for campaign: {campaign_id}")
        
        summary = content_service.get_campaign_content_summary(campaign_id)
        
        return {
            "campaign_id": campaign_id,
            "content_overview": {
                "total_pieces": summary["total_content"],
                "completed_pieces": summary["completed_content"],
                "completion_percentage": summary["completion_rate"],
                "total_words": summary["total_word_count"],
                "estimated_reading_time": summary["estimated_reading_time"]
            },
            "content_breakdown": summary["content_by_type"],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get campaign content summary for {campaign_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve content summary: {str(e)}")

@router.get("/campaign/{campaign_id}/grouped")
async def get_campaign_content_grouped(campaign_id: str) -> Dict[str, Any]:
    """
    Get campaign content grouped by type for narrative overview
    This provides the content-first view instead of task-centric
    """
    try:
        logger.info(f"📋 Retrieving grouped content for campaign: {campaign_id}")
        
        grouped_content = content_service.get_campaign_content_grouped(campaign_id)
        summary = content_service.get_campaign_content_summary(campaign_id)
        
        # Convert to response format
        content_by_type = {}
        for content_type, items in grouped_content.items():
            content_by_type[content_type] = [
                {
                    "id": item.id,
                    "title": item.title,
                    "content": item.content,
                    "summary": item.summary,
                    "status": item.status,
                    "narrative_order": item.narrative_order,
                    "word_count": item.word_count,
                    "reading_time": item.reading_time,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                }
                for item in items
            ]
        
        # Build narrative flow
        all_content = []
        for items in grouped_content.values():
            all_content.extend(items)
        
        # Sort by narrative order or creation date
        all_content.sort(key=lambda x: (x.narrative_order or 999, x.created_at or datetime.utcnow()))
        
        narrative_flow = []
        for i, content in enumerate(all_content):
            flow_item = {
                "position": i + 1,
                "content_id": content.id,
                "title": content.title,
                "content_type": content.content_type,
                "status": content.status,
                "word_count": content.word_count
            }
            
            if i > 0:
                flow_item["previous_content"] = all_content[i-1].id
            if i < len(all_content) - 1:
                flow_item["next_content"] = all_content[i+1].id
            
            narrative_flow.append(flow_item)
        
        return {
            "campaign_id": campaign_id,
            "summary": summary,
            "content_by_type": content_by_type,
            "narrative_flow": narrative_flow,
            "total_content_pieces": len(all_content),
            "content_types_count": len(grouped_content)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get grouped content for campaign {campaign_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve grouped content: {str(e)}")

@router.get("/campaign/{campaign_id}/narrative", response_model=ContentNarrativeResponse)
async def get_campaign_narrative(
    campaign_id: str,
    db: DatabaseService = Depends(get_db_service)
):
    """Get the content narrative for a campaign"""
    try:
        logger.info(f"📖 Retrieving narrative for campaign: {campaign_id}")
        
        # This would need to be implemented in DatabaseService
        narrative = db.get_campaign_narrative(campaign_id)
        
        if not narrative:
            raise HTTPException(status_code=404, detail="Content narrative not found")
        
        return ContentNarrativeResponse(**narrative)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get campaign narrative {campaign_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve campaign narrative: {str(e)}"
        )

@router.post("/campaign/{campaign_id}/migrate-tasks")
async def migrate_tasks_to_content(
    campaign_id: str,
    dry_run: bool = Query(False, description="Preview migration without actually creating deliverables")
) -> Dict[str, Any]:
    """
    Migrate existing campaign tasks to content deliverables
    Transforms task-centric data to content-centric approach
    """
    try:
        logger.info(f"🚀 Task migration requested for campaign {campaign_id} (dry_run={dry_run})")
        
        # Perform the migration
        result = migration_service.migrate_campaign_tasks(campaign_id, dry_run=dry_run)
        
        # Build response
        response = {
            "campaign_id": result.campaign_id,
            "migration_status": "completed" if not dry_run else "preview",
            "summary": {
                "tasks_processed": result.tasks_processed,
                "deliverables_created": result.deliverables_created,
                "success_rate": result.success_rate,
                "errors_count": len(result.errors)
            },
            "created_deliverables": result.created_deliverables,
            "errors": result.errors[:5],  # Limit errors shown
            "message": f"{'Preview:' if dry_run else ''} {result.deliverables_created}/{result.tasks_processed} tasks {'would be' if dry_run else 'were'} successfully migrated to content deliverables"
        }
        
        if dry_run:
            response["note"] = "This was a preview. Set dry_run=false to actually perform the migration."
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to migrate tasks for campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to migrate tasks: {str(e)}")

@router.post("/migrate-all-tasks")
async def migrate_all_tasks_to_content(
    dry_run: bool = Query(True, description="Preview migration without actually creating deliverables")
) -> Dict[str, Any]:
    """
    Migrate all existing campaign tasks to content deliverables
    Transforms the entire system from task-centric to content-centric
    """
    try:
        logger.info(f"🚀 Global task migration requested (dry_run={dry_run})")
        
        # Perform the migration for all campaigns
        results = migration_service.migrate_all_campaigns(dry_run=dry_run)
        
        if not results:
            return {
                "migration_status": "no_data",
                "message": "No campaigns with tasks found to migrate"
            }
        
        # Calculate totals
        total_campaigns = len(results)
        successful_campaigns = len([r for r in results.values() if r.success_rate > 0])
        total_tasks = sum(r.tasks_processed for r in results.values())
        total_deliverables = sum(r.deliverables_created for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        overall_success_rate = (total_deliverables / total_tasks * 100) if total_tasks > 0 else 0
        
        # Build campaign summaries
        campaign_summaries = []
        for campaign_id, result in results.items():
            campaign_summaries.append({
                "campaign_id": campaign_id,
                "tasks_processed": result.tasks_processed,
                "deliverables_created": result.deliverables_created,
                "success_rate": result.success_rate,
                "errors_count": len(result.errors)
            })
        
        response = {
            "migration_status": "completed" if not dry_run else "preview",
            "summary": {
                "total_campaigns": total_campaigns,
                "successful_campaigns": successful_campaigns,
                "total_tasks_processed": total_tasks,
                "total_deliverables_created": total_deliverables,
                "total_errors": total_errors,
                "overall_success_rate": round(overall_success_rate, 1)
            },
            "campaigns": campaign_summaries,
            "message": f"{'Preview:' if dry_run else ''} {total_deliverables}/{total_tasks} tasks from {successful_campaigns}/{total_campaigns} campaigns {'would be' if dry_run else 'were'} migrated to content deliverables"
        }
        
        if dry_run:
            response["note"] = "This was a preview. Set dry_run=false to actually perform the migration."
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to migrate all tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to migrate all tasks: {str(e)}")

@router.delete("/{deliverable_id}")
async def delete_deliverable(
    deliverable_id: str,
    db: DatabaseService = Depends(get_db_service)
):
    """Delete a content deliverable"""
    try:
        logger.info(f"🗑️ Deleting deliverable: {deliverable_id}")
        
        # This would need to be implemented in DatabaseService
        success = db.delete_content_deliverable(deliverable_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Content deliverable not found")
        
        return {"message": "Content deliverable deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete deliverable {deliverable_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete deliverable: {str(e)}"
        )

# Health endpoint moved above to avoid path parameter conflict