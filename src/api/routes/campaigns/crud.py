"""
Campaign CRUD Operations Routes
Handles campaign creation, reading, updating, and deletion.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic Models
class CampaignCreateRequest(BaseModel):
    blog_id: Optional[str] = None  # Optional for orchestration campaigns
    campaign_name: str
    company_context: str
    content_type: str = "blog"
    template_id: Optional[str] = None
    template_config: Optional[Dict[str, Any]] = None
    
    # Enhanced wizard fields
    description: Optional[str] = None
    strategy_type: Optional[str] = None
    priority: Optional[str] = None
    target_audience: Optional[str] = None
    distribution_channels: Optional[List[str]] = None
    timeline_weeks: Optional[int] = None
    scheduled_start: Optional[str] = None
    deadline: Optional[str] = None
    success_metrics: Optional[Dict[str, Any]] = None
    budget_allocation: Optional[Dict[str, Any]] = None

class CampaignSummary(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    created_at: str

class CampaignDetail(BaseModel):
    id: str
    name: str
    status: str
    strategy: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    scheduled_posts: List[Dict[str, Any]]
    performance: Dict[str, Any]

class QuickCampaignRequest(BaseModel):
    blog_id: str
    campaign_name: str

# Helper functions
async def _update_campaign_metadata(campaign_id: str, scheduled_start: Optional[str], 
                                   deadline: Optional[str], priority: Optional[str]) -> None:
    """Update campaign with wizard-specific metadata"""
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Store all wizard metadata in the metadata JSONB column
            metadata_updates = {}
            
            if scheduled_start:
                metadata_updates["scheduled_start"] = scheduled_start
                
            if deadline:
                metadata_updates["deadline"] = deadline
                
            if priority:
                metadata_updates["priority"] = priority
            
            if metadata_updates:
                query = "UPDATE campaigns SET metadata = metadata || %s, updated_at = NOW() WHERE id = %s"
                cur.execute(query, (json.dumps(metadata_updates), campaign_id))
                conn.commit()
                
    except Exception as e:
        logger.warning(f"Error updating campaign metadata: {str(e)}")

async def _create_campaign_tasks(campaign_id: str, campaign_data: dict):
    """Create campaign tasks directly in database"""
    try:
        tasks = [
            {"type": "content_planning", "agent": "PlannerAgent", "status": "pending"},
            {"type": "research", "agent": "ResearcherAgent", "status": "pending"},
            {"type": "content_creation", "agent": "WriterAgent", "status": "pending"},
            {"type": "content_editing", "agent": "EditorAgent", "status": "pending"},
            {"type": "seo_optimization", "agent": "SEOAgent", "status": "pending"},
            {"type": "image_generation", "agent": "ImageAgent", "status": "pending"},
            {"type": "social_adaptation", "agent": "SocialMediaAgent", "status": "pending"},
        ]
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            tasks_created = 0
            for task in tasks:
                cur.execute("""
                    INSERT INTO campaign_tasks (campaign_id, task_type, agent_type, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """, (
                    campaign_id,
                    task["type"],
                    task["agent"],
                    task["status"]
                ))
                tasks_created += 1
            
            conn.commit()
            return tasks_created
            
    except Exception as e:
        logger.error(f"Error creating campaign tasks: {e}")
        raise

def get_campaign_manager():
    """Get campaign manager instance with lazy loading"""
    try:
        from src.agents.specialized.campaign_manager import CampaignManagerAgent
        return CampaignManagerAgent()
    except ImportError as e:
        logger.error(f"Failed to import CampaignManagerAgent: {e}")
        raise HTTPException(status_code=500, detail="Campaign manager not available")

# Background task for automatic agent execution
async def execute_campaign_agents_background(campaign_id: str, campaign_data: dict):
    """
    Background task to automatically execute AI agents for a newly created campaign.
    """
    try:
        logger.info(f"🤖 [AGENT EXECUTION] Starting automatic agent workflow for campaign: {campaign_id}")
        
        # Import workflow manager
        from .workflow import websocket_manager
        
        # Broadcast workflow start
        await websocket_manager.broadcast_to_campaign({
            "type": "workflow_started",
            "campaign_id": campaign_id,
            "agent_type": "workflow_orchestrator",
            "status": "running",
            "message": "Starting content generation workflow",
            "progress": 0,
            "timestamp": datetime.now().isoformat()
        }, campaign_id)
        
        # Update campaign status to indicate processing has started
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE campaigns 
                SET metadata = COALESCE(metadata, '{}')::jsonb || '{"processing_status": "generating_content"}'::jsonb,
                    updated_at = NOW()
                WHERE id = %s
            """, (campaign_id,))
            conn.commit()
        
        # Import and run orchestration system
        try:
            from src.agents.orchestration.advanced_orchestrator import AdvancedOrchestrator
            from src.agents.orchestration.master_planner_agent import MasterPlannerAgent
            
            orchestrator = AdvancedOrchestrator(enable_recovery_systems=True)
            master_planner = MasterPlannerAgent()
            
            execution_plan = await master_planner.create_execution_plan(
                campaign_id=campaign_id,
                workflow_execution_id=f"campaign_{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy="adaptive",
                required_agents=["planner", "researcher", "writer", "editor", "seo"]
            )
            
            context_data = {
                "campaign_id": campaign_id,
                "company_context": campaign_data.get("company_context", ""),
                "target_audience": campaign_data.get("target_audience", "business professionals"),
                "content_objective": campaign_data.get("campaign_name", ""),
                "strategy_type": campaign_data.get("strategy_type", "thought_leadership"),
                "distribution_channels": campaign_data.get("distribution_channels", ["blog"]),
                "priority": campaign_data.get("priority", "medium")
            }
            
            logger.info(f"🤖 [ORCHESTRATION] Starting advanced workflow execution for campaign: {campaign_id}")
            
            # Execute workflow
            await orchestrator.execute_workflow(execution_plan, context_data)
            
        except Exception as orchestration_error:
            logger.error(f"Orchestration failed for campaign {campaign_id}: {str(orchestration_error)}")
            # Update campaign status to indicate failure
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE campaigns 
                    SET metadata = COALESCE(metadata, '{}')::jsonb || '{"processing_status": "failed", "error": %s}'::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                """, (str(orchestration_error), campaign_id))
                conn.commit()
            
    except Exception as e:
        logger.error(f"Background agent execution failed for campaign {campaign_id}: {str(e)}")

# CRUD Endpoints
@router.post("/", response_model=Dict[str, Any])
async def create_campaign(request: CampaignCreateRequest, background_tasks: BackgroundTasks):
    """
    Create a new AI-enhanced campaign with wizard support.
    Supports both blog-based campaigns and orchestration campaigns.
    """
    try:
        logger.info(f"🚀 [CAMPAIGN CREATE] Starting campaign creation: {request.campaign_name}")
        
        # Determine campaign type
        is_orchestration_campaign = request.blog_id is None
        campaign_type_desc = "orchestration" if is_orchestration_campaign else f"blog {request.blog_id}"
        
        logger.info(f"🚀 [CAMPAIGN CREATE] Creating {campaign_type_desc} campaign")
        
        # Prepare enhanced template configuration from wizard data
        enhanced_template_config = request.template_config or {}
        
        # Add wizard data to template configuration
        if request.strategy_type:
            enhanced_template_config["strategy_type"] = request.strategy_type
        if request.priority:
            enhanced_template_config["priority"] = request.priority
        if request.target_audience:
            enhanced_template_config["target_audience"] = request.target_audience
        if request.distribution_channels:
            enhanced_template_config["channels"] = request.distribution_channels
        if request.timeline_weeks:
            enhanced_template_config["timeline_weeks"] = request.timeline_weeks
        if request.success_metrics:
            enhanced_template_config["success_metrics"] = request.success_metrics
        if request.budget_allocation:
            enhanced_template_config["budget_allocation"] = request.budget_allocation
        
        # For orchestration campaigns, mark as orchestration mode
        if is_orchestration_campaign:
            enhanced_template_config["orchestration_mode"] = True
            enhanced_template_config["campaign_data"] = {
                "campaign_name": request.campaign_name,
                "campaign_objective": request.strategy_type or "Brand awareness and lead generation",
                "company_context": request.company_context,
                "target_market": request.target_audience or "B2B professionals",
                "industry": "B2B Services",
                "channels": request.distribution_channels or ["linkedin", "email"],
                "content_types": ["blog_posts", "social_posts", "email_content"],
                "timeline_weeks": request.timeline_weeks or 4,
                "desired_tone": "Professional and engaging",
                "key_messages": [request.description] if request.description else [],
                "success_metrics": request.success_metrics or {
                    "blog_posts": 2,
                    "social_posts": 5, 
                    "email_content": 3,
                    "seo_optimization": 1,
                    "competitor_analysis": 1,
                    "image_generation": 2,
                    "repurposed_content": 4,
                    "performance_analytics": 1
                },
                "budget_allocation": request.budget_allocation or {},
                "target_personas": [{
                    "name": "Business Decision Maker",
                    "role": "Executive/Manager",
                    "pain_points": ["Need efficient solutions", "Time constraints", "ROI concerns"],
                    "channels": request.distribution_channels or ["linkedin", "email"]
                }]
            }
        
        # Use enhanced company context
        company_context = request.description or request.company_context
        
        # Initialize campaign manager
        campaign_manager = get_campaign_manager()
        
        # Create AI-enhanced campaign plan
        campaign_plan = await campaign_manager.create_campaign_plan(
            blog_id=request.blog_id or "orchestration_campaign",
            campaign_name=request.campaign_name,
            company_context=company_context,
            content_type=request.content_type,
            template_id=request.template_id or "ai_enhanced",
            template_config=enhanced_template_config
        )
        
        # Insert the campaign into the database
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO campaigns (id, name, status, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, NOW(), NOW(), %s)
                """, (
                    campaign_plan["campaign_id"],
                    request.campaign_name,
                    "active",
                    json.dumps({
                        "strategy_type": request.strategy_type,
                        "content_type": request.content_type,
                        "template_id": request.template_id,
                        "orchestration_mode": is_orchestration_campaign
                    })
                ))
                conn.commit()
                logger.info(f"🚀 [CAMPAIGN CREATE] Campaign inserted into database: {campaign_plan['campaign_id']}")
        except Exception as e:
            logger.error(f"Failed to insert campaign into database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")
        
        # Update campaign with wizard-specific data in database
        if any([request.scheduled_start, request.deadline, request.priority]):
            await _update_campaign_metadata(
                campaign_plan["campaign_id"],
                request.scheduled_start,
                request.deadline,
                request.priority
            )
        
        # Prepare response based on campaign type
        response_data = {
            "success": True,
            "campaign_id": campaign_plan["campaign_id"],
            "message": f"AI-enhanced {campaign_type_desc} campaign created successfully",
            "campaign_type": "orchestration" if is_orchestration_campaign else "blog_based",
            "strategy": campaign_plan.get("strategy", {}),
            "timeline": campaign_plan.get("timeline", []),
            "ai_enhanced": True,
            "intelligence_version": campaign_plan.get("intelligence_version", "2.0"),
            "wizard_data": {
                "strategy_type": request.strategy_type,
                "priority": request.priority,
                "timeline_weeks": request.timeline_weeks,
                "channels_count": len(request.distribution_channels or [])
            }
        }
        
        # Add orchestration-specific data
        if is_orchestration_campaign:
            response_data.update({
                "content_tasks": campaign_plan.get("content_tasks", []),
                "content_strategy": campaign_plan.get("content_strategy", {}),
                "orchestration_mode": campaign_plan.get("orchestration_mode", True),
                "tasks": len(campaign_plan.get("content_tasks", [])),
                "competitive_insights": campaign_plan.get("competitive_insights", {}),
                "market_opportunities": campaign_plan.get("market_opportunities", {})
            })
            
            # Create tasks directly for orchestration campaigns
            try:
                tasks_created = await _create_campaign_tasks(
                    campaign_plan["campaign_id"],
                    enhanced_template_config["campaign_data"]
                )
                
                response_data["task_creation"] = {
                    "enabled": True,
                    "tasks_created": tasks_created,
                    "status": "tasks_created",
                    "method": "direct_insertion"
                }
                
            except Exception as task_error:
                logger.warning(f"⚠️ Campaign created but task creation failed: {str(task_error)}")
                response_data["task_creation"] = {
                    "enabled": False,
                    "error": str(task_error),
                    "fallback": "Campaign created without tasks - can be added manually"
                }
        else:
            response_data.update({
                "tasks": len(campaign_plan.get("tasks", [])),
                "competitive_insights": campaign_plan.get("competitive_insights", {}),
                "market_opportunities": campaign_plan.get("market_opportunities", {})
            })

        # Trigger automatic agent execution in background
        logger.info(f"🤖 [CAMPAIGN CREATE] Triggering automatic agent execution for campaign: {campaign_plan['campaign_id']}")
        background_tasks.add_task(
            execute_campaign_agents_background,
            campaign_plan["campaign_id"],
            {
                "company_context": request.company_context or request.description or "",
                "campaign_name": request.campaign_name,
                "target_audience": request.target_audience or "business professionals",
                "strategy_type": request.strategy_type or "thought_leadership",
                "distribution_channels": request.distribution_channels or ["blog"],
                "priority": request.priority or "medium"
            }
        )

        return response_data
        
    except Exception as e:
        import traceback
        logger.error(f"🚀 [CAMPAIGN CREATE] ERROR: {str(e)}")
        logger.error(f"🚀 [CAMPAIGN CREATE] ERROR type: {type(e).__name__}")
        logger.error(f"🚀 [CAMPAIGN CREATE] ERROR traceback: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=f"Failed to create AI-enhanced campaign: {str(e)}")

@router.post("/quick/{template_id}", response_model=Dict[str, Any])
async def create_quick_campaign(template_id: str, request: QuickCampaignRequest):
    """
    Create a quick campaign using a predefined template
    """
    try:
        logger.info(f"Creating quick campaign with template {template_id} for blog {request.blog_id}")
        
        # Define template configurations
        template_configs = {
            "social-blast": {
                "channels": ["linkedin", "twitter", "facebook"],
                "auto_adapt": True,
                "schedule_immediately": True
            },
            "professional-share": {
                "channels": ["linkedin"],
                "format": "professional_article",
                "auto_adapt": True,
                "schedule_immediately": True
            },
            "email-campaign": {
                "channels": ["email"],
                "format": "newsletter",
                "auto_adapt": True,
                "schedule_immediately": False
            }
        }
        
        if template_id not in template_configs:
            raise HTTPException(status_code=400, detail=f"Unknown template: {template_id}")
        
        # Fetch blog info to get company context
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT title, initial_prompt
                    FROM blog_posts 
                    WHERE id = %s
                """, (request.blog_id,))
                
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                blog_title, initial_prompt = row
                # Extract company context from initial prompt if available
                company_context = ""
                if initial_prompt and isinstance(initial_prompt, dict):
                    company_context = initial_prompt.get('company_context', '')
        except Exception as e:
            logger.warning(f"Could not fetch blog context: {str(e)}")
            company_context = ""

        # Initialize campaign manager
        campaign_manager = get_campaign_manager()
        
        # Create campaign plan using the campaign manager
        campaign_plan = await campaign_manager.create_campaign_plan(
            blog_id=request.blog_id,
            campaign_name=request.campaign_name,
            company_context=company_context,
            content_type="blog",
            template_id=template_id,
            template_config=template_configs[template_id]
        )
        
        # Auto-execute for simple templates
        auto_executed = False
        if template_configs[template_id].get("schedule_immediately"):
            try:
                from src.agents.specialized.task_scheduler import TaskSchedulerAgent
                task_scheduler = TaskSchedulerAgent()
                await task_scheduler.schedule_campaign_tasks(
                    campaign_plan["campaign_id"], 
                    campaign_plan["strategy"]
                )
                auto_executed = True
                logger.info(f"Auto-scheduled campaign {campaign_plan['campaign_id']}")
            except Exception as e:
                logger.warning(f"Failed to auto-schedule campaign: {str(e)}")
        
        return JSONResponse(content={
            "success": True,
            "campaign_id": campaign_plan["campaign_id"],
            "message": f"Quick campaign '{template_id}' created successfully",
            "template_id": template_id,
            "auto_executed": auto_executed,
            "strategy": campaign_plan["strategy"],
            "tasks": len(campaign_plan["tasks"])
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating quick campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create quick campaign: {str(e)}")

@router.get("/", response_model=List[CampaignSummary])
async def list_campaigns():
    """
    List all campaigns with real data
    """
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            # Simplified fast query - get basic campaign data
            cur.execute("""
                SELECT 
                    c.id as campaign_id,
                    COALESCE(c.name, 'Unnamed Campaign') as campaign_name,
                    c.status,
                    c.created_at,
                    COUNT(DISTINCT ct.id) as total_tasks,
                    COUNT(DISTINCT CASE WHEN ct.status = 'completed' THEN ct.id END) as completed_tasks
                FROM campaigns c
                LEFT JOIN campaign_tasks ct ON c.id = ct.campaign_id
                WHERE c.created_at >= NOW() - INTERVAL '90 days'
                GROUP BY c.id, c.name, c.status, c.created_at
                ORDER BY c.created_at DESC
                LIMIT 50
            """)
            
            rows = cur.fetchall()
            
        campaigns = []
        for row in rows:
            campaign_id, campaign_name, status, created_at, total_tasks, completed_tasks = row
            
            # Calculate progress
            progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
            
            campaigns.append(CampaignSummary(
                id=str(campaign_id),
                name=campaign_name or "Untitled Campaign",
                status=status or "draft",
                progress=round(progress, 1),
                total_tasks=int(total_tasks or 0),
                completed_tasks=int(completed_tasks or 0),
                created_at=created_at.isoformat() if created_at else datetime.now().isoformat()
            ))
        
        return campaigns
        
    except Exception as e:
        logger.error(f"Error listing campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list campaigns: {str(e)}")

@router.get("/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(campaign_id: str):
    """
    Get detailed information about a campaign with proper transaction handling
    """
    conn = None
    try:
        # Get a fresh connection to avoid transaction issues
        conn = db_config.get_db_connection()
        cur = conn.cursor()
        
        # Ensure clean transaction state
        conn.rollback()
        
        # Get campaign details and blog posts in one query
        cur.execute("""
            SELECT 
                COALESCE(c.name, 'Unnamed Campaign') as name,
                c.created_at,
                c.status,
                COUNT(DISTINCT bp.id) as blog_count
            FROM campaigns c
            LEFT JOIN briefings b ON c.id = b.campaign_id
            LEFT JOIN blog_posts bp ON c.id = bp.campaign_id
            WHERE c.id = %s
            GROUP BY c.id, c.name, c.created_at, c.status
        """, (campaign_id,))
        
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        name, created_at, campaign_status, blog_count = row
        
        # Try to get strategy, but don't fail if table doesn't exist
        strategy = {}
        try:
            cur.execute("""
                SELECT narrative_approach, hooks, themes, tone_by_channel, key_phrases, notes
                FROM content_strategies
                WHERE campaign_id = %s
            """, (campaign_id,))
            
            strategy_row = cur.fetchone()
            if strategy_row:
                strategy = {
                    "narrative_approach": strategy_row[0],
                    "hooks": strategy_row[1],
                    "themes": strategy_row[2],
                    "tone_by_channel": strategy_row[3],
                    "key_phrases": strategy_row[4],
                    "notes": strategy_row[5]
                }
        except Exception as e:
            logger.debug(f"Could not fetch strategy: {e}")
        
        # Get tasks and blog posts
        tasks = []
        
        # Get campaign tasks
        cur.execute("""
            SELECT id, task_type, status, result, error, created_at
            FROM campaign_tasks
            WHERE campaign_id = %s
            ORDER BY created_at
        """, (campaign_id,))
        
        task_rows = cur.fetchall()
        for task_row in task_rows:
            task_id, task_type, status, result, error, created_at = task_row
            
            # Extract details from result JSON if available
            task_details = {}
            if result:
                try:
                    task_details = json.loads(result) if isinstance(result, str) else result
                except:
                    pass
            
            # Use details from result or fallback to defaults
            title = task_details.get('title', task_type.replace("_", " ").title())
            channel = task_details.get('channel', '')
            content_type = task_details.get('content_type', task_type)
            assigned_agent = task_details.get('assigned_agent', 'ContentAgent')
            
            tasks.append({
                "id": str(task_id),
                "task_type": task_type,
                "status": status or "pending",
                "result": result,
                "error": error,
                "title": title,
                "channel": channel,
                "content_type": content_type,
                "assigned_agent": assigned_agent,
                "created_at": created_at.isoformat() if created_at else None
            })
        
        # Add blog posts as completed content tasks
        if blog_count > 0:
            cur.execute("""
                SELECT id, title, status, created_at
                FROM blog_posts
                WHERE campaign_id = %s
                ORDER BY created_at
            """, (campaign_id,))
            
            blog_rows = cur.fetchall()
            for blog_row in blog_rows:
                blog_id, title, blog_status, created_at = blog_row
                tasks.append({
                    "id": str(blog_id),
                    "task_type": "blog_content",
                    "status": "completed" if blog_status in ['published', 'draft'] else blog_status,
                    "result": {"title": title, "type": "blog_post"},
                    "error": None,
                    "title": title,
                    "channel": "blog",
                    "content_type": "blog_post", 
                    "assigned_agent": "WriterAgent",
                    "created_at": created_at.isoformat() if created_at else None
                })
        
        # Calculate status
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t['status'] in ['completed', 'published']])
        
        if total_tasks == 0:
            status = "draft"
        elif completed_tasks == total_tasks:
            status = "completed"  
        else:
            status = "active"
        
        # Commit to ensure clean state
        conn.commit()
        
        return CampaignDetail(
            id=campaign_id,
            name=name or "Untitled Campaign",
            status=status,
            strategy=strategy,
            timeline=[],
            tasks=tasks,
            scheduled_posts=[],
            performance={"views": 0, "clicks": 0, "engagement_rate": 0.0}
        )
        
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error getting campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get campaign: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.post("/{campaign_id}/status", response_model=Dict[str, Any])
async def update_campaign_status(campaign_id: str, status: str):
    """Update campaign status"""
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE campaigns 
                SET status = %s, updated_at = NOW() 
                WHERE id = %s
            """, (status, campaign_id))
            
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Campaign not found")
            
            conn.commit()
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": status,
            "message": f"Campaign status updated to {status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating campaign status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update campaign status: {str(e)}")

# Testing endpoints
@router.get("/simple-test")
async def simple_test():
    """Simple test endpoint"""
    return {"message": "Hello World"}

@router.get("/test-campaign/{campaign_id}")
async def test_campaign_minimal(campaign_id: str):
    """Minimal campaign test endpoint"""
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
            "task_count": len(tasks) if tasks else 0
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/test/{template_id}", response_model=Dict[str, Any])
async def test_quick_campaign(template_id: str, blog_id: str = Query(...), campaign_name: str = Query(...)):
    """Test endpoint for debugging quick campaign creation"""
    try:
        return {
            "template_id": template_id,
            "blog_id": blog_id,
            "campaign_name": campaign_name,
            "message": "Test successful"
        }
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")