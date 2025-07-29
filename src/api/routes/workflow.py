"""
Workflow API Routes - Phase 1 Implementation
Handles the complete workflow from planning to content generation.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import datetime
import logging
from enum import Enum

from ...agents.workflow.structured_blog_workflow import BlogWorkflow
from ...agents.core.agent_factory import create_agent, AgentType
from ...agents.specialized.planner_agent import PlannerAgent
from ...agents.specialized.researcher_agent import ResearcherAgent
from ...agents.specialized.writer_agent import WriterAgent
from ...agents.specialized.editor_agent import EditorAgent
from ...core.exceptions import AgentExecutionError, WorkflowExecutionError

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

class WorkflowStep(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    WRITER = "writer"
    EDITOR = "editor"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowStartRequest(BaseModel):
    title: str
    company_context: str
    content_type: str = "blog"

class WorkflowStepRequest(BaseModel):
    workflow_id: str

class WorkflowState(BaseModel):
    workflow_id: str
    current_step: WorkflowStep
    progress: int
    status: WorkflowStatus
    blog_title: str
    company_context: str
    content_type: str
    outline: Optional[List[str]] = None
    research: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    editor_feedback: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

# In-memory storage for workflow states (in production, use database)
workflow_states: Dict[str, WorkflowState] = {}

@router.post("/workflow/start", response_model=WorkflowState)
async def start_workflow(request: WorkflowStartRequest):
    """
    Start a new workflow with the given title and company context.
    """
    try:
        workflow_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()
        
        # Create initial workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_step=WorkflowStep.PLANNER,
            progress=0,
            status=WorkflowStatus.PENDING,
            blog_title=request.title,
            company_context=request.company_context,
            content_type=request.content_type,
            created_at=now,
            updated_at=now
        )
        
        # Store workflow state
        workflow_states[workflow_id] = workflow_state
        
        logger.info(f"Started workflow {workflow_id} for title: {request.title}")
        
        return workflow_state
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@router.post("/workflow/planner", response_model=WorkflowState)
async def execute_planner_step(request: WorkflowStepRequest):
    """
    Execute the planner step to create an outline.
    """
    try:
        workflow_id = request.workflow_id
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_state = workflow_states[workflow_id]
        
        # Update workflow state
        workflow_state.current_step = WorkflowStep.PLANNER
        workflow_state.status = WorkflowStatus.IN_PROGRESS
        workflow_state.progress = 25
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        # Create planner agent
        planner_agent = create_agent(
            AgentType.PLANNER,
            metadata={
                "agent_type": AgentType.PLANNER,
                "name": "WorkflowPlanner",
                "description": "Creates structured outlines for workflow content"
            }
        )
        
        # Execute planner
        planner_input = {
            "blog_title": workflow_state.blog_title,
            "company_context": workflow_state.company_context,
            "content_type": workflow_state.content_type
        }
        
        result = planner_agent.execute(planner_input)
        
        if not result.success:
            raise AgentExecutionError("PlannerAgent", "execution", result.error_message)
        
        # Update workflow state with outline
        workflow_state.outline = result.data.get("outline", [])
        workflow_state.current_step = WorkflowStep.RESEARCHER
        workflow_state.status = WorkflowStatus.COMPLETED
        workflow_state.progress = 25
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        logger.info(f"Planner step completed for workflow {workflow_id}")
        
        return workflow_state
        
    except Exception as e:
        logger.error(f"Planner step failed: {str(e)}")
        workflow_state.status = WorkflowStatus.FAILED
        workflow_state.updated_at = datetime.datetime.utcnow()
        raise HTTPException(status_code=500, detail=f"Planner step failed: {str(e)}")

@router.post("/workflow/researcher", response_model=WorkflowState)
async def execute_researcher_step(request: WorkflowStepRequest):
    """
    Execute the researcher step to gather information for each section.
    """
    try:
        workflow_id = request.workflow_id
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_state = workflow_states[workflow_id]
        
        if not workflow_state.outline:
            raise HTTPException(status_code=400, detail="Outline not found. Execute planner step first.")
        
        # Update workflow state
        workflow_state.current_step = WorkflowStep.RESEARCHER
        workflow_state.status = WorkflowStatus.IN_PROGRESS
        workflow_state.progress = 50
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        # Create researcher agent
        researcher_agent = create_agent(
            AgentType.RESEARCHER,
            metadata={
                "agent_type": AgentType.RESEARCHER,
                "name": "WorkflowResearcher",
                "description": "Researches information for workflow content"
            }
        )
        
        # Execute researcher
        researcher_input = {
            "outline": workflow_state.outline,
            "blog_title": workflow_state.blog_title,
            "company_context": workflow_state.company_context
        }
        
        result = researcher_agent.execute(researcher_input)
        
        if not result.success:
            raise AgentExecutionError("ResearcherAgent", "execution", result.error_message)
        
        # Update workflow state with research
        workflow_state.research = result.data.get("research", {})
        workflow_state.current_step = WorkflowStep.WRITER
        workflow_state.status = WorkflowStatus.COMPLETED
        workflow_state.progress = 50
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        logger.info(f"Researcher step completed for workflow {workflow_id}")
        
        return workflow_state
        
    except Exception as e:
        logger.error(f"Researcher step failed: {str(e)}")
        workflow_state.status = WorkflowStatus.FAILED
        workflow_state.updated_at = datetime.datetime.utcnow()
        raise HTTPException(status_code=500, detail=f"Researcher step failed: {str(e)}")

@router.post("/workflow/writer", response_model=WorkflowState)
async def execute_writer_step(request: WorkflowStepRequest):
    """
    Execute the writer step to generate content based on research.
    """
    try:
        workflow_id = request.workflow_id
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_state = workflow_states[workflow_id]
        
        if not workflow_state.outline or not workflow_state.research:
            raise HTTPException(status_code=400, detail="Outline and research not found. Execute previous steps first.")
        
        # Update workflow state
        workflow_state.current_step = WorkflowStep.WRITER
        workflow_state.status = WorkflowStatus.IN_PROGRESS
        workflow_state.progress = 75
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        # Create writer agent
        writer_agent = create_agent(
            AgentType.WRITER,
            metadata={
                "agent_type": AgentType.WRITER,
                "name": "WorkflowWriter",
                "description": "Generates content for workflow"
            }
        )
        
        # Execute writer
        writer_input = {
            "outline": workflow_state.outline,
            "research": workflow_state.research,
            "blog_title": workflow_state.blog_title,
            "company_context": workflow_state.company_context,
            "content_type": workflow_state.content_type
        }
        
        result = writer_agent.execute(writer_input)
        
        if not result.success:
            raise AgentExecutionError("WriterAgent", "execution", result.error_message)
        
        # Update workflow state with content
        workflow_state.content = result.data.get("content", "")
        workflow_state.current_step = WorkflowStep.EDITOR
        workflow_state.status = WorkflowStatus.COMPLETED
        workflow_state.progress = 75
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        logger.info(f"Writer step completed for workflow {workflow_id}")
        
        return workflow_state
        
    except Exception as e:
        logger.error(f"Writer step failed: {str(e)}")
        workflow_state.status = WorkflowStatus.FAILED
        workflow_state.updated_at = datetime.datetime.utcnow()
        raise HTTPException(status_code=500, detail=f"Writer step failed: {str(e)}")

@router.post("/workflow/editor", response_model=WorkflowState)
async def execute_editor_step(request: WorkflowStepRequest):
    """
    Execute the editor step to review and approve content.
    """
    try:
        workflow_id = request.workflow_id
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_state = workflow_states[workflow_id]
        
        if not workflow_state.content:
            raise HTTPException(status_code=400, detail="Content not found. Execute writer step first.")
        
        # Update workflow state
        workflow_state.current_step = WorkflowStep.EDITOR
        workflow_state.status = WorkflowStatus.IN_PROGRESS
        workflow_state.progress = 100
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        # Create editor agent
        editor_agent = create_agent(
            AgentType.EDITOR,
            metadata={
                "agent_type": AgentType.EDITOR,
                "name": "WorkflowEditor",
                "description": "Reviews and approves workflow content"
            }
        )
        
        # Execute editor
        editor_input = {
            "content": workflow_state.content,
            "blog_title": workflow_state.blog_title,
            "company_context": workflow_state.company_context,
            "content_type": workflow_state.content_type
        }
        
        result = editor_agent.execute(editor_input)
        
        if not result.success:
            raise AgentExecutionError("EditorAgent", "execution", result.error_message)
        
        # Update workflow state with editor feedback
        workflow_state.editor_feedback = result.data
        workflow_state.status = WorkflowStatus.COMPLETED
        workflow_state.progress = 100
        workflow_state.updated_at = datetime.datetime.utcnow()
        
        logger.info(f"Editor step completed for workflow {workflow_id}")
        
        return workflow_state
        
    except Exception as e:
        logger.error(f"Editor step failed: {str(e)}")
        workflow_state.status = WorkflowStatus.FAILED
        workflow_state.updated_at = datetime.datetime.utcnow()
        raise HTTPException(status_code=500, detail=f"Editor step failed: {str(e)}")

@router.get("/workflow/status/{workflow_id}", response_model=WorkflowState)
async def get_workflow_status(workflow_id: str):
    """
    Get the current status of a workflow.
    """
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow_states[workflow_id]

@router.get("/workflow/list")
async def list_workflows():
    """
    List all workflows (for debugging purposes).
    """
    return {
        "workflows": [
            {
                "workflow_id": workflow_id,
                "blog_title": state.blog_title,
                "current_step": state.current_step,
                "progress": state.progress,
                "status": state.status,
                "created_at": state.created_at,
                "updated_at": state.updated_at
            }
            for workflow_id, state in workflow_states.items()
        ]
    }

@router.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow.
    """
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del workflow_states[workflow_id]
    logger.info(f"Deleted workflow {workflow_id}")
    
    return {"message": "Workflow deleted successfully"} 