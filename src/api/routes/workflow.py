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
import asyncio
from enum import Enum

# Temporarily comment out agent imports to debug the issue
# from ...agents.workflow.structured_blog_workflow import BlogWorkflow
# from ...agents.core.agent_factory import create_agent, AgentType
# from ...agents.specialized.planner_agent import PlannerAgent
# from ...agents.specialized.researcher_agent import ResearcherAgent
# from ...agents.specialized.writer_agent import WriterAgent
# from ...agents.specialized.editor_agent import EditorAgent
# from ...core.exceptions import AgentExecutionError, WorkflowExecutionError

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
    mode: str = "advanced"
    outline: Optional[List[str]] = None
    research: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    editor_feedback: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

# In-memory storage for workflow states (in production, use database)
workflow_states: Dict[str, WorkflowState] = {}

async def auto_execute_workflow(workflow_state: WorkflowState) -> dict:
    """
    Execute the complete workflow from current step to completion.
    """
    max_steps = 10  # Safety limit to prevent infinite loops
    executed_steps = 0
    
    try:
        while executed_steps < max_steps:
            current_step = workflow_state.current_step
            mode = getattr(workflow_state, 'mode', 'advanced')
            
            logger.info(f"Executing step {executed_steps + 1}: {current_step} for workflow {workflow_state.workflow_id}")
            
            # Store the step before execution to detect changes
            step_before = current_step
            
            if current_step == WorkflowStep.PLANNER:
                await execute_planner_logic(workflow_state)
            elif current_step == WorkflowStep.RESEARCHER:
                await execute_researcher_logic(workflow_state)
            elif current_step == WorkflowStep.WRITER:
                await execute_writer_logic(workflow_state)
            elif current_step == WorkflowStep.EDITOR:
                await execute_editor_logic(workflow_state)
            else:
                # No more steps to execute
                logger.info(f"Workflow completed at step: {current_step}")
                break
            
            executed_steps += 1
            
            # Check if we've reached completion or no progress
            if workflow_state.progress >= 100 or workflow_state.current_step == step_before:
                logger.info(f"Workflow completed with progress: {workflow_state.progress}%")
                break
        
        if executed_steps >= max_steps:
            logger.warning(f"Workflow reached maximum steps limit: {max_steps}")
            
        return workflow_state_to_dict(workflow_state)
            
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        workflow_state.status = WorkflowStatus.FAILED
        workflow_state.updated_at = datetime.datetime.utcnow()
        return workflow_state_to_dict(workflow_state)

def workflow_state_to_dict(state: WorkflowState) -> dict:
    """Convert WorkflowState to dictionary for JSON response."""
    
    # Debug logging
    logger.info(f"🔍 Converting state to dict:")
    logger.info(f"  - state.current_step: {state.current_step} (type: {type(state.current_step)})")
    logger.info(f"  - state.current_step.value: {state.current_step.value}")
    logger.info(f"  - state.mode: {state.mode}")
    logger.info(f"  - hasattr(state, 'mode'): {hasattr(state, 'mode')}")
    
    result = {
        "workflow_id": state.workflow_id,
        "current_step": state.current_step.value,
        "progress": state.progress,
        "status": state.status.value,
        "blog_title": state.blog_title,
        "company_context": state.company_context,
        "content_type": getattr(state, 'content_type', 'blog'),
        "outline": getattr(state, 'outline', None),
        "research": getattr(state, 'research', None),
        "content": getattr(state, 'content', None),
        "editor_feedback": getattr(state, 'editor_feedback', None),
        "mode": getattr(state, 'mode', 'advanced'),
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat()
    }
    logger.info(f"📊 Final result: step={result['current_step']}, mode={result['mode']}, progress={result['progress']}%")
    return result

async def execute_planner_logic(workflow_state: WorkflowState):
    """Execute planner step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock planner execution
    workflow_state.outline = [
        "Introduction",
        "Section 1: Basic concepts", 
        "Section 2: Practical implementation",
        "Section 3: Best practices",
        "Conclusion"
    ]
    
    # Determine next step based on mode
    mode = getattr(workflow_state, 'mode', 'advanced')
    if mode == 'quick':
        workflow_state.current_step = WorkflowStep.WRITER
        workflow_state.progress = 50
    elif mode == 'template':
        workflow_state.current_step = WorkflowStep.WRITER
        workflow_state.progress = 25
    else:  # advanced
        workflow_state.current_step = WorkflowStep.RESEARCHER
        workflow_state.progress = 15
    
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    logger.info(f"Planner completed, moving to: {workflow_state.current_step}")

async def execute_researcher_logic(workflow_state: WorkflowState):
    """Execute researcher step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock research execution
    workflow_state.research = {
        "introduction": "Research information for the introduction",
        "section_1": "Data and statistics for section 1",
        "section_2": "Practical examples for section 2", 
        "section_3": "Documented best practices",
        "conclusion": "Summary of key points"
    }
    
    workflow_state.current_step = WorkflowStep.WRITER
    workflow_state.progress = 40
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    logger.info(f"Research completed, moving to writer")

async def execute_writer_logic(workflow_state: WorkflowState):
    """Execute writer step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock content generation
    content = f"""# {workflow_state.blog_title}

## Introduction
This is a comprehensive guide about {workflow_state.blog_title.lower()}.

## Section 1: Basic Concepts
Here we cover the fundamental concepts you need to understand.

## Section 2: Practical Implementation  
This section provides hands-on examples and implementation details.

## Section 3: Best Practices
Learn the industry best practices and recommendations.

## Conclusion
Summary of key takeaways and next steps.
"""
    
    workflow_state.content = content
    workflow_state.current_step = WorkflowStep.EDITOR
    
    mode = getattr(workflow_state, 'mode', 'advanced')
    if mode == 'quick':
        workflow_state.progress = 75
    elif mode == 'template':
        workflow_state.progress = 60
    else:  # advanced
        workflow_state.progress = 60
        
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    logger.info(f"Content generated, moving to editor")

async def execute_editor_logic(workflow_state: WorkflowState):
    """Execute editor step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock editor review
    workflow_state.editor_feedback = {
        "score": 85,
        "strengths": ["Clear structure", "Relevant content", "Good organization"],
        "weaknesses": ["Could include more examples", "Some sections need more detail"],
        "specific_issues": ["Lack of statistics", "Limited examples"],
        "recommendations": ["Add more practical examples", "Include relevant statistics"],
        "approval_recommendation": "approve",
        "revision_priority": "medium"
    }
    
    # Determine final progress based on mode
    mode = getattr(workflow_state, 'mode', 'advanced')
    if mode == 'quick':
        workflow_state.progress = 100
        workflow_state.status = WorkflowStatus.COMPLETED
    elif mode == 'template':
        # Move to SEO for template mode
        workflow_state.current_step = WorkflowStep.SEO if hasattr(WorkflowStep, 'SEO') else workflow_state.current_step
        workflow_state.progress = 80
        workflow_state.status = WorkflowStatus.COMPLETED
    else:  # advanced - continue to image generation
        workflow_state.progress = 85
        workflow_state.status = WorkflowStatus.COMPLETED
        
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    logger.info(f"Editor completed, final progress: {workflow_state.progress}%")

async def background_workflow_execution(workflow_id: str):
    """Background task to execute workflow steps."""
    try:
        logger.info(f"🔄 Background execution starting for workflow {workflow_id}")
        
        # Add a small delay to ensure the API response is sent first
        await asyncio.sleep(1)
        
        if workflow_id not in workflow_states:
            logger.error(f"❌ Workflow {workflow_id} not found in states")
            return
        
        workflow_state = workflow_states[workflow_id]
        logger.info(f"📊 Initial state: step={workflow_state.current_step}, progress={workflow_state.progress}")
        
        # Mark as in progress
        workflow_state.status = WorkflowStatus.IN_PROGRESS
        workflow_state.updated_at = datetime.datetime.utcnow()
        logger.info(f"📝 Marked workflow {workflow_id} as in-progress")
        
        # Execute the workflow
        result = await auto_execute_workflow(workflow_state)
        logger.info(f"✅ Background execution completed for {workflow_id}: progress={result.get('progress')}%")
        
        # Update the stored state with the final result
        workflow_states[workflow_id] = workflow_state
        logger.info(f"💾 Updated stored state for {workflow_id}")
        
    except Exception as e:
        logger.error(f"❌ Background execution failed for {workflow_id}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Mark workflow as failed
        if workflow_id in workflow_states:
            workflow_states[workflow_id].status = WorkflowStatus.FAILED
            workflow_states[workflow_id].updated_at = datetime.datetime.utcnow()

@router.post("/workflow/start")
async def start_workflow(request: dict):
    """
    Start a new workflow with the given title and company context.
    TEMPORARY FIX: Execute workflow synchronously to bypass background task issues.
    """
    try:
        logger.info(f"🚀 Starting workflow with request: {request}")
        
        workflow_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()
        
        # Get mode and determine starting step
        mode = request.get("mode", "advanced")
        logger.info(f"🎯 Mode: {mode}")
        
        if mode == "quick":
            current_step = WorkflowStep.WRITER
            logger.info(f"📝 Quick mode - starting with WRITER step")
        elif mode == "template":
            current_step = WorkflowStep.PLANNER
            logger.info(f"📋 Template mode - starting with PLANNER step")
        else:  # advanced
            current_step = WorkflowStep.PLANNER
            logger.info(f"🎯 Advanced mode - starting with PLANNER step")
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_step=current_step,
            progress=0,
            status=WorkflowStatus.PENDING,
            blog_title=request.get("title", "Unknown"),
            company_context=request.get("company_context", "Unknown"),
            content_type=request.get("content_type", "blog"),
            mode=mode,
            created_at=now,
            updated_at=now
        )
        
        logger.info(f"🔧 Created WorkflowState: step={workflow_state.current_step}, mode={workflow_state.mode}")
        
        # Store in memory (TODO: persist to database)
        workflow_states[workflow_id] = workflow_state
        logger.info(f"✅ Workflow {workflow_id} stored in memory")
        
        # TEMPORARY FIX: Execute workflow immediately and synchronously
        logger.info(f"⚡ Executing workflow synchronously as temporary fix")
        try:
            result = await auto_execute_workflow(workflow_state)
            logger.info(f"✅ Synchronous execution completed: progress={result.get('progress')}%")
            return result
        except Exception as exec_error:
            logger.error(f"❌ Synchronous execution failed: {str(exec_error)}")
            # Return initial state if execution fails
            return workflow_state_to_dict(workflow_state)
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        # TODO: Implement actual planner agent execution
        # For now, create a mock outline
        workflow_state.outline = [
            "Introduction",
            "Section 1: Basic concepts",
            "Section 2: Practical implementation",
            "Section 3: Best practices",
            "Conclusion"
        ]
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
        
        # TODO: Implement actual researcher agent execution
        # For now, create mock research data
        workflow_state.research = {
            "introduction": "Research information for the introduction",
            "section_1": "Data and statistics for section 1",
            "section_2": "Practical examples for section 2",
            "section_3": "Documented best practices",
            "conclusion": "Summary of key points"
        }
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
        
        # FIXED: Use mock content generation instead of agent calls
        # TODO: Implement actual writer agent execution when agents are available
        
        # Generate mock content based on outline and research
        content = f"""# {workflow_state.blog_title}

## Introduction
This is a comprehensive guide about {workflow_state.blog_title.lower()}.

## Section 1: Basic Concepts
Here we cover the fundamental concepts you need to understand.

## Section 2: Practical Implementation  
This section provides hands-on examples and implementation details.

## Section 3: Best Practices
Learn the industry best practices and recommendations.

## Conclusion
Summary of key takeaways and next steps.
"""
        
        # Update workflow state with content
        workflow_state.content = content
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
        
        # TODO: Implement actual editor agent execution
        # For now, create mock editor feedback
        workflow_state.editor_feedback = {
            "score": 85,
            "strengths": ["Clear structure", "Relevant content", "Good organization"],
            "weaknesses": ["Could include more examples", "Some sections need more detail"],
            "specific_issues": ["Lack of statistics", "Limited examples"],
            "recommendations": ["Add more practical examples", "Include relevant statistics"],
            "approval_recommendation": "approve",
            "revision_priority": "medium"
        }
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

@router.get("/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get the current status of a workflow.
    """
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_state = workflow_states[workflow_id]
    return workflow_state_to_dict(workflow_state)

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

@router.post("/workflow/test-execution")
async def test_workflow_execution(request: dict):
    """
    Test endpoint to verify auto-execution works.
    """
    try:
        logger.info(f"🧪 Test execution endpoint called with: {request}")
        
        workflow_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()
        mode = request.get("mode", "quick")  # Use quick mode for faster testing
        
        # Create a test workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_step=WorkflowStep.WRITER if mode == "quick" else WorkflowStep.PLANNER,
            progress=0,
            status=WorkflowStatus.PENDING,
            blog_title=request.get("title", "Test Blog"),
            company_context=request.get("company_context", "Test Company"),
            content_type="blog",
            mode=mode,
            created_at=now,
            updated_at=now
        )
        
        # Store it
        workflow_states[workflow_id] = workflow_state
        logger.info(f"🧪 Created test workflow {workflow_id}")
        
        # Execute immediately
        result = await auto_execute_workflow(workflow_state)
        logger.info(f"🧪 Test execution result: {result}")
        
        return {
            "test_status": "success",
            "workflow_id": workflow_id,
            "result": result,
            "message": "Auto-execution test completed"
        }
        
    except Exception as e:
        logger.error(f"🧪 Test execution failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "test_status": "failed",
            "error": str(e),
            "message": "Auto-execution test failed"
        } 