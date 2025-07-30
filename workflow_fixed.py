"""
FIXED Workflow API Routes - Phase 1 Implementation  
This file contains the corrected workflow implementation with working auto-execution.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import datetime
import logging
import asyncio
from enum import Enum

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

# In-memory storage for workflow states
workflow_states_fixed: Dict[str, WorkflowState] = {}

def workflow_state_to_dict(state: WorkflowState) -> dict:
    """Convert WorkflowState to dictionary for JSON response."""
    return {
        "workflow_id": state.workflow_id,
        "current_step": state.current_step.value,
        "progress": state.progress,
        "status": state.status.value,
        "blog_title": state.blog_title,
        "company_context": state.company_context,
        "content_type": state.content_type,
        "outline": state.outline,
        "research": state.research,
        "content": state.content,
        "editor_feedback": state.editor_feedback,
        "mode": state.mode,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat()
    }

async def execute_planner_logic(workflow_state: WorkflowState):
    """Execute planner step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock planner execution with small delay
    await asyncio.sleep(0.5)
    
    workflow_state.outline = [
        "Introduction",
        "Section 1: Basic concepts", 
        "Section 2: Practical implementation",
        "Section 3: Best practices",
        "Conclusion"
    ]
    
    # Determine next step based on mode
    mode = workflow_state.mode
    if mode == 'quick':
        workflow_state.current_step = WorkflowStep.WRITER
        workflow_state.progress = 50
    elif mode == 'template':
        workflow_state.current_step = WorkflowStep.WRITER
        workflow_state.progress = 25
    else:  # advanced
        workflow_state.current_step = WorkflowStep.RESEARCHER
        workflow_state.progress = 25
    
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    logger.info(f"✅ Planner completed, moving to: {workflow_state.current_step}")

async def execute_researcher_logic(workflow_state: WorkflowState):
    """Execute researcher step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock research execution with small delay
    await asyncio.sleep(0.5)
    
    workflow_state.research = {
        "introduction": "Research information for the introduction",
        "section_1": "Data and statistics for section 1",
        "section_2": "Practical examples for section 2", 
        "section_3": "Documented best practices",
        "conclusion": "Summary of key points"
    }
    
    workflow_state.current_step = WorkflowStep.WRITER
    workflow_state.progress = 50
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    logger.info(f"✅ Research completed, moving to writer")

async def execute_writer_logic(workflow_state: WorkflowState):
    """Execute writer step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock content generation with small delay
    await asyncio.sleep(0.5)
    
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
    
    mode = workflow_state.mode
    if mode == 'quick':
        workflow_state.progress = 75
    elif mode == 'template':
        workflow_state.progress = 75
    else:  # advanced
        workflow_state.progress = 75
        
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    logger.info(f"✅ Content generated, moving to editor")

async def execute_editor_logic(workflow_state: WorkflowState):
    """Execute editor step logic."""
    workflow_state.status = WorkflowStatus.IN_PROGRESS
    workflow_state.updated_at = datetime.datetime.utcnow()
    
    # Mock editor review with small delay
    await asyncio.sleep(0.5)
    
    workflow_state.editor_feedback = {
        "score": 85,
        "strengths": ["Clear structure", "Relevant content", "Good organization"],
        "weaknesses": ["Could include more examples", "Some sections need more detail"],
        "specific_issues": ["Lack of statistics", "Limited examples"],
        "recommendations": ["Add more practical examples", "Include relevant statistics"],
        "approval_recommendation": "approve",
        "revision_priority": "medium"
    }
    
    # Final completion
    workflow_state.progress = 100
    workflow_state.status = WorkflowStatus.COMPLETED
    workflow_state.updated_at = datetime.datetime.utcnow()
    logger.info(f"✅ Editor completed, workflow finished: {workflow_state.progress}%")

async def auto_execute_workflow(workflow_state: WorkflowState) -> dict:
    """
    Execute the complete workflow from current step to completion.
    """
    max_steps = 10  # Safety limit
    executed_steps = 0
    
    try:
        logger.info(f"🚀 Starting auto-execution for workflow {workflow_state.workflow_id}")
        
        while executed_steps < max_steps:
            current_step = workflow_state.current_step
            logger.info(f"📝 Executing step {executed_steps + 1}: {current_step}")
            
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
                logger.info(f"✅ Workflow completed at step: {current_step}")
                break
            
            executed_steps += 1
            
            # Check if completed
            if workflow_state.progress >= 100:
                logger.info(f"🎉 Workflow completed with progress: {workflow_state.progress}%")
                break
                
            # Safety check for infinite loops
            if workflow_state.current_step == step_before:
                logger.warning(f"⚠️ Step didn't progress, breaking loop")
                break
        
        return workflow_state_to_dict(workflow_state)
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {str(e)}")
        workflow_state.status = WorkflowStatus.FAILED
        workflow_state.updated_at = datetime.datetime.utcnow()
        return workflow_state_to_dict(workflow_state)

# Test the auto_execute_workflow function
async def test_auto_execution():
    """Test function to verify auto_execute_workflow works correctly"""
    print("🧪 Testing auto_execute_workflow function...")
    
    # Create test workflow state
    workflow_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    
    workflow_state = WorkflowState(
        workflow_id=workflow_id,
        current_step=WorkflowStep.PLANNER,
        progress=0,
        status=WorkflowStatus.PENDING,
        blog_title="Test Auto Execution",
        company_context="Test Company",
        content_type="blog",
        mode="advanced",
        created_at=now,
        updated_at=now
    )
    
    print(f"📊 Initial state: step={workflow_state.current_step}, progress={workflow_state.progress}%")
    
    # Execute workflow
    result = await auto_execute_workflow(workflow_state)
    
    print(f"✅ Final result: step={result.get('current_step')}, progress={result.get('progress')}%")
    print(f"📄 Content length: {len(result.get('content', ''))}")
    print(f"📋 Outline: {result.get('outline')}")
    
    return result.get('progress') == 100

if __name__ == "__main__":
    import asyncio
    
    async def main():
        success = await test_auto_execution()
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    
    asyncio.run(main())