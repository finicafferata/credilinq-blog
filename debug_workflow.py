#!/usr/bin/env python3

import sys
import traceback
import asyncio
sys.path.append('.')

async def debug_workflow():
    """Debug the workflow creation and execution"""
    try:
        from src.api.routes.workflow import (
            WorkflowState, WorkflowStep, WorkflowStatus, 
            workflow_states, auto_execute_workflow
        )
        import datetime
        import uuid
        
        print("🔍 Creating test workflow state...")
        
        # Create a test workflow state exactly like the API does
        workflow_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()
        mode = "quick"
        current_step = WorkflowStep.WRITER if mode == "quick" else WorkflowStep.PLANNER
        
        print(f"Mode: {mode}")
        print(f"Current step: {current_step}")
        print(f"Current step type: {type(current_step)}")
        
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_step=current_step,
            progress=0,
            status=WorkflowStatus.PENDING,
            blog_title="Debug Test",
            company_context="Debug company",
            content_type="blog",
            mode=mode,
            created_at=now,
            updated_at=now
        )
        
        print("✅ Workflow state created successfully")
        print(f"State: {workflow_state}")
        
        # Store in memory
        workflow_states[workflow_id] = workflow_state
        print("✅ Workflow state stored")
        
        # Try auto-execution
        print("🚀 Starting auto-execution...")
        result = await auto_execute_workflow(workflow_state)
        print("✅ Auto-execution completed!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(debug_workflow())