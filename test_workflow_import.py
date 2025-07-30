#!/usr/bin/env python3

import sys
import traceback

def test_workflow_import():
    """Test importing the workflow module to find the issue"""
    
    print("🧪 Testing workflow module import...")
    
    try:
        # Add the project root to sys.path
        sys.path.insert(0, '.')
        
        print("📝 Step 1: Basic imports...")
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel
        from typing import Dict, Any, Optional, List
        import uuid
        import datetime
        import logging
        from enum import Enum
        print("   ✅ Basic imports successful")
        
        print("📝 Step 2: Importing workflow module...")
        from src.api.routes import workflow
        print("   ✅ Workflow module imported successfully")
        
        print("📝 Step 3: Checking workflow classes...")
        print(f"   WorkflowStep: {workflow.WorkflowStep}")
        print(f"   WorkflowStatus: {workflow.WorkflowStatus}")
        print(f"   WorkflowState: {workflow.WorkflowState}")
        print("   ✅ Workflow classes accessible")
        
        print("📝 Step 4: Testing WorkflowState creation...")
        test_state = workflow.WorkflowState(
            workflow_id="test-123",
            current_step=workflow.WorkflowStep.PLANNER,
            progress=0,
            status=workflow.WorkflowStatus.PENDING,
            blog_title="Test Blog",
            company_context="Test Company",
            content_type="blog",
            mode="advanced",
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        print("   ✅ WorkflowState creation successful")
        
        print("📝 Step 5: Testing workflow_state_to_dict function...")
        result_dict = workflow.workflow_state_to_dict(test_state)
        print(f"   ✅ Dict conversion successful: {type(result_dict)}")
        
        print("📝 Step 6: Testing auto_execute_workflow function exists...")
        print(f"   auto_execute_workflow: {workflow.auto_execute_workflow}")
        print(f"   Type: {type(workflow.auto_execute_workflow)}")
        print("   ✅ Function exists and is callable")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print(f"📄 Full traceback:")
        traceback.print_exc()
        return False

def test_workflow_functions():
    """Test individual workflow functions"""
    
    print("\n🧪 Testing workflow functions...")
    
    try:
        sys.path.insert(0, '.')
        from src.api.routes import workflow
        import datetime
        
        # Create a test workflow state
        test_state = workflow.WorkflowState(
            workflow_id="test-456",
            current_step=workflow.WorkflowStep.PLANNER,
            progress=0,
            status=workflow.WorkflowStatus.PENDING,
            blog_title="Function Test Blog",
            company_context="Function Test Company",
            content_type="blog",
            mode="advanced",
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        
        print("📝 Testing individual step functions...")
        
        # Test execute_planner_logic
        print("   Testing execute_planner_logic...")
        # Note: This is async, so we'd need asyncio to test it properly
        planner_func = workflow.execute_planner_logic
        print(f"   execute_planner_logic type: {type(planner_func)}")
        print("   ✅ Planner function accessible")
        
        # Test workflow_state_to_dict
        print("   Testing workflow_state_to_dict...")
        dict_result = workflow.workflow_state_to_dict(test_state)
        print(f"   Result type: {type(dict_result)}")
        print(f"   Keys: {list(dict_result.keys())}")
        print("   ✅ Dict conversion successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debugging workflow module import and function issues")
    print("="*60)
    
    import_success = test_workflow_import()
    
    if import_success:
        function_success = test_workflow_functions()
        
        print("\n" + "="*60)
        print("📊 TEST RESULTS:")
        print(f"   Import test: {'✅ SUCCESS' if import_success else '❌ FAILED'}")
        print(f"   Function test: {'✅ SUCCESS' if function_success else '❌ FAILED'}")
        
        if import_success and function_success:
            print("\n💡 Module imports and functions work correctly!")
            print("   The error is likely in the FastAPI endpoint execution,")
            print("   not in the module itself.")
        else:
            print("\n⚠️ Found issues with the module.")
            
    else:
        print("\n❌ Cannot import workflow module - this is the root cause!")