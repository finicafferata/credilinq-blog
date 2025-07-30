#!/usr/bin/env python3

import sys
import json
import datetime
import uuid

def test_database_save():
    """Test database save function directly"""
    
    print("🧪 Testing database save function...")
    
    try:
        # Add project to path
        sys.path.insert(0, '.')
        
        # Import the database config
        from src.config.database import db_config
        
        print("📝 Step 1: Testing database connection...")
        health = db_config.health_check()
        print(f"   Database health: {health}")
        
        if not health.get('status') == 'healthy':
            print("❌ Database is not healthy, cannot test save")
            return False
        
        print("📝 Step 2: Testing direct database insert...")
        
        # Create test data
        blog_id = str(uuid.uuid4())
        title = "Database Save Test Blog"
        content_markdown = "# Test Content\n\nThis is a test blog post."
        initial_prompt = {
            "title": title,
            "company_context": "Test company",
            "content_type": "blog",
            "mode": "test",
            "workflow_type": "database_test"
        }
        created_at = datetime.datetime.utcnow().isoformat()
        updated_at = created_at
        
        print(f"   Blog ID: {blog_id}")
        print(f"   Title: {title}")
        
        # Try to save to database
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO "BlogPost" (id, title, "contentMarkdown", "initialPrompt", status, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                blog_id,
                title,
                content_markdown,
                json.dumps(initial_prompt),
                "draft",
                created_at,
                updated_at
            ))
            
            print("   ✅ Database insert successful")
            
        print("📝 Step 3: Verifying the saved blog...")
        
        # Query to verify it was saved
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT id, title, status FROM "BlogPost" WHERE id = %s', (blog_id,))
            result = cur.fetchone()
            
            if result:
                print(f"   ✅ Blog found in database: {result}")
                return True
            else:
                print(f"   ❌ Blog not found in database")
                return False
                
    except Exception as e:
        print(f"❌ Database save test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_save_function():
    """Test the actual workflow save function"""
    
    print("\n🧪 Testing workflow save function...")
    
    try:
        sys.path.insert(0, '.')
        
        # Import the save function
        from src.api.routes.workflow_fixed import save_workflow_to_database, WorkflowState, WorkflowStep, WorkflowStatus
        
        print("📝 Creating mock workflow state...")
        
        # Create mock workflow state
        workflow_state = WorkflowState(
            workflow_id=str(uuid.uuid4()),
            current_step=WorkflowStep.EDITOR,
            progress=100,
            status=WorkflowStatus.COMPLETED,
            blog_title="Workflow Save Test",
            company_context="Test company for save function",
            content_type="blog",
            mode="test",
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        
        # Create mock result
        result = {
            "progress": 100,
            "content": "# Workflow Save Test\n\nThis is test content from workflow save function.",
            "outline": ["Introduction", "Main Content", "Conclusion"],
            "research": {"section1": "Test research data"},
            "editor_feedback": {"score": 85, "status": "approved"}
        }
        
        print(f"   Workflow ID: {workflow_state.workflow_id}")
        print(f"   Title: {workflow_state.blog_title}")
        
        # Test the save function
        save_workflow_to_database(workflow_state, result)
        
        print("   ✅ Workflow save function completed")
        return True
        
    except Exception as e:
        print(f"❌ Workflow save function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing database save functionality")
    print("="*50)
    
    direct_save = test_database_save()
    workflow_save = test_workflow_save_function()
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print(f"   Direct database save: {'✅ SUCCESS' if direct_save else '❌ FAILED'}")
    print(f"   Workflow save function: {'✅ SUCCESS' if workflow_save else '❌ FAILED'}")
    
    if direct_save and workflow_save:
        print("\n🎉 Database save is working!")
        print("💡 The issue might be that the workflow endpoint isn't calling the save function")
    elif direct_save and not workflow_save:
        print("\n⚠️ Database works but workflow save function has issues")
    else:
        print("\n❌ Database connection or permissions issue")