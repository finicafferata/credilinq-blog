#!/usr/bin/env python3

import requests
import json

def test_fixed_workflow():
    """Test the working fixed workflow endpoint"""
    
    test_data = {
        "title": "Fixed Workflow Test - Working Version",
        "company_context": "Test company for the working workflow implementation",
        "mode": "advanced"
    }
    
    print("🧪 Testing FIXED workflow endpoint...")
    print(f"Request: {json.dumps(test_data, indent=2)}")
    
    try:
        # Test the fixed workflow endpoint
        response = requests.post(
            "http://localhost:8000/api/workflow-fixed/start",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"\n📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ FIXED workflow completed!")
            
            progress = result.get('progress', 0)
            current_step = result.get('current_step', 'unknown')
            mode = result.get('mode', 'unknown')
            content_length = len(result.get('content', ''))
            outline_length = len(result.get('outline', []))
            
            print(f"\n🎯 Results:")
            print(f"   Progress: {progress}%")
            print(f"   Final Step: {current_step}")
            print(f"   Mode: {mode}")
            print(f"   Content Generated: {content_length} characters")
            print(f"   Outline Sections: {outline_length}")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            if progress == 100:
                print(f"\n🎉 SUCCESS: Workflow completed successfully!")
                print(f"📄 Content preview: {result.get('content', '')[:200]}...")
                return True
            else:
                print(f"\n⚠️ Partial completion: {progress}%")
                return False
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure backend is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_fixed_workflow_list():
    """Test the fixed workflow list endpoint"""
    
    try:
        response = requests.get("http://localhost:8000/api/workflow-fixed/list", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            workflows = data.get('workflows', [])
            print(f"\n📊 Fixed workflows found: {len(workflows)}")
            
            for i, workflow in enumerate(workflows):
                print(f"   {i+1}. {workflow.get('blog_title', 'Unknown')} - {workflow.get('progress', 0)}%")
                
        else:
            print(f"❌ List request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ List error: {e}")

if __name__ == "__main__":
    print("🚀 Testing FIXED workflow implementation...")
    
    success = test_fixed_workflow()
    
    print("\n" + "="*60)
    test_fixed_workflow_list()
    
    print(f"\n{'✅ OVERALL SUCCESS' if success else '❌ OVERALL FAILED'}")
    print("\nIf this works, update the frontend to use /api/workflow-fixed/start instead!")