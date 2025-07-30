#!/usr/bin/env python3

import requests
import json
import time
import sys

def test_workflow_progress():
    """Test the workflow API endpoint and check progress"""
    
    # Test data
    test_data = {
        "title": "Test Blog Post - Progress Check",
        "company_context": "We are a tech company focused on AI solutions",
        "mode": "quick"  # Use quick mode for faster testing
    }
    
    print("🚀 Testing workflow API with progress monitoring...")
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Start workflow
        print("\n📡 Starting workflow...")
        response = requests.post(
            "http://localhost:8000/api/workflow/start",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
        result = response.json()
        workflow_id = result.get('workflow_id')
        print(f"✅ Workflow started: {workflow_id}")
        print(f"Initial state: step={result.get('current_step')}, progress={result.get('progress')}%")
        
        # Monitor progress for 30 seconds
        print("\n🔄 Monitoring progress...")
        max_checks = 15  # 30 seconds with 2-second intervals
        
        for i in range(max_checks):
            time.sleep(2)
            
            # Check status
            status_response = requests.get(
                f"http://localhost:8000/api/workflow/status/{workflow_id}",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                progress = status_data.get('progress', 0)
                current_step = status_data.get('current_step')
                status = status_data.get('status')
                
                print(f"Check {i+1}: step={current_step}, progress={progress}%, status={status}")
                
                # Check if completed
                if progress >= 100 or status == 'completed':
                    print("🎉 Workflow completed successfully!")
                    print(f"Final result: {json.dumps(status_data, indent=2)}")
                    return True
                    
                if status == 'failed':
                    print("❌ Workflow failed!")
                    print(f"Final state: {json.dumps(status_data, indent=2)}")
                    return False
            else:
                print(f"❌ Failed to get status: {status_response.status_code}")
                
        print("⏰ Timeout reached, workflow may still be processing")
        return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure backend is running on localhost:8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_workflow_progress()
    sys.exit(0 if success else 1)