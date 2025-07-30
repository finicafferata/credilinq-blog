#!/usr/bin/env python3

import requests
import json

def test_execution_endpoint():
    """Test the new test-execution endpoint"""
    
    test_data = {
        "title": "Test Auto Execution",
        "company_context": "Test company for auto execution",
        "mode": "quick"
    }
    
    print("🧪 Testing auto-execution endpoint...")
    print(f"Request: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/workflow/test-execution",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"\n📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Test completed!")
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            
            # Check if execution was successful
            if result.get('test_status') == 'success':
                workflow_result = result.get('result', {})
                progress = workflow_result.get('progress', 0)
                current_step = workflow_result.get('current_step', 'unknown')
                
                print(f"\n🎯 Execution Details:")
                print(f"   Progress: {progress}%")
                print(f"   Current Step: {current_step}")
                print(f"   Success: {'✅' if progress > 0 else '❌'}")
                
                return progress > 0
            else:
                print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
                return False
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_execution_endpoint()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")