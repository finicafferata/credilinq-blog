#!/usr/bin/env python3

import requests
import json

def test_workflow_with_debug():
    """Test workflow and capture the exact error"""
    
    test_data = {
        "title": "Debug TypeError Test",
        "company_context": "Debug company",
        "mode": "quick"  # Start with simple mode
    }
    
    print("🔍 Testing workflow to capture TypeError...")
    print(f"Request: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/workflow/start",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=10
        )
        
        print(f"\n📡 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        if response.status_code == 500:
            try:
                error_data = response.json()
                print(f"🔍 Error response:")
                print(json.dumps(error_data, indent=2))
                
                # Check if there's more detail in the response
                details = error_data.get('details', {})
                if details:
                    print(f"🔍 Error details:")
                    print(json.dumps(details, indent=2))
                    
            except json.JSONDecodeError:
                print(f"📄 Raw response text: {response.text}")
                
        elif response.status_code == 200:
            result = response.json()
            print(f"✅ Unexpected success:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
            
        return response.status_code == 500
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_workflow_list():
    """Test if workflow list endpoint works"""
    try:
        response = requests.get("http://localhost:8000/api/workflow/list", timeout=5)
        print(f"\n📊 Workflow list status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get('workflows', []))
            print(f"📋 Existing workflows: {count}")
            return True
        else:
            print(f"❌ List failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ List error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Debugging 'dict' object is not callable error")
    print("="*60)
    
    # First check if basic endpoints work
    list_works = test_workflow_list()
    
    # Then test the problematic endpoint
    error_reproduced = test_workflow_with_debug()
    
    print("\n" + "="*60)
    print("📊 DEBUG RESULTS:")
    print(f"   List endpoint works: {'✅' if list_works else '❌'}")
    print(f"   Error reproduced: {'✅' if error_reproduced else '❌'}")
    
    if error_reproduced:
        print("\n💡 Next steps:")
        print("   1. Check server logs for full traceback")
        print("   2. The error is in the /api/workflow/start endpoint")
        print("   3. Likely in auto_execute_workflow or one of its sub-functions")
    else:
        print("\n🤔 Error not reproduced - might be intermittent")