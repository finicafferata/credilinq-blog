#!/usr/bin/env python3

import requests
import json
import time
import sys

def test_workflow():
    """Test the workflow API endpoint"""
    
    # Test data
    test_data = {
        "title": "Test Blog Post",
        "company_context": "We are a tech company focused on AI solutions",
        "mode": "advanced"
    }
    
    print("🚀 Testing workflow API...")
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Start workflow
        response = requests.post(
            "http://localhost:8000/api/workflow/start",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"\n📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workflow started successfully!")
            print(f"📊 Final result: {json.dumps(result, indent=2)}")
            
            # Check if workflow completed
            if result.get('progress') == 100:
                print("🎉 Workflow completed successfully!")
            else:
                print(f"⚠️  Workflow partially completed: {result.get('progress')}%")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure backend is running on localhost:8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_workflow()