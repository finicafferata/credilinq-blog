#!/usr/bin/env python3

import requests
import json

def debug_mode_test():
    """Debug the mode selection issue"""
    
    test_data = {
        "title": "Debug Mode Test",
        "company_context": "Debug company",
        "mode": "quick"
    }
    
    print("🔍 Testing mode selection...")
    print(f"Sending: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/workflow/start",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Response received:")
            print(f"Mode in response: {result.get('mode')}")
            print(f"Current step: {result.get('current_step')}")
            print(f"Expected step for quick mode: writer")
            print(f"Match: {'✅' if result.get('current_step') == 'writer' else '❌'}")
            
            # Also check if we get to writer in the background
            import time
            print("\n🔄 Checking after 3 seconds...")
            time.sleep(3)
            
            workflow_id = result.get('workflow_id')
            status_response = requests.get(f"http://localhost:8000/api/workflow/status/{workflow_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"After 3s - step: {status_data.get('current_step')}, progress: {status_data.get('progress')}%, status: {status_data.get('status')}")
            
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_mode_test()