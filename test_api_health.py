#!/usr/bin/env python3

import requests

def test_api_health():
    """Test if the API is responding"""
    
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/api/health", timeout=5)
        print(f"Health endpoint: {health_response.status_code}")
        
        # Test workflow list endpoint
        list_response = requests.get("http://localhost:8000/api/workflow/list", timeout=5)
        print(f"Workflow list endpoint: {list_response.status_code}")
        if list_response.status_code == 200:
            data = list_response.json()
            print(f"Existing workflows: {len(data.get('workflows', []))}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("Make sure to run: python -m src.main")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_api_health()