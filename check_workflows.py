#!/usr/bin/env python3

import requests
import json

def check_workflows():
    """Check current workflows"""
    
    try:
        response = requests.get("http://localhost:8000/api/workflow/list", timeout=5)
        if response.status_code == 200:
            data = response.json()
            workflows = data.get('workflows', [])
            
            print(f"📊 Found {len(workflows)} workflows:")
            
            for i, workflow in enumerate(workflows):
                print(f"\n{i+1}. {workflow.get('workflow_id', 'Unknown ID')}")
                print(f"   Title: {workflow.get('blog_title', 'N/A')}")
                print(f"   Step: {workflow.get('current_step', 'N/A')}")
                print(f"   Progress: {workflow.get('progress', 0)}%")
                print(f"   Status: {workflow.get('status', 'N/A')}")
                
                # Check the latest Debug Mode Test workflows
                if "Debug" in workflow.get('blog_title', ''):
                    workflow_id = workflow.get('workflow_id')
                    print(f"   🔍 Checking detailed status for {workflow_id}...")
                    
                    status_resp = requests.get(f"http://localhost:8000/api/workflow/status/{workflow_id}")
                    if status_resp.status_code == 200:
                        detail = status_resp.json()
                        print(f"   📄 Detailed: step={detail.get('current_step')}, mode={detail.get('mode')}, progress={detail.get('progress')}%")
                    
        else:
            print(f"❌ Failed to get workflows: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_workflows()