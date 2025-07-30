#!/usr/bin/env python3

import os
import requests
import json

def test_workflow_with_debug_enabled():
    """Test workflow with debug mode to see the real error"""
    
    print("🔍 Testing workflow with DEBUG environment variable...")
    
    # Set DEBUG environment variable
    original_debug = os.environ.get('DEBUG')
    os.environ['DEBUG'] = 'true'
    
    test_data = {
        "title": "Debug Mode Test",
        "company_context": "Debug company",
        "mode": "quick"
    }
    
    print(f"📝 Set DEBUG=true")
    print(f"Request: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/workflow/start",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=10
        )
        
        print(f"\n📡 Response status: {response.status_code}")
        
        if response.status_code == 500:
            try:
                error_data = response.json()
                print(f"🔍 DEBUG ERROR RESPONSE:")
                print(json.dumps(error_data, indent=2))
                
                # Look for the traceback in details
                details = error_data.get('details', {})
                if 'traceback' in details:
                    print(f"\n📄 FULL TRACEBACK:")
                    print(details['traceback'])
                    
                # Look for the actual error message
                message = error_data.get('message', 'No message')
                print(f"\n🎯 ACTUAL ERROR MESSAGE: {message}")
                
            except json.JSONDecodeError:
                print(f"📄 Raw response: {response.text}")
                
        elif response.status_code == 200:
            result = response.json()
            print(f"✅ Unexpected success (debug mode worked?):")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        
    finally:
        # Restore original DEBUG setting
        if original_debug is None:
            os.environ.pop('DEBUG', None)
        else:
            os.environ['DEBUG'] = original_debug
        print(f"📝 Restored DEBUG setting")

def check_current_debug_setting():
    """Check what the current debug setting is"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        print(f"🔍 Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            debug_status = data.get('debug', 'unknown')
            print(f"📊 Current debug mode: {debug_status}")
        else:
            print(f"❌ Health check failed")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🧪 Testing with DEBUG mode to reveal the real error")
    print("="*60)
    
    print("📊 Current server debug status:")
    check_current_debug_setting()
    
    print("\n" + "="*60)
    print("🧪 Testing workflow with DEBUG=true:")
    test_workflow_with_debug_enabled()
    
    print("\n💡 Note: The DEBUG environment variable only takes effect")
    print("   when the server is restarted. This test shows what")
    print("   the error would look like with debug enabled.")