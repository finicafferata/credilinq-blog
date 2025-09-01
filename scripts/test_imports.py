#!/usr/bin/env python3
"""
Test script to check which imports are failing in the main application.
This helps diagnose Railway deployment issues.
"""

import sys
import traceback
import os

def test_import(module_name, description=""):
    """Test importing a module and report results."""
    try:
        if module_name == "src.main":
            from src.main import app
            print(f"✅ {module_name} imported successfully {description}")
            return True
        else:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully {description}")
            return True
    except Exception as e:
        print(f"❌ {module_name} failed to import {description}")
        print(f"   Error: {str(e)}")
        if "ImportError" in str(type(e)) or "ModuleNotFoundError" in str(type(e)):
            print(f"   Type: Missing module or dependency")
        elif "AttributeError" in str(type(e)):
            print(f"   Type: Module attribute error")
        else:
            print(f"   Type: {type(e).__name__}")
            # Print first few lines of traceback for complex errors
            tb_lines = traceback.format_exc().split('\n')
            for line in tb_lines[1:6]:  # Skip "Traceback..." line, show next 5
                if line.strip():
                    print(f"   {line}")
        return False

def main():
    """Test all critical imports for Railway deployment."""
    print("🔍 Testing imports for Railway deployment...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    print()
    
    # Test basic imports first
    basic_imports = [
        ("fastapi", "- Web framework"),
        ("uvicorn", "- ASGI server"), 
        ("pydantic", "- Data validation"),
        ("psycopg2", "- PostgreSQL driver"),
        ("openai", "- AI API client"),
    ]
    
    print("=== Basic Dependencies ===")
    basic_success = 0
    for module, desc in basic_imports:
        if test_import(module, desc):
            basic_success += 1
    
    print(f"\nBasic imports: {basic_success}/{len(basic_imports)} successful")
    
    # Test application modules
    print("\n=== Application Modules ===")
    app_imports = [
        ("src.config.settings", "- Settings configuration"),
        ("src.config.database", "- Database configuration"),
        ("src.api.routes.health", "- Health endpoints"),
        ("src.agents.core.agent_factory", "- Agent factory"),
    ]
    
    app_success = 0
    for module, desc in app_imports:
        if test_import(module, desc):
            app_success += 1
    
    print(f"\nApplication imports: {app_success}/{len(app_imports)} successful")
    
    # Test main application
    print("\n=== Main Application ===")
    main_success = test_import("src.main", "- Full application with agents")
    
    # Test simple fallback
    print("\n=== Simple Fallback ===")
    simple_success = test_import("src.main_railway_simple", "- Simple fallback application")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Basic dependencies: {basic_success}/{len(basic_imports)}")
    print(f"Application modules: {app_success}/{len(app_imports)}")
    print(f"Main application: {'✅' if main_success else '❌'}")
    print(f"Simple fallback: {'✅' if simple_success else '❌'}")
    
    if main_success:
        print("\n🎉 Main application should work in Railway!")
    elif simple_success:
        print("\n⚠️ Only simple mode will work - use as fallback")
    else:
        print("\n💥 Both applications failed - check dependencies")
        
    return main_success

if __name__ == "__main__":
    main()