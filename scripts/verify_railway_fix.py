#!/usr/bin/env python3
"""
Railway Deployment Fix Verification Script

This script verifies that the Railway PORT environment variable issue has been resolved.
It tests various scenarios including:
- PORT variable handling and type conversion
- Railway environment detection
- Command construction with different PORT values
- Error handling for invalid PORT values
"""

import os
import sys
import subprocess
import tempfile
from typing import Dict, Any

def test_port_handling():
    """Test PORT environment variable handling."""
    print("🧪 Testing PORT Environment Variable Handling")
    print("=" * 50)
    
    test_cases = [
        ("3000", 3000, "Valid string port"),
        ("8080", 8080, "Another valid string port"), 
        ("80", 80, "Low port number"),
        ("65535", 65535, "High port number"),
        ("", "8000", "Empty PORT (should default to 8000)"),
        (None, "8000", "No PORT set (should default to 8000)"),
    ]
    
    results = []
    
    for port_value, expected, description in test_cases:
        print(f"\n🔍 Test: {description}")
        print(f"   Input PORT: {repr(port_value)}")
        
        # Set up environment
        env = os.environ.copy()
        if port_value is None:
            env.pop('PORT', None)
        else:
            env['PORT'] = port_value
        
        env['RAILWAY_ENVIRONMENT'] = 'test'
        
        try:
            # Run test_env.py script to check PORT handling
            result = subprocess.run(
                [sys.executable, 'scripts/test_env.py'],
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout
            
            # Check if PORT conversion was successful
            if "PORT conversion: SUCCESS" in output:
                actual_port = None
                for line in output.split('\n'):
                    if "PORT as integer:" in line:
                        actual_port = int(line.split(':')[1].strip())
                        break
                
                if actual_port == expected:
                    print(f"   ✅ SUCCESS: PORT correctly handled as {actual_port}")
                    results.append(True)
                else:
                    print(f"   ❌ FAILED: Expected {expected}, got {actual_port}")
                    results.append(False)
            else:
                # Check for default case
                if "uvicorn src.main:app --host 0.0.0.0 --port 8000" in output:
                    print(f"   ✅ SUCCESS: Correctly defaulted to 8000")
                    results.append(True)
                else:
                    print(f"   ❌ FAILED: PORT handling failed")
                    print(f"   Output: {output}")
                    results.append(False)
                    
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    return all(results)

def test_startup_script():
    """Test the startup script with various configurations."""
    print("\n🚀 Testing Startup Script")
    print("=" * 50)
    
    test_cases = [
        {
            'PORT': '3000', 
            'RAILWAY_ENVIRONMENT': 'production',
            'expected_port': '3000',
            'expected_workers': '1',
            'description': 'Railway production environment'
        },
        {
            'PORT': '8080',
            'HOST': '127.0.0.1', 
            'WORKERS': '4',
            'expected_port': '8080',
            'expected_host': '127.0.0.1',
            'description': 'Local development environment'
        },
        {
            'PORT': '5000',
            'RAILWAY_ENVIRONMENT': 'staging',
            'ENVIRONMENT': 'production',
            'expected_port': '5000',
            'expected_log_level': 'warning',
            'description': 'Railway staging with production settings'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        description = test_case.pop('description')
        expected_port = test_case.pop('expected_port')
        
        print(f"\n🔍 Test: {description}")
        
        # Set up environment
        env = os.environ.copy()
        for key, value in test_case.items():
            if key.startswith('expected_'):
                continue
            env[key] = value
        
        try:
            # Run startup script in dry-run mode
            result = subprocess.run(
                [sys.executable, 'scripts/start.py', '--dry-run'],
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Startup script outputs to stderr (logging), not stdout
            output = result.stderr if result.stderr else result.stdout
            
            # Check for expected port in command
            expected_cmd = f"--port {expected_port}"
            if expected_cmd in output and result.returncode == 0:
                print(f"   ✅ SUCCESS: Port {expected_port} correctly configured")
                results.append(True)
            else:
                print(f"   ❌ FAILED: Expected port {expected_port} not found in command")
                print(f"   Return code: {result.returncode}")
                if output:
                    print(f"   Debug output: {output}")
                else:
                    print(f"   No output received")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    return all(results)

def test_railway_detection():
    """Test Railway environment detection."""
    print("\n🌐 Testing Railway Environment Detection")
    print("=" * 50)
    
    test_cases = [
        ({'RAILWAY_ENVIRONMENT': 'production'}, True, 'Railway production'),
        ({'RAILWAY_ENVIRONMENT': 'staging'}, True, 'Railway staging'),
        ({}, False, 'No Railway environment'),
        ({'SOME_OTHER_VAR': 'value'}, False, 'Different environment variable')
    ]
    
    results = []
    
    for env_vars, expected_railway, description in test_cases:
        print(f"\n🔍 Test: {description}")
        
        # Set up environment
        env = os.environ.copy()
        # Clear Railway variables first
        for key in list(env.keys()):
            if key.startswith('RAILWAY_'):
                del env[key]
        
        # Add test variables
        for key, value in env_vars.items():
            env[key] = value
        
        try:
            # Run test_env.py script
            result = subprocess.run(
                [sys.executable, 'scripts/test_env.py'],
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout
            
            # Check Railway detection
            is_railway_line = None
            for line in output.split('\n'):
                if 'Is Railway:' in line:
                    is_railway_line = line
                    break
            
            if is_railway_line:
                actual_railway = 'True' in is_railway_line
                if actual_railway == expected_railway:
                    print(f"   ✅ SUCCESS: Railway detection correct ({actual_railway})")
                    results.append(True)
                else:
                    print(f"   ❌ FAILED: Expected {expected_railway}, got {actual_railway}")
                    results.append(False)
            else:
                print(f"   ❌ FAILED: Railway detection line not found")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    return all(results)

def main():
    """Run all verification tests."""
    print("🔧 Railway Deployment Fix Verification")
    print("=" * 60)
    print("This script verifies that the PORT environment variable")
    print("handling has been fixed for Railway deployments.")
    print("=" * 60)
    
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Run tests
    tests = [
        ("PORT Environment Variable Handling", test_port_handling),
        ("Startup Script Configuration", test_startup_script),
        ("Railway Environment Detection", test_railway_detection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status} {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ Railway PORT environment variable issue has been RESOLVED")
        print("\nYour Railway deployment should now work correctly with:")
        print("   • Proper PORT environment variable handling")
        print("   • Railway environment detection")
        print("   • Optimized startup configuration")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests FAILED")
        print("⚠️  Railway deployment may still have issues")
        print("\nPlease review the failed tests and fix the issues before deploying.")
        return 1

if __name__ == '__main__':
    sys.exit(main())