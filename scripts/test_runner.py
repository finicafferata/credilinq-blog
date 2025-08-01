#!/usr/bin/env python3
"""
Test runner script for CrediLinQ Content Agent.
Runs both backend (Python) and frontend (TypeScript) tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def run_backend_tests(test_type=None, coverage=False):
    """Run Python backend tests."""
    print("\n🔧 Running Backend Tests...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", "-v"]
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if test_type:
        cmd.extend(["-m", test_type])
    
    cmd.append("tests/")
    
    return run_command(cmd, cwd=project_root)


def run_frontend_tests(coverage=False):
    """Run TypeScript frontend tests."""
    print("\n⚛️  Running Frontend Tests...")
    
    frontend_path = Path(__file__).parent.parent / "frontend"
    
    # Build vitest command
    cmd = ["npm", "run", "test"]
    
    if coverage:
        cmd = ["npm", "run", "test:coverage"]
    
    return run_command(cmd, cwd=frontend_path)


def run_linting():
    """Run linting for both backend and frontend."""
    print("\n🔍 Running Linting...")
    
    project_root = Path(__file__).parent.parent
    frontend_path = project_root / "frontend"
    
    # Backend linting (if tools are available)
    backend_success = True
    try:
        # Try to run flake8 if available
        backend_success = run_command(["python", "-m", "flake8", "src/", "--max-line-length=88"], cwd=project_root)
    except:
        print("flake8 not available, skipping backend linting")
    
    # Frontend linting
    frontend_success = run_command(["npm", "run", "lint"], cwd=frontend_path)
    
    return backend_success and frontend_success


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for CrediLinQ Content Agent")
    parser.add_argument("--backend-only", action="store_true", help="Run only backend tests")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage reports")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--all", action="store_true", help="Run all tests and linting")
    
    args = parser.parse_args()
    
    results = []
    
    if args.lint or args.all:
        results.append(("Linting", run_linting()))
    
    if args.all or (not args.frontend_only and not args.lint):
        # Determine test type for backend
        test_type = None
        if args.unit:
            test_type = "unit"
        elif args.integration:
            test_type = "integration"
        
        results.append(("Backend Tests", run_backend_tests(test_type, args.coverage)))
    
    if args.all or args.frontend_only or (not args.backend_only and not args.lint):
        results.append(("Frontend Tests", run_frontend_tests(args.coverage)))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()