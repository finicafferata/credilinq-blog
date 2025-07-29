#!/usr/bin/env python3
"""
Test runner script for both backend and frontend tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, cwd=None, description=""):
    """Run a command and return the result."""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {description} - PASSED")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def run_backend_tests(test_type="all", verbose=False):
    """Run backend tests using pytest."""
    root_dir = Path(__file__).parent.parent
    
    command = ["python", "-m", "pytest"]
    
    if verbose:
        command.append("-v")
    
    # Add coverage
    command.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test type markers
    if test_type == "unit":
        command.extend(["-m", "unit"])
    elif test_type == "integration":
        command.extend(["-m", "integration"])
    elif test_type == "api":
        command.extend(["-m", "api"])
    elif test_type == "agent":
        command.extend(["-m", "agent"])
    elif test_type == "security":
        command.extend(["-m", "security"])
    elif test_type == "database":
        command.extend(["-m", "database"])
    elif test_type == "slow":
        command.extend(["-m", "slow"])
    
    # Add test directory
    command.append("tests/")
    
    success, output = run_command(
        command, 
        cwd=root_dir,
        description=f"Backend {test_type} tests"
    )
    
    return success, output

def run_frontend_tests(test_type="all", verbose=False):
    """Run frontend tests using Vitest."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False, ""
    
    command = ["npm", "run"]
    
    if test_type == "coverage":
        command.append("test:coverage")
    elif test_type == "ui":
        command.append("test:ui")
    else:
        command.append("test")
    
    if verbose and test_type == "test":
        command.append("--")
        command.append("--reporter=verbose")
    
    success, output = run_command(
        command,
        cwd=frontend_dir,
        description=f"Frontend {test_type} tests"
    )
    
    return success, output

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Check backend dependencies
    try:
        import pytest
        import fastapi
        print("✅ Backend dependencies found")
        backend_deps = True
    except ImportError as e:
        print(f"❌ Backend dependencies missing: {e}")
        backend_deps = False
    
    # Check frontend dependencies
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        node_modules = frontend_dir / "node_modules"
        package_json = frontend_dir / "package.json"
        
        if node_modules.exists() and package_json.exists():
            print("✅ Frontend dependencies found")
            frontend_deps = True
        else:
            print("❌ Frontend dependencies missing - run 'npm install' in frontend/")
            frontend_deps = False
    else:
        print("⚠️  Frontend directory not found")
        frontend_deps = False
    
    return backend_deps, frontend_deps

def install_backend_dependencies():
    """Install backend test dependencies."""
    print("📦 Installing backend test dependencies...")
    
    deps = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "httpx>=0.24.0",
        "pytest-mock>=3.10.0"
    ]
    
    for dep in deps:
        success, _ = run_command(
            ["pip", "install", dep],
            description=f"Installing {dep}"
        )
        if not success:
            return False
    
    return True

def install_frontend_dependencies():
    """Install frontend test dependencies."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    success, _ = run_command(
        ["npm", "install"],
        cwd=frontend_dir,
        description="Installing frontend dependencies"
    )
    
    return success

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📊 COMPREHENSIVE TEST REPORT")
    print("=" * 50)
    
    root_dir = Path(__file__).parent.parent
    
    # Backend coverage report
    coverage_file = root_dir / "htmlcov" / "index.html"
    if coverage_file.exists():
        print(f"📈 Backend coverage report: {coverage_file}")
    
    # Frontend coverage report (if exists)
    frontend_coverage = root_dir / "frontend" / "coverage" / "index.html"
    if frontend_coverage.exists():
        print(f"📈 Frontend coverage report: {frontend_coverage}")
    
    print("\n🎯 Test Summary:")
    print("- Backend: pytest with coverage")
    print("- Frontend: Vitest with React Testing Library")
    print("- API: Integration tests with mock database")
    print("- Agents: Unit tests with mocked dependencies")
    print("- Security: Input validation and injection tests")

def main():
    parser = argparse.ArgumentParser(description="Run Credilinq Agent tests")
    parser.add_argument(
        "--backend", 
        choices=["all", "unit", "integration", "api", "agent", "security", "database", "slow"],
        default="all",
        help="Backend test type to run"
    )
    parser.add_argument(
        "--frontend",
        choices=["all", "coverage", "ui"],
        default="all", 
        help="Frontend test type to run"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing dependencies"
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Run only backend tests"
    )
    parser.add_argument(
        "--frontend-only", 
        action="store_true",
        help="Run only frontend tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate test report"
    )
    
    args = parser.parse_args()
    
    print("🧪 Credilinq Agent Test Runner")
    print("=" * 40)
    
    # Check dependencies
    backend_deps, frontend_deps = check_dependencies()
    
    if args.install_deps:
        if not backend_deps:
            backend_deps = install_backend_dependencies()
        if not frontend_deps:
            frontend_deps = install_frontend_dependencies()
    
    success = True
    
    # Run backend tests
    if not args.frontend_only and backend_deps:
        backend_success, _ = run_backend_tests(args.backend, args.verbose)
        success = success and backend_success
    elif not args.frontend_only:
        print("⚠️  Skipping backend tests - dependencies missing")
        success = False
    
    # Run frontend tests
    if not args.backend_only and frontend_deps:
        frontend_success, _ = run_frontend_tests(args.frontend, args.verbose)
        success = success and frontend_success
    elif not args.backend_only:
        print("⚠️  Skipping frontend tests - dependencies missing")
        success = False
    
    # Generate report
    if args.report:
        generate_test_report()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()