#!/usr/bin/env python3
"""
Production deployment script for CrediLinq AI Content Platform with LangGraph workflows
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def run_command(cmd, capture_output=True, check=True):
    """Run a command and return the result"""
    print(f"🔄 Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(
        cmd, 
        capture_output=capture_output, 
        text=True, 
        check=check
    )
    
    if capture_output:
        print(f"✅ Output: {result.stdout}")
        if result.stderr:
            print(f"⚠️  Stderr: {result.stderr}")
    
    return result

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'langgraph_workflows.py',
        'langgraph.json', 
        'requirements-railway.txt',
        'railway.toml',
        'Dockerfile'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def update_dockerfile_for_langgraph():
    """Update Dockerfile to include LangGraph workflows"""
    dockerfile_content = """# Railway-optimized build with LangGraph support
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    postgresql-client \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-railway.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy LangGraph workflows
COPY langgraph_workflows.py ./
COPY langgraph.json ./

# Make startup scripts executable
RUN chmod +x /app/scripts/start.py /app/scripts/start_railway.py

# Set environment variables for Railway + LangGraph
ENV PYTHONPATH=/app \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PORT=8000 \\
    LANGGRAPH_API_URL=http://localhost:8001 \\
    ENABLE_LANGGRAPH=true

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache

# Expose both FastAPI and LangGraph ports
EXPOSE 8000 8001

# Health check for both services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:${PORT:-8000}/health/railway || exit 1

# Start both FastAPI and LangGraph services
CMD ["python", "/app/scripts/start_production.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Updated Dockerfile with LangGraph support")

def create_production_startup_script():
    """Create a production startup script that runs both services"""
    startup_script = '''#!/usr/bin/env python3
"""
Production startup script for CrediLinq with LangGraph workflows
Starts both FastAPI backend and LangGraph API services
"""

import os
import sys
import signal
import subprocess
import threading
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment setup
os.environ.setdefault('PYTHONPATH', '/app')
os.environ.setdefault('PYTHONUNBUFFERED', '1')

# Global process references
fastapi_process = None
langgraph_process = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\\n🔄 Received signal {signum}, shutting down services...")
    
    if langgraph_process:
        langgraph_process.terminate()
        langgraph_process.wait(timeout=10)
    
    if fastapi_process:
        fastapi_process.terminate()
        fastapi_process.wait(timeout=10)
    
    print("✅ Services stopped")
    sys.exit(0)

def start_langgraph_service():
    """Start LangGraph API service"""
    global langgraph_process
    
    print("🚀 Starting LangGraph API service on port 8001...")
    
    # Use environment PORT + 1 for LangGraph, or default to 8001
    port = int(os.environ.get('PORT', 8000)) + 1
    
    cmd = [
        'python', '-m', 'langgraph', 'up', 
        '--port', str(port),
        '--host', '0.0.0.0',
        '--config', 'langgraph.json'
    ]
    
    try:
        langgraph_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/app'
        )
        
        # Wait a moment for startup
        time.sleep(3)
        
        if langgraph_process.poll() is None:
            print(f"✅ LangGraph API started successfully on port {port}")
            return True
        else:
            stdout, stderr = langgraph_process.communicate()
            print(f"❌ LangGraph API failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting LangGraph API: {e}")
        return False

def start_fastapi_service():
    """Start FastAPI service"""
    global fastapi_process
    
    print("🚀 Starting FastAPI service...")
    
    port = int(os.environ.get('PORT', 8000))
    
    cmd = [
        'python', '-m', 'uvicorn', 'src.main:app',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--workers', '1',
        '--timeout-keep-alive', '30'
    ]
    
    try:
        fastapi_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/app'
        )
        
        print(f"✅ FastAPI started on port {port}")
        return True
        
    except Exception as e:
        print(f"❌ Error starting FastAPI: {e}")
        return False

def main():
    """Main startup logic"""
    print("🚀 CrediLinq AI Content Platform with LangGraph - Production Startup")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start LangGraph service first
    if not start_langgraph_service():
        print("❌ Failed to start LangGraph service, starting FastAPI only")
    
    # Start FastAPI service
    if not start_fastapi_service():
        print("❌ Failed to start FastAPI service")
        sys.exit(1)
    
    print("✅ All services started successfully")
    print("📊 FastAPI: http://0.0.0.0:8000")
    print("🎨 LangGraph API: http://0.0.0.0:8001")
    print("Press Ctrl+C to stop all services")
    
    # Keep the main process alive
    try:
        while True:
            if fastapi_process and fastapi_process.poll() is not None:
                print("❌ FastAPI process died, restarting...")
                start_fastapi_service()
            
            if langgraph_process and langgraph_process.poll() is not None:
                print("❌ LangGraph process died, restarting...")
                start_langgraph_service()
                
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\\n🔄 Keyboard interrupt received")
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
'''
    
    os.makedirs('scripts', exist_ok=True)
    with open('scripts/start_production.py', 'w') as f:
        f.write(startup_script)
    
    # Make it executable
    os.chmod('scripts/start_production.py', 0o755)
    
    print("✅ Created production startup script")

def update_railway_config():
    """Update railway.toml for LangGraph deployment"""
    railway_config = """[build]
builder = "dockerfile"
dockerfile = "Dockerfile"

[deploy]
healthcheckPath = "/health/railway"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3

# Memory and resource limits for LangGraph + FastAPI
memoryLimit = "2GB"
cpuLimit = "1500m"

# Production startup with both services
startCommand = "python /app/scripts/start_production.py"

[environments.production.variables]
RAILWAY_FULL = "true"
ENABLE_AGENT_LOADING = "true"
ENABLE_LANGGRAPH = "true"
LANGGRAPH_API_URL = "http://localhost:8001"

[environments.production]
memoryLimit = "4GB"
cpuLimit = "2000m"

[environments.staging]
memoryLimit = "2GB"
cpuLimit = "1500m"

[environments.development]
memoryLimit = "1GB"
cpuLimit = "1000m"
"""
    
    with open('railway.toml', 'w') as f:
        f.write(railway_config)
    
    print("✅ Updated railway.toml with LangGraph configuration")

def deploy_to_railway():
    """Deploy to Railway"""
    print("🚀 Deploying to Railway...")
    
    try:
        # Check if railway CLI is available
        run_command("railway --version")
        
        # Login check
        result = run_command("railway whoami", check=False)
        if result.returncode != 0:
            print("❌ Not logged in to Railway. Please run: railway login")
            return False
        
        # Deploy
        run_command("railway up --detach")
        
        print("✅ Deployment initiated successfully")
        print("📊 Check Railway dashboard for deployment status")
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        print("💡 Make sure you have Railway CLI installed: https://docs.railway.app/develop/cli")
        return False

def main():
    """Main deployment script"""
    print("🚀 CrediLinq AI Content Platform - Production Deployment")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Prepare production files
    print("\n📋 Preparing production configuration...")
    update_dockerfile_for_langgraph()
    create_production_startup_script()
    update_railway_config()
    
    print("\n✅ Production configuration ready!")
    
    # Ask user if they want to deploy now
    deploy_now = input("\n🚀 Deploy to Railway now? (y/n): ").lower().strip()
    
    if deploy_now == 'y':
        success = deploy_to_railway()
        if success:
            print("\n🎉 Deployment completed!")
            print("\n📊 Your services will be available at:")
            print("   • FastAPI Backend: https://your-railway-domain.railway.app")
            print("   • LangGraph Studio: https://smith.langchain.com/studio/")
            print("   • Connect Studio to: https://your-railway-domain.railway.app:8001")
        else:
            print("\n❌ Deployment failed. Check the errors above.")
    else:
        print("\n📋 Deployment files prepared. You can deploy later with: railway up")

if __name__ == "__main__":
    main()