#!/usr/bin/env python3
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
    print(f"\n🔄 Received signal {signum}, shutting down services...")
    
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
        print("\n🔄 Keyboard interrupt received")
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
