#!/usr/bin/env python3
"""
Railway-specific startup script for CrediLinq AI Content Platform.
Optimized for Railway deployment with fast startup and reduced memory usage.
"""
import os
import sys
import subprocess
import logging
import signal
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global process reference for cleanup
app_process = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    if app_process:
        try:
            app_process.terminate()
            app_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't terminate gracefully, killing...")
            app_process.kill()
    sys.exit(0)

def validate_railway_environment():
    """Validate Railway environment and set defaults."""
    logger.info("🚂 Railway Environment Setup:")
    
    # Railway environment detection
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    if not is_railway:
        logger.warning("Not running in Railway environment")
    
    # Set Railway-specific defaults
    port = os.environ.get('PORT', '8080')  # Railway uses 8080 by default
    if not port or port.strip() == '':
        port = '8080'
    
    os.environ.setdefault('PORT', port)
    os.environ.setdefault('HOST', '0.0.0.0')
    os.environ.setdefault('WORKERS', '1')  # Single worker for Railway
    os.environ.setdefault('ENVIRONMENT', 'production')
    
    # Basic defaults (will be overridden based on application mode)
    os.environ.setdefault('ENABLE_CACHE', 'false')
    os.environ.setdefault('DEBUG', 'false')
    
    logger.info(f"  Port: {port}")
    logger.info(f"  Environment: {os.environ.get('ENVIRONMENT')}")
    logger.info(f"  Railway Service: {os.environ.get('RAILWAY_SERVICE_NAME', 'unknown')}")
    
    return port, is_railway

def start_simple_mode():
    """Start the simple Railway app as a fallback."""
    logger.info("🚂 Starting SIMPLE MODE as fallback")
    
    # Build simple mode command
    port = os.environ.get('PORT', '8080')
    cmd = [
        'uvicorn',
        'src.main_railway_simple:app',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--workers', '1',
        '--access-log',
        '--log-level', 'info',
        '--timeout-keep-alive', '30',
        '--timeout-graceful-shutdown', '30',
        '--no-use-colors',
    ]
    
    logger.info(f"📝 Simple mode command: {' '.join(cmd)}")
    
    try:
        # Execute simple mode directly
        import subprocess
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"❌ Simple mode also failed: {e}")
        sys.exit(1)

def main():
    """Start the Railway-optimized application."""
    global app_process
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ensure correct working directory and Python path for Railway
    app_dir = '/app'  # Railway default app directory
    if os.path.exists(app_dir):
        os.chdir(app_dir)
        # Add app directory to Python path for module imports
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        logger.info(f"📁 Set working directory to: {app_dir}")
    else:
        logger.warning(f"⚠️ Railway app directory {app_dir} not found, using current directory")
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
    
    logger.info("🚂 Starting CrediLinq AI Platform (Railway Optimized)")
    
    # Validate and configure environment
    port, is_railway = validate_railway_environment()
    
    # Determine which application module to use
    # Default to full system in production environment
    environment = os.environ.get('ENVIRONMENT', 'production').lower()
    use_full_system = os.environ.get('RAILWAY_FULL', '').lower() == 'true'
    enable_agents = os.environ.get('ENABLE_AGENT_LOADING', '').lower() == 'true'
    force_simple = os.environ.get('RAILWAY_SIMPLE', '').lower() == 'true'
    
    # Use full system by default in production unless explicitly disabled
    if (use_full_system or enable_agents or environment == 'production') and not force_simple:
        app_module = 'src.main:app'
        logger.info("🚂 Using FULL APPLICATION with AI agents and complete feature set")
        logger.info("   ✅ All AI agents will be loaded")
        logger.info("   ✅ Complete workflow orchestration")
        logger.info("   ✅ Advanced content generation")
        # Increase memory settings for full system
        os.environ.setdefault('ENABLE_ANALYTICS', 'true')
        os.environ.setdefault('ENABLE_PERFORMANCE_TRACKING', 'true')
    else:
        app_module = 'src.main_railway_simple:app'
        logger.info("🚂 Using Railway SIMPLE application module (hardcoded routes for debugging)")
        logger.info("   ⚠️  Limited functionality - agents disabled")
        # Keep minimal settings
        os.environ.setdefault('ENABLE_ANALYTICS', 'false')
        os.environ.setdefault('ENABLE_PERFORMANCE_TRACKING', 'false')
    
    # Build uvicorn command with Railway optimizations
    cmd = [
        'uvicorn',
        app_module,
        '--host', '0.0.0.0',
        '--port', str(port),
        '--workers', '1',  # Single worker for Railway stability
        '--access-log',
        '--log-level', 'info',
        '--timeout-keep-alive', '30',
        '--timeout-graceful-shutdown', '30',
        '--no-use-colors',  # Better for Railway logs
    ]
    
    logger.info(f"📝 Command: {' '.join(cmd)}")
    
    # Pre-flight checks
    logger.info("🔍 Running pre-flight checks...")
    
    # Check critical environment variables
    required_vars = ['DATABASE_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing critical environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Check optional AI API keys
    ai_keys = ['OPENAI_API_KEY', 'GEMINI_API_KEY']
    available_keys = [key for key in ai_keys if os.getenv(key)]
    
    if not available_keys:
        logger.warning("⚠️ No AI API keys found - AI features will be limited")
    else:
        logger.info(f"✅ AI API keys available: {len(available_keys)}")
    
    logger.info("✅ Pre-flight checks completed")
    
    # Test imports before starting uvicorn
    logger.info("🔍 Testing application imports...")
    try:
        if 'src.main:app' in app_module:
            # Run detailed import diagnostics first
            logger.info("🔬 Running detailed import diagnostics...")
            try:
                # Test basic dependencies
                import fastapi, uvicorn, pydantic, psycopg2, openai
                logger.info("✅ Basic dependencies imported successfully")
                
                # Test core modules
                from src.config.settings import settings
                from src.config.database import db_config
                logger.info("✅ Core configuration modules imported")
                
                # Test agent factory
                from src.agents.core.agent_factory import AgentFactory
                logger.info("✅ Agent factory imported")
                
                # Test LangGraph compatibility
                from src.agents.core.langgraph_compat import StateGraph, START, END
                logger.info(f"✅ LangGraph compatibility layer: START={repr(START)}, END={repr(END)}")
                
            except Exception as diag_e:
                logger.error(f"❌ Detailed diagnostics failed: {diag_e}")
                logger.error(f"   Error type: {type(diag_e).__name__}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
            
            # Now test full app import
            from src.main import app as test_app
            logger.info("✅ Main application imports successful")
        else:
            from src.main_railway_simple import app as test_app
            logger.info("✅ Simple application imports successful")
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        # Print the full traceback for debugging
        import traceback
        tb_lines = traceback.format_exc().split('\n')
        for i, line in enumerate(tb_lines):
            if line.strip():
                logger.error(f"   TB[{i:2d}]: {line}")
                
        if 'src.main:app' in app_module:
            logger.warning("🔄 Main app failed, falling back to simple mode...")
            app_module = 'src.main_railway_simple:app'
            logger.info("🚂 Switched to SIMPLE application mode due to import failure")
            # Test simple mode imports
            try:
                from src.main_railway_simple import app as simple_app
                logger.info("✅ Simple mode imports successful")
            except Exception as simple_e:
                logger.error(f"❌ Simple mode also failed: {simple_e}")
                sys.exit(1)
    
    try:
        # Start the application
        logger.info("🚀 Starting Railway application server...")
        
        app_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Enhanced error collection for debugging
        startup_errors = []
        
        # Monitor startup with shorter timeout for Railway
        startup_timeout = 30  # 30 seconds for Railway
        startup_time = time.time()
        startup_successful = False
        
        while app_process.poll() is None:
            line = app_process.stdout.readline()
            if line:
                print(line.rstrip())  # Forward to Railway logs
                
                # Check for successful startup
                success_indicators = [
                    'started server process',
                    'application startup complete', 
                    'uvicorn running',
                    'railway deployment started successfully'
                ]
                
                if any(indicator in line.lower() for indicator in success_indicators):
                    logger.info("✅ Railway application started successfully")
                    startup_successful = True
                    break
                
                # Check for startup errors
                error_indicators = [
                    'error', 'failed', 'exception', 'traceback',
                    'import error', 'module not found'
                ]
                
                if any(error in line.lower() for error in error_indicators):
                    logger.error(f"❌ Startup error: {line.strip()}")
                    startup_errors.append(line.strip())
            
            # Timeout check
            if time.time() - startup_time > startup_timeout:
                logger.error("❌ Railway startup timeout - terminating")
                app_process.kill()
                sys.exit(1)
            
            time.sleep(0.1)
        
        # Handle startup results
        if startup_successful:
            logger.info("🔄 Application running, monitoring...")
            try:
                # Continue monitoring
                while app_process.poll() is None:
                    line = app_process.stdout.readline()
                    if line:
                        print(line.rstrip())
                    time.sleep(0.1)
                
                # Process ended
                return_code = app_process.returncode
                if return_code == 0:
                    logger.info("✅ Application shut down gracefully")
                else:
                    logger.error(f"❌ Application exited with code {return_code}")
                    sys.exit(return_code)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested")
                app_process.terminate()
                app_process.wait(timeout=10)
        else:
            # Startup failed - try fallback to simple mode if using full system
            return_code = app_process.returncode
            logger.error(f"❌ Railway application failed to start (code: {return_code})")
            
            # Print collected errors for debugging
            if startup_errors:
                logger.error("Collected startup errors:")
                for error in startup_errors:
                    logger.error(f"  {error}")
            
            # Attempt fallback to simple mode if we were trying full system
            if 'src.main:app' in app_module:
                logger.warning("🔄 Attempting fallback to simple mode due to startup failure...")
                return start_simple_mode()
            else:
                sys.exit(return_code or 1)
        
    except Exception as e:
        logger.error(f"❌ Railway deployment error: {e}")
        if app_process:
            app_process.kill()
        sys.exit(1)

if __name__ == '__main__':
    main()