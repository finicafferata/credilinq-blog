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

def main():
    """Start the Railway-optimized application."""
    global app_process
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("🚂 Starting CrediLinq AI Platform (Railway Optimized)")
    
    # Validate and configure environment
    port, is_railway = validate_railway_environment()
    
    # Determine which application module to use
    use_full_system = os.environ.get('RAILWAY_FULL', '').lower() == 'true'
    enable_agents = os.environ.get('ENABLE_AGENT_LOADING', '').lower() == 'true'
    
    if use_full_system or enable_agents:
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
            # Startup failed
            return_code = app_process.returncode
            logger.error(f"❌ Railway application failed to start (code: {return_code})")
            sys.exit(return_code or 1)
        
    except Exception as e:
        logger.error(f"❌ Railway deployment error: {e}")
        if app_process:
            app_process.kill()
        sys.exit(1)

if __name__ == '__main__':
    main()