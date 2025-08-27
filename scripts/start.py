#!/usr/bin/env python3
"""
Railway-optimized startup script for CrediLinq AI Content Platform.
Handles PORT environment variable and other Railway-specific configurations.
"""
import os
import sys
import subprocess
import logging
import signal
import time
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global process reference for cleanup
app_process = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    if app_process:
        try:
            app_process.terminate()
            app_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't terminate gracefully, killing...")
            app_process.kill()
    sys.exit(0)

def check_railway_environment():
    """Validate Railway environment and log diagnostics."""
    logger.info("🔍 Railway Environment Diagnostics:")
    
    # Railway-specific variables
    railway_vars = {
        'RAILWAY_ENVIRONMENT': 'Railway environment name',
        'RAILWAY_SERVICE_NAME': 'Service name',
        'RAILWAY_REPLICA_ID': 'Replica identifier',
        'RAILWAY_DEPLOYMENT_ID': 'Deployment identifier'
    }
    
    for var, desc in railway_vars.items():
        value = os.getenv(var, 'Not set')
        logger.info(f"  {var}: {value}")
    
    # Memory check
    try:
        memory = psutil.virtual_memory()
        logger.info(f"  Available Memory: {memory.available / 1024 / 1024:.1f}MB")
        logger.info(f"  Total Memory: {memory.total / 1024 / 1024:.1f}MB")
    except:
        logger.warning("  Could not check memory usage")
    
    # Database URL validation
    db_url = os.getenv('DATABASE_URL', '')
    if db_url:
        # Mask sensitive parts
        if 'postgresql://' in db_url or 'postgres://' in db_url:
            parts = db_url.split('@')
            if len(parts) > 1:
                masked_url = f"{parts[0].split(':')[0]}://***:***@{parts[1]}"
                logger.info(f"  DATABASE_URL: {masked_url}")
        else:
            logger.warning("  DATABASE_URL format may be incorrect")
    else:
        logger.error("  DATABASE_URL not set!")
    
    return True

def main():
    """Start the application with Railway-optimized settings."""
    global app_process
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check for dry run mode
    dry_run = '--dry-run' in sys.argv
    debug_mode = '--debug' in sys.argv
    
    # Railway environment diagnostics
    check_railway_environment()
    
    # Get environment variables
    port = os.environ.get('PORT', '8000')
    # Handle empty string PORT (Railway edge case)
    if not port or port.strip() == '':
        port = '8000'
    
    # Validate port
    try:
        port_int = int(port)
        if port_int < 1024 or port_int > 65535:
            logger.warning(f"Port {port_int} outside recommended range (1024-65535)")
    except ValueError:
        logger.error(f"Invalid PORT value: {port}")
        port = '8000'
    
    host = os.environ.get('HOST', '0.0.0.0')
    workers_env = os.environ.get('WORKERS', '1')
    workers = int(workers_env) if workers_env and workers_env.strip() else 1
    reload = os.environ.get('RELOAD', 'false').lower() == 'true'
    
    # Railway environment detection
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    logger.info(f"🚀 Starting CrediLinq AI Platform...")
    logger.info(f"  Port: {port}")
    logger.info(f"  Host: {host}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Railway Environment: {is_railway}")
    logger.info(f"  Debug Mode: {debug_mode}")
    logger.info(f"  Dry Run Mode: {dry_run}")
    
    # Choose application module based on environment
    # Progressive Railway modes: ultra-minimal -> stable -> full
    railway_full = os.environ.get('RAILWAY_FULL', '').lower() == 'true'
    railway_stable = os.environ.get('RAILWAY_STABLE', '').lower() == 'true'
    railway_minimal = os.environ.get('RAILWAY_MINIMAL', '').lower() == 'true'
    
    if is_railway:
        # Default to stable mode for now (full mode has startup issues)
        if railway_minimal:
            app_module = 'src.main_railway_minimal:app'
            logger.info("🚂 Using Railway ultra-minimal mode (health checks only)")
        elif railway_full:
            app_module = 'src.main:app'
            logger.info("🚂 Using Railway FULL mode (all features, agents, settings)")
        else:
            # DEFAULT: Use stable mode for reliability
            app_module = 'src.main_railway_stable:app'
            logger.info("🚂 Using Railway stable mode (database + API, no agents)")
    else:
        app_module = 'src.main:app'
        logger.info("💻 Using full development mode")
    
    # Build uvicorn command
    cmd = [
        'uvicorn',
        app_module,
        '--host', host,
        '--port', str(port),
    ]
    
    # Railway-specific optimizations
    if is_railway:
        # Single worker for Railway (better for memory usage and startup time)
        cmd.extend(['--workers', '1'])
        # Add access log for Railway debugging
        cmd.extend(['--access-log', '--log-level', 'info'])
        # Add timeout settings for Railway
        cmd.extend(['--timeout-keep-alive', '30'])
        cmd.extend(['--timeout-graceful-shutdown', '30'])
        
        # Only add --reload if explicitly requested (not recommended for Railway)
        if reload:
            logger.warning("Reload mode enabled in Railway - this may cause issues")
            cmd.append('--reload')
    else:
        # Local development settings
        cmd.extend(['--workers', str(workers)])
        if reload:
            cmd.append('--reload')
    
    # Environment-specific settings
    env_name = os.environ.get('ENVIRONMENT', '').lower()
    if env_name == 'production':
        cmd.extend(['--log-level', 'warning'])
    elif debug_mode:
        cmd.extend(['--log-level', 'debug'])
    else:
        cmd.extend(['--log-level', 'info'])
    
    logger.info(f"📝 Command: {' '.join(cmd)}")
    
    # Exit early if dry run
    if dry_run:
        logger.info("✅ Dry run completed - command construction successful")
        return
    
    # Pre-flight checks
    logger.info("🔍 Running pre-flight checks...")
    
    # Check critical environment variables
    critical_vars = ['DATABASE_URL', 'OPENAI_API_KEY']
    missing_vars = [var for var in critical_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing critical environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Check if port is available (local dev only)
    if not is_railway:
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, int(port)))
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(f"❌ Port {port} is already in use")
                sys.exit(1)
            else:
                logger.warning(f"⚠️ Port check failed: {e}")
    
    logger.info("✅ Pre-flight checks completed")
    
    try:
        # Start the application with enhanced monitoring
        logger.info("🚀 Starting application server...")
        
        # Use Popen for better process control
        app_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor process output
        startup_timeout = 60  # 60 seconds for startup
        startup_time = time.time()
        
        while app_process.poll() is None:
            line = app_process.stdout.readline()
            if line:
                print(line.rstrip())  # Forward output to Railway logs
                
                # Check for successful startup indicators
                if any(indicator in line.lower() for indicator in 
                       ['started server process', 'application startup complete', 'uvicorn running']):
                    logger.info("✅ Application started successfully")
                    break
                
                # Check for startup errors
                if any(error in line.lower() for error in 
                       ['error', 'failed', 'exception', 'traceback']):
                    logger.error(f"❌ Startup error detected: {line.strip()}")
            
            # Timeout check
            if time.time() - startup_time > startup_timeout:
                logger.error("❌ Startup timeout - killing process")
                app_process.kill()
                sys.exit(1)
            
            time.sleep(0.1)
        
        # If we get here, process exited during startup
        if app_process.returncode != 0:
            logger.error(f"❌ Application exited with code {app_process.returncode}")
            sys.exit(app_process.returncode)
        
        # Wait for the process to complete (shouldn't reach here normally)
        app_process.wait()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Application shutdown requested")
        if app_process:
            app_process.terminate()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if app_process:
            app_process.kill()
        sys.exit(1)

if __name__ == '__main__':
    main()