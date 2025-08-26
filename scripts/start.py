#!/usr/bin/env python3
"""
Railway-optimized startup script for CrediLinq AI Content Platform.
Handles PORT environment variable and other Railway-specific configurations.
"""
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Start the application with Railway-optimized settings."""
    
    # Check for dry run mode
    dry_run = '--dry-run' in sys.argv
    
    # Get environment variables
    port = os.environ.get('PORT', '8000')
    # Handle empty string PORT (Railway edge case)
    if not port or port.strip() == '':
        port = '8000'
    host = os.environ.get('HOST', '0.0.0.0')
    workers_env = os.environ.get('WORKERS', '1')
    workers = int(workers_env) if workers_env and workers_env.strip() else 1
    reload = os.environ.get('RELOAD', 'false').lower() == 'true'
    
    # Railway environment detection
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    logger.info(f"Starting CrediLinq AI Platform...")
    logger.info(f"Port: {port}")
    logger.info(f"Host: {host}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Railway Environment: {is_railway}")
    logger.info(f"Dry Run Mode: {dry_run}")
    
    # Build uvicorn command
    cmd = [
        'uvicorn',
        'src.main:app',
        '--host', host,
        '--port', str(port),
    ]
    
    # Railway-specific optimizations
    if is_railway:
        # Single worker for Railway (better for memory usage)
        cmd.extend(['--workers', '1'])
        # Disable reload in production
        if not reload:
            cmd.append('--no-reload')
        # Add access log for Railway
        cmd.extend(['--access-log', '--log-level', 'info'])
    else:
        # Local development settings
        cmd.extend(['--workers', str(workers)])
        if reload:
            cmd.append('--reload')
    
    # Environment-specific settings
    if os.environ.get('ENVIRONMENT') == 'production':
        cmd.extend(['--no-reload', '--log-level', 'warning'])
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    # Exit early if dry run
    if dry_run:
        logger.info("Dry run completed - command construction successful")
        return
    
    try:
        # Start the application
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
        sys.exit(0)

if __name__ == '__main__':
    main()