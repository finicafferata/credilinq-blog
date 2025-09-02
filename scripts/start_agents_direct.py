#!/usr/bin/env python3
"""
Direct agent startup script for Railway - bypasses all complexity.
Starts the full application with agents in the simplest way possible.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Start the application directly with uvicorn."""
    
    # Ensure we're in the app directory
    app_dir = '/app'
    if os.path.exists(app_dir):
        os.chdir(app_dir)
        sys.path.insert(0, app_dir)
    
    logger.info("🚀 Starting CrediLinq AI Platform - Direct Agent Mode")
    
    # Set essential environment variables
    os.environ.setdefault('HOST', '0.0.0.0')
    os.environ.setdefault('PORT', '8080')
    
    # Import and run uvicorn directly (no subprocess)
    import uvicorn
    
    logger.info("📦 Starting application with full agent system...")
    
    try:
        # Run the application directly
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=int(os.environ.get('PORT', 8080)),
            workers=1,
            access_log=True,
            log_level="info",
            timeout_keep_alive=30,
            # Don't use reload in production
            reload=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}")
        # Fallback to simple mode on any error
        logger.info("🔄 Falling back to simple mode...")
        uvicorn.run(
            "src.main_railway_simple:app",
            host="0.0.0.0",
            port=int(os.environ.get('PORT', 8080)),
            workers=1,
            access_log=True,
            log_level="info",
            timeout_keep_alive=30,
            reload=False
        )

if __name__ == '__main__':
    main()