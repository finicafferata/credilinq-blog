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
        import traceback
        logger.error(f"❌ MAIN APP FAILED TO START: {e}")
        logger.error(f"❌ FULL TRACEBACK: {traceback.format_exc()}")
        
        # TEMPORARILY REMOVED FALLBACK TO SEE MAIN APP ERROR
        logger.error("🚨 RAILWAY DEBUG: Main app failed, NO FALLBACK - will show real error")
        raise e  # Re-raise the error instead of falling back

if __name__ == '__main__':
    main()