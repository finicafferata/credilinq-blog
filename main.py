#!/usr/bin/env python3
"""
Main entry point for the CrediLinQ Agent application.
This file imports the FastAPI app from src.main for Railway deployment.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the FastAPI app
from main import app

# This allows Railway to find the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 