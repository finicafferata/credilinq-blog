#!/usr/bin/env python3
"""
Railway deployment debugging script.
Run this to identify potential issues before deployment.
"""
import os
import sys
import psutil
import logging
import asyncio
import psycopg2
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment_variables() -> Dict[str, Any]:
    """Check critical environment variables for Railway deployment."""
    results = {"status": "ok", "issues": [], "warnings": []}
    
    # Critical variables
    critical_vars = {
        "DATABASE_URL": "PostgreSQL connection string",
        "OPENAI_API_KEY": "OpenAI API access",
        "PORT": "Railway port assignment"
    }
    
    # Optional but recommended
    optional_vars = {
        "RAILWAY_ENVIRONMENT": "Railway environment detection", 
        "SECRET_KEY": "Application security",
        "JWT_SECRET": "JWT token security",
        "ENVIRONMENT": "Runtime environment"
    }
    
    for var, description in critical_vars.items():
        value = os.getenv(var)
        if not value:
            results["issues"].append(f"Missing critical variable {var}: {description}")
            results["status"] = "error"
        elif var == "PORT":
            try:
                port_num = int(value)
                if port_num < 1024 or port_num > 65535:
                    results["issues"].append(f"PORT {port_num} outside valid range (1024-65535)")
            except ValueError:
                results["issues"].append(f"PORT '{value}' is not a valid number")
        
        logger.info(f"✅ {var}: {'***REDACTED***' if 'key' in var.lower() or 'secret' in var.lower() else value}")
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if not value:
            results["warnings"].append(f"Optional variable {var} not set: {description}")
        else:
            logger.info(f"ℹ️  {var}: {'***REDACTED***' if 'secret' in var.lower() else value}")
    
    return results

def check_database_connection() -> Dict[str, Any]:
    """Test database connection with Railway PostgreSQL."""
    results = {"status": "ok", "issues": [], "details": {}}
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        results["issues"].append("DATABASE_URL not found")
        results["status"] = "error"
        return results
    
    # Handle Railway postgres:// URLs
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    try:
        logger.info("Testing database connection...")
        conn = psycopg2.connect(database_url)
        
        # Test basic query
        with conn.cursor() as cursor:
            cursor.execute("SELECT version()")
            result = cursor.fetchone()
            results["details"]["postgres_version"] = result[0] if result else "Unknown"
            
            # Test database access
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                LIMIT 5
            """)
            tables = cursor.fetchall()
            results["details"]["table_count"] = len(tables)
            results["details"]["sample_tables"] = [row[0] for row in tables]
        
        conn.close()
        logger.info("✅ Database connection successful")
        
    except Exception as e:
        results["issues"].append(f"Database connection failed: {str(e)}")
        results["status"] = "error"
        logger.error(f"❌ Database connection failed: {e}")
    
    return results

def check_memory_usage() -> Dict[str, Any]:
    """Check current memory usage and estimate requirements."""
    results = {"status": "ok", "issues": [], "details": {}}
    
    try:
        # Get current process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Convert to MB
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        results["details"]["rss_mb"] = round(rss_mb, 2)
        results["details"]["vms_mb"] = round(vms_mb, 2)
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        total_mb = system_memory.total / 1024 / 1024
        available_mb = system_memory.available / 1024 / 1024
        
        results["details"]["system_total_mb"] = round(total_mb, 2)
        results["details"]["system_available_mb"] = round(available_mb, 2)
        
        # Railway memory limits (approximate)
        railway_limit_mb = 512  # Hobby plan limit
        if rss_mb > railway_limit_mb * 0.8:
            results["issues"].append(f"Memory usage {rss_mb:.1f}MB approaching Railway limit (~{railway_limit_mb}MB)")
            results["status"] = "warning"
        
        logger.info(f"Memory - RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB")
        
    except Exception as e:
        results["issues"].append(f"Memory check failed: {str(e)}")
        results["status"] = "error"
    
    return results

def check_dependencies() -> Dict[str, Any]:
    """Check if critical dependencies can be imported."""
    results = {"status": "ok", "issues": [], "details": {"imported": [], "failed": []}}
    
    critical_imports = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("langchain", "AI framework"),
        ("openai", "OpenAI client"),
        ("psycopg2", "PostgreSQL driver"),
        ("prisma", "Database ORM")
    ]
    
    for module, description in critical_imports:
        try:
            __import__(module)
            results["details"]["imported"].append(module)
            logger.info(f"✅ {module}: {description}")
        except ImportError as e:
            results["details"]["failed"].append({"module": module, "error": str(e)})
            results["issues"].append(f"Failed to import {module}: {e}")
            results["status"] = "error"
            logger.error(f"❌ {module}: {e}")
    
    return results

def check_application_startup() -> Dict[str, Any]:
    """Test basic application imports and configuration."""
    results = {"status": "ok", "issues": [], "details": {}}
    
    try:
        # Test basic imports
        sys.path.insert(0, "/app" if os.path.exists("/app") else ".")
        
        from src.config import settings
        results["details"]["environment"] = settings.environment
        results["details"]["debug"] = settings.debug
        results["details"]["api_version"] = settings.api_version
        
        # Test database config (try both versions)
        try:
            from src.config import db_config
            db_health = db_config.health_check()
            results["details"]["db_health"] = db_health
        except:
            # Try Railway database config
            from src.config.database_railway import railway_db
            db_health = railway_db.health_check()
            results["details"]["db_health"] = db_health
        
        if db_health.get("status") != "healthy":
            results["issues"].append(f"Database health check failed: {db_health}")
            results["status"] = "warning"
        
        logger.info(f"✅ Application config loaded - Environment: {settings.environment}")
        
    except Exception as e:
        results["issues"].append(f"Application startup test failed: {str(e)}")
        results["status"] = "error"
        logger.error(f"❌ Application startup test failed: {e}")
    
    return results

def main():
    """Run complete Railway deployment diagnostics."""
    logger.info("🔍 Starting Railway deployment diagnostics...")
    
    # Run all checks
    checks = {
        "environment": check_environment_variables(),
        "database": check_database_connection(),
        "memory": check_memory_usage(),
        "dependencies": check_dependencies(),
        "application": check_application_startup()
    }
    
    # Summarize results
    total_issues = sum(len(check.get("issues", [])) for check in checks.values())
    total_warnings = sum(len(check.get("warnings", [])) for check in checks.values())
    
    logger.info(f"\n📊 DIAGNOSTIC SUMMARY:")
    logger.info(f"Issues found: {total_issues}")
    logger.info(f"Warnings: {total_warnings}")
    
    if total_issues == 0:
        logger.info("🎉 No critical issues found! Deployment should succeed.")
    else:
        logger.info("⚠️  Critical issues found. Address these before deployment:")
        for check_name, check_result in checks.items():
            if check_result.get("issues"):
                logger.info(f"\n{check_name.upper()} ISSUES:")
                for issue in check_result["issues"]:
                    logger.info(f"  - {issue}")
    
    return checks

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Railway Deployment Diagnostics")
        print("Usage: python railway_debug.py")
        print("\nChecks:")
        print("  - Environment variables")
        print("  - Database connectivity")
        print("  - Memory usage")
        print("  - Dependency imports")
        print("  - Application startup")
        sys.exit(0)
    
    main()