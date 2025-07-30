"""
Enhanced FastAPI application for CrediLinQ AI Content Platform.
Includes comprehensive API documentation, versioning, webhooks, analytics, authentication, and monitoring.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

from .config import settings, db_config
from .api.routes import blogs, campaigns, analytics, health, documents, api_analytics, content_repurposing, content_preview, competitor_intelligence, workflow
from .api.routes import workflow_simple, workflow_new, workflow_fixed, images_debug
from .core.api_docs import configure_api_docs, custom_openapi_schema
from .core.versioning import create_versioned_app, VersionCompatibilityMiddleware
from .core.webhooks import router as webhooks_router, webhook_manager
from .core.api_analytics import APIAnalyticsMiddleware
from .core.auth import AuthenticationMiddleware
from .core.enhanced_exceptions import ErrorHandlingMiddleware
from .core.monitoring import start_monitoring, stop_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager with monitoring and services."""
    # Startup
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Perform database health check
    db_health = db_config.health_check()
    if db_health["status"] == "healthy":
        logger.info("✅ Database connections healthy")
    else:
        logger.warning(f"⚠️ Database health check failed: {db_health}")
    
    # Start monitoring services
    try:
        await start_monitoring()
        logger.info("✅ Monitoring services started")
    except Exception as e:
        logger.warning(f"⚠️ Failed to start monitoring: {e}")
    
    # Start webhook retry processor
    try:
        await webhook_manager.start_retry_processor()
        logger.info("✅ Webhook services started")
    except Exception as e:
        logger.warning(f"⚠️ Failed to start webhooks: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CrediLinQ AI Content Platform")
    
    # Stop services
    try:
        await stop_monitoring()
        await webhook_manager.stop_retry_processor()
        logger.info("✅ Services stopped gracefully")
    except Exception as e:
        logger.warning(f"⚠️ Error during shutdown: {e}")

# Create enhanced FastAPI application
app = FastAPI(
    title="CrediLinQ AI Content Platform API",
    description="""
## CrediLinQ AI Content Platform API v2.0

A comprehensive AI-powered content management and marketing automation platform with advanced features:

### 🚀 Key Features
- **AI Content Generation**: Automated blog post creation using advanced AI agents
- **Campaign Management**: Multi-channel marketing campaign orchestration
- **Performance Analytics**: Detailed insights and real-time monitoring
- **Webhook Integration**: Real-time notifications for third-party systems
- **API Versioning**: Support for multiple API versions with backward compatibility
- **Comprehensive Authentication**: API keys, OAuth2, and JWT token support

### 📚 Documentation
- **Interactive API Docs**: [/docs](/docs)
- **API Reference**: [/redoc](/redoc)
- **Version Information**: [/versions](/versions)

### 🔐 Authentication
All endpoints require authentication using API keys or JWT tokens.
Include your credentials in the `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY_OR_JWT_TOKEN
```

### 🚦 Rate Limits
API usage is subject to rate limits based on your subscription tier.
Current limits are returned in response headers.

### 🔗 Webhooks
Configure webhooks to receive real-time notifications about events like:
- Blog post completion
- Campaign status updates
- Analytics threshold alerts

### 📊 Analytics
Access comprehensive API usage analytics including:
- Request/response metrics
- Performance insights
- Security monitoring
- User behavior analysis
    """,
    version="2.0.0",
    debug=settings.debug,
    lifespan=lifespan,
    docs_url=None,  # We'll use custom docs
    redoc_url=None,  # We'll use custom redoc
    openapi_url="/openapi.json"
)

# Configure enhanced API documentation
configure_api_docs(app)

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi_schema(app)

# Add middleware in correct order (last added = first executed)
# app.add_middleware(ErrorHandlingMiddleware)  # Temporarily disabled for debugging
# app.add_middleware(APIAnalyticsMiddleware)  # Temporarily disabled for debugging
# app.add_middleware(AuthenticationMiddleware)  # Temporarily disabled for debugging
# app.add_middleware(VersionCompatibilityMiddleware)  # Temporarily disabled for debugging

# Add CORS middleware with enhanced security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "accept",
        "accept-language", 
        "content-language",
        "content-type",
        "authorization",
        "x-requested-with",
        "x-api-key",
        "api-version"
    ],
    expose_headers=[
        "x-api-version",
        "x-request-id",
        "x-rate-limit-remaining",
        "x-rate-limit-reset",
        "warning",
        "sunset"
    ]
)

# Include API routes with versioning
# V2 routes (current)
app.include_router(blogs.router, prefix="/api/v2", tags=["blogs-v2"])
app.include_router(campaigns.router, prefix="/api/v2", tags=["campaigns-v2"])  
app.include_router(analytics.router, prefix="/api/v2", tags=["analytics-v2"])
app.include_router(documents.router, prefix="/api/v2", tags=["documents-v2"])
app.include_router(webhooks_router, prefix="/api/v2", tags=["webhooks-v2"])
app.include_router(api_analytics.router, prefix="/api/v2", tags=["api-analytics-v2"])
app.include_router(content_repurposing.router, prefix="/api/v2/content", tags=["content-repurposing-v2"])
app.include_router(content_preview.router, prefix="/api/v2/content-preview", tags=["content-preview-v2"])
app.include_router(competitor_intelligence.router, prefix="/api/v2", tags=["competitor-intelligence-v2"])
app.include_router(images_debug.router, prefix="/api/v2", tags=["images-v2"])
# app.include_router(workflow.router, prefix="/api/v2", tags=["workflow-v2"])
# app.include_router(workflow_simple.router, prefix="/api/v2", tags=["workflow-simple-v2"])
# app.include_router(workflow_new.router, prefix="/api/v2", tags=["workflow-new-v2"])
app.include_router(workflow_fixed.router, prefix="/api/v2", tags=["workflow-fixed-v2"])

# V1 routes (deprecated, for backward compatibility)
app.include_router(blogs.router, prefix="/api/v1", tags=["blogs-v1"], deprecated=True)
app.include_router(campaigns.router, prefix="/api/v1", tags=["campaigns-v1"], deprecated=True)
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics-v1"], deprecated=True)

# Default routes (no version prefix) - use v2
app.include_router(blogs.router, prefix="/api", tags=["blogs"])
app.include_router(campaigns.router, prefix="/api", tags=["campaigns"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(webhooks_router, prefix="/api", tags=["webhooks"])
app.include_router(api_analytics.router, prefix="/api", tags=["api-analytics"])
app.include_router(content_repurposing.router, prefix="/api/content", tags=["content-repurposing"])
app.include_router(content_preview.router, prefix="/api/content-preview", tags=["content-preview"])
app.include_router(competitor_intelligence.router, prefix="/api", tags=["competitor-intelligence"])

# Add a simple test route directly
@app.get("/api/images/test-direct")
async def test_images_direct():
    """Test endpoint to verify images routing works."""
    return {"message": "Images router is working directly!"}

# Include the debug images router
app.include_router(images_debug.router, prefix="/api/images", tags=["images-debug"])
# app.include_router(workflow.router, prefix="/api", tags=["workflow"])
# app.include_router(workflow_simple.router, prefix="/api", tags=["workflow-simple"])
# app.include_router(workflow_new.router, prefix="/api", tags=["workflow-new"])
app.include_router(workflow_fixed.router, prefix="/api", tags=["workflow-fixed"])

# Health and system routes
app.include_router(health.router, tags=["health"])

@app.get("/")
async def root():
    """Root endpoint with comprehensive API information."""
    return {
        "message": f"{app.title} is running",
        "version": app.version,
        "status": "operational",
        "environment": settings.environment,
        "api_info": {
            "current_version": "v2",
            "supported_versions": ["v1", "v2"],
            "documentation": {
                "interactive_docs": "/docs",
                "api_reference": "/redoc",
                "openapi_spec": "/openapi.json",
                "version_info": "/versions"
            },
            "authentication": {
                "methods": ["API Key", "JWT Token"],
                "documentation": "https://docs.credilinq.com/authentication"
            },
            "rate_limits": {
                "default": "1000 requests/hour",
                "documentation": "https://docs.credilinq.com/rate-limits"
            },
            "webhooks": {
                "supported_events": [
                    "blog.created", "blog.published", "campaign.completed",
                    "analytics.threshold", "system.health_alert"
                ],
                "documentation": "https://docs.credilinq.com/webhooks"
            }
        },
        "links": {
            "documentation": "https://docs.credilinq.com",
            "status_page": "https://status.credilinq.com",
            "support": "https://support.credilinq.com",
            "github": "https://github.com/credilinq/api"
        }
    }

@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"message": "pong", "timestamp": "2025-01-15T10:30:00Z"}

@app.post("/test-workflow")
async def test_workflow(request: dict):
    """Test workflow endpoint."""
    try:
        return {
            "workflow_id": "test-123",
            "status": "ok",
            "message": "Test workflow endpoint working"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers for better error responses
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={
            "error": "NOT_FOUND",
            "message": "The requested resource was not found",
            "available_endpoints": {
                "api_documentation": "/docs",
                "health_check": "/health",
                "version_info": "/versions"
            }
        }
    )

# For Vercel deployment
handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        access_log=True
    )