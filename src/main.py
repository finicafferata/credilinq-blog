"""
Enhanced FastAPI application for CrediLinq AI Content Platform.
Includes comprehensive API documentation, versioning, webhooks, analytics, authentication, and monitoring.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

from .config import settings, db_config, secure_db
from .api.routes import blogs, campaigns, analytics, health, documents, api_analytics, content_repurposing, content_preview, settings as settings_router, review_workflow, workflow_metrics
# Temporarily disabled due to missing ML dependencies on Railway
# from .api.routes import competitor_intelligence
# from .api.routes import content_deliverables  # Temporarily disabled - missing dependencies
# from .api.routes import content_briefs  # Still temporarily disabled
from .api.routes import comments as comments_router
from .api.routes import suggestions as suggestions_router
from .api.routes import db_debug as db_debug_router
from .api.routes import workflow_fixed, images_debug, agents, auth, workflow_orchestration
# Temporarily disabled due to missing agent dependencies during migration
# from .api.routes import content_workflows
from .core.api_docs import configure_api_docs, custom_openapi_schema
# from .core.database_pool import connection_pool_maintenance
from .core.versioning import create_versioned_app, VersionCompatibilityMiddleware
from .core.webhooks import router as webhooks_router, webhook_manager
from .core.api_analytics import APIAnalyticsMiddleware
from .core.auth import AuthenticationMiddleware, initialize_auth_system
from .core.enhanced_exceptions import ErrorHandlingMiddleware
from .core.monitoring import start_monitoring, stop_monitoring
from .core.performance_middleware import (
    create_performance_middleware, 
    CompressionConfig, 
    CacheConfig
)
from .core.rate_limiting import RateLimitMiddleware
from .core.security_headers import SecurityHeadersMiddleware, ProductionSecurityMiddleware, SECURITY_CONFIGS
from .core.request_validation import RequestValidationMiddleware, VALIDATION_CONFIGS
# from .core.database_pool import startup_database_pool, shutdown_database_pool, connection_pool_maintenance
# from .core.database_auth import startup_database_auth, shutdown_database_auth
# Temporarily disabled for quick startup
from .core.enhanced_logging import enhanced_logger, RequestTrackingMiddleware
from .services.scheduler import ci_scheduler

# Agents will be lazy-loaded on first request to avoid startup delays

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
    enhanced_logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    enhanced_logger.info(f"Environment: {settings.environment}")
    
    # Initialize database connection pool
    try:
        # await startup_database_pool()
        enhanced_logger.info("✅ Database connection pool initialized")
    except Exception as e:
        enhanced_logger.error("❌ Failed to initialize database connection pool", exception=e)
        raise
    
    # Initialize database auth service
    try:
        # await startup_database_auth()
        enhanced_logger.info("✅ Database auth service initialized")
    except Exception as e:
        enhanced_logger.error("❌ Failed to initialize database auth service", exception=e)
        raise
    
    # Initialize authentication system
    try:
        await initialize_auth_system()
        enhanced_logger.info("✅ Authentication system initialized")
    except Exception as e:
        enhanced_logger.warning("⚠️ Failed to initialize authentication system", exception=e)
    
    # Start connection pool maintenance task
    maintenance_task = None
    try:
        # maintenance_task = asyncio.create_task(connection_pool_maintenance())
        enhanced_logger.info("✅ Connection pool maintenance started")
    except Exception as e:
        enhanced_logger.warning("⚠️ Failed to start connection pool maintenance", exception=e)
    
    # Perform legacy database health check
    try:
        db_health = db_config.health_check()
        if db_health["status"] == "healthy":
            enhanced_logger.info("✅ Legacy database connections healthy")
        else:
            enhanced_logger.warning(f"⚠️ Legacy database health check failed: {db_health}")
    except Exception as e:
        enhanced_logger.warning("⚠️ Legacy database health check error", exception=e)
    
    # Start monitoring services
    try:
        await start_monitoring()
        enhanced_logger.info("✅ Monitoring services started")
    except Exception as e:
        enhanced_logger.warning("⚠️ Failed to start monitoring", exception=e)
    
    # Start webhook retry processor
    try:
        await webhook_manager.start_retry_processor()
        enhanced_logger.info("✅ Webhook services started")
    except Exception as e:
        enhanced_logger.warning("⚠️ Failed to start webhooks", exception=e)
    
    # Start CI scheduler
    try:
        await ci_scheduler.start()
        enhanced_logger.info("✅ CI Scheduler started")
    except Exception as e:
        enhanced_logger.warning("⚠️ Failed to start CI Scheduler", exception=e)
    
    # Display AI provider configuration
    try:
        from .core.ai_utils import setup_ai_environment, check_provider_availability
        from .config.settings import get_settings
        
        ai_settings = get_settings()
        availability = check_provider_availability()
        
        enhanced_logger.info(f"🤖 AI Configuration:")
        enhanced_logger.info(f"   Primary Provider: {ai_settings.primary_ai_provider.upper()}")
        
        if availability.get('openai', False):
            enhanced_logger.info(f"   OpenAI: ✅ Available (Model: {ai_settings.openai_model})")
        else:
            enhanced_logger.info(f"   OpenAI: ❌ Not configured")
            
        if availability.get('gemini', False):
            enhanced_logger.info(f"   Gemini: ✅ Available (Model: {ai_settings.gemini_model})")
        else:
            enhanced_logger.info(f"   Gemini: ❌ Not configured")
            
        setup_ai_environment()
        
    except Exception as e:
        enhanced_logger.warning(f"⚠️ AI environment setup warning: {e}")
    
    enhanced_logger.info("🚀 CrediLinq AI Content Platform startup completed")
    
    yield
    
    # Shutdown
    enhanced_logger.info("🔄 Shutting down CrediLinq AI Content Platform")
    
    # Cancel maintenance task
    if maintenance_task:
        maintenance_task.cancel()
        try:
            await maintenance_task
        except asyncio.CancelledError:
            pass
    
    # Stop services in reverse order
    try:
        await stop_monitoring()
        enhanced_logger.info("✅ Monitoring services stopped")
    except Exception as e:
        enhanced_logger.warning("⚠️ Error stopping monitoring", exception=e)
    
    try:
        await webhook_manager.stop_retry_processor()
        enhanced_logger.info("✅ Webhook services stopped")
    except Exception as e:
        enhanced_logger.warning("⚠️ Error stopping webhooks", exception=e)
    
    try:
        await ci_scheduler.shutdown()
        enhanced_logger.info("✅ CI Scheduler stopped")
    except Exception as e:
        enhanced_logger.warning("⚠️ Error stopping CI Scheduler", exception=e)
    
    try:
        # await shutdown_database_auth()
        enhanced_logger.info("✅ Database auth service stopped")
    except Exception as e:
        enhanced_logger.warning("⚠️ Error stopping database auth service", exception=e)
    
    try:
        # await shutdown_database_pool()
        enhanced_logger.info("✅ Database connection pool stopped")
    except Exception as e:
        enhanced_logger.warning("⚠️ Error stopping database pool", exception=e)
    
    try:
        secure_db.close_pool()
        enhanced_logger.info("✅ Legacy database pool stopped")
    except Exception as e:
        enhanced_logger.warning("⚠️ Error stopping legacy database", exception=e)
    
    enhanced_logger.info("✅ CrediLinq AI Content Platform shutdown completed")

# Create enhanced FastAPI application
app = FastAPI(
    title="CrediLinq AI Content Platform API",
    description="""
## CrediLinq AI Content Platform API v2.0

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
app.add_middleware(ErrorHandlingMiddleware)  # Re-enabled - comprehensive error handling

# Add security middleware based on environment
security_config = SECURITY_CONFIGS.get(settings.environment, SECURITY_CONFIGS["production"])
validation_config = VALIDATION_CONFIGS.get(settings.environment, VALIDATION_CONFIGS["production"])

app.add_middleware(RequestValidationMiddleware, config=validation_config)  # Request validation and security
app.add_middleware(SecurityHeadersMiddleware, config=security_config)  # Security headers
app.add_middleware(ProductionSecurityMiddleware, config=security_config)  # Production security

app.add_middleware(RateLimitMiddleware)  # Rate limiting with sliding window algorithm
app.add_middleware(APIAnalyticsMiddleware)  # Re-enabled - request/response analytics  
app.add_middleware(AuthenticationMiddleware)  # Re-enabled - authentication handling
app.add_middleware(VersionCompatibilityMiddleware)  # Re-enabled - API versioning support
app.add_middleware(RequestTrackingMiddleware, logger=enhanced_logger)  # Enhanced request tracking and logging

# Add performance optimization middleware
performance_middleware = create_performance_middleware(
    compression_config=CompressionConfig(
        min_size=1000,
        compression_level=6,
        enable_gzip=True,
        enable_brotli=True,
        excluded_paths=['/health', '/ping', '/metrics', '/docs', '/redoc', '/openapi.json']
    ),
    cache_config=CacheConfig(
        default_ttl=300,  # 5 minutes
        cache_control_header="public, max-age=300",
        etag_enabled=True,
        excluded_paths=[
            '/api/v2/workflow', 
            '/api/v2/campaigns', 
            '/api/v2/competitor-intelligence/analyze'
        ]
    ),
    enable_compression=True,
    enable_caching=settings.enable_cache,
    enable_connection_optimization=True
)
app.add_middleware(type(performance_middleware), 
                   compression_config=performance_middleware.compression_middleware.config if hasattr(performance_middleware, 'compression_middleware') else None,
                   cache_config=performance_middleware.cache_middleware.config if hasattr(performance_middleware, 'cache_middleware') else None)

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
# Authentication routes (public - no auth required for login/register)
app.include_router(auth.router, prefix="/api/v2", tags=["authentication"])

app.include_router(blogs.router, prefix="/api/v2", tags=["blogs-v2"])
app.include_router(campaigns.router, prefix="/api/v2/campaigns", tags=["campaigns-v2"])  
app.include_router(analytics.router, prefix="/api/v2", tags=["analytics-v2"])
app.include_router(documents.router, prefix="/api/v2", tags=["documents-v2"])
app.include_router(webhooks_router, prefix="/api/v2", tags=["webhooks-v2"])
app.include_router(api_analytics.router, prefix="/api/v2", tags=["api-analytics-v2"])
app.include_router(content_repurposing.router, prefix="/api/v2/content", tags=["content-repurposing-v2"])
app.include_router(content_preview.router, prefix="/api/v2/content-preview", tags=["content-preview-v2"])
# Temporarily disabled due to missing ML dependencies on Railway
# app.include_router(competitor_intelligence.router, prefix="/api/v2", tags=["competitor-intelligence-v2"])
# app.include_router(content_briefs.router, prefix="/api/v2", tags=["content-briefs-v2"])
# app.include_router(content_deliverables.router, tags=["content-deliverables-v2"])  # Disabled - missing deps
app.include_router(images_debug.router, prefix="/api/v2", tags=["images-v2"])
# Settings routes (v2)
app.include_router(settings_router.router, prefix="/api/v2", tags=["settings-v2"])
# workflow_fixed is the main workflow implementation
app.include_router(workflow_fixed.router, prefix="/api/v2", tags=["workflow-fixed-v2"])
# Master Planner Agent workflow orchestration
app.include_router(workflow_orchestration.router, prefix="/api/v2", tags=["workflow-orchestration-v2"])
# Workflow monitoring and metrics
app.include_router(workflow_metrics.router, prefix="/api/v2", tags=["workflow-metrics-v2"])
# Content generation workflows
# app.include_router(content_workflows.router, prefix="/api/v2", tags=["content-workflows-v2"])
app.include_router(agents.router, prefix="/api/v2", tags=["agents-v2"])
app.include_router(review_workflow.router, tags=["review-workflow-v2"])  # Review workflow already has /api/v2 prefix

# V1 routes (deprecated, for backward compatibility)
app.include_router(blogs.router, prefix="/api/v1", tags=["blogs-v1"], deprecated=True)
app.include_router(campaigns.router, prefix="/api/v1", tags=["campaigns-v1"], deprecated=True)
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics-v1"], deprecated=True)

# Default routes (no version prefix) - use v2
app.include_router(blogs.router, prefix="/api", tags=["blogs"])
app.include_router(documents.router, prefix="/api", tags=["documents"])  # Move before campaigns to avoid routing conflict
app.include_router(campaigns.router, prefix="/api", tags=["campaigns"])
app.include_router(comments_router.router, prefix="/api", tags=["comments"])
app.include_router(suggestions_router.router, prefix="/api", tags=["suggestions"])
app.include_router(db_debug_router.router, prefix="/api", tags=["debug"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(webhooks_router, prefix="/api", tags=["webhooks"])
app.include_router(api_analytics.router, prefix="/api", tags=["api-analytics"])
app.include_router(content_repurposing.router, prefix="/api/content", tags=["content-repurposing"])
app.include_router(content_preview.router, prefix="/api/content-preview", tags=["content-preview"])
# app.include_router(content_briefs.router, prefix="/api", tags=["content-briefs"])
# Temporarily disabled due to missing ML dependencies on Railway
# app.include_router(competitor_intelligence.router, prefix="/api", tags=["competitor-intelligence"])
app.include_router(settings_router.router, prefix="/api", tags=["settings"])
app.include_router(review_workflow.router, tags=["review-workflow"])  # Review workflow routes

# Add a simple test route directly
@app.get("/api/images/test-direct")
async def test_images_direct():
    """Test endpoint to verify images routing works."""
    return {"message": "Images router is working directly!"}

# Include the debug images router
app.include_router(images_debug.router, prefix="/api/images", tags=["images-debug"])
# workflow_fixed handles all workflow functionality
app.include_router(workflow_fixed.router, prefix="/api", tags=["workflow-fixed"])
app.include_router(workflow_orchestration.router, prefix="/api", tags=["workflow-orchestration"])
# Workflow monitoring and metrics
app.include_router(workflow_metrics.router, prefix="/api", tags=["workflow-metrics"])

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