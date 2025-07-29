"""
Enhanced API documentation configuration for FastAPI with comprehensive Swagger UI customization.
Provides detailed API documentation, examples, and interactive testing capabilities.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json

# Custom OpenAPI schema configuration
def custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Generate enhanced OpenAPI schema with comprehensive documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="CrediLinQ AI Content Platform API",
        version="2.0.0",
        description="""
## CrediLinQ AI Content Platform API

A comprehensive AI-powered content management and marketing automation platform.

### Features
- **AI Content Generation**: Automated blog post creation using advanced AI agents
- **Campaign Management**: Marketing campaign orchestration and automation  
- **Performance Analytics**: Detailed insights into content performance and agent metrics
- **Knowledge Base**: Document management with vector search capabilities
- **Webhook Integration**: Real-time notifications for third-party systems
- **Rate Limiting**: Built-in API protection and usage monitoring

### Authentication
All API endpoints require authentication using API keys or OAuth tokens.
Include your API key in the `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY
```

### Rate Limits
- **Free Tier**: 100 requests/hour, 1000 requests/day
- **Pro Tier**: 1000 requests/hour, 10000 requests/day  
- **Enterprise**: Custom limits

### Webhooks
Configure webhooks to receive real-time notifications:
- Blog post completion events
- Campaign status updates
- Performance threshold alerts

### Error Handling
All endpoints return standardized error responses:
```json
{
    "error": "error_code",
    "message": "Human readable description",
    "details": {...},
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_123456"
}
```

### SDKs and Libraries
- **Python**: `pip install credilinq-python`
- **JavaScript**: `npm install @credilinq/api-client`
- **curl examples**: Available in each endpoint documentation
        """,
        routes=app.routes,
        servers=[
            {
                "url": "https://api.credilinq.com",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.credilinq.com", 
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ],
        tags=[
            {
                "name": "blogs",
                "description": "Blog post management and AI content generation",
                "externalDocs": {
                    "description": "Blog API Guide",
                    "url": "https://docs.credilinq.com/api/blogs"
                }
            },
            {
                "name": "campaigns", 
                "description": "Marketing campaign orchestration and automation",
                "externalDocs": {
                    "description": "Campaign API Guide", 
                    "url": "https://docs.credilinq.com/api/campaigns"
                }
            },
            {
                "name": "analytics",
                "description": "Performance metrics and business intelligence",
                "externalDocs": {
                    "description": "Analytics API Guide",
                    "url": "https://docs.credilinq.com/api/analytics"
                }
            },
            {
                "name": "webhooks",
                "description": "Real-time event notifications and integrations",
                "externalDocs": {
                    "description": "Webhook Integration Guide",
                    "url": "https://docs.credilinq.com/webhooks"
                }
            },
            {
                "name": "health",
                "description": "System health monitoring and diagnostics"
            },
            {
                "name": "auth",
                "description": "Authentication and authorization management"
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "API key authentication. Use format: `Bearer YOUR_API_KEY`"
        },
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "https://auth.credilinq.com/oauth/authorize",
                    "tokenUrl": "https://auth.credilinq.com/oauth/token",
                    "scopes": {
                        "blogs:read": "Read blog posts",
                        "blogs:write": "Create and edit blog posts",
                        "campaigns:read": "Read campaigns",
                        "campaigns:write": "Create and manage campaigns",
                        "analytics:read": "Access analytics data",
                        "webhooks:manage": "Manage webhook subscriptions"
                    }
                }
            }
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"OAuth2": ["blogs:read", "campaigns:read", "analytics:read"]}
    ]
    
    # Add custom extensions
    openapi_schema["x-logo"] = {
        "url": "https://credilinq.com/assets/logo.png",
        "altText": "CrediLinQ Logo"
    }
    
    # Add contact and license information
    openapi_schema["info"]["contact"] = {
        "name": "CrediLinQ API Support",
        "url": "https://support.credilinq.com",
        "email": "api-support@credilinq.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "Commercial License",
        "url": "https://credilinq.com/terms"
    }
    
    # Add example servers and environments
    openapi_schema["info"]["x-api-versions"] = {
        "v1": {
            "status": "deprecated",
            "sunset_date": "2025-12-31",
            "migration_guide": "https://docs.credilinq.com/migration/v1-to-v2"
        },
        "v2": {
            "status": "current",
            "release_date": "2025-01-15"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def configure_api_docs(app: FastAPI):
    """Configure enhanced API documentation with custom styling and features."""
    
    # Custom Swagger UI configuration
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Interactive API Documentation",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.3/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.10.3/swagger-ui.css",
            swagger_ui_parameters={
                "deepLinking": True,
                "displayRequestDuration": True,
                "docExpansion": "none",
                "operationsSorter": "method",
                "filter": True,
                "showExtensions": True,
                "showCommonExtensions": True,
                "defaultModelsExpandDepth": 2,
                "defaultModelExpandDepth": 2,
                "displayOperationId": True,
                "tryItOutEnabled": True,
                "persistAuthorization": True,
                "layout": "BaseLayout",
                "supportedSubmitMethods": ["get", "post", "put", "delete", "patch"],
                "validatorUrl": None,  # Disable validator badge
                "plugins": [
                    "SwaggerUIBundle.plugins.DownloadUrl"
                ],
                "presets": [
                    "SwaggerUIBundle.presets.apis",
                    "SwaggerUIStandalonePreset"
                ]
            }
        )
    
    # Enhanced ReDoc documentation
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - API Reference",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js",
            redoc_favicon_url="https://credilinq.com/favicon.ico",
            with_google_fonts=True
        )
    
    # API specification download endpoints
    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi_spec():
        return custom_openapi_schema(app)
    
    @app.get("/openapi.yaml", include_in_schema=False)
    async def get_openapi_yaml():
        import yaml
        spec = custom_openapi_schema(app)
        return HTMLResponse(
            content=yaml.dump(spec, default_flow_style=False),
            media_type="application/x-yaml",
            headers={"Content-Disposition": "attachment; filename=openapi.yaml"}
        )
    
    # API status and version information
    @app.get("/api/status", include_in_schema=False)
    async def api_status():
        return {
            "api_name": app.title,
            "version": app.version,
            "status": "operational",
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi_json": "/openapi.json",
                "openapi_yaml": "/openapi.yaml"
            },
            "support": {
                "email": "api-support@credilinq.com",
                "docs": "https://docs.credilinq.com",
                "status_page": "https://status.credilinq.com"
            }
        }


# Enhanced response models for better documentation
class APIResponse:
    """Base response model with common fields."""
    
    @staticmethod
    def success_model(data_model=None):
        """Generate success response model."""
        from pydantic import BaseModel
        from typing import Optional, Any
        
        class SuccessResponse(BaseModel):
            success: bool = True
            data: Optional[Any] = None
            message: Optional[str] = None
            timestamp: str
            request_id: str
            
            class Config:
                schema_extra = {
                    "example": {
                        "success": True,
                        "data": {"id": "123", "title": "Example"},
                        "message": "Operation completed successfully",
                        "timestamp": "2025-01-15T10:30:00Z",
                        "request_id": "req_abc123"
                    }
                }
        
        return SuccessResponse
    
    @staticmethod
    def error_model():
        """Generate error response model."""
        from pydantic import BaseModel
        from typing import Optional, Dict, Any
        
        class ErrorResponse(BaseModel):
            success: bool = False
            error: str
            message: str
            details: Optional[Dict[str, Any]] = None
            timestamp: str
            request_id: str
            
            class Config:
                schema_extra = {
                    "example": {
                        "success": False,
                        "error": "VALIDATION_ERROR",
                        "message": "The provided data is invalid",
                        "details": {
                            "field": "title",
                            "issue": "Title must be between 1 and 200 characters"
                        },
                        "timestamp": "2025-01-15T10:30:00Z",
                        "request_id": "req_abc123"
                    }
                }
        
        return ErrorResponse
    
    @staticmethod
    def paginated_model(item_model):
        """Generate paginated response model."""
        from pydantic import BaseModel
        from typing import List, Optional
        
        class PaginatedResponse(BaseModel):
            items: List[item_model]
            total: int
            page: int
            page_size: int
            has_next: bool
            has_previous: bool
            next_page: Optional[str] = None
            previous_page: Optional[str] = None
            
            class Config:
                schema_extra = {
                    "example": {
                        "items": [],
                        "total": 150,
                        "page": 2,
                        "page_size": 50,
                        "has_next": True,
                        "has_previous": True,
                        "next_page": "/api/blogs?page=3&limit=50",
                        "previous_page": "/api/blogs?page=1&limit=50"
                    }
                }
        
        return PaginatedResponse


# API Examples for documentation
API_EXAMPLES = {
    "blog_create": {
        "summary": "Create a new blog post",
        "description": "Generate a new blog post using AI with specific company context",
        "value": {
            "title": "How AI is Transforming Content Marketing in 2025",
            "company_context": "CrediLinQ is a fintech company specializing in AI-powered business solutions",
            "content_type": "blog",
            "target_audience": "marketing professionals",
            "keywords": ["AI", "content marketing", "automation"],
            "tone": "professional",
            "length": "medium"
        }
    },
    "campaign_create": {
        "summary": "Create a marketing campaign",
        "description": "Launch a multi-channel marketing campaign with automated content generation",
        "value": {
            "name": "Q1 2025 Product Launch",
            "blog_id": "550e8400-e29b-41d4-a716-446655440000",
            "channels": ["linkedin", "twitter", "email"],
            "schedule": "2025-02-01T09:00:00Z",
            "budget": 5000,
            "target_metrics": {
                "impressions": 100000,
                "engagement_rate": 0.05,
                "conversions": 200
            }
        }
    }
}


# Custom middleware for API documentation enhancement
class APIDocsMiddleware:
    """Middleware to enhance API documentation with usage examples."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # Add custom headers for API documentation
        if scope["type"] == "http" and scope["path"] in ["/docs", "/redoc"]:
            # You can add custom logic here for documentation enhancement
            pass
        
        return await self.app(scope, receive, send)