"""
API versioning system for CrediLinq platform.
Supports multiple API versions with backward compatibility and deprecation management.
"""

from typing import Dict, Any, Optional, Callable, List
from fastapi import FastAPI, APIRouter, Request, HTTPException, Depends
from fastapi.routing import APIRoute
from enum import Enum
import re
from datetime import datetime, timedelta
from pydantic import BaseModel

class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"
    
class VersionStatus(str, Enum):
    """API version lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    RETIRED = "retired"

class VersionInfo(BaseModel):
    """API version information model."""
    version: APIVersion
    status: VersionStatus
    release_date: datetime
    sunset_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    migration_guide_url: Optional[str] = None
    changelog_url: Optional[str] = None
    breaking_changes: List[str] = []
    new_features: List[str] = []

class APIVersionManager:
    """Manages API versioning, deprecation, and compatibility."""
    
    def __init__(self):
        self.versions: Dict[APIVersion, VersionInfo] = {
            APIVersion.V1: VersionInfo(
                version=APIVersion.V1,
                status=VersionStatus.DEPRECATED,
                release_date=datetime(2024, 1, 1),
                sunset_date=datetime(2025, 12, 31),
                retirement_date=datetime(2026, 6, 30),
                migration_guide_url="https://docs.credilinq.com/migration/v1-to-v2",
                breaking_changes=[
                    "Authentication changed from API key to OAuth2",
                    "Blog post creation response format updated",
                    "Error response structure standardized"
                ]
            ),
            APIVersion.V2: VersionInfo(
                version=APIVersion.V2,
                status=VersionStatus.ACTIVE,
                release_date=datetime(2025, 1, 15),
                new_features=[
                    "Enhanced webhook support",
                    "Real-time analytics",
                    "Improved rate limiting",
                    "Comprehensive API documentation"
                ]
            )
        }
        self.default_version = APIVersion.V2
    
    def get_version_info(self, version: APIVersion) -> VersionInfo:
        """Get information about a specific API version."""
        return self.versions.get(version)
    
    def get_all_versions(self) -> Dict[APIVersion, VersionInfo]:
        """Get information about all API versions."""
        return self.versions
    
    def is_version_supported(self, version: APIVersion) -> bool:
        """Check if an API version is still supported."""
        version_info = self.versions.get(version)
        if not version_info:
            return False
        return version_info.status in [VersionStatus.ACTIVE, VersionStatus.DEPRECATED]
    
    def get_deprecation_warning(self, version: APIVersion) -> Optional[str]:
        """Get deprecation warning message for a version."""
        version_info = self.versions.get(version)
        if not version_info or version_info.status != VersionStatus.DEPRECATED:
            return None
            
        warning = f"API version {version} is deprecated"
        if version_info.sunset_date:
            warning += f" and will be sunset on {version_info.sunset_date.strftime('%Y-%m-%d')}"
        if version_info.migration_guide_url:
            warning += f". Migration guide: {version_info.migration_guide_url}"
        
        return warning

# Global version manager instance
version_manager = APIVersionManager()

def extract_version_from_request(request: Request) -> APIVersion:
    """Extract API version from request headers, path, or query parameters."""
    
    # Check version in path (e.g., /v2/api/blogs)
    path_version_match = re.match(r'^/(v\d+)/', request.url.path)
    if path_version_match:
        try:
            return APIVersion(path_version_match.group(1))
        except ValueError:
            pass
    
    # Check version in Accept header (e.g., application/vnd.credilinq.v2+json)
    accept_header = request.headers.get("accept", "")
    accept_version_match = re.search(r'application/vnd\.credilinq\.(v\d+)\+json', accept_header)
    if accept_version_match:
        try:
            return APIVersion(accept_version_match.group(1))
        except ValueError:
            pass
    
    # Check version in custom header
    api_version_header = request.headers.get("api-version")
    if api_version_header:
        try:
            return APIVersion(api_version_header)
        except ValueError:
            pass
    
    # Check version in query parameter
    version_param = request.query_params.get("version")
    if version_param:
        try:
            return APIVersion(version_param)
        except ValueError:
            pass
    
    # Return default version
    return version_manager.default_version

async def get_api_version(request: Request) -> APIVersion:
    """Dependency to get and validate API version from request."""
    version = extract_version_from_request(request)
    
    if not version_manager.is_version_supported(version):
        raise HTTPException(
            status_code=400,
            detail=f"API version {version} is not supported"
        )
    
    return version

class VersionedAPIRouter(APIRouter):
    """Enhanced APIRouter with version-specific routing and deprecation handling."""
    
    def __init__(self, version: APIVersion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version
        self.version_info = version_manager.get_version_info(version)
    
    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        """Add API route with version-specific handling."""
        
        # Create version-aware endpoint wrapper
        async def version_aware_endpoint(*args, **endpoint_kwargs):
            # Add deprecation warning headers if needed
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request and self.version_info.status == VersionStatus.DEPRECATED:
                warning_message = version_manager.get_deprecation_warning(self.version)
                # We'll add this to response headers in middleware
                setattr(request.state, 'deprecation_warning', warning_message)
            
            return await endpoint(*args, **endpoint_kwargs)
        
        # Preserve original endpoint metadata
        version_aware_endpoint.__name__ = endpoint.__name__
        version_aware_endpoint.__doc__ = endpoint.__doc__
        
        super().add_api_route(path, version_aware_endpoint, **kwargs)

def create_versioned_app() -> FastAPI:
    """Create FastAPI app with version-specific routers."""
    
    app = FastAPI(
        title="CrediLinq AI Content Platform API",
        description="Multi-version API with comprehensive content management capabilities",
        version="2.0.0"
    )
    
    # Version information endpoint
    @app.get("/versions")
    async def get_version_info():
        """Get information about all supported API versions."""
        return {
            "current_version": version_manager.default_version,
            "supported_versions": {
                version: {
                    "status": info.status,
                    "release_date": info.release_date.isoformat(),
                    "sunset_date": info.sunset_date.isoformat() if info.sunset_date else None,
                    "migration_guide": info.migration_guide_url,
                    "breaking_changes": info.breaking_changes,
                    "new_features": info.new_features
                }
                for version, info in version_manager.get_all_versions().items()
            }
        }
    
    return app

class VersionCompatibilityMiddleware:
    """Middleware to handle version compatibility and deprecation warnings."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract version from request
        request = Request(scope, receive)
        api_version = extract_version_from_request(request)
        
        # Check if version is supported
        if not version_manager.is_version_supported(api_version):
            response = {
                "error": "UNSUPPORTED_API_VERSION",
                "message": f"API version {api_version} is not supported",
                "supported_versions": list(version_manager.get_all_versions().keys()),
                "current_version": version_manager.default_version
            }
            
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"x-api-version", api_version.encode()],
                ]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps(response).encode()
            })
            return
        
        # Store version in request state
        scope["state"] = getattr(scope, "state", {})
        scope["state"]["api_version"] = api_version
        
        # Custom send wrapper to add version headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                
                # Add version header
                headers.append([b"x-api-version", api_version.encode()])
                
                # Add deprecation warning if applicable
                version_info = version_manager.get_version_info(api_version)
                if version_info and version_info.status == VersionStatus.DEPRECATED:
                    warning = version_manager.get_deprecation_warning(api_version)
                    if warning:
                        headers.append([b"warning", f'299 - "Deprecated API: {warning}"'.encode()])
                        headers.append([b"sunset", version_info.sunset_date.strftime('%a, %d %b %Y %H:%M:%S GMT').encode()])
                
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

# Version-specific route decorators
def versioned_route(version: APIVersion, **route_kwargs):
    """Decorator to create version-specific routes."""
    def decorator(func):
        func._api_version = version
        func._route_kwargs = route_kwargs
        return func
    return decorator

def deprecated_route(sunset_date: datetime, migration_url: str = None):
    """Decorator to mark routes as deprecated."""
    def decorator(func):
        func._deprecated = True
        func._sunset_date = sunset_date
        func._migration_url = migration_url
        return func
    return decorator

# Version compatibility helpers
class VersionCompatibility:
    """Utilities for handling version compatibility."""
    
    @staticmethod
    def transform_response_v1_to_v2(v1_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1 response format to v2 format."""
        # Example transformation logic
        if "data" in v1_response:
            return {
                "success": True,
                "data": v1_response["data"],
                "timestamp": datetime.utcnow().isoformat(),
                "version": "v2"
            }
        return v1_response
    
    @staticmethod
    def transform_request_v1_to_v2(v1_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1 request format to v2 format."""
        # Example transformation logic
        if "api_key" in v1_request:
            # Transform old API key auth to new format
            auth_header = f"Bearer {v1_request.pop('api_key')}"
            v1_request["_auth_header"] = auth_header
        
        return v1_request

# Import json for the middleware
import json