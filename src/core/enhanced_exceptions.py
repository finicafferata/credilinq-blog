"""
Enhanced error handling and response models for the CrediLinQ API.
Provides standardized error responses, comprehensive logging, and error tracking.
"""

import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .monitoring import metrics
from ..config.settings import settings

class ErrorCategory(str, Enum):
    """Categories of API errors for better classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    RATE_LIMIT = "rate_limit"
    NOT_FOUND = "not_found"

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class APIError(BaseModel):
    """Standardized API error response model."""
    success: bool = False
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(..., description="Unique request identifier for debugging")
    timestamp: str = Field(..., description="ISO timestamp when error occurred")
    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(ErrorSeverity.MEDIUM, description="Error severity")
    documentation_url: Optional[str] = Field(None, description="Link to relevant documentation")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions to fix the error")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error_code": "BLOG_NOT_FOUND",
                "message": "The requested blog post could not be found",
                "details": {
                    "blog_id": "123e4567-e89b-12d3-a456-426614174000",
                    "requested_at": "2025-01-15T10:30:00Z"
                },
                "request_id": "req_abc123",
                "timestamp": "2025-01-15T10:30:00Z",
                "category": "not_found",
                "severity": "medium",
                "suggestions": [
                    "Verify the blog ID is correct",
                    "Check if the blog post has been deleted"
                ]
            }
        }

class ValidationError(BaseModel):
    """Validation error details."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(None, description="Invalid value that was provided")
    constraint: Optional[str] = Field(None, description="Validation constraint that was violated")

class APIValidationError(APIError):
    """Validation-specific error response."""
    validation_errors: List[ValidationError] = Field(..., description="List of validation errors")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "validation_errors": [
                    {
                        "field": "title",
                        "message": "Title must be between 1 and 200 characters",
                        "value": "",
                        "constraint": "min_length=1, max_length=200"
                    }
                ],
                "request_id": "req_abc123",
                "timestamp": "2025-01-15T10:30:00Z",
                "category": "validation",
                "severity": "medium"
            }
        }

class CustomHTTPException(HTTPException):
    """Enhanced HTTP exception with additional metadata."""
    
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggestions: Optional[List[str]] = None,
        **kwargs
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        
        # Set documentation URL based on error code
        self.documentation_url = f"https://docs.credilinq.com/errors/{error_code.lower()}"
        
        super().__init__(status_code=status_code, detail=message, **kwargs)

# Pre-defined common exceptions
class BlogNotFoundError(CustomHTTPException):
    """Blog post not found error."""
    
    def __init__(self, blog_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="BLOG_NOT_FOUND",
            message=f"Blog post with ID '{blog_id}' not found",
            details={"blog_id": blog_id},
            category=ErrorCategory.NOT_FOUND,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Verify the blog ID is correct",
                "Check if the blog post has been deleted",
                "Ensure you have permission to access this blog"
            ]
        )

class CampaignNotFoundError(CustomHTTPException):
    """Campaign not found error."""
    
    def __init__(self, campaign_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="CAMPAIGN_NOT_FOUND",
            message=f"Campaign with ID '{campaign_id}' not found",
            details={"campaign_id": campaign_id},
            category=ErrorCategory.NOT_FOUND,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Verify the campaign ID is correct",
                "Check if the campaign has been deleted"
            ]
        )

class InsufficientPermissionsError(CustomHTTPException):
    """Insufficient permissions error."""
    
    def __init__(self, required_permission: str, current_permissions: List[str] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="INSUFFICIENT_PERMISSIONS",
            message="You don't have sufficient permissions to perform this action",
            details={
                "required_permission": required_permission,
                "current_permissions": current_permissions or []
            },
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Contact your administrator to request additional permissions",
                "Use an API key with appropriate scopes",
                "Ensure you're authenticated with the correct account"
            ]
        )

class RateLimitExceededError(CustomHTTPException):
    """Rate limit exceeded error."""
    
    def __init__(self, limit: int, window: str, retry_after: int = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            message=f"Rate limit exceeded: {limit} requests per {window}",
            details={
                "limit": limit,
                "window": window,
                "retry_after_seconds": retry_after
            },
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                f"Wait {retry_after} seconds before making another request" if retry_after else "Reduce request frequency",
                "Consider upgrading your API plan for higher limits",
                "Implement exponential backoff in your client"
            ]
        )

class ExternalServiceError(CustomHTTPException):
    """External service error."""
    
    def __init__(self, service_name: str, error_details: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="EXTERNAL_SERVICE_ERROR",
            message=f"External service '{service_name}' is currently unavailable",
            details={
                "service": service_name,
                "error": error_details
            },
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Try again in a few moments",
                "Check our status page for service updates",
                "Contact support if the issue persists"
            ]
        )

class ValidationError(CustomHTTPException):
    """Request validation error."""
    
    def __init__(self, validation_errors: List[Dict[str, Any]]):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_errors": validation_errors},
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            suggestions=[
                "Check the request format and required fields",
                "Ensure all values meet the specified constraints",
                "Review the API documentation for correct usage"
            ]
        )

class ErrorTracker:
    """Tracks and analyzes API errors for monitoring and debugging."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_details: List[Dict[str, Any]] = []
        self.max_error_history = 1000
    
    def track_error(
        self,
        error: CustomHTTPException,
        request: Request,
        request_id: str,
        user_id: Optional[str] = None
    ):
        """Track an error occurrence."""
        
        # Update error counts
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
        
        # Store detailed error information
        error_detail = {
            "request_id": request_id,
            "error_code": error.error_code,
            "message": error.message,
            "status_code": error.status_code,
            "category": error.category,
            "severity": error.severity,
            "endpoint": request.url.path,
            "method": request.method,
            "user_id": user_id,
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.utcnow().isoformat(),
            "details": error.details
        }
        
        self.error_details.append(error_detail)
        
        # Keep only recent errors
        if len(self.error_details) > self.max_error_history:
            self.error_details = self.error_details[-self.max_error_history:]
        
        # Update metrics
        metrics.increment_counter(
            "api.errors",
            tags={
                "error_code": error.error_code,
                "category": error.category,
                "severity": error.severity,
                "status_code": str(error.status_code)
            }
        )
        
        # Log high severity errors
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"High severity error: {error.error_code} - {error.message}",
                extra=error_detail
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and insights."""
        recent_errors = [
            error for error in self.error_details
            if datetime.fromisoformat(error["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Calculate error rates by category
        category_counts = {}
        for error in recent_errors:
            category = error["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Find most common errors
        error_code_counts = {}
        for error in recent_errors:
            code = error["error_code"]
            error_code_counts[code] = error_code_counts.get(code, 0) + 1
        
        most_common_errors = sorted(
            error_code_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_errors_24h": len(recent_errors),
            "error_rate_per_hour": len(recent_errors) / 24,
            "errors_by_category": category_counts,
            "most_common_errors": dict(most_common_errors),
            "high_severity_errors": len([
                e for e in recent_errors 
                if e["severity"] in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            ])
        }

# Global error tracker
error_tracker = ErrorTracker()

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling and response formatting."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            return response
            
        except CustomHTTPException as exc:
            # Track the error
            user_id = getattr(request.state, 'user_id', None)
            error_tracker.track_error(exc, request, request_id, user_id)
            
            # Create standardized error response
            error_response = APIError(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
                category=exc.category,
                severity=exc.severity,
                documentation_url=exc.documentation_url,
                suggestions=exc.suggestions
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id}
            )
            
        except HTTPException as exc:
            # Convert standard HTTP exceptions to our format
            custom_exc = CustomHTTPException(
                status_code=exc.status_code,
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                category=self._categorize_http_error(exc.status_code)
            )
            
            user_id = getattr(request.state, 'user_id', None)
            error_tracker.track_error(custom_exc, request, request_id, user_id)
            
            error_response = APIError(
                error_code=custom_exc.error_code,
                message=custom_exc.message,
                details=custom_exc.details,
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
                category=custom_exc.category,
                severity=custom_exc.severity
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id}
            )
            
        except Exception as exc:
            # Handle unexpected errors
            import logging
            logger = logging.getLogger(__name__)
            
            # Log the full traceback for debugging
            logger.error(
                f"Unhandled exception in request {request_id}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "user_id": getattr(request.state, 'user_id', None)
                }
            )
            
            # Create generic error response (don't expose internal details in production)
            if settings.debug:
                error_message = str(exc)
                error_details = {"traceback": traceback.format_exc()}
            else:
                error_message = "An internal error occurred"
                error_details = None
            
            custom_exc = CustomHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="INTERNAL_SERVER_ERROR",
                message=error_message,
                details=error_details,
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
            user_id = getattr(request.state, 'user_id', None)
            error_tracker.track_error(custom_exc, request, request_id, user_id)
            
            error_response = APIError(
                error_code=custom_exc.error_code,
                message=custom_exc.message,
                details=custom_exc.details,
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
                category=custom_exc.category,
                severity=custom_exc.severity,
                suggestions=[
                    "This is likely a temporary issue. Please try again.",
                    "If the problem persists, contact support with this request ID."
                ]
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id}
            )
    
    def _categorize_http_error(self, status_code: int) -> ErrorCategory:
        """Categorize HTTP errors."""
        if status_code == 401:
            return ErrorCategory.AUTHENTICATION
        elif status_code == 403:
            return ErrorCategory.AUTHORIZATION
        elif status_code == 404:
            return ErrorCategory.NOT_FOUND
        elif status_code == 422:
            return ErrorCategory.VALIDATION
        elif status_code == 429:
            return ErrorCategory.RATE_LIMIT
        elif 400 <= status_code < 500:
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.SYSTEM

# Helper functions for creating common errors

def create_validation_error(errors: List[Dict[str, Any]]) -> CustomHTTPException:
    """Create a validation error from FastAPI validation errors."""
    formatted_errors = []
    
    for error in errors:
        formatted_errors.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", "Validation failed"),
            "type": error.get("type", "validation_error"),
            "value": error.get("input")
        })
    
    return ValidationError(formatted_errors)

def create_not_found_error(resource_type: str, resource_id: str) -> CustomHTTPException:
    """Create a generic not found error."""
    return CustomHTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        error_code=f"{resource_type.upper()}_NOT_FOUND",
        message=f"{resource_type.title()} with ID '{resource_id}' not found",
        details={f"{resource_type}_id": resource_id},
        category=ErrorCategory.NOT_FOUND
    )

def create_permission_error(action: str, resource: str) -> CustomHTTPException:
    """Create a permission denied error."""
    return CustomHTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        error_code="PERMISSION_DENIED",
        message=f"You don't have permission to {action} {resource}",
        details={"action": action, "resource": resource},
        category=ErrorCategory.AUTHORIZATION,
        severity=ErrorSeverity.HIGH
    )

# Import required modules
from datetime import timedelta