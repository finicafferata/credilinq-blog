"""
Custom exception classes for the CrediLinq Content Agent.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class CrediLinqException(Exception):
    """Base exception class for all CrediLinq Content Agent errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ========================================
# SECURITY EXCEPTIONS
# ========================================

class SecurityException(CrediLinqException):
    """Base class for security-related exceptions."""
    pass


class InputValidationError(SecurityException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str, value: Optional[str] = None):
        self.field = field
        self.value = value
        super().__init__(f"Input validation failed for '{field}': {message}")


class SQLInjectionAttempt(SecurityException):
    """Raised when potential SQL injection is detected."""
    
    def __init__(self, field: str, pattern: str):
        self.field = field
        self.pattern = pattern
        super().__init__(f"Potential SQL injection detected in '{field}': {pattern}")


class XSSAttempt(SecurityException):
    """Raised when potential XSS attack is detected."""
    
    def __init__(self, field: str, content: str):
        self.field = field
        self.content = content[:100] + "..." if len(content) > 100 else content
        super().__init__(f"Potential XSS attempt detected in '{field}': {self.content}")


class PathTraversalAttempt(SecurityException):
    """Raised when path traversal attack is detected."""
    
    def __init__(self, field: str, path: str):
        self.field = field
        self.path = path
        super().__init__(f"Path traversal attempt detected in '{field}': {path}")


class AuthenticationError(SecurityException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityException):
    """Raised when authorization fails."""
    pass


class RateLimitExceeded(SecurityException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window: str):
        self.limit = limit
        self.window = window
        super().__init__(f"Rate limit exceeded: {limit} requests per {window}")


# ========================================
# DATABASE EXCEPTIONS
# ========================================

class DatabaseException(CrediLinqException):
    """Base class for database-related exceptions."""
    pass


class DatabaseConnectionError(DatabaseException):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, retry_count: int = 0):
        self.retry_count = retry_count
        super().__init__(f"Database connection failed (attempt {retry_count}): {message}")


class DatabaseQueryError(DatabaseException):
    """Raised when database query fails."""
    
    def __init__(self, query: str, error: str):
        self.query = query
        self.error = error
        super().__init__(f"Database query failed: {error}")


class RecordNotFoundError(DatabaseException):
    """Raised when a required database record is not found."""
    
    def __init__(self, entity: str, identifier: str):
        self.entity = entity
        self.identifier = identifier
        super().__init__(f"{entity} not found with identifier: {identifier}")


class DuplicateRecordError(DatabaseException):
    """Raised when attempting to create a duplicate record."""
    
    def __init__(self, entity: str, field: str, value: str):
        self.entity = entity
        self.field = field
        self.value = value
        super().__init__(f"Duplicate {entity} found with {field}: {value}")


# ========================================
# AI AGENT EXCEPTIONS
# ========================================

class AgentException(CrediLinqException):
    """Base class for AI agent-related exceptions."""
    pass


class AgentExecutionError(AgentException):
    """Raised when agent execution fails."""
    
    def __init__(self, agent_type: str, task: str, error: str):
        self.agent_type = agent_type
        self.task = task
        self.error = error
        super().__init__(f"Agent '{agent_type}' failed executing '{task}': {error}")


class AgentTimeoutError(AgentException):
    """Raised when agent execution times out."""
    
    def __init__(self, agent_type: str, timeout_seconds: int):
        self.agent_type = agent_type
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Agent '{agent_type}' timed out after {timeout_seconds} seconds")


class AgentConfigurationError(AgentException):
    """Raised when agent configuration is invalid."""
    
    def __init__(self, agent_type: str, config_issue: str):
        self.agent_type = agent_type
        self.config_issue = config_issue
        super().__init__(f"Agent '{agent_type}' configuration error: {config_issue}")


class ContentGenerationError(AgentException):
    """Raised when content generation fails."""
    
    def __init__(self, content_type: str, reason: str):
        self.content_type = content_type
        self.reason = reason
        super().__init__(f"Failed to generate {content_type}: {reason}")


class WorkflowExecutionError(AgentException):
    """Raised when workflow execution fails."""
    
    def __init__(self, workflow_type: str, step: str, error: str):
        self.workflow_type = workflow_type
        self.step = step
        self.error = error
        super().__init__(f"Workflow '{workflow_type}' failed at step '{step}': {error}")


# ========================================
# API EXCEPTIONS
# ========================================

class APIException(CrediLinqException):
    """Base class for API-related exceptions."""
    pass


class InvalidRequestError(APIException):
    """Raised when API request is invalid."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class ResourceNotFoundError(APIException):
    """Raised when requested resource is not found."""
    
    def __init__(self, resource: str, identifier: str):
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} not found: {identifier}")


class ConflictError(APIException):
    """Raised when there's a conflict with the current state."""
    
    def __init__(self, message: str):
        super().__init__(message)


# ========================================
# EXTERNAL SERVICE EXCEPTIONS
# ========================================

class ExternalServiceException(CrediLinqException):
    """Base class for external service exceptions."""
    pass


class OpenAIServiceError(ExternalServiceException):
    """Raised when OpenAI API fails."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(f"OpenAI API error: {message}")


class SupabaseServiceError(ExternalServiceException):
    """Raised when Supabase service fails."""
    
    def __init__(self, message: str, operation: str):
        self.operation = operation
        super().__init__(f"Supabase {operation} error: {message}")


class SearchServiceError(ExternalServiceException):
    """Raised when search service (Tavily) fails."""
    
    def __init__(self, message: str, query: str):
        self.query = query
        super().__init__(f"Search service error for query '{query}': {message}")


# ========================================
# FILE HANDLING EXCEPTIONS
# ========================================

class FileException(CrediLinqException):
    """Base class for file-related exceptions."""
    pass


class FileUploadError(FileException):
    """Raised when file upload fails."""
    
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"File upload failed for '{filename}': {reason}")


class FileValidationError(FileException):
    """Raised when file validation fails."""
    
    def __init__(self, filename: str, issue: str):
        self.filename = filename
        self.issue = issue
        super().__init__(f"File validation failed for '{filename}': {issue}")


class FileSizeExceededError(FileException):
    """Raised when file size exceeds limit."""
    
    def __init__(self, filename: str, size_mb: float, limit_mb: int):
        self.filename = filename
        self.size_mb = size_mb
        self.limit_mb = limit_mb
        super().__init__(
            f"File '{filename}' size ({size_mb:.1f}MB) exceeds limit ({limit_mb}MB)"
        )


# ========================================
# HTTP EXCEPTION CONVERTERS
# ========================================

def convert_to_http_exception(error: CrediLinqException) -> HTTPException:
    """Convert CrediLinq exceptions to appropriate HTTP exceptions."""
    
    # Security exceptions -> 400 Bad Request or 403 Forbidden
    if isinstance(error, (InputValidationError, SQLInjectionAttempt, XSSAttempt, PathTraversalAttempt)):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, (AuthenticationError, AuthorizationError)):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, RateLimitExceeded):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    # Database exceptions -> 500 Internal Server Error or 404 Not Found
    if isinstance(error, RecordNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, DuplicateRecordError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, DatabaseException):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Database Error", "message": "A database error occurred"}
        )
    
    # API exceptions
    if isinstance(error, InvalidRequestError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, ResourceNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, ConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    # Agent exceptions -> 500 Internal Server Error
    if isinstance(error, AgentTimeoutError):
        return HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, AgentException):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Agent Error", "message": "An AI agent error occurred"}
        )
    
    # File exceptions
    if isinstance(error, FileSizeExceededError):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    if isinstance(error, FileException):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": error.__class__.__name__, "message": str(error)}
        )
    
    # External service exceptions -> 502 Bad Gateway
    if isinstance(error, ExternalServiceException):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "External Service Error", "message": "An external service error occurred"}
        )
    
    # Default -> 500 Internal Server Error
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"error": "Internal Error", "message": "An unexpected error occurred"}
    )