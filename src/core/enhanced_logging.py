"""
Enhanced logging system with request tracking, error logging, and performance monitoring.
Provides comprehensive logging capabilities for debugging and monitoring.
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prisma.enums import LogLevel
    from .database_pool import connection_pool
    PRISMA_AVAILABLE = True
except ImportError:
    PRISMA_AVAILABLE = False
    LogLevel = None

from ..config.settings import settings

# Context variable for request tracking
request_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('request_context', default=None)

@dataclass
class RequestContext:
    """Request context for tracking requests across the application."""
    request_id: str
    method: str
    endpoint: str
    user_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    auth_type: Optional[str] = None
    start_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.start_time == 0.0:
            self.start_time = time.time()

class EnhancedLogger:
    """Enhanced logger with request tracking and database persistence."""
    
    def __init__(self):
        self.logger = logging.getLogger("credilinq.enhanced")
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Create file handler if configured
        if settings.log_file_path:
            try:
                import os
                os.makedirs(os.path.dirname(settings.log_file_path), exist_ok=True)
                file_handler = logging.FileHandler(settings.log_file_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to create file handler: {e}")
        
        self.logger.addHandler(console_handler)
        self.logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    def _get_context_info(self) -> Dict[str, Any]:
        """Get current request context information."""
        context = request_context.get()
        if context:
            return {
                'request_id': context.get('request_id', 'unknown'),
                'method': context.get('method'),
                'endpoint': context.get('endpoint'),
                'user_id': context.get('user_id'),
                'auth_type': context.get('auth_type')
            }
        return {'request_id': 'no-context'}
    
    def _log_with_context(self, level: str, message: str, extra_data: Dict[str, Any] = None):
        """Log message with request context."""
        context_info = self._get_context_info()
        
        # Prepare log data
        log_data = {
            'message': message,
            'context': context_info,
            **(extra_data or {})
        }
        
        # Log to console/file
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra=context_info)
        
        # Log to database if available and enabled
        if PRISMA_AVAILABLE and level.upper() in ['ERROR', 'CRITICAL', 'WARNING']:
            asyncio.create_task(self._log_to_database(level, message, log_data))
    
    async def _log_to_database(self, level: str, message: str, log_data: Dict[str, Any]):
        """Log error to database for persistent storage."""
        try:
            db_client = await connection_pool.get_prisma_client()
            context = log_data.get('context', {})
            
            # Map log level to Prisma enum
            prisma_level = getattr(LogLevel, level.lower(), LogLevel.info)
            
            await db_client.errorlog.create(
                data={
                    'requestId': context.get('request_id'),
                    'level': prisma_level,
                    'message': message,
                    'context': log_data,
                    'userId': context.get('user_id'),
                    'endpoint': context.get('endpoint'),
                    'method': context.get('method'),
                    'userAgent': log_data.get('user_agent'),
                    'ipAddress': log_data.get('ip_address'),
                    'stackTrace': log_data.get('stack_trace')
                }
            )
        except Exception as e:
            # Fallback to console logging if database fails
            self.logger.error(f"Failed to log to database: {e}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context('debug', message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context('info', message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context('warning', message, kwargs)
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message with optional exception details."""
        extra_data = kwargs.copy()
        
        if exception:
            extra_data.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'stack_trace': traceback.format_exc()
            })
        
        self._log_with_context('error', message, extra_data)
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """Log critical message with optional exception details."""
        extra_data = kwargs.copy()
        
        if exception:
            extra_data.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'stack_trace': traceback.format_exc()
            })
        
        self._log_with_context('critical', message, extra_data)

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking requests and adding logging context."""
    
    def __init__(self, app, logger: EnhancedLogger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract request information
        user_agent = request.headers.get('user-agent', 'unknown')
        ip_address = self._get_client_ip(request)
        
        # Create request context
        context = RequestContext(
            request_id=request_id,
            method=request.method,
            endpoint=str(request.url.path),
            user_agent=user_agent,
            ip_address=ip_address,
            start_time=start_time
        )
        
        # Set context for this request
        token = request_context.set(asdict(context))
        
        try:
            # Process request
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Update context with response info
            context_dict = request_context.get()
            if context_dict:
                context_dict['status_code'] = response.status_code
                context_dict['duration'] = duration
            
            # Log request completion
            self.logger.info(
                f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s",
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                request_size=request.headers.get('content-length', 0),
                response_size=response.headers.get('content-length', 0)
            )
            
            # Log to database
            if PRISMA_AVAILABLE:
                asyncio.create_task(self._log_request_to_database(context_dict, response))
            
            # Add request ID to response headers for debugging
            response.headers["x-request-id"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.logger.error(
                f"{request.method} {request.url.path} - ERROR - {duration:.3f}s",
                exception=e,
                duration_ms=round(duration * 1000, 2)
            )
            
            raise
        finally:
            # Reset context
            request_context.reset(token)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (common in production with load balancers)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return getattr(request.client, 'host', 'unknown')
    
    async def _log_request_to_database(self, context: Dict[str, Any], response: Response):
        """Log request details to database."""
        try:
            db_client = await connection_pool.get_prisma_client()
            
            await db_client.requestlog.create(
                data={
                    'requestId': context['request_id'],
                    'method': context['method'],
                    'endpoint': context['endpoint'],
                    'statusCode': context.get('status_code', 500),
                    'duration': int(context.get('duration', 0) * 1000),  # Convert to milliseconds
                    'userId': context.get('user_id'),
                    'userAgent': context.get('user_agent'),
                    'ipAddress': context.get('ip_address'),
                    'authType': context.get('auth_type'),
                    'requestSize': context.get('request_size'),
                    'responseSize': context.get('response_size'),
                    'queryParams': context.get('query_params'),
                    'headers': context.get('headers'),
                    'errorMessage': context.get('error_message')
                }
            )
        except Exception as e:
            # Don't fail the request if logging fails
            print(f"Failed to log request to database: {e}")

def track_function_performance(func_name: str = None):
    """Decorator to track function performance and log slow operations."""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log slow operations (>1 second)
                if duration > 1.0:
                    enhanced_logger.warning(
                        f"Slow operation detected: {name} took {duration:.3f}s",
                        function_name=name,
                        duration_ms=round(duration * 1000, 2),
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                enhanced_logger.error(
                    f"Function {name} failed after {duration:.3f}s",
                    exception=e,
                    function_name=name,
                    duration_ms=round(duration * 1000, 2)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log slow operations (>1 second)
                if duration > 1.0:
                    enhanced_logger.warning(
                        f"Slow operation detected: {name} took {duration:.3f}s",
                        function_name=name,
                        duration_ms=round(duration * 1000, 2),
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                enhanced_logger.error(
                    f"Function {name} failed after {duration:.3f}s",
                    exception=e,
                    function_name=name,
                    duration_ms=round(duration * 1000, 2)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Global enhanced logger instance
enhanced_logger = EnhancedLogger()

# Utility functions
def get_request_id() -> str:
    """Get current request ID from context."""
    context = request_context.get()
    return context.get('request_id', 'no-context') if context else 'no-context'

def set_user_context(user_id: str, auth_type: str = None):
    """Set user information in request context."""
    context = request_context.get()
    if context:
        context['user_id'] = user_id
        if auth_type:
            context['auth_type'] = auth_type

async def get_error_logs(
    limit: int = 100,
    level: str = None,
    start_date: datetime = None,
    end_date: datetime = None
) -> List[Dict[str, Any]]:
    """Get error logs from database with filtering."""
    try:
        db_client = await connection_pool.get_prisma_client()
        
        where_clause = {}
        
        if level:
            where_clause['level'] = getattr(LogLevel, level.lower(), LogLevel.error)
        
        if start_date:
            where_clause['timestamp'] = {'gte': start_date}
        
        if end_date:
            if 'timestamp' in where_clause:
                where_clause['timestamp']['lte'] = end_date
            else:
                where_clause['timestamp'] = {'lte': end_date}
        
        logs = await db_client.errorlog.find_many(
            where=where_clause,
            take=limit,
            order_by={'timestamp': 'desc'}
        )
        
        return [
            {
                'id': log.id,
                'request_id': log.requestId,
                'level': log.level,
                'message': log.message,
                'timestamp': log.timestamp,
                'endpoint': log.endpoint,
                'user_id': log.userId,
                'resolved': log.resolved
            }
            for log in logs
        ]
        
    except Exception as e:
        enhanced_logger.error("Failed to retrieve error logs", exception=e)
        return []

async def mark_error_resolved(error_id: str) -> bool:
    """Mark an error log as resolved."""
    try:
        db_client = await connection_pool.get_prisma_client()
        
        await db_client.errorlog.update(
            where={'id': error_id},
            data={
                'resolved': True,
                'resolvedAt': datetime.utcnow()
            }
        )
        
        return True
        
    except Exception as e:
        enhanced_logger.error(f"Failed to mark error {error_id} as resolved", exception=e)
        return False