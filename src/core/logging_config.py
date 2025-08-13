"""
Enhanced logging configuration with structured logging and correlation IDs.
Provides comprehensive error tracking, performance monitoring, and audit trails.
"""

import logging
import logging.handlers
import json
import time
import uuid
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading

from ..config.settings import settings

# Thread-local storage for request correlation IDs
local_context = threading.local()

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs with correlation IDs.
    """
    
    def __init__(self):
        super().__init__()
        self.hostname = self._get_hostname()
        self.service_name = "credilinq-content-agent"
    
    def _get_hostname(self) -> str:
        """Get system hostname."""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": self.service_name,
            "hostname": self.hostname,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "process_id": record.process,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(local_context, 'correlation_id', None)
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add user context if available
        user_id = getattr(local_context, 'user_id', None)
        if user_id:
            log_entry["user_id"] = user_id
        
        # Add request context if available
        request_id = getattr(local_context, 'request_id', None)
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add exception details
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'exc_info', 'exc_text',
                'stack_info', 'getMessage'
            }
        }
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_entry["performance"] = {
                "duration_ms": record.duration_ms,
                "slow_query": record.duration_ms > 1000
            }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class CorrelationFilter(logging.Filter):
    """
    Logging filter to add correlation context to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation context to log record."""
        
        # Add correlation ID from context
        correlation_id = getattr(local_context, 'correlation_id', None)
        if correlation_id:
            record.correlation_id = correlation_id
        
        # Add user context
        user_id = getattr(local_context, 'user_id', None)
        if user_id:
            record.user_id = user_id
        
        # Add request context
        request_id = getattr(local_context, 'request_id', None)
        if request_id:
            record.request_id = request_id
        
        return True

class SecurityAuditLogger:
    """
    Dedicated logger for security events and audit trails.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("security.audit")
        self._setup_security_logger()
    
    def _setup_security_logger(self):
        """Setup security-specific logging configuration."""
        
        # Create security log file handler
        security_log_path = Path("logs/security_audit.log")
        security_log_path.parent.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            security_log_path,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        handler.setFormatter(StructuredFormatter())
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Don't propagate to root logger
    
    def log_authentication_attempt(
        self, 
        user_id: Optional[str], 
        ip_address: str, 
        user_agent: str, 
        success: bool,
        method: str = "unknown"
    ):
        """Log authentication attempt."""
        self.logger.info(
            "Authentication attempt",
            extra={
                "event_type": "authentication",
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "success": success,
                "auth_method": method,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: str
    ):
        """Log authorization failure."""
        self.logger.warning(
            "Authorization failure",
            extra={
                "event_type": "authorization_failure",
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "ip_address": ip_address,
                "severity": "medium"
            }
        )
    
    def log_suspicious_activity(
        self,
        ip_address: str,
        activity_type: str,
        details: Dict[str, Any]
    ):
        """Log suspicious activity."""
        self.logger.warning(
            f"Suspicious activity detected: {activity_type}",
            extra={
                "event_type": "suspicious_activity",
                "activity_type": activity_type,
                "ip_address": ip_address,
                "details": details,
                "severity": "high"
            }
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str
    ):
        """Log sensitive data access."""
        self.logger.info(
            f"Data access: {action} on {resource_type}",
            extra={
                "event_type": "data_access",
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action
            }
        )

class PerformanceLogger:
    """
    Dedicated logger for performance monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
        self._setup_performance_logger()
    
    def _setup_performance_logger(self):
        """Setup performance-specific logging."""
        
        # Create performance log file handler  
        perf_log_path = Path("logs/performance.log")
        perf_log_path.parent.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            perf_log_path,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5
        )
        handler.setFormatter(StructuredFormatter())
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def log_slow_query(
        self,
        query_type: str,
        duration_ms: float,
        query_details: Optional[Dict[str, Any]] = None
    ):
        """Log slow database query."""
        self.logger.warning(
            f"Slow {query_type} query detected",
            extra={
                "event_type": "slow_query",
                "query_type": query_type,
                "duration_ms": duration_ms,
                "details": query_details or {},
                "severity": "medium" if duration_ms > 5000 else "low"
            }
        )
    
    def log_api_performance(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int,
        user_id: Optional[str] = None
    ):
        """Log API endpoint performance."""
        severity = "high" if duration_ms > 10000 else "medium" if duration_ms > 5000 else "low"
        
        self.logger.info(
            f"API call performance: {method} {endpoint}",
            extra={
                "event_type": "api_performance",
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
                "status_code": status_code,
                "user_id": user_id,
                "severity": severity
            }
        )

def setup_logging():
    """
    Configure comprehensive logging for the application.
    """
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler for development
    if settings.environment == "development" or settings.debug:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if settings.log_format == "json":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        console_handler.addFilter(CorrelationFilter())
        root_logger.addHandler(console_handler)
    
    # File handler for persistent logging
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file_path,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(StructuredFormatter())
    file_handler.addFilter(CorrelationFilter())
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        "logs/errors.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    error_handler.addFilter(CorrelationFilter())
    root_logger.addHandler(error_handler)
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("psycopg2").setLevel(logging.WARNING)
    
    # Silence noisy loggers in production
    if settings.environment == "production":
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info("📝 Enhanced logging configuration completed")

def set_correlation_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Set correlation context for current thread.
    
    Args:
        correlation_id: Correlation ID for request tracing
        request_id: Request ID for this specific request
        user_id: User ID for the current user
    """
    if correlation_id:
        local_context.correlation_id = correlation_id
    if request_id:
        local_context.request_id = request_id
    if user_id:
        local_context.user_id = user_id

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return getattr(local_context, 'correlation_id', None)

def clear_correlation_context():
    """Clear correlation context for current thread."""
    for attr in ['correlation_id', 'request_id', 'user_id']:
        if hasattr(local_context, attr):
            delattr(local_context, attr)

# Global logger instances
security_audit_logger = SecurityAuditLogger()
performance_logger = PerformanceLogger()

# Context manager for correlation tracking
class CorrelationContext:
    """Context manager for automatic correlation ID management."""
    
    def __init__(self, correlation_id: Optional[str] = None, user_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.user_id = user_id
        self.previous_correlation_id = None
        self.previous_user_id = None
    
    def __enter__(self):
        # Store previous context
        self.previous_correlation_id = getattr(local_context, 'correlation_id', None)
        self.previous_user_id = getattr(local_context, 'user_id', None)
        
        # Set new context
        set_correlation_context(
            correlation_id=self.correlation_id,
            user_id=self.user_id
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        if self.previous_correlation_id:
            local_context.correlation_id = self.previous_correlation_id
        elif hasattr(local_context, 'correlation_id'):
            delattr(local_context, 'correlation_id')
        
        if self.previous_user_id:
            local_context.user_id = self.previous_user_id
        elif hasattr(local_context, 'user_id'):
            delattr(local_context, 'user_id')

# Initialize logging on import
if not hasattr(setup_logging, '_initialized'):
    setup_logging()
    setup_logging._initialized = True