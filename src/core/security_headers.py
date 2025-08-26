"""
Security headers middleware for production deployment.
Implements OWASP security headers and protection mechanisms.
"""

import time
from typing import Dict, Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    Implements OWASP recommended security headers for production deployment.
    """
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Default security headers configuration
        self.default_headers = {
            # Prevent clickjacking attacks
            "X-Frame-Options": "DENY",
            
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy (replaces Feature-Policy)
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
            
            # Content Security Policy
            "Content-Security-Policy": self._build_csp(),
            
            # HTTP Strict Transport Security (HSTS)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Server information hiding
            "Server": "CrediLinq/2.0",
            
            # Cache control for sensitive content
            "Cache-Control": "no-cache, no-store, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # Merge with custom config
        self.headers = {**self.default_headers, **self.config.get("headers", {})}
        
        # Paths that should have different cache headers
        self.public_cache_paths = [
            "/docs", "/redoc", "/openapi.json", 
            "/health", "/health/live", "/health/ready"
        ]
    
    def _build_csp(self) -> str:
        """Build Content Security Policy header."""
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' https://api.openai.com https://api.langchain.com",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests"
        ]
        return "; ".join(csp_directives)
    
    def _should_apply_cache_headers(self, path: str) -> bool:
        """Determine if cache headers should be applied to this path."""
        return not any(path.startswith(public_path) for public_path in self.public_cache_paths)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.headers.items():
            # Skip cache headers for public endpoints
            if header_name in ["Cache-Control", "Pragma", "Expires"]:
                if not self._should_apply_cache_headers(request.url.path):
                    continue
            
            response.headers[header_name] = header_value
        
        # Add custom headers based on response type
        content_type = response.headers.get("content-type", "")
        
        # API responses
        if "application/json" in content_type:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-API-Version"] = "v2.0"
        
        # HTML responses (docs)
        elif "text/html" in content_type:
            response.headers["X-Frame-Options"] = "SAMEORIGIN"  # Allow embedding docs
        
        # Add request ID for tracing
        if "x-request-id" not in response.headers:
            request_id = getattr(request.state, "request_id", None)
            if request_id:
                response.headers["x-request-id"] = request_id
        
        # Add response time header for monitoring
        if hasattr(request.state, "start_time"):
            response_time = int((time.time() - request.state.start_time) * 1000)
            response.headers["x-response-time"] = f"{response_time}ms"
        
        # Log security events
        self._log_security_events(request, response)
        
        return response
    
    def _log_security_events(self, request: Request, response: Response):
        """Log security-related events."""
        
        # Log requests with suspicious patterns
        suspicious_patterns = [
            "script", "javascript:", "<script", "eval(", "alert(",
            "../", "..\\", "etc/passwd", "cmd.exe"
        ]
        
        request_data = str(request.url) + str(request.headers)
        
        if any(pattern in request_data.lower() for pattern in suspicious_patterns):
            logger.warning(
                f"Suspicious request detected: {request.method} {request.url.path}",
                extra={
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        # Log blocked requests
        if response.status_code == 403:
            logger.info(
                f"Request blocked: {request.method} {request.url.path}",
                extra={
                    "client_ip": request.client.host if request.client else "unknown",
                    "status_code": response.status_code
                }
            )

class ProductionSecurityMiddleware(BaseHTTPMiddleware):
    """
    Additional production security measures.
    """
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        # IP whitelist for admin endpoints
        self.admin_ip_whitelist = self.config.get("admin_ip_whitelist", [])
        
        # Blocked user agents
        self.blocked_user_agents = [
            "curl", "wget", "python-requests", "bot", "crawler", "spider"
        ] if self.config.get("block_automated_tools", False) else []
        
        # Request size limits
        self.max_request_size = self.config.get("max_request_size", 10 * 1024 * 1024)  # 10MB
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply production security measures."""
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(f"Request too large: {content_length} bytes from {request.client.host}")
            return Response(
                content="Request entity too large",
                status_code=413,
                headers={"Content-Type": "text/plain"}
            )
        
        # IP whitelist for admin endpoints
        if request.url.path.startswith("/admin") and self.admin_ip_whitelist:
            client_ip = request.client.host if request.client else "unknown"
            if client_ip not in self.admin_ip_whitelist:
                logger.warning(f"Admin access denied for IP: {client_ip}")
                return Response(
                    content="Access denied",
                    status_code=403,
                    headers={"Content-Type": "text/plain"}
                )
        
        # Block suspicious user agents
        user_agent = request.headers.get("user-agent", "").lower()
        if any(blocked in user_agent for blocked in self.blocked_user_agents):
            if not request.url.path.startswith("/health"):  # Allow health checks
                logger.info(f"Blocked user agent: {user_agent}")
                return Response(
                    content="Forbidden",
                    status_code=403,
                    headers={"Content-Type": "text/plain"}
                )
        
        # Add start time for response time calculation
        request.state.start_time = time.time()
        
        # Process request
        return await call_next(request)

# Security configuration for different environments
SECURITY_CONFIGS = {
    "development": {
        "headers": {
            "Strict-Transport-Security": "",  # No HSTS in development
            "Content-Security-Policy": "default-src 'self' 'unsafe-inline' 'unsafe-eval' *"
        },
        "block_automated_tools": False,
        "max_request_size": 50 * 1024 * 1024  # 50MB for development
    },
    
    "staging": {
        "headers": {
            "Strict-Transport-Security": "max-age=300"  # Short HSTS for staging
        },
        "block_automated_tools": False,
        "max_request_size": 20 * 1024 * 1024  # 20MB
    },
    
    "production": {
        "headers": {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload"
        },
        "block_automated_tools": True,
        "max_request_size": 10 * 1024 * 1024,  # 10MB
        "admin_ip_whitelist": []  # Configure in production
    }
}