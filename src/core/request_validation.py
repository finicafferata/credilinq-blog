"""
Request validation middleware for production deployment.
Validates incoming requests for security, format, and business rules.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Union
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from datetime import datetime
import ipaddress

logger = logging.getLogger(__name__)

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive request validation middleware for production security.
    Validates request format, size, content, and applies business rules.
    """
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Request size limits
        self.max_content_length = self.config.get("max_content_length", 10 * 1024 * 1024)  # 10MB
        self.max_query_length = self.config.get("max_query_length", 2048)
        self.max_header_size = self.config.get("max_header_size", 8192)
        
        # Rate limiting per IP
        self.request_history = {}
        self.max_requests_per_minute = self.config.get("max_requests_per_minute", 100)
        
        # Pattern detection toggle
        self.enable_pattern_detection = self.config.get("enable_pattern_detection", True)
        
        # Validation rules
        self.blocked_patterns = [
            # SQL injection patterns
            r"(\bunion\b|\bselect\b|\bdrop\b|\bdelete\b|\bupdate\b|\binsert\b)",
            # XSS patterns
            r"(<script|javascript:|onerror=|onload=)",
            # Path traversal
            r"(\.\.\/|\.\.\\|%2e%2e%2f)",
            # Command injection
            r"(;|\||\&|\$\(|`)",
        ]
        
        # Content type validation
        self.allowed_content_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        ]
        
        # Blocked IP ranges (example - configure for production)
        self.blocked_ips = set()
        self.blocked_ip_ranges = []
        
        # Suspicious user agents
        self.suspicious_user_agents = [
            "sqlmap", "nikto", "nmap", "masscan", "zap",
            "burpsuite", "gobuster", "dirb", "ffuf"
        ]
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is in blocked list or ranges."""
        if ip_address in self.blocked_ips:
            return True
        
        try:
            ip = ipaddress.ip_address(ip_address)
            for ip_range in self.blocked_ip_ranges:
                if ip in ipaddress.ip_network(ip_range):
                    return True
        except ValueError:
            pass
        
        return False
    
    def _validate_rate_limit(self, client_ip: str) -> bool:
        """Validate request rate limit per IP."""
        now = datetime.now()
        minute_key = now.strftime("%Y%m%d%H%M")
        
        if client_ip not in self.request_history:
            self.request_history[client_ip] = {}
        
        # Clean old entries (older than 5 minutes)
        old_keys = [k for k in self.request_history[client_ip].keys() 
                   if abs(int(minute_key) - int(k)) > 5]
        for old_key in old_keys:
            del self.request_history[client_ip][old_key]
        
        # Check current minute
        current_count = self.request_history[client_ip].get(minute_key, 0)
        if current_count >= self.max_requests_per_minute:
            return False
        
        # Increment counter
        self.request_history[client_ip][minute_key] = current_count + 1
        return True
    
    def _validate_content_type(self, request: Request) -> bool:
        """Validate request content type."""
        content_type = request.headers.get("content-type", "")
        
        # Allow GET requests without content-type
        if request.method == "GET":
            return True
        
        # Check if content type is allowed
        return any(allowed in content_type.lower() 
                  for allowed in self.allowed_content_types)
    
    def _validate_request_size(self, request: Request) -> bool:
        """Validate request size limits."""
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return False
        
        # Check query string length
        if len(str(request.query_params)) > self.max_query_length:
            return False
        
        # Check headers size
        headers_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if headers_size > self.max_header_size:
            return False
        
        return True
    
    def _detect_malicious_patterns(self, text: str) -> List[str]:
        """Detect malicious patterns in request data."""
        detected_patterns = []
        text_lower = text.lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def _validate_user_agent(self, user_agent: str) -> bool:
        """Validate user agent for suspicious tools."""
        user_agent_lower = user_agent.lower()
        
        for suspicious in self.suspicious_user_agents:
            if suspicious in user_agent_lower:
                return False
        
        return True
    
    def _validate_json_payload(self, body: bytes) -> bool:
        """Validate JSON payload structure and content."""
        try:
            if len(body) == 0:
                return True
            
            # Parse JSON
            data = json.loads(body.decode("utf-8"))
            
            # Check for deeply nested structures (DoS protection)
            max_depth = 10
            if self._get_json_depth(data) > max_depth:
                logger.warning("Request rejected: JSON depth exceeds maximum")
                return False
            
            # Check for excessively large arrays
            max_array_size = 1000
            if self._check_large_arrays(data, max_array_size):
                logger.warning("Request rejected: JSON array size exceeds maximum")
                return False
            
            # Validate string content for malicious patterns (if enabled)
            if self.enable_pattern_detection:
                json_str = json.dumps(data)
                patterns = self._detect_malicious_patterns(json_str)
                if patterns:
                    logger.warning(f"Request rejected: Malicious patterns detected: {patterns}")
                    return False
            
            return True
            
        except json.JSONDecodeError:
            logger.warning("Request rejected: Invalid JSON payload")
            return False
        except Exception as e:
            logger.error(f"JSON validation error: {e}")
            return False
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate JSON object depth."""
        if current_depth > 20:  # Prevent infinite recursion
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _check_large_arrays(self, obj: Any, max_size: int) -> bool:
        """Check for excessively large arrays in JSON."""
        if isinstance(obj, list):
            if len(obj) > max_size:
                return True
            return any(self._check_large_arrays(item, max_size) for item in obj)
        elif isinstance(obj, dict):
            return any(self._check_large_arrays(v, max_size) for v in obj.values())
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main validation logic."""
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip validation for health check endpoints
        if request.url.path in ["/health", "/health/live", "/health/ready", "/ping"]:
            return await call_next(request)
        
        # IP-based blocking
        if self._is_ip_blocked(client_ip):
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return Response(
                content="Access denied",
                status_code=403,
                headers={"Content-Type": "text/plain"}
            )
        
        # Rate limiting validation
        if not self._validate_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Content-Type": "text/plain",
                    "Retry-After": "60"
                }
            )
        
        # Request size validation
        if not self._validate_request_size(request):
            logger.warning(f"Request size validation failed for IP: {client_ip}")
            return Response(
                content="Request too large",
                status_code=413,
                headers={"Content-Type": "text/plain"}
            )
        
        # Content type validation
        if not self._validate_content_type(request):
            logger.warning(f"Invalid content type from IP: {client_ip}")
            return Response(
                content="Unsupported content type",
                status_code=415,
                headers={"Content-Type": "text/plain"}
            )
        
        # User agent validation
        user_agent = request.headers.get("user-agent", "")
        if not self._validate_user_agent(user_agent):
            logger.warning(f"Suspicious user agent detected: {user_agent[:100]}")
            return Response(
                content="Forbidden",
                status_code=403,
                headers={"Content-Type": "text/plain"}
            )
        
        # URL validation (if pattern detection enabled)
        if self.enable_pattern_detection:
            url_patterns = self._detect_malicious_patterns(str(request.url))
            if url_patterns:
                logger.warning(f"Malicious URL patterns detected: {url_patterns}")
                return Response(
                    content="Bad request",
                    status_code=400,
                    headers={"Content-Type": "text/plain"}
                )
        
        # Validate request body for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            
            # JSON payload validation
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type.lower():
                if not self._validate_json_payload(body):
                    return Response(
                        content="Invalid request payload",
                        status_code=400,
                        headers={"Content-Type": "text/plain"}
                    )
            
            # Restore body for downstream processing
            async def receive():
                return {"type": "http.request", "body": body}
            
            request._receive = receive
        
        # Log successful validation
        logger.debug(f"Request validation passed for {request.method} {request.url.path}")
        
        # Continue to next middleware
        response = await call_next(request)
        
        # Add security headers to response
        response.headers["X-Validated"] = "true"
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
        
        return response


class APIInputValidationMixin:
    """
    Mixin class for additional API-specific validation rules.
    Can be used in route handlers for business logic validation.
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, bool]:
        """Validate password strength requirements."""
        return {
            "min_length": len(password) >= 8,
            "has_uppercase": bool(re.search(r'[A-Z]', password)),
            "has_lowercase": bool(re.search(r'[a-z]', password)),
            "has_digits": bool(re.search(r'\d', password)),
            "has_special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path traversal attempts
        filename = filename.replace("../", "").replace("..\\", "")
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Limit length
        return filename[:255]


# Configuration for different environments
VALIDATION_CONFIGS = {
    "development": {
        "max_content_length": 50 * 1024 * 1024,  # 50MB for development
        "max_requests_per_minute": 1000,
        "enable_pattern_detection": False
    },
    
    "staging": {
        "max_content_length": 20 * 1024 * 1024,  # 20MB
        "max_requests_per_minute": 200,
        "enable_pattern_detection": True
    },
    
    "production": {
        "max_content_length": 10 * 1024 * 1024,  # 10MB
        "max_requests_per_minute": 100,
        "enable_pattern_detection": True,
        "blocked_ip_ranges": ["10.0.0.0/8"],  # Example - configure as needed
        "strict_validation": True
    }
}