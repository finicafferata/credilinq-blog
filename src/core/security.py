"""
Security utilities and input validation for the CrediLinq Content Agent.
"""

import re
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status
import logging
from .exceptions import SecurityException, InputValidationError, SQLInjectionAttempt, XSSAttempt, PathTraversalAttempt

logger = logging.getLogger(__name__)

class SecurityError(SecurityException):
    """Custom exception for security-related errors."""
    pass

class InputValidator:
    """Input validation utilities to prevent various attacks."""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|;|\*|'|\")",  # SQL comment and string delimiters
        r"(\bOR\b\s+\d+\s*=\s*\d+)",  # OR 1=1 patterns
        r"(\bUNION\b.*\bSELECT\b)",  # UNION SELECT patterns
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick=
        r"<iframe[^>]*>.*?</iframe>",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",  # Directory traversal
        r"\\.\\.\\",  # Windows path traversal
        r"/etc/passwd",  # Common Linux target
        r"C:\\Windows",  # Common Windows target
    ]
    
    @staticmethod
    def validate_string_input(value: str, field_name: str, max_length: int = 1000, min_length: int = 1) -> str:
        """
        Validate and sanitize string input with less restrictive SQL injection detection.
        """
        if not isinstance(value, str):
            raise InputValidationError(field_name, f"Must be a string, got {type(value).__name__}")
        
        if len(value) < min_length:
            raise InputValidationError(field_name, f"Must be at least {min_length} characters")
        
        if len(value) > max_length:
            raise InputValidationError(field_name, f"Must be no more than {max_length} characters")
        
        # Remove leading/trailing whitespace
        value = value.strip()
        
        # Basic SQL injection check - only check for obvious patterns, not common punctuation
        dangerous_patterns = [
            '--',  # SQL comment
            ';',   # Statement separator
            '/*',  # Block comment start
            '*/',  # Block comment end
            'xp_', # Extended stored procedures
            'sp_', # Stored procedures
            'exec ', # Execute command
            'execute ', # Execute command
            'union ', # SQL union
            'select ', # SQL select
            'insert ', # SQL insert
            'update ', # SQL update
            'delete ', # SQL delete
            'drop ', # SQL drop
            'create ', # SQL create
            'alter ', # SQL alter
        ]
        
        value_lower = value.lower()
        for pattern in dangerous_patterns:
            if pattern in value_lower:
                raise SecurityException(
                    f"Potential SQL injection detected in '{field_name}': {pattern}"
                )
        
        return value

    @staticmethod
    def validate_content_text(value: str, field_name: str, max_length: int = 10000, min_length: int = 1) -> str:
        """
        Relaxed validator for natural-language content (titles, contexts).
        Allows common words like 'create', 'select', etc., while still blocking
        obvious injection vectors such as SQL comments and statement separators.
        """
        if not isinstance(value, str):
            raise InputValidationError(field_name, f"Must be a string, got {type(value).__name__}")

        if len(value) < min_length:
            raise InputValidationError(field_name, f"Must be at least {min_length} characters")

        if len(value) > max_length:
            raise InputValidationError(field_name, f"Must be no more than {max_length} characters")

        value = value.strip()

        # Block only high-signal injection primitives
        blocked_tokens = [
            '--',   # SQL comment
            ';',    # Statement separator
            '/*',   # Block comment start
            '*/',   # Block comment end
            'xp_',  # Extended stored procedures
            'sp_',  # Stored procedures
        ]

        lower = value.lower()
        for token in blocked_tokens:
            if token in lower:
                raise SecurityException(
                    f"Potential SQL injection detected in '{field_name}': {token}"
                )

        # Optionally block explicit EXEC/UNION chains without banning common words
        risky_phrases = ['exec ', 'execute ', 'union select']
        for phrase in risky_phrases:
            if phrase in lower:
                raise SecurityException(
                    f"Potential SQL injection detected in '{field_name}': {phrase.strip()}"
                )

        return value
    
    @staticmethod
    def validate_uuid(value: str, field_name: str) -> str:
        """Validate UUID format."""
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        if not re.match(uuid_pattern, value, re.IGNORECASE):
            raise SecurityError(f"Invalid UUID format for {field_name}")
        return value.lower()
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            raise SecurityError("Invalid email format")
        return email.lower()
    
    @staticmethod
    def validate_url(url: str) -> str:
        """Validate URL format and safety."""
        # Allow only HTTP/HTTPS URLs
        url_pattern = r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$"
        if not re.match(url_pattern, url):
            raise SecurityError("Invalid URL format")
        
        # Block dangerous protocols
        dangerous_protocols = ["javascript:", "data:", "file:", "ftp:"]
        for protocol in dangerous_protocols:
            if url.lower().startswith(protocol):
                raise SecurityError("Dangerous URL protocol detected")
        
        return url
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove directory traversal patterns
        filename = re.sub(r"[/\\:*?\"<>|]", "_", filename)
        # Remove leading/trailing dots and spaces
        filename = filename.strip(". ")
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        if not filename:
            raise SecurityError("Invalid filename")
        
        return filename

def validate_api_input(data: Dict[str, Any], validation_rules: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate API input data against validation rules.
    
    Args:
        data: Input data dictionary
        validation_rules: Dictionary with field validation rules
        
    Returns:
        Validated and sanitized data
        
    Example:
        rules = {
            "title": {"type": "string", "max_length": 200, "required": True},
            "email": {"type": "email", "required": False},
            "blog_id": {"type": "uuid", "required": True}
        }
    """
    validated_data = {}
    
    for field, rules in validation_rules.items():
        value = data.get(field)
        
        # Check required fields
        if rules.get("required", False) and value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Required field '{field}' is missing"
            )
        
        if value is None:
            continue
        
        try:
            # Apply validation based on type
            field_type = rules.get("type", "string")
            
            if field_type == "string":
                max_length = rules.get("max_length", 1000)
                validated_data[field] = InputValidator.validate_string_input(value, field, max_length)
            
            elif field_type == "uuid":
                validated_data[field] = InputValidator.validate_uuid(value, field)
            
            elif field_type == "email":
                validated_data[field] = InputValidator.validate_email(value)
            
            elif field_type == "url":
                validated_data[field] = InputValidator.validate_url(value)
            
            elif field_type == "int":
                min_val = rules.get("min", 0)
                max_val = rules.get("max", 2**31 - 1)
                int_val = int(value)
                if int_val < min_val or int_val > max_val:
                    raise ValueError(f"Value out of range")
                validated_data[field] = int_val
            
            elif field_type == "float":
                min_val = rules.get("min", 0.0)
                max_val = rules.get("max", 1e6)
                float_val = float(value)
                if float_val < min_val or float_val > max_val:
                    raise ValueError(f"Value out of range")
                validated_data[field] = float_val
            
            else:
                validated_data[field] = value
                
        except (SecurityError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input for field '{field}': {str(e)}"
            )
    
    return validated_data

class SecurityValidator:
    """
    Security validator class for agent input validation.
    Provides compatibility layer for agent security validation.
    """
    
    def __init__(self):
        self.input_validator = InputValidator()
    
    def validate_input(self, value: str, field_name: str = "input", max_length: int = 10000) -> str:
        """
        Validate input string for security threats.
        
        Args:
            value: Input value to validate
            field_name: Name of the field being validated
            max_length: Maximum allowed length
            
        Returns:
            str: Validated and sanitized input
            
        Raises:
            SecurityException: If validation fails
        """
        return self.input_validator.validate_string_input(value, field_name, max_length)

    def validate_content(self, value: str, field_name: str = "input", max_length: int = 10000) -> str:
        """
        Validate natural-language content using a relaxed rule set.
        """
        return self.input_validator.validate_content_text(value, field_name, max_length)
    
    def validate_vector_embedding(self, embedding: List[float]) -> bool:
        """
        Validate vector embedding data.
        
        Args:
            embedding: Vector embedding to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            SecurityException: If validation fails
        """
        if not isinstance(embedding, list):
            raise SecurityError("Embedding must be a list")
        
        if len(embedding) == 0:
            raise SecurityError("Embedding cannot be empty")
        
        if len(embedding) > 10000:  # Reasonable upper limit
            raise SecurityError("Embedding too large")
        
        for i, value in enumerate(embedding):
            if not isinstance(value, (int, float)):
                raise SecurityError(f"Embedding value at index {i} must be numeric")
            
            if abs(value) > 100:  # Reasonable range for normalized embeddings
                logger.warning(f"Unusual embedding value at index {i}: {value}")
        
        return True
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security."""
        return self.input_validator.sanitize_filename(filename)
    
    def validate_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate JSON data structure.
        
        Args:
            data: JSON data to validate
            
        Returns:
            Dict: Validated data
        """
        if not isinstance(data, dict):
            raise SecurityError("Data must be a dictionary")
        
        # Check for reasonable size limits
        if len(str(data)) > 100000:  # 100KB limit
            raise SecurityError("Data too large")
        
        return data

def secure_headers_middleware():
    """Security headers to add to responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }