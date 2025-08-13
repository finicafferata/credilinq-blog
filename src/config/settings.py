"""
Centralized configuration management using Pydantic Settings.
All configuration values are sourced from environment variables.
"""

import os
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with comprehensive environment variable support."""
    
    # ========================================
    # APPLICATION SETTINGS
    # ========================================
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    api_title: str = Field("CrediLinq Content Agent API", env="API_TITLE")
    api_version: str = Field("4.0.0", env="API_VERSION")
    api_description: str = Field(
        "AI-powered content management platform",
        env="API_DESCRIPTION"
    )
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    
    # ========================================
    # DATABASE CONFIGURATION - REQUIRED
    # ========================================
    # Database settings
    database_url: str = Field("postgresql://postgres@localhost:5432/credilinq_dev_postgres", env="DATABASE_URL")
    database_url_direct: str = Field("postgresql://postgres@localhost:5432/credilinq_dev_postgres", env="DATABASE_URL_DIRECT")
    
    # Supabase settings (opcionales para SQLite)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_db_url: Optional[str] = None
    supabase_storage_bucket: str = Field("documents", env="SUPABASE_STORAGE_BUCKET")
    
    # Database Connection Settings
    db_connection_timeout: int = Field(30, env="DB_CONNECTION_TIMEOUT")
    db_max_retries: int = Field(3, env="DB_MAX_RETRIES")
    db_pool_size: int = Field(10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    
    # Alternative Database URLs (for compatibility)
    # database_url: Optional[str] = Field(None, env="DATABASE_URL")
    # database_url_direct: Optional[str] = Field(None, env="DATABASE_URL_DIRECT")
    
    # ========================================
    # AI SERVICES - REQUIRED
    # ========================================
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(4000, env="OPENAI_MAX_TOKENS")
    
    # AI Services - Optional
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")
    
    # ========================================
    # SECURITY SETTINGS
    # ========================================
    cors_origins_str: str = Field(
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,*",
        env="CORS_ORIGINS"
    )
    secret_key: str = Field("", env="SECRET_KEY")
    jwt_secret: str = Field("", env="JWT_SECRET")
    jwt_expiration_hours: int = Field(24, env="JWT_EXPIRATION_HOURS")
    
    # Admin Account Configuration
    admin_email: Optional[str] = Field(None, env="ADMIN_EMAIL")
    admin_password: Optional[str] = Field(None, env="ADMIN_PASSWORD")
    require_admin_password_change: bool = Field(True, env="REQUIRE_ADMIN_PASSWORD_CHANGE")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(100, env="RATE_LIMIT_BURST")
    
    # ========================================
    # PERFORMANCE SETTINGS
    # ========================================
    api_timeout: int = Field(120, env="API_TIMEOUT")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    cache_ttl_seconds: int = Field(300, env="CACHE_TTL_SECONDS")
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    
    # Redis Configuration
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    cache_prefix: str = Field("credilinq", env="CACHE_PREFIX")
    
    # Agent Configuration
    agent_timeout_seconds: int = Field(300, env="AGENT_TIMEOUT_SECONDS")
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    
    # File Upload Settings
    max_file_size_mb: int = Field(10, env="MAX_FILE_SIZE_MB")
    allowed_file_types_str: str = Field("txt,pdf,docx,md", env="ALLOWED_FILE_TYPES")
    
    # ========================================
    # MONITORING & LOGGING
    # ========================================
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("detailed", env="LOG_FORMAT")
    log_file_path: str = Field("logs/application.log", env="LOG_FILE_PATH")
    enable_analytics: bool = Field(True, env="ENABLE_ANALYTICS")
    enable_performance_tracking: bool = Field(True, env="ENABLE_PERFORMANCE_TRACKING")
    
    # External Monitoring (Optional)
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    datadog_api_key: Optional[str] = Field(None, env="DATADOG_API_KEY")
    
    # ========================================
    # DEVELOPMENT SETTINGS
    # ========================================
    enable_debug_toolbar: bool = Field(False, env="ENABLE_DEBUG_TOOLBAR")
    enable_profiler: bool = Field(False, env="ENABLE_PROFILER")
    test_database_url: Optional[str] = Field(None, env="TEST_DATABASE_URL")
    
    # ========================================
    # COMPUTED PROPERTIES
    # ========================================
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins_str.split(",") if origin.strip()]
    
    @property
    def allowed_file_types(self) -> List[str]:
        """Parse allowed file types from comma-separated string."""
        return [ftype.strip() for ftype in self.allowed_file_types_str.split(",") if ftype.strip()]
    
    # Backward compatibility properties
    @property
    def default_model(self) -> str:
        """Backward compatibility for openai_model."""
        return self.openai_model
    
    @property
    def default_temperature(self) -> float:
        """Backward compatibility for openai_temperature."""
        return self.openai_temperature
    
    @property
    def max_tokens(self) -> int:
        """Backward compatibility for openai_max_tokens."""
        return self.openai_max_tokens
    
    @property
    def max_retries(self) -> int:
        """Backward compatibility for db_max_retries."""
        return self.db_max_retries
    
    @property
    def OPENAI_API_KEY(self) -> str:
        """Backward compatibility for openai_api_key (uppercase)."""
        return self.openai_api_key
    
    # ========================================
    # VALIDATORS
    # ========================================
    @validator('secret_key')
    def validate_secret_key(cls, v, values):
        """Ensure secret key meets security requirements - auto-generate if empty."""
        environment = values.get('environment', 'development')
        
        # Auto-generate if empty or insecure
        if not v or v in ["change-this-secret-key", "default", "secret", "key"]:
            import secrets
            v = secrets.token_hex(32)  # Generate 64-character hex string (256-bit)
            if environment != 'production':
                print(f"🔐 Auto-generated secure secret key for {environment} environment")
        
        # Ensure minimum length for security
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
            
        return v
    
    @validator('jwt_secret')
    def validate_jwt_secret(cls, v, values):
        """Ensure JWT secret meets security requirements - auto-generate if empty."""
        environment = values.get('environment', 'development')
        
        # Auto-generate if empty or insecure
        if not v or v in ["change-this-jwt-secret", "default", "secret", "jwt"]:
            import secrets
            v = secrets.token_hex(32)  # Generate 64-character hex string (256-bit)
            if environment != 'production':
                print(f"🔐 Auto-generated secure JWT secret for {environment} environment")
        
        # Ensure minimum length for security
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
            
        return v
    
    @validator('openai_temperature')
    def validate_temperature(cls, v):
        """Ensure temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("OpenAI temperature must be between 0.0 and 2.0")
        return v
    
    @validator('rate_limit_per_minute')
    def validate_rate_limit(cls, v):
        """Ensure rate limit is reasonable."""
        if v <= 0 or v > 10000:
            raise ValueError("Rate limit must be between 1 and 10000 requests per minute")
        return v
    
    @validator('admin_password')
    def validate_admin_password(cls, v, values):
        """Validate admin password strength if provided."""
        if v is None:
            return v  # Will be auto-generated
        
        environment = values.get('environment', 'development')
        
        # Basic length check
        if len(v) < 8:
            raise ValueError("Admin password must be at least 8 characters long")
        
        # In production, enforce stronger requirements
        if environment == 'production':
            # Check for character variety
            has_upper = any(c.isupper() for c in v)
            has_lower = any(c.islower() for c in v)
            has_digit = any(c.isdigit() for c in v)
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)
            
            missing_types = []
            if not has_upper:
                missing_types.append("uppercase letter")
            if not has_lower:
                missing_types.append("lowercase letter")
            if not has_digit:
                missing_types.append("number")
            if not has_special:
                missing_types.append("special character")
            
            if missing_types:
                raise ValueError(
                    f"Production admin password must contain at least one: {', '.join(missing_types)}"
                )
            
            # Check minimum length for production
            if len(v) < 12:
                raise ValueError("Production admin password must be at least 12 characters long")
            
            # Check for common weak passwords
            weak_patterns = [
                "password", "admin", "123456", "qwerty", "letmein",
                "welcome", "monkey", "dragon", "master", "secret", "admin123"
            ]
            
            v_lower = v.lower()
            for pattern in weak_patterns:
                if pattern in v_lower:
                    raise ValueError(f"Admin password cannot contain common pattern: {pattern}")
        
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            """Prioritize environment variables over .env file."""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """
    Get the global settings instance.
    Provides compatibility with dependency injection patterns.
    """
    return settings

# Validate critical settings on import
if settings.environment == "production":
    # Additional production validations
    critical_production_vars = ['OPENAI_API_KEY']
    optional_production_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    
    missing_critical = []
    for var in critical_production_vars:
        if not os.getenv(var):
            missing_critical.append(var)
    
    if missing_critical:
        raise ValueError(
            f"Missing critical environment variables for production: {', '.join(missing_critical)}"
        )
    
    # Check for auto-generated secrets in production (should be explicitly set)
    if not os.getenv('SECRET_KEY'):
        import warnings
        warnings.warn(
            "SECRET_KEY not explicitly set in production - using auto-generated key. "
            "For better security, set SECRET_KEY environment variable."
        )
    
    if not os.getenv('JWT_SECRET'):
        import warnings
        warnings.warn(
            "JWT_SECRET not explicitly set in production - using auto-generated key. "
            "For better security, set JWT_SECRET environment variable."
        )
    
    if not os.getenv('ADMIN_PASSWORD'):
        import warnings
        warnings.warn(
            "ADMIN_PASSWORD not set in production - admin password will be auto-generated. "
            "For better security, set ADMIN_EMAIL and ADMIN_PASSWORD environment variables."
        )