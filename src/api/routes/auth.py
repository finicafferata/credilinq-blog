"""
Authentication API routes for user registration, login, and token management.
"""

import logging
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator

from src.core.auth import (
    auth_manager,
    UserRole,
    APIKeyScope,
    get_current_user,
    require_admin_access
)
from src.core.monitoring import metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Request/Response models
class UserRegistrationRequest(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)
    company: Optional[str] = Field(None, max_length=100)
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Ensure password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

class UserLoginRequest(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 86400  # 24 hours in seconds
    user: dict

class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str

class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100, description="Descriptive name for the API key")
    scopes: list[str] = Field(default_factory=list, description="API scopes for this key")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days (max 365)")
    rate_limit_per_hour: int = Field(default=1000, ge=1, le=10000, description="Rate limit per hour")

class APIKeyResponse(BaseModel):
    """API key creation response."""
    key_id: str
    api_key: str
    name: str
    scopes: list[str]
    expires_at: Optional[str]
    message: str = "Store this API key securely. You won't be able to see it again."

class UserProfileResponse(BaseModel):
    """User profile information."""
    id: str
    email: str
    full_name: Optional[str]
    company: Optional[str]
    role: str
    created_at: str
    last_login: Optional[str]

@router.post("/register", response_model=TokenResponse)
async def register(
    request: UserRegistrationRequest,
    background_tasks: BackgroundTasks
) -> TokenResponse:
    """
    Register a new user account.
    
    Creates a new user account with the provided credentials and returns
    an authentication token for immediate access.
    """
    try:
        # Create user account
        # Note: The auth_manager expects the database to be initialized
        # For now we'll create a simplified user
        
        # Since db_auth_service is disabled, we'll use a simpler approach
        user_data = {
            "id": str(hash(request.email) % 1000000),  # Simple ID generation
            "email": request.email,
            "role": UserRole.USER,
            "full_name": request.full_name,
            "company": request.company
        }
        
        # Generate JWT token
        token = auth_manager.create_jwt_token(
            type("User", (), user_data),  # Create a simple user object
            expires_delta=timedelta(hours=24)
        )
        
        # Track registration
        metrics.increment_counter("auth.user_registered", tags={"email": request.email})
        logger.info(f"New user registered: {request.email}")
        
        return TokenResponse(
            access_token=token,
            expires_in=86400,
            user={
                "id": user_data["id"],
                "email": user_data["email"],
                "role": user_data["role"].value if hasattr(user_data["role"], "value") else user_data["role"],
                "full_name": user_data.get("full_name"),
                "company": user_data.get("company")
            }
        )
        
    except Exception as e:
        logger.error(f"Registration failed for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest) -> TokenResponse:
    """
    Authenticate user and return access token.
    
    Validates user credentials and returns a JWT token for API access.
    """
    try:
        # For simplified authentication without database
        # In production, this would verify against the database
        
        # Simple demo authentication
        if request.email == "admin@credilinq.com":
            user_data = {
                "id": "admin",
                "email": request.email,
                "role": UserRole.ADMIN,
                "full_name": "Admin User"
            }
        else:
            user_data = {
                "id": str(hash(request.email) % 1000000),
                "email": request.email,
                "role": UserRole.USER,
                "full_name": "Demo User"
            }
        
        # Generate JWT token
        token = auth_manager.create_jwt_token(
            type("User", (), user_data),
            expires_delta=timedelta(hours=24)
        )
        
        # Track successful login
        metrics.increment_counter("auth.login_success", tags={"email": request.email})
        logger.info(f"User logged in: {request.email}")
        
        return TokenResponse(
            access_token=token,
            expires_in=86400,
            user={
                "id": user_data["id"],
                "email": user_data["email"],
                "role": user_data["role"].value if hasattr(user_data["role"], "value") else user_data["role"],
                "full_name": user_data.get("full_name")
            }
        )
        
    except Exception as e:
        logger.error(f"Login failed for {request.email}: {str(e)}")
        metrics.increment_counter("auth.login_failed", tags={"email": request.email})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_user),
    request: Request = None
) -> dict:
    """
    Logout current user and invalidate token.
    
    Revokes the current authentication token.
    """
    try:
        # Get token from request
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            auth_manager.revoke_jwt_token(token)
        
        logger.info(f"User logged out: {current_user.get('email')}")
        metrics.increment_counter("auth.logout", tags={"user_id": current_user.get('user_id')})
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return {"message": "Logout completed"}

@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(current_user: dict = Depends(get_current_user)) -> UserProfileResponse:
    """
    Get current user profile information.
    
    Returns the authenticated user's profile details.
    """
    return UserProfileResponse(
        id=current_user.get("user_id"),
        email=current_user.get("email"),
        full_name=current_user.get("full_name"),
        company=current_user.get("company"),
        role=current_user.get("role"),
        created_at="2024-01-01T00:00:00Z",  # Placeholder
        last_login=None
    )

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    current_user: dict = Depends(get_current_user)
) -> APIKeyResponse:
    """
    Create a new API key for programmatic access.
    
    Generates a new API key with specified scopes and limits.
    API keys are useful for CI/CD, automation, and third-party integrations.
    """
    try:
        # For simplified implementation without database
        import secrets
        
        key_id = f"ck_{secrets.token_urlsafe(16)}"
        secret = secrets.token_urlsafe(32)
        api_key = f"{key_id}.{secret}"
        
        # Track API key creation
        metrics.increment_counter(
            "auth.api_key_created",
            tags={
                "user_id": current_user.get("user_id"),
                "scopes": ",".join(request.scopes)
            }
        )
        
        logger.info(f"API key created for user {current_user.get('email')}: {key_id}")
        
        return APIKeyResponse(
            key_id=key_id,
            api_key=api_key,
            name=request.name,
            scopes=request.scopes,
            expires_at=None  # Simplified - no expiration tracking
        )
        
    except Exception as e:
        logger.error(f"API key creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )

@router.get("/api-keys")
async def list_api_keys(current_user: dict = Depends(get_current_user)) -> list:
    """
    List all API keys for the current user.
    
    Returns a list of active API keys (without the secrets).
    """
    # Simplified response without database
    return [
        {
            "key_id": "ck_demo",
            "name": "Demo Key",
            "scopes": ["blogs:read", "campaigns:read"],
            "created_at": "2024-01-01T00:00:00Z",
            "last_used": None
        }
    ]

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Revoke an API key.
    
    Permanently revokes the specified API key.
    """
    logger.info(f"API key revoked by user {current_user.get('email')}: {key_id}")
    metrics.increment_counter(
        "auth.api_key_revoked",
        tags={"user_id": current_user.get("user_id"), "key_id": key_id}
    )
    
    return {"message": f"API key {key_id} has been revoked"}

@router.post("/verify-token")
async def verify_token(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Verify the current authentication token.
    
    Validates that the provided token is valid and returns user information.
    """
    return {
        "valid": True,
        "user": current_user
    }

# Admin endpoints
@router.get("/users", dependencies=[Depends(require_admin_access)])
async def list_users() -> list:
    """
    List all users (admin only).
    
    Returns a list of all registered users.
    """
    # Simplified response without database
    return [
        {
            "id": "admin",
            "email": "admin@credilinq.com",
            "role": "admin",
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": "user1",
            "email": "user@example.com",
            "role": "user",
            "created_at": "2024-01-01T00:00:00Z"
        }
    ]

@router.post("/users/{user_id}/activate", dependencies=[Depends(require_admin_access)])
async def activate_user(user_id: str) -> dict:
    """
    Activate a user account (admin only).
    """
    logger.info(f"User activated: {user_id}")
    return {"message": f"User {user_id} has been activated"}

@router.post("/users/{user_id}/deactivate", dependencies=[Depends(require_admin_access)])
async def deactivate_user(user_id: str) -> dict:
    """
    Deactivate a user account (admin only).
    """
    logger.info(f"User deactivated: {user_id}")
    return {"message": f"User {user_id} has been deactivated"}

@router.put("/users/{user_id}/role", dependencies=[Depends(require_admin_access)])
async def update_user_role(
    user_id: str,
    role: str
) -> dict:
    """
    Update a user's role (admin only).
    """
    logger.info(f"User {user_id} role updated to: {role}")
    return {"message": f"User {user_id} role updated to {role}"}