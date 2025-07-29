"""
Comprehensive authentication and authorization system for the CrediLinQ API.
Supports API keys, OAuth2, JWT tokens, and role-based access control.
"""

import secrets
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, validator
try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    CryptContext = None

from .cache import cache
from .monitoring import metrics
from ..config.settings import settings

# Password hashing
if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None

# Security schemes
bearer_scheme = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UserRole(str, Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    READONLY = "readonly"
    WEBHOOK_CLIENT = "webhook_client"

class APIKeyScope(str, Enum):
    """API key scopes for fine-grained permissions."""
    BLOGS_READ = "blogs:read"
    BLOGS_WRITE = "blogs:write"
    CAMPAIGNS_READ = "campaigns:read"
    CAMPAIGNS_WRITE = "campaigns:write"
    ANALYTICS_READ = "analytics:read"
    WEBHOOKS_MANAGE = "webhooks:manage"
    ADMIN_ACCESS = "admin:access"

@dataclass
class User:
    """User model for authentication."""
    id: str
    email: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class APIKey:
    """API key model for programmatic access."""
    id: str
    key_id: str  # Public identifier
    key_hash: str  # Hashed secret key
    name: str
    user_id: str
    scopes: List[APIKeyScope]
    is_active: bool = True
    created_at: datetime = None
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit_per_hour: int = 1000
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

class AuthenticationError(Exception):
    """Authentication related errors."""
    pass

class AuthorizationError(Exception):
    """Authorization related errors."""
    pass

class AuthManager:
    """Manages authentication and authorization."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.revoked_tokens: Set[str] = set()  # JWT token blacklist
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if not exists."""
        admin_id = "admin-001"
        if admin_id not in self.users:
            admin_user = User(
                id=admin_id,
                email="admin@credilinq.com",
                hashed_password=self.hash_password("admin123"),  # Change in production
                role=UserRole.ADMIN,
                metadata={"created_by": "system", "is_default": True}
            )
            self.users[admin_id] = admin_user
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        if PASSLIB_AVAILABLE and pwd_context:
            return pwd_context.hash(password)
        else:
            # Fallback to simple hashing (not secure for production)
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if PASSLIB_AVAILABLE and pwd_context:
            return pwd_context.verify(plain_password, hashed_password)
        else:
            # Fallback verification
            import hashlib
            return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
    
    def create_user(
        self,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user."""
        user_id = str(uuid.uuid4())
        
        # Check if email already exists
        for user in self.users.values():
            if user.email == email:
                raise AuthenticationError("Email already registered")
        
        user = User(
            id=user_id,
            email=email,
            hashed_password=self.hash_password(password),
            role=role,
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        
        # Check for brute force attempts
        if self._is_brute_force_attempt(email):
            raise AuthenticationError("Too many failed login attempts. Please try again later.")
        
        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break
        
        if not user or not user.is_active:
            self._record_failed_login(email)
            return None
        
        if not self.verify_password(password, user.hashed_password):
            self._record_failed_login(email)
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        # Clear failed attempts on successful login
        if email in self.failed_login_attempts:
            del self.failed_login_attempts[email]
        
        return user
    
    def _is_brute_force_attempt(self, email: str) -> bool:
        """Check if email has too many recent failed login attempts."""
        if email not in self.failed_login_attempts:
            return False
        
        # Remove attempts older than 1 hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.failed_login_attempts[email] = [
            attempt for attempt in self.failed_login_attempts[email]
            if attempt > cutoff_time
        ]
        
        # Check if too many attempts
        return len(self.failed_login_attempts[email]) >= 5
    
    def _record_failed_login(self, email: str):
        """Record a failed login attempt."""
        if email not in self.failed_login_attempts:
            self.failed_login_attempts[email] = []
        
        self.failed_login_attempts[email].append(datetime.utcnow())
        
        # Track security metrics
        metrics.increment_counter("auth.failed_login", tags={"email": email})
    
    def create_jwt_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT token for a user."""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
        
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # JWT ID for revocation
        }
        
        token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
        
        # Cache token info for faster lookups
        cache_key = f"jwt_token:{payload['jti']}"
        asyncio.create_task(
            cache.set("auth", cache_key, {"user_id": user.id, "valid": True}, ttl=int(expires_delta.total_seconds()))
        )
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
            
            # Check if token is revoked
            if payload["jti"] in self.revoked_tokens:
                return None
            
            # Verify user still exists and is active
            user = self.users.get(payload["sub"])
            if not user or not user.is_active:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def revoke_jwt_token(self, token: str) -> bool:
        """Revoke a JWT token."""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"], options={"verify_exp": False})
            self.revoked_tokens.add(payload["jti"])
            
            # Remove from cache
            cache_key = f"jwt_token:{payload['jti']}"
            asyncio.create_task(cache.delete("auth", cache_key))
            
            return True
        except jwt.JWTError:
            return False
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: List[APIKeyScope],
        expires_in_days: Optional[int] = None,
        rate_limit_per_hour: int = 1000
    ) -> tuple[APIKey, str]:
        """Create a new API key for a user."""
        
        # Generate key components
        key_id = f"ak_{secrets.token_urlsafe(16)}"
        secret_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(secret_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            id=str(uuid.uuid4()),
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at,
            rate_limit_per_hour=rate_limit_per_hour
        )
        
        self.api_keys[key_id] = api_key
        
        # Return API key object and the actual secret (only shown once)
        full_key = f"{key_id}.{secret_key}"
        return api_key, full_key
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify an API key."""
        try:
            # Parse key format: key_id.secret
            key_id, secret = api_key.split(".", 1)
        except ValueError:
            return None
        
        # Get API key record
        api_key_record = self.api_keys.get(key_id)
        if not api_key_record or not api_key_record.is_active:
            return None
        
        # Check expiration
        if api_key_record.expires_at and datetime.utcnow() > api_key_record.expires_at:
            return None
        
        # Verify secret
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()
        if not hmac.compare_digest(secret_hash, api_key_record.key_hash):
            return None
        
        # Update last used timestamp
        api_key_record.last_used = datetime.utcnow()
        
        return api_key_record
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
            return True
        return False
    
    def has_scope(self, api_key: APIKey, required_scope: APIKeyScope) -> bool:
        """Check if API key has required scope."""
        return required_scope in api_key.scopes or APIKeyScope.ADMIN_ACCESS in api_key.scopes
    
    def has_role(self, user: User, required_role: UserRole) -> bool:
        """Check if user has required role or higher."""
        role_hierarchy = {
            UserRole.READONLY: 0,
            UserRole.API_CLIENT: 1,
            UserRole.USER: 2,
            UserRole.WEBHOOK_CLIENT: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level

# Global auth manager
auth_manager = AuthManager()

# Dependency functions for FastAPI

async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """Get current user from JWT token."""
    
    try:
        # Extract token
        token = credentials.credentials
        
        # Verify token
        payload = auth_manager.verify_jwt_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Get user
        user = auth_manager.users.get(payload["sub"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Track authentication success
        metrics.increment_counter("auth.jwt_success", tags={"user_id": user.id})
        
        return {
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "auth_type": "jwt"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.increment_counter("auth.jwt_error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_current_user_from_api_key(request: Request) -> Dict[str, Any]:
    """Get current user from API key."""
    
    # Check for API key in header
    api_key = request.headers.get("x-api-key")
    if not api_key:
        # Check in Authorization header (Bearer format)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header.split(" ", 1)[1]
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Verify API key
    api_key_record = auth_manager.verify_api_key(api_key)
    if not api_key_record:
        metrics.increment_counter("auth.api_key_invalid")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Get associated user
    user = auth_manager.users.get(api_key_record.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Track successful authentication
    metrics.increment_counter("auth.api_key_success", tags={"user_id": user.id})
    
    # Store API key info in request state for rate limiting
    request.state.api_key_id = api_key_record.key_id
    request.state.rate_limit_per_hour = api_key_record.rate_limit_per_hour
    
    return {
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "auth_type": "api_key",
        "api_key_id": api_key_record.key_id,
        "scopes": api_key_record.scopes
    }

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user from any supported authentication method."""
    
    # Try API key first
    try:
        return await get_current_user_from_api_key(request)
    except HTTPException:
        pass
    
    # Try JWT token
    try:
        # Extract Bearer token from Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            return await get_current_user_from_token(credentials)
    except HTTPException:
        pass
    
    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a valid API key or JWT token."
    )

def require_scope(required_scope: APIKeyScope):
    """Dependency that requires a specific API scope."""
    
    async def scope_checker(
        current_user: Dict = Depends(get_current_user),
        request: Request = None
    ):
        # If authenticated via JWT, check user role
        if current_user.get("auth_type") == "jwt":
            user = auth_manager.users.get(current_user["user_id"])
            if user and user.role == UserRole.ADMIN:
                return current_user  # Admins have all permissions
        
        # If authenticated via API key, check scopes
        elif current_user.get("auth_type") == "api_key":
            scopes = current_user.get("scopes", [])
            if required_scope not in scopes and APIKeyScope.ADMIN_ACCESS not in scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required scope: {required_scope}"
                )
        
        return current_user
    
    return scope_checker

def require_role(required_role: UserRole):
    """Dependency that requires a specific user role."""
    
    async def role_checker(current_user: Dict = Depends(get_current_user)):
        user = auth_manager.users.get(current_user["user_id"])
        if not user or not auth_manager.has_role(user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return current_user
    
    return role_checker

async def require_admin_access(current_user: Dict = Depends(get_current_user)) -> bool:
    """Dependency that requires admin access."""
    user = auth_manager.users.get(current_user["user_id"])
    
    if not user or user.role != UserRole.ADMIN:
        # Also check for admin scope in API keys
        if current_user.get("auth_type") == "api_key":
            scopes = current_user.get("scopes", [])
            if APIKeyScope.ADMIN_ACCESS not in scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
    
    return True

# Authentication middleware
class AuthenticationMiddleware:
    """Middleware to handle authentication and add user context."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip authentication for public endpoints
        public_paths = ["/", "/health", "/docs", "/redoc", "/openapi.json", "/versions"]
        if any(request.url.path.startswith(path) for path in public_paths):
            await self.app(scope, receive, send)
            return
        
        # Try to authenticate and add user context
        try:
            user_info = await get_current_user(request)
            scope["state"] = getattr(scope, "state", {})
            scope["state"]["user"] = user_info
        except HTTPException:
            # Authentication failed - let the endpoint handle it
            pass
        
        await self.app(scope, receive, send)

# Import asyncio for async operations
import asyncio