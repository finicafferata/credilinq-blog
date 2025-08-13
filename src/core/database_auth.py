"""
Database-backed authentication storage service.
Handles persistent user and API key storage using Prisma ORM.
"""

import asyncio
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import logging

try:
    from prisma import Prisma
    from prisma.models import User, APIKey
    PRISMA_AVAILABLE = True
except ImportError:
    PRISMA_AVAILABLE = False
    Prisma = None
    User = APIKey = None

# Define UserRole enum locally to avoid circular imports
from enum import Enum

class UserRole(str, Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    READONLY = "readonly"
    WEBHOOK_CLIENT = "webhook_client"

try:
    from prisma.enums import APIKeyScope
except ImportError:
    APIKeyScope = None

from ..config.settings import settings
from .database_pool import connection_pool

logger = logging.getLogger(__name__)

class DatabaseAuthService:
    """Database service for authentication data persistence."""
    
    def __init__(self):
        if not PRISMA_AVAILABLE:
            raise ImportError("Prisma client not available. Install with: pip install prisma")
        
        self.db: Optional[Prisma] = None
        self.connection_pool = connection_pool
    
    async def connect(self):
        """Initialize database connection using connection pool."""
        if self.db is None:
            try:
                # Initialize connection pool
                await self.connection_pool.initialize_pool()
                
                # Get Prisma client from pool
                self.db = await self.connection_pool.get_prisma_client()
                logger.info("Database connection established for auth service with connection pooling")
                
                # Ensure admin user exists
                await self._ensure_admin_user_exists()
                
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
    
    async def disconnect(self):
        """Close database connection."""
        if self.db:
            # Connection pool handles the actual disconnection
            self.db = None
            logger.info("Database connection closed for auth service")
    
    async def _ensure_admin_user_exists(self):
        """Ensure the default admin user exists in database."""
        try:
            # Check if admin user exists
            admin_user = await self.db.user.find_first(
                where={"role": UserRole.ADMIN}
            )
            
            if admin_user:
                logger.info(f"Admin user found: {admin_user.email}")
                return admin_user
            
            # Create admin user if it doesn't exist
            from .auth import auth_manager  # Import here to avoid circular import
            
            # Get admin credentials from environment or generate
            admin_email = settings.admin_email or "admin@credilinq.com"
            admin_password = settings.admin_password
            
            if not admin_password:
                admin_password = auth_manager._generate_secure_admin_password()
                
                if settings.environment != "production":
                    print(f"\\n🔐 DATABASE ADMIN CREDENTIALS:")
                    print(f"   Email: {admin_email}")
                    print(f"   Password: {admin_password}")
                    print(f"   Please save these credentials securely!\\n")
            
            # Validate password
            auth_manager._validate_password_strength(admin_password)
            
            # Create admin user in database
            admin_user = await self.db.user.create(
                data={
                    "email": admin_email,
                    "hashedPassword": auth_manager.hash_password(admin_password),
                    "role": UserRole.ADMIN,
                    "metadata": {
                        "created_by": "system",
                        "is_default": True,
                        "password_auto_generated": not bool(settings.admin_password),
                        "created_at_timestamp": datetime.utcnow().isoformat()
                    }
                }
            )
            
            logger.info(f"Admin user created in database: {admin_user.email}")
            return admin_user
            
        except Exception as e:
            logger.error(f"Error ensuring admin user exists: {e}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.user.find_unique(
                where={"email": email},
                include={"apiKeys": True}
            )
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.user.find_unique(
                where={"id": user_id},
                include={"apiKeys": True}
            )
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            return None
    
    async def create_user(
        self,
        email: str,
        hashed_password: str,
        role: UserRole = UserRole.USER,
        metadata: Optional[Dict] = None
    ) -> Optional[User]:
        """Create a new user."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.user.create(
                data={
                    "email": email,
                    "hashedPassword": hashed_password,
                    "role": role,
                    "metadata": metadata or {}
                }
            )
        except Exception as e:
            logger.error(f"Error creating user {email}: {e}")
            return None
    
    async def update_user_login(self, user_id: str) -> bool:
        """Update user's last login timestamp."""
        if not self.db:
            await self.connect()
        
        try:
            await self.db.user.update(
                where={"id": user_id},
                data={"lastLogin": datetime.utcnow()}
            )
            return True
        except Exception as e:
            logger.error(f"Error updating user login {user_id}: {e}")
            return False
    
    async def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by key ID."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.apikey.find_unique(
                where={"keyId": key_id},
                include={"user": True}
            )
        except Exception as e:
            logger.error(f"Error getting API key {key_id}: {e}")
            return None
    
    async def create_api_key(
        self,
        key_id: str,
        key_hash: str,
        name: str,
        user_id: str,
        scopes: List[APIKeyScope],
        expires_at: Optional[datetime] = None,
        rate_limit_per_hour: int = 1000,
        metadata: Optional[Dict] = None
    ) -> Optional[APIKey]:
        """Create a new API key."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.apikey.create(
                data={
                    "keyId": key_id,
                    "keyHash": key_hash,
                    "name": name,
                    "userId": user_id,
                    "scopes": scopes,
                    "expiresAt": expires_at,
                    "rateLimitPerHour": rate_limit_per_hour,
                    "metadata": metadata or {}
                }
            )
        except Exception as e:
            logger.error(f"Error creating API key {key_id}: {e}")
            return None
    
    async def update_api_key_usage(self, key_id: str) -> bool:
        """Update API key's last used timestamp."""
        if not self.db:
            await self.connect()
        
        try:
            await self.db.apikey.update(
                where={"keyId": key_id},
                data={"lastUsed": datetime.utcnow()}
            )
            return True
        except Exception as e:
            logger.error(f"Error updating API key usage {key_id}: {e}")
            return False
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if not self.db:
            await self.connect()
        
        try:
            await self.db.apikey.update(
                where={"keyId": key_id},
                data={"isActive": False}
            )
            return True
        except Exception as e:
            logger.error(f"Error revoking API key {key_id}: {e}")
            return False
    
    async def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.apikey.find_many(
                where={"userId": user_id, "isActive": True}
            )
        except Exception as e:
            logger.error(f"Error getting API keys for user {user_id}: {e}")
            return []
    
    async def cleanup_expired_keys(self) -> int:
        """Remove expired API keys from database."""
        if not self.db:
            await self.connect()
        
        try:
            result = await self.db.apikey.delete_many(
                where={
                    "expiresAt": {"lt": datetime.utcnow()},
                    "isActive": False
                }
            )
            count = result.get("count", 0) if isinstance(result, dict) else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired API keys")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0
    
    async def get_all_users(self, limit: int = 100) -> List[User]:
        """Get all users (for admin purposes)."""
        if not self.db:
            await self.connect()
        
        try:
            return await self.db.user.find_many(
                take=limit,
                order_by={"createdAt": "desc"}
            )
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []
    
    async def health_check(self) -> dict:
        """Check database connection health with detailed metrics."""
        if not self.db:
            try:
                await self.connect()
            except Exception as e:
                return {"status": "unhealthy", "error": f"Connection failed: {e}"}
        
        try:
            # Test basic query
            user_count = await self.db.user.count()
            
            # Get connection pool health
            pool_health = await self.connection_pool.health_check()
            
            return {
                "status": "healthy",
                "user_count": user_count,
                "pool_health": pool_health,
                "connection_pool_initialized": self.connection_pool._pool_initialized
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

# Global database auth service instance
db_auth_service = DatabaseAuthService()

# Startup and shutdown handlers
async def startup_database_auth():
    """Initialize database auth service on startup."""
    try:
        await db_auth_service.connect()
        logger.info("Database auth service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database auth service: {e}")
        raise

async def shutdown_database_auth():
    """Cleanup database auth service on shutdown."""
    try:
        await db_auth_service.disconnect()
        logger.info("Database auth service shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down database auth service: {e}")