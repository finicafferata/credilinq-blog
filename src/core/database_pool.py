"""
Database connection pool manager with advanced configuration.
Handles connection pooling, health checks, and connection lifecycle management.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import asyncpg
    from asyncpg.pool import Pool
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = Pool = None
    ASYNCPG_AVAILABLE = False

try:
    from prisma import Prisma
    PRISMA_AVAILABLE = True
except ImportError:
    Prisma = None
    PRISMA_AVAILABLE = False

from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class PoolMetrics:
    """Connection pool metrics for monitoring."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    avg_connection_time: float = 0.0
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0

class DatabaseConnectionPool:
    """Advanced database connection pool manager."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.prisma_client: Optional[Prisma] = None
        self.metrics = PoolMetrics()
        self._connection_times: list = []
        self._health_check_interval = 60  # seconds
        self._last_health_check = None
        self._pool_initialized = False
    
    async def initialize_pool(self):
        """Initialize the connection pool with optimized settings."""
        if self._pool_initialized:
            return
        
        try:
            # Parse database URL
            db_url = settings.database_url
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
            
            # Connection pool configuration
            pool_config = {
                "dsn": db_url,
                "min_size": max(1, settings.db_pool_size // 4),  # Keep some connections always open
                "max_size": settings.db_pool_size + settings.db_max_overflow,
                "max_queries": 50000,  # Queries per connection before recycling
                "max_inactive_connection_lifetime": 300,  # 5 minutes
                "timeout": settings.db_connection_timeout,
                "command_timeout": 60,  # Command timeout
                "server_settings": {
                    "jit": "off",  # Disable JIT compilation for faster connection startup
                    "application_name": "credilinq_content_agent"
                }
            }
            
            # Create connection pool if asyncpg is available
            if ASYNCPG_AVAILABLE:
                logger.info(f"Initializing asyncpg connection pool: {pool_config['min_size']}-{pool_config['max_size']} connections")
                self.pool = await asyncpg.create_pool(**pool_config)
                logger.info("AsyncPG connection pool initialized successfully")
            
            # Initialize Prisma client for ORM operations
            if PRISMA_AVAILABLE:
                self.prisma_client = Prisma()
                await self.prisma_client.connect()
                logger.info("Prisma client initialized successfully")
            
            self._pool_initialized = True
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    async def close_pool(self):
        """Close the connection pool and cleanup resources."""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("AsyncPG connection pool closed")
            
            if self.prisma_client:
                await self.prisma_client.disconnect()
                logger.info("Prisma client disconnected")
                
            self._pool_initialized = False
            
        except Exception as e:
            logger.error(f"Error closing database connection pool: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool with metrics tracking."""
        if not self._pool_initialized:
            await self.initialize_pool()
        
        # If pool is not available, fail fast with a clear error instead of
        # letting the context manager exit without yielding (which causes
        # "generator didn't yield").
        if not self.pool:
            raise RuntimeError(
                "Database connection pool is not available. Install 'asyncpg' or disable pool usage."
            )

        start_time = time.time()
        connection = None
        
        try:
            connection = await self.pool.acquire()
            connection_time = time.time() - start_time
            self._connection_times.append(connection_time)
            
            # Keep only last 100 connection times for average calculation
            if len(self._connection_times) > 100:
                self._connection_times = self._connection_times[-100:]
            
            self.metrics.avg_connection_time = sum(self._connection_times) / len(self._connection_times)
            
            yield connection
                
        except Exception as e:
            self.metrics.failed_connections += 1
            logger.error(f"Failed to acquire database connection: {e}")
            raise
            
        finally:
            if connection and self.pool:
                try:
                    await self.pool.release(connection)
                except Exception as e:
                    logger.error(f"Failed to release database connection: {e}")
    
    async def get_prisma_client(self) -> Prisma:
        """Get the Prisma client for ORM operations."""
        if not self._pool_initialized:
            await self.initialize_pool()
        
        if not self.prisma_client:
            raise RuntimeError("Prisma client not initialized")
        
        return self.prisma_client
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute a raw SQL query using the connection pool."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_single(self, query: str, *args) -> Any:
        """Execute a query that returns a single row."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def execute_command(self, query: str, *args) -> str:
        """Execute a command (INSERT, UPDATE, DELETE) and return status."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the database connection pool."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "metrics": {},
            "checks": {}
        }
        
        try:
            # Update metrics
            await self._update_metrics()
            
            # If pool is not available (e.g., asyncpg not installed), return a clear unhealthy status
            if not self.pool:
                raise RuntimeError("Connection pool not initialized (asyncpg unavailable or disabled)")

            # Test connection acquisition
            start_time = time.time()
            async with self.get_connection() as conn:
                # Simple query to test database connectivity
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    raise Exception("Database query test failed")
            
            connection_test_time = time.time() - start_time
            health_status["checks"]["connection_test"] = {
                "status": "pass",
                "duration_ms": round(connection_test_time * 1000, 2)
            }
            
            # Test Prisma client if available
            if self.prisma_client:
                try:
                    await self.prisma_client.user.count()
                    health_status["checks"]["prisma_test"] = {"status": "pass"}
                except Exception as e:
                    health_status["checks"]["prisma_test"] = {"status": "fail", "error": str(e)}
            
            # Update metrics
            health_status["metrics"] = {
                "total_connections": self.metrics.total_connections,
                "active_connections": self.metrics.active_connections,
                "idle_connections": self.metrics.idle_connections,
                "failed_connections": self.metrics.failed_connections,
                "avg_connection_time_ms": round(self.metrics.avg_connection_time * 1000, 2),
                "pool_initialized": self._pool_initialized
            }
            
            self.metrics.last_health_check = datetime.utcnow()
            self.metrics.health_check_failures = 0
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.metrics.health_check_failures += 1
            logger.error(f"Database health check failed: {e}")
        
        return health_status
    
    async def _update_metrics(self):
        """Update connection pool metrics."""
        try:
            if self.pool:
                self.metrics.total_connections = self.pool.get_size()
                self.metrics.active_connections = self.pool.get_size() - self.pool.get_idle_size()
                self.metrics.idle_connections = self.pool.get_idle_size()
            
        except Exception as e:
            logger.warning(f"Failed to update pool metrics: {e}")
    
    async def cleanup_expired_connections(self):
        """Clean up expired or stale connections."""
        try:
            if self.pool:
                # Force cleanup of idle connections
                await self.pool.expire_connections()
                logger.debug("Expired database connections cleaned up")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup expired connections: {e}")
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get detailed pool statistics for monitoring."""
        await self._update_metrics()
        
        return {
            "pool_size": {
                "min": max(1, settings.db_pool_size // 4),
                "max": settings.db_pool_size + settings.db_max_overflow,
                "current_total": self.metrics.total_connections,
                "current_active": self.metrics.active_connections,
                "current_idle": self.metrics.idle_connections
            },
            "performance": {
                "avg_connection_time_ms": round(self.metrics.avg_connection_time * 1000, 2),
                "failed_connections": self.metrics.failed_connections,
                "health_check_failures": self.metrics.health_check_failures
            },
            "config": {
                "connection_timeout": settings.db_connection_timeout,
                "max_retries": settings.db_max_retries,
                "pool_initialized": self._pool_initialized
            },
            "last_health_check": self.metrics.last_health_check.isoformat() if self.metrics.last_health_check else None
        }

# Global connection pool instance
connection_pool = DatabaseConnectionPool()

# Startup and shutdown handlers
async def startup_database_pool():
    """Initialize database connection pool on startup."""
    try:
        await connection_pool.initialize_pool()
        logger.info("Database connection pool startup completed")
    except Exception as e:
        logger.error(f"Failed to initialize database connection pool: {e}")
        raise

async def shutdown_database_pool():
    """Cleanup database connection pool on shutdown."""
    try:
        await connection_pool.close_pool()
        logger.info("Database connection pool shutdown completed")
    except Exception as e:
        logger.error(f"Error shutting down database connection pool: {e}")

# Background task for connection pool maintenance
async def connection_pool_maintenance():
    """Background task for connection pool maintenance."""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            await connection_pool.cleanup_expired_connections()
            health = await connection_pool.health_check()
            
            if health["status"] != "healthy":
                logger.warning(f"Database pool health check failed: {health}")
            
        except asyncio.CancelledError:
            logger.info("Connection pool maintenance task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in connection pool maintenance: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying