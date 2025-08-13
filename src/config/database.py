"""
Enhanced database configuration and connection management with security and pooling.
"""

import psycopg2
import psycopg2.pool
import psycopg2.extras
import psycopg2.sql
import logging
import time
import threading
from typing import Optional, Dict, Any, Union, List, Tuple
from contextlib import contextmanager
from urllib.parse import urlparse
from .settings import settings

logger = logging.getLogger(__name__)

class SecureDatabaseService:
    """
    Secure database service with connection pooling, retry logic, and SQL injection prevention.
    """
    
    def __init__(self):
        self.database_url = settings.database_url
        self.max_retries = settings.db_max_retries
        self.connection_timeout = settings.db_connection_timeout
        self.pool_size = settings.db_pool_size
        self.max_overflow = settings.db_max_overflow
        
        # Connection pool
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._pool_lock = threading.Lock()
        
        # Initialize connection pool (skip in certain environments if DB unavailable)
        try:
            self._initialize_pool()
        except Exception as e:
            if settings.environment == "development":
                logger.warning(f"⚠️  Database connection failed in development mode: {e}")
                logger.info("🔧 Continuing without database pool - some features may be limited")
                self._pool = None
            else:
                raise
    
    def _initialize_pool(self):
        """Initialize the connection pool with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Parse database URL for connection parameters
                parsed_url = urlparse(self.database_url)
                
                connection_kwargs = {
                    'host': parsed_url.hostname,
                    'port': parsed_url.port or 5432,
                    'database': parsed_url.path[1:] if parsed_url.path else 'postgres',
                    'user': parsed_url.username,
                    'password': parsed_url.password,
                    'connect_timeout': self.connection_timeout,
                    'application_name': 'credilinq-content-agent',
                    # Security settings
                    'sslmode': 'prefer'
                }
                
                # Remove None values
                connection_kwargs = {k: v for k, v in connection_kwargs.items() if v is not None}
                
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.pool_size + self.max_overflow,
                    **connection_kwargs
                )
                
                logger.info(f"✅ Database connection pool initialized (min=1, max={self.pool_size + self.max_overflow})")
                return
                
            except Exception as e:
                logger.warning(f"Database pool initialization attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("❌ Failed to initialize database connection pool after all retries")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    @contextmanager
    def get_connection(self):
        """
        Context manager to get a database connection from the pool.
        Automatically handles connection return and error management.
        """
        conn = None
        try:
            with self._pool_lock:
                if self._pool is None:
                    raise ConnectionError("Database pool not initialized - check database configuration")
                conn = self._pool.getconn()
            
            if conn is None:
                raise ConnectionError("Could not get database connection from pool")
            
            # Set security-focused connection settings
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")  # Prevent long-running queries
                cur.execute("SET lock_timeout = '10s'")       # Prevent deadlock issues
                
            yield conn
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                try:
                    with self._pool_lock:
                        if self._pool:
                            self._pool.putconn(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
    
    def execute_query(
        self,
        query: Union[str, psycopg2.sql.SQL, psycopg2.sql.Composed],
        params: Optional[Union[Tuple, Dict[str, Any]]] = None,
        fetch: str = "all"  # "all", "one", "none"
    ) -> Optional[Union[List[Tuple], Tuple]]:
        """
        Execute a parameterized query with SQL injection prevention.
        
        Args:
            query: SQL query (preferably using psycopg2.sql for composition)
            params: Query parameters (prevents SQL injection)
            fetch: Result fetching strategy
            
        Returns:
            Query results based on fetch strategy
        """
        if isinstance(query, str) and params and any(str(p) in query for p in (params if isinstance(params, (list, tuple)) else params.values())):
            raise ValueError("Direct string interpolation detected - use parameterized queries only")
        
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        # Log query for debugging (but not params for security)
                        if settings.debug:
                            query_str = query if isinstance(query, str) else query.as_string(conn)
                            logger.debug(f"Executing query: {query_str[:100]}...")
                        
                        cur.execute(query, params)
                        
                        if fetch == "all":
                            return cur.fetchall()
                        elif fetch == "one":
                            return cur.fetchone()
                        else:
                            return None
                        
            except psycopg2.OperationalError as e:
                logger.warning(f"Database query attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Database query error: {e}")
                raise
    
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """
        Execute multiple operations in a single transaction.
        
        Args:
            operations: List of dicts with 'query', 'params' keys
            
        Returns:
            True if transaction succeeded, raises exception on failure
        """
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        for operation in operations:
                            query = operation.get('query')
                            params = operation.get('params')
                            
                            if not query:
                                raise ValueError("Each operation must have a 'query' key")
                            
                            cur.execute(query, params)
                        
                        conn.commit()
                        logger.debug(f"Transaction completed successfully ({len(operations)} operations)")
                        return True
                        
            except psycopg2.OperationalError as e:
                logger.warning(f"Transaction attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Transaction error: {e}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        health_data = {
            "status": "unhealthy",
            "database": "postgresql",
            "pool_stats": {},
            "checks": {}
        }
        
        try:
            # Test basic connectivity
            start_time = time.time()
            result = self.execute_query("SELECT version(), now(), 1 as test_value", fetch="one")
            response_time = (time.time() - start_time) * 1000
            
            if result and result.get('test_value') == 1:
                health_data["status"] = "healthy"
                health_data["checks"]["connectivity"] = "✅ Pass"
                health_data["checks"]["response_time_ms"] = round(response_time, 2)
                health_data["checks"]["version"] = result.get('version', 'Unknown')
            
            # Check pool statistics
            if self._pool:
                with self._pool_lock:
                    health_data["pool_stats"] = {
                        "total_connections": self._pool.maxconn,
                        "closed_connections": len(self._pool._pool),
                        "used_connections": len(self._pool._used)
                    }
            
            # Test query performance
            if health_data["checks"]["response_time_ms"] > 1000:
                health_data["checks"]["performance"] = "⚠️  Slow (>1s)"
            else:
                health_data["checks"]["performance"] = "✅ Good"
            
        except Exception as e:
            health_data["error"] = str(e)
            health_data["checks"]["connectivity"] = "❌ Fail"
            logger.error(f"Database health check failed: {e}")
        
        return health_data
    
    def close_pool(self):
        """Close the connection pool."""
        with self._pool_lock:
            if self._pool:
                self._pool.closeall()
                self._pool = None
                logger.info("Database connection pool closed")

class DatabaseConfig:
    """Backward compatibility wrapper."""
    
    def __init__(self):
        self.database_url = settings.database_url
        self._service = SecureDatabaseService()
        
    def get_db_connection(self):
        """Get PostgreSQL database connection - deprecated, use secure service."""
        import warnings
        warnings.warn(
            "Direct connection usage is deprecated. Use SecureDatabaseService context manager.",
            DeprecationWarning
        )
        return psycopg2.connect(self.database_url)
    
    def health_check(self) -> dict:
        """Perform database health check."""
        return self._service.health_check()

# Global instances
secure_db = SecureDatabaseService()
db_config = DatabaseConfig()  # Backward compatibility