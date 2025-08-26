"""
Railway-optimized database configuration with simplified connection handling.
Designed to avoid connection issues that could cause crashes.
"""

import os
import psycopg2
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class RailwayDatabaseService:
    """
    Simplified database service optimized for Railway deployment.
    Focuses on stability and minimal resource usage.
    """
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "")
        
        # Railway PostgreSQL URLs sometimes use postgres:// instead of postgresql://
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        
        self.connection_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
        self._connection_params = self._parse_database_url()
    
    def _parse_database_url(self) -> Dict[str, Any]:
        """Parse Railway database URL into connection parameters."""
        if not self.database_url:
            logger.error("DATABASE_URL not set")
            return {}
        
        try:
            parsed = urlparse(self.database_url)
            
            params = {
                "host": parsed.hostname,
                "port": parsed.port or 5432,
                "database": parsed.path.lstrip('/') if parsed.path else "postgres",
                "user": parsed.username,
                "password": parsed.password,
                "connect_timeout": self.connection_timeout,
                "sslmode": "require",  # Railway PostgreSQL requires SSL
                "application_name": "credilinq-railway"
            }
            
            logger.info(f"Database config: {params['host']}:{params['port']}/{params['database']}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to parse DATABASE_URL: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test database connection without pooling."""
        if not self._connection_params:
            return False
        
        try:
            conn = psycopg2.connect(**self._connection_params)
            
            # Test basic query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
            conn.close()
            logger.info("✅ Railway database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Railway database connection failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check for Railway."""
        start_time = time.time()
        
        try:
            if self.test_connection():
                response_time = round((time.time() - start_time) * 1000, 2)
                return {
                    "status": "healthy",
                    "response_time_ms": response_time,
                    "database": "postgresql",
                    "platform": "railway"
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Connection test failed"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_connection(self):
        """Get a simple database connection."""
        if not self._connection_params:
            raise RuntimeError("Database not configured")
        
        return psycopg2.connect(**self._connection_params)
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a simple query with safety checks."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if cursor.description:
                        return [dict(row) for row in cursor.fetchall()]
                    else:
                        return []
                        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

# Global instance for Railway
railway_db = RailwayDatabaseService()

# Compatibility with existing code
secure_db = railway_db
db_config = railway_db