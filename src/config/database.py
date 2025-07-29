"""
Database configuration and connection management.
"""

import psycopg2
import logging
from typing import Optional
from .settings import settings

logger = logging.getLogger(__name__)

class DatabaseConfig:
    def __init__(self):
        self.database_url = settings.database_url
        
    def get_db_connection(self):
        """Get PostgreSQL database connection."""
        return psycopg2.connect(self.database_url)
    
    def health_check(self) -> dict:
        """Perform basic database health check."""
        try:
            with self.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            
            return {"status": "healthy", "database": "postgresql"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Crear instancia global
db_config = DatabaseConfig()