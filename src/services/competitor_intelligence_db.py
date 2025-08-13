"""
Database service for Competitor Intelligence operations using Prisma.
Provides CRUD operations for competitors and related data.
"""

import asyncio
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import psycopg2
import psycopg2.extras
from ..config.database import db_config

# Database service for competitor intelligence operations
class CompetitorIntelligenceDB:
    """Database service for competitor intelligence operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_config = db_config
        # Cache for detected column naming (snake_case vs camelCase)
        self._ci_competitors_columns: Optional[Dict[str, str]] = None

    def _detect_ci_competitors_columns(self, cur) -> Dict[str, str]:
        """Detect whether the ci_competitors table uses snake_case or camelCase column names.

        Returns a mapping of logical field names to actual column identifiers
        (with proper quoting when needed).
        """
        if self._ci_competitors_columns is not None:
            return self._ci_competitors_columns

        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ci_competitors'
            """
        )
        rows = cur.fetchall()

        # Extract names robustly regardless of cursor row type
        column_names: set[str] = set()
        for row in rows:
            if isinstance(row, dict) or hasattr(row, 'get'):
                name = row.get('column_name')  # type: ignore[attr-defined]
            else:
                # Fallback: treat as sequence/tuple
                try:
                    name = row[0]  # type: ignore[index]
                except Exception:
                    name = None
            if isinstance(name, str):
                column_names.add(name)

        def pick(snake: str, camel: str) -> str:
            # Prefer whichever exists in the table; default to snake
            if camel in column_names:
                return f'"{camel}"'
            if snake in column_names:
                return snake
            # Default to snake if neither is reported (shouldn't happen)
            return snake

        columns = {
            'id': 'id',
            'name': 'name',
            'domain': 'domain',
            'tier': 'tier',
            'industry': 'industry',
            'description': 'description',
            'platforms': 'platforms',
            'monitoring_keywords': pick('monitoring_keywords', 'monitoringKeywords'),
            'is_active': pick('is_active', 'isActive'),
            'created_at': pick('created_at', 'createdAt'),
            'updated_at': pick('updated_at', 'updatedAt'),
            'last_monitored': pick('last_monitored', 'lastMonitored'),
        }

        self._ci_competitors_columns = columns
        return columns
    
    def _format_competitor_data(self, competitor_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Format competitor data from database result."""
        # Normalize DB snake_case to API camelCase expected by routes
        key_mapping = {
            "created_at": "createdAt",
            "updated_at": "updatedAt",
            "last_monitored": "lastMonitored",
            "monitoring_keywords": "monitoringKeywords",
            "is_active": "isActive",
        }

        normalized: Dict[str, Any] = {}
        for key, value in competitor_dict.items():
            out_key = key_mapping.get(key, key)
            normalized[out_key] = value

        # Convert datetime objects to ISO strings
        if normalized.get("createdAt") and hasattr(normalized["createdAt"], "isoformat"):
            normalized["createdAt"] = normalized["createdAt"].isoformat()
        if normalized.get("updatedAt") and hasattr(normalized["updatedAt"], "isoformat"):
            normalized["updatedAt"] = normalized["updatedAt"].isoformat()
        if normalized.get("lastMonitored") and hasattr(normalized["lastMonitored"], "isoformat"):
            normalized["lastMonitored"] = normalized["lastMonitored"].isoformat()

        # Parse PostgreSQL arrays to Python lists
        if normalized.get("platforms"):
            platforms_val = normalized["platforms"]
            if isinstance(platforms_val, str) and platforms_val.startswith('{') and platforms_val.endswith('}'):
                normalized["platforms"] = platforms_val[1:-1].split(',') if platforms_val != '{}' else []
            elif not isinstance(platforms_val, list):
                normalized["platforms"] = []

        return normalized
        
    async def create_competitor(self, competitor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new competitor in the database."""
        try:
            # Insert into database using raw SQL
            with self.db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cols = self._detect_ci_competitors_columns(cur)
                    # Insert competitor with explicit UUID generation
                    insert_query = f"""
                        INSERT INTO ci_competitors (
                            id, {cols['name']}, {cols['domain']}, {cols['tier']}, {cols['industry']}, {cols['description']},
                            {cols['platforms']}, {cols['monitoring_keywords']}, {cols['is_active']}, {cols['created_at']}, {cols['updated_at']}
                        ) VALUES (
                            %s, %s, %s, %s::"CICompetitorTier", %s::"CIIndustry", %s,
                            %s::"CIPlatform"[], %s, %s, NOW(), NOW()
                        ) RETURNING *
                    """
                    
                    new_id = str(uuid.uuid4())
                    cur.execute(insert_query, (
                        new_id,
                        competitor_data["name"],
                        competitor_data["domain"], 
                        competitor_data["tier"],
                        competitor_data["industry"],
                        competitor_data["description"],
                        competitor_data.get("platforms", []),
                        competitor_data.get("monitoring_keywords", []),
                        True
                    ))
                    
                    result = cur.fetchone()
                    competitor = self._format_competitor_data(dict(result))
                    
                    self.logger.info(f"Created competitor {competitor['name']} with ID {competitor['id']}")
                    return competitor
            
        except Exception as e:
            self.logger.error(f"Failed to create competitor: {str(e)}")
            raise
    
    async def list_competitors(
        self, 
        industry: Optional[str] = None,
        tier: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """List competitors with optional filtering."""
        try:
            with self.db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cols = self._detect_ci_competitors_columns(cur)
                    # Build query with filters
                    query = "SELECT * FROM ci_competitors WHERE 1=1"
                    params = []
                    
                    if industry:
                        query += " AND industry = %s"
                        params.append(industry)
                    
                    if tier:
                        query += " AND tier = %s"
                        params.append(tier)
                    
                    if active_only:
                        query += f" AND {cols['is_active']} = %s"
                        params.append(True)

                    query += f" ORDER BY {cols['created_at']} DESC"
                    
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    competitors = []
                    for result in results:
                        competitor = self._format_competitor_data(dict(result))
                        competitors.append(competitor)
                    
                    self.logger.info(f"Listed {len(competitors)} competitors")
                    return competitors
            
        except Exception as e:
            self.logger.error(f"Failed to list competitors: {str(e)}")
            raise
    
    async def get_competitor(self, competitor_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific competitor by ID."""
        try:
            with self.db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM ci_competitors WHERE id = %s", (competitor_id,))
                    result = cur.fetchone()
                    
                    if result:
                        competitor = self._format_competitor_data(dict(result))
                        
                        self.logger.info(f"Retrieved competitor {competitor_id}")
                        return competitor
                    else:
                        self.logger.warning(f"Competitor {competitor_id} not found")
                        return None
            
        except Exception as e:
            self.logger.error(f"Failed to get competitor {competitor_id}: {str(e)}")
            raise
    
    async def update_competitor(
        self, 
        competitor_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a competitor."""
        try:
            with self.db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cols = self._detect_ci_competitors_columns(cur)
                    # Build dynamic update query
                    update_fields = []
                    params = []
                    
                    if "name" in updates:
                        update_fields.append("name = %s")
                        params.append(updates["name"])
                    
                    if "domain" in updates:
                        update_fields.append("domain = %s")
                        params.append(updates["domain"])
                    
                    if "tier" in updates:
                        update_fields.append('tier = %s::"CICompetitorTier"')
                        params.append(updates["tier"])
                    
                    if "industry" in updates:
                        update_fields.append('industry = %s::"CIIndustry"')
                        params.append(updates["industry"])
                    
                    if "description" in updates:
                        update_fields.append("description = %s")
                        params.append(updates["description"])
                    
                    if "platforms" in updates:
                        update_fields.append(f"{cols['platforms']} = %s::\"CIPlatform\"[]")
                        params.append(updates["platforms"])
                    
                    if "monitoring_keywords" in updates:
                        update_fields.append(f"{cols['monitoring_keywords']} = %s")
                        params.append(updates["monitoring_keywords"])
                    
                    if "is_active" in updates:
                        update_fields.append(f"{cols['is_active']} = %s")
                        params.append(updates["is_active"])
                    
                    if not update_fields:
                        # No updates provided
                        return await self.get_competitor(competitor_id)
                    
                    # Add updated timestamp
                    update_fields.append(f"{cols['updated_at']} = NOW()")
                    params.append(competitor_id)
                    
                    update_query = f"""
                        UPDATE ci_competitors 
                        SET {', '.join(update_fields)}
                        WHERE id = %s 
                        RETURNING *
                    """
                    
                    cur.execute(update_query, params)
                    result = cur.fetchone()
                    
                    if result:
                        competitor = self._format_competitor_data(dict(result))
                        
                        self.logger.info(f"Updated competitor {competitor_id}")
                        return competitor
                    else:
                        self.logger.warning(f"Competitor {competitor_id} not found for update")
                        return None
            
        except Exception as e:
            self.logger.error(f"Failed to update competitor {competitor_id}: {str(e)}")
            raise

    async def delete_competitor(self, competitor_id: str) -> bool:
        """Delete a competitor."""
        try:
            with self.db_config.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM ci_competitors WHERE id = %s", (competitor_id,))
                    deleted_count = cur.rowcount
                    
                    if deleted_count > 0:
                        self.logger.info(f"Deleted competitor {competitor_id}")
                        return True
                    else:
                        self.logger.warning(f"Competitor {competitor_id} not found for deletion")
                        return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete competitor {competitor_id}: {str(e)}")
            raise

    async def get_competitors_count_by_industry(self, industry: str) -> int:
        """Get count of competitors for a specific industry."""
        try:
            with self.db_config.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Detect columns using a lightweight cursor
                    cur2 = conn.cursor()
                    cols = self._detect_ci_competitors_columns(cur2)
                    cur2.close()
                    cur.execute(
                        f'SELECT COUNT(*) FROM ci_competitors WHERE industry = %s AND {cols["is_active"]} = %s',
                        (industry, True)
                    )
                    count = cur.fetchone()[0]
                    return count
        except Exception as e:
            self.logger.error(f"Failed to count competitors for industry {industry}: {str(e)}")
            return 0

# Global instance
ci_db = CompetitorIntelligenceDB()