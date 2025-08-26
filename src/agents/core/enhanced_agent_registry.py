"""
Enhanced Agent Registry System - Database-backed agent registry with dynamic discovery.

This module provides an advanced agent registry that integrates with the campaign
orchestration database, supporting dynamic agent discovery, health monitoring,
and performance tracking.
"""

import asyncio
import uuid
import json
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from .base_agent import BaseAgent, AgentType, AgentMetadata, AgentStatus
from ..orchestration.campaign_database_service import CampaignDatabaseService

logger = logging.getLogger(__name__)

class AgentHealthStatus(Enum):
    """Health status of registered agents."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

@dataclass
class AgentSpecification:
    """Complete specification for registering an agent type."""
    agent_type: AgentType
    class_path: str
    capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    performance_baseline: Dict[str, float] = field(default_factory=dict)
    health_check_endpoint: Optional[str] = None
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    max_instances: int = 10
    min_instances: int = 1
    scaling_policy: str = "auto"
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RegisteredAgent:
    """Information about a registered agent instance."""
    agent_id: str
    agent_type: AgentType
    specification_id: str
    status: AgentStatus = AgentStatus.IDLE
    health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    instance_count: int = 0
    last_health_check: Optional[datetime] = None
    performance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class AgentRegistryDB:
    """
    Database-backed agent registry with dynamic discovery.
    
    This registry maintains agent specifications and instances in the database,
    providing capabilities for agent discovery, health monitoring, and performance
    tracking across the campaign orchestration system.
    """
    
    def __init__(self, db_service: Optional[CampaignDatabaseService] = None):
        """Initialize the enhanced agent registry."""
        self.db_service = db_service or CampaignDatabaseService()
        
        # In-memory caches for performance
        self._specification_cache: Dict[str, AgentSpecification] = {}
        self._agent_cache: Dict[str, RegisteredAgent] = {}
        self._capability_index: Dict[str, List[str]] = {}  # capability -> agent_ids
        
        # Health monitoring
        self._health_check_interval = timedelta(minutes=5)
        self._performance_check_interval = timedelta(minutes=10)
        self._last_cache_update = datetime.min
        self._cache_ttl = timedelta(minutes=15)
        
        logger.info("Initialized Enhanced Agent Registry with database backing")
    
    async def register_agent_type(self, agent_spec: AgentSpecification) -> str:
        """
        Register agent specification in database with capabilities and requirements.
        
        Args:
            agent_spec: Complete agent specification
            
        Returns:
            str: ID of the registered agent specification
        """
        try:
            spec_id = str(uuid.uuid4())
            spec_data = asdict(agent_spec)
            spec_data["id"] = spec_id
            spec_data["created_at"] = datetime.utcnow()
            spec_data["updated_at"] = datetime.utcnow()
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Insert agent specification
                cur.execute("""
                    INSERT INTO agent_specifications (
                        id, agent_type, class_path, capabilities, requirements,
                        performance_baseline, health_check_endpoint, configuration_schema,
                        max_instances, min_instances, scaling_policy, tags, version,
                        description, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    spec_id, agent_spec.agent_type.value, agent_spec.class_path,
                    json.dumps(agent_spec.capabilities), json.dumps(agent_spec.requirements),
                    json.dumps(agent_spec.performance_baseline), agent_spec.health_check_endpoint,
                    json.dumps(agent_spec.configuration_schema), agent_spec.max_instances,
                    agent_spec.min_instances, agent_spec.scaling_policy,
                    json.dumps(agent_spec.tags), agent_spec.version, agent_spec.description,
                    spec_data["created_at"], spec_data["updated_at"]
                ))
                
                # Update cache
                self._specification_cache[spec_id] = agent_spec
                
                # Update capability index
                for capability in agent_spec.capabilities:
                    if capability not in self._capability_index:
                        self._capability_index[capability] = []
                    self._capability_index[capability].append(spec_id)
                
                logger.info(f"Registered agent type {agent_spec.agent_type.value} with ID {spec_id}")
                return spec_id
                
        except Exception as e:
            logger.error(f"Failed to register agent type {agent_spec.agent_type.value}: {str(e)}")
            raise
    
    async def discover_agents_by_capability(self, capabilities: List[str]) -> List[RegisteredAgent]:
        """
        Find agents that match required capabilities.
        
        Args:
            capabilities: List of required capabilities
            
        Returns:
            List[RegisteredAgent]: Agents matching the capabilities
        """
        try:
            await self._refresh_cache_if_needed()
            
            # Find specifications that match capabilities
            matching_specs = []
            for spec_id, spec in self._specification_cache.items():
                if all(cap in spec.capabilities for cap in capabilities):
                    matching_specs.append(spec_id)
            
            if not matching_specs:
                logger.info(f"No agents found with capabilities: {capabilities}")
                return []
            
            # Get registered agent instances for matching specs
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                placeholders = ",".join(["%s"] * len(matching_specs))
                cur.execute(f"""
                    SELECT agent_id, agent_type, specification_id, status, health_status,
                           instance_count, last_health_check, performance_score, metadata,
                           created_at, updated_at
                    FROM registered_agents 
                    WHERE specification_id IN ({placeholders})
                    AND status IN ('idle', 'running')
                    ORDER BY performance_score DESC, last_health_check DESC
                """, matching_specs)
                
                agents = []
                for row in cur.fetchall():
                    agent = RegisteredAgent(
                        agent_id=str(row[0]),
                        agent_type=AgentType(row[1]),
                        specification_id=str(row[2]),
                        status=AgentStatus(row[3]),
                        health_status=AgentHealthStatus(row[4]),
                        instance_count=row[5] or 0,
                        last_health_check=row[6],
                        performance_score=row[7] or 0.0,
                        metadata=row[8] or {},
                        created_at=row[9],
                        updated_at=row[10]
                    )
                    agents.append(agent)
                
                logger.info(f"Found {len(agents)} agents with capabilities: {capabilities}")
                return agents
                
        except Exception as e:
            logger.error(f"Failed to discover agents by capability: {str(e)}")
            return []
    
    async def get_agent_health_status(self, agent_id: str) -> AgentHealthStatus:
        """
        Check agent availability and performance metrics.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            AgentHealthStatus: Current health status of the agent
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get agent information
                cur.execute("""
                    SELECT ra.health_status, ra.last_health_check, ra.performance_score,
                           asp.health_check_endpoint
                    FROM registered_agents ra
                    JOIN agent_specifications asp ON ra.specification_id = asp.id
                    WHERE ra.agent_id = %s
                """, (agent_id,))
                
                row = cur.fetchone()
                if not row:
                    logger.warning(f"Agent {agent_id} not found in registry")
                    return AgentHealthStatus.UNKNOWN
                
                current_health = AgentHealthStatus(row[0])
                last_check = row[1]
                performance_score = row[2] or 0.0
                health_endpoint = row[3]
                
                # Check if health data is stale
                if last_check and datetime.utcnow() - last_check > self._health_check_interval:
                    # Perform health check if endpoint available
                    if health_endpoint:
                        new_health = await self._perform_health_check(agent_id, health_endpoint)
                        await self._update_agent_health(agent_id, new_health)
                        return new_health
                    else:
                        # Use performance score to infer health
                        if performance_score >= 0.8:
                            return AgentHealthStatus.HEALTHY
                        elif performance_score >= 0.6:
                            return AgentHealthStatus.DEGRADED
                        else:
                            return AgentHealthStatus.UNHEALTHY
                
                return current_health
                
        except Exception as e:
            logger.error(f"Failed to get health status for agent {agent_id}: {str(e)}")
            return AgentHealthStatus.UNKNOWN
    
    async def update_agent_performance_score(self, agent_id: str, score: float) -> bool:
        """
        Update agent performance based on execution results.
        
        Args:
            agent_id: ID of the agent
            score: Performance score (0.0 to 1.0)
            
        Returns:
            bool: True if update successful
        """
        try:
            # Validate score
            score = max(0.0, min(1.0, score))
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Update performance score and timestamp
                cur.execute("""
                    UPDATE registered_agents 
                    SET performance_score = %s, updated_at = %s
                    WHERE agent_id = %s
                """, (score, datetime.utcnow(), agent_id))
                
                if cur.rowcount > 0:
                    # Update cache if agent is cached
                    if agent_id in self._agent_cache:
                        self._agent_cache[agent_id].performance_score = score
                        self._agent_cache[agent_id].updated_at = datetime.utcnow()
                    
                    logger.info(f"Updated performance score for agent {agent_id}: {score}")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} not found for performance update")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update performance score for agent {agent_id}: {str(e)}")
            return False
    
    async def register_agent_instance(self, agent_spec_id: str, agent_instance: BaseAgent) -> str:
        """
        Register a new agent instance.
        
        Args:
            agent_spec_id: ID of the agent specification
            agent_instance: The agent instance to register
            
        Returns:
            str: ID of the registered agent instance
        """
        try:
            agent_id = agent_instance.metadata.agent_id
            
            registered_agent = RegisteredAgent(
                agent_id=agent_id,
                agent_type=agent_instance.metadata.agent_type,
                specification_id=agent_spec_id,
                status=agent_instance.get_status(),
                health_status=AgentHealthStatus.HEALTHY,
                instance_count=1,
                last_health_check=datetime.utcnow(),
                performance_score=0.8,  # Initial score
                metadata={
                    "name": agent_instance.metadata.name,
                    "version": agent_instance.metadata.version,
                    "capabilities": agent_instance.metadata.capabilities
                }
            )
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Insert registered agent
                cur.execute("""
                    INSERT INTO registered_agents (
                        agent_id, agent_type, specification_id, status, health_status,
                        instance_count, last_health_check, performance_score, metadata,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    agent_id, registered_agent.agent_type.value, agent_spec_id,
                    registered_agent.status.value, registered_agent.health_status.value,
                    registered_agent.instance_count, registered_agent.last_health_check,
                    registered_agent.performance_score, json.dumps(registered_agent.metadata),
                    registered_agent.created_at, registered_agent.updated_at
                ))
                
                # Update cache
                self._agent_cache[agent_id] = registered_agent
                
                logger.info(f"Registered agent instance {agent_id} for spec {agent_spec_id}")
                return agent_id
                
        except Exception as e:
            logger.error(f"Failed to register agent instance: {str(e)}")
            raise
    
    async def unregister_agent_instance(self, agent_id: str) -> bool:
        """
        Unregister an agent instance.
        
        Args:
            agent_id: ID of the agent instance to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Remove from database
                cur.execute("DELETE FROM registered_agents WHERE agent_id = %s", (agent_id,))
                
                # Remove from cache
                if agent_id in self._agent_cache:
                    del self._agent_cache[agent_id]
                
                logger.info(f"Unregistered agent instance {agent_id}")
                return cur.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to unregister agent instance {agent_id}: {str(e)}")
            return False
    
    async def get_agent_specifications(self, agent_type: Optional[AgentType] = None) -> List[AgentSpecification]:
        """
        Get all agent specifications, optionally filtered by type.
        
        Args:
            agent_type: Filter by agent type (optional)
            
        Returns:
            List[AgentSpecification]: Available specifications
        """
        try:
            await self._refresh_cache_if_needed()
            
            if agent_type:
                specs = [spec for spec in self._specification_cache.values() 
                        if spec.agent_type == agent_type]
            else:
                specs = list(self._specification_cache.values())
            
            logger.info(f"Retrieved {len(specs)} agent specifications")
            return specs
            
        except Exception as e:
            logger.error(f"Failed to get agent specifications: {str(e)}")
            return []
    
    async def get_registered_agents(self, 
                                  agent_type: Optional[AgentType] = None,
                                  health_status: Optional[AgentHealthStatus] = None) -> List[RegisteredAgent]:
        """
        Get all registered agent instances with optional filtering.
        
        Args:
            agent_type: Filter by agent type (optional)
            health_status: Filter by health status (optional)
            
        Returns:
            List[RegisteredAgent]: Registered agent instances
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Build query with filters
                query = """
                    SELECT agent_id, agent_type, specification_id, status, health_status,
                           instance_count, last_health_check, performance_score, metadata,
                           created_at, updated_at
                    FROM registered_agents 
                    WHERE 1=1
                """
                params = []
                
                if agent_type:
                    query += " AND agent_type = %s"
                    params.append(agent_type.value)
                
                if health_status:
                    query += " AND health_status = %s"
                    params.append(health_status.value)
                
                query += " ORDER BY performance_score DESC, last_health_check DESC"
                
                cur.execute(query, params)
                
                agents = []
                for row in cur.fetchall():
                    agent = RegisteredAgent(
                        agent_id=str(row[0]),
                        agent_type=AgentType(row[1]),
                        specification_id=str(row[2]),
                        status=AgentStatus(row[3]),
                        health_status=AgentHealthStatus(row[4]),
                        instance_count=row[5] or 0,
                        last_health_check=row[6],
                        performance_score=row[7] or 0.0,
                        metadata=row[8] or {},
                        created_at=row[9],
                        updated_at=row[10]
                    )
                    agents.append(agent)
                
                logger.info(f"Retrieved {len(agents)} registered agents")
                return agents
                
        except Exception as e:
            logger.error(f"Failed to get registered agents: {str(e)}")
            return []
    
    async def _refresh_cache_if_needed(self) -> None:
        """Refresh cache if TTL has expired."""
        if datetime.utcnow() - self._last_cache_update > self._cache_ttl:
            await self._refresh_cache()
    
    async def _refresh_cache(self) -> None:
        """Refresh all caches from database."""
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Refresh specification cache
                cur.execute("""
                    SELECT id, agent_type, class_path, capabilities, requirements,
                           performance_baseline, health_check_endpoint, configuration_schema,
                           max_instances, min_instances, scaling_policy, tags, version,
                           description, created_at, updated_at
                    FROM agent_specifications
                """)
                
                self._specification_cache.clear()
                self._capability_index.clear()
                
                for row in cur.fetchall():
                    spec = AgentSpecification(
                        agent_type=AgentType(row[1]),
                        class_path=row[2],
                        capabilities=json.loads(row[3]) if row[3] else [],
                        requirements=json.loads(row[4]) if row[4] else [],
                        performance_baseline=json.loads(row[5]) if row[5] else {},
                        health_check_endpoint=row[6],
                        configuration_schema=json.loads(row[7]) if row[7] else {},
                        max_instances=row[8] or 10,
                        min_instances=row[9] or 1,
                        scaling_policy=row[10] or "auto",
                        tags=json.loads(row[11]) if row[11] else [],
                        version=row[12] or "1.0.0",
                        description=row[13] or "",
                        created_at=row[14],
                        updated_at=row[15]
                    )
                    
                    spec_id = str(row[0])
                    self._specification_cache[spec_id] = spec
                    
                    # Update capability index
                    for capability in spec.capabilities:
                        if capability not in self._capability_index:
                            self._capability_index[capability] = []
                        self._capability_index[capability].append(spec_id)
                
                self._last_cache_update = datetime.utcnow()
                logger.info(f"Refreshed cache with {len(self._specification_cache)} specifications")
                
        except Exception as e:
            logger.error(f"Failed to refresh cache: {str(e)}")
    
    async def _perform_health_check(self, agent_id: str, endpoint: str) -> AgentHealthStatus:
        """Perform health check on agent endpoint."""
        try:
            # This is a placeholder for actual health check implementation
            # In a real implementation, this would make HTTP requests to the endpoint
            logger.info(f"Performing health check for agent {agent_id} at {endpoint}")
            
            # Simulate health check logic
            import random
            health_score = random.uniform(0.7, 1.0)
            
            if health_score >= 0.9:
                return AgentHealthStatus.HEALTHY
            elif health_score >= 0.7:
                return AgentHealthStatus.DEGRADED
            else:
                return AgentHealthStatus.UNHEALTHY
                
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {str(e)}")
            return AgentHealthStatus.UNHEALTHY
    
    async def _update_agent_health(self, agent_id: str, health_status: AgentHealthStatus) -> None:
        """Update agent health status in database."""
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    UPDATE registered_agents 
                    SET health_status = %s, last_health_check = %s, updated_at = %s
                    WHERE agent_id = %s
                """, (health_status.value, datetime.utcnow(), datetime.utcnow(), agent_id))
                
                # Update cache
                if agent_id in self._agent_cache:
                    self._agent_cache[agent_id].health_status = health_status
                    self._agent_cache[agent_id].last_health_check = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"Failed to update health status for agent {agent_id}: {str(e)}")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        try:
            await self._refresh_cache_if_needed()
            
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get agent counts by type
                cur.execute("""
                    SELECT agent_type, COUNT(*) as count
                    FROM registered_agents
                    GROUP BY agent_type
                """)
                agent_counts = {row[0]: row[1] for row in cur.fetchall()}
                
                # Get health status distribution
                cur.execute("""
                    SELECT health_status, COUNT(*) as count
                    FROM registered_agents
                    GROUP BY health_status
                """)
                health_distribution = {row[0]: row[1] for row in cur.fetchall()}
                
                # Get performance statistics
                cur.execute("""
                    SELECT AVG(performance_score) as avg_score,
                           MIN(performance_score) as min_score,
                           MAX(performance_score) as max_score
                    FROM registered_agents
                """)
                perf_row = cur.fetchone()
                
                stats = {
                    "total_specifications": len(self._specification_cache),
                    "total_registered_agents": sum(agent_counts.values()),
                    "agents_by_type": agent_counts,
                    "health_distribution": health_distribution,
                    "performance_stats": {
                        "average_score": float(perf_row[0]) if perf_row[0] else 0.0,
                        "min_score": float(perf_row[1]) if perf_row[1] else 0.0,
                        "max_score": float(perf_row[2]) if perf_row[2] else 0.0
                    },
                    "cache_stats": {
                        "specifications_cached": len(self._specification_cache),
                        "agents_cached": len(self._agent_cache),
                        "last_cache_update": self._last_cache_update.isoformat()
                    }
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get registry stats: {str(e)}")
            return {"error": str(e)}