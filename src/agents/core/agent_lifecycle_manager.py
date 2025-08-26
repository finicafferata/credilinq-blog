"""
Agent Lifecycle Manager - Manages agent creation, monitoring, and cleanup.

This module provides comprehensive lifecycle management for agents including
spawning, health monitoring, recovery, scaling, and graceful shutdown operations.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import signal
import psutil

from .base_agent import BaseAgent, AgentStatus, AgentMetadata, AgentType
from .enhanced_agent_registry import AgentRegistryDB, AgentSpecification, RegisteredAgent, AgentHealthStatus
from ..orchestration.campaign_database_service import CampaignDatabaseService

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Overall health status of an agent."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    CRITICAL = "critical"
    DEAD = "dead"

@dataclass
class AgentInstance:
    """Information about a running agent instance."""
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent: BaseAgent
    specification_id: str = ""
    process_id: Optional[int] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    health_status: HealthStatus = HealthStatus.HEALTHY
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingResult:
    """Result of scaling operation."""
    success: bool
    target_count: int
    current_count: int
    actions_taken: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class AgentLifecycleManager:
    """
    Manages agent creation, monitoring, and cleanup.
    
    Provides comprehensive lifecycle management including:
    - Agent instance creation and initialization
    - Health monitoring and heartbeat tracking
    - Automatic recovery from failures
    - Dynamic scaling based on workload
    - Graceful shutdown and resource cleanup
    """
    
    def __init__(self, 
                 registry: Optional[AgentRegistryDB] = None,
                 db_service: Optional[CampaignDatabaseService] = None):
        """Initialize the agent lifecycle manager."""
        self.registry = registry or AgentRegistryDB()
        self.db_service = db_service or CampaignDatabaseService()
        
        # Instance tracking
        self._active_instances: Dict[str, AgentInstance] = {}
        self._instances_by_type: Dict[AgentType, List[str]] = {}
        self._shutdown_requested = False
        
        # Monitoring configuration
        self._heartbeat_interval = timedelta(seconds=30)
        self._health_check_interval = timedelta(minutes=2)
        self._recovery_timeout = timedelta(minutes=5)
        self._max_instances_per_type = 50
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "instances_spawned": 0,
            "instances_recovered": 0,
            "instances_failed": 0,
            "scaling_operations": 0,
            "cleanup_operations": 0
        }
        
        logger.info("Initialized AgentLifecycleManager")
    
    async def spawn_agent(self, agent_spec: AgentSpecification, config: Dict[str, Any]) -> AgentInstance:
        """
        Create and initialize agent instance.
        
        Args:
            agent_spec: Agent specification for creation
            config: Configuration for the agent
            
        Returns:
            AgentInstance: Created and initialized agent instance
            
        Raises:
            ValueError: If agent creation fails
            RuntimeError: If instance limit exceeded
        """
        try:
            # Check instance limits
            current_count = len(self._instances_by_type.get(agent_spec.agent_type, []))
            if current_count >= min(agent_spec.max_instances, self._max_instances_per_type):
                raise RuntimeError(
                    f"Maximum instances ({agent_spec.max_instances}) reached for "
                    f"agent type {agent_spec.agent_type.value}"
                )
            
            # Create agent metadata
            agent_metadata = AgentMetadata(
                agent_type=agent_spec.agent_type,
                name=config.get("name", f"{agent_spec.agent_type.value}_agent"),
                description=agent_spec.description,
                version=agent_spec.version,
                capabilities=agent_spec.capabilities.copy(),
                requirements=agent_spec.requirements.copy(),
                max_retries=config.get("max_retries", 3),
                timeout_seconds=config.get("timeout_seconds"),
                tags=agent_spec.tags.copy()
            )
            
            # Create agent instance using the registry
            # This is a simplified approach - in practice, we'd use the plugin system
            from .agent_factory import create_agent
            agent = create_agent(agent_spec.agent_type, agent_metadata)
            
            # Create instance tracking object
            instance = AgentInstance(
                agent=agent,
                specification_id=config.get("specification_id", ""),
                process_id=None,  # Would be set for separate processes
                metadata=config.copy()
            )
            
            # Initialize agent-specific resources
            await self._initialize_agent_resources(instance)
            
            # Register instance
            self._active_instances[instance.instance_id] = instance
            
            # Update type mapping
            if agent_spec.agent_type not in self._instances_by_type:
                self._instances_by_type[agent_spec.agent_type] = []
            self._instances_by_type[agent_spec.agent_type].append(instance.instance_id)
            
            # Register in database registry
            await self.registry.register_agent_instance(instance.specification_id, agent)
            
            # Update statistics
            self._stats["instances_spawned"] += 1
            
            logger.info(f"Spawned agent instance {instance.instance_id} of type {agent_spec.agent_type.value}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to spawn agent: {str(e)}")
            raise
    
    async def monitor_agent_health(self, instance_id: str) -> HealthStatus:
        """
        Continuous health monitoring for an agent instance.
        
        Args:
            instance_id: ID of the instance to monitor
            
        Returns:
            HealthStatus: Current health status
        """
        if instance_id not in self._active_instances:
            logger.warning(f"Instance {instance_id} not found for health monitoring")
            return HealthStatus.DEAD
        
        instance = self._active_instances[instance_id]
        
        try:
            # Update heartbeat
            instance.last_heartbeat = datetime.utcnow()
            
            # Check agent status
            agent_status = instance.agent.get_status()
            
            # Monitor resource usage
            await self._update_resource_usage(instance)
            
            # Determine health status
            health_status = await self._assess_health_status(instance)
            instance.health_status = health_status
            
            # Log health issues
            if health_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                logger.warning(f"Instance {instance_id} health: {health_status.value}")
            elif health_status == HealthStatus.DEAD:
                logger.error(f"Instance {instance_id} is dead")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health monitoring failed for instance {instance_id}: {str(e)}")
            instance.error_count += 1
            instance.health_status = HealthStatus.CRITICAL
            return HealthStatus.CRITICAL
    
    async def restart_failed_agent(self, instance_id: str) -> bool:
        """
        Automatic agent recovery from failures.
        
        Args:
            instance_id: ID of the instance to restart
            
        Returns:
            bool: True if restart successful
        """
        if instance_id not in self._active_instances:
            logger.error(f"Cannot restart unknown instance {instance_id}")
            return False
        
        instance = self._active_instances[instance_id]
        
        try:
            # Check recovery attempts
            if instance.recovery_attempts >= instance.max_recovery_attempts:
                logger.error(f"Max recovery attempts ({instance.max_recovery_attempts}) exceeded for {instance_id}")
                await self._terminate_instance(instance_id)
                return False
            
            instance.recovery_attempts += 1
            logger.info(f"Attempting recovery {instance.recovery_attempts}/{instance.max_recovery_attempts} for {instance_id}")
            
            # Stop current agent
            await self._stop_agent_safely(instance)
            
            # Re-initialize agent resources
            await self._initialize_agent_resources(instance)
            
            # Reset error count and health status
            instance.error_count = 0
            instance.health_status = HealthStatus.HEALTHY
            instance.last_heartbeat = datetime.utcnow()
            
            # Update statistics
            self._stats["instances_recovered"] += 1
            
            logger.info(f"Successfully recovered instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed for instance {instance_id}: {str(e)}")
            instance.error_count += 1
            self._stats["instances_failed"] += 1
            return False
    
    async def scale_agent_pool(self, agent_type: AgentType, target_count: int) -> ScalingResult:
        """
        Dynamic scaling based on workload.
        
        Args:
            agent_type: Type of agents to scale
            target_count: Target number of instances
            
        Returns:
            ScalingResult: Result of scaling operation
        """
        try:
            current_instances = self._instances_by_type.get(agent_type, [])
            current_count = len(current_instances)
            
            result = ScalingResult(
                success=True,
                target_count=target_count,
                current_count=current_count
            )
            
            if target_count == current_count:
                result.actions_taken.append("No scaling needed")
                return result
            
            # Get agent specification for this type
            specifications = await self.registry.get_agent_specifications(agent_type)
            if not specifications:
                result.success = False
                result.errors.append(f"No specifications found for agent type {agent_type.value}")
                return result
            
            agent_spec = specifications[0]  # Use first available specification
            
            if target_count > current_count:
                # Scale up - create new instances
                instances_to_create = target_count - current_count
                
                # Check limits
                max_allowed = min(agent_spec.max_instances, self._max_instances_per_type)
                if target_count > max_allowed:
                    target_count = max_allowed
                    instances_to_create = target_count - current_count
                    result.target_count = target_count
                    result.actions_taken.append(f"Limited target to {max_allowed} due to constraints")
                
                # Create new instances
                for i in range(instances_to_create):
                    try:
                        config = {
                            "name": f"{agent_type.value}_scaled_{i}",
                            "specification_id": "scaling_operation"
                        }
                        
                        instance = await self.spawn_agent(agent_spec, config)
                        result.actions_taken.append(f"Created instance {instance.instance_id}")
                        
                    except Exception as e:
                        result.errors.append(f"Failed to create instance {i}: {str(e)}")
                        result.success = False
            
            elif target_count < current_count:
                # Scale down - remove instances
                instances_to_remove = current_count - target_count
                
                # Ensure minimum instances
                if target_count < agent_spec.min_instances:
                    target_count = agent_spec.min_instances
                    instances_to_remove = current_count - target_count
                    result.target_count = target_count
                    result.actions_taken.append(f"Adjusted target to {target_count} to meet minimum")
                
                # Remove least healthy instances first
                instances_to_terminate = await self._select_instances_for_termination(
                    current_instances, instances_to_remove
                )
                
                for instance_id in instances_to_terminate:
                    try:
                        await self._terminate_instance(instance_id)
                        result.actions_taken.append(f"Terminated instance {instance_id}")
                    except Exception as e:
                        result.errors.append(f"Failed to terminate {instance_id}: {str(e)}")
                        result.success = False
            
            # Update final count
            result.current_count = len(self._instances_by_type.get(agent_type, []))
            
            # Update statistics
            self._stats["scaling_operations"] += 1
            
            logger.info(f"Scaling operation for {agent_type.value}: {current_count} -> {result.current_count}")
            return result
            
        except Exception as e:
            logger.error(f"Scaling failed for {agent_type.value}: {str(e)}")
            return ScalingResult(
                success=False,
                target_count=target_count,
                current_count=len(self._instances_by_type.get(agent_type, [])),
                errors=[str(e)]
            )
    
    async def graceful_shutdown(self, instance_id: str, timeout: int = 30) -> bool:
        """
        Clean agent shutdown with state preservation.
        
        Args:
            instance_id: ID of the instance to shutdown
            timeout: Timeout in seconds for graceful shutdown
            
        Returns:
            bool: True if shutdown successful
        """
        if instance_id not in self._active_instances:
            logger.warning(f"Instance {instance_id} not found for shutdown")
            return False
        
        instance = self._active_instances[instance_id]
        
        try:
            logger.info(f"Initiating graceful shutdown for instance {instance_id}")
            
            # Stop accepting new work
            instance.agent.state.status = AgentStatus.CANCELLED
            
            # Wait for current operations to complete
            start_time = datetime.utcnow()
            while (datetime.utcnow() - start_time).total_seconds() < timeout:
                if instance.agent.state.status in [AgentStatus.IDLE, AgentStatus.COMPLETED, AgentStatus.FAILED]:
                    break
                await asyncio.sleep(1)
            
            # Force stop if still running
            if instance.agent.state.status == AgentStatus.RUNNING:
                logger.warning(f"Force stopping instance {instance_id} after timeout")
            
            # Cleanup resources
            await self._cleanup_agent_resources(instance)
            
            # Remove from tracking
            await self._terminate_instance(instance_id)
            
            logger.info(f"Successfully shutdown instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed for {instance_id}: {str(e)}")
            return False
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started agent monitoring loop")
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started agent cleanup loop")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self._shutdown_requested = True
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped agent monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background loop for monitoring agent health."""
        while not self._shutdown_requested:
            try:
                # Monitor all active instances
                monitor_tasks = []
                for instance_id in list(self._active_instances.keys()):
                    task = asyncio.create_task(self.monitor_agent_health(instance_id))
                    monitor_tasks.append(task)
                
                if monitor_tasks:
                    await asyncio.gather(*monitor_tasks, return_exceptions=True)
                
                # Check for recovery needs
                await self._check_recovery_needs()
                
                # Sleep until next check
                await asyncio.sleep(self._health_check_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(30)  # Error backoff
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup operations."""
        while not self._shutdown_requested:
            try:
                # Clean up dead instances
                await self._cleanup_dead_instances()
                
                # Update registry health status
                await self._update_registry_health()
                
                # Sleep until next cleanup
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(60)  # Error backoff
    
    async def _initialize_agent_resources(self, instance: AgentInstance) -> None:
        """Initialize agent-specific resources."""
        try:
            # Set up any required resources
            # This is where you'd initialize databases, connections, etc.
            logger.debug(f"Initialized resources for instance {instance.instance_id}")
        except Exception as e:
            logger.error(f"Failed to initialize resources for {instance.instance_id}: {str(e)}")
            raise
    
    async def _cleanup_agent_resources(self, instance: AgentInstance) -> None:
        """Cleanup agent resources."""
        try:
            # Clean up any allocated resources
            logger.debug(f"Cleaned up resources for instance {instance.instance_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup resources for {instance.instance_id}: {str(e)}")
    
    async def _update_resource_usage(self, instance: AgentInstance) -> None:
        """Update resource usage metrics for an instance."""
        try:
            # Get current process info if available
            if instance.process_id:
                process = psutil.Process(instance.process_id)
                instance.resource_usage = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "threads": process.num_threads()
                }
        except (psutil.NoSuchProcess, AttributeError):
            # Process not found or not available
            instance.resource_usage = {"cpu_percent": 0, "memory_mb": 0, "threads": 0}
    
    async def _assess_health_status(self, instance: AgentInstance) -> HealthStatus:
        """Assess the health status of an instance."""
        try:
            # Check heartbeat freshness
            time_since_heartbeat = datetime.utcnow() - instance.last_heartbeat
            if time_since_heartbeat > timedelta(minutes=10):
                return HealthStatus.DEAD
            elif time_since_heartbeat > timedelta(minutes=5):
                return HealthStatus.CRITICAL
            
            # Check error count
            if instance.error_count > 10:
                return HealthStatus.CRITICAL
            elif instance.error_count > 5:
                return HealthStatus.DEGRADED
            
            # Check agent status
            agent_status = instance.agent.get_status()
            if agent_status == AgentStatus.FAILED:
                return HealthStatus.CRITICAL
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.CRITICAL
    
    async def _stop_agent_safely(self, instance: AgentInstance) -> None:
        """Safely stop an agent instance."""
        try:
            # Set status to cancelled
            instance.agent.state.status = AgentStatus.CANCELLED
            
            # Wait briefly for graceful stop
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to stop agent safely: {str(e)}")
    
    async def _terminate_instance(self, instance_id: str) -> None:
        """Terminate and remove an instance."""
        if instance_id in self._active_instances:
            instance = self._active_instances[instance_id]
            
            # Cleanup resources
            await self._cleanup_agent_resources(instance)
            
            # Remove from tracking
            del self._active_instances[instance_id]
            
            # Remove from type mapping
            agent_type = instance.agent.metadata.agent_type
            if agent_type in self._instances_by_type:
                if instance_id in self._instances_by_type[agent_type]:
                    self._instances_by_type[agent_type].remove(instance_id)
            
            # Unregister from database
            await self.registry.unregister_agent_instance(instance.agent.metadata.agent_id)
            
            self._stats["cleanup_operations"] += 1
    
    async def _select_instances_for_termination(self, instance_ids: List[str], count: int) -> List[str]:
        """Select instances for termination based on health."""
        # Sort by health status (worst first) and then by last heartbeat
        instances_with_health = []
        
        for instance_id in instance_ids:
            if instance_id in self._active_instances:
                instance = self._active_instances[instance_id]
                health_score = {
                    HealthStatus.DEAD: 0,
                    HealthStatus.CRITICAL: 1,
                    HealthStatus.DEGRADED: 2,
                    HealthStatus.HEALTHY: 3
                }.get(instance.health_status, 0)
                
                instances_with_health.append((instance_id, health_score, instance.last_heartbeat))
        
        # Sort by health score (ascending) and heartbeat (ascending)
        instances_with_health.sort(key=lambda x: (x[1], x[2]))
        
        # Return worst instances up to count
        return [instance_id for instance_id, _, _ in instances_with_health[:count]]
    
    async def _check_recovery_needs(self) -> None:
        """Check which instances need recovery."""
        for instance_id, instance in list(self._active_instances.items()):
            if instance.health_status in [HealthStatus.CRITICAL, HealthStatus.DEAD]:
                if instance.recovery_attempts < instance.max_recovery_attempts:
                    logger.info(f"Triggering recovery for unhealthy instance {instance_id}")
                    await self.restart_failed_agent(instance_id)
    
    async def _cleanup_dead_instances(self) -> None:
        """Clean up instances that are beyond recovery."""
        dead_instances = [
            instance_id for instance_id, instance in self._active_instances.items()
            if (instance.health_status == HealthStatus.DEAD and 
                instance.recovery_attempts >= instance.max_recovery_attempts)
        ]
        
        for instance_id in dead_instances:
            logger.info(f"Cleaning up dead instance {instance_id}")
            await self._terminate_instance(instance_id)
    
    async def _update_registry_health(self) -> None:
        """Update health status in the registry."""
        for instance_id, instance in self._active_instances.items():
            agent_health = {
                HealthStatus.HEALTHY: AgentHealthStatus.HEALTHY,
                HealthStatus.DEGRADED: AgentHealthStatus.DEGRADED,
                HealthStatus.CRITICAL: AgentHealthStatus.UNHEALTHY,
                HealthStatus.DEAD: AgentHealthStatus.OFFLINE
            }.get(instance.health_status, AgentHealthStatus.UNKNOWN)
            
            # This would update the registry - simplified for now
            logger.debug(f"Instance {instance_id} health: {agent_health.value}")
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle management statistics."""
        active_by_type = {}
        health_distribution = {}
        
        for instance in self._active_instances.values():
            agent_type = instance.agent.metadata.agent_type.value
            active_by_type[agent_type] = active_by_type.get(agent_type, 0) + 1
            
            health_status = instance.health_status.value
            health_distribution[health_status] = health_distribution.get(health_status, 0) + 1
        
        return {
            **self._stats,
            "active_instances": len(self._active_instances),
            "active_by_type": active_by_type,
            "health_distribution": health_distribution,
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
        }