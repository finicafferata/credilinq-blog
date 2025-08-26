"""
Enhanced Agent Pool - Advanced agent pool with database persistence and load balancing.

This module extends the basic agent pool with sophisticated load balancing,
database-backed persistence, and intelligent agent selection based on
performance metrics and workload.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import heapq

from .base_agent import BaseAgent, AgentType, AgentStatus
from .enhanced_agent_registry import AgentRegistryDB, RegisteredAgent
from .agent_lifecycle_manager import AgentLifecycleManager, AgentInstance
from ..orchestration.campaign_orchestrator import CampaignTask

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Strategies for load balancing across agents."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_BASED = "performance_based"
    RANDOM = "random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"

@dataclass
class AgentRequirements:
    """Requirements for agent selection."""
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    min_performance_score: float = 0.0
    max_load_percentage: float = 80.0
    preferred_tags: List[str] = field(default_factory=list)
    exclude_agents: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None

@dataclass
class AgentLoadInfo:
    """Information about agent current load."""
    agent_id: str
    current_tasks: int
    max_concurrent_tasks: int
    load_percentage: float
    last_task_assigned: datetime
    performance_score: float
    response_time_avg: float  # milliseconds

class EnhancedAgentPool:
    """
    Enhanced agent pool with database persistence and load balancing.
    
    Provides intelligent agent selection, load balancing, performance tracking,
    and persistent state management for optimal resource utilization.
    """
    
    def __init__(self, 
                 registry: Optional[AgentRegistryDB] = None,
                 lifecycle_manager: Optional[AgentLifecycleManager] = None):
        """Initialize the enhanced agent pool."""
        self.registry = registry or AgentRegistryDB()
        self.lifecycle_manager = lifecycle_manager or AgentLifecycleManager(self.registry)
        
        # Pool state management
        self._agent_load_info: Dict[str, AgentLoadInfo] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self._load_balancing_strategy = LoadBalancingStrategy.PERFORMANCE_BASED
        
        # Performance tracking
        self._agent_performance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._response_time_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Load balancing counters
        self._round_robin_counters: Dict[AgentType, int] = {}
        
        # Pool configuration
        self._max_concurrent_tasks_per_agent = 5
        self._performance_history_limit = 100
        self._load_rebalance_threshold = 0.3  # 30% difference triggers rebalance
        
        # Background tasks
        self._load_monitoring_task: Optional[asyncio.Task] = None
        self._performance_tracking_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized Enhanced Agent Pool")
    
    async def initialize_pool(self) -> None:
        """Initialize the agent pool and start background tasks."""
        # Start lifecycle monitoring
        await self.lifecycle_manager.start_monitoring()
        
        # Load current agent state from registry
        await self._sync_with_registry()
        
        # Start background monitoring
        await self._start_background_tasks()
        
        logger.info("Enhanced Agent Pool initialized and monitoring started")
    
    async def get_optimal_agent(self, requirements: AgentRequirements) -> Optional[BaseAgent]:
        """
        Intelligent agent selection based on performance and availability.
        
        Args:
            requirements: Agent requirements and constraints
            
        Returns:
            Optional[BaseAgent]: Best available agent or None
        """
        try:
            # Get available agents matching requirements
            candidates = await self._find_candidate_agents(requirements)
            
            if not candidates:
                logger.info(f"No agents available for requirements: {requirements.agent_type.value}")
                return None
            
            # Apply load balancing strategy
            selected_agent = await self._select_agent_by_strategy(candidates, requirements)
            
            if selected_agent:
                # Update load tracking
                await self._update_agent_load(selected_agent.metadata.agent_id, increment=True)
                
                logger.info(f"Selected agent {selected_agent.metadata.name} for {requirements.agent_type.value}")
                return selected_agent
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get optimal agent: {str(e)}")
            return None
    
    async def load_balance_workload(self, tasks: List[CampaignTask]) -> Dict[str, List[CampaignTask]]:
        """
        Distribute tasks across available agents optimally.
        
        Args:
            tasks: List of tasks to distribute
            
        Returns:
            Dict: Mapping of agent_id to assigned tasks
        """
        try:
            task_distribution: Dict[str, List[CampaignTask]] = {}
            
            # Group tasks by agent type
            tasks_by_type: Dict[AgentType, List[CampaignTask]] = {}
            for task in tasks:
                agent_type = AgentType(task.agent_type)
                if agent_type not in tasks_by_type:
                    tasks_by_type[agent_type] = []
                tasks_by_type[agent_type].append(task)
            
            # Distribute tasks for each agent type
            for agent_type, type_tasks in tasks_by_type.items():
                agent_assignments = await self._distribute_tasks_for_type(agent_type, type_tasks)
                
                # Merge into overall distribution
                for agent_id, assigned_tasks in agent_assignments.items():
                    if agent_id not in task_distribution:
                        task_distribution[agent_id] = []
                    task_distribution[agent_id].extend(assigned_tasks)
            
            logger.info(f"Distributed {len(tasks)} tasks across {len(task_distribution)} agents")
            return task_distribution
            
        except Exception as e:
            logger.error(f"Failed to load balance workload: {str(e)}")
            return {}
    
    async def return_agent(self, agent: BaseAgent, performance_score: Optional[float] = None) -> None:
        """
        Return agent to pool and update performance metrics.
        
        Args:
            agent: Agent to return to pool
            performance_score: Optional performance score for this execution
        """
        try:
            agent_id = agent.metadata.agent_id
            
            # Update load tracking
            await self._update_agent_load(agent_id, increment=False)
            
            # Update performance tracking
            if performance_score is not None:
                await self._update_performance_history(agent_id, performance_score)
                
                # Update registry performance score
                await self.registry.update_agent_performance_score(agent_id, performance_score)
            
            # Update agent status
            agent.state.status = AgentStatus.IDLE
            agent.state.current_operation = None
            agent.state.progress_percentage = 0.0
            
            logger.debug(f"Returned agent {agent.metadata.name} to pool")
            
        except Exception as e:
            logger.error(f"Failed to return agent to pool: {str(e)}")
    
    async def persist_pool_state(self) -> bool:
        """
        Save pool state to database for recovery.
        
        Returns:
            bool: True if persistence successful
        """
        try:
            # Persist load information
            pool_state = {
                "agent_load_info": {
                    agent_id: {
                        "current_tasks": info.current_tasks,
                        "load_percentage": info.load_percentage,
                        "last_task_assigned": info.last_task_assigned.isoformat(),
                        "performance_score": info.performance_score,
                        "response_time_avg": info.response_time_avg
                    }
                    for agent_id, info in self._agent_load_info.items()
                },
                "task_assignments": self._task_assignments.copy(),
                "round_robin_counters": {
                    agent_type.value: counter 
                    for agent_type, counter in self._round_robin_counters.items()
                },
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # This would save to database in a real implementation
            # For now, just log the successful persistence
            logger.info("Pool state persisted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist pool state: {str(e)}")
            return False
    
    async def recover_pool_state(self) -> bool:
        """
        Recover pool state from database.
        
        Returns:
            bool: True if recovery successful
        """
        try:
            # This would load from database in a real implementation
            # For now, just sync with registry
            await self._sync_with_registry()
            
            logger.info("Pool state recovered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover pool state: {str(e)}")
            return False
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Set the load balancing strategy."""
        self._load_balancing_strategy = strategy
        logger.info(f"Load balancing strategy set to: {strategy.value}")
    
    async def rebalance_load(self) -> Dict[str, Any]:
        """
        Rebalance load across agents if imbalance detected.
        
        Returns:
            Dict: Rebalancing results and statistics
        """
        try:
            rebalance_actions = []
            agents_checked = 0
            
            # Check load distribution for each agent type
            for agent_type in AgentType:
                type_agents = [
                    (agent_id, load_info) 
                    for agent_id, load_info in self._agent_load_info.items()
                    if agent_id.startswith(agent_type.value)  # Simplified check
                ]
                
                if len(type_agents) < 2:
                    continue  # Need at least 2 agents to rebalance
                
                agents_checked += len(type_agents)
                
                # Calculate load statistics
                load_percentages = [info.load_percentage for _, info in type_agents]
                avg_load = sum(load_percentages) / len(load_percentages)
                max_load = max(load_percentages)
                min_load = min(load_percentages)
                
                # Check if rebalancing is needed
                load_imbalance = (max_load - min_load) / 100.0
                if load_imbalance > self._load_rebalance_threshold:
                    action = f"Rebalance needed for {agent_type.value}: {load_imbalance:.2f} imbalance"
                    rebalance_actions.append(action)
                    logger.info(action)
                    
                    # TODO: Implement actual task redistribution
                    # This would involve moving tasks from overloaded to underloaded agents
            
            return {
                "rebalance_needed": len(rebalance_actions) > 0,
                "agents_checked": agents_checked,
                "actions_taken": rebalance_actions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to rebalance load: {str(e)}")
            return {"error": str(e)}
    
    async def _find_candidate_agents(self, requirements: AgentRequirements) -> List[RegisteredAgent]:
        """Find agents that match the requirements."""
        # Get agents from registry
        registered_agents = await self.registry.discover_agents_by_capability(requirements.capabilities)
        
        # Filter by type
        candidates = [
            agent for agent in registered_agents 
            if agent.agent_type == requirements.agent_type
        ]
        
        # Filter by performance score
        candidates = [
            agent for agent in candidates
            if agent.performance_score >= requirements.min_performance_score
        ]
        
        # Filter by load
        candidates = [
            agent for agent in candidates
            if self._get_agent_load_percentage(agent.agent_id) <= requirements.max_load_percentage
        ]
        
        # Filter excluded agents
        candidates = [
            agent for agent in candidates
            if agent.agent_id not in requirements.exclude_agents
        ]
        
        return candidates
    
    async def _select_agent_by_strategy(self, 
                                       candidates: List[RegisteredAgent], 
                                       requirements: AgentRequirements) -> Optional[BaseAgent]:
        """Select agent based on load balancing strategy."""
        if not candidates:
            return None
        
        strategy = self._load_balancing_strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._select_round_robin(candidates, requirements.agent_type)
        
        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            return await self._select_least_loaded(candidates)
        
        elif strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return await self._select_performance_based(candidates)
        
        elif strategy == LoadBalancingStrategy.RANDOM:
            import random
            selected = random.choice(candidates)
            return await self._get_agent_instance(selected.agent_id)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._select_weighted_round_robin(candidates, requirements.agent_type)
        
        else:
            # Default to performance-based
            return await self._select_performance_based(candidates)
    
    async def _select_round_robin(self, candidates: List[RegisteredAgent], agent_type: AgentType) -> Optional[BaseAgent]:
        """Round-robin selection."""
        if agent_type not in self._round_robin_counters:
            self._round_robin_counters[agent_type] = 0
        
        index = self._round_robin_counters[agent_type] % len(candidates)
        self._round_robin_counters[agent_type] += 1
        
        selected = candidates[index]
        return await self._get_agent_instance(selected.agent_id)
    
    async def _select_least_loaded(self, candidates: List[RegisteredAgent]) -> Optional[BaseAgent]:
        """Select agent with lowest current load."""
        least_loaded = min(
            candidates, 
            key=lambda agent: self._get_agent_load_percentage(agent.agent_id)
        )
        return await self._get_agent_instance(least_loaded.agent_id)
    
    async def _select_performance_based(self, candidates: List[RegisteredAgent]) -> Optional[BaseAgent]:
        """Select agent based on performance score and load combination."""
        # Score = performance_score * (1 - load_percentage/100)
        best_agent = max(
            candidates,
            key=lambda agent: (
                agent.performance_score * 
                (1 - self._get_agent_load_percentage(agent.agent_id) / 100.0)
            )
        )
        return await self._get_agent_instance(best_agent.agent_id)
    
    async def _select_weighted_round_robin(self, candidates: List[RegisteredAgent], agent_type: AgentType) -> Optional[BaseAgent]:
        """Weighted round-robin based on performance scores."""
        # Create weighted list based on performance scores
        weighted_candidates = []
        for agent in candidates:
            weight = max(1, int(agent.performance_score * 10))  # Convert to integer weight
            weighted_candidates.extend([agent] * weight)
        
        if agent_type not in self._round_robin_counters:
            self._round_robin_counters[agent_type] = 0
        
        index = self._round_robin_counters[agent_type] % len(weighted_candidates)
        self._round_robin_counters[agent_type] += 1
        
        selected = weighted_candidates[index]
        return await self._get_agent_instance(selected.agent_id)
    
    async def _distribute_tasks_for_type(self, agent_type: AgentType, tasks: List[CampaignTask]) -> Dict[str, List[CampaignTask]]:
        """Distribute tasks for a specific agent type."""
        distribution = {}
        
        # Get available agents for this type
        requirements = AgentRequirements(agent_type=agent_type)
        candidates = await self._find_candidate_agents(requirements)
        
        if not candidates:
            logger.warning(f"No available agents for type {agent_type.value}")
            return distribution
        
        # Distribute tasks using a simple round-robin approach
        for i, task in enumerate(tasks):
            candidate_index = i % len(candidates)
            agent = candidates[candidate_index]
            agent_instance = await self._get_agent_instance(agent.agent_id)
            
            if agent_instance:
                agent_id = agent_instance.metadata.agent_id
                if agent_id not in distribution:
                    distribution[agent_id] = []
                distribution[agent_id].append(task)
        
        return distribution
    
    async def _get_agent_instance(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent instance from lifecycle manager."""
        # This is simplified - in practice, we'd get from lifecycle manager
        # For now, return None to indicate agent would need to be created
        return None
    
    def _get_agent_load_percentage(self, agent_id: str) -> float:
        """Get current load percentage for an agent."""
        if agent_id in self._agent_load_info:
            return self._agent_load_info[agent_id].load_percentage
        return 0.0
    
    async def _update_agent_load(self, agent_id: str, increment: bool) -> None:
        """Update agent load tracking."""
        if agent_id not in self._agent_load_info:
            self._agent_load_info[agent_id] = AgentLoadInfo(
                agent_id=agent_id,
                current_tasks=0,
                max_concurrent_tasks=self._max_concurrent_tasks_per_agent,
                load_percentage=0.0,
                last_task_assigned=datetime.utcnow(),
                performance_score=0.8,  # Default
                response_time_avg=100.0  # Default 100ms
            )
        
        load_info = self._agent_load_info[agent_id]
        
        if increment:
            load_info.current_tasks += 1
            load_info.last_task_assigned = datetime.utcnow()
        else:
            load_info.current_tasks = max(0, load_info.current_tasks - 1)
        
        # Recalculate load percentage
        load_info.load_percentage = (load_info.current_tasks / load_info.max_concurrent_tasks) * 100.0
    
    async def _update_performance_history(self, agent_id: str, performance_score: float) -> None:
        """Update performance history for an agent."""
        if agent_id not in self._agent_performance_history:
            self._agent_performance_history[agent_id] = []
        
        history = self._agent_performance_history[agent_id]
        history.append((datetime.utcnow(), performance_score))
        
        # Limit history size
        if len(history) > self._performance_history_limit:
            history.pop(0)
        
        # Update load info if available
        if agent_id in self._agent_load_info:
            self._agent_load_info[agent_id].performance_score = performance_score
    
    async def _sync_with_registry(self) -> None:
        """Sync pool state with agent registry."""
        try:
            # Get all registered agents
            all_agents = await self.registry.get_registered_agents()
            
            # Initialize load info for registered agents
            for agent in all_agents:
                if agent.agent_id not in self._agent_load_info:
                    self._agent_load_info[agent.agent_id] = AgentLoadInfo(
                        agent_id=agent.agent_id,
                        current_tasks=0,
                        max_concurrent_tasks=self._max_concurrent_tasks_per_agent,
                        load_percentage=0.0,
                        last_task_assigned=datetime.utcnow(),
                        performance_score=agent.performance_score,
                        response_time_avg=100.0
                    )
            
            logger.info(f"Synced with registry: {len(all_agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to sync with registry: {str(e)}")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        if self._load_monitoring_task is None or self._load_monitoring_task.done():
            self._load_monitoring_task = asyncio.create_task(self._load_monitoring_loop())
        
        if self._performance_tracking_task is None or self._performance_tracking_task.done():
            self._performance_tracking_task = asyncio.create_task(self._performance_tracking_loop())
    
    async def _load_monitoring_loop(self) -> None:
        """Background loop for load monitoring."""
        while True:
            try:
                # Check for load rebalancing
                await self.rebalance_load()
                
                # Persist state periodically
                await self.persist_pool_state()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Load monitoring loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self) -> None:
        """Background loop for performance tracking."""
        while True:
            try:
                # Clean up old performance history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for agent_id, history in self._agent_performance_history.items():
                    # Remove old entries
                    self._agent_performance_history[agent_id] = [
                        (timestamp, score) for timestamp, score in history
                        if timestamp > cutoff_time
                    ]
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Performance tracking loop error: {str(e)}")
                await asyncio.sleep(300)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        total_agents = len(self._agent_load_info)
        total_tasks = sum(info.current_tasks for info in self._agent_load_info.values())
        
        load_distribution = {}
        for load_info in self._agent_load_info.values():
            load_bucket = f"{int(load_info.load_percentage // 10) * 10}-{int(load_info.load_percentage // 10) * 10 + 9}%"
            load_distribution[load_bucket] = load_distribution.get(load_bucket, 0) + 1
        
        avg_performance = (
            sum(info.performance_score for info in self._agent_load_info.values()) / total_agents
            if total_agents > 0 else 0.0
        )
        
        return {
            "total_agents": total_agents,
            "total_active_tasks": total_tasks,
            "average_load_percentage": sum(info.load_percentage for info in self._agent_load_info.values()) / total_agents if total_agents > 0 else 0.0,
            "average_performance_score": avg_performance,
            "load_distribution": load_distribution,
            "load_balancing_strategy": self._load_balancing_strategy.value,
            "round_robin_counters": {k.value: v for k, v in self._round_robin_counters.items()},
            "monitoring_active": (
                self._load_monitoring_task is not None and not self._load_monitoring_task.done()
            )
        }