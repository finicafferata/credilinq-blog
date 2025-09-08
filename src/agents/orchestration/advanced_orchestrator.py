#!/usr/bin/env python3
"""
Advanced Workflow Orchestrator with Dynamic Re-sequencing and Failure Recovery

This module provides sophisticated workflow orchestration capabilities including:
- Dynamic execution order adjustment based on agent completion and failures
- Intelligent failure recovery and re-sequencing strategies  
- Critical path analysis and optimization
- Alternative execution path evaluation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from .workflow_executor import WorkflowExecutor, WorkflowExecution, ExecutionStatus, AgentExecutionResult
from .master_planner_agent import MasterPlannerAgent, ExecutionPlan
from ..core.base_agent import AgentType

# Graceful degradation imports
try:
    from ...core.graceful_degradation_manager import GracefulDegradationManager, DegradationLevel, ServiceHealthStatus
    GRACEFUL_DEGRADATION_AVAILABLE = True
except ImportError:
    GracefulDegradationManager = None
    DegradationLevel = None
    ServiceHealthStatus = None
    GRACEFUL_DEGRADATION_AVAILABLE = False
    logger.warning("GracefulDegradationManager not available - degradation features disabled")

logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Failure recovery strategies."""
    SKIP_AND_CONTINUE = "skip_and_continue"
    RETRY_WITH_ALTERNATIVES = "retry_with_alternatives" 
    REPLAN_REMAINING = "replan_remaining"
    ABORT_WORKFLOW = "abort_workflow"

class PriorityLevel(Enum):
    """Agent execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class CriticalPath:
    """Critical path analysis results."""
    path_agents: List[str]
    total_estimated_time: float
    bottleneck_agent: str
    optimization_opportunities: List[str]

@dataclass
class ResequenceDecision:
    """Decision record for workflow re-sequencing."""
    decision_id: str
    timestamp: datetime
    trigger_event: str
    original_sequence: List[str]
    new_sequence: List[str]
    reasoning: str
    expected_impact: str
    confidence_score: float

class AdvancedOrchestrator:
    """
    Advanced workflow orchestrator that extends WorkflowExecutor with intelligent
    orchestration capabilities including dynamic re-sequencing and failure recovery.
    """
    
    def __init__(self, enable_recovery_systems: bool = True):
        self.workflow_executor = WorkflowExecutor(enable_recovery_systems=enable_recovery_systems)
        self.master_planner = MasterPlannerAgent()
        
        # Graceful degradation integration
        self.enable_degradation = enable_recovery_systems and GRACEFUL_DEGRADATION_AVAILABLE
        if self.enable_degradation:
            self.degradation_manager = GracefulDegradationManager()
            logger.info("✅ AdvancedOrchestrator: Graceful degradation enabled")
        else:
            self.degradation_manager = None
            logger.info("⚠️ AdvancedOrchestrator: Graceful degradation disabled")
        
        # Advanced orchestration state
        self.active_orchestrations: Dict[str, Dict[str, Any]] = {}
        self.resequence_decisions: Dict[str, List[ResequenceDecision]] = {}
        self.critical_paths: Dict[str, CriticalPath] = {}
        
        # Agent performance history for intelligent scheduling
        self.agent_performance_history: Dict[str, Dict[str, float]] = {}
        
        # Alternative execution paths
        self.alternative_paths: Dict[str, List[List[str]]] = {}
        
        logger.info("AdvancedOrchestrator initialized with dynamic re-sequencing and degradation capabilities")
    
    async def execute_adaptive_workflow(
        self,
        execution_plan: ExecutionPlan,
        context_data: Optional[Dict[str, Any]] = None,
        enable_resequencing: bool = True,
        failure_recovery_strategy: RecoveryStrategy = RecoveryStrategy.REPLAN_REMAINING
    ) -> WorkflowExecution:
        """
        Execute workflow with advanced orchestration capabilities.
        
        Args:
            execution_plan: Master Planner execution plan
            context_data: Context data for agents
            enable_resequencing: Whether to enable dynamic re-sequencing
            failure_recovery_strategy: Strategy for handling agent failures
            
        Returns:
            WorkflowExecution: Enhanced execution with orchestration metadata
        """
        workflow_id = execution_plan.workflow_execution_id
        
        logger.info(f"Starting adaptive workflow execution {workflow_id}")
        
        # Check system degradation level before starting
        degradation_level = DegradationLevel.NONE
        if self.enable_degradation and self.degradation_manager:
            # Use get_system_status to get degradation level
            system_status = await self.degradation_manager.get_system_status()
            degradation_level = system_status.get("degradation_level", DegradationLevel.NONE)
            if degradation_level != DegradationLevel.NONE:
                logger.warning(f"🚨 System degradation detected: {degradation_level.value}")
                
                # Apply degradation strategy to execution plan
                execution_plan = await self._apply_degradation_strategy(execution_plan, degradation_level)
        
        # Initialize orchestration state
        self.active_orchestrations[workflow_id] = {
            "start_time": datetime.utcnow(),
            "original_plan": execution_plan,
            "current_sequence": execution_plan.agent_sequence.copy(),
            "enable_resequencing": enable_resequencing,
            "recovery_strategy": failure_recovery_strategy,
            "resequence_count": 0,
            "recovery_attempts": 0,
            "performance_adjustments": [],
            "degradation_level": degradation_level,
            "degradation_applied": degradation_level != DegradationLevel.NONE
        }
        
        # Analyze critical path
        critical_path = await self._analyze_critical_path(execution_plan)
        self.critical_paths[workflow_id] = critical_path
        
        # Generate alternative execution paths
        alternatives = await self._generate_alternative_paths(execution_plan)
        self.alternative_paths[workflow_id] = alternatives
        
        # Start base workflow execution with monitoring
        workflow_execution = await self._execute_with_monitoring(
            execution_plan, context_data, workflow_id
        )
        
        return workflow_execution
    
    async def _execute_with_monitoring(
        self,
        execution_plan: ExecutionPlan,
        context_data: Optional[Dict[str, Any]],
        workflow_id: str
    ) -> WorkflowExecution:
        """Execute workflow with real-time monitoring and dynamic adjustments."""
        
        # Custom status callback for advanced orchestration
        async def orchestration_callback(wf_id: str, event: str, execution_state):
            await self._handle_orchestration_event(wf_id, event, execution_state)
        
        # Execute workflow with orchestration monitoring
        return await self.workflow_executor.execute_workflow(
            execution_plan, context_data, orchestration_callback
        )
    
    async def _handle_orchestration_event(
        self,
        workflow_id: str,
        event: str,
        execution_state: Optional[WorkflowExecution]
    ):
        """Handle workflow events for orchestration decisions."""
        
        if workflow_id not in self.active_orchestrations:
            return
        
        orchestration = self.active_orchestrations[workflow_id]
        
        logger.info(f"Orchestration event for {workflow_id}: {event}")
        
        # Handle different event types
        if event.startswith("agent_completed:"):
            agent_name = event.split(":")[1]
            await self._handle_agent_completion(workflow_id, agent_name, execution_state)
        
        elif event.startswith("agent_failed:"):
            agent_name = event.split(":")[1]
            await self._handle_agent_failure(workflow_id, agent_name, execution_state)
        
        elif event == "workflow_stalled":
            await self._handle_workflow_stall(workflow_id, execution_state)
        
        elif event.startswith("agent_starting:"):
            agent_name = event.split(":")[1]
            await self._handle_agent_start(workflow_id, agent_name)
    
    async def _handle_agent_completion(
        self,
        workflow_id: str,
        agent_name: str,
        execution_state: Optional[WorkflowExecution]
    ):
        """Handle successful agent completion - opportunity for optimization."""
        
        if not execution_state:
            return
        
        orchestration = self.active_orchestrations[workflow_id]
        
        # Record performance data
        if agent_name in execution_state.agent_results:
            result = execution_state.agent_results[agent_name]
            if result.execution_time_seconds:
                self._record_agent_performance(agent_name, result.execution_time_seconds)
        
        # Check for optimization opportunities
        if orchestration["enable_resequencing"]:
            await self._evaluate_sequence_optimization(workflow_id, execution_state)
    
    async def _handle_agent_failure(
        self,
        workflow_id: str,
        agent_name: str,
        execution_state: Optional[WorkflowExecution]
    ):
        """Handle agent failure with recovery strategies."""
        
        if not execution_state:
            return
        
        orchestration = self.active_orchestrations[workflow_id]
        recovery_strategy = orchestration["recovery_strategy"]
        
        logger.warning(f"Agent {agent_name} failed in workflow {workflow_id}, applying recovery strategy: {recovery_strategy.value}")
        
        if recovery_strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
            await self._skip_agent_and_continue(workflow_id, agent_name, execution_state)
        
        elif recovery_strategy == RecoveryStrategy.RETRY_WITH_ALTERNATIVES:
            await self._retry_with_alternative_agent(workflow_id, agent_name, execution_state)
        
        elif recovery_strategy == RecoveryStrategy.REPLAN_REMAINING:
            await self._replan_remaining_workflow(workflow_id, agent_name, execution_state)
        
        elif recovery_strategy == RecoveryStrategy.ABORT_WORKFLOW:
            logger.error(f"Aborting workflow {workflow_id} due to agent failure")
            self.workflow_executor.cancel_workflow(workflow_id)
        
        orchestration["recovery_attempts"] += 1
    
    async def _evaluate_sequence_optimization(
        self,
        workflow_id: str,
        execution_state: WorkflowExecution
    ):
        """Evaluate if workflow sequence can be optimized based on current state."""
        
        orchestration = self.active_orchestrations[workflow_id]
        
        # Get remaining agents
        remaining_agents = [
            agent for agent in execution_state.execution_plan.agent_sequence
            if agent["agent_name"] not in execution_state.completed_agents
        ]
        
        if len(remaining_agents) < 2:
            return  # Not enough agents left to optimize
        
        # Check if we can promote parallel-eligible agents
        promoted_agents = []
        for agent_step in remaining_agents:
            agent_name = agent_step["agent_name"]
            dependencies = agent_step["dependencies"]
            
            # Check if all dependencies are now complete
            deps_complete = all(dep in execution_state.completed_agents for dep in dependencies)
            
            # Check if agent is currently waiting but could be promoted
            if (agent_name in execution_state.agent_results and
                execution_state.agent_results[agent_name].status == ExecutionStatus.PENDING and
                deps_complete):
                promoted_agents.append(agent_name)
        
        if promoted_agents:
            decision = ResequenceDecision(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                trigger_event=f"dependencies_met_for_{len(promoted_agents)}_agents",
                original_sequence=[a["agent_name"] for a in remaining_agents],
                new_sequence=promoted_agents + [a["agent_name"] for a in remaining_agents if a["agent_name"] not in promoted_agents],
                reasoning=f"Promoting {len(promoted_agents)} agents with satisfied dependencies to parallel execution",
                expected_impact="Reduced execution time through increased parallelism",
                confidence_score=0.85
            )
            
            await self._apply_resequencing_decision(workflow_id, decision)
    
    async def _replan_remaining_workflow(
        self,
        workflow_id: str,
        failed_agent: str,
        execution_state: WorkflowExecution
    ):
        """Replan remaining workflow after agent failure."""
        
        orchestration = self.active_orchestrations[workflow_id]
        
        # Get remaining agents (excluding failed one)
        remaining_agents = [
            agent for agent in execution_state.execution_plan.agent_sequence
            if (agent["agent_name"] not in execution_state.completed_agents and 
                agent["agent_name"] not in execution_state.failed_agents)
        ]
        
        # Remove agents that depend on the failed agent
        agents_to_skip = set()
        for agent_step in remaining_agents:
            if failed_agent in agent_step["dependencies"]:
                agents_to_skip.add(agent_step["agent_name"])
        
        # Find alternative execution path
        viable_agents = [
            agent for agent in remaining_agents
            if agent["agent_name"] not in agents_to_skip
        ]
        
        if viable_agents:
            decision = ResequenceDecision(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                trigger_event=f"agent_failure_{failed_agent}",
                original_sequence=[a["agent_name"] for a in remaining_agents],
                new_sequence=[a["agent_name"] for a in viable_agents],
                reasoning=f"Replanned workflow after {failed_agent} failure, skipping {len(agents_to_skip)} dependent agents",
                expected_impact=f"Workflow continues with {len(viable_agents)} remaining agents",
                confidence_score=0.75
            )
            
            await self._apply_resequencing_decision(workflow_id, decision)
        
        logger.info(f"Replanned workflow {workflow_id}: continuing with {len(viable_agents)} agents, skipped {len(agents_to_skip)} dependent agents")
    
    async def _apply_resequencing_decision(
        self,
        workflow_id: str,
        decision: ResequenceDecision
    ):
        """Apply a resequencing decision to the active workflow."""
        
        orchestration = self.active_orchestrations[workflow_id]
        
        # Record decision
        if workflow_id not in self.resequence_decisions:
            self.resequence_decisions[workflow_id] = []
        self.resequence_decisions[workflow_id].append(decision)
        
        orchestration["resequence_count"] += 1
        orchestration["performance_adjustments"].append({
            "timestamp": decision.timestamp.isoformat(),
            "type": "resequencing", 
            "reasoning": decision.reasoning,
            "impact": decision.expected_impact
        })
        
        logger.info(f"Applied resequencing decision {decision.decision_id} for workflow {workflow_id}: {decision.reasoning}")
    
    async def _analyze_critical_path(self, execution_plan: ExecutionPlan) -> CriticalPath:
        """Analyze critical path for workflow optimization."""
        
        # Build dependency graph
        agents = {step["agent_name"]: step for step in execution_plan.agent_sequence}
        
        # Calculate estimated times (using performance history or defaults)
        estimated_times = {}
        for agent_name in agents:
            estimated_times[agent_name] = self._get_estimated_execution_time(agent_name)
        
        # Find critical path using topological sort + longest path
        critical_path = self._find_longest_path(agents, estimated_times)
        
        # Identify bottleneck (agent with longest estimated time on critical path)
        bottleneck = max(critical_path, key=lambda agent: estimated_times[agent])
        
        # Identify optimization opportunities
        optimizations = []
        
        # Look for parallel execution opportunities
        for agent_name, agent_step in agents.items():
            if agent_name not in critical_path and len(agent_step["dependencies"]) <= 1:
                optimizations.append(f"Agent {agent_name} could run in parallel")
        
        # Look for agents that could be pre-started
        for agent_name in critical_path:
            dependencies = agents[agent_name]["dependencies"]
            if len(dependencies) == 1:
                optimizations.append(f"Agent {agent_name} could start immediately after {dependencies[0]}")
        
        return CriticalPath(
            path_agents=critical_path,
            total_estimated_time=sum(estimated_times[agent] for agent in critical_path),
            bottleneck_agent=bottleneck,
            optimization_opportunities=optimizations
        )
    
    def _find_longest_path(self, agents: Dict[str, Any], estimated_times: Dict[str, float]) -> List[str]:
        """Find the critical path (longest path) through the dependency graph."""
        
        # Simple critical path algorithm - in practice, this would be more sophisticated
        visited = set()
        path_lengths = {}
        paths = {}
        
        def dfs(agent_name: str) -> float:
            if agent_name in visited:
                return path_lengths.get(agent_name, 0)
            
            visited.add(agent_name)
            
            dependencies = agents[agent_name]["dependencies"]
            if not dependencies:
                path_lengths[agent_name] = estimated_times[agent_name]
                paths[agent_name] = [agent_name]
                return path_lengths[agent_name]
            
            max_dep_length = 0
            best_dep_path = []
            
            for dep in dependencies:
                dep_length = dfs(dep)
                if dep_length > max_dep_length:
                    max_dep_length = dep_length
                    best_dep_path = paths[dep].copy()
            
            path_lengths[agent_name] = max_dep_length + estimated_times[agent_name]
            paths[agent_name] = best_dep_path + [agent_name]
            
            return path_lengths[agent_name]
        
        # Find longest path
        max_length = 0
        critical_path = []
        
        for agent_name in agents:
            length = dfs(agent_name)
            if length > max_length:
                max_length = length
                critical_path = paths[agent_name]
        
        return critical_path
    
    async def _generate_alternative_paths(self, execution_plan: ExecutionPlan) -> List[List[str]]:
        """Generate alternative execution paths for resilience."""
        
        # This is a simplified implementation - in practice, would be more sophisticated
        alternatives = []
        
        # Generate path with maximum parallelism
        agents_by_dependencies = {}
        for step in execution_plan.agent_sequence:
            dep_count = len(step["dependencies"])
            if dep_count not in agents_by_dependencies:
                agents_by_dependencies[dep_count] = []
            agents_by_dependencies[dep_count].append(step["agent_name"])
        
        # Create parallel execution groups
        parallel_path = []
        for dep_count in sorted(agents_by_dependencies.keys()):
            parallel_path.extend(agents_by_dependencies[dep_count])
        
        alternatives.append(parallel_path)
        
        # Generate conservative sequential path
        sequential_path = [step["agent_name"] for step in execution_plan.agent_sequence]
        alternatives.append(sequential_path)
        
        return alternatives
    
    def _get_estimated_execution_time(self, agent_name: str) -> float:
        """Get estimated execution time based on historical performance."""
        
        if agent_name in self.agent_performance_history:
            history = self.agent_performance_history[agent_name]
            # Use average of recent performances
            return sum(history.values()) / len(history) if history else 5.0
        
        # Default estimates
        defaults = {
            "planner": 3.0,
            "researcher": 5.0,
            "writer": 8.0,
            "editor": 4.0,
            "seo": 2.0,
            "image": 3.0,
            "social_media": 2.0,
            "campaign_manager": 4.0,
            "content_repurposer": 3.0
        }
        
        return defaults.get(agent_name, 5.0)
    
    def _record_agent_performance(self, agent_name: str, execution_time: float):
        """Record agent performance for future estimates."""
        
        if agent_name not in self.agent_performance_history:
            self.agent_performance_history[agent_name] = {}
        
        # Keep last 10 performances
        history = self.agent_performance_history[agent_name]
        timestamp = datetime.utcnow().isoformat()
        history[timestamp] = execution_time
        
        # Keep only recent entries
        if len(history) > 10:
            oldest_key = min(history.keys())
            del history[oldest_key]
    
    def get_orchestration_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get advanced orchestration analytics for a workflow."""
        
        if workflow_id not in self.active_orchestrations:
            return {}
        
        orchestration = self.active_orchestrations[workflow_id]
        decisions = self.resequence_decisions.get(workflow_id, [])
        critical_path = self.critical_paths.get(workflow_id)
        
        return {
            "orchestration_summary": {
                "resequence_count": orchestration["resequence_count"],
                "recovery_attempts": orchestration["recovery_attempts"],
                "performance_adjustments": len(orchestration["performance_adjustments"])
            },
            "resequencing_decisions": [
                {
                    "timestamp": d.timestamp.isoformat(),
                    "trigger": d.trigger_event,
                    "reasoning": d.reasoning,
                    "confidence": d.confidence_score
                } for d in decisions
            ],
            "critical_path_analysis": {
                "critical_agents": critical_path.path_agents if critical_path else [],
                "estimated_total_time": critical_path.total_estimated_time if critical_path else 0,
                "bottleneck_agent": critical_path.bottleneck_agent if critical_path else None,
                "optimization_opportunities": critical_path.optimization_opportunities if critical_path else []
            },
            "alternative_paths": self.alternative_paths.get(workflow_id, [])
        }
    
    async def _skip_agent_and_continue(self, workflow_id: str, agent_name: str, execution_state: WorkflowExecution):
        """Skip failed agent and continue with remaining workflow."""
        logger.info(f"Skipping failed agent {agent_name} and continuing workflow {workflow_id}")
        # Implementation would mark agent as skipped and adjust dependencies
        
    async def _retry_with_alternative_agent(self, workflow_id: str, agent_name: str, execution_state: WorkflowExecution):
        """Try to replace failed agent with alternative implementation."""
        logger.info(f"Attempting to retry {agent_name} with alternative implementation in workflow {workflow_id}")
        # Implementation would look for alternative agent implementations
    
    async def _handle_workflow_stall(self, workflow_id: str, execution_state: Optional[WorkflowExecution]):
        """Handle workflow stall situations."""
        logger.warning(f"Workflow {workflow_id} appears to be stalled, applying recovery measures")
        # Implementation would detect and resolve stall conditions
    
    async def _handle_agent_start(self, workflow_id: str, agent_name: str):
        """Handle agent start events for tracking."""
        
        # Report agent availability to degradation manager
        if self.enable_degradation and self.degradation_manager:
            try:
                # Register the service if not already registered
                await self.degradation_manager.register_service(
                    service_name=agent_name,
                    critical=False,  # Most agents are not critical
                    health_check_interval=60
                )
            except Exception as e:
                logger.debug(f"Agent {agent_name} registration note: {e}")
    
    # Graceful Degradation Methods
    
    async def _apply_degradation_strategy(
        self, 
        execution_plan: ExecutionPlan, 
        degradation_level: 'DegradationLevel'
    ) -> ExecutionPlan:
        """Apply degradation strategy to modify execution plan based on system health."""
        if not self.enable_degradation or not self.degradation_manager:
            return execution_plan
        
        try:
            # Generate minimal viable workflow based on degradation level
            minimal_workflow = await self.degradation_manager.get_minimal_viable_workflow(
                content_type="orchestrated_content_generation",
                unavailable_services=[],  # Will be populated based on actual failures
                priority="high"
            )
            
            if minimal_workflow:
                logger.info(f"🔧 Applied degradation strategy: reduced from {len(execution_plan.agent_sequence)} to {len(minimal_workflow)} agents")
                
                # Create new execution plan with minimal workflow
                degraded_plan = ExecutionPlan(
                    id=f"degraded_{execution_plan.id}",
                    workflow_execution_id=execution_plan.workflow_execution_id,
                    strategy=execution_plan.strategy,
                    agent_sequence=minimal_workflow,
                    estimated_duration=execution_plan.estimated_duration * 0.7  # Reduced duration
                )
                return degraded_plan
            
        except Exception as e:
            logger.error(f"Failed to apply degradation strategy: {e}")
        
        return execution_plan
    
    async def _handle_agent_failure_with_degradation(
        self, 
        workflow_id: str, 
        agent_name: str, 
        error_message: str,
        execution_state: WorkflowExecution
    ):
        """Handle agent failures with graceful degradation support."""
        if not self.enable_degradation or not self.degradation_manager:
            return
        
        try:
            # Report service failure
            degradation_level = await self.degradation_manager.report_service_failure(
                service_name=agent_name,
                error_message=error_message
            )
            
            # Check if system degradation level changed significantly
            orchestration_state = self.active_orchestrations.get(workflow_id, {})
            current_degradation = orchestration_state.get("degradation_level", DegradationLevel.NONE)
            
            if degradation_level > current_degradation:
                logger.warning(f"🚨 System degradation increased to {degradation_level.value} after {agent_name} failure")
                
                # Update orchestration state
                orchestration_state["degradation_level"] = degradation_level
                
                # Get user notification about degradation
                notification = await self.degradation_manager.get_user_notification_message(
                    content_type="orchestrated_content_generation"
                )
                
                if notification:
                    logger.info(f"📢 User notification: {notification}")
                    # In a real system, you'd send this to the frontend
                
                # Check if workflow should continue in degraded mode
                # Since there's no can_continue_with_degradation method, check system status
                system_status = await self.degradation_manager.get_system_status()
                can_continue = system_status.get("can_provide_minimal_service", False)
                
                if not can_continue:
                    logger.error(f"❌ Workflow {workflow_id} cannot continue with current degradation level")
                    execution_state.status = ExecutionStatus.FAILED
                
        except Exception as e:
            logger.error(f"Failed to handle agent failure with degradation: {e}")
    
    async def get_system_health_status(self) -> Dict[str, Any]:
        """Get current system health and degradation status."""
        if not self.enable_degradation or not self.degradation_manager:
            return {
                "status": "unknown",
                "degradation_enabled": False,
                "message": "Graceful degradation not available"
            }
        
        try:
            system_status = await self.degradation_manager.get_system_status()
            degradation_level = system_status.get("degradation_level", DegradationLevel.NONE)
            service_health = system_status.get("service_health", {})
            
            return {
                "status": "healthy" if degradation_level == DegradationLevel.NONE else "degraded",
                "degradation_level": degradation_level.value,
                "degradation_enabled": True,
                "service_health": service_health,
                "active_orchestrations": len(self.active_orchestrations),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system health status: {e}")
            return {
                "status": "error",
                "degradation_enabled": True,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        logger.info(f"Agent {agent_name} starting in workflow {workflow_id}")
        # Track agent start for performance analytics