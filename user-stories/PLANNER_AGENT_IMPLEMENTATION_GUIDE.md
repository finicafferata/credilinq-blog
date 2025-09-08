# Master Planner Agent Implementation Guide

## Overview

This guide provides step-by-step implementation details for creating a Master Planner Agent that orchestrates the execution order of your 20+ agents with complete observability and control.

## Current Architecture Integration

Your existing system already provides excellent foundations:
- `CampaignOrchestratorAgent` handles campaign lifecycle
- `WorkflowStateManager` manages state persistence
- `AsyncPerformanceTracker` tracks execution metrics
- Rich database schema with LangGraph workflow tables

The Master Planner Agent will enhance these existing components rather than replace them.

## Implementation Steps

### Step 1: Database Schema Extensions

First, extend your existing schema to support execution planning:

```sql
-- Create execution plans table
CREATE TABLE execution_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    workflow_execution_id UUID UNIQUE,
    
    -- Execution configuration
    execution_strategy VARCHAR(50) NOT NULL DEFAULT 'sequential', -- sequential, parallel, adaptive
    total_agents INTEGER NOT NULL,
    estimated_duration_minutes INTEGER,
    
    -- Plan data
    planned_sequence JSONB NOT NULL, -- Ordered execution plan with dependencies
    agent_configurations JSONB, -- Agent-specific configurations
    
    -- State tracking
    current_step INTEGER DEFAULT 0,
    completed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    failed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Metadata
    created_by_agent VARCHAR(100) NOT NULL DEFAULT 'MasterPlannerAgent',
    planning_reasoning TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create agent dependencies table
CREATE TABLE agent_dependencies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_plan_id UUID REFERENCES execution_plans(id) ON DELETE CASCADE,
    
    -- Agent identification
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    
    -- Dependency configuration  
    depends_on_agents TEXT[] DEFAULT ARRAY[]::TEXT[], -- Prerequisite agents
    dependency_type VARCHAR(20) DEFAULT 'hard', -- hard, soft, optional
    execution_order INTEGER NOT NULL,
    
    -- Parallel execution
    parallel_group_id INTEGER, -- Agents in same group can run in parallel
    is_parallel_eligible BOOLEAN DEFAULT FALSE,
    
    -- Execution constraints
    max_retries INTEGER DEFAULT 3,
    timeout_minutes INTEGER DEFAULT 30,
    resource_requirements JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create real-time workflow state tracking
CREATE TABLE workflow_state_live (
    workflow_execution_id UUID PRIMARY KEY,
    
    -- Current execution state
    current_agents_running TEXT[] DEFAULT ARRAY[]::TEXT[],
    current_step INTEGER DEFAULT 0,
    
    -- Completion tracking
    completed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    failed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    waiting_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Performance metrics
    start_time TIMESTAMPTZ,
    estimated_completion_time TIMESTAMPTZ,
    
    -- State data
    execution_metadata JSONB,
    intermediate_outputs JSONB, -- Outputs from completed agents
    
    -- Heartbeat for monitoring
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_execution_plans_campaign ON execution_plans(campaign_id);
CREATE INDEX idx_execution_plans_workflow ON execution_plans(workflow_execution_id);
CREATE INDEX idx_agent_dependencies_plan ON agent_dependencies(execution_plan_id);
CREATE INDEX idx_agent_dependencies_order ON agent_dependencies(execution_plan_id, execution_order);
CREATE INDEX idx_workflow_state_live_heartbeat ON workflow_state_live(last_heartbeat);
```

### Step 2: Master Planner Agent Implementation

Create the `MasterPlannerAgent` class extending your existing `WorkflowAgent`:

```python
# src/agents/orchestration/master_planner_agent.py

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import logging

from ..core.base_agent import WorkflowAgent, AgentMetadata, AgentType, AgentResult
from ..core.agent_factory import AgentFactory
from .workflow_state_manager import WorkflowStateManager
from .campaign_database_service import CampaignDatabaseService

logger = logging.getLogger(__name__)

@dataclass
class ExecutionPlan:
    """Complete execution plan for a campaign workflow"""
    id: str
    campaign_id: str
    workflow_execution_id: str
    strategy: str  # sequential, parallel, adaptive
    agent_sequence: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    parallel_groups: Dict[int, List[str]]
    estimated_duration: int
    created_at: datetime

@dataclass 
class AgentExecutionStep:
    """Individual agent execution step in the plan"""
    agent_name: str
    agent_type: str
    execution_order: int
    dependencies: List[str]
    parallel_group_id: Optional[int]
    max_retries: int = 3
    timeout_minutes: int = 30
    configuration: Dict[str, Any] = field(default_factory=dict)

class MasterPlannerAgent(WorkflowAgent):
    """
    Master Planner Agent that orchestrates execution order of all agents
    in a campaign workflow with complete observability and control.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
                name="MasterPlannerAgent", 
                description="Master planner for campaign workflow orchestration",
                capabilities=[
                    "execution_planning", "dependency_analysis", "parallel_optimization",
                    "error_recovery_planning", "resource_optimization"
                ],
                dependencies=["database_service", "agent_factory", "state_manager"]
            )
        
        super().__init__(metadata)
        
        # Core services
        self.db_service = CampaignDatabaseService()
        self.state_manager = WorkflowStateManager()
        self.agent_factory = AgentFactory()
        
        # Planning state
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.execution_history: Dict[str, List[str]] = {}  # Plan ID -> completed agents
        
        # Agent knowledge base - mapping agent types to capabilities and dependencies
        self.agent_knowledge_base = self._build_agent_knowledge_base()
        
        logger.info(f"Initialized {self.metadata.name} with enhanced orchestration capabilities")
    
    def _build_agent_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """
        Build knowledge base of all available agents with their capabilities and dependencies.
        This can be enhanced to read from configuration or discover dynamically.
        """
        return {
            # Core content pipeline agents
            "planner": {
                "type": AgentType.PLANNER,
                "capabilities": ["strategy_planning", "content_planning"],
                "dependencies": [],
                "outputs": ["content_strategy", "key_themes", "content_calendar"],
                "execution_time_estimate": 120,  # seconds
                "parallel_safe": False
            },
            "researcher": {
                "type": AgentType.RESEARCHER,
                "capabilities": ["market_research", "competitive_analysis"],
                "dependencies": ["planner"],
                "outputs": ["market_trends", "competitor_analysis", "industry_insights"],
                "execution_time_estimate": 180,
                "parallel_safe": True
            },
            "writer": {
                "type": AgentType.WRITER,
                "capabilities": ["content_creation", "copywriting"],
                "dependencies": ["planner", "researcher"], 
                "outputs": ["blog_content", "content_pieces"],
                "execution_time_estimate": 300,
                "parallel_safe": True
            },
            "editor": {
                "type": AgentType.EDITOR,
                "capabilities": ["content_editing", "quality_assurance"],
                "dependencies": ["writer"],
                "outputs": ["edited_content", "quality_improvements"],
                "execution_time_estimate": 180,
                "parallel_safe": False
            },
            "seo": {
                "type": AgentType.SEO,
                "capabilities": ["seo_optimization", "keyword_optimization"],
                "dependencies": ["writer", "editor"],
                "outputs": ["optimized_content", "seo_metadata"],
                "execution_time_estimate": 120,
                "parallel_safe": True
            },
            "image": {
                "type": AgentType.IMAGE,
                "capabilities": ["image_generation", "visual_content"],
                "dependencies": ["planner"],
                "outputs": ["generated_images", "image_prompts"],
                "execution_time_estimate": 240,
                "parallel_safe": True
            },
            "social_media": {
                "type": AgentType.SOCIAL_MEDIA,
                "capabilities": ["social_adaptation", "platform_optimization"],
                "dependencies": ["editor", "seo"],
                "outputs": ["social_posts", "platform_content"],
                "execution_time_estimate": 90,
                "parallel_safe": True
            },
            # Add other agents as needed...
        }
    
    async def create_execution_plan(
        self, 
        campaign_id: str,
        workflow_execution_id: str,
        strategy: str = "sequential"
    ) -> ExecutionPlan:
        """
        Create comprehensive execution plan for campaign workflow.
        
        Args:
            campaign_id: Campaign to create plan for
            workflow_execution_id: Unique workflow execution identifier
            strategy: Execution strategy (sequential, parallel, adaptive)
            
        Returns:
            ExecutionPlan: Complete execution plan with dependencies and ordering
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Creating execution plan for campaign {campaign_id} with strategy '{strategy}'")
            
            # 1. Load campaign requirements
            campaign = await self.db_service.get_campaign_with_tasks(campaign_id)
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            # 2. Analyze campaign requirements and determine needed agents
            required_agents = await self._analyze_campaign_requirements(campaign)
            
            # 3. Build dependency graph
            dependency_graph = self._build_dependency_graph(required_agents)
            
            # 4. Optimize execution order based on strategy
            if strategy == "parallel":
                execution_sequence, parallel_groups = self._optimize_for_parallel_execution(dependency_graph)
            elif strategy == "adaptive":
                execution_sequence, parallel_groups = self._optimize_adaptive_execution(dependency_graph)
            else:  # sequential
                execution_sequence, parallel_groups = self._optimize_sequential_execution(dependency_graph)
            
            # 5. Estimate total execution time
            estimated_duration = self._estimate_execution_duration(execution_sequence, parallel_groups)
            
            # 6. Create execution plan
            plan = ExecutionPlan(
                id=str(uuid.uuid4()),
                campaign_id=campaign_id,
                workflow_execution_id=workflow_execution_id,
                strategy=strategy,
                agent_sequence=execution_sequence,
                dependencies=dependency_graph,
                parallel_groups=parallel_groups,
                estimated_duration=estimated_duration,
                created_at=start_time
            )
            
            # 7. Persist execution plan to database
            await self._save_execution_plan(plan)
            
            # 8. Initialize live workflow state
            await self._initialize_live_state(plan)
            
            self.active_plans[workflow_execution_id] = plan
            
            logger.info(f"Created execution plan {plan.id} with {len(execution_sequence)} agents, "
                       f"estimated duration: {estimated_duration} minutes")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan for campaign {campaign_id}: {str(e)}")
            raise
    
    async def _analyze_campaign_requirements(self, campaign) -> List[str]:
        """
        Analyze campaign requirements and determine which agents are needed.
        """
        required_agents = ["planner"]  # Always start with planner
        
        # Analyze campaign metadata and tasks to determine needed agents
        if campaign.metadata:
            campaign_type = campaign.metadata.get("campaign_type", "blog_creation")
            content_types = campaign.metadata.get("content_types", ["blog_post"])
            
            if campaign_type == "blog_creation":
                required_agents.extend(["researcher", "writer", "editor", "seo"])
                
                if "image" in content_types or campaign.metadata.get("include_images", False):
                    required_agents.append("image")
                    
                if "social_media" in content_types:
                    required_agents.append("social_media")
            
            elif campaign_type == "content_repurposing":
                required_agents.extend(["writer", "editor", "social_media"])
            
            # Add other campaign type logic...
        
        # Analyze existing tasks for additional requirements
        for task in campaign.tasks:
            if task.task_type.value == "image_generation":
                if "image" not in required_agents:
                    required_agents.append("image")
            elif task.task_type.value == "social_media_adaptation":
                if "social_media" not in required_agents:
                    required_agents.append("social_media")
        
        logger.info(f"Campaign {campaign.id} requires agents: {required_agents}")
        return required_agents
    
    def _build_dependency_graph(self, required_agents: List[str]) -> Dict[str, List[str]]:
        """
        Build dependency graph for required agents based on knowledge base.
        """
        dependency_graph = {}
        
        for agent_name in required_agents:
            if agent_name in self.agent_knowledge_base:
                agent_info = self.agent_knowledge_base[agent_name]
                # Filter dependencies to only include agents that are also required
                dependencies = [dep for dep in agent_info["dependencies"] if dep in required_agents]
                dependency_graph[agent_name] = dependencies
            else:
                logger.warning(f"Agent {agent_name} not found in knowledge base, assuming no dependencies")
                dependency_graph[agent_name] = []
        
        # Validate no circular dependencies
        self._validate_dependency_graph(dependency_graph)
        
        return dependency_graph
    
    def _validate_dependency_graph(self, dependency_graph: Dict[str, List[str]]):
        """
        Validate dependency graph for circular dependencies using topological sort.
        """
        # Implementation of cycle detection using DFS
        white = set(dependency_graph.keys())  # Unvisited
        gray = set()  # Currently visiting
        black = set()  # Completed
        
        def visit(node):
            if node in gray:
                raise ValueError(f"Circular dependency detected involving agent: {node}")
            if node in white:
                white.remove(node)
                gray.add(node)
                for dependency in dependency_graph.get(node, []):
                    visit(dependency)
                gray.remove(node)
                black.add(node)
        
        for node in list(white):
            visit(node)
    
    def _optimize_sequential_execution(
        self, 
        dependency_graph: Dict[str, List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
        """
        Create sequential execution order using topological sort.
        """
        # Topological sort to determine execution order
        in_degree = {node: 0 for node in dependency_graph}
        for node in dependency_graph:
            for dependency in dependency_graph[node]:
                in_degree[dependency] += 1
        
        queue = [node for node in in_degree if in_degree[node] == 0]
        execution_order = []
        
        while queue:
            node = queue.pop(0)
            execution_order.append(node)
            
            for dependent in dependency_graph:
                if node in dependency_graph[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # Convert to execution sequence format
        execution_sequence = []
        for i, agent_name in enumerate(execution_order):
            agent_info = self.agent_knowledge_base.get(agent_name, {})
            execution_sequence.append({
                "agent_name": agent_name,
                "agent_type": agent_info.get("type", AgentType.GENERIC).value,
                "execution_order": i + 1,
                "dependencies": dependency_graph[agent_name],
                "parallel_group_id": None,
                "configuration": {}
            })
        
        return execution_sequence, {}  # No parallel groups in sequential
    
    def _optimize_for_parallel_execution(
        self, 
        dependency_graph: Dict[str, List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
        """
        Optimize execution plan for maximum parallelization.
        """
        # Group agents by their dependency level (topological levels)
        levels = {}
        processed = set()
        current_level = 0
        
        # Find agents with no dependencies (level 0)
        level_agents = [agent for agent, deps in dependency_graph.items() if not deps]
        
        while level_agents:
            levels[current_level] = level_agents
            processed.update(level_agents)
            
            # Find next level agents (all dependencies are processed)
            next_level_agents = []
            for agent, deps in dependency_graph.items():
                if agent not in processed and all(dep in processed for dep in deps):
                    next_level_agents.append(agent)
            
            level_agents = next_level_agents
            current_level += 1
        
        # Convert levels to execution sequence and parallel groups
        execution_sequence = []
        parallel_groups = {}
        order_counter = 1
        
        for level, agents in levels.items():
            if len(agents) > 1:
                # Multiple agents can run in parallel
                parallel_groups[level] = agents
                for agent_name in agents:
                    agent_info = self.agent_knowledge_base.get(agent_name, {})
                    execution_sequence.append({
                        "agent_name": agent_name,
                        "agent_type": agent_info.get("type", AgentType.GENERIC).value,
                        "execution_order": order_counter,
                        "dependencies": dependency_graph[agent_name],
                        "parallel_group_id": level,
                        "configuration": {}
                    })
                order_counter += 1
            else:
                # Single agent
                agent_name = agents[0]
                agent_info = self.agent_knowledge_base.get(agent_name, {})
                execution_sequence.append({
                    "agent_name": agent_name,
                    "agent_type": agent_info.get("type", AgentType.GENERIC).value,
                    "execution_order": order_counter,
                    "dependencies": dependency_graph[agent_name],
                    "parallel_group_id": None,
                    "configuration": {}
                })
                order_counter += 1
        
        return execution_sequence, parallel_groups
    
    def _optimize_adaptive_execution(
        self, 
        dependency_graph: Dict[str, List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
        """
        Create adaptive execution plan that can adjust based on runtime conditions.
        """
        # Start with parallel optimization as base
        execution_sequence, parallel_groups = self._optimize_for_parallel_execution(dependency_graph)
        
        # Add adaptive markers for agents that can be dynamically rescheduled
        for step in execution_sequence:
            agent_name = step["agent_name"]
            agent_info = self.agent_knowledge_base.get(agent_name, {})
            
            # Mark agents as adaptable if they're parallel-safe and have alternatives
            step["adaptive_scheduling"] = agent_info.get("parallel_safe", False)
            step["can_reschedule"] = len(step["dependencies"]) <= 1
        
        return execution_sequence, parallel_groups
    
    def _estimate_execution_duration(
        self, 
        execution_sequence: List[Dict[str, Any]], 
        parallel_groups: Dict[int, List[str]]
    ) -> int:
        """
        Estimate total execution duration in minutes based on agent execution times.
        """
        if not parallel_groups:
            # Sequential execution - sum all execution times
            total_seconds = sum(
                self.agent_knowledge_base.get(step["agent_name"], {}).get("execution_time_estimate", 180)
                for step in execution_sequence
            )
        else:
            # Parallel execution - sum max time per parallel group
            total_seconds = 0
            processed_groups = set()
            
            for step in execution_sequence:
                group_id = step["parallel_group_id"]
                if group_id is not None and group_id not in processed_groups:
                    # Get max execution time for this parallel group
                    group_agents = parallel_groups[group_id]
                    max_time = max(
                        self.agent_knowledge_base.get(agent, {}).get("execution_time_estimate", 180)
                        for agent in group_agents
                    )
                    total_seconds += max_time
                    processed_groups.add(group_id)
                elif group_id is None:
                    # Single agent execution
                    agent_time = self.agent_knowledge_base.get(step["agent_name"], {}).get("execution_time_estimate", 180)
                    total_seconds += agent_time
        
        # Convert to minutes and add buffer
        estimated_minutes = int(total_seconds / 60) + 5  # 5 minute buffer
        return estimated_minutes
    
    async def _save_execution_plan(self, plan: ExecutionPlan):
        """
        Save execution plan to database for persistence and recovery.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Insert execution plan
                cur.execute("""
                    INSERT INTO execution_plans (
                        id, campaign_id, workflow_execution_id, execution_strategy,
                        total_agents, estimated_duration_minutes, planned_sequence,
                        agent_configurations, created_by_agent, planning_reasoning
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    plan.id,
                    plan.campaign_id,
                    plan.workflow_execution_id,
                    plan.strategy,
                    len(plan.agent_sequence),
                    plan.estimated_duration,
                    json.dumps(plan.agent_sequence),
                    json.dumps({}),  # agent_configurations - can be enhanced
                    self.metadata.name,
                    f"Execution plan created using {plan.strategy} strategy"
                ))
                
                # Insert agent dependencies
                for step in plan.agent_sequence:
                    cur.execute("""
                        INSERT INTO agent_dependencies (
                            execution_plan_id, agent_name, agent_type, depends_on_agents,
                            execution_order, parallel_group_id, is_parallel_eligible
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        plan.id,
                        step["agent_name"],
                        step["agent_type"],
                        step["dependencies"],
                        step["execution_order"],
                        step["parallel_group_id"],
                        step["parallel_group_id"] is not None
                    ))
                
                conn.commit()
                logger.info(f"Saved execution plan {plan.id} to database")
                
        except Exception as e:
            logger.error(f"Failed to save execution plan {plan.id}: {str(e)}")
            raise
    
    async def _initialize_live_state(self, plan: ExecutionPlan):
        """
        Initialize live workflow state tracking for real-time monitoring.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Determine initial waiting agents (those with no dependencies)
                waiting_agents = [
                    step["agent_name"] for step in plan.agent_sequence 
                    if not step["dependencies"]
                ]
                
                cur.execute("""
                    INSERT INTO workflow_state_live (
                        workflow_execution_id, current_step, waiting_agents,
                        start_time, estimated_completion_time, execution_metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    plan.workflow_execution_id,
                    0,
                    waiting_agents,
                    plan.created_at,
                    plan.created_at + timedelta(minutes=plan.estimated_duration),
                    json.dumps({
                        "execution_plan_id": plan.id,
                        "strategy": plan.strategy,
                        "total_agents": len(plan.agent_sequence)
                    })
                ))
                
                conn.commit()
                logger.info(f"Initialized live state for workflow {plan.workflow_execution_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize live state: {str(e)}")
            raise
```

### Step 3: Integration with Existing Components

Modify your existing `CampaignOrchestratorAgent` to use the new Master Planner:

```python
# src/agents/orchestration/campaign_orchestrator.py (modify existing)

class CampaignOrchestratorAgent(WorkflowAgent):
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        # ... existing init code ...
        
        # Add Master Planner Agent
        self.master_planner = MasterPlannerAgent()
    
    async def orchestrate_campaign(self, campaign_id: str) -> AgentResult:
        """
        Enhanced orchestration method using Master Planner Agent.
        """
        start_time = datetime.utcnow()
        workflow_execution_id = str(uuid.uuid4())
        
        try:
            # 1. Create execution plan using Master Planner
            execution_plan = await self.master_planner.create_execution_plan(
                campaign_id=campaign_id,
                workflow_execution_id=workflow_execution_id,
                strategy="adaptive"  # or from campaign configuration
            )
            
            # 2. Execute workflow according to plan
            final_status = await self._execute_planned_workflow(execution_plan)
            
            # ... rest of existing orchestration logic ...
            
        except Exception as e:
            logger.error(f"Campaign orchestration failed for {campaign_id}: {str(e)}")
            # ... existing error handling ...
    
    async def _execute_planned_workflow(self, plan: ExecutionPlan) -> WorkflowStatus:
        """
        Execute workflow according to the master plan with real-time monitoring.
        """
        logger.info(f"Executing planned workflow {plan.workflow_execution_id} with {len(plan.agent_sequence)} agents")
        
        # Track execution state
        completed_agents = set()
        failed_agents = set()
        
        # Execute agents in planned order
        for step in plan.agent_sequence:
            agent_name = step["agent_name"]
            dependencies = step["dependencies"]
            
            # Check if dependencies are satisfied
            if not all(dep in completed_agents for dep in dependencies):
                logger.warning(f"Dependencies not satisfied for agent {agent_name}")
                continue
            
            try:
                # Execute agent
                result = await self._execute_planned_agent(step, plan)
                
                if result.success:
                    completed_agents.add(agent_name)
                    await self._update_live_state(plan.workflow_execution_id, agent_name, "completed")
                else:
                    failed_agents.add(agent_name)
                    await self._update_live_state(plan.workflow_execution_id, agent_name, "failed")
                    
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {str(e)}")
                failed_agents.add(agent_name)
                await self._update_live_state(plan.workflow_execution_id, agent_name, "failed")
        
        # Determine final status
        if len(failed_agents) == 0:
            return WorkflowStatus.COMPLETED
        elif len(completed_agents) > len(failed_agents):
            return WorkflowStatus.PARTIAL_SUCCESS  # You might add this status
        else:
            return WorkflowStatus.FAILED
```

### Step 4: Real-Time State Monitoring

Add methods for real-time state updates:

```python
# Add to MasterPlannerAgent class

async def update_agent_completion(
    self, 
    workflow_execution_id: str, 
    agent_name: str, 
    status: str,
    result: Optional[AgentResult] = None
):
    """
    Update workflow state when an agent completes execution.
    """
    try:
        with self.db_service.get_db_connection() as conn:
            cur = conn.cursor()
            
            if status == "completed":
                # Move agent from running to completed
                cur.execute("""
                    UPDATE workflow_state_live 
                    SET completed_agents = array_append(completed_agents, %s),
                        current_agents_running = array_remove(current_agents_running, %s),
                        last_heartbeat = %s,
                        intermediate_outputs = jsonb_set(
                            COALESCE(intermediate_outputs, '{}'::jsonb),
                            ARRAY[%s],
                            %s::jsonb
                        )
                    WHERE workflow_execution_id = %s
                """, (
                    agent_name, agent_name, datetime.utcnow(),
                    agent_name, json.dumps(result.data if result else {}),
                    workflow_execution_id
                ))
                
                # Determine next agents that can now run
                await self._update_waiting_agents(workflow_execution_id)
                
            elif status == "failed":
                # Move agent to failed list
                cur.execute("""
                    UPDATE workflow_state_live 
                    SET failed_agents = array_append(failed_agents, %s),
                        current_agents_running = array_remove(current_agents_running, %s),
                        last_heartbeat = %s
                    WHERE workflow_execution_id = %s
                """, (agent_name, agent_name, datetime.utcnow(), workflow_execution_id))
                
            conn.commit()
            
    except Exception as e:
        logger.error(f"Failed to update agent completion status: {str(e)}")

async def get_workflow_progress(self, workflow_execution_id: str) -> Dict[str, Any]:
    """
    Get current workflow progress for dashboard/monitoring.
    """
    try:
        with self.db_service.get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    wsl.current_step,
                    wsl.current_agents_running,
                    wsl.completed_agents,
                    wsl.failed_agents,
                    wsl.waiting_agents,
                    wsl.start_time,
                    wsl.estimated_completion_time,
                    wsl.execution_metadata,
                    ep.total_agents,
                    ep.execution_strategy
                FROM workflow_state_live wsl
                JOIN execution_plans ep ON wsl.workflow_execution_id = ep.workflow_execution_id
                WHERE wsl.workflow_execution_id = %s
            """, (workflow_execution_id,))
            
            result = cur.fetchone()
            if not result:
                return {"status": "not_found"}
            
            total_agents = result[8]
            completed_count = len(result[2]) if result[2] else 0
            failed_count = len(result[3]) if result[3] else 0
            running_count = len(result[1]) if result[1] else 0
            
            progress_percentage = int((completed_count / total_agents) * 100) if total_agents > 0 else 0
            
            return {
                "workflow_execution_id": workflow_execution_id,
                "current_step": result[0],
                "progress_percentage": progress_percentage,
                "agents_status": {
                    "running": result[1] or [],
                    "completed": result[2] or [],
                    "failed": result[3] or [],
                    "waiting": result[4] or []
                },
                "execution_info": {
                    "start_time": result[5].isoformat() if result[5] else None,
                    "estimated_completion": result[6].isoformat() if result[6] else None,
                    "strategy": result[9],
                    "total_agents": total_agents
                },
                "counts": {
                    "total": total_agents,
                    "completed": completed_count,
                    "failed": failed_count,
                    "running": running_count,
                    "waiting": len(result[4]) if result[4] else 0
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get workflow progress: {str(e)}")
        return {"status": "error", "message": str(e)}
```

### Step 5: API Endpoints for Observability

Create new API endpoints in `src/api/routes/workflows.py`:

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from ..dependencies import get_database_service

router = APIRouter(prefix="/api/v2/workflows", tags=["workflows"])

@router.get("/{workflow_execution_id}/progress")
async def get_workflow_progress(
    workflow_execution_id: str,
    db_service=Depends(get_database_service)
):
    """Get real-time workflow execution progress"""
    # Implementation using MasterPlannerAgent.get_workflow_progress()
    pass

@router.get("/{workflow_execution_id}/plan")
async def get_execution_plan(workflow_execution_id: str):
    """Get the execution plan for a workflow"""
    pass

@router.post("/{workflow_execution_id}/control")
async def control_workflow(
    workflow_execution_id: str,
    action: str,  # pause, resume, cancel
    reason: str = None
):
    """Control workflow execution (pause/resume/cancel)"""
    pass

@router.websocket("/{workflow_execution_id}/live")
async def workflow_live_updates(websocket: WebSocket, workflow_execution_id: str):
    """WebSocket endpoint for real-time workflow updates"""
    pass
```

## Testing the Implementation

### Unit Tests

Create comprehensive unit tests:

```python
# tests/agents/orchestration/test_master_planner_agent.py

import pytest
from src.agents.orchestration.master_planner_agent import MasterPlannerAgent

class TestMasterPlannerAgent:
    
    @pytest.fixture
    def planner_agent(self):
        return MasterPlannerAgent()
    
    async def test_create_execution_plan_sequential(self, planner_agent):
        # Test sequential execution plan creation
        pass
    
    async def test_create_execution_plan_parallel(self, planner_agent):
        # Test parallel execution plan creation
        pass
    
    async def test_dependency_validation(self, planner_agent):
        # Test circular dependency detection
        pass
    
    async def test_workflow_progress_tracking(self, planner_agent):
        # Test real-time progress tracking
        pass
```

### Integration Tests

Create integration tests with your existing system:

```python
# tests/integration/test_master_planner_integration.py

async def test_full_workflow_orchestration():
    """Test complete workflow from planning to execution"""
    # Create test campaign
    # Generate execution plan
    # Execute workflow
    # Verify all agents executed in correct order
    pass
```

## Deployment Checklist

- [ ] Database migration scripts created and tested
- [ ] Existing workflow backward compatibility verified
- [ ] Performance impact assessment completed
- [ ] Monitoring and alerting configured
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Integration tests passing
- [ ] Gradual rollout plan defined

This implementation provides you with a robust Master Planner Agent that enhances your existing architecture while maintaining all current functionality. The planner gives you complete control over agent execution order, comprehensive observability, and intelligent workflow management.