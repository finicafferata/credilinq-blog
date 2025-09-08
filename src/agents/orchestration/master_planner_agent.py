"""
Master Planner Agent - Core orchestration engine for campaign-centric workflows.

This agent creates execution plans for your 20+ agents with proper dependency management,
parallel execution optimization, and real-time state tracking.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import logging

# Import your existing components
from ..core.base_agent import WorkflowAgent, AgentMetadata, AgentType, AgentResult, AgentExecutionContext
from ..core.agent_factory import AgentFactory

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
                agent_type=AgentType.WORKFLOW_ORCHESTRATOR,  # Now using the proper WORKFLOW_ORCHESTRATOR
                name="MasterPlannerAgent", 
                description="Master planner for campaign workflow orchestration",
                capabilities=[
                    "execution_planning", "dependency_analysis", "parallel_optimization",
                    "error_recovery_planning", "resource_optimization"
                ],
                dependencies=["database_service", "agent_factory"]
            )
        
        super().__init__(metadata)
        
        # Core services - using your existing infrastructure
        from ..core.database_service import DatabaseService
        self.db_service = DatabaseService()
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
        Based on your existing agent architecture.
        """
        return {
            # Core content pipeline agents
            "planner": {
                "type": "PLANNER",
                "capabilities": ["strategy_planning", "content_planning"],
                "dependencies": [],
                "outputs": ["content_strategy", "key_themes", "content_calendar"],
                "execution_time_estimate": 120,  # seconds
                "parallel_safe": False,
                "priority": 1  # Always first
            },
            "researcher": {
                "type": "RESEARCHER", 
                "capabilities": ["market_research", "competitive_analysis"],
                "dependencies": ["planner"],
                "outputs": ["market_trends", "competitor_analysis", "industry_insights"],
                "execution_time_estimate": 180,
                "parallel_safe": True,
                "priority": 2
            },
            "writer": {
                "type": "WRITER",
                "capabilities": ["content_creation", "copywriting"],
                "dependencies": ["planner", "researcher"], 
                "outputs": ["blog_content", "content_pieces"],
                "execution_time_estimate": 300,
                "parallel_safe": True,
                "priority": 3
            },
            "editor": {
                "type": "EDITOR",
                "capabilities": ["content_editing", "quality_assurance"],
                "dependencies": ["writer"],
                "outputs": ["edited_content", "quality_improvements"],
                "execution_time_estimate": 180,
                "parallel_safe": False,
                "priority": 4
            },
            "seo": {
                "type": "SEO",
                "capabilities": ["seo_optimization", "keyword_optimization"],
                "dependencies": ["writer"],  # Can run parallel with editor
                "outputs": ["optimized_content", "seo_metadata"],
                "execution_time_estimate": 120,
                "parallel_safe": True,
                "priority": 4
            },
            "image": {
                "type": "IMAGE",
                "capabilities": ["image_generation", "visual_content"],
                "dependencies": ["planner"],  # Can start early
                "outputs": ["generated_images", "image_prompts"],
                "execution_time_estimate": 240,
                "parallel_safe": True,
                "priority": 2
            },
            "social_media": {
                "type": "SOCIAL_MEDIA",
                "capabilities": ["social_adaptation", "platform_optimization"],
                "dependencies": ["editor"],  # Needs final content
                "outputs": ["social_posts", "platform_content"],
                "execution_time_estimate": 90,
                "parallel_safe": True,
                "priority": 5
            },
            # Add more agents based on your system
            "campaign_manager": {
                "type": "CAMPAIGN_MANAGER",
                "capabilities": ["campaign_coordination", "task_management"],
                "dependencies": ["planner"],
                "outputs": ["campaign_tasks", "timeline"],
                "execution_time_estimate": 60,
                "parallel_safe": True,
                "priority": 2
            },
            "content_repurposer": {
                "type": "CONTENT_REPURPOSER",
                "capabilities": ["content_adaptation", "format_conversion"],
                "dependencies": ["editor"],
                "outputs": ["repurposed_content", "format_variants"],
                "execution_time_estimate": 150,
                "parallel_safe": True,
                "priority": 5
            },
            "distribution": {
                "type": "DISTRIBUTION",
                "capabilities": ["content_distribution", "channel_management"],
                "dependencies": ["social_media", "seo"],
                "outputs": ["distribution_plan", "channel_assignments"],
                "execution_time_estimate": 60,
                "parallel_safe": False,
                "priority": 6
            }
        }
    
    async def create_execution_plan(
        self, 
        campaign_id: str,
        workflow_execution_id: str,
        strategy: str = "adaptive",
        required_agents: Optional[List[str]] = None
    ) -> ExecutionPlan:
        """
        Create comprehensive execution plan for campaign workflow.
        
        Args:
            campaign_id: Campaign to create plan for
            workflow_execution_id: Unique workflow execution identifier
            strategy: Execution strategy (sequential, parallel, adaptive)
            required_agents: Specific agents needed (auto-detected if None)
            
        Returns:
            ExecutionPlan: Complete execution plan with dependencies and ordering
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Creating execution plan for campaign {campaign_id} with strategy '{strategy}'")
            
            # 1. Determine required agents (if not specified)
            if required_agents is None:
                required_agents = await self._analyze_campaign_requirements(campaign_id)
            
            # 2. Build dependency graph
            dependency_graph = self._build_dependency_graph(required_agents)
            
            # 3. Optimize execution order based on strategy
            if strategy == "parallel":
                execution_sequence, parallel_groups = self._optimize_for_parallel_execution(dependency_graph)
            elif strategy == "sequential":
                execution_sequence, parallel_groups = self._optimize_sequential_execution(dependency_graph)
            else:  # adaptive (default)
                execution_sequence, parallel_groups = self._optimize_adaptive_execution(dependency_graph)
            
            # 4. Estimate total execution time
            estimated_duration = self._estimate_execution_duration(execution_sequence, parallel_groups)
            
            # 5. Create execution plan
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
            
            # 6. Save execution plan to database
            await self._save_execution_plan(plan)
            
            # 7. Initialize live workflow state
            await self._initialize_live_state(plan)
            
            self.active_plans[workflow_execution_id] = plan
            
            logger.info(f"Created execution plan {plan.id} with {len(execution_sequence)} agents, "
                       f"estimated duration: {estimated_duration} minutes")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan for campaign {campaign_id}: {str(e)}")
            raise
    
    async def _analyze_campaign_requirements(self, campaign_id: str) -> List[str]:
        """
        Analyze campaign requirements and determine which agents are needed.
        This integrates with your existing campaign system.
        """
        try:
            # Get campaign data from your existing database
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Try to get campaign info (this will work when campaigns table exists)
                cur.execute("SELECT metadata FROM campaigns WHERE id = %s", (campaign_id,))
                result = cur.fetchone()
                
                if result and result[0]:
                    metadata = result[0] if isinstance(result[0], dict) else {}
                else:
                    # Default agents for basic campaign
                    logger.info(f"No campaign metadata found for {campaign_id}, using default agents")
                    return ["planner", "researcher", "writer", "editor", "seo"]
                
        except Exception as e:
            logger.warning(f"Could not analyze campaign requirements: {str(e)}")
            # Return default set of agents
            return ["planner", "researcher", "writer", "editor", "seo"]
        
        # Analyze metadata to determine required agents
        required_agents = ["planner"]  # Always start with planner
        
        campaign_type = metadata.get("campaign_type", "blog_creation")
        content_types = metadata.get("content_types", ["blog_post"])
        
        if campaign_type == "blog_creation":
            required_agents.extend(["researcher", "writer", "editor", "seo"])
            
            if "image" in content_types or metadata.get("include_images", False):
                required_agents.append("image")
                
            if "social_media" in content_types:
                required_agents.extend(["social_media", "content_repurposer"])
        
        elif campaign_type == "content_repurposing":
            required_agents.extend(["writer", "editor", "content_repurposer", "social_media"])
        
        elif campaign_type == "full_campaign":
            required_agents.extend([
                "researcher", "writer", "editor", "seo", "image", 
                "social_media", "content_repurposer", "distribution"
            ])
        
        # Add distribution if multiple channels
        if metadata.get("distribution_channels") and len(metadata["distribution_channels"]) > 1:
            if "distribution" not in required_agents:
                required_agents.append("distribution")
        
        logger.info(f"Campaign {campaign_id} requires agents: {required_agents}")
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
        white = set(dependency_graph.keys())  # Unvisited
        gray = set()  # Currently visiting
        
        def visit(node):
            if node in gray:
                raise ValueError(f"Circular dependency detected involving agent: {node}")
            if node in white:
                white.remove(node)
                gray.add(node)
                for dependency in dependency_graph.get(node, []):
                    visit(dependency)
                gray.remove(node)
        
        for node in list(white):
            visit(node)
    
    def _optimize_adaptive_execution(
        self, 
        dependency_graph: Dict[str, List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
        """
        Create adaptive execution plan that balances parallelism with dependencies.
        This is the recommended strategy for most workflows.
        """
        # Start with parallel optimization as base
        execution_sequence, parallel_groups = self._optimize_for_parallel_execution(dependency_graph)
        
        # Add adaptive markers and optimizations
        for step in execution_sequence:
            agent_name = step["agent_name"]
            agent_info = self.agent_knowledge_base.get(agent_name, {})
            
            # Mark agents as adaptable based on their characteristics
            step["adaptive_scheduling"] = agent_info.get("parallel_safe", False)
            step["can_reschedule"] = len(step["dependencies"]) <= 1
            step["priority"] = agent_info.get("priority", 5)
            
            # Add resource requirements for intelligent scheduling
            step["estimated_duration"] = agent_info.get("execution_time_estimate", 180)
            step["resource_requirements"] = {
                "cpu_intensive": agent_name in ["writer", "image"],
                "memory_intensive": agent_name in ["researcher", "content_repurposer"],
                "api_dependent": agent_name in ["seo", "social_media", "distribution"]
            }
        
        # Re-sort by priority and dependencies for optimal execution
        execution_sequence.sort(key=lambda x: (x["execution_order"], x.get("priority", 5)))
        
        return execution_sequence, parallel_groups
    
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
                        "agent_type": agent_info.get("type", "GENERIC"),
                        "execution_order": order_counter,
                        "dependencies": dependency_graph[agent_name],
                        "parallel_group_id": level,
                        "configuration": {}
                    })
                    order_counter += 1  # Each agent gets unique order even in parallel
            else:
                # Single agent
                agent_name = agents[0]
                agent_info = self.agent_knowledge_base.get(agent_name, {})
                execution_sequence.append({
                    "agent_name": agent_name,
                    "agent_type": agent_info.get("type", "GENERIC"),
                    "execution_order": order_counter,
                    "dependencies": dependency_graph[agent_name],
                    "parallel_group_id": None,
                    "configuration": {}
                })
                order_counter += 1
        
        return execution_sequence, parallel_groups
    
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
            # Sort by priority if available
            queue.sort(key=lambda x: self.agent_knowledge_base.get(x, {}).get("priority", 5))
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
                "agent_type": agent_info.get("type", "GENERIC"),
                "execution_order": i + 1,
                "dependencies": dependency_graph[agent_name],
                "parallel_group_id": None,
                "configuration": {}
            })
        
        return execution_sequence, {}  # No parallel groups in sequential
    
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
        Save execution plan to database using your new tables.
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
                    f"Execution plan created using {plan.strategy} strategy with {len(plan.agent_sequence)} agents"
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
                    ON CONFLICT (workflow_execution_id) DO UPDATE SET
                        current_step = EXCLUDED.current_step,
                        waiting_agents = EXCLUDED.waiting_agents,
                        start_time = EXCLUDED.start_time,
                        estimated_completion_time = EXCLUDED.estimated_completion_time,
                        execution_metadata = EXCLUDED.execution_metadata,
                        updated_at = NOW()
                """, (
                    plan.workflow_execution_id,
                    0,
                    waiting_agents,
                    plan.created_at,
                    plan.created_at + timedelta(minutes=plan.estimated_duration),
                    json.dumps({
                        "execution_plan_id": plan.id,
                        "strategy": plan.strategy,
                        "total_agents": len(plan.agent_sequence),
                        "parallel_groups": list(plan.parallel_groups.keys())
                    })
                ))
                
                conn.commit()
                logger.info(f"Initialized live state for workflow {plan.workflow_execution_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize live state: {str(e)}")
            raise
    
    async def get_execution_plan(self, workflow_execution_id: str) -> Optional[ExecutionPlan]:
        """
        Load execution plan from database.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT 
                        id, campaign_id, workflow_execution_id, execution_strategy,
                        planned_sequence, estimated_duration_minutes, created_at
                    FROM execution_plans 
                    WHERE workflow_execution_id = %s
                """, (workflow_execution_id,))
                
                result = cur.fetchone()
                if not result:
                    return None
                
                # Load dependencies
                cur.execute("""
                    SELECT agent_name, depends_on_agents 
                    FROM agent_dependencies 
                    WHERE execution_plan_id = %s
                """, (result[0],))
                
                dependencies = {}
                for dep_result in cur.fetchall():
                    dependencies[dep_result[0]] = dep_result[1]
                
                return ExecutionPlan(
                    id=result[0],
                    campaign_id=result[1],
                    workflow_execution_id=result[2],
                    strategy=result[3],
                    agent_sequence=result[4],
                    dependencies=dependencies,
                    parallel_groups={},  # Could be reconstructed if needed
                    estimated_duration=result[5],
                    created_at=result[6]
                )
                
        except Exception as e:
            logger.error(f"Failed to load execution plan: {str(e)}")
            return None
    
    async def get_workflow_status(self, workflow_execution_id: str) -> Dict[str, Any]:
        """
        Get current workflow execution status.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT 
                        current_step,
                        current_agents_running,
                        completed_agents,
                        failed_agents,
                        waiting_agents,
                        start_time,
                        estimated_completion_time,
                        actual_completion_time,
                        execution_metadata
                    FROM workflow_state_live 
                    WHERE workflow_execution_id = %s
                """, (workflow_execution_id,))
                
                result = cur.fetchone()
                if not result:
                    return {"status": "not_found"}
                
                # Calculate progress
                total_agents = len(result[2] or []) + len(result[3] or []) + len(result[1] or []) + len(result[4] or [])
                completed_count = len(result[2]) if result[2] else 0
                
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
                    "timing": {
                        "start_time": result[5].isoformat() if result[5] else None,
                        "estimated_completion": result[6].isoformat() if result[6] else None,
                        "actual_completion": result[7].isoformat() if result[7] else None
                    },
                    "metadata": result[8] or {},
                    "status": "completed" if result[7] else ("running" if result[1] else "waiting")
                }
                
        except Exception as e:
            logger.error(f"Failed to get workflow status: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    # Implement required abstract methods from base class
    def execute_workflow(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Create execution plan (synchronous wrapper).
        
        Args:
            input_data: Must contain 'campaign_id' and 'strategy' (optional)
            context: Execution context
            
        Returns:
            AgentResult: Planning result with execution plan
        """
        if "campaign_id" not in input_data:
            return AgentResult(
                success=False,
                error_message="campaign_id is required in input_data",
                error_code="MISSING_CAMPAIGN_ID"
            )
        
        campaign_id = input_data["campaign_id"]
        strategy = input_data.get("strategy", "adaptive")
        workflow_execution_id = input_data.get("workflow_execution_id", str(uuid.uuid4()))
        
        # Run async planning
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            plan = loop.run_until_complete(
                self.create_execution_plan(campaign_id, workflow_execution_id, strategy)
            )
            
            return AgentResult(
                success=True,
                data={
                    "execution_plan_id": plan.id,
                    "workflow_execution_id": plan.workflow_execution_id,
                    "strategy": plan.strategy,
                    "total_agents": len(plan.agent_sequence),
                    "estimated_duration_minutes": plan.estimated_duration,
                    "agent_sequence": plan.agent_sequence,
                    "parallel_groups": plan.parallel_groups
                },
                metadata={
                    "planner_agent": self.metadata.name,
                    "created_at": plan.created_at.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {str(e)}")
            return AgentResult(
                success=False,
                error_message=f"Planning failed: {str(e)}",
                error_code="PLANNING_ERROR"
            )
    
    async def update_agent_status(
        self,
        workflow_execution_id: str,
        agent_name: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time_seconds: Optional[float] = None
    ) -> bool:
        """
        Update the execution status of a specific agent within a workflow.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get current workflow state
                cur.execute("""
                    SELECT current_agents_running, completed_agents, failed_agents, waiting_agents,
                           intermediate_outputs
                    FROM workflow_state_live 
                    WHERE workflow_execution_id = %s
                """, (workflow_execution_id,))
                
                result = cur.fetchone()
                if not result:
                    logger.warning(f"Workflow {workflow_execution_id} not found")
                    return False
                
                current_running, completed, failed, waiting, outputs = result
                current_running = current_running or []
                completed = completed or []
                failed = failed or []
                waiting = waiting or []
                outputs = outputs or {}
                
                # Update agent lists based on status
                if status == "starting":
                    if agent_name not in current_running:
                        current_running.append(agent_name)
                    if agent_name in waiting:
                        waiting.remove(agent_name)
                        
                elif status == "completed":
                    if agent_name in current_running:
                        current_running.remove(agent_name)
                    if agent_name not in completed:
                        completed.append(agent_name)
                    if output_data:
                        outputs[agent_name] = output_data
                        
                elif status == "failed":
                    if agent_name in current_running:
                        current_running.remove(agent_name)
                    if agent_name not in failed:
                        failed.append(agent_name)
                    if error_message:
                        outputs[f"{agent_name}_error"] = {
                            "error_message": error_message,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                
                # Update workflow state
                cur.execute("""
                    UPDATE workflow_state_live 
                    SET current_agents_running = %s,
                        completed_agents = %s,
                        failed_agents = %s,
                        waiting_agents = %s,
                        intermediate_outputs = %s,
                        last_heartbeat = NOW(),
                        updated_at = NOW()
                    WHERE workflow_execution_id = %s
                """, (
                    current_running, completed, failed, waiting,
                    json.dumps(outputs), workflow_execution_id
                ))
                
                conn.commit()
                logger.info(f"Updated agent {agent_name} status to {status} for workflow {workflow_execution_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update agent status: {str(e)}")
            return False
    
    async def get_active_workflows(self) -> List[Dict[str, Any]]:
        """
        Get all currently active workflows with their status and progress.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT 
                        ep.campaign_id,
                        wsl.workflow_execution_id,
                        wsl.current_step,
                        wsl.current_agents_running,
                        wsl.completed_agents,
                        wsl.failed_agents,
                        wsl.waiting_agents,
                        wsl.start_time,
                        wsl.estimated_completion_time,
                        wsl.actual_completion_time,
                        wsl.last_heartbeat,
                        ep.total_agents,
                        ep.execution_strategy
                    FROM workflow_state_live wsl
                    JOIN execution_plans ep ON ep.workflow_execution_id = wsl.workflow_execution_id
                    WHERE wsl.actual_completion_time IS NULL  -- Only active workflows
                    AND wsl.last_heartbeat > NOW() - INTERVAL '1 hour'  -- Recent activity
                    ORDER BY wsl.start_time DESC
                """)
                
                active_workflows = []
                for result in cur.fetchall():
                    (campaign_id, workflow_id, current_step, running, completed, failed,
                     waiting, start_time, est_completion, actual_completion, last_heartbeat,
                     total_agents, strategy) = result
                    
                    # Calculate progress
                    completed_count = len(completed) if completed else 0
                    failed_count = len(failed) if failed else 0
                    total_processed = completed_count + failed_count
                    progress_percentage = (total_processed / total_agents) * 100 if total_agents > 0 else 0
                    
                    # Determine status
                    if failed_count > 0 and total_processed == total_agents:
                        status = "failed"
                    elif completed_count == total_agents:
                        status = "completed"
                    elif len(running or []) > 0:
                        status = "running"
                    else:
                        status = "waiting"
                    
                    active_workflows.append({
                        "workflow_execution_id": workflow_id,
                        "campaign_id": campaign_id,
                        "status": status,
                        "progress_percentage": round(progress_percentage, 1),
                        "current_step": current_step,
                        "total_agents": total_agents,
                        "completed_agents": completed or [],
                        "failed_agents": failed or [],
                        "waiting_agents": waiting or [],
                        "current_agents_running": running or [],
                        "start_time": start_time.isoformat() if start_time else None,
                        "estimated_completion_time": est_completion.isoformat() if est_completion else None,
                        "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
                        "execution_strategy": strategy
                    })
                
                logger.info(f"Found {len(active_workflows)} active workflows")
                return active_workflows
                
        except Exception as e:
            logger.error(f"Failed to get active workflows: {str(e)}")
            return []