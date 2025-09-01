"""
Campaign Orchestrator Agent - Core orchestration engine for campaign-centric workflows.

This agent manages the complete lifecycle of campaign execution, from task distribution
to workflow monitoring and result aggregation, leveraging the new campaign-centric
database architecture.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

from ..core.base_agent import (
    WorkflowAgent, AgentMetadata, AgentType, AgentResult, 
    AgentExecutionContext, AgentStatus
)
from .types import (
    CampaignType, TaskStatus, WorkflowStatus,
    CampaignTask, CampaignWithTasks, WorkflowExecutionCreate
)
from .campaign_database_service import CampaignDatabaseService
from .workflow_state_manager import WorkflowStateManager, CampaignWorkflowState
from ..core.agent_factory import AgentFactory
from langchain_core.runnables import RunnableConfig
# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END

logger = logging.getLogger(__name__)

class CampaignOrchestratorAgent(WorkflowAgent):
    """
    Primary orchestrator for campaign-centric workflows.
    
    This agent coordinates the execution of complex, multi-step campaigns by:
    - Analyzing campaign requirements and creating task plans
    - Distributing tasks to specialized agents based on capabilities
    - Monitoring workflow progress and handling failures
    - Aggregating results and updating campaign status
    - Maintaining workflow state for recovery and resumption
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
                name="CampaignOrchestratorAgent",
                description="Core orchestrator for campaign-centric workflows",
                capabilities=[
                    "campaign_planning", "task_distribution", "workflow_monitoring",
                    "agent_coordination", "state_management", "error_recovery"
                ],
                dependencies=["database_service", "agent_factory", "state_manager"]
            )
        
        super().__init__(metadata)
        
        # Initialize core services
        self.db_service = CampaignDatabaseService()
        self.state_manager = WorkflowStateManager()
        self.agent_factory = AgentFactory()
        
        # Workflow management
        self.active_workflows: Dict[str, CampaignWorkflowState] = {}
        self.workflow_graphs: Dict[CampaignType, StateGraph] = {}
        
        # Performance tracking
        self.execution_metrics = {
            "campaigns_orchestrated": 0,
            "tasks_distributed": 0,
            "successful_completions": 0,
            "failed_executions": 0,
            "recovery_attempts": 0
        }
        
        logger.info(f"Initialized {self.metadata.name} with database and state management")
    
    async def orchestrate_campaign(self, campaign_id: str) -> AgentResult:
        """
        Main orchestration method - execute a complete campaign workflow.
        
        Args:
            campaign_id: ID of the campaign to orchestrate
            
        Returns:
            AgentResult: Final orchestration result with campaign outcomes
        """
        start_time = datetime.utcnow()
        workflow_execution_id = str(uuid.uuid4())
        
        try:
            # Load campaign with all tasks and dependencies
            campaign = await self.db_service.get_campaign_with_tasks(campaign_id)
            if not campaign:
                return AgentResult(
                    success=False,
                    error_message=f"Campaign {campaign_id} not found",
                    error_code="CAMPAIGN_NOT_FOUND"
                )
            
            # Create workflow execution record
            workflow_execution = WorkflowExecutionCreate(
                campaign_id=campaign_id,
                orchestrator_id=self.metadata.agent_id,
                workflow_type=campaign.campaign_type.value,
                input_data={"campaign": campaign.__dict__},
                metadata={"started_by": "orchestrator", "start_time": start_time.isoformat()}
            )
            
            execution_id = await self.db_service.log_workflow_execution(workflow_execution)
            
            # Initialize workflow state
            workflow_state = CampaignWorkflowState(
                campaign_id=campaign_id,
                workflow_execution_id=execution_id,
                current_task_id=None,
                completed_tasks=[],
                failed_tasks=[],
                agent_results={},
                workflow_metadata={
                    "campaign_type": campaign.campaign_type.value,
                    "total_tasks": len(campaign.tasks),
                    "orchestrator_id": self.metadata.agent_id
                }
            )
            
            # Save initial state
            await self.state_manager.save_workflow_state(workflow_state)
            self.active_workflows[workflow_execution_id] = workflow_state
            
            # Create or get workflow graph for campaign type
            workflow_graph = self._create_workflow_graph(campaign.campaign_type)
            
            # Execute workflow
            logger.info(f"Starting orchestration for campaign {campaign_id} (type: {campaign.campaign_type.value})")
            
            if campaign.tasks:
                # Distribute tasks to agents
                distribution_result = await self.distribute_tasks(campaign.tasks)
                workflow_state.workflow_metadata["distribution_result"] = distribution_result
                
                # Monitor workflow progress
                final_status = await self.monitor_workflow_progress(execution_id)
                workflow_state.workflow_metadata["final_status"] = final_status.value
            else:
                logger.warning(f"Campaign {campaign_id} has no tasks to execute")
                final_status = WorkflowStatus.COMPLETED
            
            # Calculate execution metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self.execution_metrics["campaigns_orchestrated"] += 1
            if final_status == WorkflowStatus.COMPLETED:
                self.execution_metrics["successful_completions"] += 1
            else:
                self.execution_metrics["failed_executions"] += 1
            
            # Prepare final result
            result_data = {
                "campaign_id": campaign_id,
                "workflow_execution_id": execution_id,
                "final_status": final_status.value,
                "total_tasks": len(campaign.tasks),
                "completed_tasks": len(workflow_state.completed_tasks),
                "failed_tasks": len(workflow_state.failed_tasks),
                "execution_time_ms": execution_time,
                "agent_results": workflow_state.agent_results
            }
            
            success = final_status in [WorkflowStatus.COMPLETED]
            
            return AgentResult(
                success=success,
                data=result_data,
                execution_time_ms=execution_time,
                metadata={
                    "workflow_execution_id": execution_id,
                    "campaign_type": campaign.campaign_type.value,
                    "orchestrator_metrics": self.execution_metrics
                }
            )
            
        except Exception as e:
            logger.error(f"Campaign orchestration failed for {campaign_id}: {str(e)}")
            
            # Update failure metrics
            self.execution_metrics["failed_executions"] += 1
            
            return AgentResult(
                success=False,
                error_message=f"Orchestration failed: {str(e)}",
                error_code="ORCHESTRATION_ERROR",
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                metadata={"campaign_id": campaign_id, "error_details": str(e)}
            )
        
        finally:
            # Cleanup active workflow
            if workflow_execution_id in self.active_workflows:
                del self.active_workflows[workflow_execution_id]
    
    async def distribute_tasks(self, campaign_tasks: List[CampaignTask]) -> Dict[str, AgentResult]:
        """
        Distribute campaign tasks to appropriate specialized agents.
        
        Args:
            campaign_tasks: List of tasks to distribute
            
        Returns:
            Dict mapping task IDs to agent assignment results
        """
        distribution_results = {}
        task_queue = sorted(campaign_tasks, key=lambda t: (len(t.dependencies), t.priority), reverse=True)
        
        logger.info(f"Distributing {len(campaign_tasks)} tasks to specialized agents")
        
        for task in task_queue:
            try:
                # Check if task dependencies are satisfied
                if not self._are_dependencies_satisfied(task, distribution_results):
                    logger.info(f"Task {task.id} dependencies not satisfied, deferring")
                    task.status = TaskStatus.PENDING
                    continue
                
                # Find appropriate agent for task
                agent = await self._find_agent_for_task(task)
                if not agent:
                    logger.error(f"No suitable agent found for task {task.id} (type: {task.agent_type})")
                    task.status = TaskStatus.FAILED
                    distribution_results[task.id] = AgentResult(
                        success=False,
                        error_message=f"No agent available for task type: {task.agent_type}",
                        error_code="NO_AGENT_AVAILABLE"
                    )
                    continue
                
                # Assign task to agent
                task.assigned_agent_id = agent.metadata.agent_id
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()
                
                # Update task status in database
                await self.db_service.update_task_status(task.id, task.status, None)
                
                # Execute task asynchronously (fire and forget for now)
                logger.info(f"Assigned task {task.id} to agent {agent.metadata.name}")
                
                distribution_results[task.id] = AgentResult(
                    success=True,
                    data={"assigned_agent": agent.metadata.agent_id, "status": "assigned"},
                    metadata={"task_type": task.task_type, "agent_type": task.agent_type}
                )
                
                self.execution_metrics["tasks_distributed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to distribute task {task.id}: {str(e)}")
                task.status = TaskStatus.FAILED
                distribution_results[task.id] = AgentResult(
                    success=False,
                    error_message=f"Task distribution failed: {str(e)}",
                    error_code="DISTRIBUTION_ERROR"
                )
        
        return distribution_results
    
    async def monitor_workflow_progress(self, workflow_execution_id: str) -> WorkflowStatus:
        """
        Monitor the progress of a workflow execution until completion or failure.
        
        Args:
            workflow_execution_id: ID of the workflow execution to monitor
            
        Returns:
            WorkflowStatus: Final status of the workflow
        """
        logger.info(f"Starting workflow monitoring for execution {workflow_execution_id}")
        
        start_time = datetime.utcnow()
        timeout = timedelta(hours=2)  # Default 2-hour timeout
        check_interval = 10  # Check every 10 seconds
        
        while (datetime.utcnow() - start_time) < timeout:
            try:
                # Load current workflow state
                if workflow_execution_id not in self.active_workflows:
                    state = await self.state_manager.load_workflow_state(workflow_execution_id)
                    self.active_workflows[workflow_execution_id] = state
                else:
                    state = self.active_workflows[workflow_execution_id]
                
                # Check campaign progress from database
                campaign = await self.db_service.get_campaign_with_tasks(state.campaign_id)
                if not campaign:
                    logger.error(f"Campaign {state.campaign_id} not found during monitoring")
                    return WorkflowStatus.FAILED
                
                # Analyze task completion status
                total_tasks = len(campaign.tasks)
                completed_tasks = len([t for t in campaign.tasks if t.status == TaskStatus.COMPLETED])
                failed_tasks = len([t for t in campaign.tasks if t.status == TaskStatus.FAILED])
                in_progress_tasks = len([t for t in campaign.tasks if t.status == TaskStatus.IN_PROGRESS])
                
                # Update workflow state
                state.completed_tasks = [t.id for t in campaign.tasks if t.status == TaskStatus.COMPLETED]
                state.failed_tasks = [t.id for t in campaign.tasks if t.status == TaskStatus.FAILED]
                
                # Determine workflow status
                if failed_tasks > 0 and (failed_tasks + completed_tasks) == total_tasks:
                    # All tasks complete but some failed
                    logger.warning(f"Workflow {workflow_execution_id} completed with {failed_tasks} failures")
                    return WorkflowStatus.FAILED if failed_tasks > completed_tasks else WorkflowStatus.COMPLETED
                
                elif completed_tasks == total_tasks:
                    # All tasks completed successfully
                    logger.info(f"Workflow {workflow_execution_id} completed successfully")
                    return WorkflowStatus.COMPLETED
                
                elif in_progress_tasks == 0 and completed_tasks + failed_tasks < total_tasks:
                    # No tasks in progress but not all complete - stuck
                    logger.error(f"Workflow {workflow_execution_id} appears stuck - no progress")
                    return WorkflowStatus.FAILED
                
                # Save updated state
                await self.state_manager.save_workflow_state(state)
                
                # Log progress
                logger.info(f"Workflow progress: {completed_tasks}/{total_tasks} complete, "
                          f"{in_progress_tasks} in progress, {failed_tasks} failed")
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring workflow {workflow_execution_id}: {str(e)}")
                await asyncio.sleep(check_interval)
        
        # Timeout reached
        logger.error(f"Workflow {workflow_execution_id} monitoring timed out")
        return WorkflowStatus.FAILED
    
    def _create_workflow_graph(self, campaign_type: CampaignType) -> StateGraph:
        """
        Create a LangGraph workflow for the specified campaign type.
        
        Args:
            campaign_type: Type of campaign to create workflow for
            
        Returns:
            StateGraph: Configured workflow graph
        """
        if campaign_type in self.workflow_graphs:
            return self.workflow_graphs[campaign_type]
        
        # Create new workflow graph based on campaign type
        workflow = StateGraph(dict)
        
        if campaign_type == CampaignType.BLOG_CREATION:
            # Blog creation workflow: Planning -> Research -> Parallel(Writing + Images) -> Editing -> Publishing
            workflow.add_node("initialize", self._initialize_blog_workflow)
            workflow.add_node("planning", self._execute_planning_phase)
            workflow.add_node("research", self._execute_research_phase)
            workflow.add_node("writing", self._execute_writing_phase)
            workflow.add_node("image_generation", self._execute_image_generation_phase)
            workflow.add_node("content_merge", self._execute_content_merge_phase)
            workflow.add_node("editing", self._execute_editing_phase)
            workflow.add_node("publishing", self._execute_publishing_phase)
            workflow.add_node("finalize", self._finalize_workflow)
            
            workflow.add_edge("initialize", "planning")
            workflow.add_edge("planning", "research")
            workflow.add_edge("research", "writing")
            workflow.add_edge("research", "image_generation")
            workflow.add_edge("writing", "content_merge")
            workflow.add_edge("image_generation", "content_merge")
            workflow.add_edge("content_merge", "editing")
            workflow.add_edge("editing", "publishing")
            workflow.add_edge("publishing", "finalize")
            workflow.add_edge("finalize", END)
            
        elif campaign_type == CampaignType.CONTENT_REPURPOSING:
            # Content repurposing workflow: Analysis -> Adaptation -> Generation -> Review -> Distribution
            workflow.add_node("initialize", self._initialize_repurposing_workflow)
            workflow.add_node("analysis", self._execute_analysis_phase)
            workflow.add_node("adaptation", self._execute_adaptation_phase)
            workflow.add_node("generation", self._execute_generation_phase)
            workflow.add_node("review", self._execute_review_phase)
            workflow.add_node("distribution", self._execute_distribution_phase)
            workflow.add_node("finalize", self._finalize_workflow)
            
            workflow.add_edge("initialize", "analysis")
            workflow.add_edge("analysis", "adaptation")
            workflow.add_edge("adaptation", "generation")
            workflow.add_edge("generation", "review")
            workflow.add_edge("review", "distribution")
            workflow.add_edge("distribution", "finalize")
            workflow.add_edge("finalize", END)
            
        else:
            # Default workflow for other campaign types
            workflow.add_node("initialize", self._initialize_default_workflow)
            workflow.add_node("execute", self._execute_default_phase)
            workflow.add_node("finalize", self._finalize_workflow)
            
            workflow.add_edge("initialize", "execute")
            workflow.add_edge("execute", "finalize")
            workflow.add_edge("finalize", END)
        
        workflow.set_entry_point("initialize")
        
        # Cache the compiled workflow
        compiled_workflow = workflow.compile()
        self.workflow_graphs[campaign_type] = compiled_workflow
        
        logger.info(f"Created workflow graph for campaign type: {campaign_type.value}")
        return compiled_workflow
    
    async def _find_agent_for_task(self, task: CampaignTask) -> Optional[Any]:
        """Find an appropriate agent for the given task."""
        try:
            # Use agent factory to create agent based on task requirements
            agent = self.agent_factory.create_agent(task.agent_type)
            return agent
        except Exception as e:
            logger.error(f"Failed to find agent for task {task.id}: {str(e)}")
            return None
    
    def _are_dependencies_satisfied(self, task: CampaignTask, results: Dict[str, AgentResult]) -> bool:
        """Check if all task dependencies have been completed successfully."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in results or not results[dep_id].success:
                return False
        
        return True
    
    # Workflow phase execution methods (placeholders for now)
    async def _initialize_blog_workflow(self, state: dict) -> dict:
        """Initialize blog creation workflow."""
        logger.info("Initializing blog creation workflow")
        state["phase"] = "planning"
        state["initialized_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_planning_phase(self, state: dict) -> dict:
        """Execute planning phase of blog workflow."""
        logger.info("Executing planning phase")
        state["phase"] = "research"
        state["planning_completed_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_research_phase(self, state: dict) -> dict:
        """Execute research phase of blog workflow."""
        logger.info("Executing research phase")
        state["phase"] = "writing"
        state["research_completed_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_writing_phase(self, state: dict) -> dict:
        """Execute writing phase of blog workflow."""
        logger.info("Executing writing phase")
        state["phase"] = "editing"
        state["writing_completed_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_editing_phase(self, state: dict) -> dict:
        """Execute editing phase of blog workflow."""
        logger.info("Executing editing phase")
        state["phase"] = "publishing"
        state["editing_completed_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_image_generation_phase(self, state: dict) -> dict:
        """Execute image generation phase of blog workflow."""
        logger.info("Executing image generation phase")
        
        # Extract content context for image generation
        content_context = {
            "content": state.get("blog_content", ""),
            "blog_title": state.get("blog_title", ""),
            "outline": state.get("outline", []),
            "style": "professional",
            "count": 3
        }
        
        # Create Image Agent task
        from ..specialized.image_agent import ImageAgent
        image_agent = ImageAgent()
        
        try:
            result = image_agent.execute(content_context)
            if result.success:
                state["generated_images"] = result.data.get("images", [])
                state["image_prompts"] = result.data.get("prompts", [])
                logger.info(f"Generated {len(state.get('generated_images', []))} images")
            else:
                logger.error(f"Image generation failed: {result.error_message}")
                state["image_generation_error"] = result.error_message
        except Exception as e:
            logger.error(f"Image generation phase failed: {str(e)}")
            state["image_generation_error"] = str(e)
        
        state["image_generation_completed_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_content_merge_phase(self, state: dict) -> dict:
        """Merge content and images into final deliverable."""
        logger.info("Executing content merge phase")
        
        # Combine text content with generated images
        final_content = {
            "blog_content": state.get("blog_content", ""),
            "blog_title": state.get("blog_title", ""),
            "images": state.get("generated_images", []),
            "image_prompts": state.get("image_prompts", []),
            "outline": state.get("outline", [])
        }
        
        state["final_content"] = final_content
        state["content_merge_completed_at"] = datetime.utcnow().isoformat()
        logger.info("Content and images merged successfully")
        
        return state
    
    async def _execute_publishing_phase(self, state: dict) -> dict:
        """Execute publishing phase of blog workflow."""
        logger.info("Executing publishing phase")
        state["phase"] = "finalized"
        state["publishing_completed_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _initialize_repurposing_workflow(self, state: dict) -> dict:
        """Initialize content repurposing workflow."""
        logger.info("Initializing content repurposing workflow")
        state["phase"] = "analysis"
        return state
    
    async def _execute_analysis_phase(self, state: dict) -> dict:
        """Execute analysis phase of repurposing workflow."""
        logger.info("Executing analysis phase")
        state["phase"] = "adaptation"
        return state
    
    async def _execute_adaptation_phase(self, state: dict) -> dict:
        """Execute adaptation phase of repurposing workflow."""
        logger.info("Executing adaptation phase")
        state["phase"] = "generation"
        return state
    
    async def _execute_generation_phase(self, state: dict) -> dict:
        """Execute generation phase of repurposing workflow."""
        logger.info("Executing generation phase")
        state["phase"] = "review"
        return state
    
    async def _execute_review_phase(self, state: dict) -> dict:
        """Execute review phase of repurposing workflow."""
        logger.info("Executing review phase")
        state["phase"] = "distribution"
        return state
    
    async def _execute_distribution_phase(self, state: dict) -> dict:
        """Execute distribution phase of repurposing workflow."""
        logger.info("Executing distribution phase")
        state["phase"] = "finalized"
        return state
    
    async def _initialize_default_workflow(self, state: dict) -> dict:
        """Initialize default workflow for unknown campaign types."""
        logger.info("Initializing default workflow")
        state["phase"] = "execute"
        return state
    
    async def _execute_default_phase(self, state: dict) -> dict:
        """Execute default phase for unknown campaign types."""
        logger.info("Executing default phase")
        state["phase"] = "finalized"
        return state
    
    async def _finalize_workflow(self, state: dict) -> dict:
        """Finalize workflow execution."""
        logger.info("Finalizing workflow")
        state["phase"] = "completed"
        state["finalized_at"] = datetime.utcnow().isoformat()
        return state
    
    def execute_workflow(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Execute campaign orchestration workflow (synchronous wrapper).
        
        Args:
            input_data: Must contain 'campaign_id' key
            context: Execution context
            
        Returns:
            AgentResult: Orchestration result
        """
        if "campaign_id" not in input_data:
            return AgentResult(
                success=False,
                error_message="campaign_id is required in input_data",
                error_code="MISSING_CAMPAIGN_ID"
            )
        
        campaign_id = input_data["campaign_id"]
        
        # Run async orchestration in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if loop is already running
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.orchestrate_campaign(campaign_id))
                    return future.result()
            else:
                return loop.run_until_complete(self.orchestrate_campaign(campaign_id))
        except Exception as e:
            logger.error(f"Failed to execute campaign orchestration: {str(e)}")
            return AgentResult(
                success=False,
                error_message=f"Orchestration execution failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get current orchestration performance metrics."""
        return {
            **self.execution_metrics,
            "active_workflows": len(self.active_workflows),
            "cached_workflow_graphs": len(self.workflow_graphs),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def resume_workflow(self, workflow_execution_id: str) -> AgentResult:
        """
        Resume a paused or failed workflow from its last saved state.
        
        Args:
            workflow_execution_id: ID of the workflow execution to resume
            
        Returns:
            AgentResult: Resume operation result
        """
        try:
            # Load workflow state
            state = await self.state_manager.load_workflow_state(workflow_execution_id)
            
            logger.info(f"Resuming workflow {workflow_execution_id} for campaign {state.campaign_id}")
            
            # Resume monitoring
            final_status = await self.monitor_workflow_progress(workflow_execution_id)
            
            self.execution_metrics["recovery_attempts"] += 1
            
            return AgentResult(
                success=final_status == WorkflowStatus.COMPLETED,
                data={
                    "workflow_execution_id": workflow_execution_id,
                    "final_status": final_status.value,
                    "resumed_at": datetime.utcnow().isoformat()
                },
                metadata={"operation": "resume_workflow"}
            )
            
        except Exception as e:
            logger.error(f"Failed to resume workflow {workflow_execution_id}: {str(e)}")
            return AgentResult(
                success=False,
                error_message=f"Failed to resume workflow: {str(e)}",
                error_code="RESUME_ERROR"
            )