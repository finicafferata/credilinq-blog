"""
Campaign Orchestrator LangGraph Implementation - Enhanced multi-agent coordination.

This is the LangGraph-migrated version of the Campaign Orchestrator with enhanced capabilities:
- Multi-agent coordination with shared state
- Campaign-level workflow management and progress tracking
- Parallel task execution for campaign components
- Enhanced recovery and error handling
- Full workflow checkpointing and state persistence
"""

import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Internal imports
from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, LangGraphExecutionContext,
    WorkflowStatus, CheckpointStrategy
)
from ..workflow.blog_workflow_langgraph import BlogWorkflowOrchestrator
from ..specialized.content_repurposer import ContentRepurposer
from ..specialized.social_media_agent import SocialMediaAgent
from ..specialized.distribution_agent import DistributionAgent
from ..specialized.seo_agent import SEOAgent
from ..core.agent_factory import AgentFactory
from .types import CampaignType, TaskStatus, CampaignTask, CampaignWithTasks
from .campaign_database_service import CampaignDatabaseService

import logging
logger = logging.getLogger(__name__)

class CampaignPhase(Enum):
    """Phases of campaign execution."""
    INITIALIZATION = "initialization"
    PLANNING = "planning" 
    CONTENT_CREATION = "content_creation"
    CONTENT_OPTIMIZATION = "content_optimization"
    CONTENT_REPURPOSING = "content_repurposing"
    QUALITY_ASSURANCE = "quality_assurance"
    DISTRIBUTION = "distribution"
    MONITORING = "monitoring"
    COMPLETION = "completion"

class TaskExecutionMode(Enum):
    """Task execution modes for flexibility."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    HYBRID = "hybrid"

@dataclass
class CampaignAgentResult:
    """Result from agent execution within campaign."""
    agent_name: str
    task_id: str
    success: bool
    execution_time_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    dependencies_met: bool = True
    checkpoints: List[str] = field(default_factory=list)

class CampaignOrchestratorState(TypedDict):
    """Comprehensive state schema for campaign orchestration."""
    # Campaign identification
    campaign_id: str
    campaign_type: str
    workflow_execution_id: str
    
    # Campaign configuration
    campaign_config: Dict[str, Any]
    execution_mode: str
    max_parallel_tasks: int
    timeout_minutes: int
    
    # Workflow orchestration
    current_phase: str
    completed_phases: List[str]
    failed_phases: List[str]
    phase_checkpoints: Dict[str, Dict[str, Any]]
    
    # Task management
    all_tasks: List[Dict[str, Any]]
    pending_tasks: List[str]
    in_progress_tasks: List[str] 
    completed_tasks: List[str]
    failed_tasks: List[str]
    task_dependencies: Dict[str, List[str]]
    task_results: Dict[str, CampaignAgentResult]
    
    # Agent coordination
    active_agents: Dict[str, str]
    agent_assignments: Dict[str, str]
    agent_results: Dict[str, Any]
    
    # Content pipeline state (for content campaigns)
    content_artifacts: Dict[str, Any]
    content_quality_scores: Dict[str, float]
    distribution_channels: List[str]
    
    # Progress tracking
    overall_progress: float
    phase_progress: Dict[str, float]
    estimated_completion: Optional[str]
    
    # Quality and monitoring
    quality_gates_passed: List[str]
    quality_gates_failed: List[str]
    monitoring_metrics: Dict[str, Any]
    
    # Timing and performance
    started_at: str
    updated_at: str
    completed_at: Optional[str]
    total_execution_time_ms: float
    phase_execution_times: Dict[str, float]
    
    # Error handling and recovery
    error_state: Optional[str]
    recovery_attempts: int
    max_recovery_attempts: int
    retry_strategies: Dict[str, str]
    
    # Final output
    campaign_deliverables: Dict[str, Any]
    campaign_metrics: Dict[str, Any]
    success_criteria_met: bool

class CampaignOrchestratorLangGraph(LangGraphWorkflowBase[CampaignOrchestratorState]):
    """
    Enhanced Campaign Orchestrator using LangGraph for superior multi-agent coordination,
    state management, and campaign workflow orchestration.
    """
    
    def __init__(
        self,
        workflow_name: str = "campaign_orchestrator",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        max_retries: int = 3
    ):
        super().__init__(workflow_name, checkpoint_strategy=checkpoint_strategy, max_retries=max_retries)
        
        # Initialize services and agents
        self.db_service = CampaignDatabaseService()
        self.agent_factory = AgentFactory()
        
        # Initialize specialized workflow orchestrators
        self.blog_orchestrator = BlogWorkflowOrchestrator()
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Configuration
        self.max_concurrent_tasks = 5
        self.default_timeout_minutes = 120
        self.quality_thresholds = {
            "min_content_quality": 7.0,
            "min_seo_score": 6.0,
            "min_distribution_success": 0.8
        }
        
        self.logger.info("Campaign Orchestrator LangGraph initialized")
    
    def _initialize_agents(self):
        """Initialize specialized agents for campaign orchestration."""
        try:
            self.content_repurposer = ContentRepurposer()
            self.social_media_agent = SocialMediaAgent()
            self.distribution_agent = DistributionAgent()
            self.seo_agent = SEOAgent()
            
            self.logger.info("Specialized campaign agents initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize campaign agents: {e}")
            raise
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the enhanced LangGraph workflow for campaign orchestration."""
        workflow = StateGraph(CampaignOrchestratorState)
        
        # Add workflow nodes
        workflow.add_node("initialize_campaign", self._initialize_campaign)
        workflow.add_node("plan_campaign", self._plan_campaign_execution)
        workflow.add_node("execute_content_creation", self._execute_content_creation)
        workflow.add_node("execute_content_optimization", self._execute_content_optimization)
        workflow.add_node("execute_content_repurposing", self._execute_content_repurposing)
        workflow.add_node("execute_quality_assurance", self._execute_quality_assurance)
        workflow.add_node("execute_distribution", self._execute_distribution)
        workflow.add_node("monitor_campaign", self._monitor_campaign_progress)
        workflow.add_node("handle_recovery", self._handle_recovery)
        workflow.add_node("finalize_campaign", self._finalize_campaign)
        
        # Set entry point
        workflow.set_entry_point("initialize_campaign")
        
        # Sequential workflow edges
        workflow.add_edge("initialize_campaign", "plan_campaign")
        workflow.add_edge("plan_campaign", "execute_content_creation")
        workflow.add_edge("execute_content_creation", "execute_content_optimization")
        workflow.add_edge("execute_content_optimization", "execute_content_repurposing")
        workflow.add_edge("execute_content_repurposing", "execute_quality_assurance")
        workflow.add_edge("execute_quality_assurance", "execute_distribution")
        workflow.add_edge("execute_distribution", "monitor_campaign")
        
        # Conditional edges for recovery and completion
        workflow.add_conditional_edges(
            "monitor_campaign",
            self._should_recover_or_complete,
            {
                "recover": "handle_recovery",
                "complete": "finalize_campaign"
            }
        )
        
        workflow.add_edge("handle_recovery", "monitor_campaign")
        workflow.add_edge("finalize_campaign", END)
        
        return workflow
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> CampaignOrchestratorState:
        """Create comprehensive initial state for campaign orchestration."""
        campaign_id = input_data.get("campaign_id", str(uuid.uuid4()))
        workflow_execution_id = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()
        
        return CampaignOrchestratorState(
            # Campaign identification
            campaign_id=campaign_id,
            campaign_type=input_data.get("campaign_type", "BLOG_CREATION"),
            workflow_execution_id=workflow_execution_id,
            
            # Configuration
            campaign_config=input_data.get("campaign_config", {}),
            execution_mode=input_data.get("execution_mode", TaskExecutionMode.HYBRID.value),
            max_parallel_tasks=input_data.get("max_parallel_tasks", 3),
            timeout_minutes=input_data.get("timeout_minutes", 120),
            
            # Workflow orchestration
            current_phase=CampaignPhase.INITIALIZATION.value,
            completed_phases=[],
            failed_phases=[],
            phase_checkpoints={},
            
            # Task management
            all_tasks=input_data.get("tasks", []),
            pending_tasks=[],
            in_progress_tasks=[],
            completed_tasks=[],
            failed_tasks=[],
            task_dependencies={},
            task_results={},
            
            # Agent coordination
            active_agents={},
            agent_assignments={},
            agent_results={},
            
            # Content pipeline
            content_artifacts={},
            content_quality_scores={},
            distribution_channels=input_data.get("distribution_channels", []),
            
            # Progress tracking
            overall_progress=0.0,
            phase_progress={},
            estimated_completion=None,
            
            # Quality monitoring
            quality_gates_passed=[],
            quality_gates_failed=[],
            monitoring_metrics={},
            
            # Timing
            started_at=current_time,
            updated_at=current_time,
            completed_at=None,
            total_execution_time_ms=0.0,
            phase_execution_times={},
            
            # Error handling
            error_state=None,
            recovery_attempts=0,
            max_recovery_attempts=3,
            retry_strategies={},
            
            # Final output
            campaign_deliverables={},
            campaign_metrics={},
            success_criteria_met=False
        )
    
    async def _initialize_campaign(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Initialize the campaign orchestration with comprehensive setup."""
        self.logger.info(f"Initializing campaign orchestration: {state['campaign_id']}")
        
        start_time = datetime.utcnow()
        
        try:
            # Load campaign data from database
            campaign_data = await self._load_campaign_data(state["campaign_id"])
            
            if not campaign_data:
                state["error_state"] = f"Campaign {state['campaign_id']} not found"
                state["failed_phases"].append("initialization")
                return state
            
            # Initialize task queue and dependencies
            tasks = state.get("all_tasks", [])
            if not tasks and campaign_data.get("tasks"):
                tasks = campaign_data["tasks"]
                state["all_tasks"] = tasks
            
            # Build task dependency graph
            state["task_dependencies"] = self._build_dependency_graph(tasks)
            state["pending_tasks"] = [task["id"] for task in tasks if not task.get("dependencies")]
            
            # Initialize progress tracking
            total_phases = len(CampaignPhase)
            state["phase_progress"] = {phase.value: 0.0 for phase in CampaignPhase}
            
            # Set campaign configuration
            state["campaign_config"].update({
                "campaign_title": campaign_data.get("title", ""),
                "campaign_description": campaign_data.get("description", ""),
                "target_audience": campaign_data.get("target_audience", ""),
                "success_criteria": campaign_data.get("success_criteria", {})
            })
            
            # Initialize monitoring metrics
            state["monitoring_metrics"] = {
                "tasks_total": len(tasks),
                "tasks_pending": len(state["pending_tasks"]),
                "agents_available": len(self._get_available_agents()),
                "estimated_duration_minutes": self._estimate_campaign_duration(tasks)
            }
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["initialization"] = execution_time
            state["completed_phases"].append("initialization")
            state["current_phase"] = CampaignPhase.PLANNING.value
            state["overall_progress"] = 10.0  # 10% for initialization
            
            # Checkpoint
            state["phase_checkpoints"]["initialization"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "tasks_loaded": len(tasks),
                "dependencies_mapped": len(state["task_dependencies"]),
                "execution_time_ms": execution_time
            }
            
            self.logger.info(f"Campaign initialization completed: {len(tasks)} tasks, {len(state['pending_tasks'])} ready")
            
        except Exception as e:
            self.logger.error(f"Campaign initialization failed: {e}")
            state["error_state"] = f"Initialization failed: {str(e)}"
            state["failed_phases"].append("initialization")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _plan_campaign_execution(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Plan the execution strategy for the campaign."""
        self.logger.info("Planning campaign execution strategy")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.PLANNING.value
        
        try:
            # Analyze campaign type and requirements
            campaign_type = CampaignType(state["campaign_type"])
            
            # Create execution plan based on campaign type
            if campaign_type == CampaignType.BLOG_CREATION:
                execution_plan = await self._plan_blog_creation_campaign(state)
            elif campaign_type == CampaignType.CONTENT_REPURPOSING:
                execution_plan = await self._plan_content_repurposing_campaign(state)
            elif campaign_type == CampaignType.SOCIAL_MEDIA_CAMPAIGN:
                execution_plan = await self._plan_social_media_campaign(state)
            else:
                execution_plan = await self._plan_default_campaign(state)
            
            # Update state with execution plan
            state["campaign_config"]["execution_plan"] = execution_plan
            state["agent_assignments"] = execution_plan.get("agent_assignments", {})
            
            # Estimate completion time
            estimated_duration = execution_plan.get("estimated_duration_minutes", 60)
            estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_duration)
            state["estimated_completion"] = estimated_completion.isoformat()
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["planning"] = execution_time
            state["completed_phases"].append("planning")
            state["current_phase"] = CampaignPhase.CONTENT_CREATION.value
            state["overall_progress"] = 20.0  # 20% for planning
            
            # Checkpoint
            state["phase_checkpoints"]["planning"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "execution_plan": execution_plan,
                "agent_assignments": len(state["agent_assignments"]),
                "estimated_completion": state["estimated_completion"]
            }
            
            self.logger.info(f"Campaign planning completed: {len(state['agent_assignments'])} agents assigned")
            
        except Exception as e:
            self.logger.error(f"Campaign planning failed: {e}")
            state["error_state"] = f"Planning failed: {str(e)}"
            state["failed_phases"].append("planning")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_content_creation(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Execute the content creation phase of the campaign."""
        self.logger.info("Executing content creation phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.CONTENT_CREATION.value
        
        try:
            campaign_type = CampaignType(state["campaign_type"])
            
            if campaign_type == CampaignType.BLOG_CREATION:
                # Execute blog creation workflow
                blog_input = {
                    "blog_title": state["campaign_config"].get("campaign_title", ""),
                    "company_context": state["campaign_config"].get("company_context", ""),
                    "content_type": "blog",
                    "target_audience": state["campaign_config"].get("target_audience", "")
                }
                
                blog_result = await self.blog_orchestrator.generate_blog(blog_input)
                
                if blog_result["success"]:
                    state["content_artifacts"]["primary_blog"] = {
                        "content": blog_result["final_post"],
                        "metadata": blog_result["content_metadata"],
                        "quality_scores": blog_result["workflow_metadata"]["quality_scores"]
                    }
                    
                    # Update quality tracking
                    quality_scores = blog_result["workflow_metadata"]["quality_scores"]
                    state["content_quality_scores"]["blog_content"] = quality_scores.get("content", 0)
                    state["content_quality_scores"]["seo"] = quality_scores.get("seo", 0)
                    
                    self.logger.info("Blog creation completed successfully")
                else:
                    raise Exception(f"Blog creation failed: {blog_result.get('error_message', 'Unknown error')}")
            
            else:
                # Handle other campaign types with task-based execution
                content_tasks = [task for task in state["all_tasks"] if task.get("category") == "content_creation"]
                results = await self._execute_tasks_parallel(content_tasks, state)
                
                # Process results
                for task_id, result in results.items():
                    if result.success:
                        state["content_artifacts"][task_id] = result.data
                    else:
                        state["failed_tasks"].append(task_id)
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["content_creation"] = execution_time
            state["completed_phases"].append("content_creation")
            state["current_phase"] = CampaignPhase.CONTENT_OPTIMIZATION.value
            state["overall_progress"] = 40.0  # 40% for content creation
            
            # Checkpoint
            state["phase_checkpoints"]["content_creation"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "artifacts_created": len(state["content_artifacts"]),
                "quality_scores": state["content_quality_scores"],
                "execution_time_ms": execution_time
            }
            
            self.logger.info(f"Content creation completed: {len(state['content_artifacts'])} artifacts created")
            
        except Exception as e:
            self.logger.error(f"Content creation failed: {e}")
            state["error_state"] = f"Content creation failed: {str(e)}"
            state["failed_phases"].append("content_creation")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_content_optimization(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Execute content optimization phase with SEO and quality improvements."""
        self.logger.info("Executing content optimization phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.CONTENT_OPTIMIZATION.value
        
        try:
            optimization_results = {}
            
            # SEO optimization for each content artifact
            for artifact_id, artifact_data in state["content_artifacts"].items():
                if isinstance(artifact_data, dict) and "content" in artifact_data:
                    seo_input = {
                        "content": artifact_data["content"],
                        "title": state["campaign_config"].get("campaign_title", ""),
                        "target_keywords": state["campaign_config"].get("target_keywords", [])
                    }
                    
                    seo_result = self.seo_agent.execute(seo_input)
                    
                    if seo_result.success:
                        optimization_results[artifact_id] = {
                            "seo_score": seo_result.data.get("seo_score", 0),
                            "optimization_suggestions": seo_result.data.get("suggestions", []),
                            "optimized_content": seo_result.data.get("optimized_content", artifact_data["content"])
                        }
                        
                        # Update content with optimized version
                        artifact_data["optimized_content"] = optimization_results[artifact_id]["optimized_content"]
                        state["content_quality_scores"][f"{artifact_id}_seo"] = optimization_results[artifact_id]["seo_score"]
            
            # Store optimization results
            state["content_artifacts"]["optimization_results"] = optimization_results
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["content_optimization"] = execution_time
            state["completed_phases"].append("content_optimization")
            state["current_phase"] = CampaignPhase.CONTENT_REPURPOSING.value
            state["overall_progress"] = 55.0  # 55% for optimization
            
            # Checkpoint
            state["phase_checkpoints"]["content_optimization"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "artifacts_optimized": len(optimization_results),
                "avg_seo_score": sum(r.get("seo_score", 0) for r in optimization_results.values()) / len(optimization_results) if optimization_results else 0,
                "execution_time_ms": execution_time
            }
            
            self.logger.info(f"Content optimization completed: {len(optimization_results)} artifacts optimized")
            
        except Exception as e:
            self.logger.error(f"Content optimization failed: {e}")
            state["error_state"] = f"Content optimization failed: {str(e)}"
            state["failed_phases"].append("content_optimization")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_content_repurposing(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Execute content repurposing phase for multi-channel distribution."""
        self.logger.info("Executing content repurposing phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.CONTENT_REPURPOSING.value
        
        try:
            repurposing_results = {}
            
            # Get primary content for repurposing
            primary_content = None
            if "primary_blog" in state["content_artifacts"]:
                primary_content = state["content_artifacts"]["primary_blog"]["content"]
            elif state["content_artifacts"]:
                # Use first available content artifact
                first_artifact = next(iter(state["content_artifacts"].values()))
                if isinstance(first_artifact, dict) and "content" in first_artifact:
                    primary_content = first_artifact["content"]
            
            if primary_content:
                # Repurpose for different channels
                target_formats = ["linkedin", "twitter", "facebook", "newsletter"]
                
                for format_type in target_formats:
                    if format_type in state.get("distribution_channels", target_formats):
                        repurpose_input = {
                            "source_content": primary_content,
                            "target_format": format_type,
                            "company_context": state["campaign_config"].get("company_context", ""),
                            "maintain_key_message": True
                        }
                        
                        repurpose_result = self.content_repurposer.execute(repurpose_input)
                        
                        if repurpose_result.success:
                            repurposing_results[format_type] = {
                                "content": repurpose_result.data.get("repurposed_content", ""),
                                "format": format_type,
                                "word_count": repurpose_result.data.get("word_count", 0),
                                "adaptation_notes": repurpose_result.data.get("adaptation_notes", [])
                            }
                
                # Social media specific repurposing
                if repurposing_results:
                    social_input = {
                        "content_variants": repurposing_results,
                        "campaign_context": state["campaign_config"]
                    }
                    
                    social_result = self.social_media_agent.execute(social_input)
                    
                    if social_result.success:
                        repurposing_results["social_optimized"] = social_result.data
            
            # Store repurposing results
            state["content_artifacts"]["repurposed_content"] = repurposing_results
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["content_repurposing"] = execution_time
            state["completed_phases"].append("content_repurposing")
            state["current_phase"] = CampaignPhase.QUALITY_ASSURANCE.value
            state["overall_progress"] = 70.0  # 70% for repurposing
            
            # Checkpoint
            state["phase_checkpoints"]["content_repurposing"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "formats_created": len(repurposing_results),
                "channels_covered": list(repurposing_results.keys()),
                "execution_time_ms": execution_time
            }
            
            self.logger.info(f"Content repurposing completed: {len(repurposing_results)} formats created")
            
        except Exception as e:
            self.logger.error(f"Content repurposing failed: {e}")
            state["error_state"] = f"Content repurposing failed: {str(e)}"
            state["failed_phases"].append("content_repurposing")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_quality_assurance(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Execute comprehensive quality assurance across all campaign deliverables."""
        self.logger.info("Executing quality assurance phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.QUALITY_ASSURANCE.value
        
        try:
            qa_results = {
                "quality_gates": [],
                "quality_scores": {},
                "recommendations": []
            }
            
            # Content quality assessment
            content_quality_passed = True
            if state["content_quality_scores"]:
                avg_content_quality = sum(state["content_quality_scores"].values()) / len(state["content_quality_scores"])
                qa_results["quality_scores"]["average_content_quality"] = avg_content_quality
                
                if avg_content_quality >= self.quality_thresholds["min_content_quality"]:
                    state["quality_gates_passed"].append("content_quality")
                    qa_results["quality_gates"].append("content_quality: PASSED")
                else:
                    state["quality_gates_failed"].append("content_quality")
                    qa_results["quality_gates"].append("content_quality: FAILED")
                    qa_results["recommendations"].append(f"Content quality {avg_content_quality:.2f} below threshold {self.quality_thresholds['min_content_quality']}")
                    content_quality_passed = False
            
            # SEO quality assessment
            seo_scores = [score for key, score in state["content_quality_scores"].items() if "seo" in key.lower()]
            if seo_scores:
                avg_seo_score = sum(seo_scores) / len(seo_scores)
                qa_results["quality_scores"]["average_seo_score"] = avg_seo_score
                
                if avg_seo_score >= self.quality_thresholds["min_seo_score"]:
                    state["quality_gates_passed"].append("seo_quality")
                    qa_results["quality_gates"].append("seo_quality: PASSED")
                else:
                    state["quality_gates_failed"].append("seo_quality")
                    qa_results["quality_gates"].append("seo_quality: FAILED")
                    qa_results["recommendations"].append(f"SEO score {avg_seo_score:.2f} below threshold {self.quality_thresholds['min_seo_score']}")
            
            # Content completeness assessment
            required_artifacts = self._get_required_artifacts_for_campaign(state["campaign_type"])
            missing_artifacts = [artifact for artifact in required_artifacts if artifact not in state["content_artifacts"]]
            
            if not missing_artifacts:
                state["quality_gates_passed"].append("content_completeness")
                qa_results["quality_gates"].append("content_completeness: PASSED")
            else:
                state["quality_gates_failed"].append("content_completeness")
                qa_results["quality_gates"].append("content_completeness: FAILED")
                qa_results["recommendations"].append(f"Missing required artifacts: {', '.join(missing_artifacts)}")
            
            # Overall quality assessment
            total_gates = len(state["quality_gates_passed"]) + len(state["quality_gates_failed"])
            quality_pass_rate = len(state["quality_gates_passed"]) / total_gates if total_gates > 0 else 0
            qa_results["quality_scores"]["overall_pass_rate"] = quality_pass_rate
            
            # Store QA results
            state["content_artifacts"]["qa_results"] = qa_results
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["quality_assurance"] = execution_time
            state["completed_phases"].append("quality_assurance")
            state["current_phase"] = CampaignPhase.DISTRIBUTION.value
            state["overall_progress"] = 80.0  # 80% for QA
            
            # Checkpoint
            state["phase_checkpoints"]["quality_assurance"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "gates_passed": len(state["quality_gates_passed"]),
                "gates_failed": len(state["quality_gates_failed"]),
                "quality_pass_rate": quality_pass_rate,
                "execution_time_ms": execution_time
            }
            
            self.logger.info(f"Quality assurance completed: {len(state['quality_gates_passed'])}/{total_gates} gates passed")
            
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {e}")
            state["error_state"] = f"Quality assurance failed: {str(e)}"
            state["failed_phases"].append("quality_assurance")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _execute_distribution(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Execute content distribution across specified channels."""
        self.logger.info("Executing distribution phase")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.DISTRIBUTION.value
        
        try:
            distribution_results = {}
            
            # Get distribution-ready content
            repurposed_content = state["content_artifacts"].get("repurposed_content", {})
            
            if repurposed_content and state["distribution_channels"]:
                distribution_input = {
                    "content_variants": repurposed_content,
                    "distribution_channels": state["distribution_channels"],
                    "campaign_config": state["campaign_config"]
                }
                
                distribution_result = self.distribution_agent.execute(distribution_input)
                
                if distribution_result.success:
                    distribution_results = distribution_result.data
                    
                    # Calculate distribution success rate
                    successful_distributions = sum(1 for result in distribution_results.values() if result.get("success", False))
                    total_distributions = len(distribution_results)
                    success_rate = successful_distributions / total_distributions if total_distributions > 0 else 0
                    
                    # Check distribution quality gate
                    if success_rate >= self.quality_thresholds["min_distribution_success"]:
                        state["quality_gates_passed"].append("distribution_success")
                    else:
                        state["quality_gates_failed"].append("distribution_success")
                
                else:
                    distribution_results["error"] = distribution_result.error_message
            
            # Store distribution results
            state["content_artifacts"]["distribution_results"] = distribution_results
            
            # Phase completion
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            state["phase_execution_times"]["distribution"] = execution_time
            state["completed_phases"].append("distribution")
            state["current_phase"] = CampaignPhase.MONITORING.value
            state["overall_progress"] = 90.0  # 90% for distribution
            
            # Checkpoint
            state["phase_checkpoints"]["distribution"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "channels_targeted": len(state["distribution_channels"]),
                "distributions_attempted": len(distribution_results),
                "execution_time_ms": execution_time
            }
            
            self.logger.info(f"Distribution completed: {len(distribution_results)} channel distributions")
            
        except Exception as e:
            self.logger.error(f"Distribution failed: {e}")
            state["error_state"] = f"Distribution failed: {str(e)}"
            state["failed_phases"].append("distribution")
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _monitor_campaign_progress(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Monitor campaign progress and determine next actions."""
        self.logger.info("Monitoring campaign progress")
        
        state["current_phase"] = CampaignPhase.MONITORING.value
        
        # Update monitoring metrics
        state["monitoring_metrics"].update({
            "phases_completed": len(state["completed_phases"]),
            "phases_failed": len(state["failed_phases"]),
            "quality_gates_passed": len(state["quality_gates_passed"]),
            "quality_gates_failed": len(state["quality_gates_failed"]),
            "overall_progress": state["overall_progress"],
            "current_timestamp": datetime.utcnow().isoformat()
        })
        
        # Determine campaign health
        total_phases = len(state["completed_phases"]) + len(state["failed_phases"])
        success_rate = len(state["completed_phases"]) / total_phases if total_phases > 0 else 0
        
        # Check if recovery is needed
        recovery_needed = (
            len(state["failed_phases"]) > 0 and 
            state["recovery_attempts"] < state["max_recovery_attempts"] and
            success_rate < 0.8
        )
        
        if recovery_needed:
            self.logger.warning(f"Campaign recovery needed: {len(state['failed_phases'])} failed phases")
        else:
            state["overall_progress"] = 95.0  # 95% for monitoring
            state["current_phase"] = CampaignPhase.COMPLETION.value
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _handle_recovery(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Handle campaign recovery for failed phases."""
        self.logger.info(f"Handling campaign recovery (attempt {state['recovery_attempts'] + 1})")
        
        state["recovery_attempts"] += 1
        
        try:
            # Analyze failed phases and create recovery plan
            failed_phases = state["failed_phases"]
            recovery_plan = {}
            
            for phase in failed_phases:
                if phase == "content_creation":
                    recovery_plan[phase] = "retry_with_fallback_agent"
                elif phase == "content_optimization":
                    recovery_plan[phase] = "skip_advanced_optimization"
                elif phase == "distribution":
                    recovery_plan[phase] = "retry_with_reduced_channels"
                else:
                    recovery_plan[phase] = "retry_with_increased_timeout"
            
            # Execute recovery actions
            for phase, recovery_action in recovery_plan.items():
                self.logger.info(f"Executing recovery for {phase}: {recovery_action}")
                
                # Remove from failed phases for retry
                if phase in state["failed_phases"]:
                    state["failed_phases"].remove(phase)
                
                # Add recovery strategy
                state["retry_strategies"][phase] = recovery_action
            
            # Reset error state to allow retry
            state["error_state"] = None
            
            self.logger.info(f"Recovery plan executed for {len(recovery_plan)} phases")
            
        except Exception as e:
            self.logger.error(f"Recovery handling failed: {e}")
            state["error_state"] = f"Recovery failed: {str(e)}"
        
        state["updated_at"] = datetime.utcnow().isoformat()
        return state
    
    async def _finalize_campaign(self, state: CampaignOrchestratorState) -> CampaignOrchestratorState:
        """Finalize the campaign orchestration and prepare deliverables."""
        self.logger.info("Finalizing campaign orchestration")
        
        start_time = datetime.utcnow()
        state["current_phase"] = CampaignPhase.COMPLETION.value
        
        # Calculate final metrics
        started_at = datetime.fromisoformat(state["started_at"])
        completed_at = datetime.utcnow()
        total_execution_time = (completed_at - started_at).total_seconds() * 1000
        
        state["total_execution_time_ms"] = total_execution_time
        state["completed_at"] = completed_at.isoformat()
        
        # Assess success criteria
        total_phases = len(state["completed_phases"]) + len(state["failed_phases"])
        success_rate = len(state["completed_phases"]) / total_phases if total_phases > 0 else 0
        quality_success_rate = len(state["quality_gates_passed"]) / (len(state["quality_gates_passed"]) + len(state["quality_gates_failed"])) if (len(state["quality_gates_passed"]) + len(state["quality_gates_failed"])) > 0 else 1.0
        
        state["success_criteria_met"] = success_rate >= 0.8 and quality_success_rate >= 0.7
        
        # Prepare campaign deliverables
        state["campaign_deliverables"] = {
            "content_artifacts": state["content_artifacts"],
            "quality_reports": {
                "quality_gates_passed": state["quality_gates_passed"],
                "quality_gates_failed": state["quality_gates_failed"],
                "quality_scores": state["content_quality_scores"]
            },
            "distribution_summary": state["content_artifacts"].get("distribution_results", {}),
            "execution_summary": {
                "phases_completed": state["completed_phases"],
                "phases_failed": state["failed_phases"],
                "execution_time_ms": total_execution_time,
                "success_rate": success_rate
            }
        }
        
        # Final campaign metrics
        state["campaign_metrics"] = {
            "overall_success_rate": success_rate,
            "quality_success_rate": quality_success_rate,
            "execution_efficiency": state["overall_progress"] / 100.0,
            "recovery_attempts": state["recovery_attempts"],
            "total_artifacts_created": len(state["content_artifacts"]),
            "distribution_channels_covered": len(state["distribution_channels"]),
            "final_assessment": "SUCCESS" if state["success_criteria_met"] else "PARTIAL_SUCCESS"
        }
        
        # Final checkpoint
        state["phase_checkpoints"]["completion"] = {
            "timestamp": completed_at.isoformat(),
            "success_criteria_met": state["success_criteria_met"],
            "total_execution_time_ms": total_execution_time,
            "final_metrics": state["campaign_metrics"]
        }
        
        state["completed_phases"].append("completion")
        state["overall_progress"] = 100.0
        
        self.logger.info(f"Campaign orchestration completed: Success={state['success_criteria_met']}, "
                        f"Execution time={total_execution_time:.2f}ms, "
                        f"Success rate={success_rate:.2%}")
        
        state["updated_at"] = completed_at.isoformat()
        return state
    
    # Helper methods
    
    async def _load_campaign_data(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Load campaign data from database."""
        try:
            campaign = await self.db_service.get_campaign_with_tasks(campaign_id)
            if campaign:
                return {
                    "title": campaign.title,
                    "description": campaign.description,
                    "campaign_type": campaign.campaign_type,
                    "tasks": [task.__dict__ for task in campaign.tasks] if campaign.tasks else [],
                    "target_audience": getattr(campaign, "target_audience", ""),
                    "success_criteria": getattr(campaign, "success_criteria", {})
                }
        except Exception as e:
            self.logger.error(f"Failed to load campaign data: {e}")
        
        return None
    
    def _build_dependency_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build task dependency graph."""
        dependency_graph = {}
        
        for task in tasks:
            task_id = task.get("id", "")
            dependencies = task.get("dependencies", [])
            dependency_graph[task_id] = dependencies
        
        return dependency_graph
    
    def _get_available_agents(self) -> List[str]:
        """Get list of available agent types."""
        return [
            "planner", "researcher", "writer", "editor", "geo", "image", 
            "content_repurposer", "social_media", "distribution", "seo"
        ]
    
    def _estimate_campaign_duration(self, tasks: List[Dict[str, Any]]) -> int:
        """Estimate campaign duration in minutes."""
        base_duration = 30  # Base 30 minutes
        task_duration = len(tasks) * 10  # 10 minutes per task
        complexity_factor = 1.5 if len(tasks) > 10 else 1.0
        
        return int((base_duration + task_duration) * complexity_factor)
    
    def _get_required_artifacts_for_campaign(self, campaign_type: str) -> List[str]:
        """Get required artifacts for campaign type."""
        if campaign_type == "BLOG_CREATION":
            return ["primary_blog", "optimization_results"]
        elif campaign_type == "CONTENT_REPURPOSING":
            return ["repurposed_content", "social_optimized"]
        elif campaign_type == "SOCIAL_MEDIA_CAMPAIGN":
            return ["social_optimized", "distribution_results"]
        else:
            return ["primary_content"]
    
    async def _execute_tasks_parallel(self, tasks: List[Dict[str, Any]], state: CampaignOrchestratorState) -> Dict[str, CampaignAgentResult]:
        """Execute tasks in parallel with coordination."""
        results = {}
        
        # Limit concurrent tasks
        semaphore = asyncio.Semaphore(state["max_parallel_tasks"])
        
        async def execute_single_task(task):
            async with semaphore:
                task_id = task.get("id", "")
                agent_type = task.get("agent_type", "")
                
                try:
                    agent = self.agent_factory.create_agent(agent_type)
                    result = agent.execute(task.get("input_data", {}))
                    
                    return task_id, CampaignAgentResult(
                        agent_name=agent_type,
                        task_id=task_id,
                        success=result.success,
                        execution_time_ms=getattr(result, "execution_time_ms", 0),
                        data=result.data if result.success else {},
                        error_message=result.error_message if not result.success else None
                    )
                except Exception as e:
                    return task_id, CampaignAgentResult(
                        agent_name=agent_type,
                        task_id=task_id,
                        success=False,
                        execution_time_ms=0,
                        error_message=str(e)
                    )
        
        # Execute all tasks
        task_futures = [execute_single_task(task) for task in tasks]
        task_results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        # Process results
        for result in task_results:
            if isinstance(result, tuple):
                task_id, agent_result = result
                results[task_id] = agent_result
        
        return results
    
    # Campaign-specific planning methods
    
    async def _plan_blog_creation_campaign(self, state: CampaignOrchestratorState) -> Dict[str, Any]:
        """Plan blog creation campaign execution."""
        return {
            "campaign_type": "blog_creation",
            "agent_assignments": {
                "blog_orchestrator": "primary_workflow"
            },
            "estimated_duration_minutes": 45,
            "execution_strategy": "sequential_workflow",
            "quality_gates": ["content_quality", "seo_quality", "readability"]
        }
    
    async def _plan_content_repurposing_campaign(self, state: CampaignOrchestratorState) -> Dict[str, Any]:
        """Plan content repurposing campaign execution."""
        return {
            "campaign_type": "content_repurposing",
            "agent_assignments": {
                "content_repurposer": "format_adaptation",
                "social_media_agent": "social_optimization"
            },
            "estimated_duration_minutes": 30,
            "execution_strategy": "parallel_with_dependencies",
            "quality_gates": ["format_compliance", "message_consistency"]
        }
    
    async def _plan_social_media_campaign(self, state: CampaignOrchestratorState) -> Dict[str, Any]:
        """Plan social media campaign execution."""
        return {
            "campaign_type": "social_media_campaign",
            "agent_assignments": {
                "social_media_agent": "content_creation",
                "distribution_agent": "channel_distribution"
            },
            "estimated_duration_minutes": 25,
            "execution_strategy": "parallel_execution",
            "quality_gates": ["engagement_potential", "platform_compliance"]
        }
    
    async def _plan_default_campaign(self, state: CampaignOrchestratorState) -> Dict[str, Any]:
        """Plan default campaign execution."""
        return {
            "campaign_type": "default",
            "agent_assignments": {},
            "estimated_duration_minutes": 60,
            "execution_strategy": "task_based_execution",
            "quality_gates": ["basic_quality"]
        }
    
    # Conditional logic
    
    def _should_recover_or_complete(self, state: CampaignOrchestratorState) -> str:
        """Determine if campaign should recover or complete."""
        failed_phases = len(state["failed_phases"])
        recovery_attempts = state["recovery_attempts"]
        max_attempts = state["max_recovery_attempts"]
        
        if failed_phases > 0 and recovery_attempts < max_attempts:
            return "recover"
        return "complete"


class CampaignOrchestratorAdapter:
    """
    Adapter class to integrate CampaignOrchestratorLangGraph with existing interfaces.
    """
    
    def __init__(self):
        self.langgraph_orchestrator = CampaignOrchestratorLangGraph()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def orchestrate_campaign(self, campaign_id: str, campaign_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrate a campaign using the enhanced LangGraph workflow.
        
        Args:
            campaign_id: ID of the campaign to orchestrate
            campaign_config: Additional campaign configuration
            
        Returns:
            Dict: Campaign orchestration results
        """
        try:
            input_data = {
                "campaign_id": campaign_id,
                "campaign_config": campaign_config or {}
            }
            
            result = await self.langgraph_orchestrator.execute(input_data)
            
            if result.success:
                final_state = result.data
                
                return {
                    "success": True,
                    "campaign_id": final_state.get("campaign_id"),
                    "workflow_execution_id": final_state.get("workflow_execution_id"),
                    "success_criteria_met": final_state.get("success_criteria_met", False),
                    "campaign_deliverables": final_state.get("campaign_deliverables", {}),
                    "campaign_metrics": final_state.get("campaign_metrics", {}),
                    "execution_metadata": {
                        "total_execution_time_ms": final_state.get("total_execution_time_ms", 0),
                        "phases_completed": final_state.get("completed_phases", []),
                        "phases_failed": final_state.get("failed_phases", []),
                        "recovery_attempts": final_state.get("recovery_attempts", 0),
                        "quality_gates_passed": final_state.get("quality_gates_passed", []),
                        "overall_progress": final_state.get("overall_progress", 0)
                    }
                }
            else:
                return {
                    "success": False,
                    "error_message": result.error_message,
                    "error_code": result.error_code
                }
                
        except Exception as e:
            self.logger.error(f"Campaign orchestration failed: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "error_code": "ORCHESTRATION_EXECUTION_FAILED"
            }
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the LangGraph orchestrator."""
        return self.langgraph_orchestrator.get_workflow_info()


# Export key components
__all__ = [
    'CampaignOrchestratorLangGraph',
    'CampaignOrchestratorAdapter',
    'CampaignOrchestratorState',
    'CampaignPhase',
    'TaskExecutionMode'
]