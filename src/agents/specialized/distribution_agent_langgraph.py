"""
LangGraph-enhanced Distribution Workflow for intelligent multi-channel content distribution.
"""

import json
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
from .distribution_agent import (
    DistributionAgent, PlatformConfig, PublishedPost
)
from ...config.database import DatabaseConnection


class DistributionMode(str, Enum):
    """Distribution modes for content delivery."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    BATCH = "batch"
    DRIP_CAMPAIGN = "drip_campaign"


class PlatformStatus(str, Enum):
    """Platform availability status."""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    DISABLED = "disabled"


class PublicationStatus(str, Enum):
    """Publication status for individual posts."""
    PENDING = "pending"
    QUEUED = "queued"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DistributionTarget:
    """Distribution target configuration."""
    platform: str
    content: str
    scheduled_time: Optional[datetime] = None
    platform_config: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    status: PublicationStatus = PublicationStatus.PENDING
    error_message: Optional[str] = None


@dataclass 
class EngagementMetrics:
    """Engagement metrics for published content."""
    platform: str
    post_id: str
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    clicks: int = 0
    saves: int = 0
    reach: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
    last_updated: Optional[datetime] = None


class DistributionState(WorkflowState):
    """Enhanced state for content distribution workflow."""
    
    # Input configuration
    distribution_mode: DistributionMode = DistributionMode.IMMEDIATE
    distribution_targets: List[DistributionTarget] = field(default_factory=list)
    campaign_id: Optional[str] = None
    content_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Platform status tracking
    platform_status: Dict[str, PlatformStatus] = field(default_factory=dict)
    platform_configs: Dict[str, PlatformConfig] = field(default_factory=dict)
    
    # Distribution results
    published_posts: List[PublishedPost] = field(default_factory=list)
    failed_distributions: List[DistributionTarget] = field(default_factory=list)
    retry_queue: List[DistributionTarget] = field(default_factory=list)
    
    # Engagement tracking
    engagement_metrics: Dict[str, EngagementMetrics] = field(default_factory=dict)
    tracking_schedule: Dict[str, List[datetime]] = field(default_factory=dict)
    
    # Performance metrics
    distribution_metrics: Dict[str, Any] = field(default_factory=dict)
    success_rates: Dict[str, float] = field(default_factory=dict)
    average_processing_time: Dict[str, float] = field(default_factory=dict)
    
    # Quality control
    content_validations: Dict[str, bool] = field(default_factory=dict)
    platform_compliance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class DistributionAgentWorkflow(LangGraphWorkflowBase[DistributionState]):
    """LangGraph workflow for intelligent multi-channel content distribution."""
    
    def __init__(
        self, 
        workflow_name: str = "distribution_agent_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = DistributionAgent()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> DistributionState:
        """Create initial workflow state from context."""
        distribution_targets = []
        
        # Parse distribution targets from context
        targets_data = context.get("distribution_targets", [])
        for target_data in targets_data:
            target = DistributionTarget(
                platform=target_data.get("platform"),
                content=target_data.get("content", ""),
                scheduled_time=target_data.get("scheduled_time"),
                platform_config=target_data.get("platform_config"),
                max_retries=target_data.get("max_retries", 3)
            )
            distribution_targets.append(target)
        
        return DistributionState(
            workflow_id=context.get("workflow_id", f"dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            distribution_mode=DistributionMode(context.get("distribution_mode", "immediate")),
            distribution_targets=distribution_targets,
            campaign_id=context.get("campaign_id"),
            content_metadata=context.get("content_metadata", {}),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the content distribution workflow graph."""
        workflow = StateGraph(DistributionState)
        
        # Define workflow nodes
        workflow.add_node("validate_targets", self._validate_targets_node)
        workflow.add_node("check_platform_status", self._check_platform_status_node)
        workflow.add_node("validate_content", self._validate_content_node)
        workflow.add_node("schedule_distribution", self._schedule_distribution_node)
        workflow.add_node("execute_distribution", self._execute_distribution_node)
        workflow.add_node("handle_retries", self._handle_retries_node)
        workflow.add_node("setup_tracking", self._setup_tracking_node)
        workflow.add_node("monitor_engagement", self._monitor_engagement_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Define workflow edges
        workflow.add_edge("validate_targets", "check_platform_status")
        workflow.add_edge("check_platform_status", "validate_content")
        workflow.add_edge("validate_content", "schedule_distribution")
        workflow.add_edge("schedule_distribution", "execute_distribution")
        
        # Conditional routing for retries
        workflow.add_conditional_edges(
            "execute_distribution",
            self._should_handle_retries,
            {
                "handle_retries": "handle_retries",
                "setup_tracking": "setup_tracking"
            }
        )
        workflow.add_edge("handle_retries", "execute_distribution")  # Retry loop
        workflow.add_edge("setup_tracking", "monitor_engagement")
        workflow.add_edge("monitor_engagement", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        # Set entry point
        workflow.set_entry_point("validate_targets")
        
        return workflow
    
    async def _validate_targets_node(self, state: DistributionState) -> DistributionState:
        """Validate distribution targets and configuration."""
        try:
            self._log_progress("Validating distribution targets and configuration")
            
            validation_errors = []
            
            # Validate distribution targets
            if not state.distribution_targets:
                validation_errors.append("No distribution targets specified")
            
            valid_targets = []
            for target in state.distribution_targets:
                # Validate platform
                if not target.platform:
                    validation_errors.append("Distribution target missing platform")
                    continue
                
                # Validate content
                if not target.content or len(target.content.strip()) < 10:
                    validation_errors.append(f"Invalid content for platform {target.platform}")
                    continue
                
                # Validate scheduled time for scheduled mode
                if state.distribution_mode == DistributionMode.SCHEDULED and not target.scheduled_time:
                    validation_errors.append(f"Scheduled time required for platform {target.platform}")
                    continue
                
                valid_targets.append(target)
            
            state.distribution_targets = valid_targets
            
            # Validate distribution mode
            if state.distribution_mode == DistributionMode.DRIP_CAMPAIGN:
                if len(state.distribution_targets) < 2:
                    validation_errors.append("Drip campaign requires at least 2 distribution targets")
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 15.0
                
                state.messages.append(HumanMessage(
                    content=f"Validated {len(valid_targets)} distribution targets for {state.distribution_mode} mode."
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Target validation failed: {str(e)}"
            return state
    
    async def _check_platform_status_node(self, state: DistributionState) -> DistributionState:
        """Check status and availability of target platforms."""
        try:
            self._log_progress("Checking platform status and availability")
            
            platform_status = {}
            platform_configs = {}
            
            # Get unique platforms from targets
            platforms = list(set(target.platform for target in state.distribution_targets))
            
            for platform in platforms:
                try:
                    # Check if platform is configured in legacy agent
                    platform_config = self.legacy_agent.platform_configs.get(platform)
                    
                    if not platform_config:
                        platform_status[platform] = PlatformStatus.ERROR
                        self._log_error(f"Platform {platform} not configured")
                        continue
                    
                    # Check API availability
                    if not platform_config.api_enabled:
                        platform_status[platform] = PlatformStatus.DISABLED
                        self._log_progress(f"Platform {platform} API disabled, will simulate")
                    else:
                        # Simulate API health check
                        is_healthy = await self._check_platform_health(platform, platform_config)
                        platform_status[platform] = PlatformStatus.ACTIVE if is_healthy else PlatformStatus.ERROR
                    
                    platform_configs[platform] = platform_config
                    
                except Exception as platform_error:
                    platform_status[platform] = PlatformStatus.ERROR
                    self._log_error(f"Platform {platform} status check failed: {str(platform_error)}")
            
            state.platform_status = platform_status
            state.platform_configs = platform_configs
            state.progress_percentage = 25.0
            
            active_platforms = sum(1 for status in platform_status.values() if status == PlatformStatus.ACTIVE)
            state.messages.append(SystemMessage(
                content=f"Platform status check completed. {active_platforms}/{len(platforms)} platforms active."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Platform status check failed: {str(e)}"
            return state
    
    async def _validate_content_node(self, state: DistributionState) -> DistributionState:
        """Validate content for each platform's requirements."""
        try:
            self._log_progress("Validating content for platform requirements")
            
            content_validations = {}
            platform_compliance = {}
            
            for target in state.distribution_targets:
                platform = target.platform
                platform_config = state.platform_configs.get(platform)
                
                if not platform_config:
                    content_validations[f"{platform}_{hash(target.content)}"] = False
                    continue
                
                try:
                    # Use legacy agent's content validation
                    is_valid = self.legacy_agent._validate_content_for_platform(
                        target.content, platform_config
                    )
                    
                    content_validations[f"{platform}_{hash(target.content)}"] = is_valid
                    
                    # Enhanced compliance checking
                    compliance_details = await self._check_detailed_compliance(
                        target.content, platform_config
                    )
                    platform_compliance[f"{platform}_{hash(target.content)}"] = compliance_details
                    
                    if not is_valid:
                        target.status = PublicationStatus.FAILED
                        target.error_message = "Content validation failed"
                        
                except Exception as validation_error:
                    content_validations[f"{platform}_{hash(target.content)}"] = False
                    target.status = PublicationStatus.FAILED
                    target.error_message = f"Validation error: {str(validation_error)}"
            
            state.content_validations = content_validations
            state.platform_compliance = platform_compliance
            state.progress_percentage = 40.0
            
            valid_count = sum(1 for valid in content_validations.values() if valid)
            state.messages.append(SystemMessage(
                content=f"Content validation completed. {valid_count}/{len(content_validations)} contents valid."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content validation failed: {str(e)}"
            return state
    
    async def _schedule_distribution_node(self, state: DistributionState) -> DistributionState:
        """Schedule distribution based on mode and timing requirements."""
        try:
            self._log_progress("Scheduling content distribution")
            
            current_time = datetime.utcnow()
            
            for target in state.distribution_targets:
                # Skip failed validations
                if target.status == PublicationStatus.FAILED:
                    continue
                
                if state.distribution_mode == DistributionMode.IMMEDIATE:
                    target.scheduled_time = current_time
                    target.status = PublicationStatus.QUEUED
                    
                elif state.distribution_mode == DistributionMode.SCHEDULED:
                    if not target.scheduled_time:
                        # Set default schedule if not provided
                        target.scheduled_time = current_time + timedelta(minutes=5)
                    target.status = PublicationStatus.QUEUED
                    
                elif state.distribution_mode == DistributionMode.BATCH:
                    # Schedule all for next batch window (e.g., top of the hour)
                    next_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    target.scheduled_time = next_hour
                    target.status = PublicationStatus.QUEUED
                    
                elif state.distribution_mode == DistributionMode.DRIP_CAMPAIGN:
                    # Stagger distribution across time
                    delay_minutes = state.distribution_targets.index(target) * 30  # 30 min intervals
                    target.scheduled_time = current_time + timedelta(minutes=delay_minutes)
                    target.status = PublicationStatus.QUEUED
            
            state.progress_percentage = 55.0
            
            queued_count = sum(1 for target in state.distribution_targets if target.status == PublicationStatus.QUEUED)
            state.messages.append(SystemMessage(
                content=f"Distribution scheduled for {queued_count} targets in {state.distribution_mode} mode."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Distribution scheduling failed: {str(e)}"
            return state
    
    async def _execute_distribution_node(self, state: DistributionState) -> DistributionState:
        """Execute content distribution to target platforms."""
        try:
            self._log_progress("Executing content distribution")
            
            current_time = datetime.utcnow()
            distribution_metrics = {}
            
            # Process targets ready for distribution
            ready_targets = [
                target for target in state.distribution_targets
                if target.status == PublicationStatus.QUEUED and
                target.scheduled_time <= current_time
            ]
            
            if not ready_targets:
                self._log_progress("No targets ready for immediate distribution")
                state.progress_percentage = 70.0
                return state
            
            published_posts = []
            failed_distributions = []
            
            for target in ready_targets:
                try:
                    target.status = PublicationStatus.PUBLISHING
                    start_time = datetime.utcnow()
                    
                    # Create post data for legacy agent
                    post_data = {
                        "id": str(uuid.uuid4()),
                        "platform": target.platform,
                        "content": target.content,
                        "scheduled_time": target.scheduled_time,
                        "metadata": state.content_metadata
                    }
                    
                    # Use legacy agent for actual publication
                    published_post = await self.legacy_agent._publish_single_post(post_data)
                    
                    if published_post:
                        target.status = PublicationStatus.PUBLISHED
                        published_posts.append(published_post)
                        
                        # Track processing time
                        processing_time = (datetime.utcnow() - start_time).total_seconds()
                        distribution_metrics[target.platform] = distribution_metrics.get(target.platform, [])
                        distribution_metrics[target.platform].append(processing_time)
                        
                    else:
                        target.status = PublicationStatus.FAILED
                        target.error_message = "Publication failed"
                        failed_distributions.append(target)
                        
                except Exception as distribution_error:
                    target.status = PublicationStatus.FAILED
                    target.error_message = str(distribution_error)
                    failed_distributions.append(target)
                    self._log_error(f"Distribution failed for {target.platform}: {str(distribution_error)}")
            
            # Update state with results
            state.published_posts.extend(published_posts)
            state.failed_distributions.extend(failed_distributions)
            state.distribution_metrics = distribution_metrics
            state.progress_percentage = 70.0
            
            # Calculate success rates
            success_rates = {}
            for platform in set(target.platform for target in ready_targets):
                platform_targets = [t for t in ready_targets if t.platform == platform]
                successful = [t for t in platform_targets if t.status == PublicationStatus.PUBLISHED]
                success_rates[platform] = len(successful) / len(platform_targets) if platform_targets else 0
            
            state.success_rates = success_rates
            
            state.messages.append(SystemMessage(
                content=f"Distribution executed. {len(published_posts)} published, {len(failed_distributions)} failed."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Distribution execution failed: {str(e)}"
            return state
    
    async def _handle_retries_node(self, state: DistributionState) -> DistributionState:
        """Handle retries for failed distributions."""
        try:
            self._log_progress("Handling distribution retries")
            
            # Identify targets eligible for retry
            retry_candidates = []
            for target in state.distribution_targets:
                if (target.status == PublicationStatus.FAILED and 
                    target.retry_count < target.max_retries and
                    target.error_message != "Content validation failed"):  # Don't retry validation failures
                    retry_candidates.append(target)
            
            if not retry_candidates:
                self._log_progress("No targets eligible for retry")
                return state
            
            # Prepare retries
            for target in retry_candidates:
                target.retry_count += 1
                target.status = PublicationStatus.RETRYING
                target.scheduled_time = datetime.utcnow() + timedelta(minutes=target.retry_count * 2)  # Exponential backoff
                
                self._log_progress(f"Scheduling retry {target.retry_count}/{target.max_retries} for {target.platform}")
            
            state.retry_queue = retry_candidates
            
            state.messages.append(SystemMessage(
                content=f"Scheduled {len(retry_candidates)} targets for retry with exponential backoff."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Retry handling failed: {str(e)}"
            return state
    
    async def _setup_tracking_node(self, state: DistributionState) -> DistributionState:
        """Set up engagement tracking for published content."""
        try:
            self._log_progress("Setting up engagement tracking")
            
            tracking_schedule = {}
            
            for published_post in state.published_posts:
                platform = published_post.platform
                post_id = published_post.post_id
                
                # Check if platform supports engagement tracking
                platform_config = state.platform_configs.get(platform)
                if not platform_config or not platform_config.engagement_tracking:
                    continue
                
                # Schedule tracking at intervals: 1hr, 24hr, 7days
                base_time = published_post.published_at
                tracking_times = [
                    base_time + timedelta(hours=1),
                    base_time + timedelta(hours=24),
                    base_time + timedelta(days=7)
                ]
                
                tracking_schedule[post_id] = tracking_times
                
                # Initialize engagement metrics
                engagement_metrics = EngagementMetrics(
                    platform=platform,
                    post_id=post_id,
                    last_updated=datetime.utcnow()
                )
                state.engagement_metrics[post_id] = engagement_metrics
            
            state.tracking_schedule = tracking_schedule
            state.progress_percentage = 85.0
            
            state.messages.append(SystemMessage(
                content=f"Engagement tracking setup for {len(tracking_schedule)} published posts."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Tracking setup failed: {str(e)}"
            return state
    
    async def _monitor_engagement_node(self, state: DistributionState) -> DistributionState:
        """Monitor engagement for published content."""
        try:
            self._log_progress("Monitoring engagement for published content")
            
            current_time = datetime.utcnow()
            
            # Check for posts ready for engagement tracking
            for post_id, tracking_times in state.tracking_schedule.items():
                ready_times = [t for t in tracking_times if t <= current_time]
                
                if ready_times and post_id in state.engagement_metrics:
                    try:
                        # Use legacy agent for engagement tracking
                        engagement_data = await self.legacy_agent.track_engagement(post_id)
                        
                        if engagement_data and "engagement_metrics" in engagement_data:
                            metrics = state.engagement_metrics[post_id]
                            
                            # Update metrics from tracking data
                            tracking_metrics = engagement_data["engagement_metrics"]
                            metrics.views = tracking_metrics.get("views", 0)
                            metrics.likes = tracking_metrics.get("likes", 0)
                            metrics.shares = tracking_metrics.get("shares", tracking_metrics.get("retweets", 0))
                            metrics.comments = tracking_metrics.get("comments", tracking_metrics.get("replies", 0))
                            metrics.clicks = tracking_metrics.get("clicks", 0)
                            metrics.saves = tracking_metrics.get("saves", 0)
                            metrics.last_updated = datetime.utcnow()
                            
                            # Calculate engagement rate
                            if metrics.views > 0:
                                total_engagements = metrics.likes + metrics.shares + metrics.comments
                                metrics.engagement_rate = (total_engagements / metrics.views) * 100
                        
                    except Exception as tracking_error:
                        self._log_error(f"Engagement tracking failed for {post_id}: {str(tracking_error)}")
                        continue
            
            state.progress_percentage = 95.0
            
            tracked_posts = len([m for m in state.engagement_metrics.values() if m.last_updated])
            state.messages.append(SystemMessage(
                content=f"Engagement monitoring completed for {tracked_posts} posts."
            ))
            
            return state
            
        except Exception as e:
            # Engagement monitoring failure shouldn't fail the entire workflow
            self._log_error(f"Engagement monitoring failed: {str(e)}")
            state.messages.append(SystemMessage(
                content="Engagement monitoring failed, but distribution was successful."
            ))
            return state
    
    async def _finalize_results_node(self, state: DistributionState) -> DistributionState:
        """Finalize workflow results and prepare summary."""
        try:
            self._log_progress("Finalizing distribution results")
            
            # Calculate overall metrics
            total_targets = len(state.distribution_targets)
            published_count = len(state.published_posts)
            failed_count = len(state.failed_distributions)
            overall_success_rate = (published_count / total_targets * 100) if total_targets > 0 else 0
            
            # Calculate average processing times
            average_processing_time = {}
            for platform, times in state.distribution_metrics.items():
                if times:
                    average_processing_time[platform] = sum(times) / len(times)
            
            state.average_processing_time = average_processing_time
            
            # Generate summary metrics
            summary_metrics = {
                "total_targets": total_targets,
                "published_count": published_count,
                "failed_count": failed_count,
                "overall_success_rate": round(overall_success_rate, 2),
                "platforms_used": list(set(post.platform for post in state.published_posts)),
                "distribution_mode": state.distribution_mode.value,
                "processing_time_seconds": (datetime.utcnow() - state.created_at).total_seconds(),
                "engagement_tracked_posts": len(state.engagement_metrics)
            }
            
            # Determine final workflow status
            if published_count > 0:
                state.status = WorkflowStatus.COMPLETED
            elif published_count == 0 and total_targets > 0:
                state.status = WorkflowStatus.FAILED
                state.error_message = "No content was successfully distributed"
            else:
                state.status = WorkflowStatus.COMPLETED
                
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            # Store final metrics
            state.distribution_metrics["summary"] = summary_metrics
            
            state.messages.append(SystemMessage(
                content=f"Distribution workflow completed. {published_count}/{total_targets} targets successful "
                       f"({overall_success_rate:.1f}% success rate)."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Results finalization failed: {str(e)}"
            return state
    
    def _should_handle_retries(self, state: DistributionState) -> str:
        """Determine if retries should be handled."""
        # Check if there are any failed targets eligible for retry
        retry_eligible = any(
            target.status == PublicationStatus.FAILED and
            target.retry_count < target.max_retries and
            target.error_message != "Content validation failed"
            for target in state.distribution_targets
        )
        return "handle_retries" if retry_eligible else "setup_tracking"
    
    # Helper methods for enhanced functionality
    
    async def _check_platform_health(self, platform: str, platform_config: PlatformConfig) -> bool:
        """Check if a platform's API is healthy."""
        try:
            # Simulate API health check
            await asyncio.sleep(0.1)  # Simulate network call
            
            # For now, return True unless platform is explicitly disabled
            return platform_config.api_enabled
            
        except Exception as e:
            self._log_error(f"Health check failed for {platform}: {str(e)}")
            return False
    
    async def _check_detailed_compliance(
        self, 
        content: str, 
        platform_config: PlatformConfig
    ) -> Dict[str, Any]:
        """Check detailed compliance for platform-specific requirements."""
        try:
            compliance_details = {
                "length_compliant": True,
                "character_count": len(content),
                "word_count": len(content.split()),
                "hashtag_count": content.count('#'),
                "mention_count": content.count('@'),
                "url_count": content.count('http'),
                "compliance_score": 100
            }
            
            rules = platform_config.posting_rules
            
            # Check length compliance
            max_length = rules.get("max_length", 1000)
            if len(content) > max_length:
                compliance_details["length_compliant"] = False
                compliance_details["compliance_score"] -= 30
            
            # Check hashtag limit
            hashtag_limit = rules.get("hashtag_limit", 10)
            if compliance_details["hashtag_count"] > hashtag_limit:
                compliance_details["hashtag_limit_exceeded"] = True
                compliance_details["compliance_score"] -= 15
            
            # Additional platform-specific checks could go here
            
            return compliance_details
            
        except Exception as e:
            self._log_error(f"Compliance check failed: {str(e)}")
            return {"compliance_score": 0, "error": str(e)}
    
    async def execute_workflow(
        self,
        distribution_targets: List[Dict[str, Any]],
        distribution_mode: str = "immediate",
        campaign_id: Optional[str] = None,
        content_metadata: Optional[Dict[str, Any]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the content distribution workflow."""
        
        context = {
            "distribution_targets": distribution_targets,
            "distribution_mode": distribution_mode,
            "campaign_id": campaign_id,
            "content_metadata": content_metadata or {},
            "workflow_id": f"dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "distribution_results": {
                        "published_posts": [
                            {
                                "platform": post.platform,
                                "post_id": post.post_id,
                                "post_url": post.post_url,
                                "published_at": post.published_at.isoformat(),
                                "engagement_metrics": post.engagement_metrics
                            }
                            for post in final_state.published_posts
                        ],
                        "failed_distributions": [
                            {
                                "platform": target.platform,
                                "error_message": target.error_message,
                                "retry_count": target.retry_count
                            }
                            for target in final_state.failed_distributions
                        ]
                    },
                    "engagement_metrics": {
                        post_id: {
                            "platform": metrics.platform,
                            "views": metrics.views,
                            "likes": metrics.likes,
                            "shares": metrics.shares,
                            "comments": metrics.comments,
                            "clicks": metrics.clicks,
                            "saves": metrics.saves,
                            "engagement_rate": metrics.engagement_rate,
                            "last_updated": metrics.last_updated.isoformat() if metrics.last_updated else None
                        }
                        for post_id, metrics in final_state.engagement_metrics.items()
                    },
                    "performance_metrics": {
                        "success_rates": final_state.success_rates,
                        "average_processing_time": final_state.average_processing_time,
                        "distribution_metrics": final_state.distribution_metrics
                    },
                    "workflow_summary": final_state.distribution_metrics.get("summary", {})
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "published_count": len(final_state.published_posts),
                        "failed_count": len(final_state.failed_distributions),
                        "overall_success_rate": final_state.distribution_metrics.get("summary", {}).get("overall_success_rate", 0)
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Distribution workflow failed",
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "final_status": final_state.status.value
                    }
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=f"Workflow execution failed: {str(e)}",
                metadata={"error_type": "workflow_execution_error"}
            )