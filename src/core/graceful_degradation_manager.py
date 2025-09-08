"""
Graceful Degradation Manager (User Story 4.3)

Provides reduced functionality during agent failures, ensuring users can still
get work done even when some system components are unavailable.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ServiceHealthStatus(Enum):
    """Service health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


class DegradationLevel(Enum):
    """System degradation levels."""
    NONE = "none"                    # Full functionality
    MINIMAL = "minimal"              # Minor features disabled
    MODERATE = "moderate"            # Some key features disabled
    SEVERE = "severe"               # Only core functionality available
    EMERGENCY = "emergency"         # Absolute minimum functionality


@dataclass
class ServiceHealth:
    """Health status of a service/agent."""
    service_name: str
    status: ServiceHealthStatus
    last_success: Optional[datetime]
    failure_count: int
    error_message: str
    is_critical: bool
    recovery_estimate: Optional[datetime]


@dataclass
class DegradationStrategy:
    """Strategy for handling service degradation."""
    service_name: str
    fallback_services: List[str]
    skip_if_unavailable: bool
    alternative_workflow: Optional[str]
    user_notification: str
    impact_description: str
    recovery_actions: List[str]


@dataclass
class MinimalViableWorkflow:
    """Definition of minimal viable workflow for a content type."""
    content_type: str
    required_agents: List[str]
    optional_agents: List[str]
    fallback_agents: Dict[str, str]  # agent_name -> fallback_agent_name
    quality_threshold: float
    user_expectations: str
    limitations: List[str]


class GracefulDegradationManager:
    """
    Manages system degradation and provides reduced functionality during failures.
    
    Features:
    - Service health monitoring
    - Automatic degradation detection
    - Minimal viable workflow execution
    - User notification system
    - Auto-recovery monitoring
    - Degradation metrics tracking
    """
    
    def __init__(
        self,
        health_check_interval: int = 60,
        max_failure_threshold: int = 3,
        recovery_timeout: int = 300,
        enable_auto_recovery: bool = True
    ):
        """Initialize graceful degradation manager."""
        self.health_check_interval = health_check_interval
        self.max_failure_threshold = max_failure_threshold
        self.recovery_timeout = recovery_timeout
        self.enable_auto_recovery = enable_auto_recovery
        
        # Service health tracking
        self._service_health: Dict[str, ServiceHealth] = {}
        self._degradation_strategies: Dict[str, DegradationStrategy] = {}
        self._minimal_workflows: Dict[str, MinimalViableWorkflow] = {}
        
        # Current system state
        self._current_degradation_level = DegradationLevel.NONE
        self._degraded_services: Set[str] = set()
        self._active_notifications: Dict[str, datetime] = {}
        
        # Monitoring and recovery
        self._health_check_task = None
        self._recovery_callbacks: Dict[str, Callable] = {}
        
        # Initialize default configurations
        self._initialize_default_strategies()
        self._initialize_minimal_workflows()
        
        # Health monitoring will be started when needed
        
        logger.info("GracefulDegradationManager initialized")
        logger.info(f"Health check interval: {health_check_interval}s")
        logger.info(f"Auto-recovery enabled: {enable_auto_recovery}")
    
    async def register_service(
        self,
        service_name: str,
        is_critical: bool = False,
        health_check_func: Optional[Callable] = None
    ) -> None:
        """
        Register a service for health monitoring.
        
        Args:
            service_name: Name of the service/agent
            is_critical: Whether this service is critical for minimal functionality
            health_check_func: Optional custom health check function
        """
        self._service_health[service_name] = ServiceHealth(
            service_name=service_name,
            status=ServiceHealthStatus.HEALTHY,
            last_success=datetime.now(),
            failure_count=0,
            error_message="",
            is_critical=is_critical,
            recovery_estimate=None
        )
        
        if health_check_func:
            self._recovery_callbacks[service_name] = health_check_func
        
        logger.info(f"Registered service {service_name} (critical: {is_critical})")
    
    async def report_service_failure(
        self,
        service_name: str,
        error_message: str,
        exception: Optional[Exception] = None
    ) -> DegradationLevel:
        """
        Report a service failure and update system degradation level.
        
        Args:
            service_name: Name of the failed service
            error_message: Description of the failure
            exception: Optional exception object
            
        Returns:
            Current system degradation level
        """
        try:
            # Get or create service health record
            if service_name not in self._service_health:
                await self.register_service(service_name)
            
            service_health = self._service_health[service_name]
            
            # Update failure metrics
            service_health.failure_count += 1
            service_health.error_message = error_message
            
            # Determine new service status
            if service_health.failure_count >= self.max_failure_threshold:
                service_health.status = ServiceHealthStatus.UNAVAILABLE
                self._degraded_services.add(service_name)
            elif service_health.failure_count >= 2:
                service_health.status = ServiceHealthStatus.CRITICAL
            else:
                service_health.status = ServiceHealthStatus.DEGRADED
            
            # Set recovery estimate
            service_health.recovery_estimate = datetime.now() + timedelta(seconds=self.recovery_timeout)
            
            # Update system degradation level
            previous_level = self._current_degradation_level
            self._current_degradation_level = self._calculate_system_degradation_level()
            
            # Log the failure
            logger.warning(f"Service failure reported: {service_name}")
            logger.warning(f"Error: {error_message}")
            logger.warning(f"Failure count: {service_health.failure_count}")
            logger.warning(f"Service status: {service_health.status.value}")
            logger.warning(f"System degradation level: {self._current_degradation_level.value}")
            
            # Notify if degradation level changed
            if self._current_degradation_level != previous_level:
                await self._notify_degradation_change(previous_level, self._current_degradation_level)
            
            # Trigger user notification
            await self._notify_service_degradation(service_name, service_health)
            
            return self._current_degradation_level
            
        except Exception as e:
            logger.error(f"Failed to report service failure for {service_name}: {e}")
            return self._current_degradation_level
    
    async def report_service_recovery(self, service_name: str) -> DegradationLevel:
        """
        Report a service recovery and update system state.
        
        Args:
            service_name: Name of the recovered service
            
        Returns:
            Updated system degradation level
        """
        try:
            if service_name not in self._service_health:
                logger.warning(f"Cannot report recovery for unregistered service: {service_name}")
                return self._current_degradation_level
            
            service_health = self._service_health[service_name]
            
            # Update health status
            service_health.status = ServiceHealthStatus.HEALTHY
            service_health.last_success = datetime.now()
            service_health.failure_count = 0
            service_health.error_message = ""
            service_health.recovery_estimate = None
            
            # Remove from degraded services
            self._degraded_services.discard(service_name)
            
            # Update system degradation level
            previous_level = self._current_degradation_level
            self._current_degradation_level = self._calculate_system_degradation_level()
            
            logger.info(f"Service recovery reported: {service_name}")
            logger.info(f"System degradation level: {self._current_degradation_level.value}")
            
            # Notify if degradation level improved
            if self._current_degradation_level != previous_level:
                await self._notify_degradation_change(previous_level, self._current_degradation_level)
            
            # Notify recovery
            await self._notify_service_recovery(service_name)
            
            return self._current_degradation_level
            
        except Exception as e:
            logger.error(f"Failed to report service recovery for {service_name}: {e}")
            return self._current_degradation_level
    
    async def get_minimal_viable_workflow(
        self,
        content_type: str,
        requested_agents: List[str]
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Get minimal viable workflow configuration for degraded conditions.
        
        Args:
            content_type: Type of content being created
            requested_agents: Originally requested agents
            
        Returns:
            Tuple of (available_agents, unavailable_agents, degradation_info)
        """
        try:
            # Get minimal workflow definition
            minimal_workflow = self._minimal_workflows.get(content_type)
            if not minimal_workflow:
                # Fallback to generic minimal workflow
                minimal_workflow = self._create_generic_minimal_workflow(content_type, requested_agents)
            
            available_agents = []
            unavailable_agents = []
            fallback_substitutions = {}
            
            # Check each requested agent
            for agent_name in requested_agents:
                service_health = self._service_health.get(agent_name)
                
                if not service_health or service_health.status in [
                    ServiceHealthStatus.UNAVAILABLE, 
                    ServiceHealthStatus.CRITICAL
                ]:
                    # Agent is unavailable
                    unavailable_agents.append(agent_name)
                    
                    # Check for fallback
                    if agent_name in minimal_workflow.fallback_agents:
                        fallback_agent = minimal_workflow.fallback_agents[agent_name]
                        fallback_health = self._service_health.get(fallback_agent)
                        
                        if fallback_health and fallback_health.status == ServiceHealthStatus.HEALTHY:
                            available_agents.append(fallback_agent)
                            fallback_substitutions[agent_name] = fallback_agent
                        else:
                            # Check if agent can be skipped
                            if agent_name in minimal_workflow.optional_agents:
                                logger.info(f"Skipping optional agent {agent_name}")
                            else:
                                logger.warning(f"Required agent {agent_name} unavailable with no fallback")
                    
                    elif agent_name in minimal_workflow.optional_agents:
                        # Optional agent can be skipped
                        logger.info(f"Skipping optional agent {agent_name}")
                    else:
                        logger.warning(f"Required agent {agent_name} unavailable")
                
                else:
                    # Agent is available
                    available_agents.append(agent_name)
            
            # Ensure we have minimum required agents
            required_agents_available = [
                agent for agent in minimal_workflow.required_agents 
                if agent in available_agents or any(
                    fallback == agent for fallback in fallback_substitutions.values()
                )
            ]
            
            can_execute = len(required_agents_available) >= len(minimal_workflow.required_agents) * 0.5
            
            degradation_info = {
                'minimal_workflow': minimal_workflow,
                'can_execute_workflow': can_execute,
                'fallback_substitutions': fallback_substitutions,
                'degradation_level': self._current_degradation_level.value,
                'quality_impact': self._estimate_quality_impact(available_agents, unavailable_agents),
                'user_expectations': minimal_workflow.user_expectations,
                'limitations': minimal_workflow.limitations,
                'estimated_completion_time': self._estimate_degraded_completion_time(available_agents)
            }
            
            logger.info(f"Minimal viable workflow for {content_type}:")
            logger.info(f"  Available agents: {available_agents}")
            logger.info(f"  Unavailable agents: {unavailable_agents}")
            logger.info(f"  Can execute: {can_execute}")
            logger.info(f"  Quality impact: {degradation_info['quality_impact']:.1%}")
            
            return available_agents, unavailable_agents, degradation_info
            
        except Exception as e:
            logger.error(f"Failed to get minimal viable workflow for {content_type}: {e}")
            return requested_agents, [], {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and degradation information.
        
        Returns:
            Dictionary with system status information
        """
        try:
            healthy_services = [
                name for name, health in self._service_health.items()
                if health.status == ServiceHealthStatus.HEALTHY
            ]
            
            degraded_services = [
                name for name, health in self._service_health.items()
                if health.status in [ServiceHealthStatus.DEGRADED, ServiceHealthStatus.CRITICAL]
            ]
            
            unavailable_services = [
                name for name, health in self._service_health.items()
                if health.status == ServiceHealthStatus.UNAVAILABLE
            ]
            
            status = {
                'overall_health': self._current_degradation_level.value,
                'healthy_services': healthy_services,
                'degraded_services': degraded_services,
                'unavailable_services': unavailable_services,
                'total_services': len(self._service_health),
                'health_percentage': len(healthy_services) / max(len(self._service_health), 1) * 100,
                'can_provide_minimal_service': self._can_provide_minimal_service(),
                'active_notifications': list(self._active_notifications.keys()),
                'auto_recovery_enabled': self.enable_auto_recovery,
                'next_health_check': datetime.now() + timedelta(seconds=self.health_check_interval),
                'service_details': {
                    name: {
                        'status': health.status.value,
                        'failure_count': health.failure_count,
                        'last_success': health.last_success.isoformat() if health.last_success else None,
                        'error_message': health.error_message,
                        'is_critical': health.is_critical,
                        'recovery_estimate': health.recovery_estimate.isoformat() if health.recovery_estimate else None
                    }
                    for name, health in self._service_health.items()
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    async def get_user_notification_message(self, content_type: str) -> Dict[str, Any]:
        """
        Get user-friendly notification message about current degradation.
        
        Args:
            content_type: Type of content being created
            
        Returns:
            Dictionary with notification message and guidance
        """
        try:
            if self._current_degradation_level == DegradationLevel.NONE:
                return {
                    'level': 'info',
                    'message': '✅ All systems operational',
                    'impact': 'No impact on functionality',
                    'actions': []
                }
            
            # Get degraded services
            degraded_service_names = [
                health.service_name for health in self._service_health.values()
                if health.status != ServiceHealthStatus.HEALTHY
            ]
            
            # Get minimal workflow info
            _, unavailable_agents, degradation_info = await self.get_minimal_viable_workflow(
                content_type, list(self._service_health.keys())
            )
            
            # Build notification message
            level_messages = {
                DegradationLevel.MINIMAL: {
                    'icon': '🟡',
                    'level': 'warning',
                    'title': 'Minor Service Degradation'
                },
                DegradationLevel.MODERATE: {
                    'icon': '🟠',
                    'level': 'caution',
                    'title': 'Moderate Service Degradation'
                },
                DegradationLevel.SEVERE: {
                    'icon': '🔴',
                    'level': 'error',
                    'title': 'Severe Service Degradation'
                },
                DegradationLevel.EMERGENCY: {
                    'icon': '🚨',
                    'level': 'critical',
                    'title': 'Emergency Mode'
                }
            }
            
            level_info = level_messages.get(self._current_degradation_level, level_messages[DegradationLevel.MODERATE])
            
            message_parts = [
                f"{level_info['icon']} {level_info['title']}",
                f"Affected services: {', '.join(degraded_service_names)}"
            ]
            
            if unavailable_agents:
                message_parts.append(f"Unavailable features: {', '.join(unavailable_agents)}")
            
            if degradation_info.get('can_execute_workflow', True):
                message_parts.append("✅ Content generation will continue with reduced functionality")
            else:
                message_parts.append("❌ Content generation is severely limited")
            
            # Suggested actions
            actions = []
            
            if self._current_degradation_level in [DegradationLevel.MINIMAL, DegradationLevel.MODERATE]:
                actions.extend([
                    "Content will be generated with available features",
                    "Some enhancements may be skipped",
                    "Quality may be slightly reduced"
                ])
            
            elif self._current_degradation_level == DegradationLevel.SEVERE:
                actions.extend([
                    "Only core content generation available",
                    "Manual review recommended",
                    "Consider retrying later for full features"
                ])
            
            elif self._current_degradation_level == DegradationLevel.EMERGENCY:
                actions.extend([
                    "Minimal functionality only",
                    "Content quality will be significantly impacted",
                    "Please retry when services recover"
                ])
            
            # Recovery estimates
            recovery_times = [
                health.recovery_estimate for health in self._service_health.values()
                if health.recovery_estimate and health.status != ServiceHealthStatus.HEALTHY
            ]
            
            if recovery_times:
                earliest_recovery = min(recovery_times)
                actions.append(f"Estimated recovery: {earliest_recovery.strftime('%H:%M:%S')}")
            
            return {
                'level': level_info['level'],
                'message': '\n'.join(message_parts),
                'impact': degradation_info.get('user_expectations', 'Reduced functionality'),
                'limitations': degradation_info.get('limitations', []),
                'actions': actions,
                'quality_impact': degradation_info.get('quality_impact', 0),
                'can_proceed': degradation_info.get('can_execute_workflow', True)
            }
            
        except Exception as e:
            logger.error(f"Failed to get user notification message: {e}")
            return {
                'level': 'error',
                'message': 'System status unknown',
                'impact': 'Unable to determine impact',
                'actions': ['Please contact support']
            }
    
    # Private helper methods
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default degradation strategies for common agents."""
        strategies = {
            'seo_agent': DegradationStrategy(
                service_name='seo_agent',
                fallback_services=['basic_seo_checker'],
                skip_if_unavailable=True,
                alternative_workflow=None,
                user_notification='SEO optimization unavailable - content will be created without SEO enhancements',
                impact_description='Search engine optimization will be skipped',
                recovery_actions=['retry_seo_agent', 'manual_seo_review']
            ),
            'image_prompt_agent': DegradationStrategy(
                service_name='image_prompt_agent',
                fallback_services=['simple_image_suggester'],
                skip_if_unavailable=True,
                alternative_workflow=None,
                user_notification='Image generation unavailable - content will be created without image prompts',
                impact_description='Visual content suggestions will be unavailable',
                recovery_actions=['retry_image_agent', 'manual_image_selection']
            ),
            'video_prompt_agent': DegradationStrategy(
                service_name='video_prompt_agent',
                fallback_services=[],
                skip_if_unavailable=True,
                alternative_workflow=None,
                user_notification='Video content unavailable - focusing on text and images',
                impact_description='Video content suggestions will be skipped',
                recovery_actions=['retry_video_agent']
            ),
            'social_media_agent': DegradationStrategy(
                service_name='social_media_agent',
                fallback_services=['basic_social_formatter'],
                skip_if_unavailable=False,
                alternative_workflow='minimal_social_workflow',
                user_notification='Advanced social media optimization unavailable - using basic formatting',
                impact_description='Social media content will use basic formatting',
                recovery_actions=['retry_social_agent', 'manual_social_optimization']
            )
        }
        
        self._degradation_strategies.update(strategies)
        logger.info(f"Initialized {len(strategies)} default degradation strategies")
    
    def _initialize_minimal_workflows(self) -> None:
        """Initialize minimal viable workflows for different content types."""
        workflows = {
            'blog_post': MinimalViableWorkflow(
                content_type='blog_post',
                required_agents=['planner_agent', 'writer_agent'],
                optional_agents=['seo_agent', 'image_prompt_agent', 'video_prompt_agent', 'geo_agent'],
                fallback_agents={
                    'researcher_agent': 'basic_researcher',
                    'editor_agent': 'basic_editor',
                    'seo_agent': 'basic_seo_checker'
                },
                quality_threshold=0.7,
                user_expectations='Basic blog post with core content - may lack advanced optimizations',
                limitations=['No advanced SEO', 'Limited visual content', 'Basic formatting']
            ),
            'social_media': MinimalViableWorkflow(
                content_type='social_media',
                required_agents=['writer_agent', 'social_media_agent'],
                optional_agents=['image_prompt_agent', 'video_prompt_agent'],
                fallback_agents={
                    'social_media_agent': 'basic_social_formatter'
                },
                quality_threshold=0.6,
                user_expectations='Basic social media posts with standard formatting',
                limitations=['Platform-specific optimization may be limited', 'No advanced scheduling']
            ),
            'email_campaign': MinimalViableWorkflow(
                content_type='email_campaign',
                required_agents=['planner_agent', 'writer_agent', 'content_repurposer_agent'],
                optional_agents=['seo_agent', 'image_prompt_agent'],
                fallback_agents={
                    'content_repurposer_agent': 'basic_formatter'
                },
                quality_threshold=0.75,
                user_expectations='Basic email content with standard formatting',
                limitations=['No advanced personalization', 'Limited template options']
            )
        }
        
        self._minimal_workflows.update(workflows)
        logger.info(f"Initialized {len(workflows)} minimal viable workflows")
    
    def _create_generic_minimal_workflow(
        self, 
        content_type: str, 
        requested_agents: List[str]
    ) -> MinimalViableWorkflow:
        """Create a generic minimal workflow for unknown content types."""
        # Categorize agents by importance
        critical_agents = ['planner_agent', 'writer_agent']
        optional_agents = [
            agent for agent in requested_agents 
            if agent not in critical_agents
        ]
        
        return MinimalViableWorkflow(
            content_type=content_type,
            required_agents=[agent for agent in critical_agents if agent in requested_agents],
            optional_agents=optional_agents,
            fallback_agents={},
            quality_threshold=0.6,
            user_expectations=f'Basic {content_type} with core functionality',
            limitations=['Advanced features may be unavailable', 'Quality may be reduced']
        )
    
    def _calculate_system_degradation_level(self) -> DegradationLevel:
        """Calculate overall system degradation level."""
        if not self._service_health:
            return DegradationLevel.NONE
        
        total_services = len(self._service_health)
        healthy_services = sum(
            1 for health in self._service_health.values()
            if health.status == ServiceHealthStatus.HEALTHY
        )
        critical_services = sum(
            1 for health in self._service_health.values()
            if health.is_critical and health.status != ServiceHealthStatus.HEALTHY
        )
        unavailable_services = sum(
            1 for health in self._service_health.values()
            if health.status == ServiceHealthStatus.UNAVAILABLE
        )
        
        health_percentage = healthy_services / total_services
        
        # Determine degradation level
        if critical_services > 0:
            return DegradationLevel.EMERGENCY
        elif health_percentage < 0.3:
            return DegradationLevel.SEVERE
        elif health_percentage < 0.6 or unavailable_services > 2:
            return DegradationLevel.MODERATE
        elif health_percentage < 0.9:
            return DegradationLevel.MINIMAL
        else:
            return DegradationLevel.NONE
    
    def _can_provide_minimal_service(self) -> bool:
        """Check if system can provide minimal service."""
        # At minimum, we need writer_agent or planner_agent
        essential_agents = ['writer_agent', 'planner_agent']
        
        for agent in essential_agents:
            if agent in self._service_health:
                health = self._service_health[agent]
                if health.status in [ServiceHealthStatus.HEALTHY, ServiceHealthStatus.DEGRADED]:
                    return True
        
        return False
    
    def _estimate_quality_impact(self, available_agents: List[str], unavailable_agents: List[str]) -> float:
        """Estimate quality impact based on unavailable agents."""
        if not unavailable_agents:
            return 0.0
        
        # Weight agents by their quality impact
        agent_weights = {
            'editor_agent': 0.25,
            'seo_agent': 0.15,
            'researcher_agent': 0.20,
            'image_prompt_agent': 0.10,
            'video_prompt_agent': 0.08,
            'geo_agent': 0.12,
            'social_media_agent': 0.10
        }
        
        total_impact = sum(
            agent_weights.get(agent, 0.05) 
            for agent in unavailable_agents
        )
        
        return min(total_impact, 0.8)  # Cap at 80% impact
    
    def _estimate_degraded_completion_time(self, available_agents: List[str]) -> int:
        """Estimate completion time with available agents (in seconds)."""
        base_time = 120  # 2 minutes base
        agent_time_contributions = {
            'planner_agent': 20,
            'researcher_agent': 30,
            'writer_agent': 40,
            'editor_agent': 25,
            'seo_agent': 15,
            'image_prompt_agent': 10,
            'video_prompt_agent': 10,
            'social_media_agent': 15
        }
        
        total_time = base_time + sum(
            agent_time_contributions.get(agent, 10)
            for agent in available_agents
        )
        
        return total_time
    
    async def _notify_degradation_change(
        self, 
        previous_level: DegradationLevel, 
        new_level: DegradationLevel
    ) -> None:
        """Notify about system degradation level changes."""
        if new_level.value != previous_level.value:
            if new_level == DegradationLevel.NONE:
                logger.info("🎉 System fully recovered - all services operational")
            elif new_level.value < previous_level.value:
                logger.info(f"📈 System degradation improved: {previous_level.value} → {new_level.value}")
            else:
                logger.warning(f"📉 System degradation worsened: {previous_level.value} → {new_level.value}")
    
    async def _notify_service_degradation(self, service_name: str, health: ServiceHealth) -> None:
        """Notify about service degradation."""
        notification_key = f"{service_name}_degraded"
        
        # Avoid spam notifications
        if notification_key in self._active_notifications:
            last_notification = self._active_notifications[notification_key]
            if datetime.now() - last_notification < timedelta(minutes=5):
                return
        
        self._active_notifications[notification_key] = datetime.now()
        
        if service_name in self._degradation_strategies:
            strategy = self._degradation_strategies[service_name]
            logger.warning(f"🚨 Service degradation: {strategy.user_notification}")
        else:
            logger.warning(f"🚨 Service {service_name} is experiencing issues")
    
    async def _notify_service_recovery(self, service_name: str) -> None:
        """Notify about service recovery."""
        logger.info(f"✅ Service recovered: {service_name}")
        
        # Clear related notifications
        notification_keys = [key for key in self._active_notifications.keys() if service_name in key]
        for key in notification_keys:
            self._active_notifications.pop(key, None)
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        try:
            if self._health_check_task is None or self._health_check_task.done():
                self._health_check_task = asyncio.create_task(self._health_monitoring_task())
        except RuntimeError:
            # No event loop running, monitoring will be started when needed
            pass
    
    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.enable_auto_recovery:
                    await self._attempt_auto_recovery()
                
                # Clean up old notifications
                await self._cleanup_old_notifications()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _attempt_auto_recovery(self) -> None:
        """Attempt automatic recovery of failed services."""
        current_time = datetime.now()
        
        for service_name, health in self._service_health.items():
            if health.status != ServiceHealthStatus.HEALTHY and health.recovery_estimate:
                if current_time >= health.recovery_estimate:
                    # Attempt recovery check
                    if service_name in self._recovery_callbacks:
                        try:
                            recovery_func = self._recovery_callbacks[service_name]
                            is_recovered = await recovery_func()
                            
                            if is_recovered:
                                await self.report_service_recovery(service_name)
                            else:
                                # Extend recovery estimate
                                health.recovery_estimate = current_time + timedelta(seconds=self.recovery_timeout)
                                
                        except Exception as e:
                            logger.error(f"Auto-recovery check failed for {service_name}: {e}")
    
    async def _cleanup_old_notifications(self) -> None:
        """Clean up old notification records."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)
        
        expired_notifications = [
            key for key, timestamp in self._active_notifications.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_notifications:
            self._active_notifications.pop(key, None)
    
    async def close(self) -> None:
        """Clean shutdown of graceful degradation manager."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("GracefulDegradationManager closed")


# Global graceful degradation manager instance
graceful_degradation_manager = GracefulDegradationManager()

logger.info("Graceful Degradation Manager loaded successfully!")
logger.info("Features: Service Health Monitoring, Minimal Viable Workflows, Auto-Recovery, User Notifications")