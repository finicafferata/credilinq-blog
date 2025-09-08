"""
Enhanced Error Handling System for Parallel Research Orchestration

This module provides comprehensive error handling, recovery, and fault tolerance
for parallel research execution, ensuring robustness and reliability.

Key Features:
- Graceful agent failure handling with fallback strategies
- Timeout management with intelligent retry logic
- Circuit breaker pattern for failing agents
- Partial result preservation and recovery
- Error classification and severity assessment
- Automated recovery recommendations
- Performance impact mitigation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
from collections import defaultdict
import json

# Core agent imports
from ..core.base_agent import AgentResult, AgentExecutionContext

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # System-breaking errors
    HIGH = "high"             # Major functionality impact
    MEDIUM = "medium"         # Partial functionality impact
    LOW = "low"              # Minor impact, degraded performance
    INFO = "info"            # Informational, no functional impact


class ErrorCategory(Enum):
    """Categories of errors in parallel research."""
    AGENT_FAILURE = "agent_failure"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    DATA_VALIDATION = "data_validation"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    DEPENDENCY_FAILURE = "dependency_failure"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_SEQUENTIAL = "fallback_sequential"
    USE_CACHED_RESULTS = "use_cached_results"
    CONTINUE_WITH_PARTIAL = "continue_with_partial"
    SKIP_FAILED_AGENT = "skip_failed_agent"
    ABORT_WORKFLOW = "abort_workflow"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    traceback_info: Optional[str] = None
    agent_name: Optional[str] = None
    workflow_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Recovery information
    suggested_recovery: RecoveryStrategy = RecoveryStrategy.RETRY_WITH_BACKOFF
    recovery_attempted: bool = False
    recovery_success: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for agent fault tolerance."""
    agent_name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 3
    recovery_timeout: int = 300  # 5 minutes
    
    def should_allow_request(self) -> bool:
        """Check if requests should be allowed through the circuit breaker."""
        now = datetime.now()
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and (now - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self):
        """Record a successful request."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ParallelResearchErrorHandler:
    """Comprehensive error handling system for parallel research operations."""
    
    def __init__(self):
        """Initialize the error handling system."""
        self.error_history = []
        self.circuit_breakers = {}  # agent_name -> CircuitBreakerState
        self.error_patterns = defaultdict(int)
        self.recovery_cache = {}
        
        # Configuration
        self.max_retries = 3
        self.base_timeout = 300  # 5 minutes
        self.retry_delays = [1, 2, 4, 8, 16]  # Exponential backoff
        self.partial_result_threshold = 0.5  # Accept if 50% of agents succeed
        
        logger.info("ParallelResearchErrorHandler initialized")
    
    def classify_error(self, 
                      exception: Exception, 
                      agent_name: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> ErrorDetails:
        """Classify an error and determine its category and severity."""
        
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # Generate unique error ID
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(error_message) % 10000}"
        
        # Classify error category
        category = self._classify_error_category(exception, error_message)
        
        # Determine severity
        severity = self._determine_error_severity(category, exception, context)
        
        # Suggest recovery strategy
        recovery_strategy = self._suggest_recovery_strategy(category, severity, agent_name)
        
        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            category=category,
            severity=severity,
            message=f"{error_type}: {error_message}",
            exception=exception,
            traceback_info=traceback.format_exc(),
            agent_name=agent_name,
            context=context or {},
            suggested_recovery=recovery_strategy
        )
        
        # Store in error history
        self.error_history.append(error_details)
        self.error_patterns[f"{category.value}_{severity.value}"] += 1
        
        logger.error(f"Error classified: {error_id} - {category.value} ({severity.value})")
        logger.error(f"Message: {error_message}")
        logger.error(f"Suggested recovery: {recovery_strategy.value}")
        
        return error_details
    
    def _classify_error_category(self, exception: Exception, error_message: str) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_message_lower = error_message.lower()
        
        # Timeout errors
        if isinstance(exception, asyncio.TimeoutError) or "timeout" in error_message_lower:
            return ErrorCategory.TIMEOUT
        
        # Network errors
        if any(keyword in error_message_lower for keyword in [
            "connection", "network", "unreachable", "dns", "socket"
        ]):
            return ErrorCategory.NETWORK_ERROR
        
        # Authentication errors
        if any(keyword in error_message_lower for keyword in [
            "authentication", "unauthorized", "api key", "access denied", "forbidden"
        ]):
            return ErrorCategory.AUTHENTICATION
        
        # Rate limiting
        if any(keyword in error_message_lower for keyword in [
            "rate limit", "too many requests", "quota exceeded", "throttle"
        ]):
            return ErrorCategory.RATE_LIMITING
        
        # Resource exhaustion
        if any(keyword in error_message_lower for keyword in [
            "memory", "disk", "cpu", "resource", "out of space"
        ]):
            return ErrorCategory.RESOURCE_EXHAUSTION
        
        # Data validation errors
        if any(keyword in error_message_lower for keyword in [
            "validation", "invalid", "malformed", "parse", "format"
        ]):
            return ErrorCategory.DATA_VALIDATION
        
        # Configuration errors
        if any(keyword in error_message_lower for keyword in [
            "configuration", "config", "missing parameter", "not found"
        ]):
            return ErrorCategory.CONFIGURATION
        
        # Dependency failures
        if any(keyword in error_message_lower for keyword in [
            "service unavailable", "dependency", "external service"
        ]):
            return ErrorCategory.DEPENDENCY_FAILURE
        
        # Agent-specific failures
        if any(keyword in error_message_lower for keyword in [
            "agent", "execution failed", "workflow"
        ]):
            return ErrorCategory.AGENT_FAILURE
        
        return ErrorCategory.UNKNOWN
    
    def _determine_error_severity(self, 
                                 category: ErrorCategory, 
                                 exception: Exception,
                                 context: Optional[Dict[str, Any]]) -> ErrorSeverity:
        """Determine error severity based on category and context."""
        
        # Critical severity conditions
        if category in [ErrorCategory.RESOURCE_EXHAUSTION, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.CRITICAL
        
        # High severity conditions
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.DEPENDENCY_FAILURE]:
            return ErrorSeverity.HIGH
        
        # Context-based severity
        if context:
            # If both agents fail, it's critical
            if context.get('both_agents_failed', False):
                return ErrorSeverity.CRITICAL
            
            # If this is a retry failure, increase severity
            retry_count = context.get('retry_count', 0)
            if retry_count > 2:
                return ErrorSeverity.HIGH
        
        # Default severity based on category
        severity_mapping = {
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.RATE_LIMITING: ErrorSeverity.LOW,
            ErrorCategory.DATA_VALIDATION: ErrorSeverity.MEDIUM,
            ErrorCategory.AGENT_FAILURE: ErrorSeverity.HIGH,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
        }
        
        return severity_mapping.get(category, ErrorSeverity.MEDIUM)
    
    def _suggest_recovery_strategy(self, 
                                  category: ErrorCategory, 
                                  severity: ErrorSeverity,
                                  agent_name: Optional[str]) -> RecoveryStrategy:
        """Suggest appropriate recovery strategy."""
        
        # Critical errors - abort workflow
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT_WORKFLOW
        
        # Category-based strategies
        strategy_mapping = {
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.ABORT_WORKFLOW,
            ErrorCategory.RATE_LIMITING: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.RESOURCE_EXHAUSTION: RecoveryStrategy.FALLBACK_SEQUENTIAL,
            ErrorCategory.DATA_VALIDATION: RecoveryStrategy.CONTINUE_WITH_PARTIAL,
            ErrorCategory.CONFIGURATION: RecoveryStrategy.ABORT_WORKFLOW,
            ErrorCategory.DEPENDENCY_FAILURE: RecoveryStrategy.CIRCUIT_BREAKER,
            ErrorCategory.AGENT_FAILURE: RecoveryStrategy.SKIP_FAILED_AGENT,
            ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY_WITH_BACKOFF
        }
        
        return strategy_mapping.get(category, RecoveryStrategy.RETRY_WITH_BACKOFF)
    
    async def handle_agent_error(self,
                                agent_name: str,
                                error: Exception,
                                workflow_id: str,
                                context: Optional[Dict[str, Any]] = None,
                                agent_executor: Optional[Callable] = None) -> AgentResult:
        """Handle individual agent errors with recovery logic."""
        
        # Classify the error
        error_details = self.classify_error(error, agent_name, context)
        error_details.workflow_id = workflow_id
        
        logger.warning(f"Handling agent error for {agent_name}: {error_details.error_id}")
        
        # Check circuit breaker
        if not self._check_circuit_breaker(agent_name):
            logger.warning(f"Circuit breaker OPEN for {agent_name}, skipping execution")
            return AgentResult(
                success=False,
                error_message=f"Circuit breaker OPEN for {agent_name}",
                metadata={
                    "error_id": error_details.error_id,
                    "recovery_strategy": "circuit_breaker_open",
                    "agent_name": agent_name
                }
            )
        
        # Execute recovery strategy
        recovery_result = await self._execute_recovery_strategy(
            error_details, agent_executor, context or {}
        )
        
        # Update circuit breaker based on recovery result
        self._update_circuit_breaker(agent_name, recovery_result.success)
        
        return recovery_result
    
    def _check_circuit_breaker(self, agent_name: str) -> bool:
        """Check if circuit breaker allows requests for the agent."""
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreakerState(agent_name=agent_name)
        
        return self.circuit_breakers[agent_name].should_allow_request()
    
    def _update_circuit_breaker(self, agent_name: str, success: bool) -> None:
        """Update circuit breaker state based on execution result."""
        if agent_name not in self.circuit_breakers:
            return
        
        if success:
            self.circuit_breakers[agent_name].record_success()
            logger.info(f"Circuit breaker for {agent_name} recorded success")
        else:
            self.circuit_breakers[agent_name].record_failure()
            logger.warning(f"Circuit breaker for {agent_name} recorded failure")
            
            if self.circuit_breakers[agent_name].state == "OPEN":
                logger.error(f"Circuit breaker for {agent_name} is now OPEN")
    
    async def _execute_recovery_strategy(self,
                                        error_details: ErrorDetails,
                                        agent_executor: Optional[Callable],
                                        context: Dict[str, Any]) -> AgentResult:
        """Execute the suggested recovery strategy."""
        
        strategy = error_details.suggested_recovery
        agent_name = error_details.agent_name
        
        logger.info(f"Executing recovery strategy: {strategy.value} for {agent_name}")
        
        try:
            if strategy == RecoveryStrategy.RETRY_IMMEDIATE:
                return await self._retry_immediate(error_details, agent_executor, context)
            
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                return await self._retry_with_backoff(error_details, agent_executor, context)
            
            elif strategy == RecoveryStrategy.FALLBACK_SEQUENTIAL:
                return await self._fallback_sequential(error_details, context)
            
            elif strategy == RecoveryStrategy.USE_CACHED_RESULTS:
                return await self._use_cached_results(error_details, context)
            
            elif strategy == RecoveryStrategy.CONTINUE_WITH_PARTIAL:
                return self._continue_with_partial(error_details, context)
            
            elif strategy == RecoveryStrategy.SKIP_FAILED_AGENT:
                return self._skip_failed_agent(error_details, context)
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._handle_circuit_breaker(error_details, context)
            
            elif strategy == RecoveryStrategy.ABORT_WORKFLOW:
                return self._abort_workflow(error_details, context)
            
            else:
                logger.error(f"Unknown recovery strategy: {strategy.value}")
                return self._skip_failed_agent(error_details, context)
                
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            return AgentResult(
                success=False,
                error_message=f"Recovery failed: {str(recovery_error)}",
                metadata={
                    "error_id": error_details.error_id,
                    "recovery_strategy": strategy.value,
                    "recovery_failed": True
                }
            )
    
    async def _retry_immediate(self,
                              error_details: ErrorDetails,
                              agent_executor: Optional[Callable],
                              context: Dict[str, Any]) -> AgentResult:
        """Retry immediately without delay."""
        if not agent_executor:
            return self._skip_failed_agent(error_details, context)
        
        try:
            logger.info(f"Immediate retry for {error_details.agent_name}")
            result = await agent_executor()
            
            if result.success:
                logger.info(f"Immediate retry successful for {error_details.agent_name}")
                error_details.recovery_success = True
            
            return result
            
        except Exception as retry_error:
            logger.error(f"Immediate retry failed: {retry_error}")
            return self._skip_failed_agent(error_details, context)
    
    async def _retry_with_backoff(self,
                                 error_details: ErrorDetails,
                                 agent_executor: Optional[Callable],
                                 context: Dict[str, Any]) -> AgentResult:
        """Retry with exponential backoff."""
        if not agent_executor:
            return self._skip_failed_agent(error_details, context)
        
        max_retries = context.get('max_retries', self.max_retries)
        retry_count = error_details.retry_count
        
        if retry_count >= max_retries:
            logger.warning(f"Max retries ({max_retries}) exceeded for {error_details.agent_name}")
            return self._skip_failed_agent(error_details, context)
        
        # Calculate delay with exponential backoff
        delay_index = min(retry_count, len(self.retry_delays) - 1)
        delay = self.retry_delays[delay_index]
        
        logger.info(f"Retry with backoff for {error_details.agent_name} (attempt {retry_count + 1}/{max_retries}, delay: {delay}s)")
        
        await asyncio.sleep(delay)
        
        try:
            error_details.retry_count += 1
            result = await agent_executor()
            
            if result.success:
                logger.info(f"Retry successful for {error_details.agent_name} after {error_details.retry_count} attempts")
                error_details.recovery_success = True
            
            return result
            
        except Exception as retry_error:
            logger.error(f"Retry failed for {error_details.agent_name}: {retry_error}")
            # Recursively retry with updated count
            return await self._retry_with_backoff(error_details, agent_executor, context)
    
    async def _fallback_sequential(self,
                                  error_details: ErrorDetails,
                                  context: Dict[str, Any]) -> AgentResult:
        """Fallback to sequential execution if parallel fails."""
        logger.info(f"Fallback to sequential execution due to {error_details.category.value}")
        
        return AgentResult(
            success=False,
            error_message="Parallel execution failed, fallback to sequential recommended",
            metadata={
                "error_id": error_details.error_id,
                "recovery_strategy": "fallback_sequential",
                "recommendation": "Execute agents sequentially instead of parallel"
            }
        )
    
    async def _use_cached_results(self,
                                 error_details: ErrorDetails,
                                 context: Dict[str, Any]) -> AgentResult:
        """Use cached results from previous executions."""
        cache_key = self._generate_cache_key(error_details.agent_name, context)
        
        if cache_key in self.recovery_cache:
            cached_result = self.recovery_cache[cache_key]
            logger.info(f"Using cached results for {error_details.agent_name}")
            
            return AgentResult(
                success=True,
                data=cached_result,
                metadata={
                    "error_id": error_details.error_id,
                    "recovery_strategy": "cached_results",
                    "cache_hit": True
                }
            )
        else:
            logger.warning(f"No cached results available for {error_details.agent_name}")
            return self._skip_failed_agent(error_details, context)
    
    def _continue_with_partial(self,
                              error_details: ErrorDetails,
                              context: Dict[str, Any]) -> AgentResult:
        """Continue execution with partial results."""
        logger.info(f"Continuing with partial results despite {error_details.agent_name} failure")
        
        return AgentResult(
            success=True,  # Mark as success to continue workflow
            data={"partial_execution": True, "failed_agent": error_details.agent_name},
            metadata={
                "error_id": error_details.error_id,
                "recovery_strategy": "continue_partial",
                "partial_results": True,
                "quality_impact": "medium"
            }
        )
    
    def _skip_failed_agent(self,
                          error_details: ErrorDetails,
                          context: Dict[str, Any]) -> AgentResult:
        """Skip the failed agent and continue with other agents."""
        logger.info(f"Skipping failed agent: {error_details.agent_name}")
        
        return AgentResult(
            success=False,
            error_message=f"Agent {error_details.agent_name} skipped due to failure",
            metadata={
                "error_id": error_details.error_id,
                "recovery_strategy": "skip_failed_agent",
                "agent_skipped": error_details.agent_name,
                "continue_workflow": True
            }
        )
    
    def _handle_circuit_breaker(self,
                               error_details: ErrorDetails,
                               context: Dict[str, Any]) -> AgentResult:
        """Handle circuit breaker activation."""
        logger.warning(f"Circuit breaker activated for {error_details.agent_name}")
        
        return AgentResult(
            success=False,
            error_message=f"Circuit breaker activated for {error_details.agent_name}",
            metadata={
                "error_id": error_details.error_id,
                "recovery_strategy": "circuit_breaker",
                "circuit_breaker_state": "OPEN",
                "retry_after": 300  # seconds
            }
        )
    
    def _abort_workflow(self,
                       error_details: ErrorDetails,
                       context: Dict[str, Any]) -> AgentResult:
        """Abort the entire workflow due to critical error."""
        logger.critical(f"Aborting workflow due to critical error: {error_details.message}")
        
        return AgentResult(
            success=False,
            error_message=f"Workflow aborted: {error_details.message}",
            metadata={
                "error_id": error_details.error_id,
                "recovery_strategy": "abort_workflow",
                "workflow_aborted": True,
                "severity": error_details.severity.value
            }
        )
    
    def _generate_cache_key(self, agent_name: str, context: Dict[str, Any]) -> str:
        """Generate cache key for results."""
        topics = context.get('research_topics', [])
        audience = context.get('target_audience', 'default')
        return f"{agent_name}_{hash(str(topics))}_{audience}"
    
    async def handle_parallel_execution_error(self,
                                             researcher_result: Union[AgentResult, Exception],
                                             search_agent_result: Union[AgentResult, Exception],
                                             workflow_id: str,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors from parallel execution of both agents."""
        
        errors = []
        successful_results = {}
        
        # Handle ResearcherAgent result
        if isinstance(researcher_result, Exception):
            error_details = self.classify_error(researcher_result, "ResearcherAgent")
            error_details.workflow_id = workflow_id
            errors.append(error_details)
        elif not researcher_result.success:
            errors.append(ErrorDetails(
                error_id=f"researcher_fail_{workflow_id}",
                category=ErrorCategory.AGENT_FAILURE,
                severity=ErrorSeverity.HIGH,
                message=researcher_result.error_message or "ResearcherAgent failed",
                agent_name="ResearcherAgent",
                workflow_id=workflow_id
            ))
        else:
            successful_results["researcher"] = researcher_result
        
        # Handle SearchAgent result
        if isinstance(search_agent_result, Exception):
            error_details = self.classify_error(search_agent_result, "SearchAgent")
            error_details.workflow_id = workflow_id
            errors.append(error_details)
        elif not search_agent_result.success:
            errors.append(ErrorDetails(
                error_id=f"search_agent_fail_{workflow_id}",
                category=ErrorCategory.AGENT_FAILURE,
                severity=ErrorSeverity.HIGH,
                message=search_agent_result.error_message or "SearchAgent failed",
                agent_name="SearchAgent",
                workflow_id=workflow_id
            ))
        else:
            successful_results["search_agent"] = search_agent_result
        
        # Determine recovery strategy based on results
        if len(successful_results) == 0:
            # Both agents failed - critical situation
            logger.critical(f"Both agents failed for workflow {workflow_id}")
            return {
                "status": "critical_failure",
                "errors": errors,
                "recommendation": "abort_workflow",
                "successful_results": {}
            }
        
        elif len(successful_results) == 1:
            # One agent succeeded - partial success
            logger.warning(f"Partial success for workflow {workflow_id}: {list(successful_results.keys())} succeeded")
            return {
                "status": "partial_success",
                "errors": errors,
                "recommendation": "continue_with_partial",
                "successful_results": successful_results,
                "quality_impact": "medium"
            }
        
        else:
            # Both agents succeeded
            return {
                "status": "success",
                "errors": [],
                "recommendation": "continue",
                "successful_results": successful_results
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"message": "No error data available"}
        
        total_errors = len(self.error_history)
        
        # Group by category
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        recovery_success_rate = defaultdict(list)
        
        for error in self.error_history:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
            
            if error.recovery_attempted:
                recovery_success_rate[error.category.value].append(error.recovery_success)
        
        # Calculate recovery success rates
        recovery_rates = {}
        for category, successes in recovery_success_rate.items():
            if successes:
                recovery_rates[category] = sum(successes) / len(successes)
        
        # Circuit breaker states
        circuit_breaker_status = {
            agent: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for agent, cb in self.circuit_breakers.items()
        }
        
        return {
            "total_errors": total_errors,
            "category_distribution": dict(category_counts),
            "severity_distribution": dict(severity_counts),
            "recovery_success_rates": recovery_rates,
            "circuit_breaker_status": circuit_breaker_status,
            "error_patterns": dict(self.error_patterns),
            "recent_errors": [
                {
                    "error_id": error.error_id,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "agent": error.agent_name,
                    "timestamp": error.timestamp.isoformat(),
                    "recovery_attempted": error.recovery_attempted,
                    "recovery_success": error.recovery_success
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def get_error_report(self) -> str:
        """Generate comprehensive error report."""
        stats = self.get_error_statistics()
        
        if "message" in stats:
            return stats["message"]
        
        report_lines = [
            "=== Parallel Research Error Analysis Report ===",
            "",
            f"Total Errors Recorded: {stats['total_errors']}",
            "",
            "=== Error Distribution by Category ===",
        ]
        
        for category, count in stats['category_distribution'].items():
            percentage = (count / stats['total_errors']) * 100
            report_lines.append(f"{category}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "=== Error Distribution by Severity ===",
        ])
        
        for severity, count in stats['severity_distribution'].items():
            percentage = (count / stats['total_errors']) * 100
            report_lines.append(f"{severity}: {count} ({percentage:.1f}%)")
        
        if stats['recovery_success_rates']:
            report_lines.extend([
                "",
                "=== Recovery Success Rates by Category ===",
            ])
            
            for category, rate in stats['recovery_success_rates'].items():
                report_lines.append(f"{category}: {rate:.1%}")
        
        if stats['circuit_breaker_status']:
            report_lines.extend([
                "",
                "=== Circuit Breaker Status ===",
            ])
            
            for agent, status in stats['circuit_breaker_status'].items():
                report_lines.append(f"{agent}: {status['state']} (failures: {status['failure_count']})")
        
        return "\n".join(report_lines)
    
    def clear_old_errors(self, hours: int = 24) -> None:
        """Clear error history older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.error_history = [
            error for error in self.error_history 
            if error.timestamp > cutoff_time
        ]
        
        logger.info(f"Cleared errors older than {hours} hours")


# Global error handler instance
parallel_research_error_handler = ParallelResearchErrorHandler()

logger.info("🛡️ Parallel Research Error Handler loaded successfully!")