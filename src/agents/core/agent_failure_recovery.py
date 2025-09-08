"""
Agent Failure Recovery System

Implements User Story 4.1: Add Automatic Recovery from Agent Failures
- Detect agent failures in real-time
- Implement retry logic with exponential backoff
- Add fallback to alternative agents
- Log all failures and recovery attempts
- Alert on repeated failures

This system provides comprehensive failure recovery with circuit breakers,
fallback agents, and intelligent retry strategies.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Union, Type, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import json
from collections import defaultdict, deque

# Import existing infrastructure
from .performance_optimizer import RetryManager
from .base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentType
from ...core.enhanced_exceptions import ErrorCategory
from ...core.monitoring import metrics

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of agent failures for classification"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"
    EXTERNAL_SERVICE = "external_service"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types"""
    RETRY = "retry"
    FALLBACK_AGENT = "fallback_agent"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADED_MODE = "degraded_mode"
    FAIL_FAST = "fail_fast"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class FailureRecord:
    """Record of an agent failure"""
    agent_type: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    execution_context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    fallback_agent_used: Optional[str] = None

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2

class CircuitBreaker:
    """Circuit breaker implementation for agent failure protection"""
    
    def __init__(self, agent_name: str, config: CircuitBreakerConfig):
        self.agent_name = agent_name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
    def can_execute(self) -> bool:
        """Check if agent can execute based on circuit breaker state"""
        now = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                now - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker for {self.agent_name} moved to HALF_OPEN state")
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
            
        return False
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.agent_name} recovered to CLOSED state")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        now = datetime.now()
        self.failure_count += 1
        self.last_failure_time = now
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker for {self.agent_name} opened due to {self.failure_count} failures")
                
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker for {self.agent_name} reopened during half-open test")
        
        self.half_open_calls += 1

class FallbackAgentRegistry:
    """Registry for fallback agents and recovery strategies"""
    
    def __init__(self):
        self.fallback_mapping: Dict[str, List[str]] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.fallback_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Initialize default fallback mappings as per User Story 4.1
        self._initialize_default_fallbacks()
    
    def _initialize_default_fallbacks(self):
        """Initialize default fallback mappings from User Story 4.1"""
        self.fallback_mapping.update({
            'WriterAgent': ['SimpleWriterAgent', 'BasicContentGenerator'],
            'WriterAgentLangGraph': ['WriterAgent', 'SimpleWriterAgent'],
            'ResearcherAgent': ['BasicResearchAgent', 'SimpleSearchAgent'],
            'ResearcherAgentLangGraph': ['ResearcherAgent', 'BasicResearchAgent'],
            'SEOAgent': ['BasicSEOCheck', 'SimpleSEOAnalyzer'],
            'SEOAgentLangGraph': ['SEOAgent', 'BasicSEOCheck'],
            'EditorAgent': ['BasicEditorAgent', 'SimpleTextProcessor'],
            'EditorAgentLangGraph': ['EditorAgent', 'BasicEditorAgent']
        })
        
        # Set default recovery strategies
        for agent_type in self.fallback_mapping.keys():
            self.fallback_strategies[agent_type] = RecoveryStrategy.FALLBACK_AGENT
    
    def register_fallback(self, primary_agent: str, fallback_agents: List[str]):
        """Register fallback agents for a primary agent"""
        self.fallback_mapping[primary_agent] = fallback_agents
        logger.info(f"Registered fallback agents for {primary_agent}: {fallback_agents}")
    
    def register_agent_instance(self, agent_name: str, agent_instance: BaseAgent):
        """Register an agent instance for fallback use"""
        self.agent_instances[agent_name] = agent_instance
    
    def get_fallback_agents(self, primary_agent: str) -> List[str]:
        """Get list of fallback agents for primary agent"""
        return self.fallback_mapping.get(primary_agent, [])
    
    def get_fallback_instance(self, fallback_agent_name: str) -> Optional[BaseAgent]:
        """Get fallback agent instance"""
        return self.agent_instances.get(fallback_agent_name)

class AgentFailureRecoverySystem:
    """
    Comprehensive agent failure recovery system implementing User Story 4.1
    """
    
    def __init__(self):
        self.retry_manager = RetryManager(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True
        )
        
        self.fallback_registry = FallbackAgentRegistry()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_history: deque = deque(maxlen=1000)
        self.agent_health_status: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Metrics tracking
        self.recovery_metrics = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'fallback_activations': 0,
            'circuit_breaker_trips': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'failure_rate_threshold': 0.20,  # 20% failure rate triggers alert
            'repeated_failure_count': 3,      # Alert after 3 repeated failures
            'circuit_breaker_duration': 300   # Alert if CB open for 5+ minutes
        }
    
    def get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent"""
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreaker(
                agent_name, 
                CircuitBreakerConfig()
            )
        return self.circuit_breakers[agent_name]
    
    async def execute_with_recovery(
        self,
        agent: BaseAgent,
        execution_context: AgentExecutionContext,
        input_data: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute agent with comprehensive failure recovery
        
        This is the main entry point that implements all recovery strategies:
        - Circuit breaker protection
        - Retry logic with exponential backoff
        - Fallback agent activation
        - Comprehensive error logging
        """
        agent_name = agent.__class__.__name__
        circuit_breaker = self.get_circuit_breaker(agent_name)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for {agent_name}, attempting fallback")
            return await self._try_fallback_agents(agent_name, execution_context, input_data)
        
        # Attempt execution with retry logic
        return await self._execute_with_retry(agent, agent_name, execution_context, input_data)
    
    async def _execute_with_retry(
        self,
        agent: BaseAgent,
        agent_name: str,
        execution_context: AgentExecutionContext,
        input_data: Dict[str, Any]
    ) -> AgentResult:
        """Execute agent with retry logic"""
        
        for attempt in range(self.retry_manager.max_retries + 1):
            try:
                start_time = time.time()
                
                # Execute the agent
                result = await agent.execute_safe(input_data, execution_context)
                
                execution_time = time.time() - start_time
                
                if result.success:
                    # Record success
                    circuit_breaker = self.get_circuit_breaker(agent_name)
                    circuit_breaker.record_success()
                    
                    # Update health status
                    self._update_agent_health(agent_name, True, execution_time)
                    
                    logger.info(f"Agent {agent_name} executed successfully"
                              f"{f' (attempt {attempt + 1})' if attempt > 0 else ''}")
                    return result
                else:
                    # Agent returned failure result
                    failure_type = self._classify_failure(result.error_message)
                    
                    if attempt < self.retry_manager.max_retries:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Agent {agent_name} failed (attempt {attempt + 1}), "
                                     f"retrying in {delay:.2f}s: {result.error_message}")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Max retries exceeded
                        return await self._handle_final_failure(
                            agent_name, result, failure_type, execution_context, input_data
                        )
                        
            except Exception as e:
                failure_type = self._classify_failure(str(e))
                
                if attempt < self.retry_manager.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Agent {agent_name} exception (attempt {attempt + 1}), "
                                 f"retrying in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Max retries exceeded
                    error_result = AgentResult(
                        success=False,
                        error_message=str(e),
                        error_code="EXECUTION_EXCEPTION"
                    )
                    return await self._handle_final_failure(
                        agent_name, error_result, failure_type, execution_context, input_data
                    )
    
    async def _handle_final_failure(
        self,
        agent_name: str,
        failed_result: AgentResult,
        failure_type: FailureType,
        execution_context: AgentExecutionContext,
        input_data: Dict[str, Any]
    ) -> AgentResult:
        """Handle final failure after all retries exhausted"""
        
        # Record failure in circuit breaker
        circuit_breaker = self.get_circuit_breaker(agent_name)
        circuit_breaker.record_failure()
        
        # Log failure
        failure_record = FailureRecord(
            agent_type=agent_name,
            failure_type=failure_type,
            error_message=failed_result.error_message or "Unknown error",
            timestamp=datetime.now(),
            execution_context=execution_context.__dict__ if execution_context else {}
        )
        
        self.failure_history.append(failure_record)
        self.recovery_metrics['total_failures'] += 1
        
        # Update health status
        self._update_agent_health(agent_name, False, 0)
        
        # Check for alerts
        await self._check_failure_alerts(agent_name, failure_record)
        
        # Log comprehensive failure information
        logger.error(f"Agent {agent_name} failed after all retries: {failed_result.error_message}")
        logger.error(f"Failure classification: {failure_type.value}")
        
        # Attempt fallback agents
        return await self._try_fallback_agents(agent_name, execution_context, input_data, failure_record)
    
    async def _try_fallback_agents(
        self,
        primary_agent_name: str,
        execution_context: AgentExecutionContext,
        input_data: Dict[str, Any],
        original_failure: Optional[FailureRecord] = None
    ) -> AgentResult:
        """Try fallback agents when primary agent fails"""
        
        fallback_agents = self.fallback_registry.get_fallback_agents(primary_agent_name)
        
        if not fallback_agents:
            logger.error(f"No fallback agents available for {primary_agent_name}")
            return AgentResult(
                success=False,
                error_message=f"Primary agent {primary_agent_name} failed and no fallback agents available",
                error_code="NO_FALLBACK_AVAILABLE"
            )
        
        logger.info(f"Attempting fallback agents for {primary_agent_name}: {fallback_agents}")
        
        for fallback_name in fallback_agents:
            try:
                fallback_agent = self.fallback_registry.get_fallback_instance(fallback_name)
                
                if fallback_agent is None:
                    logger.warning(f"Fallback agent {fallback_name} not registered")
                    continue
                
                # Check if fallback agent's circuit breaker allows execution
                fallback_circuit_breaker = self.get_circuit_breaker(fallback_name)
                if not fallback_circuit_breaker.can_execute():
                    logger.warning(f"Fallback agent {fallback_name} circuit breaker is OPEN")
                    continue
                
                logger.info(f"Executing fallback agent: {fallback_name}")
                
                # Execute fallback agent (no retry for fallbacks to prevent cascading delays)
                result = await fallback_agent.execute_safe(input_data, execution_context)
                
                if result.success:
                    # Record successful fallback
                    if original_failure:
                        original_failure.recovery_attempted = True
                        original_failure.recovery_successful = True
                        original_failure.recovery_strategy = RecoveryStrategy.FALLBACK_AGENT
                        original_failure.fallback_agent_used = fallback_name
                    
                    self.recovery_metrics['successful_recoveries'] += 1
                    self.recovery_metrics['fallback_activations'] += 1
                    
                    # Update result metadata to indicate fallback was used
                    result.metadata.update({
                        'primary_agent_failed': primary_agent_name,
                        'fallback_agent_used': fallback_name,
                        'recovery_strategy': 'fallback_agent'
                    })
                    
                    logger.info(f"Fallback agent {fallback_name} succeeded for {primary_agent_name}")
                    return result
                else:
                    logger.warning(f"Fallback agent {fallback_name} also failed: {result.error_message}")
                    continue
                    
            except Exception as e:
                logger.error(f"Fallback agent {fallback_name} exception: {str(e)}")
                continue
        
        # All fallback agents failed
        logger.error(f"All fallback agents failed for {primary_agent_name}")
        
        if original_failure:
            original_failure.recovery_attempted = True
            original_failure.recovery_successful = False
            original_failure.recovery_strategy = RecoveryStrategy.FALLBACK_AGENT
        
        return AgentResult(
            success=False,
            error_message=f"Primary agent {primary_agent_name} and all fallback agents failed",
            error_code="ALL_AGENTS_FAILED",
            metadata={
                'primary_agent': primary_agent_name,
                'attempted_fallbacks': fallback_agents,
                'recovery_strategy': 'fallback_exhausted'
            }
        )
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        import random
        
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.retry_manager.base_delay * (self.retry_manager.exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.retry_manager.max_delay)
        
        # Add jitter if enabled
        if self.retry_manager.jitter:
            delay = delay * (0.5 + random.random() * 0.5)  # Add +/- 25% jitter
        
        return delay
    
    def _classify_failure(self, error_message: str) -> FailureType:
        """Classify failure type based on error message"""
        if not error_message:
            return FailureType.UNKNOWN
        
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower or 'timed out' in error_lower:
            return FailureType.TIMEOUT
        elif 'rate limit' in error_lower or 'rate-limit' in error_lower:
            return FailureType.RATE_LIMIT
        elif 'authentication' in error_lower or 'auth' in error_lower:
            return FailureType.AUTHENTICATION
        elif 'network' in error_lower or 'connection' in error_lower:
            return FailureType.NETWORK_ERROR
        elif 'memory' in error_lower or 'out of memory' in error_lower:
            return FailureType.MEMORY_ERROR
        elif 'invalid' in error_lower or 'validation' in error_lower:
            return FailureType.INVALID_INPUT
        elif 'service' in error_lower or 'external' in error_lower:
            return FailureType.EXTERNAL_SERVICE
        else:
            return FailureType.UNKNOWN
    
    def _update_agent_health(self, agent_name: str, success: bool, execution_time: float):
        """Update agent health status"""
        now = datetime.now()
        
        if agent_name not in self.agent_health_status:
            self.agent_health_status[agent_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'average_execution_time': 0.0,
                'last_success': None,
                'last_failure': None,
                'current_streak': 0,
                'streak_type': 'none'  # 'success' or 'failure'
            }
        
        health = self.agent_health_status[agent_name]
        health['total_executions'] += 1
        
        if success:
            health['successful_executions'] += 1
            health['last_success'] = now
            
            # Update average execution time (simple running average)
            n = health['successful_executions']
            current_avg = health['average_execution_time']
            health['average_execution_time'] = ((current_avg * (n-1)) + execution_time) / n
            
            # Update streak
            if health['streak_type'] == 'success':
                health['current_streak'] += 1
            else:
                health['current_streak'] = 1
                health['streak_type'] = 'success'
        else:
            health['last_failure'] = now
            
            # Update streak
            if health['streak_type'] == 'failure':
                health['current_streak'] += 1
            else:
                health['current_streak'] = 1
                health['streak_type'] = 'failure'
    
    async def _check_failure_alerts(self, agent_name: str, failure_record: FailureRecord):
        """Check if failure conditions warrant alerts"""
        
        # Check repeated failures
        recent_failures = [
            f for f in list(self.failure_history)[-10:]  # Last 10 failures
            if f.agent_type == agent_name
        ]
        
        if len(recent_failures) >= self.alert_thresholds['repeated_failure_count']:
            await self._send_alert(
                f"REPEATED_FAILURES",
                f"Agent {agent_name} has {len(recent_failures)} recent failures",
                {
                    'agent_name': agent_name,
                    'failure_count': len(recent_failures),
                    'recent_failures': [f.error_message for f in recent_failures[-3:]]
                }
            )
        
        # Check circuit breaker status
        circuit_breaker = self.get_circuit_breaker(agent_name)
        if (circuit_breaker.state == CircuitState.OPEN and 
            circuit_breaker.last_failure_time and
            (datetime.now() - circuit_breaker.last_failure_time).seconds > 
            self.alert_thresholds['circuit_breaker_duration']):
            
            self.recovery_metrics['circuit_breaker_trips'] += 1
            
            await self._send_alert(
                f"CIRCUIT_BREAKER_OPEN",
                f"Circuit breaker for {agent_name} has been open for extended period",
                {
                    'agent_name': agent_name,
                    'state': circuit_breaker.state.value,
                    'failure_count': circuit_breaker.failure_count,
                    'duration_minutes': (datetime.now() - circuit_breaker.last_failure_time).seconds // 60
                }
            )
    
    async def _send_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """Send alert for critical failures"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'message': message,
            'details': details,
            'system': 'agent_failure_recovery'
        }
        
        logger.critical(f"AGENT FAILURE ALERT: {message}", extra=alert_data)
        
        # Send to monitoring/alerting system if available
        try:
            metrics.counter('agent_failure_alerts_total', 1, tags={'alert_type': alert_type})
        except:
            pass  # Metrics not available
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all agents"""
        now = datetime.now()
        
        health_summary = {
            'overall_metrics': self.recovery_metrics,
            'agent_health': {},
            'circuit_breakers': {},
            'recent_failures': [],
            'timestamp': now.isoformat()
        }
        
        # Agent health details
        for agent_name, health in self.agent_health_status.items():
            success_rate = (health['successful_executions'] / health['total_executions']) if health['total_executions'] > 0 else 0
            
            health_summary['agent_health'][agent_name] = {
                **health,
                'success_rate': success_rate,
                'last_success_ago': (now - health['last_success']).seconds if health['last_success'] else None,
                'last_failure_ago': (now - health['last_failure']).seconds if health['last_failure'] else None
            }
        
        # Circuit breaker status
        for agent_name, cb in self.circuit_breakers.items():
            health_summary['circuit_breakers'][agent_name] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count,
                'last_failure_ago': (now - cb.last_failure_time).seconds if cb.last_failure_time else None
            }
        
        # Recent failures
        health_summary['recent_failures'] = [
            {
                'agent_type': f.agent_type,
                'failure_type': f.failure_type.value,
                'error_message': f.error_message,
                'timestamp': f.timestamp.isoformat(),
                'recovery_successful': f.recovery_successful,
                'fallback_used': f.fallback_agent_used
            }
            for f in list(self.failure_history)[-10:]  # Last 10 failures
        ]
        
        return health_summary


# Global instance
agent_failure_recovery = AgentFailureRecoverySystem()


# Convenience functions for easy integration
async def execute_agent_with_recovery(
    agent: BaseAgent,
    input_data: Dict[str, Any],
    execution_context: Optional[AgentExecutionContext] = None
) -> AgentResult:
    """
    Convenience function to execute agent with full failure recovery
    
    Usage:
        result = await execute_agent_with_recovery(writer_agent, content_data)
    """
    if execution_context is None:
        execution_context = AgentExecutionContext(
            request_id=f"recovery_{int(time.time())}",
            user_id="system"
        )
    
    return await agent_failure_recovery.execute_with_recovery(
        agent, execution_context, input_data
    )


def register_fallback_agent(primary_agent_name: str, fallback_agents: List[str]):
    """
    Register fallback agents for a primary agent
    
    Usage:
        register_fallback_agent('WriterAgent', ['SimpleWriterAgent', 'BasicContentGenerator'])
    """
    agent_failure_recovery.fallback_registry.register_fallback(primary_agent_name, fallback_agents)


def register_agent_instance(agent_name: str, agent_instance: BaseAgent):
    """
    Register an agent instance for fallback use
    
    Usage:
        register_agent_instance('SimpleWriterAgent', simple_writer_instance)
    """
    agent_failure_recovery.fallback_registry.register_agent_instance(agent_name, agent_instance)


def get_agent_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status of all agents
    
    Usage:
        health = get_agent_health_status()
        print(f"Overall success rate: {health['overall_metrics']['successful_recoveries']}")
    """
    return agent_failure_recovery.get_health_status()