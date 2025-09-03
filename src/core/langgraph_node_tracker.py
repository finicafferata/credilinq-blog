"""
LangGraph Node Performance Tracking - Decorator and utilities for tracking individual workflow nodes.
Provides granular insights into decision-making and performance at each step.
"""

import asyncio
import time
import logging
import functools
from typing import Dict, Any, Optional, Callable, TypeVar
from datetime import datetime
from dataclasses import dataclass

from .gemini_performance_tracker import global_gemini_tracker

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class NodeExecutionMetrics:
    """Metrics for individual node execution."""
    node_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0
    input_size: int = 0
    output_size: int = 0
    decision_count: int = 0
    confidence_scores: list = None
    error_occurred: bool = False
    error_message: Optional[str] = None

class LangGraphNodeTracker:
    """
    Performance tracker for individual LangGraph workflow nodes.
    Captures execution metrics and decision-making data at node level.
    """
    
    def __init__(self):
        self.active_executions: Dict[str, NodeExecutionMetrics] = {}
    
    def start_node_tracking(
        self,
        node_name: str,
        workflow_id: str,
        agent_name: str,
        input_data: Any
    ) -> str:
        """Start tracking a workflow node execution."""
        execution_id = f"node_{workflow_id}_{node_name}_{int(time.time() * 1000)}"
        
        metrics = NodeExecutionMetrics(
            node_name=node_name,
            start_time=datetime.utcnow(),
            input_size=len(str(input_data)) if input_data else 0
        )
        
        self.active_executions[execution_id] = metrics
        
        logger.debug(f"Started node tracking: {node_name} ({execution_id})")
        return execution_id
    
    def complete_node_tracking(
        self,
        execution_id: str,
        output_data: Any,
        decisions: Optional[list] = None,
        error: Optional[Exception] = None
    ) -> Optional[NodeExecutionMetrics]:
        """Complete tracking of a workflow node execution."""
        if execution_id not in self.active_executions:
            logger.warning(f"Node tracking not found: {execution_id}")
            return None
        
        metrics = self.active_executions[execution_id]
        metrics.end_time = datetime.utcnow()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        metrics.output_size = len(str(output_data)) if output_data else 0
        
        if error:
            metrics.error_occurred = True
            metrics.error_message = str(error)
        
        if decisions:
            metrics.decision_count = len(decisions)
            metrics.confidence_scores = [
                d.get('confidence_score', 0) for d in decisions 
                if isinstance(d, dict) and 'confidence_score' in d
            ]
        
        # Store completed metrics for reporting
        completed_metrics = metrics
        del self.active_executions[execution_id]
        
        logger.debug(f"Completed node tracking: {metrics.node_name} ({metrics.duration_ms:.2f}ms)")
        return completed_metrics

# Global node tracker instance
global_node_tracker = LangGraphNodeTracker()

def track_workflow_node(
    node_name: str,
    agent_name: Optional[str] = None,
    capture_decisions: bool = True,
    async_tracking: bool = True
):
    """
    Decorator for tracking LangGraph workflow node execution.
    
    Args:
        node_name: Name of the workflow node
        agent_name: Name of the agent executing the node
        capture_decisions: Whether to capture decision data from the result
        async_tracking: Whether to use async tracking (recommended)
    
    Usage:
        @track_workflow_node("generate_content", "writer_agent")
        async def generate_content_node(state: WorkflowState) -> WorkflowState:
            # Node implementation
            return updated_state
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            # Extract workflow context from state (assuming first arg is state)
            workflow_id = "unknown"
            agent_context = agent_name or "unknown_agent"
            
            if args and hasattr(args[0], 'get'):
                state = args[0]
                workflow_id = state.get('workflow_id', 'unknown')
                if 'agent_name' in state:
                    agent_context = state['agent_name']
            
            # Start tracking
            execution_id = global_node_tracker.start_node_tracking(
                node_name, workflow_id, agent_context, args[0] if args else None
            )
            
            start_time = time.time()
            
            try:
                # Execute the node function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Extract decisions if enabled and available
                decisions = None
                if capture_decisions and hasattr(result, 'get'):
                    # Look for decisions in various formats
                    if 'decisions' in result:
                        decisions = result['decisions']
                    elif 'quality_assessment' in result:
                        qa = result['quality_assessment']
                        if isinstance(qa, dict) and 'decisions' in qa:
                            decisions = qa['decisions']
                
                # Complete tracking
                global_node_tracker.complete_node_tracking(
                    execution_id, result, decisions
                )
                
                return result
                
            except Exception as e:
                # Complete tracking with error
                global_node_tracker.complete_node_tracking(
                    execution_id, None, error=e
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # Similar to async but for sync functions
            workflow_id = "unknown"
            agent_context = agent_name or "unknown_agent"
            
            if args and hasattr(args[0], 'get'):
                state = args[0]
                workflow_id = state.get('workflow_id', 'unknown')
            
            execution_id = global_node_tracker.start_node_tracking(
                node_name, workflow_id, agent_context, args[0] if args else None
            )
            
            try:
                result = func(*args, **kwargs)
                
                decisions = None
                if capture_decisions and hasattr(result, 'get'):
                    if 'decisions' in result:
                        decisions = result['decisions']
                
                global_node_tracker.complete_node_tracking(
                    execution_id, result, decisions
                )
                
                return result
                
            except Exception as e:
                global_node_tracker.complete_node_tracking(
                    execution_id, None, error=e
                )
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_agent_decision(
    execution_id: str,
    decision_point: str,
    reasoning: str,
    confidence_score: float,
    alternatives: list,
    business_impact: str,
    **kwargs
):
    """
    Track a specific agent decision within a workflow node.
    
    Args:
        execution_id: Current tracking execution ID
        decision_point: What decision was made
        reasoning: Why this decision was made
        confidence_score: Confidence level (0.0-1.0)
        alternatives: Other options considered
        business_impact: Expected business impact
        **kwargs: Additional decision metadata
    """
    # Create task for async decision tracking
    asyncio.create_task(_track_decision_async(
        execution_id, decision_point, reasoning, confidence_score,
        alternatives, business_impact, kwargs
    ))


async def _track_decision_async(
    execution_id: str,
    decision_point: str,
    reasoning: str,
    confidence_score: float,
    alternatives: list,
    business_impact: str,
    metadata: dict
):
    """Async helper for decision tracking."""
    try:
        # This would integrate with your agent_decisions table
        # For now, just log the decision
        logger.info(f"Decision tracked: {decision_point} (confidence: {confidence_score})")
        logger.debug(f"Reasoning: {reasoning}")
        
        # If you have the langgraph performance tracker available, use it
        try:
            from .langgraph_performance_tracker import global_performance_tracker
            await global_performance_tracker.track_decision(
                execution_id=execution_id,
                decision_point=decision_point,
                input_data={'context': 'workflow_node'},
                output_data={'reasoning': reasoning},
                reasoning=reasoning,
                confidence_score=confidence_score,
                alternatives_considered=alternatives,
                execution_time_ms=0,  # Would need to track this separately
                metadata={
                    'business_impact': business_impact,
                    **metadata
                }
            )
        except ImportError:
            logger.debug("LangGraph performance tracker not available for decision tracking")
    except Exception as e:
        logger.error(f"Error tracking decision: {e}")


class WorkflowNodeMetrics:
    """
    Utility class for collecting and reporting workflow node metrics.
    """
    
    @staticmethod
    def get_node_performance_summary(workflow_id: str) -> Dict[str, Any]:
        """Get performance summary for all nodes in a workflow."""
        # This would query your database for node performance data
        # For now, return a placeholder structure
        return {
            "workflow_id": workflow_id,
            "total_nodes": 0,
            "avg_execution_time_ms": 0,
            "success_rate": 100,
            "decision_count": 0,
            "avg_confidence_score": 0,
            "node_details": []
        }
    
    @staticmethod
    def get_agent_node_performance(agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent's nodes."""
        return {
            "agent_name": agent_name,
            "nodes_executed": 0,
            "avg_execution_time_ms": 0,
            "success_rate": 100,
            "most_used_nodes": [],
            "performance_trends": []
        }