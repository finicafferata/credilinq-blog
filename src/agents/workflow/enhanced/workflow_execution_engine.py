"""
Workflow Execution Engine with Error Recovery

This module provides robust workflow execution capabilities with error handling,
recovery mechanisms, and integration with the enhanced state management system.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

from langgraph.graph.state import CompiledStateGraph

from .enhanced_workflow_state import (
    EnhancedWorkflowState, CampaignWorkflowState, WorkflowStatus,
    StateManager, WorkflowCheckpoint, update_state_node, add_state_error
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Workflow execution modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STEP_BY_STEP = "step_by_step"
    DEBUG = "debug"


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY_NODE = "retry_node"
    SKIP_NODE = "skip_node"
    ROLLBACK_CHECKPOINT = "rollback_checkpoint"
    TERMINATE = "terminate"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    execution_id: str
    campaign_id: str
    workflow_id: str
    mode: ExecutionMode
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    checkpoint_interval: int = 5  # Create checkpoint every N nodes
    error_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY_NODE
    debug_mode: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of workflow execution."""
    execution_id: str
    success: bool
    final_state: Union[EnhancedWorkflowState, CampaignWorkflowState]
    execution_time: float
    nodes_executed: List[str]
    errors: List[Dict[str, Any]]
    checkpoints_created: List[str]
    recovery_actions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowExecutionEngine:
    """
    Advanced workflow execution engine with error recovery and checkpointing.
    
    This engine provides robust execution of LangGraph workflows with:
    - Error handling and recovery
    - State checkpointing and restoration
    - Execution monitoring and debugging
    - Integration with campaign orchestration
    """
    
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager()
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_callbacks: Dict[str, List[Callable]] = {}
        
    async def execute_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> ExecutionResult:
        """
        Execute a workflow with comprehensive error handling and monitoring.
        
        Args:
            workflow: Compiled LangGraph workflow to execute
            initial_state: Initial state for workflow execution
            context: Execution context with configuration
            
        Returns:
            Execution result with success status and final state
        """
        start_time = datetime.now()
        self.active_executions[context.execution_id] = context
        
        try:
            logger.info(f"Starting workflow execution: {context.execution_id}")
            
            # Initialize execution state
            execution_state = await self._initialize_execution_state(initial_state, context)
            
            # Execute workflow based on mode
            if context.mode == ExecutionMode.STEP_BY_STEP:
                final_state = await self._execute_step_by_step(workflow, execution_state, context)
            elif context.mode == ExecutionMode.DEBUG:
                final_state = await self._execute_debug_mode(workflow, execution_state, context)
            else:
                final_state = await self._execute_standard(workflow, execution_state, context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create final result
            result = ExecutionResult(
                execution_id=context.execution_id,
                success=final_state.get('status') == WorkflowStatus.COMPLETED,
                final_state=final_state,
                execution_time=execution_time,
                nodes_executed=final_state.get('execution_path', []),
                errors=final_state.get('errors', []),
                checkpoints_created=[cp.checkpoint_id for cp in final_state.get('checkpoints', [])],
                recovery_actions=final_state.get('recovery_actions', []),
                metadata={
                    'execution_mode': context.mode.value,
                    'total_nodes': len(final_state.get('execution_path', [])),
                    'checkpoint_count': len(final_state.get('checkpoints', [])),
                    'retry_count': final_state.get('retry_count', 0)
                }
            )
            
            logger.info(f"Workflow execution completed: {context.execution_id}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {context.execution_id} - {str(e)}")
            
            # Create error result
            execution_time = (datetime.now() - start_time).total_seconds()
            error_state = initial_state.copy()
            error_state = add_state_error(error_state, str(e))
            error_state['status'] = WorkflowStatus.FAILED
            
            return ExecutionResult(
                execution_id=context.execution_id,
                success=False,
                final_state=error_state,
                execution_time=execution_time,
                nodes_executed=error_state.get('execution_path', []),
                errors=error_state.get('errors', []),
                checkpoints_created=[],
                recovery_actions=[],
                metadata={'execution_failed': True, 'error': str(e)}
            )
            
        finally:
            # Cleanup
            if context.execution_id in self.active_executions:
                del self.active_executions[context.execution_id]
                
    async def _initialize_execution_state(
        self,
        initial_state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Initialize execution state with context information."""
        execution_state = initial_state.copy()
        execution_state.update({
            'execution_id': context.execution_id,
            'status': WorkflowStatus.RUNNING,
            'execution_context': context,
            'recovery_actions': [],
            'started_at': datetime.now()
        })
        
        # Create initial checkpoint if needed
        if context.checkpoint_interval > 0:
            await self._create_checkpoint(execution_state, "execution_start", is_recovery_point=True)
            
        return execution_state
        
    async def _execute_standard(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Execute workflow in standard mode."""
        try:
            # Set execution timeout if specified
            if context.timeout_seconds:
                result = await asyncio.wait_for(
                    self._invoke_workflow_with_monitoring(workflow, state, context),
                    timeout=context.timeout_seconds
                )
            else:
                result = await self._invoke_workflow_with_monitoring(workflow, state, context)
                
            result['status'] = WorkflowStatus.COMPLETED
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Workflow execution timed out: {context.execution_id}")
            state = add_state_error(state, "Workflow execution timed out")
            state['status'] = WorkflowStatus.FAILED
            return state
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return await self._handle_execution_error(state, str(e), context)
            
    async def _execute_step_by_step(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Execute workflow step by step with manual control."""
        logger.info(f"Starting step-by-step execution: {context.execution_id}")
        
        # This would require a more complex implementation with user interaction
        # For now, we'll execute with checkpoints at each node
        current_state = state
        
        try:
            # Get workflow nodes (this is a simplified approach)
            # In a real implementation, you'd need to traverse the workflow graph
            nodes_to_execute = self._get_workflow_nodes(workflow)
            
            for i, node_id in enumerate(nodes_to_execute):
                logger.info(f"Executing node {i+1}/{len(nodes_to_execute)}: {node_id}")
                
                # Create checkpoint before node execution
                await self._create_checkpoint(current_state, f"before_{node_id}")
                
                # Execute single node (simplified - actual implementation would be more complex)
                current_state = await self._execute_single_node(workflow, current_state, node_id, context)
                
                # Create checkpoint after node execution
                await self._create_checkpoint(current_state, f"after_{node_id}")
                
                # Check for user intervention points
                if context.debug_mode:
                    await self._debug_checkpoint(current_state, node_id)
                    
            current_state['status'] = WorkflowStatus.COMPLETED
            return current_state
            
        except Exception as e:
            logger.error(f"Step-by-step execution error: {str(e)}")
            return await self._handle_execution_error(current_state, str(e), context)
            
    async def _execute_debug_mode(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Execute workflow in debug mode with detailed logging."""
        logger.info(f"Starting debug mode execution: {context.execution_id}")
        
        # Enable detailed logging
        debug_logger = logging.getLogger(f"debug.{context.execution_id}")
        debug_logger.setLevel(logging.DEBUG)
        
        try:
            # Execute with detailed monitoring
            result = await self._invoke_workflow_with_detailed_monitoring(workflow, state, context)
            result['status'] = WorkflowStatus.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Debug execution error: {str(e)}")
            return await self._handle_execution_error(state, str(e), context)
            
    async def _invoke_workflow_with_monitoring(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Invoke workflow with basic monitoring."""
        
        # Create a wrapper that adds monitoring to the state
        def monitor_wrapper(original_state):
            # Add monitoring information
            monitored_state = original_state.copy()
            monitored_state['monitoring_enabled'] = True
            monitored_state['execution_context'] = context
            return monitored_state
            
        # Execute workflow
        monitored_state = monitor_wrapper(state)
        result = await asyncio.create_task(
            self._execute_workflow_async(workflow, monitored_state)
        )
        
        return result
        
    async def _invoke_workflow_with_detailed_monitoring(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Invoke workflow with detailed monitoring and logging."""
        
        debug_logger = logging.getLogger(f"debug.{context.execution_id}")
        
        def detailed_monitor_wrapper(original_state):
            monitored_state = original_state.copy()
            monitored_state['debug_mode'] = True
            monitored_state['detailed_logging'] = True
            monitored_state['execution_context'] = context
            
            # Log detailed state information
            debug_logger.debug(f"State snapshot: {monitored_state}")
            
            return monitored_state
            
        monitored_state = detailed_monitor_wrapper(state)
        result = await asyncio.create_task(
            self._execute_workflow_async(workflow, monitored_state)
        )
        
        return result
        
    async def _execute_workflow_async(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState]
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Execute workflow asynchronously."""
        # This is a simplified implementation
        # In reality, you'd need to properly invoke the LangGraph workflow
        
        try:
            # Convert state to format expected by LangGraph
            langgraph_state = self._convert_to_langgraph_state(state)
            
            # Execute workflow (this would be the actual LangGraph invocation)
            # For now, we'll simulate execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Convert result back to our enhanced state format
            result = self._convert_from_langgraph_state(langgraph_state, state)
            
            return result
            
        except Exception as e:
            logger.error(f"Async workflow execution error: {str(e)}")
            raise
            
    async def _execute_single_node(
        self,
        workflow: CompiledStateGraph,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        node_id: str,
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Execute a single node in the workflow."""
        # This is a placeholder for single node execution
        # Actual implementation would require LangGraph integration
        
        try:
            # Update state to reflect current node
            state = update_state_node(state, node_id)
            
            # Simulate node execution
            await asyncio.sleep(0.05)
            
            # Add node-specific processing results
            state['state_data'][f'{node_id}_completed'] = True
            state['state_data'][f'{node_id}_timestamp'] = datetime.now().isoformat()
            
            return state
            
        except Exception as e:
            logger.error(f"Single node execution error: {node_id} - {str(e)}")
            return add_state_error(state, f"Node {node_id} failed: {str(e)}", node_id)
            
    async def _handle_execution_error(
        self,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        error: str,
        context: ExecutionContext
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Handle execution errors based on recovery strategy."""
        
        state = add_state_error(state, error)
        recovery_action = {
            'strategy': context.error_strategy.value,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'node_id': state.get('current_node', '')
        }
        
        if context.error_strategy == ErrorRecoveryStrategy.RETRY_NODE:
            if state.get('retry_count', 0) < context.max_retries:
                state['retry_count'] = state.get('retry_count', 0) + 1
                recovery_action['action'] = 'retry_attempted'
                logger.info(f"Retrying node execution (attempt {state['retry_count']})")
                # In a real implementation, you'd retry the failed node
            else:
                state['status'] = WorkflowStatus.FAILED
                recovery_action['action'] = 'max_retries_exceeded'
                
        elif context.error_strategy == ErrorRecoveryStrategy.ROLLBACK_CHECKPOINT:
            latest_checkpoint = self.state_manager.get_latest_checkpoint(
                state['campaign_id'], state['workflow_id']
            )
            if latest_checkpoint:
                restored_state = self.state_manager.restore_from_checkpoint(
                    latest_checkpoint.checkpoint_id
                )
                if restored_state:
                    state.update(restored_state)
                    recovery_action['action'] = 'rollback_successful'
                    recovery_action['checkpoint_id'] = latest_checkpoint.checkpoint_id
                else:
                    state['status'] = WorkflowStatus.FAILED
                    recovery_action['action'] = 'rollback_failed'
            else:
                state['status'] = WorkflowStatus.FAILED
                recovery_action['action'] = 'no_checkpoint_available'
                
        elif context.error_strategy == ErrorRecoveryStrategy.SKIP_NODE:
            recovery_action['action'] = 'node_skipped'
            logger.warning(f"Skipping failed node: {state.get('current_node', '')}")
            
        elif context.error_strategy == ErrorRecoveryStrategy.TERMINATE:
            state['status'] = WorkflowStatus.FAILED
            recovery_action['action'] = 'execution_terminated'
            
        elif context.error_strategy == ErrorRecoveryStrategy.MANUAL_INTERVENTION:
            state['status'] = WorkflowStatus.PAUSED
            recovery_action['action'] = 'manual_intervention_required'
            
        # Add recovery action to state
        if 'recovery_actions' not in state:
            state['recovery_actions'] = []
        state['recovery_actions'].append(recovery_action)
        
        return state
        
    async def _create_checkpoint(
        self,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        checkpoint_name: str,
        is_recovery_point: bool = False
    ) -> WorkflowCheckpoint:
        """Create a checkpoint for the current state."""
        
        checkpoint = self.state_manager.create_checkpoint(
            campaign_id=state['campaign_id'],
            workflow_id=state['workflow_id'],
            node_id=state.get('current_node', checkpoint_name),
            state_data=state.copy(),
            execution_path=state.get('execution_path', []),
            is_recovery_point=is_recovery_point,
            metadata={'checkpoint_name': checkpoint_name}
        )
        
        # Add checkpoint to state
        if 'checkpoints' not in state:
            state['checkpoints'] = []
        state['checkpoints'].append(checkpoint)
        state['last_checkpoint'] = checkpoint.checkpoint_id
        
        if is_recovery_point:
            state['recovery_point'] = checkpoint.checkpoint_id
            
        return checkpoint
        
    async def _debug_checkpoint(
        self,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState],
        node_id: str
    ):
        """Debug checkpoint for manual inspection."""
        debug_logger = logging.getLogger(f"debug.{state.get('execution_id', 'unknown')}")
        debug_logger.info(f"Debug checkpoint at node: {node_id}")
        debug_logger.debug(f"Current state: {state}")
        
        # In a real implementation, this might pause for user input
        await asyncio.sleep(0.1)
        
    def _get_workflow_nodes(self, workflow: CompiledStateGraph) -> List[str]:
        """Get list of nodes in the workflow (simplified implementation)."""
        # This is a placeholder - actual implementation would inspect the workflow graph
        return ["node1", "node2", "node3"]
        
    def _convert_to_langgraph_state(
        self,
        state: Union[EnhancedWorkflowState, CampaignWorkflowState]
    ) -> Dict[str, Any]:
        """Convert enhanced state to LangGraph-compatible format."""
        # Extract the core fields that LangGraph expects
        langgraph_state = {}
        
        # Copy over standard blog workflow fields if they exist
        blog_fields = [
            'blog_title', 'company_context', 'content_type', 'outline',
            'research', 'geo_metadata', 'draft', 'review_notes', 'final_post'
        ]
        
        for field in blog_fields:
            if field in state:
                langgraph_state[field] = state[field]
                
        # Add any additional state data
        if 'state_data' in state:
            langgraph_state.update(state['state_data'])
            
        return langgraph_state
        
    def _convert_from_langgraph_state(
        self,
        langgraph_result: Dict[str, Any],
        original_state: Union[EnhancedWorkflowState, CampaignWorkflowState]
    ) -> Union[EnhancedWorkflowState, CampaignWorkflowState]:
        """Convert LangGraph result back to enhanced state format."""
        # Update original state with LangGraph results
        result_state = original_state.copy()
        
        # Update with LangGraph results
        result_state.update(langgraph_result)
        
        # Ensure enhanced state fields are preserved
        result_state['updated_at'] = datetime.now()
        
        return result_state
        
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an execution."""
        if execution_id not in self.active_executions:
            return None
            
        context = self.active_executions[execution_id]
        return {
            'execution_id': execution_id,
            'campaign_id': context.campaign_id,
            'workflow_id': context.workflow_id,
            'mode': context.mode.value,
            'status': 'running'
        }
        
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active execution."""
        if execution_id not in self.active_executions:
            return False
            
        # Implementation would depend on the execution framework
        logger.info(f"Pausing execution: {execution_id}")
        return True
        
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        if execution_id not in self.active_executions:
            return False
            
        logger.info(f"Resuming execution: {execution_id}")
        return True
        
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id not in self.active_executions:
            return False
            
        context = self.active_executions[execution_id]
        del self.active_executions[execution_id]
        
        logger.info(f"Cancelled execution: {execution_id}")
        return True