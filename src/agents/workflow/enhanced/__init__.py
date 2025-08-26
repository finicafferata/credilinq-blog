"""
Enhanced Workflow Components for Campaign-Centric Architecture

This package provides advanced LangGraph integration components that extend
the existing blog workflow with campaign orchestration capabilities.

Key Components:
- CampaignWorkflowBuilder: Dynamic workflow graph construction
- EnhancedWorkflowState: Advanced state management with checkpointing
- WorkflowExecutionEngine: Robust execution with error recovery
- CampaignStateManager: Campaign-specific state persistence
"""

from .campaign_workflow_builder import (
    CampaignWorkflowBuilder,
    WorkflowTemplate,
    NodeDefinition,
    EdgeDefinition
)

from .enhanced_workflow_state import (
    EnhancedWorkflowState,
    CampaignWorkflowState,
    WorkflowCheckpoint,
    StateSnapshot
)

from .workflow_execution_engine import (
    WorkflowExecutionEngine,
    ExecutionContext,
    ExecutionResult,
    ErrorRecoveryStrategy
)

from .campaign_state_manager import (
    CampaignStateManager,
    StateTransition,
    PersistenceConfig
)

__version__ = "1.0.0"
__author__ = "CrediLinq Development Team"

__all__ = [
    # Workflow Builder
    "CampaignWorkflowBuilder",
    "WorkflowTemplate", 
    "NodeDefinition",
    "EdgeDefinition",
    
    # Enhanced State Management
    "EnhancedWorkflowState",
    "CampaignWorkflowState",
    "WorkflowCheckpoint",
    "StateSnapshot",
    
    # Execution Engine
    "WorkflowExecutionEngine",
    "ExecutionContext",
    "ExecutionResult", 
    "ErrorRecoveryStrategy",
    
    # State Management
    "CampaignStateManager",
    "StateTransition",
    "PersistenceConfig"
]