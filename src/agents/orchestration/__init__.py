"""
Campaign Orchestration Package

This package contains the core campaign orchestration components for the
CrediLinq Content Agent platform, providing campaign-centric workflow
management and agent coordination.

Key Components:
- CampaignOrchestratorAgent: Main orchestrator for campaign workflows
- CampaignDatabaseService: Extended database service for campaign operations
- WorkflowStateManager: State persistence and recovery for workflows

Architecture:
The orchestration system transforms the platform from content-centric to
campaign-centric by providing centralized workflow coordination, task
distribution, and progress monitoring across multiple specialized agents.

Usage:
    from src.agents.orchestration import CampaignOrchestratorAgent
    
    # Create and configure orchestrator
    orchestrator = CampaignOrchestratorAgent()
    
    # Execute campaign workflow
    result = await orchestrator.orchestrate_campaign(campaign_id)
"""

from .campaign_orchestrator import (
    CampaignOrchestratorAgent,
    CampaignType,
    CampaignTask,
    TaskStatus,
    WorkflowStatus,
    CampaignWithTasks,
    WorkflowExecutionCreate
)

from .campaign_database_service import (
    CampaignDatabaseService,
    AgentPerformanceMetrics
)

from .workflow_state_manager import (
    WorkflowStateManager,
    CampaignWorkflowState
)

__version__ = "1.0.0"
__author__ = "CrediLinq Development Team"

__all__ = [
    # Main orchestrator
    "CampaignOrchestratorAgent",
    
    # Campaign types and data structures
    "CampaignType",
    "CampaignTask", 
    "TaskStatus",
    "WorkflowStatus",
    "CampaignWithTasks",
    "WorkflowExecutionCreate",
    
    # Database service
    "CampaignDatabaseService",
    "AgentPerformanceMetrics",
    
    # State management
    "WorkflowStateManager",
    "CampaignWorkflowState"
]