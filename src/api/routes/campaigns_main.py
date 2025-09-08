"""
Main Campaigns Router
Aggregates all campaign-related routes into a single router.
This replaces the monolithic campaigns.py file with modular organization.
"""

from fastapi import APIRouter

# Import all campaign sub-routers
from .campaigns.workflow import router as workflow_router
from .campaigns.crud import router as crud_router
from .campaigns.agents import router as agents_router
from .campaigns.tasks import router as tasks_router
from .campaigns.orchestration import router as orchestration_router
from .campaigns.scheduling import router as scheduling_router
from .campaigns.autonomous import router as autonomous_router
from .campaigns.testing import router as testing_router

# Create main campaigns router
router = APIRouter(tags=["campaigns"])

# Include all sub-routers with appropriate prefixes where needed

# Workflow management routes (no prefix needed)
router.include_router(workflow_router)

# CRUD operations (no prefix needed for basic operations)
router.include_router(crud_router)

# Agent operations (no prefix needed)
router.include_router(agents_router)

# Task management (no prefix needed)
router.include_router(tasks_router)

# Orchestration routes with prefix
router.include_router(orchestration_router, prefix="/orchestration")

# Scheduling and distribution (no prefix needed)
router.include_router(scheduling_router)

# Autonomous workflows with prefix
router.include_router(autonomous_router, prefix="/autonomous")

# Testing and debug routes (no prefix needed)
router.include_router(testing_router)

# Export the main router
__all__ = ["router"]