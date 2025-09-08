"""
Campaign Routes Module
Modular campaign API routes organized by functionality.
"""

from .workflow import router as workflow_router
from .crud import router as crud_router
from .agents import router as agents_router
from .tasks import router as tasks_router
from .orchestration import router as orchestration_router
from .scheduling import router as scheduling_router
from .autonomous import router as autonomous_router
from .testing import router as testing_router

__all__ = [
    "workflow_router",
    "crud_router", 
    "agents_router",
    "tasks_router",
    "orchestration_router",
    "scheduling_router",
    "autonomous_router",
    "testing_router"
]