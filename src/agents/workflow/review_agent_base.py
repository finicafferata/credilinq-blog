"""
Simple base class for review agents to avoid circular imports
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReviewAgentBase(ABC):
    """
    Simple base class for review workflow agents.
    Avoids circular imports with the main BaseAgent hierarchy.
    """
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute_safe(self, content_data: Dict[str, Any], **kwargs):
        """Execute the review analysis safely"""
        pass