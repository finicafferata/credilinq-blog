"""
Lazy agent loading system for Railway deployment optimization.
Agents are only loaded when first requested to avoid startup delays.
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

class LazyAgentLoader:
    """Singleton class for lazy loading agents on first request."""
    
    _instance: Optional['LazyAgentLoader'] = None
    _agents_loaded: bool = False
    _factory: Optional['AgentFactory'] = None
    _loading_lock: asyncio.Lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def ensure_agents_loaded(self) -> 'AgentFactory':
        """Ensure agents are loaded, loading them if necessary."""
        if self._agents_loaded and self._factory:
            return self._factory
            
        async with self._loading_lock:
            # Double-check pattern
            if self._agents_loaded and self._factory:
                return self._factory
                
            logger.info("🚀 Lazy loading agents on first request...")
            start_time = time.time()
            
            try:
                # Import agents to trigger registration
                from src.agents import specialized
                from src.agents.core.agent_factory import AgentFactory
                
                self._factory = AgentFactory()
                agent_types = self._factory.get_available_types()
                
                load_time = time.time() - start_time
                logger.info(f"✅ Lazy loaded {len(agent_types)} agent types in {load_time:.2f}s")
                
                self._agents_loaded = True
                return self._factory
                
            except Exception as e:
                logger.error(f"❌ Failed to lazy load agents: {e}")
                # Return a minimal factory to prevent crashes
                from src.agents.core.agent_factory import AgentFactory
                self._factory = AgentFactory()
                return self._factory
    
    @property
    def is_loaded(self) -> bool:
        """Check if agents are already loaded."""
        return self._agents_loaded
    
    def get_factory_if_loaded(self) -> Optional['AgentFactory']:
        """Get factory if already loaded, None otherwise."""
        return self._factory if self._agents_loaded else None

# Global instance
lazy_agent_loader = LazyAgentLoader()

@lru_cache(maxsize=1)
def get_agent_status() -> Dict[str, Any]:
    """Get current agent loading status (cached)."""
    return {
        "agents_loaded": lazy_agent_loader.is_loaded,
        "factory_available": lazy_agent_loader.get_factory_if_loaded() is not None,
        "railway_mode": os.environ.get('RAILWAY_ENVIRONMENT') is not None
    }