"""
Agent Plugin System

This package provides a flexible plugin architecture for extending the agent
system with custom implementations and specialized functionality.

Key Components:
- AgentPluginManager: Dynamic plugin loading and management
- AgentPlugin: Base class for all agent plugins
- PluginInfo: Plugin metadata and information

Usage:
    from src.agents.plugins import AgentPluginManager, AgentPlugin
    
    # Create plugin manager
    manager = AgentPluginManager()
    
    # Discover and load plugins
    plugins = await manager.discover_plugins()
    for plugin_name in plugins:
        await manager.load_plugin(plugin_name)
    
    # Create agent from plugin
    agent = await manager.create_agent_from_plugin(
        "my_plugin", 
        {"config": "value"}
    )
"""

from .plugin_manager import (
    AgentPluginManager,
    AgentPlugin,
    PluginInfo,
    ValidationResult
)

__version__ = "1.0.0"
__author__ = "CrediLinq Development Team"

__all__ = [
    "AgentPluginManager",
    "AgentPlugin", 
    "PluginInfo",
    "ValidationResult"
]