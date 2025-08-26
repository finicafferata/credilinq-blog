"""
Agent Plugin Manager - Dynamic plugin loading and management for agents.

This module provides a flexible plugin architecture that allows for dynamic
loading of agent implementations, enabling extensibility and modular design
of the agent system.
"""

import os
import sys
import importlib
import inspect
import uuid
from typing import Dict, List, Optional, Any, Type, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

from ..core.base_agent import BaseAgent, AgentType, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class PluginInfo:
    """Information about an agent plugin."""
    plugin_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    agent_types: List[AgentType] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    plugin_path: str = ""
    loaded_at: Optional[datetime] = None
    status: str = "unloaded"  # unloaded, loaded, active, error
    error_message: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of plugin validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compatibility_score: float = 1.0

class AgentPlugin(ABC):
    """
    Base class for agent plugins.
    
    All agent plugins must inherit from this class and implement the required
    methods to provide agent creation functionality and metadata.
    """
    
    @abstractmethod
    def get_plugin_info(self) -> PluginInfo:
        """
        Get plugin metadata and capabilities.
        
        Returns:
            PluginInfo: Complete plugin information
        """
        pass
    
    @abstractmethod
    async def create_agent(self, config: Dict[str, Any]) -> BaseAgent:
        """
        Factory method for creating plugin agents.
        
        Args:
            config: Agent configuration parameters
            
        Returns:
            BaseAgent: Created agent instance
        """
        pass
    
    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult: Validation result with errors/warnings
        """
        # Default implementation - can be overridden
        return ValidationResult(is_valid=True)
    
    async def initialize_plugin(self) -> bool:
        """
        Initialize plugin resources.
        
        Returns:
            bool: True if initialization successful
        """
        # Default implementation - can be overridden
        return True
    
    async def cleanup_plugin(self) -> bool:
        """
        Cleanup plugin resources.
        
        Returns:
            bool: True if cleanup successful
        """
        # Default implementation - can be overridden
        return True

class AgentPluginManager:
    """
    Dynamic plugin loading and management for agents.
    
    This manager handles discovery, loading, validation, and lifecycle management
    of agent plugins, providing a flexible extension mechanism for the agent system.
    """
    
    def __init__(self, plugin_directory: str = "src/agents/plugins"):
        """
        Initialize the plugin manager.
        
        Args:
            plugin_directory: Directory to search for plugins
        """
        self.plugin_directory = os.path.abspath(plugin_directory)
        
        # Plugin storage
        self._loaded_plugins: Dict[str, AgentPlugin] = {}
        self._plugin_info: Dict[str, PluginInfo] = {}
        self._agent_type_mapping: Dict[AgentType, List[str]] = {}  # agent_type -> plugin_ids
        
        # Plugin validation and compatibility
        self._compatibility_version = "1.0.0"
        self._required_methods = ["get_plugin_info", "create_agent"]
        
        # Ensure plugin directory exists
        os.makedirs(self.plugin_directory, exist_ok=True)
        
        logger.info(f"Initialized AgentPluginManager with directory: {self.plugin_directory}")
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in the plugin directory.
        
        Returns:
            List[str]: List of discovered plugin paths
        """
        discovered_plugins = []
        
        try:
            for root, dirs, files in os.walk(self.plugin_directory):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        plugin_path = os.path.join(root, file)
                        rel_path = os.path.relpath(plugin_path, self.plugin_directory)
                        
                        # Check if file contains a plugin class
                        if await self._is_plugin_file(plugin_path):
                            discovered_plugins.append(rel_path)
            
            logger.info(f"Discovered {len(discovered_plugins)} potential plugins")
            return discovered_plugins
            
        except Exception as e:
            logger.error(f"Failed to discover plugins: {str(e)}")
            return []
    
    async def load_plugin(self, plugin_name: str) -> AgentPlugin:
        """
        Dynamically load an agent plugin.
        
        Args:
            plugin_name: Name/path of the plugin to load
            
        Returns:
            AgentPlugin: Loaded plugin instance
            
        Raises:
            ImportError: If plugin cannot be loaded
            ValueError: If plugin is invalid
        """
        try:
            # Check if already loaded
            if plugin_name in self._loaded_plugins:
                logger.info(f"Plugin {plugin_name} already loaded")
                return self._loaded_plugins[plugin_name]
            
            # Construct module path
            plugin_path = os.path.join(self.plugin_directory, plugin_name)
            if not plugin_path.endswith(".py"):
                plugin_path += ".py"
            
            if not os.path.exists(plugin_path):
                raise ImportError(f"Plugin file not found: {plugin_path}")
            
            # Import the plugin module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_name.replace('/', '.').replace('.py', '')}", 
                plugin_path
            )
            
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create spec for plugin: {plugin_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the plugin class
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                raise ValueError(f"No valid plugin class found in {plugin_name}")
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Validate plugin
            validation_result = await self.validate_plugin_compatibility(plugin_instance)
            if not validation_result.is_valid:
                raise ValueError(f"Plugin validation failed: {validation_result.errors}")
            
            # Initialize plugin
            if not await plugin_instance.initialize_plugin():
                raise ValueError(f"Plugin initialization failed: {plugin_name}")
            
            # Get plugin info
            plugin_info = plugin_instance.get_plugin_info()
            plugin_info.plugin_path = plugin_path
            plugin_info.loaded_at = datetime.utcnow()
            plugin_info.status = "loaded"
            
            # Store plugin
            self._loaded_plugins[plugin_name] = plugin_instance
            self._plugin_info[plugin_name] = plugin_info
            
            # Update agent type mapping
            for agent_type in plugin_info.agent_types:
                if agent_type not in self._agent_type_mapping:
                    self._agent_type_mapping[agent_type] = []
                self._agent_type_mapping[agent_type].append(plugin_name)
            
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {str(e)}")
            
            # Update plugin info with error
            if plugin_name in self._plugin_info:
                self._plugin_info[plugin_name].status = "error"
                self._plugin_info[plugin_name].error_message = str(e)
            
            raise
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin and cleanup its resources.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if unload successful
        """
        try:
            if plugin_name not in self._loaded_plugins:
                logger.warning(f"Plugin {plugin_name} not loaded")
                return False
            
            plugin = self._loaded_plugins[plugin_name]
            
            # Cleanup plugin resources
            await plugin.cleanup_plugin()
            
            # Remove from storage
            del self._loaded_plugins[plugin_name]
            
            # Update plugin info
            if plugin_name in self._plugin_info:
                self._plugin_info[plugin_name].status = "unloaded"
                self._plugin_info[plugin_name].loaded_at = None
            
            # Update agent type mapping
            for agent_type, plugin_list in self._agent_type_mapping.items():
                if plugin_name in plugin_list:
                    plugin_list.remove(plugin_name)
            
            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {str(e)}")
            return False
    
    async def validate_plugin_compatibility(self, plugin: AgentPlugin) -> ValidationResult:
        """
        Ensure plugin compatibility with current system.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            ValidationResult: Validation result with compatibility info
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Check required methods
            for method_name in self._required_methods:
                if not hasattr(plugin, method_name):
                    result.errors.append(f"Missing required method: {method_name}")
                    result.is_valid = False
                elif not callable(getattr(plugin, method_name)):
                    result.errors.append(f"Method {method_name} is not callable")
                    result.is_valid = False
            
            # Check plugin info
            try:
                plugin_info = plugin.get_plugin_info()
                if not plugin_info.name:
                    result.warnings.append("Plugin name is empty")
                if not plugin_info.agent_types:
                    result.warnings.append("No agent types specified")
            except Exception as e:
                result.errors.append(f"Failed to get plugin info: {str(e)}")
                result.is_valid = False
            
            # Check agent creation capability
            try:
                # Test with empty config
                test_config = {}
                if hasattr(plugin, "validate_configuration"):
                    config_validation = plugin.validate_configuration(test_config)
                    if not config_validation.is_valid:
                        result.warnings.append("Plugin config validation failed with empty config")
            except Exception as e:
                result.warnings.append(f"Plugin config validation error: {str(e)}")
            
            # Calculate compatibility score
            error_penalty = len(result.errors) * 0.2
            warning_penalty = len(result.warnings) * 0.1
            result.compatibility_score = max(0.0, 1.0 - error_penalty - warning_penalty)
            
            logger.info(f"Plugin validation result: {result.is_valid}, score: {result.compatibility_score}")
            
        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            result.is_valid = False
            result.compatibility_score = 0.0
        
        return result
    
    def get_available_plugins(self) -> List[PluginInfo]:
        """
        List all available agent plugins.
        
        Returns:
            List[PluginInfo]: Available plugin information
        """
        return list(self._plugin_info.values())
    
    def get_loaded_plugins(self) -> List[PluginInfo]:
        """
        List all currently loaded plugins.
        
        Returns:
            List[PluginInfo]: Loaded plugin information
        """
        return [info for info in self._plugin_info.values() if info.status == "loaded"]
    
    def get_plugins_for_agent_type(self, agent_type: AgentType) -> List[str]:
        """
        Get plugins that can create agents of the specified type.
        
        Args:
            agent_type: Type of agent needed
            
        Returns:
            List[str]: Plugin names that support the agent type
        """
        return self._agent_type_mapping.get(agent_type, [])
    
    async def create_agent_from_plugin(
        self, 
        plugin_name: str, 
        config: Dict[str, Any]
    ) -> BaseAgent:
        """
        Create an agent using a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to use
            config: Configuration for agent creation
            
        Returns:
            BaseAgent: Created agent instance
            
        Raises:
            ValueError: If plugin not found or agent creation fails
        """
        if plugin_name not in self._loaded_plugins:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        plugin = self._loaded_plugins[plugin_name]
        
        # Validate configuration
        if hasattr(plugin, "validate_configuration"):
            validation = plugin.validate_configuration(config)
            if not validation.is_valid:
                raise ValueError(f"Invalid configuration: {validation.errors}")
        
        # Create agent
        agent = await plugin.create_agent(config)
        
        logger.info(f"Created agent {agent.metadata.name} using plugin {plugin_name}")
        return agent
    
    async def reload_plugin(self, plugin_name: str) -> AgentPlugin:
        """
        Reload a plugin (unload and load again).
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            AgentPlugin: Reloaded plugin instance
        """
        # Unload if currently loaded
        if plugin_name in self._loaded_plugins:
            await self.unload_plugin(plugin_name)
        
        # Load again
        return await self.load_plugin(plugin_name)
    
    async def _is_plugin_file(self, file_path: str) -> bool:
        """Check if a Python file contains a valid plugin class."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple heuristic: look for AgentPlugin inheritance
            return (
                "class " in content and
                ("AgentPlugin" in content or "BaseAgent" in content) and
                "def create_agent" in content
            )
            
        except Exception:
            return False
    
    def _find_plugin_class(self, module) -> Optional[Type[AgentPlugin]]:
        """Find the plugin class in a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, AgentPlugin) and 
                obj != AgentPlugin):
                return obj
        return None
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            "total_plugins_discovered": len(self._plugin_info),
            "loaded_plugins": len(self._loaded_plugins),
            "plugins_by_status": {
                status: len([p for p in self._plugin_info.values() if p.status == status])
                for status in ["loaded", "unloaded", "error"]
            },
            "agent_types_supported": len(self._agent_type_mapping),
            "plugin_directory": self.plugin_directory
        }