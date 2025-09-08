"""
Adapter modules for migrating between legacy and modern agent systems.

This package provides adapters that enable modern LangGraph agents to work
seamlessly with legacy workflow systems, facilitating gradual migration
without breaking existing functionality.
"""

from .langgraph_legacy_adapter import LangGraphLegacyAdapter, AdapterFactory

__all__ = ['LangGraphLegacyAdapter', 'AdapterFactory']