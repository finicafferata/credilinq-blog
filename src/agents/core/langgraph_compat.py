"""
LangGraph compatibility layer for different versions.
Handles differences between old and new LangGraph versions.
"""

try:
    # Try to import START and END from newer versions
    from langgraph.graph import START, END
    _HAS_CONSTANTS = True
except ImportError:
    # Fallback for older versions that use string literals
    START = "__start__"
    END = "__end__"
    _HAS_CONSTANTS = False

# Make sure we can still import StateGraph
from langgraph.graph import StateGraph

# Export everything needed
__all__ = ["StateGraph", "START", "END", "_HAS_CONSTANTS"]

def get_start_node():
    """Get the start node identifier for the current LangGraph version."""
    return START

def get_end_node():
    """Get the end node identifier for the current LangGraph version."""
    return END

def is_modern_langgraph():
    """Check if we're using a modern version of LangGraph with constants."""
    return _HAS_CONSTANTS