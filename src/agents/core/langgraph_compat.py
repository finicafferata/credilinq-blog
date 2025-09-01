"""
LangGraph compatibility layer for different versions.
Handles differences between old and new LangGraph versions, including
missing imports and API changes between langgraph==0.0.20 and newer versions.
"""

import logging

logger = logging.getLogger(__name__)

# Core imports that should always work
from langgraph.graph import StateGraph

# Constants compatibility
try:
    # Try to import START and END from newer versions
    from langgraph.graph import START, END
    _HAS_CONSTANTS = True
    logger.debug("Using modern LangGraph constants (START, END)")
except ImportError:
    # Fallback for older versions that use string literals
    START = "__start__"
    END = "__end__"
    _HAS_CONSTANTS = False
    logger.debug("Using legacy LangGraph constants (__start__, __end__)")

# CompiledStateGraph compatibility
try:
    from langgraph.graph.state import CompiledStateGraph
    _HAS_COMPILED_STATE_GRAPH = True
    logger.debug("CompiledStateGraph available")
except ImportError:
    # For older versions without CompiledStateGraph
    CompiledStateGraph = None
    _HAS_COMPILED_STATE_GRAPH = False
    logger.debug("CompiledStateGraph not available - using fallback")

# Message handling compatibility
try:
    from langgraph.graph.message import add_messages
    _HAS_MESSAGE_UTILS = True
    logger.debug("LangGraph message utilities available")
except ImportError:
    # Fallback for older versions
    def add_messages(x, y):
        """Fallback implementation for add_messages"""
        if isinstance(x, list) and isinstance(y, list):
            return x + y
        elif isinstance(x, list):
            return x + [y]
        elif isinstance(y, list):
            return [x] + y
        else:
            return [x, y]
    _HAS_MESSAGE_UTILS = False
    logger.debug("Using fallback add_messages implementation")

# Export everything needed
__all__ = [
    "StateGraph", 
    "START", 
    "END", 
    "CompiledStateGraph", 
    "add_messages",
    "_HAS_CONSTANTS",
    "_HAS_COMPILED_STATE_GRAPH", 
    "_HAS_MESSAGE_UTILS",
    "get_start_node",
    "get_end_node",
    "is_modern_langgraph",
    "safe_compile_graph"
]

def get_start_node():
    """Get the start node identifier for the current LangGraph version."""
    return START

def get_end_node():
    """Get the end node identifier for the current LangGraph version."""
    return END

def is_modern_langgraph():
    """Check if we're using a modern version of LangGraph with constants."""
    return _HAS_CONSTANTS

def has_compiled_state_graph():
    """Check if CompiledStateGraph is available."""
    return _HAS_COMPILED_STATE_GRAPH

def has_message_utils():
    """Check if message utilities are available."""
    return _HAS_MESSAGE_UTILS

def safe_compile_graph(graph, checkpointer=None):
    """
    Safely compile a StateGraph with version compatibility.
    
    Args:
        graph: StateGraph instance to compile
        checkpointer: Optional checkpointer for persistence
    
    Returns:
        Compiled graph or original graph if compilation not available
    """
    try:
        if hasattr(graph, 'compile'):
            if checkpointer:
                return graph.compile(checkpointer=checkpointer)
            else:
                return graph.compile()
        else:
            # Fallback for older versions without compile method
            logger.warning("Graph compilation not available - returning original graph")
            return graph
    except Exception as e:
        logger.warning(f"Graph compilation failed: {e} - returning original graph")
        return graph

def create_compatible_state_graph(state_schema, **kwargs):
    """
    Create a StateGraph with version compatibility handling.
    
    Args:
        state_schema: The state schema/class for the graph
        **kwargs: Additional arguments for StateGraph constructor
    
    Returns:
        StateGraph instance
    """
    try:
        return StateGraph(state_schema, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to create StateGraph with kwargs {kwargs}: {e}")
        # Try without additional kwargs for older versions
        return StateGraph(state_schema)