#!/usr/bin/env python3
"""
Deprecation Utilities for CrediLinq Agent System
Provides standardized deprecation warnings and migration guidance.
"""

import warnings
import functools
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta


class DeprecationCategory:
    """Categories for different types of deprecations."""
    LEGACY_AGENT = "legacy_agent"
    OLD_API = "old_api"  
    WORKFLOW = "workflow"
    CONFIG = "config"


def deprecated_agent(
    replacement_class: str,
    replacement_import: str,
    migration_guide_url: Optional[str] = None,
    removal_version: str = "3.0.0",
    removal_date: Optional[str] = None,
    category: str = DeprecationCategory.LEGACY_AGENT
):
    """
    Decorator for marking agent classes as deprecated.
    
    Args:
        replacement_class: Name of the replacement class
        replacement_import: Import path for the replacement
        migration_guide_url: URL to migration documentation
        removal_version: Version when this will be removed
        removal_date: Date when this will be removed
        category: Deprecation category
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def __init__(self, *args, **kwargs):
            # Issue deprecation warning
            warning_message = _build_deprecation_message(
                deprecated_item=cls.__name__,
                replacement_class=replacement_class,
                replacement_import=replacement_import,
                migration_guide_url=migration_guide_url,
                removal_version=removal_version,
                removal_date=removal_date,
                item_type="agent class"
            )
            
            warnings.warn(
                warning_message,
                DeprecationWarning,
                stacklevel=2
            )
            
            # Log deprecation usage
            _log_deprecation_usage(cls.__name__, category, {
                'replacement': replacement_class,
                'import_path': replacement_import
            })
            
            # Call original init
            original_init(self, *args, **kwargs)
        
        cls.__init__ = __init__
        
        # Add deprecation metadata to class
        cls._deprecated = True
        cls._replacement_class = replacement_class
        cls._replacement_import = replacement_import
        cls._migration_guide_url = migration_guide_url
        cls._removal_version = removal_version
        cls._removal_date = removal_date
        
        return cls
    
    return decorator


def deprecated_function(
    replacement_function: str,
    replacement_import: Optional[str] = None,
    migration_guide_url: Optional[str] = None,
    removal_version: str = "3.0.0",
    removal_date: Optional[str] = None
):
    """
    Decorator for marking functions as deprecated.
    
    Args:
        replacement_function: Name of the replacement function
        replacement_import: Import path for the replacement  
        migration_guide_url: URL to migration documentation
        removal_version: Version when this will be removed
        removal_date: Date when this will be removed
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_message = _build_deprecation_message(
                deprecated_item=func.__name__,
                replacement_class=replacement_function,
                replacement_import=replacement_import,
                migration_guide_url=migration_guide_url,
                removal_version=removal_version,
                removal_date=removal_date,
                item_type="function"
            )
            
            warnings.warn(
                warning_message,
                DeprecationWarning,
                stacklevel=2
            )
            
            return func(*args, **kwargs)
        
        # Add deprecation metadata
        wrapper._deprecated = True
        wrapper._replacement_function = replacement_function
        wrapper._replacement_import = replacement_import
        wrapper._migration_guide_url = migration_guide_url
        wrapper._removal_version = removal_version
        wrapper._removal_date = removal_date
        
        return wrapper
    
    return decorator


def _build_deprecation_message(
    deprecated_item: str,
    replacement_class: str,
    replacement_import: Optional[str],
    migration_guide_url: Optional[str],
    removal_version: str,
    removal_date: Optional[str],
    item_type: str = "class"
) -> str:
    """Build a standardized deprecation warning message."""
    
    message_parts = [
        f"🚨 DEPRECATION WARNING: {deprecated_item}",
        f"",
        f"The {item_type} '{deprecated_item}' is deprecated and will be removed in version {removal_version}."
    ]
    
    if removal_date:
        message_parts.append(f"Removal date: {removal_date}")
    
    message_parts.extend([
        f"",
        f"✅ RECOMMENDED MIGRATION:",
        f"Replace with: {replacement_class}"
    ])
    
    if replacement_import:
        message_parts.append(f"Import from: {replacement_import}")
    
    if migration_guide_url:
        message_parts.extend([
            f"",
            f"📖 Migration Guide: {migration_guide_url}"
        ])
    
    message_parts.extend([
        f"",
        f"🔧 QUICK MIGRATION:",
        f"# Old (deprecated):",
        f"from {deprecated_item.lower().replace('agent', '')} import {deprecated_item}",
        f"",
        f"# New (recommended):",
        f"from {replacement_import or 'src.agents.adapters'} import AdapterFactory",
        f"agent = AdapterFactory.create_{replacement_class.lower().replace('agent', '').replace('langgraph', '')}_adapter()",
        f"",
        f"For more information, see: https://github.com/credilinq/agent-optimization-migration"
    ])
    
    return "\n".join(message_parts)


def _log_deprecation_usage(item_name: str, category: str, metadata: Dict[str, Any]):
    """Log deprecation usage for monitoring and metrics."""
    try:
        import logging
        logger = logging.getLogger('deprecation')
        
        logger.warning(
            f"Deprecated {category} used: {item_name}",
            extra={
                'deprecated_item': item_name,
                'category': category,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Could also send to metrics/monitoring system here
        
    except Exception:
        # Don't fail if logging fails
        pass


def get_deprecation_info(obj) -> Optional[Dict[str, Any]]:
    """Get deprecation information for an object if it's deprecated."""
    if not hasattr(obj, '_deprecated'):
        return None
    
    return {
        'deprecated': obj._deprecated,
        'replacement_class': getattr(obj, '_replacement_class', None),
        'replacement_import': getattr(obj, '_replacement_import', None),
        'migration_guide_url': getattr(obj, '_migration_guide_url', None),
        'removal_version': getattr(obj, '_removal_version', None),
        'removal_date': getattr(obj, '_removal_date', None)
    }


def list_all_deprecated_items() -> Dict[str, Dict[str, Any]]:
    """
    Scan the codebase for deprecated items and return a summary.
    This would be used by migration tools and documentation generators.
    """
    # This is a placeholder - in a real implementation, this would scan
    # the codebase for deprecated items
    return {
        'QualityReviewAgent': {
            'type': 'agent_class',
            'replacement': 'EditorAgentLangGraph via AdapterFactory.create_editor_adapter()',
            'removal_version': '3.0.0',
            'migration_completed': False
        },
        'BrandReviewAgent': {
            'type': 'agent_class', 
            'replacement': 'EditorAgentLangGraph via AdapterFactory.create_brand_review_adapter()',
            'removal_version': '3.0.0',
            'migration_completed': False
        },
        'ContentQualityAgent': {
            'type': 'agent_class',
            'replacement': 'EditorAgentLangGraph via AdapterFactory.create_editor_adapter()',
            'removal_version': '3.0.0',
            'migration_completed': False
        },
        'FinalApprovalAgent': {
            'type': 'agent_class',
            'replacement': 'Workflow orchestration via ContentGenerationWorkflowLangGraph',
            'removal_version': '3.0.0',
            'migration_completed': False
        }
    }


# Convenient aliases
deprecated_class = deprecated_agent  # Alias for backwards compatibility