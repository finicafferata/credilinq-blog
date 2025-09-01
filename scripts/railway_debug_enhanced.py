#!/usr/bin/env python3
"""
Enhanced Railway debugging script to identify agent system loading issues.
Run this locally or on Railway to identify the exact import/initialization failure.
"""

import os
import sys
import time
import logging
import traceback
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, '/app' if os.path.exists('/app') else os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_dependencies():
    """Test basic Python dependencies."""
    logger.info("🔍 Testing basic dependencies...")
    
    dependencies = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'uvicorn'),
        ('pydantic', 'BaseModel'),
        ('psycopg2', 'connect'),
        ('openai', 'OpenAI'),
        ('langchain', '__version__'),
        ('langgraph', '__version__')
    ]
    
    results = {}
    for package, attr in dependencies:
        try:
            module = __import__(package)
            if hasattr(module, attr):
                version = getattr(module, '__version__', 'unknown')
                logger.info(f"  ✅ {package}: {version}")
                results[package] = {'status': 'success', 'version': version}
            else:
                logger.warning(f"  ⚠️ {package}: missing {attr}")
                results[package] = {'status': 'partial', 'issue': f'missing {attr}'}
        except ImportError as e:
            logger.error(f"  ❌ {package}: {e}")
            results[package] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_config_imports():
    """Test configuration imports."""
    logger.info("🔍 Testing configuration imports...")
    
    try:
        from src.config.settings import settings
        logger.info(f"  ✅ settings: environment={settings.environment}")
        
        from src.config.database import db_config
        logger.info("  ✅ db_config imported")
        
        # Test database connection
        health = db_config.health_check()
        logger.info(f"  ✅ database health: {health}")
        
        return {'status': 'success', 'db_health': health}
    except Exception as e:
        logger.error(f"  ❌ Config import failed: {e}")
        logger.error(f"     Traceback: {traceback.format_exc()}")
        return {'status': 'failed', 'error': str(e)}

def test_langgraph_compatibility():
    """Test LangGraph compatibility layer."""
    logger.info("🔍 Testing LangGraph compatibility...")
    
    try:
        from src.agents.core.langgraph_compat import (
            StateGraph, START, END, CompiledStateGraph,
            is_modern_langgraph, has_compiled_state_graph
        )
        
        logger.info(f"  ✅ StateGraph imported: {StateGraph}")
        logger.info(f"  ✅ Constants: START={START}, END={END}")
        logger.info(f"  ✅ Modern LangGraph: {is_modern_langgraph()}")
        logger.info(f"  ✅ CompiledStateGraph available: {has_compiled_state_graph()}")
        
        return {
            'status': 'success',
            'modern_langgraph': is_modern_langgraph(),
            'compiled_state_graph': has_compiled_state_graph()
        }
    except Exception as e:
        logger.error(f"  ❌ LangGraph compatibility failed: {e}")
        logger.error(f"     Traceback: {traceback.format_exc()}")
        return {'status': 'failed', 'error': str(e)}

def test_agent_core_imports():
    """Test core agent system imports."""
    logger.info("🔍 Testing agent core imports...")
    
    core_modules = [
        'src.agents.core.base_agent',
        'src.agents.core.agent_factory',
        'src.agents.core.database_service',
        'src.agents.core.enhanced_agent_pool'
    ]
    
    results = {}
    for module_name in core_modules:
        try:
            start_time = time.time()
            module = __import__(module_name, fromlist=[''])
            import_time = time.time() - start_time
            
            logger.info(f"  ✅ {module_name}: {import_time:.3f}s")
            results[module_name] = {
                'status': 'success', 
                'import_time': import_time
            }
        except Exception as e:
            logger.error(f"  ❌ {module_name}: {e}")
            logger.error(f"     Traceback: {traceback.format_exc()}")
            results[module_name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_specialized_agents():
    """Test specialized agent imports."""
    logger.info("🔍 Testing specialized agent imports...")
    
    try:
        # This import triggers agent registration
        from src.agents import specialized
        logger.info("  ✅ Specialized agents imported successfully")
        
        # Try to get agent factory
        from src.agents.core.agent_factory import AgentFactory
        factory = AgentFactory()
        available_agents = factory.get_available_types()
        logger.info(f"  ✅ Available agents: {len(available_agents)} types")
        for agent_type in available_agents:
            logger.info(f"    - {agent_type}")
            
        return {
            'status': 'success',
            'agent_count': len(available_agents),
            'agent_types': available_agents
        }
    except Exception as e:
        logger.error(f"  ❌ Specialized agents failed: {e}")
        logger.error(f"     Traceback: {traceback.format_exc()}")
        return {'status': 'failed', 'error': str(e)}

def test_workflow_imports():
    """Test workflow system imports."""
    logger.info("🔍 Testing workflow imports...")
    
    workflow_modules = [
        'src.agents.workflow.blog_workflow',
        'src.agents.workflow.content_generation_workflow',
        'src.agents.workflow.task_management_system'
    ]
    
    results = {}
    for module_name in workflow_modules:
        try:
            start_time = time.time()
            module = __import__(module_name, fromlist=[''])
            import_time = time.time() - start_time
            
            logger.info(f"  ✅ {module_name}: {import_time:.3f}s")
            results[module_name] = {
                'status': 'success',
                'import_time': import_time
            }
        except Exception as e:
            logger.error(f"  ❌ {module_name}: {e}")
            logger.error(f"     Traceback: {traceback.format_exc()}")
            results[module_name] = {'status': 'failed', 'error': str(e)}
    
    return results

@contextmanager
def memory_monitor():
    """Monitor memory usage during operations."""
    try:
        import psutil
        process = psutil.Process()
        
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        logger.info(f"🔍 Memory monitor started: {start_memory:.1f} MB")
        
        try:
            yield
        finally:
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_time = time.time()
            
            memory_delta = end_memory - start_memory
            time_delta = end_time - start_time
            
            logger.info(f"🔍 Memory monitor ended: {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB in {time_delta:.1f}s)")
            
            # Railway production limit is 2GB (2048MB)
            if end_memory > 1800:  # Warning at 90% of limit
                logger.warning(f"⚠️ High memory usage: {end_memory:.1f} MB (approaching Railway 2GB limit)")
    except ImportError:
        logger.warning("psutil not available, skipping memory monitoring")
        yield

def test_main_app_import():
    """Test full main application import."""
    logger.info("🔍 Testing main application import...")
    
    try:
        with memory_monitor():
            start_time = time.time()
            
            # Import the main app
            from src.main import app
            
            import_time = time.time() - start_time
            logger.info(f"  ✅ Main app imported: {import_time:.3f}s")
            logger.info(f"  ✅ App title: {app.title}")
            logger.info(f"  ✅ App version: {app.version}")
            
            return {
                'status': 'success',
                'import_time': import_time,
                'app_title': app.title,
                'app_version': app.version
            }
    except Exception as e:
        logger.error(f"  ❌ Main app import failed: {e}")
        logger.error(f"     Traceback: {traceback.format_exc()}")
        return {'status': 'failed', 'error': str(e)}

def generate_railway_diagnostics_report():
    """Generate comprehensive Railway diagnostics report."""
    logger.info("🚀 Starting Railway Diagnostics")
    logger.info(f"📍 Working directory: {os.getcwd()}")
    logger.info(f"📍 Python path: {sys.path[:3]}...")
    
    # Check environment
    railway_env = os.environ.get('RAILWAY_ENVIRONMENT')
    railway_service = os.environ.get('RAILWAY_SERVICE_NAME')
    railway_full = os.environ.get('RAILWAY_FULL')
    database_url = os.environ.get('DATABASE_URL', 'not set')
    openai_key = 'set' if os.environ.get('OPENAI_API_KEY') else 'not set'
    
    logger.info(f"🚂 Railway Environment: {railway_env}")
    logger.info(f"🚂 Railway Service: {railway_service}")
    logger.info(f"🚂 RAILWAY_FULL: {railway_full}")
    logger.info(f"🔑 Database URL: {database_url[:50]}..." if len(database_url) > 50 else database_url)
    logger.info(f"🔑 OpenAI API Key: {openai_key}")
    
    report = {
        'timestamp': time.time(),
        'environment': {
            'railway_env': railway_env,
            'railway_service': railway_service,
            'railway_full': railway_full,
            'database_configured': database_url != 'not set',
            'openai_configured': openai_key == 'set'
        },
        'tests': {}
    }
    
    # Run all tests
    tests = [
        ('basic_dependencies', test_basic_dependencies),
        ('config_imports', test_config_imports),
        ('langgraph_compatibility', test_langgraph_compatibility),
        ('agent_core_imports', test_agent_core_imports),
        ('specialized_agents', test_specialized_agents),
        ('workflow_imports', test_workflow_imports),
        ('main_app_import', test_main_app_import)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            with memory_monitor():
                result = test_func()
                report['tests'][test_name] = result
                
                if result.get('status') == 'success':
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
            report['tests'][test_name] = {'status': 'crashed', 'error': str(e)}
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RAILWAY DIAGNOSTICS SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for test in report['tests'].values() if test.get('status') == 'success')
    total = len(report['tests'])
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    failed_tests = [name for name, result in report['tests'].items() 
                   if result.get('status') != 'success']
    
    if failed_tests:
        logger.error("❌ Failed tests:")
        for test_name in failed_tests:
            error = report['tests'][test_name].get('error', 'Unknown error')
            logger.error(f"   - {test_name}: {error}")
    else:
        logger.info("🎉 All tests passed! Full system should work on Railway.")
    
    return report

if __name__ == '__main__':
    try:
        report = generate_railway_diagnostics_report()
        
        # Save report to file if possible
        try:
            import json
            report_path = '/tmp/railway_diagnostics.json' if os.path.exists('/tmp') else './railway_diagnostics.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Diagnostics report saved to: {report_path}")
        except Exception as e:
            logger.warning(f"Could not save report file: {e}")
            
    except Exception as e:
        logger.error(f"💥 Diagnostics script failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)