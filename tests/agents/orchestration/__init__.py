"""
Test package for Campaign Orchestration components.

This package contains comprehensive test suites for the campaign orchestration
system, including unit tests, integration tests, and performance tests.

Test Structure:
- test_campaign_orchestrator.py: Tests for the main orchestrator agent
- test_campaign_database_service.py: Tests for database operations
- test_workflow_state_manager.py: Tests for state management

Running Tests:
    # Run all orchestration tests
    pytest tests/agents/orchestration/
    
    # Run specific test file
    pytest tests/agents/orchestration/test_campaign_orchestrator.py
    
    # Run with coverage
    pytest tests/agents/orchestration/ --cov=src.agents.orchestration
    
    # Run integration tests only
    pytest tests/agents/orchestration/ -m integration
"""

__version__ = "1.0.0"