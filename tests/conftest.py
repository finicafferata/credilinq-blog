"""Test configuration and fixtures for pytest."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from typing import Generator, AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db_connection():
    """Mock database connection that doesn't actually connect to DB."""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Mock cursor methods
    mock_cursor.execute = Mock()
    mock_cursor.fetchone = Mock()
    mock_cursor.fetchall = Mock()
    mock_cursor.fetchmany = Mock()
    
    # Mock connection methods
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    mock_conn.close = Mock()
    
    return mock_conn, mock_cursor


@pytest.fixture
def mock_db_config(mock_db_connection):
    """Mock database config to prevent real database connections."""
    mock_conn, mock_cursor = mock_db_connection
    
    with patch('src.config.database.db_config') as mock_config:
        mock_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_config.get_db_connection.return_value.__exit__.return_value = None
        yield mock_config, mock_conn, mock_cursor


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    from src.main import app
    with TestClient(app) as client:
        yield client


# Sample test data
@pytest.fixture
def sample_blog_post():
    """Sample blog post data for testing."""
    return {
        "id": "test-blog-123",
        "title": "Test Blog Post",
        "contentMarkdown": "# Test Content\n\nThis is a test blog post.",
        "status": "draft",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_campaign():
    """Sample campaign data for testing."""
    return {
        "id": "test-campaign-123",
        "blogPostId": "test-blog-123",
        "createdAt": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_campaign_task():
    """Sample campaign task data for testing."""
    return {
        "id": "test-task-123",
        "campaignId": "test-campaign-123",
        "taskType": "content_repurposing",
        "status": "pending",
        "result": None,
        "error": None,
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_briefing():
    """Sample briefing data for testing."""
    return {
        "id": "test-briefing-123",
        "campaignId": "test-campaign-123",
        "campaignName": "Test Campaign",
        "marketingObjective": "Brand awareness",
        "targetAudience": "B2B professionals",
        "channels": ["LinkedIn", "Email"],
        "desiredTone": "Professional",
        "language": "English"
    }