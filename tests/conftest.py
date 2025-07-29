"""
Test configuration and fixtures for the Credilinq Agent backend.
"""

import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient
from supabase import Client

# Set test environment variables before importing app modules
os.environ.update({
    "ENVIRONMENT": "test",
    "OPENAI_API_KEY": "test-openai-key",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_KEY": "test-supabase-key",
    "SUPABASE_DB_URL": "postgresql://test:test@localhost:5432/test_db",
    "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
    "SUPABASE_STORAGE_BUCKET": "test-documents",
    "LOG_LEVEL": "DEBUG"
})

from src.main import app
from src.config.database import db_config


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = Mock(spec=Client)
    mock_table = Mock()
    mock_storage = Mock()
    
    # Mock table operations
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.delete.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.single.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])
    
    # Mock storage operations
    mock_storage.from_.return_value = mock_storage
    mock_storage.upload.return_value = Mock(data={"path": "test/path"})
    mock_storage.download.return_value = b"test content"
    
    mock_client.table.return_value = mock_table
    mock_client.storage.return_value = mock_storage
    
    return mock_client


@pytest.fixture(scope="function")
def mock_db_config(mock_supabase_client):
    """Mock database configuration."""
    original_supabase = db_config.supabase
    db_config.supabase = mock_supabase_client
    yield db_config
    db_config.supabase = original_supabase


@pytest.fixture(scope="function")
def client(mock_db_config) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test AI response"))]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture(scope="function") 
def sample_blog_data():
    """Sample blog data for testing."""
    return {
        "id": "test-blog-123",
        "title": "Test Blog Post",
        "company_context": "Credilinq.ai is a fintech company",
        "content_type": "blog",
        "content_markdown": "# Test Blog\n\nThis is a test blog post.",
        "status": "draft",
        "created_at": "2024-01-01T00:00:00Z",
        "initial_prompt": {
            "title": "Test Blog Post",
            "context": "Test context"
        }
    }


@pytest.fixture(scope="function")
def sample_campaign_data():
    """Sample campaign data for testing."""
    return {
        "id": "test-campaign-123",
        "blog_id": "test-blog-123",
        "created_at": "2024-01-01T00:00:00Z",
        "tasks": [
            {
                "id": "task-1",
                "task_type": "repurpose",
                "target_format": "linkedin_post",
                "status": "pending",
                "result": None,
                "error": None
            },
            {
                "id": "task-2", 
                "task_type": "create_image_prompt",
                "target_asset": "Blog Header",
                "status": "pending",
                "result": None,
                "error": None
            }
        ]
    }


@pytest.fixture(scope="function")
def sample_analytics_data():
    """Sample analytics data for testing."""
    return {
        "blog_id": "test-blog-123",
        "views": 100,
        "unique_visitors": 75,
        "engagement_rate": 0.25,
        "social_shares": 15,
        "avg_time_on_page": 180
    }


class MockAgent:
    """Mock agent class for testing."""
    def __init__(self, return_value="Mock agent response"):
        self.return_value = return_value
        
    def execute(self, *args, **kwargs):
        return self.return_value


@pytest.fixture(scope="function")
def mock_agents():
    """Mock agent instances for testing."""
    return {
        "planner": MockAgent("Mock planner response"),
        "researcher": MockAgent("Mock researcher response"),
        "writer": MockAgent("Mock writer response"),
        "editor": MockAgent("Mock editor response"),
        "campaign_manager": MockAgent([
            {"task_type": "repurpose", "target_format": "linkedin_post", "status": "pending"},
            {"task_type": "create_image_prompt", "target_asset": "Blog Header", "status": "pending"}
        ]),
        "repurpose_agent": MockAgent("Repurposed content"),
        "image_agent": MockAgent("Image prompt generated")
    }


# Test database utilities
class TestDatabaseManager:
    """Utility class for managing test database state."""
    
    @staticmethod
    def reset_mock_data(mock_client):
        """Reset mock database to clean state."""
        mock_client.table.return_value.execute.return_value = Mock(data=[])
    
    @staticmethod
    def set_mock_data(mock_client, table_name, data):
        """Set mock data for a specific table."""
        mock_client.table.return_value.execute.return_value = Mock(data=data)


@pytest.fixture(scope="function")
def test_db_manager():
    """Test database manager utility."""
    return TestDatabaseManager


# Performance testing utilities
@pytest.fixture(scope="function")
def benchmark_timer():
    """Timer utility for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
    
    return Timer


# Async test utilities
@pytest.fixture(scope="function")
async def async_mock_client():
    """Async mock client for testing async operations."""
    mock_client = Mock()
    mock_client.execute = asyncio.coroutine(lambda: Mock(data=[]))
    return mock_client


# Security testing fixtures
@pytest.fixture(scope="function")
def malicious_inputs():
    """Sample malicious inputs for security testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE blog_posts; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --"
        ],
        "xss_attacks": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
    }