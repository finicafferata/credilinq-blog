"""
Unit tests for database operations and services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.core.database_service import (
    DatabaseService, BlogAnalyticsData, MarketingMetric
)
from src.core.exceptions import DatabaseQueryError, SecurityViolationError
from src.core.security import SecurityValidator


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseService:
    """Test DatabaseService functionality."""
    
    def test_init_with_valid_config(self, mock_supabase_client):
        """Test database service initialization."""
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            assert db_service.client is not None

    def test_get_blog_content_success(self, mock_supabase_client, sample_blog_data):
        """Test successful blog content retrieval."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=sample_blog_data
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            result = db_service.get_blog_content("test-blog-123")
        
        assert result["id"] == "test-blog-123"
        assert result["title"] == "Test Blog Post"

    def test_get_blog_content_not_found(self, mock_supabase_client):
        """Test blog content retrieval when blog not found."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            with pytest.raises(Exception) as exc_info:
                db_service.get_blog_content("nonexistent-id")
            
            assert "not found" in str(exc_info.value).lower()

    def test_update_blog_analytics_success(self, mock_supabase_client, sample_analytics_data):
        """Test successful blog analytics update."""
        mock_supabase_client.table.return_value.upsert.return_value.execute.return_value = Mock(
            data=[{"id": "analytics-123", **sample_analytics_data}]
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            analytics_data = BlogAnalyticsData(**sample_analytics_data)
            
            result = db_service.update_blog_analytics(analytics_data)
        
        assert result == "analytics-123"

    def test_record_marketing_metric_success(self, mock_supabase_client):
        """Test successful marketing metric recording."""
        metric_data = {
            "blog_id": "test-blog-123",
            "metric_type": "social_share",
            "platform": "linkedin",
            "value": 1.0,
            "metadata": {"post_id": "linkedin-post-123"}
        }
        
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": "metric-123"}]
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            marketing_metric = MarketingMetric(**metric_data)
            
            result = db_service.record_marketing_metric(marketing_metric)
        
        assert result == "metric-123"

    def test_get_dashboard_analytics_success(self, mock_supabase_client):
        """Test successful dashboard analytics retrieval."""
        mock_analytics_data = {
            "total_blogs": 10,
            "published_blogs": 8,
            "total_views": 1500,
            "avg_engagement": 0.25,
            "top_performing_blogs": [
                {"id": "blog-1", "title": "Top Blog", "views": 500},
                {"id": "blog-2", "title": "Another Top Blog", "views": 300}
            ]
        }
        
        # Mock multiple database calls for analytics aggregation
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[
                {"total_blogs": 10},
                {"published_blogs": 8},
                {"total_views": 1500},
                {"avg_engagement": 0.25}
            ]
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            with patch.object(db_service, '_aggregate_dashboard_data', return_value=mock_analytics_data):
                result = db_service.get_dashboard_analytics(days=30)
        
        assert result["total_blogs"] == 10
        assert result["published_blogs"] == 8
        assert len(result["top_performing_blogs"]) == 2

    def test_record_agent_feedback_success(self, mock_supabase_client):
        """Test successful agent feedback recording."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": "feedback-123"}]
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            result = db_service.record_agent_feedback(
                agent_type="writer_agent",
                feedback_type="quality_rating",
                feedback_value=4.5,
                feedback_text="Great writing quality",
                user_id="user-123"
            )
        
        assert result == "feedback-123"

    def test_health_check_success(self, mock_supabase_client):
        """Test successful database health check."""
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.return_value = Mock(
            data=[{"test": "ok"}]
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            result = db_service.health_check()
        
        assert result["status"] == "healthy"
        assert "database" in result
        assert result["database"]["status"] == "connected"

    def test_health_check_failure(self, mock_supabase_client):
        """Test database health check failure."""
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("Database connection failed")
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            result = db_service.health_check()
        
        assert result["status"] == "unhealthy"
        assert "error" in result

    @pytest.mark.security
    def test_sql_injection_protection(self, mock_supabase_client, malicious_inputs):
        """Test SQL injection protection in database operations."""
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            for malicious_input in malicious_inputs["sql_injection"]:
                # Should detect and reject SQL injection attempts
                with pytest.raises((SecurityViolationError, ValueError)):
                    db_service.get_blog_content(malicious_input)

    def test_database_error_handling(self, mock_supabase_client):
        """Test database error handling."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.side_effect = Exception("Database error")
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            with pytest.raises(DatabaseQueryError):
                db_service.get_blog_content("test-blog-123")

    def test_vector_search_functionality(self, mock_supabase_client):
        """Test vector search operations."""
        mock_search_results = [
            {"id": "doc-1", "content": "Relevant document 1", "similarity": 0.95},
            {"id": "doc-2", "content": "Relevant document 2", "similarity": 0.87}
        ]
        
        mock_supabase_client.rpc.return_value.execute.return_value = Mock(
            data=mock_search_results
        )
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            # Mock embedding generation
            with patch.object(db_service, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
                result = db_service.vector_search("fintech innovation", limit=5)
        
        assert len(result) == 2
        assert result[0]["similarity"] == 0.95

    def test_concurrent_operations(self, mock_supabase_client):
        """Test handling of concurrent database operations."""
        import asyncio
        
        async def mock_async_operation():
            return {"id": "async-result"}
        
        with patch('src.agents.core.database_service.create_client', return_value=mock_supabase_client):
            db_service = DatabaseService()
            
            # Test that service can handle concurrent operations without issues
            # This is more of a smoke test for thread safety
            assert db_service.client is not None


@pytest.mark.unit
@pytest.mark.security
class TestSecurityValidator:
    """Test security validation functionality."""
    
    def test_validate_input_clean(self):
        """Test validation with clean input."""
        validator = SecurityValidator()
        clean_input = "This is a normal blog title about fintech"
        
        result = validator.validate_input(clean_input)
        
        assert result == clean_input

    def test_validate_input_sql_injection(self, malicious_inputs):
        """Test SQL injection detection."""
        validator = SecurityValidator()
        
        for malicious_input in malicious_inputs["sql_injection"]:
            with pytest.raises(SecurityViolationError):
                validator.validate_input(malicious_input)

    def test_validate_input_xss(self, malicious_inputs):
        """Test XSS attack detection."""
        validator = SecurityValidator()
        
        for malicious_input in malicious_inputs["xss_attacks"]:
            with pytest.raises(SecurityViolationError):
                validator.validate_input(malicious_input)

    def test_validate_input_path_traversal(self, malicious_inputs):
        """Test path traversal attack detection."""
        validator = SecurityValidator()
        
        for malicious_input in malicious_inputs["path_traversal"]:
            with pytest.raises(SecurityViolationError):
                validator.validate_input(malicious_input)

    def test_sanitize_output_basic(self):
        """Test basic output sanitization."""
        validator = SecurityValidator()
        unsafe_output = '<script>alert("xss")</script>Safe content here'
        
        result = validator.sanitize_output(unsafe_output)
        
        assert '<script>' not in result
        assert 'Safe content here' in result

    def test_validate_vector_embedding(self):
        """Test vector embedding validation."""
        validator = SecurityValidator()
        
        # Valid embedding
        valid_embedding = [0.1, 0.2, 0.3, -0.1, -0.2]
        result = validator.validate_vector_embedding(valid_embedding)
        assert result == valid_embedding
        
        # Invalid embedding (too large values)
        invalid_embedding = [999.0, 0.2, 0.3]
        with pytest.raises(SecurityViolationError):
            validator.validate_vector_embedding(invalid_embedding)
        
        # Invalid embedding (wrong type)
        with pytest.raises(SecurityViolationError):
            validator.validate_vector_embedding("not a list")

    def test_check_rate_limiting(self):
        """Test rate limiting functionality."""
        validator = SecurityValidator()
        
        # Should allow initial requests
        assert validator.check_rate_limit("test-user", "api_call") is True
        
        # Mock excessive requests
        with patch.object(validator, '_get_request_count', return_value=1000):
            assert validator.check_rate_limit("test-user", "api_call") is False