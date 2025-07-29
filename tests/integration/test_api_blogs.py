"""
Integration tests for blog API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.core.exceptions import RecordNotFoundError, DatabaseQueryError


@pytest.mark.api
@pytest.mark.integration
class TestBlogAPI:
    """Test blog API endpoints."""

    def test_create_blog_success(self, client: TestClient, mock_db_config, sample_blog_data):
        """Test successful blog creation."""
        # Setup mock response
        mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[sample_blog_data]
        )
        
        request_data = {
            "title": "Test Blog Post",
            "company_context": "Credilinq.ai is a fintech company",
            "content_type": "blog"
        }
        
        with patch('src.api.routes.blogs.BlogWorkflow') as mock_workflow:
            mock_workflow.return_value.execute.return_value = {
                "content_markdown": "# Test Blog\n\nThis is a test blog post."
            }
            
            response = client.post("/api/blogs", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Blog Post"
        assert data["status"] == "draft"

    def test_create_blog_invalid_data(self, client: TestClient):
        """Test blog creation with invalid data."""
        request_data = {
            "title": "",  # Empty title should fail validation
            "company_context": "Test context",
            "content_type": "blog"
        }
        
        response = client.post("/api/blogs", json=request_data)
        assert response.status_code == 422

    def test_get_blog_success(self, client: TestClient, mock_db_config, sample_blog_data):
        """Test successful blog retrieval."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=sample_blog_data
        )
        
        response = client.get("/api/blogs/test-blog-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-blog-123"
        assert data["title"] == "Test Blog Post"

    def test_get_blog_not_found(self, client: TestClient, mock_db_config):
        """Test blog retrieval when blog not found."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None
        )
        
        response = client.get("/api/blogs/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_blogs_success(self, client: TestClient, mock_db_config, sample_blog_data):
        """Test successful blog listing."""
        mock_db_config.supabase.table.return_value.select.return_value.order.return_value.execute.return_value = Mock(
            data=[sample_blog_data, {**sample_blog_data, "id": "test-blog-456", "title": "Another Blog"}]
        )
        
        response = client.get("/api/blogs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "test-blog-123"

    def test_update_blog_success(self, client: TestClient, mock_db_config, sample_blog_data):
        """Test successful blog update."""
        updated_blog_data = {**sample_blog_data, "content_markdown": "# Updated Blog\n\nUpdated content."}
        
        mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[updated_blog_data]
        )
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=updated_blog_data
        )
        
        request_data = {"content_markdown": "# Updated Blog\n\nUpdated content."}
        
        response = client.put("/api/blogs/test-blog-123", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "Updated content" in data["content_markdown"]

    def test_update_blog_not_found(self, client: TestClient, mock_db_config):
        """Test blog update when blog not found."""
        mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[]
        )
        
        request_data = {"content_markdown": "Updated content"}
        
        response = client.put("/api/blogs/nonexistent-id", json=request_data)
        
        assert response.status_code == 404

    def test_delete_blog_success(self, client: TestClient, mock_db_config):
        """Test successful blog deletion."""
        mock_db_config.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = Mock(
            data=[{"id": "test-blog-123"}]
        )
        
        response = client.delete("/api/blogs/test-blog-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Blog deleted successfully"
        assert data["id"] == "test-blog-123"

    def test_delete_blog_not_found(self, client: TestClient, mock_db_config):
        """Test blog deletion when blog not found."""
        mock_db_config.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = Mock(
            data=[]
        )
        
        response = client.delete("/api/blogs/nonexistent-id")
        
        assert response.status_code == 404

    def test_publish_blog_success(self, client: TestClient, mock_db_config, sample_blog_data):
        """Test successful blog publishing."""
        published_blog_data = {**sample_blog_data, "status": "published"}
        
        mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[published_blog_data]
        )
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=published_blog_data
        )
        
        response = client.post("/api/blogs/test-blog-123/publish")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "published"

    @pytest.mark.security
    def test_create_blog_sql_injection_protection(self, client: TestClient, malicious_inputs):
        """Test SQL injection protection in blog creation."""
        for malicious_input in malicious_inputs["sql_injection"]:
            request_data = {
                "title": malicious_input,
                "company_context": "Test context",
                "content_type": "blog"
            }
            
            response = client.post("/api/blogs", json=request_data)
            # Should either reject the input or sanitize it, not cause a server error
            assert response.status_code in [400, 422, 201]

    @pytest.mark.security  
    def test_create_blog_xss_protection(self, client: TestClient, malicious_inputs):
        """Test XSS protection in blog creation."""
        for malicious_input in malicious_inputs["xss_attacks"]:
            request_data = {
                "title": malicious_input,
                "company_context": "Test context",
                "content_type": "blog"
            }
            
            response = client.post("/api/blogs", json=request_data)
            # Should sanitize or reject XSS attempts
            if response.status_code == 201:
                data = response.json()
                # Verify that malicious scripts are not present in response
                assert "<script>" not in data.get("title", "")
                assert "javascript:" not in data.get("title", "")

    @pytest.mark.slow
    def test_blog_creation_performance(self, client: TestClient, mock_db_config, benchmark_timer):
        """Test blog creation performance."""
        request_data = {
            "title": "Performance Test Blog",
            "company_context": "Credilinq.ai is a fintech company",
            "content_type": "blog"
        }
        
        # Setup mock response
        mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": "perf-test", "title": "Performance Test Blog", "status": "draft"}]
        )
        
        timer = benchmark_timer()
        timer.start()
        
        with patch('src.api.routes.blogs.BlogWorkflow') as mock_workflow:
            mock_workflow.return_value.execute.return_value = {
                "content_markdown": "# Performance Test\n\nContent generated quickly."
            }
            response = client.post("/api/blogs", json=request_data)
        
        elapsed_time = timer.stop()
        
        assert response.status_code == 201
        # Blog creation should complete within reasonable time (adjust threshold as needed)
        assert elapsed_time < 5.0  # 5 seconds max