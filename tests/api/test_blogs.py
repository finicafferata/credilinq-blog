"""Tests for blog API routes."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class TestBlogRoutes:
    """Test cases for blog API endpoints."""

    def test_create_blog_success(self, test_client, mock_db_config):
        """Test successful blog creation."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock successful insert
        mock_cursor.fetchone.return_value = ("new-blog-123",)
        
        request_data = {
            "title": "Test Blog Post",
            "content": "This is test content for the blog post.",
            "company_context": "Tech company specializing in AI solutions",
            "target_audience": "B2B professionals",
            "tone": "professional",
            "keywords": ["AI", "technology", "innovation"]
        }
        
        response = test_client.post("/api/blogs", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["blog_id"] == "new-blog-123"
        assert "Blog post created successfully" in data["message"]

    def test_create_blog_validation_error(self, test_client):
        """Test blog creation with invalid data."""
        request_data = {
            "title": "",  # Empty title should fail validation
            "content": "Content"
        }
        
        response = test_client.post("/api/blogs", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_get_blog_success(self, test_client, mock_db_config, sample_blog_post):
        """Test successful blog retrieval."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        mock_cursor.fetchone.return_value = (
            sample_blog_post["id"],
            sample_blog_post["title"],
            sample_blog_post["contentMarkdown"],
            sample_blog_post.get("initialPrompt"),
            sample_blog_post["status"],
            sample_blog_post.get("geoMetadata"),
            sample_blog_post.get("geoOptimized", False),
            sample_blog_post.get("geoScore"),
            sample_blog_post["createdAt"],
            sample_blog_post["updatedAt"]
        )
        
        response = test_client.get(f"/api/blogs/{sample_blog_post['id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_blog_post["id"]
        assert data["title"] == sample_blog_post["title"]
        assert data["status"] == sample_blog_post["status"]

    def test_get_blog_not_found(self, test_client, mock_db_config):
        """Test blog retrieval when blog doesn't exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = None
        
        response = test_client.get("/api/blogs/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_blogs_success(self, test_client, mock_db_config, sample_blog_post):
        """Test successful blog listing."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        mock_cursor.fetchall.return_value = [
            (
                sample_blog_post["id"],
                sample_blog_post["title"],
                sample_blog_post["status"],
                sample_blog_post["createdAt"],
                sample_blog_post["updatedAt"]
            )
        ]
        
        response = test_client.get("/api/blogs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_blog_post["id"]
        assert data[0]["title"] == sample_blog_post["title"]

    def test_list_blogs_empty(self, test_client, mock_db_config):
        """Test blog listing when no blogs exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = []
        
        response = test_client.get("/api/blogs")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_update_blog_success(self, test_client, mock_db_config, sample_blog_post):
        """Test successful blog update."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog exists check
        mock_cursor.fetchone.return_value = (sample_blog_post["id"],)
        
        request_data = {
            "title": "Updated Blog Title",
            "content": "Updated content for the blog post.",
            "status": "published"
        }
        
        response = test_client.put(f"/api/blogs/{sample_blog_post['id']}", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "updated successfully" in data["message"].lower()

    def test_update_blog_not_found(self, test_client, mock_db_config):
        """Test blog update when blog doesn't exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = None
        
        request_data = {
            "title": "Updated Title",
            "content": "Updated content"
        }
        
        response = test_client.put("/api/blogs/nonexistent-id", json=request_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_blog_success(self, test_client, mock_db_config, sample_blog_post):
        """Test successful blog deletion."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog exists check
        mock_cursor.fetchone.return_value = (sample_blog_post["id"],)
        
        response = test_client.delete(f"/api/blogs/{sample_blog_post['id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"].lower()

    def test_delete_blog_not_found(self, test_client, mock_db_config):
        """Test blog deletion when blog doesn't exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = None
        
        response = test_client.delete("/api/blogs/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_campaign_from_blog_success(self, test_client, mock_db_config, sample_blog_post):
        """Test successful campaign creation from blog post."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog status check and campaign existence check
        mock_cursor.fetchone.side_effect = [
            ("published",),  # Blog status
            None  # No existing campaign
        ]
        
        request_data = {
            "campaign_name": "Test Campaign from Blog"
        }
        
        response = test_client.post(f"/api/blogs/{sample_blog_post['id']}/campaign", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "Campaign created successfully" in data["message"]
        assert data["blog_id"] == sample_blog_post["id"]

    def test_create_campaign_from_blog_invalid_status(self, test_client, mock_db_config, sample_blog_post):
        """Test campaign creation from blog with invalid status."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog with draft status (should fail)
        mock_cursor.fetchone.return_value = ("draft",)
        
        request_data = {
            "campaign_name": "Test Campaign"
        }
        
        response = test_client.post(f"/api/blogs/{sample_blog_post['id']}/campaign", json=request_data)
        
        assert response.status_code == 400
        assert "Cannot create campaign" in response.json()["detail"]

    def test_create_campaign_from_blog_already_exists(self, test_client, mock_db_config, sample_blog_post):
        """Test campaign creation when campaign already exists for blog."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog status and existing campaign
        mock_cursor.fetchone.side_effect = [
            ("published",),  # Blog status
            ("existing-campaign-123",)  # Existing campaign
        ]
        
        request_data = {
            "campaign_name": "Test Campaign"
        }
        
        response = test_client.post(f"/api/blogs/{sample_blog_post['id']}/campaign", json=request_data)
        
        assert response.status_code == 400
        assert "Campaign already exists" in response.json()["detail"]

    def test_search_blogs_success(self, test_client, mock_db_config, sample_blog_post):
        """Test successful blog search."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        mock_cursor.fetchall.return_value = [
            (
                sample_blog_post["id"],
                sample_blog_post["title"],
                sample_blog_post["status"],
                sample_blog_post["createdAt"]
            )
        ]
        
        response = test_client.get("/api/blogs/search?query=test")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_blog_post["id"]

    def test_search_blogs_no_results(self, test_client, mock_db_config):
        """Test blog search with no results."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = []
        
        response = test_client.get("/api/blogs/search?query=nonexistent")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_get_blog_analytics(self, test_client):
        """Test blog analytics retrieval."""
        with patch('src.api.routes.blogs.get_blog_analytics') as mock_analytics:
            mock_analytics.return_value = {
                "total_views": 1000,
                "unique_visitors": 750,
                "engagement_rate": 0.15,
                "bounce_rate": 0.35,
                "average_time_on_page": 180
            }
            
            response = test_client.get("/api/blogs/test-blog-123/analytics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_views"] == 1000
        assert data["engagement_rate"] == 0.15