"""
Integration tests for campaign API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.mark.api
@pytest.mark.integration
class TestCampaignAPI:
    """Test campaign API endpoints."""

    def test_create_campaign_success(self, client: TestClient, mock_db_config, sample_campaign_data):
        """Test successful campaign creation."""
        # Mock database responses
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None  # No existing campaign
        )
        mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": "test-campaign-123", "blog_id": "test-blog-123"}]
        )
        
        request_data = {"blog_id": "test-blog-123"}
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent') as mock_agent:
            mock_agent.return_value.execute.return_value = [
                {"task_type": "repurpose", "target_format": "linkedin_post", "status": "pending"},
                {"task_type": "create_image_prompt", "target_asset": "Blog Header", "status": "pending"}
            ]
            
            response = client.post("/api/campaigns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["blog_id"] == "test-blog-123"
        assert len(data["tasks"]) == 2

    def test_create_campaign_existing(self, client: TestClient, mock_db_config, sample_campaign_data):
        """Test campaign creation when campaign already exists."""
        # Mock existing campaign
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=sample_campaign_data
        )
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
            data=sample_campaign_data["tasks"]
        )
        
        request_data = {"blog_id": "test-blog-123"}
        
        response = client.post("/api/campaigns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-campaign-123"
        assert data["blog_id"] == "test-blog-123"

    def test_get_campaign_success(self, client: TestClient, mock_db_config, sample_campaign_data):
        """Test successful campaign retrieval."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=sample_campaign_data
        )
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
            data=sample_campaign_data["tasks"]
        )
        
        response = client.get("/api/campaigns/test-blog-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["blog_id"] == "test-blog-123"
        assert len(data["tasks"]) == 2

    def test_get_campaign_not_found(self, client: TestClient, mock_db_config):
        """Test campaign retrieval when campaign not found."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None
        )
        
        response = client.get("/api/campaigns/nonexistent-blog-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_execute_campaign_task_success(self, client: TestClient, mock_db_config):
        """Test successful campaign task execution."""
        # Mock task data
        task_data = {
            "id": "task-1",
            "campaign_id": "test-campaign-123",
            "task_type": "repurpose",
            "target_format": "linkedin_post",
            "status": "pending"
        }
        
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=task_data
        )
        mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[{**task_data, "status": "in_progress"}]
        )
        
        request_data = {"task_id": "task-1"}
        
        response = client.post("/api/campaigns/tasks/execute", json=request_data)
        
        assert response.status_code == 202
        data = response.json()
        assert data["message"] == "Task execution started"
        assert data["task_id"] == "task-1"

    def test_execute_campaign_task_not_found(self, client: TestClient, mock_db_config):
        """Test campaign task execution when task not found."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None
        )
        
        request_data = {"task_id": "nonexistent-task"}
        
        response = client.post("/api/campaigns/tasks/execute", json=request_data)
        
        assert response.status_code == 404

    def test_update_campaign_task_success(self, client: TestClient, mock_db_config):
        """Test successful campaign task update."""
        updated_task = {
            "id": "task-1",
            "task_type": "repurpose",
            "target_format": "linkedin_post",
            "status": "completed",
            "result": "Updated task content",
            "error": None,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T01:00:00Z"
        }
        
        mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[updated_task]
        )
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=updated_task
        )
        
        request_data = {
            "content": "Updated task content",
            "status": "completed"
        }
        
        response = client.put("/api/campaigns/tasks/task-1", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] == "Updated task content"

    def test_campaign_agent_execution_error(self, client: TestClient, mock_db_config):
        """Test campaign creation with agent execution error."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None
        )
        
        request_data = {"blog_id": "test-blog-123"}
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent') as mock_agent:
            mock_agent.return_value.execute.side_effect = Exception("Agent execution failed")
            
            response = client.post("/api/campaigns", json=request_data)
        
        assert response.status_code == 500
        assert "Campaign generation failed" in response.json()["detail"]

    @pytest.mark.security
    def test_campaign_input_validation(self, client: TestClient, malicious_inputs):
        """Test input validation in campaign endpoints."""
        for malicious_input in malicious_inputs["sql_injection"]:
            request_data = {"blog_id": malicious_input}
            
            response = client.post("/api/campaigns", json=request_data)
            # Should handle malicious input gracefully
            assert response.status_code in [400, 404, 422, 500]