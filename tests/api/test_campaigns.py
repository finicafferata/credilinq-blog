"""Tests for campaign API routes."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException


class TestCampaignRoutes:
    """Test cases for campaign API endpoints."""

    def test_list_campaigns_success(self, test_client, mock_db_config, sample_campaign, sample_briefing):
        """Test successful campaign listing."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock database response
        mock_cursor.fetchall.return_value = [
            (
                sample_campaign["id"],
                sample_briefing["campaignName"],
                "active",
                sample_campaign["createdAt"],
                2,  # total_tasks
                1   # completed_tasks
            )
        ]
        
        response = test_client.get("/api/campaigns")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_campaign["id"]
        assert data[0]["name"] == sample_briefing["campaignName"]
        assert data[0]["status"] == "active"
        assert data[0]["total_tasks"] == 2
        assert data[0]["completed_tasks"] == 1
        assert data[0]["progress"] == 50.0

    def test_list_campaigns_empty(self, test_client, mock_db_config):
        """Test campaign listing when no campaigns exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = []
        
        response = test_client.get("/api/campaigns")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_get_campaign_success(self, test_client, mock_db_config, sample_campaign, sample_campaign_task):
        """Test successful campaign retrieval."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock multiple database calls
        mock_cursor.fetchone.side_effect = [
            ("Test Campaign", sample_campaign["createdAt"]),  # Campaign basic info
            ("Data-driven approach", {"hooks": []}, {"themes": []}, {"LinkedIn": "Professional"}, {"key": "phrases"}, "Notes"),  # Strategy
        ]
        mock_cursor.fetchall.return_value = [
            (sample_campaign_task["id"], sample_campaign_task["taskType"], "pending", None, None)
        ]
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent.get_campaign_status') as mock_status, \
             patch('src.agents.specialized.task_scheduler.TaskSchedulerAgent.get_scheduled_posts') as mock_posts, \
             patch('src.agents.specialized.distribution_agent.DistributionAgent.get_campaign_performance') as mock_perf:
            
            mock_status.return_value = {"status": "active"}
            mock_posts.return_value = []
            mock_perf.return_value = {"views": 100, "clicks": 10}
            
            response = test_client.get(f"/api/campaigns/{sample_campaign['id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_campaign["id"]
        assert data["name"] == "Test Campaign"
        assert data["status"] == "active"
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_type"] == "content_repurposing"

    def test_get_campaign_not_found(self, test_client, mock_db_config):
        """Test campaign retrieval when campaign doesn't exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = None
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent.get_campaign_status') as mock_status:
            mock_status.return_value = {"status": "active"}
            
            response = test_client.get("/api/campaigns/nonexistent-id")
        
        assert response.status_code == 404
        assert "Campaign not found" in response.json()["detail"]

    def test_create_campaign_success(self, test_client, mock_db_config):
        """Test successful campaign creation."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        request_data = {
            "blog_id": "test-blog-123",
            "campaign_name": "Test Campaign",
            "company_context": "Tech company",
            "content_type": "blog"
        }
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent.create_campaign_plan') as mock_create:
            mock_create.return_value = {
                "campaign_id": "new-campaign-123",
                "strategy": {"target_audience": "B2B"},
                "timeline": [],
                "tasks": [{"task_type": "content_repurposing"}]
            }
            
            response = test_client.post("/api/campaigns", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["campaign_id"] == "new-campaign-123"
        assert "Campaign created successfully" in data["message"]

    def test_create_quick_campaign_success(self, test_client, mock_db_config):
        """Test successful quick campaign creation."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog fetch
        mock_cursor.fetchone.return_value = ("Test Blog", {"company_context": "Tech company"})
        
        request_data = {
            "blog_id": "test-blog-123",
            "campaign_name": "Quick Campaign"
        }
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent.create_campaign_plan') as mock_create, \
             patch('src.agents.specialized.task_scheduler.TaskSchedulerAgent.schedule_campaign_tasks') as mock_schedule:
            
            mock_create.return_value = {
                "campaign_id": "quick-campaign-123",
                "strategy": {"target_audience": "B2B"},
                "tasks": []
            }
            mock_schedule.return_value = {"scheduled_posts": [], "schedule": []}
            
            response = test_client.post("/api/campaigns/quick/social-blast", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["campaign_id"] == "quick-campaign-123"
        assert data["template_id"] == "social-blast"

    def test_create_quick_campaign_invalid_template(self, test_client):
        """Test quick campaign creation with invalid template."""
        request_data = {
            "blog_id": "test-blog-123",
            "campaign_name": "Quick Campaign"
        }
        
        response = test_client.post("/api/campaigns/quick/invalid-template", json=request_data)
        
        assert response.status_code == 400
        assert "Unknown template" in response.json()["detail"]

    def test_update_task_status_success(self, test_client, mock_db_config, sample_campaign_task):
        """Test successful task status update."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock task existence check and update
        mock_cursor.fetchone.side_effect = [
            (sample_campaign_task["id"],),  # Task exists
            (sample_campaign_task["id"], "content_repurposing", "in_progress", None, None)  # Updated task
        ]
        
        request_data = {
            "task_id": sample_campaign_task["id"],
            "status": "in_progress"
        }
        
        response = test_client.put(
            f"/api/campaigns/{sample_campaign_task['campaignId']}/tasks/{sample_campaign_task['id']}/status",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["status"] == "in_progress"

    def test_update_task_status_invalid_status(self, test_client):
        """Test task status update with invalid status."""
        request_data = {
            "task_id": "test-task-123",
            "status": "invalid_status"
        }
        
        response = test_client.put(
            "/api/campaigns/test-campaign-123/tasks/test-task-123/status",
            json=request_data
        )
        
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]

    def test_update_task_status_not_found(self, test_client, mock_db_config):
        """Test task status update when task doesn't exist."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = None
        
        request_data = {
            "task_id": "nonexistent-task",
            "status": "completed"
        }
        
        response = test_client.put(
            "/api/campaigns/test-campaign-123/tasks/nonexistent-task/status",
            json=request_data
        )
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]

    def test_schedule_campaign_success(self, test_client, mock_db_config):
        """Test successful campaign scheduling."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock strategy fetch
        mock_cursor.fetchone.return_value = (
            "Data-driven approach", {"hooks": []}, {"themes": []}, 
            {"LinkedIn": "Professional"}, {"key": "phrases"}, "Notes"
        )
        
        request_data = {"campaign_id": "test-campaign-123"}
        
        with patch('src.agents.specialized.task_scheduler.TaskSchedulerAgent.schedule_campaign_tasks') as mock_schedule:
            mock_schedule.return_value = {
                "scheduled_posts": [{"platform": "LinkedIn", "content": "Test post"}],
                "schedule": ["2025-01-01T10:00:00Z"]
            }
            
            response = test_client.post("/api/campaigns/test-campaign-123/schedule", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Campaign scheduled successfully" in data["message"]

    def test_get_campaign_performance(self, test_client):
        """Test campaign performance retrieval."""
        with patch('src.agents.specialized.distribution_agent.DistributionAgent.get_campaign_performance') as mock_perf:
            mock_perf.return_value = {
                "views": 1000,
                "clicks": 50,
                "conversions": 5,
                "engagement_rate": 0.05
            }
            
            response = test_client.get("/api/campaigns/test-campaign-123/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert data["views"] == 1000
        assert data["clicks"] == 50
        assert data["engagement_rate"] == 0.05