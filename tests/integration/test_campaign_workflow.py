"""Integration tests for campaign workflow."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestCampaignWorkflow:
    """Test complete campaign workflow from creation to completion."""

    def test_complete_campaign_workflow(self, test_client, mock_db_config):
        """Test the complete workflow: blog creation -> campaign creation -> task execution."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Step 1: Create a blog post
        blog_data = {
            "title": "Test Blog for Campaign",
            "content": "This is test content for campaign workflow.",
            "company_context": "Tech startup",
            "target_audience": "B2B professionals",
            "tone": "professional",
            "keywords": ["technology", "innovation"]
        }
        
        # Mock blog creation
        mock_cursor.fetchone.return_value = ("blog-123",)
        blog_response = test_client.post("/api/blogs", json=blog_data)
        assert blog_response.status_code == 200
        blog_id = blog_response.json()["blog_id"]
        
        # Step 2: Create campaign from blog
        # Mock blog status check and campaign creation
        mock_cursor.fetchone.side_effect = [
            ("published",),  # Blog status check
            None  # No existing campaign
        ]
        
        campaign_data = {"campaign_name": "Integration Test Campaign"}
        campaign_response = test_client.post(f"/api/blogs/{blog_id}/campaign", json=campaign_data)
        assert campaign_response.status_code == 200
        campaign_id = campaign_response.json()["campaign_id"]
        
        # Step 3: List campaigns and verify it appears
        mock_cursor.fetchall.return_value = [
            (campaign_id, "Integration Test Campaign", "active", "2025-01-01T00:00:00Z", 2, 0)
        ]
        
        campaigns_response = test_client.get("/api/campaigns")
        assert campaigns_response.status_code == 200
        campaigns = campaigns_response.json()
        assert len(campaigns) == 1
        assert campaigns[0]["name"] == "Integration Test Campaign"
        
        # Step 4: Get campaign details
        mock_cursor.fetchone.side_effect = [
            ("Integration Test Campaign", "2025-01-01T00:00:00Z"),  # Campaign info
            ("Data-driven approach", {"hooks": []}, {"themes": []}, {"LinkedIn": "Professional"}, {"key": "phrases"}, "Notes")  # Strategy
        ]
        mock_cursor.fetchall.return_value = [
            ("task-1", "content_repurposing", "pending", None, None),
            ("task-2", "image_generation", "pending", None, None)
        ]
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent.get_campaign_status') as mock_status, \
             patch('src.agents.specialized.task_scheduler.TaskSchedulerAgent.get_scheduled_posts') as mock_posts, \
             patch('src.agents.specialized.distribution_agent.DistributionAgent.get_campaign_performance') as mock_perf:
            
            mock_status.return_value = {"status": "active"}
            mock_posts.return_value = []
            mock_perf.return_value = {"views": 0, "clicks": 0}
            
            campaign_detail_response = test_client.get(f"/api/campaigns/{campaign_id}")
        
        assert campaign_detail_response.status_code == 200
        campaign_details = campaign_detail_response.json()
        assert len(campaign_details["tasks"]) == 2
        
        # Step 5: Update task status
        mock_cursor.fetchone.side_effect = [
            ("task-1",),  # Task exists
            ("task-1", "content_repurposing", "completed", "LinkedIn post created", None)  # Updated task
        ]
        
        task_update = {"task_id": "task-1", "status": "completed"}
        task_response = test_client.put(f"/api/campaigns/{campaign_id}/tasks/task-1/status", json=task_update)
        assert task_response.status_code == 200
        assert task_response.json()["task"]["status"] == "completed"

    def test_quick_campaign_workflow(self, test_client, mock_db_config):
        """Test quick campaign creation workflow."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Mock blog fetch for quick campaign
        mock_cursor.fetchone.return_value = ("Test Blog", {"company_context": "Tech company"})
        
        request_data = {
            "blog_id": "test-blog-123",
            "campaign_name": "Quick Social Blast"
        }
        
        with patch('src.agents.specialized.campaign_manager.CampaignManagerAgent.create_campaign_plan') as mock_create, \
             patch('src.agents.specialized.task_scheduler.TaskSchedulerAgent.schedule_campaign_tasks') as mock_schedule:
            
            mock_create.return_value = {
                "campaign_id": "quick-campaign-123",
                "strategy": {"target_audience": "B2B professionals"},
                "tasks": [{"task_type": "social_media_post"}]
            }
            mock_schedule.return_value = {"scheduled_posts": [], "schedule": []}
            
            response = test_client.post("/api/campaigns/quick/social-blast", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["auto_executed"] is True
        assert data["template_id"] == "social-blast"

    def test_campaign_scheduling_workflow(self, test_client, mock_db_config):
        """Test campaign scheduling workflow."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        campaign_id = "test-campaign-123"
        
        # Mock strategy fetch
        mock_cursor.fetchone.return_value = (
            "Data-driven approach", {"hooks": ["Innovation"]}, {"themes": ["Technology"]}, 
            {"LinkedIn": "Professional"}, {"AI": "keywords"}, "Campaign notes"
        )
        
        request_data = {"campaign_id": campaign_id}
        
        with patch('src.agents.specialized.task_scheduler.TaskSchedulerAgent.schedule_campaign_tasks') as mock_schedule:
            mock_schedule.return_value = {
                "scheduled_posts": [
                    {"platform": "LinkedIn", "content": "Test post", "scheduled_at": "2025-01-05T10:00:00Z"}
                ],
                "schedule": ["2025-01-05T10:00:00Z"]
            }
            
            response = test_client.post(f"/api/campaigns/{campaign_id}/schedule", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["scheduled_posts"]) == 1

    def test_campaign_distribution_workflow(self, test_client):
        """Test campaign distribution workflow."""
        campaign_id = "test-campaign-123"
        request_data = {"campaign_id": campaign_id}
        
        with patch('src.agents.specialized.distribution_agent.DistributionAgent.publish_scheduled_posts') as mock_publish:
            mock_publish.return_value = {
                "published": 2,
                "failed": 0,
                "posts": [
                    {"platform": "LinkedIn", "status": "published"},
                    {"platform": "Twitter", "status": "published"}
                ]
            }
            
            response = test_client.post(f"/api/campaigns/{campaign_id}/distribute", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["published"] == 2
        assert data["failed"] == 0

    def test_campaign_performance_tracking(self, test_client):
        """Test campaign performance tracking workflow."""
        campaign_id = "test-campaign-123"
        
        with patch('src.agents.specialized.distribution_agent.DistributionAgent.get_campaign_performance') as mock_perf:
            mock_perf.return_value = {
                "views": 2500,
                "clicks": 125,
                "conversions": 12,
                "engagement_rate": 0.05,
                "click_through_rate": 0.025,
                "conversion_rate": 0.096
            }
            
            response = test_client.get(f"/api/campaigns/{campaign_id}/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert data["views"] == 2500
        assert data["clicks"] == 125
        assert data["conversions"] == 12
        assert data["engagement_rate"] == 0.05

    def test_error_handling_in_workflow(self, test_client, mock_db_config):
        """Test error handling throughout the workflow."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        # Test campaign creation with invalid blog status
        mock_cursor.fetchone.return_value = ("draft",)  # Invalid status
        
        campaign_data = {"campaign_name": "Test Campaign"}
        response = test_client.post("/api/blogs/test-blog-123/campaign", json=campaign_data)
        assert response.status_code == 400
        assert "Cannot create campaign" in response.json()["detail"]
        
        # Test campaign scheduling with non-existent campaign
        mock_cursor.fetchone.return_value = None
        
        request_data = {"campaign_id": "nonexistent-campaign"}
        response = test_client.post("/api/campaigns/nonexistent-campaign/schedule", json=request_data)
        assert response.status_code == 404

    def test_concurrent_operations(self, test_client, mock_db_config):
        """Test handling of concurrent operations on the same campaign."""
        mock_config, mock_conn, mock_cursor = mock_db_config
        
        campaign_id = "concurrent-test-campaign"
        
        # Mock concurrent task updates
        mock_cursor.fetchone.side_effect = [
            ("task-1",),  # First task exists
            ("task-1", "content_repurposing", "in_progress", None, None),  # First update
            ("task-2",),  # Second task exists
            ("task-2", "image_generation", "completed", "Image created", None)  # Second update
        ]
        
        # Update first task
        task_update_1 = {"task_id": "task-1", "status": "in_progress"}
        response_1 = test_client.put(f"/api/campaigns/{campaign_id}/tasks/task-1/status", json=task_update_1)
        assert response_1.status_code == 200
        
        # Update second task
        task_update_2 = {"task_id": "task-2", "status": "completed"}
        response_2 = test_client.put(f"/api/campaigns/{campaign_id}/tasks/task-2/status", json=task_update_2)
        assert response_2.status_code == 200
        
        # Both updates should succeed
        assert response_1.json()["task"]["status"] == "in_progress"
        assert response_2.json()["task"]["status"] == "completed"