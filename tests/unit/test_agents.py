"""
Unit tests for agent workflows and functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.specialized.campaign_manager import CampaignManagerAgent
from src.agents.specialized.repurpose_agent import ContentRepurposingAgent
from src.agents.specialized.image_agent import ImagePromptAgent
from src.agents.workflow.structured_blog_workflow import BlogWorkflow


@pytest.mark.unit
@pytest.mark.agent
class TestCampaignManagerAgent:
    """Test CampaignManagerAgent functionality."""
    
    def test_execute_success(self, mock_db_config, sample_blog_data):
        """Test successful campaign plan generation."""
        # Mock database response for blog data
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=sample_blog_data
        )
        
        agent = CampaignManagerAgent()
        
        with patch.object(agent, 'llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = '''[
                {"task_type": "repurpose", "target_format": "linkedin_post", "status": "pending"},
                {"task_type": "create_image_prompt", "target_asset": "Blog Header", "status": "pending"}
            ]'''
            mock_llm.invoke.return_value = mock_response
            
            result = agent.execute("test-blog-123")
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["task_type"] == "repurpose"
        assert result[1]["task_type"] == "create_image_prompt"

    def test_execute_blog_not_found(self, mock_db_config):
        """Test campaign generation when blog is not found."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=None
        )
        
        agent = CampaignManagerAgent()
        
        with pytest.raises(Exception) as exc_info:
            agent.execute("nonexistent-blog-id")
        
        assert "Blog not found" in str(exc_info.value)

    def test_execute_llm_error(self, mock_db_config, sample_blog_data):
        """Test campaign generation with LLM error."""
        mock_db_config.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = Mock(
            data=sample_blog_data
        )
        
        agent = CampaignManagerAgent()
        
        with patch.object(agent, 'llm') as mock_llm:
            mock_llm.invoke.side_effect = Exception("LLM service unavailable")
            
            with pytest.raises(Exception) as exc_info:
                agent.execute("test-blog-123")
            
            assert "LLM service unavailable" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.agent
class TestContentRepurposingAgent:
    """Test ContentRepurposingAgent functionality."""
    
    def test_execute_linkedin_post(self):
        """Test LinkedIn post repurposing."""
        agent = ContentRepurposingAgent()
        original_content = "# Test Blog\n\nThis is a comprehensive blog post about fintech innovations."
        target_format = "linkedin_post"
        
        with patch.object(agent, 'llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = "🚀 Exciting developments in fintech! Here's what you need to know about the latest innovations..."
            mock_llm.invoke.return_value = mock_response
            
            result = agent.execute(original_content, target_format)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "fintech" in result.lower()

    def test_execute_twitter_thread(self):
        """Test Twitter thread repurposing."""
        agent = ContentRepurposingAgent()
        original_content = "# Financial Technology Trends\n\nBlockchain, AI, and embedded finance are reshaping the industry."
        target_format = "twitter_thread"
        
        with patch.object(agent, 'llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = "1/5 🧵 Financial technology is evolving rapidly...\n\n2/5 Blockchain technology is enabling..."
            mock_llm.invoke.return_value = mock_response
            
            result = agent.execute(original_content, target_format)
        
        assert isinstance(result, str)
        assert "1/" in result  # Twitter thread format

    def test_execute_empty_content(self):
        """Test repurposing with empty content."""
        agent = ContentRepurposingAgent()
        
        result = agent.execute("", "linkedin_post")
        
        assert "Error" in result

    def test_execute_invalid_format(self):
        """Test repurposing with invalid target format."""
        agent = ContentRepurposingAgent()
        content = "Test content"
        
        result = agent.execute(content, "invalid_format")
        
        # Should handle gracefully and provide fallback
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.unit
@pytest.mark.agent  
class TestImagePromptAgent:
    """Test ImagePromptAgent functionality."""
    
    def test_execute_blog_header(self):
        """Test blog header image prompt generation."""
        agent = ImagePromptAgent()
        topic = "The Future of Embedded Finance"
        content = "# The Future of Embedded Finance\n\nEmbedded finance is revolutionizing how businesses integrate financial services..."
        
        with patch.object(agent, 'llm') as mock_llm:
            # Mock concept extraction
            mock_llm.invoke.side_effect = [
                Mock(content='["embedded finance", "technology", "business integration", "financial services", "innovation"]'),
                Mock(content='{"tone": "professional", "industry": "fintech", "audience": "business leaders"}'),
                Mock(content="A modern, professional image showing digital financial interfaces and business technology integration...")
            ]
            
            result = agent.execute(topic, content)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "professional" in result.lower()

    def test_execute_concept_extraction_fallback(self):
        """Test fallback when concept extraction fails."""
        agent = ImagePromptAgent()
        topic = "Fintech Innovation"
        content = "Content about fintech innovation and digital transformation."
        
        with patch.object(agent, 'llm') as mock_llm:
            # Mock LLM failure for concept extraction, success for others
            mock_llm.invoke.side_effect = [
                Exception("LLM error"),  # Concept extraction fails
                Mock(content='{"tone": "professional", "industry": "technology", "audience": "professionals"}'),
                Mock(content="Professional technology image with modern design elements...")
            ]
            
            result = agent.execute(topic, content)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should use fallback concepts
        assert any(keyword in result.lower() for keyword in ["technology", "business", "professional"])

    def test_create_image_variations(self):
        """Test image prompt variations generation."""
        agent = ImagePromptAgent()
        base_prompt = "Professional fintech illustration with modern design"
        
        variations = agent.create_image_variations(base_prompt, num_variations=3)
        
        assert isinstance(variations, list)
        assert len(variations) == 3
        assert all(base_prompt in variation for variation in variations)
        assert all("style" in variation.lower() for variation in variations)

    def test_analyze_prompt_quality(self):
        """Test prompt quality analysis."""
        agent = ImagePromptAgent()
        
        # High quality prompt
        good_prompt = "Professional, high quality, 4K resolution fintech illustration with modern design, clean composition, and vibrant colors"
        analysis = agent.analyze_prompt_quality(good_prompt)
        
        assert analysis["estimated_effectiveness"] == "high"
        assert analysis["has_style"] is True
        assert analysis["has_technical_specs"] is True
        
        # Low quality prompt
        poor_prompt = "image"
        analysis = agent.analyze_prompt_quality(poor_prompt)
        
        assert analysis["estimated_effectiveness"] == "low"


@pytest.mark.unit
@pytest.mark.agent
class TestBlogWorkflow:
    """Test BlogWorkflow functionality."""
    
    def test_execute_full_workflow(self, mock_db_config):
        """Test complete blog generation workflow."""
        workflow = BlogWorkflow()
        request_data = {
            "title": "The Future of Digital Finance",
            "company_context": "Credilinq.ai is a leading fintech company",
            "content_type": "blog"
        }
        
        # Mock all agent responses
        with patch('src.agents.specialized.planner_agent.PlannerAgent') as mock_planner, \
             patch('src.agents.specialized.research_agent.ResearchAgent') as mock_researcher, \
             patch('src.agents.specialized.writer_agent.WriterAgent') as mock_writer, \
             patch('src.agents.specialized.editor_agent.EditorAgent') as mock_editor:
            
            mock_planner.return_value.execute.return_value = "Blog outline with key sections"
            mock_researcher.return_value.execute.return_value = "Research findings and data"
            mock_writer.return_value.execute.return_value = "# The Future of Digital Finance\n\nDetailed blog content..."
            mock_editor.return_value.execute.return_value = "# The Future of Digital Finance\n\nPolished final content..."
            
            # Mock database operations
            mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
                data=[{"id": "workflow-blog-123"}]
            )
            mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
                data=[{"id": "workflow-blog-123", "status": "completed"}]
            )
            
            result = workflow.execute(request_data)
        
        assert isinstance(result, dict)
        assert "content_markdown" in result
        assert "# The Future of Digital Finance" in result["content_markdown"]

    def test_execute_linkedin_workflow(self, mock_db_config):
        """Test LinkedIn post generation workflow."""
        workflow = BlogWorkflow()
        request_data = {
            "title": "Fintech Innovation Trends",
            "company_context": "Credilinq.ai expertise in embedded finance",
            "content_type": "linkedin"
        }
        
        with patch('src.agents.specialized.planner_agent.PlannerAgent') as mock_planner, \
             patch('src.agents.specialized.research_agent.ResearchAgent') as mock_researcher, \
             patch('src.agents.specialized.writer_agent.WriterAgent') as mock_writer, \
             patch('src.agents.specialized.editor_agent.EditorAgent') as mock_editor:
            
            mock_planner.return_value.execute.return_value = "LinkedIn post outline"
            mock_researcher.return_value.execute.return_value = "Industry insights"
            mock_writer.return_value.execute.return_value = "🚀 Fintech innovation is accelerating..."
            mock_editor.return_value.execute.return_value = "🚀 Fintech innovation is accelerating with new trends..."
            
            # Mock database operations
            mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
                data=[{"id": "linkedin-post-123"}]
            )
            mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
                data=[{"id": "linkedin-post-123", "status": "completed"}]
            )
            
            result = workflow.execute(request_data)
        
        assert isinstance(result, dict)
        assert "content_markdown" in result
        assert "🚀" in result["content_markdown"]  # LinkedIn emoji style

    def test_execute_workflow_agent_failure(self, mock_db_config):
        """Test workflow with agent failure handling."""
        workflow = BlogWorkflow()
        request_data = {
            "title": "Test Blog",
            "company_context": "Test context",
            "content_type": "blog"
        }
        
        with patch('src.agents.specialized.planner_agent.PlannerAgent') as mock_planner:
            mock_planner.return_value.execute.side_effect = Exception("Planner agent failed")
            
            # Mock database insert for initial blog creation
            mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
                data=[{"id": "failed-blog-123"}]
            )
            
            with pytest.raises(Exception) as exc_info:
                workflow.execute(request_data)
            
            assert "Planner agent failed" in str(exc_info.value)

    @pytest.mark.slow
    def test_workflow_performance(self, mock_db_config, benchmark_timer):
        """Test workflow performance."""
        workflow = BlogWorkflow()
        request_data = {
            "title": "Performance Test Blog",
            "company_context": "Test context",
            "content_type": "blog"
        }
        
        # Mock fast agent responses
        with patch('src.agents.specialized.planner_agent.PlannerAgent') as mock_planner, \
             patch('src.agents.specialized.research_agent.ResearchAgent') as mock_researcher, \
             patch('src.agents.specialized.writer_agent.WriterAgent') as mock_writer, \
             patch('src.agents.specialized.editor_agent.EditorAgent') as mock_editor:
            
            mock_planner.return_value.execute.return_value = "Quick outline"
            mock_researcher.return_value.execute.return_value = "Quick research"
            mock_writer.return_value.execute.return_value = "# Quick Blog\n\nQuick content"
            mock_editor.return_value.execute.return_value = "# Quick Blog\n\nEdited content"
            
            # Mock database operations
            mock_db_config.supabase.table.return_value.insert.return_value.execute.return_value = Mock(
                data=[{"id": "perf-blog-123"}]
            )
            mock_db_config.supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
                data=[{"id": "perf-blog-123", "status": "completed"}]
            )
            
            timer = benchmark_timer()
            timer.start()
            
            result = workflow.execute(request_data)
            
            elapsed_time = timer.stop()
        
        assert isinstance(result, dict)
        # Workflow should complete within reasonable time with mock agents
        assert elapsed_time < 2.0  # 2 seconds with mocks