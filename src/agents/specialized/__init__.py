"""Specialized AI agents for specific tasks."""

# Import from LangGraph versions - temporarily disabled for migration
# from .campaign_manager_langgraph import CampaignManagerAgent
# from .content_agent_langgraph import ContentGenerationAgent  
# from .content_repurposer_langgraph import ContentRepurposer
# from .search_agent_langgraph import WebSearchAgent
# from .planner_agent_langgraph import PlannerAgent
# from .researcher_agent_langgraph import ResearcherAgent
# from .writer_agent_langgraph import WriterAgent
# from .editor_agent_langgraph import EditorAgent
# from .image_prompt_agent_langgraph import ImagePromptAgent
# from .video_prompt_agent_langgraph import VideoPromptAgent
# from .seo_agent_langgraph import SEOAgent
# from .geo_analysis_agent_langgraph import GEOAnalysisAgent
# from .social_media_agent_langgraph import SocialMediaAgent
# from .ai_content_generator_langgraph import AIContentGeneratorAgent

print('⚠️  Agent imports temporarily disabled during LangGraph migration')

# Import agent factory for registration
from ..core.agent_factory import register_agent
from ..core.base_agent import AgentType

# Register all agents with the factory - temporarily disabled
def _register_agents():
    """Register all specialized agents with the global factory."""
    print('⚠️  Agent registration temporarily disabled during LangGraph migration')
    pass

# Auto-register agents when module is imported
# _register_agents()  # Disabled - using factory registration instead

__all__ = [
    "CampaignManagerAgent",
    "ContentGenerationAgent", 
    "ContentRepurposingAgent",
    "WebSearchAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "WriterAgent",
    "EditorAgent",
    "ImagePromptAgent",
    "VideoPromptAgent",
    "SEOAgent",
    "GEOAnalysisAgent",
    "SocialMediaAgent",
    "AIContentGeneratorAgent"
]