"""Specialized AI agents for specific tasks."""

from .campaign_manager import CampaignManagerAgent
from .content_agent import ContentGenerationAgent  
from .content_repurposer import ContentRepurposer
from .search_agent import WebSearchAgent
from .planner_agent import PlannerAgent
from .researcher_agent import ResearcherAgent
from .writer_agent import WriterAgent
from .editor_agent import EditorAgent
from .image_prompt_agent import ImagePromptAgent
from .video_prompt_agent import VideoPromptAgent
from .seo_agent import SEOAgent
from .geo_analysis_agent import GEOAnalysisAgent
from .social_media_agent import SocialMediaAgent
from .ai_content_generator import AIContentGeneratorAgent

# Import agent factory for registration
from ..core.agent_factory import register_agent
from ..core.base_agent import AgentType

# Register all agents with the factory
def _register_agents():
    """Register all specialized agents with the global factory."""
    try:
        register_agent(AgentType.PLANNER, PlannerAgent)
        register_agent(AgentType.RESEARCHER, ResearcherAgent)
        register_agent(AgentType.WRITER, WriterAgent)
        register_agent(AgentType.EDITOR, EditorAgent)
        register_agent(AgentType.CAMPAIGN_MANAGER, CampaignManagerAgent)
        register_agent(AgentType.CONTENT_REPURPOSER, ContentRepurposer)
        register_agent(AgentType.IMAGE_PROMPT, ImagePromptAgent)
        register_agent(AgentType.VIDEO_PROMPT, VideoPromptAgent)
        register_agent(AgentType.SEO, SEOAgent)
        register_agent(AgentType.CONTENT_OPTIMIZER, GEOAnalysisAgent)
        register_agent(AgentType.SOCIAL_MEDIA, SocialMediaAgent)
        register_agent(AgentType.SEARCH, WebSearchAgent)
        register_agent(AgentType.AI_CONTENT_GENERATOR, AIContentGeneratorAgent)
        print("✅ All specialized agents registered successfully")
    except Exception as e:
        print(f"❌ Error registering agents: {e}")
        import traceback
        traceback.print_exc()

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