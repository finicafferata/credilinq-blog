"""Specialized AI agents for specific tasks."""

from .campaign_manager import CampaignManagerAgent
from .content_agent import ContentGenerationAgent  
from .repurpose_agent import ContentRepurposingAgent
from .search_agent import WebSearchAgent
from .planner_agent import PlannerAgent
from .researcher_agent import ResearcherAgent
from .writer_agent import WriterAgent
from .editor_agent import EditorAgent
from .image_agent import ImageAgent
from .seo_agent import SEOAgent
from .social_media_agent import SocialMediaAgent

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
        register_agent(AgentType.CONTENT_REPURPOSER, ContentRepurposingAgent)
        register_agent(AgentType.IMAGE_PROMPT_GENERATOR, ImageAgent)
        register_agent(AgentType.SEO, SEOAgent)
        register_agent(AgentType.SOCIAL_MEDIA, SocialMediaAgent)
        register_agent(AgentType.SEARCH, WebSearchAgent)
        print("✅ All specialized agents registered successfully")
    except Exception as e:
        print(f"❌ Error registering agents: {e}")
        import traceback
        traceback.print_exc()

# Auto-register agents when module is imported
_register_agents()

__all__ = [
    "CampaignManagerAgent",
    "ContentGenerationAgent", 
    "ContentRepurposingAgent",
    "WebSearchAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "WriterAgent",
    "EditorAgent",
    "ImageAgent",
    "SEOAgent",
    "SocialMediaAgent"
]