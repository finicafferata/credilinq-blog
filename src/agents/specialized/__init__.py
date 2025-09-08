"""Specialized AI agents for specific tasks."""

print('🔄 Enabling agent imports for workflow execution...')

# Import base classes first
from ..core.base_agent import BaseAgent, AgentResult, AgentMetadata, AgentType

# Import agent factory for registration  
from ..core.agent_factory import AgentFactory, register_agent

# Import real agent implementations from implementations/ directory
print('✅ Loading real agent implementations...')

# Import real agents where available
try:
    from .planner_agent_langgraph import PlannerAgentLangGraph as PlannerAgent
    print('✅ Imported PlannerAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import PlannerAgentLangGraph: {e}')
    try:
        from ..implementations.planner_agent_real import RealPlannerAgent as PlannerAgent
        print('✅ Fallback to RealPlannerAgent (simple)')
    except ImportError:
        PlannerAgent = None

# Use LangGraph versions (more sophisticated than simple versions)
try:
    from .researcher_agent_langgraph import ResearcherAgentLangGraph as ResearcherAgent
    print('✅ Imported ResearcherAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import ResearcherAgentLangGraph: {e}')
    try:
        from ..implementations.researcher_agent_real import RealResearcherAgent as ResearcherAgent
        print('✅ Fallback to RealResearcherAgent (simple)')
    except ImportError:
        ResearcherAgent = None

try:
    from .writer_agent_langgraph import WriterAgentLangGraph as WriterAgent
    print('✅ Imported WriterAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import WriterAgentLangGraph: {e}')
    try:
        from ..implementations.writer_agent_real import RealWriterAgent as WriterAgent
        print('✅ Fallback to RealWriterAgent (simple)')
    except ImportError:
        WriterAgent = None

try:
    from .editor_agent_langgraph import EditorAgentLangGraph as EditorAgent
    print('✅ Imported EditorAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import EditorAgentLangGraph: {e}')
    try:
        from ..implementations.editor_agent_real import RealEditorAgent as EditorAgent
        print('✅ Fallback to RealEditorAgent (simple)')
    except ImportError:
        EditorAgent = None

try:
    from .seo_agent_langgraph import SEOAgentLangGraph as SEOAgent
    print('✅ Imported SEOAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import SEOAgentLangGraph: {e}')
    try:
        from ..implementations.seo_agent_real import RealSEOAgent as SEOAgent
        print('✅ Fallback to RealSEOAgent (simple)')
    except ImportError:
        SEOAgent = None

# Import LangGraph agents (more sophisticated than simple Real agents)
try:
    from .image_prompt_agent_langgraph import ImagePromptAgentLangGraph as ImagePromptAgent
    print('✅ Imported ImagePromptAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import ImagePromptAgentLangGraph: {e}')
    try:
        from ..implementations.image_prompt_real import RealImagePromptAgent as ImagePromptAgent
        print('✅ Fallback to RealImagePromptAgent')
    except ImportError:
        ImagePromptAgent = None

try:
    from .video_prompt_agent_langgraph import VideoPromptAgentLangGraph as VideoPromptAgent
    print('✅ Imported VideoPromptAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import VideoPromptAgentLangGraph: {e}')
    VideoPromptAgent = None

try:
    from .social_media_agent_langgraph import SocialMediaAgentLangGraph as SocialMediaAgent
    print('✅ Imported SocialMediaAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import SocialMediaAgentLangGraph: {e}')
    SocialMediaAgent = None

try:
    from .search_agent_langgraph import SearchAgentWorkflow as SearchAgent
    print('✅ Imported SearchAgentWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import SearchAgentWorkflow: {e}')
    SearchAgent = None

try:
    from .content_agent_langgraph import ContentAgentWorkflow as ContentAgent
    print('✅ Imported ContentAgentWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import ContentAgentWorkflow: {e}')
    ContentAgent = None

try:
    from .content_brief_agent_langgraph import ContentBriefAgentWorkflow as ContentBriefAgent
    print('✅ Imported ContentBriefAgentWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import ContentBriefAgentWorkflow: {e}')
    ContentBriefAgent = None

try:
    from .distribution_agent_langgraph import DistributionAgentWorkflow as DistributionAgent
    print('✅ Imported DistributionAgentWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import DistributionAgentWorkflow: {e}')
    DistributionAgent = None

try:
    from .document_processor_langgraph import DocumentProcessorWorkflow as DocumentProcessorAgent
    print('✅ Imported DocumentProcessorWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import DocumentProcessorWorkflow: {e}')
    DocumentProcessorAgent = None

try:
    from .task_scheduler_langgraph import TaskSchedulerWorkflow as TaskSchedulerAgent
    print('✅ Imported TaskSchedulerWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import TaskSchedulerWorkflow: {e}')
    TaskSchedulerAgent = None

# Removed: AIContentGeneratorAgent - functionality covered by WriterAgentLangGraph

# Additional LangGraph agents
try:
    from .campaign_manager_langgraph import CampaignManagerWorkflow as CampaignManagerAgent
    print('✅ Imported CampaignManagerWorkflow (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import CampaignManagerWorkflow: {e}')
    CampaignManagerAgent = None

try:
    from .geo_analysis_agent_langgraph import GeoAnalysisAgentLangGraph as GeoAnalysisAgent
    print('✅ Imported GeoAnalysisAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import GeoAnalysisAgentLangGraph: {e}')
    GeoAnalysisAgent = None

# Removed redundant agent types - functionality merged into existing agents

# Use LangGraph content repurposer (more sophisticated than simple version)
try:
    from .content_repurposer_langgraph import ContentRepurposerAgentLangGraph as ContentRepurposerAgent
    print('✅ Imported ContentRepurposerAgentLangGraph (LangGraph)')
except ImportError as e:
    print(f'⚠️ Failed to import ContentRepurposerAgentLangGraph: {e}')
    try:
        from ..implementations.content_repurposer_real import RealContentRepurposerAgent as ContentRepurposerAgent
        print('✅ Fallback to RealContentRepurposerAgent (simple)')
    except ImportError:
        ContentRepurposerAgent = StubRepurposerAgent

# Stub agents for missing functionality (fallback if real implementations fail)
class StubImageAgent(BaseAgent):
    def __init__(self, metadata=None):
        super().__init__(metadata or AgentMetadata(
            agent_type=AgentType.IMAGE_PROMPT, name="StubImageAgent", description="Placeholder image agent"
        ))
    def execute(self, input_data, context=None):
        return AgentResult(success=True, data={"message": "Image generation placeholder"})

class StubCampaignAgent(BaseAgent):
    def __init__(self, metadata=None):
        super().__init__(metadata or AgentMetadata(
            agent_type=AgentType.CAMPAIGN_MANAGER, name="StubCampaignAgent", description="Placeholder campaign agent"
        ))
    def execute(self, input_data, context=None):
        return AgentResult(success=True, data={"message": "Campaign management placeholder"})

class StubRepurposerAgent(BaseAgent):
    def __init__(self, metadata=None):
        super().__init__(metadata or AgentMetadata(
            agent_type=AgentType.CONTENT_REPURPOSER, name="StubRepurposerAgent", description="Placeholder content repurposer"
        ))
    def execute(self, input_data, context=None):
        return AgentResult(success=True, data={"message": "Content repurposing placeholder"})

# All agents now imported with LangGraph priority and fallbacks

def _register_agents():
    """Register all available agents with the global factory."""
    print('🔧 Registering agents with factory...')
    
    # Register agents that were successfully imported using module-level function
    agents_to_register = [
        # Core content pipeline (Real implementations)
        (AgentType.PLANNER, PlannerAgent),
        (AgentType.RESEARCHER, ResearcherAgent),
        (AgentType.WRITER, WriterAgent),
        (AgentType.EDITOR, EditorAgent),
        (AgentType.SEO, SEOAgent),
        (AgentType.CONTENT_REPURPOSER, ContentRepurposerAgent),
        
        # LangGraph workflow agents
        (AgentType.IMAGE_PROMPT, ImagePromptAgent),
        (AgentType.VIDEO_PROMPT, VideoPromptAgent),
        (AgentType.SOCIAL_MEDIA, SocialMediaAgent),
        (AgentType.SEARCH, SearchAgent),
        (AgentType.CONTENT_AGENT, ContentAgent),
        (AgentType.CONTENT_BRIEF, ContentBriefAgent),
        (AgentType.DISTRIBUTION_AGENT, DistributionAgent),
        # Removed: AIContentGeneratorAgent - functionality covered by WriterAgent
        (AgentType.DOCUMENT_PROCESSOR, DocumentProcessorAgent),
        (AgentType.TASK_SCHEDULER, TaskSchedulerAgent),
        
        # Additional LangGraph agents
        (AgentType.GEO_ANALYSIS, GeoAnalysisAgent),
        
        # Campaign management
        (AgentType.CAMPAIGN_MANAGER, CampaignManagerAgent),
        
        # Removed redundant agent types - functionality covered by existing agents:
        # CONTENT_GENERATOR -> WriterAgent
        # CONTENT_OPTIMIZER -> EditorAgent + SEOAgent  
        # WORKFLOW_ORCHESTRATOR -> CampaignManagerAgent
    ]
    
    registered_count = 0
    for agent_type, agent_class in agents_to_register:
        if agent_class is not None:
            try:
                register_agent(agent_type, agent_class)
                print(f'✅ Registered {agent_type.value}')
                registered_count += 1
            except Exception as e:
                print(f'❌ Failed to register {agent_type.value}: {e}')
        else:
            print(f'⚠️  Skipping {agent_type.value} (not imported)')
    
    print(f'🎉 Agent registration complete: {registered_count} agents registered')

# Auto-register agents when module is imported
try:
    _register_agents()
except Exception as e:
    print(f'❌ Agent registration failed: {e}')

__all__ = [
    # Core content pipeline agents
    "PlannerAgent",
    "ResearcherAgent",
    "WriterAgent",
    "EditorAgent",
    "SEOAgent",
    "ContentRepurposerAgent",
    
    # Specialized content agents
    "ImagePromptAgent",
    "VideoPromptAgent",
    "SocialMediaAgent",
    "SearchAgent",
    "ContentAgent",
    "ContentBriefAgent",
    "DistributionAgent", 
    "DocumentProcessorAgent",
    "TaskSchedulerAgent",
    
    # Campaign and analysis agents
    "CampaignManagerAgent",
    "GeoAnalysisAgent"
]