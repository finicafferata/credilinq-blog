"""Agent workflow orchestration."""

# from .blog_workflow import app as legacy_blog_agent_app, BlogWriterState  # Disabled due to langgraph dependency
from .structured_blog_workflow import BlogWorkflowCompatibility

# Content generation workflow components
from .content_generation_workflow import ContentGenerationWorkflow, content_generation_workflow
from .content_workflow_manager import ContentWorkflowManager, content_workflow_manager
from .task_management_system import TaskManagementSystem, task_management_system

# Enhanced workflow components (temporarily disabled due to circular imports)
try:
    # from .enhanced.workflow_execution_engine import WorkflowExecutionEngine
    # from .enhanced.campaign_workflow_builder import CampaignWorkflowBuilder
    # from .enhanced.enhanced_workflow_state import CampaignWorkflowState, WorkflowStatus
    # from .enhanced.campaign_state_manager import CampaignStateManager
    ENHANCED_WORKFLOWS_AVAILABLE = False  # Temporarily disabled
except ImportError:
    ENHANCED_WORKFLOWS_AVAILABLE = False

# Create a compatibility layer that provides the same interface as the legacy workflow
# but uses the new structured approach
_structured_workflow = BlogWorkflowCompatibility()

class BlogAgentApp:
    """
    Compatibility wrapper that provides the legacy interface while using the new structured workflow.
    """
    
    def __init__(self):
        self.structured_workflow = _structured_workflow
    
    def invoke(self, input_data):
        """
        Invoke the blog workflow with legacy interface compatibility.
        
        Args:
            input_data: Dictionary with blog creation parameters
            
        Returns:
            Dictionary with 'final_post' key for compatibility
        """
        try:
            # Use the new structured workflow
            result = self.structured_workflow.execute(input_data)
            return result
        except Exception as e:
            # Return error information
            return {
                "final_post": f"Error: Structured workflow failed. Error: {str(e)}",
                "success": False,
                "error": str(e)
            }

# Create the compatible instance
blog_agent_app = BlogAgentApp()

# Export all workflow components
__all__ = [
    "blog_agent_app", 
    "BlogWorkflowCompatibility",
    "ContentGenerationWorkflow",
    "content_generation_workflow", 
    "ContentWorkflowManager",
    "content_workflow_manager",
    "TaskManagementSystem", 
    "task_management_system"
]

# Add enhanced workflows to exports if available (temporarily disabled)
# if ENHANCED_WORKFLOWS_AVAILABLE:
#     __all__.extend([
#         "WorkflowExecutionEngine",
#         "CampaignWorkflowBuilder", 
#         "CampaignWorkflowState",
#         "WorkflowStatus",
#         "CampaignStateManager"
#     ])