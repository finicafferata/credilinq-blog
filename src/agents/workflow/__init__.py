"""Agent workflow orchestration."""

# from .blog_workflow import app as legacy_blog_agent_app, BlogWriterState  # Disabled due to langgraph dependency
from .structured_blog_workflow import BlogWorkflowCompatibility

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

__all__ = ["blog_agent_app", "BlogWorkflowCompatibility"]