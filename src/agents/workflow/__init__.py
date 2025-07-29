"""Agent workflow orchestration."""

from .blog_workflow import app as legacy_blog_agent_app, BlogWriterState
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
        self.legacy_workflow = legacy_blog_agent_app
    
    def invoke(self, input_data):
        """
        Invoke the blog workflow with legacy interface compatibility.
        
        Args:
            input_data: Dictionary with blog creation parameters
            
        Returns:
            Dictionary with 'final_post' key for compatibility
        """
        try:
            # Try the new structured workflow first
            result = self.structured_workflow.execute(input_data)
            return result
        except Exception as e:
            # Fallback to legacy workflow if structured fails
            try:
                return self.legacy_workflow.invoke(input_data)
            except Exception as fallback_error:
                # If both fail, return error information
                return {
                    "final_post": f"Error: Both structured and legacy workflows failed. "
                                f"Structured error: {str(e)}. Legacy error: {str(fallback_error)}",
                    "success": False,
                    "error": str(e)
                }

# Create the compatible instance
blog_agent_app = BlogAgentApp()

__all__ = ["blog_agent_app", "BlogWriterState"]