"""
Planner Agent - Creates structured outlines for content generation.
"""

from typing import List, Dict, Any, Optional
import ast
import re  # Added for robust parsing
from src.core.llm_client import create_llm
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ..core.langgraph_base import LangGraphAgentMixin, LangGraphExecutionContext
from ...core.security import SecurityValidator


class PlannerAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for creating structured outlines for blog posts and content.
    Now supports both LangChain and LangGraph execution modes.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.PLANNER,
                name="PlannerAgent",
                description="Creates structured outlines for content generation",
                capabilities=[
                    "outline_creation",
                    "content_structure_planning",
                    "format_adaptation",
                    "section_organization"
                ],
                version="2.1.0"  # Version bumped to reflect improvements
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
    
    def _initialize(self):
        """Initialize the LLM and other resources."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = create_llm(
                model="gemini-1.5-flash",
                temperature=0.7,
                api_key=settings.primary_api_key
            )
            self.logger.info("PlannerAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PlannerAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for planning."""
        super()._validate_input(input_data)
        
        required_fields = ["blog_title", "company_context", "content_type"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation (relaxed for natural-language fields)
        title = str(input_data["blog_title"]).strip()
        context_text = str(input_data["company_context"]).strip()
        try:
            self.security_validator.validate_content(title, "blog_title")
            self.security_validator.validate_content(context_text, "company_context")
        except Exception as e:
            self.logger.error(f"Planner input validation failed: title='{title[:80]}', error={e}")
            raise
        # content_type is a small controlled string, keep normal validation
        self.security_validator.validate_input(str(input_data["content_type"]), "content_type")
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Create a structured outline for the given content.
        
        Args:
            input_data: Dictionary containing:
                - blog_title: Title of the content
                - company_context: Company/brand context
                - content_type: Type of content (blog, linkedin, etc.)
            context: Execution context
            
        Returns:
            AgentResult: Result containing the outline
        """
        try:
            # Initialize if not already done
            if self.llm is None:
                self._initialize()
            
            blog_title = input_data["blog_title"]
            company_context = input_data["company_context"]
            content_type = input_data.get("content_type", "blog").lower()
            
            # Storing content_type for the estimator method
            self._current_content_type = content_type
            
            self.logger.info(f"Creating outline for: {blog_title} (type: {content_type})")
            
            # Create content-type specific outline
            outline = self._create_outline(blog_title, company_context, content_type)
            
            result_data = {
                "outline": outline,
                "content_type": content_type,
                "outline_length": len(outline),
                "outline_structure": self._analyze_outline_structure(outline)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "planner",
                    "outline_sections": len(outline),
                    "content_format": content_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create outline: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="OUTLINE_CREATION_FAILED"
            )
    
    def _create_outline(
        self, 
        blog_title: str, 
        company_context: str, 
        content_type: str
    ) -> List[str]:
        """Create an outline based on content type and requirements."""
        
        # Select appropriate prompt based on content type
        if content_type == "linkedin":
            prompt = self._create_linkedin_outline_prompt(blog_title, company_context)
        elif content_type == "article":
            prompt = self._create_article_outline_prompt(blog_title, company_context)
        else:  # default to blog
            prompt = self._create_blog_outline_prompt(blog_title, company_context)
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            outline = self._parse_outline_response(response.content.strip())
            
            self.logger.info(f"Created outline with {len(outline)} sections")
            return outline
            
        except Exception as e:
            self.logger.warning(f"LLM outline creation failed: {str(e)}, using fallback")
            return self._create_fallback_outline(content_type)
    
    def _create_blog_outline_prompt(self, title: str, context: str) -> str:
        """Create prompt for blog post outline."""
        return f"""
        Act as an expert Content Strategist specializing in high-converting financial services content. Create an outline following proven B2B content structures that drive engagement and conversions.

        **Blog Title:** "{title}"
        **Company Context:** "{context}"

        **PROVEN STRUCTURE REQUIREMENTS:**
        
        **Opening Section:**
        - Start with compelling hook that includes an authority quote opportunity
        - Frame the core problem/challenge that resonates with the target audience
        
        **Problem Definition Sections (2-3 sections):**
        - Break down main challenges into numbered, specific sections
        - Format as: "1. [Specific Challenge]", "2. [Another Challenge]" 
        - Each should promise concrete pain points and real examples
        
        **Solution Sections (2-3 sections):**
        - Present practical strategies/solutions
        - Format as: "5 Practical Strategies to [Solve Problem]" or "How [Company] Solves [Specific Issue]"
        - Include opportunity for customer success stories
        
        **Company Solution Section:**
        - Dedicated section showcasing how the company addresses the challenges
        - Should allow for comparison tables and specific benefits
        
        **Closing Section:**
        - Strong call-to-action focused on urgency and next steps
        - Format as: "Don't Let [Problem] Hold Back Your [Goal]" or "Take Action Today"

        **Negative Constraints:**
        - **Do not** use generic titles like "Introduction", "Benefits", or "Conclusion"
        - **Do not** create more than 8 sections (optimal is 6-7)
        - **Avoid** theoretical sections - focus on practical, actionable content
        - **Do not** include placeholders like "[Insert Detail Here]"

        **Output Format:**
        Return ONLY a Python list of strings inside <outline> tags. Example:
        <outline>
        ["Opening Hook: The Hidden Challenge Behind Growing Sales", "1. Delayed Marketplace Payouts Create Cash Flow Gaps", "2. High Upfront Inventory Costs Strain Working Capital", "5 Practical Strategies to Improve Cash Flow", "How CrediLinq Solves eCommerce Cash Flow Gaps", "Why Traditional Loans Don't Work for eCommerce", "Don't Let Cash Flow Hold Back Your Growth"]
        </outline>
        """
    
    def _create_linkedin_outline_prompt(self, title: str, context: str) -> str:
        """Create prompt for LinkedIn post outline."""
        return f"""
        Act as a social media marketing expert specializing in LinkedIn B2B content. Create a short, punchy outline for a professional post.

        **Topic:** "{title}"
        **Company Context & Goal:** "{context}"

        **Instructions:**
        - Create 3-5 concise sections optimized for high engagement on the LinkedIn feed.
        - Start with an attention-grabbing hook to stop the scroll.
        - The core sections should provide a key insight or benefit that directly relates to the company context.
        - End with an engaging question or a clear call-to-action.
        - Keep section titles focused and punchy.

        **Negative Constraints:**
        - **Avoid** overly formal or academic language.
        - **Do not** make it longer than 5 sections.

        **Output Format:**
        Return ONLY a Python list of strings inside <outline> tags. Example:
        <outline>
        ["The Surprising Truth About [Topic]", "Key Insight: Why X is Y", "How Our Solution Addresses This", "What are your thoughts?"]
        </outline>
        """
    
    def _create_article_outline_prompt(self, title: str, context: str) -> str:
        """Create prompt for article outline."""
        return f"""
        Act as a subject-matter expert and technical writer. Create a comprehensive, well-structured outline for an informative article.

        **Article Topic:** "{title}"
        **Company's Perspective/Context:** "{context}"

        **Instructions:**
        - Create 6-10 detailed sections for comprehensive, analytical coverage.
        - Structure should include background/context, deep analysis, practical implications, and a forward-looking conclusion.
        - Section titles must be informative and reflect the content's depth. Ensure the outline directly incorporates the company's perspective.

        **Negative Constraints:**
        - **Do not** use generic section titles like "Introduction", "Analysis", or "Conclusion". Be specific, e.g., "Analyzing the Market Shift" instead of "Analysis".
        - **Avoid** placeholders or vague concepts. Every section title should promise concrete information.

        **Output Format:**
        Return ONLY a Python list of strings inside <outline> tags. Example:
        <outline>
        ["Abstract and Key Takeaways", "1. Historical Context of [Topic]", "2. In-Depth Analysis of Current Trends", "3. Case Study: Applying [Company's Perspective]", "4. Future Outlook and Recommendations"]
        </outline>
        """
    
    def _parse_outline_response(self, response: str) -> List[str]:
        """Parse the LLM response to extract the outline."""
        try:
            # V2 Parsing: First, try to find content within <outline> tags for robustness
            match = re.search(r"<outline>(.*?)</outline>", response, re.DOTALL)
            if match:
                # If tags are found, use the content within them
                content_to_parse = match.group(1).strip()
            else:
                # If no tags, use the whole response (for backward compatibility or model error)
                content_to_parse = response

            # Try to parse as Python list
            outline = ast.literal_eval(content_to_parse)
            if isinstance(outline, list) and all(isinstance(item, str) for item in outline):
                return outline
        except (ValueError, SyntaxError, AttributeError):
            # If ast.literal_eval fails, proceed to fallback
            self.logger.warning("Failed to parse response as a Python list, falling back to line-by-line parsing.")
            pass
        
        # Fallback: parse line by line from the original response
        lines = response.strip().split('\n')
        outline = []
        
        for line in lines:
            line = line.strip()
            # Ignore common non-outline lines
            if line and not line.lower().startswith(('<outline>', '</outline>', '`', 'python')):
                # Remove list markers and quotes, then clean up
                line = re.sub(r'^[\[\s\-*•"\']+|[\]\s,"\']+ what do you think?', '', line).strip()
                if line:
                    outline.append(line)
        
        return outline if outline else self._create_fallback_outline("blog")
    
    def _create_fallback_outline(self, content_type: str) -> List[str]:
        """Create a fallback outline when LLM fails."""
        self.logger.info(f"Using fallback outline for content type: {content_type}")
        if content_type == "linkedin":
            return [
                "Attention-Grabbing Hook",
                "Key Insight or Problem",
                "Solution or Benefits",
                "Call to Action"
            ]
        elif content_type == "article":
            return [
                "Introduction",
                "Background and Context",
                "Current Situation Analysis",
                "Key Findings",
                "Implications",
                "Recommendations",
                "Conclusion"
            ]
        else:  # blog
            return [
                "Introduction: Setting the Stage",
                "Understanding the Core Problem",
                "A New Solution on the Horizon",
                "Key Benefits for Your Business",
                "Step-by-Step Implementation Guide",
                "Best Practices to Maximize Results",
                "Conclusion: The Future is Now"
            ]
    
    def _analyze_outline_structure(self, outline: List[str]) -> Dict[str, Any]:
        """Analyze the structure of the created outline."""
        structure = {
            "total_sections": len(outline),
            "has_introduction": any("intro" in section.lower() for section in outline),
            "has_conclusion": any("conclu" in section.lower() or "action" in section.lower() or "takeaway" in section.lower() for section in outline),
            "section_types": self._categorize_sections(outline),
            "estimated_length": self._estimate_content_length(outline)
        }
        
        return structure
    
    def _categorize_sections(self, outline: List[str]) -> Dict[str, int]:
        """Categorize sections by type."""
        categories = {
            "introductory": 0,
            "content": 0,
            "conclusive": 0
        }
        
        intro_keywords = ["intro", "hook", "opening", "overview", "unpac", "setting the stage"]
        conclusion_keywords = ["conclu", "summary", "action", "takeaway", "final", "next step", "outlook"]
        
        # Assume first section is intro and last is conclusion if not otherwise specified
        if len(outline) > 1:
            if any(keyword in outline[0].lower() for keyword in intro_keywords):
                categories["introductory"] += 1
            else: # Fallback categorization
                 categories["introductory"] += 1

            if any(keyword in outline[-1].lower() for keyword in conclusion_keywords):
                categories["conclusive"] += 1
            else: # Fallback categorization
                categories["conclusive"] += 1

            # Categorize the middle sections
            content_sections = outline[1:-1]
            for section in content_sections:
                section_lower = section.lower()
                if any(keyword in section_lower for keyword in conclusion_keywords):
                     categories["conclusive"] += 1
                else:
                    categories["content"] += 1
            # Add the non-keyword first and last sections to content if they dont match
            if not any(keyword in outline[0].lower() for keyword in intro_keywords):
                categories["content"] += 1
            if not any(keyword in outline[-1].lower() for keyword in conclusion_keywords):
                categories["content"] +=1
        elif outline: # If only one section
             categories["content"] += 1


        return categories
    
    def _estimate_content_length(self, outline: List[str]) -> Dict[str, Any]:
        """Estimate content length based on outline."""
        base_words_per_section = {
            "blog": 250,
            "linkedin": 50,
            "article": 350
        }
        
        content_type = getattr(self, '_current_content_type', 'blog')
        words_per_section = base_words_per_section.get(content_type, 250)
        
        estimated_words = len(outline) * words_per_section
        estimated_reading_time = max(1, estimated_words // 200)  # 200 words per minute
        
        return {
            "estimated_words": estimated_words,
            "estimated_reading_time_minutes": estimated_reading_time,
            "sections": len(outline)
        }
    
    def create_custom_outline(
        self,
        topic: str,
        target_sections: int,
        content_focus: str,
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Create a custom outline with specific requirements.
        
        Args:
            topic: Topic for the outline
            target_sections: Desired number of sections
            content_focus: Focus area (technical, marketing, educational, etc.)
            context: Execution context
            
        Returns:
            AgentResult: Custom outline result
        """
        # This custom function can be enhanced with its own specific prompt logic
        # For now, it leverages the existing execute flow.
        self.logger.info(f"Creating custom outline for '{topic}' with focus on {content_focus}.")
        
        input_data = {
            "blog_title": topic,
            "company_context": f"Create a piece of content with a strong focus on '{content_focus}'. The target number of sections is {target_sections}.",
            "content_type": "article",  # Defaulting to 'article' for custom detailed outlines
        }
        
        return self.execute(input_data, context)
    
    async def execute_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[LangGraphExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Async execution method supporting both LangChain and LangGraph workflows.
        
        Args:
            input_data: Input data for planning
            context: LangGraph execution context
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Planning result
        """
        # If LangGraph is enabled, use LangGraph execution
        if self._langgraph_enabled and self._workflow:
            try:
                return await self.execute_langgraph(input_data, context)
            except Exception as e:
                self.logger.error(f"LangGraph execution failed: {e}")
                
                # Fallback to LangChain if enabled
                if getattr(self, '_fallback_to_langchain', True):
                    self.logger.info("Falling back to LangChain execution")
                    agent_context = context.to_agent_context() if context else None
                    return self.execute_safe(input_data, agent_context, **kwargs)
                raise
        
        # Use standard LangChain execution
        agent_context = context.to_agent_context() if context else None
        return self.execute_safe(input_data, agent_context, **kwargs)
    
    def get_execution_modes(self) -> Dict[str, bool]:
        """Get available execution modes for this agent."""
        return {
            "langchain": True,
            "langgraph": self._langgraph_enabled,
            "hybrid": self._langgraph_enabled and getattr(self, '_fallback_to_langchain', True)
        }