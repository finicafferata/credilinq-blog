"""
LangGraph-based PlannerAgent workflow implementation.

This provides a LangGraph workflow version of the PlannerAgent that can run
alongside or replace the existing LangChain implementation.
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass
import json
import ast
import re

from ..core.langgraph_compat import StateGraph, START, END
from langchain_core.messages import SystemMessage
from src.core.llm_client import create_llm

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState, WorkflowStatus
from ..core.base_agent import AgentResult, AgentType
from ...core.security import SecurityValidator


# State definition for PlannerAgent LangGraph workflow
class PlannerState(TypedDict):
    """State for the PlannerAgent LangGraph workflow."""
    # Workflow management
    workflow_id: str
    status: str
    current_step: Optional[str]
    error_message: Optional[str]
    
    # Input data
    blog_title: str
    company_context: str
    content_type: str
    
    # Processing data
    outline: Optional[List[str]]
    outline_analysis: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    
    # Output data
    final_outline: Optional[List[str]]
    outline_metadata: Optional[Dict[str, Any]]


@dataclass
class PlannerWorkflowConfig:
    """Configuration for PlannerAgent workflow."""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_outline_sections: int = 10
    min_outline_sections: int = 3
    enable_validation: bool = True
    enable_fallback: bool = True


class PlannerAgentWorkflow(LangGraphWorkflowBase[PlannerState]):
    """
    LangGraph workflow implementation for content outline planning.
    """
    
    def __init__(
        self, 
        workflow_name: str = "planner_agent_workflow",
        config: Optional[PlannerWorkflowConfig] = None,
        **kwargs
    ):
        self.config = config or PlannerWorkflowConfig()
        self.security_validator = SecurityValidator()
        self.llm = None
        super().__init__(workflow_name, **kwargs)
    
    def _initialize_llm(self):
        """Initialize the LLM if not already done."""
        if self.llm is None:
            try:
                from ...config.settings import get_settings
                settings = get_settings()
                
                self.llm = create_llm(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    api_key=settings.primary_api_key
                )
                self.logger.info(f"LLM initialized: {self.config.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM: {e}")
                raise
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> PlannerState:
        """Create initial state from input data."""
        return PlannerState(
            workflow_id=input_data.get('workflow_id', 'unknown'),
            status=WorkflowStatus.RUNNING.value,
            current_step=None,
            error_message=None,
            blog_title=input_data['blog_title'],
            company_context=input_data['company_context'],
            content_type=input_data.get('content_type', 'blog'),
            outline=None,
            outline_analysis=None,
            validation_results=None,
            final_outline=None,
            outline_metadata=None
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow structure."""
        workflow = StateGraph(PlannerState)
        
        # Add workflow nodes
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("generate_outline", self._generate_outline_node)
        workflow.add_node("analyze_outline", self._analyze_outline_node)
        workflow.add_node("validate_outline", self._validate_outline_node)
        workflow.add_node("finalize_outline", self._finalize_outline_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("validate_input")
        
        # Add edges
        workflow.add_edge("validate_input", "generate_outline")
        workflow.add_edge("generate_outline", "analyze_outline")
        workflow.add_edge("analyze_outline", "validate_outline")
        
        # Conditional edges for validation
        workflow.add_conditional_edges(
            "validate_outline",
            self._should_retry_outline,
            {
                "retry": "generate_outline",
                "finalize": "finalize_outline",
                "error": "handle_error"
            }
        )
        
        # Terminal edges
        workflow.add_edge("finalize_outline", END)
        workflow.add_edge("handle_error", END)
        
        return workflow
    
    def _validate_input_node(self, state: PlannerState) -> PlannerState:
        """Validate input data for the workflow."""
        try:
            self.logger.info(f"Validating input for workflow {state['workflow_id']}")
            
            # Check required fields
            required_fields = ['blog_title', 'company_context', 'content_type']
            missing_fields = [field for field in required_fields if not state.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Security validation
            title = str(state['blog_title']).strip()
            context = str(state['company_context']).strip()
            
            self.security_validator.validate_content(title, "blog_title")
            self.security_validator.validate_content(context, "company_context")
            self.security_validator.validate_input(str(state['content_type']), "content_type")
            
            state['current_step'] = "validate_input"
            self.logger.info("Input validation completed successfully")
            
        except Exception as e:
            state['error_message'] = f"Input validation failed: {str(e)}"
            state['status'] = WorkflowStatus.FAILED.value
            self.logger.error(f"Input validation error: {e}")
        
        return state
    
    def _generate_outline_node(self, state: PlannerState) -> PlannerState:
        """Generate content outline using LLM."""
        try:
            self.logger.info("Generating content outline")
            state['current_step'] = "generate_outline"
            
            # Initialize LLM if needed
            self._initialize_llm()
            
            blog_title = state['blog_title']
            company_context = state['company_context'] 
            content_type = state['content_type'].lower()
            
            # Generate outline based on content type
            outline = self._create_outline(blog_title, company_context, content_type)
            
            state['outline'] = outline
            self.logger.info(f"Generated outline with {len(outline)} sections")
            
        except Exception as e:
            state['error_message'] = f"Outline generation failed: {str(e)}"
            state['status'] = WorkflowStatus.FAILED.value
            self.logger.error(f"Outline generation error: {e}")
        
        return state
    
    def _analyze_outline_node(self, state: PlannerState) -> PlannerState:
        """Analyze the generated outline structure."""
        try:
            self.logger.info("Analyzing outline structure")
            state['current_step'] = "analyze_outline"
            
            outline = state['outline']
            if not outline:
                raise ValueError("No outline to analyze")
            
            analysis = self._analyze_outline_structure(outline, state['content_type'])
            state['outline_analysis'] = analysis
            
            self.logger.info(f"Outline analysis completed: {analysis['total_sections']} sections")
            
        except Exception as e:
            state['error_message'] = f"Outline analysis failed: {str(e)}"
            state['status'] = WorkflowStatus.FAILED.value
            self.logger.error(f"Outline analysis error: {e}")
        
        return state
    
    def _validate_outline_node(self, state: PlannerState) -> PlannerState:
        """Validate the outline meets quality criteria."""
        try:
            self.logger.info("Validating outline quality")
            state['current_step'] = "validate_outline"
            
            outline = state['outline']
            analysis = state['outline_analysis']
            
            if not outline or not analysis:
                raise ValueError("Missing outline or analysis data")
            
            validation_results = self._validate_outline_quality(outline, analysis, state['content_type'])
            state['validation_results'] = validation_results
            
            self.logger.info(f"Outline validation completed: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
            
        except Exception as e:
            state['error_message'] = f"Outline validation failed: {str(e)}"
            state['status'] = WorkflowStatus.FAILED.value
            self.logger.error(f"Outline validation error: {e}")
        
        return state
    
    def _finalize_outline_node(self, state: PlannerState) -> PlannerState:
        """Finalize the outline and prepare output."""
        try:
            self.logger.info("Finalizing outline")
            state['current_step'] = "finalize_outline"
            
            outline = state['outline']
            analysis = state['outline_analysis']
            validation = state['validation_results']
            
            # Create final outline
            state['final_outline'] = outline
            
            # Create comprehensive metadata
            state['outline_metadata'] = {
                'outline_length': len(outline),
                'content_type': state['content_type'],
                'structure_analysis': analysis,
                'validation_results': validation,
                'workflow_id': state['workflow_id'],
                'agent_type': 'planner_langgraph',
                'generation_method': 'langgraph_workflow'
            }
            
            state['status'] = WorkflowStatus.COMPLETED.value
            self.logger.info("Outline finalization completed successfully")
            
        except Exception as e:
            state['error_message'] = f"Outline finalization failed: {str(e)}"
            state['status'] = WorkflowStatus.FAILED.value
            self.logger.error(f"Outline finalization error: {e}")
        
        return state
    
    def _handle_error_node(self, state: PlannerState) -> PlannerState:
        """Handle workflow errors and provide fallback."""
        try:
            self.logger.warning(f"Handling workflow error: {state.get('error_message')}")
            state['current_step'] = "handle_error"
            
            if self.config.enable_fallback:
                # Provide fallback outline
                fallback_outline = self._create_fallback_outline(state['content_type'])
                state['final_outline'] = fallback_outline
                state['outline_metadata'] = {
                    'outline_length': len(fallback_outline),
                    'content_type': state['content_type'],
                    'is_fallback': True,
                    'original_error': state['error_message'],
                    'workflow_id': state['workflow_id'],
                    'agent_type': 'planner_langgraph',
                    'generation_method': 'fallback'
                }
                state['status'] = WorkflowStatus.COMPLETED.value
                self.logger.info("Fallback outline provided")
            else:
                state['status'] = WorkflowStatus.FAILED.value
                self.logger.error("Error handling completed without fallback")
            
        except Exception as e:
            state['error_message'] = f"Error handling failed: {str(e)}"
            state['status'] = WorkflowStatus.FAILED.value
            self.logger.error(f"Error handling error: {e}")
        
        return state
    
    def _should_retry_outline(self, state: PlannerState) -> str:
        """Determine if outline should be retried or finalized."""
        validation_results = state.get('validation_results', {})
        
        if state.get('error_message'):
            return "error"
        
        if not validation_results.get('is_valid', False):
            retry_count = state.get('retry_count', 0)
            if retry_count < self.max_retries:
                state['retry_count'] = retry_count + 1
                self.logger.info(f"Retrying outline generation (attempt {retry_count + 1})")
                return "retry"
            else:
                self.logger.warning("Max retries reached, proceeding with current outline")
                return "finalize"
        
        return "finalize"
    
    def _create_outline(self, blog_title: str, company_context: str, content_type: str) -> List[str]:
        """Create outline based on content type."""
        if content_type == "linkedin":
            prompt = self._create_linkedin_outline_prompt(blog_title, company_context)
        elif content_type == "article":
            prompt = self._create_article_outline_prompt(blog_title, company_context)
        else:  # default to blog
            prompt = self._create_blog_outline_prompt(blog_title, company_context)
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            outline = self._parse_outline_response(response.content.strip())
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
            # Try to find content within <outline> tags
            match = re.search(r"<outline>(.*?)</outline>", response, re.DOTALL)
            if match:
                content_to_parse = match.group(1).strip()
            else:
                content_to_parse = response

            # Try to parse as Python list
            outline = ast.literal_eval(content_to_parse)
            if isinstance(outline, list) and all(isinstance(item, str) for item in outline):
                return outline
        except (ValueError, SyntaxError, AttributeError):
            self.logger.warning("Failed to parse response as Python list, falling back to line-by-line parsing")
        
        # Fallback: parse line by line
        lines = response.strip().split('\n')
        outline = []
        
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('<outline>', '</outline>', '`', 'python')):
                line = re.sub(r'^[\[\s\-*•"\']+|[\]\s,"\']+ what do you think?', '', line).strip()
                if line:
                    outline.append(line)
        
        return outline if outline else self._create_fallback_outline("blog")
    
    def _analyze_outline_structure(self, outline: List[str], content_type: str) -> Dict[str, Any]:
        """Analyze the structure of the created outline."""
        structure = {
            "total_sections": len(outline),
            "has_introduction": any("intro" in section.lower() for section in outline),
            "has_conclusion": any("conclu" in section.lower() or "action" in section.lower() or "takeaway" in section.lower() for section in outline),
            "section_types": self._categorize_sections(outline),
            "estimated_length": self._estimate_content_length(outline, content_type),
            "quality_score": self._calculate_outline_quality_score(outline)
        }
        
        return structure
    
    def _categorize_sections(self, outline: List[str]) -> Dict[str, int]:
        """Categorize sections by type."""
        categories = {
            "introductory": 0,
            "content": 0,
            "conclusive": 0
        }
        
        intro_keywords = ["intro", "hook", "opening", "overview", "unpack", "setting the stage"]
        conclusion_keywords = ["conclu", "summary", "action", "takeaway", "final", "next step", "outlook"]
        
        for i, section in enumerate(outline):
            section_lower = section.lower()
            if i == 0 and any(keyword in section_lower for keyword in intro_keywords):
                categories["introductory"] += 1
            elif i == len(outline) - 1 and any(keyword in section_lower for keyword in conclusion_keywords):
                categories["conclusive"] += 1
            else:
                categories["content"] += 1
        
        return categories
    
    def _estimate_content_length(self, outline: List[str], content_type: str) -> Dict[str, Any]:
        """Estimate content length based on outline."""
        base_words_per_section = {
            "blog": 250,
            "linkedin": 50,
            "article": 350
        }
        
        words_per_section = base_words_per_section.get(content_type, 250)
        estimated_words = len(outline) * words_per_section
        estimated_reading_time = max(1, estimated_words // 200)
        
        return {
            "estimated_words": estimated_words,
            "estimated_reading_time_minutes": estimated_reading_time,
            "sections": len(outline)
        }
    
    def _calculate_outline_quality_score(self, outline: List[str]) -> float:
        """Calculate a quality score for the outline."""
        score = 0.0
        
        # Check for proper length
        if 4 <= len(outline) <= 8:
            score += 25
        elif len(outline) < 4:
            score -= 10
        
        # Check for specific and actionable titles
        specific_words = ["how", "why", "what", "when", "strategies", "steps", "guide", "tips"]
        specific_count = sum(1 for section in outline if any(word in section.lower() for word in specific_words))
        score += min(25, specific_count * 5)
        
        # Check for numbered sections (good structure)
        numbered_sections = sum(1 for section in outline if re.match(r'^\d+\.', section.strip()))
        if numbered_sections > 0:
            score += 15
        
        # Check for variety in section lengths
        lengths = [len(section) for section in outline]
        if len(set(lengths)) > 1:  # Different lengths indicate variety
            score += 10
        
        # Check for call-to-action or engaging elements
        engaging_words = ["action", "don't", "transform", "discover", "unlock", "proven"]
        engaging_count = sum(1 for section in outline if any(word in section.lower() for word in engaging_words))
        score += min(15, engaging_count * 3)
        
        # Check for company/brand integration
        brand_integration = sum(1 for section in outline if "company" in section.lower() or "solution" in section.lower())
        if brand_integration > 0:
            score += 10
        
        return min(100.0, score)
    
    def _validate_outline_quality(
        self, 
        outline: List[str], 
        analysis: Dict[str, Any], 
        content_type: str
    ) -> Dict[str, Any]:
        """Validate outline meets quality criteria."""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "quality_score": analysis.get("quality_score", 0)
        }
        
        # Check minimum length
        if len(outline) < self.config.min_outline_sections:
            validation_results["issues"].append(f"Outline too short: {len(outline)} sections (minimum: {self.config.min_outline_sections})")
            validation_results["is_valid"] = False
        
        # Check maximum length
        if len(outline) > self.config.max_outline_sections:
            validation_results["warnings"].append(f"Outline may be too long: {len(outline)} sections (recommended max: {self.config.max_outline_sections})")
        
        # Check for empty sections
        empty_sections = [i for i, section in enumerate(outline) if not section.strip()]
        if empty_sections:
            validation_results["issues"].append(f"Empty sections found at positions: {empty_sections}")
            validation_results["is_valid"] = False
        
        # Check quality score threshold
        quality_score = analysis.get("quality_score", 0)
        if quality_score < 50:
            validation_results["warnings"].append(f"Low quality score: {quality_score}")
        
        # Content type specific validation
        if content_type == "linkedin" and len(outline) > 5:
            validation_results["warnings"].append("LinkedIn posts should be concise (5 sections max)")
        
        if content_type == "blog":
            if not analysis.get("has_introduction") and not analysis.get("has_conclusion"):
                validation_results["warnings"].append("Blog should have clear introduction and conclusion")
        
        return validation_results
    
    def _create_fallback_outline(self, content_type: str) -> List[str]:
        """Create a fallback outline when generation fails."""
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


# Convenience function to create and configure the workflow
def create_planner_workflow(
    config: Optional[PlannerWorkflowConfig] = None,
    **kwargs
) -> PlannerAgentWorkflow:
    """Create a configured PlannerAgent LangGraph workflow."""
    return PlannerAgentWorkflow(
        workflow_name="planner_agent_langgraph",
        config=config,
        **kwargs
    )