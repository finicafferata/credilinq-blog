"""
Writer Agent - Generates content based on outline and research.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator


class WriterAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for writing content based on outline and research.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WRITER,
                name="WriterAgent",
                description="Generates high-quality content based on outline and research",
                capabilities=[
                    "content_generation",
                    "markdown_formatting",
                    "tone_adaptation", 
                    "format_optimization",
                    "revision_incorporation"
                ],
                version="2.0.0"
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
    
    def _initialize(self):
        """Initialize the LLM."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            self.logger.info("WriterAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize WriterAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for writing."""
        super()._validate_input(input_data)
        
        required_fields = ["outline", "research", "blog_title", "company_context"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types
        if not isinstance(input_data["outline"], list):
            raise ValueError("Outline must be a list")
        
        if not isinstance(input_data["research"], dict):
            raise ValueError("Research must be a dictionary")
        
        # Security validation
        self.security_validator.validate_input(str(input_data["blog_title"]))
        self.security_validator.validate_input(str(input_data["company_context"]))
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate content based on outline and research.
        
        Args:
            input_data: Dictionary containing:
                - outline: List of sections
                - research: Research results by section
                - blog_title: Title of the content
                - company_context: Company context
                - content_type: Type of content (optional)
                - review_notes: Editor feedback for revisions (optional)
            context: Execution context
            
        Returns:
            AgentResult: Generated content
        """
        try:
            # Initialize if not already done
            if self.llm is None:
                self._initialize()
            
            outline = input_data["outline"]
            research = input_data["research"]
            blog_title = input_data["blog_title"]
            company_context = input_data["company_context"]
            content_type = input_data.get("content_type", "blog").lower()
            review_notes = input_data.get("review_notes")
            
            self.logger.info(f"Writing {content_type} content: {blog_title}")
            
            # Generate content based on type
            if content_type == "linkedin":
                content = self._generate_linkedin_content(
                    outline, research, blog_title, company_context, review_notes
                )
            elif content_type == "article":
                content = self._generate_article_content(
                    outline, research, blog_title, company_context, review_notes
                )
            else:  # default to blog
                content = self._generate_blog_content(
                    outline, research, blog_title, company_context, review_notes
                )
            
            # Analyze generated content
            content_analysis = self._analyze_content(content, content_type)
            
            result_data = {
                "content": content,
                "content_type": content_type,
                "word_count": content_analysis["word_count"],
                "reading_time": content_analysis["reading_time"],
                "sections_covered": len(outline),
                "content_analysis": content_analysis
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "writer",
                    "content_format": content_type,
                    "word_count": content_analysis["word_count"],
                    "revision": bool(review_notes)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CONTENT_GENERATION_FAILED"
            )
    
    def _generate_blog_content(
        self,
        outline: list,
        research: dict,
        blog_title: str,
        company_context: str,
        review_notes: Optional[str] = None
    ) -> str:
        """Generate blog post content."""
        
        research_text = self._format_research_for_prompt(research)
        revision_notes = self._format_revision_notes(review_notes)
        
        prompt = f"""You are 'ContextMark', an expert blog writer with 15+ years of experience. Write a comprehensive, engaging blog post.

BLOG TITLE: {blog_title}

OUTLINE TO FOLLOW:
{self._format_outline_for_prompt(outline)}

RESEARCH TO USE:
---
{research_text}
---

COMPANY CONTEXT & TONE:
{company_context}

{revision_notes}

BLOG-SPECIFIC REQUIREMENTS:
- Write in professional yet conversational tone
- Use the research as your primary source of truth
- Follow the outline structure closely
- Include engaging introduction and strong conclusion
- Use proper Markdown formatting with headers (## for main sections)
- Aim for 1500-2500 words for comprehensive coverage
- Include actionable insights and practical advice
- Provide detailed explanations and examples
- Use bullet points and numbered lists where appropriate
- Maintain consistency with company voice
- Ensure content is original and well-researched
- Include relevant statistics or data from research
- Make content scannable with subheadings

Write the complete blog post in Markdown format now."""

        response = self.llm.invoke([SystemMessage(content=prompt)])
        return response.content.strip()
    
    def _generate_linkedin_content(
        self,
        outline: list,
        research: dict,
        blog_title: str,
        company_context: str,
        review_notes: Optional[str] = None
    ) -> str:
        """Generate LinkedIn post content."""
        
        research_text = self._format_research_for_prompt(research)
        revision_notes = self._format_revision_notes(review_notes)
        
        prompt = f"""You are 'ContextMark', an expert LinkedIn content creator with 15+ years of experience in professional social media. Write an engaging, professional LinkedIn post.

LINKEDIN POST TITLE: {blog_title}

OUTLINE TO FOLLOW:
{self._format_outline_for_prompt(outline)}

RESEARCH TO USE:
---
{research_text}
---

COMPANY CONTEXT & TONE:
{company_context}

{revision_notes}

LINKEDIN-SPECIFIC REQUIREMENTS:
- Write in a professional yet personable tone that encourages engagement
- Use the research as your primary source of truth
- Follow the outline structure closely
- Start with a compelling hook that grabs attention
- Include relevant emojis sparingly and professionally
- Aim for 800-1200 words (LinkedIn optimal length)
- Include actionable insights that professionals can implement
- End with a call-to-action or thought-provoking question
- Use line breaks and short paragraphs for mobile readability
- Include relevant hashtags at the end (3-5 maximum)
- Maintain consistency with company voice and professional brand
- Use storytelling elements where appropriate
- Include specific examples or case studies from research

Write the complete LinkedIn post now. Use minimal formatting - just line breaks and emojis where appropriate."""

        response = self.llm.invoke([SystemMessage(content=prompt)])
        return response.content.strip()
    
    def _generate_article_content(
        self,
        outline: list,
        research: dict,
        blog_title: str,
        company_context: str,
        review_notes: Optional[str] = None
    ) -> str:
        """Generate article content."""
        
        research_text = self._format_research_for_prompt(research)
        revision_notes = self._format_revision_notes(review_notes)
        
        prompt = f"""You are 'ContextMark', an expert article writer specializing in analytical and informative content. Write a comprehensive article.

ARTICLE TITLE: {blog_title}

OUTLINE TO FOLLOW:
{self._format_outline_for_prompt(outline)}

RESEARCH TO USE:
---
{research_text}
---

COMPANY CONTEXT & TONE:
{company_context}

{revision_notes}

ARTICLE-SPECIFIC REQUIREMENTS:
- Write in an analytical, informative tone
- Use the research as your primary source of truth
- Follow the outline structure closely
- Include data-driven insights and analysis
- Use proper Markdown formatting with headers
- Aim for 2000-3000 words for in-depth coverage
- Include citations or references where appropriate
- Provide objective analysis and multiple perspectives
- Use charts, graphs, or data visualizations concepts
- Maintain journalistic integrity and accuracy
- Include expert opinions or case studies from research
- End with actionable recommendations or future outlook

Write the complete article in Markdown format now."""

        response = self.llm.invoke([SystemMessage(content=prompt)])
        return response.content.strip()
    
    def _format_research_for_prompt(self, research: dict) -> str:
        """Format research data for the prompt."""
        formatted_research = []
        
        for section, research_data in research.items():
            if isinstance(research_data, dict):
                content = research_data.get("content", "")
            else:
                content = str(research_data)
            
            formatted_research.append(f"## Section: {section}\nResearch: {content}")
        
        return "\n\n".join(formatted_research)
    
    def _format_outline_for_prompt(self, outline: list) -> str:
        """Format outline for the prompt."""
        return "\n".join([f"{i+1}. {section}" for i, section in enumerate(outline)])
    
    def _format_revision_notes(self, review_notes: Optional[str]) -> str:
        """Format revision notes for the prompt."""
        if review_notes:
            return f"""
IMPORTANT - EDITOR'S REVISION NOTES:
{review_notes}
Please incorporate these revisions in your rewrite.
"""
        return ""
    
    def _analyze_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Analyze the generated content."""
        words = len(content.split())
        chars = len(content)
        lines = len(content.split('\n'))
        
        # Estimate reading time (average 200 words per minute)
        reading_time = max(1, words // 200)
        
        # Count markdown headers
        header_count = content.count('#')
        
        # Check for specific elements
        has_bullet_points = '•' in content or '-' in content
        has_numbered_lists = any(f"{i}." in content for i in range(1, 11))
        has_links = '[' in content and '](' in content
        has_emphasis = '**' in content or '*' in content
        
        analysis = {
            "word_count": words,
            "character_count": chars,
            "line_count": lines,
            "reading_time": reading_time,
            "header_count": header_count,
            "has_bullet_points": has_bullet_points,
            "has_numbered_lists": has_numbered_lists,
            "has_links": has_links,
            "has_emphasis": has_emphasis,
            "content_quality": self._assess_content_quality(words, content_type),
            "format_score": self._calculate_format_score(content)
        }
        
        return analysis
    
    def _assess_content_quality(self, word_count: int, content_type: str) -> str:
        """Assess content quality based on word count and type."""
        target_ranges = {
            "blog": (1500, 2500),
            "linkedin": (800, 1200),
            "article": (2000, 3000)
        }
        
        min_words, max_words = target_ranges.get(content_type, (1000, 2000))
        
        if min_words <= word_count <= max_words:
            return "optimal"
        elif word_count < min_words * 0.8:
            return "too_short"
        elif word_count > max_words * 1.2:
            return "too_long"
        else:
            return "acceptable"
    
    def _calculate_format_score(self, content: str) -> float:
        """Calculate formatting score based on content structure."""
        score = 0.0
        
        # Check for headers
        if '#' in content:
            score += 20
        
        # Check for lists
        if '-' in content or '•' in content:
            score += 15
        
        # Check for emphasis
        if '**' in content or '*' in content:
            score += 15
        
        # Check for proper paragraph breaks
        paragraph_breaks = content.count('\n\n')
        if paragraph_breaks > 0:
            score += min(20, paragraph_breaks * 2)
        
        # Check for links
        if '[' in content and '](' in content:
            score += 10
        
        # Check for code blocks or quotes
        if '```' in content or '>' in content:
            score += 10
        
        # Check content length structure
        lines = content.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        if 50 <= avg_line_length <= 100:  # Good readability
            score += 10
        
        return min(100.0, score)
    
    def revise_content(
        self,
        original_content: str,
        revision_notes: str,
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Revise existing content based on feedback.
        
        Args:
            original_content: Original content to revise
            revision_notes: Specific revision instructions
            context: Execution context
            
        Returns:
            AgentResult: Revised content
        """
        try:
            prompt = f"""You are an expert content editor. Revise the following content based on the specific feedback provided.

ORIGINAL CONTENT:
{original_content}

REVISION INSTRUCTIONS:
{revision_notes}

REVISION REQUIREMENTS:
- Address all points in the revision instructions
- Maintain the original tone and style unless specifically requested to change
- Ensure the revised content flows naturally
- Keep the core message and structure intact unless revision requires changes
- Improve clarity and readability where possible
- Maintain proper formatting and structure

Provide the complete revised content:"""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            revised_content = response.content.strip()
            
            # Analyze the revision
            revision_analysis = self._analyze_content(revised_content, "revision")
            
            result_data = {
                "revised_content": revised_content,
                "original_word_count": len(original_content.split()),
                "revised_word_count": revision_analysis["word_count"],
                "revision_analysis": revision_analysis
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "operation": "revision",
                    "word_count_change": revision_analysis["word_count"] - len(original_content.split())
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CONTENT_REVISION_FAILED"
            )