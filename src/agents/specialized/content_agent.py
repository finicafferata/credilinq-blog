"""
Content Agent - Generates high-quality content based on outline, research, and company context.
"""

from typing import List, Dict, Any, Optional
import ast
import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator


class ContentGenerationAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for generating high-quality content based on structured outlines,
    research data, and company context. Produces professional content optimized for
    different formats (blog posts, LinkedIn posts, articles).
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_GENERATOR,
                name="ContentGenerationAgent",
                description="Generates high-quality content based on outline, research, and company context",
                capabilities=[
                    "content_generation",
                    "format_optimization",
                    "tone_adaptation",
                    "research_integration",
                    "markdown_formatting",
                    "multi_format_support"
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
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=settings.OPENAI_API_KEY
            )
            self.logger.info("ContentGenerationAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ContentGenerationAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for content generation."""
        super()._validate_input(input_data)
        
        required_fields = ["title", "outline", "company_context"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation
        for field in required_fields:
            if isinstance(input_data[field], str):
                self.security_validator.validate_input(input_data[field])
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate high-quality content based on provided outline and research.
        
        Args:
            input_data: Dictionary containing:
                - title: Content title/topic
                - outline: List of sections to cover
                - company_context: Company/brand context
                - content_type: Type of content (blog, linkedin, article)
                - research: Optional research data
                - geo_metadata: Optional GEO optimization data
            context: Execution context
            
        Returns:
            AgentResult: Result containing the generated content
        """
        try:
            # Initialize if not already done
            if self.llm is None:
                self._initialize()
            
            title = input_data["title"]
            outline = input_data["outline"]
            company_context = input_data["company_context"]
            content_type = input_data.get("content_type", "blog").lower()
            research = input_data.get("research", {})
            geo_metadata = input_data.get("geo_metadata", {})
            
            # Store content type for word count estimation
            self._current_content_type = content_type
            
            self.logger.info(f"Generating {content_type} content for: {title}")
            
            # Generate content based on type
            content = self._generate_content(title, outline, company_context, content_type, research, geo_metadata)
            
            result_data = {
                "content": content,
                "content_type": content_type,
                "word_count": len(content.split()),
                "sections_covered": len(outline) if isinstance(outline, list) else 0,
                "content_analysis": self._analyze_content(content, content_type),
                "geo_optimized": bool(geo_metadata)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "content_generator",
                    "content_format": content_type,
                    "word_count": result_data["word_count"],
                    "geo_optimized": result_data["geo_optimized"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate content: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CONTENT_GENERATION_FAILED"
            )

    def _generate_content(
        self,
        title: str,
        outline: List[str],
        company_context: str,
        content_type: str,
        research: Dict[str, str],
        geo_metadata: Dict[str, Any] = None
    ) -> str:
        """Generate content based on content type and requirements."""
        
        if geo_metadata is None:
            geo_metadata = {}
        
        # Select appropriate prompt based on content type
        if content_type == "linkedin":
            prompt = self._create_linkedin_content_prompt(title, outline, company_context, research, geo_metadata)
        elif content_type == "article":
            prompt = self._create_article_content_prompt(title, outline, company_context, research, geo_metadata)
        else:  # default to blog
            prompt = self._create_blog_content_prompt(title, outline, company_context, research, geo_metadata)
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            content = self._parse_content_response(response.content.strip())
            
            self.logger.info(f"Generated content with {len(content.split())} words")
            return content
            
        except Exception as e:
            self.logger.warning(f"LLM content generation failed: {str(e)}, using fallback")
            return self._create_fallback_content(title, outline, content_type)
    
    def _create_blog_content_prompt(
        self, 
        title: str, 
        outline: List[str], 
        company_context: str, 
        research: Dict[str, str],
        geo_metadata: Dict[str, Any] = None
    ) -> str:
        """Create prompt for blog post content."""
        research_text = self._format_research_data(research)
        outline_text = "\n".join([f"- {section}" for section in outline])
        geo_guidance = self._format_geo_guidance(geo_metadata) if geo_metadata else ""
        
        return f"""
        Act as an expert Content Writer and SEO specialist with 15+ years of experience creating high-performing blog content optimized for AI Answer Engines. Your task is to write a comprehensive, engaging blog post.

        **Blog Title:** "{title}"
        **Company Context & Voice:** "{company_context}"

        **Outline to Follow:**
        {outline_text}

        **Research Data:**
        {research_text}

        {geo_guidance}

        **Instructions:**
        - Write a comprehensive blog post that follows the outline structure exactly.
        - Use a professional yet conversational tone that reflects the company's voice.
        - Integrate the research data naturally throughout the content to support your points.
        - Create engaging introductions for each section that connect to the overall narrative.
        - Include actionable insights, practical examples, and concrete takeaways.
        - Use proper Markdown formatting with ## for main sections and ### for subsections.
        - Aim for 1500-2500 words for comprehensive coverage.
        - Include bullet points and numbered lists where appropriate for readability.
        - End with a strong conclusion that includes a clear call-to-action.
        - If GEO guidance is provided, incorporate it naturally for AI discoverability.

        **Negative Constraints:**
        - **Do not** deviate from the provided outline structure.
        - **Do not** include placeholder text or incomplete sections.
        - **Avoid** generic conclusions without actionable next steps.
        - **Do not** ignore the research data - integrate it meaningfully.

        **Output Format:**
        Return the complete blog post content inside <content> tags:
        <content>
        [Your complete blog post in Markdown format]
        </content>
        """
    
    def _create_linkedin_content_prompt(
        self, 
        title: str, 
        outline: List[str], 
        company_context: str, 
        research: Dict[str, str],
        geo_metadata: Dict[str, Any] = None
    ) -> str:
        """Create prompt for LinkedIn post content."""
        research_text = self._format_research_data(research)
        outline_text = "\n".join([f"- {section}" for section in outline])
        
        return f"""
        Act as a LinkedIn content strategist and B2B social media expert with 10+ years of experience creating viral professional content. Create an engaging LinkedIn post optimized for maximum engagement.

        **Topic:** "{title}"
        **Company Context & Brand Voice:** "{company_context}"

        **Content Structure:**
        {outline_text}

        **Supporting Research:**
        {research_text}

        **Instructions:**
        - Create a compelling LinkedIn post that stops the scroll and drives engagement.
        - Follow the outline structure while maintaining LinkedIn's conversational flow.
        - Start with an attention-grabbing hook in the first 2 lines.
        - Use short paragraphs (1-2 sentences) with line breaks for mobile readability.
        - Include relevant emojis sparingly and professionally (2-3 maximum).
        - Integrate research insights to add credibility and value.
        - Aim for 800-1300 characters for optimal LinkedIn engagement.
        - End with a thought-provoking question or clear call-to-action.
        - Include 3-5 relevant hashtags at the end.

        **Negative Constraints:**
        - **Avoid** overly formal corporate language - keep it conversational.
        - **Do not** exceed 1300 characters or it will be cut off.
        - **Avoid** excessive emojis or hashtags - keep it professional.

        **Output Format:**
        Return the complete LinkedIn post inside <content> tags:
        <content>
        [Your complete LinkedIn post]
        </content>
        """
    
    def _create_article_content_prompt(
        self, 
        title: str, 
        outline: List[str], 
        company_context: str, 
        research: Dict[str, str],
        geo_metadata: Dict[str, Any] = None
    ) -> str:
        """Create prompt for article content."""
        research_text = self._format_research_data(research)
        outline_text = "\n".join([f"- {section}" for section in outline])
        
        return f"""
        Act as a subject-matter expert and authoritative technical writer with deep industry knowledge. Create a comprehensive, analytical article that establishes thought leadership.

        **Article Topic:** "{title}"
        **Company's Perspective:** "{company_context}"

        **Article Structure:**
        {outline_text}

        **Research Foundation:**
        {research_text}

        **Instructions:**
        - Write an authoritative, well-researched article that demonstrates deep expertise.
        - Follow the outline structure with detailed analysis in each section.
        - Use a professional, analytical tone with industry-specific terminology.
        - Include data-driven insights, case studies, and concrete examples.
        - Provide in-depth explanations of complex concepts.
        - Use proper Markdown formatting with clear section headers.
        - Aim for 2000-3500 words for comprehensive industry coverage.
        - Include relevant statistics, trends, and forward-looking insights.
        - Conclude with actionable recommendations and future implications.

        **Negative Constraints:**
        - **Avoid** superficial treatment of topics - provide deep analysis.
        - **Do not** make unsupported claims - back assertions with research.
        - **Avoid** casual language - maintain professional authority.

        **Output Format:**
        Return the complete article inside <content> tags:
        <content>
        [Your complete article in Markdown format]
        </content>
        """
    
    def _format_research_data(self, research: Dict[str, str]) -> str:
        """Format research data for inclusion in prompts."""
        if not research:
            return "No specific research data provided - use general industry knowledge."
        
        formatted_research = []
        for section, data in research.items():
            formatted_research.append(f"**{section}:**\n{data}")
        
        return "\n\n".join(formatted_research)
    
    def _format_geo_guidance(self, geo_metadata: Dict[str, Any]) -> str:
        """Format GEO metadata for inclusion in prompts."""
        if not geo_metadata:
            return ""
        
        geo_sections = ["**GEO OPTIMIZATION GUIDANCE:**"]
        
        if geo_metadata.get('optimized_title'):
            geo_sections.append(f"- **Optimized Title:** {geo_metadata['optimized_title']}")
        
        if geo_metadata.get('key_takeaways'):
            takeaways = geo_metadata['key_takeaways']
            if isinstance(takeaways, list):
                geo_sections.append("- **Key Takeaways to Include:**")
                for takeaway in takeaways:
                    geo_sections.append(f"  • {takeaway}")
        
        if geo_metadata.get('faq_for_structured_data'):
            faqs = geo_metadata['faq_for_structured_data']
            if isinstance(faqs, list):
                geo_sections.append("- **FAQ Questions to Address:**")
                for faq in faqs:
                    if isinstance(faq, dict) and faq.get('question'):
                        geo_sections.append(f"  • {faq['question']}")
        
        if geo_metadata.get('entity_and_concept_map'):
            entities = geo_metadata['entity_and_concept_map']
            if isinstance(entities, dict):
                geo_sections.append("- **Important Concepts to Define:**")
                for concept, definition in entities.items():
                    geo_sections.append(f"  • {concept}: {definition}")
        
        if geo_metadata.get('citable_expert_quote', {}).get('quote'):
            quote = geo_metadata['citable_expert_quote']['quote']
            geo_sections.append(f"- **Expert Quote to Incorporate:** \"{quote}\"")
        
        return "\n".join(geo_sections)
    
    def _parse_content_response(self, response: str) -> str:
        """Parse the LLM response to extract the content."""
        try:
            # Try to find content within <content> tags
            match = re.search(r"<content>(.*?)</content>", response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                return content
        except (AttributeError, ValueError):
            self.logger.warning("Failed to parse content from tags, using full response")
        
        # Fallback: use the entire response
        return response.strip()
    
    def _create_fallback_content(self, title: str, outline: List[str], content_type: str) -> str:
        """Create fallback content when LLM fails."""
        self.logger.info(f"Using fallback content for type: {content_type}")
        
        if content_type == "linkedin":
            return f"""🚀 {title}

Here's what professionals need to know:

{chr(10).join([f"✓ {section}" for section in outline[:3]])}

What's your experience with this? Share in the comments!

#ProfessionalGrowth #Industry #Leadership"""
        
        elif content_type == "article":
            content_sections = []
            content_sections.append(f"# {title}\n")
            content_sections.append("## Abstract\n")
            content_sections.append(f"This article examines {title.lower()} and its implications for the industry.\n")
            
            for section in outline:
                content_sections.append(f"## {section}\n")
                content_sections.append(f"Detailed analysis of {section.lower()} will be provided here.\n")
            
            return "\n".join(content_sections)
        
        else:  # blog
            content_sections = []
            content_sections.append(f"# {title}\n")
            content_sections.append("## Introduction\n")
            content_sections.append(f"In this comprehensive guide, we'll explore {title.lower()} and provide actionable insights.\n")
            
            for section in outline[1:-1]:  # Skip intro/conclusion
                content_sections.append(f"## {section}\n")
                content_sections.append(f"This section covers {section.lower()} in detail.\n")
            
            content_sections.append("## Conclusion\n")
            content_sections.append("Key takeaways and next steps for implementation.\n")
            
            return "\n".join(content_sections)
    
    def _analyze_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Analyze the generated content for quality metrics."""
        words = content.split()
        sentences = content.count('.') + content.count('!') + content.count('?')
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        # Count markdown headers
        headers = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))
        
        # Estimate reading time (200 words per minute)
        reading_time = max(1, len(words) // 200)
        
        # Content type specific analysis
        type_analysis = {}
        if content_type == "linkedin":
            type_analysis = {
                "character_count": len(content),
                "hashtag_count": content.count('#'),
                "emoji_count": len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', content))
            }
        elif content_type in ["blog", "article"]:
            type_analysis = {
                "markdown_headers": headers,
                "bullet_points": content.count('- ') + content.count('* '),
                "numbered_lists": len(re.findall(r'^\d+\.', content, re.MULTILINE))
            }
        
        return {
            "word_count": len(words),
            "sentence_count": sentences,
            "paragraph_count": paragraphs,
            "estimated_reading_time_minutes": reading_time,
            "content_type_specific": type_analysis
        }
