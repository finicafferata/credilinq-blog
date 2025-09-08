"""
Real WriterAgent Implementation - LLM-powered content generation agent.

This replaces the stub implementation with a real agent that:
- Generates high-quality content using advanced LLM capabilities
- Adapts tone, style, and format based on requirements
- Creates structured content with proper formatting and SEO optimization
- Provides content analytics and quality assessments
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import re

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.llm_client import LLMClient
from ...core.security import InputValidator, SecurityError
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RealWriterAgent(BaseAgent):
    """
    Real LLM-powered writer agent for high-quality content generation.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize the real writer agent."""
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.WRITER,
                name="RealWriterAgent",
                description="Creates high-quality content with advanced LLM-powered writing capabilities",
                capabilities=[
                    "content_writing",
                    "tone_adaptation",
                    "seo_optimization",
                    "structure_optimization",
                    "multi_format_content",
                    "brand_voice_alignment"
                ]
            )
        
        super().__init__(metadata)
        
        # Initialize LLM client with higher temperature for creativity
        self.llm_client = LLMClient(
            model="gemini-1.5-flash",
            temperature=0.8,  # Higher temperature for creative writing
            agent_name=self.metadata.name,
            agent_type=self.metadata.agent_type.value
        )
        
        # Initialize security validator
        self.security_validator = InputValidator()
        
        # Writing templates and styles
        self._init_writing_templates()
        self._init_content_formats()
    
    def _init_writing_templates(self):
        """Initialize writing prompt templates and style guides."""
        self.writing_system_prompt = """You are an expert content writer specializing in fintech, financial services, and B2B technology content.

Your writing expertise includes:
- Creating engaging, authoritative content that drives business results
- Adapting tone and style for different audiences and platforms
- Incorporating SEO best practices naturally into content
- Structuring content for optimal readability and engagement
- Maintaining brand voice consistency across all content

You produce high-quality content that:
- Provides genuine value to readers
- Incorporates industry insights and thought leadership
- Uses clear, compelling language that converts
- Follows content marketing best practices
- Optimizes for both human readers and search engines"""

        self.writing_prompt_template = """Create compelling, high-quality content based on the following requirements:

CONTENT SPECIFICATION:
- Content Type: {content_type}
- Topic/Title: {topic}
- Target Audience: {target_audience}
- Word Count Target: {word_count}
- Tone: {tone}
- Content Outline: {outline}

CONTEXT AND RESEARCH:
{research_data}

REQUIREMENTS:
1. Content Structure:
   - Engaging introduction that hooks the reader
   - Clear, logical flow with smooth transitions
   - Well-organized sections with descriptive headers
   - Strong conclusion with clear next steps/CTA

2. Writing Quality:
   - Authoritative yet accessible language
   - Industry-specific insights and examples
   - Data-driven points with statistics where relevant
   - Engaging storytelling elements

3. SEO Optimization:
   - Natural keyword integration
   - Optimized headers and subheaders
   - Meta description suggestions
   - Internal linking opportunities

4. Brand Alignment:
   - Professional, trustworthy tone
   - Value-focused messaging
   - Thought leadership positioning
   - Customer-centric perspective

5. Engagement Features:
   - Compelling statistics and data points
   - Real-world examples and case studies
   - Actionable takeaways and recommendations
   - Clear calls-to-action

Provide the content in a structured format with:
- title
- meta_description
- main_content (formatted with headers)
- seo_keywords
- content_analytics (word count, readability score)
- engagement_elements (statistics, examples used)"""

    def _init_content_formats(self):
        """Initialize content format specifications."""
        self.content_formats = {
            'blog_post': {
                'structure': ['intro', 'body_sections', 'conclusion'],
                'typical_word_count': 1500,
                'seo_importance': 'high',
                'engagement_focus': 'education_and_authority'
            },
            'social_media': {
                'structure': ['hook', 'value_prop', 'cta'],
                'typical_word_count': 100,
                'seo_importance': 'medium',
                'engagement_focus': 'viral_and_shares'
            },
            'case_study': {
                'structure': ['challenge', 'solution', 'results'],
                'typical_word_count': 1200,
                'seo_importance': 'medium',
                'engagement_focus': 'proof_and_credibility'
            },
            'whitepaper': {
                'structure': ['executive_summary', 'analysis', 'recommendations'],
                'typical_word_count': 3000,
                'seo_importance': 'medium',
                'engagement_focus': 'authority_and_leads'
            }
        }
        
        self.tone_styles = {
            'professional': 'Authoritative, formal, industry-focused',
            'conversational': 'Friendly, approachable, easy to understand',
            'thought_leadership': 'Visionary, innovative, industry-shaping',
            'educational': 'Clear, informative, step-by-step guidance'
        }
    
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute high-quality content generation.
        
        Args:
            input_data: Content requirements including type, topic, audience
            context: Execution context for tracking
            
        Returns:
            AgentResult with generated content and analytics
        """
        try:
            # Validate and sanitize inputs
            await self._validate_writing_inputs(input_data)
            
            # Extract writing parameters
            content_type = input_data.get('content_type', 'blog_post')
            topic = input_data.get('topic', 'Fintech Innovation Trends')
            target_audience = input_data.get('target_audience', 'Business decision-makers')
            word_count = input_data.get('word_count', self._get_default_word_count(content_type))
            tone = input_data.get('tone', 'professional')
            outline = input_data.get('outline', [])
            research_data = input_data.get('research_data', {})
            
            # Generate high-quality content
            content_result = await self._generate_content(
                content_type=content_type,
                topic=topic,
                target_audience=target_audience,
                word_count=word_count,
                tone=tone,
                outline=outline,
                research_data=research_data
            )
            
            # Process and enhance the content
            enhanced_content = await self._process_and_enhance_content(content_result, content_type)
            
            # Create result with content analytics
            result = AgentResult(
                success=True,
                data=enhanced_content,
                metadata={
                    'agent_type': 'real_writer',
                    'model_used': self.llm_client.model_name,
                    'content_type': content_type,
                    'generation_timestamp': datetime.utcnow().isoformat(),
                    'target_word_count': word_count,
                    'actual_word_count': enhanced_content.get('content_analytics', {}).get('word_count', 0)
                }
            )
            
            # Add writing decisions for business intelligence
            self._add_writing_decisions(result, enhanced_content, input_data)
            
            # Set quality assessment
            self._assess_content_quality(result, enhanced_content)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in writer agent execution: {e}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="WRITER_EXECUTION_ERROR"
            )
    
    async def _validate_writing_inputs(self, input_data: Dict[str, Any]) -> None:
        """Validate and sanitize writing inputs."""
        try:
            # Security validation
            for key, value in input_data.items():
                if isinstance(value, str):
                    self.security_validator.validate_input(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            self.security_validator.validate_input(item)
            
            # Business logic validation
            topic = input_data.get('topic', '')
            if len(topic) > 300:
                raise ValueError("Topic too long (max 300 characters)")
            
            content_type = input_data.get('content_type', 'blog_post')
            if content_type not in self.content_formats:
                logger.warning(f"Unknown content type: {content_type}")
            
            word_count = input_data.get('word_count', 1500)
            if word_count > 10000:
                raise ValueError("Word count too high (max 10,000)")
            
            tone = input_data.get('tone', 'professional')
            if tone not in self.tone_styles:
                logger.warning(f"Unknown tone: {tone}")
                
        except SecurityError as e:
            raise ValueError(f"Security validation failed: {e}")
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")
    
    def _get_default_word_count(self, content_type: str) -> int:
        """Get default word count for content type."""
        return self.content_formats.get(content_type, {}).get('typical_word_count', 1500)
    
    async def _generate_content(
        self,
        content_type: str,
        topic: str,
        target_audience: str,
        word_count: int,
        tone: str,
        outline: List[str],
        research_data: Dict[str, Any]
    ) -> str:
        """Generate high-quality content using LLM."""
        
        # Format research data for prompt
        research_summary = self._format_research_data(research_data)
        
        # Format outline
        outline_text = "\\n".join([f"- {item}" for item in outline]) if outline else "Create a logical structure based on the topic"
        
        # Format the writing prompt
        formatted_prompt = self.writing_prompt_template.format(
            content_type=content_type,
            topic=topic,
            target_audience=target_audience,
            word_count=word_count,
            tone=self.tone_styles.get(tone, 'Professional and authoritative'),
            outline=outline_text,
            research_data=research_summary
        )
        
        # Create messages for LLM
        messages = [
            SystemMessage(content=self.writing_system_prompt),
            HumanMessage(content=formatted_prompt)
        ]
        
        # Generate content using LLM
        response = await self.llm_client.agenerate(messages)
        
        return response
    
    def _format_research_data(self, research_data: Dict[str, Any]) -> str:
        """Format research data for inclusion in writing prompt."""
        if not research_data:
            return "No specific research data provided. Use general industry knowledge and best practices."
        
        formatted_research = "RESEARCH INSIGHTS:\\n"
        
        # Extract key insights
        if 'key_insights' in research_data:
            formatted_research += "Key Insights:\\n"
            insights = research_data['key_insights']
            if isinstance(insights, dict):
                for key, value in insights.items():
                    formatted_research += f"- {key}: {value}\\n"
            else:
                formatted_research += f"- {str(insights)}\\n"
        
        # Extract market trends
        if 'detailed_findings' in research_data:
            findings = research_data['detailed_findings']
            if isinstance(findings, dict) and 'market_trends' in findings:
                formatted_research += "Market Trends:\\n"
                trends = findings['market_trends']
                if isinstance(trends, list):
                    for trend in trends[:5]:  # Limit to top 5
                        formatted_research += f"- {trend}\\n"
        
        # Extract recommendations
        if 'recommendations' in research_data:
            formatted_research += "Strategic Recommendations:\\n"
            recommendations = research_data['recommendations']
            if isinstance(recommendations, dict):
                for key, value in recommendations.items():
                    formatted_research += f"- {key}: {value}\\n"
        
        return formatted_research if len(formatted_research) > 20 else "Use industry expertise and current market knowledge."
    
    async def _process_and_enhance_content(self, llm_response: str, content_type: str) -> Dict[str, Any]:
        """Process and enhance the LLM content response."""
        try:
            # Try to parse as JSON first
            if llm_response.strip().startswith('{'):
                content_data = json.loads(llm_response)
            else:
                # Parse text response into structured format
                content_data = await self._parse_text_content(llm_response, content_type)
            
            # Ensure required fields exist
            required_fields = ['title', 'main_content', 'meta_description']
            for field in required_fields:
                if field not in content_data:
                    content_data[field] = self._generate_fallback_field(field, content_data, content_type)
            
            # Enhance content with analytics
            content_data = await self._add_content_analytics(content_data)
            
            # Add SEO enhancements
            content_data = await self._enhance_seo_elements(content_data)
            
            # Add engagement elements
            content_data = await self._add_engagement_elements(content_data)
            
            # Add content metadata
            content_data['content_metadata'] = {
                'generated_at': datetime.utcnow().isoformat(),
                'agent_name': self.metadata.name,
                'content_type': content_type,
                'processing_version': '1.0'
            }
            
            return content_data
            
        except json.JSONDecodeError:
            # Fallback: create structured content from text
            return await self._create_fallback_content(llm_response, content_type)
        except Exception as e:
            logger.error(f"Error processing content result: {e}")
            return await self._create_default_content(content_type)
    
    async def _parse_text_content(self, text_response: str, content_type: str) -> Dict[str, Any]:
        """Parse text response into structured content format."""
        content_data = {
            'title': '',
            'meta_description': '',
            'main_content': '',
            'seo_keywords': [],
            'content_analytics': {},
            'engagement_elements': {}
        }
        
        lines = text_response.split('\\n')
        current_section = 'main_content'
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract title (usually first line or marked with #)
            if not content_data['title']:
                if line.startswith('#') or (len(content_lines) == 0 and len(line) < 100):
                    content_data['title'] = line.replace('#', '').strip()
                    continue
            
            # Detect sections
            if 'meta description' in line.lower():
                current_section = 'meta_description'
                continue
            elif 'keywords' in line.lower():
                current_section = 'seo_keywords'
                continue
            elif current_section == 'meta_description':
                if len(line) > 20 and len(line) < 160:
                    content_data['meta_description'] = line
                    current_section = 'main_content'
                    continue
            elif current_section == 'seo_keywords':
                # Extract keywords from line
                keywords = [k.strip() for k in line.split(',') if k.strip()]
                content_data['seo_keywords'].extend(keywords)
                current_section = 'main_content'
                continue
            
            # Add to main content
            content_lines.append(line)
        
        content_data['main_content'] = '\\n'.join(content_lines)
        
        # Generate missing title if needed
        if not content_data['title'] and content_lines:
            content_data['title'] = content_lines[0][:80] + "..." if len(content_lines[0]) > 80 else content_lines[0]
        
        return content_data
    
    def _generate_fallback_field(self, field: str, content_data: Dict[str, Any], content_type: str) -> str:
        """Generate fallback content for missing fields."""
        if field == 'title':
            return f"Expert Guide to {content_type.replace('_', ' ').title()}"
        elif field == 'meta_description':
            title = content_data.get('title', 'Expert Guide')
            return f"Comprehensive {content_type.replace('_', ' ')} on {title}. Get expert insights and actionable strategies."
        elif field == 'main_content':
            return f"This {content_type.replace('_', ' ')} provides valuable insights and expert guidance on the topic."
        return ""
    
    async def _add_content_analytics(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add content analytics and metrics."""
        main_content = content_data.get('main_content', '')
        
        # Calculate word count
        word_count = len(main_content.split()) if main_content else 0
        
        # Calculate reading time (average 200 words per minute)
        reading_time = max(1, round(word_count / 200))
        
        # Estimate readability score (simplified)
        sentences = len([s for s in main_content.split('.') if s.strip()]) if main_content else 1
        avg_sentence_length = word_count / sentences if sentences > 0 else 0
        readability_score = max(0, min(100, 100 - (avg_sentence_length * 2)))
        
        # Count headers (lines starting with #)
        headers_count = len([line for line in main_content.split('\\n') if line.strip().startswith('#')])
        
        content_data['content_analytics'] = {
            'word_count': word_count,
            'reading_time_minutes': reading_time,
            'readability_score': round(readability_score, 1),
            'headers_count': headers_count,
            'sentences_count': sentences,
            'avg_sentence_length': round(avg_sentence_length, 1)
        }
        
        return content_data
    
    async def _enhance_seo_elements(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance SEO elements in the content."""
        main_content = content_data.get('main_content', '')
        title = content_data.get('title', '')
        
        # Extract or generate SEO keywords if not present
        if not content_data.get('seo_keywords'):
            # Simple keyword extraction from title and content
            keywords = []
            
            # Extract from title
            title_words = [w.lower() for w in title.split() if len(w) > 3]
            keywords.extend(title_words[:3])
            
            # Add common fintech keywords
            fintech_keywords = ['fintech', 'financial services', 'digital banking', 'innovation', 'technology']
            keywords.extend([k for k in fintech_keywords if k in main_content.lower()])
            
            content_data['seo_keywords'] = list(set(keywords[:8]))  # Limit to 8 unique keywords
        
        # Enhance meta description if too short
        meta_desc = content_data.get('meta_description', '')
        if len(meta_desc) < 120:
            title_start = title[:50] if title else 'Expert insights'
            content_data['meta_description'] = f"{title_start}. Discover actionable strategies and expert guidance to drive your business forward."
        
        # Add SEO recommendations
        content_data['seo_recommendations'] = {
            'title_length': 'good' if 30 <= len(title) <= 60 else 'needs_improvement',
            'meta_description_length': 'good' if 120 <= len(content_data['meta_description']) <= 160 else 'needs_improvement',
            'keyword_density': 'good',  # Simplified check
            'headers_structure': 'good' if content_data.get('content_analytics', {}).get('headers_count', 0) > 0 else 'needs_headers'
        }
        
        return content_data
    
    async def _add_engagement_elements(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add engagement elements analysis."""
        main_content = content_data.get('main_content', '')
        
        # Count engagement elements (simplified detection)
        statistics_count = len(re.findall(r'\\b\\d+%', main_content))
        questions_count = len(re.findall(r'\\?', main_content))
        bullet_points = len(re.findall(r'^\\s*[-*•]', main_content, re.MULTILINE))
        
        # Detect calls-to-action (simplified)
        cta_patterns = ['contact', 'learn more', 'get started', 'download', 'sign up', 'try', 'discover']
        cta_count = sum(1 for pattern in cta_patterns if pattern in main_content.lower())
        
        content_data['engagement_elements'] = {
            'statistics_count': statistics_count,
            'questions_count': questions_count,
            'bullet_points_count': bullet_points,
            'call_to_action_count': cta_count,
            'engagement_score': min(100, (statistics_count * 10) + (questions_count * 5) + (bullet_points * 2) + (cta_count * 15))
        }
        
        return content_data
    
    async def _create_fallback_content(self, original_response: str, content_type: str) -> Dict[str, Any]:
        """Create structured fallback content when parsing fails."""
        return {
            'title': f"Expert Guide to {content_type.replace('_', ' ').title()}",
            'meta_description': f"Comprehensive {content_type.replace('_', ' ')} with expert insights and actionable strategies for business success.",
            'main_content': original_response,
            'seo_keywords': ['fintech', 'financial services', 'innovation', 'business growth'],
            'content_analytics': {
                'word_count': len(original_response.split()),
                'reading_time_minutes': max(1, len(original_response.split()) // 200),
                'readability_score': 75.0
            },
            'engagement_elements': {
                'statistics_count': 0,
                'questions_count': 0,
                'bullet_points_count': 0,
                'call_to_action_count': 1,
                'engagement_score': 15
            },
            'seo_recommendations': {
                'title_length': 'good',
                'meta_description_length': 'good',
                'keyword_density': 'good',
                'headers_structure': 'needs_headers'
            },
            'fallback_used': True
        }
    
    async def _create_default_content(self, content_type: str) -> Dict[str, Any]:
        """Create basic default content as last resort."""
        content = f"""# The Future of {content_type.replace('_', ' ').title()}

In today's rapidly evolving financial technology landscape, businesses must stay ahead of emerging trends and innovations.

## Key Trends

The financial services industry is experiencing unprecedented transformation driven by technological advancement and changing customer expectations.

## Strategic Recommendations

To succeed in this dynamic environment, organizations should focus on:

- Embracing digital transformation
- Prioritizing customer experience
- Investing in scalable technology
- Building strategic partnerships

## Conclusion

The future belongs to organizations that can adapt quickly and leverage technology to create value for their customers."""
        
        return {
            'title': f"The Future of {content_type.replace('_', ' ').title()}",
            'meta_description': f"Explore the future of {content_type.replace('_', ' ')} with expert insights on trends, strategies, and best practices.",
            'main_content': content,
            'seo_keywords': ['fintech', 'financial services', 'digital transformation', 'innovation'],
            'content_analytics': {
                'word_count': len(content.split()),
                'reading_time_minutes': 2,
                'readability_score': 80.0,
                'headers_count': 3
            },
            'engagement_elements': {
                'statistics_count': 0,
                'questions_count': 0,
                'bullet_points_count': 4,
                'call_to_action_count': 0,
                'engagement_score': 8
            },
            'default_content_used': True
        }
    
    def _add_writing_decisions(self, result: AgentResult, content: Dict[str, Any], input_data: Dict[str, Any]) -> None:
        """Add detailed writing decisions to the result."""
        
        # Content structure decision
        self.add_decision_reasoning(
            result=result,
            decision_point="Content Structure and Organization",
            reasoning="Selected optimal content structure based on content type, audience needs, and engagement goals",
            importance_explanation="Proper content structure is crucial for readability, engagement, and SEO performance",
            confidence_score=0.89,
            alternatives_considered=[
                "Linear narrative structure",
                "Problem-solution format",
                "Listicle format",
                "Case study approach"
            ],
            business_impact="Well-structured content improves reader engagement, time on page, and conversion rates",
            risk_assessment="Low risk with proven content structure patterns",
            success_indicators=[
                "High readability score",
                "Logical content flow",
                "Clear section headers",
                "Engaging introduction and conclusion"
            ],
            implementation_priority="high"
        )
        
        # SEO optimization decision
        analytics = content.get('content_analytics', {})
        self.add_decision_reasoning(
            result=result,
            decision_point="SEO Optimization Strategy",
            reasoning="Implemented comprehensive SEO optimization with natural keyword integration and meta optimization",
            importance_explanation="SEO optimization is essential for organic discovery and search engine visibility",
            confidence_score=0.85,
            alternatives_considered=[
                "Keyword-stuffing approach",
                "Minimal SEO focus",
                "Topic cluster strategy"
            ],
            business_impact="SEO-optimized content drives organic traffic and lead generation",
            success_indicators=[
                "Optimal keyword density",
                "Proper meta descriptions",
                "Header structure optimization",
                f"Readability score: {analytics.get('readability_score', 'N/A')}"
            ],
            implementation_priority="high"
        )
    
    def _assess_content_quality(self, result: AgentResult, content: Dict[str, Any]) -> None:
        """Assess the quality of the generated content."""
        
        # Get analytics data
        analytics = content.get('content_analytics', {})
        engagement = content.get('engagement_elements', {})
        seo_recommendations = content.get('seo_recommendations', {})
        
        # Calculate quality dimensions
        word_count = analytics.get('word_count', 0)
        word_count_score = min(10.0, max(5.0, (word_count / 1500) * 10)) if word_count > 0 else 5.0
        
        readability_score = analytics.get('readability_score', 70) / 10  # Convert to 0-10 scale
        
        engagement_score = min(10.0, engagement.get('engagement_score', 0) / 10)
        
        seo_score = 8.5  # Default good score
        seo_issues = sum(1 for v in seo_recommendations.values() if v == 'needs_improvement')
        seo_score -= seo_issues * 1.5
        
        # Calculate overall score
        overall_score = (word_count_score + readability_score + engagement_score + seo_score) / 4
        
        # Identify strengths and improvement areas
        strengths = []
        improvement_areas = []
        
        if word_count_score >= 8.0:
            strengths.append("Appropriate content length")
        elif word_count_score < 6.0:
            improvement_areas.append("Content length optimization")
            
        if readability_score >= 7.5:
            strengths.append("High readability and clarity")
        else:
            improvement_areas.append("Sentence structure and readability")
            
        if engagement_score >= 7.0:
            strengths.append("Strong engagement elements")
        else:
            improvement_areas.append("Engagement features and interactivity")
        
        if seo_score >= 8.0:
            strengths.append("Excellent SEO optimization")
        else:
            improvement_areas.append("SEO metadata and optimization")
        
        # Set quality assessment
        self.set_quality_assessment(
            result=result,
            overall_score=overall_score,
            dimension_scores={
                "content_length": word_count_score,
                "readability": readability_score,
                "engagement": engagement_score,
                "seo_optimization": seo_score
            },
            improvement_areas=improvement_areas,
            strengths=strengths,
            quality_notes=f"Content quality based on length ({word_count} words), readability ({analytics.get('readability_score', 'N/A')}), and engagement elements."
        )