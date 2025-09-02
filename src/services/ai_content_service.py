"""
Real AI Content Service - Generates actual content using Google Gemini API
Provides a simple interface for generating marketing content without complex agent orchestration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)

class AIContentService:
    """Service for generating real AI content using Google Gemini."""
    
    def __init__(self):
        """Initialize the AI content service."""
        # Check for either GEMINI_API_KEY or GOOGLE_API_KEY
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY not found - AI content generation will be disabled")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Google Gemini client initialized for real content generation")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.model = None
    
    def is_available(self) -> bool:
        """Check if AI service is available."""
        return self.model is not None
    
    async def generate_blog_post(
        self, 
        topic: str, 
        target_audience: str = "financial services professionals",
        tone: str = "professional",
        company_context: str = "CrediLinq AI platform for embedded finance"
    ) -> Dict[str, Any]:
        """Generate a complete blog post using AI."""
        if not self.is_available():
            return {
                "title": f"Mock Blog Post: {topic}",
                "content": f"This is mock content for {topic}. Real AI generation is not available (missing GOOGLE_API_KEY).",
                "summary": "Mock content generated",
                "word_count": 50,
                "status": "mock_generation"
            }
        
        try:
            # Generate the blog post
            prompt = f"""
            Write a comprehensive blog post for {company_context} targeting {target_audience}.
            
            Topic: {topic}
            Tone: {tone}
            Target Audience: {target_audience}
            
            The blog post should be:
            - 800-1200 words long
            - Professional and engaging
            - Include actionable insights
            - Focus on embedded finance and fintech solutions
            - Include a compelling call-to-action
            - Be SEO optimized with relevant keywords
            
            Format as a complete blog post with title, introduction, main content sections, and conclusion.
            """
            
            response = await self._generate_content(prompt, max_tokens=2000)
            
            # Extract title and content
            content = response.strip()
            lines = content.split('\n')
            title = lines[0].strip().replace('# ', '').replace('Title: ', '')
            
            # Remove title from content
            main_content = '\n'.join(lines[1:]).strip()
            
            # Generate summary
            summary_prompt = f"Write a 2-3 sentence summary of this blog post:\n\n{content}"
            summary = await self._generate_content(summary_prompt, max_tokens=200)
            
            word_count = len(main_content.split())
            
            logger.info(f"✅ Generated blog post: '{title}' ({word_count} words)")
            
            return {
                "title": title,
                "content": main_content,
                "summary": summary.strip(),
                "word_count": word_count,
                "status": "ai_generated",
                "generated_at": datetime.now().isoformat(),
                "topic": topic,
                "target_audience": target_audience,
                "tone": tone
            }
            
        except Exception as e:
            logger.error(f"Error generating blog post: {e}")
            return {
                "title": f"Blog Post: {topic}",
                "content": f"Error generating content: {str(e)}",
                "summary": "Content generation failed",
                "word_count": 0,
                "status": "generation_failed",
                "error": str(e)
            }
    
    async def generate_social_media_post(
        self,
        platform: str,
        topic: str,
        company_context: str = "CrediLinq AI platform for embedded finance",
        include_hashtags: bool = True
    ) -> Dict[str, Any]:
        """Generate social media content for specific platforms."""
        if not self.is_available():
            return {
                "content": f"Mock {platform} post about {topic} #MockContent",
                "hashtags": ["#MockContent", "#AI", "#FinTech"],
                "status": "mock_generation"
            }
        
        try:
            platform_specs = {
                "linkedin": {"max_chars": 3000, "tone": "professional", "format": "business networking"},
                "twitter": {"max_chars": 280, "tone": "concise", "format": "tweet"},
                "facebook": {"max_chars": 2000, "tone": "engaging", "format": "social media post"}
            }
            
            spec = platform_specs.get(platform.lower(), platform_specs["linkedin"])
            
            prompt = f"""
            Create a {platform} post for {company_context} about {topic}.
            
            Requirements:
            - Maximum {spec['max_chars']} characters
            - {spec['tone']} tone
            - Format appropriate for {spec['format']}
            - Focus on embedded finance and fintech
            - Include call-to-action
            {"- Include relevant hashtags" if include_hashtags else ""}
            - Engaging and professional
            """
            
            content = await self._generate_content(prompt, max_tokens=300)
            
            # Extract hashtags if included
            hashtags = []
            if include_hashtags and '#' in content:
                import re
                hashtags = re.findall(r'#\w+', content)
            
            logger.info(f"✅ Generated {platform} post about {topic}")
            
            return {
                "content": content.strip(),
                "hashtags": hashtags,
                "platform": platform,
                "topic": topic,
                "char_count": len(content),
                "status": "ai_generated",
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating {platform} post: {e}")
            return {
                "content": f"Error generating {platform} content: {str(e)}",
                "hashtags": [],
                "status": "generation_failed",
                "error": str(e)
            }
    
    async def generate_email_campaign(
        self,
        purpose: str,
        target_audience: str = "potential partners",
        company_context: str = "CrediLinq AI platform for embedded finance"
    ) -> Dict[str, Any]:
        """Generate email campaign content."""
        if not self.is_available():
            return {
                "subject": f"Mock Email: {purpose}",
                "content": f"This is mock email content for {purpose} targeting {target_audience}.",
                "status": "mock_generation"
            }
        
        try:
            prompt = f"""
            Create an email campaign for {company_context} with the following details:
            
            Purpose: {purpose}
            Target Audience: {target_audience}
            
            The email should include:
            - Compelling subject line
            - Personalized greeting
            - Clear value proposition
            - Call-to-action
            - Professional closing
            - Focus on embedded finance benefits
            - Appropriate length for business email (300-500 words)
            
            Format:
            Subject: [subject line]
            
            [email body]
            """
            
            response = await self._generate_content(prompt, max_tokens=800)
            
            # Parse subject and content
            lines = response.strip().split('\n')
            subject = ""
            content_lines = []
            
            for line in lines:
                if line.startswith("Subject:"):
                    subject = line.replace("Subject:", "").strip()
                elif line.strip() and not line.startswith("Subject:"):
                    content_lines.append(line)
            
            content = '\n'.join(content_lines).strip()
            
            if not subject:
                subject = f"Partnership Opportunity - {purpose}"
            
            logger.info(f"✅ Generated email campaign: '{subject}'")
            
            return {
                "subject": subject,
                "content": content,
                "purpose": purpose,
                "target_audience": target_audience,
                "word_count": len(content.split()),
                "status": "ai_generated",
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating email campaign: {e}")
            return {
                "subject": f"Email Campaign: {purpose}",
                "content": f"Error generating email content: {str(e)}",
                "status": "generation_failed",
                "error": str(e)
            }
    
    async def generate_content_for_task(
        self,
        task_type: str,
        target_format: str,
        target_asset: str,
        campaign_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate content based on campaign task specifications."""
        campaign_context = campaign_context or {}
        
        # Determine content type and generate accordingly
        if "blog" in target_format.lower() or "post" in target_format.lower():
            return await self.generate_blog_post(
                topic=target_asset,
                target_audience=campaign_context.get("target_audience", "financial services professionals"),
                company_context=campaign_context.get("company_context", "CrediLinq AI platform")
            )
        
        elif "social" in target_format.lower() or "linkedin" in target_format.lower() or "twitter" in target_format.lower():
            platform = target_format.lower().replace("_post", "").replace("_", "")
            return await self.generate_social_media_post(
                platform=platform,
                topic=target_asset,
                company_context=campaign_context.get("company_context", "CrediLinq AI platform")
            )
        
        elif "email" in target_format.lower():
            return await self.generate_email_campaign(
                purpose=target_asset,
                target_audience=campaign_context.get("target_audience", "potential partners"),
                company_context=campaign_context.get("company_context", "CrediLinq AI platform")
            )
        
        else:
            # General content generation
            if not self.is_available():
                return {
                    "content": f"Mock content for {target_asset} in {target_format} format",
                    "status": "mock_generation"
                }
            
            try:
                prompt = f"""
                Create marketing content for CrediLinq AI platform:
                
                Asset: {target_asset}
                Format: {target_format}
                Type: {task_type}
                
                The content should be:
                - Professional and engaging
                - Focused on embedded finance solutions
                - Appropriate for the specified format
                - Include relevant call-to-action
                - Target financial services professionals
                """
                
                content = await self._generate_content(prompt, max_tokens=1000)
                
                return {
                    "content": content.strip(),
                    "format": target_format,
                    "asset": target_asset,
                    "type": task_type,
                    "status": "ai_generated",
                    "generated_at": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error generating general content: {e}")
                return {
                    "content": f"Error generating content for {target_asset}: {str(e)}",
                    "status": "generation_failed",
                    "error": str(e)
                }
    
    async def _generate_content(self, prompt: str, max_tokens: int = 1000) -> str:
        """Internal method to generate content using Gemini API."""
        if not self.model:
            raise Exception("Gemini model not initialized")
        
        try:
            # Add system context to the prompt for Gemini
            full_prompt = f"""You are a professional content writer specializing in fintech and embedded finance solutions. Create high-quality, engaging content that drives business value.

{prompt}"""
            
            response = self.model.generate_content(full_prompt)
            
            if response.text:
                return response.text
            else:
                raise Exception("Gemini returned empty response")
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"AI content generation failed: {str(e)}")

# Create global instance
ai_content_service = AIContentService()