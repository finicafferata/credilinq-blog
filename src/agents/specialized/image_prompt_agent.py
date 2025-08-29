"""
Image Prompt Agent - Generates creative prompts for image generation services
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata, AgentExecutionContext
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

class ImagePromptAgent(BaseAgent):
    """
    Agent specialized in generating creative and detailed prompts for image generation services.
    Outputs prompts for DALL-E, Midjourney, Stable Diffusion, and other image generation APIs.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None, name: str = "ImagePromptAgent", description: str = "Generates creative prompts for image generation services"):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.IMAGE_PROMPT,
                name=name,
                description=description
            )
        super().__init__(metadata=metadata)
        self.agent_type = AgentType.IMAGE_PROMPT
        
    def execute(self, input_data: Dict[str, Any], context: Optional[AgentExecutionContext] = None, **kwargs) -> AgentResult:
        """
        Generate detailed image prompts based on content.
        
        Args:
            input_data: Dictionary containing:
                - content: Main content/blog post
                - blog_title: Title of the content
                - style: Visual style preference (professional, creative, modern, elegant)
                - image_type: Type of image needed (header, section, infographic, social)
                - count: Number of prompts to generate (default: 3)
                - target_platform: Platform for the image (web, social, print)
                - brand_guidelines: Brand colors, fonts, style preferences (optional)
            context: Execution context
        
        Returns:
            AgentResult with generated image prompts
        """
        try:
            logger.info(f"ImagePromptAgent executing for: {input_data.get('blog_title', 'Unknown')}")
            
            content = input_data.get('content', '')
            blog_title = input_data.get('blog_title', '')
            style = input_data.get('style', 'professional')
            image_type = input_data.get('image_type', 'header')
            count = input_data.get('count', 3)
            target_platform = input_data.get('target_platform', 'web')
            brand_guidelines = input_data.get('brand_guidelines', {})
            
            if not content and not blog_title:
                raise AgentExecutionError("No content or title provided for image prompt generation")
            
            # Generate prompts based on type and style
            prompts = self._generate_image_prompts(
                content=content,
                blog_title=blog_title,
                style=style,
                image_type=image_type,
                count=count,
                target_platform=target_platform,
                brand_guidelines=brand_guidelines
            )
            
            result_data = {
                "prompts": prompts,
                "image_prompts": prompts,  # Keep for backward compatibility
                "style": style,
                "image_type": image_type,
                "target_platform": target_platform,
                "count": len(prompts),
                "recommended_tools": self._get_recommended_tools(style, image_type),
                "technical_specs": self._get_technical_specs(target_platform, image_type)
            }
            
            logger.info(f"ImagePromptAgent completed successfully. Generated {len(prompts)} prompts.")
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"ImagePromptAgent execution failed: {str(e)}")
            raise AgentExecutionError("ImagePromptAgent", "execution", str(e))
    
    def _generate_image_prompts(self, content: str, blog_title: str, style: str, image_type: str, 
                              count: int, target_platform: str, brand_guidelines: Dict) -> List[Dict[str, Any]]:
        """
        Generate detailed image prompts based on content and specifications.
        """
        prompts = []
        
        # Extract key themes from content
        themes = self._extract_themes(content, blog_title)
        
        # Style-specific modifiers
        style_modifiers = {
            "professional": "clean, corporate, minimalist, high-quality photography",
            "creative": "artistic, vibrant colors, creative composition, unique perspective",
            "modern": "sleek, contemporary, geometric shapes, negative space",
            "elegant": "sophisticated, luxury, refined typography, premium quality"
        }
        
        # Image type specific requirements
        type_requirements = {
            "header": "wide aspect ratio, prominent title space, hero image composition",
            "section": "supportive illustration, clear visual metaphor, complementary to text",
            "infographic": "data visualization, charts, icons, structured layout",
            "social": "square format, engaging thumbnail, social media optimized",
            "background": "subtle pattern, texture, non-distracting background"
        }
        
        base_style = style_modifiers.get(style, style_modifiers["professional"])
        requirements = type_requirements.get(image_type, type_requirements["header"])
        
        for i in range(count):
            theme = themes[i % len(themes)] if themes else blog_title
            
            # Build comprehensive prompt
            prompt_parts = [
                f"{theme}",
                f"Style: {base_style}",
                f"Format: {requirements}",
                f"Platform: {target_platform} optimized"
            ]
            
            # Add brand guidelines if provided
            if brand_guidelines:
                if 'colors' in brand_guidelines:
                    prompt_parts.append(f"Brand colors: {brand_guidelines['colors']}")
                if 'style_notes' in brand_guidelines:
                    prompt_parts.append(f"Brand style: {brand_guidelines['style_notes']}")
            
            # Add technical specifications
            tech_specs = self._get_technical_specs(target_platform, image_type)
            prompt_parts.append(f"Resolution: {tech_specs['resolution']}")
            
            # Create final prompt
            final_prompt = ", ".join(prompt_parts)
            
            prompt_data = {
                "id": f"img_prompt_{i+1}",
                "prompt": final_prompt,
                "short_description": theme,
                "style": style,
                "image_type": image_type,
                "estimated_tokens": len(final_prompt.split()),
                "complexity": "medium" if len(prompt_parts) < 6 else "high",
                "alternative_prompts": self._generate_alternatives(theme, style, requirements)
            }
            
            prompts.append(prompt_data)
        
        return prompts
    
    def _extract_themes(self, content: str, blog_title: str) -> List[str]:
        """
        Extract key visual themes from content for image generation.
        """
        themes = [blog_title]
        
        # Simple keyword extraction for visual concepts
        visual_keywords = [
            "technology", "business", "growth", "innovation", "team", "success",
            "data", "analytics", "digital", "transformation", "strategy", "future",
            "leadership", "collaboration", "efficiency", "automation", "AI",
            "finance", "investment", "market", "customer", "solution"
        ]
        
        content_lower = content.lower()
        for keyword in visual_keywords:
            if keyword in content_lower:
                themes.append(f"{keyword.title()} concept visualization")
        
        return themes[:5]  # Limit to 5 themes
    
    def _get_recommended_tools(self, style: str, image_type: str) -> List[str]:
        """
        Recommend the best image generation tools for the specified style and type.
        """
        tools = {
            "professional": ["DALL-E 3", "Midjourney", "Adobe Firefly"],
            "creative": ["Midjourney", "Stable Diffusion", "Leonardo AI"],
            "modern": ["DALL-E 3", "Midjourney", "RunwayML"],
            "elegant": ["Midjourney", "Adobe Firefly", "DALL-E 3"]
        }
        
        return tools.get(style, ["DALL-E 3", "Midjourney"])
    
    def _get_technical_specs(self, target_platform: str, image_type: str) -> Dict[str, str]:
        """
        Get technical specifications for different platforms and image types.
        """
        specs = {
            "web": {
                "header": {"resolution": "1200x600", "format": "WebP/JPEG", "max_size": "500KB"},
                "section": {"resolution": "800x600", "format": "WebP/PNG", "max_size": "300KB"},
                "infographic": {"resolution": "1024x1024", "format": "PNG", "max_size": "800KB"}
            },
            "social": {
                "header": {"resolution": "1080x1080", "format": "JPEG", "max_size": "1MB"},
                "section": {"resolution": "1080x1350", "format": "JPEG", "max_size": "1MB"},
                "infographic": {"resolution": "1080x1080", "format": "PNG", "max_size": "2MB"}
            },
            "print": {
                "header": {"resolution": "3000x1800", "format": "PNG", "max_size": "5MB"},
                "section": {"resolution": "2400x1800", "format": "PNG", "max_size": "3MB"},
                "infographic": {"resolution": "3600x3600", "format": "PNG", "max_size": "8MB"}
            }
        }
        
        return specs.get(target_platform, {}).get(image_type, {
            "resolution": "1024x1024", 
            "format": "PNG", 
            "max_size": "1MB"
        })
    
    def _generate_alternatives(self, theme: str, style: str, requirements: str) -> List[str]:
        """
        Generate alternative prompt variations for the same theme.
        """
        alternatives = [
            f"{theme}, {style} style, alternative composition",
            f"{theme}, {style} aesthetic, different angle",
            f"{theme}, {style} design, creative interpretation"
        ]
        
        return alternatives[:2]  # Return 2 alternatives
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context for prompt generation.
        """
        if not context.get('content') and not context.get('blog_title'):
            logger.warning("Missing content and blog_title")
            return False
        
        style = context.get('style', 'professional')
        if style not in ['professional', 'creative', 'modern', 'elegant']:
            logger.warning(f"Invalid style: {style}")
            return False
        
        return True