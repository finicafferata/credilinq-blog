"""
Image Agent - Generates images based on blog content
"""

import os
import logging
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata
from ...core.exceptions import AgentExecutionError
import requests
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class ImageAgent(BaseAgent):
    """
    Agent specialized in generating images for blog content.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None, name: str = "ImageAgent", description: str = "Generates images for blog content"):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.IMAGE,
                name=name,
                description=description
            )
        super().__init__(metadata=metadata)
        self.agent_type = AgentType.IMAGE
        
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Generate images based on blog content.
        
        Args:
            context: Dictionary containing:
                - content: Blog content
                - blog_title: Title of the blog
                - outline: Blog outline sections
                - style: Image style preference (optional)
                - count: Number of images to generate (default: 3)
                - regenerate_id: ID of image to regenerate (optional)
        
        Returns:
            AgentResult with generated images
        """
        try:
            logger.info(f"ImageAgent executing for blog: {context.get('blog_title', 'Unknown')}")
            
            content = context.get('content', '')
            blog_title = context.get('blog_title', '')
            outline = context.get('outline', [])
            style = context.get('style', 'professional')
            count = context.get('count', 3)
            regenerate_id = context.get('regenerate_id')
            
            if not content and not blog_title:
                raise AgentExecutionError("No content or title provided for image generation")
            
            if regenerate_id:
                # Regenerate specific image
                image_prompts = self._generate_image_prompts(content, blog_title, outline, style)
                generated_images = self._generate_images(image_prompts[:1], regenerate_id)
            else:
                # Generate new images
                image_prompts = self._generate_image_prompts(content, blog_title, outline, style)
                generated_images = self._generate_images(image_prompts[:count])
            
            result_data = {
                "images": generated_images,
                "prompts": image_prompts,
                "style": style,
                "count": len(generated_images)
            }
            
            logger.info(f"ImageAgent completed successfully. Generated {len(generated_images)} images.")
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"ImageAgent execution failed: {str(e)}")
            raise AgentExecutionError("ImageAgent", "execution", str(e))
    
    def _generate_image_prompts(self, content: str, blog_title: str, outline: List[str], style: str) -> List[str]:
        """
        Generate image prompts based on blog content.
        """
        prompts = []
        
        # Generate prompt for main blog image
        main_prompt = f"Professional blog header image for: {blog_title}. Style: {style}, clean design, modern typography"
        prompts.append(main_prompt)
        
        # Generate prompts for each section
        for section in outline[:3]:  # Limit to first 3 sections
            section_prompt = f"Professional illustration for blog section: {section}. Style: {style}, clean design"
            prompts.append(section_prompt)
        
        # Generate infographic prompt
        infographic_prompt = f"Professional infographic for: {blog_title}. Style: {style}, data visualization, clean design"
        prompts.append(infographic_prompt)
        
        return prompts
    
    def _generate_images(self, prompts: List[str], regenerate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate images using image generation API (mock implementation).
        In production, this would integrate with DALL-E, Midjourney, or similar.
        """
        images = []
        
        for i, prompt in enumerate(prompts):
            # Mock image generation - in production this would call a real API
            image_id = regenerate_id if regenerate_id else f"img_{i+1}"
            image_data = {
                "id": image_id,
                "prompt": prompt,
                "url": f"https://via.placeholder.com/800x600/4F46E5/FFFFFF?text=Generated+Image+{image_id}",
                "alt_text": f"Generated image for: {prompt[:50]}...",
                "style": "professional",
                "size": "800x600"
            }
            images.append(image_data)
        
        return images
    
    def _call_image_api(self, prompt: str, style: str = "professional") -> str:
        """
        Call external image generation API.
        This is a placeholder for real API integration.
        """
        # TODO: Integrate with real image generation API
        # Example APIs: OpenAI DALL-E, Midjourney, Stable Diffusion
        
        # For now, return a placeholder URL
        return f"https://via.placeholder.com/800x600/4F46E5/FFFFFF?text={prompt[:30]}"
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context for image generation.
        """
        required_fields = ['content', 'blog_title']
        
        for field in required_fields:
            if not context.get(field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True
