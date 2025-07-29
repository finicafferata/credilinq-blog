from typing import Dict, List, Optional, Any
import os
import logging
import re
import requests
import base64
from io import BytesIO
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

class ImageGenerationAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent that generates images using AI image generation APIs (DALL-E, Stable Diffusion).
    Also creates optimized prompts for image generation based on content analysis.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.IMAGE_PROMPT_GENERATOR,
                name="ImageGenerationAgent",
                description="Generates images and optimized prompts for visual content creation",
                capabilities=[
                    "image_generation",
                    "prompt_optimization",
                    "content_analysis",
                    "visual_concept_extraction",
                    "multi_provider_support"
                ],
                version="2.1.0"  # Version bumped to reflect improvements
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
        
        # Initialize image generation settings
        self.enabled_providers = []
        self.default_provider = None
        
    def _initialize(self):
        """Initialize the LLM and image generation providers."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.8,  # More creativity for image prompts
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Check available image generation providers
            self._initialize_image_providers(settings)
            
            self.logger.info(f"ImageGenerationAgent initialized with providers: {self.enabled_providers}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ImageGenerationAgent: {str(e)}")
            raise
    
    def _initialize_image_providers(self, settings):
        """Initialize available image generation providers."""
        # Check DALL-E availability (uses OpenAI API key)
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            self.enabled_providers.append('dalle')
            if not self.default_provider:
                self.default_provider = 'dalle'
        
        # Check Stability AI availability
        if hasattr(settings, 'stability_api_key') and getattr(settings, 'stability_api_key', None):
            self.enabled_providers.append('stability')
            if not self.default_provider:
                self.default_provider = 'stability'
        
        # If no providers are available, log warning but continue
        if not self.enabled_providers:
            self.logger.warning("No image generation providers configured. Image generation will return mock responses.")
        
        # Style and purpose configurations for each image type
        self.image_styles = {
            "Blog Header": {
                "style": "professional, clean, header banner",
                "aspect_ratio": "16:9",
                "purpose": "main visual for blog post",
                "elements": ["title text space", "professional aesthetic", "brand-friendly"]
            },
            "LinkedIn Post Image": {
                "style": "professional, business-oriented, social media",
                "aspect_ratio": "1:1",
                "purpose": "social media engagement",
                "elements": ["eye-catching", "informational", "professional branding"]
            },
            "Instagram Post Image": {
                "style": "vibrant, modern, social media optimized",
                "aspect_ratio": "1:1",
                "purpose": "social media engagement",
                "elements": ["colorful", "trendy", "visually appealing", "mobile-friendly"]
            },
            "Twitter Card": {
                "style": "simple, clean, attention-grabbing",
                "aspect_ratio": "2:1",
                "purpose": "twitter card preview",
                "elements": ["minimal text", "clear message", "brand colors"]
            },
            "Thumbnail": {
                "style": "bold, high-contrast, clickable",
                "aspect_ratio": "16:9",
                "purpose": "video or article thumbnail",
                "elements": ["large text", "dramatic", "compelling visual"]
            },
            "Infographic": {
                "style": "informational, structured, data-focused",
                "aspect_ratio": "9:16",
                "purpose": "educational content",
                "elements": ["charts", "diagrams", "structured layout", "educational"]
            }
        }
        
        # Keywords to improve prompt quality
        self.quality_enhancers = [
            "high quality", "detailed", "professional photography",
            "sharp focus", "well-lit", "4K resolution",
            "clean composition", "modern design"
        ]
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for image generation."""
        super()._validate_input(input_data)
        
        # Handle both legacy and new input formats
        if "content_topic" not in input_data and "topic" in input_data:
            input_data["content_topic"] = input_data["topic"]
        if "content_body" not in input_data and "content" in input_data:
            input_data["content_body"] = input_data["content"]
        
        required_fields = ["content_topic", "content_body"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation
        self.security_validator.validate_input(str(input_data["content_topic"]))
        self.security_validator.validate_input(str(input_data["content_body"]))

    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generates images and optimized prompts based on content analysis.
        
        Args:
            input_data: Dictionary containing:
                - content_topic: The main title or topic
                - content_body: The complete content
                - image_type: Type of image to generate (optional)
                - generate_image: Whether to actually generate image (optional)
                - provider: Preferred image generation provider (optional)
            context: Execution context
            
        Returns:
            AgentResult: Generated prompt and optionally actual image
        """
        try:
            content_topic = input_data["content_topic"]
            content_body = input_data["content_body"]
            image_type = input_data.get("image_type", "Blog Header")
            generate_image = input_data.get("generate_image", False)
            provider = input_data.get("provider", self.default_provider)
            
            self.logger.info(f"Generating image prompt for topic: {content_topic}")
            
            # Determine the image type
            target_asset = self._determine_image_type(content_topic, content_body, image_type)
            
            # Extract visual concepts
            key_concepts = self._extract_visual_concepts(content_topic, content_body)
            
            # Identify tone and industry
            tone_and_industry = self._analyze_tone_and_industry(content_topic, content_body)
            
            # Generate base prompt
            base_prompt = self._generate_base_prompt(
                content_topic, 
                key_concepts, 
                tone_and_industry, 
                target_asset
            )
            
            # Optimize final prompt
            final_prompt = self._optimize_prompt(base_prompt, target_asset)
            
            result_data = {
                "image_prompt": final_prompt,
                "target_asset": target_asset,
                "key_concepts": key_concepts,
                "tone_analysis": tone_and_industry,
                "prompt_quality": self.analyze_prompt_quality(final_prompt)
            }
            
            # Generate actual image if requested
            if generate_image and self.enabled_providers:
                try:
                    image_result = self._generate_image_with_provider(final_prompt, provider)
                    result_data.update(image_result)
                except Exception as e:
                    self.logger.warning(f"Image generation failed: {str(e)}")
                    result_data["image_generation_error"] = str(e)
            
            self.logger.info(f"Image prompt successfully generated for {target_asset}")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "image_generation",
                    "target_asset": target_asset,
                    "image_generated": "image_url" in result_data,
                    "provider_used": provider if generate_image else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="IMAGE_GENERATION_FAILED"
            )

    def _determine_image_type(self, topic: str, content: str, suggested_type: str = None) -> str:
        """Determines the most appropriate image type based on content."""
        if suggested_type and suggested_type in self.image_styles:
            return suggested_type
        return "Blog Header"  # Default logic; could be extended in the future

    def _extract_visual_concepts(self, topic: str, content: str) -> List[str]:
        """Extracts key visual concepts from content using AI."""
        prompt = f"""
        Act as a professional Visual Content Strategist and Creative Director with 10+ years of experience in digital marketing and brand visualization. Analyze this content to extract powerful visual concepts for image generation.

        **Content Analysis:**
        **Title:** "{topic}"
        **Content Preview:** {content[:2500]}

        **Instructions:**
        - Identify 5 specific visual concepts that can effectively represent this content
        - Focus on concrete, visually representable elements rather than abstract ideas
        - Consider visual metaphors that would resonate with the target audience
        - Think about composition elements that would create compelling visuals
        - Prioritize concepts that work well for professional/business imagery

        **Visual Concept Categories to Consider:**
        - Physical objects and settings (office spaces, devices, tools)
        - Visual metaphors (growth charts, connecting networks, building blocks)
        - Human elements (professionals, team dynamics, presentations)
        - Environmental contexts (modern workspaces, conference rooms, digital interfaces)
        - Symbolic representations (progress indicators, success symbols, innovation imagery)

        **Negative Constraints:**
        - **Avoid** purely abstract concepts that can't be visualized
        - **Do not** include concepts that would be inappropriate for business contexts
        - **Avoid** overly complex scenarios that would be hard to represent clearly

        **Output Format:**
        Return ONLY a Python list inside <concepts> tags:
        <concepts>
        ["specific visual concept 1", "concrete visual element 2", "visual metaphor 3", "environmental setting 4", "symbolic representation 5"]
        </concepts>
        """
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            # Parse concepts from tags
            import re
            match = re.search(r"<concepts>(.*?)</concepts>", response.content, re.DOTALL)
            if match:
                import ast
                concepts = ast.literal_eval(match.group(1).strip())
                return concepts if isinstance(concepts, list) else []
        except Exception as e:
            self.logger.warning(f"Visual concept extraction failed: {str(e)}, using fallback")
            
        return self._extract_concepts_fallback(topic, content)

    def _extract_concepts_fallback(self, topic: str, content: str) -> List[str]:
        """Fallback method for extracting visual concepts."""
        business_keywords = [
            "business", "company", "technology", "digital", "financial",
            "strategy", "growth", "innovation", "team", "leadership",
            "data", "analytics", "market", "customer", "product"
        ]
        
        text_lower = (topic + " " + content).lower()
        found_concepts = [kw for kw in business_keywords if kw in text_lower]
        
        if len(found_concepts) < 3:
            found_concepts.extend(["modern office", "professionals", "technology"])
            
        return found_concepts[:5]

    def _analyze_tone_and_industry(self, topic: str, content: str) -> Dict[str, str]:
        """Analyzes the tone and industry context of the content."""
        prompt = f"""
        Analyze this content and determine:
        1. TONE: (professional, casual, technical, creative, etc.)
        2. INDUSTRY: (technology, finance, marketing, healthcare, etc.)
        3. AUDIENCE: (executives, technical staff, general audience, students, etc.)
        
        TITLE: {topic}
        CONTENT: {content[:1500]}
        
        Respond in JSON format:
        {{"tone": "value", "industry": "value", "audience": "value"}}
        """
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            import json
            analysis = json.loads(response.content.strip())
            return analysis
        except:
            return {
                "tone": "professional",
                "industry": "technology",
                "audience": "professionals"
            }

    def _generate_base_prompt(self, topic: str, concepts: List[str], analysis: Dict[str, str], target_asset: str) -> str:
        """Generates the base prompt for the image."""
        style_config = self.image_styles.get(target_asset, self.image_styles["Blog Header"])
        
        prompt = f"""
        Act as an expert AI Image Prompt Engineer and Visual Designer who specializes in creating highly effective prompts for AI image generation models like DALL-E and Midjourney. Your task is to craft the perfect prompt for generating a compelling {target_asset.lower()}.

        **Content Brief:**
        **Topic:** "{topic}"
        **Target Asset:** {target_asset}
        **Tone:** {analysis.get('tone', 'professional')}
        **Industry Context:** {analysis.get('industry', 'general')}
        **Target Audience:** {analysis.get('audience', 'professionals')}

        **Visual Concepts to Integrate:**
        {', '.join(concepts[:4])}

        **Technical Specifications:**
        - **Style Requirement:** {style_config['style']}
        - **Aspect Ratio:** {style_config['aspect_ratio']}
        - **Primary Purpose:** {style_config['purpose']}
        - **Essential Elements:** {', '.join(style_config['elements'])}

        **Prompt Engineering Instructions:**
        - Create a detailed, specific prompt that will generate a high-quality, professional image
        - Include specific visual elements, composition details, and aesthetic choices
        - Incorporate lighting, color palette, and atmospheric details
        - Ensure the prompt is optimized for the target asset type and purpose
        - Balance artistic creativity with business appropriateness
        - Include technical quality descriptors for best results

        **Professional Standards:**
        - Maintain brand-safe, business-appropriate imagery
        - Ensure visual clarity and professional impact
        - Consider the psychology of visual communication for the target audience
        - Optimize for the specific use case ({style_config['purpose']})

        **Negative Constraints:**
        - **Avoid** including specific text or typography in the image unless essential
        - **Do not** include copyrighted elements or specific brand logos
        - **Avoid** cluttered compositions that would detract from the main message

        **Output Format:**
        Return ONLY the final optimized image generation prompt inside <prompt> tags:
        <prompt>
        [Your detailed, professional image generation prompt]
        </prompt>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            # Parse prompt from tags
            import re
            match = re.search(r"<prompt>(.*?)</prompt>", response.content, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                # Fallback: return the whole response if no tags found
                return response.content.strip()
        except Exception as e:
            self.logger.warning(f"Base prompt generation failed: {str(e)}, using fallback")
            return self._generate_fallback_prompt(topic, concepts, style_config)
    
    def _generate_fallback_prompt(self, topic: str, concepts: List[str], style_config: Dict[str, Any]) -> str:
        """Generate a fallback prompt when AI generation fails."""
        self.logger.info(f"Using fallback prompt generation for: {topic}")
        
        # Create a professional fallback prompt
        main_concept = concepts[0] if concepts else "professional business setting"
        style = style_config.get('style', 'professional, clean')
        
        fallback_prompt = f"""Professional image about "{topic}" featuring {main_concept}, {style} aesthetic, 
        high-quality composition, modern design, business-appropriate, clean and impactful visual, 
        {style_config.get('aspect_ratio', '16:9')} aspect ratio, suitable for {style_config.get('purpose', 'business use')}"""
        
        return fallback_prompt

    def _optimize_prompt(self, base_prompt: str, target_asset: str) -> str:
        """Optimizes the final prompt with technical and quality enhancements."""
        style_config = self.image_styles[target_asset]
        
        technical_specs = [
            f"aspect ratio {style_config['aspect_ratio']}",
            "high quality",
            "professional composition",
            "clean and modern design"
        ]
        
        optimized_prompt = f"""
        MAIN PROMPT:
        {base_prompt}
        
        TECHNICAL SPECIFICATIONS:
        - {', '.join(technical_specs)}
        - {style_config['style']}
        - Optimized for {style_config['purpose']}
        
        QUALITY ENHANCERS:
        {', '.join(self.quality_enhancers[:4])}
        
        STYLE NOTES:
        - Avoid text in image unless explicitly required
        - Use a professional color palette
        - Ensure visual clarity and impact
        - Brand-safe and business-appropriate
        """
        return optimized_prompt

    def _generate_image_with_provider(self, prompt: str, provider: str = None) -> Dict[str, Any]:
        """
        Generate image using the specified provider.
        
        Args:
            prompt: Image generation prompt
            provider: Provider to use ('dalle' or 'stability')
            
        Returns:
            Dict containing image_url, provider_used, and generation_metadata
        """
        if not provider or provider not in self.enabled_providers:
            provider = self.default_provider
        
        if not provider:
            raise AgentExecutionError("image_generation", "provider_selection", "No image generation provider available")
        
        self.logger.info(f"Generating image with {provider}: {prompt[:100]}...")
        
        try:
            if provider == 'dalle':
                return self._generate_with_dalle(prompt)
            elif provider == 'stability':
                return self._generate_with_stability(prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Image generation failed with {provider}: {str(e)}")
            # Return fallback mock response
            return self._generate_mock_response(prompt, provider)
    
    def _generate_with_dalle(self, prompt: str) -> Dict[str, Any]:
        """Generate image using DALL-E API."""
        try:
            from openai import OpenAI
            from ...config.settings import get_settings
            
            settings = get_settings()
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Clean prompt for DALL-E (remove technical specifications)
            dalle_prompt = self._clean_prompt_for_dalle(prompt)
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            revised_prompt = getattr(response.data[0], 'revised_prompt', dalle_prompt)
            
            return {
                "image_url": image_url,
                "provider_used": "dalle",
                "generation_metadata": {
                    "model": "dall-e-3",
                    "size": "1024x1024",
                    "quality": "standard",
                    "original_prompt": prompt,
                    "revised_prompt": revised_prompt,
                    "cost_estimate": 0.04  # USD per image for DALL-E 3
                }
            }
            
        except Exception as e:
            self.logger.error(f"DALL-E generation failed: {str(e)}")
            raise
    
    def _generate_with_stability(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Stability AI API."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            api_key = getattr(settings, 'stability_api_key', None)
            if not api_key:
                raise ValueError("Stability API key not configured")
            
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            body = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            }
            
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert base64 image to URL (you'd typically upload to storage)
            image_data = data["artifacts"][0]["base64"]
            image_url = self._upload_image_to_storage(image_data)
            
            return {
                "image_url": image_url,
                "provider_used": "stability",
                "generation_metadata": {
                    "model": "stable-diffusion-xl-1024-v1-0",
                    "size": "1024x1024",
                    "cfg_scale": 7,
                    "steps": 30,
                    "cost_estimate": 0.02  # USD per image for Stability AI
                }
            }
            
        except Exception as e:
            self.logger.error(f"Stability AI generation failed: {str(e)}")
            raise
    
    def _clean_prompt_for_dalle(self, prompt: str) -> str:
        """Clean prompt for DALL-E by removing technical specifications."""
        # DALL-E doesn't like technical specifications, so extract the main description
        lines = prompt.split('\n')
        main_prompt = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and not line.startswith('TECHNICAL') and not line.startswith('QUALITY'):
                if 'MAIN PROMPT:' in line:
                    continue
                main_prompt += line + " "
        
        # Fallback to original if cleaning failed
        if len(main_prompt.strip()) < 20:
            main_prompt = prompt
        
        # Ensure it's under DALL-E's character limit
        return main_prompt[:1000]
    
    def _upload_image_to_storage(self, base64_data: str) -> str:
        """
        Upload base64 image data to storage and return URL.
        This is a placeholder - implement with your preferred storage solution.
        """
        # TODO: Implement actual storage upload (S3, Supabase Storage, etc.)
        # For now, return a placeholder URL
        import hashlib
        image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:12]
        return f"https://storage.example.com/generated-images/{image_hash}.png"
    
    def _generate_mock_response(self, prompt: str, provider: str) -> Dict[str, Any]:
        """Generate a mock response when actual generation fails."""
        import hashlib
        mock_id = hashlib.md5(prompt.encode()).hexdigest()[:12]
        
        return {
            "image_url": f"https://via.placeholder.com/1024x1024/4F46E5/FFFFFF?text=Generated+Image+{mock_id}",
            "provider_used": f"{provider}_mock",
            "generation_metadata": {
                "model": "mock",
                "size": "1024x1024",
                "note": "This is a placeholder image. Configure image generation providers for actual images.",
                "cost_estimate": 0.0
            }
        }

    def create_image_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """Creates prompt variations to generate multiple image options."""
        variations = []
        style_variations = [
            "photorealistic style",
            "digital art style", 
            "illustration style",
            "minimalist design",
            "corporate aesthetic"
        ]
        
        for i in range(num_variations):
            if i < len(style_variations):
                variation = f"{base_prompt}, {style_variations[i]}"
            else:
                variation = f"{base_prompt}, variation {i+1}"
            variations.append(variation)
            
        return variations

    def analyze_prompt_quality(self, prompt: str) -> Dict[str, Any]:
        """Analyzes the quality and completeness of an image prompt."""
        
        # Enhanced quality analysis metrics
        quality_indicators = {
            "style_descriptors": ["style", "aesthetic", "design", "artistic", "visual", "modern", "professional"],
            "technical_specs": ["resolution", "quality", "aspect", "4k", "hd", "sharp", "detailed", "high-quality"],
            "visual_elements": ["color", "composition", "lighting", "texture", "atmosphere", "mood", "contrast"],
            "subject_clarity": ["featuring", "showing", "depicting", "with", "containing", "including"],
            "professional_terms": ["professional", "business", "corporate", "clean", "polished", "sophisticated"],
            "composition_terms": ["centered", "balanced", "dynamic", "focal", "background", "foreground", "layout"]
        }
        
        analysis = {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "has_style_descriptors": any(word in prompt.lower() for word in quality_indicators["style_descriptors"]),
            "has_technical_specs": any(word in prompt.lower() for word in quality_indicators["technical_specs"]),
            "has_visual_elements": any(word in prompt.lower() for word in quality_indicators["visual_elements"]),
            "has_subject_clarity": any(word in prompt.lower() for word in quality_indicators["subject_clarity"]),
            "has_professional_terms": any(word in prompt.lower() for word in quality_indicators["professional_terms"]),
            "has_composition_terms": any(word in prompt.lower() for word in quality_indicators["composition_terms"]),
            "estimated_effectiveness": "medium"
        }
        
        # Calculate comprehensive quality score
        quality_factors = [
            analysis["length"] > 100,  # Adequate length
            analysis["word_count"] > 15,  # Sufficient detail
            analysis["has_style_descriptors"],
            analysis["has_technical_specs"],
            analysis["has_visual_elements"],
            analysis["has_subject_clarity"],
            analysis["has_professional_terms"],
            analysis["has_composition_terms"]
        ]
        
        score = sum(quality_factors)
        total_possible = len(quality_factors)
        
        # Calculate effectiveness based on comprehensive scoring
        effectiveness_ratio = score / total_possible
        
        if effectiveness_ratio >= 0.75:
            analysis["estimated_effectiveness"] = "excellent"
        elif effectiveness_ratio >= 0.6:
            analysis["estimated_effectiveness"] = "high"
        elif effectiveness_ratio >= 0.4:
            analysis["estimated_effectiveness"] = "medium"
        else:
            analysis["estimated_effectiveness"] = "low"
        
        # Add quality score and recommendations
        analysis["quality_score"] = score
        analysis["max_possible_score"] = total_possible
        analysis["improvement_suggestions"] = self._generate_improvement_suggestions(analysis)
        
        return analysis
    
    def _generate_improvement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving prompt quality."""
        suggestions = []
        
        if not analysis["has_style_descriptors"]:
            suggestions.append("Add style descriptors (e.g., 'modern', 'professional', 'minimalist')")
        
        if not analysis["has_technical_specs"]:
            suggestions.append("Include technical quality specs (e.g., 'high-quality', 'detailed', '4K')")
        
        if not analysis["has_visual_elements"]:
            suggestions.append("Specify visual elements (e.g., 'warm lighting', 'balanced composition')")
        
        if not analysis["has_subject_clarity"]:
            suggestions.append("Clarify the main subject with specific descriptors")
        
        if analysis["length"] < 100:
            suggestions.append("Expand prompt length for more detailed specifications")
        
        if not analysis["has_composition_terms"]:
            suggestions.append("Add composition guidance (e.g., 'centered', 'balanced layout')")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    # Legacy compatibility methods
    def execute_legacy(self, content_topic: str, content_body: str) -> str:
        """
        Legacy interface for backward compatibility.
        Returns only the image prompt as a string.
        """
        input_data = {
            "content_topic": content_topic,
            "content_body": content_body,
            "generate_image": False
        }
        
        result = self.execute_safe(input_data)
        if result.success:
            return result.data.get("image_prompt", "Error generating prompt")
        else:
            return f"Error: {result.error_message}"
    
    def generate_image_with_prompt(self, content_topic: str, content_body: str, provider: str = None) -> Dict[str, Any]:
        """
        Generate both prompt and image.
        Convenience method for full image generation workflow.
        """
        input_data = {
            "content_topic": content_topic,
            "content_body": content_body,
            "generate_image": True,
            "provider": provider
        }
        
        result = self.execute_safe(input_data)
        return result.data if result.success else {"error": result.error_message}


# Keep the original class name for backward compatibility
ImagePromptAgent = ImageGenerationAgent
