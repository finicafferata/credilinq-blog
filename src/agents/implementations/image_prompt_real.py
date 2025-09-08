"""
Real ImagePromptAgent Implementation - LLM-powered image prompt generation.

This agent generates detailed, optimized prompts for AI image generation tools
like DALL-E, Midjourney, Stable Diffusion, etc.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.llm_client import LLMClient
from ...core.security import InputValidator, SecurityError
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RealImagePromptAgent(BaseAgent):
    """
    Real LLM-powered image prompt generator for AI art generation.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.IMAGE_PROMPT,
                name="RealImagePromptAgent",
                description="LLM-powered AI image prompt generation and optimization",
                version="1.0.0",
                capabilities=[
                    "prompt_generation",
                    "style_optimization",
                    "technical_parameter_tuning",
                    "platform_optimization",
                    "brand_alignment"
                ],
                dependencies=["llm_client", "input_validator"],
                performance_targets={
                    "response_time": 3.0,
                    "quality_score": 0.88
                }
            )
        super().__init__(metadata)
        self.llm_client = LLMClient()
        self.input_validator = InputValidator()

    async def execute(self, input_data: Dict[str, Any], context: Optional[AgentExecutionContext] = None) -> AgentResult:
        """
        Generate optimized image prompts for AI art generation.
        
        Args:
            input_data: Must contain:
                - content_description: Description of desired image content
                - image_style: Desired artistic style (optional)
                - target_platform: AI platform (dall-e, midjourney, stable-diffusion)
                - aspect_ratio: Desired aspect ratio (optional)
                - brand_context: Brand context for alignment (optional)
        """
        start_time = datetime.now()
        
        try:
            # Input validation and security check
            self._validate_input(input_data)
            
            # Extract parameters
            content_description = input_data["content_description"]
            image_style = input_data.get("image_style", "professional")
            target_platform = input_data.get("target_platform", "dall-e")
            aspect_ratio = input_data.get("aspect_ratio", "1:1")
            brand_context = input_data.get("brand_context", {})
            
            # Performance tracking
            if context and hasattr(context, 'performance_tracker'):
                await context.performance_tracker.start_agent_execution(self.metadata.name, input_data)
            
            logger.info(f"Generating image prompts for {target_platform} platform")
            
            # Generate multiple prompt variations
            primary_prompt = await self._generate_primary_prompt(
                content_description, image_style, target_platform, aspect_ratio, brand_context
            )
            
            alternative_prompts = await self._generate_alternative_prompts(
                content_description, image_style, target_platform, primary_prompt
            )
            
            # Generate technical parameters
            technical_params = self._generate_technical_parameters(target_platform, image_style, aspect_ratio)
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            quality_score = self._assess_prompt_quality(primary_prompt, alternative_prompts, target_platform)
            
            # Compile comprehensive result
            result_data = {
                "primary_prompt": primary_prompt,
                "alternative_prompts": alternative_prompts,
                "technical_parameters": technical_params,
                "platform_optimization": {
                    "target_platform": target_platform,
                    "optimized_for": self._get_platform_features(target_platform),
                    "recommended_settings": technical_params
                },
                "prompt_analysis": {
                    "primary_word_count": len(primary_prompt["prompt"].split()),
                    "style_elements": primary_prompt.get("style_elements", []),
                    "technical_elements": primary_prompt.get("technical_elements", []),
                    "brand_alignment": self._assess_brand_alignment(primary_prompt, brand_context)
                },
                "usage_recommendations": {
                    "best_for": self._get_usage_recommendations(image_style, content_description),
                    "variations_count": len(alternative_prompts),
                    "testing_strategy": "Start with primary prompt, test alternatives for variety"
                },
                "performance_metrics": {
                    "execution_time": execution_time,
                    "quality_score": quality_score,
                    "prompts_generated": 1 + len(alternative_prompts)
                }
            }
            
            # Decision reasoning
            decision_reasoning = self._generate_decision_reasoning(
                content_description, target_platform, primary_prompt, quality_score
            )
            result_data["decision_reasoning"] = decision_reasoning
            
            # Performance tracking completion
            if context and hasattr(context, 'performance_tracker'):
                await context.performance_tracker.complete_agent_execution(
                    self.metadata.name, True, execution_time, quality_score
                )
            
            logger.info(f"Image prompt generation completed in {execution_time:.2f}s with quality score {quality_score:.3f}")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "execution_time": execution_time,
                    "quality_score": quality_score,
                    "target_platform": target_platform
                },
                agent_id=self.metadata.name,
                timestamp=datetime.now()
            )
            
        except SecurityError as e:
            logger.error(f"Security validation failed in {self.metadata.name}: {e}")
            return AgentResult(
                success=False,
                error=f"Security validation failed: {str(e)}",
                agent_id=self.metadata.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Image prompt generation failed: {e}")
            
            if context and hasattr(context, 'performance_tracker'):
                await context.performance_tracker.complete_agent_execution(
                    self.metadata.name, False, execution_time, 0.0
                )
            
            return AgentResult(
                success=False,
                error=f"Image prompt generation failed: {str(e)}",
                agent_id=self.metadata.name,
                timestamp=datetime.now(),
                metadata={"execution_time": execution_time}
            )

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for image prompt generation."""
        # Security validation
        self.input_validator.validate_input(input_data)
        
        # Business logic validation
        if "content_description" not in input_data:
            raise ValueError("Missing required field: content_description")
        
        if len(input_data["content_description"]) < 10:
            raise ValueError("Content description too short (minimum 10 characters)")
        
        if len(input_data["content_description"]) > 1000:
            raise ValueError("Content description too long (maximum 1000 characters)")
        
        # Validate platform
        supported_platforms = ["dall-e", "midjourney", "stable-diffusion", "leonardo", "runway"]
        target_platform = input_data.get("target_platform", "dall-e")
        if target_platform not in supported_platforms:
            raise ValueError(f"Unsupported platform: {target_platform}. Supported: {supported_platforms}")
        
        # Validate aspect ratio
        valid_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "2:3", "3:2"]
        aspect_ratio = input_data.get("aspect_ratio", "1:1")
        if aspect_ratio not in valid_ratios:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}. Valid ratios: {valid_ratios}")

    async def _generate_primary_prompt(self, content_description: str, image_style: str,
                                     target_platform: str, aspect_ratio: str, 
                                     brand_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the primary optimized image prompt."""
        
        # Platform-specific optimization guidelines
        platform_guidelines = {
            "dall-e": {
                "style": "Natural language descriptions work best",
                "length": "Medium length (50-200 words)",
                "technical": "Focus on clear, descriptive language",
                "avoid": "Avoid overly complex artistic terms"
            },
            "midjourney": {
                "style": "Artistic and technical terms encouraged",
                "length": "Concise but detailed (20-100 words)",
                "technical": "Use parameters like --v, --ar, --stylize",
                "avoid": "Avoid overly long descriptions"
            },
            "stable-diffusion": {
                "style": "Keyword-heavy, comma-separated",
                "length": "Variable length acceptable",
                "technical": "Use quality tags, artist references",
                "avoid": "Avoid negative prompts in main prompt"
            }
        }
        
        guidelines = platform_guidelines.get(target_platform, platform_guidelines["dall-e"])
        
        system_prompt = f"""You are an expert AI art prompt engineer specializing in {target_platform} optimization.

Your task is to create an optimized image generation prompt based on:
- Content description: {content_description}
- Desired style: {image_style}
- Target platform: {target_platform}
- Aspect ratio: {aspect_ratio}
- Brand context: {json.dumps(brand_context, default=str)}

Platform-specific guidelines for {target_platform}:
- Style approach: {guidelines['style']}
- Optimal length: {guidelines['length']}
- Technical focus: {guidelines['technical']}
- Avoid: {guidelines['avoid']}

Create a comprehensive prompt that includes:
1. Core visual description
2. Style and artistic elements
3. Technical quality indicators
4. Composition and framing
5. Platform-specific optimizations

Provide your response in JSON format:
{{
    "prompt": "The complete optimized prompt text",
    "style_elements": ["style element 1", "style element 2"],
    "technical_elements": ["technical element 1", "technical element 2"],
    "composition_notes": "Notes about composition and framing",
    "platform_specific": "Platform-specific optimizations applied",
    "brand_alignment": "How the prompt aligns with brand context"
}}"""
        
        human_prompt = f"""Generate an optimized {target_platform} prompt for: {content_description}

Style: {image_style}
Aspect ratio: {aspect_ratio}
Brand context: {brand_context.get('brand_voice', 'N/A')}

Please create a high-quality, optimized prompt following the guidelines."""
        
        try:
            response = await self.llm_client.generate_response([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                result["generated_at"] = datetime.now().isoformat()
                result["tokens_used"] = getattr(response, 'usage', {}).get('total_tokens', 80)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "prompt": response.content,
                    "style_elements": [image_style],
                    "technical_elements": [f"optimized for {target_platform}"],
                    "generated_at": datetime.now().isoformat(),
                    "tokens_used": 80,
                    "parsing_note": "Used fallback parsing"
                }
                
        except Exception as e:
            logger.error(f"Failed to generate primary prompt: {e}")
            raise

    async def _generate_alternative_prompts(self, content_description: str, image_style: str,
                                          target_platform: str, primary_prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative prompt variations."""
        
        variations = [
            {"variation_type": "simplified", "description": "Simpler, more direct approach"},
            {"variation_type": "detailed", "description": "More detailed and specific"},
            {"variation_type": "creative", "description": "Creative interpretation with artistic flair"}
        ]
        
        alternative_prompts = []
        
        for variation in variations:
            try:
                system_prompt = f"""Create a {variation['variation_type']} variation of an image prompt.

Original prompt concept: {content_description}
Primary prompt: {primary_prompt['prompt'][:200]}...
Variation type: {variation['description']}

Create a {variation['variation_type']} version that maintains the core concept but offers a different approach.
Keep it optimized for {target_platform}.

Respond with just the alternative prompt text, no JSON needed."""
                
                human_prompt = f"Create a {variation['variation_type']} variation of the image prompt."
                
                response = await self.llm_client.generate_response([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ])
                
                alternative_prompts.append({
                    "variation_type": variation["variation_type"],
                    "prompt": response.content,
                    "description": variation["description"],
                    "generated_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Failed to generate {variation['variation_type']} variation: {e}")
                continue
        
        return alternative_prompts

    def _generate_technical_parameters(self, target_platform: str, image_style: str, 
                                     aspect_ratio: str) -> Dict[str, Any]:
        """Generate platform-specific technical parameters."""
        
        platform_params = {
            "dall-e": {
                "size": self._get_dalle_size(aspect_ratio),
                "quality": "hd" if "professional" in image_style.lower() else "standard",
                "style": "vivid" if "vibrant" in image_style.lower() else "natural"
            },
            "midjourney": {
                "aspect_ratio": f"--ar {aspect_ratio}",
                "version": "--v 6",
                "stylize": "--stylize 100" if "artistic" in image_style.lower() else "--stylize 50",
                "quality": "--quality 1"
            },
            "stable-diffusion": {
                "width": self._get_sd_dimensions(aspect_ratio)[0],
                "height": self._get_sd_dimensions(aspect_ratio)[1],
                "steps": 30,
                "cfg_scale": 7.5,
                "sampler": "DPM++ 2M Karras"
            }
        }
        
        return platform_params.get(target_platform, {})

    def _get_dalle_size(self, aspect_ratio: str) -> str:
        """Convert aspect ratio to DALL-E size parameter."""
        ratio_to_size = {
            "1:1": "1024x1024",
            "16:9": "1792x1024",
            "9:16": "1024x1792"
        }
        return ratio_to_size.get(aspect_ratio, "1024x1024")

    def _get_sd_dimensions(self, aspect_ratio: str) -> tuple:
        """Convert aspect ratio to Stable Diffusion dimensions."""
        ratio_to_dims = {
            "1:1": (512, 512),
            "16:9": (768, 432),
            "9:16": (432, 768),
            "4:3": (640, 480),
            "3:4": (480, 640)
        }
        return ratio_to_dims.get(aspect_ratio, (512, 512))

    def _get_platform_features(self, platform: str) -> List[str]:
        """Get key features for each platform."""
        features = {
            "dall-e": ["Natural language processing", "High coherence", "Safety filtering"],
            "midjourney": ["Artistic style", "Parameter control", "Community features"],
            "stable-diffusion": ["Open source", "Customizable", "Fine-tuning support"]
        }
        return features.get(platform, ["AI image generation"])

    def _assess_brand_alignment(self, prompt: Dict[str, Any], brand_context: Dict[str, Any]) -> str:
        """Assess how well the prompt aligns with brand context."""
        if not brand_context:
            return "No brand context provided"
        
        brand_voice = brand_context.get("brand_voice", "")
        if "professional" in brand_voice.lower() and "professional" in prompt.get("prompt", "").lower():
            return "Good alignment with professional brand voice"
        elif "creative" in brand_voice.lower() and any(word in prompt.get("prompt", "").lower() for word in ["artistic", "creative", "innovative"]):
            return "Good alignment with creative brand voice"
        else:
            return "Basic alignment - consider brand-specific refinements"

    def _get_usage_recommendations(self, image_style: str, content_description: str) -> List[str]:
        """Generate usage recommendations based on style and content."""
        recommendations = []
        
        if "professional" in image_style.lower():
            recommendations.append("Suitable for business and corporate use")
        if "artistic" in image_style.lower():
            recommendations.append("Great for creative and marketing materials")
        if "social" in content_description.lower():
            recommendations.append("Optimized for social media platforms")
        if "product" in content_description.lower():
            recommendations.append("Suitable for product marketing and e-commerce")
        
        if not recommendations:
            recommendations.append("General purpose image generation")
        
        return recommendations

    def _assess_prompt_quality(self, primary_prompt: Dict[str, Any], 
                             alternative_prompts: List[Dict[str, Any]], 
                             target_platform: str) -> float:
        """Assess the quality of generated prompts."""
        quality_factors = []
        
        # Check primary prompt quality
        prompt_text = primary_prompt.get("prompt", "")
        if len(prompt_text.split()) >= 10:  # Reasonable length
            quality_factors.append(0.3)
        
        # Check for style elements
        if primary_prompt.get("style_elements"):
            quality_factors.append(0.2)
        
        # Check for technical elements
        if primary_prompt.get("technical_elements"):
            quality_factors.append(0.2)
        
        # Check alternative variations
        if len(alternative_prompts) >= 2:
            quality_factors.append(0.15)
        
        # Platform-specific optimization
        if target_platform in ["midjourney", "dall-e", "stable-diffusion"]:
            quality_factors.append(0.15)
        
        return min(sum(quality_factors), 1.0)

    def _generate_decision_reasoning(self, content_description: str, target_platform: str,
                                   primary_prompt: Dict[str, Any], quality_score: float) -> Dict[str, Any]:
        """Generate decision reasoning for business intelligence."""
        return {
            "decision_summary": f"Generated optimized {target_platform} prompts with {quality_score:.1%} quality score",
            "key_factors": [
                f"Target platform: {target_platform}",
                f"Content focus: {content_description[:100]}{'...' if len(content_description) > 100 else ''}",
                f"Prompt optimization: Platform-specific guidelines applied",
                f"Quality assessment: {quality_score:.1%}"
            ],
            "business_impact": {
                "efficiency_gain": "Automated prompt engineering reduces manual effort",
                "quality_improvement": "Platform-optimized prompts increase success rate",
                "cost_optimization": "Better prompts reduce iteration costs"
            },
            "recommendations": [
                "Test primary prompt first" if quality_score > 0.8 else "Consider manual refinement",
                "Use alternative prompts for variety" if len(primary_prompt.get("style_elements", [])) > 2 else "Generate additional variations",
                "Monitor generation results and iterate" if quality_score > 0.7 else "Refine prompt strategy"
            ],
            "confidence_score": min(quality_score * 1.1, 1.0),
            "decision_timestamp": datetime.now().isoformat()
        }