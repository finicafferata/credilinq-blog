"""
Real ContentRepurposer Implementation - LLM-powered content repurposing agent.

This agent repurposes existing content into different formats while maintaining
quality, brand voice, and key messaging across multiple channels.
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


class RealContentRepurposerAgent(BaseAgent):
    """
    Real LLM-powered content repurposer for multi-format content adaptation.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_REPURPOSER,
                name="RealContentRepurposerAgent",
                description="LLM-powered content repurposing and format adaptation",
                version="1.0.0",
                capabilities=[
                    "multi_format_adaptation",
                    "social_media_optimization", 
                    "content_summarization",
                    "audience_targeting",
                    "brand_voice_preservation"
                ],
                dependencies=["llm_client", "input_validator"],
                performance_targets={
                    "response_time": 5.0,
                    "quality_score": 0.85
                }
            )
        super().__init__(metadata)
        self.llm_client = LLMClient()
        self.input_validator = InputValidator()

    async def execute(self, input_data: Dict[str, Any], context: Optional[AgentExecutionContext] = None) -> AgentResult:
        """
        Execute content repurposing with LLM-powered adaptation.
        
        Args:
            input_data: Must contain:
                - original_content: Source content to repurpose
                - target_formats: List of desired output formats
                - target_audience: Target audience for adaptation
                - brand_voice: Brand voice guidelines (optional)
        """
        start_time = datetime.now()
        
        try:
            # Input validation and security check
            self._validate_input(input_data)
            
            # Extract required parameters
            original_content = input_data["original_content"]
            target_formats = input_data["target_formats"]
            target_audience = input_data.get("target_audience", "general audience")
            brand_voice = input_data.get("brand_voice", "professional and engaging")
            
            # Performance tracking
            if context and hasattr(context, 'performance_tracker'):
                await context.performance_tracker.start_agent_execution(self.metadata.name, input_data)
            
            # Execute repurposing for each target format
            repurposed_content = {}
            total_tokens = 0
            
            for format_type in target_formats:
                logger.info(f"Repurposing content for format: {format_type}")
                
                format_result = await self._repurpose_for_format(
                    original_content, format_type, target_audience, brand_voice
                )
                
                repurposed_content[format_type] = format_result
                total_tokens += format_result.get("tokens_used", 0)
            
            # Calculate execution time and quality
            execution_time = (datetime.now() - start_time).total_seconds()
            quality_score = self._assess_repurposing_quality(original_content, repurposed_content)
            
            # Compile comprehensive result
            result_data = {
                "repurposed_content": repurposed_content,
                "original_content_summary": {
                    "word_count": len(original_content.split()),
                    "format": input_data.get("original_format", "unknown")
                },
                "repurposing_summary": {
                    "formats_created": len(target_formats),
                    "total_adaptations": len(repurposed_content),
                    "formats": list(target_formats)
                },
                "performance_metrics": {
                    "execution_time": execution_time,
                    "tokens_used": total_tokens,
                    "quality_score": quality_score,
                    "cost_estimate": total_tokens * 0.000002  # Rough cost estimate
                },
                "adaptation_metadata": {
                    "target_audience": target_audience,
                    "brand_voice": brand_voice,
                    "created_at": datetime.now().isoformat(),
                    "agent_version": self.metadata.version
                }
            }
            
            # Decision reasoning for business intelligence
            decision_reasoning = self._generate_decision_reasoning(
                original_content, target_formats, repurposed_content, quality_score
            )
            result_data["decision_reasoning"] = decision_reasoning
            
            # Performance tracking completion
            if context and hasattr(context, 'performance_tracker'):
                await context.performance_tracker.complete_agent_execution(
                    self.metadata.name, True, execution_time, quality_score
                )
            
            logger.info(f"Content repurposing completed in {execution_time:.2f}s with quality score {quality_score:.3f}")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "execution_time": execution_time,
                    "quality_score": quality_score,
                    "tokens_used": total_tokens
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
            logger.error(f"Content repurposing failed: {e}")
            
            if context and hasattr(context, 'performance_tracker'):
                await context.performance_tracker.complete_agent_execution(
                    self.metadata.name, False, execution_time, 0.0
                )
            
            return AgentResult(
                success=False,
                error=f"Content repurposing failed: {str(e)}",
                agent_id=self.metadata.name,
                timestamp=datetime.now(),
                metadata={"execution_time": execution_time}
            )

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for content repurposing."""
        # Security validation
        self.input_validator.validate_input(input_data)
        
        # Business logic validation
        required_fields = ["original_content", "target_formats"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate content length
        if len(input_data["original_content"]) < 50:
            raise ValueError("Original content too short for effective repurposing (minimum 50 characters)")
        
        if len(input_data["original_content"]) > 10000:
            raise ValueError("Original content too long (maximum 10,000 characters)")
        
        # Validate target formats
        supported_formats = [
            "social_media_post", "linkedin_post", "twitter_thread", "instagram_caption",
            "email_newsletter", "blog_summary", "executive_summary", "bullet_points",
            "infographic_text", "video_script", "podcast_outline", "press_release"
        ]
        
        for format_type in input_data["target_formats"]:
            if format_type not in supported_formats:
                raise ValueError(f"Unsupported format type: {format_type}. Supported formats: {supported_formats}")

    async def _repurpose_for_format(self, original_content: str, format_type: str, 
                                  target_audience: str, brand_voice: str) -> Dict[str, Any]:
        """Repurpose content for a specific format using LLM."""
        
        # Format-specific prompts and constraints
        format_configs = {
            "social_media_post": {
                "max_length": 300,
                "style": "engaging and concise",
                "include_hashtags": True,
                "call_to_action": True
            },
            "linkedin_post": {
                "max_length": 1300,
                "style": "professional and insightful",
                "include_hashtags": True,
                "professional_tone": True
            },
            "twitter_thread": {
                "max_length": 280,
                "style": "conversational and informative",
                "thread_format": True,
                "numbered_posts": True
            },
            "email_newsletter": {
                "max_length": 800,
                "style": "informative and personal",
                "subject_line": True,
                "clear_structure": True
            },
            "executive_summary": {
                "max_length": 500,
                "style": "concise and strategic",
                "bullet_points": True,
                "key_insights": True
            },
            "blog_summary": {
                "max_length": 400,
                "style": "informative and engaging",
                "key_points": True,
                "conclusion": True
            }
        }
        
        config = format_configs.get(format_type, {
            "max_length": 500,
            "style": "clear and engaging",
            "adaptive": True
        })
        
        # Construct format-specific prompt
        system_prompt = f"""You are an expert content repurposer specializing in {format_type} adaptation.

Your task is to repurpose the given content while:
1. Maintaining the core message and key insights
2. Adapting to {format_type} format requirements
3. Optimizing for {target_audience}
4. Preserving {brand_voice} brand voice
5. Following {config['style']} writing style

Format Requirements:
- Maximum length: {config['max_length']} characters
- Style: {config['style']}
- Target audience: {target_audience}
- Brand voice: {brand_voice}

Additional requirements: {json.dumps({k: v for k, v in config.items() if k not in ['max_length', 'style']}, indent=2)}

Provide the repurposed content in valid JSON format with these fields:
{{
    "content": "The repurposed content text",
    "title": "Appropriate title for the format",
    "key_points": ["key point 1", "key point 2"],
    "hashtags": ["#tag1", "#tag2"] (if applicable),
    "call_to_action": "Suggested call-to-action",
    "character_count": actual_character_count,
    "adaptation_notes": "Brief notes on adaptation choices"
}}"""
        
        human_prompt = f"""Original content to repurpose:

{original_content}

Please repurpose this content for {format_type} following the requirements above."""
        
        try:
            # Execute LLM call
            response = await self.llm_client.generate_response([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "content": response.content,
                    "title": f"Repurposed for {format_type}",
                    "adaptation_notes": "JSON parsing failed, returning raw content",
                    "character_count": len(response.content)
                }
            
            # Add metadata
            result["tokens_used"] = getattr(response, 'usage', {}).get('total_tokens', 100)
            result["format_type"] = format_type
            result["created_at"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to repurpose content for {format_type}: {e}")
            return {
                "content": f"Error repurposing for {format_type}: {str(e)}",
                "error": str(e),
                "format_type": format_type,
                "success": False
            }

    def _assess_repurposing_quality(self, original_content: str, 
                                   repurposed_content: Dict[str, Any]) -> float:
        """Assess the quality of content repurposing."""
        quality_factors = []
        
        # Check adaptation coverage
        successful_adaptations = sum(1 for content in repurposed_content.values() 
                                   if not content.get("error"))
        total_adaptations = len(repurposed_content)
        adaptation_success_rate = successful_adaptations / total_adaptations if total_adaptations > 0 else 0
        quality_factors.append(adaptation_success_rate * 0.4)
        
        # Check content length appropriateness
        appropriate_lengths = 0
        for format_type, content in repurposed_content.items():
            if not content.get("error"):
                content_length = content.get("character_count", len(content.get("content", "")))
                # Simple heuristic for appropriate length
                if 50 <= content_length <= 2000:  # Reasonable range for most formats
                    appropriate_lengths += 1
        
        length_quality = appropriate_lengths / total_adaptations if total_adaptations > 0 else 0
        quality_factors.append(length_quality * 0.3)
        
        # Check for key content preservation (simple word overlap check)
        original_words = set(original_content.lower().split())
        if len(original_words) > 0:
            overlap_scores = []
            for content in repurposed_content.values():
                if not content.get("error"):
                    repurposed_text = content.get("content", "")
                    repurposed_words = set(repurposed_text.lower().split())
                    overlap = len(original_words.intersection(repurposed_words)) / len(original_words)
                    overlap_scores.append(min(overlap * 2, 1.0))  # Scale overlap score
            
            avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
            quality_factors.append(avg_overlap * 0.3)
        else:
            quality_factors.append(0.0)
        
        return min(sum(quality_factors), 1.0)

    def _generate_decision_reasoning(self, original_content: str, target_formats: List[str],
                                   repurposed_content: Dict[str, Any], quality_score: float) -> Dict[str, Any]:
        """Generate decision reasoning for business intelligence."""
        return {
            "decision_summary": f"Repurposed content into {len(target_formats)} formats with {quality_score:.1%} quality",
            "key_factors": [
                f"Original content length: {len(original_content)} characters",
                f"Target formats: {', '.join(target_formats)}",
                f"Successful adaptations: {sum(1 for c in repurposed_content.values() if not c.get('error'))}/{len(repurposed_content)}",
                f"Quality assessment: {quality_score:.1%}"
            ],
            "business_impact": {
                "content_multiplier": len(target_formats),
                "channel_reach": f"Content adapted for {len(target_formats)} different channels",
                "efficiency_gain": f"Single source content expanded to {len(target_formats)} formats"
            },
            "recommendations": [
                "Review adapted content for brand consistency" if quality_score < 0.8 else "Content quality meets standards",
                "Consider A/B testing different format variations" if len(target_formats) > 2 else "Single format focus maintained",
                "Monitor engagement metrics across formats" if quality_score > 0.8 else "Improve content quality before distribution"
            ],
            "confidence_score": min(quality_score * 1.2, 1.0),
            "decision_timestamp": datetime.now().isoformat()
        }