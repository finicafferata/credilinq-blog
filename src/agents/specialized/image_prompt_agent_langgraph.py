"""
LangGraph-based Image Prompt Agent with advanced workflow capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

@dataclass
class ImagePromptAgentState(WorkflowState):
    """State for Image Prompt Agent LangGraph workflow."""
    # Input requirements
    content: str = ""
    blog_title: str = ""
    style: str = "professional"
    image_type: str = "header"
    count: int = 3
    target_platform: str = "web"
    brand_guidelines: Dict[str, Any] = field(default_factory=dict)
    
    # Processing state
    extracted_themes: List[str] = field(default_factory=list)
    style_analysis: Dict[str, Any] = field(default_factory=dict)
    technical_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Generated outputs
    image_prompts: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow control
    requires_refinement: bool = False
    refinement_feedback: str = ""

class ImagePromptAgentLangGraph(LangGraphWorkflowBase[ImagePromptAgentState]):
    """
    LangGraph-based Image Prompt Agent with sophisticated multi-phase workflow.
    """
    
    def __init__(self, workflow_name: str = "ImagePromptAgent_workflow"):
        super().__init__(
            workflow_name=workflow_name
        )
        
        logger.info("ImagePromptAgentLangGraph initialized with advanced workflow capabilities")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(ImagePromptAgentState)
        
        # Define workflow nodes
        workflow.add_node("analyze_content", self._analyze_content)
        workflow.add_node("extract_themes", self._extract_themes)
        workflow.add_node("analyze_style", self._analyze_style)
        workflow.add_node("determine_requirements", self._determine_requirements)
        workflow.add_node("generate_prompts", self._generate_prompts)
        workflow.add_node("quality_assessment", self._quality_assessment)
        workflow.add_node("refine_prompts", self._refine_prompts)
        workflow.add_node("finalize_output", self._finalize_output)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_content")
        
        workflow.add_edge("analyze_content", "extract_themes")
        workflow.add_edge("extract_themes", "analyze_style")
        workflow.add_edge("analyze_style", "determine_requirements")
        workflow.add_edge("determine_requirements", "generate_prompts")
        workflow.add_edge("generate_prompts", "quality_assessment")
        
        # Conditional routing based on quality
        workflow.add_conditional_edges(
            "quality_assessment",
            self._should_refine,
            {
                "refine": "refine_prompts",
                "finalize": "finalize_output"
            }
        )
        
        workflow.add_edge("refine_prompts", "quality_assessment")
        workflow.set_finish_point("finalize_output")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> ImagePromptAgentState:
        """Create initial workflow state from input."""
        return ImagePromptAgentState(
            content=input_data.get("content", ""),
            blog_title=input_data.get("blog_title", ""),
            style=input_data.get("style", "professional"),
            image_type=input_data.get("image_type", "header"),
            count=input_data.get("count", 3),
            target_platform=input_data.get("target_platform", "web"),
            brand_guidelines=input_data.get("brand_guidelines", {}),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="analyze_content"
        )
    
    def _analyze_content(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Analyze the input content for visual concepts."""
        logger.info("Starting content analysis for image prompt generation")
        
        try:
            # Content analysis for visual elements
            visual_concepts = []
            
            if state.content:
                # Extract visual keywords
                visual_keywords = [
                    "technology", "business", "growth", "innovation", "team", "success",
                    "data", "analytics", "digital", "transformation", "strategy", "future",
                    "leadership", "collaboration", "efficiency", "automation", "AI",
                    "finance", "investment", "market", "customer", "solution", "process",
                    "workflow", "optimization", "performance", "results", "achievement"
                ]
                
                content_lower = state.content.lower()
                for keyword in visual_keywords:
                    if keyword in content_lower:
                        visual_concepts.append({
                            "concept": keyword,
                            "context": self._extract_context(content_lower, keyword),
                            "visual_potential": "high" if keyword in ["technology", "growth", "innovation"] else "medium"
                        })
            
            # Update state
            state.metadata["visual_concepts"] = visual_concepts
            state.metadata["content_length"] = len(state.content)
            state.metadata["has_technical_content"] = any(
                tech in state.content.lower() for tech in ["ai", "software", "technology", "digital"]
            )
            
            state.current_step = "extract_themes"
            
            logger.info(f"Content analysis completed: found {len(visual_concepts)} visual concepts")
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            state.errors.append(f"Content analysis error: {e}")
        
        return state
    
    def _extract_themes(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Extract visual themes from content and title."""
        logger.info("Extracting visual themes")
        
        try:
            themes = []
            
            # Primary theme from title
            if state.blog_title:
                themes.append({
                    "theme": state.blog_title,
                    "type": "primary",
                    "visual_weight": 1.0,
                    "description": f"Main visual representation of: {state.blog_title}"
                })
            
            # Secondary themes from visual concepts
            visual_concepts = state.metadata.get("visual_concepts", [])
            for concept in visual_concepts[:4]:  # Limit to top 4
                themes.append({
                    "theme": f"{concept['concept'].title()} visualization",
                    "type": "secondary",
                    "visual_weight": 0.7 if concept["visual_potential"] == "high" else 0.5,
                    "description": f"Visual metaphor for {concept['concept']} in context of {concept['context'][:50]}..."
                })
            
            # Industry-specific themes
            if state.metadata.get("has_technical_content"):
                themes.append({
                    "theme": "Technology innovation concept",
                    "type": "contextual",
                    "visual_weight": 0.8,
                    "description": "Modern technology and innovation visual elements"
                })
            
            state.extracted_themes = [theme["theme"] for theme in themes]
            state.metadata["theme_details"] = themes
            state.current_step = "analyze_style"
            
            logger.info(f"Extracted {len(themes)} visual themes")
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            state.errors.append(f"Theme extraction error: {e}")
        
        return state
    
    def _analyze_style(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Analyze and expand style requirements."""
        logger.info(f"Analyzing style requirements: {state.style}")
        
        try:
            # Style definitions with comprehensive elements
            style_definitions = {
                "professional": {
                    "visual_elements": "clean lines, corporate colors, minimal design",
                    "color_palette": "blue, white, grey tones, professional color scheme",
                    "composition": "balanced layout, clear hierarchy, organized structure",
                    "mood": "trustworthy, reliable, competent",
                    "lighting": "even lighting, professional photography style",
                    "typography_style": "sans-serif, clean, readable fonts"
                },
                "creative": {
                    "visual_elements": "dynamic shapes, artistic elements, creative compositions",
                    "color_palette": "vibrant colors, bold contrasts, artistic color schemes",
                    "composition": "asymmetrical layouts, creative angles, artistic flow",
                    "mood": "innovative, inspiring, energetic",
                    "lighting": "dramatic lighting, creative shadows, artistic effects",
                    "typography_style": "creative fonts, artistic typography, unique styles"
                },
                "modern": {
                    "visual_elements": "geometric shapes, minimalist design, contemporary elements",
                    "color_palette": "monochromatic schemes, accent colors, modern palettes",
                    "composition": "clean layouts, negative space, geometric arrangements",
                    "mood": "sophisticated, current, forward-thinking",
                    "lighting": "clean lighting, subtle shadows, modern aesthetics",
                    "typography_style": "modern sans-serif, geometric fonts, clean typography"
                },
                "elegant": {
                    "visual_elements": "refined details, luxury elements, sophisticated design",
                    "color_palette": "muted tones, gold accents, premium color schemes",
                    "composition": "classical proportions, elegant layouts, refined balance",
                    "mood": "sophisticated, premium, refined",
                    "lighting": "soft lighting, elegant shadows, luxury photography style",
                    "typography_style": "serif fonts, elegant typography, premium styling"
                }
            }
            
            current_style = style_definitions.get(state.style, style_definitions["professional"])
            
            # Add brand guidelines integration
            if state.brand_guidelines:
                if "colors" in state.brand_guidelines:
                    current_style["color_palette"] = f"Brand colors: {state.brand_guidelines['colors']}, {current_style['color_palette']}"
                if "style_notes" in state.brand_guidelines:
                    current_style["additional_notes"] = state.brand_guidelines["style_notes"]
            
            state.style_analysis = current_style
            state.current_step = "determine_requirements"
            
            logger.info("Style analysis completed")
            
        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            state.errors.append(f"Style analysis error: {e}")
        
        return state
    
    def _determine_requirements(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Determine technical requirements based on platform and image type."""
        logger.info(f"Determining requirements for {state.target_platform} - {state.image_type}")
        
        try:
            # Platform-specific requirements
            platform_specs = {
                "web": {
                    "header": {"resolution": "1200x600", "aspect_ratio": "2:1", "optimization": "web-optimized"},
                    "section": {"resolution": "800x600", "aspect_ratio": "4:3", "optimization": "fast-loading"},
                    "infographic": {"resolution": "1024x1024", "aspect_ratio": "1:1", "optimization": "high-detail"}
                },
                "social": {
                    "header": {"resolution": "1080x1080", "aspect_ratio": "1:1", "optimization": "engagement-focused"},
                    "section": {"resolution": "1080x1350", "aspect_ratio": "4:5", "optimization": "mobile-first"},
                    "infographic": {"resolution": "1080x1080", "aspect_ratio": "1:1", "optimization": "shareable"}
                },
                "print": {
                    "header": {"resolution": "3000x1800", "aspect_ratio": "5:3", "optimization": "high-resolution"},
                    "section": {"resolution": "2400x1800", "aspect_ratio": "4:3", "optimization": "print-quality"},
                    "infographic": {"resolution": "3600x3600", "aspect_ratio": "1:1", "optimization": "professional-print"}
                }
            }
            
            requirements = platform_specs.get(state.target_platform, {}).get(
                state.image_type, 
                {"resolution": "1024x1024", "aspect_ratio": "1:1", "optimization": "standard"}
            )
            
            # Add image type specific requirements
            type_specific = {
                "header": {"prominence": "high", "text_space": "required", "focal_point": "center-left"},
                "section": {"prominence": "medium", "text_space": "optional", "focal_point": "center"},
                "infographic": {"prominence": "high", "text_space": "integrated", "focal_point": "structured"},
                "social": {"prominence": "high", "text_space": "minimal", "focal_point": "center"},
                "background": {"prominence": "low", "text_space": "full", "focal_point": "subtle"}
            }
            
            requirements.update(type_specific.get(state.image_type, type_specific["header"]))
            
            state.technical_requirements = requirements
            state.current_step = "generate_prompts"
            
            logger.info("Technical requirements determined")
            
        except Exception as e:
            logger.error(f"Requirements determination failed: {e}")
            state.errors.append(f"Requirements determination error: {e}")
        
        return state
    
    def _generate_prompts(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Generate detailed image prompts."""
        logger.info(f"Generating {state.count} image prompts")
        
        try:
            prompts = []
            
            for i in range(state.count):
                # Select theme
                theme_idx = i % len(state.extracted_themes) if state.extracted_themes else 0
                theme = state.extracted_themes[theme_idx] if state.extracted_themes else state.blog_title
                
                # Build comprehensive prompt
                prompt_components = [
                    f"Subject: {theme}",
                    f"Style: {state.style_analysis.get('visual_elements', '')}",
                    f"Colors: {state.style_analysis.get('color_palette', '')}",
                    f"Composition: {state.style_analysis.get('composition', '')}",
                    f"Mood: {state.style_analysis.get('mood', '')}",
                    f"Lighting: {state.style_analysis.get('lighting', '')}",
                    f"Resolution: {state.technical_requirements.get('resolution', '1024x1024')}",
                    f"Aspect ratio: {state.technical_requirements.get('aspect_ratio', '1:1')}",
                    f"Optimization: {state.technical_requirements.get('optimization', 'standard')}"
                ]
                
                # Add brand guidelines
                if "additional_notes" in state.style_analysis:
                    prompt_components.append(f"Brand guidelines: {state.style_analysis['additional_notes']}")
                
                final_prompt = ", ".join(filter(None, prompt_components))
                
                # Generate alternatives
                alternatives = [
                    f"{theme}, {state.style} aesthetic, alternative composition",
                    f"{theme}, {state.style} style, different perspective",
                    f"{theme}, creative {state.style} interpretation"
                ]
                
                prompt_data = {
                    "id": f"img_prompt_{i+1}",
                    "prompt": final_prompt,
                    "theme": theme,
                    "style": state.style,
                    "image_type": state.image_type,
                    "technical_specs": state.technical_requirements,
                    "estimated_tokens": len(final_prompt.split()),
                    "complexity": "high" if len(prompt_components) > 8 else "medium",
                    "alternatives": alternatives[:2],
                    "recommended_tools": self._get_recommended_tools(state.style, state.image_type)
                }
                
                prompts.append(prompt_data)
            
            state.image_prompts = prompts
            state.current_step = "quality_assessment"
            
            logger.info(f"Generated {len(prompts)} detailed image prompts")
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            state.errors.append(f"Prompt generation error: {e}")
        
        return state
    
    def _quality_assessment(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Assess quality of generated prompts."""
        logger.info("Conducting quality assessment")
        
        try:
            quality_scores = {}
            overall_quality = 0.0
            
            for prompt in state.image_prompts:
                score = 0.0
                feedback = []
                
                # Prompt completeness (0-30 points)
                prompt_text = prompt.get("prompt", "")
                if len(prompt_text) > 100:
                    score += 30
                elif len(prompt_text) > 50:
                    score += 20
                else:
                    score += 10
                    feedback.append("Prompt could be more detailed")
                
                # Style consistency (0-25 points)
                if state.style in prompt_text.lower():
                    score += 25
                else:
                    score += 10
                    feedback.append("Style integration could be improved")
                
                # Technical specifications (0-20 points)
                if prompt.get("technical_specs"):
                    score += 20
                else:
                    score += 5
                    feedback.append("Missing technical specifications")
                
                # Theme relevance (0-15 points)
                theme = prompt.get("theme", "")
                if theme and len(theme) > 5:
                    score += 15
                else:
                    score += 5
                    feedback.append("Theme could be more specific")
                
                # Alternative options (0-10 points)
                alternatives = prompt.get("alternatives", [])
                if len(alternatives) >= 2:
                    score += 10
                else:
                    score += 5
                    feedback.append("More alternatives would be beneficial")
                
                quality_scores[prompt["id"]] = {
                    "score": score,
                    "percentage": (score / 100) * 100,
                    "feedback": feedback
                }
                
                overall_quality += score
            
            # Calculate overall quality
            overall_quality = overall_quality / len(state.image_prompts) if state.image_prompts else 0
            
            state.quality_scores = quality_scores
            state.metadata["overall_quality"] = overall_quality
            
            # Determine if refinement is needed
            state.requires_refinement = overall_quality < 75.0
            if state.requires_refinement:
                state.refinement_feedback = "Overall quality below threshold. Improving prompt detail and specificity."
            
            state.current_step = "quality_assessment_complete"
            
            logger.info(f"Quality assessment completed. Overall quality: {overall_quality:.1f}%")
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            state.errors.append(f"Quality assessment error: {e}")
        
        return state
    
    def _should_refine(self, state: ImagePromptAgentState) -> str:
        """Determine if prompts need refinement."""
        if state.requires_refinement and state.refinement_count < 2:
            logger.info("Prompts require refinement")
            return "refine"
        else:
            logger.info("Proceeding to finalization")
            return "finalize"
    
    def _refine_prompts(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Refine prompts based on quality assessment."""
        logger.info("Refining prompts based on quality feedback")
        
        try:
            state.refinement_count += 1
            
            for prompt in state.image_prompts:
                prompt_id = prompt["id"]
                quality_info = state.quality_scores.get(prompt_id, {})
                feedback = quality_info.get("feedback", [])
                
                # Apply refinements based on feedback
                if "more detailed" in " ".join(feedback).lower():
                    # Enhance prompt detail
                    enhanced_details = [
                        "high-quality photography",
                        "professional composition",
                        "studio lighting",
                        "crisp details",
                        "modern aesthetic"
                    ]
                    prompt["prompt"] += ", " + ", ".join(enhanced_details[:2])
                
                if "style integration" in " ".join(feedback).lower():
                    # Reinforce style elements
                    style_elements = state.style_analysis.get('visual_elements', '')
                    if style_elements and style_elements not in prompt["prompt"]:
                        prompt["prompt"] = f"{style_elements}, {prompt['prompt']}"
                
                if "technical specifications" in " ".join(feedback).lower():
                    # Add missing technical specs
                    if not prompt.get("technical_specs"):
                        prompt["technical_specs"] = state.technical_requirements
            
            state.current_step = "quality_assessment"  # Re-assess after refinement
            
            logger.info(f"Prompts refined (iteration {state.refinement_count})")
            
        except Exception as e:
            logger.error(f"Prompt refinement failed: {e}")
            state.errors.append(f"Prompt refinement error: {e}")
        
        return state
    
    def _finalize_output(self, state: ImagePromptAgentState) -> ImagePromptAgentState:
        """Finalize the output with recommendations and metadata."""
        logger.info("Finalizing image prompt output")
        
        try:
            # Generate recommendations
            recommendations = {
                "recommended_tools": self._get_recommended_tools(state.style, state.image_type),
                "technical_specs": state.technical_requirements,
                "style_guide": state.style_analysis,
                "usage_tips": [
                    f"Best suited for {state.target_platform} platform",
                    f"Optimized for {state.image_type} image type",
                    f"Style: {state.style} - {state.style_analysis.get('mood', 'professional')}"
                ]
            }
            
            if state.brand_guidelines:
                recommendations["brand_integration"] = "Brand guidelines have been integrated into prompts"
            
            state.recommendations = recommendations
            state.current_step = "completed"
            state.is_complete = True
            
            logger.info("Image prompt generation workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Output finalization failed: {e}")
            state.errors.append(f"Output finalization error: {e}")
        
        return state
    
    def _get_recommended_tools(self, style: str, image_type: str) -> List[str]:
        """Get recommended tools for image generation."""
        tools = {
            "professional": ["DALL-E 3", "Midjourney", "Adobe Firefly"],
            "creative": ["Midjourney", "Stable Diffusion", "Leonardo AI"],
            "modern": ["DALL-E 3", "Midjourney", "RunwayML"],
            "elegant": ["Midjourney", "Adobe Firefly", "DALL-E 3"]
        }
        
        return tools.get(style, ["DALL-E 3", "Midjourney"])
    
    def _extract_context(self, content: str, keyword: str) -> str:
        """Extract context around a keyword from content."""
        try:
            keyword_pos = content.find(keyword)
            if keyword_pos == -1:
                return ""
            
            start = max(0, keyword_pos - 50)
            end = min(len(content), keyword_pos + 50)
            context = content[start:end].strip()
            
            return context if context else keyword
        except:
            return keyword
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the LangGraph workflow."""
        try:
            logger.info("Starting ImagePromptAgent LangGraph execution")
            
            # Run the workflow
            result = self._run_workflow(context)
            
            if result and result.is_complete:
                return AgentResult(
                    success=True,
                    data={
                        "image_prompts": result.image_prompts,
                        "style": result.style,
                        "image_type": result.image_type,
                        "target_platform": result.target_platform,
                        "count": len(result.image_prompts),
                        "recommendations": result.recommendations,
                        "quality_scores": result.quality_scores,
                        "workflow_metadata": {
                            "total_steps": result.step_count,
                            "refinement_iterations": result.refinement_count,
                            "overall_quality": result.metadata.get("overall_quality", 0)
                        }
                    }
                )
            else:
                error_msg = "Workflow did not complete successfully"
                if result and result.errors:
                    error_msg += f": {'; '.join(result.errors)}"
                
                return AgentResult(
                    success=False,
                    error_message=error_msg,
                    data={"partial_results": result.image_prompts if result else []}
                )
                
        except Exception as e:
            logger.error(f"ImagePromptAgent LangGraph execution failed: {e}")
            return AgentResult(
                success=False,
                error_message=f"Execution failed: {str(e)}",
                data={}
            )