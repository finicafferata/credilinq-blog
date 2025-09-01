"""
LangGraph-based Video Prompt Agent with advanced workflow capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

@dataclass
class VideoPromptAgentState(WorkflowState):
    """State for Video Prompt Agent LangGraph workflow."""
    # Input requirements
    content: str = ""
    blog_title: str = ""
    video_type: str = "explainer"
    duration: int = 60
    style: str = "corporate"
    target_platform: str = "YouTube"
    include_text: bool = True
    voiceover: bool = True
    
    # Processing state
    narrative_structure: Dict[str, Any] = field(default_factory=dict)
    scene_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    technical_specs: Dict[str, Any] = field(default_factory=dict)
    
    # Generated outputs
    video_structure: Dict[str, Any] = field(default_factory=dict)
    scene_prompts: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    voiceover_script: Dict[str, Any] = field(default_factory=dict)
    text_overlays: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality assessment
    scene_quality_scores: Dict[str, float] = field(default_factory=dict)
    narrative_flow_score: float = 0.0
    
    # Workflow control
    requires_restructuring: bool = False
    restructure_feedback: str = ""

class VideoPromptAgentLangGraph(LangGraphWorkflowBase[VideoPromptAgentState]):
    """
    LangGraph-based Video Prompt Agent with sophisticated multi-phase workflow.
    """
    
    def __init__(self, workflow_name: str = "VideoPromptAgent_workflow"):
        super().__init__(
            workflow_name=workflow_name
        )
        
        logger.info("VideoPromptAgentLangGraph initialized with advanced workflow capabilities")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(VideoPromptAgentState)
        
        # Define workflow nodes
        workflow.add_node("analyze_content", self._analyze_content)
        workflow.add_node("design_structure", self._design_structure)
        workflow.add_node("create_scene_breakdown", self._create_scene_breakdown)
        workflow.add_node("generate_scene_prompts", self._generate_scene_prompts)
        workflow.add_node("design_transitions", self._design_transitions)
        workflow.add_node("create_voiceover", self._create_voiceover)
        workflow.add_node("design_text_overlays", self._design_text_overlays)
        workflow.add_node("quality_assessment", self._quality_assessment)
        workflow.add_node("restructure_video", self._restructure_video)
        workflow.add_node("finalize_output", self._finalize_output)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_content")
        
        workflow.add_edge("analyze_content", "design_structure")
        workflow.add_edge("design_structure", "create_scene_breakdown")
        workflow.add_edge("create_scene_breakdown", "generate_scene_prompts")
        workflow.add_edge("generate_scene_prompts", "design_transitions")
        
        # Conditional paths based on requirements
        workflow.add_conditional_edges(
            "design_transitions",
            self._should_add_voiceover,
            {
                "add_voiceover": "create_voiceover",
                "skip_voiceover": "design_text_overlays"
            }
        )
        
        workflow.add_conditional_edges(
            "create_voiceover",
            self._should_add_text_overlays,
            {
                "add_overlays": "design_text_overlays",
                "skip_overlays": "quality_assessment"
            }
        )
        
        workflow.add_edge("design_text_overlays", "quality_assessment")
        
        # Quality-based conditional routing
        workflow.add_conditional_edges(
            "quality_assessment",
            self._should_restructure,
            {
                "restructure": "restructure_video",
                "finalize": "finalize_output"
            }
        )
        
        workflow.add_edge("restructure_video", "create_scene_breakdown")
        workflow.set_finish_point("finalize_output")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> VideoPromptAgentState:
        """Create initial workflow state from input."""
        return VideoPromptAgentState(
            content=input_data.get("content", ""),
            blog_title=input_data.get("blog_title", ""),
            video_type=input_data.get("video_type", "explainer"),
            duration=input_data.get("duration", 60),
            style=input_data.get("style", "corporate"),
            target_platform=input_data.get("target_platform", "YouTube"),
            include_text=input_data.get("include_text", True),
            voiceover=input_data.get("voiceover", True),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="analyze_content"
        )
    
    def _analyze_content(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Analyze content for video narrative potential."""
        logger.info("Analyzing content for video narrative structure")
        
        try:
            # Extract key narrative elements
            narrative_elements = {
                "main_topic": state.blog_title,
                "key_points": self._extract_key_points(state.content),
                "complexity": self._assess_complexity(state.content),
                "target_audience": self._determine_audience(state.content, state.video_type),
                "call_to_action": self._extract_cta(state.content)
            }
            
            # Platform-specific analysis
            platform_requirements = self._get_platform_requirements(state.target_platform, state.duration)
            
            state.metadata["narrative_elements"] = narrative_elements
            state.metadata["platform_requirements"] = platform_requirements
            state.current_step = "design_structure"
            
            logger.info(f"Content analysis completed for {state.video_type} video")
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            state.errors.append(f"Content analysis error: {e}")
        
        return state
    
    def _design_structure(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Design the overall video structure and timing."""
        logger.info(f"Designing video structure for {state.duration}s {state.video_type} video")
        
        try:
            # Define segments based on video type and duration
            segment_templates = {
                "explainer": {
                    "short": ["hook", "problem", "solution", "cta"],
                    "medium": ["hook", "problem", "solution", "benefits", "proof", "cta"],
                    "long": ["intro", "hook", "problem", "solution", "details", "benefits", "proof", "testimonial", "cta"]
                },
                "demo": {
                    "short": ["intro", "overview", "demo", "cta"],
                    "medium": ["intro", "problem", "overview", "demo", "benefits", "cta"],
                    "long": ["intro", "problem", "overview", "demo", "features", "benefits", "comparison", "cta"]
                },
                "social": {
                    "short": ["hook", "main_point", "cta"],
                    "medium": ["hook", "problem", "solution", "cta"],
                    "long": ["hook", "problem", "solution", "benefits", "cta"]
                },
                "testimonial": {
                    "short": ["intro", "story", "results", "cta"],
                    "medium": ["intro", "background", "challenge", "solution", "results", "cta"],
                    "long": ["intro", "background", "challenge", "journey", "solution", "results", "impact", "cta"]
                }
            }
            
            # Determine duration category
            if state.duration <= 30:
                duration_category = "short"
            elif state.duration <= 90:
                duration_category = "medium"
            else:
                duration_category = "long"
            
            # Get appropriate segments
            segments = segment_templates.get(state.video_type, segment_templates["explainer"])[duration_category]
            
            # Calculate timing for each segment
            base_duration = state.duration // len(segments)
            remaining_time = state.duration % len(segments)
            
            structure = {
                "video_type": state.video_type,
                "total_duration": state.duration,
                "target_platform": state.target_platform,
                "segments": []
            }
            
            cumulative_time = 0
            for i, segment_name in enumerate(segments):
                # Distribute extra seconds to key segments
                extra_time = 1 if i < remaining_time else 0
                segment_duration = base_duration + extra_time
                
                segment = {
                    "name": segment_name,
                    "duration": segment_duration,
                    "start_time": cumulative_time,
                    "end_time": cumulative_time + segment_duration,
                    "purpose": self._get_segment_purpose(segment_name),
                    "visual_focus": self._get_visual_focus(segment_name, state.video_type),
                    "pacing": self._get_pacing(segment_name),
                    "priority": self._get_segment_priority(segment_name)
                }
                
                structure["segments"].append(segment)
                cumulative_time += segment_duration
            
            state.video_structure = structure
            state.current_step = "create_scene_breakdown"
            
            logger.info(f"Video structure designed with {len(segments)} segments")
            
        except Exception as e:
            logger.error(f"Structure design failed: {e}")
            state.errors.append(f"Structure design error: {e}")
        
        return state
    
    def _create_scene_breakdown(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Break down each segment into detailed scenes."""
        logger.info("Creating detailed scene breakdown")
        
        try:
            scenes = []
            scene_id = 1
            
            for segment in state.video_structure.get("segments", []):
                # Determine number of scenes per segment based on duration
                num_scenes = max(1, segment["duration"] // 15)  # Rough guideline: 1 scene per 15 seconds
                scene_duration = segment["duration"] // num_scenes
                remaining_duration = segment["duration"] % num_scenes
                
                for i in range(num_scenes):
                    extra_time = 1 if i < remaining_duration else 0
                    current_scene_duration = scene_duration + extra_time
                    scene_start = segment["start_time"] + (i * scene_duration)
                    
                    scene = {
                        "scene_id": scene_id,
                        "segment_name": segment["name"],
                        "duration": current_scene_duration,
                        "start_time": scene_start,
                        "end_time": scene_start + current_scene_duration,
                        "visual_focus": segment["visual_focus"],
                        "purpose": segment["purpose"],
                        "pacing": segment["pacing"],
                        "camera_suggestion": self._suggest_camera_angle(segment["name"], i, num_scenes),
                        "scene_type": self._determine_scene_type(segment["name"], i, num_scenes)
                    }
                    
                    scenes.append(scene)
                    scene_id += 1
            
            state.scene_breakdown = scenes
            state.current_step = "generate_scene_prompts"
            
            logger.info(f"Created {len(scenes)} detailed scenes")
            
        except Exception as e:
            logger.error(f"Scene breakdown failed: {e}")
            state.errors.append(f"Scene breakdown error: {e}")
        
        return state
    
    def _generate_scene_prompts(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Generate detailed prompts for each scene."""
        logger.info("Generating detailed scene prompts")
        
        try:
            # Get style-specific elements
            style_elements = self._get_style_elements(state.style)
            
            scene_prompts = []
            
            for scene in state.scene_breakdown:
                # Build comprehensive scene prompt
                prompt_components = [
                    f"Scene {scene['scene_id']} - {scene['segment_name'].title()} segment",
                    f"Duration: {scene['duration']} seconds",
                    f"Visual focus: {scene['visual_focus']}",
                    f"Setting: {style_elements['setting']}",
                    f"Visual style: {style_elements['visual_style']}",
                    f"Color palette: {style_elements['colors']}",
                    f"Lighting: {style_elements['lighting']}",
                    f"Camera: {scene['camera_suggestion']}",
                    f"Pacing: {scene['pacing']}",
                    f"Mood: {style_elements['mood']}"
                ]
                
                main_prompt = "\n".join(prompt_components)
                
                # Add platform-specific optimizations
                platform_opts = self._get_platform_optimizations(state.target_platform)
                
                scene_prompt = {
                    "scene_id": scene["scene_id"],
                    "segment_name": scene["segment_name"],
                    "duration": scene["duration"],
                    "start_time": scene["start_time"],
                    "main_prompt": main_prompt,
                    "detailed_description": self._create_detailed_description(scene, style_elements),
                    "camera_direction": scene["camera_suggestion"],
                    "visual_elements": self._get_scene_visual_elements(scene, style_elements),
                    "technical_specs": platform_opts,
                    "alternative_approaches": self._generate_scene_alternatives(scene, style_elements)
                }
                
                scene_prompts.append(scene_prompt)
            
            state.scene_prompts = scene_prompts
            state.current_step = "design_transitions"
            
            logger.info(f"Generated {len(scene_prompts)} detailed scene prompts")
            
        except Exception as e:
            logger.error(f"Scene prompt generation failed: {e}")
            state.errors.append(f"Scene prompt generation error: {e}")
        
        return state
    
    def _design_transitions(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Design transitions between scenes."""
        logger.info("Designing scene transitions")
        
        try:
            transitions = []
            style_transitions = self._get_style_transitions(state.style)
            
            for i in range(len(state.scene_prompts) - 1):
                current_scene = state.scene_prompts[i]
                next_scene = state.scene_prompts[i + 1]
                
                # Determine transition type based on context
                transition_type = self._determine_transition_type(
                    current_scene["segment_name"],
                    next_scene["segment_name"],
                    style_transitions
                )
                
                transition = {
                    "from_scene": current_scene["scene_id"],
                    "to_scene": next_scene["scene_id"],
                    "type": transition_type["name"],
                    "duration": transition_type["duration"],
                    "description": transition_type["description"],
                    "technical_specs": {
                        "ease": transition_type.get("ease", "ease-in-out"),
                        "direction": transition_type.get("direction", "forward")
                    }
                }
                
                transitions.append(transition)
            
            state.transitions = transitions
            state.current_step = "transitions_complete"
            
            logger.info(f"Designed {len(transitions)} scene transitions")
            
        except Exception as e:
            logger.error(f"Transition design failed: {e}")
            state.errors.append(f"Transition design error: {e}")
        
        return state
    
    def _should_add_voiceover(self, state: VideoPromptAgentState) -> str:
        """Determine if voiceover should be added."""
        return "add_voiceover" if state.voiceover else "skip_voiceover"
    
    def _should_add_text_overlays(self, state: VideoPromptAgentState) -> str:
        """Determine if text overlays should be added."""
        return "add_overlays" if state.include_text else "skip_overlays"
    
    def _create_voiceover(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Create voiceover script for the video."""
        logger.info("Creating voiceover script")
        
        try:
            script_sections = []
            total_words = 0
            
            # Average speaking pace: 150-160 words per minute
            words_per_second = 2.5
            
            for scene in state.scene_prompts:
                target_words = int(scene["duration"] * words_per_second)
                
                script_section = {
                    "scene_id": scene["scene_id"],
                    "segment_name": scene["segment_name"],
                    "duration": scene["duration"],
                    "target_words": target_words,
                    "suggested_script": self._generate_scene_script(scene, target_words),
                    "tone_direction": self._get_tone_direction(scene["segment_name"]),
                    "pacing_notes": f"Deliver in {scene['duration']} seconds, {scene.get('pacing', 'moderate')} pace",
                    "emphasis_points": self._identify_emphasis_points(scene["segment_name"])
                }
                
                script_sections.append(script_section)
                total_words += target_words
            
            voiceover_script = {
                "total_duration": state.duration,
                "estimated_word_count": total_words,
                "speaking_pace": f"{words_per_second} words/second",
                "overall_tone": self._get_overall_tone(state.style, state.video_type),
                "sections": script_sections,
                "production_notes": [
                    f"Target {state.style} style delivery",
                    f"Optimized for {state.target_platform}",
                    f"Total duration: {state.duration} seconds"
                ]
            }
            
            state.voiceover_script = voiceover_script
            state.current_step = "voiceover_complete"
            
            logger.info("Voiceover script created successfully")
            
        except Exception as e:
            logger.error(f"Voiceover creation failed: {e}")
            state.errors.append(f"Voiceover creation error: {e}")
        
        return state
    
    def _design_text_overlays(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Design text overlays for the video."""
        logger.info("Designing text overlays")
        
        try:
            text_overlays = []
            
            for scene in state.scene_prompts:
                overlay_text = self._generate_overlay_text(scene["segment_name"], scene["duration"])
                
                if overlay_text:
                    overlay = {
                        "scene_id": scene["scene_id"],
                        "segment_name": scene["segment_name"],
                        "text_elements": [
                            {
                                "text": overlay_text,
                                "start_time": scene["start_time"] + 0.5,  # Slight delay
                                "duration": min(scene["duration"] * 0.8, scene["duration"] - 1),
                                "position": self._get_text_position(scene["segment_name"]),
                                "style": self._get_text_style(state.style),
                                "animation": self._get_text_animation(scene["segment_name"])
                            }
                        ]
                    }
                    
                    text_overlays.append(overlay)
            
            state.text_overlays = text_overlays
            state.current_step = "text_overlays_complete"
            
            logger.info(f"Designed {len(text_overlays)} text overlays")
            
        except Exception as e:
            logger.error(f"Text overlay design failed: {e}")
            state.errors.append(f"Text overlay design error: {e}")
        
        return state
    
    def _quality_assessment(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Assess the quality of the video structure and prompts."""
        logger.info("Conducting video quality assessment")
        
        try:
            scene_scores = {}
            total_score = 0
            
            for scene in state.scene_prompts:
                score = self._assess_scene_quality(scene, state)
                scene_scores[f"scene_{scene['scene_id']}"] = score
                total_score += score["overall"]
            
            # Assess narrative flow
            narrative_flow = self._assess_narrative_flow(state)
            
            # Calculate overall quality
            avg_scene_quality = total_score / len(state.scene_prompts) if state.scene_prompts else 0
            overall_quality = (avg_scene_quality + narrative_flow) / 2
            
            state.scene_quality_scores = scene_scores
            state.narrative_flow_score = narrative_flow
            state.metadata["overall_quality"] = overall_quality
            
            # Determine if restructuring is needed
            state.requires_restructuring = overall_quality < 75.0
            if state.requires_restructuring:
                state.restructure_feedback = f"Quality score {overall_quality:.1f}% below threshold. Improving structure and scene detail."
            
            state.current_step = "quality_assessment_complete"
            
            logger.info(f"Quality assessment completed. Overall quality: {overall_quality:.1f}%")
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            state.errors.append(f"Quality assessment error: {e}")
        
        return state
    
    def _should_restructure(self, state: VideoPromptAgentState) -> str:
        """Determine if video needs restructuring."""
        if state.requires_restructuring and state.refinement_count < 2:
            logger.info("Video requires restructuring")
            return "restructure"
        else:
            logger.info("Proceeding to finalization")
            return "finalize"
    
    def _restructure_video(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Restructure video based on quality assessment."""
        logger.info("Restructuring video based on quality feedback")
        
        try:
            state.refinement_count += 1
            
            # Apply improvements based on quality scores
            for scene in state.scene_prompts:
                scene_id = f"scene_{scene['scene_id']}"
                quality_info = state.scene_quality_scores.get(scene_id, {})
                
                # Enhance based on specific weaknesses
                if quality_info.get("detail_score", 0) < 70:
                    scene["main_prompt"] += ", enhanced detail, professional quality"
                
                if quality_info.get("coherence_score", 0) < 70:
                    scene["detailed_description"] = self._enhance_scene_coherence(scene)
            
            state.current_step = "create_scene_breakdown"  # Restart from scene breakdown
            
            logger.info(f"Video restructured (iteration {state.refinement_count})")
            
        except Exception as e:
            logger.error(f"Video restructuring failed: {e}")
            state.errors.append(f"Video restructuring error: {e}")
        
        return state
    
    def _finalize_output(self, state: VideoPromptAgentState) -> VideoPromptAgentState:
        """Finalize the video prompt output."""
        logger.info("Finalizing video prompt output")
        
        try:
            # Compile comprehensive output
            final_output = {
                "video_overview": state.video_structure,
                "scenes": state.scene_prompts,
                "transitions": state.transitions,
                "technical_specifications": self._get_platform_optimizations(state.target_platform),
                "recommended_tools": self._get_recommended_tools(state.style, state.video_type),
                "production_timeline": self._estimate_production_timeline(len(state.scene_prompts))
            }
            
            if state.voiceover_script:
                final_output["voiceover_script"] = state.voiceover_script
            
            if state.text_overlays:
                final_output["text_overlays"] = state.text_overlays
            
            # Update state with final output
            state.metadata["final_output"] = final_output
            state.current_step = "completed"
            state.is_complete = True
            
            logger.info("Video prompt generation workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Output finalization failed: {e}")
            state.errors.append(f"Output finalization error: {e}")
        
        return state
    
    # Helper methods (implementation would continue with all the helper methods)
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Simple implementation - in production this would be more sophisticated
        sentences = content.split('. ')[:5]  # Take first 5 sentences as key points
        return [sentence.strip() for sentence in sentences if len(sentence) > 20]
    
    def _assess_complexity(self, content: str) -> str:
        """Assess content complexity."""
        word_count = len(content.split())
        if word_count < 200:
            return "simple"
        elif word_count < 500:
            return "medium"
        else:
            return "complex"
    
    def _determine_audience(self, content: str, video_type: str) -> str:
        """Determine target audience."""
        business_terms = ["ROI", "strategy", "efficiency", "productivity", "growth"]
        technical_terms = ["API", "algorithm", "software", "technology", "system"]
        
        content_lower = content.lower()
        
        if any(term in content_lower for term in technical_terms):
            return "technical"
        elif any(term in content_lower for term in business_terms):
            return "business"
        else:
            return "general"
    
    def _extract_cta(self, content: str) -> str:
        """Extract call-to-action from content."""
        # Simple CTA extraction
        cta_phrases = ["learn more", "contact us", "get started", "sign up", "download"]
        content_lower = content.lower()
        
        for phrase in cta_phrases:
            if phrase in content_lower:
                return phrase.title()
        
        return "Learn More"
    
    def _get_platform_requirements(self, platform: str, duration: int) -> Dict[str, Any]:
        """Get platform-specific requirements."""
        requirements = {
            "YouTube": {
                "optimal_duration": "2-10 minutes",
                "attention_span": "15 seconds hook critical",
                "format": "landscape 16:9",
                "engagement_features": ["cards", "end_screens", "chapters"]
            },
            "LinkedIn": {
                "optimal_duration": "30-90 seconds",
                "attention_span": "3 seconds hook critical",
                "format": "square 1:1 or landscape 16:9",
                "engagement_features": ["native_video", "auto_play", "captions"]
            },
            "TikTok": {
                "optimal_duration": "15-60 seconds",
                "attention_span": "1 second hook critical",
                "format": "vertical 9:16",
                "engagement_features": ["trending_sounds", "effects", "hashtags"]
            }
        }
        
        return requirements.get(platform, requirements["YouTube"])
    
    # Additional helper methods would be implemented here...
    # (Due to length constraints, showing structure rather than full implementation)
    
    def _get_segment_purpose(self, segment: str) -> str:
        """Get purpose of video segment."""
        purposes = {
            "hook": "Capture immediate attention",
            "intro": "Introduce topic and brand",
            "problem": "Present challenge or pain point",
            "solution": "Introduce solution or approach",
            "benefits": "Highlight value and advantages",
            "proof": "Provide evidence or examples",
            "cta": "Direct to specific action"
        }
        return purposes.get(segment, "Support narrative flow")
    
    def _get_visual_focus(self, segment: str, video_type: str) -> str:
        """Get visual focus for segment."""
        return f"Dynamic visuals emphasizing {segment} for {video_type} content"
    
    def _get_pacing(self, segment: str) -> str:
        """Get pacing for segment."""
        pacing_map = {
            "hook": "fast",
            "problem": "moderate",
            "solution": "steady",
            "cta": "deliberate"
        }
        return pacing_map.get(segment, "moderate")
    
    def _get_segment_priority(self, segment: str) -> str:
        """Get segment priority."""
        high_priority = ["hook", "solution", "cta"]
        return "high" if segment in high_priority else "medium"
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the LangGraph workflow."""
        try:
            logger.info("Starting VideoPromptAgent LangGraph execution")
            
            # Run the workflow
            result = self._run_workflow(context)
            
            if result and result.is_complete:
                return AgentResult(
                    success=True,
                    data={
                        "video_prompts": result.metadata.get("final_output", {}),
                        "video_structure": result.video_structure,
                        "scene_count": len(result.scene_prompts),
                        "total_duration": result.duration,
                        "style": result.style,
                        "video_type": result.video_type,
                        "target_platform": result.target_platform,
                        "workflow_metadata": {
                            "total_steps": result.step_count,
                            "restructure_iterations": result.refinement_count,
                            "narrative_flow_score": result.narrative_flow_score,
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
                    data={"partial_results": result.scene_prompts if result else []}
                )
                
        except Exception as e:
            logger.error(f"VideoPromptAgent LangGraph execution failed: {e}")
            return AgentResult(
                success=False,
                error_message=f"Execution failed: {str(e)}",
                data={}
            )