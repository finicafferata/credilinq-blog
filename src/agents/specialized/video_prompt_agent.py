"""
Video Prompt Agent - Generates detailed prompts for video generation services
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata, AgentExecutionContext
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

class VideoPromptAgent(BaseAgent):
    """
    Agent specialized in generating comprehensive prompts for video generation services.
    Creates detailed scene descriptions, transitions, and specifications for video AI tools.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None, name: str = "VideoPromptAgent", description: str = "Generates comprehensive prompts for video generation services"):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.VIDEO_PROMPT,
                name=name,
                description=description
            )
        super().__init__(metadata=metadata)
        self.agent_type = AgentType.VIDEO_PROMPT
        
    def execute(self, input_data: Dict[str, Any], context: Optional[AgentExecutionContext] = None, **kwargs) -> AgentResult:
        """
        Generate detailed video prompts with scenes, transitions, and specifications.
        
        Args:
            input_data: Dictionary containing:
                - content: Main content to visualize
                - blog_title: Title of the content
                - video_type: Type of video (explainer, demo, social, testimonial)
                - duration: Target duration in seconds (default: 60)
                - style: Visual style (corporate, creative, documentary, animated)
                - target_platform: Platform (YouTube, LinkedIn, TikTok, Instagram)
                - include_text: Whether to include text overlays
                - voiceover: Whether video should include voiceover guidance
            context: Execution context
        
        Returns:
            AgentResult with comprehensive video prompts and specifications
        """
        try:
            logger.info(f"VideoPromptAgent executing for: {input_data.get('blog_title', 'Unknown')}")
            
            content = input_data.get('content', '')
            blog_title = input_data.get('blog_title', '')
            video_type = input_data.get('video_type', 'explainer')
            duration = input_data.get('duration', 60)
            style = input_data.get('style', 'corporate')
            target_platform = input_data.get('target_platform', 'YouTube')
            include_text = input_data.get('include_text', True)
            voiceover = input_data.get('voiceover', True)
            
            if not content and not blog_title:
                raise AgentExecutionError("No content or title provided for video prompt generation")
            
            # Generate comprehensive video prompt structure
            video_structure = self._create_video_structure(
                content=content,
                blog_title=blog_title,
                video_type=video_type,
                duration=duration,
                style=style,
                target_platform=target_platform
            )
            
            # Generate scene-by-scene prompts
            scenes = self._generate_scene_prompts(video_structure, style, include_text)
            
            # Generate transitions
            transitions = self._generate_transitions(len(scenes), style)
            
            # Generate technical specifications
            tech_specs = self._get_video_specifications(target_platform, duration)
            
            # Create individual video prompt objects for content integration
            video_prompts_list = [{
                "prompt": f"Create a {video_type} video about {blog_title}: {video_structure.get('main_concept', '')}",
                "duration": f"{duration} seconds",
                "video_type": video_type,
                "scenes": scenes,
                "transitions": transitions,
                "style": style,
                "target_platform": target_platform,
                "voiceover_script": self._generate_voiceover_script(scenes) if voiceover else None,
                "text_overlays": self._generate_text_overlays(scenes) if include_text else None
            }]
            
            result_data = {
                "prompts": video_prompts_list,  # For workflow integration
                "video_prompts": {  # Keep detailed structure for backward compatibility
                    "overview": video_structure,
                    "scenes": scenes,
                    "transitions": transitions,
                    "technical_specs": tech_specs
                },
                "style": style,
                "video_type": video_type,
                "duration": duration,
                "target_platform": target_platform,
                "recommended_tools": self._get_recommended_video_tools(style, video_type),
                "voiceover_script": self._generate_voiceover_script(scenes) if voiceover else None,
                "text_overlays": self._generate_text_overlays(scenes) if include_text else None
            }
            
            logger.info(f"VideoPromptAgent completed successfully. Generated {len(scenes)} scenes.")
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"VideoPromptAgent execution failed: {str(e)}")
            raise AgentExecutionError("VideoPromptAgent", "execution", str(e))
    
    def _create_video_structure(self, content: str, blog_title: str, video_type: str, 
                              duration: int, style: str, target_platform: str) -> Dict[str, Any]:
        """
        Create overall video structure and narrative flow.
        """
        # Define video segments based on duration and type
        if duration <= 30:
            segments = ["hook", "main_point", "call_to_action"]
        elif duration <= 60:
            segments = ["hook", "problem", "solution", "benefits", "call_to_action"]
        else:
            segments = ["intro", "hook", "problem", "solution", "details", "benefits", "proof", "call_to_action"]
        
        # Calculate timing for each segment
        segment_duration = duration // len(segments)
        remaining_time = duration % len(segments)
        
        structure = {
            "title": blog_title,
            "total_duration": duration,
            "video_type": video_type,
            "style": style,
            "target_platform": target_platform,
            "segments": []
        }
        
        for i, segment in enumerate(segments):
            # Add extra seconds to the main segments
            extra_time = 1 if i < remaining_time else 0
            segment_time = segment_duration + extra_time
            
            structure["segments"].append({
                "name": segment,
                "duration": segment_time,
                "start_time": sum(s["duration"] for s in structure["segments"]),
                "purpose": self._get_segment_purpose(segment),
                "visual_focus": self._get_visual_focus(segment, video_type)
            })
        
        return structure
    
    def _generate_scene_prompts(self, video_structure: Dict, style: str, include_text: bool) -> List[Dict[str, Any]]:
        """
        Generate detailed prompts for each scene in the video.
        """
        scenes = []
        
        # Style-specific visual elements
        style_elements = {
            "corporate": {
                "setting": "modern office environment, professional lighting",
                "colors": "blue and white color scheme, clean aesthetics",
                "props": "laptops, charts, professional attire"
            },
            "creative": {
                "setting": "dynamic studio space, creative lighting",
                "colors": "vibrant color palette, artistic elements",
                "props": "design tools, creative materials, artistic backgrounds"
            },
            "documentary": {
                "setting": "natural environments, authentic locations",
                "colors": "natural color grading, realistic lighting",
                "props": "real-world objects, authentic settings"
            },
            "animated": {
                "setting": "animated environments, stylized backgrounds",
                "colors": "brand-consistent color scheme, animated elements",
                "props": "animated characters, motion graphics, icons"
            }
        }
        
        current_style = style_elements.get(style, style_elements["corporate"])
        
        for i, segment in enumerate(video_structure["segments"]):
            scene_prompt = {
                "scene_id": i + 1,
                "segment_name": segment["name"],
                "duration": segment["duration"],
                "start_time": segment["start_time"],
                "main_prompt": self._create_scene_prompt(segment, current_style),
                "camera_direction": self._get_camera_direction(segment["name"]),
                "visual_elements": self._get_visual_elements(segment["name"], current_style),
                "pacing": self._get_pacing_direction(segment["name"]),
                "mood": self._get_mood_direction(segment["name"], style)
            }
            
            if include_text:
                scene_prompt["text_overlay"] = self._generate_scene_text(segment)
            
            scenes.append(scene_prompt)
        
        return scenes
    
    def _create_scene_prompt(self, segment: Dict, style_elements: Dict) -> str:
        """
        Create a detailed prompt for a specific scene.
        """
        base_prompt = f"""
        Scene for {segment['name']} segment ({segment['duration']} seconds):
        Setting: {style_elements['setting']}
        Visual style: {style_elements['colors']}
        Props/Elements: {style_elements['props']}
        Purpose: {segment['purpose']}
        Visual focus: {segment['visual_focus']}
        """
        
        return base_prompt.strip()
    
    def _generate_transitions(self, num_scenes: int, style: str) -> List[Dict[str, str]]:
        """
        Generate transition suggestions between scenes.
        """
        transitions = []
        
        transition_types = {
            "corporate": ["smooth fade", "slide transition", "clean cut"],
            "creative": ["dynamic wipe", "creative dissolve", "zoom transition"],
            "documentary": ["natural cut", "fade to black", "cross dissolve"],
            "animated": ["animated transition", "morph effect", "stylized wipe"]
        }
        
        available_transitions = transition_types.get(style, transition_types["corporate"])
        
        for i in range(num_scenes - 1):
            transition = {
                "from_scene": i + 1,
                "to_scene": i + 2,
                "type": available_transitions[i % len(available_transitions)],
                "duration": "0.5-1.0 seconds",
                "description": f"Transition from scene {i + 1} to scene {i + 2}"
            }
            transitions.append(transition)
        
        return transitions
    
    def _get_video_specifications(self, target_platform: str, duration: int) -> Dict[str, Any]:
        """
        Get technical specifications for different video platforms.
        """
        specs = {
            "YouTube": {
                "resolution": "1920x1080",
                "aspect_ratio": "16:9",
                "framerate": "30fps",
                "format": "MP4",
                "max_file_size": "128GB",
                "recommended_bitrate": "8-12 Mbps"
            },
            "LinkedIn": {
                "resolution": "1920x1080",
                "aspect_ratio": "16:9 or 1:1",
                "framerate": "30fps",
                "format": "MP4",
                "max_file_size": "5GB",
                "recommended_bitrate": "5-8 Mbps"
            },
            "TikTok": {
                "resolution": "1080x1920",
                "aspect_ratio": "9:16",
                "framerate": "30fps",
                "format": "MP4",
                "max_file_size": "287.6MB",
                "recommended_bitrate": "3-5 Mbps"
            },
            "Instagram": {
                "resolution": "1080x1080",
                "aspect_ratio": "1:1",
                "framerate": "30fps",
                "format": "MP4",
                "max_file_size": "250MB",
                "recommended_bitrate": "3-5 Mbps"
            }
        }
        
        return specs.get(target_platform, specs["YouTube"])
    
    def _get_recommended_video_tools(self, style: str, video_type: str) -> List[str]:
        """
        Recommend the best video generation tools for the specified style and type.
        """
        tools = {
            "corporate": ["RunwayML", "Synthesia", "Luma AI", "Pika Labs"],
            "creative": ["RunwayML", "Pika Labs", "Stable Video Diffusion", "Gen-2"],
            "documentary": ["RunwayML", "Luma AI", "Pika Labs"],
            "animated": ["RunwayML", "Pika Labs", "Stable Video Diffusion", "LeiaPix"]
        }
        
        return tools.get(style, ["RunwayML", "Pika Labs"])
    
    def _generate_voiceover_script(self, scenes: List[Dict]) -> Dict[str, Any]:
        """
        Generate voiceover script for the video scenes.
        """
        script = {
            "total_duration": sum(scene["duration"] for scene in scenes),
            "scenes": []
        }
        
        for scene in scenes:
            scene_script = {
                "scene_id": scene["scene_id"],
                "duration": scene["duration"],
                "suggested_text": self._get_voiceover_text(scene["segment_name"]),
                "tone": "professional and engaging",
                "pacing": "moderate speed, clear pronunciation",
                "emphasis_words": []
            }
            script["scenes"].append(scene_script)
        
        return script
    
    def _generate_text_overlays(self, scenes: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate text overlay suggestions for scenes.
        """
        overlays = []
        
        for scene in scenes:
            overlay = {
                "scene_id": scene["scene_id"],
                "text_elements": [
                    {
                        "text": self._get_overlay_text(scene["segment_name"]),
                        "position": "center",
                        "duration": scene["duration"] * 0.8,  # Show for 80% of scene
                        "animation": "fade in",
                        "style": "clean, readable font, brand colors"
                    }
                ]
            }
            overlays.append(overlay)
        
        return overlays
    
    def _get_segment_purpose(self, segment: str) -> str:
        """Get the purpose of each video segment."""
        purposes = {
            "intro": "Introduce the topic and brand",
            "hook": "Capture viewer attention immediately",
            "problem": "Present the challenge or pain point",
            "solution": "Introduce your solution",
            "details": "Explain how it works",
            "benefits": "Show value and outcomes",
            "proof": "Provide evidence or testimonials",
            "call_to_action": "Direct viewers to next step",
            "main_point": "Deliver the key message"
        }
        return purposes.get(segment, "Support the narrative")
    
    def _get_visual_focus(self, segment: str, video_type: str) -> str:
        """Get visual focus for each segment."""
        focus_map = {
            "hook": "Dynamic visuals, attention-grabbing elements",
            "problem": "Problem visualization, pain point representation",
            "solution": "Product/service demonstration, solution visualization",
            "benefits": "Positive outcomes, success indicators",
            "call_to_action": "Clear next steps, contact information"
        }
        return focus_map.get(segment, "Supporting visuals")
    
    def _get_camera_direction(self, segment: str) -> str:
        """Get camera direction for each segment."""
        directions = {
            "intro": "Wide establishing shot",
            "hook": "Dynamic close-up, engaging angle",
            "problem": "Medium shot, focused composition",
            "solution": "Product-focused shot, clean framing",
            "call_to_action": "Direct, face-on composition"
        }
        return directions.get(segment, "Medium shot, steady framing")
    
    def _get_visual_elements(self, segment: str, style_elements: Dict) -> List[str]:
        """Get specific visual elements for each segment."""
        return [
            style_elements["setting"],
            style_elements["colors"], 
            style_elements["props"]
        ]
    
    def _get_pacing_direction(self, segment: str) -> str:
        """Get pacing direction for each segment."""
        pacing = {
            "hook": "Fast-paced, energetic",
            "problem": "Moderate pace, contemplative",
            "solution": "Steady pace, clear presentation",
            "call_to_action": "Deliberate pace, emphasis on key points"
        }
        return pacing.get(segment, "Moderate, engaging pace")
    
    def _get_mood_direction(self, segment: str, style: str) -> str:
        """Get mood direction for each segment."""
        moods = {
            "hook": "Exciting, intriguing",
            "problem": "Serious, relatable",
            "solution": "Optimistic, confident",
            "call_to_action": "Motivating, decisive"
        }
        return moods.get(segment, f"{style} and professional")
    
    def _generate_scene_text(self, segment: Dict) -> str:
        """Generate text overlay for a scene."""
        text_map = {
            "hook": "Attention-grabbing statement",
            "problem": "Problem description",
            "solution": "Solution headline",
            "benefits": "Key benefits",
            "call_to_action": "Next step instruction"
        }
        return text_map.get(segment["name"], segment["purpose"])
    
    def _get_voiceover_text(self, segment: str) -> str:
        """Get suggested voiceover text for each segment."""
        suggestions = {
            "hook": "[Engaging opening statement that hooks the viewer]",
            "problem": "[Describe the problem or challenge your audience faces]",
            "solution": "[Present your solution clearly and confidently]",
            "benefits": "[Highlight the key benefits and value proposition]",
            "call_to_action": "[Clear, compelling call to action]"
        }
        return suggestions.get(segment, "[Engaging narration for this segment]")
    
    def _get_overlay_text(self, segment: str) -> str:
        """Get text overlay suggestions for each segment."""
        overlays = {
            "hook": "Key Question or Statement",
            "problem": "The Challenge",
            "solution": "The Solution",
            "benefits": "Benefits",
            "call_to_action": "Get Started"
        }
        return overlays.get(segment, "Key Point")
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context for video prompt generation.
        """
        if not context.get('content') and not context.get('blog_title'):
            logger.warning("Missing content and blog_title")
            return False
        
        video_type = context.get('video_type', 'explainer')
        valid_types = ['explainer', 'demo', 'social', 'testimonial', 'tutorial', 'promotional']
        if video_type not in valid_types:
            logger.warning(f"Invalid video_type: {video_type}")
            return False
        
        duration = context.get('duration', 60)
        if not isinstance(duration, int) or duration <= 0:
            logger.warning(f"Invalid duration: {duration}")
            return False
        
        return True