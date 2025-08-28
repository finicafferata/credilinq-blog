"""
LangGraph-enhanced Content Repurposing Workflow for intelligent multi-platform content adaptation.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
from .content_repurposer import (
    ContentRepurposer, ContentPlatform, ContentTone, RepurposedContent,
    ContentSeries, PLATFORM_SPECS
)
from ...config.database import DatabaseConnection


class ContentRepurposingState(WorkflowState):
    """Enhanced state for content repurposing workflow."""
    
    # Input data
    original_content: str = ""
    target_platforms: List[ContentPlatform] = field(default_factory=list)
    source_context: Dict[str, Any] = field(default_factory=dict)
    customization_options: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    platform_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Processing results
    repurposed_content: Dict[str, RepurposedContent] = field(default_factory=dict)
    optimization_suggestions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Quality metrics
    engagement_scores: Dict[str, float] = field(default_factory=dict)
    character_counts: Dict[str, int] = field(default_factory=dict)
    optimization_notes: Dict[str, List[str]] = field(default_factory=dict)
    
    # Series creation (optional)
    series_config: Optional[Dict[str, Any]] = None
    content_series: Optional[ContentSeries] = None
    
    # Performance tracking
    processing_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class ContentRepurposerWorkflow(LangGraphWorkflowBase[ContentRepurposingState]):
    """LangGraph workflow for advanced content repurposing with multi-platform optimization."""
    
    def __init__(
        self, 
        workflow_name: str = "content_repurposer_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = ContentRepurposer()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> ContentRepurposingState:
        """Create initial workflow state from context."""
        return ContentRepurposingState(
            workflow_id=context.get("workflow_id", f"repurpose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            original_content=context.get("original_content", ""),
            target_platforms=[
                ContentPlatform(platform) if isinstance(platform, str) else platform
                for platform in context.get("target_platforms", [ContentPlatform.LINKEDIN_POST])
            ],
            source_context=context.get("source_context", {}),
            customization_options=context.get("customization_options", {}),
            series_config=context.get("series_config"),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the content repurposing workflow graph."""
        workflow = StateGraph(ContentRepurposingState)
        
        # Define workflow nodes
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("develop_strategies", self._develop_platform_strategies_node)
        workflow.add_node("repurpose_content", self._repurpose_content_node)
        workflow.add_node("optimize_content", self._optimize_content_node)
        workflow.add_node("generate_series", self._generate_series_node)
        workflow.add_node("validate_quality", self._validate_quality_node)
        workflow.add_node("finalize_output", self._finalize_output_node)
        
        # Define workflow edges
        workflow.add_edge("validate_input", "analyze_content")
        workflow.add_edge("analyze_content", "develop_strategies")
        workflow.add_edge("develop_strategies", "repurpose_content")
        workflow.add_edge("repurpose_content", "optimize_content")
        
        # Conditional routing for series generation
        workflow.add_conditional_edges(
            "optimize_content",
            self._should_generate_series,
            {
                "generate_series": "generate_series",
                "validate_quality": "validate_quality"
            }
        )
        workflow.add_edge("generate_series", "validate_quality")
        workflow.add_edge("validate_quality", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        # Set entry point
        workflow.set_entry_point("validate_input")
        
        return workflow
    
    async def _validate_input_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Validate input parameters and content requirements."""
        try:
            self._log_progress("Validating input parameters and content requirements")
            
            validation_errors = []
            
            # Validate original content
            if not state.original_content or len(state.original_content.strip()) < 50:
                validation_errors.append("Original content must be at least 50 characters long")
            
            # Validate target platforms
            if not state.target_platforms:
                state.target_platforms = [ContentPlatform.LINKEDIN_POST]
                self._log_progress("No target platforms specified, defaulting to LinkedIn")
            
            # Validate platform support
            unsupported_platforms = [
                platform for platform in state.target_platforms
                if platform not in PLATFORM_SPECS
            ]
            if unsupported_platforms:
                validation_errors.append(f"Unsupported platforms: {unsupported_platforms}")
            
            # Set default customization options
            default_options = {
                "tone": ContentTone.PROFESSIONAL,
                "include_cta": True,
                "hashtag_count": 5,
                "optimize_length": True,
                "add_engagement_hooks": True
            }
            
            for key, default_value in default_options.items():
                if key not in state.customization_options:
                    state.customization_options[key] = default_value
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 15.0
                
                state.messages.append(HumanMessage(
                    content=f"Input validated successfully. Processing content for {len(state.target_platforms)} platforms."
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Input validation failed: {str(e)}"
            return state
    
    async def _analyze_content_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Analyze original content to extract key information for repurposing."""
        try:
            self._log_progress("Analyzing content structure and key elements")
            
            # Use the legacy agent's content analysis
            content_analysis = await self.legacy_agent._analyze_content(state.original_content)
            
            # Enhanced analysis for LangGraph workflow
            enhanced_analysis = {
                **content_analysis,
                "content_length": len(state.original_content),
                "word_count": len(state.original_content.split()),
                "readability_score": self._calculate_readability_score(state.original_content),
                "sentiment_indicators": self._analyze_sentiment_indicators(state.original_content),
                "content_structure": self._analyze_content_structure(state.original_content),
                "repurposing_potential": self._assess_repurposing_potential(state.original_content)
            }
            
            state.content_analysis = enhanced_analysis
            state.progress_percentage = 30.0
            
            state.messages.append(SystemMessage(
                content=f"Content analysis completed. Main topic: {enhanced_analysis.get('main_topic', 'General')}. "
                       f"Readability score: {enhanced_analysis.get('readability_score', 'N/A')}. "
                       f"Repurposing potential: {enhanced_analysis.get('repurposing_potential', 'Medium')}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content analysis failed: {str(e)}"
            return state
    
    async def _develop_platform_strategies_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Develop platform-specific strategies for content adaptation."""
        try:
            self._log_progress("Developing platform-specific adaptation strategies")
            
            platform_strategies = {}
            
            for platform in state.target_platforms:
                platform_spec = PLATFORM_SPECS.get(platform)
                if not platform_spec:
                    continue
                
                # Develop strategy based on platform characteristics and content analysis
                strategy = {
                    "adaptation_approach": self._determine_adaptation_approach(
                        platform, state.content_analysis
                    ),
                    "length_target": self._calculate_optimal_length(
                        platform, state.content_analysis, state.customization_options
                    ),
                    "tone_adjustment": self._determine_tone_adjustment(
                        platform, state.content_analysis, state.customization_options.get("tone")
                    ),
                    "engagement_hooks": self._select_engagement_hooks(
                        platform, state.content_analysis
                    ),
                    "hashtag_strategy": self._develop_hashtag_strategy(
                        platform, state.content_analysis
                    ),
                    "cta_strategy": self._develop_cta_strategy(
                        platform, state.content_analysis
                    ),
                    "formatting_rules": self._get_formatting_rules(platform),
                    "optimization_priorities": self._get_optimization_priorities(platform)
                }
                
                platform_strategies[platform.value] = strategy
            
            state.platform_strategies = platform_strategies
            state.progress_percentage = 45.0
            
            state.messages.append(SystemMessage(
                content=f"Platform strategies developed for {len(platform_strategies)} platforms. "
                       f"Ready for content adaptation phase."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Strategy development failed: {str(e)}"
            return state
    
    async def _repurpose_content_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Repurpose content for each target platform using developed strategies."""
        try:
            self._log_progress("Repurposing content for target platforms")
            
            repurposed_content = {}
            processing_metrics = {}
            
            for platform in state.target_platforms:
                try:
                    start_time = datetime.utcnow()
                    
                    # Use the legacy agent's repurposing functionality with strategy guidance
                    repurposed = await self.legacy_agent._adapt_for_platform(
                        original_content=state.original_content,
                        platform=platform,
                        content_analysis=state.content_analysis,
                        source_context=state.source_context,
                        customization_options={
                            **state.customization_options,
                            **state.platform_strategies.get(platform.value, {})
                        }
                    )
                    
                    repurposed_content[platform.value] = repurposed
                    
                    # Track processing metrics
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    processing_metrics[platform.value] = {
                        "processing_time_seconds": processing_time,
                        "content_length_original": len(state.original_content),
                        "content_length_adapted": len(repurposed.content),
                        "compression_ratio": len(repurposed.content) / len(state.original_content),
                        "adaptation_success": True
                    }
                    
                except Exception as platform_error:
                    self._log_error(f"Failed to repurpose for {platform}: {str(platform_error)}")
                    processing_metrics[platform.value] = {
                        "processing_time_seconds": 0,
                        "adaptation_success": False,
                        "error_message": str(platform_error)
                    }
                    continue
            
            state.repurposed_content = repurposed_content
            state.processing_metrics = processing_metrics
            state.progress_percentage = 70.0
            
            successful_adaptations = len([m for m in processing_metrics.values() if m.get("adaptation_success")])
            state.messages.append(SystemMessage(
                content=f"Content repurposing completed. {successful_adaptations}/{len(state.target_platforms)} "
                       f"platforms processed successfully."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content repurposing failed: {str(e)}"
            return state
    
    async def _optimize_content_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Optimize repurposed content for maximum engagement and platform compliance."""
        try:
            self._log_progress("Optimizing content for engagement and platform compliance")
            
            optimization_suggestions = {}
            engagement_scores = {}
            character_counts = {}
            optimization_notes = {}
            
            for platform_key, content in state.repurposed_content.items():
                platform = ContentPlatform(platform_key)
                
                # Calculate engagement score
                engagement_score = content.estimated_engagement.get("score", 50)
                engagement_scores[platform_key] = engagement_score
                
                # Track character count
                character_counts[platform_key] = content.character_count
                
                # Generate optimization suggestions
                suggestions = []
                notes = []
                
                # Platform compliance checks
                platform_spec = PLATFORM_SPECS.get(platform)
                if platform_spec and content.character_count > platform_spec.max_length:
                    suggestions.append("Content exceeds platform maximum length")
                    notes.append(f"Reduce by {content.character_count - platform_spec.max_length} characters")
                
                # Engagement optimization suggestions
                if engagement_score < 60:
                    suggestions.append("Low engagement potential detected")
                    if "?" not in content.content:
                        suggestions.append("Add questions to encourage interaction")
                    if len(content.hashtags) < 3:
                        suggestions.append("Increase hashtag count for better discoverability")
                
                # Quality improvements
                if not content.call_to_action and state.customization_options.get("include_cta", True):
                    suggestions.append("Missing call-to-action")
                    notes.append("Add engaging CTA to drive audience response")
                
                # Platform-specific optimizations
                if platform == ContentPlatform.LINKEDIN_POST:
                    if "professional" not in content.content.lower():
                        suggestions.append("Consider adding professional insights")
                elif platform == ContentPlatform.TWITTER_THREAD:
                    if content.content.count('\n\n') < 2:
                        suggestions.append("Break into more distinct thread segments")
                elif platform == ContentPlatform.INSTAGRAM_POST:
                    emoji_count = len([c for c in content.content if ord(c) > 127])
                    if emoji_count < 3:
                        suggestions.append("Add more visual elements (emojis)")
                
                optimization_suggestions[platform_key] = suggestions
                optimization_notes[platform_key] = notes
            
            state.optimization_suggestions = optimization_suggestions
            state.engagement_scores = engagement_scores
            state.character_counts = character_counts
            state.optimization_notes = optimization_notes
            state.progress_percentage = 85.0
            
            avg_engagement = sum(engagement_scores.values()) / len(engagement_scores) if engagement_scores else 0
            state.messages.append(SystemMessage(
                content=f"Content optimization completed. Average engagement score: {avg_engagement:.1f}. "
                       f"Optimization suggestions generated for all platforms."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content optimization failed: {str(e)}"
            return state
    
    async def _generate_series_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Generate content series if requested."""
        try:
            self._log_progress("Generating content series from repurposed content")
            
            if not state.series_config:
                return state
            
            # Use the legacy agent's series creation functionality
            content_series = await self.legacy_agent.create_content_series(
                base_content=state.original_content,
                series_config=state.series_config
            )
            
            state.content_series = content_series
            state.messages.append(SystemMessage(
                content=f"Content series generated: '{content_series.title}' with {content_series.total_posts} posts."
            ))
            
            return state
            
        except Exception as e:
            # Series generation failure shouldn't fail the entire workflow
            self._log_error(f"Content series generation failed: {str(e)}")
            state.messages.append(SystemMessage(
                content="Content series generation failed, continuing with individual content pieces."
            ))
            return state
    
    async def _validate_quality_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Validate quality of repurposed content and generate quality scores."""
        try:
            self._log_progress("Validating content quality and calculating quality scores")
            
            quality_scores = {}
            
            for platform_key, content in state.repurposed_content.items():
                platform = ContentPlatform(platform_key)
                platform_spec = PLATFORM_SPECS.get(platform)
                
                quality_score = 0
                max_score = 100
                
                # Length compliance (20 points)
                if platform_spec:
                    if content.character_count <= platform_spec.max_length:
                        if abs(content.character_count - platform_spec.optimal_length) <= 100:
                            quality_score += 20
                        else:
                            quality_score += 15
                    else:
                        quality_score += 5
                
                # Content structure (25 points)
                if content.content and len(content.content.strip()) > 0:
                    quality_score += 10
                if len(content.content.split('.')) > 1:  # Multiple sentences
                    quality_score += 10
                if any(hook in content.content.lower() for hook in ["tip", "insight", "key", "important"]):
                    quality_score += 5
                
                # Engagement elements (25 points)
                if content.call_to_action:
                    quality_score += 10
                if "?" in content.content:
                    quality_score += 8
                if len(content.hashtags) >= 3:
                    quality_score += 7
                
                # Platform optimization (30 points)
                engagement_score = content.estimated_engagement.get("score", 50)
                quality_score += min(30, engagement_score * 0.3)
                
                quality_scores[platform_key] = min(quality_score, max_score)
            
            state.quality_scores = quality_scores
            state.progress_percentage = 95.0
            
            avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
            state.messages.append(SystemMessage(
                content=f"Quality validation completed. Average quality score: {avg_quality:.1f}/100."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Quality validation failed: {str(e)}"
            return state
    
    async def _finalize_output_node(self, state: ContentRepurposingState) -> ContentRepurposingState:
        """Finalize workflow output and prepare results."""
        try:
            self._log_progress("Finalizing workflow output and preparing results")
            
            # Calculate overall metrics
            total_platforms = len(state.target_platforms)
            successful_adaptations = len(state.repurposed_content)
            avg_engagement = sum(state.engagement_scores.values()) / len(state.engagement_scores) if state.engagement_scores else 0
            avg_quality = sum(state.quality_scores.values()) / len(state.quality_scores) if state.quality_scores else 0
            
            # Generate comprehensive recommendations
            recommendations = []
            
            if avg_quality < 70:
                recommendations.append("Consider revising content to improve overall quality scores")
            if avg_engagement < 60:
                recommendations.append("Add more engaging hooks and interactive elements")
            if successful_adaptations < total_platforms:
                recommendations.append("Review failed adaptations and retry with adjusted parameters")
            
            # Add platform-specific recommendations
            for platform_key, suggestions in state.optimization_suggestions.items():
                if suggestions:
                    recommendations.extend([f"{platform_key}: {suggestion}" for suggestion in suggestions[:2]])
            
            state.status = WorkflowStatus.COMPLETED if successful_adaptations > 0 else WorkflowStatus.FAILED
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            # Prepare final result summary
            result_summary = {
                "workflow_id": state.workflow_id,
                "total_platforms_requested": total_platforms,
                "successful_adaptations": successful_adaptations,
                "average_engagement_score": round(avg_engagement, 1),
                "average_quality_score": round(avg_quality, 1),
                "content_series_created": state.content_series is not None,
                "processing_time_seconds": (state.completed_at - state.created_at).total_seconds(),
                "recommendations": recommendations[:5]  # Top 5 recommendations
            }
            
            state.messages.append(SystemMessage(
                content=f"Content repurposing workflow completed successfully. "
                       f"{successful_adaptations}/{total_platforms} platforms processed. "
                       f"Average quality: {avg_quality:.1f}/100."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Output finalization failed: {str(e)}"
            return state
    
    def _should_generate_series(self, state: ContentRepurposingState) -> str:
        """Determine if content series should be generated."""
        return "generate_series" if state.series_config else "validate_quality"
    
    # Helper methods for enhanced analysis and strategy development
    
    def _calculate_readability_score(self, content: str) -> str:
        """Calculate basic readability score."""
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        if sentences == 0:
            return "Unknown"
        
        avg_words_per_sentence = words / sentences
        
        if avg_words_per_sentence < 15:
            return "Easy"
        elif avg_words_per_sentence < 25:
            return "Medium"
        else:
            return "Complex"
    
    def _analyze_sentiment_indicators(self, content: str) -> List[str]:
        """Analyze sentiment indicators in content."""
        positive_indicators = ["success", "growth", "opportunity", "benefit", "advantage", "improve"]
        negative_indicators = ["problem", "challenge", "mistake", "fail", "difficult", "issue"]
        neutral_indicators = ["analysis", "data", "report", "study", "research", "findings"]
        
        content_lower = content.lower()
        indicators = []
        
        if any(indicator in content_lower for indicator in positive_indicators):
            indicators.append("positive")
        if any(indicator in content_lower for indicator in negative_indicators):
            indicators.append("negative")
        if any(indicator in content_lower for indicator in neutral_indicators):
            indicators.append("analytical")
        
        return indicators if indicators else ["neutral"]
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of the content."""
        paragraphs = content.split('\n\n')
        sentences = content.split('.')
        
        return {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "has_headings": any(line.startswith('#') for line in content.split('\n')),
            "has_lists": any(line.strip().startswith(('-', '*', '1.')) for line in content.split('\n')),
            "has_questions": '?' in content,
            "has_calls_to_action": any(phrase in content.lower() for phrase in ["contact", "learn more", "sign up", "follow"])
        }
    
    def _assess_repurposing_potential(self, content: str) -> str:
        """Assess the repurposing potential of content."""
        word_count = len(content.split())
        structure = self._analyze_content_structure(content)
        
        score = 0
        
        # Length factor
        if word_count > 300:
            score += 2
        elif word_count > 100:
            score += 1
        
        # Structure factor
        if structure["paragraph_count"] > 3:
            score += 1
        if structure["has_lists"]:
            score += 1
        if structure["has_questions"]:
            score += 1
        
        # Content richness
        if any(keyword in content.lower() for keyword in ["example", "tip", "strategy", "insight", "analysis"]):
            score += 1
        
        if score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _determine_adaptation_approach(self, platform: ContentPlatform, content_analysis: Dict[str, Any]) -> str:
        """Determine the best adaptation approach for the platform."""
        content_type = content_analysis.get("content_type", "educational")
        word_count = content_analysis.get("word_count", 0)
        
        if platform == ContentPlatform.TWITTER_THREAD:
            return "sequential_breakdown" if word_count > 200 else "single_tweet_summary"
        elif platform == ContentPlatform.LINKEDIN_POST:
            return "professional_insight" if content_type == "educational" else "personal_experience"
        elif platform == ContentPlatform.INSTAGRAM_POST:
            return "visual_storytelling"
        else:
            return "direct_adaptation"
    
    def _calculate_optimal_length(self, platform: ContentPlatform, content_analysis: Dict[str, Any], customization_options: Dict[str, Any]) -> int:
        """Calculate optimal length for platform-specific content."""
        platform_spec = PLATFORM_SPECS.get(platform)
        if not platform_spec:
            return 500
        
        base_optimal = platform_spec.optimal_length
        
        # Adjust based on content complexity
        if content_analysis.get("expertise_level") == "advanced":
            return min(base_optimal * 1.2, platform_spec.max_length)
        elif content_analysis.get("expertise_level") == "beginner":
            return base_optimal * 0.8
        
        return base_optimal
    
    def _determine_tone_adjustment(self, platform: ContentPlatform, content_analysis: Dict[str, Any], preferred_tone: Optional[ContentTone]) -> ContentTone:
        """Determine appropriate tone adjustment for the platform."""
        if preferred_tone:
            return preferred_tone
        
        current_tone = content_analysis.get("tone", "professional")
        platform_spec = PLATFORM_SPECS.get(platform)
        
        if platform_spec:
            return platform_spec.preferred_tone
        
        return ContentTone.PROFESSIONAL
    
    def _select_engagement_hooks(self, platform: ContentPlatform, content_analysis: Dict[str, Any]) -> List[str]:
        """Select appropriate engagement hooks for the platform."""
        platform_spec = PLATFORM_SPECS.get(platform)
        if platform_spec and hasattr(platform_spec, 'engagement_hooks'):
            return platform_spec.engagement_hooks
        
        # Default hooks based on content topic
        topic = content_analysis.get("main_topic", "business")
        return [
            f"What's your experience with {topic}?",
            f"Here's what I learned about {topic}:",
            f"3 key insights about {topic}:"
        ]
    
    def _develop_hashtag_strategy(self, platform: ContentPlatform, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Develop hashtag strategy for the platform."""
        return {
            "count": 5 if platform == ContentPlatform.INSTAGRAM_POST else 3,
            "mix": "popular_and_niche",
            "keywords": content_analysis.get("keywords", []),
            "branded": False,
            "trending": True
        }
    
    def _develop_cta_strategy(self, platform: ContentPlatform, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Develop call-to-action strategy for the platform."""
        platform_spec = PLATFORM_SPECS.get(platform)
        
        return {
            "style": platform_spec.call_to_action_style if platform_spec else "Ask a question",
            "position": "end",
            "tone": "engaging",
            "type": "engagement" if platform in [ContentPlatform.LINKEDIN_POST, ContentPlatform.FACEBOOK_POST] else "viral"
        }
    
    def _get_formatting_rules(self, platform: ContentPlatform) -> List[str]:
        """Get platform-specific formatting rules."""
        rules = {
            ContentPlatform.LINKEDIN_POST: [
                "Use line breaks for readability",
                "Professional emoji usage",
                "Hashtags at the end"
            ],
            ContentPlatform.TWITTER_THREAD: [
                "Number each tweet (1/, 2/, etc.)",
                "Keep each tweet under 280 characters",
                "Use strategic line breaks"
            ],
            ContentPlatform.INSTAGRAM_POST: [
                "Use emojis throughout",
                "Add line breaks for visual appeal",
                "Mix of popular and niche hashtags"
            ]
        }
        
        return rules.get(platform, ["Standard formatting"])
    
    def _get_optimization_priorities(self, platform: ContentPlatform) -> List[str]:
        """Get optimization priorities for the platform."""
        priorities = {
            ContentPlatform.LINKEDIN_POST: ["professional_insights", "engagement", "credibility"],
            ContentPlatform.TWITTER_THREAD: ["virality", "shareability", "conciseness"],
            ContentPlatform.INSTAGRAM_POST: ["visual_appeal", "saves", "comments"],
            ContentPlatform.FACEBOOK_POST: ["shares", "comments", "community_building"]
        }
        
        return priorities.get(platform, ["engagement", "quality"])
    
    async def execute_workflow(
        self,
        original_content: str,
        target_platforms: List[ContentPlatform],
        source_context: Optional[Dict[str, Any]] = None,
        customization_options: Optional[Dict[str, Any]] = None,
        series_config: Optional[Dict[str, Any]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the content repurposing workflow."""
        
        context = {
            "original_content": original_content,
            "target_platforms": target_platforms,
            "source_context": source_context or {},
            "customization_options": customization_options or {},
            "series_config": series_config,
            "workflow_id": f"repurpose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "repurposed_content": {
                        platform_key: {
                            "platform": content.platform.value,
                            "content": content.content,
                            "hashtags": content.hashtags,
                            "call_to_action": content.call_to_action,
                            "word_count": content.word_count,
                            "character_count": content.character_count,
                            "estimated_engagement": content.estimated_engagement,
                            "optimization_notes": content.optimization_notes
                        }
                        for platform_key, content in final_state.repurposed_content.items()
                    },
                    "content_series": final_state.content_series.__dict__ if final_state.content_series else None,
                    "quality_metrics": {
                        "engagement_scores": final_state.engagement_scores,
                        "quality_scores": final_state.quality_scores,
                        "processing_metrics": final_state.processing_metrics
                    },
                    "optimization_suggestions": final_state.optimization_suggestions,
                    "workflow_metrics": {
                        "total_processing_time": (final_state.completed_at - final_state.created_at).total_seconds(),
                        "successful_adaptations": len(final_state.repurposed_content),
                        "requested_platforms": len(target_platforms)
                    }
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "platforms_processed": list(final_state.repurposed_content.keys()),
                        "average_quality_score": sum(final_state.quality_scores.values()) / len(final_state.quality_scores) if final_state.quality_scores else 0
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Workflow failed",
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "final_status": final_state.status.value
                    }
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=f"Workflow execution failed: {str(e)}",
                metadata={"error_type": "workflow_execution_error"}
            )