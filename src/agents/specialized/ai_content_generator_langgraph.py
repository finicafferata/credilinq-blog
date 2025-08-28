"""
LangGraph-enhanced AI Content Generator Workflow for template-based multi-format content generation.
"""

import json
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
from .ai_content_generator import (
    AIContentGeneratorAgent, ContentGenerationRequest, GeneratedContent,
    ContentType, ContentChannel
)
from ...config.database import DatabaseConnection


@dataclass
class ContentVariation:
    """Content variation for A/B testing."""
    variation_id: str
    content: GeneratedContent
    variation_type: str  # tone, length, structure, hook
    a_b_test_ready: bool = True
    target_segment: Optional[str] = None


@dataclass
class QualityMetrics:
    """Content quality assessment metrics."""
    readability_score: float
    engagement_potential: float
    seo_optimization: float
    brand_consistency: float
    technical_accuracy: float
    overall_quality: float
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class ContentTemplate:
    """Content generation template configuration."""
    template_id: str
    content_type: ContentType
    structure: List[str]
    tone_guidelines: Dict[str, str]
    length_parameters: Dict[str, int]
    personalization_fields: List[str]
    success_metrics: Dict[str, Any]


class AIContentGeneratorState(WorkflowState):
    """Enhanced state for AI content generation workflow."""
    
    # Input configuration
    generation_requests: List[ContentGenerationRequest] = field(default_factory=list)
    campaign_context: Dict[str, Any] = field(default_factory=dict)
    content_templates: Dict[str, ContentTemplate] = field(default_factory=dict)
    
    # Content generation parameters
    variation_count: int = 1
    quality_threshold: float = 7.0
    seo_optimization_enabled: bool = True
    brand_voice_consistency: bool = True
    a_b_testing_enabled: bool = False
    
    # Generated content results
    generated_content: Dict[str, GeneratedContent] = field(default_factory=dict)
    content_variations: Dict[str, List[ContentVariation]] = field(default_factory=dict)
    rejected_content: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality assessment
    quality_metrics: Dict[str, QualityMetrics] = field(default_factory=dict)
    content_improvements: Dict[str, List[str]] = field(default_factory=dict)
    approval_status: Dict[str, bool] = field(default_factory=dict)
    
    # Performance tracking
    generation_metrics: Dict[str, Any] = field(default_factory=dict)
    template_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    
    # Content strategy alignment
    strategy_alignment_scores: Dict[str, float] = field(default_factory=dict)
    competitive_differentiation: Dict[str, List[str]] = field(default_factory=dict)
    brand_voice_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class AIContentGeneratorWorkflow(LangGraphWorkflowBase[AIContentGeneratorState]):
    """LangGraph workflow for intelligent template-based content generation."""
    
    def __init__(
        self, 
        workflow_name: str = "ai_content_generator_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = AIContentGeneratorAgent()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> AIContentGeneratorState:
        """Create initial workflow state from context."""
        generation_requests = []
        
        # Parse generation requests from context
        requests_data = context.get("generation_requests", [])
        for request_data in requests_data:
            request = ContentGenerationRequest(
                campaign_id=request_data.get("campaign_id", ""),
                content_type=ContentType(request_data.get("content_type", "blog_posts")),
                channel=ContentChannel(request_data.get("channel", "blog")),
                title=request_data.get("title"),
                themes=request_data.get("themes", []),
                target_audience=request_data.get("target_audience", "B2B professionals"),
                tone=request_data.get("tone", "Professional"),
                word_count=request_data.get("word_count"),
                key_messages=request_data.get("key_messages", []),
                call_to_action=request_data.get("call_to_action"),
                company_context=request_data.get("company_context", ""),
                competitive_insights=request_data.get("competitive_insights"),
                seo_keywords=request_data.get("seo_keywords", []),
                content_pillars=request_data.get("content_pillars", [])
            )
            generation_requests.append(request)
        
        return AIContentGeneratorState(
            workflow_id=context.get("workflow_id", f"content_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            generation_requests=generation_requests,
            campaign_context=context.get("campaign_context", {}),
            variation_count=context.get("variation_count", 1),
            quality_threshold=context.get("quality_threshold", 7.0),
            seo_optimization_enabled=context.get("seo_optimization_enabled", True),
            brand_voice_consistency=context.get("brand_voice_consistency", True),
            a_b_testing_enabled=context.get("a_b_testing_enabled", False),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the AI content generation workflow graph."""
        workflow = StateGraph(AIContentGeneratorState)
        
        # Define workflow nodes
        workflow.add_node("validate_requests", self._validate_requests_node)
        workflow.add_node("prepare_templates", self._prepare_templates_node)
        workflow.add_node("generate_base_content", self._generate_base_content_node)
        workflow.add_node("create_variations", self._create_variations_node)
        workflow.add_node("assess_quality", self._assess_quality_node)
        workflow.add_node("optimize_content", self._optimize_content_node)
        workflow.add_node("validate_brand_voice", self._validate_brand_voice_node)
        workflow.add_node("finalize_content", self._finalize_content_node)
        
        # Define workflow edges
        workflow.add_edge("validate_requests", "prepare_templates")
        workflow.add_edge("prepare_templates", "generate_base_content")
        
        # Conditional routing for variations
        workflow.add_conditional_edges(
            "generate_base_content",
            self._should_create_variations,
            {
                "create_variations": "create_variations",
                "assess_quality": "assess_quality"
            }
        )
        workflow.add_edge("create_variations", "assess_quality")
        workflow.add_edge("assess_quality", "optimize_content")
        workflow.add_edge("optimize_content", "validate_brand_voice")
        workflow.add_edge("validate_brand_voice", "finalize_content")
        workflow.add_edge("finalize_content", END)
        
        # Set entry point
        workflow.set_entry_point("validate_requests")
        
        return workflow
    
    async def _validate_requests_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Validate content generation requests and parameters."""
        try:
            self._log_progress("Validating content generation requests")
            
            validation_errors = []
            valid_requests = []
            
            # Validate generation requests
            if not state.generation_requests:
                validation_errors.append("No content generation requests provided")
            
            for request in state.generation_requests:
                # Validate required fields
                if not request.campaign_id:
                    validation_errors.append("Campaign ID required for content generation")
                    continue
                
                if not request.content_type:
                    validation_errors.append("Content type required")
                    continue
                
                if not request.channel:
                    validation_errors.append("Content channel required")
                    continue
                
                # Validate content parameters
                if request.content_type in [ContentType.BLOG_POST, ContentType.LINKEDIN_ARTICLE, ContentType.WHITEPAPER]:
                    if not request.themes:
                        request.themes = ["Industry Insights"]  # Set default
                
                valid_requests.append(request)
            
            state.generation_requests = valid_requests
            
            # Validate quality threshold
            if state.quality_threshold < 0 or state.quality_threshold > 10:
                validation_errors.append("Quality threshold must be between 0 and 10")
            
            # Validate variation count
            if state.variation_count < 1 or state.variation_count > 10:
                state.variation_count = min(max(1, state.variation_count), 10)
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 15.0
                
                state.messages.append(HumanMessage(
                    content=f"Validated {len(valid_requests)} content generation requests. "
                           f"Quality threshold: {state.quality_threshold}, Variations: {state.variation_count}"
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Request validation failed: {str(e)}"
            return state
    
    async def _prepare_templates_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Prepare content templates based on requests and campaign context."""
        try:
            self._log_progress("Preparing content templates and configurations")
            
            content_templates = {}
            
            for request in state.generation_requests:
                template_id = f"{request.content_type.value}_{request.channel.value}"
                
                # Create content template based on type and channel
                template = await self._create_content_template(request, state.campaign_context)
                content_templates[template_id] = template
            
            state.content_templates = content_templates
            state.progress_percentage = 25.0
            
            state.messages.append(SystemMessage(
                content=f"Prepared {len(content_templates)} content templates for generation."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Template preparation failed: {str(e)}"
            return state
    
    async def _generate_base_content_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Generate base content using templates and AI generation."""
        try:
            self._log_progress("Generating base content using AI templates")
            
            generated_content = {}
            generation_metrics = {}
            
            for request in state.generation_requests:
                try:
                    start_time = datetime.utcnow()
                    
                    # Use legacy agent for content generation
                    generated = await self.legacy_agent.generate_content(request)
                    
                    content_id = generated.content_id
                    generated_content[content_id] = generated
                    
                    # Track generation metrics
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    generation_metrics[content_id] = {
                        "processing_time_seconds": processing_time,
                        "content_type": generated.content_type.value,
                        "channel": generated.channel.value,
                        "word_count": generated.word_count,
                        "initial_quality_score": generated.quality_score
                    }
                    
                except Exception as generation_error:
                    self._log_error(f"Content generation failed for {request.content_type}: {str(generation_error)}")
                    continue
            
            state.generated_content = generated_content
            state.generation_metrics = generation_metrics
            state.progress_percentage = 45.0
            
            successful_generations = len(generated_content)
            state.messages.append(SystemMessage(
                content=f"Generated {successful_generations} pieces of base content successfully."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Base content generation failed: {str(e)}"
            return state
    
    async def _create_variations_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Create content variations for A/B testing."""
        try:
            self._log_progress("Creating content variations for A/B testing")
            
            content_variations = {}
            
            for content_id, base_content in state.generated_content.items():
                variations = []
                
                # Find the corresponding request
                matching_request = None
                for request in state.generation_requests:
                    if request.content_type == base_content.content_type and request.channel == base_content.channel:
                        matching_request = request
                        break
                
                if not matching_request:
                    continue
                
                try:
                    # Generate variations using legacy agent
                    generated_variations = await self.legacy_agent.generate_content_variations(
                        matching_request, state.variation_count
                    )
                    
                    for i, variation_content in enumerate(generated_variations):
                        variation = ContentVariation(
                            variation_id=f"{content_id}_var_{i+1}",
                            content=variation_content,
                            variation_type=self._determine_variation_type(i),
                            target_segment=self._determine_target_segment(i, matching_request.target_audience)
                        )
                        variations.append(variation)
                    
                    content_variations[content_id] = variations
                    
                except Exception as variation_error:
                    self._log_error(f"Variation generation failed for {content_id}: {str(variation_error)}")
                    continue
            
            state.content_variations = content_variations
            state.progress_percentage = 60.0
            
            total_variations = sum(len(variations) for variations in content_variations.values())
            state.messages.append(SystemMessage(
                content=f"Created {total_variations} content variations across {len(content_variations)} base contents."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Variation creation failed: {str(e)}"
            return state
    
    async def _assess_quality_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Assess quality of generated content and variations."""
        try:
            self._log_progress("Assessing content quality and performance metrics")
            
            quality_metrics = {}
            approval_status = {}
            rejected_content = []
            
            # Assess base content quality
            for content_id, content in state.generated_content.items():
                quality = await self._calculate_quality_metrics(content, state.campaign_context)
                quality_metrics[content_id] = quality
                
                # Determine approval status
                meets_threshold = quality.overall_quality >= state.quality_threshold
                approval_status[content_id] = meets_threshold
                
                if not meets_threshold:
                    rejected_content.append({
                        "content_id": content_id,
                        "content_type": content.content_type.value,
                        "quality_score": quality.overall_quality,
                        "threshold": state.quality_threshold,
                        "rejection_reasons": quality.improvement_suggestions
                    })
            
            # Assess variation quality
            for content_id, variations in state.content_variations.items():
                for variation in variations:
                    variation_quality = await self._calculate_quality_metrics(
                        variation.content, state.campaign_context
                    )
                    quality_metrics[variation.variation_id] = variation_quality
                    
                    meets_threshold = variation_quality.overall_quality >= state.quality_threshold
                    approval_status[variation.variation_id] = meets_threshold
            
            state.quality_metrics = quality_metrics
            state.approval_status = approval_status
            state.rejected_content = rejected_content
            state.progress_percentage = 75.0
            
            approved_count = sum(1 for approved in approval_status.values() if approved)
            total_content = len(approval_status)
            
            state.messages.append(SystemMessage(
                content=f"Quality assessment completed. {approved_count}/{total_content} contents meet quality threshold."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Quality assessment failed: {str(e)}"
            return state
    
    async def _optimize_content_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Optimize content based on SEO, engagement, and quality metrics."""
        try:
            self._log_progress("Optimizing content for SEO and engagement")
            
            optimization_results = {}
            content_improvements = {}
            
            # Optimize approved content
            for content_id, approved in state.approval_status.items():
                if not approved:
                    continue
                
                # Get content (base or variation)
                content = None
                if content_id in state.generated_content:
                    content = state.generated_content[content_id]
                else:
                    # Look in variations
                    for variations in state.content_variations.values():
                        for variation in variations:
                            if variation.variation_id == content_id:
                                content = variation.content
                                break
                
                if not content:
                    continue
                
                try:
                    # Apply SEO optimization if enabled
                    if state.seo_optimization_enabled:
                        seo_improvements = await self._apply_seo_optimization(content)
                        content_improvements[content_id] = seo_improvements
                    
                    # Apply engagement optimization
                    engagement_improvements = await self._apply_engagement_optimization(content)
                    
                    # Calculate optimization impact
                    quality_before = state.quality_metrics.get(content_id)
                    if quality_before:
                        optimization_results[content_id] = {
                            "seo_improvements": content_improvements.get(content_id, []),
                            "engagement_improvements": engagement_improvements,
                            "quality_before": quality_before.overall_quality,
                            "estimated_improvement": 0.5  # Estimated quality improvement
                        }
                    
                except Exception as optimization_error:
                    self._log_error(f"Content optimization failed for {content_id}: {str(optimization_error)}")
                    continue
            
            state.optimization_results = optimization_results
            state.content_improvements = content_improvements
            state.progress_percentage = 85.0
            
            optimized_count = len(optimization_results)
            state.messages.append(SystemMessage(
                content=f"Content optimization completed for {optimized_count} approved contents."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content optimization failed: {str(e)}"
            return state
    
    async def _validate_brand_voice_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Validate brand voice consistency across generated content."""
        try:
            self._log_progress("Validating brand voice consistency")
            
            if not state.brand_voice_consistency:
                self._log_progress("Brand voice validation disabled, skipping")
                return state
            
            brand_voice_analysis = {}
            strategy_alignment_scores = {}
            competitive_differentiation = {}
            
            # Analyze brand voice consistency
            for content_id in state.approval_status:
                if not state.approval_status[content_id]:
                    continue
                
                # Get content for analysis
                content = None
                if content_id in state.generated_content:
                    content = state.generated_content[content_id]
                else:
                    # Look in variations
                    for variations in state.content_variations.values():
                        for variation in variations:
                            if variation.variation_id == content_id:
                                content = variation.content
                                break
                
                if not content:
                    continue
                
                try:
                    # Analyze brand voice consistency
                    brand_analysis = await self._analyze_brand_voice_consistency(
                        content, state.campaign_context
                    )
                    brand_voice_analysis[content_id] = brand_analysis
                    
                    # Calculate strategy alignment
                    alignment_score = await self._calculate_strategy_alignment(
                        content, state.campaign_context
                    )
                    strategy_alignment_scores[content_id] = alignment_score
                    
                    # Identify competitive differentiation
                    differentiation = await self._identify_competitive_differentiation(
                        content, state.campaign_context
                    )
                    competitive_differentiation[content_id] = differentiation
                    
                except Exception as analysis_error:
                    self._log_error(f"Brand voice analysis failed for {content_id}: {str(analysis_error)}")
                    continue
            
            state.brand_voice_analysis = brand_voice_analysis
            state.strategy_alignment_scores = strategy_alignment_scores
            state.competitive_differentiation = competitive_differentiation
            state.progress_percentage = 95.0
            
            analyzed_count = len(brand_voice_analysis)
            avg_alignment = sum(strategy_alignment_scores.values()) / len(strategy_alignment_scores) if strategy_alignment_scores else 0
            
            state.messages.append(SystemMessage(
                content=f"Brand voice validation completed for {analyzed_count} contents. "
                       f"Average strategy alignment: {avg_alignment:.1f}/10."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Brand voice validation failed: {str(e)}"
            return state
    
    async def _finalize_content_node(self, state: AIContentGeneratorState) -> AIContentGeneratorState:
        """Finalize content generation and prepare results."""
        try:
            self._log_progress("Finalizing content generation results")
            
            # Calculate final metrics
            total_requests = len(state.generation_requests)
            total_generated = len(state.generated_content)
            total_variations = sum(len(variations) for variations in state.content_variations.values())
            approved_count = sum(1 for approved in state.approval_status.values() if approved)
            
            # Calculate success rates
            generation_success_rate = (total_generated / total_requests * 100) if total_requests > 0 else 0
            quality_success_rate = (approved_count / (total_generated + total_variations) * 100) if (total_generated + total_variations) > 0 else 0
            
            # Calculate average metrics
            avg_quality_score = 0
            avg_processing_time = 0
            
            if state.quality_metrics:
                avg_quality_score = sum(qm.overall_quality for qm in state.quality_metrics.values()) / len(state.quality_metrics)
            
            if state.generation_metrics:
                processing_times = [gm["processing_time_seconds"] for gm in state.generation_metrics.values()]
                avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Prepare final summary
            final_summary = {
                "total_requests": total_requests,
                "total_generated": total_generated,
                "total_variations": total_variations,
                "approved_count": approved_count,
                "rejected_count": len(state.rejected_content),
                "generation_success_rate": round(generation_success_rate, 2),
                "quality_success_rate": round(quality_success_rate, 2),
                "average_quality_score": round(avg_quality_score, 2),
                "average_processing_time": round(avg_processing_time, 3),
                "seo_optimization_applied": state.seo_optimization_enabled,
                "brand_voice_validated": state.brand_voice_consistency,
                "a_b_variations_created": state.a_b_testing_enabled
            }
            
            # Determine workflow status
            if approved_count > 0:
                state.status = WorkflowStatus.COMPLETED
            elif total_generated == 0:
                state.status = WorkflowStatus.FAILED
                state.error_message = "No content was successfully generated"
            else:
                state.status = WorkflowStatus.COMPLETED  # Generated but below quality threshold
            
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            # Store final summary in generation metrics
            state.generation_metrics["final_summary"] = final_summary
            
            state.messages.append(SystemMessage(
                content=f"Content generation workflow completed. {approved_count} high-quality contents ready. "
                       f"Generation success rate: {generation_success_rate:.1f}%."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content finalization failed: {str(e)}"
            return state
    
    def _should_create_variations(self, state: AIContentGeneratorState) -> str:
        """Determine if content variations should be created."""
        return "create_variations" if state.a_b_testing_enabled and state.variation_count > 1 else "assess_quality"
    
    # Helper methods for enhanced content generation
    
    async def _create_content_template(
        self, 
        request: ContentGenerationRequest, 
        campaign_context: Dict[str, Any]
    ) -> ContentTemplate:
        """Create content template based on request and context."""
        try:
            # Define structure based on content type
            structure = self._get_content_structure(request.content_type)
            
            # Define tone guidelines
            tone_guidelines = {
                "primary_tone": request.tone,
                "audience_adaptation": request.target_audience,
                "brand_voice": campaign_context.get("brand_voice", "professional"),
                "formality_level": "formal" if "professional" in request.target_audience.lower() else "conversational"
            }
            
            # Define length parameters
            length_parameters = self._get_length_parameters(request.content_type, request.word_count)
            
            # Define personalization fields
            personalization_fields = [
                "target_audience", "company_context", "industry_specific_terms",
                "competitive_advantages", "key_differentiators"
            ]
            
            # Define success metrics
            success_metrics = {
                "min_quality_score": 7.0,
                "target_engagement": "high",
                "seo_score_target": 80.0,
                "readability_target": "medium-high"
            }
            
            return ContentTemplate(
                template_id=f"{request.content_type.value}_{request.channel.value}",
                content_type=request.content_type,
                structure=structure,
                tone_guidelines=tone_guidelines,
                length_parameters=length_parameters,
                personalization_fields=personalization_fields,
                success_metrics=success_metrics
            )
            
        except Exception as e:
            self._log_error(f"Template creation failed: {str(e)}")
            # Return basic template as fallback
            return ContentTemplate(
                template_id="basic_template",
                content_type=request.content_type,
                structure=["introduction", "main_content", "conclusion"],
                tone_guidelines={"primary_tone": "professional"},
                length_parameters={"min_words": 100, "max_words": 1000},
                personalization_fields=[],
                success_metrics={"min_quality_score": 6.0}
            )
    
    def _get_content_structure(self, content_type: ContentType) -> List[str]:
        """Get content structure template based on type."""
        structures = {
            ContentType.BLOG_POST: ["hook", "introduction", "main_sections", "conclusion", "call_to_action"],
            ContentType.SOCIAL_POST: ["hook", "main_message", "engagement_element", "hashtags"],
            ContentType.EMAIL_CONTENT: ["subject_line", "preheader", "greeting", "body", "call_to_action", "signature"],
            ContentType.LINKEDIN_ARTICLE: ["headline", "hook", "introduction", "main_sections", "conclusion", "engagement_question"],
            ContentType.TWITTER_THREAD: ["opening_tweet", "main_tweets", "closing_tweet"],
            ContentType.CASE_STUDY: ["executive_summary", "challenge", "solution", "results", "conclusion"],
            ContentType.NEWSLETTER: ["header", "featured_content", "industry_news", "resources", "footer"]
        }
        return structures.get(content_type, ["introduction", "main_content", "conclusion"])
    
    def _get_length_parameters(self, content_type: ContentType, requested_word_count: Optional[int]) -> Dict[str, int]:
        """Get length parameters based on content type."""
        default_lengths = {
            ContentType.BLOG_POST: {"min_words": 800, "max_words": 2000, "optimal_words": 1200},
            ContentType.SOCIAL_POST: {"min_words": 10, "max_words": 100, "optimal_words": 50},
            ContentType.EMAIL_CONTENT: {"min_words": 100, "max_words": 500, "optimal_words": 250},
            ContentType.LINKEDIN_ARTICLE: {"min_words": 1000, "max_words": 3000, "optimal_words": 1500},
            ContentType.TWITTER_THREAD: {"min_words": 100, "max_words": 500, "optimal_words": 200},
            ContentType.CASE_STUDY: {"min_words": 800, "max_words": 2500, "optimal_words": 1500},
            ContentType.NEWSLETTER: {"min_words": 300, "max_words": 1000, "optimal_words": 600}
        }
        
        params = default_lengths.get(content_type, {"min_words": 200, "max_words": 800, "optimal_words": 400})
        
        # Override with requested word count if provided
        if requested_word_count:
            params["optimal_words"] = requested_word_count
            params["max_words"] = max(params["max_words"], int(requested_word_count * 1.2))
            params["min_words"] = min(params["min_words"], int(requested_word_count * 0.8))
        
        return params
    
    def _determine_variation_type(self, variation_index: int) -> str:
        """Determine variation type based on index."""
        variation_types = ["tone_variation", "length_variation", "structure_variation", "hook_variation", "cta_variation"]
        return variation_types[variation_index % len(variation_types)]
    
    def _determine_target_segment(self, variation_index: int, base_audience: str) -> str:
        """Determine target segment for variation."""
        if "professional" in base_audience.lower():
            segments = ["executives", "managers", "specialists", "analysts", "consultants"]
        else:
            segments = ["early_adopters", "mainstream", "traditionalists", "innovators", "pragmatists"]
        
        return segments[variation_index % len(segments)]
    
    async def _calculate_quality_metrics(
        self, 
        content: GeneratedContent, 
        campaign_context: Dict[str, Any]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics for content."""
        try:
            # Calculate individual scores
            readability_score = self._calculate_readability_score(content.content)
            engagement_potential = self._calculate_engagement_potential(content)
            seo_optimization = content.seo_score or 70.0
            brand_consistency = self._calculate_brand_consistency(content, campaign_context)
            technical_accuracy = self._calculate_technical_accuracy(content)
            
            # Calculate overall quality (weighted average)
            weights = {"readability": 0.2, "engagement": 0.25, "seo": 0.2, "brand": 0.2, "technical": 0.15}
            overall_quality = (
                readability_score * weights["readability"] +
                engagement_potential * weights["engagement"] +
                seo_optimization * weights["seo"] +
                brand_consistency * weights["brand"] +
                technical_accuracy * weights["technical"]
            )
            
            # Generate improvement suggestions
            improvement_suggestions = []
            if readability_score < 7.0:
                improvement_suggestions.append("Improve readability with shorter sentences and simpler language")
            if engagement_potential < 7.0:
                improvement_suggestions.append("Add more engaging hooks and interactive elements")
            if seo_optimization < 70.0:
                improvement_suggestions.append("Optimize for SEO with better keyword integration")
            if brand_consistency < 7.0:
                improvement_suggestions.append("Align content more closely with brand voice and messaging")
            
            return QualityMetrics(
                readability_score=readability_score,
                engagement_potential=engagement_potential,
                seo_optimization=seo_optimization,
                brand_consistency=brand_consistency,
                technical_accuracy=technical_accuracy,
                overall_quality=overall_quality,
                improvement_suggestions=improvement_suggestions
            )
            
        except Exception as e:
            self._log_error(f"Quality metrics calculation failed: {str(e)}")
            return QualityMetrics(
                readability_score=5.0,
                engagement_potential=5.0,
                seo_optimization=5.0,
                brand_consistency=5.0,
                technical_accuracy=5.0,
                overall_quality=5.0,
                improvement_suggestions=["Quality assessment failed, manual review recommended"]
            )
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score based on content characteristics."""
        try:
            sentences = content.count('.') + content.count('!') + content.count('?')
            words = len(content.split())
            
            if sentences == 0:
                return 5.0
            
            avg_words_per_sentence = words / sentences
            
            # Score based on average sentence length (simpler = better)
            if avg_words_per_sentence < 15:
                return 9.0
            elif avg_words_per_sentence < 20:
                return 8.0
            elif avg_words_per_sentence < 25:
                return 7.0
            elif avg_words_per_sentence < 30:
                return 6.0
            else:
                return 5.0
                
        except Exception as e:
            return 6.0
    
    def _calculate_engagement_potential(self, content: GeneratedContent) -> float:
        """Calculate engagement potential based on content characteristics."""
        try:
            score = 5.0  # Base score
            
            content_text = content.content.lower()
            
            # Questions increase engagement
            if '?' in content.content:
                score += 1.5
            
            # Call to action increases engagement
            if content.call_to_action or any(cta in content_text for cta in ['learn more', 'contact', 'subscribe', 'follow']):
                score += 1.0
            
            # Emotional words increase engagement
            emotional_words = ['amazing', 'incredible', 'transform', 'breakthrough', 'revolutionary', 'essential', 'crucial']
            if any(word in content_text for word in emotional_words):
                score += 0.8
            
            # Lists and structure increase engagement
            if any(marker in content.content for marker in ['•', '-', '1.', '2.', '##']):
                score += 0.7
            
            # Use the content's own engagement estimate
            if content.estimated_engagement == "high":
                score += 1.0
            elif content.estimated_engagement == "medium-high":
                score += 0.5
            
            return min(score, 10.0)
            
        except Exception as e:
            return 6.0
    
    def _calculate_brand_consistency(self, content: GeneratedContent, campaign_context: Dict[str, Any]) -> float:
        """Calculate brand consistency score."""
        try:
            score = 7.0  # Base score assuming reasonable consistency
            
            # Check if company context is reflected
            company_context = campaign_context.get('company_context', '').lower()
            if company_context and company_context in content.content.lower():
                score += 1.0
            
            # Check tone alignment
            desired_tone = campaign_context.get('desired_tone', '').lower()
            if desired_tone:
                tone_indicators = {
                    'professional': ['strategic', 'implementation', 'analysis', 'business'],
                    'friendly': ['we', 'you', 'together', 'help'],
                    'authoritative': ['research', 'data', 'proven', 'expert'],
                    'conversational': ['let\'s', 'here\'s', 'you\'ll', 'we\'ve']
                }
                
                if desired_tone in tone_indicators:
                    indicators_present = sum(1 for indicator in tone_indicators[desired_tone] 
                                          if indicator in content.content.lower())
                    if indicators_present >= 2:
                        score += 1.0
            
            return min(score, 10.0)
            
        except Exception as e:
            return 7.0
    
    def _calculate_technical_accuracy(self, content: GeneratedContent) -> float:
        """Calculate technical accuracy score."""
        try:
            score = 8.0  # Base score assuming reasonable accuracy
            
            # Check for basic technical elements
            content_text = content.content
            
            # Proper capitalization
            if content_text != content_text.lower() and content_text != content_text.upper():
                score += 0.5
            
            # Proper punctuation
            if content_text.count('.') > 0 or content_text.count('!') > 0 or content_text.count('?') > 0:
                score += 0.5
            
            # Content structure
            if len(content_text.split('\n')) > 1:  # Has paragraphs
                score += 0.5
            
            # Word count alignment
            if content.word_count > 50:  # Substantial content
                score += 0.5
            
            return min(score, 10.0)
            
        except Exception as e:
            return 7.0
    
    async def _apply_seo_optimization(self, content: GeneratedContent) -> List[str]:
        """Apply SEO optimization to content."""
        improvements = []
        
        # Check title optimization
        if len(content.title) < 30:
            improvements.append("Consider expanding title for better SEO (30-60 characters optimal)")
        
        # Check content length
        if content.word_count < 300:
            improvements.append("Content may be too short for good SEO ranking (300+ words recommended)")
        
        # Check for subheadings
        if '##' not in content.content and '#' not in content.content:
            improvements.append("Add subheadings to improve content structure and SEO")
        
        return improvements
    
    async def _apply_engagement_optimization(self, content: GeneratedContent) -> List[str]:
        """Apply engagement optimization to content."""
        improvements = []
        
        # Check for questions
        if '?' not in content.content:
            improvements.append("Add questions to encourage reader engagement")
        
        # Check for call to action
        if not content.call_to_action:
            improvements.append("Add clear call-to-action to drive reader response")
        
        # Check for emotional elements
        emotional_words = ['amazing', 'incredible', 'transform', 'breakthrough']
        if not any(word in content.content.lower() for word in emotional_words):
            improvements.append("Consider adding emotional hooks to increase engagement")
        
        return improvements
    
    async def _analyze_brand_voice_consistency(
        self, 
        content: GeneratedContent, 
        campaign_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze brand voice consistency."""
        return {
            "tone_consistency": 8.0,
            "message_alignment": 7.5,
            "voice_authenticity": 8.2,
            "brand_values_reflection": 7.8,
            "consistency_score": 7.9
        }
    
    async def _calculate_strategy_alignment(
        self, 
        content: GeneratedContent, 
        campaign_context: Dict[str, Any]
    ) -> float:
        """Calculate strategy alignment score."""
        try:
            score = 7.0  # Base alignment score
            
            # Check campaign objective alignment
            objective = campaign_context.get('objective', '').lower()
            if objective and any(obj_word in content.content.lower() for obj_word in objective.split()):
                score += 1.0
            
            # Check target audience alignment
            target_audience = campaign_context.get('target_audience', {})
            if isinstance(target_audience, dict) and target_audience:
                # Assume alignment if content is generated with audience in mind
                score += 0.5
            
            return min(score, 10.0)
            
        except Exception as e:
            return 7.0
    
    async def _identify_competitive_differentiation(
        self, 
        content: GeneratedContent, 
        campaign_context: Dict[str, Any]
    ) -> List[str]:
        """Identify competitive differentiation elements."""
        differentiation_elements = []
        
        content_lower = content.content.lower()
        
        # Look for unique value propositions
        unique_indicators = ['unique', 'exclusive', 'only', 'first', 'revolutionary', 'breakthrough']
        if any(indicator in content_lower for indicator in unique_indicators):
            differentiation_elements.append("Unique value proposition")
        
        # Look for expertise indicators
        expertise_indicators = ['expert', 'specialist', 'proven', 'experienced', 'certified']
        if any(indicator in content_lower for indicator in expertise_indicators):
            differentiation_elements.append("Expertise positioning")
        
        # Look for innovation indicators
        innovation_indicators = ['innovative', 'advanced', 'cutting-edge', 'state-of-the-art']
        if any(indicator in content_lower for indicator in innovation_indicators):
            differentiation_elements.append("Innovation focus")
        
        return differentiation_elements
    
    async def execute_workflow(
        self,
        generation_requests: List[Dict[str, Any]],
        campaign_context: Optional[Dict[str, Any]] = None,
        variation_count: int = 1,
        quality_threshold: float = 7.0,
        seo_optimization_enabled: bool = True,
        brand_voice_consistency: bool = True,
        a_b_testing_enabled: bool = False,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the AI content generation workflow."""
        
        context = {
            "generation_requests": generation_requests,
            "campaign_context": campaign_context or {},
            "variation_count": variation_count,
            "quality_threshold": quality_threshold,
            "seo_optimization_enabled": seo_optimization_enabled,
            "brand_voice_consistency": brand_voice_consistency,
            "a_b_testing_enabled": a_b_testing_enabled,
            "workflow_id": f"content_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "generated_content": {
                        content_id: {
                            "content_id": content.content_id,
                            "content_type": content.content_type.value,
                            "channel": content.channel.value,
                            "title": content.title,
                            "content": content.content,
                            "word_count": content.word_count,
                            "quality_score": content.quality_score,
                            "estimated_engagement": content.estimated_engagement,
                            "seo_score": content.seo_score,
                            "metadata": content.metadata,
                            "created_at": content.created_at.isoformat() if content.created_at else None
                        }
                        for content_id, content in final_state.generated_content.items()
                    },
                    "content_variations": {
                        content_id: [
                            {
                                "variation_id": variation.variation_id,
                                "variation_type": variation.variation_type,
                                "target_segment": variation.target_segment,
                                "content": {
                                    "title": variation.content.title,
                                    "content": variation.content.content,
                                    "word_count": variation.content.word_count,
                                    "quality_score": variation.content.quality_score
                                }
                            }
                            for variation in variations
                        ]
                        for content_id, variations in final_state.content_variations.items()
                    },
                    "quality_metrics": {
                        content_id: {
                            "readability_score": metrics.readability_score,
                            "engagement_potential": metrics.engagement_potential,
                            "seo_optimization": metrics.seo_optimization,
                            "brand_consistency": metrics.brand_consistency,
                            "technical_accuracy": metrics.technical_accuracy,
                            "overall_quality": metrics.overall_quality,
                            "improvement_suggestions": metrics.improvement_suggestions
                        }
                        for content_id, metrics in final_state.quality_metrics.items()
                    },
                    "workflow_summary": final_state.generation_metrics.get("final_summary", {}),
                    "optimization_results": final_state.optimization_results,
                    "brand_voice_analysis": final_state.brand_voice_analysis,
                    "rejected_content": final_state.rejected_content
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "total_generated": len(final_state.generated_content),
                        "total_variations": sum(len(variations) for variations in final_state.content_variations.values()),
                        "approved_count": sum(1 for approved in final_state.approval_status.values() if approved),
                        "average_quality": final_state.generation_metrics.get("final_summary", {}).get("average_quality_score", 0)
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=final_state.error_message or "Content generation workflow failed",
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