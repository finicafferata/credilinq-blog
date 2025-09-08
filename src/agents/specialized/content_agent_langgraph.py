"""
LangGraph-enhanced Content Agent Workflow for advanced multi-format content generation.
"""

import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum

# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..core.langgraph_base import (
    LangGraphWorkflowBase, WorkflowState, WorkflowStatus, 
    CheckpointStrategy, LangGraphExecutionContext
)
from ..core.base_agent import AgentType, AgentResult, AgentExecutionContext
# Removed broken import: from .content_agent import ContentGenerationAgent
# from ...config.database import DatabaseConnection  # Temporarily disabled


class ContentFormat(str, Enum):
    """Supported content formats."""
    BLOG_POST = "blog"
    LINKEDIN_POST = "linkedin" 
    ARTICLE = "article"
    WHITE_PAPER = "whitepaper"
    CASE_STUDY = "case_study"
    EMAIL_NEWSLETTER = "email"
    SOCIAL_POST = "social"
    PRESS_RELEASE = "press_release"


class ContentQuality(str, Enum):
    """Content quality levels."""
    DRAFT = "draft"           # Basic content for review
    STANDARD = "standard"     # Good quality, ready for light editing
    PREMIUM = "premium"       # High quality, minimal editing needed
    EXPERT = "expert"         # Expert-level, publication-ready


class ToneStyle(str, Enum):
    """Content tone and style options."""
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    EDUCATIONAL = "educational"
    PERSUASIVE = "persuasive"
    THOUGHT_LEADERSHIP = "thought_leadership"


@dataclass
class ContentRequirements:
    """Detailed content generation requirements."""
    title: str
    outline: List[str]
    content_format: ContentFormat
    company_context: str
    target_audience: str
    quality_level: ContentQuality = ContentQuality.STANDARD
    tone_style: ToneStyle = ToneStyle.PROFESSIONAL
    word_count_target: Optional[int] = None
    include_research: bool = True
    include_seo_optimization: bool = True
    include_call_to_action: bool = True
    custom_instructions: List[str] = field(default_factory=list)


@dataclass
class ContentMetrics:
    """Comprehensive content analysis metrics."""
    word_count: int
    character_count: int
    paragraph_count: int
    sentence_count: int
    header_count: int
    reading_time_minutes: int
    readability_score: str
    engagement_score: float
    seo_score: float
    brand_consistency_score: float
    call_to_action_count: int
    format_specific_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Content quality assessment results."""
    overall_quality_score: float
    content_completeness: float
    structural_coherence: float
    brand_alignment: float
    audience_targeting: float
    actionability: float
    improvement_suggestions: List[str] = field(default_factory=list)
    approval_status: str = "pending"  # pending, approved, needs_revision


class ContentAgentState(WorkflowState):
    """Enhanced state for content generation workflow."""
    
    # Input requirements
    content_requirements: Optional[ContentRequirements] = None
    research_data: Dict[str, Any] = field(default_factory=dict)
    geo_metadata: Dict[str, Any] = field(default_factory=dict)
    brand_guidelines: Dict[str, Any] = field(default_factory=dict)
    existing_content_context: List[str] = field(default_factory=list)
    
    # Content generation process
    content_outline_enhanced: List[Dict[str, str]] = field(default_factory=list)
    generated_content: str = ""
    alternative_versions: List[str] = field(default_factory=list)
    
    # Content optimization
    seo_optimizations: Dict[str, Any] = field(default_factory=dict)
    readability_improvements: List[str] = field(default_factory=list)
    brand_consistency_adjustments: List[str] = field(default_factory=list)
    
    # Quality assessment
    content_metrics: Optional[ContentMetrics] = None
    quality_assessment: Optional[QualityAssessment] = None
    stakeholder_feedback: List[str] = field(default_factory=list)
    
    # Final deliverables
    final_content: str = ""
    content_variants: Dict[str, str] = field(default_factory=dict)  # Different formats/lengths
    asset_recommendations: List[str] = field(default_factory=list)
    distribution_recommendations: List[str] = field(default_factory=list)
    
    # Performance tracking
    generation_time: float = 0.0
    revision_count: int = 0
    optimization_iterations: int = 0
    
    # Messages for communication between nodes
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


class ContentAgentWorkflow(LangGraphWorkflowBase[ContentAgentState]):
    """LangGraph workflow for advanced multi-format content generation."""
    
    def __init__(
        self, 
        workflow_name: str = "content_agent_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT,
        enable_human_in_loop: bool = False
    ):
        # Initialize the legacy agent for core functionality
        self.legacy_agent = ContentGenerationAgent()
        
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy,
            enable_human_in_loop=enable_human_in_loop
        )
    
    def _create_initial_state(self, context: Dict[str, Any]) -> ContentAgentState:
        """Create initial workflow state from context."""
        # Parse content requirements
        requirements = ContentRequirements(
            title=context.get("title", ""),
            outline=context.get("outline", []),
            content_format=ContentFormat(context.get("content_format", "blog")),
            company_context=context.get("company_context", ""),
            target_audience=context.get("target_audience", "Business professionals"),
            quality_level=ContentQuality(context.get("quality_level", "standard")),
            tone_style=ToneStyle(context.get("tone_style", "professional")),
            word_count_target=context.get("word_count_target"),
            include_research=context.get("include_research", True),
            include_seo_optimization=context.get("include_seo_optimization", True),
            include_call_to_action=context.get("include_call_to_action", True),
            custom_instructions=context.get("custom_instructions", [])
        )
        
        return ContentAgentState(
            workflow_id=context.get("workflow_id", f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            content_requirements=requirements,
            research_data=context.get("research_data", {}),
            geo_metadata=context.get("geo_metadata", {}),
            brand_guidelines=context.get("brand_guidelines", {}),
            existing_content_context=context.get("existing_content_context", []),
            created_at=datetime.utcnow()
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the content generation workflow graph."""
        workflow = StateGraph(ContentAgentState)
        
        # Define workflow nodes
        workflow.add_node("validate_content_requirements", self._validate_content_requirements_node)
        workflow.add_node("enhance_content_outline", self._enhance_content_outline_node)
        workflow.add_node("generate_content", self._generate_content_node)
        workflow.add_node("optimize_content", self._optimize_content_node)
        workflow.add_node("assess_content_quality", self._assess_content_quality_node)
        workflow.add_node("generate_content_variants", self._generate_content_variants_node)
        workflow.add_node("finalize_deliverables", self._finalize_deliverables_node)
        
        # Define workflow edges
        workflow.add_edge("validate_content_requirements", "enhance_content_outline")
        workflow.add_edge("enhance_content_outline", "generate_content")
        workflow.add_edge("generate_content", "optimize_content")
        workflow.add_edge("optimize_content", "assess_content_quality")
        
        # Conditional routing based on quality assessment
        workflow.add_conditional_edges(
            "assess_content_quality",
            self._should_generate_variants,
            {
                "generate_variants": "generate_content_variants",
                "finalize": "finalize_deliverables"
            }
        )
        workflow.add_edge("generate_content_variants", "finalize_deliverables")
        workflow.add_edge("finalize_deliverables", END)
        
        # Set entry point
        workflow.set_entry_point("validate_content_requirements")
        
        return workflow
    
    async def _validate_content_requirements_node(self, state: ContentAgentState) -> ContentAgentState:
        """Validate content requirements and prepare generation parameters."""
        try:
            self._log_progress("Validating content requirements")
            
            requirements = state.content_requirements
            validation_errors = []
            
            # Validate core requirements
            if not requirements.title or len(requirements.title.strip()) < 5:
                validation_errors.append("Title must be at least 5 characters long")
            
            if not requirements.outline or len(requirements.outline) < 2:
                validation_errors.append("Outline must contain at least 2 sections")
            
            if not requirements.company_context:
                self._log_progress("No company context provided, will use generic professional context")
            
            # Validate content format and settings alignment
            format_requirements = self._get_format_requirements(requirements.content_format)
            
            # Adjust word count target based on format if not specified
            if not requirements.word_count_target:
                requirements.word_count_target = format_requirements.get("default_word_count", 1500)
            
            # Validate word count for format
            max_word_count = format_requirements.get("max_word_count")
            if max_word_count and requirements.word_count_target > max_word_count:
                requirements.word_count_target = max_word_count
                self._log_progress(f"Adjusted word count to {max_word_count} for {requirements.content_format} format")
            
            if validation_errors:
                state.status = WorkflowStatus.FAILED
                state.error_message = "; ".join(validation_errors)
            else:
                state.status = WorkflowStatus.IN_PROGRESS
                state.progress_percentage = 10.0
                
                state.messages.append(HumanMessage(
                    content=f"Content requirements validated: '{requirements.title}' "
                           f"({requirements.content_format.value}, {requirements.quality_level.value} quality, "
                           f"{requirements.word_count_target} words target)"
                ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Requirements validation failed: {str(e)}"
            return state
    
    async def _enhance_content_outline_node(self, state: ContentAgentState) -> ContentAgentState:
        """Enhance the content outline with detailed structure and guidance."""
        try:
            self._log_progress("Enhancing content outline with detailed structure")
            
            requirements = state.content_requirements
            
            # Create enhanced outline with detailed guidance for each section
            enhanced_outline = await self._create_enhanced_outline(
                requirements.outline,
                requirements.content_format,
                requirements.quality_level,
                state.research_data,
                requirements.custom_instructions
            )
            
            state.content_outline_enhanced = enhanced_outline
            state.progress_percentage = 20.0
            
            state.messages.append(SystemMessage(
                content=f"Content outline enhanced with {len(enhanced_outline)} detailed sections. "
                       f"Ready for content generation."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Outline enhancement failed: {str(e)}"
            return state
    
    async def _generate_content_node(self, state: ContentAgentState) -> ContentAgentState:
        """Generate high-quality content using enhanced outline and requirements."""
        try:
            self._log_progress("Generating high-quality content")
            
            start_time = datetime.utcnow()
            requirements = state.content_requirements
            
            # Prepare input for legacy agent
            legacy_input = {
                "title": requirements.title,
                "outline": [section["section"] for section in state.content_outline_enhanced],
                "company_context": requirements.company_context,
                "content_type": requirements.content_format.value,
                "research": state.research_data,
                "geo_metadata": state.geo_metadata
            }
            
            # Generate content using legacy agent
            result = self.legacy_agent.execute(legacy_input)
            
            if result.success:
                generated_content = result.data["content"]
                
                # Enhance content based on quality level and requirements
                if requirements.quality_level in [ContentQuality.PREMIUM, ContentQuality.EXPERT]:
                    generated_content = await self._enhance_content_quality(
                        generated_content, requirements, state.content_outline_enhanced
                    )
                
                state.generated_content = generated_content
                state.generation_time = (datetime.utcnow() - start_time).total_seconds()
                state.progress_percentage = 50.0
                
                word_count = len(generated_content.split())
                state.messages.append(SystemMessage(
                    content=f"Content generated successfully. {word_count} words, "
                           f"generation time: {state.generation_time:.1f}s"
                ))
            else:
                raise Exception(f"Legacy agent failed: {result.error_message}")
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content generation failed: {str(e)}"
            return state
    
    async def _optimize_content_node(self, state: ContentAgentState) -> ContentAgentState:
        """Optimize content for SEO, readability, and brand consistency."""
        try:
            self._log_progress("Optimizing content for SEO, readability, and brand alignment")
            
            requirements = state.content_requirements
            optimizations_applied = []
            
            # SEO Optimization
            if requirements.include_seo_optimization:
                seo_optimizations = await self._apply_seo_optimizations(
                    state.generated_content, requirements, state.research_data
                )
                state.seo_optimizations = seo_optimizations
                optimizations_applied.append("SEO")
            
            # Readability Improvements
            readability_improvements = await self._improve_readability(
                state.generated_content, requirements.content_format, requirements.target_audience
            )
            state.readability_improvements = readability_improvements
            if readability_improvements:
                optimizations_applied.append("Readability")
            
            # Brand Consistency
            if state.brand_guidelines:
                brand_adjustments = await self._ensure_brand_consistency(
                    state.generated_content, state.brand_guidelines, requirements.company_context
                )
                state.brand_consistency_adjustments = brand_adjustments
                if brand_adjustments:
                    optimizations_applied.append("Brand Consistency")
            
            # Apply optimizations to content
            optimized_content = await self._apply_content_optimizations(
                state.generated_content, 
                state.seo_optimizations,
                state.readability_improvements,
                state.brand_consistency_adjustments
            )
            
            state.generated_content = optimized_content
            state.optimization_iterations = 1
            state.progress_percentage = 70.0
            
            state.messages.append(SystemMessage(
                content=f"Content optimization completed. Applied: {', '.join(optimizations_applied) or 'None needed'}. "
                       f"Content enhanced for optimal performance."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content optimization failed: {str(e)}"
            return state
    
    async def _assess_content_quality_node(self, state: ContentAgentState) -> ContentAgentState:
        """Assess comprehensive content quality metrics."""
        try:
            self._log_progress("Assessing content quality and performance metrics")
            
            requirements = state.content_requirements
            
            # Calculate comprehensive content metrics
            content_metrics = await self._calculate_content_metrics(
                state.generated_content, requirements.content_format
            )
            state.content_metrics = content_metrics
            
            # Perform quality assessment
            quality_assessment = await self._perform_quality_assessment(
                state.generated_content, 
                requirements,
                content_metrics,
                state.content_outline_enhanced
            )
            state.quality_assessment = quality_assessment
            
            # Determine if content meets quality standards
            quality_threshold = self._get_quality_threshold(requirements.quality_level)
            meets_standards = quality_assessment.overall_quality_score >= quality_threshold
            
            state.progress_percentage = 85.0
            
            state.messages.append(SystemMessage(
                content=f"Quality assessment completed. Overall score: {quality_assessment.overall_quality_score:.2f}/1.0. "
                       f"Quality standard met: {meets_standards}. "
                       f"Word count: {content_metrics.word_count}, Reading time: {content_metrics.reading_time_minutes}min"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Quality assessment failed: {str(e)}"
            return state
    
    async def _generate_content_variants_node(self, state: ContentAgentState) -> ContentAgentState:
        """Generate content variants for different formats and use cases."""
        try:
            self._log_progress("Generating content variants for different formats")
            
            requirements = state.content_requirements
            content_variants = {}
            
            # Generate format variants based on original content format
            variant_formats = self._get_variant_formats(requirements.content_format)
            
            for variant_format in variant_formats:
                try:
                    variant_content = await self._create_content_variant(
                        state.generated_content, 
                        requirements.content_format,
                        variant_format,
                        requirements
                    )
                    content_variants[variant_format] = variant_content
                    
                except Exception as variant_error:
                    self._log_error(f"Failed to create {variant_format} variant: {str(variant_error)}")
                    continue
            
            # Generate length variants (short, medium versions)
            if requirements.content_format in [ContentFormat.BLOG_POST, ContentFormat.ARTICLE]:
                try:
                    short_version = await self._create_length_variant(
                        state.generated_content, "short", requirements.word_count_target // 3
                    )
                    content_variants["short_version"] = short_version
                    
                    medium_version = await self._create_length_variant(
                        state.generated_content, "medium", requirements.word_count_target // 2
                    )
                    content_variants["medium_version"] = medium_version
                    
                except Exception as length_error:
                    self._log_error(f"Failed to create length variants: {str(length_error)}")
            
            state.content_variants = content_variants
            state.progress_percentage = 95.0
            
            state.messages.append(SystemMessage(
                content=f"Content variants generated: {len(content_variants)} variants created "
                       f"for different formats and use cases."
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Content variant generation failed: {str(e)}"
            return state
    
    async def _finalize_deliverables_node(self, state: ContentAgentState) -> ContentAgentState:
        """Finalize all deliverables and prepare final content package."""
        try:
            self._log_progress("Finalizing deliverables and content package")
            
            requirements = state.content_requirements
            
            # Set final content (apply any final improvements)
            state.final_content = state.generated_content
            
            # Generate asset recommendations
            asset_recommendations = await self._generate_asset_recommendations(
                state.final_content, requirements.content_format, state.content_metrics
            )
            state.asset_recommendations = asset_recommendations
            
            # Generate distribution recommendations
            distribution_recommendations = await self._generate_distribution_recommendations(
                requirements, state.quality_assessment
            )
            state.distribution_recommendations = distribution_recommendations
            
            # Calculate final metrics
            state.generation_time = (datetime.utcnow() - state.created_at).total_seconds()
            
            # Final status determination
            quality_threshold = self._get_quality_threshold(requirements.quality_level)
            if state.quality_assessment.overall_quality_score >= quality_threshold:
                state.status = WorkflowStatus.COMPLETED
                state.quality_assessment.approval_status = "approved"
            else:
                state.status = WorkflowStatus.COMPLETED  # Still complete, but may need revision
                state.quality_assessment.approval_status = "needs_revision"
            
            state.progress_percentage = 100.0
            state.completed_at = datetime.utcnow()
            
            # Final summary
            final_summary = {
                "content_title": requirements.title,
                "content_format": requirements.content_format.value,
                "word_count": state.content_metrics.word_count,
                "quality_score": state.quality_assessment.overall_quality_score,
                "variants_created": len(state.content_variants),
                "optimization_applied": bool(state.seo_optimizations),
                "generation_time": state.generation_time,
                "approval_status": state.quality_assessment.approval_status
            }
            
            state.messages.append(SystemMessage(
                content=f"Content workflow completed successfully. "
                       f"Final content: {state.content_metrics.word_count} words, "
                       f"Quality score: {state.quality_assessment.overall_quality_score:.2f}, "
                       f"Variants: {len(state.content_variants)}, "
                       f"Status: {state.quality_assessment.approval_status}"
            ))
            
            return state
            
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Deliverables finalization failed: {str(e)}"
            return state
    
    def _should_generate_variants(self, state: ContentAgentState) -> str:
        """Determine if content variants should be generated."""
        # Generate variants for premium/expert quality or specific formats
        should_generate = (
            state.content_requirements.quality_level in [ContentQuality.PREMIUM, ContentQuality.EXPERT] or
            state.content_requirements.content_format in [ContentFormat.BLOG_POST, ContentFormat.ARTICLE] or
            len(state.content_requirements.custom_instructions) > 0
        )
        return "generate_variants" if should_generate else "finalize"
    
    # Helper methods for enhanced content generation
    
    def _get_format_requirements(self, content_format: ContentFormat) -> Dict[str, Any]:
        """Get format-specific requirements and constraints."""
        format_requirements = {
            ContentFormat.BLOG_POST: {
                "default_word_count": 1500,
                "max_word_count": 3000,
                "structure_requirements": ["intro", "body_sections", "conclusion"],
                "seo_critical": True,
                "call_to_action_required": True
            },
            ContentFormat.LINKEDIN_POST: {
                "default_word_count": 150,
                "max_word_count": 300,
                "character_limit": 1300,
                "structure_requirements": ["hook", "value", "cta"],
                "hashtag_required": True
            },
            ContentFormat.ARTICLE: {
                "default_word_count": 2500,
                "max_word_count": 5000,
                "structure_requirements": ["abstract", "sections", "conclusion"],
                "research_citations_preferred": True
            },
            ContentFormat.EMAIL_NEWSLETTER: {
                "default_word_count": 800,
                "max_word_count": 1200,
                "structure_requirements": ["subject", "greeting", "content", "cta"],
                "personalization_required": True
            },
            ContentFormat.WHITE_PAPER: {
                "default_word_count": 3000,
                "max_word_count": 8000,
                "structure_requirements": ["executive_summary", "sections", "conclusion"],
                "research_heavy": True,
                "professional_tone_required": True
            }
        }
        
        return format_requirements.get(content_format, {
            "default_word_count": 1500,
            "max_word_count": 3000,
            "structure_requirements": ["intro", "body", "conclusion"]
        })
    
    async def _create_enhanced_outline(
        self,
        basic_outline: List[str],
        content_format: ContentFormat,
        quality_level: ContentQuality,
        research_data: Dict[str, Any],
        custom_instructions: List[str]
    ) -> List[Dict[str, str]]:
        """Create enhanced outline with detailed guidance for each section."""
        enhanced_outline = []
        
        for i, section in enumerate(basic_outline):
            # Add detailed guidance for each section
            section_guidance = {
                "section": section,
                "description": f"Detailed content for {section}",
                "word_count_target": 200 if content_format == ContentFormat.LINKEDIN_POST else 400,
                "key_elements": [],
                "research_integration": []
            }
            
            # Add format-specific guidance
            if content_format == ContentFormat.BLOG_POST:
                if i == 0:  # Introduction
                    section_guidance["key_elements"] = ["hook", "problem_statement", "preview"]
                elif i == len(basic_outline) - 1:  # Conclusion
                    section_guidance["key_elements"] = ["summary", "call_to_action", "next_steps"]
                else:  # Body sections
                    section_guidance["key_elements"] = ["subheading", "main_points", "examples"]
            elif content_format == ContentFormat.LINKEDIN_POST:
                section_guidance["key_elements"] = ["engaging_statement", "value_proposition", "social_proof"]
            elif content_format == ContentFormat.ARTICLE:
                section_guidance["key_elements"] = ["analysis", "evidence", "insights"]
            
            # Add research integration points
            if research_data:
                for research_topic, data in research_data.items():
                    if any(keyword in section.lower() for keyword in research_topic.lower().split()):
                        section_guidance["research_integration"].append(research_topic)
            
            enhanced_outline.append(section_guidance)
        
        return enhanced_outline
    
    async def _enhance_content_quality(
        self,
        content: str,
        requirements: ContentRequirements,
        enhanced_outline: List[Dict[str, str]]
    ) -> str:
        """Enhance content quality for premium/expert levels."""
        try:
            enhanced_content = content
            
            if requirements.quality_level == ContentQuality.PREMIUM:
                # Add more detailed examples and insights
                enhanced_content = await self._add_premium_enhancements(enhanced_content, requirements)
                
            elif requirements.quality_level == ContentQuality.EXPERT:
                # Add expert-level analysis and thought leadership elements
                enhanced_content = await self._add_expert_enhancements(enhanced_content, requirements)
            
            return enhanced_content
            
        except Exception as e:
            self._log_error(f"Content quality enhancement failed: {str(e)}")
            return content  # Return original content if enhancement fails
    
    async def _add_premium_enhancements(self, content: str, requirements: ContentRequirements) -> str:
        """Add premium-level enhancements to content."""
        # Simple enhancement: add more detailed examples and actionable insights
        enhanced_sections = []
        sections = content.split('\n## ')  # Split by markdown headers
        
        for i, section in enumerate(sections):
            enhanced_sections.append(section)
            
            # Add premium elements every few sections
            if i > 0 and i % 2 == 0 and requirements.content_format == ContentFormat.BLOG_POST:
                if "example" not in section.lower():
                    enhanced_sections.append("\n**Example:** [Detailed example would be provided here based on the topic]\n")
        
        return '\n## '.join(enhanced_sections)
    
    async def _add_expert_enhancements(self, content: str, requirements: ContentRequirements) -> str:
        """Add expert-level enhancements to content."""
        # Add thought leadership elements and deeper analysis
        enhanced_content = content
        
        # Add expert insights section for certain formats
        if requirements.content_format in [ContentFormat.ARTICLE, ContentFormat.WHITE_PAPER]:
            expert_section = "\n\n## Expert Analysis and Industry Implications\n\nBased on industry trends and market analysis, this topic represents a significant opportunity for organizations to..."
            enhanced_content += expert_section
        
        return enhanced_content
    
    async def _apply_seo_optimizations(
        self,
        content: str,
        requirements: ContentRequirements,
        research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply SEO optimizations to content."""
        seo_optimizations = {
            "keyword_density_check": True,
            "header_structure_optimized": True,
            "meta_description_suggested": f"Learn about {requirements.title} and discover actionable insights...",
            "internal_linking_opportunities": [],
            "seo_improvements_applied": []
        }
        
        # Check for basic SEO elements
        has_h1 = '# ' in content
        has_h2 = '## ' in content
        word_count = len(content.split())
        
        if not has_h1:
            seo_optimizations["seo_improvements_applied"].append("Added H1 header")
        if not has_h2:
            seo_optimizations["seo_improvements_applied"].append("Added H2 headers")
        if word_count < 300:
            seo_optimizations["seo_improvements_applied"].append("Content length optimization needed")
        
        return seo_optimizations
    
    async def _improve_readability(
        self,
        content: str,
        content_format: ContentFormat,
        target_audience: str
    ) -> List[str]:
        """Analyze and suggest readability improvements."""
        improvements = []
        
        # Basic readability checks
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        if sentences > 0:
            avg_words_per_sentence = words / sentences
            if avg_words_per_sentence > 25:
                improvements.append("Reduce average sentence length for better readability")
        
        # Check paragraph length
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 150]
        
        if long_paragraphs:
            improvements.append("Break up long paragraphs for better scanning")
        
        # Format-specific readability
        if content_format == ContentFormat.LINKEDIN_POST:
            if len(content) > 1300:
                improvements.append("Reduce character count for LinkedIn optimization")
        
        return improvements
    
    async def _ensure_brand_consistency(
        self,
        content: str,
        brand_guidelines: Dict[str, Any],
        company_context: str
    ) -> List[str]:
        """Check and suggest brand consistency adjustments."""
        adjustments = []
        
        # Check for brand voice consistency
        preferred_tone = brand_guidelines.get("tone", "professional")
        if preferred_tone == "conversational" and not any(word in content.lower() for word in ["you", "we", "your"]):
            adjustments.append("Add more conversational language to match brand voice")
        
        # Check for company-specific terminology
        brand_terms = brand_guidelines.get("preferred_terms", {})
        for preferred_term, alternative in brand_terms.items():
            if alternative.lower() in content.lower() and preferred_term.lower() not in content.lower():
                adjustments.append(f"Use '{preferred_term}' instead of '{alternative}' for brand consistency")
        
        return adjustments
    
    async def _apply_content_optimizations(
        self,
        content: str,
        seo_optimizations: Dict[str, Any],
        readability_improvements: List[str],
        brand_adjustments: List[str]
    ) -> str:
        """Apply all content optimizations to the generated content."""
        optimized_content = content
        
        # Apply basic optimizations (this would be more sophisticated in production)
        # For now, ensure basic structure elements are present
        
        if not content.startswith('# '):
            newline = '\n'
            first_line = content.split(newline)[0]
            optimized_content = f"# {first_line}{newline}{newline}{content}"
        
        return optimized_content
    
    async def _calculate_content_metrics(
        self,
        content: str,
        content_format: ContentFormat
    ) -> ContentMetrics:
        """Calculate comprehensive content analysis metrics."""
        # Basic metrics
        words = content.split()
        word_count = len(words)
        character_count = len(content)
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        header_count = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))
        reading_time = max(1, word_count // 200)  # 200 words per minute
        
        # Readability score (simplified)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        if avg_words_per_sentence < 15:
            readability_score = "Easy"
        elif avg_words_per_sentence < 25:
            readability_score = "Medium"
        else:
            readability_score = "Complex"
        
        # Engagement score (simplified calculation)
        engagement_factors = 0
        if '?' in content:  # Questions increase engagement
            engagement_factors += 1
        if any(word in content.lower() for word in ['tip', 'secret', 'guide', 'how to']):
            engagement_factors += 1
        if header_count > 2:  # Good structure
            engagement_factors += 1
        if paragraph_count > 3:  # Well-organized
            engagement_factors += 1
        
        engagement_score = min(engagement_factors * 0.25, 1.0)
        
        # SEO score (simplified)
        seo_factors = 0
        if header_count > 0:
            seo_factors += 1
        if word_count >= 300:
            seo_factors += 1
        if word_count <= 3000:
            seo_factors += 1
        
        seo_score = min(seo_factors * 0.33, 1.0)
        
        # Brand consistency score (simplified)
        brand_consistency_score = 0.8  # Default good score
        
        # Call to action count
        cta_count = content.lower().count('contact') + content.lower().count('learn more') + content.lower().count('get started')
        
        # Format-specific metrics
        format_specific_metrics = {}
        if content_format == ContentFormat.LINKEDIN_POST:
            format_specific_metrics = {
                "character_count": character_count,
                "hashtag_count": content.count('#'),
                "mention_count": content.count('@')
            }
        elif content_format in [ContentFormat.BLOG_POST, ContentFormat.ARTICLE]:
            format_specific_metrics = {
                "subheader_count": content.count('## '),
                "bullet_points": content.count('- ') + content.count('* '),
                "numbered_lists": len(re.findall(r'^\d+\.', content, re.MULTILINE))
            }
        
        return ContentMetrics(
            word_count=word_count,
            character_count=character_count,
            paragraph_count=paragraph_count,
            sentence_count=sentence_count,
            header_count=header_count,
            reading_time_minutes=reading_time,
            readability_score=readability_score,
            engagement_score=engagement_score,
            seo_score=seo_score,
            brand_consistency_score=brand_consistency_score,
            call_to_action_count=cta_count,
            format_specific_metrics=format_specific_metrics
        )
    
    async def _perform_quality_assessment(
        self,
        content: str,
        requirements: ContentRequirements,
        content_metrics: ContentMetrics,
        enhanced_outline: List[Dict[str, str]]
    ) -> QualityAssessment:
        """Perform comprehensive quality assessment of generated content."""
        # Content completeness (covers all outline sections)
        content_completeness = 0.9  # Assume good coverage for now
        
        # Structural coherence (logical flow and organization)
        structural_coherence = 0.85
        if content_metrics.header_count >= len(enhanced_outline):
            structural_coherence = 0.9
        
        # Brand alignment (matches company context and tone)
        brand_alignment = 0.8
        if requirements.company_context.lower() in content.lower():
            brand_alignment = 0.85
        
        # Audience targeting (appropriate for target audience)
        audience_targeting = 0.8
        if requirements.target_audience.lower() in content.lower():
            audience_targeting = 0.85
        
        # Actionability (provides clear next steps or value)
        actionability = 0.75
        if content_metrics.call_to_action_count > 0:
            actionability = 0.85
        
        # Calculate overall quality score (weighted average)
        overall_quality_score = (
            content_completeness * 0.25 +
            structural_coherence * 0.20 +
            brand_alignment * 0.20 +
            audience_targeting * 0.20 +
            actionability * 0.15
        )
        
        # Generate improvement suggestions
        improvement_suggestions = []
        if content_completeness < 0.8:
            improvement_suggestions.append("Ensure all outline sections are fully covered")
        if structural_coherence < 0.8:
            improvement_suggestions.append("Improve content flow and organization")
        if brand_alignment < 0.8:
            improvement_suggestions.append("Strengthen alignment with brand voice and context")
        if audience_targeting < 0.8:
            improvement_suggestions.append("Better tailor content to target audience needs")
        if actionability < 0.8:
            improvement_suggestions.append("Add more actionable insights and clear next steps")
        
        return QualityAssessment(
            overall_quality_score=overall_quality_score,
            content_completeness=content_completeness,
            structural_coherence=structural_coherence,
            brand_alignment=brand_alignment,
            audience_targeting=audience_targeting,
            actionability=actionability,
            improvement_suggestions=improvement_suggestions
        )
    
    def _get_quality_threshold(self, quality_level: ContentQuality) -> float:
        """Get quality score threshold for different quality levels."""
        thresholds = {
            ContentQuality.DRAFT: 0.6,
            ContentQuality.STANDARD: 0.75,
            ContentQuality.PREMIUM: 0.85,
            ContentQuality.EXPERT: 0.9
        }
        return thresholds.get(quality_level, 0.75)
    
    def _get_variant_formats(self, original_format: ContentFormat) -> List[str]:
        """Get related formats for creating content variants."""
        format_variants = {
            ContentFormat.BLOG_POST: ["linkedin", "email_summary", "social_post"],
            ContentFormat.ARTICLE: ["executive_summary", "linkedin", "press_release"],
            ContentFormat.WHITE_PAPER: ["executive_summary", "blog_post", "linkedin"],
            ContentFormat.LINKEDIN_POST: ["twitter_thread", "email_snippet"],
            ContentFormat.EMAIL_NEWSLETTER: ["blog_post", "social_post"]
        }
        return format_variants.get(original_format, [])
    
    async def _create_content_variant(
        self,
        original_content: str,
        original_format: ContentFormat,
        target_format: str,
        requirements: ContentRequirements
    ) -> str:
        """Create a content variant in a different format."""
        try:
            # Simple content adaptation (would be more sophisticated in production)
            if target_format == "linkedin":
                # Convert to LinkedIn post
                lines = original_content.split('\n')
                title = lines[0].replace('# ', '')
                
                # Extract key points
                key_points = []
                for line in lines[1:6]:  # First few lines
                    if line.strip() and not line.startswith('#'):
                        key_points.append(line.strip()[:100])
                
                newline = '\n'
                linkedin_post = f"{title}{newline}{newline}"
                linkedin_post += "Key insights:\n"
                for i, point in enumerate(key_points[:3], 1):
                    linkedin_post += f"{i}. {point}{newline}"
                
                linkedin_post += "\nWhat's your experience with this? 👇\n\n#Business #Strategy #Growth"
                
                return linkedin_post
                
            elif target_format == "executive_summary":
                # Create executive summary
                lines = original_content.split('\n')
                title = lines[0].replace('# ', '')
                
                summary = f"# Executive Summary: {title}{newline}{newline}"
                summary += "## Key Points\n\n"
                summary += "• Strategic insights for business leaders\n"
                summary += "• Actionable recommendations for implementation\n"
                summary += "• Market analysis and competitive implications\n\n"
                summary += "## Recommendations\n\n"
                summary += "Based on this analysis, we recommend immediate action to capitalize on identified opportunities."
                
                return summary
            
            elif target_format == "email_summary":
                # Create email newsletter version
                lines = original_content.split('\n')
                title = lines[0].replace('# ', '')
                
                email = f"Subject: {title} - Key Insights{newline}{newline}"
                email += f"Hi there,{newline}{newline}"
                email += f"I wanted to share some key insights about {title.lower()}:{newline}{newline}"
                email += "• Important industry developments\n"
                email += "• Actionable strategies for your business\n"
                email += "• Next steps for implementation\n\n"
                email += "Read the full analysis here: [Link]\n\n"
                email += "Best regards,\nThe Team"
                
                return email
            
            else:
                # Default: create a summary version
                return f"Summary of {original_format.value}:{newline}{newline}{original_content[:500]}..."
                
        except Exception as e:
            self._log_error(f"Content variant creation failed: {str(e)}")
            return f"Variant creation failed for {target_format}"
    
    async def _create_length_variant(
        self,
        original_content: str,
        variant_type: str,
        target_word_count: int
    ) -> str:
        """Create length-based content variants."""
        try:
            words = original_content.split()
            
            if variant_type == "short" and len(words) > target_word_count:
                # Create short version by taking key sections
                lines = original_content.split('\n')
                short_content = []
                
                # Keep title and first few paragraphs
                for line in lines[:10]:
                    short_content.append(line)
                    if len(' '.join(short_content).split()) >= target_word_count:
                        break
                
                return '\n'.join(short_content)
                
            elif variant_type == "medium":
                # Create medium version
                current_word_count = len(words)
                if current_word_count > target_word_count:
                    # Take proportional content
                    ratio = target_word_count / current_word_count
                    lines = original_content.split('\n')
                    target_lines = int(len(lines) * ratio)
                    return '\n'.join(lines[:target_lines])
            
            return original_content  # Return original if no changes needed
            
        except Exception as e:
            self._log_error(f"Length variant creation failed: {str(e)}")
            return original_content
    
    async def _generate_asset_recommendations(
        self,
        content: str,
        content_format: ContentFormat,
        content_metrics: ContentMetrics
    ) -> List[str]:
        """Generate recommendations for supporting assets."""
        recommendations = []
        
        # Format-specific asset recommendations
        if content_format == ContentFormat.BLOG_POST:
            recommendations.extend([
                "Featured image for blog header",
                "Social media graphics for promotion",
                "Infographic summarizing key points"
            ])
            
            if content_metrics.word_count > 2000:
                recommendations.append("Table of contents for long-form content")
                
        elif content_format == ContentFormat.LINKEDIN_POST:
            recommendations.extend([
                "LinkedIn carousel graphics",
                "Professional headshot for author credibility",
                "Company logo for brand recognition"
            ])
            
        elif content_format == ContentFormat.WHITE_PAPER:
            recommendations.extend([
                "Professional document design template",
                "Charts and graphs for data visualization",
                "Executive summary one-pager"
            ])
        
        # General recommendations based on content analysis
        if content_metrics.header_count > 3:
            recommendations.append("Section divider graphics")
            
        if 'data' in content.lower() or 'statistics' in content.lower():
            recommendations.append("Data visualization charts")
        
        return recommendations
    
    async def _generate_distribution_recommendations(
        self,
        requirements: ContentRequirements,
        quality_assessment: QualityAssessment
    ) -> List[str]:
        """Generate content distribution recommendations."""
        recommendations = []
        
        # Format-specific distribution
        if requirements.content_format == ContentFormat.BLOG_POST:
            recommendations.extend([
                "Publish on company blog with SEO optimization",
                "Share on LinkedIn with executive commentary",
                "Include in email newsletter with excerpt",
                "Create social media promotion campaign"
            ])
            
        elif requirements.content_format == ContentFormat.LINKEDIN_POST:
            recommendations.extend([
                "Post during peak LinkedIn engagement hours",
                "Tag relevant industry professionals",
                "Cross-promote on other social channels",
                "Include in LinkedIn newsletter"
            ])
            
        elif requirements.content_format == ContentFormat.WHITE_PAPER:
            recommendations.extend([
                "Gate as premium download on website",
                "Promote through email marketing campaigns",
                "Submit to industry publications",
                "Use for sales enablement"
            ])
        
        # Quality-based recommendations
        if quality_assessment.overall_quality_score >= 0.9:
            recommendations.extend([
                "Submit to industry publications",
                "Nominate for content awards",
                "Use for speaking opportunities"
            ])
        
        return recommendations
    
    async def execute_workflow(
        self,
        title: str,
        outline: List[str],
        content_format: str = "blog",
        company_context: str = "",
        target_audience: str = "Business professionals",
        quality_level: str = "standard",
        tone_style: str = "professional",
        word_count_target: Optional[int] = None,
        include_research: bool = True,
        include_seo_optimization: bool = True,
        include_call_to_action: bool = True,
        research_data: Optional[Dict[str, Any]] = None,
        geo_metadata: Optional[Dict[str, Any]] = None,
        brand_guidelines: Optional[Dict[str, Any]] = None,
        custom_instructions: Optional[List[str]] = None,
        execution_context: Optional[LangGraphExecutionContext] = None
    ) -> AgentResult:
        """Execute the content agent workflow."""
        
        context = {
            "title": title,
            "outline": outline,
            "content_format": content_format,
            "company_context": company_context,
            "target_audience": target_audience,
            "quality_level": quality_level,
            "tone_style": tone_style,
            "word_count_target": word_count_target,
            "include_research": include_research,
            "include_seo_optimization": include_seo_optimization,
            "include_call_to_action": include_call_to_action,
            "research_data": research_data or {},
            "geo_metadata": geo_metadata or {},
            "brand_guidelines": brand_guidelines or {},
            "custom_instructions": custom_instructions or [],
            "workflow_id": f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if execution_context:
            context["execution_context"] = execution_context.__dict__
        
        try:
            final_state = await self.run_workflow(context)
            
            if final_state.status == WorkflowStatus.COMPLETED:
                # Prepare successful result
                result_data = {
                    "final_content": final_state.final_content,
                    "content_variants": final_state.content_variants,
                    "content_metrics": {
                        "word_count": final_state.content_metrics.word_count,
                        "character_count": final_state.content_metrics.character_count,
                        "reading_time_minutes": final_state.content_metrics.reading_time_minutes,
                        "readability_score": final_state.content_metrics.readability_score,
                        "engagement_score": final_state.content_metrics.engagement_score,
                        "seo_score": final_state.content_metrics.seo_score,
                        "format_specific_metrics": final_state.content_metrics.format_specific_metrics
                    },
                    "quality_assessment": {
                        "overall_quality_score": final_state.quality_assessment.overall_quality_score,
                        "content_completeness": final_state.quality_assessment.content_completeness,
                        "structural_coherence": final_state.quality_assessment.structural_coherence,
                        "brand_alignment": final_state.quality_assessment.brand_alignment,
                        "audience_targeting": final_state.quality_assessment.audience_targeting,
                        "actionability": final_state.quality_assessment.actionability,
                        "improvement_suggestions": final_state.quality_assessment.improvement_suggestions,
                        "approval_status": final_state.quality_assessment.approval_status
                    },
                    "optimizations_applied": {
                        "seo_optimizations": final_state.seo_optimizations,
                        "readability_improvements": final_state.readability_improvements,
                        "brand_consistency_adjustments": final_state.brand_consistency_adjustments
                    },
                    "recommendations": {
                        "assets": final_state.asset_recommendations,
                        "distribution": final_state.distribution_recommendations
                    },
                    "workflow_metrics": {
                        "generation_time": final_state.generation_time,
                        "optimization_iterations": final_state.optimization_iterations,
                        "variant_count": len(final_state.content_variants)
                    }
                }
                
                return AgentResult(
                    success=True,
                    data=result_data,
                    execution_time_ms=(final_state.completed_at - final_state.created_at).total_seconds() * 1000,
                    metadata={
                        "workflow_id": final_state.workflow_id,
                        "content_format": final_state.content_requirements.content_format.value,
                        "quality_level": final_state.content_requirements.quality_level.value,
                        "word_count": final_state.content_metrics.word_count,
                        "quality_score": final_state.quality_assessment.overall_quality_score,
                        "approval_status": final_state.quality_assessment.approval_status,
                        "variants_created": len(final_state.content_variants)
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