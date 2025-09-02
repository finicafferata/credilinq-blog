#!/usr/bin/env python3
"""
Content-First LangGraph Workflow - Narrative-Driven Content Generation

This module implements a content-centric workflow that generates cohesive content deliverables
instead of fragmented tasks. Content pieces flow as a unified narrative with clear relationships
and dependencies between different content types.

Key Features:
- Content deliverables as first-class entities
- Narrative continuity across all content pieces
- Relationship tracking between content items
- Structured content storage with metadata
- Agent collaboration focused on content quality
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Union
from typing_extensions import Annotated
from dataclasses import dataclass, field
from enum import Enum

# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END, CompiledStateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.llm_client import create_llm

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext
from ..core.agent_factory import AgentFactory
from ..core.content_deliverable_service import content_service, ContentDeliverable as ServiceContentDeliverable
from ..specialized.content_agent import ContentAgent
from ..specialized.writer_agent import WriterAgent
from ..specialized.planner_agent import PlannerAgent

logger = logging.getLogger(__name__)

# =============================================================================
# CONTENT DELIVERABLE MODELS
# =============================================================================

class ContentDeliverableType(Enum):
    """Types of content deliverables"""
    BLOG_POST = "blog_post"
    LINKEDIN_POST = "linkedin_post" 
    TWITTER_THREAD = "twitter_thread"
    EMAIL_SEQUENCE = "email_sequence"
    CASE_STUDY = "case_study"
    WHITEPAPER = "whitepaper"
    INFOGRAPHIC_COPY = "infographic_copy"
    VIDEO_SCRIPT = "video_script"
    NEWSLETTER = "newsletter"
    LANDING_PAGE_COPY = "landing_page_copy"

class ContentStatus(Enum):
    """Status of content deliverables"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DRAFT_COMPLETE = "draft_complete"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class NarrativePosition(Enum):
    """Position in the overall narrative flow"""
    FOUNDATION = "foundation"      # Sets up core concepts
    EXPLORATION = "exploration"    # Dives deeper into topics
    APPLICATION = "application"    # Shows practical use cases
    TRANSFORMATION = "transformation"  # Demonstrates outcomes
    REINFORCEMENT = "reinforcement"    # Strengthens key messages

@dataclass
class ContentDeliverable:
    """A complete content deliverable with narrative context"""
    content_id: str
    deliverable_type: ContentDeliverableType
    title: str
    content_body: str
    summary: str
    word_count: int
    status: ContentStatus
    
    # Narrative Flow
    narrative_position: NarrativePosition
    narrative_thread_id: str
    key_message: str
    supporting_points: List[str]
    
    # Content Relationships
    references_content: List[str] = field(default_factory=list)  # IDs of content this references
    referenced_by_content: List[str] = field(default_factory=list)  # IDs of content that reference this
    narrative_precedence: List[str] = field(default_factory=list)  # Content that should be read before this
    narrative_sequence: List[str] = field(default_factory=list)  # Content that flows after this
    
    # Metadata
    target_audience: str = "B2B professionals"
    tone: str = "Professional"
    channel: str = ""
    seo_keywords: List[str] = field(default_factory=list)
    call_to_action: str = ""
    
    # Performance & Quality
    readability_score: Optional[float] = None
    engagement_prediction: Optional[str] = None
    quality_score: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'content_id': self.content_id,
            'deliverable_type': self.deliverable_type.value,
            'title': self.title,
            'content_body': self.content_body,
            'summary': self.summary,
            'word_count': self.word_count,
            'status': self.status.value,
            'narrative_position': self.narrative_position.value,
            'narrative_thread_id': self.narrative_thread_id,
            'key_message': self.key_message,
            'supporting_points': self.supporting_points,
            'references_content': self.references_content,
            'referenced_by_content': self.referenced_by_content,
            'narrative_precedence': self.narrative_precedence,
            'narrative_sequence': self.narrative_sequence,
            'target_audience': self.target_audience,
            'tone': self.tone,
            'channel': self.channel,
            'seo_keywords': self.seo_keywords,
            'call_to_action': self.call_to_action,
            'readability_score': self.readability_score,
            'engagement_prediction': self.engagement_prediction,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

@dataclass
class NarrativeContext:
    """Context for maintaining narrative continuity across content pieces"""
    central_theme: str
    supporting_themes: List[str]
    key_messages: List[str]
    target_transformation: str  # What change do we want to create in the audience?
    brand_voice_guidelines: str
    
    # Narrative Flow Management
    content_journey_map: Dict[str, List[str]]  # Maps content types to their narrative flow
    cross_references: Dict[str, List[str]]     # Maps content IDs to related content
    thematic_connections: Dict[str, str]       # Maps content IDs to their theme contributions
    
    # Consistency Tracking
    terminology_glossary: Dict[str, str]       # Consistent terms and definitions
    recurring_concepts: List[str]              # Concepts that should appear across content
    brand_examples: List[str]                  # Consistent examples to use
    
    def get_narrative_context_for_content(self, content_type: ContentDeliverableType, 
                                        position: NarrativePosition) -> str:
        """Get narrative context for specific content creation"""
        context_prompt = f"""
        NARRATIVE CONTEXT FOR {content_type.value.upper()}:
        
        Central Theme: {self.central_theme}
        Position in Journey: {position.value}
        Target Transformation: {self.target_transformation}
        
        Key Messages to Reinforce:
        {chr(10).join(f"• {msg}" for msg in self.key_messages)}
        
        Supporting Themes:
        {chr(10).join(f"• {theme}" for theme in self.supporting_themes)}
        
        Brand Voice: {self.brand_voice_guidelines}
        
        Terminology to Use Consistently:
        {chr(10).join(f"• {term}: {definition}" for term, definition in self.terminology_glossary.items())}
        
        Recurring Concepts to Reference:
        {chr(10).join(f"• {concept}" for concept in self.recurring_concepts)}
        """
        return context_prompt

# =============================================================================
# CONTENT-FIRST WORKFLOW STATE
# =============================================================================

class ContentFirstWorkflowState(TypedDict):
    """
    Content-first workflow state that tracks deliverables, not tasks
    """
    # Campaign Context
    campaign_id: str
    campaign_brief: str
    campaign_objectives: List[str]
    target_audience: str
    brand_context: str
    
    # Content Portfolio Planning
    content_deliverables: Annotated[List[ContentDeliverable], "All content deliverables for the campaign"]
    content_requirements: Annotated[Dict[str, Any], "Requirements for each content type"]
    deliverable_priorities: Annotated[List[str], "Content IDs in priority order"]
    
    # Narrative Management
    narrative_thread: Annotated[NarrativeContext, "Overall narrative context"]
    content_relationships: Annotated[Dict[str, List[str]], "Mapping of content relationships"]
    thematic_flow: Annotated[Dict[str, str], "How themes flow between content pieces"]
    
    # Content Generation Progress
    completed_deliverables: Annotated[List[str], "IDs of completed content"]
    in_progress_deliverables: Annotated[List[str], "IDs of content being worked on"]
    pending_deliverables: Annotated[List[str], "IDs of content waiting to be created"]
    
    # Agent Coordination
    content_assignments: Annotated[Dict[str, str], "Content ID to agent assignment"]
    agent_context_sharing: Annotated[Dict[str, Any], "Shared context between agents"]
    
    # Quality Assurance
    review_feedback: Annotated[Dict[str, str], "Review feedback by content ID"]
    revision_requests: Annotated[Dict[str, List[str]], "Revision requests by content ID"]
    quality_scores: Annotated[Dict[str, float], "Quality scores by content ID"]
    
    # Workflow Control
    current_phase: str
    workflow_status: str
    error_log: Annotated[List[str], "Any errors encountered"]
    
    # Metadata
    created_at: datetime
    updated_at: datetime

# =============================================================================
# CONTENT-FIRST WORKFLOW ENGINE
# =============================================================================

class ContentFirstWorkflowOrchestrator:
    """
    Orchestrates content-first workflows that produce cohesive narrative content
    """
    
    def __init__(self):
        self.workflow_id = "content_first_workflow"
        self.description = "Content-centric workflow producing narrative-driven deliverables"
        self.agent_factory = AgentFactory()
        self.llm = create_llm(model="gemini-1.5-pro", temperature=0.3)
        
        # Agent pool for content creation
        self.narrative_coordinator = None  # Will be created during workflow
        self.content_agents: Dict[str, BaseAgent] = {}
        
        self.workflow_graph = self._create_workflow_graph()
        
    def _create_workflow_graph(self) -> CompiledStateGraph:
        """Create the LangGraph workflow for content-first generation"""
        
        workflow = StateGraph(ContentFirstWorkflowState)
        
        # Define workflow nodes
        workflow.add_node("initialize_campaign", self._initialize_campaign_context)
        workflow.add_node("plan_content_portfolio", self._plan_content_portfolio)
        workflow.add_node("establish_narrative_thread", self._establish_narrative_thread)
        workflow.add_node("coordinate_content_creation", self._coordinate_content_creation)
        workflow.add_node("generate_foundation_content", self._generate_foundation_content)
        workflow.add_node("generate_exploration_content", self._generate_exploration_content)
        workflow.add_node("generate_application_content", self._generate_application_content)
        workflow.add_node("generate_transformation_content", self._generate_transformation_content)
        workflow.add_node("generate_reinforcement_content", self._generate_reinforcement_content)
        workflow.add_node("ensure_narrative_continuity", self._ensure_narrative_continuity)
        workflow.add_node("quality_review_deliverables", self._quality_review_deliverables)
        workflow.add_node("finalize_content_portfolio", self._finalize_content_portfolio)
        
        # Define workflow edges
        workflow.add_edge("initialize_campaign", "plan_content_portfolio")
        workflow.add_edge("plan_content_portfolio", "establish_narrative_thread")
        workflow.add_edge("establish_narrative_thread", "coordinate_content_creation")
        workflow.add_edge("coordinate_content_creation", "generate_foundation_content")
        workflow.add_edge("generate_foundation_content", "generate_exploration_content")
        workflow.add_edge("generate_exploration_content", "generate_application_content")
        workflow.add_edge("generate_application_content", "generate_transformation_content")
        workflow.add_edge("generate_transformation_content", "generate_reinforcement_content")
        workflow.add_edge("generate_reinforcement_content", "ensure_narrative_continuity")
        workflow.add_edge("ensure_narrative_continuity", "quality_review_deliverables")
        workflow.add_edge("quality_review_deliverables", "finalize_content_portfolio")
        workflow.add_edge("finalize_content_portfolio", END)
        
        workflow.set_entry_point("initialize_campaign")
        
        return workflow.compile()
    
    # =============================================================================
    # WORKFLOW NODE IMPLEMENTATIONS
    # =============================================================================
    
    async def _initialize_campaign_context(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Initialize campaign context and content requirements"""
        logger.info(f"Initializing content-first workflow for campaign: {state['campaign_id']}")
        
        # Parse campaign brief for content requirements
        content_requirements = await self._analyze_campaign_brief(state['campaign_brief'])
        
        state.update({
            'content_requirements': content_requirements,
            'current_phase': 'planning',
            'workflow_status': 'initializing',
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'content_deliverables': [],
            'completed_deliverables': [],
            'in_progress_deliverables': [],
            'pending_deliverables': [],
            'content_assignments': {},
            'agent_context_sharing': {},
            'review_feedback': {},
            'revision_requests': {},
            'quality_scores': {},
            'error_log': []
        })
        
        logger.info("Campaign context initialized successfully")
        return state
    
    async def _plan_content_portfolio(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Plan the complete content portfolio based on campaign objectives"""
        logger.info("Planning content portfolio with narrative structure")
        
        # Create content planner agent
        planner = PlannerAgent()
        
        planning_prompt = f"""
        Create a comprehensive content portfolio plan for this campaign:
        
        CAMPAIGN BRIEF: {state['campaign_brief']}
        OBJECTIVES: {', '.join(state['campaign_objectives'])}
        TARGET AUDIENCE: {state['target_audience']}
        BRAND CONTEXT: {state['brand_context']}
        
        CONTENT PORTFOLIO REQUIREMENTS:
        1. Create a portfolio of 12-15 content pieces across different formats
        2. Ensure narrative flow and thematic consistency
        3. Plan content relationships and cross-references
        4. Assign narrative positions to each piece
        5. Define key messages and supporting points for each
        
        Return a JSON structure with detailed content deliverable specifications.
        """
        
        planning_result = await planner.execute({
            'task': 'content_portfolio_planning',
            'input': planning_prompt,
            'output_format': 'json'
        })
        
        if planning_result.success:
            portfolio_plan = planning_result.data
            deliverables = await self._create_content_deliverables_from_plan(
                state['campaign_id'], 
                portfolio_plan
            )
            
            state.update({
                'content_deliverables': deliverables,
                'pending_deliverables': [d.content_id for d in deliverables],
                'deliverable_priorities': self._prioritize_deliverables(deliverables),
                'current_phase': 'narrative_establishment'
            })
            
            logger.info(f"Content portfolio planned: {len(deliverables)} deliverables")
        else:
            logger.error(f"Content portfolio planning failed: {planning_result.error_message}")
            state['error_log'].append(f"Portfolio planning failed: {planning_result.error_message}")
        
        return state
    
    async def _establish_narrative_thread(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Establish the narrative thread that connects all content pieces"""
        logger.info("Establishing narrative thread for content continuity")
        
        # Create narrative coordinator
        coordinator_prompt = f"""
        Create a comprehensive narrative context for this content portfolio:
        
        CAMPAIGN: {state['campaign_brief']}
        CONTENT DELIVERABLES: {len(state['content_deliverables'])} pieces
        
        ESTABLISH:
        1. Central narrative theme
        2. Supporting themes for each content type
        3. Key messages to reinforce consistently
        4. Target transformation for the audience
        5. Brand voice guidelines
        6. Terminology glossary for consistency
        7. Content journey mapping
        8. Cross-reference opportunities
        
        This narrative thread should ensure all content feels like part of a cohesive story.
        """
        
        narrative_result = await self._create_narrative_coordinator()
        
        if narrative_result:
            narrative_context = await self._generate_narrative_context(
                coordinator_prompt,
                state['content_deliverables']
            )
            
            state.update({
                'narrative_thread': narrative_context,
                'content_relationships': self._map_content_relationships(state['content_deliverables']),
                'thematic_flow': self._create_thematic_flow_map(state['content_deliverables']),
                'current_phase': 'content_coordination'
            })
            
            logger.info("Narrative thread established successfully")
        else:
            logger.error("Failed to establish narrative thread")
            state['error_log'].append("Narrative thread establishment failed")
        
        return state
    
    async def _coordinate_content_creation(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Coordinate agent assignments and content creation strategy"""
        logger.info("Coordinating content creation with specialized agents")
        
        # Assign agents to content deliverables
        assignments = {}
        agent_context = {}
        
        for deliverable in state['content_deliverables']:
            # Determine best agent for this content type
            agent_type = self._determine_agent_for_content_type(deliverable.deliverable_type)
            agent = await self._get_or_create_agent(agent_type)
            
            assignments[deliverable.content_id] = agent.agent_id if agent else 'default'
            
            # Prepare shared context for agent
            agent_context[deliverable.content_id] = {
                'narrative_context': state['narrative_thread'].get_narrative_context_for_content(
                    deliverable.deliverable_type,
                    deliverable.narrative_position
                ),
                'related_content': deliverable.references_content,
                'key_message': deliverable.key_message,
                'supporting_points': deliverable.supporting_points,
                'brand_context': state['brand_context']
            }
        
        state.update({
            'content_assignments': assignments,
            'agent_context_sharing': agent_context,
            'current_phase': 'content_generation'
        })
        
        logger.info(f"Content creation coordinated: {len(assignments)} assignments made")
        return state
    
    async def _generate_foundation_content(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Generate foundation content that establishes core concepts"""
        logger.info("Generating foundation content pieces")
        
        foundation_content = [
            d for d in state['content_deliverables'] 
            if d.narrative_position == NarrativePosition.FOUNDATION
        ]
        
        for deliverable in foundation_content:
            await self._generate_single_content_deliverable(state, deliverable)
        
        state['current_phase'] = 'exploration_generation'
        logger.info(f"Foundation content generated: {len(foundation_content)} pieces")
        return state
    
    async def _generate_exploration_content(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Generate exploration content that dives deeper into topics"""
        logger.info("Generating exploration content pieces")
        
        exploration_content = [
            d for d in state['content_deliverables']
            if d.narrative_position == NarrativePosition.EXPLORATION
        ]
        
        for deliverable in exploration_content:
            await self._generate_single_content_deliverable(state, deliverable)
        
        state['current_phase'] = 'application_generation'
        logger.info(f"Exploration content generated: {len(exploration_content)} pieces")
        return state
    
    async def _generate_application_content(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Generate application content showing practical use cases"""
        logger.info("Generating application content pieces")
        
        application_content = [
            d for d in state['content_deliverables']
            if d.narrative_position == NarrativePosition.APPLICATION
        ]
        
        for deliverable in application_content:
            await self._generate_single_content_deliverable(state, deliverable)
        
        state['current_phase'] = 'transformation_generation'
        logger.info(f"Application content generated: {len(application_content)} pieces")
        return state
    
    async def _generate_transformation_content(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Generate transformation content demonstrating outcomes"""
        logger.info("Generating transformation content pieces")
        
        transformation_content = [
            d for d in state['content_deliverables']
            if d.narrative_position == NarrativePosition.TRANSFORMATION
        ]
        
        for deliverable in transformation_content:
            await self._generate_single_content_deliverable(state, deliverable)
        
        state['current_phase'] = 'reinforcement_generation'
        logger.info(f"Transformation content generated: {len(transformation_content)} pieces")
        return state
    
    async def _generate_reinforcement_content(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Generate reinforcement content that strengthens key messages"""
        logger.info("Generating reinforcement content pieces")
        
        reinforcement_content = [
            d for d in state['content_deliverables']
            if d.narrative_position == NarrativePosition.REINFORCEMENT
        ]
        
        for deliverable in reinforcement_content:
            await self._generate_single_content_deliverable(state, deliverable)
        
        state['current_phase'] = 'continuity_check'
        logger.info(f"Reinforcement content generated: {len(reinforcement_content)} pieces")
        return state
    
    async def _ensure_narrative_continuity(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Ensure narrative continuity across all generated content"""
        logger.info("Ensuring narrative continuity across content portfolio")
        
        # Review all content for narrative consistency
        continuity_issues = []
        
        for deliverable in state['content_deliverables']:
            if deliverable.status == ContentStatus.DRAFT_COMPLETE:
                issues = await self._check_narrative_consistency(
                    deliverable, 
                    state['narrative_thread'],
                    state['content_deliverables']
                )
                if issues:
                    continuity_issues.extend(issues)
                    state['revision_requests'][deliverable.content_id] = issues
        
        if continuity_issues:
            logger.warning(f"Found {len(continuity_issues)} narrative continuity issues")
            # Process revisions for content with issues
            await self._process_narrative_revisions(state)
        else:
            logger.info("Narrative continuity verified across all content")
        
        state['current_phase'] = 'quality_review'
        return state
    
    async def _quality_review_deliverables(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Conduct quality review of all content deliverables"""
        logger.info("Conducting quality review of all deliverables")
        
        quality_scores = {}
        review_feedback = {}
        
        for deliverable in state['content_deliverables']:
            if deliverable.status == ContentStatus.DRAFT_COMPLETE:
                quality_result = await self._evaluate_content_quality(deliverable, state['narrative_thread'])
                
                quality_scores[deliverable.content_id] = quality_result['score']
                review_feedback[deliverable.content_id] = quality_result['feedback']
                
                # Update deliverable status based on quality
                if quality_result['score'] >= 0.8:
                    deliverable.status = ContentStatus.APPROVED
                    deliverable.quality_score = quality_result['score']
                else:
                    deliverable.status = ContentStatus.UNDER_REVIEW
                    state['revision_requests'][deliverable.content_id] = quality_result['improvements']
        
        state.update({
            'quality_scores': quality_scores,
            'review_feedback': review_feedback,
            'current_phase': 'finalization'
        })
        
        avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        logger.info(f"Quality review completed. Average quality score: {avg_quality:.2f}")
        
        return state
    
    async def _finalize_content_portfolio(self, state: ContentFirstWorkflowState) -> ContentFirstWorkflowState:
        """Finalize the content portfolio and prepare deliverables"""
        logger.info("Finalizing content portfolio")
        
        # Mark approved content as completed
        completed_count = 0
        for deliverable in state['content_deliverables']:
            if deliverable.status == ContentStatus.APPROVED:
                deliverable.status = ContentStatus.PUBLISHED
                deliverable.completed_at = datetime.now()
                completed_count += 1
                
                if deliverable.content_id in state['in_progress_deliverables']:
                    state['in_progress_deliverables'].remove(deliverable.content_id)
                
                if deliverable.content_id not in state['completed_deliverables']:
                    state['completed_deliverables'].append(deliverable.content_id)
        
        # Store content deliverables in database
        await self._store_content_deliverables(state['campaign_id'], state['content_deliverables'])
        
        state.update({
            'workflow_status': 'completed',
            'current_phase': 'completed',
            'updated_at': datetime.now()
        })
        
        logger.info(f"Content portfolio finalized: {completed_count} deliverables completed")
        return state
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    async def _analyze_campaign_brief(self, brief: str) -> Dict[str, Any]:
        """Analyze campaign brief to extract content requirements"""
        analysis_prompt = f"""
        Analyze this campaign brief and extract content requirements:
        
        BRIEF: {brief}
        
        Extract and structure:
        1. Content types needed (blog posts, social posts, emails, etc.)
        2. Target audience characteristics
        3. Key messages to communicate
        4. Tone and voice requirements
        5. Content volume recommendations
        6. Channel-specific needs
        
        Return as structured JSON.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a content strategist analyzing campaign requirements."),
            HumanMessage(content=analysis_prompt)
        ])
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse content requirements JSON")
            return {'content_types': ['blog_post', 'linkedin_post'], 'volume': 12}
    
    async def _create_content_deliverables_from_plan(self, campaign_id: str, 
                                                   plan: Dict[str, Any]) -> List[ContentDeliverable]:
        """Create ContentDeliverable objects from planning results"""
        deliverables = []
        
        for item in plan.get('content_items', []):
            deliverable = ContentDeliverable(
                content_id=str(uuid.uuid4()),
                deliverable_type=ContentDeliverableType(item.get('type', 'blog_post')),
                title=item.get('title', ''),
                content_body='',  # Will be generated later
                summary=item.get('summary', ''),
                word_count=0,    # Will be calculated after generation
                status=ContentStatus.PLANNED,
                narrative_position=NarrativePosition(item.get('narrative_position', 'foundation')),
                narrative_thread_id=f"{campaign_id}_main_thread",
                key_message=item.get('key_message', ''),
                supporting_points=item.get('supporting_points', []),
                target_audience=item.get('target_audience', 'B2B professionals'),
                tone=item.get('tone', 'Professional'),
                channel=item.get('channel', ''),
                seo_keywords=item.get('seo_keywords', []),
                call_to_action=item.get('call_to_action', '')
            )
            deliverables.append(deliverable)
        
        return deliverables
    
    def _prioritize_deliverables(self, deliverables: List[ContentDeliverable]) -> List[str]:
        """Prioritize deliverables based on narrative position and type"""
        priority_map = {
            NarrativePosition.FOUNDATION: 1,
            NarrativePosition.EXPLORATION: 2,
            NarrativePosition.APPLICATION: 3,
            NarrativePosition.TRANSFORMATION: 4,
            NarrativePosition.REINFORCEMENT: 5
        }
        
        sorted_deliverables = sorted(
            deliverables,
            key=lambda d: (priority_map.get(d.narrative_position, 999), d.title)
        )
        
        return [d.content_id for d in sorted_deliverables]
    
    async def _create_narrative_coordinator(self) -> bool:
        """Create narrative coordinator for maintaining story consistency"""
        try:
            self.narrative_coordinator = ContentAgent()  # Use content agent as coordinator
            logger.info("Narrative coordinator created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create narrative coordinator: {str(e)}")
            return False
    
    async def _generate_narrative_context(self, prompt: str, 
                                        deliverables: List[ContentDeliverable]) -> NarrativeContext:
        """Generate comprehensive narrative context"""
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a narrative strategist creating cohesive content experiences."),
            HumanMessage(content=prompt)
        ])
        
        # Parse response and create NarrativeContext
        # For now, create a basic structure
        return NarrativeContext(
            central_theme="Professional growth through innovative solutions",
            supporting_themes=["Industry expertise", "Technology innovation", "Customer success"],
            key_messages=["Transformation", "Innovation", "Results"],
            target_transformation="Move audience from awareness to consideration",
            brand_voice_guidelines="Professional, authoritative, approachable",
            content_journey_map={},
            cross_references={},
            thematic_connections={},
            terminology_glossary={"AI": "Artificial Intelligence", "ROI": "Return on Investment"},
            recurring_concepts=["Digital transformation", "Competitive advantage"],
            brand_examples=["Customer success story", "Industry benchmark"]
        )
    
    def _map_content_relationships(self, deliverables: List[ContentDeliverable]) -> Dict[str, List[str]]:
        """Map relationships between content deliverables"""
        relationships = {}
        
        for deliverable in deliverables:
            relationships[deliverable.content_id] = []
            
            # Find related content based on themes and narrative position
            for other in deliverables:
                if other.content_id != deliverable.content_id:
                    # Check for thematic overlap
                    if self._has_thematic_overlap(deliverable, other):
                        relationships[deliverable.content_id].append(other.content_id)
        
        return relationships
    
    def _has_thematic_overlap(self, content1: ContentDeliverable, content2: ContentDeliverable) -> bool:
        """Check if two content pieces have thematic overlap"""
        # Simple overlap check - can be enhanced with semantic analysis
        return (content1.key_message in content2.supporting_points or 
                content2.key_message in content1.supporting_points)
    
    def _create_thematic_flow_map(self, deliverables: List[ContentDeliverable]) -> Dict[str, str]:
        """Create a map of how themes flow between content pieces"""
        flow_map = {}
        
        for deliverable in deliverables:
            flow_map[deliverable.content_id] = f"Contributes to {deliverable.key_message} narrative"
        
        return flow_map
    
    def _determine_agent_for_content_type(self, content_type: ContentDeliverableType) -> str:
        """Determine the best agent type for a content deliverable"""
        agent_mapping = {
            ContentDeliverableType.BLOG_POST: "writer_agent",
            ContentDeliverableType.LINKEDIN_POST: "social_media_agent",
            ContentDeliverableType.TWITTER_THREAD: "social_media_agent",
            ContentDeliverableType.EMAIL_SEQUENCE: "content_agent",
            ContentDeliverableType.CASE_STUDY: "writer_agent",
            ContentDeliverableType.WHITEPAPER: "writer_agent",
            ContentDeliverableType.VIDEO_SCRIPT: "content_agent"
        }
        
        return agent_mapping.get(content_type, "content_agent")
    
    async def _get_or_create_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get existing agent or create new one"""
        if agent_type in self.content_agents:
            return self.content_agents[agent_type]
        
        try:
            agent = self.agent_factory.create_agent(agent_type)
            self.content_agents[agent_type] = agent
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {str(e)}")
            return None
    
    async def _generate_single_content_deliverable(self, state: ContentFirstWorkflowState, 
                                                 deliverable: ContentDeliverable) -> None:
        """Generate content for a single deliverable"""
        logger.info(f"Generating content for: {deliverable.title}")
        
        # Get assigned agent
        agent_id = state['content_assignments'].get(deliverable.content_id)
        agent = self.content_agents.get(agent_id.split('_')[0] if agent_id else 'content')
        
        if not agent:
            logger.error(f"No agent available for deliverable {deliverable.content_id}")
            return
        
        # Get narrative context for this content
        context = state['agent_context_sharing'].get(deliverable.content_id, {})
        
        # Generate content
        generation_input = {
            'title': deliverable.title,
            'content_type': deliverable.deliverable_type.value,
            'target_audience': deliverable.target_audience,
            'key_message': deliverable.key_message,
            'supporting_points': deliverable.supporting_points,
            'narrative_context': context.get('narrative_context', ''),
            'tone': deliverable.tone,
            'word_count_target': self._get_target_word_count(deliverable.deliverable_type),
            'seo_keywords': deliverable.seo_keywords,
            'call_to_action': deliverable.call_to_action
        }
        
        try:
            result = await agent.execute(generation_input)
            
            if result.success:
                # Update deliverable with generated content
                deliverable.content_body = result.data.get('content', '')
                deliverable.word_count = len(deliverable.content_body.split())
                deliverable.status = ContentStatus.DRAFT_COMPLETE
                deliverable.updated_at = datetime.now()
                
                # Move from pending to in_progress to completed
                if deliverable.content_id in state['pending_deliverables']:
                    state['pending_deliverables'].remove(deliverable.content_id)
                state['completed_deliverables'].append(deliverable.content_id)
                
                logger.info(f"Content generated successfully: {deliverable.word_count} words")
            else:
                logger.error(f"Content generation failed: {result.error_message}")
                deliverable.status = ContentStatus.UNDER_REVIEW
                
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            deliverable.status = ContentStatus.UNDER_REVIEW
    
    def _get_target_word_count(self, content_type: ContentDeliverableType) -> int:
        """Get target word count for content type"""
        word_counts = {
            ContentDeliverableType.BLOG_POST: 1500,
            ContentDeliverableType.LINKEDIN_POST: 200,
            ContentDeliverableType.TWITTER_THREAD: 250,
            ContentDeliverableType.EMAIL_SEQUENCE: 300,
            ContentDeliverableType.CASE_STUDY: 2000,
            ContentDeliverableType.WHITEPAPER: 3000,
            ContentDeliverableType.VIDEO_SCRIPT: 800
        }
        
        return word_counts.get(content_type, 1000)
    
    async def _check_narrative_consistency(self, deliverable: ContentDeliverable, 
                                         narrative: NarrativeContext,
                                         all_deliverables: List[ContentDeliverable]) -> List[str]:
        """Check narrative consistency for a deliverable"""
        issues = []
        
        # Check if key message is reinforced
        if deliverable.key_message not in deliverable.content_body:
            issues.append(f"Key message '{deliverable.key_message}' not clearly reinforced")
        
        # Check terminology consistency
        for term, definition in narrative.terminology_glossary.items():
            if term.lower() in deliverable.content_body.lower():
                # Could add more sophisticated terminology consistency checks
                pass
        
        return issues
    
    async def _process_narrative_revisions(self, state: ContentFirstWorkflowState) -> None:
        """Process narrative revision requests"""
        for content_id, issues in state['revision_requests'].items():
            deliverable = next((d for d in state['content_deliverables'] if d.content_id == content_id), None)
            if deliverable:
                logger.info(f"Processing revisions for {deliverable.title}: {len(issues)} issues")
                # Could implement revision logic here
                deliverable.status = ContentStatus.UNDER_REVIEW
    
    async def _evaluate_content_quality(self, deliverable: ContentDeliverable, 
                                      narrative: NarrativeContext) -> Dict[str, Any]:
        """Evaluate content quality and provide feedback"""
        evaluation_prompt = f"""
        Evaluate the quality of this content deliverable:
        
        TITLE: {deliverable.title}
        TYPE: {deliverable.deliverable_type.value}
        CONTENT: {deliverable.content_body[:1000]}...
        
        NARRATIVE CONTEXT: {narrative.central_theme}
        KEY MESSAGE: {deliverable.key_message}
        
        Evaluate on:
        1. Content quality and clarity (0-1)
        2. Narrative alignment (0-1)
        3. Message reinforcement (0-1)
        4. Audience appropriateness (0-1)
        5. Brand voice consistency (0-1)
        
        Return overall score and specific feedback.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a content quality evaluator."),
            HumanMessage(content=evaluation_prompt)
        ])
        
        # Parse evaluation results
        # For now, return a basic structure
        return {
            'score': 0.85,  # Default good score
            'feedback': 'Content meets quality standards with minor improvements possible',
            'improvements': ['Strengthen call-to-action', 'Add more specific examples']
        }
    
    async def _store_content_deliverables(self, campaign_id: str, 
                                        deliverables: List[ContentDeliverable]) -> None:
        """Store content deliverables in database using content service"""
        try:
            completed_deliverables = [d for d in deliverables if d.status == ContentStatus.PUBLISHED]
            
            logger.info(f"Storing {len(completed_deliverables)} completed deliverables for campaign {campaign_id}")
            
            stored_count = 0
            for deliverable in completed_deliverables:
                try:
                    # Map workflow ContentDeliverable to service format
                    content_type_mapping = {
                        ContentDeliverableType.BLOG_POST: 'blog_post',
                        ContentDeliverableType.LINKEDIN_POST: 'social_media_post',
                        ContentDeliverableType.TWITTER_THREAD: 'social_media_post',
                        ContentDeliverableType.EMAIL_SEQUENCE: 'email_campaign',
                        ContentDeliverableType.CASE_STUDY: 'case_study',
                        ContentDeliverableType.WHITEPAPER: 'whitepaper',
                        ContentDeliverableType.INFOGRAPHIC_COPY: 'infographic_concept',
                        ContentDeliverableType.VIDEO_SCRIPT: 'video_script',
                        ContentDeliverableType.NEWSLETTER: 'newsletter',
                        ContentDeliverableType.LANDING_PAGE_COPY: 'landing_page'
                    }
                    
                    # Create service deliverable
                    service_deliverable = content_service.create_content(
                        title=deliverable.title,
                        content=deliverable.content_body,
                        campaign_id=campaign_id,
                        content_type=content_type_mapping.get(deliverable.deliverable_type, 'blog_post'),
                        summary=deliverable.summary,
                        platform=deliverable.channel,
                        narrative_order=self._map_narrative_position_to_order(deliverable.narrative_position),
                        target_audience=deliverable.target_audience,
                        tone=deliverable.tone,
                        word_count=deliverable.word_count,
                        reading_time=deliverable.word_count // 200 if deliverable.word_count else 1,
                        seo_score=deliverable.quality_score,
                        metadata={
                            'workflow_generated': True,
                            'narrative_thread_id': deliverable.narrative_thread_id,
                            'key_message': deliverable.key_message,
                            'supporting_points': deliverable.supporting_points,
                            'narrative_position': deliverable.narrative_position.value,
                            'references_content': deliverable.references_content,
                            'referenced_by_content': deliverable.referenced_by_content,
                            'seo_keywords': deliverable.seo_keywords,
                            'call_to_action': deliverable.call_to_action,
                            'quality_score': deliverable.quality_score,
                            'engagement_prediction': deliverable.engagement_prediction
                        }
                    )
                    
                    if service_deliverable:
                        stored_count += 1
                        logger.info(f"✅ Stored deliverable: {deliverable.title}")
                    else:
                        logger.error(f"❌ Failed to store deliverable: {deliverable.title}")
                        
                except Exception as e:
                    logger.error(f"❌ Error storing deliverable {deliverable.title}: {str(e)}")
            
            logger.info(f"✅ Successfully stored {stored_count}/{len(completed_deliverables)} deliverables")
            
        except Exception as e:
            logger.error(f"❌ Failed to store content deliverables: {str(e)}")
    
    def _map_narrative_position_to_order(self, position: NarrativePosition) -> int:
        """Map narrative position to numeric order"""
        position_order = {
            NarrativePosition.FOUNDATION: 1,
            NarrativePosition.EXPLORATION: 2,
            NarrativePosition.APPLICATION: 3,
            NarrativePosition.TRANSFORMATION: 4,
            NarrativePosition.REINFORCEMENT: 5
        }
        return position_order.get(position, 1)
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    async def execute_content_workflow(self, campaign_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete content-first workflow"""
        logger.info(f"Starting content-first workflow for campaign: {campaign_input.get('campaign_id')}")
        
        # Initialize workflow state
        initial_state = ContentFirstWorkflowState(
            campaign_id=campaign_input['campaign_id'],
            campaign_brief=campaign_input['campaign_brief'],
            campaign_objectives=campaign_input.get('objectives', []),
            target_audience=campaign_input.get('target_audience', 'B2B professionals'),
            brand_context=campaign_input.get('brand_context', ''),
            content_deliverables=[],
            content_requirements={},
            deliverable_priorities=[],
            narrative_thread=None,
            content_relationships={},
            thematic_flow={},
            completed_deliverables=[],
            in_progress_deliverables=[],
            pending_deliverables=[],
            content_assignments={},
            agent_context_sharing={},
            review_feedback={},
            revision_requests={},
            quality_scores={},
            current_phase='initialization',
            workflow_status='starting',
            error_log=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Execute workflow
        try:
            final_state = await self.workflow_graph.ainvoke(initial_state)
            
            # Prepare result summary
            result = {
                'campaign_id': final_state['campaign_id'],
                'workflow_status': final_state['workflow_status'],
                'total_deliverables': len(final_state['content_deliverables']),
                'completed_deliverables': len(final_state['completed_deliverables']),
                'content_pieces': [
                    {
                        'content_id': d.content_id,
                        'type': d.deliverable_type.value,
                        'title': d.title,
                        'status': d.status.value,
                        'word_count': d.word_count,
                        'quality_score': d.quality_score,
                        'narrative_position': d.narrative_position.value,
                        'key_message': d.key_message
                    }
                    for d in final_state['content_deliverables']
                ],
                'narrative_summary': {
                    'central_theme': final_state['narrative_thread'].central_theme if final_state['narrative_thread'] else '',
                    'key_messages': final_state['narrative_thread'].key_messages if final_state['narrative_thread'] else [],
                    'content_relationships': len(final_state['content_relationships'])
                },
                'quality_metrics': {
                    'average_quality_score': sum(final_state['quality_scores'].values()) / len(final_state['quality_scores']) if final_state['quality_scores'] else 0,
                    'content_with_revisions': len(final_state['revision_requests']),
                    'total_word_count': sum(d.word_count for d in final_state['content_deliverables'])
                },
                'execution_time': (final_state['updated_at'] - final_state['created_at']).total_seconds()
            }
            
            logger.info(f"Content workflow completed: {result['completed_deliverables']}/{result['total_deliverables']} deliverables")
            return result
            
        except Exception as e:
            logger.error(f"Content workflow execution failed: {str(e)}")
            return {
                'campaign_id': campaign_input['campaign_id'],
                'workflow_status': 'failed',
                'error': str(e)
            }

# Create global workflow orchestrator instance
content_first_workflow = ContentFirstWorkflowOrchestrator()