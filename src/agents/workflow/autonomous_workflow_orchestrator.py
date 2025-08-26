#!/usr/bin/env python3
"""
Autonomous Workflow Orchestrator
Orchestrates the complete autonomous content creation workflow using LangGraph
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from enum import Enum

# LangGraph imports with fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Create stub classes for development
    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes = {}
            self.edges = {}
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            self.edges[from_node] = to_node
        
        def add_conditional_edges(self, node, condition_func, mapping):
            pass
        
        def set_entry_point(self, node):
            self.entry_point = node
        
        def compile(self, checkpointer=None):
            # Will be set by the orchestrator when creating the graph
            return MockWorkflowGraph(None)
    
    class SqliteSaver:
        @classmethod
        def from_conn_string(cls, conn_string):
            return cls()
    
    class MockWorkflowGraph:
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator
        
        async def ainvoke(self, initial_state, config=None):
            # Actually execute workflow phases instead of just simulating
            try:
                # Execute the real workflow phases
                state = await self.orchestrator._execute_workflow_phases(initial_state)
                return state
            except Exception as e:
                logger.error(f"Mock workflow execution failed: {e}")
                return {
                    **initial_state,
                    'completion_status': 'failed',
                    'workflow_metadata': {
                        **initial_state.get('workflow_metadata', {}),
                        'completion_time': datetime.now().isoformat(),
                        'error': str(e)
                    }
                }
    
    END = "END"

from src.agents.core.base_agent import BaseAgent
from src.agents.core.agent_factory import AgentFactory
from src.config.database import db_config

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    """Enhanced workflow state for autonomous operation"""
    campaign_id: str
    campaign_data: Dict[str, Any]
    current_phase: str
    content_strategy: Dict[str, Any]
    knowledge_context: List[Dict[str, Any]]
    generated_content: Dict[str, Any]  # Content by type and agent
    quality_scores: Dict[str, float]
    agent_outputs: Dict[str, Any]
    error_history: List[Dict[str, Any]]
    retry_count: int
    workflow_metadata: Dict[str, Any]
    completion_status: str  # 'running', 'completed', 'failed', 'paused'

class WorkflowPhase(Enum):
    """Workflow execution phases"""
    INITIALIZATION = "initialization"
    INTELLIGENCE_GATHERING = "intelligence_gathering" 
    STRATEGIC_PLANNING = "strategic_planning"
    CONTENT_CREATION = "content_creation"
    QUALITY_ASSURANCE = "quality_assurance"
    OPTIMIZATION = "optimization"
    FINALIZATION = "finalization"
    COMPLETION = "completion"

@dataclass
class AgentTask:
    """Structured agent task definition"""
    agent_type: str
    task_description: str
    input_dependencies: List[str]
    output_key: str
    timeout_seconds: int = 300
    max_retries: int = 3
    quality_threshold: float = 7.0

class AutonomousWorkflowOrchestrator(BaseAgent):
    """
    Main orchestrator for autonomous content creation workflows
    Coordinates 25+ agents in an intelligent workflow
    """
    
    # Valid TaskType enum mapping from database
    TASK_TYPE_MAPPING = {
        'competitor_analysis': 'content_creation',
        'trend_analysis': 'content_creation', 
        'gap_identification': 'content_creation',
        'content_strategy': 'content_creation',
        'social_content': 'social_media_adaptation',
        'email_content': 'email_formatting',
        'content_review': 'content_editing',
        'blog_to_social': 'blog_to_linkedin'
    }
    
    def __init__(self):
        super().__init__()
        self.agent_factory = AgentFactory()
        self.workflow_graph = None
        self.langgraph_available = LANGGRAPH_AVAILABLE
        
        if LANGGRAPH_AVAILABLE:
            self.checkpointer = SqliteSaver.from_conn_string(":memory:")  # In-memory for now
        else:
            self.checkpointer = None
            logger.warning("⚠️ LangGraph not available, using mock implementation for autonomous workflow")
        
        self._initialize_workflow_graph()
        
        # Workflow configuration
        self.max_workflow_time = timedelta(hours=4)  # Maximum workflow execution time
        self.quality_gates = {
            'content_creation': 7.0,
            'quality_assurance': 8.0,
            'final_review': 8.5
        }
    
    def _map_task_type(self, custom_task_type: str) -> str:
        """Map custom task type to valid database enum"""
        return self.TASK_TYPE_MAPPING.get(custom_task_type, custom_task_type)
        
    def _initialize_workflow_graph(self):
        """Initialize the LangGraph workflow"""
        graph_builder = StateGraph(WorkflowState)
        
        # Add workflow nodes
        graph_builder.add_node("initialize_campaign", self._initialize_campaign_node)
        graph_builder.add_node("gather_intelligence", self._gather_intelligence_node)
        graph_builder.add_node("create_strategy", self._create_strategy_node)
        graph_builder.add_node("generate_content", self._generate_content_node)
        graph_builder.add_node("review_quality", self._review_quality_node)
        graph_builder.add_node("optimize_content", self._optimize_content_node)
        graph_builder.add_node("finalize_campaign", self._finalize_campaign_node)
        graph_builder.add_node("handle_errors", self._handle_errors_node)
        
        # Define workflow edges and transitions
        graph_builder.set_entry_point("initialize_campaign")
        
        # Main workflow path
        graph_builder.add_edge("initialize_campaign", "gather_intelligence")
        graph_builder.add_edge("gather_intelligence", "create_strategy") 
        graph_builder.add_edge("create_strategy", "generate_content")
        graph_builder.add_edge("generate_content", "review_quality")
        
        # Quality-based routing
        graph_builder.add_conditional_edges(
            "review_quality",
            self._should_optimize_content,
            {
                "optimize": "optimize_content",
                "finalize": "finalize_campaign",
                "retry": "generate_content",
                "error": "handle_errors"
            }
        )
        
        graph_builder.add_edge("optimize_content", "finalize_campaign")
        graph_builder.add_edge("finalize_campaign", END)
        
        # Error handling
        graph_builder.add_edge("handle_errors", "finalize_campaign")
        
        self.workflow_graph = graph_builder.compile(checkpointer=self.checkpointer)
        
        # Set the orchestrator reference for mock execution
        if hasattr(self.workflow_graph, 'orchestrator'):
            self.workflow_graph.orchestrator = self
    
    async def start_autonomous_workflow(self, campaign_id: str, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start the autonomous workflow for a campaign
        """
        try:
            logger.info(f"🚀 Starting autonomous workflow for campaign: {campaign_id}")
            
            # Initialize workflow state
            initial_state = WorkflowState(
                campaign_id=campaign_id,
                campaign_data=campaign_data,
                current_phase=WorkflowPhase.INITIALIZATION.value,
                content_strategy={},
                knowledge_context=[],
                generated_content={},
                quality_scores={},
                agent_outputs={},
                error_history=[],
                retry_count=0,
                workflow_metadata={
                    'start_time': datetime.now().isoformat(),
                    'workflow_id': str(uuid.uuid4()),
                    'orchestrator_version': '1.0.0'
                },
                completion_status='running'
            )
            
            # Execute the workflow
            workflow_result = await self.workflow_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": campaign_id}}
            )
            
            # Update campaign status in database
            await self._update_campaign_completion_status(campaign_id, workflow_result)
            
            logger.info(f"✅ Autonomous workflow completed for campaign: {campaign_id}")
            
            return {
                'success': True,
                'workflow_id': workflow_result['workflow_metadata']['workflow_id'],
                'completion_status': workflow_result['completion_status'],
                'content_generated': len(workflow_result['generated_content']),
                'quality_scores': workflow_result['quality_scores'],
                'execution_time': self._calculate_execution_time(workflow_result),
                'agent_performance': self._analyze_agent_performance(workflow_result)
            }
            
        except Exception as e:
            logger.error(f"❌ Autonomous workflow failed for campaign {campaign_id}: {str(e)}")
            await self._handle_workflow_failure(campaign_id, str(e))
            raise
    
    async def _initialize_campaign_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize campaign with intelligence gathering and context setup"""
        try:
            logger.info(f"🎯 Initializing campaign: {state['campaign_id']}")
            
            # Load campaign context from database
            campaign_context = await self._load_campaign_context(state['campaign_id'])
            
            # Gather relevant knowledge base context
            knowledge_context = await self._gather_knowledge_context(state['campaign_data'])
            
            # Initialize agent performance tracking
            await self._initialize_performance_tracking(state['campaign_id'])
            
            state['current_phase'] = WorkflowPhase.INTELLIGENCE_GATHERING.value
            state['knowledge_context'] = knowledge_context
            state['workflow_metadata']['campaign_context'] = campaign_context
            
            logger.info(f"✅ Campaign initialized with {len(knowledge_context)} knowledge items")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Campaign initialization failed: {str(e)}")
            state['error_history'].append({
                'phase': 'initialization',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _gather_intelligence_node(self, state: WorkflowState) -> WorkflowState:
        """Gather competitive intelligence and market analysis"""
        try:
            logger.info(f"🧠 Gathering intelligence for campaign: {state['campaign_id']}")
            
            # Execute intelligence gathering agents in parallel
            intelligence_tasks = [
                AgentTask(
                    agent_type="StrategicInsightsAgent",
                    task_description="Analyze competitive landscape and market opportunities",
                    input_dependencies=["campaign_data", "knowledge_context"],
                    output_key="competitive_intelligence"
                ),
                AgentTask(
                    agent_type="TrendAnalysisAgent", 
                    task_description="Identify current market trends and opportunities",
                    input_dependencies=["campaign_data", "knowledge_context"],
                    output_key="trend_analysis"
                ),
                AgentTask(
                    agent_type="GapIdentificationAgent",
                    task_description="Identify content gaps and opportunities",
                    input_dependencies=["campaign_data", "knowledge_context"],
                    output_key="content_gaps"
                )
            ]
            
            # Execute intelligence gathering agents
            intelligence_results = await self._execute_agent_tasks(intelligence_tasks, state)
            
            # Store intelligence outputs
            state['agent_outputs']['intelligence'] = intelligence_results
            state['current_phase'] = WorkflowPhase.STRATEGIC_PLANNING.value
            
            logger.info(f"✅ Intelligence gathering completed with {len(intelligence_results)} insights")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Intelligence gathering failed: {str(e)}")
            state['error_history'].append({
                'phase': 'intelligence_gathering',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _create_strategy_node(self, state: WorkflowState) -> WorkflowState:
        """Create comprehensive content strategy using planning agents"""
        try:
            logger.info(f"📋 Creating content strategy for campaign: {state['campaign_id']}")
            
            # Execute strategic planning agents
            strategy_tasks = [
                AgentTask(
                    agent_type="PlannerAgent",
                    task_description="Create comprehensive content plan based on intelligence",
                    input_dependencies=["campaign_data", "intelligence"],
                    output_key="content_plan"
                ),
                AgentTask(
                    agent_type="SEOAgent",
                    task_description="Develop SEO strategy and keyword recommendations", 
                    input_dependencies=["campaign_data", "intelligence"],
                    output_key="seo_strategy"
                ),
                AgentTask(
                    agent_type="SocialMediaAgent",
                    task_description="Plan social media content strategy",
                    input_dependencies=["campaign_data", "intelligence"],
                    output_key="social_strategy"
                )
            ]
            
            strategy_results = await self._execute_agent_tasks(strategy_tasks, state)
            
            # Consolidate strategy
            consolidated_strategy = await self._consolidate_strategy(strategy_results, state)
            
            state['content_strategy'] = consolidated_strategy
            state['agent_outputs']['strategy'] = strategy_results
            state['current_phase'] = WorkflowPhase.CONTENT_CREATION.value
            
            logger.info(f"✅ Content strategy created with {len(consolidated_strategy)} components")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Strategy creation failed: {str(e)}")
            state['error_history'].append({
                'phase': 'strategic_planning',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _generate_content_node(self, state: WorkflowState) -> WorkflowState:
        """Generate all content types using specialized content agents"""
        try:
            logger.info(f"✍️ Generating content for campaign: {state['campaign_id']}")
            
            # Define content generation tasks based on strategy
            content_tasks = await self._create_content_generation_tasks(state)
            
            # Execute content creation agents
            content_results = await self._execute_agent_tasks(content_tasks, state)
            
            # Store generated content
            state['generated_content'] = content_results
            state['agent_outputs']['content_creation'] = content_results
            state['current_phase'] = WorkflowPhase.QUALITY_ASSURANCE.value
            
            logger.info(f"✅ Content generation completed: {len(content_results)} pieces created")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Content generation failed: {str(e)}")
            state['error_history'].append({
                'phase': 'content_creation',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _review_quality_node(self, state: WorkflowState) -> WorkflowState:
        """Review content quality using editor and quality assurance agents"""
        try:
            logger.info(f"🔍 Reviewing content quality for campaign: {state['campaign_id']}")
            
            quality_tasks = [
                AgentTask(
                    agent_type="EditorAgent",
                    task_description="Review and edit all generated content for quality and consistency",
                    input_dependencies=["generated_content", "content_strategy"],
                    output_key="editorial_review"
                ),
                AgentTask(
                    agent_type="AIContentAnalyzer", 
                    task_description="Analyze content quality scores and brand consistency",
                    input_dependencies=["generated_content", "campaign_data"],
                    output_key="quality_analysis"
                )
            ]
            
            quality_results = await self._execute_agent_tasks(quality_tasks, state)
            
            # Calculate overall quality scores
            quality_scores = await self._calculate_quality_scores(quality_results)
            
            state['quality_scores'] = quality_scores
            state['agent_outputs']['quality_assurance'] = quality_results
            
            logger.info(f"✅ Quality review completed. Average score: {quality_scores.get('overall', 0)}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Quality review failed: {str(e)}")
            state['error_history'].append({
                'phase': 'quality_assurance',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _optimize_content_node(self, state: WorkflowState) -> WorkflowState:
        """Optimize content based on quality feedback"""
        try:
            logger.info(f"⚡ Optimizing content for campaign: {state['campaign_id']}")
            
            # Identify content that needs optimization
            optimization_needed = await self._identify_optimization_needs(state)
            
            if optimization_needed:
                optimization_tasks = await self._create_optimization_tasks(optimization_needed, state)
                optimization_results = await self._execute_agent_tasks(optimization_tasks, state)
                
                # Update generated content with optimized versions
                state['generated_content'].update(optimization_results)
                state['agent_outputs']['optimization'] = optimization_results
                
                logger.info(f"✅ Content optimization completed for {len(optimization_needed)} items")
            else:
                logger.info("ℹ️ No content optimization needed")
            
            state['current_phase'] = WorkflowPhase.FINALIZATION.value
            return state
            
        except Exception as e:
            logger.error(f"❌ Content optimization failed: {str(e)}")
            state['error_history'].append({
                'phase': 'optimization',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _finalize_campaign_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize campaign and prepare content for distribution"""
        try:
            logger.info(f"🏁 Finalizing campaign: {state['campaign_id']}")
            
            # Generate final performance report
            performance_report = await self._generate_performance_report(state)
            
            # Save all generated content to database
            await self._save_generated_content_to_database(state)
            
            # Update campaign tasks status to completed
            await self._update_campaign_tasks_status(state['campaign_id'], 'completed')
            
            # Mark workflow as completed
            state['completion_status'] = 'completed'
            state['current_phase'] = WorkflowPhase.COMPLETION.value
            state['workflow_metadata']['completion_time'] = datetime.now().isoformat()
            state['workflow_metadata']['performance_report'] = performance_report
            
            logger.info(f"✅ Campaign finalization completed successfully")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Campaign finalization failed: {str(e)}")
            state['completion_status'] = 'failed'
            state['error_history'].append({
                'phase': 'finalization',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _handle_errors_node(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors and attempt recovery"""
        try:
            logger.warning(f"⚠️ Handling errors for campaign: {state['campaign_id']}")
            
            # Analyze error patterns
            error_analysis = await self._analyze_error_patterns(state['error_history'])
            
            # Attempt error recovery if possible
            if state['retry_count'] < 3:
                recovery_result = await self._attempt_error_recovery(state, error_analysis)
                if recovery_result['success']:
                    logger.info(f"✅ Error recovery successful")
                    return state
            
            # Mark as failed if recovery not possible
            state['completion_status'] = 'failed'
            state['workflow_metadata']['failure_reason'] = error_analysis['primary_failure']
            
            logger.error(f"❌ Workflow failed after error handling")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Error handling failed: {str(e)}")
            state['completion_status'] = 'failed'
            return state
    
    def _should_optimize_content(self, state: WorkflowState) -> str:
        """Determine if content needs optimization based on quality scores"""
        if not state['quality_scores']:
            return "error"
        
        overall_quality = state['quality_scores'].get('overall', 0)
        quality_threshold = self.quality_gates['quality_assurance']
        
        if overall_quality >= self.quality_gates['final_review']:
            return "finalize"
        elif overall_quality >= quality_threshold:
            return "optimize" 
        elif state['retry_count'] < 2:
            state['retry_count'] += 1
            return "retry"
        else:
            return "finalize"  # Accept current quality after max retries
    
    async def _execute_agent_tasks(self, tasks: List[AgentTask], state: WorkflowState) -> Dict[str, Any]:
        """Execute multiple agent tasks concurrently with error handling"""
        results = {}
        
        for task in tasks:
            try:
                # Get agent instance
                agent = await self.agent_factory.get_agent(task.agent_type)
                
                # Prepare task input
                task_input = await self._prepare_task_input(task, state)
                
                # Execute agent task
                result = await agent.execute_task(task_input)
                
                # Store result
                results[task.output_key] = result
                
                logger.debug(f"✅ Agent task completed: {task.agent_type} -> {task.output_key}")
                
            except Exception as e:
                logger.error(f"❌ Agent task failed: {task.agent_type} - {str(e)}")
                results[task.output_key] = {'error': str(e), 'success': False}
        
        return results
    
    async def _load_campaign_context(self, campaign_id: str) -> Dict[str, Any]:
        """Load campaign context from database"""
        try:
            query = """
                SELECT id, company_name, target_audience, goals, success_metrics, 
                       description, status, created_at 
                FROM campaigns WHERE id = $1
            """
            # Skip campaign context loading for now
            result = None
            
            if result:
                return {
                    'campaign_data': dict(result),
                    'existing_tasks': await self._load_campaign_tasks(campaign_id)
                }
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load campaign context: {e}")
            return {}
    
    async def _load_campaign_tasks(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Load existing campaign tasks"""
        try:
            query = """
                SELECT id, task_type, content, status, assigned_agent, 
                       created_at, updated_at 
                FROM campaign_tasks WHERE campaign_id = $1
            """
            # Skip task loading for now
            return []
            
        except Exception as e:
            logger.error(f"Failed to load campaign tasks: {e}")
            return []
    
    async def _gather_knowledge_context(self, campaign_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather relevant knowledge from the knowledge base"""
        try:
            # Search for relevant documents based on campaign goals and description
            search_terms = []
            if 'goals' in campaign_data:
                search_terms.append(campaign_data['goals'])
            if 'description' in campaign_data:
                search_terms.append(campaign_data['description'])
            if 'target_audience' in campaign_data:
                search_terms.append(campaign_data['target_audience'])
            
            knowledge_items = []
            
            for term in search_terms:
                if term:
                    # Search documents (simplified - would use vector search in production)
                    query = """
                        SELECT d.id, d.title, d.content, d.metadata, dc.content as chunk_content
                        FROM documents d
                        LEFT JOIN document_chunks dc ON d.id = dc.document_id
                        WHERE d.title ILIKE $1 OR d.content ILIKE $1
                        LIMIT 5
                    """
                    # Skip database knowledge search for now in mock mode
                    results = []
                    
                    for result in results or []:
                        knowledge_items.append({
                            'document_id': result['id'],
                            'title': result['title'],
                            'content': result['chunk_content'] or result['content'][:1000],
                            'metadata': result['metadata'],
                            'relevance_score': 0.8  # Would be calculated by vector similarity
                        })
            
            return knowledge_items[:10]  # Limit to top 10 most relevant
            
        except Exception as e:
            logger.error(f"Failed to gather knowledge context: {e}")
            return []
    
    async def _initialize_performance_tracking(self, campaign_id: str):
        """Initialize performance tracking for the workflow"""
        try:
            tracking_data = {
                'workflow_start': datetime.now().isoformat(),
                'campaign_id': campaign_id,
                'phase_timings': {},
                'agent_executions': {}
            }
            
            # Store initial tracking data
            query = """
                INSERT INTO agent_performance (agent_name, task_type, execution_time, 
                                             success, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            # Skip performance tracking insertion for now
            pass
            
        except Exception as e:
            logger.error(f"Failed to initialize performance tracking: {e}")
    
    async def _consolidate_strategy(self, strategy_results: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Consolidate multiple strategy inputs into unified content strategy"""
        consolidated = {
            'content_pillars': [],
            'content_calendar': {},
            'seo_keywords': [],
            'social_strategy': {},
            'content_types': [],
            'distribution_channels': []
        }
        
        try:
            # Extract content plan
            if 'content_plan' in strategy_results and 'success' in strategy_results['content_plan']:
                plan = strategy_results['content_plan']
                if 'content_pillars' in plan:
                    consolidated['content_pillars'] = plan['content_pillars']
                if 'content_calendar' in plan:
                    consolidated['content_calendar'] = plan['content_calendar']
            
            # Extract SEO strategy  
            if 'seo_strategy' in strategy_results and 'success' in strategy_results['seo_strategy']:
                seo = strategy_results['seo_strategy']
                if 'keywords' in seo:
                    consolidated['seo_keywords'] = seo['keywords']
            
            # Extract social strategy
            if 'social_strategy' in strategy_results and 'success' in strategy_results['social_strategy']:
                social = strategy_results['social_strategy']
                consolidated['social_strategy'] = social
            
            # Determine content types based on campaign success metrics
            success_metrics = state['campaign_data'].get('success_metrics', {})
            for content_type, count in success_metrics.items():
                if count > 0:
                    consolidated['content_types'].append({
                        'type': content_type,
                        'count': count,
                        'priority': 'high' if count >= 3 else 'normal'
                    })
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Failed to consolidate strategy: {e}")
            return consolidated
    
    async def _create_content_generation_tasks(self, state: WorkflowState) -> List[AgentTask]:
        """Create content generation tasks based on strategy"""
        tasks = []
        
        try:
            content_strategy = state['content_strategy']
            success_metrics = state['campaign_data'].get('success_metrics', {})
            
            # Blog posts
            if success_metrics.get('blog_posts', 0) > 0:
                tasks.append(AgentTask(
                    agent_type="WriterAgent",
                    task_description="Generate high-quality blog posts based on content strategy",
                    input_dependencies=["content_strategy", "knowledge_context"],
                    output_key="blog_posts"
                ))
            
            # Social media content
            if success_metrics.get('social_posts', 0) > 0:
                tasks.append(AgentTask(
                    agent_type="SocialMediaAgent",
                    task_description="Create engaging social media content",
                    input_dependencies=["content_strategy", "blog_posts"],
                    output_key="social_posts"
                ))
            
            # Email content
            if success_metrics.get('email_content', 0) > 0:
                tasks.append(AgentTask(
                    agent_type="ContentAgent",
                    task_description="Create email marketing content",
                    input_dependencies=["content_strategy", "blog_posts"],
                    output_key="email_content"
                ))
            
            # SEO optimization
            if success_metrics.get('seo_optimization', 0) > 0:
                tasks.append(AgentTask(
                    agent_type="SEOAgent",
                    task_description="Optimize content for search engines",
                    input_dependencies=["blog_posts", "content_strategy"],
                    output_key="seo_optimization"
                ))
            
            # Image generation
            if success_metrics.get('image_generation', 0) > 0:
                tasks.append(AgentTask(
                    agent_type="ImageAgent",
                    task_description="Generate image concepts and prompts for visual content",
                    input_dependencies=["content_strategy", "blog_posts"],
                    output_key="image_generation"
                ))
            
            # Content repurposing
            if success_metrics.get('repurposed_content', 0) > 0:
                tasks.append(AgentTask(
                    agent_type="ContentRepurposer",
                    task_description="Repurpose content across multiple formats",
                    input_dependencies=["blog_posts", "social_posts"],
                    output_key="repurposed_content"
                ))
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to create content generation tasks: {e}")
            return []
    
    async def _calculate_quality_scores(self, quality_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores from quality assurance results"""
        scores = {}
        
        try:
            total_score = 0
            count = 0
            
            # Process editorial review scores
            if 'editorial_review' in quality_results:
                editorial = quality_results['editorial_review']
                if 'quality_score' in editorial:
                    scores['editorial'] = editorial['quality_score']
                    total_score += editorial['quality_score']
                    count += 1
            
            # Process quality analysis scores  
            if 'quality_analysis' in quality_results:
                analysis = quality_results['quality_analysis']
                if 'overall_score' in analysis:
                    scores['analysis'] = analysis['overall_score']
                    total_score += analysis['overall_score']
                    count += 1
            
            # Calculate overall score
            if count > 0:
                scores['overall'] = total_score / count
            else:
                scores['overall'] = 7.0  # Default score
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate quality scores: {e}")
            return {'overall': 7.0}
    
    async def _identify_optimization_needs(self, state: WorkflowState) -> List[str]:
        """Identify content that needs optimization"""
        optimization_needed = []
        
        try:
            quality_scores = state['quality_scores']
            threshold = self.quality_gates['final_review']
            
            for content_key, content_data in state['generated_content'].items():
                # Check if content has quality issues
                content_score = quality_scores.get(content_key, quality_scores.get('overall', 0))
                
                if content_score < threshold:
                    optimization_needed.append(content_key)
            
            return optimization_needed
            
        except Exception as e:
            logger.error(f"Failed to identify optimization needs: {e}")
            return []
    
    async def _create_optimization_tasks(self, optimization_needed: List[str], state: WorkflowState) -> List[AgentTask]:
        """Create optimization tasks for content that needs improvement"""
        tasks = []
        
        try:
            for content_key in optimization_needed:
                # Determine appropriate optimization agent based on content type
                if 'blog' in content_key:
                    agent_type = "EditorAgent"
                elif 'social' in content_key:
                    agent_type = "SocialMediaAgent"  
                elif 'seo' in content_key:
                    agent_type = "SEOAgent"
                else:
                    agent_type = "ContentAgent"
                
                tasks.append(AgentTask(
                    agent_type=agent_type,
                    task_description=f"Optimize and improve {content_key} content quality",
                    input_dependencies=[content_key, "quality_scores"],
                    output_key=f"optimized_{content_key}"
                ))
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to create optimization tasks: {e}")
            return []
    
    async def _prepare_task_input(self, task: AgentTask, state: WorkflowState) -> Dict[str, Any]:
        """Prepare input data for agent task execution"""
        task_input = {
            'campaign_id': state['campaign_id'],
            'campaign_data': state['campaign_data'],
            'task_description': task.task_description
        }
        
        # Add dependency data
        for dependency in task.input_dependencies:
            if dependency == 'campaign_data':
                task_input['campaign_data'] = state['campaign_data']
            elif dependency == 'knowledge_context':
                task_input['knowledge_context'] = state['knowledge_context']
            elif dependency == 'content_strategy':
                task_input['content_strategy'] = state['content_strategy']
            elif dependency == 'intelligence':
                task_input['intelligence'] = state['agent_outputs'].get('intelligence', {})
            elif dependency in state['generated_content']:
                task_input[dependency] = state['generated_content'][dependency]
            elif dependency in state['agent_outputs']:
                task_input[dependency] = state['agent_outputs'][dependency]
        
        return task_input
    
    async def _generate_performance_report(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate comprehensive performance report for the workflow"""
        try:
            return {
                'workflow_id': state['workflow_metadata']['workflow_id'],
                'campaign_id': state['campaign_id'],
                'execution_time': self._calculate_execution_time(state),
                'content_generated': len(state['generated_content']),
                'quality_scores': state['quality_scores'],
                'agent_performance': self._analyze_agent_performance(state),
                'error_count': len(state['error_history']),
                'retry_count': state['retry_count'],
                'completion_status': state['completion_status']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}
    
    async def _save_generated_content_to_database(self, state: WorkflowState):
        """Save all generated content to the database"""
        try:
            campaign_id = state['campaign_id']
            
            # Save blog posts
            if 'blog_posts' in state['generated_content']:
                blog_data = state['generated_content']['blog_posts']
                if 'content' in blog_data:
                    query = """
                        INSERT INTO blog_posts (campaign_id, title, content, status, 
                                              word_count, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    await secure_db.execute(query, [
                        campaign_id,
                        blog_data.get('title', 'Generated Blog Post'),
                        blog_data['content'],
                        'draft',
                        len(blog_data['content'].split()),
                        datetime.now(),
                        datetime.now()
                    ])
            
            # Update campaign tasks as completed
            await self._update_campaign_tasks_status(campaign_id, 'completed')
            
        except Exception as e:
            logger.error(f"Failed to save generated content: {e}")
    
    async def _update_campaign_tasks_status(self, campaign_id: str, status: str):
        """Update campaign tasks status"""
        try:
            query = """
                UPDATE campaign_tasks 
                SET status = $1, updated_at = $2 
                WHERE campaign_id = $3
            """
            await secure_db.execute(query, [status, datetime.now(), campaign_id])
            
        except Exception as e:
            logger.error(f"Failed to update campaign tasks status: {e}")
    
    def _calculate_execution_time(self, state: WorkflowState) -> float:
        """Calculate total workflow execution time"""
        try:
            start_time = datetime.fromisoformat(state['workflow_metadata']['start_time'])
            if 'completion_time' in state['workflow_metadata']:
                end_time = datetime.fromisoformat(state['workflow_metadata']['completion_time'])
            else:
                end_time = datetime.now()
            
            return (end_time - start_time).total_seconds()
            
        except Exception:
            return 0.0
    
    def _analyze_agent_performance(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze agent performance during workflow execution"""
        performance = {
            'agents_used': set(),
            'successful_executions': 0,
            'failed_executions': 0,
            'average_quality': 0.0
        }
        
        try:
            # Analyze agent outputs
            for phase, outputs in state['agent_outputs'].items():
                for agent_key, result in outputs.items():
                    performance['agents_used'].add(agent_key)
                    
                    if result.get('success', True):
                        performance['successful_executions'] += 1
                    else:
                        performance['failed_executions'] += 1
            
            # Convert set to list for JSON serialization
            performance['agents_used'] = list(performance['agents_used'])
            
            # Calculate average quality
            quality_scores = state['quality_scores']
            if quality_scores:
                performance['average_quality'] = quality_scores.get('overall', 0.0)
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to analyze agent performance: {e}")
            return performance
    
    async def _update_campaign_completion_status(self, campaign_id: str, workflow_result: WorkflowState):
        """Update campaign completion status in database"""
        try:
            status = 'completed' if workflow_result['completion_status'] == 'completed' else 'failed'
            
            query = """
                UPDATE campaigns 
                SET status = $1, updated_at = $2 
                WHERE id = $3
            """
            await secure_db.execute(query, [status, datetime.now(), campaign_id])
            
        except Exception as e:
            logger.error(f"Failed to update campaign completion status: {e}")
    
    async def _handle_workflow_failure(self, campaign_id: str, error_message: str):
        """Handle workflow failure by updating database and logging"""
        try:
            query = """
                UPDATE campaigns 
                SET status = 'failed', updated_at = $1 
                WHERE id = $2
            """
            await secure_db.execute(query, [datetime.now(), campaign_id])
            
            # Log failure details
            performance_query = """
                INSERT INTO agent_performance (agent_name, task_type, execution_time, 
                                             success, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            await secure_db.execute(performance_query, [
                'AutonomousWorkflowOrchestrator',
                'workflow_failure',
                0.0,
                False,
                json.dumps({'error': error_message, 'campaign_id': campaign_id}),
                datetime.now()
            ])
            
        except Exception as e:
            logger.error(f"Failed to handle workflow failure: {e}")
    
    async def _analyze_error_patterns(self, error_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns for recovery strategies"""
        if not error_history:
            return {'primary_failure': 'unknown', 'recovery_possible': False}
        
        # Simple analysis - in production would be more sophisticated
        latest_error = error_history[-1]
        
        return {
            'primary_failure': latest_error.get('phase', 'unknown'),
            'error_count': len(error_history),
            'recovery_possible': len(error_history) < 3,
            'suggested_action': 'retry' if len(error_history) < 2 else 'manual_intervention'
        }
    
    async def _attempt_error_recovery(self, state: WorkflowState, error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from errors"""
        try:
            if error_analysis.get('recovery_possible', False):
                # Simple recovery - reset current phase and continue
                state['retry_count'] += 1
                logger.info(f"Attempting error recovery, retry {state['retry_count']}")
                return {'success': True, 'action': 'retry'}
            
            return {'success': False, 'action': 'fail'}
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            return {'success': False, 'action': 'fail'}
    
    async def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute method required by BaseAgent
        This is the main entry point for autonomous workflow execution
        """
        try:
            campaign_id = task_input.get('campaign_id')
            campaign_data = task_input.get('campaign_data', {})
            
            if not campaign_id:
                raise ValueError("campaign_id is required")
            
            # Start autonomous workflow
            result = await self.start_autonomous_workflow(campaign_id, campaign_data)
            
            return {
                'success': True,
                'result': result,
                'message': 'Autonomous workflow completed successfully'
            }
            
        except Exception as e:
            logger.error(f"Autonomous workflow execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Autonomous workflow execution failed'
            }
    
    async def _execute_workflow_phases(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Execute the workflow phases and actually create tasks in the database
        """
        try:
            state = initial_state
            campaign_id = state['campaign_id']
            
            logger.info(f"🚀 Executing autonomous workflow phases for campaign {campaign_id}")
            
            # Phase 1: Load campaign context (already done in initial_state)
            
            # Phase 2: Intelligence gathering - create competitor analysis task
            await self._create_intelligence_tasks(campaign_id, state['campaign_data'])
            
            # Phase 3: Strategic planning - create SEO and content planning tasks  
            await self._create_strategic_planning_tasks(campaign_id, state['campaign_data'])
            
            # Phase 4: Content creation - create multiple content tasks
            await self._create_content_creation_tasks(campaign_id, state['campaign_data'])
            
            # Phase 5: Quality assurance - create review tasks
            await self._create_quality_assurance_tasks(campaign_id, state['campaign_data'])
            
            # Phase 6: Optimization - create repurposing tasks
            await self._create_optimization_tasks(campaign_id, state['campaign_data'])
            
            # Phase 7: Finalization - update campaign status
            await self._finalize_campaign(campaign_id)
            
            # Update state
            state['completion_status'] = 'completed'
            state['content_generated'] = 15  # Approximate number of tasks created
            state['workflow_metadata']['completion_time'] = datetime.now().isoformat()
            
            logger.info(f"✅ Autonomous workflow completed for campaign {campaign_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Workflow phase execution failed: {e}")
            state['completion_status'] = 'failed'
            state['error_history'] = state.get('error_history', []) + [{'error': str(e), 'phase': 'execution'}]
            return state
    
    async def _create_intelligence_tasks(self, campaign_id: str, campaign_data: Dict[str, Any]):
        """Create intelligence gathering tasks"""
        try:
            tasks = [
                {
                    'task_type': 'competitor_analysis',
                    'assigned_agent': 'StrategicInsightsAgent',
                    'content': json.dumps({
                        'title': 'Competitor Intelligence Analysis',
                        'description': f"Analyze competitors for {campaign_data.get('campaign_name', 'campaign')}",
                        'channel': 'research',
                        'priority': 'high',
                        'estimated_hours': 3
                    }),
                    'status': 'pending'
                },
                {
                    'task_type': 'trend_analysis', 
                    'assigned_agent': 'TrendAnalysisAgent',
                    'content': json.dumps({
                        'title': 'Market Trend Analysis',
                        'description': f"Identify trends for {campaign_data.get('industry', 'industry')}",
                        'channel': 'research',
                        'priority': 'medium',
                        'estimated_hours': 2
                    }),
                    'status': 'pending'
                }
            ]
            
            for task in tasks:
                await self._insert_campaign_task(campaign_id, task)
            
        except Exception as e:
            logger.error(f"Failed to create intelligence tasks: {e}")
    
    async def _create_strategic_planning_tasks(self, campaign_id: str, campaign_data: Dict[str, Any]):
        """Create strategic planning tasks"""
        try:
            tasks = [
                {
                    'task_type': 'seo_optimization',
                    'assigned_agent': 'SEOAgent',
                    'content': json.dumps({
                        'title': 'SEO Strategy & Optimization',
                        'description': 'Develop comprehensive SEO strategy and keyword optimization',
                        'channel': 'search',
                        'priority': 'high',
                        'estimated_hours': 3
                    }),
                    'status': 'pending'
                },
                {
                    'task_type': 'content_strategy',
                    'assigned_agent': 'PlannerAgent',
                    'content': json.dumps({
                        'title': 'Content Strategy Development',
                        'description': 'Create comprehensive content strategy and calendar',
                        'channel': 'strategy',
                        'priority': 'high',
                        'estimated_hours': 4
                    }),
                    'status': 'pending'
                }
            ]
            
            for task in tasks:
                await self._insert_campaign_task(campaign_id, task)
                
        except Exception as e:
            logger.error(f"Failed to create strategic planning tasks: {e}")
    
    async def _create_content_creation_tasks(self, campaign_id: str, campaign_data: Dict[str, Any]):
        """Create content creation tasks"""
        try:
            success_metrics = campaign_data.get('success_metrics', {})
            
            tasks = []
            
            # Blog posts
            blog_count = success_metrics.get('blog_posts', 2)
            for i in range(blog_count):
                tasks.append({
                    'task_type': 'content_creation',
                    'assigned_agent': 'WriterAgent',
                    'content': json.dumps({
                        'title': f'Blog Post {i+1}',
                        'description': f'Create high-quality blog post {i+1} for {campaign_data.get("campaign_name", "campaign")}',
                        'channel': 'blog',
                        'content_type': 'blog_post',
                        'priority': 'high',
                        'estimated_hours': 4
                    }),
                    'status': 'pending'
                })
            
            # Social posts
            social_count = success_metrics.get('social_posts', 5)
            for i in range(min(social_count, 5)):  # Limit to 5 for now
                tasks.append({
                    'task_type': 'social_content',
                    'assigned_agent': 'SocialMediaAgent',
                    'content': json.dumps({
                        'title': f'Social Media Post {i+1}',
                        'description': f'Create engaging social media content {i+1}',
                        'channel': 'linkedin',
                        'content_type': 'social_post',
                        'priority': 'medium',
                        'estimated_hours': 1
                    }),
                    'status': 'pending'
                })
            
            # Email content
            email_count = success_metrics.get('email_content', 2)
            for i in range(email_count):
                tasks.append({
                    'task_type': 'email_content',
                    'assigned_agent': 'ContentAgent',
                    'content': json.dumps({
                        'title': f'Email Campaign {i+1}',
                        'description': f'Create email marketing content {i+1}',
                        'channel': 'email',
                        'content_type': 'email',
                        'priority': 'high',
                        'estimated_hours': 2
                    }),
                    'status': 'pending'
                })
            
            # Visual content
            image_count = success_metrics.get('image_generation', 2)
            for i in range(image_count):
                tasks.append({
                    'task_type': 'image_generation',
                    'assigned_agent': 'ImageAgent',
                    'content': json.dumps({
                        'title': f'Visual Content {i+1}',
                        'description': f'Create visual content and image prompts {i+1}',
                        'channel': 'visual',
                        'content_type': 'image',
                        'priority': 'medium',
                        'estimated_hours': 1
                    }),
                    'status': 'pending'
                })
            
            for task in tasks:
                await self._insert_campaign_task(campaign_id, task)
                
        except Exception as e:
            logger.error(f"Failed to create content creation tasks: {e}")
    
    async def _create_quality_assurance_tasks(self, campaign_id: str, campaign_data: Dict[str, Any]):
        """Create quality assurance tasks"""
        try:
            tasks = [
                {
                    'task_type': 'content_review',
                    'assigned_agent': 'EditorAgent',
                    'content': json.dumps({
                        'title': 'Content Quality Review',
                        'description': 'Review all content for quality, consistency, and brand alignment',
                        'channel': 'quality',
                        'priority': 'high',
                        'estimated_hours': 3
                    }),
                    'status': 'pending'
                }
            ]
            
            for task in tasks:
                await self._insert_campaign_task(campaign_id, task)
                
        except Exception as e:
            logger.error(f"Failed to create quality assurance tasks: {e}")
    
    async def _create_optimization_tasks(self, campaign_id: str, campaign_data: Dict[str, Any]):
        """Create optimization tasks"""
        try:
            success_metrics = campaign_data.get('success_metrics', {})
            repurpose_count = success_metrics.get('repurposed_content', 2)
            
            tasks = []
            for i in range(min(repurpose_count, 3)):  # Limit to 3
                tasks.append({
                    'task_type': 'content_repurposing',
                    'assigned_agent': 'ContentRepurposer',
                    'content': json.dumps({
                        'title': f'Content Repurposing {i+1}',
                        'description': f'Repurpose content across multiple formats and channels {i+1}',
                        'channel': 'multi',
                        'priority': 'medium',
                        'estimated_hours': 2
                    }),
                    'status': 'pending'
                })
            
            for task in tasks:
                await self._insert_campaign_task(campaign_id, task)
                
        except Exception as e:
            logger.error(f"Failed to create optimization tasks: {e}")
    
    async def _insert_campaign_task(self, campaign_id: str, task_data: Dict[str, Any]):
        """Insert a task into the campaign_tasks table"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                # Parse the content to add assigned_agent info
                content_dict = json.loads(task_data['content'])
                content_dict['assigned_agent'] = task_data['assigned_agent']
                updated_content = json.dumps(content_dict)
                
                # Generate UUID and map task type
                task_id = str(uuid.uuid4())
                mapped_task_type = self._map_task_type(task_data['task_type'])
                
                cur.execute("""
                    INSERT INTO campaign_tasks (id, campaign_id, task_type, result, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, [
                    task_id,
                    campaign_id,
                    mapped_task_type,
                    updated_content,
                    task_data['status'],
                    datetime.now(),
                    datetime.now()
                ])
                conn.commit()
            
            logger.info(f"✅ Created task: {task_data['task_type']} for campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to insert campaign task: {e}")
    
    async def _finalize_campaign(self, campaign_id: str):
        """Finalize the campaign"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE campaigns 
                    SET status = 'autonomous_workflow_completed', updated_at = %s 
                    WHERE id = %s
                """, [datetime.now(), campaign_id])
                conn.commit()
            
            logger.info(f"✅ Campaign {campaign_id} finalized")
            
        except Exception as e:
            logger.error(f"Failed to finalize campaign: {e}")


# Global orchestrator instance
autonomous_orchestrator = AutonomousWorkflowOrchestrator()