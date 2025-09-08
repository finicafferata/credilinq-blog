"""
LangGraph to Legacy Adapter
Enables modern LangGraph agents to work seamlessly in legacy workflow systems.

This adapter implements the bridge pattern to translate between:
- Legacy interface (ReviewAgentBase with execute_safe)
- Modern LangGraph agents (LangGraphWorkflowBase with execute)

Key Features:
- Maintains backward compatibility with existing workflows
- Translates data formats between legacy and modern systems
- Preserves error handling and logging patterns
- Enables gradual migration without breaking changes
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, Union, List
from datetime import datetime
import json

from src.agents.workflow.review_agent_base import ReviewAgentBase
from src.agents.workflow.review_workflow_models import ReviewAgentResult, ReviewDecision
from src.agents.core.langgraph_base import LangGraphWorkflowBase, WorkflowStatus
from src.agents.core.base_agent import AgentResult

logger = logging.getLogger(__name__)


class LangGraphLegacyAdapter(ReviewAgentBase):
    """
    Adapter that allows LangGraph agents to work in legacy ReviewAgentBase workflows.
    
    This adapter:
    1. Accepts legacy execute_safe() calls
    2. Translates input to LangGraph format
    3. Executes the LangGraph agent
    4. Translates output back to legacy format
    """
    
    def __init__(
        self,
        langgraph_agent: Union[LangGraphWorkflowBase, Any],
        agent_name: Optional[str] = None,
        agent_description: Optional[str] = None,
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None
    ):
        """
        Initialize the adapter with a LangGraph agent.
        
        Args:
            langgraph_agent: The modern LangGraph agent to wrap
            agent_name: Override name for the agent
            agent_description: Override description
            input_transformer: Custom function to transform legacy input to LangGraph format
            output_transformer: Custom function to transform LangGraph output to legacy format
        """
        # Get agent info from the wrapped agent if available
        if agent_name is None:
            agent_name = getattr(langgraph_agent, 'name', langgraph_agent.__class__.__name__)
        
        if agent_description is None:
            agent_description = getattr(
                langgraph_agent,
                'description',
                f"LangGraph adapter for {agent_name}"
            )
        
        super().__init__(
            name=agent_name,
            description=agent_description,
            version="2.0.0-adapter"
        )
        
        self.langgraph_agent = langgraph_agent
        self.input_transformer = input_transformer or self._default_input_transformer
        self.output_transformer = output_transformer or self._default_output_transformer
        
        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0
        self.error_count = 0
        
        logger.info(f"Initialized LangGraph adapter for {agent_name}")
    
    async def execute_safe(self, content_data: Dict[str, Any], **kwargs) -> ReviewAgentResult:
        """
        Legacy interface method that adapts to LangGraph execution.
        
        Args:
            content_data: Legacy format input data
            **kwargs: Additional parameters
            
        Returns:
            ReviewAgentResult in legacy format
        """
        start_time = datetime.utcnow()
        self.execution_count += 1
        
        try:
            # Step 1: Transform legacy input to LangGraph format
            langgraph_input = await self._transform_input(content_data, **kwargs)
            
            self.logger.debug(f"Transformed input for {self.name}: {list(langgraph_input.keys())}")
            
            # Step 2: Execute LangGraph agent
            if hasattr(self.langgraph_agent, 'execute'):
                # Standard LangGraph agent
                langgraph_result = await self.langgraph_agent.execute(langgraph_input)
            elif hasattr(self.langgraph_agent, 'invoke'):
                # Alternative invoke method
                langgraph_result = await self.langgraph_agent.invoke(langgraph_input)
            elif asyncio.iscoroutinefunction(self.langgraph_agent):
                # Direct async function
                langgraph_result = await self.langgraph_agent(langgraph_input)
            else:
                # Fallback for other patterns
                langgraph_result = await self._execute_fallback(langgraph_input)
            
            # Step 3: Transform LangGraph output to legacy format
            legacy_result = await self._transform_output(langgraph_result, content_data)
            
            # Track performance
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_execution_time += execution_time
            
            self.logger.info(
                f"Successfully executed {self.name} via adapter "
                f"(execution_time: {execution_time:.2f}s, "
                f"score: {legacy_result.score:.2f})"
            )
            
            return legacy_result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error in LangGraph adapter for {self.name}: {str(e)}")
            
            # Return a safe failure result
            from src.agents.workflow.review_workflow_models import ReviewStage
            
            return ReviewAgentResult(
                stage=ReviewStage.CONTENT_QUALITY,
                content_id=content_data.get('content_id', 'unknown'),
                automated_score=0.0,
                confidence=0.0,
                feedback=["Please check the agent configuration and try again"],
                issues_found=[f"Adapter execution failed: {str(e)}"],
                recommendations=["Please check the agent configuration and try again"],
                metrics={
                    "adapter_error": str(e),
                    "agent_name": self.name,
                    "execution_count": self.execution_count,
                    "error_count": self.error_count,
                    "decision": ReviewDecision.REJECT.value
                },
                requires_human_review=True,
                auto_approved=False,
                execution_time_ms=0,
                model_used="adapter_error",
                tokens_used=0,
                cost=0.0
            )
    
    async def _transform_input(self, content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Transform legacy input format to LangGraph format.
        
        Args:
            content_data: Legacy format input
            **kwargs: Additional parameters
            
        Returns:
            LangGraph-compatible input dictionary
        """
        # Use custom transformer if provided
        if self.input_transformer != self._default_input_transformer:
            return await self._ensure_async(self.input_transformer, content_data, **kwargs)
        
        # Default transformation logic
        langgraph_input = {}
        
        # Map common fields
        if 'content' in content_data:
            langgraph_input['content'] = content_data['content']
            langgraph_input['content_to_edit'] = content_data['content']  # For EditorAgent
            
        if 'title' in content_data:
            langgraph_input['title'] = content_data['title']
            
        if 'target_audience' in content_data:
            langgraph_input['target_audience'] = content_data['target_audience']
            
        if 'content_type' in content_data:
            langgraph_input['content_type'] = content_data['content_type']
            
        # Map review-specific fields
        if 'brand_guidelines' in content_data:
            langgraph_input['brand_voice'] = content_data.get('brand_tone', 'professional')
            langgraph_input['style_guide'] = {
                'brand_guidelines': content_data['brand_guidelines'],
                'tone': content_data.get('brand_tone', 'professional')
            }
            
        # Map quality requirements
        if 'quality_requirements' in content_data:
            langgraph_input['editing_objectives'] = [
                'improve_quality',
                'enhance_clarity',
                'ensure_consistency'
            ]
            
        # Add metadata
        langgraph_input['metadata'] = {
            'source': 'legacy_adapter',
            'original_stage': content_data.get('review_stage', 'quality_review'),
            'campaign_id': content_data.get('campaign_id'),
            'task_id': content_data.get('task_id')
        }
        
        # Pass through any additional kwargs
        langgraph_input.update(kwargs)
        
        return langgraph_input
    
    async def _transform_output(
        self,
        langgraph_result: Union[AgentResult, Dict[str, Any], Any],
        original_input: Dict[str, Any]
    ) -> ReviewAgentResult:
        """
        Transform LangGraph output to legacy ReviewAgentResult format.
        
        Args:
            langgraph_result: Output from LangGraph agent
            original_input: Original legacy input for context
            
        Returns:
            ReviewAgentResult in legacy format
        """
        # Use custom transformer if provided
        if self.output_transformer != self._default_output_transformer:
            return await self._ensure_async(
                self.output_transformer,
                langgraph_result,
                original_input
            )
        
        # Extract data based on result type
        if isinstance(langgraph_result, AgentResult):
            # Standard AgentResult from LangGraph
            success = langgraph_result.success
            data = langgraph_result.data or {}
            metadata = langgraph_result.metadata or {}
            error_message = langgraph_result.error_message
            
        elif isinstance(langgraph_result, dict):
            # Dictionary result
            success = langgraph_result.get('success', True)
            data = langgraph_result.get('data', langgraph_result)
            metadata = langgraph_result.get('metadata', {})
            error_message = langgraph_result.get('error_message')
            
        else:
            # Unknown format - try to extract what we can
            success = True
            data = {'result': str(langgraph_result)}
            metadata = {}
            error_message = None
        
        # Extract scores and quality metrics
        score = self._extract_score(data, metadata)
        issues = self._extract_issues(data, metadata)
        suggestions = self._extract_suggestions(data, metadata)
        
        # Determine review decision
        if error_message:
            decision = ReviewDecision.REJECT
            passed = False
        elif score >= 0.85:
            decision = ReviewDecision.APPROVE
            passed = True
        elif score >= 0.70:
            decision = ReviewDecision.REQUEST_CHANGES
            passed = True
        else:
            decision = ReviewDecision.REJECT
            passed = False
        
        # Build legacy result
        from src.agents.workflow.review_workflow_models import ReviewStage
        
        # Map stage string to ReviewStage enum
        stage_str = original_input.get('review_stage', 'quality_review')
        stage_map = {
            'quality_review': ReviewStage.CONTENT_QUALITY,
            'content_quality': ReviewStage.CONTENT_QUALITY,
            'editorial_review': ReviewStage.EDITORIAL_REVIEW,
            'brand_check': ReviewStage.BRAND_CHECK,
            'seo_analysis': ReviewStage.SEO_ANALYSIS,
            'geo_analysis': ReviewStage.GEO_ANALYSIS,
            'visual_review': ReviewStage.VISUAL_REVIEW,
            'social_media_review': ReviewStage.SOCIAL_MEDIA_REVIEW,
            'final_approval': ReviewStage.FINAL_APPROVAL
        }
        stage = stage_map.get(stage_str, ReviewStage.CONTENT_QUALITY)
        
        return ReviewAgentResult(
            stage=stage,
            content_id=original_input.get('content_id', 'unknown'),
            automated_score=score,
            confidence=min(score + 0.1, 1.0),  # Slightly higher confidence than score
            feedback=suggestions,
            issues_found=issues,
            recommendations=suggestions,
            metrics={
                **metadata,
                'adapter_version': self.version,
                'langgraph_agent': self.langgraph_agent.__class__.__name__ if hasattr(self.langgraph_agent, '__class__') else str(type(self.langgraph_agent)),
                'execution_count': self.execution_count,
                'decision': decision.value
            },
            requires_human_review=not passed or score < 0.8,
            auto_approved=score >= self.auto_approve_threshold,
            execution_time_ms=int((end_time - start_time) * 1000) if 'start_time' in locals() and 'end_time' in locals() else 0,
            model_used=data.get('model_used', 'unknown'),
            tokens_used=data.get('tokens_used', 0),
            cost=data.get('cost', 0.0)
        )
    
    def _extract_score(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Extract quality score from LangGraph result."""
        # Try multiple possible score locations
        score_keys = [
            'quality_score', 'score', 'overall_score',
            'final_score', 'confidence', 'quality'
        ]
        
        for key in score_keys:
            if key in data:
                return float(data[key])
            if key in metadata:
                return float(metadata[key])
        
        # Check nested quality metrics
        if 'quality_metrics' in data:
            metrics = data['quality_metrics']
            if 'overall' in metrics:
                return float(metrics['overall'])
            # Average of all metrics
            if metrics:
                return sum(float(v) for v in metrics.values()) / len(metrics)
        
        # Default to a moderate score
        return 0.75
    
    def _extract_issues(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Extract issues from LangGraph result."""
        issues = []
        
        # Direct issues list
        if 'issues' in data:
            issues.extend(data['issues'])
        
        # Improvement areas
        if 'improvement_areas' in data:
            issues.extend(data['improvement_areas'])
        
        # Error messages
        if 'errors' in data:
            issues.extend(data['errors'])
        
        # Quality issues
        if 'quality_issues' in data:
            issues.extend(data['quality_issues'])
        
        return issues or ["No specific issues identified"]
    
    def _extract_suggestions(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Extract suggestions from LangGraph result."""
        suggestions = []
        
        # Direct suggestions
        if 'suggestions' in data:
            suggestions.extend(data['suggestions'])
        
        # Recommendations
        if 'recommendations' in data:
            suggestions.extend(data['recommendations'])
        
        # Editing feedback
        if 'editing_feedback' in data:
            suggestions.extend(data['editing_feedback'])
        
        # Optimization suggestions
        if 'optimizations' in data:
            suggestions.extend(data['optimizations'])
        
        return suggestions or ["Content meets quality standards"]
    
    async def _execute_fallback(self, langgraph_input: Dict[str, Any]) -> AgentResult:
        """
        Fallback execution method for non-standard agent patterns.
        """
        self.logger.warning(f"Using fallback execution for {self.name}")
        
        # Try to call the agent directly
        if callable(self.langgraph_agent):
            result = self.langgraph_agent(langgraph_input)
            if asyncio.iscoroutine(result):
                result = await result
            
            # If result is a dict, convert to AgentResult
            if isinstance(result, dict):
                return AgentResult(
                    success=result.get('success', True),
                    data=result.get('data', result),  # Use entire result as data if no 'data' key
                    metadata=result.get('metadata', {'fallback_used': True})
                )
            
            # If it's already an AgentResult, return as-is
            return result
        
        # Return a default failure result
        return AgentResult(
            success=False,
            data={},
            metadata={
                'fallback_used': True,
                'error_message': f"Unable to execute agent {self.name}"
            }
        )
    
    async def _ensure_async(self, func: callable, *args, **kwargs):
        """Ensure a function runs asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _default_input_transformer(self, content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Default input transformation (can be overridden)."""
        # This is called by _transform_input as the default
        return content_data
    
    def _default_output_transformer(
        self,
        langgraph_result: Any,
        original_input: Dict[str, Any]
    ) -> ReviewAgentResult:
        """Default output transformation (can be overridden)."""
        # This is called by _transform_output as the default
        return langgraph_result
    
    @property
    def auto_approve_threshold(self) -> float:
        """Get the auto-approval threshold."""
        return 0.85
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get adapter performance statistics."""
        avg_execution_time = (
            self.total_execution_time / self.execution_count
            if self.execution_count > 0
            else 0
        )
        
        return {
            'agent_name': self.name,
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'success_rate': 1 - (self.error_count / max(self.execution_count, 1)),
            'average_execution_time': avg_execution_time,
            'total_execution_time': self.total_execution_time
        }


class AdapterFactory:
    """
    Factory for creating adapters for different LangGraph agents.
    Provides pre-configured adapters for common agent types.
    """
    
    @staticmethod
    def create_editor_adapter():
        """Create adapter for EditorAgentLangGraph."""
        from src.agents.specialized.editor_agent_langgraph import EditorAgentLangGraph
        
        def input_transformer(content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Transform legacy input for EditorAgent."""
            return {
                'content_to_edit': content_data.get('content', ''),
                'editing_objectives': [
                    'improve_quality',
                    'enhance_clarity',
                    'ensure_consistency',
                    'check_grammar',
                    'optimize_readability'
                ],
                'target_audience': content_data.get('target_audience', 'general'),
                'content_type': content_data.get('content_type', 'article'),
                'brand_voice': content_data.get('brand_tone', 'professional'),
                'style_guide': {
                    'guidelines': content_data.get('brand_guidelines', {}),
                    'tone': content_data.get('brand_tone', 'professional')
                }
            }
        
        editor_agent = EditorAgentLangGraph()
        return LangGraphLegacyAdapter(
            langgraph_agent=editor_agent,
            agent_name="EditorAgent",
            agent_description="Advanced multi-phase editing with quality optimization",
            input_transformer=input_transformer
        )
    
    @staticmethod
    def create_seo_adapter():
        """Create adapter for SEOAgentLangGraph."""
        from src.agents.specialized.seo_agent_langgraph import SEOAgentLangGraph
        
        def input_transformer(content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Transform legacy input for SEOAgent."""
            return {
                'content': content_data.get('content', ''),
                'title': content_data.get('title', ''),
                'target_keywords': content_data.get('seo_keywords', []),
                'content_type': content_data.get('content_type', 'blog_post'),
                'target_audience': content_data.get('target_audience', 'general'),
                'competitor_urls': content_data.get('competitor_urls', [])
            }
        
        seo_agent = SEOAgentLangGraph()
        return LangGraphLegacyAdapter(
            langgraph_agent=seo_agent,
            agent_name="SEOAgent",
            agent_description="Comprehensive SEO optimization with keyword analysis",
            input_transformer=input_transformer
        )
    
    @staticmethod
    def create_writer_adapter():
        """Create adapter for WriterAgentLangGraph."""
        from src.agents.specialized.writer_agent_langgraph import WriterAgentLangGraph
        
        def input_transformer(content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Transform legacy input for WriterAgent."""
            return {
                'content_brief': {
                    'topic': content_data.get('title', ''),
                    'description': content_data.get('description', ''),
                    'key_points': content_data.get('key_messages', []),
                    'target_audience': content_data.get('target_audience', 'general')
                },
                'research_data': content_data.get('research_data', {}),
                'target_audience': content_data.get('target_audience', 'general'),
                'content_type': content_data.get('content_type', 'article'),
                'word_count_target': content_data.get('word_count', 1000),
                'tone': content_data.get('tone', 'professional'),
                'seo_keywords': content_data.get('seo_keywords', [])
            }
        
        writer_agent = WriterAgentLangGraph()
        return LangGraphLegacyAdapter(
            langgraph_agent=writer_agent,
            agent_name="WriterAgent",
            agent_description="Advanced content creation with iterative quality improvement",
            input_transformer=input_transformer
        )
    
    @staticmethod
    def create_brand_review_adapter():
        """Create adapter for brand review using EditorAgent with brand focus."""
        from src.agents.specialized.editor_agent_langgraph import EditorAgentLangGraph
        
        def brand_input_transformer(content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Transform legacy input for brand-focused EditorAgent."""
            return {
                'content_to_edit': content_data.get('content', ''),
                'editing_objectives': [
                    'ensure_brand_consistency',
                    'check_tone_alignment',
                    'verify_messaging',
                    'validate_brand_voice',
                    'assess_brand_compliance'
                ],
                'target_audience': content_data.get('target_audience', 'general'),
                'content_type': content_data.get('content_type', 'article'),
                'brand_voice': content_data.get('tone', 'professional'),
                'style_guide': {
                    'brand_guidelines': content_data.get('brand_guidelines', {}),
                    'tone_requirements': content_data.get('tone', 'professional'),
                    'focus_area': 'brand_consistency'
                }
            }
        
        editor_agent = EditorAgentLangGraph()
        return LangGraphLegacyAdapter(
            langgraph_agent=editor_agent,
            agent_name="BrandReviewAgent",
            agent_description="Brand consistency review using EditorAgent with brand focus",
            input_transformer=brand_input_transformer
        )
    
    @staticmethod
    def create_generic_adapter(
        langgraph_agent: Any,
        agent_name: str,
        agent_description: Optional[str] = None
    ) -> LangGraphLegacyAdapter:
        """
        Create a generic adapter for any LangGraph agent.
        
        Args:
            langgraph_agent: The LangGraph agent to wrap
            agent_name: Name for the adapted agent
            agent_description: Optional description
            
        Returns:
            Configured adapter instance
        """
        return LangGraphLegacyAdapter(
            langgraph_agent=langgraph_agent,
            agent_name=agent_name,
            agent_description=agent_description
        )