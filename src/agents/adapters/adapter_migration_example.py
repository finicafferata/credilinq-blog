"""
Example: Migrating ContentGenerationWorkflow to use LangGraph agents via adapters.

This file demonstrates how to gradually migrate legacy workflows to use modern
LangGraph agents without breaking existing functionality.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.adapters.langgraph_legacy_adapter import (
    LangGraphLegacyAdapter,
    AdapterFactory
)
from src.agents.workflow.review_agent_base import ReviewAgentBase

logger = logging.getLogger(__name__)


class MigratedContentGenerationWorkflow:
    """
    Example of ContentGenerationWorkflow using LangGraph agents through adapters.
    
    This demonstrates a gradual migration approach where legacy agents are
    replaced one by one with their LangGraph equivalents.
    """
    
    def __init__(self, use_langgraph: bool = True):
        """
        Initialize the workflow with option to use LangGraph or legacy agents.
        
        Args:
            use_langgraph: If True, use LangGraph agents via adapters
        """
        self.use_langgraph = use_langgraph
        self.agents = self._initialize_agents()
        
    def _initialize_agents(self) -> Dict[str, ReviewAgentBase]:
        """
        Initialize agents, using adapters for LangGraph agents when enabled.
        """
        agents = {}
        
        if self.use_langgraph:
            logger.info("Initializing workflow with LangGraph agents via adapters")
            
            # Use adapted LangGraph agents
            agents['quality_review'] = self._create_quality_review_adapter()
            agents['editor'] = AdapterFactory.create_editor_adapter()
            agents['seo'] = AdapterFactory.create_seo_adapter()
            agents['writer'] = AdapterFactory.create_writer_adapter()
            agents['brand_review'] = self._create_brand_review_adapter()
            
        else:
            logger.info("Initializing workflow with legacy agents")
            
            # Use legacy agents (fallback)
            from src.agents.specialized.quality_review_agent import QualityReviewAgent
            from src.agents.specialized.brand_review_agent import BrandReviewAgent
            
            agents['quality_review'] = QualityReviewAgent()
            agents['brand_review'] = BrandReviewAgent()
            # Note: Would add other legacy agents here
            
        return agents
    
    def _create_quality_review_adapter(self) -> LangGraphLegacyAdapter:
        """
        Create an adapter for quality review using EditorAgent.
        
        The EditorAgent provides superior quality assessment compared to
        the legacy QualityReviewAgent.
        """
        from src.agents.specialized.editor_agent_langgraph import EditorAgentLangGraph
        
        # Custom input transformer for quality review
        def quality_input_transformer(content_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            return {
                'content_to_edit': content_data.get('content', ''),
                'editing_objectives': [
                    'assess_quality',
                    'check_grammar',
                    'evaluate_structure',
                    'verify_factual_accuracy',
                    'measure_readability'
                ],
                'target_audience': content_data.get('target_audience', 'general'),
                'content_type': content_data.get('content_type', 'article'),
                'style_guide': {
                    'quality_thresholds': {
                        'grammar': 0.8,
                        'readability': 0.7,
                        'structure': 0.8,
                        'factual_accuracy': 0.9
                    }
                }
            }
        
        # Custom output transformer for quality review format
        def quality_output_transformer(langgraph_result: Any, original_input: Dict[str, Any]):
            from src.agents.workflow.review_workflow_models import ReviewAgentResult, ReviewDecision
            
            # Extract quality metrics from EditorAgent result
            if hasattr(langgraph_result, 'data'):
                data = langgraph_result.data
            else:
                data = langgraph_result if isinstance(langgraph_result, dict) else {}
            
            # Calculate overall quality score
            quality_scores = {
                'grammar': data.get('grammar_score', 0.8),
                'readability': data.get('readability_score', 0.75),
                'structure': data.get('clarity_score', 0.8),
                'factual_accuracy': data.get('consistency_score', 0.85)
            }
            
            overall_score = sum(quality_scores.values()) / len(quality_scores)
            
            # Determine issues and suggestions
            issues = []
            suggestions = []
            
            for metric, score in quality_scores.items():
                if score < 0.7:
                    issues.append(f"Low {metric} score: {score:.2f}")
                    suggestions.append(f"Improve {metric} to meet quality standards")
            
            # Add any issues from the agent
            if 'improvement_areas' in data:
                issues.extend(data['improvement_areas'])
            
            if 'editing_feedback' in data:
                suggestions.extend(data['editing_feedback'])
            
            from src.agents.workflow.review_workflow_models import ReviewStage
            
            return ReviewAgentResult(
                stage=ReviewStage.CONTENT_QUALITY,
                content_id=original_input.get('content_id', 'unknown'),
                automated_score=overall_score,
                confidence=min(overall_score + 0.1, 1.0),
                feedback=suggestions or ['Content meets quality standards'],
                issues_found=issues or ['No quality issues found'],
                recommendations=suggestions or ['Content meets quality standards'],
                metrics={
                    'quality_metrics': quality_scores,
                    'agent_type': 'EditorAgentLangGraph_adapted',
                    'decision': (
                        ReviewDecision.APPROVE if overall_score >= 0.85
                        else ReviewDecision.REQUEST_CHANGES if overall_score >= 0.7
                        else ReviewDecision.REJECT
                    ).value
                },
                requires_human_review=overall_score < 0.8,
                auto_approved=overall_score >= 0.85,
                execution_time_ms=100,
                model_used='EditorAgentLangGraph'
            )
        
        editor_agent = EditorAgentLangGraph()
        return LangGraphLegacyAdapter(
            langgraph_agent=editor_agent,
            agent_name="QualityReviewAgent_LangGraph",
            agent_description="Quality assessment using advanced EditorAgent",
            input_transformer=quality_input_transformer,
            output_transformer=quality_output_transformer
        )
    
    def _create_brand_review_adapter(self) -> LangGraphLegacyAdapter:
        """
        Create an adapter for brand review using EditorAgent with brand focus.
        """
        from src.agents.specialized.editor_agent_langgraph import EditorAgentLangGraph
        from src.agents.specialized.seo_agent_langgraph import SEOAgentLangGraph
        
        class BrandReviewComposite:
            """Composite agent that combines Editor and SEO for brand review."""
            
            def __init__(self):
                self.editor = EditorAgentLangGraph()
                self.seo = SEOAgentLangGraph()
            
            async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Run editor with brand focus
                editor_result = await self.editor.execute({
                    **input_data,
                    'editing_objectives': [
                        'ensure_brand_consistency',
                        'check_tone_alignment',
                        'verify_messaging'
                    ]
                })
                
                # Extract brand-relevant scores
                brand_score = 0.0
                if editor_result.success:
                    data = editor_result.data or {}
                    brand_score = data.get('consistency_score', 0.75)
                
                return {
                    'success': True,
                    'data': {
                        'brand_consistency_score': brand_score,
                        'tone_alignment': data.get('tone_score', 0.8),
                        'messaging_clarity': data.get('clarity_score', 0.85),
                        'issues': data.get('improvement_areas', []),
                        'suggestions': data.get('editing_feedback', [])
                    },
                    'metadata': {
                        'agent_type': 'BrandReviewComposite',
                        'editor_result': editor_result.metadata
                    }
                }
        
        def brand_output_transformer(result: Any, original_input: Dict[str, Any]):
            from src.agents.workflow.review_workflow_models import ReviewAgentResult, ReviewDecision
            
            data = result.get('data', {}) if isinstance(result, dict) else {}
            brand_score = data.get('brand_consistency_score', 0.75)
            
            from src.agents.workflow.review_workflow_models import ReviewStage
            
            return ReviewAgentResult(
                stage=ReviewStage.BRAND_CHECK,
                content_id=original_input.get('content_id', 'unknown'),
                automated_score=brand_score,
                confidence=min(brand_score + 0.1, 1.0),
                feedback=data.get('suggestions', []),
                issues_found=data.get('issues', []),
                recommendations=data.get('suggestions', []),
                metrics={
                    **data,
                    'decision': (
                        ReviewDecision.APPROVE if brand_score >= 0.85
                        else ReviewDecision.REQUEST_CHANGES if brand_score >= 0.7
                        else ReviewDecision.REJECT
                    ).value
                },
                requires_human_review=brand_score < 0.75,
                auto_approved=brand_score >= 0.85,
                execution_time_ms=150,
                model_used='BrandReviewComposite'
            )
        
        composite_agent = BrandReviewComposite()
        return LangGraphLegacyAdapter(
            langgraph_agent=composite_agent,
            agent_name="BrandReviewAgent_LangGraph",
            agent_description="Brand consistency review using composite agents",
            output_transformer=brand_output_transformer
        )
    
    async def execute_workflow(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the content generation workflow with adapted agents.
        
        Args:
            task_data: Task information including content, metadata, etc.
            
        Returns:
            Workflow execution results
        """
        results = {
            'task_id': task_data.get('task_id'),
            'stages': {},
            'overall_success': True,
            'final_content': task_data.get('content')
        }
        
        # Execute each stage with the appropriate agent
        for stage_name, agent in self.agents.items():
            try:
                logger.info(f"Executing stage: {stage_name}")
                
                # Prepare stage input
                stage_input = {
                    'content': results['final_content'],
                    'review_stage': stage_name,
                    **task_data
                }
                
                # Execute agent (works for both legacy and adapted agents)
                stage_result = await agent.execute_safe(stage_input)
                
                # Store results
                results['stages'][stage_name] = {
                    'score': stage_result.automated_score,
                    'passed': not stage_result.requires_human_review,
                    'issues': stage_result.issues_found,
                    'suggestions': stage_result.feedback,
                    'decision': stage_result.metrics.get('decision', 'unknown')
                }
                
                # Update content if agent provided edited version
                if hasattr(stage_result, 'metadata') and 'edited_content' in stage_result.metadata:
                    results['final_content'] = stage_result.metadata['edited_content']
                
                # Check if stage failed
                if stage_result.requires_human_review and stage_result.automated_score < 0.6:
                    results['overall_success'] = False
                    logger.warning(f"Stage {stage_name} failed with score {stage_result.automated_score}")
                    
            except Exception as e:
                logger.error(f"Error in stage {stage_name}: {str(e)}")
                results['stages'][stage_name] = {
                    'error': str(e),
                    'passed': False
                }
                results['overall_success'] = False
        
        # Calculate overall score
        stage_scores = [
            stage['score'] 
            for stage in results['stages'].values() 
            if 'score' in stage and isinstance(stage['score'], (int, float))
        ]
        results['overall_score'] = sum(stage_scores) / len(stage_scores) if stage_scores else 0.0
        
        return results


# Example usage
async def demonstrate_migration():
    """
    Demonstrate the migration from legacy to LangGraph agents.
    """
    import json
    
    # Sample task data
    task_data = {
        'task_id': 'task_001',
        'content': '''
        # Introduction to Cloud Computing
        
        Cloud computing has revolutionized how businesses operate in the digital age.
        It provides scalable, on-demand access to computing resources without the need
        for maintaining physical infrastructure.
        
        ## Key Benefits
        
        1. Cost Efficiency: Pay only for what you use
        2. Scalability: Easily scale up or down based on demand
        3. Flexibility: Access resources from anywhere
        4. Reliability: Built-in redundancy and backup
        
        ## Conclusion
        
        Cloud computing is essential for modern businesses looking to stay competitive
        and agile in today's fast-paced market.
        ''',
        'title': 'Introduction to Cloud Computing',
        'target_audience': 'IT professionals',
        'content_type': 'technical_article',
        'brand_tone': 'professional',
        'brand_guidelines': {
            'company': 'TechCorp',
            'voice': 'authoritative yet approachable'
        },
        'seo_keywords': ['cloud computing', 'scalability', 'infrastructure'],
        'word_count': 150
    }
    
    # Test with LangGraph agents
    logger.info("=" * 50)
    logger.info("Testing with LangGraph agents via adapters")
    logger.info("=" * 50)
    
    workflow_langgraph = MigratedContentGenerationWorkflow(use_langgraph=True)
    results_langgraph = await workflow_langgraph.execute_workflow(task_data)
    
    print("\nLangGraph Workflow Results:")
    print(f"Overall Success: {results_langgraph['overall_success']}")
    print(f"Overall Score: {results_langgraph['overall_score']:.2f}")
    print("\nStage Results:")
    for stage, result in results_langgraph['stages'].items():
        print(f"  {stage}:")
        if 'score' in result:
            print(f"    Score: {result['score']:.2f}")
            print(f"    Passed: {result['passed']}")
            print(f"    Decision: {result['decision']}")
        else:
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    # Compare with legacy agents (if available)
    try:
        logger.info("\n" + "=" * 50)
        logger.info("Testing with legacy agents for comparison")
        logger.info("=" * 50)
        
        workflow_legacy = MigratedContentGenerationWorkflow(use_langgraph=False)
        results_legacy = await workflow_legacy.execute_workflow(task_data)
        
        print("\n" + "=" * 50)
        print("Comparison: LangGraph vs Legacy")
        print("=" * 50)
        print(f"LangGraph Overall Score: {results_langgraph['overall_score']:.2f}")
        print(f"Legacy Overall Score: {results_legacy['overall_score']:.2f}")
        print(f"Score Improvement: {(results_langgraph['overall_score'] - results_legacy['overall_score']):.2f}")
        
    except ImportError:
        print("\nLegacy agents not available for comparison")
    
    return results_langgraph


if __name__ == "__main__":
    import asyncio
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    asyncio.run(demonstrate_migration())