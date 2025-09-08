"""
LangGraph-based Writer Agent with advanced multi-phase writing workflow.

This agent creates high-quality content using sophisticated workflows with
research integration, iterative drafting, and quality optimization.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class WriterState(WorkflowState):
    """State for Writer LangGraph workflow."""
    # Input requirements
    content_brief: Dict[str, Any] = field(default_factory=dict)
    research_data: Dict[str, Any] = field(default_factory=dict)
    target_audience: str = "general"
    content_type: str = "blog_post"
    word_count_target: int = 1000
    tone: str = "professional"
    
    # Writing process
    content_outline: Dict[str, Any] = field(default_factory=dict)
    draft_versions: List[Dict[str, Any]] = field(default_factory=list)
    current_draft: str = ""
    
    # Quality metrics
    readability_score: float = 0.0
    engagement_score: float = 0.0
    seo_score: float = 0.0
    brand_consistency_score: float = 0.0
    
    # Workflow control
    requires_revision: bool = False
    revision_feedback: List[str] = field(default_factory=list)
    writing_iterations: int = 0
    max_iterations: int = 3

class WriterAgentLangGraph(LangGraphWorkflowBase[WriterState]):
    """
    LangGraph-based Writer with sophisticated multi-phase writing workflow.
    """
    
    def __init__(self, workflow_name: str = "Writer_workflow"):
        super().__init__(workflow_name=workflow_name)
        logger.info("WriterAgentLangGraph initialized with advanced writing capabilities")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(WriterState)
        
        # Define workflow nodes
        workflow.add_node("analyze_brief", self._analyze_brief)
        workflow.add_node("create_outline", self._create_outline)
        workflow.add_node("write_first_draft", self._write_first_draft)
        workflow.add_node("assess_quality", self._assess_quality)
        workflow.add_node("optimize_content", self._optimize_content)
        workflow.add_node("revise_content", self._revise_content)
        workflow.add_node("finalize_content", self._finalize_content)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_brief")
        
        workflow.add_edge("analyze_brief", "create_outline")
        workflow.add_edge("create_outline", "write_first_draft")
        workflow.add_edge("write_first_draft", "assess_quality")
        workflow.add_edge("assess_quality", "optimize_content")
        
        # Conditional routing based on quality assessment
        workflow.add_conditional_edges(
            "optimize_content",
            self._should_revise,
            {
                "revise": "revise_content",
                "finalize": "finalize_content"
            }
        )
        
        workflow.add_edge("revise_content", "assess_quality")
        workflow.set_finish_point("finalize_content")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> WriterState:
        """Create initial workflow state from input."""
        return WriterState(
            content_brief=input_data.get("content_brief", {}),
            research_data=input_data.get("research_data", {}),
            target_audience=input_data.get("target_audience", "general"),
            content_type=input_data.get("content_type", "blog_post"),
            word_count_target=input_data.get("word_count_target", 1000),
            tone=input_data.get("tone", "professional"),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="analyze_brief"
        )
    
    def _analyze_brief(self, state: WriterState) -> WriterState:
        """Analyze the content brief and requirements."""
        logger.info("Analyzing content brief and writing requirements")
        
        brief = state.content_brief
        
        # Extract key requirements from brief
        requirements = {
            "title": brief.get("title", "Untitled Content"),
            "key_messages": brief.get("key_messages", []),
            "keywords": brief.get("keywords", []),
            "call_to_action": brief.get("call_to_action", ""),
            "content_goals": brief.get("goals", ["inform", "engage"]),
            "constraints": brief.get("constraints", [])
        }
        
        # Analyze research data if available
        research_insights = []
        if state.research_data:
            research_insights = state.research_data.get("key_insights", [])
        
        # Update brief analysis
        state.content_brief.update({
            "analyzed_requirements": requirements,
            "research_integration": research_insights,
            "writing_strategy": self._determine_writing_strategy(state)
        })
        
        state.current_step = "create_outline"
        
        return state
    
    def _create_outline(self, state: WriterState) -> WriterState:
        """Create a structured content outline."""
        logger.info("Creating content outline")
        
        brief = state.content_brief
        title = brief.get("analyzed_requirements", {}).get("title", "Untitled Content")
        
        # Create outline structure based on content type
        if state.content_type == "blog_post":
            outline = {
                "title": title,
                "introduction": {
                    "hook": "Engaging opening statement",
                    "problem_statement": "Problem this content addresses",
                    "preview": "What readers will learn"
                },
                "main_sections": [
                    {
                        "heading": "Understanding the Fundamentals",
                        "key_points": ["Point 1", "Point 2", "Point 3"],
                        "supporting_data": []
                    },
                    {
                        "heading": "Practical Applications",
                        "key_points": ["Application 1", "Application 2"],
                        "supporting_data": []
                    },
                    {
                        "heading": "Best Practices and Tips",
                        "key_points": ["Best practice 1", "Best practice 2"],
                        "supporting_data": []
                    }
                ],
                "conclusion": {
                    "summary": "Key takeaways",
                    "call_to_action": brief.get("analyzed_requirements", {}).get("call_to_action", "Learn more")
                }
            }
        else:
            # Generic outline for other content types
            outline = {
                "title": title,
                "sections": [
                    {"heading": "Introduction", "content_focus": "Overview and context"},
                    {"heading": "Main Content", "content_focus": "Core information"},
                    {"heading": "Conclusion", "content_focus": "Summary and next steps"}
                ]
            }
        
        # Integrate research insights
        if state.research_data.get("key_insights"):
            for i, insight in enumerate(state.research_data["key_insights"][:3]):
                if i < len(outline.get("main_sections", [])):
                    outline["main_sections"][i]["supporting_data"].append(insight)
        
        state.content_outline = outline
        state.current_step = "write_first_draft"
        
        return state
    
    def _write_first_draft(self, state: WriterState) -> WriterState:
        """Write the first draft based on outline."""
        logger.info("Writing first draft")
        
        outline = state.content_outline
        
        # Generate content sections
        content_sections = []
        
        # Introduction
        if "introduction" in outline:
            intro = outline["introduction"]
            introduction = f"""
{intro.get('hook', 'Welcome to this comprehensive guide.')}

{intro.get('problem_statement', 'Many professionals face challenges in this area.')}

{intro.get('preview', 'In this article, we will explore key concepts and practical solutions.')}
"""
            content_sections.append(f"## Introduction\n{introduction.strip()}")
        
        # Main sections
        for section in outline.get("main_sections", []):
            section_content = f"## {section['heading']}\n\n"
            
            # Add key points
            for point in section.get("key_points", []):
                section_content += f"### {point}\n\n"
                section_content += f"Detailed explanation of {point.lower()} with relevant examples and insights.\n\n"
            
            # Add supporting data
            for data in section.get("supporting_data", []):
                section_content += f"*Research insight: {data}*\n\n"
            
            content_sections.append(section_content.strip())
        
        # Conclusion
        if "conclusion" in outline:
            conclusion = outline["conclusion"]
            conclusion_text = f"""
## Conclusion

{conclusion.get('summary', 'To summarize the key points covered in this article...')}

### Next Steps

{conclusion.get('call_to_action', 'Take action on these insights to achieve your goals.')}
"""
            content_sections.append(conclusion_text.strip())
        
        # Combine all sections
        draft_content = f"# {outline['title']}\n\n" + "\n\n".join(content_sections)
        
        # Create draft record
        draft = {
            "version": 1,
            "content": draft_content,
            "word_count": len(draft_content.split()),
            "created_at": datetime.now().isoformat(),
            "notes": "First draft - based on outline"
        }
        
        state.draft_versions.append(draft)
        state.current_draft = draft_content
        state.current_step = "assess_quality"
        
        return state
    
    def _assess_quality(self, state: WriterState) -> WriterState:
        """Assess the quality of the current draft."""
        logger.info("Assessing content quality")
        
        content = state.current_draft
        word_count = len(content.split())
        
        # Readability assessment (simplified)
        readability_score = self._assess_readability(content)
        
        # Engagement assessment
        engagement_score = self._assess_engagement(content)
        
        # SEO assessment
        seo_score = self._assess_seo_quality(content, state.content_brief)
        
        # Brand consistency assessment
        brand_consistency_score = self._assess_brand_consistency(content, state.tone)
        
        # Update state with scores
        state.readability_score = readability_score
        state.engagement_score = engagement_score
        state.seo_score = seo_score
        state.brand_consistency_score = brand_consistency_score
        
        # Generate improvement feedback
        feedback = []
        if readability_score < 0.7:
            feedback.append("Improve readability - simplify complex sentences")
        if engagement_score < 0.7:
            feedback.append("Enhance engagement - add more compelling examples")
        if seo_score < 0.7:
            feedback.append("Optimize for SEO - improve keyword usage")
        if brand_consistency_score < 0.7:
            feedback.append("Improve brand voice consistency")
        if abs(word_count - state.word_count_target) > state.word_count_target * 0.2:
            feedback.append(f"Adjust length - target {state.word_count_target} words, current {word_count}")
        
        state.revision_feedback = feedback
        state.current_step = "optimize_content"
        
        return state
    
    def _optimize_content(self, state: WriterState) -> WriterState:
        """Optimize content based on quality assessment."""
        logger.info("Optimizing content based on quality metrics")
        
        content = state.current_draft
        
        # Apply basic optimizations
        optimized_content = content
        
        # SEO optimization
        if state.seo_score < 0.8:
            keywords = state.content_brief.get("analyzed_requirements", {}).get("keywords", [])
            if keywords and keywords[0] not in content.lower():
                # Simple keyword integration
                optimized_content = content.replace(
                    "## Introduction",
                    f"## Introduction\n\n*This article focuses on {keywords[0]} best practices.*"
                )
        
        # Engagement optimization
        if state.engagement_score < 0.8:
            # Add engaging elements
            if "### Next Steps" in optimized_content:
                optimized_content = optimized_content.replace(
                    "### Next Steps",
                    "### 🚀 Ready to Take Action?"
                )
        
        # Update current draft if optimized
        if optimized_content != content:
            state.current_draft = optimized_content
            
            # Create new draft version
            draft = {
                "version": len(state.draft_versions) + 1,
                "content": optimized_content,
                "word_count": len(optimized_content.split()),
                "created_at": datetime.now().isoformat(),
                "notes": "Optimized based on quality assessment"
            }
            state.draft_versions.append(draft)
        
        state.current_step = "finalize_content"
        
        return state
    
    def _should_revise(self, state: WriterState) -> str:
        """Determine if content needs revision."""
        avg_quality = (
            state.readability_score + 
            state.engagement_score + 
            state.seo_score + 
            state.brand_consistency_score
        ) / 4
        
        needs_revision = (
            avg_quality < 0.75 or
            len(state.revision_feedback) > 2
        ) and state.writing_iterations < state.max_iterations
        
        if needs_revision:
            state.requires_revision = True
            state.writing_iterations += 1
            return "revise"
        else:
            return "finalize"
    
    def _revise_content(self, state: WriterState) -> WriterState:
        """Revise content based on feedback."""
        logger.info(f"Revising content (iteration {state.writing_iterations})")
        
        content = state.current_draft
        revised_content = content
        
        # Apply revisions based on feedback
        for feedback_item in state.revision_feedback:
            if "readability" in feedback_item.lower():
                # Simplify content (basic simulation)
                revised_content = revised_content.replace(" furthermore, ", " also, ")
                revised_content = revised_content.replace(" however, ", " but ")
            
            elif "engagement" in feedback_item.lower():
                # Add engaging elements
                if "Detailed explanation" in revised_content:
                    revised_content = revised_content.replace(
                        "Detailed explanation",
                        "Here's what you need to know about"
                    )
            
            elif "length" in feedback_item.lower():
                current_words = len(revised_content.split())
                if current_words < state.word_count_target:
                    # Expand content
                    revised_content += "\n\n### Additional Insights\n\nFurther analysis reveals important considerations for implementation."
        
        # Update current draft
        state.current_draft = revised_content
        
        # Create revision record
        revision = {
            "version": len(state.draft_versions) + 1,
            "content": revised_content,
            "word_count": len(revised_content.split()),
            "created_at": datetime.now().isoformat(),
            "notes": f"Revision {state.writing_iterations} - addressed: {'; '.join(state.revision_feedback)}"
        }
        state.draft_versions.append(revision)
        
        state.current_step = "assess_quality"
        
        return state
    
    def _finalize_content(self, state: WriterState) -> WriterState:
        """Finalize the written content."""
        logger.info("Finalizing content")
        
        final_content = state.current_draft
        word_count = len(final_content.split())
        
        # Add final metadata
        state.metadata.update({
            "final_word_count": word_count,
            "target_word_count": state.word_count_target,
            "word_count_variance": abs(word_count - state.word_count_target) / state.word_count_target,
            "draft_versions_created": len(state.draft_versions),
            "writing_iterations": state.writing_iterations,
            "final_readability_score": state.readability_score,
            "final_engagement_score": state.engagement_score,
            "final_seo_score": state.seo_score,
            "final_brand_consistency_score": state.brand_consistency_score,
            "overall_quality_score": (state.readability_score + state.engagement_score + 
                                    state.seo_score + state.brand_consistency_score) / 4,
            "content_complete": True
        })
        
        state.current_step = "completed"
        
        return state
    
    def _determine_writing_strategy(self, state: WriterState) -> Dict[str, Any]:
        """Determine writing strategy based on brief and audience."""
        strategy = {
            "structure_type": "problem-solution" if "problem" in str(state.content_brief).lower() else "informational",
            "tone_adaptation": state.tone,
            "engagement_tactics": self._select_engagement_tactics(state.target_audience),
            "content_depth": "comprehensive" if state.word_count_target > 1500 else "concise"
        }
        return strategy
    
    def _select_engagement_tactics(self, audience: str) -> List[str]:
        """Select engagement tactics based on audience."""
        tactics_by_audience = {
            "general": ["clear examples", "practical tips", "actionable steps"],
            "technical": ["detailed analysis", "code examples", "technical comparisons"],
            "business": ["ROI focus", "case studies", "implementation roadmaps"],
            "executive": ["strategic insights", "high-level summaries", "decision frameworks"]
        }
        return tactics_by_audience.get(audience, tactics_by_audience["general"])
    
    def _assess_readability(self, content: str) -> float:
        """Assess content readability (simplified)."""
        words = content.split()
        sentences = content.split('.')
        
        if not sentences or not words:
            return 0.5
        
        # Simple readability metric
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Ideal range: 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            return 0.9
        elif 10 <= avg_words_per_sentence <= 25:
            return 0.7
        else:
            return 0.5
    
    def _assess_engagement(self, content: str) -> float:
        """Assess content engagement potential."""
        engagement_indicators = [
            "you", "your", "how to", "why", "what", "example", "tip", "step",
            "?", "!", "discover", "learn", "achieve", "improve", "solution"
        ]
        
        content_lower = content.lower()
        found_indicators = sum(1 for indicator in engagement_indicators if indicator in content_lower)
        
        # Score based on engagement indicators
        return min(found_indicators / 10, 1.0)
    
    def _assess_seo_quality(self, content: str, brief: Dict[str, Any]) -> float:
        """Assess SEO quality of content."""
        keywords = brief.get("analyzed_requirements", {}).get("keywords", [])
        if not keywords:
            return 0.8  # Default score if no keywords specified
        
        content_lower = content.lower()
        keyword_usage = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        
        # Score based on keyword usage
        return min(keyword_usage / len(keywords), 1.0) if keywords else 0.8
    
    def _assess_brand_consistency(self, content: str, tone: str) -> float:
        """Assess brand voice consistency."""
        tone_indicators = {
            "professional": ["expertise", "analysis", "recommend", "industry", "best practice"],
            "casual": ["you'll", "let's", "simply", "easy", "fun"],
            "authoritative": ["proven", "essential", "critical", "must", "required"],
            "friendly": ["help", "support", "together", "community", "share"]
        }
        
        indicators = tone_indicators.get(tone, tone_indicators["professional"])
        content_lower = content.lower()
        
        found_indicators = sum(1 for indicator in indicators if indicator in content_lower)
        
        # Score based on tone consistency
        return min(found_indicators / 3, 1.0)  # Expect at least 3 tone indicators