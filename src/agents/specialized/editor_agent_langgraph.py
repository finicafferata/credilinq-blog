"""
LangGraph-based Editor Agent with advanced multi-phase editing workflow.

This agent provides comprehensive content editing using sophisticated workflows with
quality assessment, style optimization, and iterative refinement.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class EditorState(WorkflowState):
    """State for Editor LangGraph workflow."""
    # Input requirements
    content_to_edit: str = ""
    editing_objectives: List[str] = field(default_factory=list)
    target_audience: str = "general"
    content_type: str = "article"
    style_guide: Dict[str, Any] = field(default_factory=dict)
    brand_voice: str = "professional"
    
    # Content analysis
    initial_analysis: Dict[str, Any] = field(default_factory=dict)
    quality_assessment: Dict[str, float] = field(default_factory=dict)
    improvement_areas: List[str] = field(default_factory=list)
    
    # Editing process
    editing_plan: Dict[str, Any] = field(default_factory=dict)
    edit_versions: List[Dict[str, Any]] = field(default_factory=list)
    current_edited_content: str = ""
    
    # Quality metrics
    grammar_score: float = 0.0
    readability_score: float = 0.0
    clarity_score: float = 0.0
    consistency_score: float = 0.0
    engagement_score: float = 0.0
    
    # Workflow control
    requires_additional_editing: bool = False
    editing_feedback: List[str] = field(default_factory=list)
    editing_iterations: int = 0
    max_iterations: int = 3

class EditorAgentLangGraph(LangGraphWorkflowBase[EditorState]):
    """
    LangGraph-based Editor with sophisticated multi-phase editing workflow.
    """
    
    def __init__(self, workflow_name: str = "Editor_workflow"):
        super().__init__(workflow_name=workflow_name)
        logger.info("EditorAgentLangGraph initialized with advanced editing capabilities")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(EditorState)
        
        # Define workflow nodes
        workflow.add_node("analyze_content", self._analyze_content)
        workflow.add_node("assess_quality", self._assess_quality)
        workflow.add_node("create_editing_plan", self._create_editing_plan)
        workflow.add_node("perform_structural_edits", self._perform_structural_edits)
        workflow.add_node("perform_language_edits", self._perform_language_edits)
        workflow.add_node("optimize_readability", self._optimize_readability)
        workflow.add_node("ensure_consistency", self._ensure_consistency)
        workflow.add_node("final_quality_check", self._final_quality_check)
        workflow.add_node("additional_editing", self._additional_editing)
        workflow.add_node("finalize_edits", self._finalize_edits)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_content")
        
        workflow.add_edge("analyze_content", "assess_quality")
        workflow.add_edge("assess_quality", "create_editing_plan")
        workflow.add_edge("create_editing_plan", "perform_structural_edits")
        workflow.add_edge("perform_structural_edits", "perform_language_edits")
        workflow.add_edge("perform_language_edits", "optimize_readability")
        workflow.add_edge("optimize_readability", "ensure_consistency")
        workflow.add_edge("ensure_consistency", "final_quality_check")
        
        # Conditional routing based on quality check
        workflow.add_conditional_edges(
            "final_quality_check",
            self._should_continue_editing,
            {
                "continue": "additional_editing",
                "finalize": "finalize_edits"
            }
        )
        
        workflow.add_edge("additional_editing", "perform_language_edits")
        workflow.set_finish_point("finalize_edits")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> EditorState:
        """Create initial workflow state from input."""
        return EditorState(
            content_to_edit=input_data.get("content_to_edit", input_data.get("content", "")),
            editing_objectives=input_data.get("editing_objectives", ["improve_quality", "enhance_clarity"]),
            target_audience=input_data.get("target_audience", "general"),
            content_type=input_data.get("content_type", "article"),
            style_guide=input_data.get("style_guide", {}),
            brand_voice=input_data.get("brand_voice", "professional"),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="analyze_content"
        )
    
    def _analyze_content(self, state: EditorState) -> EditorState:
        """Analyze the content structure and characteristics."""
        logger.info("Analyzing content for editing requirements")
        
        content = state.content_to_edit
        
        initial_analysis = {
            "content_length": {
                "word_count": len(content.split()),
                "character_count": len(content),
                "paragraph_count": content.count('\n\n') + 1,
                "sentence_count": len([s for s in content.split('.') if s.strip()])
            },
            "content_structure": {
                "has_headings": bool('##' in content or '#' in content),
                "has_lists": bool('-' in content or '*' in content or '1.' in content),
                "has_quotes": bool('"' in content or '"' in content),
                "has_links": bool('http' in content or 'www' in content)
            },
            "linguistic_features": {
                "avg_sentence_length": len(content.split()) / max(1, len([s for s in content.split('.') if s.strip()])),
                "complexity_indicators": self._detect_complexity_indicators(content),
                "tone_indicators": self._detect_tone_indicators(content),
                "technical_terms": self._extract_technical_terms(content)
            },
            "potential_issues": self._identify_potential_issues(content)
        }
        
        state.initial_analysis = initial_analysis
        state.current_step = "assess_quality"
        
        return state
    
    def _assess_quality(self, state: EditorState) -> EditorState:
        """Assess current content quality across multiple dimensions."""
        logger.info("Assessing content quality")
        
        content = state.content_to_edit
        
        # Grammar and mechanics assessment
        grammar_score = self._assess_grammar(content)
        
        # Readability assessment
        readability_score = self._assess_readability(content)
        
        # Clarity assessment
        clarity_score = self._assess_clarity(content)
        
        # Consistency assessment
        consistency_score = self._assess_consistency(content)
        
        # Engagement assessment
        engagement_score = self._assess_engagement(content)
        
        # Update state with quality scores
        state.grammar_score = grammar_score
        state.readability_score = readability_score
        state.clarity_score = clarity_score
        state.consistency_score = consistency_score
        state.engagement_score = engagement_score
        
        quality_assessment = {
            "grammar": grammar_score,
            "readability": readability_score,
            "clarity": clarity_score,
            "consistency": consistency_score,
            "engagement": engagement_score,
            "overall": (grammar_score + readability_score + clarity_score + 
                       consistency_score + engagement_score) / 5
        }
        
        # Identify improvement areas
        improvement_areas = []
        threshold = 0.7
        
        if grammar_score < threshold:
            improvement_areas.append("grammar_and_mechanics")
        if readability_score < threshold:
            improvement_areas.append("readability")
        if clarity_score < threshold:
            improvement_areas.append("clarity_and_flow")
        if consistency_score < threshold:
            improvement_areas.append("consistency")
        if engagement_score < threshold:
            improvement_areas.append("engagement")
        
        state.quality_assessment = quality_assessment
        state.improvement_areas = improvement_areas
        state.current_step = "create_editing_plan"
        
        return state
    
    def _create_editing_plan(self, state: EditorState) -> EditorState:
        """Create a comprehensive editing plan."""
        logger.info("Creating comprehensive editing plan")
        
        editing_plan = {
            "editing_priorities": self._prioritize_improvements(state.improvement_areas, state.quality_assessment),
            "structural_changes": [],
            "language_improvements": [],
            "style_adjustments": [],
            "content_enhancements": []
        }
        
        # Plan structural changes
        if "clarity_and_flow" in state.improvement_areas:
            editing_plan["structural_changes"].extend([
                "Reorganize paragraphs for better logical flow",
                "Add transition sentences between sections",
                "Improve paragraph structure and coherence"
            ])
        
        # Plan language improvements
        if "grammar_and_mechanics" in state.improvement_areas:
            editing_plan["language_improvements"].extend([
                "Fix grammatical errors and typos",
                "Improve sentence structure and syntax",
                "Ensure proper punctuation and capitalization"
            ])
        
        if "readability" in state.improvement_areas:
            editing_plan["language_improvements"].extend([
                "Simplify complex sentences",
                "Replace jargon with clearer terms",
                "Vary sentence length for better rhythm"
            ])
        
        # Plan style adjustments
        if "consistency" in state.improvement_areas:
            editing_plan["style_adjustments"].extend([
                "Ensure consistent tone throughout",
                "Standardize terminology and formatting",
                "Align with brand voice guidelines"
            ])
        
        # Plan content enhancements
        if "engagement" in state.improvement_areas:
            editing_plan["content_enhancements"].extend([
                "Add engaging examples and anecdotes",
                "Strengthen opening and closing statements",
                "Include compelling calls-to-action"
            ])
        
        state.editing_plan = editing_plan
        state.current_step = "perform_structural_edits"
        
        return state
    
    def _perform_structural_edits(self, state: EditorState) -> EditorState:
        """Perform structural editing improvements."""
        logger.info("Performing structural edits")
        
        content = state.content_to_edit
        edited_content = content
        
        # Apply structural improvements
        structural_changes = state.editing_plan.get("structural_changes", [])
        
        if "Reorganize paragraphs for better logical flow" in structural_changes:
            # Simulate paragraph reorganization
            paragraphs = edited_content.split('\n\n')
            if len(paragraphs) > 2:
                # Simple reordering simulation (move conclusion to end if not already there)
                conclusion_indicators = ['in conclusion', 'to summarize', 'finally', 'in summary']
                for i, para in enumerate(paragraphs[:-1]):  # Don't check last paragraph
                    if any(indicator in para.lower() for indicator in conclusion_indicators):
                        # Move conclusion paragraph to end
                        conclusion_para = paragraphs.pop(i)
                        paragraphs.append(conclusion_para)
                        break
                edited_content = '\n\n'.join(paragraphs)
        
        if "Add transition sentences between sections" in structural_changes:
            # Add simple transitions
            paragraphs = edited_content.split('\n\n')
            if len(paragraphs) > 1:
                transitions = ["Furthermore, ", "Additionally, ", "Moreover, ", "However, ", "Nevertheless, "]
                for i in range(1, len(paragraphs)):
                    if not any(trans.lower() in paragraphs[i].lower()[:20] for trans in transitions):
                        # Add transition to paragraph that doesn't have one
                        if i < len(transitions):
                            paragraphs[i] = transitions[i-1] + paragraphs[i].lower()[0] + paragraphs[i][1:]
                        break
                edited_content = '\n\n'.join(paragraphs)
        
        # Create edit version record
        edit_version = {
            "version": len(state.edit_versions) + 1,
            "edit_type": "structural",
            "content": edited_content,
            "changes_made": structural_changes,
            "timestamp": datetime.now().isoformat()
        }
        
        state.edit_versions.append(edit_version)
        state.current_edited_content = edited_content
        state.current_step = "perform_language_edits"
        
        return state
    
    def _perform_language_edits(self, state: EditorState) -> EditorState:
        """Perform language-level editing improvements."""
        logger.info("Performing language edits")
        
        content = state.current_edited_content
        edited_content = content
        
        language_improvements = state.editing_plan.get("language_improvements", [])
        
        # Fix common grammatical issues
        if "Fix grammatical errors and typos" in language_improvements:
            # Basic grammar improvements
            grammar_fixes = [
                ("it's", "its"),  # Common mistake
                ("  ", " "),      # Double spaces
                ("i ", "I "),     # Capitalize I
                (" ,", ","),      # Space before comma
                (" .", "."),      # Space before period
            ]
            
            for old, new in grammar_fixes:
                if old in edited_content and old != new:
                    edited_content = edited_content.replace(old, new)
                    break  # Apply one fix per iteration
        
        # Improve readability
        if "Simplify complex sentences" in language_improvements:
            # Replace complex conjunctions with simpler ones
            readability_improvements = [
                ("nevertheless", "however"),
                ("furthermore", "also"),
                ("consequently", "so"),
                ("therefore", "so"),
                ("in addition to", "besides")
            ]
            
            for complex_word, simple_word in readability_improvements:
                if complex_word in edited_content.lower():
                    # Case-sensitive replacement
                    words = edited_content.split()
                    for i, word in enumerate(words):
                        if word.lower() == complex_word:
                            words[i] = simple_word
                            break
                    edited_content = ' '.join(words)
                    break
        
        # Vary sentence length
        if "Vary sentence length for better rhythm" in language_improvements:
            sentences = edited_content.split('. ')
            if len(sentences) > 2:
                # Find very long sentences and break them
                for i, sentence in enumerate(sentences):
                    if len(sentence.split()) > 25:  # Very long sentence
                        # Simple sentence breaking (add period in middle)
                        words = sentence.split()
                        mid_point = len(words) // 2
                        # Find a good break point near the middle
                        for j in range(mid_point - 2, mid_point + 3):
                            if j < len(words) and words[j] in ['and', 'but', 'or', 'so']:
                                first_part = ' '.join(words[:j])
                                second_part = ' '.join(words[j+1:])
                                sentences[i] = first_part + '. ' + second_part.capitalize()
                                break
                        break
                edited_content = '. '.join(sentences)
        
        # Create edit version record
        edit_version = {
            "version": len(state.edit_versions) + 1,
            "edit_type": "language",
            "content": edited_content,
            "changes_made": language_improvements,
            "timestamp": datetime.now().isoformat()
        }
        
        state.edit_versions.append(edit_version)
        state.current_edited_content = edited_content
        state.current_step = "optimize_readability"
        
        return state
    
    def _optimize_readability(self, state: EditorState) -> EditorState:
        """Optimize content for readability."""
        logger.info("Optimizing readability")
        
        content = state.current_edited_content
        edited_content = content
        
        # Readability optimizations
        readability_changes = []
        
        # Add subheadings for long content
        if len(content.split()) > 500 and not ('##' in content):
            paragraphs = edited_content.split('\n\n')
            if len(paragraphs) >= 4:
                # Add a subheading in the middle
                mid_point = len(paragraphs) // 2
                paragraphs.insert(mid_point, "## Key Points")
                edited_content = '\n\n'.join(paragraphs)
                readability_changes.append("Added subheading for better structure")
        
        # Break up long paragraphs
        paragraphs = edited_content.split('\n\n')
        modified = False
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) > 100:  # Very long paragraph
                sentences = paragraph.split('. ')
                if len(sentences) > 4:
                    # Break into two paragraphs
                    mid_point = len(sentences) // 2
                    first_part = '. '.join(sentences[:mid_point]) + '.'
                    second_part = '. '.join(sentences[mid_point:])
                    paragraphs[i] = first_part + '\n\n' + second_part
                    modified = True
                    readability_changes.append("Split long paragraph for better readability")
                    break
        
        if modified:
            edited_content = '\n\n'.join(paragraphs)
        
        # Create edit version record
        edit_version = {
            "version": len(state.edit_versions) + 1,
            "edit_type": "readability",
            "content": edited_content,
            "changes_made": readability_changes,
            "timestamp": datetime.now().isoformat()
        }
        
        state.edit_versions.append(edit_version)
        state.current_edited_content = edited_content
        state.current_step = "ensure_consistency"
        
        return state
    
    def _ensure_consistency(self, state: EditorState) -> EditorState:
        """Ensure style and voice consistency."""
        logger.info("Ensuring style and voice consistency")
        
        content = state.current_edited_content
        edited_content = content
        
        consistency_changes = []
        
        # Brand voice alignment
        brand_voice = state.brand_voice.lower()
        
        if brand_voice == "professional":
            # Replace casual language with professional alternatives
            professional_replacements = [
                ("really good", "excellent"),
                ("pretty much", "essentially"),
                ("a lot of", "many"),
                ("stuff", "content"),
                ("things", "elements")
            ]
            
            for casual, professional in professional_replacements:
                if casual in edited_content.lower():
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(casual), re.IGNORECASE)
                    edited_content = pattern.sub(professional, edited_content, count=1)
                    consistency_changes.append(f"Replaced '{casual}' with '{professional}' for professional tone")
                    break
        
        elif brand_voice == "friendly":
            # Add friendly elements if missing
            if not any(friendly in edited_content.lower() for friendly in ["you", "your", "we", "us"]):
                # Make it more personal
                edited_content = edited_content.replace("This article discusses", "We'll explore")
                consistency_changes.append("Added personal pronouns for friendly tone")
        
        # Consistency in formatting
        if consistency_changes or True:  # Always check formatting
            # Ensure consistent list formatting
            if '- ' in edited_content and '* ' in edited_content:
                edited_content = edited_content.replace('* ', '- ')
                consistency_changes.append("Standardized bullet point formatting")
        
        # Create edit version record
        edit_version = {
            "version": len(state.edit_versions) + 1,
            "edit_type": "consistency",
            "content": edited_content,
            "changes_made": consistency_changes,
            "timestamp": datetime.now().isoformat()
        }
        
        state.edit_versions.append(edit_version)
        state.current_edited_content = edited_content
        state.current_step = "final_quality_check"
        
        return state
    
    def _final_quality_check(self, state: EditorState) -> EditorState:
        """Perform final quality assessment."""
        logger.info("Performing final quality check")
        
        # Re-assess quality after editing
        final_grammar_score = self._assess_grammar(state.current_edited_content)
        final_readability_score = self._assess_readability(state.current_edited_content)
        final_clarity_score = self._assess_clarity(state.current_edited_content)
        final_consistency_score = self._assess_consistency(state.current_edited_content)
        final_engagement_score = self._assess_engagement(state.current_edited_content)
        
        # Update final scores
        state.grammar_score = final_grammar_score
        state.readability_score = final_readability_score
        state.clarity_score = final_clarity_score
        state.consistency_score = final_consistency_score
        state.engagement_score = final_engagement_score
        
        # Generate editing feedback
        feedback = []
        
        # Compare improvements
        initial_overall = state.quality_assessment.get("overall", 0.5)
        final_overall = (final_grammar_score + final_readability_score + final_clarity_score + 
                        final_consistency_score + final_engagement_score) / 5
        
        improvement = final_overall - initial_overall
        
        if improvement > 0.1:
            feedback.append(f"Significant quality improvement achieved (+{improvement:.1%})")
        elif improvement > 0.05:
            feedback.append(f"Moderate quality improvement achieved (+{improvement:.1%})")
        else:
            feedback.append("Minor improvements made")
        
        # Identify remaining issues
        if final_grammar_score < 0.8:
            feedback.append("Grammar still needs attention")
        if final_readability_score < 0.8:
            feedback.append("Readability could be further improved")
        if final_clarity_score < 0.8:
            feedback.append("Clarity enhancements recommended")
        
        state.editing_feedback = feedback
        state.current_step = "finalize_edits"
        
        return state
    
    def _should_continue_editing(self, state: EditorState) -> str:
        """Determine if additional editing is needed."""
        overall_quality = (state.grammar_score + state.readability_score + 
                          state.clarity_score + state.consistency_score + 
                          state.engagement_score) / 5
        
        needs_more_editing = (
            overall_quality < 0.8 or
            len([f for f in state.editing_feedback if "needs attention" in f or "could be improved" in f]) > 1
        ) and state.editing_iterations < state.max_iterations
        
        if needs_more_editing:
            state.requires_additional_editing = True
            state.editing_iterations += 1
            return "continue"
        else:
            return "finalize"
    
    def _additional_editing(self, state: EditorState) -> EditorState:
        """Perform additional editing based on feedback."""
        logger.info(f"Performing additional editing (iteration {state.editing_iterations})")
        
        # Focus on the most critical issues identified in feedback
        for feedback_item in state.editing_feedback:
            if "grammar" in feedback_item.lower():
                # Additional grammar pass
                content = state.current_edited_content
                # Apply more aggressive grammar fixes
                grammar_improvements = [
                    ("who's", "whose"),
                    ("there", "their"),  # Context-dependent, simplified
                    ("its", "it's")      # Reverse previous fix if context needs it
                ]
                
                for old, new in grammar_improvements[:1]:  # Apply one fix
                    if old in content:
                        state.current_edited_content = content.replace(old, new, 1)
                        break
                break
        
        state.current_step = "final_quality_check"
        return state
    
    def _finalize_edits(self, state: EditorState) -> EditorState:
        """Finalize the editing process."""
        logger.info("Finalizing editing process")
        
        # Calculate final metrics
        original_word_count = len(state.content_to_edit.split())
        final_word_count = len(state.current_edited_content.split())
        
        final_overall_quality = (state.grammar_score + state.readability_score + 
                                state.clarity_score + state.consistency_score + 
                                state.engagement_score) / 5
        
        # Add final metadata
        state.metadata.update({
            "original_word_count": original_word_count,
            "final_word_count": final_word_count,
            "word_count_change": final_word_count - original_word_count,
            "edit_versions_created": len(state.edit_versions),
            "editing_iterations": state.editing_iterations,
            "improvement_areas_addressed": len(state.improvement_areas),
            "final_quality_scores": {
                "grammar": state.grammar_score,
                "readability": state.readability_score,
                "clarity": state.clarity_score,
                "consistency": state.consistency_score,
                "engagement": state.engagement_score,
                "overall": final_overall_quality
            },
            "quality_improvement": final_overall_quality - state.quality_assessment.get("overall", 0.5),
            "editing_complete": True
        })
        
        state.current_step = "completed"
        return state
    
    # Helper assessment methods
    def _assess_grammar(self, content: str) -> float:
        """Assess grammar quality (simplified)."""
        # Simple grammar indicators
        grammar_issues = [
            "it's" if "its" not in content else "",  # Possessive vs contraction
            "  ",      # Double spaces
            " ,",      # Space before comma
            " .",      # Space before period
            "i ",      # Uncapitalized I
        ]
        
        issue_count = sum(1 for issue in grammar_issues if issue and issue in content)
        max_issues = len([i for i in grammar_issues if i])
        
        return max(0.5, 1.0 - (issue_count / max(max_issues, 1)))
    
    def _assess_readability(self, content: str) -> float:
        """Assess readability (simplified Flesch-like metric)."""
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        
        if not sentences or not words:
            return 0.5
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Optimal range: 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            return 0.9
        elif 10 <= avg_words_per_sentence <= 25:
            return 0.7
        else:
            return 0.5
    
    def _assess_clarity(self, content: str) -> float:
        """Assess content clarity."""
        clarity_indicators = [
            "however", "therefore", "furthermore", "consequently", "moreover",
            "for example", "such as", "in other words", "specifically"
        ]
        
        content_lower = content.lower()
        found_indicators = sum(1 for indicator in clarity_indicators if indicator in content_lower)
        
        # Score based on presence of clarity indicators
        return min(0.3 + (found_indicators / 10), 1.0)
    
    def _assess_consistency(self, content: str) -> float:
        """Assess style consistency."""
        # Check for mixed list formatting
        has_dashes = '- ' in content
        has_asterisks = '* ' in content
        
        consistency_score = 1.0
        
        if has_dashes and has_asterisks:
            consistency_score -= 0.2  # Mixed list formats
        
        # Check for mixed quotation marks
        has_straight_quotes = '"' in content
        has_curly_quotes = '"' in content or '"' in content
        
        if has_straight_quotes and has_curly_quotes:
            consistency_score -= 0.1
        
        return max(0.5, consistency_score)
    
    def _assess_engagement(self, content: str) -> float:
        """Assess content engagement potential."""
        engagement_indicators = [
            "you", "your", "we", "our", "?", "!", "discover", "learn", 
            "imagine", "consider", "think about", "what if", "why"
        ]
        
        content_lower = content.lower()
        found_indicators = sum(1 for indicator in engagement_indicators if indicator in content_lower)
        
        return min(found_indicators / 8, 1.0)
    
    def _detect_complexity_indicators(self, content: str) -> List[str]:
        """Detect linguistic complexity indicators."""
        complex_indicators = [
            "nevertheless", "furthermore", "consequently", "notwithstanding",
            "inasmuch", "whereas", "heretofore", "aforementioned"
        ]
        
        content_lower = content.lower()
        found = [indicator for indicator in complex_indicators if indicator in content_lower]
        return found
    
    def _detect_tone_indicators(self, content: str) -> List[str]:
        """Detect tone indicators in content."""
        tone_indicators = {
            "formal": ["therefore", "furthermore", "consequently", "indeed"],
            "casual": ["pretty much", "really", "stuff", "things", "a lot of"],
            "friendly": ["you", "your", "we", "our", "let's"],
            "authoritative": ["must", "should", "essential", "critical", "required"]
        }
        
        content_lower = content.lower()
        detected_tones = []
        
        for tone, indicators in tone_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                detected_tones.append(tone)
        
        return detected_tones
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content."""
        # Common technical terms across domains
        technical_terms = [
            "algorithm", "framework", "implementation", "optimization", "methodology",
            "infrastructure", "architecture", "protocol", "interface", "integration",
            "analytics", "metrics", "dashboard", "workflow", "automation"
        ]
        
        content_lower = content.lower()
        found_terms = [term for term in technical_terms if term in content_lower]
        return found_terms
    
    def _identify_potential_issues(self, content: str) -> List[str]:
        """Identify potential content issues."""
        issues = []
        
        # Length issues
        word_count = len(content.split())
        if word_count < 100:
            issues.append("Content may be too short")
        elif word_count > 3000:
            issues.append("Content may be too long for target audience")
        
        # Structure issues
        if '\n\n' not in content:
            issues.append("No paragraph breaks detected")
        
        if not ('.' in content[-50:]):  # No period in last 50 characters
            issues.append("Missing conclusion or proper ending")
        
        # Readability issues
        sentences = [s for s in content.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = len(content.split()) / len(sentences)
            if avg_sentence_length > 30:
                issues.append("Average sentence length may be too long")
        
        return issues
    
    def _prioritize_improvements(self, improvement_areas: List[str], quality_scores: Dict[str, float]) -> List[str]:
        """Prioritize improvement areas based on scores and impact."""
        priorities = []
        
        # Critical issues (score < 0.6)
        critical_areas = [area for area in improvement_areas 
                         if quality_scores.get(area.replace("_and_", "_").replace("_", ""), 0.7) < 0.6]
        priorities.extend([(area, "critical") for area in critical_areas])
        
        # Important issues (score < 0.7)
        important_areas = [area for area in improvement_areas 
                          if area not in critical_areas and 
                          quality_scores.get(area.replace("_and_", "_").replace("_", ""), 0.7) < 0.7]
        priorities.extend([(area, "important") for area in important_areas])
        
        # Minor issues
        minor_areas = [area for area in improvement_areas 
                      if area not in critical_areas and area not in important_areas]
        priorities.extend([(area, "minor") for area in minor_areas])
        
        return [area for area, priority in priorities]