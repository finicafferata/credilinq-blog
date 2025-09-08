"""
Content Quality Agent - Specialized agent for comprehensive content quality analysis.

This agent evaluates content across multiple quality dimensions including:
- Readability and clarity
- Structure and organization  
- Grammar and language quality
- Audience alignment
- Engagement potential
- Technical accuracy

🚨 DEPRECATED: This agent is deprecated and will be removed in version 3.0.0.
Use EditorAgentLangGraph via AdapterFactory.create_editor_adapter() instead.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ..core.base_agent import BaseAgent, AgentResult, AgentMetadata, AgentType, AgentExecutionContext
from src.utils.deprecation import deprecated_agent

logger = logging.getLogger(__name__)

@dataclass
class QualityDimension:
    """Individual quality assessment dimension."""
    name: str
    score: float  # 0.0 to 10.0
    description: str
    suggestions: List[str]
    weight: float = 1.0  # Weighting factor for overall score

@deprecated_agent(
    replacement_class="EditorAgentLangGraph", 
    replacement_import="src.agents.adapters.langgraph_legacy_adapter.AdapterFactory",
    migration_guide_url="https://github.com/credilinq/agent-optimization-migration/blob/main/content-quality-migration.md",
    removal_version="3.0.0",
    removal_date="2025-12-01"
)
class ContentQualityAgent(BaseAgent[Dict[str, Any]]):
    """
    Specialized agent for comprehensive content quality analysis.
    
    Evaluates content across multiple quality dimensions and provides
    detailed feedback for improvement.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_AGENT,
                name="ContentQualityAgent",
                description="Comprehensive content quality analysis and scoring",
                capabilities=[
                    "readability_analysis",
                    "structure_evaluation", 
                    "grammar_checking",
                    "audience_alignment",
                    "engagement_scoring",
                    "technical_accuracy"
                ],
                version="1.0.0"
            )
        super().__init__(metadata)
        
        # Quality dimension weights
        self.dimension_weights = {
            "readability": 2.0,
            "structure": 1.8,
            "grammar": 1.5,
            "audience_alignment": 2.2,
            "engagement": 1.7,
            "technical_accuracy": 1.3
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 9.0,
            "good": 7.5,
            "acceptable": 6.0,
            "needs_improvement": 4.0
        }
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute comprehensive content quality analysis.
        
        Args:
            input_data: Content analysis input containing:
                - content: The content text to analyze
                - title: Content title
                - content_type: Type of content (blog_post, article, etc.)
                - target_audience: Target audience description
                - evaluation_criteria: Optional custom criteria
                
        Returns:
            AgentResult with quality analysis results
        """
        try:
            content = input_data.get("content", "")
            title = input_data.get("title", "")
            content_type = input_data.get("content_type", "article")
            target_audience = input_data.get("target_audience", "general")
            
            if not content:
                return AgentResult(
                    success=False,
                    error_message="Content is required for quality analysis",
                    error_code="MISSING_CONTENT"
                )
            
            # Perform quality analysis across multiple dimensions
            quality_dimensions = self._analyze_quality_dimensions(
                content, title, content_type, target_audience
            )
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(quality_dimensions)
            
            # Generate quality assessment
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(quality_dimensions)
            
            # Create detailed quality analysis
            quality_analysis = {
                "overall_score": overall_score,
                "quality_level": quality_level,
                "dimension_scores": {dim.name: dim.score for dim in quality_dimensions},
                "dimension_details": [
                    {
                        "name": dim.name,
                        "score": dim.score,
                        "description": dim.description,
                        "suggestions": dim.suggestions,
                        "weight": dim.weight
                    }
                    for dim in quality_dimensions
                ],
                "content_statistics": self._calculate_content_statistics(content),
                "readability_metrics": self._calculate_readability_metrics(content)
            }
            
            result = AgentResult(
                success=True,
                data={
                    "quality_score": overall_score,
                    "quality_level": quality_level,
                    "suggestions": improvement_suggestions,
                    "quality_analysis": quality_analysis,
                    "content_type": content_type,
                    "analyzed_at": datetime.utcnow().isoformat()
                }
            )
            
            # Add detailed reasoning for quality assessment
            self.add_decision_reasoning(
                result,
                decision_point="Content Quality Assessment",
                reasoning=f"Comprehensive quality analysis across {len(quality_dimensions)} dimensions resulted in overall score of {overall_score:.2f}/10.0",
                importance_explanation=f"Quality assessment is critical for content effectiveness and audience engagement. A {quality_level} rating indicates {'strong' if overall_score >= 7.5 else 'moderate' if overall_score >= 6.0 else 'significant improvement needed'} content quality.",
                confidence_score=0.85,
                alternatives_considered=[
                    "Simple readability check only",
                    "Basic grammar and spell check",
                    "Automated content scoring tools"
                ],
                business_impact=f"High-quality content (score {overall_score:.1f}) directly impacts user engagement, brand credibility, and content performance metrics.",
                risk_assessment="Low quality content risks poor user experience, reduced engagement, and negative brand perception" if overall_score < 6.0 else "Quality content supports positive user experience and brand goals",
                success_indicators=[
                    "Overall quality score above 7.5",
                    "All dimension scores above 6.0", 
                    "Clear improvement suggestions provided",
                    "Actionable feedback for content optimization"
                ],
                implementation_priority="high" if overall_score < 6.0 else "medium" if overall_score < 8.0 else "low"
            )
            
            # Set quality assessment metadata
            self.set_quality_assessment(
                result,
                overall_score=overall_score,
                dimension_scores={dim.name: dim.score for dim in quality_dimensions},
                improvement_areas=[dim.name for dim in quality_dimensions if dim.score < 6.0],
                strengths=[dim.name for dim in quality_dimensions if dim.score >= 8.0],
                quality_notes=f"Content analyzed across {len(quality_dimensions)} quality dimensions with {quality_level} overall rating"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content quality analysis failed: {e}")
            return AgentResult(
                success=False,
                error_message=f"Quality analysis failed: {str(e)}",
                error_code="QUALITY_ANALYSIS_FAILED"
            )
    
    def _analyze_quality_dimensions(
        self, 
        content: str, 
        title: str, 
        content_type: str, 
        target_audience: str
    ) -> List[QualityDimension]:
        """Analyze content across multiple quality dimensions."""
        dimensions = []
        
        # 1. Readability Analysis
        readability_score, readability_suggestions = self._analyze_readability(content)
        dimensions.append(QualityDimension(
            name="readability",
            score=readability_score,
            description="How easy the content is to read and understand",
            suggestions=readability_suggestions,
            weight=self.dimension_weights["readability"]
        ))
        
        # 2. Structure Evaluation
        structure_score, structure_suggestions = self._analyze_structure(content, title)
        dimensions.append(QualityDimension(
            name="structure",
            score=structure_score,
            description="How well-organized and structured the content is",
            suggestions=structure_suggestions,
            weight=self.dimension_weights["structure"]
        ))
        
        # 3. Grammar and Language Quality
        grammar_score, grammar_suggestions = self._analyze_grammar(content)
        dimensions.append(QualityDimension(
            name="grammar",
            score=grammar_score,
            description="Grammar, spelling, and language quality",
            suggestions=grammar_suggestions,
            weight=self.dimension_weights["grammar"]
        ))
        
        # 4. Audience Alignment
        audience_score, audience_suggestions = self._analyze_audience_alignment(content, target_audience)
        dimensions.append(QualityDimension(
            name="audience_alignment",
            score=audience_score,
            description="How well the content matches the target audience",
            suggestions=audience_suggestions,
            weight=self.dimension_weights["audience_alignment"]
        ))
        
        # 5. Engagement Potential
        engagement_score, engagement_suggestions = self._analyze_engagement(content, title)
        dimensions.append(QualityDimension(
            name="engagement",
            score=engagement_score,
            description="Potential for audience engagement and interest",
            suggestions=engagement_suggestions,
            weight=self.dimension_weights["engagement"]
        ))
        
        # 6. Technical Accuracy (basic analysis)
        technical_score, technical_suggestions = self._analyze_technical_accuracy(content, content_type)
        dimensions.append(QualityDimension(
            name="technical_accuracy",
            score=technical_score,
            description="Technical accuracy and factual consistency",
            suggestions=technical_suggestions,
            weight=self.dimension_weights["technical_accuracy"]
        ))
        
        return dimensions
    
    def _analyze_readability(self, content: str) -> tuple[float, List[str]]:
        """Analyze content readability."""
        suggestions = []
        
        # Basic readability metrics
        sentences = len([s for s in re.split(r'[.!?]+', content) if s.strip()])
        words = len(content.split())
        
        if sentences == 0:
            return 2.0, ["Content appears to have no complete sentences"]
        
        avg_sentence_length = words / sentences
        
        # Simple readability scoring based on sentence length
        if avg_sentence_length <= 15:
            readability_score = 9.0
        elif avg_sentence_length <= 20:
            readability_score = 8.0
        elif avg_sentence_length <= 25:
            readability_score = 6.5
        elif avg_sentence_length <= 30:
            readability_score = 5.0
        else:
            readability_score = 3.0
            suggestions.append("Consider breaking up very long sentences for better readability")
        
        # Check for complex words (basic heuristic)
        complex_words = len([word for word in content.split() if len(word) > 10])
        complex_word_ratio = complex_words / words if words > 0 else 0
        
        if complex_word_ratio > 0.15:
            readability_score = max(readability_score - 1.5, 1.0)
            suggestions.append("Consider using simpler words where possible")
        
        # Check paragraph structure
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) == 1 and words > 200:
            suggestions.append("Break content into multiple paragraphs for better readability")
            readability_score = max(readability_score - 1.0, 1.0)
        
        if not suggestions:
            if readability_score >= 8.0:
                suggestions.append("Excellent readability - content is easy to read and understand")
            else:
                suggestions.append("Good readability with room for minor improvements")
        
        return min(readability_score, 10.0), suggestions
    
    def _analyze_structure(self, content: str, title: str) -> tuple[float, List[str]]:
        """Analyze content structure and organization."""
        suggestions = []
        structure_score = 7.0  # Base score
        
        # Check for title
        if not title or len(title.strip()) < 5:
            structure_score -= 1.5
            suggestions.append("Add a descriptive title to improve content structure")
        
        # Check for headings/sections
        heading_patterns = [r'#+ ', r'^[A-Z][^.]+:$', r'^\d+\.', r'^[A-Z][A-Z\s]+$']
        has_headings = any(re.search(pattern, content, re.MULTILINE) for pattern in heading_patterns)
        
        if not has_headings and len(content.split()) > 300:
            structure_score -= 1.0
            suggestions.append("Add headings or sections to break up long content")
        elif has_headings:
            structure_score += 0.5
            suggestions.append("Good use of headings to organize content")
        
        # Check for introduction and conclusion patterns
        has_intro = any(word in content.lower()[:200] for word in ['introduction', 'overview', 'begin', 'start'])
        has_conclusion = any(word in content.lower()[-200:] for word in ['conclusion', 'summary', 'finally', 'end'])
        
        if has_intro:
            structure_score += 0.3
        if has_conclusion:
            structure_score += 0.3
        
        if not has_intro and len(content.split()) > 200:
            suggestions.append("Consider adding a clear introduction")
        if not has_conclusion and len(content.split()) > 200:
            suggestions.append("Consider adding a conclusion or summary")
        
        # Check for lists or bullet points
        has_lists = bool(re.search(r'^\s*[-*•]\s+', content, re.MULTILINE) or 
                        re.search(r'^\s*\d+\.\s+', content, re.MULTILINE))
        if has_lists:
            structure_score += 0.4
            suggestions.append("Good use of lists to organize information")
        
        # Check paragraph distribution
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if avg_paragraph_length > 100:
                suggestions.append("Consider shorter paragraphs for better structure")
                structure_score -= 0.5
        
        if not suggestions:
            suggestions.append("Content has good overall structure and organization")
        
        return min(max(structure_score, 1.0), 10.0), suggestions
    
    def _analyze_grammar(self, content: str) -> tuple[float, List[str]]:
        """Basic grammar and language quality analysis."""
        suggestions = []
        grammar_score = 8.0  # Assume good grammar as baseline
        
        # Check for common grammar issues (basic patterns)
        
        # Double spaces
        if '  ' in content:
            grammar_score -= 0.3
            suggestions.append("Remove double spaces for better formatting")
        
        # Missing spaces after punctuation
        if re.search(r'[.!?][A-Za-z]', content):
            grammar_score -= 0.5
            suggestions.append("Add spaces after punctuation marks")
        
        # Repeated words
        words = content.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 3:
                grammar_score -= 0.2
                suggestions.append("Check for accidentally repeated words")
                break
        
        # Basic capitalization check
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        for sentence in sentences[:5]:  # Check first 5 sentences
            if sentence and not sentence[0].isupper():
                grammar_score -= 0.4
                suggestions.append("Ensure sentences start with capital letters")
                break
        
        # Check for very short sentences that might be fragments
        very_short_sentences = [s for s in sentences if len(s.split()) <= 2]
        if len(very_short_sentences) > len(sentences) * 0.2:
            grammar_score -= 0.5
            suggestions.append("Review very short sentences - some may be fragments")
        
        # Passive voice detection (basic)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(content.lower().count(word) for word in passive_indicators)
        total_words = len(content.split())
        
        if passive_count / total_words > 0.05:
            suggestions.append("Consider using more active voice for stronger writing")
            grammar_score -= 0.3
        
        if not suggestions:
            suggestions.append("Grammar and language quality appear good")
        
        return min(max(grammar_score, 1.0), 10.0), suggestions
    
    def _analyze_audience_alignment(self, content: str, target_audience: str) -> tuple[float, List[str]]:
        """Analyze how well content matches target audience."""
        suggestions = []
        alignment_score = 7.0  # Base score
        
        audience_lower = target_audience.lower()
        content_lower = content.lower()
        
        # Professional/Business audience indicators
        if any(term in audience_lower for term in ['business', 'professional', 'corporate', 'b2b']):
            professional_terms = ['strategy', 'efficiency', 'roi', 'optimization', 'solution', 'implementation']
            professional_count = sum(1 for term in professional_terms if term in content_lower)
            
            if professional_count >= 2:
                alignment_score += 0.8
                suggestions.append("Good use of professional terminology for business audience")
            else:
                suggestions.append("Consider adding more business-focused terminology")
                alignment_score -= 0.5
        
        # Technical audience indicators
        if any(term in audience_lower for term in ['technical', 'developer', 'engineer', 'expert']):
            if len([word for word in content.split() if len(word) > 8]) / len(content.split()) > 0.1:
                alignment_score += 0.5
                suggestions.append("Appropriate technical depth for expert audience")
            else:
                suggestions.append("Consider adding more technical detail for expert audience")
        
        # General audience indicators
        if any(term in audience_lower for term in ['general', 'consumer', 'public', 'beginner']):
            # Check for jargon
            technical_terms = ['api', 'algorithm', 'optimization', 'methodology', 'implementation']
            jargon_count = sum(1 for term in technical_terms if term in content_lower)
            
            if jargon_count > 5:
                alignment_score -= 1.0
                suggestions.append("Reduce technical jargon for general audience")
            else:
                alignment_score += 0.3
                suggestions.append("Good accessibility for general audience")
        
        # Tone analysis
        formal_indicators = ['therefore', 'furthermore', 'however', 'consequently']
        informal_indicators = ["you'll", "we'll", "it's", "don't", "can't"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in content_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in content_lower)
        
        if 'formal' in audience_lower or 'academic' in audience_lower:
            if formal_count > informal_count:
                alignment_score += 0.4
                suggestions.append("Appropriate formal tone for target audience")
            else:
                suggestions.append("Consider a more formal tone for this audience")
        
        if not suggestions:
            suggestions.append("Content appears reasonably aligned with target audience")
        
        return min(max(alignment_score, 1.0), 10.0), suggestions
    
    def _analyze_engagement(self, content: str, title: str) -> tuple[float, List[str]]:
        """Analyze content engagement potential."""
        suggestions = []
        engagement_score = 6.0  # Base score
        
        # Title engagement
        if title:
            title_lower = title.lower()
            engaging_words = ['how', 'why', 'what', 'guide', 'tips', 'secrets', 'ultimate', 'best']
            if any(word in title_lower for word in engaging_words):
                engagement_score += 0.8
                suggestions.append("Title uses engaging language")
            
            # Check for numbers in title
            if re.search(r'\d+', title):
                engagement_score += 0.4
                suggestions.append("Good use of numbers in title for engagement")
        
        # Question usage
        question_count = content.count('?')
        if question_count > 0:
            engagement_score += min(question_count * 0.3, 1.0)
            suggestions.append("Good use of questions to engage readers")
        else:
            suggestions.append("Consider adding rhetorical questions to increase engagement")
        
        # Call-to-action detection
        cta_patterns = [
            r'\b(try|start|begin|learn|discover|find out|check out|read more)\b',
            r'\b(contact|visit|subscribe|download|get)\b'
        ]
        has_cta = any(re.search(pattern, content.lower()) for pattern in cta_patterns)
        
        if has_cta:
            engagement_score += 0.6
            suggestions.append("Good inclusion of call-to-action elements")
        else:
            suggestions.append("Consider adding call-to-action elements")
        
        # Personal pronouns (engagement indicators)
        personal_pronouns = content.lower().count('you') + content.lower().count('your')
        total_words = len(content.split())
        
        if personal_pronouns / total_words > 0.01:
            engagement_score += 0.5
            suggestions.append("Good use of direct address to engage readers")
        
        # Story/example indicators
        story_indicators = ['example', 'story', 'case study', 'imagine', 'picture this']
        if any(indicator in content.lower() for indicator in story_indicators):
            engagement_score += 0.7
            suggestions.append("Excellent use of examples or stories for engagement")
        
        # Emotional words
        emotional_words = ['amazing', 'incredible', 'fantastic', 'powerful', 'revolutionary', 'breakthrough']
        emotional_count = sum(1 for word in emotional_words if word in content.lower())
        
        if emotional_count > 0:
            engagement_score += min(emotional_count * 0.2, 0.8)
        else:
            suggestions.append("Consider adding more emotionally engaging language")
        
        if not suggestions:
            suggestions.append("Content has moderate engagement potential")
        
        return min(max(engagement_score, 1.0), 10.0), suggestions
    
    def _analyze_technical_accuracy(self, content: str, content_type: str) -> tuple[float, List[str]]:
        """Basic technical accuracy analysis."""
        suggestions = []
        technical_score = 7.5  # Assume reasonable accuracy
        
        # Check for obvious inconsistencies
        # This is a basic implementation - real technical accuracy would require domain expertise
        
        # Date consistency check
        dates = re.findall(r'\b\d{4}\b', content)
        current_year = datetime.now().year
        future_dates = [int(date) for date in dates if int(date) > current_year + 1]
        
        if future_dates:
            technical_score -= 0.5
            suggestions.append("Check dates for accuracy - some appear to be in the future")
        
        # Number consistency (basic check)
        percentages = re.findall(r'\b(\d+(?:\.\d+)?)%', content)
        high_percentages = [float(p) for p in percentages if float(p) > 100]
        
        if high_percentages:
            technical_score -= 0.3
            suggestions.append("Review percentages - some exceed 100%")
        
        # URL/link validation (basic format check)
        urls = re.findall(r'https?://[^\s<>"]+', content)
        malformed_urls = [url for url in urls if not re.match(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', url)]
        
        if malformed_urls:
            technical_score -= 0.4
            suggestions.append("Check URL formats for accuracy")
        
        # Check for citation/source indicators
        citation_indicators = ['source:', 'according to', 'study shows', 'research indicates']
        has_citations = any(indicator in content.lower() for indicator in citation_indicators)
        
        if has_citations:
            technical_score += 0.5
            suggestions.append("Good inclusion of sources and citations")
        elif content_type in ['research', 'analysis', 'report']:
            suggestions.append("Consider adding sources and citations for technical content")
        
        if not suggestions:
            suggestions.append("No obvious technical accuracy issues detected")
        
        return min(max(technical_score, 1.0), 10.0), suggestions
    
    def _calculate_overall_score(self, dimensions: List[QualityDimension]) -> float:
        """Calculate weighted overall quality score."""
        total_weighted_score = sum(dim.score * dim.weight for dim in dimensions)
        total_weight = sum(dim.weight for dim in dimensions)
        
        return min(total_weighted_score / total_weight if total_weight > 0 else 0, 10.0)
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds["excellent"]:
            return "excellent"
        elif score >= self.quality_thresholds["good"]:
            return "good"
        elif score >= self.quality_thresholds["acceptable"]:
            return "acceptable"
        elif score >= self.quality_thresholds["needs_improvement"]:
            return "needs_improvement"
        else:
            return "poor"
    
    def _generate_improvement_suggestions(self, dimensions: List[QualityDimension]) -> List[str]:
        """Generate prioritized improvement suggestions."""
        suggestions = []
        
        # Get suggestions from lowest scoring dimensions first
        sorted_dimensions = sorted(dimensions, key=lambda x: x.score)
        
        for dimension in sorted_dimensions[:3]:  # Top 3 areas for improvement
            if dimension.score < 7.0:
                suggestions.extend(dimension.suggestions[:2])  # Top 2 suggestions per dimension
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _calculate_content_statistics(self, content: str) -> Dict[str, Any]:
        """Calculate basic content statistics."""
        words = content.split()
        sentences = [s for s in re.split(r'[.!?]+', content) if s.strip()]
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "character_count": len(content),
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
            "average_paragraph_length": len(words) / len(paragraphs) if paragraphs else 0
        }
    
    def _calculate_readability_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate readability metrics."""
        words = content.split()
        sentences = [s for s in re.split(r'[.!?]+', content) if s.strip()]
        
        # Simple readability indicators
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Syllable estimation (basic)
        vowels = 'aeiouAEIOU'
        syllable_count = sum(
            max(1, sum(1 for char in word if char in vowels))
            for word in words
        )
        
        return {
            "average_word_length": avg_word_length,
            "estimated_syllables": syllable_count,
            "flesch_reading_ease_estimate": max(0, 206.835 - 1.015 * (len(words) / len(sentences) if sentences else 1) - 84.6 * (syllable_count / len(words) if words else 1)),
            "complexity_score": avg_word_length * 2 + (len(words) / len(sentences) if sentences else 20) / 5
        }