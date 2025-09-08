"""
Quality Review Agent
Performs comprehensive content quality assessment including grammar, readability, structure, and factual accuracy.

🚨 DEPRECATED: This agent is deprecated and will be removed in version 3.0.0.
Use EditorAgentLangGraph via AdapterFactory.create_editor_adapter() instead.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.agents.workflow.review_workflow_models import ReviewStage, ReviewAgentResult
from src.agents.workflow.review_agent_base import ReviewAgentBase
from src.core.llm_client import LLMClient
from src.utils.deprecation import deprecated_agent


@deprecated_agent(
    replacement_class="EditorAgentLangGraph",
    replacement_import="src.agents.adapters.langgraph_legacy_adapter.AdapterFactory",
    migration_guide_url="https://github.com/credilinq/agent-optimization-migration/blob/main/quality-review-migration.md",
    removal_version="3.0.0",
    removal_date="2025-12-01"
)
class QualityReviewAgent(ReviewAgentBase):
    """
    Agent responsible for content quality assessment in the review workflow.
    Analyzes grammar, readability, structure, factual accuracy, and overall quality.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__(
            name="QualityReviewAgent",
            description="Comprehensive content quality assessment agent",
            version="1.0.0"
        )
        self.model = model
        try:
            self.llm_client = LLMClient()
        except ValueError:
            # LLM client not available (missing API key)
            self.llm_client = None
            self.logger.warning("LLM client not available - some features will be limited")
        
        # Quality scoring thresholds
        self.auto_approve_threshold = 0.85
        self.quality_thresholds = {
            "grammar": 0.8,
            "readability": 0.7,
            "structure": 0.8,
            "factual_accuracy": 0.9,
            "overall_quality": 0.8
        }
    
    async def execute_safe(self, content_data: Dict[str, Any], **kwargs) -> ReviewAgentResult:
        """
        Execute quality review analysis on content.
        
        Args:
            content_data: Dictionary containing content to review (title, body, etc.)
            **kwargs: Additional parameters
            
        Returns:
            ReviewAgentResult with quality assessment
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract content for analysis
            content_text = self._extract_content_text(content_data)
            content_id = content_data.get("id", "unknown")
            
            # Perform comprehensive quality analysis
            quality_metrics = await self._analyze_quality(content_text, content_data)
            
            # Calculate overall score and confidence
            overall_score = self._calculate_overall_score(quality_metrics)
            confidence = self._calculate_confidence(quality_metrics)
            
            # Generate feedback and recommendations
            feedback = self._generate_feedback(quality_metrics)
            recommendations = self._generate_recommendations(quality_metrics)
            issues_found = self._identify_issues(quality_metrics)
            
            # Determine if human review is required
            requires_human_review = overall_score < self.auto_approve_threshold
            auto_approved = overall_score >= self.auto_approve_threshold
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ReviewAgentResult(
                stage=ReviewStage.QUALITY_CHECK,
                content_id=content_id,
                automated_score=overall_score,
                confidence=confidence,
                feedback=feedback,
                recommendations=recommendations,
                issues_found=issues_found,
                metrics=quality_metrics,
                requires_human_review=requires_human_review,
                auto_approved=auto_approved,
                execution_time_ms=execution_time,
                model_used=self.model,
                tokens_used=0,  # Will be updated after LLM call
                cost=0.0       # Will be updated after LLM call
            )
            
        except Exception as e:
            self.logger.error(f"Quality review failed: {str(e)}")
            raise
    
    async def _analyze_quality(self, content_text: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality analysis using AI and rule-based checks."""
        
        # Parallel execution of different quality checks
        tasks = [
            self._analyze_grammar_and_language(content_text),
            self._analyze_readability(content_text),
            self._analyze_structure(content_data),
            self._analyze_factual_accuracy(content_text),
            self._analyze_consistency(content_text, content_data)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        quality_metrics = {
            "grammar": results[0] if not isinstance(results[0], Exception) else {"score": 0.5, "issues": ["Grammar analysis failed"]},
            "readability": results[1] if not isinstance(results[1], Exception) else {"score": 0.5, "issues": ["Readability analysis failed"]},
            "structure": results[2] if not isinstance(results[2], Exception) else {"score": 0.5, "issues": ["Structure analysis failed"]},
            "factual_accuracy": results[3] if not isinstance(results[3], Exception) else {"score": 0.5, "issues": ["Factual analysis failed"]},
            "consistency": results[4] if not isinstance(results[4], Exception) else {"score": 0.5, "issues": ["Consistency analysis failed"]},
            "word_count": len(content_text.split()),
            "character_count": len(content_text),
            "sentence_count": len(re.split(r'[.!?]+', content_text))
        }
        
        return quality_metrics
    
    async def _analyze_grammar_and_language(self, content_text: str) -> Dict[str, Any]:
        """Analyze grammar, spelling, and language quality using AI."""
        
        prompt = f"""
        Analyze the following content for grammar, spelling, and language quality.
        Provide a detailed assessment with a score from 0.0 to 1.0.

        Content:
        {content_text[:2000]}...

        Provide your analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "grammar_issues": ["list of specific grammar issues"],
            "spelling_issues": ["list of spelling errors"],
            "language_quality_issues": ["list of language quality concerns"],
            "positive_aspects": ["list of strong language elements"],
            "improvement_suggestions": ["specific suggestions for improvement"]
        }}
        """
        
        try:
            response = await self.llm_client.generate_async(
                prompt=prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            analysis = json.loads(response.strip())
            
            # Add rule-based checks
            analysis["issues"] = (
                analysis.get("grammar_issues", []) +
                analysis.get("spelling_issues", []) +
                analysis.get("language_quality_issues", [])
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Grammar analysis failed: {str(e)}")
            return {
                "score": 0.5,
                "issues": [f"Grammar analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual grammar review recommended"]
            }
    
    async def _analyze_readability(self, content_text: str) -> Dict[str, Any]:
        """Analyze content readability and accessibility."""
        
        # Calculate basic readability metrics
        words = content_text.split()
        sentences = re.split(r'[.!?]+', content_text)
        syllables = self._count_syllables(content_text)
        
        # Flesch Reading Ease Score
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables_per_word = syllables / max(len(words), 1)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))
        
        # Convert to 0-1 scale (60+ is considered good readability)
        readability_score = min(1.0, flesch_score / 60.0)
        
        # Additional readability checks
        issues = []
        suggestions = []
        
        if avg_sentence_length > 20:
            issues.append("Sentences are too long (average > 20 words)")
            suggestions.append("Break down long sentences for better readability")
        
        if avg_syllables_per_word > 1.7:
            issues.append("Complex vocabulary may reduce readability")
            suggestions.append("Consider using simpler words where appropriate")
        
        # Check for passive voice (simple heuristic)
        passive_indicators = len(re.findall(r'\b(?:was|were|is|are|been)\s+\w+ed\b', content_text, re.IGNORECASE))
        if passive_indicators > len(sentences) * 0.3:
            issues.append("High use of passive voice detected")
            suggestions.append("Consider using more active voice constructions")
        
        return {
            "score": readability_score,
            "flesch_reading_ease": flesch_score,
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables_per_word,
            "issues": issues,
            "improvement_suggestions": suggestions
        }
    
    async def _analyze_structure(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content structure and organization."""
        
        content_text = self._extract_content_text(content_data)
        title = content_data.get("title", "")
        
        # Structure analysis
        structure_score = 1.0
        issues = []
        suggestions = []
        
        # Check for title
        if not title or len(title.strip()) < 10:
            structure_score -= 0.2
            issues.append("Title is missing or too short")
            suggestions.append("Add a compelling title (10+ characters)")
        
        # Check for headings/sections
        heading_pattern = r'^#{1,6}\s+.+$'
        headings = re.findall(heading_pattern, content_text, re.MULTILINE)
        
        if len(headings) == 0 and len(content_text.split()) > 300:
            structure_score -= 0.2
            issues.append("Long content lacks section headings")
            suggestions.append("Add section headings to improve structure")
        
        # Check for paragraphs
        paragraphs = [p.strip() for p in content_text.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        
        if avg_paragraph_length > 100:
            structure_score -= 0.1
            issues.append("Paragraphs are too long")
            suggestions.append("Break down long paragraphs (aim for 50-80 words)")
        
        # Check for introduction and conclusion
        if len(paragraphs) > 3:
            first_para_words = len(paragraphs[0].split())
            last_para_words = len(paragraphs[-1].split())
            
            if first_para_words < 20:
                structure_score -= 0.1
                issues.append("Introduction paragraph is too brief")
                suggestions.append("Expand the introduction to better engage readers")
            
            if last_para_words < 15:
                structure_score -= 0.1
                issues.append("Conclusion is too brief")
                suggestions.append("Add a stronger conclusion to summarize key points")
        
        structure_score = max(0.0, structure_score)
        
        return {
            "score": structure_score,
            "heading_count": len(headings),
            "paragraph_count": len(paragraphs),
            "avg_paragraph_length": avg_paragraph_length,
            "issues": issues,
            "improvement_suggestions": suggestions
        }
    
    async def _analyze_factual_accuracy(self, content_text: str) -> Dict[str, Any]:
        """Analyze content for potential factual accuracy issues."""
        
        prompt = f"""
        Review the following content for potential factual accuracy issues, outdated information, 
        and unsupported claims. Focus on financial services and B2B content context.

        Content:
        {content_text[:1500]}...

        Provide your analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "potential_issues": ["list of potential factual concerns"],
            "unsupported_claims": ["claims that need sources or verification"],
            "outdated_information": ["potentially outdated facts or figures"],
            "verification_needed": ["items that require fact-checking"],
            "confidence_level": 0.0-1.0
        }}
        """
        
        try:
            response = await self.llm_client.generate_async(
                prompt=prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            analysis = json.loads(response.strip())
            
            # Combine all issues
            all_issues = (
                analysis.get("potential_issues", []) +
                analysis.get("unsupported_claims", []) +
                analysis.get("outdated_information", [])
            )
            
            analysis["issues"] = all_issues
            return analysis
            
        except Exception as e:
            self.logger.error(f"Factual accuracy analysis failed: {str(e)}")
            return {
                "score": 0.7,  # Conservative score when analysis fails
                "issues": [f"Factual accuracy analysis error: {str(e)}"],
                "verification_needed": ["Manual fact-checking recommended"],
                "confidence_level": 0.3
            }
    
    async def _analyze_consistency(self, content_text: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content for internal consistency and coherence."""
        
        # Check title-content alignment
        title = content_data.get("title", "")
        consistency_score = 1.0
        issues = []
        suggestions = []
        
        # Simple keyword overlap between title and content
        if title:
            title_words = set(title.lower().split())
            content_words = set(content_text.lower().split())
            
            # Remove common stop words
            stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an"}
            title_keywords = title_words - stop_words
            
            if title_keywords:
                keyword_overlap = len(title_keywords.intersection(content_words)) / len(title_keywords)
                if keyword_overlap < 0.5:
                    consistency_score -= 0.2
                    issues.append("Title and content keywords don't align well")
                    suggestions.append("Ensure content addresses topics mentioned in title")
        
        # Check for topic drift (simplified)
        paragraphs = [p.strip() for p in content_text.split('\n\n') if p.strip()]
        if len(paragraphs) > 3:
            # This is a simplified check - in practice, you'd use more sophisticated NLP
            first_para_words = set(paragraphs[0].lower().split())
            last_para_words = set(paragraphs[-1].lower().split())
            
            word_overlap = len(first_para_words.intersection(last_para_words)) / max(len(first_para_words), 1)
            if word_overlap < 0.1:
                consistency_score -= 0.1
                issues.append("Content may drift from initial topic")
                suggestions.append("Ensure content maintains focus throughout")
        
        return {
            "score": max(0.0, consistency_score),
            "issues": issues,
            "improvement_suggestions": suggestions
        }
    
    def _extract_content_text(self, content_data: Dict[str, Any]) -> str:
        """Extract readable text from content data."""
        # Handle different content formats
        if "body" in content_data:
            return str(content_data["body"])
        elif "content" in content_data:
            return str(content_data["content"])
        elif "text" in content_data:
            return str(content_data["text"])
        else:
            # Try to concatenate all text fields
            text_parts = []
            for key, value in content_data.items():
                if isinstance(value, str) and len(value) > 20:
                    text_parts.append(value)
            return " ".join(text_parts)
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified algorithm)."""
        # This is a simplified syllable counter
        words = re.findall(r'\b\w+\b', text.lower())
        total_syllables = 0
        
        for word in words:
            syllable_count = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e'):
                syllable_count -= 1
            if syllable_count == 0:
                syllable_count = 1
            total_syllables += syllable_count
        
        return total_syllables
    
    def _calculate_overall_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "grammar": 0.25,
            "readability": 0.20,
            "structure": 0.20,
            "factual_accuracy": 0.25,
            "consistency": 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics and "score" in quality_metrics[metric]:
                weighted_score += quality_metrics[metric]["score"] * weight
                total_weight += weight
        
        return weighted_score / max(total_weight, 1.0)
    
    def _calculate_confidence(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the quality assessment."""
        confidence_factors = []
        
        # Higher confidence when we have fewer analysis errors
        for metric_name, metric_data in quality_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                # Higher confidence for extreme scores (very good or very bad)
                score = metric_data["score"]
                confidence_factors.append(min(score, 1.0 - score) * 2)
        
        return sum(confidence_factors) / max(len(confidence_factors), 1) if confidence_factors else 0.5
    
    def _generate_feedback(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate human-readable feedback."""
        feedback = []
        
        for metric_name, metric_data in quality_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                score = metric_data["score"]
                
                if score >= 0.9:
                    feedback.append(f"Excellent {metric_name.replace('_', ' ')} quality")
                elif score >= 0.8:
                    feedback.append(f"Good {metric_name.replace('_', ' ')} quality")
                elif score >= 0.7:
                    feedback.append(f"Acceptable {metric_name.replace('_', ' ')} quality")
                else:
                    feedback.append(f"Poor {metric_name.replace('_', ' ')} quality - requires improvement")
        
        return feedback
    
    def _generate_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for metric_name, metric_data in quality_metrics.items():
            if isinstance(metric_data, dict):
                suggestions = metric_data.get("improvement_suggestions", [])
                recommendations.extend(suggestions)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_issues(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Identify specific issues found during analysis."""
        issues = []
        
        for metric_name, metric_data in quality_metrics.items():
            if isinstance(metric_data, dict):
                metric_issues = metric_data.get("issues", [])
                issues.extend(metric_issues)
        
        return issues