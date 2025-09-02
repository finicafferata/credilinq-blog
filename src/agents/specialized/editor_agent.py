"""
Editor Agent - Reviews and refines content for quality and consistency.
"""

from typing import Dict, Any, Optional, List
from src.core.llm_client import create_llm
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator


class EditorAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for reviewing and refining content quality.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.EDITOR,
                name="EditorAgent",
                description="Reviews and refines content for quality, consistency, and standards",
                capabilities=[
                    "content_review",
                    "quality_assessment",
                    "style_consistency",
                    "grammar_check",
                    "revision_suggestions",
                    "approval_decisions"
                ],
                version="2.0.0"
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
    
    def _initialize(self):
        """Initialize the LLM."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = create_llm(
                model="gpt-3.5-turbo",
                temperature=0.3,  # Lower temperature for more consistent editing
                api_key=settings.OPENAI_API_KEY
            )
            
            self.logger.info("EditorAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EditorAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for editing."""
        super()._validate_input(input_data)
        
        required_fields = ["content", "blog_title", "company_context"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation
        self.security_validator.validate_input(str(input_data["content"]))
        self.security_validator.validate_input(str(input_data["blog_title"]))
        self.security_validator.validate_input(str(input_data["company_context"]))
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Review content and provide approval or revision suggestions.
        
        Args:
            input_data: Dictionary containing:
                - content: Content to review
                - blog_title: Title for context
                - company_context: Company context for alignment
                - content_type: Type of content (optional)
                - quality_standards: Specific quality requirements (optional)
            context: Execution context
            
        Returns:
            AgentResult: Review results with approval or revision notes
        """
        try:
            content = input_data["content"]
            blog_title = input_data["blog_title"]
            company_context = input_data["company_context"]
            content_type = input_data.get("content_type", "blog")
            quality_standards = input_data.get("quality_standards", {})
            
            self.logger.info(f"Reviewing {content_type} content: {blog_title}")
            
            # Perform comprehensive review
            review_result = self._comprehensive_review(
                content, blog_title, company_context, content_type, quality_standards
            )
            
            # Make approval decision
            approval_decision = self._make_approval_decision(review_result)
            
            result_data = {
                "approved": approval_decision["approved"],
                "review_score": review_result["overall_score"],
                "review_details": review_result,
                "revision_notes": approval_decision.get("revision_notes"),
                "final_content": content if approval_decision["approved"] else None,
                "quality_assessment": self._assess_quality_level(review_result["overall_score"])
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "editor",
                    "content_approved": approval_decision["approved"],
                    "review_score": review_result["overall_score"],
                    "content_type": content_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content review failed: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CONTENT_REVIEW_FAILED"
            )
    
    def _comprehensive_review(
        self,
        content: str,
        blog_title: str,
        company_context: str,
        content_type: str,
        quality_standards: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive content review."""
        
        # Automated quality checks
        automated_score = self._automated_quality_check(content, content_type)
        
        # LLM-based review
        llm_review = self._llm_content_review(content, blog_title, company_context, content_type)
        
        # Combine results
        review_result = {
            "overall_score": (automated_score["total_score"] + llm_review.get("score", 70)) / 2,
            "automated_checks": automated_score,
            "editorial_review": llm_review,
            "quality_dimensions": self._analyze_quality_dimensions(content, content_type),
            "improvement_suggestions": self._generate_improvement_suggestions(automated_score, llm_review)
        }
        
        return review_result
    
    def _automated_quality_check(self, content: str, content_type: str) -> Dict[str, Any]:
        """Perform automated quality checks."""
        words = content.split()
        word_count = len(words)
        char_count = len(content)
        
        # Check content length
        length_score = self._check_content_length(word_count, content_type)
        
        # Check formatting
        format_score = self._check_formatting(content)
        
        # Check readability
        readability_score = self._check_readability(content)
        
        # Check structure
        structure_score = self._check_content_structure(content, content_type)
        
        # Check consistency
        consistency_score = self._check_consistency(content)
        
        total_score = (
            length_score * 0.2 +
            format_score * 0.25 +
            readability_score * 0.25 +
            structure_score * 0.2 +
            consistency_score * 0.1
        )
        
        return {
            "total_score": total_score,
            "word_count": word_count,
            "char_count": char_count,
            "length_score": length_score,
            "format_score": format_score,
            "readability_score": readability_score,
            "structure_score": structure_score,
            "consistency_score": consistency_score,
            "detailed_checks": {
                "has_headers": "#" in content,
                "has_lists": any(marker in content for marker in ["-", "•", "1.", "2."]),
                "has_emphasis": any(marker in content for marker in ["**", "*", "_"]),
                "paragraph_breaks": content.count("\n\n"),
                "average_sentence_length": self._calculate_avg_sentence_length(content)
            }
        }
    
    def _llm_content_review(
        self,
        content: str,
        blog_title: str,
        company_context: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Use LLM for content review."""
        
        prompt = f"""You are a senior editor reviewing content for publication. Evaluate this {content_type} comprehensively.

COMPANY CONTEXT:
{company_context}

CONTENT TITLE: {blog_title}

CONTENT TO REVIEW:
---
{content}
---

EVALUATION CRITERIA:
- Content quality and accuracy
- Alignment with company voice and context
- Proper structure and flow
- Engagement and readability
- Actionable value for readers
- Professional presentation
- Grammar and style
- Completeness and depth

Provide your review in this JSON format:
{{
  "score": <number 0-100>,
  "strengths": ["strength1", "strength2", ...],
  "weaknesses": ["weakness1", "weakness2", ...],
  "specific_issues": ["issue1", "issue2", ...],
  "recommendations": ["rec1", "rec2", ...],
  "approval_recommendation": "approve" or "revise",
  "revision_priority": "high", "medium", or "low"
}}

Focus on constructive feedback and specific actionable recommendations."""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            
            # Try to parse JSON response
            import json
            try:
                review_data = json.loads(response.content.strip())
                return review_data
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_review_fallback(response.content.strip())
                
        except Exception as e:
            self.logger.warning(f"LLM review failed: {str(e)}")
            return {
                "score": 70,
                "strengths": ["Content structure"],
                "weaknesses": ["Needs manual review"],
                "specific_issues": ["LLM review unavailable"],
                "recommendations": ["Manual editorial review recommended"],
                "approval_recommendation": "revise",
                "revision_priority": "medium"
            }
    
    def _parse_review_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON LLM responses."""
        # Simple text parsing fallback
        score = 70  # Default score
        
        # Look for score indicators
        if "excellent" in response_text.lower() or "outstanding" in response_text.lower():
            score = 90
        elif "good" in response_text.lower() or "solid" in response_text.lower():
            score = 80
        elif "needs improvement" in response_text.lower() or "revisions" in response_text.lower():
            score = 60
        
        approval = "approve" if "approved" in response_text.lower() else "revise"
        
        return {
            "score": score,
            "strengths": ["Content structure"],
            "weaknesses": ["Needs detailed review"],
            "specific_issues": [],
            "recommendations": ["Manual review recommended"],
            "approval_recommendation": approval,
            "revision_priority": "medium"
        }
    
    def _check_content_length(self, word_count: int, content_type: str) -> float:
        """Check if content length is appropriate."""
        target_ranges = {
            "blog": (1500, 2500),
            "linkedin": (800, 1200),
            "article": (2000, 3000)
        }
        
        min_words, max_words = target_ranges.get(content_type, (1000, 2000))
        
        if min_words <= word_count <= max_words:
            return 100.0
        elif word_count < min_words:
            ratio = word_count / min_words
            return max(50.0, ratio * 100)
        else:  # word_count > max_words
            excess_ratio = (word_count - max_words) / max_words
            penalty = min(30, excess_ratio * 50)
            return max(70.0, 100 - penalty)
    
    def _check_formatting(self, content: str) -> float:
        """Check formatting quality."""
        score = 0.0
        
        # Headers
        if "#" in content:
            score += 25
        
        # Lists
        if any(marker in content for marker in ["-", "•", "1.", "2.", "3."]):
            score += 20
        
        # Emphasis
        if any(marker in content for marker in ["**", "*", "_"]):
            score += 15
        
        # Paragraph breaks
        paragraph_breaks = content.count("\n\n")
        score += min(20, paragraph_breaks * 2)
        
        # Links or references
        if "[" in content and "]" in content:
            score += 10
        
        # Code blocks or quotes
        if "```" in content or ">" in content:
            score += 10
        
        return min(100.0, score)
    
    def _check_readability(self, content: str) -> float:
        """Check content readability."""
        sentences = content.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 50.0
        
        words = content.split()
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal sentence length is 15-20 words
        if 15 <= avg_sentence_length <= 20:
            sentence_score = 100
        elif 10 <= avg_sentence_length <= 25:
            sentence_score = 80
        else:
            sentence_score = 60
        
        # Check paragraph length
        paragraphs = content.split('\n\n')
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        
        # Optimal paragraph length is 50-100 words
        if 50 <= avg_paragraph_length <= 100:
            paragraph_score = 100
        elif 30 <= avg_paragraph_length <= 120:
            paragraph_score = 80
        else:
            paragraph_score = 60
        
        return (sentence_score + paragraph_score) / 2
    
    def _check_content_structure(self, content: str, content_type: str) -> float:
        """Check content structure."""
        score = 0.0
        
        # Check for introduction
        first_100_words = ' '.join(content.split()[:100]).lower()
        intro_keywords = ['introduction', 'begin', 'start', 'overview', 'today', 'explore']
        if any(keyword in first_100_words for keyword in intro_keywords):
            score += 25
        
        # Check for conclusion
        last_100_words = ' '.join(content.split()[-100:]).lower()
        conclusion_keywords = ['conclusion', 'summary', 'final', 'end', 'takeaway', 'action']
        if any(keyword in last_100_words for keyword in conclusion_keywords):
            score += 25
        
        # Check for headers/sections
        header_count = content.count('#')
        if content_type == "blog" and header_count >= 3:
            score += 30
        elif content_type == "linkedin" and header_count >= 1:
            score += 30
        elif content_type == "article" and header_count >= 4:
            score += 30
        else:
            score += header_count * 5
        
        # Check for logical flow
        if '\n\n' in content:  # Paragraph breaks indicate structure
            score += 20
        
        return min(100.0, score)
    
    def _check_consistency(self, content: str) -> float:
        """Check content consistency."""
        # Basic consistency checks
        score = 100.0
        
        # Check for consistent header formatting
        headers = [line for line in content.split('\n') if line.startswith('#')]
        if headers:
            # Check if all headers follow consistent pattern
            header_levels = [len(h) - len(h.lstrip('#')) for h in headers]
            if len(set(header_levels)) <= 3:  # Maximum 3 header levels is good
                score += 0  # No penalty
            else:
                score -= 10
        
        return max(60.0, score)
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length."""
        sentences = content.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def _analyze_quality_dimensions(self, content: str, content_type: str) -> Dict[str, float]:
        """Analyze different quality dimensions."""
        return {
            "clarity": self._assess_clarity(content),
            "completeness": self._assess_completeness(content, content_type),
            "engagement": self._assess_engagement(content),
            "professionalism": self._assess_professionalism(content),
            "actionability": self._assess_actionability(content)
        }
    
    def _assess_clarity(self, content: str) -> float:
        """Assess content clarity."""
        # Simple heuristics for clarity
        words = content.split()
        
        # Check for clear language (avoid overly complex words)
        complex_words = sum(1 for word in words if len(word) > 12)
        complexity_ratio = complex_words / len(words) if words else 0
        
        clarity_score = max(60.0, 100 - (complexity_ratio * 200))
        return clarity_score
    
    def _assess_completeness(self, content: str, content_type: str) -> float:
        """Assess content completeness."""
        word_count = len(content.split())
        
        target_ranges = {
            "blog": (1500, 2500),
            "linkedin": (800, 1200),
            "article": (2000, 3000)
        }
        
        min_words, max_words = target_ranges.get(content_type, (1000, 2000))
        
        if min_words <= word_count <= max_words:
            return 100.0
        elif word_count < min_words:
            return (word_count / min_words) * 100
        else:
            return max(80.0, 100 - ((word_count - max_words) / max_words * 20))
    
    def _assess_engagement(self, content: str) -> float:
        """Assess content engagement level."""
        engagement_indicators = [
            content.count('?'),  # Questions engage readers
            content.count('!'),  # Exclamations show enthusiasm
            len([line for line in content.split('\n') if line.strip().startswith('-')]),  # Lists
            content.lower().count('you'),  # Direct address
            content.lower().count('example'),  # Examples engage
        ]
        
        engagement_score = min(100.0, sum(engagement_indicators) * 10)
        return max(50.0, engagement_score)
    
    def _assess_professionalism(self, content: str) -> float:
        """Assess professional tone."""
        # Check for professional language patterns
        professional_score = 80.0  # Default professional score
        
        # Penalize informal language
        informal_words = ['gonna', 'wanna', 'kinda', 'sorta', 'yeah', 'nah']
        informal_count = sum(content.lower().count(word) for word in informal_words)
        professional_score -= informal_count * 5
        
        # Bonus for professional structure
        if '#' in content:  # Headers indicate structure
            professional_score += 10
        
        return max(50.0, min(100.0, professional_score))
    
    def _assess_actionability(self, content: str) -> float:
        """Assess how actionable the content is."""
        action_words = ['how to', 'step', 'guide', 'implement', 'apply', 'use', 'start', 'begin']
        action_score = sum(content.lower().count(word) for word in action_words) * 10
        
        return min(100.0, max(40.0, action_score))
    
    def _generate_improvement_suggestions(
        self,
        automated_score: Dict[str, Any],
        llm_review: Dict[str, Any]
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Based on automated checks
        if automated_score["length_score"] < 70:
            if automated_score["word_count"] < 1000:
                suggestions.append("Consider expanding the content for more comprehensive coverage")
            else:
                suggestions.append("Consider condensing the content for better readability")
        
        if automated_score["format_score"] < 70:
            suggestions.append("Improve formatting with headers, lists, and emphasis")
        
        if automated_score["readability_score"] < 70:
            suggestions.append("Improve readability with shorter sentences and paragraphs")
        
        if automated_score["structure_score"] < 70:
            suggestions.append("Enhance content structure with clear introduction and conclusion")
        
        # Based on LLM review
        if llm_review.get("recommendations"):
            suggestions.extend(llm_review["recommendations"])
        
        return suggestions
    
    def _make_approval_decision(self, review_result: Dict[str, Any]) -> Dict[str, Any]:
        """Make final approval decision based on review."""
        overall_score = review_result["overall_score"]
        llm_recommendation = review_result.get("editorial_review", {}).get("approval_recommendation", "revise")
        
        # Decision logic
        if overall_score >= 85 and llm_recommendation == "approve":
            return {
                "approved": True,
                "confidence": "high",
                "reason": "Content meets high quality standards"
            }
        elif overall_score >= 75:
            return {
                "approved": True,
                "confidence": "medium",
                "reason": "Content meets acceptable quality standards"
            }
        else:
            revision_notes = self._compile_revision_notes(review_result)
            return {
                "approved": False,
                "revision_notes": revision_notes,
                "reason": "Content requires improvements before publication",
                "priority": review_result.get("editorial_review", {}).get("revision_priority", "medium")
            }
    
    def _compile_revision_notes(self, review_result: Dict[str, Any]) -> str:
        """Compile detailed revision notes."""
        notes = []
        
        # Add specific issues
        editorial_review = review_result.get("editorial_review", {})
        if editorial_review.get("specific_issues"):
            notes.append("Specific Issues to Address:")
            for issue in editorial_review["specific_issues"]:
                notes.append(f"• {issue}")
            notes.append("")
        
        # Add improvement suggestions
        if review_result.get("improvement_suggestions"):
            notes.append("Improvement Suggestions:")
            for suggestion in review_result["improvement_suggestions"]:
                notes.append(f"• {suggestion}")
            notes.append("")
        
        # Add score-based feedback
        overall_score = review_result["overall_score"]
        notes.append(f"Overall Quality Score: {overall_score:.1f}/100")
        
        if overall_score < 75:
            notes.append("Priority: Address the above issues to improve content quality.")
        
        return "\n".join(notes)
    
    def _assess_quality_level(self, score: float) -> str:
        """Convert numeric score to quality level."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "acceptable"
        elif score >= 60:
            return "needs_improvement"
        else:
            return "poor"