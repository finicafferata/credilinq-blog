"""
Brand Review Agent
Performs brand consistency and alignment assessment for content.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.agents.workflow.review_workflow_models import ReviewStage, ReviewAgentResult
from src.agents.workflow.review_agent_base import ReviewAgentBase
from src.core.llm_client import LLMClient


class BrandReviewAgent(ReviewAgentBase):
    """
    Agent responsible for brand consistency and alignment assessment in the review workflow.
    Analyzes tone, messaging, brand voice, compliance, and visual consistency.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__(
            name="BrandReviewAgent",
            description="Brand consistency and alignment assessment agent",
            version="1.0.0"
        )
        self.model = model
        try:
            self.llm_client = LLMClient()
        except ValueError:
            # LLM client not available (missing API key)
            self.llm_client = None
            self.logger.warning("LLM client not available - some features will be limited")
        
        # Brand scoring thresholds
        self.auto_approve_threshold = 0.85
        
        # Brand guidelines for financial services B2B content
        self.brand_guidelines = {
            "tone": {
                "preferred": ["professional", "authoritative", "trustworthy", "knowledgeable", "helpful"],
                "avoid": ["casual", "overly promotional", "aggressive", "uncertain", "unprofessional"]
            },
            "messaging": {
                "focus": ["value proposition", "expertise", "trust", "solutions", "results"],
                "avoid": ["feature-heavy", "technical jargon", "generic claims", "weak guarantees"]
            },
            "voice": {
                "characteristics": ["confident", "clear", "direct", "empathetic", "solution-focused"],
                "avoid": ["passive", "vague", "overly complex", "impersonal", "sales-heavy"]
            },
            "compliance": {
                "requirements": ["accurate disclaimers", "regulatory compliance", "fact-based claims"],
                "avoid": ["unrealistic promises", "misleading statements", "incomplete disclosures"]
            }
        }
    
    async def execute_safe(self, content_data: Dict[str, Any], **kwargs) -> ReviewAgentResult:
        """
        Execute brand review analysis on content.
        
        Args:
            content_data: Dictionary containing content to review (title, body, etc.)
            **kwargs: Additional parameters including campaign_context
            
        Returns:
            ReviewAgentResult with brand assessment
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract content and context
            content_text = self._extract_content_text(content_data)
            content_id = content_data.get("id", "unknown")
            campaign_context = kwargs.get("campaign_context", {})
            
            # Perform comprehensive brand analysis
            brand_metrics = await self._analyze_brand_alignment(content_text, content_data, campaign_context)
            
            # Calculate overall score and confidence
            overall_score = self._calculate_overall_score(brand_metrics)
            confidence = self._calculate_confidence(brand_metrics)
            
            # Generate feedback and recommendations
            feedback = self._generate_feedback(brand_metrics)
            recommendations = self._generate_recommendations(brand_metrics)
            issues_found = self._identify_issues(brand_metrics)
            
            # Determine if human review is required
            requires_human_review = overall_score < self.auto_approve_threshold
            auto_approved = overall_score >= self.auto_approve_threshold
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ReviewAgentResult(
                stage=ReviewStage.BRAND_CHECK,
                content_id=content_id,
                automated_score=overall_score,
                confidence=confidence,
                feedback=feedback,
                recommendations=recommendations,
                issues_found=issues_found,
                metrics=brand_metrics,
                requires_human_review=requires_human_review,
                auto_approved=auto_approved,
                execution_time_ms=execution_time,
                model_used=self.model,
                tokens_used=0,  # Will be updated after LLM call
                cost=0.0       # Will be updated after LLM call
            )
            
        except Exception as e:
            self.logger.error(f"Brand review failed: {str(e)}")
            raise
    
    async def _analyze_brand_alignment(self, content_text: str, content_data: Dict[str, Any], campaign_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive brand alignment analysis."""
        
        # Parallel execution of different brand checks
        tasks = [
            self._analyze_tone_and_voice(content_text),
            self._analyze_messaging_alignment(content_text, campaign_context),
            self._analyze_brand_compliance(content_text, content_data),
            self._analyze_visual_consistency(content_data),
            self._analyze_competitive_differentiation(content_text)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        brand_metrics = {
            "tone_voice": results[0] if not isinstance(results[0], Exception) else {"score": 0.5, "issues": ["Tone analysis failed"]},
            "messaging": results[1] if not isinstance(results[1], Exception) else {"score": 0.5, "issues": ["Messaging analysis failed"]},
            "compliance": results[2] if not isinstance(results[2], Exception) else {"score": 0.5, "issues": ["Compliance analysis failed"]},
            "visual_consistency": results[3] if not isinstance(results[3], Exception) else {"score": 0.5, "issues": ["Visual analysis failed"]},
            "differentiation": results[4] if not isinstance(results[4], Exception) else {"score": 0.5, "issues": ["Differentiation analysis failed"]},
            "content_type": content_data.get("type", "unknown"),
            "target_audience": campaign_context.get("target_audience", "unknown")
        }
        
        return brand_metrics
    
    async def _analyze_tone_and_voice(self, content_text: str) -> Dict[str, Any]:
        """Analyze content tone and brand voice alignment."""
        
        # Get brand guidelines for tone analysis
        preferred_tone = ", ".join(self.brand_guidelines["tone"]["preferred"])
        avoid_tone = ", ".join(self.brand_guidelines["tone"]["avoid"])
        preferred_voice = ", ".join(self.brand_guidelines["voice"]["characteristics"])
        
        prompt = f"""
        Analyze the tone and voice of the following financial services B2B content.
        
        Brand Guidelines:
        - Preferred tone: {preferred_tone}
        - Avoid tone: {avoid_tone}
        - Preferred voice characteristics: {preferred_voice}
        
        Content:
        {content_text[:1500]}...
        
        Provide your analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "tone_assessment": {{
                "dominant_tone": "identified tone",
                "tone_consistency": 0.0-1.0,
                "alignment_with_brand": 0.0-1.0
            }},
            "voice_assessment": {{
                "voice_characteristics": ["list of identified characteristics"],
                "brand_alignment": 0.0-1.0,
                "consistency_throughout": 0.0-1.0
            }},
            "tone_issues": ["specific tone problems"],
            "voice_issues": ["specific voice problems"],
            "strengths": ["positive tone/voice elements"],
            "improvement_suggestions": ["specific suggestions"]
        }}
        """
        
        try:
            response = await self.llm_client.generate_async(
                prompt=prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            
            import json
            analysis = json.loads(response.strip())
            
            # Add rule-based tone checks
            tone_indicators = self._analyze_tone_indicators(content_text)
            analysis.update(tone_indicators)
            
            # Combine all issues
            all_issues = analysis.get("tone_issues", []) + analysis.get("voice_issues", [])
            analysis["issues"] = all_issues
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Tone/voice analysis failed: {str(e)}")
            return {
                "score": 0.5,
                "issues": [f"Tone analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual tone review recommended"]
            }
    
    async def _analyze_messaging_alignment(self, content_text: str, campaign_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze messaging alignment with brand and campaign objectives."""
        
        campaign_goals = campaign_context.get("goals", "Not specified")
        target_audience = campaign_context.get("target_audience", "Not specified")
        
        preferred_messaging = ", ".join(self.brand_guidelines["messaging"]["focus"])
        avoid_messaging = ", ".join(self.brand_guidelines["messaging"]["avoid"])
        
        prompt = f"""
        Analyze the messaging alignment of the following content for a financial services B2B campaign.
        
        Campaign Context:
        - Goals: {campaign_goals}
        - Target Audience: {target_audience}
        
        Brand Messaging Guidelines:
        - Focus on: {preferred_messaging}
        - Avoid: {avoid_messaging}
        
        Content:
        {content_text[:1500]}...
        
        Provide your analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "messaging_assessment": {{
                "key_messages": ["identified key messages"],
                "value_proposition_clarity": 0.0-1.0,
                "audience_alignment": 0.0-1.0,
                "campaign_goal_alignment": 0.0-1.0
            }},
            "brand_consistency": {{
                "messaging_focus_alignment": 0.0-1.0,
                "avoided_messaging_present": ["any problematic messaging found"],
                "brand_voice_consistency": 0.0-1.0
            }},
            "messaging_issues": ["specific messaging problems"],
            "strengths": ["strong messaging elements"],
            "improvement_suggestions": ["actionable messaging improvements"]
        }}
        """
        
        try:
            response = await self.llm_client.generate_async(
                prompt=prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            
            import json
            analysis = json.loads(response.strip())
            
            # Add rule-based messaging checks
            messaging_indicators = self._analyze_messaging_indicators(content_text)
            analysis.update(messaging_indicators)
            
            analysis["issues"] = analysis.get("messaging_issues", [])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Messaging analysis failed: {str(e)}")
            return {
                "score": 0.5,
                "issues": [f"Messaging analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual messaging review recommended"]
            }
    
    async def _analyze_brand_compliance(self, content_text: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compliance with brand guidelines and regulatory requirements."""
        
        compliance_requirements = ", ".join(self.brand_guidelines["compliance"]["requirements"])
        avoid_compliance = ", ".join(self.brand_guidelines["compliance"]["avoid"])
        
        prompt = f"""
        Analyze the following financial services content for brand compliance and regulatory considerations.
        
        Compliance Requirements:
        - Must include: {compliance_requirements}
        - Must avoid: {avoid_compliance}
        
        Content:
        {content_text[:1500]}...
        
        Provide your analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "compliance_assessment": {{
                "regulatory_compliance": 0.0-1.0,
                "disclaimer_presence": true/false,
                "fact_based_claims": 0.0-1.0,
                "risk_disclosure": true/false
            }},
            "compliance_issues": ["specific compliance problems"],
            "missing_elements": ["required elements not found"],
            "problematic_claims": ["potentially problematic statements"],
            "compliance_strengths": ["strong compliance elements"],
            "improvement_suggestions": ["compliance improvements needed"]
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
            
            # Add rule-based compliance checks
            compliance_indicators = self._analyze_compliance_indicators(content_text)
            analysis.update(compliance_indicators)
            
            analysis["issues"] = analysis.get("compliance_issues", [])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Compliance analysis failed: {str(e)}")
            return {
                "score": 0.7,  # Conservative score for compliance when analysis fails
                "issues": [f"Compliance analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual compliance review required"]
            }
    
    async def _analyze_visual_consistency(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual elements for brand consistency."""
        
        visual_score = 1.0
        issues = []
        suggestions = []
        strengths = []
        
        # Check for image data
        images = content_data.get("images", [])
        featured_image = content_data.get("featured_image")
        
        if not images and not featured_image:
            visual_score -= 0.3
            issues.append("No visual elements found in content")
            suggestions.append("Consider adding branded visuals to enhance engagement")
        else:
            strengths.append("Visual elements present in content")
        
        # Check for alt text (accessibility and brand consistency)
        if images:
            missing_alt_text = [img for img in images if not img.get("alt_text")]
            if missing_alt_text:
                visual_score -= 0.2
                issues.append(f"{len(missing_alt_text)} images missing alt text")
                suggestions.append("Add descriptive alt text to all images")
        
        # Check for brand colors/themes in metadata (if available)
        visual_metadata = content_data.get("visual_metadata", {})
        if visual_metadata:
            brand_colors_present = visual_metadata.get("brand_colors_detected", False)
            if brand_colors_present:
                strengths.append("Brand colors detected in visual elements")
            else:
                visual_score -= 0.1
                suggestions.append("Ensure visual elements align with brand color palette")
        
        return {
            "score": max(0.0, visual_score),
            "visual_elements_count": len(images) + (1 if featured_image else 0),
            "accessibility_score": 1.0 if not issues else 0.7,
            "issues": issues,
            "strengths": strengths,
            "improvement_suggestions": suggestions
        }
    
    async def _analyze_competitive_differentiation(self, content_text: str) -> Dict[str, Any]:
        """Analyze competitive differentiation and unique value positioning."""
        
        prompt = f"""
        Analyze the following financial services B2B content for competitive differentiation and unique value positioning.
        
        Content:
        {content_text[:1500]}...
        
        Provide your analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "differentiation_assessment": {{
                "unique_value_propositions": ["identified unique selling points"],
                "competitive_advantages": ["mentioned competitive advantages"],
                "differentiation_clarity": 0.0-1.0,
                "generic_messaging_present": true/false
            }},
            "positioning_strength": {{
                "market_position_clarity": 0.0-1.0,
                "expertise_demonstration": 0.0-1.0,
                "credibility_indicators": ["credibility elements found"]
            }},
            "differentiation_issues": ["differentiation problems"],
            "strengths": ["strong differentiation elements"],
            "improvement_suggestions": ["differentiation improvements"]
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
            
            # Add rule-based differentiation checks
            differentiation_indicators = self._analyze_differentiation_indicators(content_text)
            analysis.update(differentiation_indicators)
            
            analysis["issues"] = analysis.get("differentiation_issues", [])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Differentiation analysis failed: {str(e)}")
            return {
                "score": 0.6,
                "issues": [f"Differentiation analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual differentiation review needed"]
            }
    
    def _analyze_tone_indicators(self, content_text: str) -> Dict[str, Any]:
        """Rule-based tone indicator analysis."""
        indicators = {}
        
        # Check for professional language indicators
        professional_indicators = len(re.findall(r'\b(expertise|experience|professional|established|proven|certified)\b', content_text, re.IGNORECASE))
        indicators["professional_language_score"] = min(1.0, professional_indicators / 5)
        
        # Check for overly promotional language
        promotional_indicators = len(re.findall(r'\b(best|amazing|incredible|revolutionary|groundbreaking|guaranteed)\b', content_text, re.IGNORECASE))
        if promotional_indicators > 3:
            indicators["overly_promotional"] = True
        else:
            indicators["overly_promotional"] = False
        
        return indicators
    
    def _analyze_messaging_indicators(self, content_text: str) -> Dict[str, Any]:
        """Rule-based messaging analysis."""
        indicators = {}
        
        # Check for value-focused messaging
        value_keywords = len(re.findall(r'\b(value|benefit|solution|result|outcome|improvement|efficiency|savings)\b', content_text, re.IGNORECASE))
        indicators["value_focus_score"] = min(1.0, value_keywords / 8)
        
        # Check for feature-heavy messaging
        feature_keywords = len(re.findall(r'\b(feature|functionality|capability|specification|technical|system)\b', content_text, re.IGNORECASE))
        if feature_keywords > value_keywords:
            indicators["feature_heavy"] = True
        else:
            indicators["feature_heavy"] = False
        
        return indicators
    
    def _analyze_compliance_indicators(self, content_text: str) -> Dict[str, Any]:
        """Rule-based compliance analysis."""
        indicators = {}
        
        # Check for disclaimer indicators
        disclaimer_indicators = len(re.findall(r'\b(disclaimer|terms|conditions|subject to|may vary|not guaranteed)\b', content_text, re.IGNORECASE))
        indicators["disclaimer_indicators"] = disclaimer_indicators
        
        # Check for absolute claims that might need qualification
        absolute_claims = len(re.findall(r'\b(always|never|guaranteed|100%|completely|totally|absolutely)\b', content_text, re.IGNORECASE))
        indicators["absolute_claims"] = absolute_claims
        
        if absolute_claims > 2:
            indicators["high_risk_claims"] = True
        else:
            indicators["high_risk_claims"] = False
        
        return indicators
    
    def _analyze_differentiation_indicators(self, content_text: str) -> Dict[str, Any]:
        """Rule-based differentiation analysis."""
        indicators = {}
        
        # Check for unique positioning words
        unique_indicators = len(re.findall(r'\b(unique|exclusive|only|first|leading|innovative|proprietary)\b', content_text, re.IGNORECASE))
        indicators["uniqueness_score"] = min(1.0, unique_indicators / 3)
        
        # Check for generic business language
        generic_phrases = len(re.findall(r'\b(synergistic|leverage|optimize|streamline|best-in-class|world-class)\b', content_text, re.IGNORECASE))
        if generic_phrases > 3:
            indicators["generic_language_high"] = True
        else:
            indicators["generic_language_high"] = False
        
        return indicators
    
    def _extract_content_text(self, content_data: Dict[str, Any]) -> str:
        """Extract readable text from content data."""
        if "body" in content_data:
            return str(content_data["body"])
        elif "content" in content_data:
            return str(content_data["content"])
        elif "text" in content_data:
            return str(content_data["text"])
        else:
            text_parts = []
            for key, value in content_data.items():
                if isinstance(value, str) and len(value) > 20:
                    text_parts.append(value)
            return " ".join(text_parts)
    
    def _calculate_overall_score(self, brand_metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall brand score."""
        weights = {
            "tone_voice": 0.25,
            "messaging": 0.30,
            "compliance": 0.25,
            "visual_consistency": 0.10,
            "differentiation": 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in brand_metrics and "score" in brand_metrics[metric]:
                weighted_score += brand_metrics[metric]["score"] * weight
                total_weight += weight
        
        return weighted_score / max(total_weight, 1.0)
    
    def _calculate_confidence(self, brand_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the brand assessment."""
        confidence_factors = []
        
        for metric_name, metric_data in brand_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                score = metric_data["score"]
                # Higher confidence for more definitive scores
                confidence_factors.append(abs(score - 0.5) * 2)
        
        return sum(confidence_factors) / max(len(confidence_factors), 1) if confidence_factors else 0.5
    
    def _generate_feedback(self, brand_metrics: Dict[str, Any]) -> List[str]:
        """Generate human-readable feedback."""
        feedback = []
        
        for metric_name, metric_data in brand_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                score = metric_data["score"]
                readable_name = metric_name.replace('_', ' ').title()
                
                if score >= 0.9:
                    feedback.append(f"Excellent {readable_name} - strongly aligned with brand")
                elif score >= 0.8:
                    feedback.append(f"Good {readable_name} - well aligned with brand guidelines")
                elif score >= 0.7:
                    feedback.append(f"Acceptable {readable_name} - minor brand alignment improvements needed")
                else:
                    feedback.append(f"Poor {readable_name} - significant brand alignment issues")
        
        return feedback
    
    def _generate_recommendations(self, brand_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for metric_name, metric_data in brand_metrics.items():
            if isinstance(metric_data, dict):
                suggestions = metric_data.get("improvement_suggestions", [])
                recommendations.extend(suggestions)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_issues(self, brand_metrics: Dict[str, Any]) -> List[str]:
        """Identify specific issues found during analysis."""
        issues = []
        
        for metric_name, metric_data in brand_metrics.items():
            if isinstance(metric_data, dict):
                metric_issues = metric_data.get("issues", [])
                issues.extend(metric_issues)
        
        return issues