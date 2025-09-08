"""
Final Approval Agent
Performs final content approval assessment, aggregating all previous review stages.

🚨 DEPRECATED: This agent is deprecated and will be removed in version 3.0.0.
Use ContentGenerationWorkflowLangGraph workflow orchestration instead.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.agents.workflow.review_workflow_models import ReviewStage, ReviewAgentResult
from src.agents.workflow.review_agent_base import ReviewAgentBase
from src.core.llm_client import LLMClient
from src.utils.deprecation import deprecated_agent


@deprecated_agent(
    replacement_class="ContentGenerationWorkflowLangGraph",
    replacement_import="src.agents.workflow.content_generation_workflow_langgraph",
    migration_guide_url="https://github.com/credilinq/agent-optimization-migration/blob/main/final-approval-migration.md",
    removal_version="3.0.0",
    removal_date="2025-12-01"
)
class FinalApprovalAgent(ReviewAgentBase):
    """
    Agent responsible for final content approval in the review workflow.
    Aggregates results from quality, brand, and SEO reviews to make final approval decision.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__(
            name="FinalApprovalAgent",
            description="Final content approval and publication readiness agent",
            version="1.0.0"
        )
        self.model = model
        try:
            self.llm_client = LLMClient()
        except ValueError:
            # LLM client not available (missing API key)
            self.llm_client = None
            self.logger.warning("LLM client not available - some features will be limited")
        
        # Final approval scoring thresholds
        self.auto_approve_threshold = 0.90
        self.min_stage_scores = {
            "quality_check": 0.70,
            "brand_check": 0.75,
            "seo_review": 0.65
        }
        
        # Approval criteria weights
        self.stage_weights = {
            "quality_check": 0.40,
            "brand_check": 0.35,
            "seo_review": 0.25
        }
    
    async def execute_safe(self, content_data: Dict[str, Any], **kwargs) -> ReviewAgentResult:
        """
        Execute final approval analysis on content.
        
        Args:
            content_data: Dictionary containing content to review
            **kwargs: Additional parameters including previous_reviews
            
        Returns:
            ReviewAgentResult with final approval assessment
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract content and previous review results
            content_text = self._extract_content_text(content_data)
            content_id = content_data.get("id", "unknown")
            previous_reviews = kwargs.get("previous_reviews", {})
            campaign_context = kwargs.get("campaign_context", {})
            
            # Perform final approval analysis
            approval_metrics = await self._analyze_final_approval(
                content_text, content_data, previous_reviews, campaign_context
            )
            
            # Calculate overall score and confidence
            overall_score = self._calculate_overall_score(approval_metrics, previous_reviews)
            confidence = self._calculate_confidence(approval_metrics, previous_reviews)
            
            # Generate feedback and recommendations
            feedback = self._generate_feedback(approval_metrics, previous_reviews)
            recommendations = self._generate_recommendations(approval_metrics, previous_reviews)
            issues_found = self._identify_issues(approval_metrics, previous_reviews)
            
            # Determine if human review is required
            requires_human_review = overall_score < self.auto_approve_threshold
            auto_approved = overall_score >= self.auto_approve_threshold
            
            # Check if any critical issues prevent approval
            has_critical_issues = self._has_critical_issues(previous_reviews)
            if has_critical_issues:
                requires_human_review = True
                auto_approved = False
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ReviewAgentResult(
                stage=ReviewStage.FINAL_APPROVAL,
                content_id=content_id,
                automated_score=overall_score,
                confidence=confidence,
                feedback=feedback,
                recommendations=recommendations,
                issues_found=issues_found,
                metrics=approval_metrics,
                requires_human_review=requires_human_review,
                auto_approved=auto_approved,
                execution_time_ms=execution_time,
                model_used=self.model,
                tokens_used=0,  # Will be updated after LLM call
                cost=0.0       # Will be updated after LLM call
            )
            
        except Exception as e:
            self.logger.error(f"Final approval review failed: {str(e)}")
            raise
    
    async def _analyze_final_approval(self, content_text: str, content_data: Dict[str, Any], 
                                      previous_reviews: Dict[str, Any], campaign_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive final approval analysis."""
        
        # Parallel execution of final approval checks
        tasks = [
            self._analyze_overall_quality(previous_reviews),
            self._analyze_publication_readiness(content_text, content_data, previous_reviews),
            self._analyze_compliance_final_check(content_text, previous_reviews),
            self._analyze_business_alignment(content_text, campaign_context, previous_reviews),
            self._analyze_risk_assessment(content_text, content_data, previous_reviews)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        approval_metrics = {
            "overall_quality": results[0] if not isinstance(results[0], Exception) else {"score": 0.5, "issues": ["Quality analysis failed"]},
            "publication_readiness": results[1] if not isinstance(results[1], Exception) else {"score": 0.5, "issues": ["Readiness analysis failed"]},
            "compliance_check": results[2] if not isinstance(results[2], Exception) else {"score": 0.5, "issues": ["Compliance check failed"]},
            "business_alignment": results[3] if not isinstance(results[3], Exception) else {"score": 0.5, "issues": ["Alignment analysis failed"]},
            "risk_assessment": results[4] if not isinstance(results[4], Exception) else {"score": 0.5, "issues": ["Risk analysis failed"]},
            "previous_stages_summary": self._summarize_previous_stages(previous_reviews),
            "approval_timestamp": datetime.utcnow().isoformat()
        }
        
        return approval_metrics
    
    async def _analyze_overall_quality(self, previous_reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall quality based on previous review stages."""
        
        quality_score = 1.0
        issues = []
        suggestions = []
        strengths = []
        
        # Check each stage's results
        for stage_name, stage_weight in self.stage_weights.items():
            if stage_name in previous_reviews:
                stage_result = previous_reviews[stage_name]
                stage_score = stage_result.get("automated_score", 0.0)
                min_required = self.min_stage_scores.get(stage_name, 0.5)
                
                if stage_score < min_required:
                    quality_score -= (stage_weight * 0.5)  # Penalty for below minimum
                    issues.append(f"{stage_name.replace('_', ' ').title()} score below threshold ({stage_score:.2f} < {min_required})")
                    suggestions.append(f"Address {stage_name.replace('_', ' ')} issues before approval")
                elif stage_score >= 0.8:
                    strengths.append(f"Excellent {stage_name.replace('_', ' ')} quality ({stage_score:.2f})")
                
                # Check for unresolved issues
                stage_issues = stage_result.get("issues_found", [])
                if stage_issues:
                    critical_issues = [issue for issue in stage_issues if any(word in issue.lower() for word in ["critical", "serious", "major", "error"])]
                    if critical_issues:
                        quality_score -= 0.2
                        issues.extend(critical_issues[:3])  # Limit to 3 most critical
            else:
                quality_score -= stage_weight  # Full penalty for missing stage
                issues.append(f"Missing {stage_name.replace('_', ' ')} review")
                suggestions.append(f"Complete {stage_name.replace('_', ' ')} review before approval")
        
        return {
            "score": max(0.0, quality_score),
            "stages_reviewed": len([s for s in self.stage_weights.keys() if s in previous_reviews]),
            "total_stages_required": len(self.stage_weights),
            "issues": issues,
            "strengths": strengths,
            "improvement_suggestions": suggestions
        }
    
    async def _analyze_publication_readiness(self, content_text: str, content_data: Dict[str, Any], 
                                             previous_reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze publication readiness and completeness."""
        
        readiness_score = 1.0
        issues = []
        suggestions = []
        strengths = []
        
        # Check content completeness
        if not content_text or len(content_text.strip()) < 100:
            readiness_score -= 0.3
            issues.append("Content is too short or empty")
            suggestions.append("Ensure content meets minimum length requirements")
        else:
            strengths.append(f"Content has adequate length ({len(content_text.split())} words)")
        
        # Check title presence
        title = content_data.get("title", "")
        if not title or len(title.strip()) < 10:
            readiness_score -= 0.2
            issues.append("Title is missing or too short")
            suggestions.append("Add a compelling title")
        else:
            strengths.append("Title is present and appropriate length")
        
        # Check meta description (if available)
        meta_description = content_data.get("meta_description", "")
        if not meta_description:
            readiness_score -= 0.1
            suggestions.append("Consider adding meta description for SEO")
        else:
            strengths.append("Meta description is present")
        
        # Check for images/media
        images = content_data.get("images", [])
        if not images and not content_data.get("featured_image"):
            readiness_score -= 0.1
            suggestions.append("Consider adding visual elements to enhance engagement")
        else:
            strengths.append("Visual elements are present")
        
        # Check previous review completion
        incomplete_reviews = []
        for stage in self.stage_weights.keys():
            if stage not in previous_reviews:
                incomplete_reviews.append(stage)
        
        if incomplete_reviews:
            readiness_score -= 0.3
            issues.append(f"Incomplete review stages: {', '.join(incomplete_reviews)}")
            suggestions.append("Complete all review stages before publication")
        
        # Check for unresolved critical issues
        critical_issues = []
        for stage_name, stage_result in previous_reviews.items():
            stage_issues = stage_result.get("issues_found", [])
            critical_issues.extend([issue for issue in stage_issues if "critical" in issue.lower() or "error" in issue.lower()])
        
        if critical_issues:
            readiness_score -= 0.2
            issues.extend(critical_issues[:3])
            suggestions.append("Resolve all critical issues before publication")
        
        return {
            "score": max(0.0, readiness_score),
            "content_complete": len(content_text.strip()) >= 100,
            "metadata_complete": bool(title and len(title.strip()) >= 10),
            "reviews_complete": len(incomplete_reviews) == 0,
            "critical_issues_count": len(critical_issues),
            "issues": issues,
            "strengths": strengths,
            "improvement_suggestions": suggestions
        }
    
    async def _analyze_compliance_final_check(self, content_text: str, previous_reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Perform final compliance and legal check."""
        
        # Use AI to perform comprehensive final compliance check
        prompt = f"""
        Perform a final compliance and legal review of this content for B2B financial services:

        Content: {content_text[:2000]}...

        Previous review findings:
        {self._summarize_compliance_findings(previous_reviews)}

        Check for:
        1. Legal compliance issues
        2. Regulatory concerns
        3. Factual accuracy
        4. Misleading statements
        5. Missing disclaimers

        Provide analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "compliance_status": "compliant|requires_review|non_compliant",
            "legal_concerns": ["list of legal issues"],
            "regulatory_issues": ["regulatory compliance concerns"],
            "factual_issues": ["factual accuracy problems"],
            "disclaimer_needed": true/false,
            "risk_level": "low|medium|high",
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
            
            # Combine all compliance issues
            all_issues = (
                analysis.get("legal_concerns", []) +
                analysis.get("regulatory_issues", []) +
                analysis.get("factual_issues", [])
            )
            
            analysis["issues"] = all_issues
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Compliance final check failed: {str(e)}")
            return {
                "score": 0.8,  # Conservative score when analysis fails
                "compliance_status": "requires_review",
                "issues": [f"Compliance analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual compliance review required"],
                "risk_level": "medium"
            }
    
    async def _analyze_business_alignment(self, content_text: str, campaign_context: Dict[str, Any], 
                                          previous_reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze alignment with business objectives and campaign goals."""
        
        campaign_goals = campaign_context.get("goals", "Not specified")
        target_audience = campaign_context.get("target_audience", "Not specified")
        
        prompt = f"""
        Analyze how well this content aligns with business objectives and campaign goals:

        Content: {content_text[:1500]}...
        
        Campaign Context:
        - Goals: {campaign_goals}
        - Target Audience: {target_audience}
        
        Previous Review Summary:
        {self._summarize_business_findings(previous_reviews)}

        Evaluate:
        1. Goal alignment and achievement potential
        2. Target audience relevance and appeal
        3. Business value and impact
        4. Brand consistency and messaging
        5. Call-to-action effectiveness

        Provide analysis in this JSON format:
        {{
            "score": 0.0-1.0,
            "goal_alignment": 0.0-1.0,
            "audience_relevance": 0.0-1.0,
            "business_value": 0.0-1.0,
            "message_consistency": 0.0-1.0,
            "cta_effectiveness": 0.0-1.0,
            "alignment_strengths": ["strong alignment elements"],
            "alignment_gaps": ["areas needing improvement"],
            "improvement_suggestions": ["business alignment improvements"]
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
            
            analysis["issues"] = analysis.get("alignment_gaps", [])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Business alignment analysis failed: {str(e)}")
            return {
                "score": 0.7,
                "issues": [f"Business alignment analysis error: {str(e)}"],
                "improvement_suggestions": ["Manual business alignment review needed"]
            }
    
    async def _analyze_risk_assessment(self, content_text: str, content_data: Dict[str, Any], 
                                       previous_reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Perform final risk assessment for publication."""
        
        risk_score = 1.0  # Start with low risk
        risks = []
        mitigations = []
        
        # Check for high-risk keywords or topics
        high_risk_indicators = [
            "guaranteed", "risk-free", "100%", "never", "always", "instant",
            "secret", "exclusive", "limited time", "act now", "urgent"
        ]
        
        content_lower = content_text.lower()
        found_risk_indicators = [indicator for indicator in high_risk_indicators if indicator in content_lower]
        
        if found_risk_indicators:
            risk_score -= 0.2
            risks.append(f"High-risk language detected: {', '.join(found_risk_indicators[:3])}")
            mitigations.append("Review and qualify absolute statements")
        
        # Check previous reviews for risk indicators
        for stage_name, stage_result in previous_reviews.items():
            stage_issues = stage_result.get("issues_found", [])
            risk_issues = [issue for issue in stage_issues if any(word in issue.lower() for word in ["risk", "compliance", "legal", "error"])]
            
            if risk_issues:
                risk_score -= 0.1
                risks.extend(risk_issues[:2])
        
        # Check content type specific risks
        content_type = content_data.get("type", "blog")
        if content_type == "financial_advice" and "disclaimer" not in content_lower:
            risk_score -= 0.3
            risks.append("Financial content missing appropriate disclaimers")
            mitigations.append("Add financial disclaimer and risk warnings")
        
        # Check for missing author or source attribution
        if not content_data.get("author") and not content_data.get("source"):
            risk_score -= 0.1
            risks.append("Missing author or source attribution")
            mitigations.append("Add author credentials or source information")
        
        # Determine overall risk level
        if risk_score >= 0.8:
            risk_level = "low"
        elif risk_score >= 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "score": max(0.0, risk_score),
            "risk_level": risk_level,
            "risk_indicators_found": found_risk_indicators,
            "publication_risks": risks,
            "risk_mitigations": mitigations,
            "issues": risks,
            "improvement_suggestions": mitigations
        }
    
    def _summarize_previous_stages(self, previous_reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize results from previous review stages."""
        summary = {}
        
        for stage_name, stage_result in previous_reviews.items():
            summary[stage_name] = {
                "score": stage_result.get("automated_score", 0.0),
                "requires_human_review": stage_result.get("requires_human_review", False),
                "issues_count": len(stage_result.get("issues_found", [])),
                "recommendations_count": len(stage_result.get("recommendations", []))
            }
        
        return summary
    
    def _summarize_compliance_findings(self, previous_reviews: Dict[str, Any]) -> str:
        """Summarize compliance-related findings from previous reviews."""
        findings = []
        
        for stage_name, stage_result in previous_reviews.items():
            issues = stage_result.get("issues_found", [])
            compliance_issues = [issue for issue in issues if any(word in issue.lower() for word in ["compliance", "legal", "disclaimer", "regulation"])]
            if compliance_issues:
                findings.extend(compliance_issues)
        
        return "\n".join(findings) if findings else "No significant compliance issues found in previous reviews."
    
    def _summarize_business_findings(self, previous_reviews: Dict[str, Any]) -> str:
        """Summarize business-related findings from previous reviews."""
        findings = []
        
        for stage_name, stage_result in previous_reviews.items():
            if stage_name == "brand_check":
                score = stage_result.get("automated_score", 0.0)
                findings.append(f"Brand alignment score: {score:.2f}")
                
                recommendations = stage_result.get("recommendations", [])
                if recommendations:
                    findings.append(f"Brand recommendations: {'; '.join(recommendations[:2])}")
        
        return "\n".join(findings) if findings else "Previous reviews show good business alignment."
    
    def _has_critical_issues(self, previous_reviews: Dict[str, Any]) -> bool:
        """Check if there are any critical issues that prevent approval."""
        for stage_name, stage_result in previous_reviews.items():
            # Check if any stage has very low score
            score = stage_result.get("automated_score", 0.0)
            if score < 0.3:
                return True
            
            # Check for critical issues
            issues = stage_result.get("issues_found", [])
            critical_issues = [issue for issue in issues if any(word in issue.lower() for word in ["critical", "error", "failure", "illegal"])]
            if critical_issues:
                return True
        
        return False
    
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
    
    def _calculate_overall_score(self, approval_metrics: Dict[str, Any], previous_reviews: Dict[str, Any]) -> float:
        """Calculate weighted overall approval score."""
        # Combine current approval analysis with weighted previous stage scores
        approval_weights = {
            "overall_quality": 0.30,
            "publication_readiness": 0.25,
            "compliance_check": 0.25,
            "business_alignment": 0.15,
            "risk_assessment": 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        # Add approval analysis scores
        for metric, weight in approval_weights.items():
            if metric in approval_metrics and "score" in approval_metrics[metric]:
                weighted_score += approval_metrics[metric]["score"] * weight
                total_weight += weight
        
        # Factor in previous stage scores with their weights
        for stage_name, stage_weight in self.stage_weights.items():
            if stage_name in previous_reviews:
                stage_score = previous_reviews[stage_name].get("automated_score", 0.0)
                weighted_score += stage_score * stage_weight * 0.3  # 30% weight for previous stages
                total_weight += stage_weight * 0.3
        
        return weighted_score / max(total_weight, 1.0)
    
    def _calculate_confidence(self, approval_metrics: Dict[str, Any], previous_reviews: Dict[str, Any]) -> float:
        """Calculate confidence in the final approval assessment."""
        confidence_factors = []
        
        # Higher confidence when all stages are complete
        completed_stages = len(previous_reviews)
        required_stages = len(self.stage_weights)
        stage_completion_confidence = completed_stages / required_stages
        confidence_factors.append(stage_completion_confidence)
        
        # Higher confidence when scores are more definitive
        for metric_name, metric_data in approval_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                score = metric_data["score"]
                confidence_factors.append(abs(score - 0.5) * 2)  # More confident with extreme scores
        
        return sum(confidence_factors) / max(len(confidence_factors), 1) if confidence_factors else 0.5
    
    def _generate_feedback(self, approval_metrics: Dict[str, Any], previous_reviews: Dict[str, Any]) -> List[str]:
        """Generate human-readable feedback."""
        feedback = []
        
        # Summarize overall status
        overall_score = self._calculate_overall_score(approval_metrics, previous_reviews)
        if overall_score >= 0.9:
            feedback.append("Content ready for publication - all quality standards met")
        elif overall_score >= 0.8:
            feedback.append("Content nearly ready - minor improvements recommended")
        elif overall_score >= 0.7:
            feedback.append("Content requires attention - several issues need resolution")
        else:
            feedback.append("Content not ready for publication - significant improvements needed")
        
        # Add stage-specific feedback
        for metric_name, metric_data in approval_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                score = metric_data["score"]
                readable_name = metric_name.replace('_', ' ').title()
                
                if score >= 0.8:
                    feedback.append(f"Excellent {readable_name}")
                elif score >= 0.6:
                    feedback.append(f"Good {readable_name}")
                else:
                    feedback.append(f"{readable_name} needs improvement")
        
        return feedback
    
    def _generate_recommendations(self, approval_metrics: Dict[str, Any], previous_reviews: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Add recommendations from current analysis
        for metric_name, metric_data in approval_metrics.items():
            if isinstance(metric_data, dict):
                suggestions = metric_data.get("improvement_suggestions", [])
                recommendations.extend(suggestions)
        
        # Add high-priority recommendations from previous stages
        for stage_name, stage_result in previous_reviews.items():
            stage_score = stage_result.get("automated_score", 0.0)
            if stage_score < 0.7:  # Only include recommendations from low-scoring stages
                stage_recommendations = stage_result.get("recommendations", [])
                recommendations.extend(stage_recommendations[:2])  # Limit to top 2 per stage
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_issues(self, approval_metrics: Dict[str, Any], previous_reviews: Dict[str, Any]) -> List[str]:
        """Identify specific issues found during final approval analysis."""
        issues = []
        
        # Add issues from current analysis
        for metric_name, metric_data in approval_metrics.items():
            if isinstance(metric_data, dict):
                metric_issues = metric_data.get("issues", [])
                issues.extend(metric_issues)
        
        # Add critical issues from previous stages
        for stage_name, stage_result in previous_reviews.items():
            stage_issues = stage_result.get("issues_found", [])
            critical_issues = [issue for issue in stage_issues if any(word in issue.lower() for word in ["critical", "error", "compliance"])]
            issues.extend(critical_issues)
        
        return issues