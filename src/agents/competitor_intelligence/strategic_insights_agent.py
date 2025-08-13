"""
Strategic Insights Agent for generating high-level competitive intelligence and strategic recommendations.
Synthesizes data from other agents to provide actionable business insights and strategic guidance.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import asdict
import json

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

from ..core.base_agent import BaseAgent
from .models import (
    CompetitorInsight, Trend, ContentGap, MarketAnalysis,
    Competitor, ContentItem, Industry, AlertPriority,
    CompetitorIntelligenceReport, CompetitorTier
)
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class StrategicInsightsAgent(BaseAgent):
    """
    Specialized agent for generating strategic insights and high-level competitive intelligence.
    Synthesizes data from multiple sources to provide actionable business recommendations.
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..core.base_agent import AgentMetadata, AgentType
        
        metadata = AgentMetadata(
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            name="StrategicInsightsAgent"
        )
        super().__init__(metadata)
        
        # Initialize AI for strategic analysis (lazy loading to avoid requiring API keys at startup)
        self.strategy_llm = None
        
        # Strategic analysis configuration
        self.strategy_config = {
            "insight_confidence_threshold": 0.7,
            "high_impact_threshold": 80,
            "competitive_threat_threshold": 75,
            "opportunity_score_threshold": 70,
            "trend_significance_threshold": 0.6,
            "max_insights_per_report": 25,
            "strategic_time_horizon_months": 12
        }
        
        # Strategic frameworks and templates
        self.strategic_frameworks = {
            "porter_five_forces": {
                "competitive_rivalry": "intensity_of_competition",
                "supplier_power": "supplier_bargaining_power", 
                "buyer_power": "customer_bargaining_power",
                "threat_of_substitution": "substitute_products_threat",
                "barriers_to_entry": "new_entrants_threat"
            },
            "swot_categories": {
                "strengths": "competitive_advantages",
                "weaknesses": "areas_for_improvement",
                "opportunities": "market_opportunities",
                "threats": "competitive_threats"
            },
            "growth_strategies": [
                "market_penetration",
                "market_development", 
                "product_development",
                "diversification"
            ]
        }
        
        # Industry-specific strategic considerations
        self.industry_strategies = {
            Industry.FINTECH: {
                "key_success_factors": [
                    "regulatory_compliance", "security_trust", "user_experience",
                    "partnership_ecosystem", "technological_innovation"
                ],
                "competitive_moats": [
                    "regulatory_licenses", "network_effects", "data_advantages",
                    "brand_trust", "integration_ecosystem"
                ]
            },
            Industry.SAAS: {
                "key_success_factors": [
                    "product_market_fit", "customer_retention", "scalability",
                    "integration_capabilities", "customer_success"
                ],
                "competitive_moats": [
                    "switching_costs", "network_effects", "data_network_effects",
                    "api_ecosystem", "customer_lock_in"
                ]
            },
            Industry.MARKETING: {
                "key_success_factors": [
                    "thought_leadership", "client_results", "innovation",
                    "talent_quality", "service_delivery"
                ],
                "competitive_moats": [
                    "brand_reputation", "client_relationships", "proprietary_methodologies",
                    "talent_network", "case_study_portfolio"
                ]
            }
        }
    
    def _get_strategy_llm(self):
        """Lazy initialize the strategy LLM."""
        if self.strategy_llm is None:
            try:
                self.strategy_llm = ChatOpenAI(
                    model="gpt-4",  # Use GPT-4 for more sophisticated strategic analysis
                    temperature=0.1,  # Low temperature for consistent strategic insights
                    max_tokens=2500
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI LLM: {e}")
                return None
        return self.strategy_llm
    
    async def generate_strategic_insights(
        self,
        competitors: List[Competitor],
        market_trends: List[Trend],
        content_gaps: List[ContentGap],
        performance_analysis: Dict[str, Any],
        industry: Industry
    ) -> List[CompetitorInsight]:
        """
        Generate comprehensive strategic insights from all available competitive intelligence data.
        """
        
        async with async_performance_tracker("generate_strategic_insights"):
            self.logger.info(f"Generating strategic insights for {len(competitors)} competitors")
            
            # Analyze competitive landscape
            competitive_landscape = await self._analyze_competitive_landscape(
                competitors, performance_analysis, industry
            )
            
            # Identify market opportunities
            market_opportunities = await self._identify_market_opportunities(
                market_trends, content_gaps, competitive_landscape
            )
            
            # Assess competitive threats
            competitive_threats = await self._assess_competitive_threats(
                competitors, market_trends, performance_analysis
            )
            
            # Generate positioning insights
            positioning_insights = await self._generate_positioning_insights(
                competitive_landscape, market_opportunities, industry
            )
            
            # Synthesize strategic recommendations
            strategic_recommendations = await self._synthesize_strategic_recommendations(
                competitive_landscape,
                market_opportunities,
                competitive_threats,
                positioning_insights,
                industry
            )
            
            # Create insight objects
            all_insights = []
            all_insights.extend(competitive_landscape.get("insights", []))
            all_insights.extend(market_opportunities.get("insights", []))
            all_insights.extend(competitive_threats.get("insights", []))
            all_insights.extend(positioning_insights)
            all_insights.extend(strategic_recommendations)
            
            # Filter and rank insights
            high_value_insights = await self._filter_and_rank_insights(all_insights)
            
            # Track metrics
            metrics.increment_counter(
                "strategic_insights.generated",
                tags={
                    "industry": industry.value,
                    "competitors_analyzed": str(len(competitors)),
                    "insights_generated": str(len(high_value_insights))
                }
            )
            
            return high_value_insights
    
    async def _analyze_competitive_landscape(
        self,
        competitors: List[Competitor],
        performance_analysis: Dict[str, Any],
        industry: Industry
    ) -> Dict[str, Any]:
        """Analyze the overall competitive landscape and positioning."""
        
        # Categorize competitors by tier and performance
        competitor_segments = {
            "market_leaders": [],
            "strong_competitors": [],
            "emerging_players": [],
            "niche_players": []
        }
        
        # Get performance data
        engagement_analysis = performance_analysis.get("engagement_analysis", {})
        
        for competitor in competitors:
            competitor_data = engagement_analysis.get(competitor.id, {})
            avg_engagement = competitor_data.get("avg_engagement", 0)
            
            # Segment competitors based on tier and performance
            if competitor.tier == CompetitorTier.DIRECT and avg_engagement > 30:
                competitor_segments["market_leaders"].append({
                    "competitor": competitor,
                    "performance": competitor_data
                })
            elif avg_engagement > 20:
                competitor_segments["strong_competitors"].append({
                    "competitor": competitor,
                    "performance": competitor_data
                })
            elif competitor.tier == CompetitorTier.INDIRECT:
                competitor_segments["niche_players"].append({
                    "competitor": competitor,
                    "performance": competitor_data
                })
            else:
                competitor_segments["emerging_players"].append({
                    "competitor": competitor,
                    "performance": competitor_data
                })
        
        # Analyze market concentration
        total_competitors = len(competitors)
        market_leaders_count = len(competitor_segments["market_leaders"])
        market_concentration = market_leaders_count / max(total_competitors, 1)
        
        # Generate competitive landscape insights
        landscape_insights = []
        
        # Market concentration insight
        if market_concentration > 0.3:  # High concentration
            insight = CompetitorInsight(
                id="landscape_concentration",
                competitor_id="market",
                insight_type="market_structure",
                title="Concentrated Market with Dominant Players",
                description=f"Market shows high concentration with {market_leaders_count} dominant players out of {total_competitors} total competitors",
                confidence_score=0.9,
                impact_level="high",
                supporting_evidence=[
                    f"Market concentration ratio: {market_concentration:.1%}",
                    f"Dominant players: {market_leaders_count}/{total_competitors}"
                ],
                recommendations=[
                    "Focus on differentiation rather than direct competition",
                    "Identify underserved market segments",
                    "Consider partnerships with non-competing market leaders"
                ]
            )
            landscape_insights.append(insight)
        
        # Performance gap insight
        if competitor_segments["market_leaders"]:
            leader_performance = max([
                comp["performance"].get("avg_engagement", 0)
                for comp in competitor_segments["market_leaders"]
            ])
            
            avg_market_performance = np.mean([
                comp["performance"].get("avg_engagement", 0)
                for segment in competitor_segments.values()
                for comp in segment
            ])
            
            if leader_performance > avg_market_performance * 2:
                insight = CompetitorInsight(
                    id="performance_gap",
                    competitor_id="market",
                    insight_type="performance_analysis",
                    title="Significant Performance Gap Between Leaders and Market",
                    description=f"Top performers achieve {leader_performance:.1f} engagement vs {avg_market_performance:.1f} market average",
                    confidence_score=0.85,
                    impact_level="high",
                    supporting_evidence=[
                        f"Leader performance: {leader_performance:.1f}",
                        f"Market average: {avg_market_performance:.1f}",
                        f"Performance gap: {leader_performance/avg_market_performance:.1f}x"
                    ],
                    recommendations=[
                        "Study top performer strategies and tactics",
                        "Identify performance improvement opportunities",
                        "Benchmark against market leaders, not average"
                    ]
                )
                landscape_insights.append(insight)
        
        return {
            "competitor_segments": competitor_segments,
            "market_concentration": market_concentration,
            "insights": landscape_insights,
            "competitive_intensity": self._calculate_competitive_intensity(competitor_segments)
        }
    
    async def _identify_market_opportunities(
        self,
        market_trends: List[Trend],
        content_gaps: List[ContentGap],
        competitive_landscape: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify strategic market opportunities from trends and gaps."""
        
        # Analyze high-opportunity trends
        high_opportunity_trends = [
            trend for trend in market_trends
            if (trend.opportunity_score or 0) > self.strategy_config["opportunity_score_threshold"]
        ]
        
        # Analyze high-value content gaps
        high_value_gaps = [
            gap for gap in content_gaps
            if gap.opportunity_score > self.strategy_config["opportunity_score_threshold"]
        ]
        
        # Generate opportunity insights
        opportunity_insights = []
        
        # Trending topic opportunities
        if high_opportunity_trends:
            top_trend = max(high_opportunity_trends, key=lambda t: t.opportunity_score or 0)
            
            insight = CompetitorInsight(
                id=f"trend_opportunity_{top_trend.id}",
                competitor_id="market",
                insight_type="market_opportunity",
                title=f"High-Value Trend Opportunity: {top_trend.topic}",
                description=f"Emerging trend '{top_trend.topic}' shows {top_trend.strength.value} strength with {top_trend.opportunity_score:.0f}% opportunity score",
                confidence_score=min((top_trend.opportunity_score or 50) / 100, 1.0),
                impact_level="high" if (top_trend.opportunity_score or 0) > 85 else "medium",
                supporting_evidence=[
                    f"Trend strength: {top_trend.strength.value}",
                    f"Growth rate: {top_trend.growth_rate:.1%}",
                    f"Opportunity score: {top_trend.opportunity_score:.0f}%"
                ],
                recommendations=[
                    f"Develop content strategy around '{top_trend.topic}'",
                    "Position as thought leader in this emerging area",
                    "Monitor trend development for optimal timing"
                ]
            )
            opportunity_insights.append(insight)
        
        # Content gap opportunities
        if high_value_gaps:
            top_gap = max(high_value_gaps, key=lambda g: g.opportunity_score)
            
            insight = CompetitorInsight(
                id=f"content_gap_{top_gap.id}",
                competitor_id="market",
                insight_type="content_opportunity",
                title=f"Underserved Content Opportunity: {top_gap.topic}",
                description=f"Content gap in '{top_gap.topic}' with {top_gap.opportunity_score:.0f}% opportunity score and {top_gap.potential_reach} potential reach",
                confidence_score=0.8,
                impact_level="high" if top_gap.opportunity_score > 85 else "medium",
                supporting_evidence=[
                    f"Opportunity score: {top_gap.opportunity_score:.0f}%",
                    f"Potential reach: {top_gap.potential_reach:,}",
                    f"Missing content types: {len(top_gap.content_types_missing)}"
                ],
                recommendations=[
                    f"Create comprehensive content about '{top_gap.topic}'",
                    f"Focus on {', '.join([ct.value for ct in top_gap.content_types_missing[:2]])} content",
                    "Establish thought leadership in this underserved area"
                ]
            )
            opportunity_insights.append(insight)
        
        # Blue ocean opportunities (low competition, high opportunity)
        blue_ocean_gaps = [
            gap for gap in high_value_gaps
            if gap.difficulty_score < 50  # Low competition
        ]
        
        if blue_ocean_gaps:
            insight = CompetitorInsight(
                id="blue_ocean_opportunity",
                competitor_id="market",
                insight_type="strategic_opportunity",
                title=f"Blue Ocean Opportunities Identified",
                description=f"Found {len(blue_ocean_gaps)} low-competition, high-opportunity content areas",
                confidence_score=0.85,
                impact_level="high",
                supporting_evidence=[
                    f"Blue ocean topics: {len(blue_ocean_gaps)}",
                    f"Average opportunity score: {np.mean([g.opportunity_score for g in blue_ocean_gaps]):.0f}%",
                    f"Average difficulty: {np.mean([g.difficulty_score for g in blue_ocean_gaps]):.0f}%"
                ],
                recommendations=[
                    "Prioritize blue ocean opportunities for quick wins",
                    "Establish early market presence in these areas",
                    "Build competitive moats before others enter"
                ]
            )
            opportunity_insights.append(insight)
        
        return {
            "high_opportunity_trends": high_opportunity_trends,
            "high_value_gaps": high_value_gaps,
            "blue_ocean_opportunities": blue_ocean_gaps,
            "insights": opportunity_insights
        }
    
    async def _assess_competitive_threats(
        self,
        competitors: List[Competitor],
        market_trends: List[Trend],
        performance_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess competitive threats and emerging risks."""
        
        engagement_analysis = performance_analysis.get("engagement_analysis", {})
        posting_patterns = performance_analysis.get("posting_patterns", {})
        
        # Identify high-performing competitors as potential threats
        threat_competitors = []
        
        for competitor in competitors:
            competitor_data = engagement_analysis.get(competitor.id, {})
            pattern_data = posting_patterns.get(competitor.id, {})
            
            # Threat scoring
            threat_score = 0
            
            # Performance threat
            avg_engagement = competitor_data.get("avg_engagement", 0)
            if avg_engagement > 25:
                threat_score += 30
            
            # Consistency threat
            consistency = competitor_data.get("engagement_consistency", 0)
            if consistency > 0.7:
                threat_score += 20
            
            # Growth threat
            trend = competitor_data.get("engagement_trend", "stable")
            if trend == "improving":
                threat_score += 25
            
            # Activity threat
            activity_trend = pattern_data.get("activity_trend", "stable")
            if activity_trend == "increasing":
                threat_score += 15
            
            # Tier-based adjustment
            if competitor.tier == CompetitorTier.DIRECT:
                threat_score += 10
            
            if threat_score > self.strategy_config["competitive_threat_threshold"]:
                threat_competitors.append({
                    "competitor": competitor,
                    "threat_score": threat_score,
                    "threat_factors": self._identify_threat_factors(
                        competitor_data, pattern_data, competitor.tier
                    )
                })
        
        # Generate threat insights
        threat_insights = []
        
        if threat_competitors:
            # Sort by threat score
            threat_competitors.sort(key=lambda x: x["threat_score"], reverse=True)
            top_threat = threat_competitors[0]
            
            insight = CompetitorInsight(
                id=f"competitive_threat_{top_threat['competitor'].id}",
                competitor_id=top_threat['competitor'].id,
                insight_type="competitive_threat",
                title=f"High Competitive Threat: {top_threat['competitor'].name}",
                description=f"{top_threat['competitor'].name} poses significant competitive threat with {top_threat['threat_score']} threat score",
                confidence_score=0.8,
                impact_level="high",
                supporting_evidence=top_threat["threat_factors"],
                recommendations=[
                    "Monitor this competitor's activities closely",
                    "Analyze their successful strategies and tactics",
                    "Consider defensive positioning strategies",
                    "Identify areas where you can differentiate"
                ]
            )
            threat_insights.append(insight)
        
        # Emerging market threats from trends
        viral_trends = [trend for trend in market_trends if trend.strength.value == "viral"]
        if viral_trends:
            insight = CompetitorInsight(
                id="viral_trend_threat",
                competitor_id="market",
                insight_type="market_threat",
                title="Viral Trends Creating Market Disruption",
                description=f"Identified {len(viral_trends)} viral trends that could disrupt current market dynamics",
                confidence_score=0.75,
                impact_level="medium",
                supporting_evidence=[
                    f"Viral trends: {len(viral_trends)}",
                    f"Topics: {', '.join([t.topic for t in viral_trends[:3]])}"
                ],
                recommendations=[
                    "Monitor viral trends for potential disruption",
                    "Assess ability to capitalize on or defend against trends",
                    "Consider rapid response strategies for viral topics"
                ]
            )
            threat_insights.append(insight)
        
        return {
            "threat_competitors": threat_competitors,
            "viral_trends": viral_trends,
            "insights": threat_insights
        }
    
    async def _generate_positioning_insights(
        self,
        competitive_landscape: Dict[str, Any],
        market_opportunities: Dict[str, Any],
        industry: Industry
    ) -> List[CompetitorInsight]:
        """Generate strategic positioning insights."""
        
        positioning_insights = []
        
        # Get industry-specific strategic factors
        industry_strategy = self.industry_strategies.get(industry, {})
        key_success_factors = industry_strategy.get("key_success_factors", [])
        
        # Analyze market positioning opportunities
        competitor_segments = competitive_landscape.get("competitor_segments", {})
        market_leaders = competitor_segments.get("market_leaders", [])
        
        # Differentiation positioning insight
        if len(market_leaders) > 2:
            insight = CompetitorInsight(
                id="differentiation_positioning",
                competitor_id="strategic",
                insight_type="positioning_strategy",
                title="Differentiation Strategy Recommended",
                description=f"With {len(market_leaders)} established market leaders, differentiation positioning offers better success potential than direct competition",
                confidence_score=0.85,
                impact_level="high",
                supporting_evidence=[
                    f"Market leaders: {len(market_leaders)}",
                    f"Market concentration: {competitive_landscape.get('market_concentration', 0):.1%}",
                    f"Competitive intensity: {competitive_landscape.get('competitive_intensity', 'medium')}"
                ],
                recommendations=[
                    "Focus on unique value proposition development",
                    "Identify underserved customer segments",
                    "Leverage content gaps for thought leadership positioning",
                    "Build distinctive capabilities and competitive moats"
                ]
            )
            positioning_insights.append(insight)
        
        # Niche positioning opportunity
        blue_ocean_opportunities = market_opportunities.get("blue_ocean_opportunities", [])
        if blue_ocean_opportunities:
            insight = CompetitorInsight(
                id="niche_positioning",
                competitor_id="strategic",
                insight_type="positioning_strategy",
                title="Niche Market Leadership Opportunity",
                description=f"Identified {len(blue_ocean_opportunities)} underserved niches suitable for market leadership positioning",
                confidence_score=0.8,
                impact_level="medium",
                supporting_evidence=[
                    f"Blue ocean opportunities: {len(blue_ocean_opportunities)}",
                    f"Average opportunity score: {np.mean([op.opportunity_score for op in blue_ocean_opportunities]):.0f}%"
                ],
                recommendations=[
                    "Select 1-2 niche areas for focused positioning",
                    "Build deep expertise and thought leadership",
                    "Establish market presence before competitors enter",
                    "Scale successful niches to adjacent markets"
                ]
            )
            positioning_insights.append(insight)
        
        # Innovation positioning insight
        high_opportunity_trends = market_opportunities.get("high_opportunity_trends", [])
        if high_opportunity_trends:
            innovation_trends = [
                trend for trend in high_opportunity_trends
                if any(keyword in trend.topic.lower() for keyword in ["ai", "innovation", "technology", "digital", "automation"])
            ]
            
            if innovation_trends:
                insight = CompetitorInsight(
                    id="innovation_positioning",
                    competitor_id="strategic",
                    insight_type="positioning_strategy",
                    title="Innovation Leader Positioning Opportunity",
                    description=f"Emerging technology trends provide opportunity for innovation leadership positioning",
                    confidence_score=0.75,
                    impact_level="medium",
                    supporting_evidence=[
                        f"Innovation trends: {len(innovation_trends)}",
                        f"Technology topics gaining momentum"
                    ],
                    recommendations=[
                        "Develop thought leadership in emerging technologies",
                        "Create educational content about innovation trends",
                        "Position as forward-thinking industry innovator",
                        "Build partnerships with technology providers"
                    ]
                )
                positioning_insights.append(insight)
        
        return positioning_insights
    
    async def _synthesize_strategic_recommendations(
        self,
        competitive_landscape: Dict[str, Any],
        market_opportunities: Dict[str, Any],
        competitive_threats: Dict[str, Any],
        positioning_insights: List[CompetitorInsight],
        industry: Industry
    ) -> List[CompetitorInsight]:
        """Synthesize comprehensive strategic recommendations."""
        
        # Use AI to generate sophisticated strategic recommendations
        try:
            # Prepare strategic context
            context = {
                "market_structure": {
                    "concentration": competitive_landscape.get("market_concentration", 0),
                    "competitive_intensity": competitive_landscape.get("competitive_intensity", "medium"),
                    "market_leaders_count": len(competitive_landscape.get("competitor_segments", {}).get("market_leaders", []))
                },
                "opportunities": {
                    "high_value_trends": len(market_opportunities.get("high_opportunity_trends", [])),
                    "content_gaps": len(market_opportunities.get("high_value_gaps", [])),
                    "blue_ocean_count": len(market_opportunities.get("blue_ocean_opportunities", []))
                },
                "threats": {
                    "high_threat_competitors": len([t for t in competitive_threats.get("threat_competitors", []) if t["threat_score"] > 80]),
                    "viral_trends": len(competitive_threats.get("viral_trends", []))
                }
            }
            
            prompt = f"""
            As a strategic business consultant, analyze this competitive intelligence data for the {industry.value} industry and provide 3-4 high-level strategic recommendations.
            
            Market Context:
            - Market concentration: {context['market_structure']['concentration']:.1%}
            - Competitive intensity: {context['market_structure']['competitive_intensity']}
            - Market leaders: {context['market_structure']['market_leaders_count']}
            
            Opportunities:
            - High-value trends: {context['opportunities']['high_value_trends']}
            - Content gaps: {context['opportunities']['content_gaps']}
            - Blue ocean opportunities: {context['opportunities']['blue_ocean_count']}
            
            Threats:
            - High-threat competitors: {context['threats']['high_threat_competitors']}
            - Viral/disruptive trends: {context['threats']['viral_trends']}
            
            Provide strategic recommendations in this format:
            1. [Strategic Focus Area]: [Recommendation] - [Rationale]
            2. [Strategic Focus Area]: [Recommendation] - [Rationale]
            
            Focus on actionable, high-impact strategies based on the competitive landscape.
            """
            
            llm = self._get_strategy_llm()
            if llm is None:
                # Fallback to rule-based recommendations if LLM not available
                return self._generate_fallback_recommendations(
                    competitive_landscape,
                    market_opportunities,
                    competitive_threats,
                    industry
                )
            
            response = await llm.agenerate([
                [HumanMessage(content=prompt)]
            ])
            
            strategic_analysis = response.generations[0][0].text.strip()
            
            # Parse AI recommendations into insights
            recommendations = self._parse_strategic_recommendations(strategic_analysis)
            
            return recommendations
            
        except Exception as e:
            self.logger.debug(f"Failed to generate AI strategic recommendations: {str(e)}")
            
            # Fallback to rule-based recommendations
            return self._generate_fallback_recommendations(
                competitive_landscape,
                market_opportunities,
                competitive_threats,
                industry
            )
    
    def _parse_strategic_recommendations(self, ai_analysis: str) -> List[CompetitorInsight]:
        """Parse AI-generated strategic recommendations into insight objects."""
        
        recommendations = []
        lines = ai_analysis.split('\n')
        
        current_recommendation = None
        for line in lines:
            line = line.strip()
            
            # Look for numbered recommendations
            if line and (line[0].isdigit() or line.startswith('-')):
                if current_recommendation:
                    # Save previous recommendation
                    recommendations.append(current_recommendation)
                
                # Parse new recommendation
                parts = line.split(':', 1)
                if len(parts) == 2:
                    title_part = parts[0].strip('123456789.- ')
                    description = parts[1].strip()
                    
                    current_recommendation = CompetitorInsight(
                        id=f"strategic_rec_{len(recommendations)}",
                        competitor_id="strategic",
                        insight_type="strategic_recommendation",
                        title=title_part,
                        description=description,
                        confidence_score=0.8,
                        impact_level="high",
                        supporting_evidence=["AI strategic analysis"],
                        recommendations=[description]
                    )
            elif current_recommendation and line:
                # Continuation of current recommendation
                current_recommendation.description += " " + line
        
        # Add final recommendation
        if current_recommendation:
            recommendations.append(current_recommendation)
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def _generate_fallback_recommendations(
        self,
        competitive_landscape: Dict[str, Any],
        market_opportunities: Dict[str, Any],
        competitive_threats: Dict[str, Any],
        industry: Industry
    ) -> List[CompetitorInsight]:
        """Generate fallback strategic recommendations using rule-based logic."""
        
        recommendations = []
        
        # Market concentration-based recommendation
        market_concentration = competitive_landscape.get("market_concentration", 0)
        if market_concentration > 0.4:
            recommendations.append(CompetitorInsight(
                id="differentiation_strategy",
                competitor_id="strategic",
                insight_type="strategic_recommendation",
                title="Pursue Differentiation Strategy",
                description="High market concentration suggests differentiation over direct competition",
                confidence_score=0.8,
                impact_level="high",
                supporting_evidence=[f"Market concentration: {market_concentration:.1%}"],
                recommendations=["Focus on unique value propositions", "Target underserved segments"]
            ))
        
        # Opportunity-based recommendation
        blue_ocean_count = len(market_opportunities.get("blue_ocean_opportunities", []))
        if blue_ocean_count > 0:
            recommendations.append(CompetitorInsight(
                id="blue_ocean_strategy",
                competitor_id="strategic",
                insight_type="strategic_recommendation",
                title="Capitalize on Blue Ocean Opportunities",
                description=f"Pursue {blue_ocean_count} identified low-competition, high-opportunity areas",
                confidence_score=0.85,
                impact_level="high",
                supporting_evidence=[f"Blue ocean opportunities: {blue_ocean_count}"],
                recommendations=["Establish early market presence", "Build competitive moats"]
            ))
        
        return recommendations
    
    async def _filter_and_rank_insights(
        self,
        insights: List[CompetitorInsight]
    ) -> List[CompetitorInsight]:
        """Filter and rank insights by strategic value."""
        
        # Filter by confidence threshold
        filtered_insights = [
            insight for insight in insights
            if insight.confidence_score >= self.strategy_config["insight_confidence_threshold"]
        ]
        
        # Calculate strategic value score
        for insight in filtered_insights:
            strategic_value = 0
            
            # Impact level contribution
            impact_scores = {"low": 25, "medium": 50, "high": 100}
            strategic_value += impact_scores.get(insight.impact_level, 50)
            
            # Confidence contribution
            strategic_value += insight.confidence_score * 50
            
            # Insight type contribution
            type_scores = {
                "strategic_recommendation": 100,
                "competitive_threat": 80,
                "market_opportunity": 90,
                "positioning_strategy": 85,
                "strategic_analysis": 75
            }
            strategic_value += type_scores.get(insight.insight_type, 50)
            
            insight.metadata = insight.metadata or {}
            insight.metadata["strategic_value"] = strategic_value
        
        # Sort by strategic value
        ranked_insights = sorted(
            filtered_insights,
            key=lambda i: i.metadata.get("strategic_value", 0),
            reverse=True
        )
        
        return ranked_insights[:self.strategy_config["max_insights_per_report"]]
    
    # Helper methods
    
    def _calculate_competitive_intensity(self, competitor_segments: Dict[str, List]) -> str:
        """Calculate competitive intensity based on market structure."""
        
        total_competitors = sum(len(segment) for segment in competitor_segments.values())
        market_leaders = len(competitor_segments.get("market_leaders", []))
        strong_competitors = len(competitor_segments.get("strong_competitors", []))
        
        # Calculate intensity score
        intensity_score = 0
        
        # More competitors = higher intensity
        if total_competitors > 10:
            intensity_score += 30
        elif total_competitors > 5:
            intensity_score += 20
        else:
            intensity_score += 10
        
        # More strong players = higher intensity
        strong_players = market_leaders + strong_competitors
        if strong_players > 5:
            intensity_score += 40
        elif strong_players > 2:
            intensity_score += 25
        else:
            intensity_score += 10
        
        # Market concentration affects intensity
        concentration = market_leaders / max(total_competitors, 1)
        if concentration > 0.5:
            intensity_score += 30  # High concentration = intense competition
        elif concentration > 0.3:
            intensity_score += 20
        else:
            intensity_score += 10
        
        # Classify intensity
        if intensity_score > 80:
            return "very_high"
        elif intensity_score > 60:
            return "high"
        elif intensity_score > 40:
            return "medium"
        else:
            return "low"
    
    def _identify_threat_factors(
        self,
        competitor_data: Dict[str, Any],
        pattern_data: Dict[str, Any],
        competitor_tier: CompetitorTier
    ) -> List[str]:
        """Identify specific threat factors for a competitor."""
        
        threat_factors = []
        
        # Performance factors
        avg_engagement = competitor_data.get("avg_engagement", 0)
        if avg_engagement > 30:
            threat_factors.append(f"High engagement performance: {avg_engagement:.1f}")
        
        # Consistency factors
        consistency = competitor_data.get("engagement_consistency", 0)
        if consistency > 0.8:
            threat_factors.append(f"High content consistency: {consistency:.1%}")
        
        # Growth factors
        trend = competitor_data.get("engagement_trend", "stable")
        if trend == "improving":
            threat_factors.append("Improving engagement trend")
        
        # Activity factors
        activity_trend = pattern_data.get("activity_trend", "stable")
        if activity_trend == "increasing":
            threat_factors.append("Increasing content activity")
        
        # Posting consistency
        posting_consistency = pattern_data.get("consistency_score", 0)
        if posting_consistency > 0.7:
            threat_factors.append(f"Consistent posting schedule: {posting_consistency:.1%}")
        
        # Tier-based factors
        if competitor_tier == CompetitorTier.DIRECT:
            threat_factors.append("Direct competitor in same market")
        
        return threat_factors
    
    async def generate_competitive_intelligence_report(
        self,
        strategic_insights: List[CompetitorInsight],
        market_analysis: MarketAnalysis,
        competitors: List[Competitor],
        trends: List[Trend],
        content_gaps: List[ContentGap]
    ) -> CompetitorIntelligenceReport:
        """Generate comprehensive competitive intelligence report."""
        
        # Prepare report data
        analysis_period = {
            "start_date": (datetime.utcnow() - timedelta(days=90)).isoformat(),
            "end_date": datetime.utcnow().isoformat()
        }
        
        competitors_analyzed = [
            {
                "id": comp.id,
                "name": comp.name,
                "tier": comp.tier.value,
                "industry": comp.industry.value
            }
            for comp in competitors
        ]
        
        key_insights = [
            {
                "type": insight.insight_type,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence_score,
                "impact": insight.impact_level
            }
            for insight in strategic_insights[:10]  # Top 10 insights
        ]
        
        trending_topics = [
            {
                "topic": trend.topic,
                "strength": trend.strength.value,
                "growth_rate": trend.growth_rate,
                "opportunity_score": trend.opportunity_score
            }
            for trend in trends[:10]  # Top 10 trends
        ]
        
        content_gaps_data = [
            {
                "topic": gap.topic,
                "opportunity_score": gap.opportunity_score,
                "potential_reach": gap.potential_reach,
                "difficulty_score": gap.difficulty_score
            }
            for gap in content_gaps[:10]  # Top 10 gaps
        ]
        
        # Extract performance benchmarks from market analysis
        performance_benchmarks = {
            "content_velocity": market_analysis.content_velocity,
            "average_quality": market_analysis.quality_trends.get("average_quality", 0),
            "top_performing_content_types": list(market_analysis.content_type_distribution.keys())[:3],
            "trending_keywords": market_analysis.trending_keywords[:10]
        }
        
        # Generate strategic recommendations
        recommendations = [
            insight.recommendations[0] if insight.recommendations else insight.description
            for insight in strategic_insights
            if insight.insight_type == "strategic_recommendation"
        ][:5]
        
        # Market overview
        market_overview = {
            "total_content_analyzed": market_analysis.total_content_analyzed,
            "analysis_period_days": (market_analysis.analysis_period[1] - market_analysis.analysis_period[0]).days,
            "competitive_intensity": "high",  # Would be calculated from analysis
            "market_trends": len(trends),
            "opportunities_identified": len(content_gaps)
        }
        
        return CompetitorIntelligenceReport(
            report_id=f"ci_report_{int(datetime.utcnow().timestamp())}",
            generated_at=datetime.utcnow(),
            analysis_period=analysis_period,
            competitors_analyzed=competitors_analyzed,
            key_insights=key_insights,
            trending_topics=trending_topics,
            content_gaps=content_gaps_data,
            performance_benchmarks=performance_benchmarks,
            recommendations=recommendations,
            market_overview=market_overview
        )
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the strategic insights agent's main functionality.
        Routes to appropriate analysis method based on input.
        """
        return {
            "status": "ready",
            "agent_type": "strategic_insights",
            "available_operations": [
                "generate_strategic_insights",
                "generate_competitive_intelligence_report"
            ]
        }