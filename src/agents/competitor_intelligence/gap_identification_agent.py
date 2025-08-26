"""
Gap Identification Agent for discovering content gaps and market opportunities.
Analyzes competitor content coverage to identify underserved topics and content types.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import asdict
import json
from itertools import combinations

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.base_agent import BaseAgent
from .models import (
    ContentGap, ContentItem, ContentType, Platform, Industry,
    Competitor, Trend, GapAnalysisRequest
)
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class GapIdentificationAgent(BaseAgent):
    """
    Specialized agent for identifying content gaps and market opportunities.
    Analyzes competitor content patterns to find underserved topics and niches.
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..core.base_agent import AgentMetadata, AgentType
        
        metadata = AgentMetadata(
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            name="GapIdentificationAgent"
        )
        super().__init__(metadata)
        
        # Initialize AI for gap analysis (lazy loading to avoid requiring API keys at startup)
        self.analysis_llm = None
        
        # Gap analysis configuration
        self.gap_config = {
            "minimum_opportunity_score": 60.0,
            "minimum_potential_reach": 100,
            "content_coverage_threshold": 0.3,  # 30% coverage considered saturated
            "platform_diversity_weight": 0.25,
            "competitor_analysis_depth": 10,  # Analyze top 10 competitors
            "topic_similarity_threshold": 0.7
        }
        
        # Industry-specific content frameworks
        self.content_frameworks = {
            Industry.FINTECH: {
                "core_topics": [
                    "digital payments", "financial literacy", "regulatory compliance",
                    "cybersecurity", "investment strategies", "lending solutions",
                    "mobile banking", "cryptocurrency", "financial planning"
                ],
                "content_types": [
                    ContentType.BLOG_POST, ContentType.WHITEPAPER, 
                    ContentType.CASE_STUDY, ContentType.WEBINAR
                ],
                "key_platforms": [Platform.WEBSITE, Platform.LINKEDIN, Platform.MEDIUM]
            },
            Industry.SAAS: {
                "core_topics": [
                    "software integration", "user experience", "scalability",
                    "api documentation", "customer success", "product updates",
                    "automation workflows", "data analytics", "security"
                ],
                "content_types": [
                    ContentType.BLOG_POST, ContentType.CASE_STUDY,
                    ContentType.VIDEO, ContentType.WEBINAR
                ],
                "key_platforms": [Platform.WEBSITE, Platform.YOUTUBE, Platform.LINKEDIN]
            },
            Industry.MARKETING: {
                "core_topics": [
                    "content strategy", "social media marketing", "seo optimization",
                    "marketing automation", "lead generation", "brand building",
                    "customer acquisition", "retention strategies", "analytics"
                ],
                "content_types": [
                    ContentType.BLOG_POST, ContentType.SOCIAL_MEDIA_POST,
                    ContentType.VIDEO, ContentType.PODCAST
                ],
                "key_platforms": [
                    Platform.WEBSITE, Platform.LINKEDIN, Platform.TWITTER,
                    Platform.INSTAGRAM, Platform.YOUTUBE
                ]
            }
        }
        
        # Cache for analysis results
        self.analysis_cache = {}
    
    def _get_analysis_llm(self):
        """Lazy initialize the analysis LLM."""
        if self.analysis_llm is None:
            try:
                self.analysis_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=2000
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI LLM: {e}")
                return None
        return self.analysis_llm
    
    async def identify_content_gaps(
        self,
        request: GapAnalysisRequest,
        competitor_content: Dict[str, List[ContentItem]],
        market_trends: List[Trend]
    ) -> List[ContentGap]:
        """
        Identify content gaps and opportunities based on competitor analysis.
        Returns ranked list of content gaps with opportunity scoring.
        """
        
        async with async_performance_tracker(f"identify_gaps_{request.industry}"):
            self.logger.info(f"Analyzing content gaps for {len(competitor_content)} competitors")
            
            # Analyze topic coverage across competitors
            topic_coverage = await self._analyze_topic_coverage(
                competitor_content, 
                request.your_content_topics,
                request.industry
            )
            
            # Analyze content type gaps
            content_type_gaps = await self._analyze_content_type_gaps(
                competitor_content,
                request.content_types or list(ContentType),
                request.industry
            )
            
            # Analyze platform distribution gaps
            platform_gaps = await self._analyze_platform_gaps(
                competitor_content,
                request.industry
            )
            
            # Identify trending topic opportunities
            trend_opportunities = await self._identify_trend_opportunities(
                market_trends,
                topic_coverage,
                request.industry
            )
            
            # Synthesize gaps into ContentGap objects
            content_gaps = await self._synthesize_content_gaps(
                topic_coverage,
                content_type_gaps,
                platform_gaps,
                trend_opportunities,
                request
            )
            
            # Score and rank gaps
            ranked_gaps = await self._rank_content_gaps(content_gaps, competitor_content)
            
            # Filter by minimum thresholds
            filtered_gaps = [
                gap for gap in ranked_gaps
                if gap.opportunity_score >= self.gap_config["minimum_opportunity_score"]
                and gap.potential_reach >= self.gap_config["minimum_potential_reach"]
            ]
            
            # Track metrics
            metrics.increment_counter(
                "content_gaps.identified",
                tags={
                    "industry": request.industry.value,
                    "competitors_analyzed": str(len(competitor_content)),
                    "gaps_found": str(len(filtered_gaps))
                }
            )
            
            self.logger.info(f"Identified {len(filtered_gaps)} high-opportunity content gaps")
            
            return filtered_gaps[:25]  # Return top 25 gaps
    
    async def _analyze_topic_coverage(
        self,
        competitor_content: Dict[str, List[ContentItem]],
        your_topics: List[str],
        industry: Industry
    ) -> Dict[str, Any]:
        """Analyze how well competitors cover different topics."""
        
        # Extract all topics from competitor content
        all_competitor_topics = []
        topic_by_competitor = defaultdict(set)
        
        for competitor_id, content_items in competitor_content.items():
            competitor_topics = set()
            
            for item in content_items:
                # Extract topics from keywords and content
                item_topics = await self._extract_content_topics(item)
                competitor_topics.update(item_topics)
                all_competitor_topics.extend(item_topics)
            
            topic_by_competitor[competitor_id] = competitor_topics
        
        # Analyze topic frequency and coverage
        topic_frequency = Counter(all_competitor_topics)
        total_competitors = len(competitor_content)
        
        # Get industry framework topics
        framework = self.content_frameworks.get(industry, {})
        core_topics = framework.get("core_topics", [])
        
        # Analyze coverage for each topic
        topic_analysis = {}
        
        # Analyze core industry topics
        for topic in core_topics:
            related_mentions = [
                freq for t, freq in topic_frequency.items()
                if self._topics_are_related(topic, t)
            ]
            
            total_mentions = sum(related_mentions)
            competitor_coverage = len([
                comp_id for comp_id, topics in topic_by_competitor.items()
                if any(self._topics_are_related(topic, t) for t in topics)
            ])
            coverage_ratio = competitor_coverage / total_competitors
            
            topic_analysis[topic] = {
                "total_mentions": total_mentions,
                "competitor_coverage": competitor_coverage,
                "coverage_ratio": coverage_ratio,
                "is_saturated": coverage_ratio > self.gap_config["content_coverage_threshold"],
                "your_coverage": topic.lower() in [t.lower() for t in your_topics],
                "gap_score": self._calculate_topic_gap_score(
                    coverage_ratio, total_mentions, topic in your_topics
                )
            }
        
        # Analyze trending/emerging topics from competitor content
        emerging_topics = [
            topic for topic, freq in topic_frequency.most_common(50)
            if topic not in core_topics and freq >= 3
        ]
        
        for topic in emerging_topics:
            competitor_coverage = len([
                comp_id for comp_id, topics in topic_by_competitor.items()
                if topic in topics
            ])
            coverage_ratio = competitor_coverage / total_competitors
            
            topic_analysis[topic] = {
                "total_mentions": topic_frequency[topic],
                "competitor_coverage": competitor_coverage,
                "coverage_ratio": coverage_ratio,
                "is_saturated": coverage_ratio > self.gap_config["content_coverage_threshold"],
                "your_coverage": topic.lower() in [t.lower() for t in your_topics],
                "is_emerging": True,
                "gap_score": self._calculate_topic_gap_score(
                    coverage_ratio, topic_frequency[topic], topic.lower() in your_topics
                )
            }
        
        return {
            "topic_analysis": topic_analysis,
            "total_topics_analyzed": len(topic_analysis),
            "saturated_topics": [
                t for t, data in topic_analysis.items() 
                if data["is_saturated"]
            ],
            "opportunity_topics": [
                t for t, data in topic_analysis.items()
                if not data["is_saturated"] and data["gap_score"] > 70
            ],
            "your_coverage_gaps": [
                t for t, data in topic_analysis.items()
                if not data["your_coverage"] and data["gap_score"] > 60
            ]
        }
    
    async def _analyze_content_type_gaps(
        self,
        competitor_content: Dict[str, List[ContentItem]],
        target_content_types: List[ContentType],
        industry: Industry
    ) -> Dict[ContentType, Any]:
        """Analyze gaps in content type coverage."""
        
        # Count content types by competitor
        type_by_competitor = defaultdict(lambda: defaultdict(int))
        
        for competitor_id, content_items in competitor_content.items():
            for item in content_items:
                type_by_competitor[competitor_id][item.content_type] += 1
        
        # Analyze each content type
        content_type_analysis = {}
        total_competitors = len(competitor_content)
        
        for content_type in target_content_types:
            competitors_using = len([
                comp_id for comp_id, types in type_by_competitor.items()
                if content_type in types
            ])
            
            total_content = sum([
                types[content_type] 
                for types in type_by_competitor.values()
            ])
            
            coverage_ratio = competitors_using / total_competitors
            avg_content_per_competitor = total_content / max(competitors_using, 1)
            
            # Get industry framework expectations
            framework = self.content_frameworks.get(industry, {})
            expected_types = framework.get("content_types", [])
            is_expected = content_type in expected_types
            
            content_type_analysis[content_type] = {
                "competitors_using": competitors_using,
                "coverage_ratio": coverage_ratio,
                "total_content": total_content,
                "avg_content_per_competitor": avg_content_per_competitor,
                "is_expected_for_industry": is_expected,
                "is_underutilized": coverage_ratio < 0.4,  # Less than 40% coverage
                "opportunity_score": self._calculate_content_type_opportunity(
                    coverage_ratio, is_expected, total_content
                )
            }
        
        return content_type_analysis
    
    async def _analyze_platform_gaps(
        self,
        competitor_content: Dict[str, List[ContentItem]],
        industry: Industry
    ) -> Dict[Platform, Any]:
        """Analyze platform distribution and identify gaps."""
        
        # Count platform usage by competitor
        platform_by_competitor = defaultdict(lambda: defaultdict(int))
        
        for competitor_id, content_items in competitor_content.items():
            for item in content_items:
                platform_by_competitor[competitor_id][item.platform] += 1
        
        # Analyze each platform
        platform_analysis = {}
        total_competitors = len(competitor_content)
        
        for platform in Platform:
            competitors_using = len([
                comp_id for comp_id, platforms in platform_by_competitor.items()
                if platform in platforms
            ])
            
            total_content = sum([
                platforms[platform] 
                for platforms in platform_by_competitor.values()
            ])
            
            coverage_ratio = competitors_using / total_competitors
            
            # Get industry framework expectations
            framework = self.content_frameworks.get(industry, {})
            key_platforms = framework.get("key_platforms", [])
            is_key_platform = platform in key_platforms
            
            platform_analysis[platform] = {
                "competitors_using": competitors_using,
                "coverage_ratio": coverage_ratio,
                "total_content": total_content,
                "is_key_for_industry": is_key_platform,
                "is_underutilized": coverage_ratio < 0.3,  # Less than 30% coverage
                "opportunity_score": self._calculate_platform_opportunity(
                    coverage_ratio, is_key_platform, total_content
                )
            }
        
        return platform_analysis
    
    async def _identify_trend_opportunities(
        self,
        market_trends: List[Trend],
        topic_coverage: Dict[str, Any],
        industry: Industry
    ) -> List[Dict[str, Any]]:
        """Identify opportunities based on market trends and topic gaps."""
        
        trend_opportunities = []
        topic_analysis = topic_coverage.get("topic_analysis", {})
        
        for trend in market_trends:
            # Check if trend topics have coverage gaps
            trend_gap_score = 0
            related_topics = []
            
            for keyword in trend.keywords:
                # Find related topics in our analysis
                for analyzed_topic, data in topic_analysis.items():
                    if self._topics_are_related(keyword, analyzed_topic):
                        related_topics.append(analyzed_topic)
                        trend_gap_score += data["gap_score"]
            
            if related_topics:
                avg_gap_score = trend_gap_score / len(related_topics)
            else:
                # New trending topic not in our analysis - high opportunity
                avg_gap_score = 85
            
            # Calculate opportunity score
            opportunity_score = min(
                avg_gap_score * 0.7 +  # Gap contribution
                (trend.opportunity_score or 50) * 0.3,  # Trend strength contribution
                100
            )
            
            if opportunity_score > 60:
                trend_opportunities.append({
                    "trend": trend,
                    "related_topics": related_topics,
                    "gap_score": avg_gap_score,
                    "opportunity_score": opportunity_score,
                    "recommended_content_types": self._recommend_content_types_for_trend(
                        trend, industry
                    ),
                    "recommended_platforms": self._recommend_platforms_for_trend(
                        trend, industry
                    )
                })
        
        # Sort by opportunity score
        trend_opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        return trend_opportunities[:15]  # Top 15 opportunities
    
    async def _synthesize_content_gaps(
        self,
        topic_coverage: Dict[str, Any],
        content_type_gaps: Dict[ContentType, Any],
        platform_gaps: Dict[Platform, Any],
        trend_opportunities: List[Dict[str, Any]],
        request: GapAnalysisRequest
    ) -> List[ContentGap]:
        """Synthesize different analyses into ContentGap objects."""
        
        content_gaps = []
        
        # Create gaps from topic analysis
        topic_analysis = topic_coverage.get("topic_analysis", {})
        
        for topic, data in topic_analysis.items():
            if data["gap_score"] > 60 and not data["your_coverage"]:
                
                # Determine missing content types for this topic
                missing_content_types = []
                for content_type, type_data in content_type_gaps.items():
                    if type_data["is_underutilized"]:
                        missing_content_types.append(content_type)
                
                # Determine missing platforms
                missing_platforms = []
                for platform, platform_data in platform_gaps.items():
                    if platform_data["is_underutilized"]:
                        missing_platforms.append(platform)
                
                # Calculate potential reach
                potential_reach = self._estimate_potential_reach(
                    data["gap_score"],
                    len(missing_content_types),
                    len(missing_platforms)
                )
                
                # Generate approach suggestion
                suggested_approach = await self._generate_content_approach(
                    topic,
                    missing_content_types[:3],
                    request.industry
                )
                
                content_gap = ContentGap(
                    id=f"topic_gap_{hash(topic)}",
                    topic=topic,
                    description=f"Underserved topic with {data['competitor_coverage']}/{len(request.competitors)} competitor coverage",
                    opportunity_score=data["gap_score"],
                    difficulty_score=self._calculate_difficulty_score(data),
                    potential_reach=potential_reach,
                    content_types_missing=missing_content_types[:5],
                    platforms_missing=missing_platforms[:5],
                    keywords=self._generate_topic_keywords(topic),
                    competitors_covering=[],  # Would need competitor mapping
                    suggested_approach=suggested_approach,
                    metadata={
                        "source": "topic_analysis",
                        "competitor_coverage": data["competitor_coverage"],
                        "coverage_ratio": data["coverage_ratio"],
                        "total_mentions": data["total_mentions"]
                    }
                )
                
                content_gaps.append(content_gap)
        
        # Create gaps from trend opportunities
        for opportunity in trend_opportunities:
            trend = opportunity["trend"]
            
            content_gap = ContentGap(
                id=f"trend_gap_{trend.id}",
                topic=trend.topic,
                description=f"Trending topic opportunity with {trend.strength.value} strength",
                opportunity_score=opportunity["opportunity_score"],
                difficulty_score=self._calculate_trend_difficulty(trend),
                potential_reach=self._estimate_trend_reach(trend),
                content_types_missing=opportunity["recommended_content_types"],
                platforms_missing=opportunity["recommended_platforms"],
                keywords=trend.keywords[:10],
                competitors_covering=[],  # Would need mapping
                suggested_approach=await self._generate_trend_approach(
                    trend,
                    request.industry
                ),
                metadata={
                    "source": "trend_analysis",
                    "trend_strength": trend.strength.value,
                    "trend_growth_rate": trend.growth_rate,
                    "related_topics": opportunity["related_topics"]
                }
            )
            
            content_gaps.append(content_gap)
        
        # Create gaps from content type analysis
        for content_type, data in content_type_gaps.items():
            if data["is_underutilized"] and data["opportunity_score"] > 70:
                
                content_gap = ContentGap(
                    id=f"content_type_gap_{content_type.value}",
                    topic=f"{content_type.value.replace('_', ' ').title()} Content Opportunity",
                    description=f"Underutilized content type with only {data['coverage_ratio']:.1%} competitor adoption",
                    opportunity_score=data["opportunity_score"],
                    difficulty_score=50,  # Moderate difficulty for content type gaps
                    potential_reach=self._estimate_content_type_reach(data),
                    content_types_missing=[content_type],
                    platforms_missing=[],
                    keywords=self._generate_content_type_keywords(content_type, request.industry),
                    competitors_covering=[],
                    suggested_approach=await self._generate_content_type_approach(
                        content_type,
                        request.industry
                    ),
                    metadata={
                        "source": "content_type_analysis",
                        "coverage_ratio": data["coverage_ratio"],
                        "competitors_using": data["competitors_using"],
                        "is_expected_for_industry": data["is_expected_for_industry"]
                    }
                )
                
                content_gaps.append(content_gap)
        
        return content_gaps
    
    async def _rank_content_gaps(
        self,
        content_gaps: List[ContentGap],
        competitor_content: Dict[str, List[ContentItem]]
    ) -> List[ContentGap]:
        """Rank content gaps by strategic value and feasibility."""
        
        for gap in content_gaps:
            # Calculate composite ranking score
            ranking_score = (
                gap.opportunity_score * 0.4 +  # Opportunity weight
                (100 - gap.difficulty_score) * 0.3 +  # Feasibility weight  
                min(gap.potential_reach / 1000, 50) * 0.2 +  # Reach weight
                len(gap.keywords) * 2 * 0.1  # Keyword diversity weight
            )
            
            gap.metadata["ranking_score"] = ranking_score
        
        # Sort by ranking score
        return sorted(
            content_gaps,
            key=lambda g: g.metadata.get("ranking_score", 0),
            reverse=True
        )
    
    # Helper methods
    
    async def _extract_content_topics(self, content_item: ContentItem) -> List[str]:
        """Extract topics from a content item."""
        
        # Use existing keywords if available
        if content_item.keywords:
            return content_item.keywords[:10]
        
        # Simple keyword extraction from title and content
        text = f"{content_item.title} {content_item.content[:500]}"
        
        try:
            # Use AI to extract topics
            prompt = f"""
            Extract 5-8 main topics/themes from this content. Return as comma-separated list.
            
            Content: {text}
            
            Topics:
            """
            
            llm = self._get_analysis_llm()
            if llm is None:
                # Fallback: simple keyword extraction
                import re
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
                return list(set(words))[:8]
            
            response = await llm.agenerate([
                [HumanMessage(content=prompt)]
            ])
            
            topics_text = response.generations[0][0].text.strip()
            topics = [
                topic.strip().lower() 
                for topic in topics_text.split(',')
                if topic.strip()
            ]
            
            return topics[:8]
            
        except Exception as e:
            self.logger.debug(f"Failed to extract topics: {str(e)}")
            
            # Fallback: simple keyword extraction
            import re
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            return list(set(words))[:8]
    
    def _topics_are_related(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are related."""
        
        topic1_words = set(topic1.lower().split())
        topic2_words = set(topic2.lower().split())
        
        # Check for word overlap
        overlap = len(topic1_words & topic2_words)
        total_words = len(topic1_words | topic2_words)
        
        similarity = overlap / max(total_words, 1)
        
        return similarity > 0.3 or topic1.lower() in topic2.lower() or topic2.lower() in topic1.lower()
    
    def _calculate_topic_gap_score(
        self,
        coverage_ratio: float,
        total_mentions: int,
        your_coverage: bool
    ) -> float:
        """Calculate gap score for a topic."""
        
        # Lower coverage = higher gap score
        coverage_gap_score = (1 - coverage_ratio) * 70
        
        # Higher mentions = higher opportunity
        mention_score = min(total_mentions * 2, 25)
        
        # Penalty if you already cover it
        coverage_penalty = 20 if your_coverage else 0
        
        return max(0, coverage_gap_score + mention_score - coverage_penalty)
    
    def _calculate_content_type_opportunity(
        self,
        coverage_ratio: float,
        is_expected: bool,
        total_content: int
    ) -> float:
        """Calculate opportunity score for a content type."""
        
        base_score = (1 - coverage_ratio) * 60
        
        # Bonus for industry-expected content types
        industry_bonus = 20 if is_expected else 0
        
        # Bonus for proven content types (high total content)
        proven_bonus = min(total_content / 10, 15)
        
        return min(base_score + industry_bonus + proven_bonus, 100)
    
    def _calculate_platform_opportunity(
        self,
        coverage_ratio: float,
        is_key_platform: bool,
        total_content: int
    ) -> float:
        """Calculate opportunity score for a platform."""
        
        base_score = (1 - coverage_ratio) * 50
        
        # Major bonus for key industry platforms
        key_platform_bonus = 30 if is_key_platform else 10
        
        # Activity bonus
        activity_bonus = min(total_content / 20, 15)
        
        return min(base_score + key_platform_bonus + activity_bonus, 100)
    
    def _calculate_difficulty_score(self, topic_data: Dict[str, Any]) -> float:
        """Calculate difficulty score for creating content about a topic."""
        
        base_difficulty = 40  # Base difficulty
        
        # Higher competitor coverage = higher difficulty
        competition_difficulty = topic_data["coverage_ratio"] * 30
        
        # More mentions = potentially more competitive
        volume_difficulty = min(topic_data["total_mentions"] / 20, 20)
        
        return min(base_difficulty + competition_difficulty + volume_difficulty, 90)
    
    def _calculate_trend_difficulty(self, trend: Trend) -> float:
        """Calculate difficulty score for trend-based content."""
        
        strength_difficulty = {
            TrendStrength.WEAK: 30,
            TrendStrength.MODERATE: 45,
            TrendStrength.STRONG: 65,
            TrendStrength.VIRAL: 80
        }
        
        return strength_difficulty.get(trend.strength, 50)
    
    def _estimate_potential_reach(
        self,
        gap_score: float,
        content_types_count: int,
        platforms_count: int
    ) -> int:
        """Estimate potential reach for a content gap."""
        
        base_reach = gap_score * 10  # Base reach from gap score
        
        # Multipliers for content type and platform diversity
        type_multiplier = 1 + (content_types_count * 0.2)
        platform_multiplier = 1 + (platforms_count * 0.15)
        
        return int(base_reach * type_multiplier * platform_multiplier)
    
    def _estimate_trend_reach(self, trend: Trend) -> int:
        """Estimate potential reach for trend-based content."""
        
        base_reach = (trend.opportunity_score or 50) * 20
        
        strength_multipliers = {
            TrendStrength.WEAK: 1.0,
            TrendStrength.MODERATE: 1.5,
            TrendStrength.STRONG: 2.0,
            TrendStrength.VIRAL: 3.0
        }
        
        multiplier = strength_multipliers.get(trend.strength, 1.0)
        
        return int(base_reach * multiplier)
    
    def _estimate_content_type_reach(self, type_data: Dict[str, Any]) -> int:
        """Estimate potential reach for content type gaps."""
        
        return int(type_data["opportunity_score"] * 15)
    
    def _recommend_content_types_for_trend(
        self,
        trend: Trend,
        industry: Industry
    ) -> List[ContentType]:
        """Recommend content types for a trending topic."""
        
        # Get industry framework
        framework = self.content_frameworks.get(industry, {})
        expected_types = framework.get("content_types", list(ContentType))
        
        # Recommend based on trend strength
        if trend.strength in [TrendStrength.VIRAL, TrendStrength.STRONG]:
            return [
                ContentType.SOCIAL_MEDIA_POST,
                ContentType.BLOG_POST,
                ContentType.VIDEO
            ]
        else:
            return [
                ContentType.BLOG_POST,
                ContentType.WHITEPAPER,
                ContentType.WEBINAR
            ]
    
    def _recommend_platforms_for_trend(
        self,
        trend: Trend,
        industry: Industry
    ) -> List[Platform]:
        """Recommend platforms for a trending topic."""
        
        framework = self.content_frameworks.get(industry, {})
        key_platforms = framework.get("key_platforms", [Platform.WEBSITE, Platform.LINKEDIN])
        
        # Add social platforms for viral trends
        if trend.strength == TrendStrength.VIRAL:
            return key_platforms + [Platform.TWITTER, Platform.INSTAGRAM]
        
        return key_platforms
    
    def _generate_topic_keywords(self, topic: str) -> List[str]:
        """Generate related keywords for a topic."""
        
        # Simple keyword generation (can be enhanced with NLP)
        base_keywords = [topic]
        
        # Add variations
        words = topic.split()
        if len(words) > 1:
            base_keywords.extend(words)
        
        # Add common prefixes/suffixes
        base_keywords.extend([
            f"{topic} strategy",
            f"{topic} guide",
            f"{topic} tips",
            f"{topic} best practices"
        ])
        
        return base_keywords[:10]
    
    def _generate_content_type_keywords(
        self,
        content_type: ContentType,
        industry: Industry
    ) -> List[str]:
        """Generate keywords for content type opportunities."""
        
        type_name = content_type.value.replace('_', ' ')
        
        keywords = [
            type_name,
            f"{industry.value} {type_name}",
            f"how to create {type_name}",
            f"{type_name} strategy",
            f"best {type_name} examples"
        ]
        
        return keywords
    
    async def _generate_content_approach(
        self,
        topic: str,
        content_types: List[ContentType],
        industry: Industry
    ) -> str:
        """Generate suggested approach for creating content about a topic."""
        
        try:
            types_str = ", ".join([ct.value.replace('_', ' ') for ct in content_types])
            
            prompt = f"""
            Generate a brief content strategy recommendation for the topic "{topic}" in the {industry.value} industry.
            Focus on creating {types_str} content.
            Keep it practical and actionable in 2-3 sentences.
            
            Strategy:
            """
            
            llm = self._get_analysis_llm()
            if llm is None:
                return f"Create comprehensive {content_types[0].value.replace('_', ' ')} content covering {topic} fundamentals and best practices."
            
            response = await llm.agenerate([
                [HumanMessage(content=prompt)]
            ])
            
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            self.logger.debug(f"Failed to generate approach: {str(e)}")
            return f"Create comprehensive {content_types[0].value.replace('_', ' ')} content covering {topic} fundamentals and best practices."
    
    async def _generate_trend_approach(
        self,
        trend: Trend,
        industry: Industry
    ) -> str:
        """Generate approach for trend-based content."""
        
        try:
            prompt = f"""
            Generate a content strategy for capitalizing on the trending topic "{trend.topic}" in {industry.value}.
            The trend has {trend.strength.value} strength and {trend.growth_rate:.1%} growth rate.
            Keep it actionable in 2-3 sentences.
            
            Strategy:
            """
            
            llm = self._get_analysis_llm()
            if llm is None:
                return f"Create timely content about {trend.topic} to capitalize on its {trend.strength.value} momentum."
            
            response = await llm.agenerate([
                [HumanMessage(content=prompt)]
            ])
            
            return response.generations[0][0].text.strip ()
            
        except Exception as e:
            self.logger.debug(f"Failed to generate trend approach: {str(e)}")
            return f"Create timely content about {trend.topic} to capitalize on its {trend.strength.value} momentum."
    
    async def _generate_content_type_approach(
        self,
        content_type: ContentType,
        industry: Industry
    ) -> str:
        """Generate approach for content type opportunities."""
        
        type_name = content_type.value.replace('_', ' ')
        
        return f"Develop high-quality {type_name} content to differentiate from competitors who are underutilizing this format in {industry.value}."
    
    async def perform_competitive_positioning_analysis(
        self,
        content_gaps: List[ContentGap],
        competitor_content: Dict[str, List[ContentItem]]
    ) -> Dict[str, Any]:
        """Perform competitive positioning analysis based on identified gaps."""
        
        positioning_analysis = {
            "content_differentiation_opportunities": [],
            "blue_ocean_topics": [],
            "competitive_advantages": [],
            "strategic_recommendations": []
        }
        
        # Identify blue ocean opportunities (topics with minimal competition)
        blue_ocean_gaps = [
            gap for gap in content_gaps[:10]
            if gap.metadata.get("coverage_ratio", 1.0) < 0.2  # Less than 20% coverage
        ]
        
        positioning_analysis["blue_ocean_topics"] = [
            {
                "topic": gap.topic,
                "opportunity_score": gap.opportunity_score,
                "potential_reach": gap.potential_reach,
                "competition_level": "minimal"
            }
            for gap in blue_ocean_gaps
        ]
        
        # Identify differentiation opportunities
        differentiation_gaps = [
            gap for gap in content_gaps[:15]
            if gap.opportunity_score > 75 and len(gap.content_types_missing) > 2
        ]
        
        positioning_analysis["content_differentiation_opportunities"] = [
            {
                "topic": gap.topic,
                "unique_angles": gap.content_types_missing,
                "platforms": gap.platforms_missing,
                "suggested_approach": gap.suggested_approach
            }
            for gap in differentiation_gaps
        ]
        
        # Generate strategic recommendations
        recommendations = []
        
        if blue_ocean_gaps:
            recommendations.append(
                f"Focus on {len(blue_ocean_gaps)} blue ocean topics with minimal competition"
            )
        
        if differentiation_gaps:
            recommendations.append(
                f"Leverage {len(differentiation_gaps)} differentiation opportunities through unique content formats"
            )
        
        high_reach_gaps = [gap for gap in content_gaps if gap.potential_reach > 500]
        if high_reach_gaps:
            recommendations.append(
                f"Prioritize {len(high_reach_gaps)} high-reach opportunities for maximum impact"
            )
        
        positioning_analysis["strategic_recommendations"] = recommendations
        
        return positioning_analysis
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the gap identification agent's main functionality.
        Routes to appropriate analysis method based on input.
        """
        return {
            "status": "ready",
            "agent_type": "gap_identification",
            "available_operations": [
                "identify_content_gaps",
                "analyze_competitive_positioning"
            ]
        }