"""
Performance Analysis Agent for competitive benchmarking and performance insights.
Analyzes competitor content performance, engagement patterns, and strategic positioning.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import asdict
import statistics
from scipy import stats

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..core.base_agent import BaseAgent
from .models import (
    ContentItem, Competitor, CompetitorInsight, MarketAnalysis,
    Platform, ContentType, Industry, CompetitorTier
)
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class PerformanceAnalysisAgent(BaseAgent):
    """
    Specialized agent for analyzing competitor content performance and generating benchmarks.
    Provides competitive intelligence through performance pattern analysis.
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..core.base_agent import AgentMetadata, AgentType
        
        metadata = AgentMetadata(
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            name="PerformanceAnalysisAgent"
        )
        super().__init__(metadata)
        
        # Initialize AI for insights generation (lazy loading to avoid requiring API keys at startup)
        self.analysis_llm = None
        
        # Performance analysis configuration
        self.analysis_config = {
            "benchmark_window_days": 90,
            "minimum_content_for_analysis": 10,
            "outlier_threshold": 2.0,  # Standard deviations
            "trend_detection_periods": 4,  # Number of time periods for trend analysis
            "confidence_threshold": 0.8,
            "performance_metrics": [
                "engagement_rate", "reach", "clicks", "shares", 
                "comments", "likes", "conversion_rate"
            ]
        }
        
        # Performance scoring weights
        self.performance_weights = {
            "likes": 1.0,
            "shares": 3.0,      # Shares are more valuable
            "comments": 2.0,    # Comments show deeper engagement
            "clicks": 2.5,      # Clicks indicate intent
            "views": 0.1,       # Views are less valuable but show reach
            "saves": 2.5,       # Saves indicate value
            "mentions": 1.5     # Mentions show amplification
        }
        
        # Platform-specific benchmarks (industry averages)
        self.platform_benchmarks = {
            Platform.LINKEDIN: {
                "engagement_rate": 0.054,  # 5.4% average
                "click_rate": 0.009,       # 0.9% average
                "share_rate": 0.003        # 0.3% average
            },
            Platform.TWITTER: {
                "engagement_rate": 0.045,  # 4.5% average
                "retweet_rate": 0.012,     # 1.2% average
                "click_rate": 0.008        # 0.8% average
            },
            Platform.INSTAGRAM: {
                "engagement_rate": 0.067,  # 6.7% average
                "save_rate": 0.004,        # 0.4% average
                "story_completion": 0.73   # 73% average
            },
            Platform.WEBSITE: {
                "bounce_rate": 0.43,       # 43% average
                "time_on_page": 150,       # 2.5 minutes average
                "conversion_rate": 0.025   # 2.5% average
            }
        }
    
    def _get_analysis_llm(self):
        """Lazy initialize the analysis LLM."""
        if self.analysis_llm is None:
            try:
                self.analysis_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.2,
                    max_tokens=1500
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI LLM: {e}")
                return None
        return self.analysis_llm
    
    async def analyze_competitor_performance(
        self,
        competitors: List[Competitor],
        content_data: Dict[str, List[ContentItem]],
        analysis_period_days: int = 90
    ) -> Dict[str, Any]:
        """
        Comprehensive performance analysis of competitors.
        Returns benchmarks, insights, and competitive positioning.
        """
        
        async with async_performance_tracker("competitor_performance_analysis"):
            self.logger.info(f"Analyzing performance for {len(competitors)} competitors")
            
            # Filter content by analysis period
            cutoff_date = datetime.utcnow() - timedelta(days=analysis_period_days)
            filtered_content = {}
            
            for comp_id, content_items in content_data.items():
                filtered_content[comp_id] = [
                    item for item in content_items
                    if item.published_at >= cutoff_date
                ]
            
            # Perform comprehensive analysis
            engagement_analysis = await self._analyze_engagement_patterns(
                competitors, filtered_content
            )
            
            content_performance = await self._analyze_content_performance(
                competitors, filtered_content
            )
            
            posting_patterns = await self._analyze_posting_patterns(
                competitors, filtered_content
            )
            
            competitive_benchmarks = await self._generate_competitive_benchmarks(
                competitors, filtered_content
            )
            
            performance_insights = await self._generate_performance_insights(
                competitors,
                engagement_analysis,
                content_performance,
                posting_patterns
            )
            
            # Track analysis metrics
            metrics.increment_counter(
                "performance_analysis.completed",
                tags={
                    "competitors_analyzed": str(len(competitors)),
                    "content_items": str(sum(len(content) for content in filtered_content.values())),
                    "insights_generated": str(len(performance_insights))
                }
            )
            
            return {
                "analysis_period": {
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat(),
                    "days": analysis_period_days
                },
                "engagement_analysis": engagement_analysis,
                "content_performance": content_performance,
                "posting_patterns": posting_patterns,
                "competitive_benchmarks": competitive_benchmarks,
                "performance_insights": performance_insights,
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def _analyze_engagement_patterns(
        self,
        competitors: List[Competitor],
        content_data: Dict[str, List[ContentItem]]
    ) -> Dict[str, Any]:
        """Analyze engagement patterns across competitors."""
        
        engagement_analysis = {}
        
        for competitor in competitors:
            content_items = content_data.get(competitor.id, [])
            
            if len(content_items) < self.analysis_config["minimum_content_for_analysis"]:
                continue
            
            # Calculate engagement metrics
            engagement_scores = []
            platform_engagement = defaultdict(list)
            content_type_engagement = defaultdict(list)
            
            for item in content_items:
                # Calculate composite engagement score
                score = self._calculate_engagement_score(item)
                engagement_scores.append(score)
                
                platform_engagement[item.platform].append(score)
                content_type_engagement[item.content_type].append(score)
            
            if not engagement_scores:
                continue
            
            # Calculate statistics
            avg_engagement = statistics.mean(engagement_scores)
            median_engagement = statistics.median(engagement_scores)
            engagement_std = statistics.stdev(engagement_scores) if len(engagement_scores) > 1 else 0
            
            # Identify top performing content
            top_threshold = avg_engagement + engagement_std
            top_performers = [
                item for item, score in zip(content_items, engagement_scores)
                if score >= top_threshold
            ]
            
            # Calculate platform performance
            platform_performance = {}
            for platform, scores in platform_engagement.items():
                if scores:
                    platform_performance[platform.value] = {
                        "avg_engagement": statistics.mean(scores),
                        "content_count": len(scores),
                        "performance_vs_benchmark": self._compare_to_benchmark(
                            platform, statistics.mean(scores)
                        )
                    }
            
            # Calculate content type performance
            content_type_performance = {}
            for content_type, scores in content_type_engagement.items():
                if scores:
                    content_type_performance[content_type.value] = {
                        "avg_engagement": statistics.mean(scores),
                        "content_count": len(scores),
                        "best_performing": max(scores),
                        "consistency": 1 - (statistics.stdev(scores) / statistics.mean(scores))
                        if len(scores) > 1 and statistics.mean(scores) > 0 else 0
                    }
            
            engagement_analysis[competitor.id] = {
                "competitor_name": competitor.name,
                "total_content": len(content_items),
                "avg_engagement": avg_engagement,
                "median_engagement": median_engagement,
                "engagement_consistency": 1 - (engagement_std / avg_engagement) if avg_engagement > 0 else 0,
                "top_performers_count": len(top_performers),
                "top_performer_titles": [item.title for item in top_performers[:5]],
                "platform_performance": platform_performance,
                "content_type_performance": content_type_performance,
                "engagement_trend": self._calculate_engagement_trend(content_items)
            }
        
        return engagement_analysis
    
    async def _analyze_content_performance(
        self,
        competitors: List[Competitor],
        content_data: Dict[str, List[ContentItem]]
    ) -> Dict[str, Any]:
        """Analyze content performance patterns and success factors."""
        
        content_performance = {}
        
        for competitor in competitors:
            content_items = content_data.get(competitor.id, [])
            
            if len(content_items) < self.analysis_config["minimum_content_for_analysis"]:
                continue
            
            # Analyze content characteristics
            performance_by_length = self._analyze_performance_by_length(content_items)
            performance_by_timing = self._analyze_performance_by_timing(content_items)
            keyword_performance = await self._analyze_keyword_performance(content_items)
            viral_content_analysis = self._identify_viral_content(content_items)
            
            # Content quality analysis
            quality_scores = [
                item.quality_score for item in content_items
                if item.quality_score is not None
            ]
            
            quality_analysis = {
                "avg_quality_score": statistics.mean(quality_scores) if quality_scores else None,
                "quality_vs_engagement_correlation": self._calculate_quality_engagement_correlation(
                    content_items
                ),
                "high_quality_content_count": len([
                    score for score in quality_scores if score > 80
                ])
            }
            
            content_performance[competitor.id] = {
                "competitor_name": competitor.name,
                "performance_by_length": performance_by_length,
                "performance_by_timing": performance_by_timing,
                "keyword_performance": keyword_performance,
                "viral_content_analysis": viral_content_analysis,
                "quality_analysis": quality_analysis,
                "content_themes": await self._identify_content_themes(content_items)
            }
        
        return content_performance
    
    async def _analyze_posting_patterns(
        self,
        competitors: List[Competitor],
        content_data: Dict[str, List[ContentItem]]
    ) -> Dict[str, Any]:
        """Analyze posting frequency and timing patterns."""
        
        posting_patterns = {}
        
        for competitor in competitors:
            content_items = content_data.get(competitor.id, [])
            
            if len(content_items) < 5:
                continue
            
            # Analyze posting frequency
            posting_frequency = self._calculate_posting_frequency(content_items)
            
            # Analyze optimal timing
            timing_analysis = self._analyze_optimal_timing(content_items)
            
            # Analyze content mix
            content_mix = self._analyze_content_mix(content_items)
            
            # Analyze posting consistency
            consistency_score = self._calculate_posting_consistency(content_items)
            
            posting_patterns[competitor.id] = {
                "competitor_name": competitor.name,
                "posting_frequency": posting_frequency,
                "timing_analysis": timing_analysis,
                "content_mix": content_mix,
                "consistency_score": consistency_score,
                "activity_trend": self._calculate_activity_trend(content_items)
            }
        
        return posting_patterns
    
    async def _generate_competitive_benchmarks(
        self,
        competitors: List[Competitor],
        content_data: Dict[str, List[ContentItem]]
    ) -> Dict[str, Any]:
        """Generate competitive benchmarks across different metrics."""
        
        # Collect all metrics
        all_engagement_scores = []
        platform_metrics = defaultdict(list)
        content_type_metrics = defaultdict(list)
        tier_metrics = defaultdict(list)
        
        for competitor in competitors:
            content_items = content_data.get(competitor.id, [])
            
            for item in content_items:
                engagement_score = self._calculate_engagement_score(item)
                all_engagement_scores.append(engagement_score)
                
                platform_metrics[item.platform].append(engagement_score)
                content_type_metrics[item.content_type].append(engagement_score)
                tier_metrics[competitor.tier].append(engagement_score)
        
        # Calculate benchmarks
        benchmarks = {
            "overall": {
                "avg_engagement": statistics.mean(all_engagement_scores) if all_engagement_scores else 0,
                "median_engagement": statistics.median(all_engagement_scores) if all_engagement_scores else 0,
                "top_quartile": np.percentile(all_engagement_scores, 75) if all_engagement_scores else 0,
                "bottom_quartile": np.percentile(all_engagement_scores, 25) if all_engagement_scores else 0
            },
            "by_platform": {},
            "by_content_type": {},
            "by_competitor_tier": {}
        }
        
        # Platform benchmarks
        for platform, scores in platform_metrics.items():
            if scores:
                benchmarks["by_platform"][platform.value] = {
                    "avg_engagement": statistics.mean(scores),
                    "median_engagement": statistics.median(scores),
                    "sample_size": len(scores),
                    "vs_industry_benchmark": self._compare_to_benchmark(platform, statistics.mean(scores))
                }
        
        # Content type benchmarks
        for content_type, scores in content_type_metrics.items():
            if scores:
                benchmarks["by_content_type"][content_type.value] = {
                    "avg_engagement": statistics.mean(scores),
                    "median_engagement": statistics.median(scores),
                    "sample_size": len(scores),
                    "performance_rank": self._rank_content_type_performance(content_type, scores)
                }
        
        # Competitor tier benchmarks
        for tier, scores in tier_metrics.items():
            if scores:
                benchmarks["by_competitor_tier"][tier.value] = {
                    "avg_engagement": statistics.mean(scores),
                    "median_engagement": statistics.median(scores),
                    "sample_size": len(scores)
                }
        
        return benchmarks
    
    async def _generate_performance_insights(
        self,
        competitors: List[Competitor],
        engagement_analysis: Dict[str, Any],
        content_performance: Dict[str, Any],
        posting_patterns: Dict[str, Any]
    ) -> List[CompetitorInsight]:
        """Generate strategic insights from performance analysis."""
        
        insights = []
        
        # Top performer analysis
        if engagement_analysis:
            # Find top performer
            top_competitor_id = max(
                engagement_analysis.keys(),
                key=lambda k: engagement_analysis[k]["avg_engagement"]
            )
            
            top_competitor_data = engagement_analysis[top_competitor_id]
            
            insight = CompetitorInsight(
                id=f"top_performer_{top_competitor_id}",
                competitor_id=top_competitor_id,
                insight_type="top_performer_analysis",
                title=f"{top_competitor_data['competitor_name']} Leading in Engagement",
                description=f"Achieving {top_competitor_data['avg_engagement']:.1f} average engagement with {top_competitor_data['engagement_consistency']:.1%} consistency",
                confidence_score=0.9,
                impact_level="high",
                supporting_evidence=[
                    f"Average engagement: {top_competitor_data['avg_engagement']:.1f}",
                    f"Top performers: {top_competitor_data['top_performers_count']} pieces",
                    f"Consistency score: {top_competitor_data['engagement_consistency']:.1%}"
                ],
                recommendations=[
                    "Analyze their top-performing content formats and topics",
                    "Study their posting timing and frequency patterns",
                    "Consider adopting similar content types or platforms"
                ]
            )
            insights.append(insight)
        
        # Content strategy insights
        for competitor_id, performance_data in content_performance.items():
            if "viral_content_analysis" in performance_data:
                viral_analysis = performance_data["viral_content_analysis"]
                
                if viral_analysis["viral_count"] > 0:
                    insight = CompetitorInsight(
                        id=f"viral_strategy_{competitor_id}",
                        competitor_id=competitor_id,
                        insight_type="content_strategy",
                        title=f"{performance_data['competitor_name']} Viral Content Strategy",
                        description=f"Successfully created {viral_analysis['viral_count']} viral pieces with common patterns",
                        confidence_score=0.8,
                        impact_level="medium",
                        supporting_evidence=[
                            f"Viral content count: {viral_analysis['viral_count']}",
                            f"Common themes: {', '.join(viral_analysis['common_themes'][:3])}"
                        ],
                        recommendations=[
                            "Study their viral content themes and formats",
                            "Experiment with similar content approaches",
                            "Monitor their content for pattern recognition"
                        ]
                    )
                    insights.append(insight)
        
        # Posting pattern insights
        for competitor_id, pattern_data in posting_patterns.items():
            if pattern_data["consistency_score"] > 0.8:  # High consistency
                insight = CompetitorInsight(
                    id=f"posting_consistency_{competitor_id}",
                    competitor_id=competitor_id,
                    insight_type="posting_strategy",
                    title=f"{pattern_data['competitor_name']} Consistent Posting Strategy",
                    description=f"Maintains high posting consistency ({pattern_data['consistency_score']:.1%}) with optimal timing",
                    confidence_score=0.85,
                    impact_level="medium",
                    supporting_evidence=[
                        f"Consistency score: {pattern_data['consistency_score']:.1%}",
                        f"Posting frequency: {pattern_data['posting_frequency']} posts/week"
                    ],
                    recommendations=[
                        "Adopt similar posting consistency practices",
                        "Study their optimal timing patterns",
                        "Consider their content mix ratios"
                    ]
                )
                insights.append(insight)
        
        # Generate AI-powered insights
        strategic_insights = await self._generate_ai_insights(
            engagement_analysis,
            content_performance,
            posting_patterns
        )
        
        insights.extend(strategic_insights)
        
        return insights[:15]  # Return top 15 insights
    
    # Helper methods
    
    def _calculate_engagement_score(self, content_item: ContentItem) -> float:
        """Calculate weighted engagement score for a content item."""
        
        metrics = content_item.engagement_metrics
        score = 0
        
        for metric, value in metrics.items():
            weight = self.performance_weights.get(metric, 1.0)
            score += value * weight
        
        # Normalize by follower count if available
        follower_count = metrics.get("follower_count", 1000)  # Default assumption
        normalized_score = (score / follower_count) * 100
        
        return normalized_score
    
    def _compare_to_benchmark(self, platform: Platform, score: float) -> Dict[str, Any]:
        """Compare performance to industry benchmarks."""
        
        benchmark = self.platform_benchmarks.get(platform, {})
        
        if not benchmark:
            return {"comparison": "no_benchmark", "performance": "unknown"}
        
        # Compare to engagement rate benchmark (most common metric)
        benchmark_score = benchmark.get("engagement_rate", 0.05) * 100  # Convert to percentage
        
        if score > benchmark_score * 1.5:
            performance = "excellent"
        elif score > benchmark_score * 1.2:
            performance = "above_average"
        elif score > benchmark_score * 0.8:
            performance = "average"
        else:
            performance = "below_average"
        
        return {
            "comparison": performance,
            "vs_benchmark": f"{score/benchmark_score:.1f}x" if benchmark_score > 0 else "N/A",
            "benchmark_value": benchmark_score
        }
    
    def _calculate_engagement_trend(self, content_items: List[ContentItem]) -> str:
        """Calculate engagement trend over time."""
        
        if len(content_items) < 4:
            return "insufficient_data"
        
        # Sort by publication date
        sorted_items = sorted(content_items, key=lambda x: x.published_at)
        
        # Split into periods
        period_size = len(sorted_items) // 4
        periods = [
            sorted_items[i:i + period_size]
            for i in range(0, len(sorted_items), period_size)
        ][:4]
        
        # Calculate average engagement for each period
        period_averages = []
        for period in periods:
            if period:
                scores = [self._calculate_engagement_score(item) for item in period]
                period_averages.append(statistics.mean(scores))
        
        if len(period_averages) < 2:
            return "insufficient_data"
        
        # Determine trend
        recent_avg = statistics.mean(period_averages[-2:])
        early_avg = statistics.mean(period_averages[:2])
        
        if recent_avg > early_avg * 1.2:
            return "improving"
        elif recent_avg < early_avg * 0.8:
            return "declining"
        else:
            return "stable"
    
    def _analyze_performance_by_length(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Analyze how content length affects performance."""
        
        length_performance = {"short": [], "medium": [], "long": []}
        
        for item in content_items:
            content_length = len(item.content.split())
            engagement_score = self._calculate_engagement_score(item)
            
            if content_length < 300:
                length_performance["short"].append(engagement_score)
            elif content_length < 800:
                length_performance["medium"].append(engagement_score)
            else:
                length_performance["long"].append(engagement_score)
        
        results = {}
        for length_category, scores in length_performance.items():
            if scores:
                results[length_category] = {
                    "avg_engagement": statistics.mean(scores),
                    "sample_size": len(scores),
                    "best_performing": max(scores)
                }
        
        return results
    
    def _analyze_performance_by_timing(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Analyze optimal posting times."""
        
        hour_performance = defaultdict(list)
        day_performance = defaultdict(list)
        
        for item in content_items:
            engagement_score = self._calculate_engagement_score(item)
            
            hour = item.published_at.hour
            day = item.published_at.strftime('%A')
            
            hour_performance[hour].append(engagement_score)
            day_performance[day].append(engagement_score)
        
        # Find best performing times
        best_hours = sorted(
            [(hour, statistics.mean(scores)) for hour, scores in hour_performance.items() if len(scores) >= 2],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        best_days = sorted(
            [(day, statistics.mean(scores)) for day, scores in day_performance.items() if len(scores) >= 2],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "best_hours": [{"hour": hour, "avg_engagement": score} for hour, score in best_hours],
            "best_days": [{"day": day, "avg_engagement": score} for day, score in best_days]
        }
    
    async def _analyze_keyword_performance(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Analyze which keywords correlate with high performance."""
        
        keyword_scores = defaultdict(list)
        
        for item in content_items:
            engagement_score = self._calculate_engagement_score(item)
            
            for keyword in item.keywords:
                keyword_scores[keyword].append(engagement_score)
        
        # Calculate average performance for each keyword
        keyword_performance = {}
        for keyword, scores in keyword_scores.items():
            if len(scores) >= 3:  # Require at least 3 occurrences
                keyword_performance[keyword] = {
                    "avg_engagement": statistics.mean(scores),
                    "usage_count": len(scores),
                    "consistency": 1 - (statistics.stdev(scores) / statistics.mean(scores))
                    if len(scores) > 1 and statistics.mean(scores) > 0 else 0
                }
        
        # Find top performing keywords
        top_keywords = sorted(
            keyword_performance.items(),
            key=lambda x: x[1]["avg_engagement"],
            reverse=True
        )[:10]
        
        return {
            "top_performing_keywords": [
                {"keyword": keyword, **data} 
                for keyword, data in top_keywords
            ],
            "total_keywords_analyzed": len(keyword_performance)
        }
    
    def _identify_viral_content(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Identify viral content and analyze common characteristics."""
        
        engagement_scores = [self._calculate_engagement_score(item) for item in content_items]
        
        if len(engagement_scores) < 3:
            return {"viral_count": 0, "viral_threshold": 0}
        
        # Calculate viral threshold (2 standard deviations above mean)
        mean_engagement = statistics.mean(engagement_scores)
        std_engagement = statistics.stdev(engagement_scores) if len(engagement_scores) > 1 else 0
        viral_threshold = mean_engagement + (2 * std_engagement)
        
        # Identify viral content
        viral_content = []
        for item, score in zip(content_items, engagement_scores):
            if score >= viral_threshold:
                viral_content.append(item)
        
        # Analyze common characteristics
        common_themes = []
        if viral_content:
            all_keywords = []
            for item in viral_content:
                all_keywords.extend(item.keywords)
            
            keyword_counts = Counter(all_keywords)
            common_themes = [keyword for keyword, count in keyword_counts.most_common(5)]
        
        return {
            "viral_count": len(viral_content),
            "viral_threshold": viral_threshold,
            "viral_rate": len(viral_content) / len(content_items) if content_items else 0,
            "common_themes": common_themes,
            "viral_titles": [item.title for item in viral_content[:3]]
        }
    
    def _calculate_quality_engagement_correlation(self, content_items: List[ContentItem]) -> Optional[float]:
        """Calculate correlation between content quality and engagement."""
        
        quality_scores = []
        engagement_scores = []
        
        for item in content_items:
            if item.quality_score is not None:
                quality_scores.append(item.quality_score)
                engagement_scores.append(self._calculate_engagement_score(item))
        
        if len(quality_scores) < 5:
            return None
        
        # Calculate Pearson correlation
        try:
            correlation, _ = stats.pearsonr(quality_scores, engagement_scores)
            return correlation
        except:
            return None
    
    async def _identify_content_themes(self, content_items: List[ContentItem]) -> List[Dict[str, Any]]:
        """Identify common content themes and their performance."""
        
        # Group by keywords
        theme_groups = defaultdict(list)
        
        for item in content_items:
            # Use first keyword as primary theme (simplified)
            if item.keywords:
                primary_theme = item.keywords[0]
                theme_groups[primary_theme].append(item)
        
        # Analyze each theme
        themes = []
        for theme, items in theme_groups.items():
            if len(items) >= 3:  # Require at least 3 pieces
                engagement_scores = [self._calculate_engagement_score(item) for item in items]
                
                themes.append({
                    "theme": theme,
                    "content_count": len(items),
                    "avg_engagement": statistics.mean(engagement_scores),
                    "best_performing_title": max(
                        items, key=lambda x: self._calculate_engagement_score(x)
                    ).title
                })
        
        # Sort by average engagement
        themes.sort(key=lambda x: x["avg_engagement"], reverse=True)
        
        return themes[:10]  # Top 10 themes
    
    def _calculate_posting_frequency(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Calculate posting frequency metrics."""
        
        if len(content_items) < 2:
            return {"posts_per_week": 0, "posting_interval_days": 0}
        
        # Sort by date
        sorted_items = sorted(content_items, key=lambda x: x.published_at)
        
        # Calculate time span
        time_span = sorted_items[-1].published_at - sorted_items[0].published_at
        weeks = time_span.days / 7
        
        posts_per_week = len(content_items) / max(weeks, 0.1)
        
        # Calculate average interval between posts
        intervals = []
        for i in range(1, len(sorted_items)):
            interval = (sorted_items[i].published_at - sorted_items[i-1].published_at).days
            intervals.append(interval)
        
        avg_interval = statistics.mean(intervals) if intervals else 0
        
        return {
            "posts_per_week": posts_per_week,
            "posting_interval_days": avg_interval,
            "total_posts": len(content_items),
            "analysis_period_weeks": weeks
        }
    
    def _analyze_optimal_timing(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Analyze optimal posting timing based on performance."""
        
        return self._analyze_performance_by_timing(content_items)
    
    def _analyze_content_mix(self, content_items: List[ContentItem]) -> Dict[str, Any]:
        """Analyze the mix of content types and platforms."""
        
        content_type_count = Counter([item.content_type for item in content_items])
        platform_count = Counter([item.platform for item in content_items])
        
        return {
            "content_types": dict(content_type_count),
            "platforms": dict(platform_count),
            "diversity_score": len(content_type_count) / len(ContentType) +
                             len(platform_count) / len(Platform)
        }
    
    def _calculate_posting_consistency(self, content_items: List[ContentItem]) -> float:
        """Calculate posting consistency score."""
        
        if len(content_items) < 4:
            return 0
        
        # Calculate intervals between posts
        sorted_items = sorted(content_items, key=lambda x: x.published_at)
        intervals = []
        
        for i in range(1, len(sorted_items)):
            interval = (sorted_items[i].published_at - sorted_items[i-1].published_at).days
            intervals.append(interval)
        
        if not intervals:
            return 0
        
        # Consistency is inverse of coefficient of variation
        mean_interval = statistics.mean(intervals)
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        
        if mean_interval == 0:
            return 0
        
        cv = std_interval / mean_interval
        consistency_score = max(0, 1 - cv)  # Higher consistency = lower variation
        
        return consistency_score
    
    def _calculate_activity_trend(self, content_items: List[ContentItem]) -> str:
        """Calculate whether posting activity is increasing, decreasing, or stable."""
        
        if len(content_items) < 6:
            return "insufficient_data"
        
        # Sort by date and split into two halves
        sorted_items = sorted(content_items, key=lambda x: x.published_at)
        mid_point = len(sorted_items) // 2
        
        early_items = sorted_items[:mid_point]
        recent_items = sorted_items[mid_point:]
        
        # Calculate posting rate for each half
        early_span = (early_items[-1].published_at - early_items[0].published_at).days
        recent_span = (recent_items[-1].published_at - recent_items[0].published_at).days
        
        early_rate = len(early_items) / max(early_span, 1)
        recent_rate = len(recent_items) / max(recent_span, 1)
        
        if recent_rate > early_rate * 1.3:
            return "increasing"
        elif recent_rate < early_rate * 0.7:
            return "decreasing"
        else:
            return "stable"
    
    def _rank_content_type_performance(self, content_type: ContentType, scores: List[float]) -> int:
        """Rank content type performance (1 = best performing)."""
        
        avg_score = statistics.mean(scores)
        
        # Simple ranking based on average score
        # This would be enhanced with more sophisticated ranking logic
        if avg_score > 50:
            return 1  # Top tier
        elif avg_score > 30:
            return 2  # Second tier
        elif avg_score > 15:
            return 3  # Third tier
        else:
            return 4  # Bottom tier
    
    async def _generate_ai_insights(
        self,
        engagement_analysis: Dict[str, Any],
        content_performance: Dict[str, Any],
        posting_patterns: Dict[str, Any]
    ) -> List[CompetitorInsight]:
        """Generate AI-powered strategic insights."""
        
        insights = []
        
        try:
            # Prepare data summary for AI analysis
            analysis_summary = {
                "top_performers": [],
                "content_strategies": [],
                "timing_patterns": []
            }
            
            # Extract key insights for AI analysis
            for comp_id, data in engagement_analysis.items():
                if data["avg_engagement"] > 20:  # High performers
                    analysis_summary["top_performers"].append({
                        "name": data["competitor_name"],
                        "avg_engagement": data["avg_engagement"],
                        "consistency": data["engagement_consistency"],
                        "top_platforms": list(data["platform_performance"].keys())[:2]
                    })
            
            if len(analysis_summary["top_performers"]) > 0:
                prompt = f"""
                Analyze this competitor performance data and generate 2-3 strategic insights:
                
                Top Performers: {analysis_summary["top_performers"]}
                
                Focus on:
                1. What makes top performers successful
                2. Common patterns across high-performing competitors
                3. Strategic opportunities based on the data
                
                For each insight, provide:
                - Title (concise)
                - Description (2-3 sentences)
                - Strategic recommendation (1-2 sentences)
                """
                
                llm = self._get_analysis_llm()
                if llm is None:
                    return []  # Return empty insights if LLM not available
                
                response = await llm.agenerate([
                    [HumanMessage(content=prompt)]
                ])
                
                ai_insights_text = response.generations[0][0].text.strip()
                
                # Parse AI response into insights (simplified parsing)
                insight_sections = ai_insights_text.split('\n\n')
                
                for i, section in enumerate(insight_sections[:3]):
                    if len(section) > 50:  # Valid insight
                        insight = CompetitorInsight(
                            id=f"ai_insight_{i}",
                            competitor_id="multiple",
                            insight_type="strategic_analysis",
                            title=f"AI Strategic Insight #{i+1}",
                            description=section[:500],  # Truncate if too long
                            confidence_score=0.75,
                            impact_level="medium",
                            supporting_evidence=["AI analysis of competitor performance data"],
                            recommendations=["Implement strategic recommendations from analysis"]
                        )
                        insights.append(insight)
            
        except Exception as e:
            self.logger.debug(f"Failed to generate AI insights: {str(e)}")
        
        return insights
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the performance analysis agent's main functionality.
        Routes to appropriate analysis method based on input.
        """
        return {
            "status": "ready",
            "agent_type": "performance_analysis",
            "available_operations": [
                "analyze_competitor_performance",
                "benchmark_engagement",
                "analyze_posting_patterns"
            ]
        }