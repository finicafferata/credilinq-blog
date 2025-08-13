"""
Trend analysis engine for competitor intelligence.
Identifies market patterns, content trends, and competitive insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import psycopg2.extras
from ..config.database import db_config
import statistics
import math

logger = logging.getLogger(__name__)

@dataclass
class Trend:
    """Represents an identified trend."""
    id: str
    title: str
    description: str
    trend_type: str  # 'content', 'keyword', 'platform', 'sentiment', 'engagement'
    strength: str    # 'weak', 'moderate', 'strong', 'viral'
    confidence: float  # 0.0 to 1.0
    data_points: List[Dict[str, Any]]
    timeframe: str
    industries: List[str]
    competitors_involved: List[str]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class MarketInsight:
    """Represents a strategic market insight."""
    id: str
    title: str
    insight_type: str  # 'opportunity', 'threat', 'pattern', 'prediction'
    description: str
    confidence: float
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    supporting_data: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]
    created_at: datetime

class TrendAnalysisEngine:
    """Engine for analyzing trends and generating insights."""
    
    def __init__(self):
        self.min_data_points = 3
        self.trend_confidence_threshold = 0.6
        
    async def analyze_content_trends(
        self, 
        industry: Optional[str] = None,
        days_back: int = 30
    ) -> List[Trend]:
        """Analyze content trends across competitors."""
        trends = []
        
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get content data
                    query = """
                        SELECT 
                            ci.title, ci.content, ci.keywords, ci."contentType",
                            ci."publishedAt", ci."discoveredAt", ci."qualityScore",
                            c.name as competitor_name, c.industry, c.id as competitor_id
                        FROM ci_content_items ci
                        JOIN ci_competitors c ON ci."competitorId" = c.id
                        WHERE ci."discoveredAt" >= NOW() - INTERVAL '%s days'
                        AND c.is_active = true
                    """
                    params = [days_back]
                    
                    if industry:
                        query += " AND c.industry = %s"
                        params.append(industry)
                        
                    query += " ORDER BY ci.\"discoveredAt\" DESC"
                    
                    cur.execute(query, params)
                    content_items = [dict(row) for row in cur.fetchall()]
                    
                    if len(content_items) < self.min_data_points:
                        return trends
                        
                    # Analyze keyword trends
                    keyword_trends = await self._analyze_keyword_trends(content_items)
                    trends.extend(keyword_trends)
                    
                    # Analyze content type trends
                    content_type_trends = await self._analyze_content_type_trends(content_items)
                    trends.extend(content_type_trends)
                    
                    # Analyze quality trends
                    quality_trends = await self._analyze_quality_trends(content_items)
                    trends.extend(quality_trends)
                    
                    # Analyze publishing frequency trends
                    frequency_trends = await self._analyze_publishing_frequency(content_items)
                    trends.extend(frequency_trends)
                    
        except Exception as e:
            logger.error(f"Error analyzing content trends: {str(e)}")
            
        return trends
        
    async def _analyze_keyword_trends(self, content_items: List[Dict]) -> List[Trend]:
        """Analyze trending keywords and topics."""
        trends = []
        
        try:
            # Collect all keywords with timestamps
            keyword_timeline = defaultdict(list)
            
            for item in content_items:
                if item.get('keywords'):
                    timestamp = item['discoveredAt']
                    for keyword in item['keywords']:
                        keyword_timeline[keyword.lower()].append({
                            'timestamp': timestamp,
                            'competitor': item['competitor_name'],
                            'content_type': item['contentType'],
                            'quality_score': item.get('qualityScore', 0)
                        })
                        
            # Analyze trending keywords
            for keyword, occurrences in keyword_timeline.items():
                if len(occurrences) >= 3:  # Minimum occurrences
                    # Calculate trend strength
                    recent_count = len([o for o in occurrences 
                                      if (datetime.utcnow() - o['timestamp']).days <= 7])
                    total_count = len(occurrences)
                    
                    if recent_count > 0:
                        trend_strength = self._calculate_trend_strength(
                            recent_count, total_count, len(content_items)
                        )
                        
                        competitors_using = list(set(o['competitor'] for o in occurrences))
                        
                        trend = Trend(
                            id=f"keyword_trend_{keyword}_{datetime.utcnow().strftime('%Y%m%d')}",
                            title=f"Trending Topic: {keyword.title()}",
                            description=f"'{keyword}' mentioned by {len(competitors_using)} competitors",
                            trend_type='keyword',
                            strength=trend_strength,
                            confidence=min(len(occurrences) / 10, 1.0),
                            data_points=[{
                                'keyword': keyword,
                                'occurrences': total_count,
                                'recent_occurrences': recent_count,
                                'competitors': competitors_using,
                                'timeline': [o['timestamp'].isoformat() for o in occurrences]
                            }],
                            timeframe='30 days',
                            industries=[],
                            competitors_involved=competitors_using,
                            metadata={
                                'analysis_type': 'keyword_frequency',
                                'total_mentions': total_count,
                                'unique_competitors': len(competitors_using)
                            },
                            created_at=datetime.utcnow()
                        )
                        trends.append(trend)
                        
        except Exception as e:
            logger.error(f"Error analyzing keyword trends: {str(e)}")
            
        return trends
        
    async def _analyze_content_type_trends(self, content_items: List[Dict]) -> List[Trend]:
        """Analyze trends in content types."""
        trends = []
        
        try:
            # Group by week and content type
            weekly_content_types = defaultdict(lambda: defaultdict(int))
            
            for item in content_items:
                week_start = item['discoveredAt'] - timedelta(days=item['discoveredAt'].weekday())
                week_key = week_start.strftime('%Y-W%U')
                content_type = item['contentType']
                weekly_content_types[week_key][content_type] += 1
                
            # Analyze trends for each content type
            content_type_data = defaultdict(list)
            for week, types in weekly_content_types.items():
                for content_type, count in types.items():
                    content_type_data[content_type].append({
                        'week': week,
                        'count': count
                    })
                    
            for content_type, data_points in content_type_data.items():
                if len(data_points) >= 2:
                    # Calculate growth trend
                    counts = [d['count'] for d in data_points]
                    if len(counts) > 1:
                        growth_rate = (counts[-1] - counts[0]) / max(counts[0], 1)
                        
                        if abs(growth_rate) > 0.2:  # 20% change threshold
                            trend_direction = "increasing" if growth_rate > 0 else "decreasing"
                            
                            trend = Trend(
                                id=f"content_type_trend_{content_type}_{datetime.utcnow().strftime('%Y%m%d')}",
                                title=f"{content_type.replace('_', ' ').title()} Content Trend",
                                description=f"{content_type.replace('_', ' ').title()} content is {trend_direction} ({growth_rate:+.1%})",
                                trend_type='content_type',
                                strength='moderate' if abs(growth_rate) > 0.5 else 'weak',
                                confidence=min(len(data_points) / 4, 1.0),
                                data_points=data_points,
                                timeframe='30 days',
                                industries=[],
                                competitors_involved=[],
                                metadata={
                                    'content_type': content_type,
                                    'growth_rate': growth_rate,
                                    'direction': trend_direction
                                },
                                created_at=datetime.utcnow()
                            )
                            trends.append(trend)
                            
        except Exception as e:
            logger.error(f"Error analyzing content type trends: {str(e)}")
            
        return trends
        
    async def _analyze_quality_trends(self, content_items: List[Dict]) -> List[Trend]:
        """Analyze content quality trends."""
        trends = []
        
        try:
            # Group by competitor and calculate quality metrics
            competitor_quality = defaultdict(list)
            
            for item in content_items:
                if item.get('qualityScore') is not None:
                    competitor_quality[item['competitor_name']].append({
                        'score': float(item['qualityScore']),
                        'timestamp': item['discoveredAt'],
                        'content_type': item['contentType']
                    })
                    
            # Analyze quality trends for each competitor
            for competitor, quality_data in competitor_quality.items():
                if len(quality_data) >= 3:
                    scores = [d['score'] for d in quality_data]
                    avg_quality = statistics.mean(scores)
                    quality_std = statistics.stdev(scores) if len(scores) > 1 else 0
                    
                    # Check for quality improvement/decline
                    recent_scores = [d['score'] for d in quality_data[-3:]]
                    older_scores = [d['score'] for d in quality_data[:-3]] if len(quality_data) > 3 else scores
                    
                    if older_scores:
                        recent_avg = statistics.mean(recent_scores)
                        older_avg = statistics.mean(older_scores)
                        quality_change = (recent_avg - older_avg) / max(older_avg, 1)
                        
                        if abs(quality_change) > 0.15:  # 15% change threshold
                            trend_direction = "improving" if quality_change > 0 else "declining"
                            
                            trend = Trend(
                                id=f"quality_trend_{competitor}_{datetime.utcnow().strftime('%Y%m%d')}",
                                title=f"Content Quality Trend: {competitor}",
                                description=f"{competitor}'s content quality is {trend_direction} ({quality_change:+.1%})",
                                trend_type='quality',
                                strength='moderate' if abs(quality_change) > 0.3 else 'weak',
                                confidence=min(len(quality_data) / 5, 1.0),
                                data_points=[{
                                    'competitor': competitor,
                                    'average_quality': avg_quality,
                                    'recent_average': recent_avg,
                                    'change_percentage': quality_change,
                                    'data_points': len(quality_data)
                                }],
                                timeframe='30 days',
                                industries=[],
                                competitors_involved=[competitor],
                                metadata={
                                    'quality_std': quality_std,
                                    'direction': trend_direction,
                                    'change_rate': quality_change
                                },
                                created_at=datetime.utcnow()
                            )
                            trends.append(trend)
                            
        except Exception as e:
            logger.error(f"Error analyzing quality trends: {str(e)}")
            
        return trends
        
    async def _analyze_publishing_frequency(self, content_items: List[Dict]) -> List[Trend]:
        """Analyze publishing frequency trends."""
        trends = []
        
        try:
            # Group by competitor and week
            competitor_frequency = defaultdict(lambda: defaultdict(int))
            
            for item in content_items:
                week_start = item['discoveredAt'] - timedelta(days=item['discoveredAt'].weekday())
                week_key = week_start.strftime('%Y-W%U')
                competitor_frequency[item['competitor_name']][week_key] += 1
                
            # Analyze frequency trends
            for competitor, weekly_counts in competitor_frequency.items():
                if len(weekly_counts) >= 2:
                    counts = list(weekly_counts.values())
                    avg_frequency = statistics.mean(counts)
                    
                    # Look for frequency spikes
                    max_week = max(counts)
                    if max_week > avg_frequency * 2 and max_week >= 3:  # Spike detection
                        trend = Trend(
                            id=f"frequency_spike_{competitor}_{datetime.utcnow().strftime('%Y%m%d')}",
                            title=f"Publishing Spike: {competitor}",
                            description=f"{competitor} published {max_week} items in one week (avg: {avg_frequency:.1f})",
                            trend_type='frequency',
                            strength='strong' if max_week > avg_frequency * 3 else 'moderate',
                            confidence=0.8,
                            data_points=[{
                                'competitor': competitor,
                                'max_weekly_posts': max_week,
                                'average_weekly_posts': avg_frequency,
                                'spike_ratio': max_week / max(avg_frequency, 1)
                            }],
                            timeframe='30 days',
                            industries=[],
                            competitors_involved=[competitor],
                            metadata={
                                'analysis_type': 'publishing_frequency',
                                'spike_detection': True
                            },
                            created_at=datetime.utcnow()
                        )
                        trends.append(trend)
                        
        except Exception as e:
            logger.error(f"Error analyzing publishing frequency: {str(e)}")
            
        return trends
        
    def _calculate_trend_strength(self, recent_count: int, total_count: int, dataset_size: int) -> str:
        """Calculate trend strength based on frequency and recency."""
        recency_ratio = recent_count / max(total_count, 1)
        frequency_ratio = total_count / max(dataset_size, 1)
        
        strength_score = (recency_ratio * 0.6) + (frequency_ratio * 0.4)
        
        if strength_score > 0.7:
            return 'viral'
        elif strength_score > 0.4:
            return 'strong'
        elif strength_score > 0.2:
            return 'moderate'
        else:
            return 'weak'
            
    async def analyze_social_media_trends(
        self, 
        industry: Optional[str] = None,
        days_back: int = 30
    ) -> List[Trend]:
        """Analyze social media trends."""
        trends = []
        
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get social media data
                    query = """
                        SELECT 
                            sp.platform, sp.content, sp.hashtags, sp."postType",
                            sp."viralityScore", sp."sentimentScore", sp."publishedAt",
                            c.name as competitor_name, c.industry, c.id as competitor_id
                        FROM ci_social_posts sp
                        JOIN ci_competitors c ON sp."competitorId" = c.id
                        WHERE sp."publishedAt" >= NOW() - INTERVAL '%s days'
                        AND c.is_active = true
                    """
                    params = [days_back]
                    
                    if industry:
                        query += " AND c.industry = %s"
                        params.append(industry)
                        
                    query += " ORDER BY sp.\"publishedAt\" DESC"
                    
                    cur.execute(query, params)
                    social_posts = [dict(row) for row in cur.fetchall()]
                    
                    if len(social_posts) >= self.min_data_points:
                        # Analyze hashtag trends
                        hashtag_trends = await self._analyze_hashtag_trends(social_posts)
                        trends.extend(hashtag_trends)
                        
                        # Analyze platform trends
                        platform_trends = await self._analyze_platform_trends(social_posts)
                        trends.extend(platform_trends)
                        
                        # Analyze sentiment trends
                        sentiment_trends = await self._analyze_sentiment_trends(social_posts)
                        trends.extend(sentiment_trends)
                        
        except Exception as e:
            logger.error(f"Error analyzing social media trends: {str(e)}")
            
        return trends
        
    async def _analyze_hashtag_trends(self, social_posts: List[Dict]) -> List[Trend]:
        """Analyze trending hashtags."""
        trends = []
        
        try:
            hashtag_usage = defaultdict(list)
            
            for post in social_posts:
                if post.get('hashtags'):
                    for hashtag in post['hashtags']:
                        hashtag_clean = hashtag.lower().replace('#', '')
                        hashtag_usage[hashtag_clean].append({
                            'timestamp': post['publishedAt'],
                            'competitor': post['competitor_name'],
                            'platform': post['platform'],
                            'virality_score': post.get('viralityScore', 0)
                        })
                        
            # Find trending hashtags
            for hashtag, usage_data in hashtag_usage.items():
                if len(usage_data) >= 2:  # Minimum usage
                    recent_usage = len([u for u in usage_data 
                                      if (datetime.utcnow() - u['timestamp']).days <= 7])
                    
                    avg_virality = statistics.mean([u['virality_score'] for u in usage_data])
                    competitors_using = list(set(u['competitor'] for u in usage_data))
                    
                    if recent_usage > 0 and (len(competitors_using) > 1 or avg_virality > 3):
                        trend = Trend(
                            id=f"hashtag_trend_{hashtag}_{datetime.utcnow().strftime('%Y%m%d')}",
                            title=f"Trending Hashtag: #{hashtag}",
                            description=f"#{hashtag} used by {len(competitors_using)} competitors",
                            trend_type='hashtag',
                            strength='strong' if len(competitors_using) > 2 else 'moderate',
                            confidence=min(len(usage_data) / 5, 1.0),
                            data_points=[{
                                'hashtag': hashtag,
                                'total_usage': len(usage_data),
                                'recent_usage': recent_usage,
                                'competitors': competitors_using,
                                'avg_virality': avg_virality
                            }],
                            timeframe='30 days',
                            industries=[],
                            competitors_involved=competitors_using,
                            metadata={
                                'analysis_type': 'hashtag_trending',
                                'cross_competitor': len(competitors_using) > 1
                            },
                            created_at=datetime.utcnow()
                        )
                        trends.append(trend)
                        
        except Exception as e:
            logger.error(f"Error analyzing hashtag trends: {str(e)}")
            
        return trends
        
    async def _analyze_platform_trends(self, social_posts: List[Dict]) -> List[Trend]:
        """Analyze platform usage trends."""
        trends = []
        
        try:
            # Group by platform and week
            weekly_platform_usage = defaultdict(lambda: defaultdict(int))
            
            for post in social_posts:
                week_start = post['publishedAt'] - timedelta(days=post['publishedAt'].weekday())
                week_key = week_start.strftime('%Y-W%U')
                weekly_platform_usage[week_key][post['platform']] += 1
                
            # Analyze trends for each platform
            platform_data = defaultdict(list)
            for week, platforms in weekly_platform_usage.items():
                for platform, count in platforms.items():
                    platform_data[platform].append({
                        'week': week,
                        'count': count
                    })
                    
            for platform, data_points in platform_data.items():
                if len(data_points) >= 2:
                    counts = [d['count'] for d in data_points]
                    if len(counts) > 1:
                        growth_rate = (counts[-1] - counts[0]) / max(counts[0], 1)
                        
                        if abs(growth_rate) > 0.3:  # 30% change threshold
                            trend_direction = "increasing" if growth_rate > 0 else "decreasing"
                            
                            trend = Trend(
                                id=f"platform_trend_{platform}_{datetime.utcnow().strftime('%Y%m%d')}",
                                title=f"{platform.title()} Usage Trend",
                                description=f"{platform.title()} usage is {trend_direction} ({growth_rate:+.1%})",
                                trend_type='platform',
                                strength='strong' if abs(growth_rate) > 0.6 else 'moderate',
                                confidence=min(len(data_points) / 3, 1.0),
                                data_points=data_points,
                                timeframe='30 days',
                                industries=[],
                                competitors_involved=[],
                                metadata={
                                    'platform': platform,
                                    'growth_rate': growth_rate,
                                    'direction': trend_direction
                                },
                                created_at=datetime.utcnow()
                            )
                            trends.append(trend)
                            
        except Exception as e:
            logger.error(f"Error analyzing platform trends: {str(e)}")
            
        return trends
        
    async def _analyze_sentiment_trends(self, social_posts: List[Dict]) -> List[Trend]:
        """Analyze sentiment trends across competitors."""
        trends = []
        
        try:
            competitor_sentiment = defaultdict(list)
            
            for post in social_posts:
                if post.get('sentimentScore') is not None:
                    competitor_sentiment[post['competitor_name']].append({
                        'score': float(post['sentimentScore']),
                        'timestamp': post['publishedAt'],
                        'platform': post['platform']
                    })
                    
            # Analyze sentiment trends
            for competitor, sentiment_data in competitor_sentiment.items():
                if len(sentiment_data) >= 3:
                    scores = [d['score'] for d in sentiment_data]
                    avg_sentiment = statistics.mean(scores)
                    
                    # Check for sentiment shifts
                    recent_scores = [d['score'] for d in sentiment_data[-3:]]
                    older_scores = [d['score'] for d in sentiment_data[:-3]] if len(sentiment_data) > 3 else scores
                    
                    if older_scores:
                        recent_avg = statistics.mean(recent_scores)
                        older_avg = statistics.mean(older_scores)
                        sentiment_change = recent_avg - older_avg
                        
                        if abs(sentiment_change) > 0.2:  # Significant sentiment change
                            trend_direction = "more positive" if sentiment_change > 0 else "more negative"
                            
                            trend = Trend(
                                id=f"sentiment_trend_{competitor}_{datetime.utcnow().strftime('%Y%m%d')}",
                                title=f"Sentiment Shift: {competitor}",
                                description=f"{competitor}'s social sentiment is becoming {trend_direction}",
                                trend_type='sentiment',
                                strength='strong' if abs(sentiment_change) > 0.4 else 'moderate',
                                confidence=min(len(sentiment_data) / 5, 1.0),
                                data_points=[{
                                    'competitor': competitor,
                                    'average_sentiment': avg_sentiment,
                                    'recent_sentiment': recent_avg,
                                    'sentiment_change': sentiment_change,
                                    'data_points': len(sentiment_data)
                                }],
                                timeframe='30 days',
                                industries=[],
                                competitors_involved=[competitor],
                                metadata={
                                    'analysis_type': 'sentiment_analysis',
                                    'direction': trend_direction,
                                    'change_magnitude': sentiment_change
                                },
                                created_at=datetime.utcnow()
                            )
                            trends.append(trend)
                            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {str(e)}")
            
        return trends
        
    async def generate_market_insights(
        self, 
        trends: List[Trend],
        industry: Optional[str] = None
    ) -> List[MarketInsight]:
        """Generate strategic market insights from trends."""
        insights = []
        
        try:
            # Analyze cross-competitor patterns
            cross_competitor_insights = await self._analyze_cross_competitor_patterns(trends)
            insights.extend(cross_competitor_insights)
            
            # Identify content opportunities
            content_opportunities = await self._identify_content_opportunities(trends)
            insights.extend(content_opportunities)
            
            # Detect competitive threats
            competitive_threats = await self._detect_competitive_threats(trends)
            insights.extend(competitive_threats)
            
            # Predict emerging trends
            emerging_trends = await self._predict_emerging_trends(trends)
            insights.extend(emerging_trends)
            
        except Exception as e:
            logger.error(f"Error generating market insights: {str(e)}")
            
        return insights
        
    async def _analyze_cross_competitor_patterns(self, trends: List[Trend]) -> List[MarketInsight]:
        """Analyze patterns across multiple competitors."""
        insights = []
        
        # Find trends involving multiple competitors
        multi_competitor_trends = [t for t in trends if len(t.competitors_involved) > 1]
        
        if multi_competitor_trends:
            keyword_trends = [t for t in multi_competitor_trends if t.trend_type == 'keyword']
            
            if keyword_trends:
                insight = MarketInsight(
                    id=f"cross_competitor_keywords_{datetime.utcnow().strftime('%Y%m%d')}",
                    title="Market-wide Topic Convergence",
                    insight_type="pattern",
                    description=f"Multiple competitors are focusing on similar topics, indicating market trends",
                    confidence=0.8,
                    impact_level="medium",
                    supporting_data=[{
                        'trending_topics': [t.data_points[0]['keyword'] for t in keyword_trends],
                        'competitors_involved': len(set().union(*[t.competitors_involved for t in keyword_trends])),
                        'trend_count': len(keyword_trends)
                    }],
                    recommendations=[
                        "Monitor these trending topics closely for opportunities",
                        "Consider creating content around these themes",
                        "Analyze competitor positioning on these topics"
                    ],
                    metadata={'analysis_type': 'cross_competitor_keywords'},
                    created_at=datetime.utcnow()
                )
                insights.append(insight)
                
        return insights
        
    async def _identify_content_opportunities(self, trends: List[Trend]) -> List[MarketInsight]:
        """Identify content opportunities from trends."""
        insights = []
        
        # Look for declining quality trends
        quality_trends = [t for t in trends if t.trend_type == 'quality' and 'declining' in t.description]
        
        if quality_trends:
            insight = MarketInsight(
                id=f"content_opportunity_{datetime.utcnow().strftime('%Y%m%d')}",
                title="Content Quality Gap Opportunity",
                insight_type="opportunity",
                description=f"Competitors showing declining content quality - opportunity for differentiation",
                confidence=0.7,
                impact_level="medium",
                supporting_data=[{
                    'declining_competitors': [t.competitors_involved[0] for t in quality_trends],
                    'avg_decline': statistics.mean([t.metadata.get('change_rate', 0) for t in quality_trends])
                }],
                recommendations=[
                    "Increase content quality standards",
                    "Focus on comprehensive, well-researched content",
                    "Capitalize on competitors' content quality decline"
                ],
                metadata={'analysis_type': 'content_quality_opportunity'},
                created_at=datetime.utcnow()
            )
            insights.append(insight)
            
        return insights
        
    async def _detect_competitive_threats(self, trends: List[Trend]) -> List[MarketInsight]:
        """Detect competitive threats from trend data."""
        insights = []
        
        # Look for publishing frequency spikes
        frequency_spikes = [t for t in trends if t.trend_type == 'frequency' and 'spike' in t.title.lower()]
        
        if frequency_spikes:
            for trend in frequency_spikes:
                if trend.strength in ['strong', 'viral']:
                    insight = MarketInsight(
                        id=f"competitive_threat_{trend.competitors_involved[0]}_{datetime.utcnow().strftime('%Y%m%d')}",
                        title=f"Competitive Threat: {trend.competitors_involved[0]}",
                        insight_type="threat",
                        description=f"{trend.competitors_involved[0]} showing increased content activity",
                        confidence=0.8,
                        impact_level="high" if trend.strength == 'viral' else "medium",
                        supporting_data=[trend.data_points[0]],
                        recommendations=[
                            "Monitor competitor's content strategy closely",
                            "Analyze their content themes and messaging",
                            "Consider increasing own content frequency"
                        ],
                        metadata={'analysis_type': 'competitive_activity_threat'},
                        created_at=datetime.utcnow()
                    )
                    insights.append(insight)
                    
        return insights
        
    async def _predict_emerging_trends(self, trends: List[Trend]) -> List[MarketInsight]:
        """Predict emerging trends from current data."""
        insights = []
        
        # Look for early-stage trends with high confidence
        emerging = [t for t in trends if t.strength in ['moderate', 'strong'] and t.confidence > 0.7]
        
        if emerging:
            keyword_emerging = [t for t in emerging if t.trend_type == 'keyword']
            
            if keyword_emerging:
                insight = MarketInsight(
                    id=f"emerging_trends_{datetime.utcnow().strftime('%Y%m%d')}",
                    title="Emerging Market Trends",
                    insight_type="prediction",
                    description=f"Early indicators suggest emerging trends in {len(keyword_emerging)} topic areas",
                    confidence=0.6,
                    impact_level="medium",
                    supporting_data=[{
                        'emerging_topics': [t.data_points[0]['keyword'] for t in keyword_emerging],
                        'confidence_scores': [t.confidence for t in keyword_emerging]
                    }],
                    recommendations=[
                        "Early adoption of these emerging topics",
                        "Create thought leadership content",
                        "Monitor for mainstream adoption"
                    ],
                    metadata={'analysis_type': 'emerging_trend_prediction'},
                    created_at=datetime.utcnow()
                )
                insights.append(insight)
                
        return insights

# Global instance
trend_analysis_engine = TrendAnalysisEngine()