"""
Trend Analysis Agent for detecting trending topics and patterns in competitor content.
Analyzes content velocity, keyword emergence, and topic momentum across industry verticals.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import asdict
import math
import re
from statistics import mean, stdev

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from src.core.llm_client import create_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from ..core.base_agent import BaseAgent
from .models import (
    Trend, TrendStrength, ContentItem, Industry, Platform,
    ContentType, MarketAnalysis
)
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class TrendAnalysisAgent(BaseAgent):
    """
    Specialized agent for analyzing content trends and detecting emerging topics.
    Uses statistical analysis, NLP, and machine learning to identify trending patterns.
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..core.base_agent import AgentMetadata, AgentType
        
        metadata = AgentMetadata(
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            name="TrendAnalysisAgent"
        )
        super().__init__(metadata)
        
        # Initialize AI for trend analysis (lazy loading)
        self.analysis_llm = None
        
    def _get_analysis_llm(self):
        """Lazy initialize the analysis LLM."""
        if self.analysis_llm is None:
            try:
                self.analysis_llm = create_llm(
                    model="gemini-1.5-flash",
                    temperature=0.2,
                    max_tokens=1500
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI LLM: {e}")
                return None
        return self.analysis_llm
        
        # Trend detection parameters
        self.trend_config = {
            "minimum_mentions": 5,
            "minimum_growth_rate": 0.15,  # 15% growth
            "temporal_window_days": 7,
            "viral_threshold": 2.0,  # Standard deviations above mean
            "cluster_min_samples": 3,
            "similarity_threshold": 0.7
        }
        
        # Industry-specific keyword filters
        self.industry_keywords = {
            Industry.FINTECH: [
                "fintech", "payments", "blockchain", "crypto", "neobank", 
                "digital banking", "financial technology", "regtech", "insurtech",
                "open banking", "defi", "robo advisor", "buy now pay later"
            ],
            Industry.SAAS: [
                "saas", "software as a service", "cloud computing", "api",
                "integration", "automation", "workflow", "productivity",
                "collaboration", "scalability", "subscription model"
            ],
            Industry.ECOMMERCE: [
                "ecommerce", "online shopping", "marketplace", "retail",
                "customer experience", "conversion", "cart abandonment",
                "personalization", "omnichannel", "fulfillment"
            ],
            Industry.MARKETING: [
                "marketing", "digital marketing", "content marketing", "seo",
                "social media", "advertising", "branding", "growth hacking",
                "marketing automation", "attribution", "roi", "engagement"
            ]
        }
        
        # Cache for trend calculations
        self.trend_cache = {}
        self.last_analysis_time = None
    
    async def analyze_trending_topics(
        self,
        content_items: List[ContentItem],
        industry: Industry,
        time_window_days: int = 30
    ) -> List[Trend]:
        """
        Analyze content items to detect trending topics and emerging patterns.
        Returns ranked list of trends with strength and growth metrics.
        """
        
        async with async_performance_tracker(f"analyze_trends_{industry}"):
            # Filter content by time window and industry relevance
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            relevant_content = [
                item for item in content_items
                if item.published_at >= cutoff_date
                and self._is_industry_relevant(item, industry)
            ]
            
            if len(relevant_content) < 10:
                self.logger.warning(f"Insufficient content for trend analysis: {len(relevant_content)} items")
                return []
            
            self.logger.info(f"Analyzing {len(relevant_content)} content items for trends")
            
            # Extract and analyze keywords
            keyword_trends = await self._analyze_keyword_trends(relevant_content, time_window_days)
            
            # Cluster topics for thematic analysis
            topic_clusters = await self._cluster_topics(relevant_content)
            
            # Detect viral content patterns
            viral_patterns = await self._detect_viral_patterns(relevant_content)
            
            # Combine analyses into trend objects
            trends = await self._synthesize_trends(
                keyword_trends,
                topic_clusters, 
                viral_patterns,
                industry,
                time_window_days
            )
            
            # Score and rank trends
            ranked_trends = await self._rank_trends(trends, relevant_content)
            
            # Track metrics
            metrics.increment_counter(
                "trends.analyzed",
                tags={
                    "industry": industry.value,
                    "content_count": str(len(relevant_content)),
                    "trends_found": str(len(ranked_trends))
                }
            )
            
            return ranked_trends
    
    async def _analyze_keyword_trends(
        self,
        content_items: List[ContentItem],
        time_window_days: int
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze keyword frequency trends over time."""
        
        # Group content by time periods (weekly buckets)
        time_buckets = defaultdict(list)
        bucket_size = max(1, time_window_days // 7)  # Weekly buckets
        
        for item in content_items:
            days_ago = (datetime.utcnow() - item.published_at).days
            bucket = days_ago // bucket_size
            time_buckets[bucket].append(item)
        
        # Extract keywords from each time bucket
        keyword_timeline = defaultdict(list)
        
        for bucket_id, bucket_content in time_buckets.items():
            # Combine all content in bucket
            combined_text = " ".join([
                f"{item.title} {item.content}" 
                for item in bucket_content
            ])
            
            # Extract keywords
            keywords = await self._extract_trending_keywords(combined_text)
            
            # Count keyword frequencies
            keyword_counts = Counter(keywords)
            
            for keyword, count in keyword_counts.items():
                keyword_timeline[keyword].append({
                    "bucket": bucket_id,
                    "count": count,
                    "content_volume": len(bucket_content)
                })
        
        # Calculate trend metrics for each keyword
        keyword_trends = {}
        
        for keyword, timeline in keyword_timeline.items():
            if len(timeline) < 2:  # Need at least 2 data points
                continue
                
            # Sort by bucket (most recent first)
            timeline.sort(key=lambda x: x["bucket"])
            
            # Calculate growth rate
            counts = [point["count"] for point in timeline]
            growth_rate = self._calculate_growth_rate(counts)
            
            # Calculate momentum (acceleration)
            momentum = self._calculate_momentum(counts)
            
            # Calculate total mentions
            total_mentions = sum(counts)
            
            if total_mentions >= self.trend_config["minimum_mentions"]:
                keyword_trends[keyword] = {
                    "timeline": timeline,
                    "growth_rate": growth_rate,
                    "momentum": momentum,
                    "total_mentions": total_mentions,
                    "current_velocity": counts[-1] if counts else 0,
                    "peak_count": max(counts) if counts else 0
                }
        
        return keyword_trends
    
    async def _cluster_topics(
        self,
        content_items: List[ContentItem]
    ) -> List[Dict[str, Any]]:
        """Cluster content items by topic similarity."""
        
        if len(content_items) < 3:
            return []
        
        # Prepare text data
        texts = []
        for item in content_items:
            # Combine title and content preview
            text = f"{item.title} {item.content[:500]}"
            texts.append(text)
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(
                eps=0.3,
                min_samples=self.trend_config["cluster_min_samples"],
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(tfidf_matrix)
            
            # Analyze clusters
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    clusters[label].append({
                        "content": content_items[idx],
                        "text": texts[idx],
                        "vector_idx": idx
                    })
            
            # Process each cluster
            topic_clusters = []
            feature_names = vectorizer.get_feature_names_out()
            
            for cluster_id, cluster_items in clusters.items():
                if len(cluster_items) < self.trend_config["cluster_min_samples"]:
                    continue
                
                # Get cluster centroid
                cluster_indices = [item["vector_idx"] for item in cluster_items]
                cluster_vectors = tfidf_matrix[cluster_indices]
                centroid = cluster_vectors.mean(axis=0).A1
                
                # Extract top keywords for cluster
                top_indices = centroid.argsort()[-10:][::-1]
                cluster_keywords = [feature_names[i] for i in top_indices if centroid[i] > 0]
                
                # Calculate cluster metrics
                cluster_content = [item["content"] for item in cluster_items]
                avg_engagement = self._calculate_average_engagement(cluster_content)
                content_velocity = len(cluster_content)
                
                # Generate cluster topic using AI
                cluster_topic = await self._generate_cluster_topic(
                    cluster_keywords[:5],
                    [item["content"].title for item in cluster_items[:3]]
                )
                
                topic_clusters.append({
                    "id": f"cluster_{cluster_id}",
                    "topic": cluster_topic,
                    "keywords": cluster_keywords[:10],
                    "content_count": len(cluster_items),
                    "content_items": cluster_content,
                    "avg_engagement": avg_engagement,
                    "content_velocity": content_velocity,
                    "coherence_score": self._calculate_cluster_coherence(cluster_vectors)
                })
            
            return topic_clusters
            
        except Exception as e:
            self.logger.error(f"Topic clustering failed: {str(e)}")
            return []
    
    async def _detect_viral_patterns(
        self,
        content_items: List[ContentItem]
    ) -> List[Dict[str, Any]]:
        """Detect viral content patterns and anomalies."""
        
        # Extract engagement metrics
        engagement_scores = []
        for item in content_items:
            # Calculate composite engagement score
            metrics = item.engagement_metrics
            score = (
                metrics.get("likes", 0) * 1.0 +
                metrics.get("shares", 0) * 2.0 +
                metrics.get("comments", 0) * 1.5 +
                metrics.get("views", 0) * 0.1
            )
            engagement_scores.append({
                "content": item,
                "score": score,
                "normalized_score": 0  # Will be calculated
            })
        
        if not engagement_scores:
            return []
        
        # Calculate statistical thresholds
        scores = [item["score"] for item in engagement_scores]
        if len(scores) < 3:
            return []
        
        mean_score = mean(scores)
        std_score = stdev(scores) if len(scores) > 1 else 0
        
        # Normalize scores
        for item in engagement_scores:
            if std_score > 0:
                item["normalized_score"] = (item["score"] - mean_score) / std_score
            else:
                item["normalized_score"] = 0
        
        # Identify viral content (outliers)
        viral_threshold = self.trend_config["viral_threshold"]
        viral_content = [
            item for item in engagement_scores
            if item["normalized_score"] > viral_threshold
        ]
        
        # Analyze viral patterns
        viral_patterns = []
        
        if viral_content:
            # Group by content type
            viral_by_type = defaultdict(list)
            for item in viral_content:
                viral_by_type[item["content"].content_type].append(item)
            
            # Analyze each content type
            for content_type, viral_items in viral_by_type.items():
                # Extract common characteristics
                common_keywords = self._extract_common_keywords(
                    [item["content"] for item in viral_items]
                )
                
                # Calculate pattern metrics
                avg_viral_score = mean([item["normalized_score"] for item in viral_items])
                pattern_strength = len(viral_items) / len(content_items)
                
                viral_patterns.append({
                    "content_type": content_type,
                    "viral_items": viral_items,
                    "common_keywords": common_keywords,
                    "avg_viral_score": avg_viral_score,
                    "pattern_strength": pattern_strength,
                    "sample_titles": [
                        item["content"].title 
                        for item in viral_items[:3]
                    ]
                })
        
        return viral_patterns
    
    async def _synthesize_trends(
        self,
        keyword_trends: Dict[str, Dict[str, Any]],
        topic_clusters: List[Dict[str, Any]],
        viral_patterns: List[Dict[str, Any]],
        industry: Industry,
        time_window_days: int
    ) -> List[Trend]:
        """Synthesize different analyses into comprehensive trend objects."""
        
        trends = []
        
        # Convert keyword trends to Trend objects
        for keyword, trend_data in keyword_trends.items():
            if trend_data["growth_rate"] >= self.trend_config["minimum_growth_rate"]:
                
                # Determine trend strength
                strength = self._classify_trend_strength(
                    trend_data["growth_rate"],
                    trend_data["momentum"],
                    trend_data["total_mentions"]
                )
                
                # Calculate opportunity score
                opportunity_score = self._calculate_opportunity_score(
                    trend_data["growth_rate"],
                    trend_data["total_mentions"],
                    strength
                )
                
                trend = Trend(
                    id=f"keyword_trend_{hash(keyword)}",
                    topic=keyword,
                    keywords=[keyword] + self._get_related_keywords(keyword, keyword_trends),
                    industry=industry,
                    strength=strength,
                    growth_rate=trend_data["growth_rate"],
                    first_detected=datetime.utcnow() - timedelta(days=time_window_days),
                    last_updated=datetime.utcnow(),
                    opportunity_score=opportunity_score,
                    metadata={
                        "source": "keyword_analysis",
                        "total_mentions": trend_data["total_mentions"],
                        "momentum": trend_data["momentum"],
                        "current_velocity": trend_data["current_velocity"],
                        "timeline": trend_data["timeline"]
                    }
                )
                
                trends.append(trend)
        
        # Convert topic clusters to trends
        for cluster in topic_clusters:
            if cluster["content_count"] >= self.trend_config["minimum_mentions"]:
                
                # Calculate growth rate from content velocity
                growth_rate = min(cluster["content_velocity"] / time_window_days * 7, 3.0)  # Cap at 300%
                
                strength = self._classify_trend_strength(
                    growth_rate,
                    cluster["coherence_score"],
                    cluster["content_count"]
                )
                
                opportunity_score = self._calculate_opportunity_score(
                    growth_rate,
                    cluster["content_count"],
                    strength
                )
                
                trend = Trend(
                    id=f"topic_trend_{cluster['id']}",
                    topic=cluster["topic"],
                    keywords=cluster["keywords"],
                    industry=industry,
                    strength=strength,
                    growth_rate=growth_rate,
                    first_detected=datetime.utcnow() - timedelta(days=time_window_days),
                    last_updated=datetime.utcnow(),
                    related_content=[item.id for item in cluster["content_items"]],
                    opportunity_score=opportunity_score,
                    metadata={
                        "source": "topic_clustering",
                        "content_count": cluster["content_count"],
                        "avg_engagement": cluster["avg_engagement"],
                        "coherence_score": cluster["coherence_score"]
                    }
                )
                
                trends.append(trend)
        
        # Incorporate viral patterns
        for pattern in viral_patterns:
            viral_trend = Trend(
                id=f"viral_pattern_{pattern['content_type']}",
                topic=f"Viral {pattern['content_type'].value} patterns",
                keywords=pattern["common_keywords"],
                industry=industry,
                strength=TrendStrength.VIRAL,
                growth_rate=pattern["pattern_strength"] * 5,  # Amplify for viral content
                first_detected=datetime.utcnow() - timedelta(days=time_window_days),
                last_updated=datetime.utcnow(),
                related_content=[item["content"].id for item in pattern["viral_items"]],
                opportunity_score=90.0,  # High opportunity for viral patterns
                metadata={
                    "source": "viral_analysis",
                    "viral_score": pattern["avg_viral_score"],
                    "pattern_strength": pattern["pattern_strength"],
                    "sample_titles": pattern["sample_titles"]
                }
            )
            
            trends.append(viral_trend)
        
        return trends
    
    async def _rank_trends(
        self,
        trends: List[Trend],
        content_items: List[ContentItem]
    ) -> List[Trend]:
        """Rank trends by relevance and potential impact."""
        
        # Calculate composite ranking score
        for trend in trends:
            score = 0
            
            # Growth rate contribution (40%)
            score += min(trend.growth_rate * 40, 100)
            
            # Strength contribution (30%)
            strength_scores = {
                TrendStrength.WEAK: 10,
                TrendStrength.MODERATE: 30,
                TrendStrength.STRONG: 60,
                TrendStrength.VIRAL: 100
            }
            score += strength_scores.get(trend.strength, 0) * 0.3
            
            # Opportunity score contribution (20%)
            score += (trend.opportunity_score or 0) * 0.2
            
            # Recency bonus (10%)
            days_since_update = (datetime.utcnow() - trend.last_updated).days
            recency_bonus = max(0, 10 * (1 - days_since_update / 30))
            score += recency_bonus
            
            trend.metadata["ranking_score"] = score
        
        # Sort by ranking score
        ranked_trends = sorted(
            trends,
            key=lambda t: t.metadata.get("ranking_score", 0),
            reverse=True
        )
        
        return ranked_trends[:50]  # Return top 50 trends
    
    # Helper methods
    
    def _is_industry_relevant(self, content: ContentItem, industry: Industry) -> bool:
        """Check if content is relevant to the specified industry."""
        
        if industry not in self.industry_keywords:
            return True  # If no keywords defined, include all content
        
        text = f"{content.title} {content.content}".lower()
        keywords = self.industry_keywords[industry]
        
        # Check if any industry keywords appear in content
        return any(keyword.lower() in text for keyword in keywords)
    
    async def _extract_trending_keywords(self, text: str) -> List[str]:
        """Extract potential trending keywords from text."""
        
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words and short words
        stop_words = {
            'the', 'and', 'are', 'for', 'with', 'this', 'that', 'you', 'can',
            'not', 'but', 'have', 'has', 'was', 'will', 'been', 'from', 'they'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_growth_rate(self, counts: List[int]) -> float:
        """Calculate growth rate from a series of counts."""
        
        if len(counts) < 2:
            return 0.0
        
        # Simple growth rate calculation
        old_value = sum(counts[:len(counts)//2])
        new_value = sum(counts[len(counts)//2:])
        
        if old_value == 0:
            return float('inf') if new_value > 0 else 0.0
        
        return (new_value - old_value) / old_value
    
    def _calculate_momentum(self, counts: List[int]) -> float:
        """Calculate momentum (acceleration) of trend."""
        
        if len(counts) < 3:
            return 0.0
        
        # Calculate second derivative approximation
        momentum = 0.0
        for i in range(1, len(counts) - 1):
            momentum += counts[i+1] - 2*counts[i] + counts[i-1]
        
        return momentum / (len(counts) - 2)
    
    def _calculate_average_engagement(self, content_items: List[ContentItem]) -> float:
        """Calculate average engagement score for content items."""
        
        if not content_items:
            return 0.0
        
        total_engagement = 0
        for item in content_items:
            metrics = item.engagement_metrics
            engagement = (
                metrics.get("likes", 0) +
                metrics.get("shares", 0) * 2 +
                metrics.get("comments", 0) * 1.5
            )
            total_engagement += engagement
        
        return total_engagement / len(content_items)
    
    def _calculate_cluster_coherence(self, vectors) -> float:
        """Calculate coherence score for a cluster of vectors."""
        
        if vectors.shape[0] < 2:
            return 1.0
        
        # Calculate average pairwise similarity
        similarities = cosine_similarity(vectors)
        
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        avg_similarity = similarities[mask].mean()
        
        return max(0.0, avg_similarity)
    
    async def _generate_cluster_topic(
        self,
        keywords: List[str],
        sample_titles: List[str]
    ) -> str:
        """Generate a descriptive topic name for a cluster."""
        
        try:
            prompt = f"""
            Based on these keywords and sample titles, generate a concise topic name (2-4 words):
            
            Keywords: {', '.join(keywords)}
            Sample titles: {'; '.join(sample_titles)}
            
            Topic name:
            """
            
            response = await self.analysis_llm.agenerate([
                [HumanMessage(content=prompt)]
            ])
            
            topic = response.generations[0][0].text.strip()
            
            # Fallback to keywords if AI response is invalid
            if not topic or len(topic) > 50:
                topic = " ".join(keywords[:3])
            
            return topic
            
        except Exception as e:
            self.logger.debug(f"Failed to generate cluster topic: {str(e)}")
            return " ".join(keywords[:3])
    
    def _extract_common_keywords(self, content_items: List[ContentItem]) -> List[str]:
        """Extract common keywords from a list of content items."""
        
        all_keywords = []
        for item in content_items:
            all_keywords.extend(item.keywords)
        
        # Count keyword frequency
        keyword_counts = Counter(all_keywords)
        
        # Return most common keywords
        return [keyword for keyword, count in keyword_counts.most_common(10)]
    
    def _classify_trend_strength(
        self,
        growth_rate: float,
        momentum: float,
        mentions: int
    ) -> TrendStrength:
        """Classify trend strength based on metrics."""
        
        # Viral threshold
        if growth_rate > 2.0 or momentum > 10:
            return TrendStrength.VIRAL
        
        # Strong threshold
        if growth_rate > 1.0 and mentions > 20:
            return TrendStrength.STRONG
        
        # Moderate threshold
        if growth_rate > 0.5 and mentions > 10:
            return TrendStrength.MODERATE
        
        return TrendStrength.WEAK
    
    def _calculate_opportunity_score(
        self,
        growth_rate: float,
        mentions: int,
        strength: TrendStrength
    ) -> float:
        """Calculate opportunity score for a trend."""
        
        base_score = min(growth_rate * 30, 60)  # Growth contribution
        volume_score = min(mentions * 2, 25)   # Volume contribution
        
        strength_multipliers = {
            TrendStrength.WEAK: 0.8,
            TrendStrength.MODERATE: 1.0,
            TrendStrength.STRONG: 1.3,
            TrendStrength.VIRAL: 1.5
        }
        
        multiplier = strength_multipliers.get(strength, 1.0)
        
        return min((base_score + volume_score) * multiplier, 100.0)
    
    def _get_related_keywords(
        self,
        keyword: str,
        keyword_trends: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Find keywords related to the given keyword."""
        
        related = []
        
        for other_keyword, trend_data in keyword_trends.items():
            if other_keyword != keyword and len(related) < 5:
                # Simple similarity check (can be enhanced)
                if (keyword in other_keyword or 
                    other_keyword in keyword or
                    len(set(keyword.split()) & set(other_keyword.split())) > 0):
                    related.append(other_keyword)
        
        return related
    
    async def generate_market_analysis(
        self,
        trends: List[Trend],
        content_items: List[ContentItem],
        industry: Industry
    ) -> MarketAnalysis:
        """Generate comprehensive market analysis from trends and content."""
        
        analysis_period = (
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        # Analyze content distribution
        content_type_dist = Counter([item.content_type for item in content_items])
        platform_dist = Counter([item.platform for item in content_items])
        
        # Calculate engagement benchmarks
        engagement_metrics = defaultdict(list)
        for item in content_items:
            for metric, value in item.engagement_metrics.items():
                engagement_metrics[metric].append(value)
        
        engagement_benchmarks = {
            metric: mean(values) if values else 0
            for metric, values in engagement_metrics.items()
        }
        
        # Extract trending keywords
        trending_keywords = []
        for trend in trends[:10]:  # Top 10 trends
            trending_keywords.extend(trend.keywords[:3])
        
        # Calculate content velocity
        days_in_period = (analysis_period[1] - analysis_period[0]).days
        content_velocity = len(content_items) / max(days_in_period, 1)
        
        # Identify top topics
        top_topics = [
            {
                "topic": trend.topic,
                "strength": trend.strength.value,
                "growth_rate": trend.growth_rate,
                "opportunity_score": trend.opportunity_score
            }
            for trend in trends[:15]
        ]
        
        # Calculate quality trends (simplified)
        quality_scores = [
            item.quality_score for item in content_items 
            if item.quality_score is not None
        ]
        quality_trends = {
            "average_quality": mean(quality_scores) if quality_scores else 0,
            "quality_trend": "stable"  # Simplified
        }
        
        return MarketAnalysis(
            id=f"market_analysis_{industry}_{int(datetime.utcnow().timestamp())}",
            industry=industry,
            analysis_period=analysis_period,
            total_content_analyzed=len(content_items),
            top_topics=top_topics,
            content_type_distribution=dict(content_type_dist),
            platform_distribution=dict(platform_dist),
            engagement_benchmarks=engagement_benchmarks,
            trending_keywords=list(set(trending_keywords)),
            content_velocity=content_velocity,
            quality_trends=quality_trends,
            metadata={
                "trends_analyzed": len(trends),
                "analysis_method": "multi_agent_trend_analysis",
                "confidence_level": "high"
            }
        )
    
    async def get_trend_predictions(
        self,
        trends: List[Trend],
        prediction_days: int = 14
    ) -> Dict[str, Any]:
        """Generate predictions for trend evolution."""
        
        predictions = {}
        
        for trend in trends[:10]:  # Predict top 10 trends
            # Simple linear prediction based on growth rate
            current_momentum = trend.growth_rate
            predicted_growth = current_momentum * (prediction_days / 7)  # Weekly normalization
            
            # Predict peak date
            if current_momentum > 0.5:
                days_to_peak = min(prediction_days, int(1 / max(current_momentum - 0.3, 0.1)))
                peak_date = datetime.utcnow() + timedelta(days=days_to_peak)
            else:
                peak_date = None
            
            predictions[trend.id] = {
                "trend_topic": trend.topic,
                "predicted_growth": predicted_growth,
                "peak_date": peak_date.isoformat() if peak_date else None,
                "confidence": min(trend.opportunity_score / 100, 1.0),
                "recommendation": self._generate_trend_recommendation(trend, predicted_growth)
            }
        
        return {
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat(),
            "prediction_horizon_days": prediction_days,
            "methodology": "statistical_trend_analysis"
        }
    
    def _generate_trend_recommendation(self, trend: Trend, predicted_growth: float) -> str:
        """Generate actionable recommendation for a trend."""
        
        if predicted_growth > 1.0:
            return f"High opportunity: Create content around '{trend.topic}' immediately"
        elif predicted_growth > 0.5:
            return f"Moderate opportunity: Consider content about '{trend.topic}' in upcoming strategy"
        elif predicted_growth > 0:
            return f"Monitor: '{trend.topic}' shows steady interest, good for evergreen content"
        else:
            return f"Declining: '{trend.topic}' may be losing momentum, use cautiously"
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the trend analysis agent's main functionality.
        Routes to appropriate analysis method based on input.
        """
        return {
            "status": "ready",
            "agent_type": "trend_analysis",
            "available_operations": [
                "analyze_trending_topics",
                "generate_market_analysis",
                "predict_trend_trajectory"
            ]
        }