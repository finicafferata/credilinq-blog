from typing import List, Dict, Any


def generate_review_suggestions(content: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simple heuristic content reviewer stub.
    Returns two lists: comments and suggestions.
    This is a placeholder; can be replaced with OpenAI or your analyzer.
    """
    comments: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []

    # Heuristic example: if the intro phrase appears, suggest a rewrite
    target = "In today's rapidly changing business landscape"
    if target in content:
        start = content.find(target)
        end = start + len(target)
        suggestions.append({
            "author": "AI Assistant",
            "originalText": target,
            "suggestedText": "In today's dynamic business environment",
            "reason": "More concise and impactful phrasing",
            "position": {"start": start, "end": end},
        })

    # General comment example
    if len(content.split()) < 200:
        comments.append({
            "author": "AI Assistant",
            "content": "Consider expanding this section with more concrete examples and data.",
            "position": None,
        })

    return {"comments": comments, "suggestions": suggestions}

"""
AI-Powered Content Analysis Service
Provides sophisticated content analysis using AI models for quality scoring,
topic extraction, sentiment analysis, and competitive insights.
"""

import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import tiktoken

from ..config.settings import get_settings

settings = get_settings()

class ContentQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"

class ContentType(Enum):
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    PRESS_RELEASE = "press_release"
    CASE_STUDY = "case_study"
    WHITEPAPER = "whitepaper"
    SOCIAL_POST = "social_post"
    VIDEO_TRANSCRIPT = "video_transcript"
    PODCAST_TRANSCRIPT = "podcast_transcript"

@dataclass
class TopicExtraction:
    """Extracted topics from content analysis."""
    primary_topics: List[str]
    secondary_topics: List[str]
    entities: List[str]
    keywords: List[str]
    themes: List[str]
    confidence: float

@dataclass
class QualityMetrics:
    """Content quality analysis metrics."""
    overall_score: float  # 0-100
    readability_score: float
    engagement_potential: float
    seo_score: float
    information_density: float
    originality_score: float
    structure_score: float
    quality_rating: ContentQuality

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 to 1.0
    emotional_tone: List[str]
    confidence: float
    key_phrases: List[str]

@dataclass
class CompetitiveInsights:
    """AI-generated competitive insights."""
    content_strategy: str
    target_audience: str
    positioning: str
    strengths: List[str]
    weaknesses: List[str]
    differentiation_opportunities: List[str]
    threat_level: str  # low, medium, high, critical

@dataclass
class ContentAnalysisResult:
    """Complete AI content analysis result."""
    content_id: str
    content_type: ContentType
    topics: TopicExtraction
    quality: QualityMetrics
    sentiment: SentimentAnalysis
    competitive_insights: CompetitiveInsights
    analysis_timestamp: datetime
    processing_time_ms: int

class AIContentAnalyzer:
    """AI-powered content analysis service."""
    
    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                max_tokens=2000
            )
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            # Fallback for when OpenAI API key is not available
            self.llm = None
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except:
                self.encoding = None
        self.max_content_length = 8000  # tokens
        
    async def analyze_content(
        self, 
        content: str, 
        content_url: str = None,
        competitor_name: str = None,
        content_type: ContentType = ContentType.ARTICLE
    ) -> ContentAnalysisResult:
        """
        Perform comprehensive AI analysis of content.
        
        Args:
            content: The content text to analyze
            content_url: URL of the content (optional)
            competitor_name: Name of the competitor (optional)
            content_type: Type of content being analyzed
            
        Returns:
            ContentAnalysisResult with complete analysis
        """
        start_time = datetime.utcnow()
        
        # Truncate content if too long
        truncated_content = self._truncate_content(content)
        
        # Run analysis components in parallel
        tasks = [
            self._extract_topics(truncated_content),
            self._analyze_quality(truncated_content, content_type),
            self._analyze_sentiment(truncated_content),
            self._generate_competitive_insights(truncated_content, competitor_name or "Unknown")
        ]
        
        results = await asyncio.gather(*tasks)
        topics, quality, sentiment, competitive_insights = results
        
        end_time = datetime.utcnow()
        processing_time = int((end_time - start_time).total_seconds() * 1000)
        
        return ContentAnalysisResult(
            content_id=self._generate_content_id(content_url or content[:50]),
            content_type=content_type,
            topics=topics,
            quality=quality,
            sentiment=sentiment,
            competitive_insights=competitive_insights,
            analysis_timestamp=end_time,
            processing_time_ms=processing_time
        )
    
    async def _extract_topics(self, content: str) -> TopicExtraction:
        """Extract topics, entities, and themes from content."""
        system_prompt = """You are an expert content analyst. Extract topics, entities, keywords, and themes from the given content.

Return a JSON object with:
- primary_topics: 3-5 main topics (most important)
- secondary_topics: 3-7 supporting topics
- entities: Named entities (companies, people, products, locations)
- keywords: Important keywords and phrases
- themes: Overarching themes and concepts
- confidence: Your confidence in the analysis (0.0-1.0)

Focus on business-relevant topics and competitive intelligence insights."""

        user_prompt = f"Analyze this content and extract topics:\n\n{content}"
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            data = json.loads(response)
            
            return TopicExtraction(
                primary_topics=data.get('primary_topics', []),
                secondary_topics=data.get('secondary_topics', []),
                entities=data.get('entities', []),
                keywords=data.get('keywords', []),
                themes=data.get('themes', []),
                confidence=data.get('confidence', 0.8)
            )
        except Exception as e:
            # Fallback analysis
            return self._basic_topic_extraction(content)
    
    async def _analyze_quality(self, content: str, content_type: ContentType) -> QualityMetrics:
        """Analyze content quality across multiple dimensions."""
        system_prompt = f"""You are an expert content quality analyst. Analyze the quality of this {content_type.value} across multiple dimensions.

Rate each dimension from 0-100 and provide an overall quality assessment:

Return JSON with:
- overall_score: Overall quality score (0-100)
- readability_score: How easy it is to read and understand
- engagement_potential: Likelihood to engage readers
- seo_score: SEO optimization quality
- information_density: Amount of valuable information
- originality_score: Uniqueness and originality
- structure_score: Organization and flow
- quality_rating: One of: excellent, good, average, poor, very_poor

Consider factors like:
- Writing clarity and style
- Information value and depth
- Structure and organization
- Audience engagement
- Professional quality
- Competitive differentiation"""

        user_prompt = f"Analyze the quality of this {content_type.value}:\n\n{content}"
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            data = json.loads(response)
            
            return QualityMetrics(
                overall_score=data.get('overall_score', 70),
                readability_score=data.get('readability_score', 70),
                engagement_potential=data.get('engagement_potential', 70),
                seo_score=data.get('seo_score', 70),
                information_density=data.get('information_density', 70),
                originality_score=data.get('originality_score', 70),
                structure_score=data.get('structure_score', 70),
                quality_rating=ContentQuality(data.get('quality_rating', 'average'))
            )
        except Exception as e:
            # Fallback quality analysis
            return self._basic_quality_analysis(content)
    
    async def _analyze_sentiment(self, content: str) -> SentimentAnalysis:
        """Analyze sentiment and emotional tone of content."""
        system_prompt = """You are an expert sentiment analyst. Analyze the sentiment and emotional tone of the content.

Return JSON with:
- overall_sentiment: positive, negative, or neutral
- sentiment_score: Numeric score from -1.0 (very negative) to 1.0 (very positive)
- emotional_tone: List of emotional tones (e.g., confident, urgent, friendly, professional)
- confidence: Your confidence in the analysis (0.0-1.0)
- key_phrases: Important phrases that indicate sentiment

Focus on business context and competitive positioning."""

        user_prompt = f"Analyze the sentiment of this content:\n\n{content}"
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            data = json.loads(response)
            
            return SentimentAnalysis(
                overall_sentiment=data.get('overall_sentiment', 'neutral'),
                sentiment_score=data.get('sentiment_score', 0.0),
                emotional_tone=data.get('emotional_tone', []),
                confidence=data.get('confidence', 0.8),
                key_phrases=data.get('key_phrases', [])
            )
        except Exception as e:
            # Fallback sentiment analysis
            return self._basic_sentiment_analysis(content)
    
    async def _generate_competitive_insights(self, content: str, competitor_name: str) -> CompetitiveInsights:
        """Generate competitive intelligence insights from content."""
        system_prompt = f"""You are a competitive intelligence analyst. Analyze this content from {competitor_name} and provide strategic insights.

Return JSON with:
- content_strategy: What content strategy does this reveal?
- target_audience: Who is the target audience?
- positioning: How is the company positioning itself?
- strengths: What strengths does this content demonstrate? (list)
- weaknesses: What weaknesses or gaps are apparent? (list)
- differentiation_opportunities: How could competitors differentiate? (list)
- threat_level: low, medium, high, or critical (based on content quality and strategy)

Focus on actionable competitive intelligence."""

        user_prompt = f"Analyze this content from {competitor_name} for competitive insights:\n\n{content}"
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            data = json.loads(response)
            
            return CompetitiveInsights(
                content_strategy=data.get('content_strategy', 'Unknown strategy'),
                target_audience=data.get('target_audience', 'General audience'),
                positioning=data.get('positioning', 'Standard positioning'),
                strengths=data.get('strengths', []),
                weaknesses=data.get('weaknesses', []),
                differentiation_opportunities=data.get('differentiation_opportunities', []),
                threat_level=data.get('threat_level', 'medium')
            )
        except Exception as e:
            # Fallback competitive analysis
            return self._basic_competitive_analysis(competitor_name)
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with retry logic."""
        if self.llm is None:
            raise Exception("OpenAI API key not configured")
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit within token limits."""
        if self.encoding is None:
            # Simple character-based truncation if tiktoken is not available
            max_chars = self.max_content_length * 4  # Rough estimate: 4 chars per token
            return content[:max_chars] if len(content) > max_chars else content
            
        tokens = self.encoding.encode(content)
        if len(tokens) <= self.max_content_length:
            return content
        
        # Truncate to max length
        truncated_tokens = tokens[:self.max_content_length]
        return self.encoding.decode(truncated_tokens)
    
    def _generate_content_id(self, identifier: str) -> str:
        """Generate a unique content ID."""
        import hashlib
        return hashlib.md5(identifier.encode()).hexdigest()[:12]
    
    def _basic_topic_extraction(self, content: str) -> TopicExtraction:
        """Fallback topic extraction using simple heuristics."""
        words = re.findall(r'\b[A-Za-z]{3,}\b', content.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words as keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, freq in keywords]
        
        return TopicExtraction(
            primary_topics=keywords[:3],
            secondary_topics=keywords[3:6],
            entities=[],
            keywords=keywords,
            themes=[],
            confidence=0.5
        )
    
    def _basic_quality_analysis(self, content: str) -> QualityMetrics:
        """Fallback quality analysis using heuristics."""
        # Simple heuristics
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Basic scoring
        readability = min(100, max(0, 100 - (avg_sentence_length - 15) * 2))
        overall = min(100, max(20, word_count / 10))
        
        return QualityMetrics(
            overall_score=overall,
            readability_score=readability,
            engagement_potential=70,
            seo_score=60,
            information_density=overall,
            originality_score=70,
            structure_score=70,
            quality_rating=ContentQuality.AVERAGE
        )
    
    def _basic_sentiment_analysis(self, content: str) -> SentimentAnalysis:
        """Fallback sentiment analysis using simple word matching."""
        positive_words = ['great', 'excellent', 'amazing', 'good', 'best', 'innovative', 'leading']
        negative_words = ['bad', 'poor', 'worst', 'terrible', 'awful', 'disappointing']
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            score = 0.3
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = -0.3
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return SentimentAnalysis(
            overall_sentiment=sentiment,
            sentiment_score=score,
            emotional_tone=['professional'],
            confidence=0.6,
            key_phrases=[]
        )
    
    def _basic_competitive_analysis(self, competitor_name: str) -> CompetitiveInsights:
        """Fallback competitive analysis."""
        return CompetitiveInsights(
            content_strategy='Standard content marketing approach',
            target_audience='Business professionals',
            positioning='Industry participant',
            strengths=['Professional content'],
            weaknesses=['Limited differentiation'],
            differentiation_opportunities=['Unique value proposition', 'Thought leadership'],
            threat_level='medium'
        )

# Global instance
ai_content_analyzer = AIContentAnalyzer()