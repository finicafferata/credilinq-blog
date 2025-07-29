"""
Data models for the competitor intelligence system.
Defines all data structures used across the multi-agent system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import uuid

class ContentType(str, Enum):
    """Types of content monitored."""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA_POST = "social_media_post"
    VIDEO = "video"
    PODCAST = "podcast"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    WEBINAR = "webinar"
    EMAIL_NEWSLETTER = "email_newsletter"
    PRESS_RELEASE = "press_release"
    PRODUCT_UPDATE = "product_update"

class Platform(str, Enum):
    """Platforms where content is monitored."""
    WEBSITE = "website"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    MEDIUM = "medium"
    SUBSTACK = "substack"
    TIKTOK = "tiktok"
    REDDIT = "reddit"

class Industry(str, Enum):
    """Industry categories for trend analysis."""
    FINTECH = "fintech"
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    MARKETING = "marketing"
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    RETAIL = "retail"
    MEDIA = "media"

class TrendStrength(str, Enum):
    """Strength levels for trends."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VIRAL = "viral"

class AlertPriority(str, Enum):
    """Priority levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CompetitorTier(str, Enum):
    """Competitor classification tiers."""
    DIRECT = "direct"          # Direct competitors
    INDIRECT = "indirect"      # Indirect competitors
    ASPIRATIONAL = "aspirational"  # Companies you aspire to compete with
    ADJACENT = "adjacent"      # Adjacent market players

@dataclass
class Competitor:
    """Represents a competitor being monitored."""
    id: str
    name: str
    domain: str
    tier: CompetitorTier
    industry: Industry
    description: str
    platforms: List[Platform] = field(default_factory=list)
    monitoring_keywords: List[str] = field(default_factory=list)
    last_monitored: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ContentItem:
    """Represents a piece of competitor content."""
    id: str
    competitor_id: str
    title: str
    content: str
    content_type: ContentType
    platform: Platform
    url: str
    published_at: datetime
    discovered_at: datetime
    author: Optional[str] = None
    engagement_metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class Trend:
    """Represents a detected industry trend."""
    id: str
    topic: str
    keywords: List[str]
    industry: Industry
    strength: TrendStrength
    growth_rate: float  # Percentage growth
    first_detected: datetime
    last_updated: datetime
    peak_date: Optional[datetime] = None
    related_content: List[str] = field(default_factory=list)  # Content IDs
    competitors_using: List[str] = field(default_factory=list)  # Competitor IDs
    opportunity_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ContentGap:
    """Represents an identified content gap opportunity."""
    id: str
    topic: str
    description: str
    opportunity_score: float
    difficulty_score: float
    potential_reach: int
    content_types_missing: List[ContentType]
    platforms_missing: List[Platform]
    keywords: List[str]
    competitors_covering: List[str] = field(default_factory=list)
    suggested_approach: Optional[str] = None
    identified_at: datetime = field(default_factory=lambda: datetime.utcnow())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class CompetitorInsight:
    """Strategic insight about a competitor."""
    id: str
    competitor_id: str
    insight_type: str  # "content_strategy", "performance_pattern", "positioning_shift", etc.
    title: str
    description: str
    confidence_score: float
    impact_level: str  # "low", "medium", "high"
    supporting_evidence: List[str] = field(default_factory=list)  # Content IDs or data points
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.utcnow())
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class MarketAnalysis:
    """Market-wide content analysis."""
    id: str
    industry: Industry
    analysis_period: tuple[datetime, datetime]
    total_content_analyzed: int
    top_topics: List[Dict[str, Any]]
    content_type_distribution: Dict[ContentType, int]
    platform_distribution: Dict[Platform, int]
    engagement_benchmarks: Dict[str, float]
    trending_keywords: List[str]
    content_velocity: float  # Posts per day across all competitors
    quality_trends: Dict[str, float]
    generated_at: datetime = field(default_factory=lambda: datetime.utcnow())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class Alert:
    """Intelligence alert for stakeholders."""
    id: str
    alert_type: str
    priority: AlertPriority
    title: str
    message: str
    data: Dict[str, Any]
    competitor_ids: List[str] = field(default_factory=list)
    trend_ids: List[str] = field(default_factory=list)
    content_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    recipients: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

# Pydantic models for API requests/responses

class CompetitorCreate(BaseModel):
    """Model for creating a new competitor."""
    name: str = Field(..., description="Competitor name")
    domain: str = Field(..., description="Primary domain/website")
    tier: CompetitorTier = Field(..., description="Competitor tier")
    industry: Industry = Field(..., description="Industry category")
    description: str = Field(..., description="Brief description of competitor")
    platforms: List[Platform] = Field(default_factory=list, description="Platforms to monitor")
    monitoring_keywords: List[str] = Field(default_factory=list, description="Keywords to track")
    
    @validator('domain')
    def validate_domain(cls, v):
        if not v.startswith(('http://', 'https://')):
            v = f'https://{v}'
        return v

class MonitoringConfig(BaseModel):
    """Configuration for monitoring settings."""
    check_frequency_hours: int = Field(default=24, description="How often to check for new content")
    content_types: List[ContentType] = Field(default_factory=list, description="Content types to monitor")
    platforms: List[Platform] = Field(default_factory=list, description="Platforms to monitor")
    keywords: List[str] = Field(default_factory=list, description="Keywords to track")
    sentiment_analysis: bool = Field(default=True, description="Enable sentiment analysis")
    quality_scoring: bool = Field(default=True, description="Enable content quality scoring")
    max_content_age_days: int = Field(default=90, description="Maximum age of content to analyze")

class TrendQuery(BaseModel):
    """Query parameters for trend analysis."""
    industry: Optional[Industry] = Field(None, description="Filter by industry")
    keywords: List[str] = Field(default_factory=list, description="Keywords to analyze")
    time_range_days: int = Field(default=30, description="Time range for analysis")
    min_strength: TrendStrength = Field(default=TrendStrength.WEAK, description="Minimum trend strength")
    include_predictions: bool = Field(default=True, description="Include trend predictions")

class GapAnalysisRequest(BaseModel):
    """Request for content gap analysis."""
    competitors: List[str] = Field(..., description="Competitor IDs to analyze")
    your_content_topics: List[str] = Field(..., description="Topics you currently cover")
    industry: Industry = Field(..., description="Industry focus")
    analysis_depth: str = Field(default="standard", description="Analysis depth: basic, standard, deep")
    content_types: List[ContentType] = Field(default_factory=list, description="Content types to analyze")
    
class CompetitorIntelligenceReport(BaseModel):
    """Comprehensive competitor intelligence report."""
    report_id: str
    generated_at: datetime
    analysis_period: Dict[str, str]
    competitors_analyzed: List[Dict[str, Any]]
    key_insights: List[Dict[str, Any]]
    trending_topics: List[Dict[str, Any]]
    content_gaps: List[Dict[str, Any]]
    performance_benchmarks: Dict[str, Any]
    recommendations: List[str]
    market_overview: Dict[str, Any]
    
class AlertSubscription(BaseModel):
    """Model for alert subscriptions."""
    user_id: str
    alert_types: List[str] = Field(..., description="Types of alerts to receive")
    competitors: List[str] = Field(default_factory=list, description="Specific competitors to monitor")
    keywords: List[str] = Field(default_factory=list, description="Keywords to track")
    priority_threshold: AlertPriority = Field(default=AlertPriority.MEDIUM, description="Minimum alert priority")
    delivery_channels: List[str] = Field(default=["email"], description="How to deliver alerts")
    frequency_limit: int = Field(default=10, description="Maximum alerts per day")

# Response models
class CompetitorSummary(BaseModel):
    """Summary of competitor data."""
    id: str
    name: str
    tier: CompetitorTier
    industry: Industry
    content_count: int
    last_activity: Optional[datetime]
    avg_engagement: float
    trending_score: float

class TrendSummary(BaseModel):
    """Summary of trend data."""
    id: str
    topic: str
    strength: TrendStrength
    growth_rate: float
    opportunity_score: Optional[float]
    competitors_count: int
    content_count: int

class InsightSummary(BaseModel):
    """Summary of strategic insights."""
    id: str
    competitor_name: str
    insight_type: str
    title: str
    confidence_score: float
    impact_level: str
    generated_at: datetime