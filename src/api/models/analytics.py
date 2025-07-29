"""Analytics-related API models."""

from pydantic import BaseModel
from typing import Optional


class BlogAnalyticsRequest(BaseModel):
    views: int = 0
    unique_visitors: int = 0
    engagement_rate: float = 0.0
    avg_time_on_page: Optional[int] = None
    bounce_rate: float = 0.0
    social_shares: int = 0
    comments_count: int = 0
    conversion_rate: float = 0.0
    seo_score: float = 0.0
    readability_score: float = 0.0


class MarketingMetricRequest(BaseModel):
    metric_type: str
    metric_value: float
    source: Optional[str] = None
    medium: Optional[str] = None
    campaign_name: Optional[str] = None


class AgentFeedbackRequest(BaseModel):
    agent_type: str
    feedback_type: str
    feedback_value: Optional[float] = None
    feedback_text: Optional[str] = None
    user_id: Optional[str] = None