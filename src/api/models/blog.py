"""Blog-related API models."""

from pydantic import BaseModel
from typing import Optional


class BlogCreateRequest(BaseModel):
    title: str
    company_context: str
    content_type: str = "blog"  # "blog" or "linkedin"


class BlogEditRequest(BaseModel):
    content_markdown: str


class BlogReviseRequest(BaseModel):
    instruction: str
    text_to_revise: str


class BlogSearchRequest(BaseModel):
    query: Optional[str] = None
    status: Optional[str] = None
    limit: Optional[int] = 50


class BlogSummary(BaseModel):
    id: str
    title: str
    status: str
    created_at: str
    word_count: Optional[int] = None
    reading_time: Optional[int] = None
    seo_score: Optional[float] = None
    published_at: Optional[str] = None


class BlogDetail(BlogSummary):
    content_markdown: str
    initial_prompt: dict
    updated_at: Optional[str] = None
    geo_optimized: Optional[bool] = False
    geo_score: Optional[int] = None
    geo_metadata: Optional[dict] = None


class BlogMetadata(BaseModel):
    """Enhanced blog metadata for analytics"""
    id: str
    word_count: int
    reading_time: int
    seo_score: Optional[float] = None
    geo_optimized: bool = False
    geo_score: Optional[int] = None
    content_quality_score: Optional[float] = None
    readability_score: Optional[float] = None