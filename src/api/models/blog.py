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


class BlogDetail(BlogSummary):
    content_markdown: str
    initial_prompt: dict