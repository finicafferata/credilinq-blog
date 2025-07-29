"""Campaign-related API models."""

from pydantic import BaseModel
from typing import List, Optional


class CampaignCreateRequest(BaseModel):
    blog_id: str


class CampaignTaskExecuteRequest(BaseModel):
    task_id: str


class CampaignTaskUpdateRequest(BaseModel):
    content: Optional[str] = None
    status: str


class CampaignTaskResponse(BaseModel):
    id: str
    task_type: str
    target_format: Optional[str]
    target_asset: Optional[str]
    status: str
    result: Optional[str]
    image_url: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str


class CampaignResponse(BaseModel):
    id: str
    blog_id: str
    created_at: str
    tasks: List[CampaignTaskResponse]