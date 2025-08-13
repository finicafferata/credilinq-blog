"""
Lightweight Suggestions API for the editor.
Uses an in-memory store to avoid DB changes for now.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import uuid
from ...config.database import db_config
import logging


router = APIRouter(prefix="/blogs", tags=["suggestions"])
logger = logging.getLogger(__name__)


class SuggestionPosition(BaseModel):
    start: int
    end: int


class SuggestionModel(BaseModel):
    id: str
    author: str
    originalText: str
    suggestedText: str
    reason: str
    timestamp: datetime
    status: str = Field(default="pending", pattern="^(pending|accepted|rejected)$")
    position: SuggestionPosition


class CreateSuggestionRequest(BaseModel):
    author: str = Field(default="AI Editor")
    originalText: str
    suggestedText: str
    reason: str
    position: SuggestionPosition


# In-memory store: blog_id -> List[SuggestionModel]
_SUGGESTIONS_STORE: Dict[str, List[Dict[str, Any]]] = {}


@router.get("/{blog_id}/suggestions", response_model=List[SuggestionModel])
def list_suggestions(blog_id: str):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, blog_id, author, original_text, suggested_text, reason, timestamp, status, position_start, position_end
                FROM public.blog_suggestion
                WHERE blog_id = %s
                ORDER BY timestamp DESC
                """,
                (str(blog_uuid),),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r[0]),
                    "author": r[2],
                    "originalText": r[3],
                    "suggestedText": r[4],
                    "reason": r[5],
                    "timestamp": r[6],
                    "status": r[7],
                    "position": {"start": r[8], "end": r[9]},
                }
                for r in rows
            ]
    except Exception as e:
        logger.exception("Error listing suggestions for blog_id=%s", blog_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{blog_id}/suggestions", response_model=SuggestionModel)
def add_suggestion(blog_id: str, request: CreateSuggestionRequest):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            sid = uuid.uuid4()
            cur.execute(
                """
                INSERT INTO public.blog_suggestion
                (id, blog_id, author, original_text, suggested_text, reason, timestamp, status, position_start, position_end)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), 'pending', %s, %s)
                """,
                (
                    str(sid),
                    str(blog_uuid),
                    request.author or "AI Editor",
                    request.originalText,
                    request.suggestedText,
                    request.reason,
                    request.position.start,
                    request.position.end,
                ),
            )
            conn.commit()
            return {
                "id": str(sid),
                "author": request.author or "AI Editor",
                "originalText": request.originalText,
                "suggestedText": request.suggestedText,
                "reason": request.reason,
                "timestamp": datetime.utcnow(),
                "status": "pending",
                "position": request.position.dict(),
            }
    except Exception as e:
        logger.exception("Error adding suggestion for blog_id=%s", blog_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{blog_id}/suggestions/{suggestion_id}/accept", response_model=SuggestionModel)
def accept_suggestion(blog_id: str, suggestion_id: str):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE public.blog_suggestion SET status = 'accepted' WHERE id = %s",
                (suggestion_id,),
            )
            conn.commit()
            # return updated item
            items = list_suggestions(str(blog_uuid))
            for s in items:
                if s["id"] == suggestion_id:
                    return s
            raise HTTPException(status_code=404, detail="Suggestion not found")
    except Exception as e:
        logger.exception("Error accepting suggestion=%s blog_id=%s", suggestion_id, blog_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{blog_id}/suggestions/{suggestion_id}/reject", response_model=SuggestionModel)
def reject_suggestion(blog_id: str, suggestion_id: str):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE public.blog_suggestion SET status = 'rejected' WHERE id = %s",
                (suggestion_id,),
            )
            conn.commit()
            items = list_suggestions(str(blog_uuid))
            for s in items:
                if s["id"] == suggestion_id:
                    return s
            raise HTTPException(status_code=404, detail="Suggestion not found")
    except Exception as e:
        logger.exception("Error rejecting suggestion=%s blog_id=%s", suggestion_id, blog_id)
        raise HTTPException(status_code=500, detail=str(e))

