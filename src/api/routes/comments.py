"""
Lightweight Comments API for the editor.
This uses an in-memory store for now to avoid DB schema changes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import psycopg2
import logging


router = APIRouter(prefix="/blogs", tags=["comments"])
logger = logging.getLogger(__name__)


class CommentPosition(BaseModel):
    start: int
    end: int
    selectedText: str


class CommentModel(BaseModel):
    id: str
    author: str
    content: str
    timestamp: datetime
    resolved: bool = False
    position: Optional[CommentPosition] = None
    replies: Optional[List["CommentModel"]] = None  # recursive type for replies


CommentModel.update_forward_refs()


class CreateCommentRequest(BaseModel):
    author: str = Field(default="You")
    content: str
    position: Optional[CommentPosition] = None


class ReplyRequest(BaseModel):
    author: str = Field(default="You")
    content: str


from ...config.database import db_config


@router.get("/{blog_id}/comments", response_model=List[CommentModel])
def list_comments(blog_id: str):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, blog_id, author, content, timestamp, resolved,
                       position_start, position_end, position_selected_text
                FROM public.blog_comment
                WHERE blog_id = %s
                ORDER BY timestamp DESC
                """,
                (str(blog_uuid),),
            )
            rows = cur.fetchall()
            comments: List[Dict[str, Any]] = []
            for (cid, _bid, author, content, ts, resolved, ps, pe, pst) in rows:
                # Load replies
                cur.execute(
                    """
                    SELECT id, author, content, timestamp
                    FROM public.blog_comment_reply WHERE comment_id = %s ORDER BY timestamp ASC
                    """,
                    (cid,),
                )
                reply_rows = cur.fetchall()
                replies = [
                    {
                        "id": rid,
                        "author": rauthor,
                        "content": rcontent,
                        "timestamp": rts,
                        "resolved": False,
                    }
                    for (rid, rauthor, rcontent, rts) in reply_rows
                ]
                comments.append(
                    {
                        "id": str(cid),
                        "author": author,
                        "content": content,
                        "timestamp": ts,
                        "resolved": resolved,
                        "position": (
                            {"start": ps, "end": pe, "selectedText": pst}
                            if ps is not None and pe is not None and pst is not None
                            else None
                        ),
                        "replies": replies,
                    }
                )
            return comments
    except Exception as e:
        logger.exception("Error listing comments for blog_id=%s", blog_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{blog_id}/comments", response_model=CommentModel)
def add_comment(blog_id: str, request: CreateCommentRequest):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cid = uuid.uuid4()
            ps = request.position.start if request.position else None
            pe = request.position.end if request.position else None
            pst = request.position.selectedText if request.position else None
            cur.execute(
                """
                INSERT INTO public.blog_comment
                (id, blog_id, author, content, timestamp, resolved, position_start, position_end, position_selected_text)
                VALUES (%s, %s, %s, %s, NOW(), FALSE, %s, %s, %s)
                """,
                (str(cid), str(blog_uuid), request.author or "You", request.content, ps, pe, pst),
            )
            conn.commit()
            return {
                "id": str(cid),
                "author": request.author or "You",
                "content": request.content,
                "timestamp": datetime.utcnow(),
                "resolved": False,
                "position": request.position.dict() if request.position else None,
                "replies": [],
            }
    except Exception as e:
        logger.exception("Error adding comment for blog_id=%s", blog_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{blog_id}/comments/{comment_id}/reply", response_model=CommentModel)
def reply_comment(blog_id: str, comment_id: str, request: ReplyRequest):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            rid = uuid.uuid4()
            cur.execute(
                """
                INSERT INTO public.blog_comment_reply (id, comment_id, author, content, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (str(rid), comment_id, request.author or "You", request.content),
            )
            conn.commit()
            # Return the updated parent comment by filtering
            updated = [c for c in list_comments(blog_id=str(blog_uuid)) if c["id"] == comment_id]
            return updated[0] if updated else list_comments(blog_id=str(blog_uuid))
    except Exception as e:
        logger.exception("Error replying to comment=%s blog_id=%s", comment_id, blog_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{blog_id}/comments/{comment_id}/resolve", response_model=CommentModel)
def resolve_comment(blog_id: str, comment_id: str):
    try:
        blog_uuid = uuid.UUID(blog_id)
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE public.blog_comment SET resolved = TRUE WHERE id = %s",
                (comment_id,),
            )
            conn.commit()
            comments = list_comments(str(blog_uuid))
            for c in comments:
                if c["id"] == comment_id:
                    return c
            raise HTTPException(status_code=404, detail="Comment not found")
    except Exception as e:
        logger.exception("Error resolving comment=%s blog_id=%s", comment_id, blog_id)
        raise HTTPException(status_code=500, detail=str(e))

