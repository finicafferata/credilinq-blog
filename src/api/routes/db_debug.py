from fastapi import APIRouter, HTTPException
from typing import Any, Dict
from urllib.parse import urlparse
from .. import __init__  # noqa: F401  # ensure package context
from ...config.database import db_config
from ...config.settings import settings

router = APIRouter(prefix="/debug", tags=["debug"], include_in_schema=False)


def _mask_db_url(db_url: str) -> str:
    try:
        u = urlparse(db_url)
        netloc = u.hostname or ""
        if u.port:
            netloc += f":{u.port}"
        return f"{u.scheme}://{netloc}{u.path}"
    except Exception:
        return "unknown"


@router.get("/db")
def debug_db() -> Dict[str, Any]:
    """Return runtime DB diagnostics to quickly identify connection/schema issues."""
    info: Dict[str, Any] = {
        "configured_database_url": _mask_db_url(settings.database_url),
    }
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT current_database(), current_schema(), session_user")
            db_name, schema, user = cur.fetchone()
            cur.execute("SHOW search_path")
            search_path = cur.fetchone()[0]
            # Check presence of our tables via to_regclass
            cur.execute("SELECT to_regclass('public.blog_comment'), to_regclass('public.blog_comment_reply'), to_regclass('public.blog_suggestion')")
            comment_reg, reply_reg, sugg_reg = cur.fetchone()
            info.update({
                "current_database": db_name,
                "current_schema": schema,
                "session_user": user,
                "search_path": search_path,
                "tables": {
                    "public.blog_comment": bool(comment_reg),
                    "public.blog_comment_reply": bool(reply_reg),
                    "public.blog_suggestion": bool(sugg_reg),
                },
            })
            return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

