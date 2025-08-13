"""
Webhook service for outbound notifications of CI change events.
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import uuid
from typing import Dict, Any, List, Optional

import aiohttp
import psycopg2.extras

from ..config.database import db_config

logger = logging.getLogger(__name__)


class WebhookService:
    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # Subscription management
    async def create_subscription(self, name: str, target_url: str, event_types: List[str], secret_hmac: Optional[str]) -> Dict[str, Any]:
        subscription_id = str(uuid.uuid4())
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO ci_webhook_subscriptions (id, name, target_url, secret_hmac, event_types, is_active)
                    VALUES (%s, %s, %s, %s, %s, TRUE)
                    RETURNING *
                    """,
                    (subscription_id, name, target_url, secret_hmac, event_types),
                )
                row = cur.fetchone()
                return dict(row)

    async def list_subscriptions(self) -> List[Dict[str, Any]]:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM ci_webhook_subscriptions ORDER BY created_at DESC")
                return [dict(r) for r in cur.fetchall()]

    async def remove_subscription(self, subscription_id: str) -> bool:
        with db_config.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM ci_webhook_subscriptions WHERE id = %s", (subscription_id,))
                return cur.rowcount > 0

    # Delivery
    async def deliver_event(self, event: Dict[str, Any]) -> None:
        """Deliver event to all active subscribers of its type."""
        event_type = event.get("change_type") or event.get("type") or "change"

        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM ci_webhook_subscriptions
                    WHERE is_active = TRUE AND (event_types = '{}' OR %s = ANY(event_types))
                    """,
                    (event_type,),
                )
                subs = [dict(r) for r in cur.fetchall()]

        if not subs:
            return

        payload = json.dumps(event, default=str).encode("utf-8")
        session = await self._get_session()

        async def send_to(sub: Dict[str, Any]) -> None:
            signature = None
            secret = sub.get("secret_hmac")
            if secret:
                signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

            headers = {
                "Content-Type": "application/json",
                "X-CI-Event-Type": event_type,
                "X-CI-Event-Id": str(event.get("id", "")),
            }
            if signature:
                headers["X-CI-Signature"] = f"sha256={signature}"

            status = "failed"
            http_code: Optional[int] = None
            error: Optional[str] = None

            try:
                async with session.post(sub["target_url"], data=payload, headers=headers) as resp:
                    http_code = resp.status
                    status = "success" if 200 <= resp.status < 300 else "failed"
                    if not (200 <= resp.status < 300):
                        error = await resp.text()
            except Exception as e:
                error = str(e)

            # Log delivery
            with db_config.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ci_delivery_log (subscription_id, event_id, event_type, status, http_code, error)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (sub["id"], str(event.get("id", "")), event_type, status, http_code, error),
                    )

        await asyncio.gather(*(send_to(s) for s in subs))


webhook_service = WebhookService()


