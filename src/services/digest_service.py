"""
Digest service for daily/weekly summaries via Email or Microsoft Teams.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import aiohttp
import psycopg2.extras
from aiosmtplib import SMTP
from email.message import EmailMessage
import os

from ..config.database import db_config

logger = logging.getLogger(__name__)


class DigestService:
    async def subscribe(self, channel: str, address_or_webhook: str, frequency: str = "daily", timezone: str = "UTC") -> Dict[str, Any]:
        subscription_id = str(uuid.uuid4())
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO ci_digest_subscriptions (id, channel, address_or_webhook, frequency, timezone, is_active)
                    VALUES (%s, %s, %s, %s, %s, TRUE)
                    RETURNING *
                    """,
                    (subscription_id, channel, address_or_webhook, frequency, timezone),
                )
                return dict(cur.fetchone())

    async def list_subscriptions(self) -> List[Dict[str, Any]]:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM ci_digest_subscriptions WHERE is_active = TRUE ORDER BY created_at DESC")
                return [dict(r) for r in cur.fetchall()]

    async def send_daily_digests(self) -> None:
        subs = await self.list_subscriptions()
        if not subs:
            return
        tasks = [self._send_digest_to(sub) for sub in subs if sub.get("frequency") == "daily"]
        await asyncio.gather(*tasks)

    async def _collect_digest_items(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT e.*, c.name as competitor_name
                    FROM ci_change_events e
                    JOIN ci_competitors c ON e.competitor_id = c.id
                    WHERE e.detected_at >= %s
                    ORDER BY e.detected_at DESC
                    """,
                    (cutoff,),
                )
                return [dict(r) for r in cur.fetchall()]

    async def _send_digest_to(self, sub: Dict[str, Any]) -> None:
        items = await self._collect_digest_items(24 if sub.get("frequency") == "daily" else 7 * 24)
        if not items:
            return
        # Compose summary text
        lines = [
            f"{i['detected_at'].strftime('%Y-%m-%d %H:%M')} | {i['competitor_name']} | {i['change_type']} | {i.get('url','')}"
            for i in items[:20]
        ]
        body_text = "\n".join(lines)

        if sub["channel"] == "email":
            await self._send_email(sub["address_or_webhook"], "Daily Competitor Digest", body_text)
        elif sub["channel"] == "teams":
            await self._send_teams_card(sub["address_or_webhook"], items[:10])

    async def send_test_digest(self, channel: str, address_or_webhook: str, hours_back: int = 24) -> None:
        items = await self._collect_digest_items(hours_back)
        if channel == "email":
            lines = [
                f"{i['detected_at'].strftime('%Y-%m-%d %H:%M')} | {i['competitor_name']} | {i['change_type']} | {i.get('url','')}"
                for i in items[:20]
            ]
            await self._send_email(address_or_webhook, "Test Competitor Digest", "\n".join(lines) or "No recent items")
        elif channel == "teams":
            await self._send_teams_card(address_or_webhook, items[:10])

    async def _send_email(self, to_addr: str, subject: str, body_text: str) -> None:
        host = os.getenv("SMTP_HOST")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER")
        password = os.getenv("SMTP_PASSWORD")
        sender = os.getenv("EMAIL_FROM", user or "no-reply@example.com")

        if not host or not user or not password:
            logger.warning("SMTP not configured; skipping email digest")
            return

        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body_text)

        smtp = SMTP(hostname=host, port=port, use_tls=False)
        await smtp.connect()
        await smtp.starttls()
        await smtp.login(user, password)
        await smtp.send_message(msg)
        await smtp.quit()

    async def _send_teams_card(self, webhook_url: str, items: List[Dict[str, Any]]) -> None:
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "Daily Competitor Digest",
            "themeColor": "0076D7",
            "title": "Daily Competitor Digest",
            "sections": [
                {
                    "facts": [
                        {
                            "name": f"{i['competitor_name']} {i['change_type']}",
                            "value": i.get("url", "") or ""
                        }
                        for i in items
                    ]
                }
            ]
        }
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=card)


digest_service = DigestService()


