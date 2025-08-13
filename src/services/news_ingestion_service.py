"""
News Ingestion Service
Fetches external news (Google News RSS) for competitors and stores events.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import urllib.parse

import aiohttp
import feedparser
import psycopg2.extras

from ..config.database import db_config


logger = logging.getLogger(__name__)


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


class NewsIngestionService:
    async def ingest_for_competitor(self, competitor_id: str, days_back: int = 7) -> Dict[str, Any]:
        comp = self._get_competitor(competitor_id)
        if not comp:
            raise ValueError("Competitor not found")

        name = comp["name"].strip()
        domain = comp["domain"].replace("https://", "").replace("http://", "").strip('/ ')

        # Query both by name and company domain brand
        queries = [name]
        # Add quoted name if it contains spaces
        if ' ' in name:
            queries.append(f'"{name}"')
        # Include domain-based query to catch brand mentions
        queries.append(domain.split('/')[0])

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        total_inserted = 0
        items: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession() as session:
            for q in queries:
                feed_url = GOOGLE_NEWS_RSS.format(query=urllib.parse.quote_plus(q))
                try:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            logger.warning(f"Google News RSS error {resp.status} for query {q}")
                            continue
                        xml = await resp.text()
                        feed = feedparser.parse(xml)
                except Exception as e:
                    logger.debug(f"RSS fetch failed for {q}: {e}")
                    continue

                for entry in feed.entries[:15]:
                    try:
                        published = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published = datetime(*entry.updated_parsed[:6])
                        if published and published < cutoff:
                            continue

                        title = entry.title
                        link = entry.link
                        source = getattr(entry, 'source', {}).get('title') if hasattr(entry, 'source') else None

                        # Deduplicate by URL
                        if self._news_exists(competitor_id, link):
                            continue

                        self._insert_news_event(
                            competitor_id=competitor_id,
                            title=title,
                            url=link,
                            published_at=published.isoformat() if published else None,
                            source_name=source or 'news',
                        )
                        total_inserted += 1
                        items.append({"title": title, "url": link, "publishedAt": published.isoformat() if published else None})
                    except Exception as e:
                        logger.debug(f"Skipping entry due to error: {e}")
                        continue

        return {"success": True, "inserted": total_inserted, "items": items}

    async def ingest_for_all(self, days_back: int = 7) -> Dict[str, Any]:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id FROM ci_competitors WHERE is_active = TRUE ORDER BY created_at DESC")
                ids = [r["id"] for r in cur.fetchall()]
        results = []
        inserted = 0
        for cid in ids:
            try:
                res = await self.ingest_for_competitor(cid, days_back)
                results.append({"competitorId": cid, "inserted": res.get("inserted", 0)})
                inserted += res.get("inserted", 0)
            except Exception as e:
                results.append({"competitorId": cid, "error": str(e)})
        return {"success": True, "total": len(ids), "inserted": inserted, "details": results}

    def _get_competitor(self, competitor_id: str) -> Optional[Dict[str, Any]]:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id, name, domain FROM ci_competitors WHERE id = %s", (competitor_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def _news_exists(self, competitor_id: str, url: str) -> bool:
        with db_config.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM ci_change_events WHERE competitor_id = %s AND change_type = 'news' AND url = %s LIMIT 1",
                    (competitor_id, url),
                )
                return cur.fetchone() is not None

    def _insert_news_event(self, competitor_id: str, title: str, url: str, published_at: Optional[str], source_name: str) -> None:
        with db_config.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ci_change_events (
                        competitor_id, source, change_type, old_value, new_value, url, confidence, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        competitor_id,
                        "news",
                        "news",
                        None,
                        {"title": title, "publishedAt": published_at, "source": source_name},
                        url,
                        0.9,
                        {"detector": "news"},
                    ),
                )


news_ingestion_service = NewsIngestionService()


