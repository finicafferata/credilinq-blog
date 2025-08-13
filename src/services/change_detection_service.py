"""
Change Detection Service
Detects pricing/plan changes on competitor websites and records events.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import difflib

import psycopg2.extras
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from ..config.database import db_config
from .web_scraper import CompetitorWebScraper


logger = logging.getLogger(__name__)


@dataclass
class PricingPlan:
    name: str
    price: Optional[str]
    period: Optional[str]
    features: List[str]


class ChangeDetectionService:
    async def detect_pricing_for_competitor(self, competitor_id: str) -> Dict[str, Any]:
        """Fetch competitor domain and run pricing detection."""
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id, domain, name FROM ci_competitors WHERE id = %s", (competitor_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError("Competitor not found")
                domain = row["domain"]
                comp_name = row["name"]

        result = await self._detect_pricing(domain)
        if result.get("plans"):
            await self._upsert_pricing_event(competitor_id, result)
        return {"competitorId": competitor_id, "competitorName": comp_name, **result}

    async def detect_pricing_for_all(self) -> Dict[str, Any]:
        """Run detection for all active competitors."""
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id FROM ci_competitors WHERE is_active = TRUE ORDER BY created_at DESC")
                ids = [r["id"] for r in cur.fetchall()]

        successes = 0
        failures = 0
        details = []
        for cid in ids:
            try:
                res = await self.detect_pricing_for_competitor(cid)
                details.append({"competitorId": cid, "status": "ok", "plans": len(res.get("plans", []))})
                successes += 1
            except Exception as e:
                details.append({"competitorId": cid, "status": "error", "error": str(e)})
                failures += 1
        return {"success": True, "processed": len(ids), "successes": successes, "failures": failures, "details": details}

    async def _detect_pricing(self, domain_or_url: str) -> Dict[str, Any]:
        """Try common pricing paths and extract plan cards."""
        pricing_paths = ["/pricing", "/plans", "/pricing-plans", "/pricing/", "/plans/"]
        if not domain_or_url.startswith(("http://", "https://")):
            base = f"https://{domain_or_url}"
        else:
            base = domain_or_url

        async with CompetitorWebScraper() as scraper:
            html: Optional[str] = None
            used_url: Optional[str] = None
            for path in pricing_paths:
                test_url = urljoin(base, path)
                html = await scraper._fetch_page(test_url)
                if html:
                    used_url = test_url
                    break

        if not html:
            return {"success": False, "message": "Pricing page not found", "plans": []}

        soup = BeautifulSoup(html, 'html.parser')
        plans = self._extract_plans(soup)
        return {"success": True, "url": used_url, "plans": [p.__dict__ for p in plans], "detectedAt": datetime.utcnow().isoformat()}

    def _extract_plans(self, soup: BeautifulSoup) -> List[PricingPlan]:
        """Generic extraction: find cards containing price patterns and headings."""
        cards = []
        selectors = [
            '.pricing', '.plan', '.tier', '.card', '.pricing-card', '.plan-card', '.package'
        ]
        price_regex = r"\$\s?\d+[\d,\.]*|\d+[\d,\.]*\s?USD|free|custom|starting"

        # Collect potential plan containers
        containers = set()
        for sel in selectors:
            for elem in soup.select(sel):
                containers.add(elem)

        for elem in containers:
            text = elem.get_text(" ", strip=True).lower()
            if len(text) < 10:
                continue
            # Find a heading for plan name
            name = None
            for h in ['h1', 'h2', 'h3', '.plan-name', '.tier-name']:
                found = elem.select_one(h) if h.startswith('.') else elem.find(h)
                if found and 2 <= len(found.get_text(strip=True)) <= 50:
                    name = found.get_text(strip=True)
                    break
            # Find price
            import re as _re
            price_match = _re.search(price_regex, text)
            price = price_match.group(0) if price_match else None
            # Period
            period = None
            if 'per month' in text or '/month' in text:
                period = 'month'
            elif 'per year' in text or '/year' in text:
                period = 'year'
            # Features
            features = []
            ul = elem.find('ul')
            if ul:
                for li in ul.find_all('li'):
                    val = li.get_text(strip=True)
                    if val:
                        features.append(val)
            if name or price or features:
                cards.append(PricingPlan(name=name or 'Plan', price=price, period=period, features=features[:10]))
        # De-duplicate by (name, price)
        uniq = {}
        for p in cards:
            key = (p.name or "Plan", p.price or "")
            if key not in uniq:
                uniq[key] = p
        return list(uniq.values())

    async def _upsert_pricing_event(self, competitor_id: str, result: Dict[str, Any]) -> None:
        """Compare with last pricing event and insert new event if changed."""
        new_value = {"plans": result.get("plans", []), "url": result.get("url")}

        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, new_value, detected_at
                    FROM ci_change_events
                    WHERE competitor_id = %s AND change_type = 'pricing'
                    ORDER BY detected_at DESC LIMIT 1
                    """,
                    (competitor_id,),
                )
                prev = cur.fetchone()

                # Simple comparison
                prev_val = prev.get("new_value") if prev else None
                if prev_val and json.dumps(prev_val, sort_keys=True) == json.dumps(new_value, sort_keys=True):
                    return  # No change

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
                        "website",
                        "pricing",
                        prev_val,
                        new_value,
                        result.get("url"),
                        0.95,
                        {"detector": "pricing"},
                    ),
                )


    # ---- Copy / Features changes ----
    async def detect_copy_for_competitor(self, competitor_id: str) -> Dict[str, Any]:
        """Detect homepage/features copy changes for a competitor."""
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id, domain, name FROM ci_competitors WHERE id = %s", (competitor_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError("Competitor not found")
                domain = row["domain"]

        base = domain if domain.startswith(("http://", "https://")) else f"https://{domain}"
        paths = ["/", "/features", "/product", "/solutions"]
        async with CompetitorWebScraper() as scraper:
            results = []
            for path in paths:
                url = urljoin(base, path)
                html = await scraper._fetch_page(url)
                if not html:
                    continue
                soup = BeautifulSoup(html, 'html.parser')
                text = self._extract_main_text(soup)
                if len(text) < 200:
                    continue
                changed, summary, prev_hash, new_hash = await self._compare_and_record_copy(competitor_id, url, text)
                results.append({
                    "url": url,
                    "changed": changed,
                    "prev_hash": prev_hash,
                    "new_hash": new_hash,
                    "summary": summary,
                })
        return {"success": True, "pages": results}

    async def detect_copy_for_all(self) -> Dict[str, Any]:
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id FROM ci_competitors WHERE is_active = TRUE ORDER BY created_at DESC")
                ids = [r["id"] for r in cur.fetchall()]
        processed = []
        for cid in ids:
            try:
                res = await self.detect_copy_for_competitor(cid)
                processed.append({"competitorId": cid, "pages": len(res.get("pages", []))})
            except Exception as e:
                processed.append({"competitorId": cid, "error": str(e)})
        return {"success": True, "processed": processed}

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        # Remove boilerplate elements
        for sel in ['script', 'style', 'nav', 'footer', 'header', 'aside']:
            for el in soup.select(sel):
                el.decompose()
        main = soup.select_one('main') or soup.find('body') or soup
        text = main.get_text(" ", strip=True)
        # Normalize spaces
        return " ".join(text.split())

    async def _compare_and_record_copy(self, competitor_id: str, url: str, text: str):
        new_hash = self._hash_text(text)
        with db_config.get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, new_value, detected_at
                    FROM ci_change_events
                    WHERE competitor_id = %s AND change_type = 'product' AND url = %s
                    ORDER BY detected_at DESC LIMIT 1
                    """,
                    (competitor_id, url),
                )
                prev = cur.fetchone()
                prev_hash = prev.get("new_value", {}).get("hash") if prev else None
                if prev_hash == new_hash:
                    return False, None, prev_hash, new_hash

                # Create diff summary
                prev_text = prev.get("new_value", {}).get("text", "") if prev else ""
                summary = self._summarize_diff(prev_text, text)

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
                        "website",
                        "product",
                        {"hash": prev_hash} if prev_hash else None,
                        {"hash": new_hash, "text": text[:4000]},
                        url,
                        0.85,
                        {"detector": "copy"},
                    ),
                )
                return True, summary, prev_hash, new_hash

    def _summarize_diff(self, old: str, new: str) -> Optional[Dict[str, Any]]:
        if not old:
            return {"type": "baseline", "changes": []}
        old_lines = old.split('. ')
        new_lines = new.split('. ')
        diff = difflib.unified_diff(old_lines, new_lines, lineterm='')
        changes = []
        count = 0
        for line in diff:
            if line.startswith('+ ') and count < 5:
                changes.append({"added": line[2:]})
                count += 1
            elif line.startswith('- ') and count < 5:
                changes.append({"removed": line[2:]})
                count += 1
        return {"type": "diff", "changes": changes}

    def _hash_text(self, text: str) -> str:
        import hashlib as _hashlib
        return _hashlib.md5(text.encode('utf-8')).hexdigest()

change_detection_service = ChangeDetectionService()


