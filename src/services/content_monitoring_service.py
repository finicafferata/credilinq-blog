"""
Content monitoring service for competitor intelligence.
Coordinates web scraping and content storage.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .web_scraper import CompetitorWebScraper, ScrapedContent
import uuid
from .webhook_service import webhook_service
from .competitor_intelligence_db import ci_db
from .alert_service import alert_service
import psycopg2.extras
from ..config.database import db_config

logger = logging.getLogger(__name__)

class ContentMonitoringService:
    """Service for monitoring and storing competitor content."""
    
    def __init__(self):
        self.scraper = CompetitorWebScraper()
        
    async def monitor_competitor(self, competitor_id: str) -> Dict[str, Any]:
        """Monitor a single competitor for new content."""
        try:
            # Get competitor details
            competitor = await ci_db.get_competitor(competitor_id)
            if not competitor:
                raise ValueError(f"Competitor {competitor_id} not found")
                
            results = {
                'competitor_id': competitor_id,
                'competitor_name': competitor['name'],
                'new_content_count': 0,
                'updated_content_count': 0,
                'errors': [],
                'monitoring_time': datetime.utcnow().isoformat()
            }
            
            async with self.scraper:
                # Monitor website content
                website_content = await self.scraper.scrape_competitor_content(
                    competitor['domain'],
                    content_types=['blog_post', 'press_release', 'product_update'],
                    max_pages=10
                )
                
                # Monitor RSS feeds
                feed_content = await self.scraper.monitor_competitor_feeds(
                    competitor['domain']
                )
                
                all_content = website_content + feed_content
                
                # Store new content in database
                for content in all_content:
                    try:
                        stored = await self._store_content_item(competitor_id, content)
                        if stored == 'new':
                            results['new_content_count'] += 1
                        elif stored == 'updated':
                            results['updated_content_count'] += 1
                    except Exception as e:
                        error_msg = f"Error storing content {content.url}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)

                # If nothing was discovered via HTML or feeds, fallback: scan the blog index directly
                if results['new_content_count'] == 0 and results['updated_content_count'] == 0 and competitor.get('domain'):
                    try:
                        fallback_urls = [
                            f"{competitor['domain'].rstrip('/')}/blog",
                            f"{competitor['domain'].rstrip('/')}/blog/news",
                        ]
                        for u in fallback_urls:
                            html = await self.scraper._fetch_page(u)  # internal helper, safe here
                            if not html:
                                continue
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(html, 'html.parser')
                            # collect links inside blog listing
                            for a in soup.find_all('a', href=True):
                                href = a['href']
                                if '/blog' in href and len(href) > 10:
                                    full = href if href.startswith('http') else competitor['domain'].rstrip('/') + '/' + href.lstrip('/')
                                    page = await self.scraper._fetch_page(full)
                                    if not page:
                                        continue
                                    psoup = BeautifulSoup(page, 'html.parser')
                                    item = self.scraper._extract_blog_content(psoup, full)
                                    if item and len(item.content) > 100:
                                        status = await self._store_content_item(competitor_id, item)
                                        if status == 'new':
                                            results['new_content_count'] += 1
                                        elif status == 'updated':
                                            results['updated_content_count'] += 1
                    except Exception:
                        pass
                        
                # Generate alerts for new content
                if all_content:
                    try:
                        alert_ids = await alert_service.analyze_content_for_alerts(competitor_id, all_content)
                        results['alerts_generated'] = len(alert_ids)
                    except Exception as e:
                        error_msg = f"Error generating alerts: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                # Update competitor last monitored time
                await ci_db.update_competitor(competitor_id, {
                    'last_monitored': datetime.utcnow().isoformat()
                })
                
            logger.info(f"Monitored {competitor['name']}: {results['new_content_count']} new, {results['updated_content_count']} updated, {results.get('alerts_generated', 0)} alerts")
            return results
            
        except Exception as e:
            error_msg = f"Error monitoring competitor {competitor_id}: {str(e)}"
            logger.error(error_msg)
            return {
                'competitor_id': competitor_id,
                'error': error_msg,
                'monitoring_time': datetime.utcnow().isoformat()
            }
            
    async def _store_content_item(self, competitor_id: str, content: ScrapedContent) -> str:
        """Store content item in database. Returns 'new', 'updated', or 'exists'."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Check if content already exists (by URL or content hash)
                    cur.execute("""
                        SELECT id, content, updated_at, content_hash
                        FROM ci_content_items 
                        WHERE competitor_id = %s 
                          AND (url = %s OR content_hash = %s)
                        ORDER BY discovered_at DESC
                        LIMIT 1
                    """, (competitor_id, content.url, content.content_hash))
                    
                    existing = cur.fetchone()
                    
                    if existing:
                        # Check if content has changed
                        if existing.get('content_hash') != content.content_hash:
                            # Update existing content
                            cur.execute("""
                                UPDATE ci_content_items 
                                SET 
                                    title = %s,
                                    content = %s,
                                    content_hash = %s,
                                    updated_at = NOW(),
                                    metadata = %s
                                WHERE id = %s
                                RETURNING id
                            """, (
                                content.title,
                                content.content,
                                content.content_hash,
                                psycopg2.extras.Json({
                                    **content.metadata,
                                    'last_updated': datetime.utcnow().isoformat(),
                                    'update_reason': 'content_changed'
                                }),
                                existing['id']
                            ))
                            conn.commit()
                            return 'updated'
                        else:
                            cur.execute("""
                                UPDATE ci_content_items 
                                SET updated_at = NOW()
                                WHERE id = %s
                            """, (existing['id'],))
                            conn.commit()
                            return 'exists'
                    else:
                        # Insert new content (snake_case schema + JSON metadata + updated_at NOW)
                        cur.execute("""
                            INSERT INTO ci_content_items (
                                id, competitor_id, title, content, content_type, 
                                platform, url, published_at, discovered_at, 
                                author, keywords, 
                                sentiment_score, quality_score, metadata, content_hash,
                                updated_at
                            ) VALUES (
                                %s, %s, %s, %s, %s::"CIContentType", 
                                %s::"CIPlatform", %s, %s, %s, 
                                %s, %s, 
                                %s, %s, %s, %s,
                                NOW()
                            )
                            RETURNING id
                        """, (
                            str(uuid.uuid4()),
                            competitor_id,
                            content.title,
                            content.content,
                            content.content_type,
                            self._determine_platform(content.url),
                            content.url,
                            content.published_date,
                            content.discovered_at,
                            content.author,
                            self._extract_keywords(content.content),
                            None,  # sentiment_score - to be calculated later  
                            self._calculate_quality_score(content),
                            psycopg2.extras.Json({
                                **content.metadata,
                                'discovery_method': 'web_scraping',
                                'last_checked': datetime.utcnow().isoformat()
                            }),
                            content.content_hash
                        ))
                        row = cur.fetchone()
                        new_id = row['id'] if row and isinstance(row, dict) else (row[0] if row else None)

                        # Record change event (basic new content event)
                        try:
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
                                    "news" if self._determine_platform(content.url) != "website" else "product",
                                    None,
                                    {
                                        "title": content.title,
                                        "content_id": new_id,
                                        "content_type": content.content_type,
                                    },
                                    content.url,
                                    0.8,
                                    {"detector": "content_monitor"},
                                ),
                            )
                        except Exception:
                            pass

                        conn.commit()
                        return 'new'
                        
        except Exception as e:
            logger.error(f"Database error storing content: {str(e)}")
            raise
            
    def _determine_platform(self, url: str) -> str:
        """Determine platform from URL."""
        url_lower = url.lower()
        if 'linkedin.com' in url_lower:
            return 'linkedin'
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'twitter'
        elif 'medium.com' in url_lower:
            return 'medium'
        elif 'substack.com' in url_lower:
            return 'substack'
        else:
            return 'website'
            
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content using simple frequency analysis."""
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b\w{4,}\b', content.lower())
        
        # Common stop words to exclude
        stop_words = {
            'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 
            'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very',
            'when', 'come', 'here', 'what', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        # Filter out stop words and get most common
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_counts = Counter(filtered_words)
        
        # Return top 10 keywords
        return [word for word, count in word_counts.most_common(10)]
        
    def _calculate_quality_score(self, content: ScrapedContent) -> float:
        """Calculate a quality score for the content."""
        score = 0.0
        
        # Word count factor (longer content generally better, but diminishing returns)
        word_count = content.metadata.get('word_count', 0)
        if word_count > 1000:
            score += 1.0
        elif word_count > 500:
            score += 0.7
        elif word_count > 200:
            score += 0.4
        else:
            score += 0.1
            
        # Has author
        if content.author:
            score += 0.5
            
        # Has publication date
        if content.published_date:
            score += 0.3
            
        # Has images/multimedia
        if content.metadata.get('has_images'):
            score += 0.2
        if content.metadata.get('has_videos'):
            score += 0.3
            
        # Recent content gets bonus
        if content.published_date:
            days_old = (datetime.utcnow() - content.published_date).days
            if days_old <= 7:
                score += 0.5
            elif days_old <= 30:
                score += 0.3
                
        # Normalize to 0-10 scale
        return min(score * 3.0, 10.0)
        
    async def monitor_all_active_competitors(self) -> Dict[str, Any]:
        """Monitor all active competitors."""
        try:
            # Get all active competitors
            competitors = await ci_db.list_competitors(active_only=True)
            
            results = {
                'total_competitors': len(competitors),
                'successful_monitoring': 0,
                'failed_monitoring': 0,
                'total_new_content': 0,
                'total_updated_content': 0,
                'monitoring_start': datetime.utcnow().isoformat(),
                'competitor_results': [],
                'errors': []
            }
            
            # Monitor each competitor
            for competitor in competitors:
                try:
                    competitor_result = await self.monitor_competitor(competitor['id'])
                    results['competitor_results'].append(competitor_result)
                    
                    if 'error' in competitor_result:
                        results['failed_monitoring'] += 1
                        results['errors'].append(competitor_result['error'])
                    else:
                        results['successful_monitoring'] += 1
                        results['total_new_content'] += competitor_result.get('new_content_count', 0)
                        results['total_updated_content'] += competitor_result.get('updated_content_count', 0)
                        
                except Exception as e:
                    error_msg = f"Failed to monitor {competitor.get('name', 'Unknown')}: {str(e)}"
                    logger.error(error_msg)
                    results['failed_monitoring'] += 1
                    results['errors'].append(error_msg)
                    
            results['monitoring_end'] = datetime.utcnow().isoformat()
            
            logger.info(f"Completed monitoring: {results['successful_monitoring']}/{results['total_competitors']} successful")
            return results
            
        except Exception as e:
            error_msg = f"Error in batch monitoring: {str(e)}"
            logger.error(error_msg)
            return {
                'error': error_msg,
                'monitoring_time': datetime.utcnow().isoformat()
            }
            
    async def get_competitor_content(
        self, 
        competitor_id: str, 
        content_type: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get stored content for a competitor."""
        try:
            with db_config.get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Build query (snake_case schema)
                    query = """
                        SELECT * FROM ci_content_items 
                        WHERE competitor_id = %s 
                        AND discovered_at >= %s - INTERVAL '%s days'
                    """
                    params = [competitor_id, datetime.utcnow(), days_back]
                    
                    if content_type:
                        query += " AND content_type = %s"
                        params.append(content_type)
                        
                    query += " ORDER BY discovered_at DESC LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    content_items = []
                    for result in results:
                        item = dict(result)
                        # Format datetime objects
                        for field in ['published_at', 'discovered_at', 'updated_at']:
                            if item.get(field):
                                item[field] = item[field].isoformat()
                        content_items.append(item)
                        
                    return content_items
                    
        except Exception as e:
            logger.error(f"Error retrieving content for competitor {competitor_id}: {str(e)}")
            return []

# Global service instance
content_monitoring_service = ContentMonitoringService()