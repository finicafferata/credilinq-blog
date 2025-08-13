"""
Web scraping service for competitor content monitoring.
Safely extracts content from competitor websites.
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import hashlib
import re
from dataclasses import dataclass
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Represents scraped content from a website."""
    url: str
    title: str
    content: str
    author: Optional[str]
    published_date: Optional[datetime]
    content_type: str  # 'blog_post', 'product_update', 'press_release', etc.
    metadata: Dict[str, Any]
    content_hash: str
    discovered_at: datetime

class CompetitorWebScraper:
    """Web scraper for competitor content monitoring."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limits = {}  # domain -> last_request_time
        self.min_delay = 2.0  # Minimum delay between requests to same domain
        self.max_concurrent = 5  # Maximum concurrent requests
        self.timeout = 30  # Request timeout in seconds
        self.min_content_length = 80  # Accept slightly shorter articles
        
        # User agent rotation for ethical scraping
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.user_agents[0]}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def _respect_rate_limit(self, domain: str):
        """Respect rate limiting for domain."""
        if domain in self.rate_limits:
            elapsed = time.time() - self.rate_limits[domain]
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
        
        self.rate_limits[domain] = time.time()
        
    async def _fetch_page(self, url: str, headers: Optional[Dict] = None) -> Optional[str]:
        """Fetch a single page with error handling."""
        try:
            domain = urlparse(url).netloc
            await self._respect_rate_limit(domain)
            
            request_headers = {'User-Agent': self.user_agents[hash(url) % len(self.user_agents)]}
            if headers:
                request_headers.update(headers)
                
            async with self.session.get(url, headers=request_headers) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.info(f"Successfully fetched {url}")
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
        
        return None
        
    def _extract_blog_content(self, soup: BeautifulSoup, url: str) -> Optional[ScrapedContent]:
        """Extract blog post content from HTML."""
        try:
            # Common blog post selectors
            content_selectors = [
                'article',
                '.post-content',
                '.entry-content', 
                '.content',
                '.post-body',
                '.article-content',
                'main article',
                '[role="main"] article',
                'main',
                '[class*="prose"]',
                '[class*="rich-text"]',
                '[class*="article"]',
                '[class*="content"]'
            ]
            
            title_selectors = [
                'h1',
                '.post-title',
                '.entry-title',
                '.article-title',
                'header h1',
                'title'
            ]
            
            # Extract title
            title = None
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
                    
            if not title:
                title = soup.find('title')
                title = title.get_text().strip() if title else url
                
            # Extract main content
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script, style, and navigation elements
                    for elem in content_elem(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        elem.decompose()
                    # Prefer concatenating paragraphs for better structure
                    paragraphs = [p.get_text().strip() for p in content_elem.find_all(['p', 'li']) if p.get_text().strip()]
                    if len(paragraphs) >= 2:
                        content = "\n\n".join(paragraphs)
                    else:
                        content = content_elem.get_text().strip()
                    break
                    
            if not content:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    for elem in body(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        elem.decompose()
                    # Gather paragraphs from body as a last resort
                    body_paras = [p.get_text().strip() for p in body.find_all('p') if p.get_text().strip()]
                    content = "\n\n".join(body_paras) if body_paras else body.get_text().strip()
                    
            # If content is still too short, try JSON-LD (many SPA sites embed articleBody)
            if not content or len(content) < self.min_content_length:
                try:
                    for script in soup.find_all('script', type='application/ld+json'):
                        import json
                        data = json.loads(script.string or '{}')
                        # Handle list or single object
                        candidates = data if isinstance(data, list) else [data]
                        for obj in candidates:
                            types = obj.get('@type')
                            if not types:
                                continue
                            if isinstance(types, list):
                                is_article = any(t.lower().endswith('article') for t in types if isinstance(t, str))
                            else:
                                is_article = isinstance(types, str) and types.lower().endswith('article')
                            if not is_article:
                                continue
                            article_body = obj.get('articleBody') or obj.get('description')
                            if article_body and len(article_body) > 40:
                                content = article_body.strip()
                                # Prefer headline from JSON-LD if available
                                headline = obj.get('headline') or obj.get('name')
                                if headline:
                                    title = headline.strip()
                                # Published date
                                pub = obj.get('datePublished') or obj.get('dateCreated')
                                if pub:
                                    try:
                                        published_date = self._parse_date(pub)
                                    except Exception:
                                        pass
                                # Author
                                auth = obj.get('author')
                                if isinstance(auth, dict):
                                    author = auth.get('name')
                                elif isinstance(auth, list) and auth and isinstance(auth[0], dict):
                                    author = auth[0].get('name')
                                break
                except Exception:
                    pass

            # As último fallback, usa meta descriptions si el contenido es muy corto
            if not content or len(content) < self.min_content_length:
                og_desc = soup.find('meta', attrs={'property': 'og:description'})
                if og_desc and og_desc.get('content'):
                    content = (content + "\n\n" if content else "") + og_desc.get('content').strip()
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    content = (content + "\n\n" if content else "") + meta_desc.get('content').strip()

            # Extract metadata
            author = None
            author_selectors = [
                '.author',
                '.post-author',
                '.by-author',
                '[rel="author"]',
                '.article-author'
            ]
            
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = author_elem.get_text().strip()
                    break
                    
            # Extract published date
            published_date = None
            date_selectors = [
                'time[datetime]',
                '.post-date',
                '.published',
                '.date',
                '.entry-date'
            ]
            
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_text = date_elem.get('datetime') or date_elem.get_text().strip()
                    try:
                        # Try to parse various date formats
                        published_date = self._parse_date(date_text)
                        break
                    except:
                        continue
                        
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Extract additional metadata
            metadata = {
                'meta_description': '',
                'meta_keywords': '',
                'word_count': len(content.split()),
                'has_images': len(soup.find_all('img')) > 0,
                'has_videos': len(soup.find_all(['video', 'iframe'])) > 0,
                'domain': urlparse(url).netloc
            }
            
            # Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata['meta_description'] = meta_desc.get('content', '')
                
            # Meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata['meta_keywords'] = meta_keywords.get('content', '')
                
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                author=author,
                published_date=published_date,
                content_type='blog_post',
                metadata=metadata,
                content_hash=content_hash,
                discovered_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
            
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse date from various formats."""
        import dateutil.parser
        try:
            return dateutil.parser.parse(date_string)
        except:
            return None
            
    async def _discover_content_urls(self, base_url: str, content_types: List[str]) -> List[str]:
        """Discover content URLs from website."""
        urls = []
        
        try:
            html = await self._fetch_page(base_url)
            if not html:
                return urls
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Common patterns for content discovery
            link_patterns = {
                'blog_post': ['/blog/', '/news/', '/articles/', '/posts/', '/insights/', '/learn/', '/resources/'],
                'press_release': ['/press/', '/news/', '/media/', '/newsroom/', '/press-releases/'],
                'case_study': ['/case-studies/', '/customers/', '/success/', '/stories/', '/case-study/'],
                'product_update': ['/updates/', '/changelog/', '/releases/', '/product/', '/features/']
            }
            
            # Try direct navigation to common blog/content sections
            direct_paths = ['/blog', '/news', '/insights', '/resources', '/learn', '/press']
            for path in direct_paths:
                test_url = urljoin(base_url, path)
                try:
                    # Quick check if path exists
                    test_html = await self._fetch_page(test_url)
                    if test_html:
                        test_soup = BeautifulSoup(test_html, 'html.parser')
                        # Look for article links on this page
                        for link in test_soup.find_all('a', href=True):
                            href = link['href']
                            if href.startswith(path) or any(pattern in href.lower() for pattern_list in link_patterns.values() for pattern in pattern_list):
                                full_url = urljoin(base_url, href)
                                urls.append(full_url)
                except:
                    continue
            
            # Find all links on main page
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                # Check if URL matches content patterns
                for content_type in content_types:
                    if content_type in link_patterns:
                        for pattern in link_patterns[content_type]:
                            if pattern in href.lower():
                                urls.append(full_url)
                                break
                                
            # Also look for links with certain text patterns
            content_link_texts = [
                'blog', 'blog-posts', 'article', 'news', 'press release', 'case study', 
                'learn', 'insights', 'resources', 'updates', 'stories'
            ]
            
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().lower().strip()
                if any(pattern in link_text for pattern in content_link_texts):
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    urls.append(full_url)
                                
            # Deduplicate and limit
            urls = list(set(urls))
            
            # Filter out common non-content URLs
            filtered_urls = []
            skip_patterns = ['/contact', '/about', '/privacy', '/terms', '/jobs', '/careers', '/login', '/signup']
            for url in urls:
                if not any(skip in url.lower() for skip in skip_patterns):
                    filtered_urls.append(url)
            
            # Limit to reasonable number
            urls = filtered_urls[:30]  # Limit to 30 URLs per scan
            
            logger.info(f"Discovered {len(urls)} potential content URLs from {base_url}")
            
        except Exception as e:
            logger.error(f"Error discovering URLs from {base_url}: {str(e)}")
            
        return urls
        
    async def scrape_competitor_content(
        self, 
        competitor_domain: str, 
        content_types: List[str] = None,
        max_pages: int = 20
    ) -> List[ScrapedContent]:
        """Scrape content from competitor website."""
        if content_types is None:
            content_types = ['blog_post', 'press_release', 'product_update']
            
        scraped_content = []
        
        try:
            # Ensure domain has protocol
            if not competitor_domain.startswith(('http://', 'https://')):
                competitor_domain = f"https://{competitor_domain}"
                
            logger.info(f"Starting content scraping for {competitor_domain}")
            
            # Discover content URLs
            content_urls = await self._discover_content_urls(competitor_domain, content_types)
            
            # Limit the number of pages to scrape
            content_urls = content_urls[:max_pages]
            
            # Scrape each URL
            for url in content_urls:
                try:
                    html = await self._fetch_page(url)
                    if html:
                        soup = BeautifulSoup(html, 'html.parser')
                        content = self._extract_blog_content(soup, url)
                        # If initial extraction is too short, try AMP/static variants
                        if not content or len(content.content) <= self.min_content_length:
                            amp_candidates = []
                            if not url.endswith('/amp'):
                                amp_candidates.append(url.rstrip('/') + '/amp')
                            amp_candidates.append(url + ('&' if '?' in url else '?') + 'output=amp')
                            for amp_url in amp_candidates:
                                amp_html = await self._fetch_page(amp_url)
                                if amp_html:
                                    amp_soup = BeautifulSoup(amp_html, 'html.parser')
                                    content = self._extract_blog_content(amp_soup, url)
                                    if content and len(content.content) > self.min_content_length:
                                        break
                        if content and len(content.content) > self.min_content_length:  # Minimum content length
                            scraped_content.append(content)
                            
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping competitor {competitor_domain}: {str(e)}")
            
        logger.info(f"Scraped {len(scraped_content)} content items from {competitor_domain}")
        return scraped_content
        
    async def monitor_competitor_feeds(self, competitor_domain: str) -> List[ScrapedContent]:
        """Monitor RSS/Atom feeds for new content."""
        feed_urls = [
            f"{competitor_domain}/feed",
            f"{competitor_domain}/rss",
            f"{competitor_domain}/atom.xml",
            f"{competitor_domain}/blog/feed",
            f"{competitor_domain}/news/feed"
        ]
        
        content = []
        
        for feed_url in feed_urls:
            try:
                html = await self._fetch_page(feed_url)
                if html and ('<?xml' in html or '<rss' in html or '<feed' in html):
                    # Parse RSS/Atom feed
                    feed_content = await self._parse_feed(html, competitor_domain)
                    content.extend(feed_content)
                    break  # Found valid feed
                    
            except Exception as e:
                logger.debug(f"Feed not found at {feed_url}: {str(e)}")
                continue
                
        return content
        
    async def _parse_feed(self, feed_xml: str, base_domain: str) -> List[ScrapedContent]:
        """Parse RSS/Atom feed content."""
        import feedparser
        
        content = []
        try:
            feed = feedparser.parse(feed_xml)
            
            for entry in feed.entries[:10]:  # Limit to latest 10 entries
                try:
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6])
                        
                    content_text = ""
                    if hasattr(entry, 'content') and entry.content:
                        content_text = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content_text = entry.summary
                        
                    # Clean HTML from content
                    soup = BeautifulSoup(content_text, 'html.parser')
                    content_text = soup.get_text().strip()
                    
                    content_hash = hashlib.md5(content_text.encode('utf-8')).hexdigest()
                    
                    scraped_item = ScrapedContent(
                        url=entry.link,
                        title=entry.title,
                        content=content_text,
                        author=getattr(entry, 'author', None),
                        published_date=published_date,
                        content_type='blog_post',
                        metadata={
                            'source': 'rss_feed',
                            'domain': urlparse(base_domain).netloc,
                            'word_count': len(content_text.split())
                        },
                        content_hash=content_hash,
                        discovered_at=datetime.utcnow()
                    )
                    
                    content.append(scraped_item)
                    
                except Exception as e:
                    logger.error(f"Error parsing feed entry: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing feed: {str(e)}")
            
        return content

# Global scraper instance
web_scraper = CompetitorWebScraper()