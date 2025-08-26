"""
Content Monitoring Agent for automated competitor content discovery and tracking.
Crawls competitor websites, blogs, and social media for new content.
"""

import asyncio
import aiohttp
import feedparser
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import re
from dataclasses import asdict

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

from ..core.base_agent import BaseAgent
from .models import (
    Competitor, ContentItem, ContentType, Platform, 
    MonitoringConfig, Industry
)
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class ContentMonitoringAgent(BaseAgent):
    """
    Specialized agent for monitoring competitor content across multiple platforms.
    Uses web scraping, RSS feeds, APIs, and AI to discover and analyze content.
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..core.base_agent import AgentMetadata, AgentType
        
        metadata = AgentMetadata(
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,  # Use available type
            name="ContentMonitoringAgent"
        )
        super().__init__(metadata)
        
        # Initialize AI for content analysis (lazy loading)
        self.analysis_llm = None
        
        # Text splitter for large content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        
        # Content discovery patterns
        self.content_patterns = {
            ContentType.BLOG_POST: [
                r'/blog/', r'/articles/', r'/insights/', r'/news/',
                r'/resources/', r'/content/', r'/posts/'
            ],
            ContentType.CASE_STUDY: [
                r'/case-study/', r'/case-studies/', r'/success-stories/',
                r'/customer-stories/', r'/examples/'
            ],
            ContentType.WHITEPAPER: [
                r'/whitepaper/', r'/whitepapers/', r'/reports/',
                r'/research/', r'/guides/', r'/ebooks/'
            ],
            ContentType.PRESS_RELEASE: [
                r'/press/', r'/news/', r'/announcements/',
                r'/media/', r'/press-releases/'
            ]
        }
        
        # Platform-specific handlers
        self.platform_handlers = {
            Platform.WEBSITE: self._monitor_website,
            Platform.LINKEDIN: self._monitor_linkedin,
            Platform.TWITTER: self._monitor_twitter,
            Platform.MEDIUM: self._monitor_medium,
            Platform.YOUTUBE: self._monitor_youtube
        }
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Content cache to avoid duplicates
        self.content_hashes: Set[str] = set()
    
    def _get_analysis_llm(self):
        """Lazy initialize the analysis LLM."""
        if self.analysis_llm is None:
            try:
                self.analysis_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=1000
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI LLM: {e}")
                return None
        return self.analysis_llm
    
    async def start_monitoring_session(self):
        """Initialize monitoring session with proper headers and settings."""
        headers = {
            'User-Agent': 'CrediLinq Content Intelligence Bot 1.0 (+https://credilinq.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=connector
        )
    
    async def close_monitoring_session(self):
        """Clean up monitoring session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def monitor_competitor(
        self,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """
        Monitor a single competitor across all configured platforms.
        Returns newly discovered content items.
        """
        
        async with async_performance_tracker(f"monitor_competitor_{competitor.id}"):
            if not self.session:
                await self.start_monitoring_session()
            
            new_content = []
            
            # Monitor each platform
            for platform in competitor.platforms:
                if platform in self.platform_handlers:
                    try:
                        platform_content = await self.platform_handlers[platform](
                            competitor, config
                        )
                        new_content.extend(platform_content)
                        
                        # Track metrics
                        metrics.increment_counter(
                            "competitor.content.discovered",
                            tags={
                                "competitor": competitor.name,
                                "platform": platform.value,
                                "count": str(len(platform_content))
                            }
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            f"Failed to monitor {platform.value} for {competitor.name}: {str(e)}"
                        )
                        
                        metrics.increment_counter(
                            "competitor.monitoring.failed",
                            tags={
                                "competitor": competitor.name,
                                "platform": platform.value
                            }
                        )
            
            # Update competitor's last monitored timestamp
            competitor.last_monitored = datetime.utcnow()
            
            self.logger.info(
                f"Discovered {len(new_content)} new content items for {competitor.name}"
            )
            
            return new_content
    
    async def _monitor_website(
        self,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Monitor competitor's main website and blog."""
        
        content_items = []
        
        # Check main website
        website_content = await self._scrape_website(
            competitor.domain,
            competitor,
            config
        )
        content_items.extend(website_content)
        
        # Check RSS feeds
        rss_feeds = await self._discover_rss_feeds(competitor.domain)
        for feed_url in rss_feeds:
            feed_content = await self._monitor_rss_feed(
                feed_url,
                competitor,
                config
            )
            content_items.extend(feed_content)
        
        return content_items
    
    async def _scrape_website(
        self,
        domain: str,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Scrape competitor website for new content."""
        
        content_items = []
        
        try:
            # Get sitemap if available
            sitemap_urls = await self._get_sitemap_urls(domain)
            
            # If no sitemap, discover URLs through navigation
            if not sitemap_urls:
                sitemap_urls = await self._discover_content_urls(domain)
            
            # Limit URLs to avoid overwhelming
            sitemap_urls = sitemap_urls[:50]  # Process max 50 URLs per check
            
            # Process each URL
            for url in sitemap_urls:
                try:
                    content_item = await self._process_webpage(
                        url,
                        competitor,
                        config
                    )
                    
                    if content_item and self._is_new_content(content_item):
                        content_items.append(content_item)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to process {url}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Website scraping failed for {domain}: {str(e)}")
        
        return content_items
    
    async def _get_sitemap_urls(self, domain: str) -> List[str]:
        """Retrieve URLs from sitemap."""
        
        sitemap_urls = [
            f"{domain}/sitemap.xml",
            f"{domain}/sitemap_index.xml",
            f"{domain}/robots.txt"  # Check robots.txt for sitemap location
        ]
        
        urls = []
        
        for sitemap_url in sitemap_urls:
            try:
                async with self.session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        if 'sitemap' in sitemap_url:
                            # Parse XML sitemap
                            urls.extend(self._parse_sitemap_xml(content))
                        else:
                            # Parse robots.txt
                            sitemap_lines = [
                                line.split(': ', 1)[1] 
                                for line in content.split('\n') 
                                if line.startswith('Sitemap:')
                            ]
                            
                            for sitemap_line in sitemap_lines:
                                async with self.session.get(sitemap_line.strip()) as sitemap_response:
                                    if sitemap_response.status == 200:
                                        sitemap_content = await sitemap_response.text()
                                        urls.extend(self._parse_sitemap_xml(sitemap_content))
                        
                        break  # Found sitemap, no need to check others
                        
            except Exception as e:
                self.logger.debug(f"Failed to fetch sitemap {sitemap_url}: {str(e)}")
                continue
        
        return urls[:100]  # Limit to 100 URLs
    
    def _parse_sitemap_xml(self, xml_content: str) -> List[str]:
        """Parse XML sitemap to extract URLs."""
        
        urls = []
        
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            
            # Handle sitemap index
            sitemaps = soup.find_all('sitemap')
            if sitemaps:
                for sitemap in sitemaps:
                    loc = sitemap.find('loc')
                    if loc:
                        urls.append(loc.text.strip())
            
            # Handle URL list
            url_tags = soup.find_all('url')
            for url_tag in url_tags:
                loc = url_tag.find('loc')
                if loc:
                    urls.append(loc.text.strip())
                    
        except Exception as e:
            self.logger.debug(f"Failed to parse sitemap XML: {str(e)}")
        
        return urls
    
    async def _discover_content_urls(self, domain: str) -> List[str]:
        """Discover content URLs through navigation discovery."""
        
        discovered_urls = set()
        
        try:
            # Start with main page
            async with self.session.get(domain) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find navigation links
                nav_links = soup.find_all('a', href=True)
                
                for link in nav_links:
                    href = link['href']
                    
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin(domain, href)
                    elif not href.startswith('http'):
                        continue
                    
                    # Check if URL matches content patterns
                    if self._matches_content_patterns(href):
                        discovered_urls.add(href)
                        
                        # Also check this page for more links
                        if len(discovered_urls) < 30:  # Limit discovery depth
                            sub_urls = await self._discover_urls_from_page(href)
                            discovered_urls.update(sub_urls)
        
        except Exception as e:
            self.logger.debug(f"URL discovery failed for {domain}: {str(e)}")
        
        return list(discovered_urls)[:50]
    
    def _matches_content_patterns(self, url: str) -> bool:
        """Check if URL matches known content patterns."""
        
        url_lower = url.lower()
        
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return True
        
        return False
    
    async def _discover_urls_from_page(self, url: str) -> Set[str]:
        """Discover URLs from a specific page."""
        
        discovered_urls = set()
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return discovered_urls
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find links that look like content
                links = soup.find_all('a', href=True)
                
                for link in links[:20]:  # Limit to 20 links per page
                    href = link['href']
                    
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    elif not href.startswith('http'):
                        continue
                    
                    if self._matches_content_patterns(href):
                        discovered_urls.add(href)
        
        except Exception as e:
            self.logger.debug(f"Failed to discover URLs from {url}: {str(e)}")
        
        return discovered_urls
    
    async def _process_webpage(
        self,
        url: str,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> Optional[ContentItem]:
        """Process a single webpage and extract content."""
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract basic information
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                
                if not title or not content or len(content) < 100:
                    return None
                
                # Determine content type based on URL and content
                content_type = self._classify_content_type(url, title, content)
                
                # Extract metadata
                author = self._extract_author(soup)
                published_date = self._extract_published_date(soup)
                
                # Generate content hash for duplicate detection
                content_hash = hashlib.md5(
                    f"{title}{content}".encode('utf-8')
                ).hexdigest()
                
                # Create content item
                content_item = ContentItem(
                    id=f"content_{content_hash}",
                    competitor_id=competitor.id,
                    title=title,
                    content=content,
                    content_type=content_type,
                    platform=Platform.WEBSITE,
                    url=url,
                    published_at=published_date or datetime.utcnow(),
                    discovered_at=datetime.utcnow(),
                    author=author,
                    metadata={
                        'content_hash': content_hash,
                        'word_count': len(content.split()),
                        'competitor_name': competitor.name
                    }
                )
                
                # Analyze content with AI if enabled
                if config.quality_scoring:
                    content_item.quality_score = await self._assess_content_quality(
                        title, content
                    )
                
                if config.sentiment_analysis:
                    content_item.sentiment_score = await self._analyze_sentiment(
                        content
                    )
                
                # Extract keywords
                content_item.keywords = await self._extract_keywords(
                    title, content
                )
                
                return content_item
                
        except Exception as e:
            self.logger.debug(f"Failed to process webpage {url}: {str(e)}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title from webpage."""
        
        # Try different title sources in order of preference
        title_sources = [
            soup.find('h1'),
            soup.find('title'),
            soup.find('meta', property='og:title'),
            soup.find('meta', name='twitter:title')
        ]
        
        for source in title_sources:
            if source:
                if source.name == 'meta':
                    title = source.get('content', '').strip()
                else:
                    title = source.get_text().strip()
                
                if title and len(title) > 10:
                    return title[:200]  # Limit title length
        
        return None
    
    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main content from webpage."""
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content area
        content_selectors = [
            'article',
            '.post-content',
            '.entry-content', 
            '.content',
            'main',
            '.post-body',
            '.article-body'
        ]
        
        for selector in content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                text = content_area.get_text().strip()
                if len(text) > 200:  # Minimum content length
                    return self._clean_text(text)
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = '\n'.join([p.get_text().strip() for p in paragraphs])
            if len(text) > 200:
                return self._clean_text(text)
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return '\n'.join(lines)[:10000]  # Limit content length
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author from webpage."""
        
        author_selectors = [
            '.author',
            '.by-author',
            '[rel="author"]',
            '.post-author',
            '[itemprop="author"]'
        ]
        
        for selector in author_selectors:
            author_element = soup.select_one(selector)
            if author_element:
                author = author_element.get_text().strip()
                if author and len(author) < 100:
                    return author
        
        # Check meta tags
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author:
            return meta_author.get('content', '').strip()
        
        return None
    
    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract published date from webpage."""
        
        # Check structured data
        time_element = soup.find('time')
        if time_element:
            datetime_attr = time_element.get('datetime')
            if datetime_attr:
                try:
                    return datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                except:
                    pass
        
        # Check meta tags
        meta_selectors = [
            {'property': 'article:published_time'},
            {'name': 'publish_date'},
            {'name': 'date'},
            {'property': 'og:published_time'}
        ]
        
        for meta_attrs in meta_selectors:
            meta_tag = soup.find('meta', attrs=meta_attrs)
            if meta_tag:
                date_content = meta_tag.get('content', '')
                try:
                    return datetime.fromisoformat(date_content.replace('Z', '+00:00'))
                except:
                    continue
        
        return None
    
    def _classify_content_type(
        self,
        url: str,
        title: str,
        content: str
    ) -> ContentType:
        """Classify content type based on URL, title, and content."""
        
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Check URL patterns first
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return content_type
        
        # Check title and content for keywords
        if any(keyword in title_lower for keyword in ['case study', 'success story', 'customer story']):
            return ContentType.CASE_STUDY
        
        if any(keyword in title_lower for keyword in ['whitepaper', 'white paper', 'report', 'guide', 'ebook']):
            return ContentType.WHITEPAPER
        
        if any(keyword in title_lower for keyword in ['press release', 'announcement', 'news']):
            return ContentType.PRESS_RELEASE
        
        if any(keyword in content_lower for keyword in ['webinar', 'live session', 'online event']):
            return ContentType.WEBINAR
        
        # Default to blog post
        return ContentType.BLOG_POST
    
    async def _assess_content_quality(
        self,
        title: str,
        content: str
    ) -> float:
        """Assess content quality using AI."""
        
        try:
            # Truncate content for analysis
            analysis_content = content[:2000] if len(content) > 2000 else content
            
            quality_prompt = f"""
            Analyze the quality of this content on a scale of 0-100.
            Consider factors like:
            - Writing quality and clarity
            - Depth of information
            - Originality and insights
            - Structure and organization
            - Value to readers

            Title: {title}
            Content: {analysis_content}

            Return only a number between 0-100.
            """
            
            response = await self.analysis_llm.agenerate([
                [HumanMessage(content=quality_prompt)]
            ])
            
            score_text = response.generations[0][0].text.strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                return min(100, max(0, score))  # Ensure 0-100 range
            
        except Exception as e:
            self.logger.debug(f"Quality assessment failed: {str(e)}")
        
        return 50.0  # Default neutral score
    
    async def _analyze_sentiment(self, content: str) -> float:
        """Analyze content sentiment."""
        
        try:
            # Truncate content for analysis
            analysis_content = content[:1500] if len(content) > 1500 else content
            
            sentiment_prompt = f"""
            Analyze the sentiment of this content.
            Return a number between -1 (very negative) and 1 (very positive).
            0 is neutral.

            Content: {analysis_content}

            Return only a decimal number between -1 and 1.
            """
            
            response = await self.analysis_llm.agenerate([
                [HumanMessage(content=sentiment_prompt)]
            ])
            
            sentiment_text = response.generations[0][0].text.strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'-?\d*\.?\d+', sentiment_text)
            if numbers:
                sentiment = float(numbers[0])
                return max(-1, min(1, sentiment))  # Ensure -1 to 1 range
        
        except Exception as e:
            self.logger.debug(f"Sentiment analysis failed: {str(e)}")
        
        return 0.0  # Default neutral sentiment
    
    async def _extract_keywords(
        self,
        title: str,
        content: str
    ) -> List[str]:
        """Extract keywords from content using AI."""
        
        try:
            # Truncate content for analysis
            analysis_content = content[:1500] if len(content) > 1500 else content
            
            keyword_prompt = f"""
            Extract 10-15 most important keywords/phrases from this content.
            Focus on topics, concepts, and important terms.
            Return as a comma-separated list.

            Title: {title}
            Content: {analysis_content}

            Keywords:
            """
            
            response = await self.analysis_llm.agenerate([
                [HumanMessage(content=keyword_prompt)]
            ])
            
            keywords_text = response.generations[0][0].text.strip()
            
            # Parse keywords
            keywords = [
                keyword.strip().lower() 
                for keyword in keywords_text.split(',')
                if keyword.strip()
            ]
            
            return keywords[:15]  # Limit to 15 keywords
            
        except Exception as e:
            self.logger.debug(f"Keyword extraction failed: {str(e)}")
        
        return []
    
    def _is_new_content(self, content_item: ContentItem) -> bool:
        """Check if content is new (not seen before)."""
        
        content_hash = content_item.metadata.get('content_hash')
        if not content_hash:
            return True
        
        if content_hash in self.content_hashes:
            return False
        
        # Add to cache
        self.content_hashes.add(content_hash)
        
        # Check if content is within age limit
        age_limit = timedelta(days=90)  # Default 90 days
        if datetime.utcnow() - content_item.published_at > age_limit:
            return False
        
        return True
    
    async def _discover_rss_feeds(self, domain: str) -> List[str]:
        """Discover RSS feeds for a domain."""
        
        rss_feeds = []
        
        # Common RSS feed locations
        common_feeds = [
            '/feed',
            '/rss',
            '/blog/feed',
            '/feed.xml',
            '/rss.xml',
            '/atom.xml',
            '/index.xml'
        ]
        
        for feed_path in common_feeds:
            feed_url = urljoin(domain, feed_path)
            
            try:
                async with self.session.get(feed_url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'xml' in content_type or 'rss' in content_type:
                            rss_feeds.append(feed_url)
                            
            except Exception as e:
                self.logger.debug(f"Failed to check RSS feed {feed_url}: {str(e)}")
                continue
        
        return rss_feeds
    
    async def _monitor_rss_feed(
        self,
        feed_url: str,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Monitor RSS feed for new content."""
        
        content_items = []
        
        try:
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    return content_items
                
                feed_content = await response.text()
                
            # Parse RSS feed
            feed = feedparser.parse(feed_content)
            
            for entry in feed.entries[:20]:  # Process max 20 entries
                try:
                    # Extract entry information
                    title = entry.get('title', '')
                    content = self._extract_rss_content(entry)
                    url = entry.get('link', '')
                    
                    if not title or not content or not url:
                        continue
                    
                    # Parse published date
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6])
                    else:
                        published_date = datetime.utcnow()
                    
                    # Generate content hash
                    content_hash = hashlib.md5(
                        f"{title}{content}".encode('utf-8')
                    ).hexdigest()
                    
                    # Create content item
                    content_item = ContentItem(
                        id=f"rss_{content_hash}",
                        competitor_id=competitor.id,
                        title=title,
                        content=content,
                        content_type=ContentType.BLOG_POST,  # Assume blog post for RSS
                        platform=Platform.WEBSITE,
                        url=url,
                        published_at=published_date,
                        discovered_at=datetime.utcnow(),
                        author=entry.get('author', None),
                        metadata={
                            'content_hash': content_hash,
                            'source': 'rss',
                            'feed_url': feed_url,
                            'competitor_name': competitor.name
                        }
                    )
                    
                    if self._is_new_content(content_item):
                        content_items.append(content_item)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to process RSS entry: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.error(f"RSS feed monitoring failed for {feed_url}: {str(e)}")
        
        return content_items
    
    def _extract_rss_content(self, entry) -> str:
        """Extract content from RSS entry."""
        
        # Try different content fields
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if hasattr(entry, field):
                content_data = getattr(entry, field)
                
                if isinstance(content_data, list) and content_data:
                    content_data = content_data[0]
                
                if isinstance(content_data, dict):
                    content = content_data.get('value', '')
                elif isinstance(content_data, str):
                    content = content_data
                else:
                    continue
                
                if content and len(content) > 50:
                    # Clean HTML if present
                    soup = BeautifulSoup(content, 'html.parser')
                    return soup.get_text().strip()
        
        return ''
    
    # Platform-specific monitoring methods (simplified implementations)
    
    async def _monitor_linkedin(
        self,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Monitor LinkedIn for competitor content."""
        # Note: LinkedIn monitoring would require LinkedIn API access
        # This is a placeholder for the implementation
        self.logger.info(f"LinkedIn monitoring not implemented yet for {competitor.name}")
        return []
    
    async def _monitor_twitter(
        self,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Monitor Twitter for competitor content."""
        # Note: Twitter monitoring would require Twitter API access
        # This is a placeholder for the implementation
        self.logger.info(f"Twitter monitoring not implemented yet for {competitor.name}")
        return []
    
    async def _monitor_medium(
        self,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Monitor Medium for competitor content."""
        # Note: Medium has RSS feeds that can be used
        # This is a placeholder for the implementation
        self.logger.info(f"Medium monitoring not implemented yet for {competitor.name}")
        return []
    
    async def _monitor_youtube(
        self,
        competitor: Competitor,
        config: MonitoringConfig
    ) -> List[ContentItem]:
        """Monitor YouTube for competitor content."""
        # Note: YouTube monitoring would require YouTube Data API
        # This is a placeholder for the implementation
        self.logger.info(f"YouTube monitoring not implemented yet for {competitor.name}")
        return []
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics."""
        
        return {
            "active_session": self.session is not None,
            "content_cache_size": len(self.content_hashes),
            "supported_platforms": list(self.platform_handlers.keys()),
            "monitoring_capabilities": ["web_scraping", "rss_monitoring", "social_media_tracking"],
            "last_activity": datetime.utcnow().isoformat()
        }
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the content monitoring agent's main functionality.
        Routes to appropriate monitoring method based on input.
        """
        # For now, return a simple status
        return {
            "status": "ready",
            "agent_type": "content_monitoring",
            "available_operations": [
                "monitor_competitor",
                "start_monitoring_session",
                "close_monitoring_session",
                "get_monitoring_status"
            ]
        }