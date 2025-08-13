"""
Social media monitoring service for competitor intelligence.
Monitors competitor activity across social platforms.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import re
import json
from dataclasses import dataclass
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SocialMediaPost:
    """Represents a social media post."""
    platform: str
    post_id: str
    url: str
    content: str
    author: str
    author_handle: str
    published_at: datetime
    engagement_metrics: Dict[str, Any]
    hashtags: List[str]
    mentions: List[str]
    media_urls: List[str]
    post_type: str  # 'text', 'image', 'video', 'link', 'repost'
    metadata: Dict[str, Any]

class SocialMediaMonitor:
    """Monitor competitor social media activity."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limits = {}
        self.min_delay = 3.0  # Longer delay for social media
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=3)  # Lower concurrent limit
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def _respect_rate_limit(self, platform: str):
        """Respect rate limiting per platform."""
        if platform in self.rate_limits:
            elapsed = asyncio.get_event_loop().time() - self.rate_limits[platform]
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
        
        self.rate_limits[platform] = asyncio.get_event_loop().time()
        
    async def _fetch_page(self, url: str, headers: Optional[Dict] = None) -> Optional[str]:
        """Fetch a page with error handling."""
        try:
            platform = self._get_platform_from_url(url)
            await self._respect_rate_limit(platform)
            
            request_headers = self.session.headers.copy()
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
                    
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
            
    def _get_platform_from_url(self, url: str) -> str:
        """Determine platform from URL."""
        domain = urlparse(url).netloc.lower()
        if 'linkedin.com' in domain:
            return 'linkedin'
        elif 'twitter.com' in domain or 'x.com' in domain:
            return 'twitter'
        elif 'facebook.com' in domain:
            return 'facebook'
        elif 'instagram.com' in domain:
            return 'instagram'
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return 'youtube'
        elif 'medium.com' in domain:
            return 'medium'
        else:
            return 'unknown'
            
    async def monitor_linkedin_company(self, company_handle: str) -> List[SocialMediaPost]:
        """Monitor LinkedIn company page for posts."""
        posts = []
        
        try:
            # LinkedIn public company page URL
            url = f"https://www.linkedin.com/company/{company_handle}/posts/"
            
            html = await self._fetch_page(url)
            if not html:
                return posts
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for post containers (LinkedIn uses dynamic class names)
            post_containers = soup.find_all(['div'], class_=re.compile(r'feed-shared-update-v2'))
            
            for container in post_containers[:10]:  # Limit to recent posts
                try:
                    post = await self._parse_linkedin_post(container, company_handle)
                    if post:
                        posts.append(post)
                except Exception as e:
                    logger.debug(f"Error parsing LinkedIn post: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error monitoring LinkedIn for {company_handle}: {str(e)}")
            
        return posts
        
    async def _parse_linkedin_post(self, container, company_handle: str) -> Optional[SocialMediaPost]:
        """Parse a LinkedIn post container."""
        try:
            # Extract post content
            content_elem = container.find(['span', 'div'], class_=re.compile(r'break-words'))
            content = content_elem.get_text().strip() if content_elem else ""
            
            # Extract post URL/ID
            post_link = container.find('a', href=re.compile(r'/feed/update/'))
            post_url = ""
            post_id = ""
            
            if post_link:
                post_url = urljoin("https://www.linkedin.com", post_link['href'])
                post_id = re.search(r'update:([^/?]+)', post_url)
                post_id = post_id.group(1) if post_id else ""
                
            # Extract timestamp (approximation since LinkedIn doesn't show exact times publicly)
            time_elem = container.find(['time', 'span'], class_=re.compile(r'visually-hidden'))
            published_at = datetime.utcnow() - timedelta(hours=1)  # Default recent
            
            # Extract engagement metrics (if visible)
            engagement = {}
            like_elem = container.find(['button', 'span'], class_=re.compile(r'social-counts'))
            if like_elem:
                engagement_text = like_elem.get_text()
                # Parse numbers from text like "15 reactions"
                numbers = re.findall(r'\d+', engagement_text)
                if numbers:
                    engagement['reactions'] = int(numbers[0])
                    
            # Extract hashtags and mentions
            hashtags = re.findall(r'#\w+', content)
            mentions = re.findall(r'@\w+', content)
            
            # Detect media
            media_urls = []
            images = container.find_all('img', src=True)
            for img in images:
                if 'media' in img['src'] or 'image' in img['src']:
                    media_urls.append(img['src'])
                    
            return SocialMediaPost(
                platform='linkedin',
                post_id=post_id,
                url=post_url,
                content=content,
                author=company_handle,
                author_handle=company_handle,
                published_at=published_at,
                engagement_metrics=engagement,
                hashtags=hashtags,
                mentions=mentions,
                media_urls=media_urls,
                post_type=self._determine_post_type(content, media_urls),
                metadata={'source': 'company_page', 'scraping_method': 'html_parse'}
            )
            
        except Exception as e:
            logger.debug(f"Error parsing LinkedIn post: {str(e)}")
            return None
            
    async def monitor_twitter_profile(self, handle: str) -> List[SocialMediaPost]:
        """Monitor Twitter/X profile for tweets."""
        posts = []
        
        try:
            # Twitter profile URL
            url = f"https://twitter.com/{handle}"
            
            html = await self._fetch_page(url)
            if not html:
                return posts
                
            # Twitter heavily uses JavaScript, so we look for any embedded data
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for script tags with JSON data
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'tweet' in script.string.lower():
                    try:
                        # Try to find tweet data in script content
                        tweet_matches = re.findall(r'"full_text":"([^"]+)"', script.string)
                        for i, tweet_text in enumerate(tweet_matches[:5]):  # Limit to 5
                            post = SocialMediaPost(
                                platform='twitter',
                                post_id=f"twitter_{handle}_{i}",
                                url=f"https://twitter.com/{handle}",
                                content=tweet_text.replace('\\n', '\n'),
                                author=handle,
                                author_handle=handle,
                                published_at=datetime.utcnow() - timedelta(hours=i),
                                engagement_metrics={},
                                hashtags=re.findall(r'#\w+', tweet_text),
                                mentions=re.findall(r'@\w+', tweet_text),
                                media_urls=[],
                                post_type='text',
                                metadata={'source': 'profile_page', 'extracted_from': 'script'}
                            )
                            posts.append(post)
                            break
                    except Exception as e:
                        continue
                        
        except Exception as e:
            logger.error(f"Error monitoring Twitter for {handle}: {str(e)}")
            
        return posts
        
    async def monitor_medium_profile(self, handle: str) -> List[SocialMediaPost]:
        """Monitor Medium profile for articles."""
        posts = []
        
        try:
            # Medium profile URL
            url = f"https://medium.com/@{handle}"
            
            html = await self._fetch_page(url)
            if not html:
                return posts
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for article links
            article_links = soup.find_all('a', href=re.compile(r'/@' + handle + '/'))
            
            for link in article_links[:5]:  # Limit to recent articles
                try:
                    article_url = urljoin("https://medium.com", link['href'])
                    
                    # Extract article title
                    title_elem = link.find(['h1', 'h2', 'h3'])
                    title = title_elem.get_text().strip() if title_elem else ""
                    
                    # Extract preview text
                    preview_elem = link.find_next(['p', 'div'])
                    preview = preview_elem.get_text().strip() if preview_elem else ""
                    
                    post = SocialMediaPost(
                        platform='medium',
                        post_id=re.search(r'/([^/]+)$', article_url).group(1) if re.search(r'/([^/]+)$', article_url) else "",
                        url=article_url,
                        content=f"{title}\n\n{preview}",
                        author=handle,
                        author_handle=handle,
                        published_at=datetime.utcnow() - timedelta(days=1),
                        engagement_metrics={},
                        hashtags=[],
                        mentions=[],
                        media_urls=[],
                        post_type='article',
                        metadata={'source': 'profile_page', 'title': title}
                    )
                    posts.append(post)
                    
                except Exception as e:
                    logger.debug(f"Error parsing Medium article: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error monitoring Medium for {handle}: {str(e)}")
            
        return posts
        
    async def monitor_youtube_channel(self, channel_handle: str) -> List[SocialMediaPost]:
        """Monitor YouTube channel for videos."""
        posts = []
        
        try:
            # YouTube channel URL
            url = f"https://www.youtube.com/@{channel_handle}/videos"
            
            html = await self._fetch_page(url)
            if not html:
                # Try alternative URL format
                url = f"https://www.youtube.com/c/{channel_handle}/videos"
                html = await self._fetch_page(url)
                
            if not html:
                return posts
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for video data in script tags
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'videoDetails' in script.string:
                    try:
                        # Extract video information from YouTube's JSON data
                        video_matches = re.findall(r'"title":{"runs":\[{"text":"([^"]+)"}', script.string)
                        for i, video_title in enumerate(video_matches[:3]):  # Limit to 3 recent
                            post = SocialMediaPost(
                                platform='youtube',
                                post_id=f"youtube_{channel_handle}_{i}",
                                url=f"https://www.youtube.com/@{channel_handle}",
                                content=video_title,
                                author=channel_handle,
                                author_handle=channel_handle,
                                published_at=datetime.utcnow() - timedelta(days=i),
                                engagement_metrics={},
                                hashtags=[],
                                mentions=[],
                                media_urls=[],
                                post_type='video',
                                metadata={'source': 'channel_page', 'video_title': video_title}
                            )
                            posts.append(post)
                    except Exception as e:
                        continue
                        
        except Exception as e:
            logger.error(f"Error monitoring YouTube for {channel_handle}: {str(e)}")
            
        return posts
        
    def _determine_post_type(self, content: str, media_urls: List[str]) -> str:
        """Determine the type of social media post."""
        if media_urls:
            if any('video' in url.lower() for url in media_urls):
                return 'video'
            else:
                return 'image'
        elif 'http' in content:
            return 'link'
        else:
            return 'text'
            
    async def monitor_competitor_social_media(
        self, 
        competitor_name: str,
        social_handles: Dict[str, str]
    ) -> List[SocialMediaPost]:
        """Monitor all social media platforms for a competitor."""
        all_posts = []
        
        for platform, handle in social_handles.items():
            if not handle:
                continue
                
            try:
                logger.info(f"Monitoring {platform} for {competitor_name} (@{handle})")
                
                if platform == 'linkedin':
                    posts = await self.monitor_linkedin_company(handle)
                elif platform == 'twitter':
                    posts = await self.monitor_twitter_profile(handle)
                elif platform == 'medium':
                    posts = await self.monitor_medium_profile(handle)
                elif platform == 'youtube':
                    posts = await self.monitor_youtube_channel(handle)
                else:
                    logger.warning(f"Unsupported platform: {platform}")
                    continue
                    
                all_posts.extend(posts)
                logger.info(f"Found {len(posts)} posts on {platform} for {competitor_name}")
                
            except Exception as e:
                logger.error(f"Error monitoring {platform} for {competitor_name}: {str(e)}")
                continue
                
        return all_posts
        
    async def extract_social_handles_from_website(self, domain: str) -> Dict[str, str]:
        """Extract social media handles from competitor website."""
        handles = {}
        
        try:
            if not domain.startswith(('http://', 'https://')):
                domain = f"https://{domain}"
                
            html = await self._fetch_page(domain)
            if not html:
                return handles
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for social media links
            social_patterns = {
                'linkedin': [
                    r'linkedin\.com/company/([^/?]+)',
                    r'linkedin\.com/in/([^/?]+)'
                ],
                'twitter': [
                    r'twitter\.com/([^/?]+)',
                    r'x\.com/([^/?]+)'
                ],
                'facebook': [
                    r'facebook\.com/([^/?]+)'
                ],
                'instagram': [
                    r'instagram\.com/([^/?]+)'
                ],
                'youtube': [
                    r'youtube\.com/@([^/?]+)',
                    r'youtube\.com/c/([^/?]+)',
                    r'youtube\.com/channel/([^/?]+)'
                ],
                'medium': [
                    r'medium\.com/@([^/?]+)'
                ]
            }
            
            # Find all links
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                for platform, patterns in social_patterns.items():
                    for pattern in patterns:
                        match = re.search(pattern, href, re.IGNORECASE)
                        if match and platform not in handles:
                            handle = match.group(1)
                            # Clean up handle
                            handle = handle.split('?')[0]  # Remove query params
                            handle = handle.split('#')[0]  # Remove fragments
                            handles[platform] = handle
                            logger.info(f"Found {platform} handle: {handle}")
                            break
                            
        except Exception as e:
            logger.error(f"Error extracting social handles from {domain}: {str(e)}")
            
        return handles

# Global instance
social_media_monitor = SocialMediaMonitor()