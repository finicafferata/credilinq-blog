"""
Social Media Agent - Adapts content for different social media platforms
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

class SocialMediaAgent(BaseAgent):
    """
    Agent specialized in adapting content for social media platforms.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None, name: str = "SocialMediaAgent", description: str = "Adapts content for social media platforms"):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.SOCIAL_MEDIA,
                name=name,
                description=description
            )
        super().__init__(metadata=metadata)
        self.agent_type = AgentType.SOCIAL_MEDIA
        
        # Platform-specific configurations
        self.platform_configs = {
            "linkedin": {
                "max_length": 1300,
                "optimal_length": 1000,
                "hashtag_limit": 5,
                "tone": "professional",
                "content_type": "business",
                "call_to_action": True
            },
            "twitter": {
                "max_length": 280,
                "optimal_length": 240,
                "hashtag_limit": 2,
                "tone": "conversational",
                "content_type": "news",
                "call_to_action": False
            },
            "facebook": {
                "max_length": 632,
                "optimal_length": 400,
                "hashtag_limit": 3,
                "tone": "friendly",
                "content_type": "engagement",
                "call_to_action": True
            },
            "instagram": {
                "max_length": 2200,
                "optimal_length": 150,
                "hashtag_limit": 30,
                "tone": "visual",
                "content_type": "visual",
                "call_to_action": True
            }
        }
        
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Adapt content for different social media platforms.
        
        Args:
            context: Dictionary containing:
                - content: Blog content
                - blog_title: Title of the blog
                - outline: Blog outline sections
                - platforms: List of platforms to target (optional, defaults to all)
                - target_audience: Target audience (optional)
        
        Returns:
            AgentResult with adapted content for each platform
        """
        try:
            logger.info(f"SocialMediaAgent executing for blog: {context.get('blog_title', 'Unknown')}")
            
            content = context.get('content', '')
            blog_title = context.get('blog_title', '')
            outline = context.get('outline', [])
            platforms = context.get('platforms', ['linkedin', 'twitter', 'facebook', 'instagram'])
            target_audience = context.get('target_audience', 'professionals')
            
            if not content and not blog_title:
                raise AgentExecutionError("No content or title provided for social media adaptation")
            
            # Generate posts for each platform
            social_posts = {}
            for platform in platforms:
                if platform in self.platform_configs:
                    post = self._generate_platform_post(
                        content, blog_title, outline, platform, target_audience
                    )
                    social_posts[platform] = post
            
            result_data = {
                "social_posts": social_posts,
                "platforms_processed": list(social_posts.keys()),
                "total_posts": len(social_posts)
            }
            
            logger.info(f"SocialMediaAgent completed successfully. Generated posts for {len(social_posts)} platforms.")
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Generated social media posts for {len(social_posts)} platforms"
            )
            
        except Exception as e:
            logger.error(f"SocialMediaAgent execution failed: {str(e)}")
            raise AgentExecutionError("SocialMediaAgent", "execution", str(e))
    
    def _generate_platform_post(self, content: str, blog_title: str, outline: List[str], platform: str, target_audience: str) -> Dict[str, Any]:
        """
        Generate a post optimized for a specific platform.
        """
        config = self.platform_configs[platform]
        
        # Extract key points from content
        key_points = self._extract_key_points(content, outline)
        
        # Generate main post content
        post_content = self._create_post_content(
            blog_title, key_points, platform, config, target_audience
        )
        
        # Generate hashtags
        hashtags = self._generate_hashtags(blog_title, content, platform, config)
        
        # Generate call to action
        cta = self._generate_call_to_action(platform, config) if config["call_to_action"] else None
        
        # Calculate engagement metrics
        engagement_metrics = self._calculate_engagement_metrics(post_content, platform)
        
        return {
            "platform": platform,
            "content": post_content,
            "hashtags": hashtags,
            "call_to_action": cta,
            "character_count": len(post_content),
            "engagement_metrics": engagement_metrics,
            "optimization_score": self._calculate_optimization_score(post_content, platform, config)
        }
    
    def _extract_key_points(self, content: str, outline: List[str]) -> List[str]:
        """
        Extract key points from content for social media posts.
        """
        key_points = []
        
        # Extract from outline sections
        for section in outline[:3]:  # Use first 3 sections
            key_points.append(section)
        
        # Extract from content paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs[:2]:  # Use first 2 paragraphs
            if para.strip() and not para.startswith('#'):
                # Extract first sentence as key point
                sentences = para.split('.')
                if sentences[0].strip():
                    key_points.append(sentences[0].strip() + '.')
        
        return key_points[:5]  # Limit to 5 key points
    
    def _create_post_content(self, blog_title: str, key_points: List[str], platform: str, config: Dict[str, Any], target_audience: str) -> str:
        """
        Create post content optimized for the platform.
        """
        if platform == "linkedin":
            return self._create_linkedin_post(blog_title, key_points, target_audience)
        elif platform == "twitter":
            return self._create_twitter_post(blog_title, key_points)
        elif platform == "facebook":
            return self._create_facebook_post(blog_title, key_points, target_audience)
        elif platform == "instagram":
            return self._create_instagram_post(blog_title, key_points)
        else:
            return self._create_generic_post(blog_title, key_points)
    
    def _create_linkedin_post(self, blog_title: str, key_points: List[str], target_audience: str) -> str:
        """
        Create a LinkedIn post optimized for professionals.
        """
        post = f"📈 {blog_title}\n\n"
        
        if key_points:
            post += "Key insights from our latest article:\n\n"
            for i, point in enumerate(key_points[:3], 1):
                post += f"• {point}\n"
        
        post += f"\nPerfect for {target_audience} looking to stay ahead in their industry.\n\n"
        post += "What's your take on this topic? Share your thoughts below! 👇"
        
        return post
    
    def _create_twitter_post(self, blog_title: str, key_points: List[str]) -> str:
        """
        Create a Twitter post optimized for brevity and engagement.
        """
        post = f"🚀 {blog_title}\n\n"
        
        if key_points:
            post += f"💡 {key_points[0] if key_points else 'Essential insights for professionals'}\n\n"
        
        post += "Read the full article to discover more strategies that can transform your approach."
        
        return post
    
    def _create_facebook_post(self, blog_title: str, key_points: List[str], target_audience: str) -> str:
        """
        Create a Facebook post optimized for engagement.
        """
        post = f"🎯 {blog_title}\n\n"
        
        if key_points:
            post += "Here's what you need to know:\n\n"
            for i, point in enumerate(key_points[:2], 1):
                post += f"📌 {point}\n"
        
        post += f"\nThis is a must-read for {target_audience}!\n\n"
        post += "Tag someone who would benefit from this information! 👥"
        
        return post
    
    def _create_instagram_post(self, blog_title: str, key_points: List[str]) -> str:
        """
        Create an Instagram post optimized for visual appeal.
        """
        post = f"✨ {blog_title}\n\n"
        
        if key_points:
            post += "Key takeaways:\n\n"
            for i, point in enumerate(key_points[:2], 1):
                post += f"🔹 {point}\n"
        
        post += "\nSwipe to learn more! 📖\n\n"
        post += "Follow for daily insights! 👆"
        
        return post
    
    def _create_generic_post(self, blog_title: str, key_points: List[str]) -> str:
        """
        Create a generic post for unknown platforms.
        """
        post = f"📝 {blog_title}\n\n"
        
        if key_points:
            post += "Highlights:\n\n"
            for point in key_points[:2]:
                post += f"• {point}\n"
        
        post += "\nRead the full article for complete insights!"
        
        return post
    
    def _generate_hashtags(self, blog_title: str, content: str, platform: str, config: Dict[str, Any]) -> List[str]:
        """
        Generate relevant hashtags for the platform.
        """
        # Extract potential hashtags from title and content
        text = f"{blog_title} {content}".lower()
        
        # Common business hashtags
        business_hashtags = [
            "#business", "#strategy", "#growth", "#innovation", "#leadership",
            "#marketing", "#digital", "#technology", "#entrepreneur", "#success"
        ]
        
        # Platform-specific hashtags
        platform_hashtags = {
            "linkedin": ["#linkedin", "#networking", "#professional", "#career"],
            "twitter": ["#business", "#tech", "#innovation"],
            "facebook": ["#business", "#growth", "#success"],
            "instagram": ["#business", "#entrepreneur", "#success", "#growth"]
        }
        
        # Extract keywords for custom hashtags
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        word_freq = {}
        
        for word in words:
            if word not in ['business', 'strategy', 'growth', 'innovation', 'leadership']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create custom hashtags
        custom_hashtags = []
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:3]:
            custom_hashtags.append(f"#{word}")
        
        # Combine hashtags
        all_hashtags = business_hashtags + platform_hashtags.get(platform, []) + custom_hashtags
        
        # Limit based on platform
        limit = config.get("hashtag_limit", 5)
        return all_hashtags[:limit]
    
    def _generate_call_to_action(self, platform: str, config: Dict[str, Any]) -> str:
        """
        Generate appropriate call to action for the platform.
        """
        cta_options = {
            "linkedin": [
                "What's your experience with this? Share below! 👇",
                "How do you approach this in your business? Comment your thoughts! 💭",
                "Tag someone who needs to see this! 👥"
            ],
            "facebook": [
                "What do you think? Share your thoughts! 💬",
                "Tag a friend who would benefit from this! 👥",
                "Like and share if you found this helpful! 👍"
            ],
            "instagram": [
                "Double tap if you agree! ❤️",
                "Share this with someone who needs to see it! 📤",
                "Follow for more insights! 👆"
            ]
        }
        
        import random
        platform_ctas = cta_options.get(platform, ["Read more to learn! 📖"])
        return random.choice(platform_ctas)
    
    def _calculate_engagement_metrics(self, post_content: str, platform: str) -> Dict[str, Any]:
        """
        Calculate estimated engagement metrics for the post.
        """
        # Simple engagement prediction based on content characteristics
        word_count = len(post_content.split())
        has_emoji = bool(re.search(r'[😀-🙏🌀-🗿🚀-🛿]', post_content))
        has_question = '?' in post_content
        has_cta = any(word in post_content.lower() for word in ['comment', 'share', 'like', 'follow', 'tag'])
        
        # Base engagement score
        engagement_score = 50
        
        # Adjust based on factors
        if has_emoji:
            engagement_score += 10
        if has_question:
            engagement_score += 15
        if has_cta:
            engagement_score += 20
        if 50 <= word_count <= 150:
            engagement_score += 10
        
        # Platform-specific adjustments
        platform_multipliers = {
            "linkedin": 1.2,
            "twitter": 1.0,
            "facebook": 1.3,
            "instagram": 1.1
        }
        
        final_score = min(engagement_score * platform_multipliers.get(platform, 1.0), 100)
        
        return {
            "estimated_engagement": round(final_score, 1),
            "estimated_reach": round(final_score * 10, 0),
            "estimated_clicks": round(final_score * 0.3, 1),
            "has_emoji": has_emoji,
            "has_question": has_question,
            "has_cta": has_cta
        }
    
    def _calculate_optimization_score(self, post_content: str, platform: str, config: Dict[str, Any]) -> int:
        """
        Calculate optimization score for the post (0-100).
        """
        score = 0
        max_score = 100
        
        # Length optimization (30 points)
        content_length = len(post_content)
        optimal_length = config.get("optimal_length", 100)
        max_length = config.get("max_length", 1000)
        
        if content_length <= max_length:
            if abs(content_length - optimal_length) <= 50:
                score += 30
            elif abs(content_length - optimal_length) <= 100:
                score += 20
            else:
                score += 10
        else:
            score += 5
        
        # Content quality (40 points)
        has_emoji = bool(re.search(r'[😀-🙏🌀-🗿🚀-🛿]', post_content))
        has_question = '?' in post_content
        has_cta = any(word in post_content.lower() for word in ['comment', 'share', 'like', 'follow', 'tag'])
        
        if has_emoji:
            score += 10
        if has_question:
            score += 15
        if has_cta:
            score += 15
        
        # Platform-specific optimization (30 points)
        if platform == "linkedin" and "professional" in post_content.lower():
            score += 30
        elif platform == "twitter" and len(post_content) <= 280:
            score += 30
        elif platform == "facebook" and ("engagement" in post_content.lower() or "share" in post_content.lower()):
            score += 30
        elif platform == "instagram" and ("visual" in post_content.lower() or "✨" in post_content):
            score += 30
        else:
            score += 15
        
        return min(score, max_score)
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context for social media adaptation.
        """
        required_fields = ['content', 'blog_title']
        
        for field in required_fields:
            if not context.get(field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True 