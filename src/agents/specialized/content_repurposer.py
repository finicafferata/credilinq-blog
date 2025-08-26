"""
Advanced Content Repurposing Agent for Multi-Platform Content Generation.
Intelligently adapts content for different social media platforms and formats.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import json

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from ..core.base_agent import BaseAgent
from ...config.settings import settings
from ...core.monitoring import metrics, async_performance_tracker

class ContentPlatform(str, Enum):
    """Supported content platforms."""
    BLOG = "blog"
    LINKEDIN_POST = "linkedin_post"
    LINKEDIN_ARTICLE = "linkedin_article"
    TWITTER_THREAD = "twitter_thread"
    TWITTER_SINGLE = "twitter_single"
    INSTAGRAM_POST = "instagram_post"
    FACEBOOK_POST = "facebook_post"
    YOUTUBE_DESCRIPTION = "youtube_description"
    EMAIL_NEWSLETTER = "email_newsletter"
    PODCAST_SCRIPT = "podcast_script"

class ContentTone(str, Enum):
    """Content tone variations."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    INSPIRATIONAL = "inspirational"

class ContentLength(str, Enum):
    """Content length categories."""
    MICRO = "micro"      # <50 words
    SHORT = "short"      # 50-150 words
    MEDIUM = "medium"    # 150-500 words
    LONG = "long"        # 500-1500 words
    EXTENDED = "extended" # >1500 words

@dataclass
class PlatformSpecs:
    """Platform-specific content specifications."""
    platform: ContentPlatform
    max_length: int
    optimal_length: int
    character_limit: Optional[int] = None
    supports_hashtags: bool = True
    supports_mentions: bool = True
    supports_links: bool = True
    supports_media: bool = True
    preferred_tone: ContentTone = ContentTone.PROFESSIONAL
    engagement_hooks: List[str] = None
    call_to_action_style: str = "Ask a question"
    
    def __post_init__(self):
        if self.engagement_hooks is None:
            self.engagement_hooks = []

# Platform specifications database
PLATFORM_SPECS = {
    ContentPlatform.LINKEDIN_POST: PlatformSpecs(
        platform=ContentPlatform.LINKEDIN_POST,
        max_length=3000,
        optimal_length=1300,
        character_limit=3000,
        preferred_tone=ContentTone.PROFESSIONAL,
        engagement_hooks=[
            "What's your experience with...",
            "Here's what I learned...",
            "3 key insights from...",
            "The biggest mistake I see..."
        ],
        call_to_action_style="Ask for engagement with a professional question"
    ),
    
    ContentPlatform.TWITTER_THREAD: PlatformSpecs(
        platform=ContentPlatform.TWITTER_THREAD,
        max_length=280,  # per tweet
        optimal_length=250,
        character_limit=280,
        preferred_tone=ContentTone.CONVERSATIONAL,
        engagement_hooks=[
            "🧵 Thread:",
            "Here's what most people get wrong about...",
            "5 things I wish I knew about...",
            "Quick story:"
        ],
        call_to_action_style="Encourage retweets and replies"
    ),
    
    ContentPlatform.TWITTER_SINGLE: PlatformSpecs(
        platform=ContentPlatform.TWITTER_SINGLE,
        max_length=280,
        optimal_length=250,
        character_limit=280,
        preferred_tone=ContentTone.CONVERSATIONAL,
        engagement_hooks=[
            "Hot take:",
            "Unpopular opinion:",
            "Quick tip:",
            "Today I learned:"
        ],
        call_to_action_style="Encourage engagement with likes and retweets"
    ),
    
    ContentPlatform.INSTAGRAM_POST: PlatformSpecs(
        platform=ContentPlatform.INSTAGRAM_POST,
        max_length=2200,
        optimal_length=1500,
        preferred_tone=ContentTone.FRIENDLY,
        engagement_hooks=[
            "Behind the scenes:",
            "Here's the truth about...",
            "Save this post if...",
            "Double tap if you agree..."
        ],
        call_to_action_style="Encourage saves, shares, and comments"
    ),
    
    ContentPlatform.FACEBOOK_POST: PlatformSpecs(
        platform=ContentPlatform.FACEBOOK_POST,
        max_length=63206,
        optimal_length=500,
        preferred_tone=ContentTone.FRIENDLY,
        engagement_hooks=[
            "I wanted to share something important...",
            "Here's what happened when...",
            "Can we talk about...",
            "This might be controversial, but..."
        ],
        call_to_action_style="Encourage comments and shares"
    )
}

class RepurposedContent(BaseModel):
    """Model for repurposed content output."""
    platform: ContentPlatform
    content: str
    title: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    call_to_action: Optional[str] = None
    estimated_engagement: Dict[str, Any] = Field(default_factory=dict)
    optimization_notes: List[str] = Field(default_factory=list)
    word_count: int = 0
    character_count: int = 0
    
    @validator('content')
    def validate_content_length(cls, v, values):
        platform = values.get('platform')
        if platform and platform in PLATFORM_SPECS:
            spec = PLATFORM_SPECS[platform]
            if len(v) > spec.max_length:
                raise ValueError(f"Content exceeds {platform} maximum length of {spec.max_length}")
        return v

class ContentSeriesPost(BaseModel):
    """Model for content series post."""
    series_title: str
    post_number: int
    total_posts: int
    platform: ContentPlatform
    content: str
    hashtags: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    
class ContentSeries(BaseModel):
    """Model for cross-platform content series."""
    series_id: str
    title: str
    description: str
    total_posts: int
    posts: List[ContentSeriesPost]
    publishing_schedule: Dict[str, Any] = Field(default_factory=dict)
    cross_promotion_strategy: Dict[str, Any] = Field(default_factory=dict)

class ContentRepurposer(BaseAgent):
    """Advanced content repurposing agent with multi-platform intelligence."""
    
    def __init__(self):
        super().__init__(
            agent_type="content_repurposer",
            capabilities=[
                "multi_platform_adaptation",
                "length_optimization", 
                "tone_adjustment",
                "engagement_optimization",
                "hashtag_generation",
                "series_creation"
            ]
        )
        
        # Initialize language models for different tasks
        self.adaptation_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        
        self.optimization_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1000
        )
        
        # Content adaptation templates
        self.adaptation_templates = self._create_adaptation_templates()
        
        # Engagement optimization patterns
        self.engagement_patterns = self._load_engagement_patterns()
    
    def _create_adaptation_templates(self) -> Dict[ContentPlatform, ChatPromptTemplate]:
        """Create platform-specific adaptation templates."""
        templates = {}
        
        # LinkedIn Post Template
        linkedin_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert LinkedIn content strategist. Your task is to repurpose content for LinkedIn posts that drive professional engagement.

LinkedIn Best Practices:
- Professional yet approachable tone
- Include personal insights or experiences
- Use line breaks for readability
- Add 3-5 relevant hashtags at the end
- Include a call-to-action that encourages professional discussion
- Optimal length: 1300 characters
- Use emojis sparingly and professionally

Content should be structured as:
1. Hook (attention-grabbing opening)
2. Value/insight (main content)
3. Personal take or experience
4. Call to action
5. Hashtags
            """),
            HumanMessage(content="""
Original content: {original_content}

Target audience: {target_audience}
Company context: {company_context}
Key message: {key_message}

Repurpose this content for a LinkedIn post that will engage professionals and drive meaningful conversations.
            """)
        ])
        templates[ContentPlatform.LINKEDIN_POST] = linkedin_template
        
        # Twitter Thread Template
        twitter_thread_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert Twitter content creator. Create engaging Twitter threads that maximize engagement and virality.

Twitter Thread Best Practices:
- Start with a compelling hook in the first tweet
- Each tweet should be under 280 characters
- Use thread numbers (1/, 2/, 3/, etc.)
- Include relevant emojis and hashtags
- End with a strong call-to-action
- Make each tweet valuable on its own
- Use line breaks and spacing for readability
- Include 2-3 strategic hashtags per tweet

Structure:
1/ Hook tweet (introduce the topic)
2-X/ Value tweets (main content broken down)
Final/ CTA tweet (encourage retweets, follows, etc.)
            """),
            HumanMessage(content="""
Original content: {original_content}

Target audience: {target_audience}
Key message: {key_message}

Create a Twitter thread (4-8 tweets) that breaks down this content into engaging, shareable tweets.
            """)
        ])
        templates[ContentPlatform.TWITTER_THREAD] = twitter_thread_template
        
        # Instagram Post Template
        instagram_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert Instagram content creator. Create visually-oriented content that drives engagement.

Instagram Best Practices:
- Visual storytelling approach
- Use emojis strategically throughout
- Include 5-10 relevant hashtags
- Encourage saves and shares
- Use line breaks for visual appeal
- Ask questions to drive comments
- Optimal length: 1500 characters
- Include a clear call-to-action

Structure:
1. Visual hook (describe what image/video would show)
2. Story or valuable content
3. Personal connection
4. Call-to-action
5. Hashtags (mix of popular and niche)
            """),
            HumanMessage(content="""
Original content: {original_content}

Target audience: {target_audience}
Visual concept: {visual_concept}
Key message: {key_message}

Create an Instagram post that would work well with visual content and drive high engagement.
            """)
        ])
        templates[ContentPlatform.INSTAGRAM_POST] = instagram_template
        
        return templates
    
    def _load_engagement_patterns(self) -> Dict[str, List[str]]:
        """Load engagement optimization patterns."""
        return {
            "hooks": [
                "Here's what most people get wrong about {topic}:",
                "I used to think {misconception}, but then I learned:",
                "3 things I wish I knew about {topic} when I started:",
                "The biggest mistake in {industry} that nobody talks about:",
                "Hot take: {controversial_opinion}",
                "Here's what happened when I {action}:",
                "If you're struggling with {problem}, this will help:",
                "The truth about {topic} that nobody wants to admit:"
            ],
            "cta_patterns": {
                ContentPlatform.LINKEDIN_POST: [
                    "What's your experience with this?",
                    "Do you agree? Share your thoughts below.",
                    "What would you add to this list?",
                    "Have you faced this challenge too?"
                ],
                ContentPlatform.TWITTER_THREAD: [
                    "What did I miss? Reply with your thoughts!",
                    "Retweet if this helped you!",
                    "What's your take on this?",
                    "Follow for more insights like this!"
                ],
                ContentPlatform.INSTAGRAM_POST: [
                    "Save this post for later!",
                    "Double tap if you agree!",
                    "Share this with someone who needs to see it!",
                    "What's your experience? Comment below!"
                ]
            },
            "value_frameworks": [
                "Problem → Solution → Benefit",
                "Before → After → How",
                "Mistake → Lesson → Action",
                "Question → Answer → Application",
                "Myth → Truth → Impact"
            ]
        }
    
    async def repurpose_content(
        self,
        original_content: str,
        target_platforms: List[ContentPlatform],
        source_context: Dict[str, Any],
        customization_options: Optional[Dict[str, Any]] = None
    ) -> Dict[ContentPlatform, RepurposedContent]:
        """Repurpose content for multiple platforms with intelligent adaptation."""
        
        async with async_performance_tracker("content_repurposing"):
            results = {}
            
            # Extract key information from original content
            content_analysis = await self._analyze_content(original_content)
            
            # Process each target platform
            for platform in target_platforms:
                try:
                    repurposed = await self._adapt_for_platform(
                        original_content=original_content,
                        platform=platform,
                        content_analysis=content_analysis,
                        source_context=source_context,
                        customization_options=customization_options or {}
                    )
                    
                    results[platform] = repurposed
                    
                    # Track metrics
                    metrics.increment_counter(
                        "content.repurposed",
                        tags={
                            "source_platform": source_context.get("platform", "unknown"),
                            "target_platform": platform,
                            "success": "true"
                        }
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to repurpose for {platform}: {str(e)}")
                    metrics.increment_counter(
                        "content.repurpose_failed",
                        tags={"platform": platform}
                    )
                    continue
            
            return results
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to extract key information for adaptation."""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
Analyze the provided content and extract key information for content repurposing.

Return a JSON object with:
- main_topic: The primary subject matter
- key_points: List of 3-5 main points or takeaways
- tone: Current tone (professional, casual, etc.)
- target_audience: Implied target audience
- content_type: Type of content (educational, promotional, etc.)
- keywords: Relevant keywords and phrases
- emotional_appeal: Emotional elements present
- call_to_action: Any existing call-to-action
- expertise_level: Required expertise level to understand
- engagement_potential: Elements that could drive engagement
            """),
            HumanMessage(content="Analyze this content: {content}")
        ])
        
        try:
            response = await self.adaptation_llm.agenerate([
                analysis_prompt.format_messages(content=content)
            ])
            
            # Parse the response
            analysis_text = response.generations[0][0].text
            
            # Try to extract JSON, fallback to basic analysis
            try:
                import json
                analysis = json.loads(analysis_text)
            except:
                analysis = {
                    "main_topic": "Content repurposing",
                    "key_points": ["Key insight 1", "Key insight 2", "Key insight 3"],
                    "tone": "professional",
                    "target_audience": "professionals",
                    "content_type": "educational",
                    "keywords": [],
                    "emotional_appeal": "informational",
                    "call_to_action": None,
                    "expertise_level": "intermediate",
                    "engagement_potential": ["informative", "actionable"]
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            return {
                "main_topic": "Content repurposing",
                "key_points": ["Key insight"],
                "tone": "professional",
                "target_audience": "professionals",
                "content_type": "educational"
            }
    
    async def _adapt_for_platform(
        self,
        original_content: str,
        platform: ContentPlatform,
        content_analysis: Dict[str, Any],
        source_context: Dict[str, Any],
        customization_options: Dict[str, Any]
    ) -> RepurposedContent:
        """Adapt content for a specific platform."""
        
        # Get platform specifications
        platform_spec = PLATFORM_SPECS.get(platform)
        if not platform_spec:
            raise ValueError(f"Unsupported platform: {platform}")
        
        # Prepare adaptation context
        adaptation_context = {
            "original_content": original_content,
            "target_audience": content_analysis.get("target_audience", "professionals"),
            "company_context": source_context.get("company_context", ""),
            "key_message": content_analysis.get("main_topic", ""),
            "visual_concept": customization_options.get("visual_concept", "relevant professional image"),
            "tone_preference": customization_options.get("tone", platform_spec.preferred_tone),
            "hashtag_count": customization_options.get("hashtag_count", 5),
            "include_cta": customization_options.get("include_cta", True)
        }
        
        # Get the appropriate template
        template = self.adaptation_templates.get(platform)
        if not template:
            # Use generic adaptation
            adapted_content = await self._generic_adaptation(
                original_content, platform, platform_spec, adaptation_context
            )
        else:
            # Use platform-specific template
            response = await self.adaptation_llm.agenerate([
                template.format_messages(**adaptation_context)
            ])
            adapted_content = response.generations[0][0].text
        
        # Optimize for platform specifications
        optimized_content = await self._optimize_for_platform(
            adapted_content, platform_spec, customization_options
        )
        
        # Generate hashtags and mentions
        hashtags = await self._generate_hashtags(
            content_analysis, platform, customization_options.get("hashtag_count", 5)
        )
        
        # Generate call-to-action if needed
        cta = None
        if customization_options.get("include_cta", True):
            cta = await self._generate_cta(platform, content_analysis)
        
        # Calculate metrics
        word_count = len(optimized_content.split())
        character_count = len(optimized_content)
        
        # Generate optimization notes
        optimization_notes = self._generate_optimization_notes(
            optimized_content, platform_spec, content_analysis
        )
        
        # Estimate engagement potential
        engagement_estimate = self._estimate_engagement(
            optimized_content, platform, content_analysis, hashtags
        )
        
        return RepurposedContent(
            platform=platform,
            content=optimized_content,
            hashtags=hashtags,
            call_to_action=cta,
            estimated_engagement=engagement_estimate,
            optimization_notes=optimization_notes,
            word_count=word_count,
            character_count=character_count
        )
    
    async def _generic_adaptation(
        self,
        content: str,
        platform: ContentPlatform,
        platform_spec: PlatformSpecs,
        context: Dict[str, Any]
    ) -> str:
        """Generic content adaptation for platforms without specific templates."""
        
        generic_prompt = f"""
        Adapt the following content for {platform}:
        
        Platform requirements:
        - Maximum length: {platform_spec.max_length} characters
        - Optimal length: {platform_spec.optimal_length} characters
        - Preferred tone: {platform_spec.preferred_tone}
        - Supports hashtags: {platform_spec.supports_hashtags}
        
        Original content: {content}
        
        Create adapted content that fits the platform's style and requirements.
        """
        
        response = await self.optimization_llm.agenerate([
            [HumanMessage(content=generic_prompt)]
        ])
        
        return response.generations[0][0].text
    
    async def _optimize_for_platform(
        self,
        content: str,
        platform_spec: PlatformSpecs,
        customization_options: Dict[str, Any]
    ) -> str:
        """Optimize content length and format for platform specifications."""
        
        # Check if content needs length optimization
        if len(content) > platform_spec.max_length:
            # Truncate and optimize
            optimization_prompt = f"""
            The following content is too long for {platform_spec.platform} 
            (current: {len(content)} chars, max: {platform_spec.max_length} chars).
            
            Please shorten it while maintaining the key message and impact:
            
            {content}
            
            Target length: {platform_spec.optimal_length} characters
            """
            
            response = await self.optimization_llm.agenerate([
                [HumanMessage(content=optimization_prompt)]
            ])
            
            content = response.generations[0][0].text
        
        # Apply platform-specific formatting
        content = self._apply_platform_formatting(content, platform_spec)
        
        return content
    
    def _apply_platform_formatting(self, content: str, platform_spec: PlatformSpecs) -> str:
        """Apply platform-specific formatting rules."""
        
        if platform_spec.platform == ContentPlatform.TWITTER_THREAD:
            # Add thread numbering if not present
            lines = content.split('\n\n')
            if len(lines) > 1:
                formatted_lines = []
                for i, line in enumerate(lines, 1):
                    if not line.strip().startswith(f"{i}/"):
                        line = f"{i}/ {line}"
                    formatted_lines.append(line)
                content = '\n\n'.join(formatted_lines)
        
        elif platform_spec.platform == ContentPlatform.LINKEDIN_POST:
            # Ensure proper line breaks for LinkedIn
            content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 line breaks
            
        elif platform_spec.platform == ContentPlatform.INSTAGRAM_POST:
            # Add more line breaks for visual appeal
            content = content.replace('. ', '.\n\n')
            content = re.sub(r'\n{4,}', '\n\n\n', content)  # Max 3 line breaks
        
        return content.strip()
    
    async def _generate_hashtags(
        self,
        content_analysis: Dict[str, Any],
        platform: ContentPlatform,
        count: int = 5
    ) -> List[str]:
        """Generate relevant hashtags for the content and platform."""
        
        hashtag_prompt = f"""
        Generate {count} relevant hashtags for {platform} based on this content analysis:
        
        Topic: {content_analysis.get('main_topic', '')}
        Keywords: {', '.join(content_analysis.get('keywords', []))}
        Audience: {content_analysis.get('target_audience', '')}
        
        Return only the hashtags, one per line, without the # symbol.
        Focus on a mix of popular and niche hashtags for better reach.
        """
        
        try:
            response = await self.optimization_llm.agenerate([
                [HumanMessage(content=hashtag_prompt)]
            ])
            
            hashtags_text = response.generations[0][0].text
            hashtags = [
                tag.strip().replace('#', '') 
                for tag in hashtags_text.split('\n') 
                if tag.strip()
            ]
            
            return hashtags[:count]  # Limit to requested count
            
        except Exception as e:
            self.logger.error(f"Hashtag generation failed: {str(e)}")
            # Return default hashtags based on platform
            default_hashtags = {
                ContentPlatform.LINKEDIN_POST: ["business", "professional", "growth", "leadership", "insights"],
                ContentPlatform.TWITTER_THREAD: ["thread", "tips", "knowledge", "learning", "growth"],
                ContentPlatform.INSTAGRAM_POST: ["inspiration", "motivation", "business", "success", "entrepreneurship"]
            }
            return default_hashtags.get(platform, ["content", "business", "tips"])[:count]
    
    async def _generate_cta(
        self,
        platform: ContentPlatform,
        content_analysis: Dict[str, Any]
    ) -> str:
        """Generate platform-appropriate call-to-action."""
        
        cta_patterns = self.engagement_patterns["cta_patterns"].get(platform, [])
        
        if cta_patterns:
            # Use pattern-based CTA
            import random
            base_cta = random.choice(cta_patterns)
            
            # Customize based on content
            topic = content_analysis.get("main_topic", "this topic")
            cta = base_cta.replace("{topic}", topic.lower())
            
            return cta
        else:
            # Generate custom CTA
            cta_prompt = f"""
            Generate a compelling call-to-action for {platform} based on this content about {content_analysis.get('main_topic', 'business topics')}.
            
            Make it engaging and appropriate for the platform's audience.
            Keep it under 50 characters.
            """
            
            try:
                response = await self.optimization_llm.agenerate([
                    [HumanMessage(content=cta_prompt)]
                ])
                
                return response.generations[0][0].text.strip()
                
            except Exception as e:
                self.logger.error(f"CTA generation failed: {str(e)}")
                return "What are your thoughts on this?"
    
    def _generate_optimization_notes(
        self,
        content: str,
        platform_spec: PlatformSpecs,
        content_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization notes for content improvement."""
        
        notes = []
        
        # Length optimization notes
        content_length = len(content)
        if content_length > platform_spec.optimal_length:
            notes.append(f"Content is {content_length - platform_spec.optimal_length} characters over optimal length")
        elif content_length < platform_spec.optimal_length * 0.7:
            notes.append("Content could be expanded for better engagement")
        
        # Platform-specific notes
        if platform_spec.platform == ContentPlatform.LINKEDIN_POST:
            if not any(word in content.lower() for word in ["experience", "insight", "learned", "perspective"]):
                notes.append("Consider adding personal experience or insights for better LinkedIn engagement")
        
        elif platform_spec.platform == ContentPlatform.TWITTER_THREAD:
            if content.count('\n\n') < 2:
                notes.append("Consider breaking into multiple tweets for better thread engagement")
        
        elif platform_spec.platform == ContentPlatform.INSTAGRAM_POST:
            if content.count('🔥') + content.count('💡') + content.count('✨') == 0:
                notes.append("Consider adding relevant emojis for better Instagram visual appeal")
        
        # Engagement optimization notes
        if "?" not in content:
            notes.append("Consider adding a question to encourage engagement")
        
        return notes
    
    def _estimate_engagement(
        self,
        content: str,
        platform: ContentPlatform,
        content_analysis: Dict[str, Any],
        hashtags: List[str]
    ) -> Dict[str, Any]:
        """Estimate engagement potential based on content characteristics."""
        
        engagement_score = 50  # Base score
        
        # Content quality factors
        if len(content.split()) > 50:
            engagement_score += 10  # Substantial content
        
        if "?" in content:
            engagement_score += 15  # Questions drive engagement
        
        if any(hook in content.lower() for hook in ["tip", "mistake", "secret", "truth"]):
            engagement_score += 20  # High-engagement hooks
        
        # Platform-specific factors
        platform_multipliers = {
            ContentPlatform.TWITTER_THREAD: 1.3,
            ContentPlatform.LINKEDIN_POST: 1.1,
            ContentPlatform.INSTAGRAM_POST: 1.2
        }
        
        engagement_score *= platform_multipliers.get(platform, 1.0)
        
        # Hashtag impact
        if len(hashtags) >= 3:
            engagement_score += 10
        
        # Cap at 100
        engagement_score = min(100, engagement_score)
        
        return {
            "score": round(engagement_score),
            "predicted_reach": "medium" if engagement_score > 60 else "low" if engagement_score < 40 else "high",
            "key_factors": [
                "Quality content structure",
                "Engaging hooks" if engagement_score > 70 else "Consider adding hooks",
                "Good hashtag usage" if len(hashtags) >= 3 else "Add more hashtags",
                "Call-to-action present" if "?" in content else "Add call-to-action"
            ]
        }
    
    async def create_content_series(
        self,
        base_content: str,
        series_config: Dict[str, Any]
    ) -> ContentSeries:
        """Create a cross-platform content series from base content."""
        
        series_title = series_config.get("title", "Content Series")
        num_posts = series_config.get("num_posts", 5)
        platforms = series_config.get("platforms", [ContentPlatform.LINKEDIN_POST])
        
        series_id = f"series_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Break down content into series parts
        content_parts = await self._break_into_series(base_content, num_posts)
        
        posts = []
        for i, (platform, content_part) in enumerate(zip(platforms * num_posts, content_parts), 1):
            
            # Adapt each part for the platform
            content_analysis = await self._analyze_content(content_part)
            adapted_content = await self._adapt_for_platform(
                original_content=content_part,
                platform=platform,
                content_analysis=content_analysis,
                source_context=series_config.get("source_context", {}),
                customization_options=series_config.get("customization_options", {})
            )
            
            # Generate series-specific hashtags
            series_hashtags = await self._generate_series_hashtags(
                series_title, i, num_posts, platform
            )
            
            post = ContentSeriesPost(
                series_title=series_title,
                post_number=i,
                total_posts=num_posts,
                platform=platform,
                content=adapted_content.content,
                hashtags=series_hashtags,
                cross_references=[f"Part {j}/{num_posts}" for j in range(1, num_posts + 1) if j != i]
            )
            
            posts.append(post)
        
        # Create publishing schedule
        publishing_schedule = self._create_publishing_schedule(posts, series_config)
        
        # Create cross-promotion strategy
        cross_promotion = self._create_cross_promotion_strategy(posts, series_config)
        
        return ContentSeries(
            series_id=series_id,
            title=series_title,
            description=series_config.get("description", f"A {num_posts}-part series on {series_title}"),
            total_posts=num_posts,
            posts=posts,
            publishing_schedule=publishing_schedule,
            cross_promotion_strategy=cross_promotion
        )
    
    async def _break_into_series(self, content: str, num_posts: int) -> List[str]:
        """Break content into logical parts for a series."""
        
        breakdown_prompt = f"""
        Break the following content into {num_posts} logical parts for a content series.
        Each part should be substantial enough for a standalone post but connected to the overall theme.
        
        Return each part separated by "---PART---"
        
        Original content:
        {content}
        """
        
        try:
            response = await self.optimization_llm.agenerate([
                [HumanMessage(content=breakdown_prompt)]
            ])
            
            parts_text = response.generations[0][0].text
            parts = [part.strip() for part in parts_text.split("---PART---") if part.strip()]
            
            # Ensure we have the right number of parts
            if len(parts) != num_posts:
                # Split content by paragraphs and distribute
                paragraphs = content.split('\n\n')
                parts_per_post = max(1, len(paragraphs) // num_posts)
                
                parts = []
                for i in range(0, len(paragraphs), parts_per_post):
                    part = '\n\n'.join(paragraphs[i:i + parts_per_post])
                    if part.strip():
                        parts.append(part.strip())
                
                # Adjust to exact count
                parts = parts[:num_posts]
                while len(parts) < num_posts:
                    parts.append(f"Additional insights on {content[:100]}...")
            
            return parts
            
        except Exception as e:
            self.logger.error(f"Content breakdown failed: {str(e)}")
            # Fallback: simple content splitting
            words = content.split()
            words_per_part = len(words) // num_posts
            
            parts = []
            for i in range(num_posts):
                start_idx = i * words_per_part
                end_idx = (i + 1) * words_per_part if i < num_posts - 1 else len(words)
                part = ' '.join(words[start_idx:end_idx])
                parts.append(part)
            
            return parts
    
    async def _generate_series_hashtags(
        self,
        series_title: str,
        post_number: int,
        total_posts: int,
        platform: ContentPlatform
    ) -> List[str]:
        """Generate hashtags for a series post."""
        
        base_hashtags = await self._generate_hashtags(
            {"main_topic": series_title, "keywords": [series_title.lower()]},
            platform,
            count=3
        )
        
        # Add series-specific hashtags
        series_hashtag = re.sub(r'[^a-zA-Z0-9]', '', series_title.replace(' ', ''))
        series_hashtags = [
            series_hashtag,
            f"{series_hashtag}Part{post_number}",
            "ContentSeries"
        ]
        
        return base_hashtags + series_hashtags
    
    def _create_publishing_schedule(
        self,
        posts: List[ContentSeriesPost],
        series_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create optimal publishing schedule for the series."""
        
        frequency = series_config.get("frequency", "daily")  # daily, weekly, bi-weekly
        start_date = series_config.get("start_date", datetime.now().isoformat())
        
        schedule = {
            "frequency": frequency,
            "start_date": start_date,
            "posts_schedule": []
        }
        
        # Calculate posting intervals
        intervals = {
            "daily": 1,
            "weekly": 7,
            "bi-weekly": 14
        }
        
        interval_days = intervals.get(frequency, 1)
        
        for i, post in enumerate(posts):
            post_date = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if 'T' in start_date else datetime.now()
            post_date = post_date.replace(tzinfo=None) + timedelta(days=i * interval_days)
            
            schedule["posts_schedule"].append({
                "post_number": post.post_number,
                "platform": post.platform,
                "scheduled_date": post_date.isoformat(),
                "optimal_time": self._get_optimal_posting_time(post.platform)
            })
        
        return schedule
    
    def _get_optimal_posting_time(self, platform: ContentPlatform) -> str:
        """Get optimal posting time for platform."""
        
        optimal_times = {
            ContentPlatform.LINKEDIN_POST: "09:00",  # 9 AM - business hours
            ContentPlatform.TWITTER_THREAD: "12:00",  # Noon - lunch break
            ContentPlatform.INSTAGRAM_POST: "18:00",  # 6 PM - evening engagement
            ContentPlatform.FACEBOOK_POST: "15:00"   # 3 PM - afternoon break
        }
        
        return optimal_times.get(platform, "12:00")
    
    def _create_cross_promotion_strategy(
        self,
        posts: List[ContentSeriesPost],
        series_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create cross-promotion strategy for the series."""
        
        return {
            "teaser_strategy": {
                "create_teaser_post": True,
                "teaser_content": f"Starting a new {len(posts)}-part series on {posts[0].series_title}. Follow along for insights!",
                "teaser_platforms": list(set(post.platform for post in posts))
            },
            "cross_references": {
                "reference_previous_posts": True,
                "reference_upcoming_posts": True,
                "use_series_hashtag": True
            },
            "finale_strategy": {
                "create_summary_post": True,
                "include_all_links": True,
                "thank_audience": True
            },
            "engagement_strategy": {
                "ask_questions_each_post": True,
                "encourage_series_following": True,
                "create_discussion_threads": True
            }
        }
    
    async def one_click_repurpose(
        self,
        blog_content: str,
        blog_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """One-click repurposing from blog to multiple social platforms."""
        
        # Define the repurposing chain: Blog → LinkedIn → Twitter → Instagram
        repurposing_chain = [
            ContentPlatform.LINKEDIN_POST,
            ContentPlatform.TWITTER_THREAD,
            ContentPlatform.INSTAGRAM_POST,
            ContentPlatform.FACEBOOK_POST
        ]
        
        # Repurpose for all platforms
        repurposed_content = await self.repurpose_content(
            original_content=blog_content,
            target_platforms=repurposing_chain,
            source_context={
                "platform": "blog",
                "company_context": blog_context.get("company_context", ""),
                **blog_context
            },
            customization_options={
                "include_cta": True,
                "hashtag_count": 5,
                "tone": ContentTone.PROFESSIONAL
            }
        )
        
        # Calculate total reach potential
        total_reach_score = sum(
            content.estimated_engagement.get("score", 0) 
            for content in repurposed_content.values()
        ) / len(repurposed_content)
        
        # Generate content calendar
        content_calendar = self._create_content_calendar(repurposed_content)
        
        # Create performance predictions
        performance_predictions = self._predict_performance(repurposed_content)
        
        return {
            "repurposed_content": {
                platform: asdict(content) 
                for platform, content in repurposed_content.items()
            },
            "content_calendar": content_calendar,
            "performance_predictions": performance_predictions,
            "total_reach_score": round(total_reach_score),
            "recommendations": self._generate_repurposing_recommendations(repurposed_content),
            "created_at": datetime.now().isoformat()
        }
    
    def _create_content_calendar(
        self, 
        repurposed_content: Dict[ContentPlatform, RepurposedContent]
    ) -> Dict[str, Any]:
        """Create a suggested content calendar for repurposed content."""
        
        base_date = datetime.now()
        calendar = {}
        
        # Suggested posting schedule
        posting_schedule = {
            ContentPlatform.LINKEDIN_POST: 0,      # Post immediately
            ContentPlatform.TWITTER_THREAD: 1,    # 1 day later
            ContentPlatform.INSTAGRAM_POST: 2,    # 2 days later
            ContentPlatform.FACEBOOK_POST: 3      # 3 days later
        }
        
        for platform, days_offset in posting_schedule.items():
            if platform in repurposed_content:
                post_date = base_date + timedelta(days=days_offset)
                optimal_time = self._get_optimal_posting_time(platform)
                
                calendar[platform.value] = {
                    "suggested_date": post_date.strftime("%Y-%m-%d"),
                    "optimal_time": optimal_time,
                    "day_of_week": post_date.strftime("%A"),
                    "reasoning": f"Optimal timing for {platform.value} audience engagement"
                }
        
        return calendar
    
    def _predict_performance(
        self,
        repurposed_content: Dict[ContentPlatform, RepurposedContent]
    ) -> Dict[str, Any]:
        """Predict performance metrics for repurposed content."""
        
        predictions = {}
        
        for platform, content in repurposed_content.items():
            engagement_score = content.estimated_engagement.get("score", 50)
            
            # Platform-specific reach multipliers
            reach_multipliers = {
                ContentPlatform.LINKEDIN_POST: 1.2,
                ContentPlatform.TWITTER_THREAD: 1.5,
                ContentPlatform.INSTAGRAM_POST: 1.1,
                ContentPlatform.FACEBOOK_POST: 0.8
            }
            
            base_reach = 100  # Base reach estimate
            predicted_reach = base_reach * reach_multipliers.get(platform, 1.0) * (engagement_score / 50)
            
            predictions[platform.value] = {
                "estimated_reach": round(predicted_reach),
                "estimated_engagement_rate": f"{engagement_score/10:.1f}%",
                "predicted_interactions": round(predicted_reach * engagement_score / 1000),
                "confidence_level": "medium" if engagement_score > 60 else "low"
            }
        
        return predictions
    
    def _generate_repurposing_recommendations(
        self,
        repurposed_content: Dict[ContentPlatform, RepurposedContent]
    ) -> List[str]:
        """Generate recommendations for improving repurposed content."""
        
        recommendations = []
        
        # Analyze content quality across platforms
        avg_engagement_score = sum(
            content.estimated_engagement.get("score", 0) 
            for content in repurposed_content.values()
        ) / len(repurposed_content)
        
        if avg_engagement_score < 60:
            recommendations.append("Consider adding more engaging hooks and questions to increase interaction potential")
        
        # Platform-specific recommendations
        for platform, content in repurposed_content.items():
            if len(content.hashtags) < 3:
                recommendations.append(f"Add more relevant hashtags for {platform.value} to improve discoverability")
            
            if not content.call_to_action:
                recommendations.append(f"Include a call-to-action in {platform.value} content to drive engagement")
            
            if content.character_count > PLATFORM_SPECS[platform].optimal_length:
                recommendations.append(f"Consider shortening {platform.value} content for better engagement")
        
        # Cross-platform recommendations
        if len(repurposed_content) > 2:
            recommendations.append("Schedule posts across different days to maximize reach and avoid audience fatigue")
        
        recommendations.append("Monitor performance and adjust future repurposing based on engagement data")
        
        return recommendations