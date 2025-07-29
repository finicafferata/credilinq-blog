"""
Repurpose Agent - Transforms content into platform-optimized formats with expert prompts.
"""

from typing import List, Dict, Any, Optional
import ast
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator


class ContentRepurposingAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for transforming content into platform-optimized formats.
    Uses expert-level prompts to create engaging, platform-specific content that
    maximizes reach and engagement across different social media channels.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_REPURPOSER,
                name="ContentRepurposingAgent",
                description="Transforms content into platform-optimized formats with expert prompts",
                capabilities=[
                    "content_repurposing",
                    "platform_optimization",
                    "engagement_maximization",
                    "multi_format_adaptation",
                    "tone_adjustment",
                    "hashtag_optimization"
                ],
                version="2.1.0"  # Version bumped to reflect improvements
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
        
        # Enhanced platform-specific configuration with engagement optimization
        self.platform_config = {
            "LinkedIn Post": {
                "max_chars": 1300,
                "style": "professional_thought_leader",
                "optimal_length": "800-1200",
                "engagement_tactics": ["storytelling", "data_insights", "call_to_discussion"],
                "hashtag_strategy": "industry_specific",
                "max_hashtags": 5,
                "format_structure": ["hook", "value", "proof", "cta"]
            },
            "Tweet Thread": {
                "max_chars_per_tweet": 280,
                "thread_length": "3-7 tweets",
                "style": "conversational_expert",
                "engagement_tactics": ["thread_hooks", "numbered_insights", "retweet_worthy"],
                "hashtag_strategy": "trending_relevant",
                "max_hashtags": 3,
                "format_structure": ["hook_tweet", "value_tweets", "conclusion_cta"]
            },
            "Instagram Caption": {
                "max_chars": 2200,
                "style": "visual_storytelling",
                "optimal_length": "1000-1500",
                "engagement_tactics": ["emoji_storytelling", "user_questions", "community_building"],
                "hashtag_strategy": "discovery_mix",
                "max_hashtags": 10,
                "format_structure": ["visual_hook", "story", "value", "community_cta"]
            },
            "Facebook Post": {
                "max_chars": 500,
                "style": "conversational_community",
                "optimal_length": "200-400",
                "engagement_tactics": ["personal_stories", "community_questions", "shareability"],
                "hashtag_strategy": "minimal_relevant",
                "max_hashtags": 2,
                "format_structure": ["personal_hook", "value", "community_question"]
            },
            "Email Newsletter": {
                "max_words": 300,
                "style": "professional_personal",
                "structure_required": True,
                "engagement_tactics": ["personalization", "actionable_tips", "exclusive_content"],
                "format_structure": ["subject", "greeting", "value", "cta", "signature"]
            },
            "YouTube Description": {
                "max_chars": 1000,
                "style": "seo_optimized",
                "engagement_tactics": ["keyword_rich", "timestamp_value", "playlist_promotion"],
                "hashtag_strategy": "seo_discovery",
                "max_hashtags": 15,
                "format_structure": ["summary", "key_points", "links", "tags"]
            }
        }

    def _initialize(self):
        """Initialize the LLM and other resources."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,  # Higher creativity for content adaptation
                openai_api_key=settings.OPENAI_API_KEY
            )
            self.logger.info("ContentRepurposingAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ContentRepurposingAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for content repurposing."""
        super()._validate_input(input_data)
        
        required_fields = ["content", "target_format"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate target format
        target_format = input_data["target_format"]
        if target_format not in self.platform_config:
            supported = list(self.platform_config.keys())
            raise ValueError(f"Unsupported format '{target_format}'. Supported: {supported}")
        
        # Security validation
        self.security_validator.validate_input(str(input_data["content"]))
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Transform content into platform-optimized format with engagement maximization.
        
        Args:
            input_data: Dictionary containing:
                - content: Original content to repurpose
                - target_format: Target platform format
                - company_context: Optional brand context
                - engagement_goals: Optional specific engagement objectives
            context: Execution context
            
        Returns:
            AgentResult: Result containing the repurposed content and analysis
        """
        try:
            # Initialize if not already done
            if self.llm is None:
                self._initialize()
            
            content = input_data["content"]
            target_format = input_data["target_format"]
            company_context = input_data.get("company_context", "")
            engagement_goals = input_data.get("engagement_goals", "maximize engagement")
            
            self.logger.info(f"Repurposing content for {target_format} ({len(content.split())} words)")
            
            # Extract key insights and value propositions
            content_analysis = self._analyze_content(content)
            
            # Generate platform-optimized content
            repurposed_content = self._generate_repurposed_content(
                content, target_format, company_context, engagement_goals, content_analysis
            )
            
            # Analyze the repurposed content
            output_analysis = self._analyze_repurposed_content(repurposed_content, target_format)
            
            result_data = {
                "repurposed_content": repurposed_content,
                "target_format": target_format,
                "content_analysis": content_analysis,
                "output_analysis": output_analysis,
                "optimization_applied": self._get_optimization_summary(target_format)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "content_repurposer",
                    "target_platform": target_format,
                    "content_length": len(repurposed_content)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to repurpose content: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CONTENT_REPURPOSING_FAILED"
            )

    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to extract key insights and value propositions."""
        
        prompt = f"""
        Act as a content strategist analyzing this content. Extract key insights for repurposing.

        **Content to Analyze:**
        {content[:2000]}

        **Analysis Required:**
        - Identify the 3 most compelling value propositions
        - Extract 5 key actionable insights
        - Determine the primary emotional appeal (inspiration, education, entertainment, etc.)
        - Identify the target audience type
        - Note any data points, statistics, or credible sources mentioned

        **Output Format:**
        Return analysis as JSON inside <analysis> tags:
        <analysis>
        {{
            "value_propositions": ["Prop 1", "Prop 2", "Prop 3"],
            "key_insights": ["Insight 1", "Insight 2", "Insight 3", "Insight 4", "Insight 5"],
            "emotional_appeal": "primary emotion",
            "target_audience": "audience description",
            "credibility_elements": ["data point 1", "source 2"]
        }}
        </analysis>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            # Parse JSON from response
            match = re.search(r"<analysis>(.*?)</analysis>", response.content, re.DOTALL)
            if match:
                analysis = json.loads(match.group(1).strip())
                return analysis
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {str(e)}, using fallback")
        
        # Fallback analysis
        return self._fallback_content_analysis(content)
    
    def _fallback_content_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback content analysis when AI fails."""
        # Simple extraction based on content structure
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        return {
            "value_propositions": sentences[:3] if sentences else ["Value-driven content"],
            "key_insights": sentences[:5] if len(sentences) >= 5 else sentences + ["Additional insights"],
            "emotional_appeal": "educational",
            "target_audience": "professional audience",
            "credibility_elements": ["industry expertise", "practical experience"]
        }
    
    def _generate_repurposed_content(
        self,
        content: str,
        target_format: str,
        company_context: str,
        engagement_goals: str,
        content_analysis: Dict[str, Any]
    ) -> str:
        """Generate platform-optimized content using expert prompts."""
        
        config = self.platform_config[target_format]
        
        # Route to platform-specific expert prompts
        if target_format == "LinkedIn Post":
            return self._create_linkedin_post(content, config, company_context, content_analysis)
        elif target_format == "Tweet Thread":
            return self._create_tweet_thread(content, config, company_context, content_analysis)
        elif target_format == "Instagram Caption":
            return self._create_instagram_caption(content, config, company_context, content_analysis)
        elif target_format == "Facebook Post":
            return self._create_facebook_post(content, config, company_context, content_analysis)
        elif target_format == "Email Newsletter":
            return self._create_email_newsletter(content, config, company_context, content_analysis)
        elif target_format == "YouTube Description":
            return self._create_youtube_description(content, config, company_context, content_analysis)
        else:
            return self._create_generic_repurpose(content, config, company_context, content_analysis)

    def _create_linkedin_post(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create LinkedIn post with professional thought leadership approach."""
        
        value_props = content_analysis.get("value_propositions", [])
        key_insights = content_analysis.get("key_insights", [])
        
        prompt = f"""
        Act as a LinkedIn thought leader and professional content strategist with 10+ years of experience creating viral B2B content. Transform this content into a compelling LinkedIn post that stops the scroll and drives meaningful professional engagement.

        **Source Content Analysis:**
        - Value Propositions: {', '.join(value_props[:2])}
        - Key Insights: {', '.join(key_insights[:3])}
        - Target: {content_analysis.get('target_audience', 'professionals')}

        **Company Context:** "{company_context}"

        **Original Content Preview:**
        {content[:1500]}

        **LinkedIn Post Requirements:**
        - Maximum {config['max_chars']} characters for optimal algorithm performance
        - Follow the proven engagement structure: {' → '.join(config['format_structure'])}
        - Use {config['style']} tone that builds authority and trust
        - Apply engagement tactics: {', '.join(config['engagement_tactics'])}
        - Include {config['max_hashtags']} strategic hashtags using {config['hashtag_strategy']} approach
        - Target length: {config['optimal_length']} characters for maximum reach

        **Professional Guidelines:**
        - Start with a scroll-stopping hook in the first 2 lines
        - Use line breaks and white space for mobile readability  
        - Include a data point or statistic if available from the content
        - End with a thought-provoking question or clear call-to-action
        - Maintain authentic professional voice aligned with company context

        **Negative Constraints:**
        - **Avoid** generic corporate speak or buzzwords
        - **Do not** exceed character limits or algorithm will suppress reach
        - **Avoid** more than 2 emojis - keep it professional

        **Output Format:**
        Return ONLY the complete LinkedIn post inside <post> tags:
        <post>
        [Your LinkedIn post content]
        </post>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return self._parse_tagged_response(response.content, "post")
        except Exception as e:
            self.logger.warning(f"LinkedIn post generation failed: {str(e)}")
            return self._create_fallback_linkedin_post(content, value_props)

    def _create_tweet_thread(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create Twitter thread with viral engagement optimization."""
        
        key_insights = content_analysis.get("key_insights", [])
        
        prompt = f"""
        Act as a Twitter growth expert and viral content creator with deep understanding of the Twitter algorithm. Create a compelling thread that maximizes engagement and retweets.

        **Content Analysis:**
        - Key Insights: {', '.join(key_insights[:4])}
        - Emotional Appeal: {content_analysis.get('emotional_appeal', 'educational')}

        **Company Context:** "{company_context}"

        **Source Material:**
        {content[:1200]}

        **Twitter Thread Requirements:**
        - Each tweet maximum {config['max_chars_per_tweet']} characters
        - Thread length: {config['thread_length']} for optimal engagement
        - Style: {config['style']} with authority and authenticity
        - Apply tactics: {', '.join(config['engagement_tactics'])}
        - Structure: {' → '.join(config['format_structure'])}
        - Include {config['max_hashtags']} trending hashtags

        **Viral Thread Formula:**
        1. Hook tweet: Bold claim or surprising insight that stops the scroll
        2. Value tweets: Break down key insights with numbered points
        3. Proof/examples: Add credibility with data or real examples
        4. Conclusion: Summarize with clear takeaway and CTA

        **Engagement Optimization:**
        - Use thread connectors (1/n, 2/n) for clarity
        - Include 🧵 emoji to signal thread format
        - Ask engaging questions to boost replies
        - Use strategic line breaks for readability

        **Output Format:**
        Return the complete thread inside <thread> tags:
        <thread>
        1/🧵 [First tweet]

        2/🧵 [Second tweet]

        [Continue for all tweets...]
        </thread>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return self._parse_tagged_response(response.content, "thread")
        except Exception as e:
            self.logger.warning(f"Tweet thread generation failed: {str(e)}")
            return self._create_fallback_tweet_thread(content, key_insights)

    def _create_instagram_caption(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create Instagram caption with visual storytelling approach."""
        
        value_props = content_analysis.get("value_propositions", [])
        
        prompt = f"""
        Act as an Instagram content creator and visual storytelling expert who understands how to create engaging captions that drive comments, saves, and shares.

        **Content Insights:**
        - Value Props: {', '.join(value_props[:2])}
        - Emotional Appeal: {content_analysis.get('emotional_appeal', 'inspirational')}
        - Target: {content_analysis.get('target_audience', 'engaged community')}

        **Brand Context:** "{company_context}"

        **Source Content:**
        {content[:1000]}

        **Instagram Caption Requirements:**
        - Maximum {config['max_chars']} characters
        - Style: {config['style']} with authentic personality
        - Length: {config['optimal_length']} characters for optimal reach
        - Structure: {' → '.join(config['format_structure'])}
        - Tactics: {', '.join(config['engagement_tactics'])}
        - Hashtags: {config['max_hashtags']} using {config['hashtag_strategy']} strategy

        **Visual Storytelling Elements:**
        - Start with an emoji or visual element that complements the image
        - Use storytelling techniques to create emotional connection
        - Include personal anecdotes or relatable situations
        - Break text into digestible paragraphs with line breaks
        - End with a community question to boost engagement

        **Caption Structure:**
        - Visual Hook (emoji + opening line)
        - Story/Context (2-3 sentences)
        - Value/Insight (main takeaway)
        - Community Call-to-Action (question or request)
        - Hashtags (separated from main caption)

        **Output Format:**
        Return the complete caption inside <caption> tags:
        <caption>
        [Your Instagram caption]
        </caption>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return self._parse_tagged_response(response.content, "caption")
        except Exception as e:
            self.logger.warning(f"Instagram caption generation failed: {str(e)}")
            return self._create_fallback_instagram_caption(content, value_props)

    def _create_facebook_post(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create Facebook post optimized for community engagement."""
        
        key_insights = content_analysis.get("key_insights", [])
        
        prompt = f"""
        Act as a Facebook community manager who excels at creating posts that generate meaningful discussions and build strong communities.

        **Content Elements:**
        - Key Insights: {', '.join(key_insights[:2])}
        - Community Focus: {content_analysis.get('target_audience', 'engaged community')}

        **Brand Voice:** "{company_context}"

        **Source Material:**
        {content[:800]}

        **Facebook Post Requirements:**
        - Maximum {config['max_chars']} characters for optimal reach
        - Style: {config['style']} that feels personal and authentic
        - Target length: {config['optimal_length']} characters
        - Structure: {' → '.join(config['format_structure'])}
        - Engagement tactics: {', '.join(config['engagement_tactics'])}
        - Minimal hashtags: {config['max_hashtags']} maximum

        **Community Engagement Focus:**
        - Use conversational tone that invites discussion
        - Share relatable experiences or stories
        - Ask open-ended questions that generate thoughtful responses
        - Include a clear call-to-action for community interaction
        - Focus on building relationships rather than promotion

        **Output Format:**
        Return the complete post inside <fbpost> tags:
        <fbpost>
        [Your Facebook post content]
        </fbpost>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return self._parse_tagged_response(response.content, "fbpost")
        except Exception as e:
            self.logger.warning(f"Facebook post generation failed: {str(e)}")
            return self._create_fallback_facebook_post(content, key_insights)

    def _create_email_newsletter(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create email newsletter with personalized approach."""
        
        value_props = content_analysis.get("value_propositions", [])
        key_insights = content_analysis.get("key_insights", [])
        
        prompt = f"""
        Act as an email marketing specialist who creates newsletters that subscribers eagerly anticipate and never unsubscribe from.

        **Content Analysis:**
        - Value Props: {', '.join(value_props[:2])}
        - Key Insights: {', '.join(key_insights[:3])}

        **Brand Context:** "{company_context}"

        **Source Content:**
        {content[:1500]}

        **Email Newsletter Requirements:**
        - Maximum {config['max_words']} words for optimal engagement
        - Style: {config['style']} with warmth and expertise
        - Structure: {' → '.join(config['format_structure'])}
        - Tactics: {', '.join(config['engagement_tactics'])}

        **Newsletter Structure:**
        Subject: [Compelling 40-character subject line]
        
        [Personal greeting]
        [Brief, engaging introduction]
        [Main value content with key insights]
        [Clear call-to-action]
        [Personal sign-off]

        **Email Best Practices:**
        - Subject line should create curiosity without being clickbait
        - Personal greeting that feels authentic
        - Scannable content with bullet points or short paragraphs
        - One clear, compelling call-to-action
        - Friendly, personal sign-off that builds relationship

        **Output Format:**
        Return the complete email inside <email> tags:
        <email>
        [Your complete email newsletter]
        </email>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return self._parse_tagged_response(response.content, "email")
        except Exception as e:
            self.logger.warning(f"Email newsletter generation failed: {str(e)}")
            return self._create_fallback_email_newsletter(content, value_props)

    def _create_youtube_description(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create YouTube description optimized for SEO and discovery."""
        
        key_insights = content_analysis.get("key_insights", [])
        
        prompt = f"""
        Act as a YouTube SEO specialist who understands how to optimize descriptions for maximum discoverability and engagement.

        **Content Elements:**
        - Key Topics: {', '.join(key_insights[:4])}
        - Target Audience: {content_analysis.get('target_audience', 'YouTube viewers')}

        **Channel Context:** "{company_context}"

        **Video Content Overview:**
        {content[:1000]}

        **YouTube Description Requirements:**
        - Maximum {config['max_chars']} characters
        - Style: {config['style']} for search optimization
        - Structure: {' → '.join(config['format_structure'])}
        - Tactics: {', '.join(config['engagement_tactics'])}
        - Hashtags: {config['max_hashtags']} for {config['hashtag_strategy']}

        **SEO Optimization:**
        - Include key topics in first 125 characters (preview text)
        - Use relevant keywords naturally throughout
        - Add timestamps if applicable
        - Include links to related content
        - Strategic hashtag placement for discovery

        **Output Format:**
        Return the complete description inside <description> tags:
        <description>
        [Your YouTube description]
        </description>
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            return self._parse_tagged_response(response.content, "description")
        except Exception as e:
            self.logger.warning(f"YouTube description generation failed: {str(e)}")
            return self._create_fallback_youtube_description(content, key_insights)

    def _create_generic_repurpose(
        self, 
        content: str, 
        config: Dict[str, Any], 
        company_context: str, 
        content_analysis: Dict[str, Any]
    ) -> str:
        """Create generic repurposed content as fallback."""
        
        key_insights = content_analysis.get("key_insights", [])
        
        return f"""Based on the original content, here are the key takeaways:

{chr(10).join([f"• {insight}" for insight in key_insights[:3]])}

{company_context}

For more insights like this, stay connected with our content."""

    def _parse_tagged_response(self, response: str, tag: str) -> str:
        """Parse response content from specific tags."""
        try:
            match = re.search(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        
        # Fallback: return the entire response
        return response.strip()

    def _create_fallback_linkedin_post(self, content: str, value_props: List[str]) -> str:
        """Fallback LinkedIn post when generation fails."""
        preview = content[:200] + "..." if len(content) > 200 else content
        prop = value_props[0] if value_props else "Key insights from our latest content"
        
        return f"""💡 {prop}

{preview}

What's your experience with this? Share your thoughts below.

#Professional #Industry #Growth #Innovation #Leadership"""

    def _create_fallback_tweet_thread(self, content: str, insights: List[str]) -> str:
        """Fallback tweet thread when generation fails."""
        insight1 = insights[0] if insights else "Key insight from latest content"
        insight2 = insights[1] if len(insights) > 1 else "Additional valuable takeaway"
        
        return f"""1/🧵 {insight1}

2/🧵 {insight2}

3/🧵 What are your thoughts on this? 

#Thread #Insights #Growth"""

    def _create_fallback_instagram_caption(self, content: str, value_props: List[str]) -> str:
        """Fallback Instagram caption when generation fails."""
        prop = value_props[0] if value_props else "Sharing valuable insights"
        
        return f"""✨ {prop}

Swipe to learn more about this important topic. 

What's your experience? Share in the comments! 👇

#inspiration #growth #community #learning #motivation #success #mindset #tips #lifestyle #content"""

    def _create_fallback_facebook_post(self, content: str, insights: List[str]) -> str:
        """Fallback Facebook post when generation fails."""
        insight = insights[0] if insights else "Important insights to share"
        
        return f"""{insight}

What do you think about this? I'd love to hear your perspective in the comments.

#community #discussion"""

    def _create_fallback_email_newsletter(self, content: str, value_props: List[str]) -> str:
        """Fallback email newsletter when generation fails."""
        prop = value_props[0] if value_props else "Valuable insights"
        
        return f"""Subject: {prop[:40]}

Hi there,

Hope you're having a great week! 

{prop}

{content[:300]}...

Best regards,
The Team"""

    def _create_fallback_youtube_description(self, content: str, insights: List[str]) -> str:
        """Fallback YouTube description when generation fails."""
        insight = insights[0] if insights else "Educational content"
        
        return f"""{insight}

In this video, we explore key topics that matter to you.

Key Points:
{chr(10).join([f"• {insight}" for insight in insights[:3]])}

Subscribe for more content like this!

#education #learning #content #insights #growth"""

    def _analyze_repurposed_content(self, content: str, target_format: str) -> Dict[str, Any]:
        """Analyze the quality and characteristics of repurposed content."""
        config = self.platform_config[target_format]
        
        analysis = {
            "character_count": len(content),
            "word_count": len(content.split()),
            "hashtag_count": content.count('#'),
            "emoji_count": len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', content)),
            "line_breaks": content.count('\n'),
            "engagement_elements": {
                "questions": content.count('?'),
                "calls_to_action": len(re.findall(r'\b(share|comment|like|follow|subscribe|join|learn|discover)\b', content.lower())),
                "personal_pronouns": len(re.findall(r'\b(you|your|we|our|us|me|my|I)\b', content.lower()))
            }
        }
        
        # Platform-specific analysis
        if target_format == "LinkedIn Post":
            analysis["within_optimal_range"] = 800 <= len(content) <= 1200
            analysis["professional_tone"] = any(word in content.lower() for word in ['professional', 'industry', 'business', 'strategy'])
        elif target_format == "Tweet Thread":
            tweets = [t.strip() for t in content.split('\n') if t.strip()]
            analysis["thread_length"] = len(tweets)
            analysis["avg_tweet_length"] = sum(len(t) for t in tweets) / len(tweets) if tweets else 0
        
        return analysis

    def _get_optimization_summary(self, target_format: str) -> Dict[str, Any]:
        """Get summary of optimizations applied for the target format."""
        config = self.platform_config[target_format]
        
        return {
            "platform": target_format,
            "style_applied": config["style"],
            "engagement_tactics": config["engagement_tactics"],
            "format_structure": config["format_structure"],
            "character_limit": config.get("max_chars", config.get("max_words", "N/A")),
            "hashtag_strategy": config.get("hashtag_strategy", "not applicable")
        }