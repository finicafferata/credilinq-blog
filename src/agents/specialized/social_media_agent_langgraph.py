"""
SocialMediaAgent LangGraph Implementation - Advanced social media content adaptation.
"""

from typing import Dict, Any, Optional, List, TypedDict, Tuple
from enum import Enum
import re
import json
import asyncio
from dataclasses import dataclass
from langchain_core.messages import SystemMessage
from src.core.llm_client import create_llm
# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.langgraph_base import (
    LangGraphWorkflowBase,
    WorkflowState,
    LangGraphExecutionContext,
    CheckpointStrategy,
    WorkflowStatus
)
from ..core.base_agent import AgentResult, AgentType, AgentMetadata
from ...core.security import SecurityValidator


class SocialMediaPhase(str, Enum):
    """Phases of the social media adaptation workflow."""
    INITIALIZATION = "initialization"
    CONTENT_ANALYSIS = "content_analysis"
    PLATFORM_ADAPTATION = "platform_adaptation"
    HASHTAG_RESEARCH = "hashtag_research"
    ENGAGEMENT_OPTIMIZATION = "engagement_optimization"
    SCHEDULING_OPTIMIZATION = "scheduling_optimization"
    FINAL_REVIEW = "final_review"


@dataclass
class PlatformConfig:
    """Configuration for social media platforms."""
    name: str
    max_length: int
    optimal_length: int
    hashtag_limit: int
    tone: str
    content_type: str
    call_to_action: bool
    visual_priority: bool = False
    thread_support: bool = False
    link_preview: bool = True


class SocialMediaState(TypedDict):
    """State for the social media workflow."""
    # Input data
    content: str
    blog_title: str
    outline: List[str]
    target_platforms: List[str]
    target_audience: str
    brand_voice: str
    content_goals: List[str]
    
    # Content analysis
    key_points: List[str]
    content_themes: List[str]
    call_to_actions: List[str]
    visual_suggestions: List[str]
    
    # Platform adaptations
    platform_posts: Dict[str, Dict[str, Any]]
    post_variations: Dict[str, List[str]]
    thread_versions: Dict[str, List[str]]
    
    # Hashtag research
    platform_hashtags: Dict[str, List[str]]
    trending_hashtags: Dict[str, List[str]]
    branded_hashtags: List[str]
    
    # Engagement optimization
    engagement_strategies: Dict[str, List[str]]
    optimal_posting_times: Dict[str, str]
    audience_targeting: Dict[str, str]
    
    # Scheduling and campaign
    posting_schedule: Dict[str, List[str]]
    campaign_strategy: Dict[str, Any]
    cross_promotion: Dict[str, List[str]]
    
    # Results
    social_media_score: float
    performance_predictions: Dict[str, Dict[str, float]]
    optimization_recommendations: List[str]
    
    # Workflow metadata
    current_phase: str
    adaptation_quality: str
    errors: List[str]
    warnings: List[str]


class SocialMediaAgentWorkflow(LangGraphWorkflowBase[SocialMediaState]):
    """
    LangGraph-based SocialMediaAgent with comprehensive platform adaptation.
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        workflow_name: str = "social_media_agent_workflow",
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.DATABASE_PERSISTENT
    ):
        """
        Initialize the SocialMediaAgent workflow.
        
        Args:
            llm: Language model for content adaptation
            workflow_name: Name of the workflow
            checkpoint_strategy: When to save checkpoints
        """
        self.llm = llm
        self.security_validator = SecurityValidator()
        
        # Platform configurations
        self.platforms = {
            "linkedin": PlatformConfig(
                name="linkedin",
                max_length=3000,
                optimal_length=1300,
                hashtag_limit=5,
                tone="professional",
                content_type="business",
                call_to_action=True,
                link_preview=True
            ),
            "twitter": PlatformConfig(
                name="twitter",
                max_length=280,
                optimal_length=240,
                hashtag_limit=2,
                tone="conversational",
                content_type="news",
                call_to_action=False,
                thread_support=True
            ),
            "facebook": PlatformConfig(
                name="facebook",
                max_length=63206,
                optimal_length=400,
                hashtag_limit=3,
                tone="friendly",
                content_type="engagement",
                call_to_action=True,
                link_preview=True
            ),
            "instagram": PlatformConfig(
                name="instagram",
                max_length=2200,
                optimal_length=150,
                hashtag_limit=30,
                tone="visual",
                content_type="visual",
                call_to_action=True,
                visual_priority=True
            ),
            "tiktok": PlatformConfig(
                name="tiktok",
                max_length=2200,
                optimal_length=100,
                hashtag_limit=5,
                tone="casual",
                content_type="viral",
                call_to_action=True,
                visual_priority=True
            )
        }
        
        # Initialize base class
        super().__init__(
            workflow_name=workflow_name,
            checkpoint_strategy=checkpoint_strategy
        )
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create and configure the LangGraph workflow structure."""
        workflow = StateGraph(SocialMediaState)
        
        # Add nodes for each phase
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("content_analysis", self.content_analysis_node)
        workflow.add_node("platform_adaptation", self.platform_adaptation_node)
        workflow.add_node("hashtag_research", self.hashtag_research_node)
        workflow.add_node("engagement_optimization", self.engagement_optimization_node)
        workflow.add_node("scheduling_optimization", self.scheduling_optimization_node)
        workflow.add_node("final_review", self.final_review_node)
        
        # Define edges
        workflow.set_entry_point("initialization")
        
        workflow.add_edge("initialization", "content_analysis")
        workflow.add_edge("content_analysis", "platform_adaptation")
        workflow.add_edge("platform_adaptation", "hashtag_research")
        workflow.add_edge("hashtag_research", "engagement_optimization")
        workflow.add_edge("engagement_optimization", "scheduling_optimization")
        workflow.add_edge("scheduling_optimization", "final_review")
        workflow.add_edge("final_review", END)
        
        return workflow
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> SocialMediaState:
        """Create the initial state for the workflow."""
        return SocialMediaState(
            # Input data
            content=input_data.get("content", ""),
            blog_title=input_data.get("blog_title", ""),
            outline=input_data.get("outline", []),
            target_platforms=input_data.get("target_platforms", ["linkedin", "twitter", "facebook"]),
            target_audience=input_data.get("target_audience", "professionals"),
            brand_voice=input_data.get("brand_voice", "professional"),
            content_goals=input_data.get("content_goals", ["awareness", "engagement"]),
            
            # Content analysis - will be filled during workflow
            key_points=[],
            content_themes=[],
            call_to_actions=[],
            visual_suggestions=[],
            
            # Platform adaptations - will be filled during workflow
            platform_posts={},
            post_variations={},
            thread_versions={},
            
            # Hashtag research - will be filled during workflow
            platform_hashtags={},
            trending_hashtags={},
            branded_hashtags=[],
            
            # Engagement optimization - will be filled during workflow
            engagement_strategies={},
            optimal_posting_times={},
            audience_targeting={},
            
            # Scheduling and campaign - will be filled during workflow
            posting_schedule={},
            campaign_strategy={},
            cross_promotion={},
            
            # Results - will be filled during workflow
            social_media_score=0.0,
            performance_predictions={},
            optimization_recommendations=[],
            
            # Workflow metadata
            current_phase=SocialMediaPhase.INITIALIZATION,
            adaptation_quality="pending",
            errors=[],
            warnings=[]
        )
    
    def initialization_node(self, state: SocialMediaState) -> SocialMediaState:
        """Initialize social media workflow."""
        try:
            state["current_phase"] = SocialMediaPhase.INITIALIZATION
            
            # Security validation
            self.security_validator.validate_content(state["content"], "content")
            self.security_validator.validate_content(state["blog_title"], "title")
            
            # Initialize state fields
            state["target_platforms"] = state.get("target_platforms", ["linkedin", "twitter", "facebook"])
            state["target_audience"] = state.get("target_audience", "professionals")
            state["brand_voice"] = state.get("brand_voice", "professional")
            state["content_goals"] = state.get("content_goals", ["awareness", "engagement"])
            state["outline"] = state.get("outline", [])
            state["errors"] = []
            state["warnings"] = []
            
            # Validate platforms
            valid_platforms = [p for p in state["target_platforms"] if p in self.platforms]
            if len(valid_platforms) != len(state["target_platforms"]):
                invalid = set(state["target_platforms"]) - set(valid_platforms)
                state["warnings"].append(f"Invalid platforms ignored: {', '.join(invalid)}")
            state["target_platforms"] = valid_platforms or ["linkedin"]
            
            self.logger.info(
                f"Social media adaptation initialized for {len(state['target_platforms'])} platforms"
            )
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            state["errors"].append(f"Initialization error: {str(e)}")
        
        return state
    
    def content_analysis_node(self, state: SocialMediaState) -> SocialMediaState:
        """Analyze content for social media adaptation."""
        try:
            state["current_phase"] = SocialMediaPhase.CONTENT_ANALYSIS
            
            content = state["content"]
            title = state["blog_title"]
            
            # Extract key points
            state["key_points"] = self._extract_key_points(content, title)
            
            # Identify themes
            state["content_themes"] = self._identify_themes(content, title)
            
            # Extract potential CTAs
            state["call_to_actions"] = self._extract_call_to_actions(content)
            
            # Generate visual suggestions
            state["visual_suggestions"] = self._generate_visual_suggestions(
                content, title, state["content_themes"]
            )
            
            self.logger.info(
                f"Content analysis completed - {len(state['key_points'])} key points identified"
            )
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            state["errors"].append(f"Content analysis error: {str(e)}")
            # Fallback
            state["key_points"] = [state["blog_title"]]
            state["content_themes"] = ["general"]
            state["call_to_actions"] = ["Learn more"]
            state["visual_suggestions"] = []
        
        return state
    
    async def platform_adaptation_node(self, state: SocialMediaState) -> SocialMediaState:
        """Adapt content for each target platform."""
        try:
            state["current_phase"] = SocialMediaPhase.PLATFORM_ADAPTATION
            
            state["platform_posts"] = {}
            state["post_variations"] = {}
            state["thread_versions"] = {}
            
            # Process platforms in parallel for efficiency
            tasks = []
            for platform in state["target_platforms"]:
                task = self._adapt_content_for_platform(
                    platform,
                    state["content"],
                    state["blog_title"],
                    state["key_points"],
                    state["call_to_actions"],
                    state["brand_voice"],
                    state["target_audience"]
                )
                tasks.append((platform, task))
            
            # Execute adaptations
            for platform, task in tasks:
                try:
                    if asyncio.iscoroutine(task):
                        adaptation = await task
                    else:
                        adaptation = task
                    
                    state["platform_posts"][platform] = adaptation["main_post"]
                    state["post_variations"][platform] = adaptation["variations"]
                    
                    # Generate thread versions for supported platforms
                    if self.platforms[platform].thread_support:
                        state["thread_versions"][platform] = adaptation.get("thread", [])
                        
                except Exception as e:
                    self.logger.warning(f"Adaptation failed for {platform}: {str(e)}")
                    state["warnings"].append(f"Failed to adapt content for {platform}")
            
            self.logger.info(f"Platform adaptation completed for {len(state['platform_posts'])} platforms")
            
        except Exception as e:
            self.logger.error(f"Platform adaptation failed: {str(e)}")
            state["errors"].append(f"Platform adaptation error: {str(e)}")
            # Fallback
            state["platform_posts"] = {}
            state["post_variations"] = {}
            state["thread_versions"] = {}
        
        return state
    
    def hashtag_research_node(self, state: SocialMediaState) -> SocialMediaState:
        """Research and optimize hashtags for each platform."""
        try:
            state["current_phase"] = SocialMediaPhase.HASHTAG_RESEARCH
            
            state["platform_hashtags"] = {}
            state["trending_hashtags"] = {}
            
            # Generate hashtags for each platform
            for platform in state["target_platforms"]:
                platform_config = self.platforms[platform]
                
                # Generate relevant hashtags
                hashtags = self._generate_hashtags(
                    state["content_themes"],
                    state["key_points"],
                    platform,
                    platform_config.hashtag_limit
                )
                
                state["platform_hashtags"][platform] = hashtags
                
                # Mock trending hashtags (in production, would use API)
                state["trending_hashtags"][platform] = self._get_trending_hashtags(platform)
            
            # Generate branded hashtags
            state["branded_hashtags"] = self._generate_branded_hashtags(
                state["blog_title"],
                state["content_themes"]
            )
            
            self.logger.info("Hashtag research completed for all platforms")
            
        except Exception as e:
            self.logger.error(f"Hashtag research failed: {str(e)}")
            state["errors"].append(f"Hashtag research error: {str(e)}")
            # Fallback
            state["platform_hashtags"] = {}
            state["trending_hashtags"] = {}
            state["branded_hashtags"] = []
        
        return state
    
    def engagement_optimization_node(self, state: SocialMediaState) -> SocialMediaState:
        """Optimize content for engagement."""
        try:
            state["current_phase"] = SocialMediaPhase.ENGAGEMENT_OPTIMIZATION
            
            state["engagement_strategies"] = {}
            state["audience_targeting"] = {}
            
            # Generate engagement strategies for each platform
            for platform in state["target_platforms"]:
                strategies = self._generate_engagement_strategies(
                    platform,
                    state["target_audience"],
                    state["content_goals"],
                    state["brand_voice"]
                )
                state["engagement_strategies"][platform] = strategies
                
                # Audience targeting recommendations
                targeting = self._generate_audience_targeting(
                    platform,
                    state["target_audience"],
                    state["content_themes"]
                )
                state["audience_targeting"][platform] = targeting
            
            self.logger.info("Engagement optimization completed")
            
        except Exception as e:
            self.logger.error(f"Engagement optimization failed: {str(e)}")
            state["errors"].append(f"Engagement optimization error: {str(e)}")
            # Fallback
            state["engagement_strategies"] = {}
            state["audience_targeting"] = {}
        
        return state
    
    def scheduling_optimization_node(self, state: SocialMediaState) -> SocialMediaState:
        """Optimize posting schedule and campaign strategy."""
        try:
            state["current_phase"] = SocialMediaPhase.SCHEDULING_OPTIMIZATION
            
            # Generate optimal posting times
            state["optimal_posting_times"] = {}
            for platform in state["target_platforms"]:
                optimal_time = self._get_optimal_posting_time(
                    platform,
                    state["target_audience"]
                )
                state["optimal_posting_times"][platform] = optimal_time
            
            # Create posting schedule
            state["posting_schedule"] = self._create_posting_schedule(
                state["target_platforms"],
                state["platform_posts"],
                state["post_variations"]
            )
            
            # Campaign strategy
            state["campaign_strategy"] = self._create_campaign_strategy(
                state["target_platforms"],
                state["content_goals"],
                state["engagement_strategies"]
            )
            
            # Cross-promotion opportunities
            state["cross_promotion"] = self._identify_cross_promotion(
                state["target_platforms"],
                state["platform_posts"]
            )
            
            self.logger.info("Scheduling optimization completed")
            
        except Exception as e:
            self.logger.error(f"Scheduling optimization failed: {str(e)}")
            state["errors"].append(f"Scheduling optimization error: {str(e)}")
            # Fallback
            state["optimal_posting_times"] = {}
            state["posting_schedule"] = {}
            state["campaign_strategy"] = {}
            state["cross_promotion"] = {}
        
        return state
    
    def final_review_node(self, state: SocialMediaState) -> SocialMediaState:
        """Final review and scoring."""
        try:
            state["current_phase"] = SocialMediaPhase.FINAL_REVIEW
            
            # Calculate social media score
            state["social_media_score"] = self._calculate_social_media_score(
                state["platform_posts"],
                state["platform_hashtags"],
                state["engagement_strategies"]
            )
            
            # Performance predictions
            state["performance_predictions"] = self._predict_performance(
                state["platform_posts"],
                state["engagement_strategies"],
                state["target_audience"]
            )
            
            # Generate optimization recommendations
            state["optimization_recommendations"] = self._generate_optimization_recommendations(
                state["platform_posts"],
                state["engagement_strategies"],
                state["social_media_score"]
            )
            
            # Determine adaptation quality
            state["adaptation_quality"] = self._determine_adaptation_quality(
                state["social_media_score"],
                len(state["platform_posts"])
            )
            
            self.logger.info(f"Social media adaptation completed - Score: {state['social_media_score']:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"Final review failed: {str(e)}")
            state["errors"].append(f"Final review error: {str(e)}")
            # Fallback
            state["social_media_score"] = 50.0
            state["performance_predictions"] = {}
            state["optimization_recommendations"] = []
            state["adaptation_quality"] = "fair"
        
        return state
    
    # Helper methods for social media optimization
    def _extract_key_points(self, content: str, title: str) -> List[str]:
        """Extract key points from content."""
        # Look for bullet points or numbered lists
        bullet_points = re.findall(r'^[-*•]\s+(.+)$', content, re.MULTILINE)
        numbered_points = re.findall(r'^\d+\.\s+(.+)$', content, re.MULTILINE)
        
        key_points = bullet_points + numbered_points
        
        # If no structured points, extract from headings
        if not key_points:
            headings = re.findall(r'^#{2,6}\s+(.+)$', content, re.MULTILINE)
            key_points = headings[:5]  # Limit to top 5
        
        # Fallback to sentences
        if not key_points:
            sentences = [s.strip() for s in content.split('.') if 20 < len(s.strip()) < 100]
            key_points = sentences[:3]
        
        # Always include the title as a key point
        if title not in key_points:
            key_points = [title] + key_points
        
        return key_points[:8]  # Limit to 8 key points
    
    def _identify_themes(self, content: str, title: str) -> List[str]:
        """Identify main themes in the content."""
        text = (title + " " + content).lower()
        
        # Common business/tech themes
        theme_keywords = {
            "ai": ["ai", "artificial intelligence", "machine learning", "automation"],
            "business": ["business", "strategy", "growth", "revenue", "profit"],
            "technology": ["technology", "tech", "digital", "software", "innovation"],
            "finance": ["finance", "financial", "money", "investment", "banking"],
            "marketing": ["marketing", "advertising", "branding", "promotion"],
            "productivity": ["productivity", "efficiency", "workflow", "optimization"],
            "leadership": ["leadership", "management", "team", "culture"],
            "data": ["data", "analytics", "insights", "metrics", "reporting"]
        }
        
        themes = []
        for theme, keywords in theme_keywords.items():
            if any(keyword in text for keyword in keywords):
                themes.append(theme)
        
        return themes or ["business"]
    
    def _extract_call_to_actions(self, content: str) -> List[str]:
        """Extract potential call-to-action phrases."""
        cta_patterns = [
            r'(learn more about [^.]+)',
            r'(get started with [^.]+)',
            r'(try [^.]+)',
            r'(discover [^.]+)',
            r'(explore [^.]+)',
            r'(find out [^.]+)'
        ]
        
        ctas = []
        for pattern in cta_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            ctas.extend(matches)
        
        # Add generic CTAs
        generic_ctas = [
            "Learn more",
            "Get started today",
            "Discover more",
            "Read the full article",
            "Share your thoughts",
            "What do you think?"
        ]
        
        return (ctas + generic_ctas)[:10]
    
    def _generate_visual_suggestions(self, content: str, title: str, themes: List[str]) -> List[str]:
        """Generate visual content suggestions."""
        suggestions = []
        
        # Theme-based visuals
        theme_visuals = {
            "ai": ["AI brain diagram", "Automation workflow", "Tech infographic"],
            "business": ["Growth chart", "Business meeting", "Strategy diagram"],
            "technology": ["Code screenshot", "Tech stack diagram", "Device mockup"],
            "finance": ["Financial charts", "Money graphics", "Investment diagram"],
            "marketing": ["Marketing funnel", "Social media graphics", "Brand elements"],
            "data": ["Data visualization", "Analytics dashboard", "Chart graphics"]
        }
        
        for theme in themes:
            if theme in theme_visuals:
                suggestions.extend(theme_visuals[theme])
        
        # Content-specific suggestions
        if "step" in content.lower() or "how to" in title.lower():
            suggestions.append("Step-by-step infographic")
        
        if "comparison" in content.lower() or "vs" in content.lower():
            suggestions.append("Comparison table")
        
        if "statistics" in content.lower() or "%" in content:
            suggestions.append("Statistical infographic")
        
        return suggestions[:8]
    
    async def _adapt_content_for_platform(
        self,
        platform: str,
        content: str,
        title: str,
        key_points: List[str],
        ctas: List[str],
        brand_voice: str,
        audience: str
    ) -> Dict[str, Any]:
        """Adapt content for a specific platform."""
        config = self.platforms[platform]
        
        if self.llm:
            return await self._ai_adapt_content(
                platform, config, content, title, key_points, ctas, brand_voice, audience
            )
        else:
            return self._manual_adapt_content(
                platform, config, content, title, key_points, ctas
            )
    
    async def _ai_adapt_content(
        self,
        platform: str,
        config: PlatformConfig,
        content: str,
        title: str,
        key_points: List[str],
        ctas: List[str],
        brand_voice: str,
        audience: str
    ) -> Dict[str, Any]:
        """Use AI to adapt content for platform."""
        try:
            prompt = f"""Adapt this content for {platform}:

Title: {title}
Content: {content[:2000]}
Key Points: {', '.join(key_points[:5])}
Target Audience: {audience}
Brand Voice: {brand_voice}

Platform Requirements:
- Max length: {config.max_length} characters
- Optimal length: {config.optimal_length} characters
- Tone: {config.tone}
- Content type: {config.content_type}
- Needs CTA: {config.call_to_action}

Create:
1. Main post (within optimal length)
2. 2 variations (different angles)
{"3. Thread version (3-5 posts)" if config.thread_support else ""}

Format as JSON with keys: main_post, variations, {"thread" if config.thread_support else ""}"""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_ai_response(response.content, config)
                
        except Exception as e:
            self.logger.warning(f"AI adaptation failed for {platform}: {str(e)}")
            return self._manual_adapt_content(platform, config, content, title, key_points, ctas)
    
    def _manual_adapt_content(
        self,
        platform: str,
        config: PlatformConfig,
        content: str,
        title: str,
        key_points: List[str],
        ctas: List[str]
    ) -> Dict[str, Any]:
        """Manual content adaptation fallback."""
        # Create base post from title and first key points
        base_text = f"{title}\n\n"
        
        # Add key points within length limit
        current_length = len(base_text)
        for point in key_points[:3]:
            point_text = f"• {point}\n"
            if current_length + len(point_text) < config.optimal_length - 100:
                base_text += point_text
                current_length += len(point_text)
        
        # Add CTA if required
        if config.call_to_action and ctas:
            cta_text = f"\n{ctas[0]}"
            if current_length + len(cta_text) < config.optimal_length:
                base_text += cta_text
        
        # Truncate if necessary
        if len(base_text) > config.max_length:
            base_text = base_text[:config.max_length - 3] + "..."
        
        main_post = base_text.strip()
        
        # Create variations
        variations = [
            self._create_variation(main_post, "question"),
            self._create_variation(main_post, "statistic")
        ]
        
        result = {
            "main_post": main_post,
            "variations": variations
        }
        
        # Add thread version if supported
        if config.thread_support:
            result["thread"] = self._create_thread_version(title, key_points, config)
        
        return result
    
    def _parse_ai_response(self, response_text: str, config: PlatformConfig) -> Dict[str, Any]:
        """Parse AI response when JSON parsing fails."""
        lines = response_text.strip().split('\n')
        
        result = {
            "main_post": "Content adaptation failed",
            "variations": ["Variation 1", "Variation 2"]
        }
        
        # Simple parsing attempt
        current_section = None
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "main_post" in line.lower() or "main post" in line.lower():
                if content_lines and current_section:
                    result[current_section] = '\n'.join(content_lines)
                current_section = "main_post"
                content_lines = []
            elif "variation" in line.lower():
                if content_lines and current_section:
                    if current_section == "variations":
                        result["variations"].append('\n'.join(content_lines))
                    else:
                        result[current_section] = '\n'.join(content_lines)
                current_section = "variations"
                content_lines = []
            elif "thread" in line.lower() and config.thread_support:
                if content_lines and current_section:
                    result[current_section] = '\n'.join(content_lines)
                current_section = "thread"
                content_lines = []
            else:
                content_lines.append(line)
        
        # Handle last section
        if content_lines and current_section:
            if current_section == "variations" and isinstance(result["variations"], list):
                result["variations"].append('\n'.join(content_lines))
            else:
                result[current_section] = '\n'.join(content_lines)
        
        return result
    
    def _create_variation(self, base_post: str, variation_type: str) -> str:
        """Create a variation of the base post."""
        if variation_type == "question":
            return f"What do you think about this?\n\n{base_post}"
        elif variation_type == "statistic":
            return f"Here's an interesting insight:\n\n{base_post}"
        else:
            return base_post
    
    def _create_thread_version(self, title: str, key_points: List[str], config: PlatformConfig) -> List[str]:
        """Create a thread version of the content."""
        thread = []
        
        # First post - introduction
        thread.append(f"{title} 🧵\n\nHere's what you need to know:")
        
        # Subsequent posts - key points
        for i, point in enumerate(key_points[:4], 2):
            post = f"{i}/{min(len(key_points) + 2, 6)} {point}"
            if len(post) > config.max_length:
                post = post[:config.max_length - 3] + "..."
            thread.append(post)
        
        # Final post - conclusion
        thread.append(f"{len(thread) + 1}/{len(thread) + 1} What are your thoughts? Share below! 👇")
        
        return thread
    
    def _generate_hashtags(self, themes: List[str], key_points: List[str], platform: str, limit: int) -> List[str]:
        """Generate relevant hashtags for the platform."""
        hashtags = []
        
        # Theme-based hashtags
        theme_hashtags = {
            "ai": ["#AI", "#ArtificialIntelligence", "#MachineLearning", "#Automation", "#Tech"],
            "business": ["#Business", "#Strategy", "#Growth", "#Success", "#Entrepreneur"],
            "technology": ["#Technology", "#Tech", "#Digital", "#Innovation", "#Software"],
            "finance": ["#Finance", "#Financial", "#Investment", "#Money", "#Banking"],
            "marketing": ["#Marketing", "#DigitalMarketing", "#Branding", "#Growth", "#Strategy"],
            "productivity": ["#Productivity", "#Efficiency", "#Workflow", "#Tips", "#Success"],
            "leadership": ["#Leadership", "#Management", "#Team", "#Culture", "#Success"],
            "data": ["#Data", "#Analytics", "#Insights", "#DataScience", "#BigData"]
        }
        
        # Add theme-specific hashtags
        for theme in themes:
            if theme in theme_hashtags:
                hashtags.extend(theme_hashtags[theme][:3])
        
        # Platform-specific hashtags
        platform_hashtags = {
            "linkedin": ["#LinkedIn", "#Professional", "#Career", "#Business"],
            "twitter": ["#Twitter", "#Thread", "#TechTalk"],
            "instagram": ["#Insta", "#Visual", "#Story"],
            "facebook": ["#Facebook", "#Community", "#Discussion"],
            "tiktok": ["#TikTok", "#Viral", "#Trend"]
        }
        
        if platform in platform_hashtags:
            hashtags.extend(platform_hashtags[platform][:2])
        
        # Remove duplicates and limit
        unique_hashtags = list(dict.fromkeys(hashtags))
        return unique_hashtags[:limit]
    
    def _get_trending_hashtags(self, platform: str) -> List[str]:
        """Get trending hashtags for platform (mock data)."""
        trending = {
            "linkedin": ["#MondayMotivation", "#ThoughtLeadership", "#Innovation"],
            "twitter": ["#Trending", "#TechNews", "#StartupLife"],
            "instagram": ["#InstaGood", "#PhotoOfTheDay", "#Inspiration"],
            "facebook": ["#Community", "#Viral", "#Trending"],
            "tiktok": ["#ForYou", "#Trending", "#Viral"]
        }
        return trending.get(platform, [])
    
    def _generate_branded_hashtags(self, title: str, themes: List[str]) -> List[str]:
        """Generate branded hashtags."""
        branded = ["#CrediLinq", "#FinancialTech"]
        
        # Add content-specific branded tags
        if "ai" in themes:
            branded.append("#CrediLinqAI")
        if "business" in themes:
            branded.append("#CrediLinqBusiness")
        
        return branded
    
    def _generate_engagement_strategies(self, platform: str, audience: str, goals: List[str], brand_voice: str) -> List[str]:
        """Generate engagement strategies for platform."""
        strategies = []
        
        # Platform-specific strategies
        platform_strategies = {
            "linkedin": [
                "Ask thought-provoking questions",
                "Share professional insights",
                "Tag relevant industry leaders",
                "Use LinkedIn native video",
                "Create carousel posts"
            ],
            "twitter": [
                "Create engaging threads",
                "Use relevant GIFs",
                "Retweet with comments",
                "Host Twitter Spaces",
                "Use polls for engagement"
            ],
            "facebook": [
                "Create shareable content",
                "Use Facebook Live",
                "Create events",
                "Share behind-the-scenes",
                "Use Facebook Stories"
            ],
            "instagram": [
                "Use Instagram Stories",
                "Create Reels",
                "Use IGTV for longer content",
                "Partner with influencers",
                "Create user-generated content"
            ],
            "tiktok": [
                "Create trending challenges",
                "Use popular sounds",
                "Create educational content",
                "Collaborate with creators",
                "Use trending hashtags"
            ]
        }
        
        strategies.extend(platform_strategies.get(platform, []))
        
        # Goal-specific strategies
        if "awareness" in goals:
            strategies.append("Share valuable insights regularly")
        if "engagement" in goals:
            strategies.append("Ask questions and respond to comments")
        if "leads" in goals:
            strategies.append("Include clear call-to-actions")
        
        return strategies[:5]
    
    def _generate_audience_targeting(self, platform: str, audience: str, themes: List[str]) -> str:
        """Generate audience targeting recommendations."""
        targeting_base = f"Target {audience} interested in {', '.join(themes)}"
        
        platform_additions = {
            "linkedin": "Focus on job titles and company sizes",
            "twitter": "Use interest-based targeting",
            "facebook": "Leverage detailed demographics",
            "instagram": "Use visual interest targeting",
            "tiktok": "Target by trending content engagement"
        }
        
        addition = platform_additions.get(platform, "Use platform-specific targeting")
        return f"{targeting_base}. {addition}."
    
    def _get_optimal_posting_time(self, platform: str, audience: str) -> str:
        """Get optimal posting time for platform and audience."""
        # Mock optimal times (would be data-driven in production)
        times = {
            "linkedin": "Tuesday-Thursday, 8-10 AM",
            "twitter": "Monday-Friday, 12-3 PM",
            "facebook": "Tuesday-Thursday, 1-3 PM",
            "instagram": "Monday, Wednesday, Friday, 11 AM-1 PM",
            "tiktok": "Tuesday-Thursday, 6-10 AM and 7-9 PM"
        }
        
        return times.get(platform, "Weekdays, 9 AM - 5 PM")
    
    def _create_posting_schedule(self, platforms: List[str], posts: Dict[str, Dict[str, Any]], variations: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create a posting schedule across platforms."""
        schedule = {}
        
        for platform in platforms:
            if platform in posts:
                platform_schedule = [
                    f"Day 1: {posts[platform].get('text', 'Main post')}",
                    f"Day 3: Follow-up engagement post",
                    f"Day 7: Recap and insights"
                ]
                
                if platform in variations and variations[platform]:
                    platform_schedule.insert(1, f"Day 2: {variations[platform][0][:50]}...")
                
                schedule[platform] = platform_schedule
        
        return schedule
    
    def _create_campaign_strategy(self, platforms: List[str], goals: List[str], engagement_strategies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Create overall campaign strategy."""
        return {
            "duration": "2 weeks",
            "primary_platforms": platforms[:2],  # Focus on top 2
            "secondary_platforms": platforms[2:],
            "goals": goals,
            "success_metrics": [
                "Engagement rate > 3%",
                "Click-through rate > 1%",
                "Follower growth > 2%"
            ],
            "content_themes": ["educational", "engaging", "actionable"],
            "posting_frequency": "3-4 times per week"
        }
    
    def _identify_cross_promotion(self, platforms: List[str], posts: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify cross-promotion opportunities."""
        cross_promo = {}
        
        for platform in platforms:
            opportunities = []
            
            # Platform-specific cross-promotion
            if platform == "instagram" and "linkedin" in platforms:
                opportunities.append("Share Instagram Stories on LinkedIn")
            
            if platform == "twitter" and "linkedin" in platforms:
                opportunities.append("Create Twitter threads from LinkedIn posts")
            
            if platform == "facebook" and "instagram" in platforms:
                opportunities.append("Cross-post visual content")
            
            if not opportunities:
                opportunities.append("Repurpose content with platform-specific adaptations")
            
            cross_promo[platform] = opportunities
        
        return cross_promo
    
    def _calculate_social_media_score(self, posts: Dict[str, Dict[str, Any]], hashtags: Dict[str, List[str]], strategies: Dict[str, List[str]]) -> float:
        """Calculate overall social media adaptation score."""
        score = 0.0
        max_score = 100.0
        
        # Platform coverage (30 points)
        num_platforms = len(posts)
        if num_platforms >= 4:
            score += 30
        elif num_platforms >= 3:
            score += 25
        elif num_platforms >= 2:
            score += 20
        elif num_platforms >= 1:
            score += 15
        
        # Content quality (40 points)
        avg_content_score = 0
        for platform, post in posts.items():
            content_text = post.get("text", "")
            if content_text:
                # Length optimization
                config = self.platforms[platform]
                length_score = 10 if len(content_text) <= config.optimal_length else 5
                
                # CTA presence
                cta_score = 10 if any(cta in content_text.lower() for cta in ["learn", "discover", "try", "get"]) else 5
                
                # Hashtag usage
                platform_hashtags = hashtags.get(platform, [])
                hashtag_score = 10 if len(platform_hashtags) > 0 else 0
                
                platform_score = (length_score + cta_score + hashtag_score) / 30 * 40
                avg_content_score += platform_score
        
        if posts:
            score += avg_content_score / len(posts)
        
        # Strategy completeness (30 points)
        strategy_score = 0
        for platform in posts.keys():
            if platform in strategies and len(strategies[platform]) >= 3:
                strategy_score += 30 / len(posts)
            elif platform in strategies:
                strategy_score += 20 / len(posts)
        
        score += min(strategy_score, 30)
        
        return min(score, max_score)
    
    def _predict_performance(self, posts: Dict[str, Dict[str, Any]], strategies: Dict[str, List[str]], audience: str) -> Dict[str, Dict[str, float]]:
        """Predict performance metrics for each platform."""
        predictions = {}
        
        for platform in posts.keys():
            # Mock predictions based on platform characteristics
            platform_multipliers = {
                "linkedin": {"engagement": 1.2, "reach": 1.0, "clicks": 1.3},
                "twitter": {"engagement": 1.5, "reach": 1.4, "clicks": 1.1},
                "facebook": {"engagement": 1.0, "reach": 1.2, "clicks": 1.0},
                "instagram": {"engagement": 1.8, "reach": 1.1, "clicks": 0.8},
                "tiktok": {"engagement": 2.0, "reach": 1.6, "clicks": 0.6}
            }
            
            multipliers = platform_multipliers.get(platform, {"engagement": 1.0, "reach": 1.0, "clicks": 1.0})
            
            predictions[platform] = {
                "estimated_engagement_rate": 3.5 * multipliers["engagement"],
                "estimated_reach": 1000 * multipliers["reach"],
                "estimated_click_rate": 1.2 * multipliers["clicks"],
                "confidence": 0.75
            }
        
        return predictions
    
    def _generate_optimization_recommendations(self, posts: Dict[str, Dict[str, Any]], strategies: Dict[str, List[str]], score: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if score < 60:
            recommendations.append("Increase content length for better engagement")
            recommendations.append("Add more relevant hashtags")
            recommendations.append("Include stronger call-to-actions")
        
        if score < 80:
            recommendations.append("Create platform-specific variations")
            recommendations.append("Add visual content suggestions")
            recommendations.append("Optimize posting times")
        
        # Platform-specific recommendations
        for platform in posts.keys():
            config = self.platforms[platform]
            post_content = posts[platform].get("text", "")
            
            if len(post_content) > config.optimal_length * 1.2:
                recommendations.append(f"Shorten {platform} post for better readability")
            
            if config.visual_priority and platform in posts:
                recommendations.append(f"Add visual elements for {platform}")
        
        return recommendations[:8]
    
    def _determine_adaptation_quality(self, score: float, num_platforms: int) -> str:
        """Determine overall adaptation quality."""
        if score >= 85 and num_platforms >= 3:
            return "excellent"
        elif score >= 70 and num_platforms >= 2:
            return "good"
        elif score >= 55:
            return "fair"
        else:
            return "needs_improvement"
    
    async def execute_workflow(
        self,
        initial_state: Dict[str, Any],
        context: Optional[LangGraphExecutionContext] = None
    ) -> WorkflowState:
        """Execute the social media workflow."""
        try:
            # Convert input to SocialMediaState
            social_state = SocialMediaState(
                content=initial_state["content"],
                blog_title=initial_state["blog_title"],
                outline=initial_state.get("outline", []),
                target_platforms=initial_state.get("target_platforms", ["linkedin", "twitter", "facebook"]),
                target_audience=initial_state.get("target_audience", "professionals"),
                brand_voice=initial_state.get("brand_voice", "professional"),
                content_goals=initial_state.get("content_goals", ["awareness", "engagement"]),
                key_points=[],
                content_themes=[],
                call_to_actions=[],
                visual_suggestions=[],
                platform_posts={},
                post_variations={},
                thread_versions={},
                platform_hashtags={},
                trending_hashtags={},
                branded_hashtags=[],
                engagement_strategies={},
                optimal_posting_times={},
                audience_targeting={},
                posting_schedule={},
                campaign_strategy={},
                cross_promotion={},
                social_media_score=0.0,
                performance_predictions={},
                optimization_recommendations=[],
                current_phase=SocialMediaPhase.INITIALIZATION,
                adaptation_quality="fair",
                errors=[],
                warnings=[]
            )
            
            # Execute the graph
            config = {"configurable": {"thread_id": context.session_id if context else "default"}}
            final_state = await self.graph.ainvoke(social_state, config)
            
            # Convert to WorkflowState
            return WorkflowState(
                status=WorkflowStatus.COMPLETED if final_state["platform_posts"] else WorkflowStatus.FAILED,
                phase=final_state["current_phase"],
                data={
                    "platform_posts": final_state["platform_posts"],
                    "post_variations": final_state["post_variations"],
                    "platform_hashtags": final_state["platform_hashtags"],
                    "engagement_strategies": final_state["engagement_strategies"],
                    "posting_schedule": final_state["posting_schedule"],
                    "campaign_strategy": final_state["campaign_strategy"],
                    "performance_predictions": final_state["performance_predictions"],
                    "social_media_score": final_state["social_media_score"],
                    "optimization_recommendations": final_state["optimization_recommendations"],
                    "visual_suggestions": final_state["visual_suggestions"]
                },
                errors=final_state.get("errors", []),
                metadata={
                    "warnings": final_state.get("warnings", []),
                    "adaptation_quality": final_state.get("adaptation_quality", ""),
                    "platforms_adapted": len(final_state.get("platform_posts", {}))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return WorkflowState(
                status=WorkflowStatus.FAILED,
                phase=SocialMediaPhase.INITIALIZATION,
                data={},
                errors=[str(e)],
                metadata={"error_type": type(e).__name__}
            )


# Adapter for backward compatibility
class SocialMediaAgentLangGraph:
    """Adapter to make LangGraph workflow compatible with existing SocialMediaAgent interface."""
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        """Initialize the adapter."""
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.SOCIAL_MEDIA,
                name="SocialMediaAgentLangGraph",
                description="LangGraph-powered social media optimizer with comprehensive platform adaptation",
                capabilities=[
                    "multi_platform_adaptation",
                    "hashtag_research",
                    "engagement_optimization",
                    "posting_schedule",
                    "performance_prediction",
                    "cross_promotion"
                ],
                version="3.0.0"
            )
        
        self.metadata = metadata
        self.workflow = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the workflow."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            llm = create_llm(
                model="gemini-1.5-flash",
                temperature=0.7,
                api_key=settings.primary_api_key
            )
            
            self.workflow = SocialMediaAgentWorkflow(llm=llm)
            
        except Exception as e:
            # Fallback without LLM
            self.workflow = SocialMediaAgentWorkflow(llm=None)
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Any] = None
    ) -> AgentResult:
        """Execute the social media workflow."""
        try:
            # Execute workflow
            result = await self.workflow.execute_workflow(
                input_data,
                LangGraphExecutionContext(
                    session_id=context.session_id if context else "default",
                    user_id=context.user_id if context else None
                )
            )
            
            # Convert to AgentResult
            return AgentResult(
                success=result.status == WorkflowStatus.COMPLETED,
                data=result.data,
                metadata={
                    "agent_type": "social_media_langgraph",
                    "workflow_status": result.status,
                    "final_phase": result.phase,
                    **result.metadata
                },
                error_message="; ".join(result.errors) if result.errors else None
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="SOCIAL_MEDIA_WORKFLOW_FAILED"
            )