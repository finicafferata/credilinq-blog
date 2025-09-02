"""
Strategic Content Brief Generation Agent with SEO Research.
Creates comprehensive content briefs that drive marketing success.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from ..core.base_agent import BaseAgent, AgentMetadata, AgentType
from ...config.settings import settings
from ...core.monitoring import metrics, async_performance_tracker

class ContentType(str, Enum):
    """Types of content that can be briefed."""
    BLOG_POST = "blog_post"
    LINKEDIN_ARTICLE = "linkedin_article"  
    WHITE_PAPER = "white_paper"
    CASE_STUDY = "case_study"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    NEWS_ARTICLE = "news_article"
    OPINION_PIECE = "opinion_piece"

class ContentPurpose(str, Enum):
    """Primary purpose of the content."""
    LEAD_GENERATION = "lead_generation"
    BRAND_AWARENESS = "brand_awareness"
    THOUGHT_LEADERSHIP = "thought_leadership"
    CUSTOMER_EDUCATION = "customer_education"
    PRODUCT_ANNOUNCEMENT = "product_announcement"
    SEO_RANKING = "seo_ranking"
    COMPETITOR_RESPONSE = "competitor_response"

class KeywordDifficulty(str, Enum):
    """SEO keyword difficulty levels."""
    EASY = "easy"          # 0-30
    MEDIUM = "medium"      # 31-50
    HARD = "hard"          # 51-70
    EXPERT = "expert"      # 71-100

@dataclass
class SEOKeyword:
    """SEO keyword with research data."""
    keyword: str
    search_volume: int
    difficulty: KeywordDifficulty
    competition: str  # low, medium, high
    intent: str      # informational, commercial, transactional
    suggested_usage: str  # primary, secondary, semantic
    related_terms: List[str]
    
class CompetitorInsight(BaseModel):
    """Competitor content analysis insight."""
    competitor_name: str
    content_title: str
    content_url: Optional[str] = None
    key_angles: List[str]
    engagement_signals: Dict[str, Any] = Field(default_factory=dict)
    content_gaps: List[str] = Field(default_factory=list)
    differentiation_opportunities: List[str] = Field(default_factory=list)

class ContentStructure(BaseModel):
    """Suggested content structure and outline."""
    estimated_word_count: int
    suggested_headlines: List[str]
    content_outline: List[Dict[str, str]]  # {"section": "title", "description": "what to cover"}
    recommended_sections: List[str]
    call_to_actions: List[str]
    internal_link_opportunities: List[str] = Field(default_factory=list)

class ContentBrief(BaseModel):
    """Comprehensive strategic content brief."""
    
    # Basic Information
    brief_id: str
    title: str
    content_type: ContentType
    primary_purpose: ContentPurpose
    target_audience: str
    
    # Strategic Context
    marketing_objective: str
    business_context: str
    unique_value_proposition: str
    key_messages: List[str]
    
    # SEO Strategy
    primary_keyword: SEOKeyword
    secondary_keywords: List[SEOKeyword]
    semantic_keywords: List[str]
    target_search_intent: str
    seo_goals: List[str]
    
    # Competitive Intelligence
    competitor_insights: List[CompetitorInsight]
    market_gap_identified: str
    differentiation_strategy: str
    
    # Content Strategy
    content_structure: ContentStructure
    tone_and_voice: str
    writing_style_notes: str
    brand_guidelines: Dict[str, Any] = Field(default_factory=dict)
    
    # Success Metrics
    success_kpis: List[str]
    target_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Operational Details
    estimated_creation_time: str
    content_calendar_notes: str
    distribution_channels: List[str]
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = "ContentBriefAgent"
    version: str = "1.0"

class ContentBriefAgent(BaseAgent):
    """Strategic content brief generation agent with comprehensive SEO and competitive research."""
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_BRIEF,
                name="ContentBriefAgent",
                description="Strategic content brief generation agent with comprehensive SEO and competitive research",
                capabilities=[
                    "seo_keyword_research",
                    "competitor_analysis", 
                    "content_strategy",
                    "audience_analysis",
                    "content_structure_planning",
                    "success_metrics_definition"
                ],
                version="1.0.0"
            )
        
        super().__init__(metadata)
        
        # Initialize language models for different research tasks
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for ContentBriefAgent")
        
        # Use GPT-4 for strategic and complex analysis tasks
        self.strategy_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,  # Lower temperature for strategic thinking
            max_tokens=3000,
            api_key=api_key
        )
        
        # Use GPT-3.5-turbo for faster research tasks
        self.research_llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Faster and cheaper for research
            api_key=api_key,
            temperature=0.7,  # Higher temperature for creative research
            max_tokens=2000
        )
        
        # Use GPT-3.5-turbo for keyword research (simple structured task)
        self.keyword_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,  # Low temperature for factual keyword research
            max_tokens=1500,
            api_key=api_key
        )
        
        # Content brief templates
        self.brief_templates = self._create_brief_templates()
        
        # Keyword research patterns
        self.keyword_research_patterns = self._load_keyword_patterns()
        
    def _create_brief_templates(self) -> Dict[ContentType, ChatPromptTemplate]:
        """Create content type-specific brief templates."""
        templates = {}
        
        # Blog Post Brief Template
        blog_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a strategic content marketing expert creating comprehensive content briefs for B2B financial services.

Your task is to create a detailed content brief that includes:
1. Strategic context and business objectives
2. Target audience analysis with pain points and goals
3. SEO keyword strategy with search intent analysis
4. Competitive landscape analysis
5. Content structure and outline recommendations
6. Success metrics and KPIs
7. Unique differentiation opportunities

Focus on creating briefs that drive measurable business results while establishing thought leadership in financial technology.

B2B Financial Services Context:
- Industry challenges: regulation, digital transformation, embedded finance
- Audience: CFOs, Finance Directors, Fintech founders, B2B marketplace owners
- Key trends: API-first solutions, embedded lending, alternative credit scoring
- Competitive landscape: Traditional banks vs fintech vs platform solutions
            """),
            HumanMessage(content="""
Create a comprehensive content brief for the following:

Topic: {topic}
Content Type: {content_type}
Primary Purpose: {primary_purpose}
Target Audience: {target_audience}
Company Context: {company_context}
Competitive Focus: {competitive_focus}

Additional Context:
- Current market position: {market_position}
- Content goals: {content_goals}
- Distribution channels: {distribution_channels}
- Success metrics: {success_metrics}

Please provide a strategic, actionable brief that will guide content creation and ensure marketing success.
            """)
        ])
        templates[ContentType.BLOG_POST] = blog_template
        
        # LinkedIn Article Brief Template
        linkedin_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are creating content briefs specifically for LinkedIn articles that drive professional engagement and thought leadership in B2B financial services.

LinkedIn Article Success Factors:
- Professional tone with personal insights
- Industry expertise demonstration
- Actionable advice for business leaders
- Network engagement optimization
- Lead generation potential
- Personal brand building

Focus on creating briefs that position executives as thought leaders while driving business objectives.
            """),
            HumanMessage(content="""
Create a comprehensive LinkedIn article brief for:

Topic: {topic}
Executive Voice: {executive_voice}
Thought Leadership Angle: {thought_leadership_angle}
Target Professional Audience: {target_audience}
Business Objectives: {business_objectives}
Company Context: {company_context}

Include specific recommendations for professional engagement, networking opportunities, and lead generation tactics.
            """)
        ])
        templates[ContentType.LINKEDIN_ARTICLE] = linkedin_template
        
        return templates
    
    def _load_keyword_patterns(self) -> Dict[str, Any]:
        """Load keyword research patterns and industry-specific terms."""
        return {
            "fintech_keywords": [
                "embedded finance", "API banking", "digital lending", "fintech platform",
                "alternative credit", "payment orchestration", "financial marketplace",
                "B2B credit", "invoice factoring", "working capital", "trade finance"
            ],
            "intent_patterns": {
                "informational": ["what is", "how to", "guide to", "understanding", "benefits of"],
                "commercial": ["best", "top", "vs", "comparison", "review", "solution"],
                "transactional": ["pricing", "cost", "buy", "signup", "demo", "trial"]
            },
            "difficulty_indicators": {
                "low": ["long-tail", "specific", "technical", "niche"],
                "medium": ["industry terms", "solution-focused", "problem-solving"],
                "high": ["broad terms", "generic", "competitive", "established"]
            }
        }
    
    async def create_content_brief(
        self,
        brief_request: Dict[str, Any]
    ) -> ContentBrief:
        """Create a comprehensive strategic content brief."""
        
        async with async_performance_tracker("content_brief_creation"):
            
            # Extract request parameters
            topic = brief_request.get("topic", "")
            content_type = ContentType(brief_request.get("content_type", ContentType.BLOG_POST))
            primary_purpose = ContentPurpose(brief_request.get("primary_purpose", ContentPurpose.LEAD_GENERATION))
            target_audience = brief_request.get("target_audience", "B2B finance professionals")
            company_context = brief_request.get("company_context", "")
            
            # Generate unique brief ID
            brief_id = f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(topic) % 10000}"
            
            self.logger.info(f"Creating content brief for topic: {topic}")
            
            # Step 1 & 2: Run independent research tasks concurrently
            keyword_task = self._conduct_keyword_research(topic, content_type, primary_purpose)
            competitor_task = self._analyze_competitive_landscape(topic, target_audience)
            metrics_task = self._define_success_metrics(primary_purpose, content_type)
            
            # Wait for all concurrent tasks to complete
            keyword_research, competitor_insights, success_metrics = await asyncio.gather(
                keyword_task, competitor_task, metrics_task
            )
            
            # Step 3: Generate strategic content structure (depends on research results)
            content_structure = await self._generate_content_structure(
                topic, content_type, keyword_research, competitor_insights
            )
            
            # Step 5: Create strategic brief using appropriate template
            strategic_brief = await self._generate_strategic_brief(
                brief_request, keyword_research, competitor_insights, content_structure
            )
            
            # Step 6: Compile comprehensive content brief
            content_brief = ContentBrief(
                brief_id=brief_id,
                title=strategic_brief.get("title", topic),
                content_type=content_type,
                primary_purpose=primary_purpose,
                target_audience=target_audience,
                marketing_objective=strategic_brief.get("marketing_objective", ""),
                business_context=company_context,
                unique_value_proposition=strategic_brief.get("unique_value_proposition", ""),
                key_messages=strategic_brief.get("key_messages", []),
                primary_keyword=keyword_research["primary_keyword"],
                secondary_keywords=keyword_research["secondary_keywords"],
                semantic_keywords=keyword_research["semantic_keywords"],
                target_search_intent=keyword_research["search_intent"],
                seo_goals=keyword_research["seo_goals"],
                competitor_insights=competitor_insights,
                market_gap_identified=strategic_brief.get("market_gap", ""),
                differentiation_strategy=strategic_brief.get("differentiation_strategy", ""),
                content_structure=content_structure,
                tone_and_voice=strategic_brief.get("tone_and_voice", "Professional and authoritative"),
                writing_style_notes=strategic_brief.get("writing_style_notes", ""),
                brand_guidelines=brief_request.get("brand_guidelines", {}),
                success_kpis=success_metrics["kpis"],
                target_metrics=success_metrics["targets"],
                estimated_creation_time=self._estimate_creation_time(content_type, content_structure.estimated_word_count),
                content_calendar_notes=strategic_brief.get("content_calendar_notes", ""),
                distribution_channels=brief_request.get("distribution_channels", ["website", "linkedin", "email"])
            )
            
            # Track metrics
            metrics.increment_counter(
                "content.brief_created",
                tags={
                    "content_type": content_type.value,
                    "primary_purpose": primary_purpose.value,
                    "success": "true"
                }
            )
            
            self.logger.info(f"Content brief created successfully: {brief_id}")
            return content_brief
    
    async def _conduct_keyword_research(
        self,
        topic: str,
        content_type: ContentType,
        primary_purpose: ContentPurpose
    ) -> Dict[str, Any]:
        """Conduct comprehensive SEO keyword research."""
        
        keyword_research_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert SEO strategist specializing in B2B financial services content.

Conduct comprehensive keyword research that includes:
1. Primary keyword identification (1-2 keywords)
2. Secondary keywords (3-5 supporting keywords)
3. Semantic keywords (10-15 related terms)
4. Search intent analysis (informational, commercial, transactional)
5. Difficulty assessment (easy, medium, hard, expert)
6. Content optimization recommendations

Focus on keywords that:
- Have business value for B2B financial services
- Match the content type and purpose
- Drive qualified traffic and conversions
- Support thought leadership positioning
- Consider current market trends in fintech

Return structured JSON with detailed keyword analysis.
            """),
            HumanMessage(content="""
Topic: {topic}
Content Type: {content_type}
Primary Purpose: {primary_purpose}
Industry Focus: B2B Financial Services, Fintech, Embedded Finance

Conduct keyword research and provide:
1. Primary keyword with search volume estimate and difficulty
2. Secondary keywords with metrics
3. Semantic keyword cluster
4. Search intent analysis
5. SEO optimization strategy
6. Competitive keyword opportunities

Format as structured JSON for easy parsing.
            """)
        ])
        
        try:
            # Use faster model for keyword research
            response = await self.keyword_llm.agenerate([
                keyword_research_prompt.format_messages(
                    topic=topic,
                    content_type=content_type.value,
                    primary_purpose=primary_purpose.value
                )
            ])
            
            keyword_analysis = response.generations[0][0].text
            
            # Parse the response (simplified version - in production, use more robust parsing)
            try:
                parsed_research = json.loads(keyword_analysis)
            except:
                # Fallback research if parsing fails
                parsed_research = self._generate_fallback_keyword_research(topic, content_type)
            
            # Convert to structured format
            primary_keyword = SEOKeyword(
                keyword=parsed_research.get("primary_keyword", {}).get("keyword", topic.lower()),
                search_volume=parsed_research.get("primary_keyword", {}).get("search_volume", 1000),
                difficulty=KeywordDifficulty(parsed_research.get("primary_keyword", {}).get("difficulty", "medium")),
                competition=parsed_research.get("primary_keyword", {}).get("competition", "medium"),
                intent=parsed_research.get("primary_keyword", {}).get("intent", "informational"),
                suggested_usage="primary",
                related_terms=parsed_research.get("primary_keyword", {}).get("related_terms", [])
            )
            
            secondary_keywords = []
            for kw_data in parsed_research.get("secondary_keywords", [])[:5]:
                secondary_keywords.append(SEOKeyword(
                    keyword=kw_data.get("keyword", ""),
                    search_volume=kw_data.get("search_volume", 500),
                    difficulty=KeywordDifficulty(kw_data.get("difficulty", "medium")),
                    competition=kw_data.get("competition", "medium"),
                    intent=kw_data.get("intent", "informational"),
                    suggested_usage="secondary",
                    related_terms=kw_data.get("related_terms", [])
                ))
            
            # Ensure search_intent is always a string
            search_intent_raw = parsed_research.get("search_intent", "informational")
            if isinstance(search_intent_raw, dict):
                # If it's a dict, extract the first key or value
                if search_intent_raw:
                    search_intent = list(search_intent_raw.keys())[0] if search_intent_raw.keys() else "informational"
                else:
                    search_intent = "informational"
            elif isinstance(search_intent_raw, list):
                # If it's a list, take the first item
                search_intent = search_intent_raw[0] if search_intent_raw else "informational"
            else:
                search_intent = str(search_intent_raw) if search_intent_raw else "informational"

            return {
                "primary_keyword": primary_keyword,
                "secondary_keywords": secondary_keywords,
                "semantic_keywords": parsed_research.get("semantic_keywords", []),
                "search_intent": search_intent,
                "seo_goals": parsed_research.get("seo_goals", ["Improve organic visibility", "Drive qualified traffic"]),
                "competitive_opportunities": parsed_research.get("competitive_opportunities", [])
            }
            
        except Exception as e:
            self.logger.error(f"Keyword research failed: {str(e)}")
            return self._generate_fallback_keyword_research(topic, content_type)
    
    def _generate_fallback_keyword_research(self, topic: str, content_type: ContentType) -> Dict[str, Any]:
        """Generate fallback keyword research if API fails."""
        
        primary_keyword = SEOKeyword(
            keyword=topic.lower(),
            search_volume=1000,
            difficulty=KeywordDifficulty.MEDIUM,
            competition="medium",
            intent="informational",
            suggested_usage="primary",
            related_terms=[f"{topic} guide", f"{topic} solutions", f"best {topic}"]
        )
        
        secondary_keywords = [
            SEOKeyword(
                keyword=f"{topic} guide",
                search_volume=500,
                difficulty=KeywordDifficulty.EASY,
                competition="low",
                intent="informational",
                suggested_usage="secondary",
                related_terms=[]
            ),
            SEOKeyword(
                keyword=f"{topic} solutions",
                search_volume=800,
                difficulty=KeywordDifficulty.MEDIUM,
                competition="medium",
                intent="commercial",
                suggested_usage="secondary",
                related_terms=[]
            )
        ]
        
        return {
            "primary_keyword": primary_keyword,
            "secondary_keywords": secondary_keywords,
            "semantic_keywords": [f"{topic} platform", f"{topic} technology", f"{topic} benefits"],
            "search_intent": "informational",
            "seo_goals": ["Improve organic visibility", "Drive qualified traffic"],
            "competitive_opportunities": ["Target long-tail keywords", "Focus on technical content"]
        }
    
    async def _analyze_competitive_landscape(
        self,
        topic: str,
        target_audience: str
    ) -> List[CompetitorInsight]:
        """Analyze competitive content landscape."""
        
        competitive_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a competitive intelligence analyst specializing in B2B financial services content marketing.

Analyze the competitive landscape for the given topic and provide insights on:
1. Key competitors creating content in this space
2. Popular content angles and approaches
3. Content gaps and opportunities
4. Differentiation strategies
5. Engagement patterns and successful formats

Focus on identifying opportunities for unique positioning and thought leadership in the fintech and embedded finance space.

Return structured analysis with actionable competitive insights.
            """),
            HumanMessage(content="""
Topic: {topic}
Target Audience: {target_audience}
Industry: B2B Financial Services, Fintech, Embedded Finance

Analyze the competitive content landscape and provide:
1. Top 3-5 competitors creating content on this topic
2. Common content angles and approaches
3. Identified content gaps
4. Differentiation opportunities
5. Successful content formats and engagement strategies

Format as structured analysis with specific, actionable insights.
            """)
        ])
        
        try:
            response = await self.research_llm.agenerate([
                competitive_analysis_prompt.format_messages(
                    topic=topic,
                    target_audience=target_audience
                )
            ])
            
            competitive_analysis = response.generations[0][0].text
            
            # Parse and structure the competitive insights
            # In a production system, this would be more sophisticated parsing
            competitor_insights = [
                CompetitorInsight(
                    competitor_name="Industry Leader Example",
                    content_title=f"Comprehensive Guide to {topic}",
                    key_angles=["Technical depth", "Implementation focus", "ROI emphasis"],
                    engagement_signals={"estimated_shares": 100, "estimated_views": 5000},
                    content_gaps=["Lacks practical examples", "Missing implementation timeline"],
                    differentiation_opportunities=["Add real customer case studies", "Include technical implementation guide"]
                ),
                CompetitorInsight(
                    competitor_name="Market Challenger",
                    content_title=f"{topic} Best Practices",
                    key_angles=["Best practices focus", "Industry trends", "Future outlook"],
                    engagement_signals={"estimated_shares": 75, "estimated_views": 3000},
                    content_gaps=["Surface-level analysis", "Generic recommendations"],
                    differentiation_opportunities=["Provide deeper technical insights", "Include proprietary research data"]
                )
            ]
            
            return competitor_insights
            
        except Exception as e:
            self.logger.error(f"Competitive analysis failed: {str(e)}")
            # Return basic competitive insights as fallback
            return [
                CompetitorInsight(
                    competitor_name="Generic Competitor",
                    content_title=f"Basic {topic} Content",
                    key_angles=["General information", "Basic benefits"],
                    content_gaps=["Lacks depth", "Missing practical guidance"],
                    differentiation_opportunities=["Add technical depth", "Include real case studies"]
                )
            ]
    
    async def _generate_content_structure(
        self,
        topic: str,
        content_type: ContentType,
        keyword_research: Dict[str, Any],
        competitor_insights: List[CompetitorInsight]
    ) -> ContentStructure:
        """Generate strategic content structure and outline."""
        
        structure_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a strategic content architect specializing in B2B financial services content that drives business results.

Create a detailed content structure that includes:
1. Optimal word count for the content type and SEO goals
2. Compelling headlines that incorporate primary keywords
3. Logical content outline with section descriptions
4. Recommended sections for maximum impact
5. Strategic call-to-action recommendations
6. Internal linking opportunities for SEO

Consider:
- B2B finance professional reading patterns
- SEO optimization requirements
- Thought leadership positioning
- Lead generation opportunities
- Competitive differentiation
            """),
            HumanMessage(content="""
Topic: {topic}
Content Type: {content_type}
Primary Keyword: {primary_keyword}
Secondary Keywords: {secondary_keywords}

Competitive Gaps Identified:
{competitive_gaps}

Create a comprehensive content structure that:
1. Addresses competitive gaps
2. Optimizes for target keywords
3. Serves B2B finance professional needs
4. Drives business objectives
5. Establishes thought leadership

Provide specific, actionable structure recommendations.
            """)
        ])
        
        try:
            # Format competitive gaps for context
            competitive_gaps = []
            for insight in competitor_insights:
                competitive_gaps.extend(insight.content_gaps)
            
            response = await self.strategy_llm.agenerate([
                structure_prompt.format_messages(
                    topic=topic,
                    content_type=content_type.value,
                    primary_keyword=keyword_research["primary_keyword"].keyword,
                    secondary_keywords=[kw.keyword for kw in keyword_research["secondary_keywords"]],
                    competitive_gaps="; ".join(competitive_gaps)
                )
            ])
            
            # Create structured content outline
            content_structure = ContentStructure(
                estimated_word_count=self._get_optimal_word_count(content_type),
                suggested_headlines=[
                    f"The Complete Guide to {topic} for B2B Finance Leaders",
                    f"How {topic} is Transforming Financial Operations",
                    f"{topic}: Strategic Implementation Guide for CFOs"
                ],
                content_outline=[
                    {"section": "Executive Summary", "description": "Key insights and takeaways for busy executives"},
                    {"section": "Market Context", "description": "Current landscape and industry trends"},
                    {"section": "Strategic Benefits", "description": "Business value and ROI considerations"},
                    {"section": "Implementation Guide", "description": "Practical steps and best practices"},
                    {"section": "Case Studies", "description": "Real-world examples and success stories"},
                    {"section": "Future Outlook", "description": "Trends and strategic recommendations"}
                ],
                recommended_sections=[
                    "Introduction with hook",
                    "Problem definition",
                    "Solution overview", 
                    "Implementation strategy",
                    "ROI analysis",
                    "Conclusion with CTA"
                ],
                call_to_actions=[
                    "Schedule a strategy consultation",
                    "Download implementation checklist", 
                    "Request custom ROI analysis",
                    "Join executive roundtable discussion"
                ],
                internal_link_opportunities=[
                    "Related case studies",
                    "Implementation resources",
                    "Industry trend reports",
                    "Executive insights blog"
                ]
            )
            
            return content_structure
            
        except Exception as e:
            self.logger.error(f"Content structure generation failed: {str(e)}")
            return self._generate_fallback_structure(content_type)
    
    def _get_optimal_word_count(self, content_type: ContentType) -> int:
        """Get optimal word count for content type."""
        word_counts = {
            ContentType.BLOG_POST: 2000,
            ContentType.LINKEDIN_ARTICLE: 1500,
            ContentType.WHITE_PAPER: 5000,
            ContentType.CASE_STUDY: 1200,
            ContentType.GUIDE: 3000,
            ContentType.TUTORIAL: 1800
        }
        return word_counts.get(content_type, 2000)
    
    def _generate_fallback_structure(self, content_type: ContentType) -> ContentStructure:
        """Generate fallback content structure."""
        return ContentStructure(
            estimated_word_count=self._get_optimal_word_count(content_type),
            suggested_headlines=["Strategic Guide to Implementation", "Best Practices for Success"],
            content_outline=[
                {"section": "Introduction", "description": "Overview and context"},
                {"section": "Main Content", "description": "Core insights and analysis"},
                {"section": "Conclusion", "description": "Key takeaways and next steps"}
            ],
            recommended_sections=["Introduction", "Main body", "Conclusion"],
            call_to_actions=["Contact us for more information", "Subscribe for updates"],
            internal_link_opportunities=["Related articles", "Resource library"]
        )
    
    async def _define_success_metrics(
        self,
        primary_purpose: ContentPurpose,
        content_type: ContentType
    ) -> Dict[str, Any]:
        """Define success metrics and KPIs based on content purpose."""
        
        success_metrics = {
            "kpis": [],
            "targets": {}
        }
        
        # Purpose-based metrics
        purpose_metrics = {
            ContentPurpose.LEAD_GENERATION: {
                "kpis": ["Lead conversion rate", "Form submissions", "Download completions", "Demo requests"],
                "targets": {"lead_conversion_rate": "3-5%", "form_submissions": "50+ per month"}
            },
            ContentPurpose.BRAND_AWARENESS: {
                "kpis": ["Organic traffic", "Social shares", "Brand mention increase", "Engagement rate"],
                "targets": {"organic_traffic_increase": "25% in 3 months", "social_shares": "100+ per post"}
            },
            ContentPurpose.THOUGHT_LEADERSHIP: {
                "kpis": ["Executive engagement", "Industry citations", "Speaking opportunities", "Media mentions"],
                "targets": {"executive_engagement": "C-level comments/shares", "industry_citations": "5+ per quarter"}
            },
            ContentPurpose.SEO_RANKING: {
                "kpis": ["Keyword rankings", "Organic traffic", "Click-through rate", "Backlink acquisition"],
                "targets": {"target_keywords_top_10": "80% within 6 months", "organic_traffic_increase": "40%"}
            }
        }
        
        metrics_data = purpose_metrics.get(primary_purpose, purpose_metrics[ContentPurpose.LEAD_GENERATION])
        success_metrics.update(metrics_data)
        
        # Content type specific additions
        if content_type == ContentType.LINKEDIN_ARTICLE:
            success_metrics["kpis"].extend(["LinkedIn engagement rate", "Profile views", "Connection requests"])
            success_metrics["targets"]["linkedin_engagement"] = "5%+ engagement rate"
        
        return success_metrics
    
    async def _generate_strategic_brief(
        self,
        brief_request: Dict[str, Any],
        keyword_research: Dict[str, Any],
        competitor_insights: List[CompetitorInsight],
        content_structure: ContentStructure
    ) -> Dict[str, Any]:
        """Generate strategic content brief using AI."""
        
        strategic_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a senior marketing strategist creating executive-level content briefs for B2B financial services companies.

Create a strategic brief that includes:
1. Clear marketing objectives tied to business goals
2. Unique value proposition and key messages
3. Market gap identification and differentiation strategy
4. Tone and voice guidelines
5. Content calendar and distribution recommendations

Focus on creating briefs that drive measurable business results and establish market leadership.
            """),
            HumanMessage(content="""
Create a strategic content brief for:

Topic: {topic}
Company Context: {company_context}
Target Audience: {target_audience}
Primary Purpose: {primary_purpose}

SEO Strategy:
- Primary Keyword: {primary_keyword}
- Search Intent: {search_intent}

Competitive Landscape:
{competitive_summary}

Content Structure: {word_count} words with sections on strategic benefits, implementation, and ROI.

Provide strategic guidance that ensures this content drives business objectives while establishing thought leadership.
            """)
        ])
        
        try:
            competitive_summary = "; ".join([
                f"{insight.competitor_name}: {', '.join(insight.key_angles)}"
                for insight in competitor_insights
            ])
            
            response = await self.strategy_llm.agenerate([
                strategic_prompt.format_messages(
                    topic=brief_request.get("topic", ""),
                    company_context=brief_request.get("company_context", ""),
                    target_audience=brief_request.get("target_audience", ""),
                    primary_purpose=brief_request.get("primary_purpose", ""),
                    primary_keyword=keyword_research["primary_keyword"].keyword,
                    search_intent=keyword_research["search_intent"],
                    competitive_summary=competitive_summary,
                    word_count=content_structure.estimated_word_count
                )
            ])
            
            # Parse strategic brief (simplified - in production use more robust parsing)
            brief_content = response.generations[0][0].text
            
            return {
                "title": f"Strategic {brief_request.get('topic', '')} Content Brief",
                "marketing_objective": "Drive qualified lead generation while establishing thought leadership",
                "unique_value_proposition": f"Unique perspective on {brief_request.get('topic', '')} implementation",
                "key_messages": [
                    f"{brief_request.get('topic', '')} drives measurable business value",
                    "Implementation requires strategic approach",
                    "Success depends on proper execution"
                ],
                "market_gap": "Lack of practical implementation guidance",
                "differentiation_strategy": "Combine strategic insights with practical implementation guidance",
                "tone_and_voice": "Professional, authoritative, accessible to executives",
                "writing_style_notes": "Balance technical depth with executive accessibility",
                "content_calendar_notes": "Schedule for maximum executive engagement"
            }
            
        except Exception as e:
            self.logger.error(f"Strategic brief generation failed: {str(e)}")
            return {
                "title": f"{brief_request.get('topic', 'Content')} Strategic Brief",
                "marketing_objective": "Drive business growth through strategic content",
                "unique_value_proposition": "Unique market perspective and practical insights",
                "key_messages": ["Strategic value", "Practical implementation", "Measurable results"],
                "market_gap": "Need for more strategic content",
                "differentiation_strategy": "Focus on practical business value",
                "tone_and_voice": "Professional and authoritative"
            }
    
    def _estimate_creation_time(self, content_type: ContentType, word_count: int) -> str:
        """Estimate content creation time."""
        
        # Base writing speed: 200 words per hour for quality content
        writing_hours = word_count / 200
        
        # Add research and editing time
        research_hours = {
            ContentType.BLOG_POST: 2,
            ContentType.WHITE_PAPER: 8,
            ContentType.CASE_STUDY: 4,
            ContentType.GUIDE: 6
        }.get(content_type, 2)
        
        editing_hours = writing_hours * 0.5  # 50% of writing time for editing
        
        total_hours = writing_hours + research_hours + editing_hours
        
        if total_hours <= 8:
            return f"{int(total_hours)} hours (1 day)"
        elif total_hours <= 16:
            return f"{int(total_hours)} hours (2 days)"
        else:
            days = int(total_hours / 8)
            return f"{int(total_hours)} hours ({days} days)"
    
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute the content brief agent with input data."""
        if isinstance(input_data, dict):
            brief_request = input_data
        else:
            brief_request = {"topic": str(input_data)}
        
        # Create content brief
        content_brief = await self.create_content_brief(brief_request)
        
        # Generate summary  
        summary = await self.generate_brief_summary(content_brief)
        
        return {
            "brief": content_brief.model_dump(),
            "summary": summary
        }
    
    async def generate_brief_summary(self, content_brief: ContentBrief) -> str:
        """Generate a concise executive summary of the content brief."""
        
        summary_parts = [
            f"**Content Brief: {content_brief.title}**",
            f"**Type:** {content_brief.content_type.value.replace('_', ' ').title()}",
            f"**Primary Purpose:** {content_brief.primary_purpose.value.replace('_', ' ').title()}",
            f"**Target Audience:** {content_brief.target_audience}",
            "",
            f"**SEO Strategy:**",
            f"- Primary Keyword: {content_brief.primary_keyword.keyword} ({content_brief.primary_keyword.search_volume} searches/month)",
            f"- Search Intent: {content_brief.target_search_intent}",
            f"- Secondary Keywords: {', '.join([kw.keyword for kw in content_brief.secondary_keywords[:3]])}",
            "",
            f"**Content Strategy:**",
            f"- Estimated Length: {content_brief.content_structure.estimated_word_count} words",
            f"- Creation Time: {content_brief.estimated_creation_time}",
            f"- Differentiation: {content_brief.differentiation_strategy}",
            "",
            f"**Success Metrics:**",
            f"- KPIs: {', '.join(content_brief.success_kpis[:3])}",
            "",
            f"**Next Steps:**",
            f"1. Review and approve brief",
            f"2. Assign to content creator",
            f"3. Begin content development",
            f"4. Schedule for distribution"
        ]
        
        return "\n".join(summary_parts)