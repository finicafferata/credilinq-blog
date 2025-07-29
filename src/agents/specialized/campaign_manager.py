"""
Campaign Manager Agent - Creates comprehensive marketing campaigns from content analysis.
"""

from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator


class CampaignManagerAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for analyzing content and generating comprehensive marketing campaigns
    with strategic task planning, timeline management, and multi-channel distribution.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CAMPAIGN_MANAGER,
                name="CampaignManagerAgent",
                description="Creates strategic marketing campaigns from content analysis",
                capabilities=[
                    "campaign_strategy",
                    "content_analysis",
                    "task_orchestration",
                    "multi_channel_planning",
                    "timeline_management",
                    "performance_optimization"
                ],
                version="2.1.0"  # Version bumped to reflect improvements
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
    
    def _initialize(self):
        """Initialize the LLM and other resources."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,  # Lower temperature for strategic planning
                openai_api_key=settings.OPENAI_API_KEY
            )
            self.logger.info("CampaignManagerAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CampaignManagerAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for campaign generation."""
        super()._validate_input(input_data)
        
        required_fields = ["content", "company_context"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation
        for field in required_fields:
            if isinstance(input_data[field], str):
                self.security_validator.validate_input(input_data[field])
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate a comprehensive marketing campaign from content analysis.
        
        Args:
            input_data: Dictionary containing:
                - content: The source content to build campaign around
                - company_context: Company/brand context and goals
                - campaign_goals: Optional specific campaign objectives
                - target_audience: Optional audience specification
                - timeline_weeks: Optional campaign duration (default: 4)
            context: Execution context
            
        Returns:
            AgentResult: Result containing the campaign strategy and tasks
        """
        try:
            # Initialize if not already done
            if self.llm is None:
                self._initialize()
            
            content = input_data["content"]
            company_context = input_data["company_context"]
            campaign_goals = input_data.get("campaign_goals", "Maximize reach and engagement")
            target_audience = input_data.get("target_audience", "Professional audience")
            timeline_weeks = input_data.get("timeline_weeks", 4)
            
            self.logger.info(f"Generating campaign strategy for content ({len(content.split())} words)")
            
            # Generate comprehensive campaign strategy
            campaign_strategy = self._generate_campaign_strategy(
                content, company_context, campaign_goals, target_audience, timeline_weeks
            )
            
            # Generate detailed task plan
            task_plan = self._generate_task_plan(campaign_strategy, timeline_weeks)
            
            result_data = {
                "campaign_strategy": campaign_strategy,
                "task_plan": task_plan,
                "timeline_weeks": timeline_weeks,
                "total_tasks": len(task_plan),
                "campaign_analysis": self._analyze_campaign_scope(campaign_strategy, task_plan)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "campaign_manager",
                    "campaign_duration": timeline_weeks,
                    "total_tasks": len(task_plan)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate campaign: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CAMPAIGN_GENERATION_FAILED"
            )
    
    def _generate_campaign_strategy(
        self,
        content: str,
        company_context: str,
        campaign_goals: str,
        target_audience: str,
        timeline_weeks: int
    ) -> Dict[str, Any]:
        """Generate comprehensive campaign strategy based on content analysis."""
        
        prompt = self._create_campaign_strategy_prompt(
            content, company_context, campaign_goals, target_audience, timeline_weeks
        )
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            strategy = self._parse_campaign_response(response.content.strip())
            
            self.logger.info(f"Generated campaign strategy with {len(strategy.get('channels', []))} channels")
            return strategy
            
        except Exception as e:
            self.logger.warning(f"LLM campaign generation failed: {str(e)}, using fallback")
            return self._create_fallback_campaign_strategy(timeline_weeks)
    
    def _create_campaign_strategy_prompt(
        self,
        content: str,
        company_context: str,
        campaign_goals: str,
        target_audience: str,
        timeline_weeks: int
    ) -> str:
        """Create prompt for campaign strategy generation."""
        content_preview = content[:1000] + "..." if len(content) > 1000 else content
        
        return f"""
        Act as a senior Marketing Strategist and Campaign Manager with 15+ years of experience in multi-channel digital marketing. Analyze the provided content and create a comprehensive marketing campaign strategy.

        **Source Content Analysis:**
        {content_preview}

        **Company Context & Brand:** "{company_context}"
        **Campaign Goals:** "{campaign_goals}"
        **Target Audience:** "{target_audience}"
        **Campaign Duration:** {timeline_weeks} weeks

        **Instructions:**
        - Analyze the content to identify key themes, value propositions, and messaging opportunities.
        - Create a strategic campaign that maximizes content ROI across multiple channels.
        - Design a cohesive narrative that adapts the core message for different platforms.
        - Include specific tactics for each channel with clear success metrics.
        - Consider the customer journey from awareness to conversion.
        - Align all campaign elements with the company's brand voice and goals.

        **Required Campaign Elements:**
        1. **Campaign Theme & Core Message**
        2. **Target Channel Strategy** (LinkedIn, Twitter, Blog syndication, Email, etc.)
        3. **Content Repurposing Plan** (formats, adaptations, messaging angles)
        4. **Visual Asset Requirements** (images, graphics, videos)
        5. **Engagement Tactics** (polls, discussions, user-generated content)
        6. **Timeline & Phasing Strategy**
        7. **Success Metrics & KPIs**

        **Negative Constraints:**
        - **Do not** create generic, one-size-fits-all campaigns.
        - **Avoid** platforms that don't align with the target audience.
        - **Do not** ignore the company's specific context and goals.

        **Output Format:**
        Return the complete campaign strategy as a JSON object inside <strategy> tags:
        <strategy>
        {{
            "campaign_name": "Strategic campaign title",
            "core_message": "Primary value proposition and messaging theme",
            "target_channels": [
                {{
                    "channel": "LinkedIn",
                    "strategy": "Specific approach for this channel",
                    "content_types": ["Post", "Article", "Video"],
                    "posting_frequency": "3 times per week",
                    "success_metrics": ["Engagement rate", "Lead generation"]
                }}
            ],
            "content_variants": [
                {{
                    "format": "LinkedIn Post",
                    "messaging_angle": "Thought leadership focus",
                    "target_length": "800-1200 characters",
                    "key_elements": ["Hook", "Insight", "Call-to-action"]
                }}
            ],
            "visual_assets": [
                {{
                    "asset_type": "Blog Header Image",
                    "purpose": "Brand awareness and content promotion",
                    "specifications": "1200x630px, professional design"
                }}
            ],
            "timeline_phases": [
                {{
                    "phase": "Launch Week",
                    "week": 1,
                    "focus": "Awareness building",
                    "key_activities": ["Blog publication", "Social media launch"]
                }}
            ],
            "success_metrics": [
                "Content engagement rate",
                "Lead generation",
                "Brand awareness lift"
            ]
        }}
        </strategy>
        """
    
    def _parse_campaign_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the campaign strategy."""
        try:
            # Try to find strategy within <strategy> tags
            match = re.search(r"<strategy>(.*?)</strategy>", response, re.DOTALL)
            if match:
                strategy_json = match.group(1).strip()
                strategy = json.loads(strategy_json)
                return strategy
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            self.logger.warning(f"Failed to parse campaign strategy JSON: {str(e)}")
        
        # Fallback: try to extract key information from text
        return self._extract_strategy_from_text(response)
    
    def _extract_strategy_from_text(self, response: str) -> Dict[str, Any]:
        """Extract campaign strategy from unstructured text response."""
        # Basic extraction of campaign elements
        lines = response.split('\n')
        
        strategy = {
            "campaign_name": "AI-Generated Marketing Campaign",
            "core_message": "Strategic content marketing initiative",
            "target_channels": [
                {
                    "channel": "LinkedIn",
                    "strategy": "Professional thought leadership",
                    "content_types": ["Post", "Article"],
                    "posting_frequency": "3 times per week",
                    "success_metrics": ["Engagement rate", "Profile views"]
                },
                {
                    "channel": "Blog Syndication",
                    "strategy": "SEO and organic reach",
                    "content_types": ["Blog post", "Guest articles"],
                    "posting_frequency": "1 time per week",
                    "success_metrics": ["Organic traffic", "Backlinks"]
                }
            ],
            "content_variants": [
                {
                    "format": "LinkedIn Post",
                    "messaging_angle": "Professional insights",
                    "target_length": "800-1200 characters",
                    "key_elements": ["Hook", "Value proposition", "Call-to-action"]
                }
            ],
            "visual_assets": [
                {
                    "asset_type": "Social Media Graphics",
                    "purpose": "Engagement and brand recognition",
                    "specifications": "Professional design with brand colors"
                }
            ],
            "timeline_phases": [
                {
                    "phase": "Content Launch",
                    "week": 1,
                    "focus": "Initial promotion",
                    "key_activities": ["Blog publication", "Social promotion"]
                }
            ],
            "success_metrics": ["Engagement rate", "Lead generation", "Brand awareness"]
        }
        
        return strategy
    
    def _generate_task_plan(self, campaign_strategy: Dict[str, Any], timeline_weeks: int) -> List[Dict[str, Any]]:
        """Generate detailed task plan from campaign strategy."""
        tasks = []
        task_id = 1
        
        # Generate content repurposing tasks
        for variant in campaign_strategy.get("content_variants", []):
            tasks.append({
                "task_id": task_id,
                "task_type": "content_repurpose",
                "target_format": variant["format"],
                "messaging_angle": variant["messaging_angle"],
                "specifications": {
                    "target_length": variant.get("target_length", "Standard length"),
                    "key_elements": variant.get("key_elements", [])
                },
                "priority": "high",
                "estimated_duration_hours": 2,
                "status": "pending",
                "week": 1
            })
            task_id += 1
        
        # Generate visual asset tasks
        for asset in campaign_strategy.get("visual_assets", []):
            tasks.append({
                "task_id": task_id,
                "task_type": "create_visual_asset",
                "asset_type": asset["asset_type"],
                "purpose": asset["purpose"],
                "specifications": asset.get("specifications", "Standard design"),
                "priority": "medium",
                "estimated_duration_hours": 3,
                "status": "pending",
                "week": 1
            })
            task_id += 1
        
        # Generate channel-specific distribution tasks
        for channel in campaign_strategy.get("target_channels", []):
            # Create posting schedule tasks
            frequency_map = {
                "daily": 7 * timeline_weeks,
                "3 times per week": 3 * timeline_weeks,
                "twice per week": 2 * timeline_weeks,
                "weekly": timeline_weeks,
                "1 time per week": timeline_weeks
            }
            
            frequency = channel.get("posting_frequency", "weekly").lower()
            post_count = frequency_map.get(frequency, timeline_weeks)
            
            for week in range(1, min(timeline_weeks + 1, 5)):  # Limit to first 4 weeks
                tasks.append({
                    "task_id": task_id,
                    "task_type": "content_distribution",
                    "channel": channel["channel"],
                    "strategy": channel["strategy"],
                    "content_types": channel.get("content_types", []),
                    "priority": "high",
                    "estimated_duration_hours": 1,
                    "status": "pending",
                    "week": week
                })
                task_id += 1
        
        # Generate analytics and optimization tasks
        for week in range(2, timeline_weeks + 1):
            if week % 2 == 0:  # Every other week
                tasks.append({
                    "task_id": task_id,
                    "task_type": "performance_analysis",
                    "focus": "Campaign performance review and optimization",
                    "metrics": campaign_strategy.get("success_metrics", ["Engagement", "Reach"]),
                    "priority": "medium",
                    "estimated_duration_hours": 2,
                    "status": "pending",
                    "week": week
                })
                task_id += 1
        
        return sorted(tasks, key=lambda x: (x.get("week", 1), x.get("priority", "medium")))
    
    def _create_fallback_campaign_strategy(self, timeline_weeks: int) -> Dict[str, Any]:
        """Create fallback campaign strategy when LLM fails."""
        self.logger.info(f"Using fallback campaign strategy for {timeline_weeks} weeks")
        
        return {
            "campaign_name": "Multi-Channel Content Campaign",
            "core_message": "Strategic content marketing for professional audience",
            "target_channels": [
                {
                    "channel": "LinkedIn",
                    "strategy": "Professional networking and thought leadership",
                    "content_types": ["Post", "Article"],
                    "posting_frequency": "3 times per week",
                    "success_metrics": ["Engagement rate", "Profile views", "Connections"]
                },
                {
                    "channel": "Company Blog",
                    "strategy": "SEO optimization and organic reach",
                    "content_types": ["Blog post", "Guest articles"],
                    "posting_frequency": "Weekly",
                    "success_metrics": ["Organic traffic", "Time on page", "Backlinks"]
                }
            ],
            "content_variants": [
                {
                    "format": "LinkedIn Post",
                    "messaging_angle": "Professional insights and industry trends",
                    "target_length": "800-1200 characters",
                    "key_elements": ["Compelling hook", "Key insight", "Call-to-action"]
                },
                {
                    "format": "Twitter Thread",
                    "messaging_angle": "Quick tips and actionable advice",
                    "target_length": "280 characters per tweet",
                    "key_elements": ["Thread starter", "Value points", "Engagement question"]
                }
            ],
            "visual_assets": [
                {
                    "asset_type": "Blog Header Image",
                    "purpose": "Brand awareness and content promotion",
                    "specifications": "1200x630px, professional design with brand colors"
                },
                {
                    "asset_type": "Social Media Graphics",
                    "purpose": "Engagement and visual appeal",
                    "specifications": "Square format 1080x1080px for Instagram/LinkedIn"
                }
            ],
            "timeline_phases": [
                {
                    "phase": "Launch Week",
                    "week": 1,
                    "focus": "Content publication and initial promotion",
                    "key_activities": ["Blog publication", "Social media announcement", "Email newsletter"]
                },
                {
                    "phase": "Amplification Phase",
                    "week": 2,
                    "focus": "Expand reach and engagement",
                    "key_activities": ["Repurposed content", "Community engagement", "Influencer outreach"]
                }
            ],
            "success_metrics": [
                "Total reach and impressions",
                "Engagement rate across channels",
                "Lead generation and conversions",
                "Brand awareness lift"
            ]
        }
    
    def _analyze_campaign_scope(self, strategy: Dict[str, Any], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the scope and complexity of the generated campaign."""
        
        # Analyze channels
        channels = strategy.get("target_channels", [])
        channel_names = [ch.get("channel", "Unknown") for ch in channels]
        
        # Analyze tasks by type
        task_types = {}
        total_hours = 0
        weeks_covered = set()
        
        for task in tasks:
            task_type = task.get("task_type", "unknown")
            task_types[task_type] = task_types.get(task_type, 0) + 1
            total_hours += task.get("estimated_duration_hours", 1)
            weeks_covered.add(task.get("week", 1))
        
        # Calculate complexity score
        complexity_factors = {
            "channel_count": len(channels),
            "task_count": len(tasks),
            "content_variants": len(strategy.get("content_variants", [])),
            "visual_assets": len(strategy.get("visual_assets", [])),
            "timeline_phases": len(strategy.get("timeline_phases", []))
        }
        
        complexity_score = sum(complexity_factors.values())
        
        if complexity_score < 10:
            complexity_level = "Simple"
        elif complexity_score < 20:
            complexity_level = "Moderate"
        else:
            complexity_level = "Complex"
        
        return {
            "campaign_channels": channel_names,
            "total_estimated_hours": total_hours,
            "weeks_with_activities": len(weeks_covered),
            "task_breakdown": task_types,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "recommended_team_size": max(1, complexity_score // 8)
        }
