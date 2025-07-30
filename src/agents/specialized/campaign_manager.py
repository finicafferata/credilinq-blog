#!/usr/bin/env python3
"""
Campaign Manager Agent
Responsible for creating strategic campaign plans and coordinating all campaign activities.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.agents.core.base_agent import BaseAgent
from src.config.database import db_config

logger = logging.getLogger(__name__)

@dataclass
class CampaignStrategy:
    """Campaign strategy configuration"""
    target_audience: str
    key_messages: List[str]
    distribution_channels: List[str]
    timeline_weeks: int
    budget_allocation: Dict[str, float]
    success_metrics: Dict[str, Any]

@dataclass
class CampaignTask:
    """Individual campaign task"""
    task_type: str
    platform: str
    content_type: str
    priority: str
    estimated_duration_hours: int
    dependencies: List[str]
    assigned_agent: str

class CampaignManagerAgent(BaseAgent):
    """
    Campaign Manager Agent - Orchestrates the entire campaign workflow
    """
    
    def __init__(self):
        super().__init__()
        self.agent_name = "CampaignManager"
        self.description = "Strategic campaign planning and coordination"
        
    async def create_campaign_plan(self, blog_id: str, campaign_name: str, 
                                 company_context: str, content_type: str = "blog",
                                 template_id: Optional[str] = None, 
                                 template_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive campaign plan for a blog post with optional template support
        """
        try:
            logger.info(f"Creating campaign plan for blog {blog_id} with template {template_id}")
            
            # 1. Analyze the blog content and context
            blog_analysis = await self._analyze_blog_content(blog_id, company_context)
            
            # 2. Generate campaign strategy (with template if provided)
            if template_id and template_config:
                strategy = await self._generate_template_strategy(blog_analysis, template_id, template_config)
            else:
                strategy = await self._generate_campaign_strategy(blog_analysis, content_type)
            
            # 3. Create campaign timeline
            timeline = await self._create_campaign_timeline(strategy)
            
            # 4. Generate task breakdown
            tasks = await self._generate_campaign_tasks(strategy, timeline)
            
            # 5. Save campaign to database
            campaign_id = await self._save_campaign_to_db(blog_id, campaign_name, strategy)
            
            # 6. Save tasks to database
            await self._save_tasks_to_db(campaign_id, tasks)
            
            return {
                "campaign_id": campaign_id,
                "strategy": {
                    "target_audience": strategy.target_audience,
                    "key_messages": strategy.key_messages,
                    "distribution_channels": strategy.distribution_channels,
                    "timeline_weeks": strategy.timeline_weeks,
                    "budget_allocation": strategy.budget_allocation,
                    "success_metrics": strategy.success_metrics
                },
                "timeline": timeline,
                "tasks": [
                    {
                        "task_type": task.task_type,
                        "platform": task.platform,
                        "content_type": task.content_type,
                        "priority": task.priority,
                        "estimated_duration_hours": task.estimated_duration_hours,
                        "dependencies": task.dependencies,
                        "assigned_agent": task.assigned_agent
                    } for task in tasks
                ],
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Error creating campaign plan: {str(e)}")
            raise Exception(f"Failed to create campaign plan: {str(e)}")
    
    async def _analyze_blog_content(self, blog_id: str, company_context: str) -> Dict[str, Any]:
        """
        Analyze blog content to understand key themes and opportunities
        """
        try:
            # Get blog content from database
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT title, "contentMarkdown", "initialPrompt"
                    FROM "BlogPost" 
                    WHERE id = %s
                """, (blog_id,))
                row = cur.fetchone()
                
                if not row:
                    raise Exception("Blog not found")
                
                title, content_markdown, initial_prompt = row
                
                # Analyze content using AI
                analysis_prompt = f"""
                Analyze this blog content for campaign planning:
                
                Title: {title}
                Company Context: {company_context}
                Content: {content_markdown[:1000]}...
                
                Provide analysis in JSON format:
                {{
                    "key_themes": ["theme1", "theme2"],
                    "target_audience": "description",
                    "key_messages": ["message1", "message2"],
                    "content_opportunities": ["opportunity1", "opportunity2"],
                    "estimated_engagement": "high/medium/low"
                }}
                """
                
                # For now, provide a structured analysis without AI calls
                analysis = {
                    "key_themes": ["business growth", "market expansion"],
                    "target_audience": "B2B professionals and business owners",
                    "key_messages": [
                        f"Learn about {title}",
                        "Discover actionable insights",
                        "Transform your business approach"
                    ],
                    "content_opportunities": ["social media", "email marketing", "professional networks"],
                    "estimated_engagement": "medium"
                }
                
                return {
                    "title": title,
                    "content_markdown": content_markdown,
                    "initial_prompt": initial_prompt,
                    "analysis": analysis
                }
                
        except Exception as e:
            logger.error(f"Error analyzing blog content: {str(e)}")
            raise
    
    async def _generate_template_strategy(self, blog_analysis: Dict[str, Any], 
                                        template_id: str, template_config: Dict[str, Any]) -> CampaignStrategy:
        """
        Generate a campaign strategy based on a predefined template
        """
        try:
            logger.info(f"Generating strategy for template {template_id}")
            
            # Template-specific strategy generation
            if template_id == "social-blast":
                return CampaignStrategy(
                    target_audience=blog_analysis.get('target_audience', 'General business audience'),
                    key_messages=[
                        f"Key insight: {blog_analysis.get('key_insight', 'Share valuable content')}",
                        f"Call to action: {blog_analysis.get('cta', 'Learn more')}"
                    ],
                    distribution_channels=template_config.get('channels', ['linkedin', 'twitter', 'facebook']),
                    timeline_weeks=1,
                    budget_allocation={"content_creation": 70, "promotion": 30},
                    success_metrics={"impressions": 10000, "engagement_rate": 5.0, "clicks": 500}
                )
            elif template_id == "professional-share":
                return CampaignStrategy(
                    target_audience=blog_analysis.get('target_audience', 'Business professionals'),
                    key_messages=[f"Professional insight: {blog_analysis.get('summary', 'Industry expertise')}"],
                    distribution_channels=['linkedin'],
                    timeline_weeks=1,
                    budget_allocation={"content_creation": 80, "promotion": 20},
                    success_metrics={"linkedin_views": 5000, "engagement_rate": 7.0, "profile_visits": 200}
                )
            elif template_id == "email-campaign":
                return CampaignStrategy(
                    target_audience=blog_analysis.get('target_audience', 'Email subscribers'),
                    key_messages=[f"Newsletter: {blog_analysis.get('title', 'Latest insights')}"],
                    distribution_channels=['email'],
                    timeline_weeks=1,
                    budget_allocation={"content_creation": 90, "promotion": 10},
                    success_metrics={"open_rate": 25.0, "click_rate": 5.0, "forwards": 50}
                )
            else:
                # Fall back to default strategy
                return await self._generate_campaign_strategy(blog_analysis, "blog")
                
        except Exception as e:
            logger.error(f"Error generating template strategy: {str(e)}")
            # Fall back to default strategy
            return await self._generate_campaign_strategy(blog_analysis, "blog")
    
    async def _generate_campaign_strategy(self, blog_analysis: Dict[str, Any], 
                                       content_type: str) -> CampaignStrategy:
        """
        Generate strategic campaign plan based on blog analysis
        """
        try:
            analysis = blog_analysis["analysis"]
            
            strategy_prompt = f"""
            Create a campaign strategy for this content:
            
            Blog Title: {blog_analysis['title']}
            Key Themes: {analysis['key_themes']}
            Target Audience: {analysis['target_audience']}
            Content Type: {content_type}
            
            Generate strategy in JSON format:
            {{
                "target_audience": "detailed audience description",
                "key_messages": ["message1", "message2", "message3"],
                "distribution_channels": ["linkedin", "twitter", "email"],
                "timeline_weeks": 4,
                "budget_allocation": {{
                    "content_creation": 0.4,
                    "distribution": 0.3,
                    "promotion": 0.2,
                    "analytics": 0.1
                }},
                "success_metrics": {{
                    "impressions": 10000,
                    "engagement_rate": 0.05,
                    "conversions": 100,
                    "brand_awareness": "high"
                }}
            }}
            """
            
            # For now, provide a default strategy structure without AI calls
            strategy_data = {
                "target_audience": analysis["target_audience"],
                "key_messages": analysis["key_messages"],
                "distribution_channels": ["linkedin", "twitter", "email"],
                "timeline_weeks": 2,
                "budget_allocation": {
                    "content_creation": 0.4,
                    "distribution": 0.3,
                    "promotion": 0.2,
                    "analytics": 0.1
                },
                "success_metrics": {
                    "impressions": 5000,
                    "engagement_rate": 0.05,
                    "conversions": 50,
                    "brand_awareness": "medium"
                }
            }
            
            return CampaignStrategy(
                target_audience=strategy_data["target_audience"],
                key_messages=strategy_data["key_messages"],
                distribution_channels=strategy_data["distribution_channels"],
                timeline_weeks=strategy_data["timeline_weeks"],
                budget_allocation=strategy_data["budget_allocation"],
                success_metrics=strategy_data["success_metrics"]
            )
            
        except Exception as e:
            logger.error(f"Error generating campaign strategy: {str(e)}")
            raise
    
    async def _create_campaign_timeline(self, strategy: CampaignStrategy) -> List[Dict[str, Any]]:
        """
        Create campaign timeline with phases
        """
        try:
            timeline_prompt = f"""
            Create a {strategy.timeline_weeks}-week campaign timeline for:
            Channels: {strategy.distribution_channels}
            Key Messages: {strategy.key_messages}
            
            Generate timeline in JSON format:
            [
                {{
                    "phase": "awareness",
                    "week": 1,
                    "focus": "description",
                    "channels": ["channel1"],
                    "goals": {{"metric": "target_value"}}
                }}
            ]
            """
            
            # For now, provide a default timeline structure
            timeline = [
                {
                    "phase": "preparation",
                    "week": 1,
                    "focus": "Content creation and adaptation",
                    "channels": strategy.distribution_channels,
                    "goals": {"content_ready": 100}
                },
                {
                    "phase": "launch",
                    "week": 2,
                    "focus": "Distribution and initial promotion",
                    "channels": strategy.distribution_channels,
                    "goals": {"reach": 1000, "engagement": 50}
                }
            ]
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error creating campaign timeline: {str(e)}")
            raise
    
    async def _generate_campaign_tasks(self, strategy: CampaignStrategy, 
                                     timeline: List[Dict[str, Any]]) -> List[CampaignTask]:
        """
        Generate detailed task breakdown for the campaign
        """
        try:
            tasks = []
            
            # Content creation tasks
            for channel in strategy.distribution_channels:
                if channel == "linkedin":
                    tasks.extend([
                        CampaignTask(
                            task_type="content_creation",
                            platform="linkedin",
                            content_type="post",
                            priority="high",
                            estimated_duration_hours=2,
                            dependencies=[],
                            assigned_agent="ContentAgent"
                        ),
                        CampaignTask(
                            task_type="content_creation",
                            platform="linkedin",
                            content_type="carousel",
                            priority="medium",
                            estimated_duration_hours=4,
                            dependencies=["linkedin_post"],
                            assigned_agent="ImageAgent"
                        )
                    ])
                
                elif channel == "twitter":
                    tasks.extend([
                        CampaignTask(
                            task_type="content_creation",
                            platform="twitter",
                            content_type="thread",
                            priority="high",
                            estimated_duration_hours=3,
                            dependencies=[],
                            assigned_agent="ContentAgent"
                        ),
                        CampaignTask(
                            task_type="content_creation",
                            platform="twitter",
                            content_type="image",
                            priority="medium",
                            estimated_duration_hours=2,
                            dependencies=["twitter_thread"],
                            assigned_agent="ImageAgent"
                        )
                    ])
                
                elif channel == "email":
                    tasks.extend([
                        CampaignTask(
                            task_type="content_creation",
                            platform="email",
                            content_type="newsletter",
                            priority="high",
                            estimated_duration_hours=4,
                            dependencies=[],
                            assigned_agent="ContentAgent"
                        )
                    ])
            
            # Distribution tasks
            for task in tasks:
                if task.task_type == "content_creation":
                    distribution_task = CampaignTask(
                        task_type="distribution",
                        platform=task.platform,
                        content_type=task.content_type,
                        priority=task.priority,
                        estimated_duration_hours=1,
                        dependencies=[f"{task.platform}_{task.content_type}"],
                        assigned_agent="DistributionAgent"
                    )
                    tasks.append(distribution_task)
            
            # Analytics tasks
            analytics_task = CampaignTask(
                task_type="analytics",
                platform="all",
                content_type="report",
                priority="medium",
                estimated_duration_hours=2,
                dependencies=[f"{t.platform}_{t.content_type}" for t in tasks if t.task_type == "distribution"],
                assigned_agent="AnalyticsAgent"
            )
            tasks.append(analytics_task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error generating campaign tasks: {str(e)}")
            raise
    
    async def _save_campaign_to_db(self, blog_id: str, campaign_name: str, 
                                  strategy: CampaignStrategy) -> str:
        """
        Save campaign to database
        """
        try:
            campaign_id = str(uuid.uuid4())
            
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO campaign (id, blog_id, name, status, strategy)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    campaign_id,
                    blog_id,
                    campaign_name,
                    "draft",
                    json.dumps({
                        "target_audience": strategy.target_audience,
                        "key_messages": strategy.key_messages,
                        "distribution_channels": strategy.distribution_channels,
                        "timeline_weeks": strategy.timeline_weeks,
                        "budget_allocation": strategy.budget_allocation,
                        "success_metrics": strategy.success_metrics
                    })
                ))
                conn.commit()
                
                logger.info(f"Campaign saved with ID: {campaign_id}")
                return campaign_id
                
        except Exception as e:
            logger.error(f"Error saving campaign to database: {str(e)}")
            raise
    
    async def _save_tasks_to_db(self, campaign_id: str, tasks: List[CampaignTask]) -> None:
        """
        Save campaign tasks to database
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                for task in tasks:
                    task_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO campaign_task (id, campaign_id, task_type, status, content, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        task_id,
                        campaign_id,
                        task.task_type,
                        "pending",
                        f"{task.platform}_{task.content_type}",
                        json.dumps({
                            "platform": task.platform,
                            "content_type": task.content_type,
                            "priority": task.priority,
                            "estimated_duration_hours": task.estimated_duration_hours,
                            "dependencies": task.dependencies,
                            "assigned_agent": task.assigned_agent
                        })
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(tasks)} tasks for campaign {campaign_id}")
                
        except Exception as e:
            logger.error(f"Error saving tasks to database: {str(e)}")
            raise
    
    async def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get current status of a campaign
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get campaign info
                cur.execute("""
                    SELECT c.name, c.status, c.strategy, 
                           COUNT(ct.id) as total_tasks,
                           COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                    FROM campaign c
                    LEFT JOIN campaign_task ct ON c.id = ct.campaign_id
                    WHERE c.id = %s
                    GROUP BY c.id, c.name, c.status, c.strategy
                """, (campaign_id,))
                
                row = cur.fetchone()
                if not row:
                    raise Exception("Campaign not found")
                
                name, status, strategy_json, total_tasks, completed_tasks = row
                
                # Handle strategy JSON parsing with error handling
                if strategy_json:
                    if isinstance(strategy_json, str):
                        strategy = json.loads(strategy_json)
                    elif isinstance(strategy_json, dict):
                        strategy = strategy_json  # Already parsed
                    else:
                        logger.warning(f"Unexpected strategy_json type: {type(strategy_json)}")
                        strategy = {}
                else:
                    strategy = {}
                
                progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                
                return {
                    "campaign_id": campaign_id,
                    "name": name,
                    "status": status,
                    "progress": progress,
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "strategy": strategy
                }
                
        except Exception as e:
            logger.error(f"Error getting campaign status: {str(e)}")
            raise
    
    async def update_campaign_status(self, campaign_id: str, new_status: str) -> bool:
        """
        Update campaign status
        """
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE campaign SET status = %s WHERE id = %s
                """, (new_status, campaign_id))
                conn.commit()
                
                return cur.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating campaign status: {str(e)}")
            raise
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the campaign manager agent (required by BaseAgent)
        """
        try:
            # For now, return a simple result
            from src.agents.core.base_agent import AgentResult
            return AgentResult(
                success=True,
                data={"message": "CampaignManagerAgent executed successfully"},
                metadata={"agent_type": "campaign_manager"}
            )
        except Exception as e:
            from src.agents.core.base_agent import AgentResult
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="CAMPAIGN_MANAGER_EXECUTION_FAILED"
            )
