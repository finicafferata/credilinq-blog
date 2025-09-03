"""
Agent Insights Service - Real data from agent_performance and agent_decisions tables.
Replaces all mock data with actual database insights.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

from ..config.database import db_config

logger = logging.getLogger(__name__)


@dataclass
class AgentInsight:
    """Real agent insight with performance metrics and reasoning."""
    agent_name: str
    agent_type: str
    performance: Dict[str, Any] = field(default_factory=dict)
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    cost_analysis: Dict[str, float] = field(default_factory=dict)


class AgentInsightsService:
    """Service to extract real insights from agent performance and decisions."""
    
    def __init__(self):
        self.db_config = db_config
        
    async def get_campaign_agent_insights(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get comprehensive agent insights for a campaign from real data.
        
        Args:
            campaign_id: Campaign ID to analyze
            
        Returns:
            Dict containing real agent performance insights
        """
        try:
            with self.db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get agent performance data for this campaign (enhanced for Gemini)
                performance_query = """
                    SELECT 
                        ap.agent_name,
                        ap.agent_type,
                        COUNT(*) as total_executions,
                        AVG(ap.duration) as avg_duration_ms,
                        COUNT(CASE WHEN ap.status = 'success' THEN 1 END) as successful_executions,
                        SUM(ap.input_tokens) as total_input_tokens,
                        SUM(ap.output_tokens) as total_output_tokens,
                        SUM(ap.cost) as total_cost,
                        MAX(ap.start_time) as last_activity,
                        MIN(ap.start_time) as first_activity,
                        ap.model_used,
                        AVG(CASE WHEN ap.response_metadata::json->>'usage_metrics' IS NOT NULL 
                            THEN (ap.response_metadata::json->'usage_metrics'->>'total_token_count')::int 
                            ELSE NULL END) as avg_total_tokens,
                        STRING_AGG(DISTINCT ap.model_used, ', ') as models_used
                    FROM agent_performance ap
                    WHERE ap.campaign_id = %s
                    GROUP BY ap.agent_name, ap.agent_type, ap.model_used
                    ORDER BY total_executions DESC
                """
                
                cur.execute(performance_query, (campaign_id,))
                performance_rows = cur.fetchall()
                
                agent_insights = []
                total_cost = 0
                total_executions = 0
                
                for row in performance_rows:
                    (agent_name, agent_type, executions, avg_duration, successful, 
                     input_tokens, output_tokens, cost, last_activity, first_activity,
                     model_used, avg_total_tokens, models_used) = row
                    
                    # Calculate metrics
                    success_rate = (successful / executions) * 100 if executions > 0 else 0
                    avg_duration = float(avg_duration) if avg_duration else 0
                    cost = float(cost) if cost else 0
                    
                    total_cost += cost
                    total_executions += executions
                    
                    # Get recent decisions for this agent
                    decisions_query = """
                        SELECT 
                            ad.decision_point,
                            ad.reasoning,
                            ad.confidence_score,
                            ad.execution_time,
                            ad.timestamp
                        FROM agent_decisions ad
                        JOIN agent_performance ap ON ad.performance_id = ap.id
                        WHERE ap.campaign_id = %s 
                        AND ap.agent_name = %s 
                        AND ap.agent_type = %s
                        ORDER BY ad.timestamp DESC
                        LIMIT 5
                    """
                    
                    cur.execute(decisions_query, (campaign_id, agent_name, agent_type))
                    decision_rows = cur.fetchall()
                    
                    recent_decisions = []
                    avg_confidence = 0
                    
                    for decision_row in decision_rows:
                        (decision_point, reasoning, confidence, exec_time, timestamp) = decision_row
                        recent_decisions.append({
                            "decision": decision_point,
                            "reasoning": reasoning,
                            "confidence": float(confidence) if confidence else 0,
                            "execution_time_ms": exec_time,
                            "timestamp": timestamp.isoformat() if timestamp else None
                        })
                        
                        if confidence:
                            avg_confidence += float(confidence)
                    
                    avg_confidence = avg_confidence / len(recent_decisions) if recent_decisions else 0
                    
                    agent_insights.append({
                        "agent_name": agent_name,
                        "agent_type": agent_type,
                        "performance": {
                            "total_executions": executions,
                            "success_rate": round(success_rate, 2),
                            "avg_duration_ms": round(avg_duration, 2),
                            "total_cost": round(cost, 4),
                            "input_tokens": input_tokens or 0,
                            "output_tokens": output_tokens or 0,
                            "avg_total_tokens": round(avg_total_tokens, 2) if avg_total_tokens else 0,
                            "last_activity": last_activity.isoformat() if last_activity else None,
                            "uptime_hours": self._calculate_uptime_hours(first_activity, last_activity)
                        },
                        "gemini_metrics": {
                            "primary_model": model_used or "gemini-1.5-flash",
                            "models_used": models_used.split(', ') if models_used else [model_used or "gemini-1.5-flash"],
                            "avg_tokens_per_execution": round((input_tokens + output_tokens) / executions, 2) if executions > 0 else 0,
                            "cost_per_1k_tokens": round(cost / ((input_tokens + output_tokens) / 1000), 4) if (input_tokens + output_tokens) > 0 else 0
                        },
                        "quality_metrics": {
                            "average_confidence": round(avg_confidence, 2),
                            "decision_count": len(recent_decisions),
                            "reasoning_quality": self._assess_reasoning_quality(recent_decisions)
                        },
                        "recent_decisions": recent_decisions,
                        "cost_efficiency": {
                            "cost_per_execution": round(cost / executions, 4) if executions > 0 else 0,
                            "tokens_per_dollar": round((input_tokens + output_tokens) / cost, 2) if cost > 0 else 0,
                            "efficiency_score": self._calculate_efficiency_score(cost, executions, avg_duration, success_rate)
                        }
                    })
                
                # Get keyword extraction and readability metrics
                seo_metrics = await self._get_seo_geo_metrics(campaign_id, cur)
                
                return {
                    "campaign_id": campaign_id,
                    "agent_insights": agent_insights,
                    "summary": {
                        "total_agents": len(agent_insights),
                        "total_executions": total_executions,
                        "total_cost": round(total_cost, 4),
                        "avg_success_rate": round(
                            sum(insight["performance"]["success_rate"] for insight in agent_insights) / len(agent_insights), 2
                        ) if agent_insights else 0,
                        "cost_per_execution": round(total_cost / total_executions, 4) if total_executions > 0 else 0
                    },
                    "seo_geo_analysis": seo_metrics,
                    "generated_at": datetime.utcnow().isoformat(),
                    "data_source": "real_agent_performance_tables"
                }
                
        except Exception as e:
            logger.error(f"Error getting agent insights for campaign {campaign_id}: {e}")
            # Return empty structure instead of mock data
            return {
                "campaign_id": campaign_id,
                "agent_insights": [],
                "summary": {},
                "seo_geo_analysis": {},
                "error": str(e),
                "data_source": "error_fallback"
            }
    
    async def get_agent_performance_details(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed performance data for a specific agent from real data.
        
        Args:
            agent_id: Agent identifier (name or type)
            
        Returns:
            Dict containing real agent performance details
        """
        try:
            with self.db_config.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Get agent performance data
                cur.execute("""
                    SELECT 
                        agent_name,
                        agent_type,
                        COUNT(*) as total_executions,
                        AVG(duration) as avg_duration,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_executions,
                        SUM(input_tokens) as total_input_tokens,
                        SUM(output_tokens) as total_output_tokens,
                        SUM(cost) as total_cost,
                        MAX(start_time) as last_activity,
                        MIN(start_time) as first_activity
                    FROM agent_performance
                    WHERE agent_name LIKE %s OR agent_type LIKE %s
                    GROUP BY agent_name, agent_type
                """, (f"%{agent_id}%", f"%{agent_id}%"))
                
                row = cur.fetchone()
                if not row:
                    return {
                        "agent_id": agent_id,
                        "error": "Agent not found in performance data",
                        "data_source": "no_data_available"
                    }
                
                (agent_name, agent_type, total, avg_duration, successful, 
                 input_tokens, output_tokens, cost, last_activity, first_activity) = row
                
                success_rate = (successful / total * 100) if total > 0 else 0
                
                # Get recent tasks
                cur.execute("""
                    SELECT 
                        execution_id,
                        start_time,
                        end_time,
                        status,
                        duration,
                        blog_post_id,
                        campaign_id,
                        cost,
                        input_tokens,
                        output_tokens
                    FROM agent_performance
                    WHERE agent_name = %s OR agent_type = %s
                    ORDER BY start_time DESC
                    LIMIT 10
                """, (agent_name, agent_type))
                
                task_rows = cur.fetchall()
                recent_tasks = []
                
                for task_row in task_rows:
                    (execution_id, start_time, end_time, status, duration, 
                     blog_post_id, campaign_id, task_cost, task_input_tokens, task_output_tokens) = task_row
                    
                    recent_tasks.append({
                        "execution_id": execution_id,
                        "start_time": start_time.isoformat() + 'Z' if start_time else None,
                        "end_time": end_time.isoformat() + 'Z' if end_time else None,
                        "status": status,
                        "duration_ms": duration,
                        "blog_post_id": str(blog_post_id) if blog_post_id else None,
                        "campaign_id": str(campaign_id) if campaign_id else None,
                        "cost": float(task_cost) if task_cost else 0,
                        "tokens": {
                            "input": task_input_tokens or 0,
                            "output": task_output_tokens or 0,
                            "total": (task_input_tokens or 0) + (task_output_tokens or 0)
                        }
                    })
                
                return {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_type": agent_type,
                    "performance": {
                        "total_executions": total,
                        "success_rate": round(success_rate, 2),
                        "avg_duration_ms": round(float(avg_duration), 2) if avg_duration else 0,
                        "total_cost": round(float(cost), 4) if cost else 0,
                        "total_tokens": {
                            "input": input_tokens or 0,
                            "output": output_tokens or 0,
                            "total": (input_tokens or 0) + (output_tokens or 0)
                        },
                        "uptime_hours": self._calculate_uptime_hours(first_activity, last_activity),
                        "last_activity": last_activity.isoformat() if last_activity else None
                    },
                    "recent_tasks": recent_tasks,
                    "capabilities": self._get_agent_capabilities(agent_type),
                    "data_source": "real_agent_performance_table"
                }
                
        except Exception as e:
            logger.error(f"Error getting agent performance for {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "error": str(e),
                "data_source": "error_fallback"
            }
    
    async def _get_seo_geo_metrics(self, campaign_id: str, cursor) -> Dict[str, Any]:
        """Extract SEO/GEO analysis from agent decisions."""
        try:
            # Get SEO and GEO related decisions
            cursor.execute("""
                SELECT 
                    ad.decision_point,
                    ad.reasoning,
                    ad.confidence_score,
                    ad.alternatives_considered,
                    ad.input_data,
                    ad.output_data,
                    ap.agent_type
                FROM agent_decisions ad
                JOIN agent_performance ap ON ad.performance_id = ap.id
                WHERE ap.campaign_id = %s 
                AND (ap.agent_type IN ('seo', 'geo_analysis') OR ad.decision_point ILIKE '%seo%' OR ad.decision_point ILIKE '%geo%')
                ORDER BY ad.timestamp DESC
            """, (campaign_id,))
            
            seo_geo_data = cursor.fetchall()
            
            if not seo_geo_data:
                return {"message": "No SEO/GEO analysis data available for this campaign"}
            
            seo_insights = []
            geo_insights = []
            keyword_data = []
            readability_scores = []
            
            for row in seo_geo_data:
                (decision_point, reasoning, confidence, alternatives, input_data, output_data, agent_type) = row
                
                insight = {
                    "decision": decision_point,
                    "reasoning": reasoning,
                    "confidence": float(confidence) if confidence else 0,
                    "agent_type": agent_type
                }
                
                # Parse output data for specific metrics
                if output_data:
                    try:
                        output = json.loads(output_data) if isinstance(output_data, str) else output_data
                        
                        # Extract keyword data
                        if "keywords" in output:
                            keyword_data.extend(output["keywords"])
                        
                        # Extract readability scores
                        if "readability_score" in output:
                            readability_scores.append(output["readability_score"])
                        
                        insight["metrics"] = output
                        
                    except json.JSONDecodeError:
                        pass
                
                if agent_type == 'seo' or 'seo' in decision_point.lower():
                    seo_insights.append(insight)
                elif agent_type == 'geo_analysis' or 'geo' in decision_point.lower():
                    geo_insights.append(insight)
            
            return {
                "seo_analysis": {
                    "insights": seo_insights,
                    "avg_confidence": round(
                        sum(i["confidence"] for i in seo_insights) / len(seo_insights), 2
                    ) if seo_insights else 0
                },
                "geo_analysis": {
                    "insights": geo_insights,
                    "avg_confidence": round(
                        sum(i["confidence"] for i in geo_insights) / len(geo_insights), 2
                    ) if geo_insights else 0
                },
                "keyword_extraction": {
                    "keywords": list(set(keyword_data)),
                    "total_keywords": len(set(keyword_data))
                },
                "readability_metrics": {
                    "scores": readability_scores,
                    "avg_score": round(sum(readability_scores) / len(readability_scores), 2) if readability_scores else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting SEO/GEO metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_uptime_hours(self, first_activity: datetime, last_activity: datetime) -> float:
        """Calculate agent uptime in hours."""
        if not first_activity or not last_activity:
            return 0
        
        delta = last_activity - first_activity
        return round(delta.total_seconds() / 3600, 2)
    
    def _assess_reasoning_quality(self, decisions: List[Dict]) -> float:
        """Assess the quality of agent reasoning based on decision data."""
        if not decisions:
            return 0
        
        quality_score = 0
        for decision in decisions:
            # Base score from confidence
            confidence_score = decision.get("confidence", 0) * 0.4
            
            # Reasoning length and detail (simple heuristic)
            reasoning = decision.get("reasoning", "")
            if reasoning:
                reasoning_score = min(len(reasoning.split()), 50) / 50 * 0.3
            else:
                reasoning_score = 0
            
            # Execution time bonus (faster = better, within reason)
            exec_time = decision.get("execution_time_ms", 0)
            if exec_time > 0 and exec_time < 5000:  # Under 5 seconds is good
                time_score = 0.3
            elif exec_time < 15000:  # Under 15 seconds is okay
                time_score = 0.15
            else:
                time_score = 0
            
            quality_score += confidence_score + reasoning_score + time_score
        
        return round(quality_score / len(decisions), 2)
    
    def _calculate_efficiency_score(self, cost: float, executions: int, avg_duration: float, success_rate: float) -> float:
        """Calculate an overall efficiency score for the agent (0-100)."""
        try:
            # Normalize metrics to 0-1 scale
            cost_score = max(0, 1 - (cost / max(cost, 0.1)))  # Lower cost = higher score
            speed_score = max(0, 1 - (avg_duration / 30000))  # Under 30s = good
            success_score = success_rate / 100  # Already 0-1
            
            # Weight the components
            efficiency_score = (
                cost_score * 0.3 +      # 30% cost efficiency
                speed_score * 0.3 +     # 30% speed
                success_score * 0.4     # 40% success rate
            ) * 100
            
            return round(min(100, max(0, efficiency_score)), 2)
            
        except (ZeroDivisionError, TypeError):
            return 50.0  # Neutral score if calculation fails
    
    def _get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get agent capabilities based on type."""
        capabilities_map = {
            "planner": ["content_planning", "strategy_development", "timeline_creation"],
            "researcher": ["web_research", "data_gathering", "fact_checking"],
            "writer": ["content_creation", "copywriting", "adaptation"],
            "editor": ["content_review", "quality_assurance", "proofreading"],
            "seo": ["keyword_optimization", "search_ranking", "meta_tags"],
            "geo_analysis": ["ai_search_optimization", "citation_analysis", "trustworthiness"],
            "social_media": ["platform_adaptation", "engagement_optimization", "hashtag_strategy"],
            "image_prompt": ["visual_content_planning", "image_prompt_generation", "style_recommendations"],
            "video_prompt": ["video_content_planning", "storyboard_creation", "platform_optimization"],
            "distribution": ["multi_channel_publishing", "scheduling", "performance_tracking"],
            "content_repurposer": ["format_adaptation", "channel_optimization", "message_consistency"]
        }
        
        return capabilities_map.get(agent_type, ["general_task_execution", "content_processing"])


# Global instance
agent_insights_service = AgentInsightsService()