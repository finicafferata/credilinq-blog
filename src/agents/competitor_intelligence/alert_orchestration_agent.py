"""
Alert Orchestration Agent for managing competitive intelligence alerts and notifications.
Monitors competitive events, triggers alerts, and manages notification delivery.
"""

import asyncio
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

class AlertDeliveryError(Exception):
    """Exception raised when alert delivery fails."""
    pass

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

from ..core.base_agent import BaseAgent
from .models import (
    Alert, AlertPriority, CompetitorInsight, Trend, ContentGap,
    ContentItem, Competitor, AlertSubscription, TrendStrength
)
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class AlertOrchestrationAgent(BaseAgent):
    """
    Specialized agent for orchestrating competitive intelligence alerts and notifications.
    Monitors competitive events, evaluates alert conditions, and manages delivery.
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..core.base_agent import AgentMetadata, AgentType
        
        metadata = AgentMetadata(
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            name="AlertOrchestrationAgent"
        )
        super().__init__(metadata)
        
        # Initialize AI for alert summarization (lazy loading to avoid requiring API keys at startup)
        self.summarization_llm = None
        
        # Alert configuration
        self.alert_config = {
            "max_alerts_per_hour": 10,
            "max_alerts_per_day": 50,
            "deduplication_window_hours": 24,
            "batch_delivery_interval_minutes": 30,
            "escalation_threshold_hours": 4,
            "alert_retention_days": 90
        }
        
        # Alert condition templates
        self.alert_conditions = {
            "competitor_viral_content": {
                "condition": "engagement_score > viral_threshold",
                "priority": AlertPriority.HIGH,
                "cooldown_hours": 6,
                "description": "Competitor content achieving viral status"
            },
            "trending_topic_emergence": {
                "condition": "trend_strength >= STRONG and growth_rate > 1.0",
                "priority": AlertPriority.MEDIUM,
                "cooldown_hours": 12,
                "description": "New strong trend emerging in industry"
            },
            "content_gap_opportunity": {
                "condition": "opportunity_score > 85 and difficulty_score < 40",
                "priority": AlertPriority.MEDIUM,
                "cooldown_hours": 24,
                "description": "High-value, low-competition content opportunity"
            },
            "competitor_strategy_shift": {
                "condition": "content_pattern_change > 0.7",
                "priority": AlertPriority.HIGH,
                "cooldown_hours": 48,
                "description": "Significant change in competitor content strategy"
            },
            "market_disruption": {
                "condition": "multiple_viral_trends and industry_shift_detected",
                "priority": AlertPriority.CRITICAL,
                "cooldown_hours": 2,
                "description": "Potential market disruption detected"
            }
        }
        
        # Delivery channels
        self.delivery_channels = {
            "email": self._send_email_alert,
            "webhook": self._send_webhook_alert,
            "slack": self._send_slack_alert,
            "dashboard": self._send_dashboard_alert
        }
        
        # Alert cache and state management
        self.alert_cache = {}
        self.delivery_queue = defaultdict(list)
        self.rate_limits = defaultdict(lambda: {"count": 0, "reset_time": datetime.utcnow()})
        self.active_subscriptions = {}
    
    def _get_summarization_llm(self):
        """Lazy initialize the summarization LLM."""
        if self.summarization_llm is None:
            try:
                self.summarization_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI LLM: {e}")
                return None
        return self.summarization_llm
    
    async def monitor_and_alert(
        self,
        competitors: List[Competitor],
        recent_content: Dict[str, List[ContentItem]],
        trends: List[Trend],
        content_gaps: List[ContentGap],
        insights: List[CompetitorInsight],
        subscriptions: List[AlertSubscription]
    ) -> List[Alert]:
        """
        Monitor competitive intelligence data and generate alerts based on conditions.
        """
        
        async with async_performance_tracker("alert_monitoring"):
            self.logger.info(f"Monitoring {len(competitors)} competitors for alert conditions")
            
            # Update subscription cache
            self._update_subscription_cache(subscriptions)
            
            # Evaluate alert conditions
            triggered_alerts = []
            
            # Monitor competitor content for viral activity
            viral_alerts = await self._monitor_viral_content(competitors, recent_content)
            triggered_alerts.extend(viral_alerts)
            
            # Monitor trending topics
            trend_alerts = await self._monitor_trending_topics(trends)
            triggered_alerts.extend(trend_alerts)
            
            # Monitor content gaps
            gap_alerts = await self._monitor_content_gaps(content_gaps)
            triggered_alerts.extend(gap_alerts)
            
            # Monitor strategic insights
            insight_alerts = await self._monitor_strategic_insights(insights)
            triggered_alerts.extend(insight_alerts)
            
            # Deduplicate and prioritize alerts
            final_alerts = await self._process_and_prioritize_alerts(triggered_alerts)
            
            # Queue alerts for delivery
            await self._queue_alerts_for_delivery(final_alerts)
            
            # Process delivery queue
            await self._process_delivery_queue()
            
            # Track metrics
            metrics.increment_counter(
                "alerts.generated",
                tags={
                    "total_alerts": str(len(final_alerts)),
                    "critical_alerts": str(len([a for a in final_alerts if a.priority == AlertPriority.CRITICAL])),
                    "high_alerts": str(len([a for a in final_alerts if a.priority == AlertPriority.HIGH]))
                }
            )
            
            return final_alerts
    
    async def _monitor_viral_content(
        self,
        competitors: List[Competitor],
        recent_content: Dict[str, List[ContentItem]]
    ) -> List[Alert]:
        """Monitor for competitor content achieving viral status."""
        
        viral_alerts = []
        
        for competitor in competitors:
            content_items = recent_content.get(competitor.id, [])
            
            # Calculate viral threshold for this competitor
            engagement_scores = []
            for item in content_items:
                score = self._calculate_engagement_score(item)
                engagement_scores.append(score)
            
            if len(engagement_scores) < 5:  # Need sufficient data
                continue
            
            # Calculate viral threshold (2 standard deviations above mean)
            import statistics
            mean_engagement = statistics.mean(engagement_scores)
            std_engagement = statistics.stdev(engagement_scores) if len(engagement_scores) > 1 else 0
            viral_threshold = mean_engagement + (2 * std_engagement)
            
            # Check for viral content
            for item in content_items:
                engagement_score = self._calculate_engagement_score(item)
                
                if engagement_score >= viral_threshold and engagement_score > 50:  # Absolute minimum
                    
                    # Check cooldown
                    alert_key = f"viral_{competitor.id}_{item.id}"
                    if self._is_in_cooldown(alert_key, 6):  # 6 hour cooldown
                        continue
                    
                    alert = Alert(
                        id=f"viral_alert_{competitor.id}_{int(datetime.utcnow().timestamp())}",
                        alert_type="competitor_viral_content",
                        priority=AlertPriority.HIGH,
                        title=f"🔥 Viral Content Alert: {competitor.name}",
                        message=f"{competitor.name} has viral content '{item.title}' with {engagement_score:.1f} engagement score (threshold: {viral_threshold:.1f})",
                        data={
                            "competitor_id": competitor.id,
                            "competitor_name": competitor.name,
                            "content_id": item.id,
                            "content_title": item.title,
                            "content_url": item.url,
                            "engagement_score": engagement_score,
                            "viral_threshold": viral_threshold,
                            "platform": item.platform.value,
                            "published_at": item.published_at.isoformat()
                        },
                        competitor_ids=[competitor.id],
                        content_ids=[item.id],
                        expires_at=datetime.utcnow() + timedelta(hours=48)
                    )
                    
                    viral_alerts.append(alert)
                    self._set_cooldown(alert_key, 6)
        
        return viral_alerts
    
    async def _monitor_trending_topics(self, trends: List[Trend]) -> List[Alert]:
        """Monitor for emerging trending topics."""
        
        trend_alerts = []
        
        for trend in trends:
            # Check for strong trends with high growth
            if (trend.strength in [TrendStrength.STRONG, TrendStrength.VIRAL] and 
                trend.growth_rate > 1.0):
                
                alert_key = f"trend_{trend.id}"
                if self._is_in_cooldown(alert_key, 12):  # 12 hour cooldown
                    continue
                
                # Determine priority based on trend strength
                if trend.strength == TrendStrength.VIRAL:
                    priority = AlertPriority.CRITICAL
                    emoji = "🚀"
                else:
                    priority = AlertPriority.HIGH
                    emoji = "📈"
                
                alert = Alert(
                    id=f"trend_alert_{trend.id}",
                    alert_type="trending_topic_emergence",
                    priority=priority,
                    title=f"{emoji} Trending Topic Alert: {trend.topic}",
                    message=f"'{trend.topic}' is trending with {trend.strength.value} strength and {trend.growth_rate:.1%} growth rate. Opportunity score: {trend.opportunity_score:.0f}%",
                    data={
                        "trend_id": trend.id,
                        "topic": trend.topic,
                        "strength": trend.strength.value,
                        "growth_rate": trend.growth_rate,
                        "opportunity_score": trend.opportunity_score,
                        "keywords": trend.keywords[:5],
                        "first_detected": trend.first_detected.isoformat()
                    },
                    trend_ids=[trend.id],
                    expires_at=datetime.utcnow() + timedelta(hours=24)
                )
                
                trend_alerts.append(alert)
                self._set_cooldown(alert_key, 12)
        
        return trend_alerts
    
    async def _monitor_content_gaps(self, content_gaps: List[ContentGap]) -> List[Alert]:
        """Monitor for high-value content gap opportunities."""
        
        gap_alerts = []
        
        for gap in content_gaps:
            # Check for high-opportunity, low-difficulty gaps
            if gap.opportunity_score > 85 and gap.difficulty_score < 40:
                
                alert_key = f"gap_{gap.id}"
                if self._is_in_cooldown(alert_key, 24):  # 24 hour cooldown
                    continue
                
                alert = Alert(
                    id=f"gap_alert_{gap.id}",
                    alert_type="content_gap_opportunity",
                    priority=AlertPriority.MEDIUM,
                    title=f"💡 Content Opportunity: {gap.topic}",
                    message=f"High-value content opportunity detected for '{gap.topic}' with {gap.opportunity_score:.0f}% opportunity score and only {gap.difficulty_score:.0f}% difficulty",
                    data={
                        "gap_id": gap.id,
                        "topic": gap.topic,
                        "opportunity_score": gap.opportunity_score,
                        "difficulty_score": gap.difficulty_score,
                        "potential_reach": gap.potential_reach,
                        "missing_content_types": [ct.value for ct in gap.content_types_missing],
                        "suggested_approach": gap.suggested_approach
                    },
                    expires_at=datetime.utcnow() + timedelta(days=7)
                )
                
                gap_alerts.append(alert)
                self._set_cooldown(alert_key, 24)
        
        return gap_alerts
    
    async def _monitor_strategic_insights(self, insights: List[CompetitorInsight]) -> List[Alert]:
        """Monitor for critical strategic insights."""
        
        insight_alerts = []
        
        for insight in insights:
            # Check for high-impact insights
            if (insight.impact_level == "high" and 
                insight.confidence_score > 0.8 and
                insight.insight_type in ["competitive_threat", "strategic_opportunity"]):
                
                alert_key = f"insight_{insight.id}"
                if self._is_in_cooldown(alert_key, 12):  # 12 hour cooldown
                    continue
                
                # Determine priority and emoji
                if insight.insight_type == "competitive_threat":
                    priority = AlertPriority.HIGH
                    emoji = "⚠️"
                else:
                    priority = AlertPriority.MEDIUM
                    emoji = "💰"
                
                alert = Alert(
                    id=f"insight_alert_{insight.id}",
                    alert_type="strategic_insight",
                    priority=priority,
                    title=f"{emoji} Strategic Alert: {insight.title}",
                    message=f"{insight.description} (Confidence: {insight.confidence_score:.0%})",
                    data={
                        "insight_id": insight.id,
                        "insight_type": insight.insight_type,
                        "competitor_id": insight.competitor_id,
                        "confidence_score": insight.confidence_score,
                        "impact_level": insight.impact_level,
                        "supporting_evidence": insight.supporting_evidence,
                        "recommendations": insight.recommendations
                    },
                    competitor_ids=[insight.competitor_id] if insight.competitor_id != "market" else [],
                    expires_at=datetime.utcnow() + timedelta(days=3)
                )
                
                insight_alerts.append(alert)
                self._set_cooldown(alert_key, 12)
        
        return insight_alerts
    
    async def _process_and_prioritize_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Process, deduplicate, and prioritize alerts."""
        
        if not alerts:
            return alerts
        
        # Deduplicate similar alerts
        deduplicated_alerts = await self._deduplicate_alerts(alerts)
        
        # Apply rate limiting
        rate_limited_alerts = self._apply_rate_limits(deduplicated_alerts)
        
        # Sort by priority and timestamp
        priority_order = {
            AlertPriority.CRITICAL: 4,
            AlertPriority.HIGH: 3,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 1
        }
        
        sorted_alerts = sorted(
            rate_limited_alerts,
            key=lambda a: (priority_order.get(a.priority, 0), a.created_at),
            reverse=True
        )
        
        return sorted_alerts
    
    async def _deduplicate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Remove duplicate or very similar alerts."""
        
        deduplicated = []
        seen_signatures = set()
        
        for alert in alerts:
            # Create signature for deduplication
            signature = self._create_alert_signature(alert)
            
            if signature not in seen_signatures:
                deduplicated.append(alert)
                seen_signatures.add(signature)
        
        return deduplicated
    
    def _create_alert_signature(self, alert: Alert) -> str:
        """Create a signature for alert deduplication."""
        
        # Combine alert type, main subject, and key data
        signature_parts = [
            alert.alert_type,
            alert.competitor_ids[0] if alert.competitor_ids else "global",
            alert.data.get("topic", alert.data.get("content_title", ""))[:50]
        ]
        
        return "|".join(str(part) for part in signature_parts)
    
    def _apply_rate_limits(self, alerts: List[Alert]) -> List[Alert]:
        """Apply rate limiting to prevent alert spam."""
        
        now = datetime.utcnow()
        rate_limited = []
        
        for alert in alerts:
            # Check hourly limit
            hour_key = f"hour_{now.hour}"
            if self.rate_limits[hour_key]["reset_time"] < now - timedelta(hours=1):
                self.rate_limits[hour_key] = {"count": 0, "reset_time": now}
            
            if self.rate_limits[hour_key]["count"] >= self.alert_config["max_alerts_per_hour"]:
                # Skip this alert due to rate limiting
                continue
            
            # Check daily limit
            day_key = f"day_{now.date()}"
            if self.rate_limits[day_key]["reset_time"] < now - timedelta(days=1):
                self.rate_limits[day_key] = {"count": 0, "reset_time": now}
            
            if self.rate_limits[day_key]["count"] >= self.alert_config["max_alerts_per_day"]:
                # Skip this alert due to rate limiting
                continue
            
            # Alert passes rate limits
            rate_limited.append(alert)
            self.rate_limits[hour_key]["count"] += 1
            self.rate_limits[day_key]["count"] += 1
        
        return rate_limited
    
    async def _queue_alerts_for_delivery(self, alerts: List[Alert]) -> None:
        """Queue alerts for delivery to subscribers."""
        
        for alert in alerts:
            # Find matching subscriptions
            matching_subscriptions = self._find_matching_subscriptions(alert)
            
            for subscription in matching_subscriptions:
                # Check if alert meets subscriber's priority threshold
                priority_values = {
                    AlertPriority.LOW: 1,
                    AlertPriority.MEDIUM: 2,
                    AlertPriority.HIGH: 3,
                    AlertPriority.CRITICAL: 4
                }
                
                if priority_values.get(alert.priority, 0) >= priority_values.get(subscription.priority_threshold, 0):
                    
                    # Queue for each delivery channel
                    for channel in subscription.delivery_channels:
                        if channel in self.delivery_channels:
                            self.delivery_queue[channel].append({
                                "alert": alert,
                                "subscription": subscription,
                                "queued_at": datetime.utcnow()
                            })
    
    def _find_matching_subscriptions(self, alert: Alert) -> List[AlertSubscription]:
        """Find subscriptions that match the alert criteria."""
        
        matching_subscriptions = []
        
        for subscription in self.active_subscriptions.values():
            # Check alert type match
            if alert.alert_type not in subscription.alert_types and "all" not in subscription.alert_types:
                continue
            
            # Check competitor match (if specified)
            if (subscription.competitors and 
                alert.competitor_ids and 
                not any(comp_id in subscription.competitors for comp_id in alert.competitor_ids)):
                continue
            
            # Check keyword match (if specified)
            if subscription.keywords:
                alert_text = f"{alert.title} {alert.message}".lower()
                if not any(keyword.lower() in alert_text for keyword in subscription.keywords):
                    continue
            
            matching_subscriptions.append(subscription)
        
        return matching_subscriptions
    
    async def _process_delivery_queue(self) -> None:
        """Process queued alerts for delivery."""
        
        for channel, queued_alerts in self.delivery_queue.items():
            if not queued_alerts:
                continue
            
            # Group alerts by subscriber for batching
            subscriber_alerts = defaultdict(list)
            
            for queued_alert in queued_alerts:
                subscriber_id = queued_alert["subscription"].user_id
                subscriber_alerts[subscriber_id].append(queued_alert)
            
            # Deliver alerts
            for subscriber_id, alerts in subscriber_alerts.items():
                try:
                    await self.delivery_channels[channel](alerts)
                    
                    # Mark alerts as sent
                    for queued_alert in alerts:
                        queued_alert["alert"].sent_at = datetime.utcnow()
                    
                except Exception as e:
                    self.logger.error(f"Failed to deliver alerts via {channel}: {str(e)}")
            
            # Clear processed alerts
            self.delivery_queue[channel] = []
    
    # Helper methods
    
    def _calculate_engagement_score(self, content_item: ContentItem) -> float:
        """Calculate engagement score for content item."""
        
        metrics = content_item.engagement_metrics
        
        # Weighted engagement calculation
        score = (
            metrics.get("likes", 0) * 1.0 +
            metrics.get("shares", 0) * 3.0 +
            metrics.get("comments", 0) * 2.0 +
            metrics.get("clicks", 0) * 2.5 +
            metrics.get("views", 0) * 0.1
        )
        
        # Normalize by follower count
        follower_count = metrics.get("follower_count", 1000)
        normalized_score = (score / follower_count) * 100
        
        return normalized_score
    
    def _is_in_cooldown(self, alert_key: str, cooldown_hours: int) -> bool:
        """Check if alert is in cooldown period."""
        
        if alert_key not in self.alert_cache:
            return False
        
        last_alert_time = self.alert_cache[alert_key]
        cooldown_end = last_alert_time + timedelta(hours=cooldown_hours)
        
        return datetime.utcnow() < cooldown_end
    
    def _set_cooldown(self, alert_key: str, cooldown_hours: int) -> None:
        """Set cooldown period for alert type."""
        
        self.alert_cache[alert_key] = datetime.utcnow()
    
    def _update_subscription_cache(self, subscriptions: List[AlertSubscription]) -> None:
        """Update the subscription cache."""
        
        self.active_subscriptions = {
            sub.user_id: sub for sub in subscriptions
        }
    
    # Delivery channel implementations
    
    async def _send_email_alert(self, alerts: List[Dict[str, Any]]) -> None:
        """Send alerts via email."""
        
        if not alerts:
            return
            
        subscription = alerts[0]["subscription"]
        
        # Create email content
        subject = f"CrediLinq Intelligence Alert - {len(alerts)} new alert(s)"
        
        # Group alerts by priority for better organization
        alerts_by_priority = defaultdict(list)
        for alert_data in alerts:
            alert = alert_data["alert"]
            alerts_by_priority[alert.priority].append(alert)
        
        # Generate email body
        body_parts = ["<html><body>"]
        body_parts.append("<h2>CrediLinq Competitive Intelligence Alerts</h2>")
        
        for priority in [AlertPriority.CRITICAL, AlertPriority.HIGH, AlertPriority.MEDIUM, AlertPriority.LOW]:
            priority_alerts = alerts_by_priority.get(priority, [])
            if not priority_alerts:
                continue
                
            body_parts.append(f"<h3>{priority.value.upper()} Priority ({len(priority_alerts)} alerts)</h3>")
            
            for alert in priority_alerts:
                body_parts.append(f"<div style='margin-bottom: 20px; padding: 15px; border-left: 4px solid #007cba;'>")
                body_parts.append(f"<h4>{alert.title}</h4>")
                body_parts.append(f"<p>{alert.message}</p>")
                body_parts.append(f"<small>Generated: {alert.created_at.strftime('%Y-%m-%d %H:%M UTC')}</small>")
                body_parts.append("</div>")
        
        body_parts.append("<hr>")
        body_parts.append("<p><small>This is an automated alert from CrediLinq Competitive Intelligence. To manage your alert preferences, visit your dashboard.</small></p>")
        body_parts.append("</body></html>")
        
        email_body = "\n".join(body_parts)
        
        # Log email (in real implementation, would send via SMTP)
        self.logger.info(f"Email alert prepared for {subscription.user_id}: {len(alerts)} alerts")
        
        # IMPLEMENTATION NOTE: Email sending integration
        # In production, this would integrate with:
        # - SMTP server configuration (settings.smtp_host, smtp_port, etc.)
        # - Email service providers (SendGrid, AWS SES, etc.)
        # - Email templates and personalization
        
        # Mock implementation for development
        try:
            # Future implementation:
            # await self._send_smtp_email(subscription.email, subject, html_content)
            self.logger.info(f"✅ Email alert mock-sent to {subscription.user_id}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            raise AlertDeliveryError(f"Email delivery failed: {e}")
    
    async def _send_webhook_alert(self, alerts: List[Dict[str, Any]]) -> None:
        """Send alerts via webhook."""
        
        if not alerts:
            return
        
        subscription = alerts[0]["subscription"]
        
        # Prepare webhook payload
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "subscriber_id": subscription.user_id,
            "alert_count": len(alerts),
            "alerts": [
                {
                    "id": alert_data["alert"].id,
                    "type": alert_data["alert"].alert_type,
                    "priority": alert_data["alert"].priority.value,
                    "title": alert_data["alert"].title,
                    "message": alert_data["alert"].message,
                    "data": alert_data["alert"].data,
                    "created_at": alert_data["alert"].created_at.isoformat()
                }
                for alert_data in alerts
            ]
        }
        
        self.logger.info(f"Webhook alert prepared for {subscription.user_id}: {len(alerts)} alerts")
        
        # IMPLEMENTATION NOTE: Webhook sending integration
        # In production, this would:
        # - Make HTTP POST request to subscriber's webhook URL
        # - Include proper authentication headers
        # - Handle retries and delivery failures
        # - Validate webhook response
        
        # Mock implementation for development
        try:
            # Future implementation:
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         subscription.webhook_url,
            #         json=payload,
            #         headers={"Content-Type": "application/json"},
            #         timeout=30
            #     )
            #     response.raise_for_status()
            
            self.logger.info(f"✅ Webhook alert mock-sent to {subscription.user_id}")
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            raise AlertDeliveryError(f"Webhook delivery failed: {e}")
    
    async def _send_slack_alert(self, alerts: List[Dict[str, Any]]) -> None:
        """Send alerts via Slack."""
        
        if not alerts:
            return
        
        subscription = alerts[0]["subscription"]
        
        # Create Slack message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔔 {len(alerts)} New Intelligence Alert(s)"
                }
            }
        ]
        
        for alert_data in alerts[:5]:  # Limit to 5 alerts per message
            alert = alert_data["alert"]
            
            # Priority emoji
            priority_emoji = {
                AlertPriority.CRITICAL: "🚨",
                AlertPriority.HIGH: "⚠️",
                AlertPriority.MEDIUM: "ℹ️",
                AlertPriority.LOW: "💡"
            }
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{priority_emoji.get(alert.priority, '📢')} *{alert.title}*\n{alert.message}"
                }
            })
        
        if len(alerts) > 5:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_And {len(alerts) - 5} more alerts..._"
                }
            })
        
        self.logger.info(f"Slack alert prepared for {subscription.user_id}: {len(alerts)} alerts")
        
        # IMPLEMENTATION NOTE: Slack API integration
        # In production, this would integrate with Slack Bot API:
        # - Use Slack SDK (slack-sdk) 
        # - Send formatted messages to channels or DMs
        # - Include proper authentication tokens
        # - Handle rate limiting and retries
        
        # Mock implementation for development
        try:
            # Future implementation:
            # from slack_sdk.web.async_client import AsyncWebClient
            # slack_client = AsyncWebClient(token=settings.slack_bot_token)
            # await slack_client.chat_postMessage(
            #     channel=subscription.slack_channel,
            #     text=f"🚨 {len(alerts)} new alerts",
            #     blocks=slack_blocks
            # )
            
            self.logger.info(f"✅ Slack alert mock-sent to {subscription.user_id}")
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            raise AlertDeliveryError(f"Slack delivery failed: {e}")
    
    async def _send_dashboard_alert(self, alerts: List[Dict[str, Any]]) -> None:
        """Send alerts to dashboard (in-app notifications)."""
        
        if not alerts:
            return
        
        subscription = alerts[0]["subscription"]
        
        # Store alerts in database for dashboard display
        for alert_data in alerts:
            alert = alert_data["alert"]
            
            # IMPLEMENTATION NOTE: Database notifications integration
            # In production, this would:
            # - Insert into notifications table in database
            # - Track read/unread status
            # - Set expiration dates for cleanup
            # - Support notification preferences
            
            # Mock implementation for development
            try:
                # Future implementation:
                # notification = {
                #     "id": str(uuid.uuid4()),
                #     "user_id": subscription.user_id,
                #     "alert_id": alert.id,
                #     "title": alert.title,
                #     "message": alert.message,
                #     "type": alert.alert_type,
                #     "priority": alert.priority.value,
                #     "read": False,
                #     "created_at": datetime.utcnow(),
                #     "expires_at": datetime.utcnow() + timedelta(days=30)
                # }
                # await secure_db.execute_query(
                #     "INSERT INTO notifications (...) VALUES (...)",
                #     notification
                # )
                
                self.logger.info(f"✅ Dashboard notification mock-stored for {subscription.user_id}: {alert.title}")
            except Exception as e:
                self.logger.error(f"Failed to store dashboard notification: {e}")
        
        # IMPLEMENTATION NOTE: WebSocket real-time notifications
        # In production, this would send real-time updates via WebSocket
        # to users currently online in the dashboard
    
    async def get_alert_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get alert statistics for the specified period."""
        
        # TODO: In real implementation, this would query database
        # For now, return mock statistics
        
        return {
            "period_days": days,
            "total_alerts": 145,
            "alerts_by_priority": {
                "critical": 8,
                "high": 32,
                "medium": 67,
                "low": 38
            },
            "alerts_by_type": {
                "competitor_viral_content": 23,
                "trending_topic_emergence": 45,
                "content_gap_opportunity": 31,
                "strategic_insight": 46
            },
            "delivery_success_rate": 0.94,
            "average_delivery_time_seconds": 45,
            "top_alert_triggers": [
                {"competitor": "TechFlow", "alerts": 18},
                {"competitor": "InnovateCorp", "alerts": 15},
                {"topic": "AI Marketing", "alerts": 12}
            ]
        }
    
    async def manage_subscription(
        self,
        user_id: str,
        subscription_data: Dict[str, Any]
    ) -> AlertSubscription:
        """Create or update alert subscription."""
        
        subscription = AlertSubscription(
            user_id=user_id,
            alert_types=subscription_data.get("alert_types", ["all"]),
            competitors=subscription_data.get("competitors", []),
            keywords=subscription_data.get("keywords", []),
            priority_threshold=AlertPriority(subscription_data.get("priority_threshold", "medium")),
            delivery_channels=subscription_data.get("delivery_channels", ["email"]),
            frequency_limit=subscription_data.get("frequency_limit", 10)
        )
        
        # Update subscription cache
        self.active_subscriptions[user_id] = subscription
        
        self.logger.info(f"Updated alert subscription for user {user_id}")
        
        return subscription
    
    def execute(self, input_data, context=None, **kwargs):
        """
        Execute the alert orchestration agent's main functionality.
        Routes to appropriate monitoring method based on input.
        """
        return {
            "status": "ready",
            "agent_type": "alert_orchestration",
            "available_operations": [
                "monitor_and_alert",
                "get_alert_statistics",
                "create_or_update_subscription"
            ]
        }