"""
External Integrations Service
Provides integration with external tools and platforms including Slack, Microsoft Teams,
email notifications, webhooks, and third-party APIs for seamless workflow integration.
"""

import asyncio
import json
import aiohttp
import smtplib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
import logging

from ..config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    ZAPIER = "zapier"
    IFTTT = "ifttt"

class MessagePriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class EventType(Enum):
    NEW_TREND = "new_trend"
    HIGH_PRIORITY_ALERT = "high_priority_alert"
    COMPETITOR_UPDATE = "competitor_update"
    REPORT_GENERATED = "report_generated"
    SYSTEM_HEALTH = "system_health"
    CUSTOM = "custom"

@dataclass
class IntegrationConfig:
    """Configuration for external integration."""
    integration_type: IntegrationType
    name: str
    enabled: bool = True
    webhook_url: str = None
    api_token: str = None
    channel: str = None
    email_settings: Dict[str, str] = None
    custom_settings: Dict[str, Any] = None
    event_filters: List[EventType] = None
    priority_threshold: MessagePriority = MessagePriority.NORMAL

@dataclass
class NotificationMessage:
    """Notification message structure."""
    title: str
    content: str
    event_type: EventType
    priority: MessagePriority
    data: Dict[str, Any] = None
    attachments: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class IntegrationResult:
    """Result of integration attempt."""
    success: bool
    integration_name: str
    message_id: str = None
    error: str = None
    response_data: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class ExternalIntegrationsService:
    """Service for managing external integrations and notifications."""
    
    def __init__(self):
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.session: aiohttp.ClientSession = None
    
    async def _initialize_session(self):
        """Initialize HTTP session for external API calls."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'CrediLinq-CompetitorIntelligence/1.0'}
            )
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    def register_integration(self, config: IntegrationConfig) -> bool:
        """Register a new external integration."""
        try:
            self.integrations[config.name] = config
            logger.info(f"Registered {config.integration_type.value} integration: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register integration {config.name}: {e}")
            return False
    
    def remove_integration(self, name: str) -> bool:
        """Remove an integration."""
        if name in self.integrations:
            del self.integrations[name]
            logger.info(f"Removed integration: {name}")
            return True
        return False
    
    async def send_notification(
        self, 
        message: NotificationMessage,
        integration_names: List[str] = None
    ) -> List[IntegrationResult]:
        """Send notification through specified integrations or all enabled ones."""
        if integration_names is None:
            # Send to all enabled integrations that match event type and priority
            target_integrations = [
                config for config in self.integrations.values()
                if config.enabled and self._should_send_message(config, message)
            ]
        else:
            # Send to specified integrations
            target_integrations = [
                self.integrations[name] for name in integration_names
                if name in self.integrations and self.integrations[name].enabled
            ]
        
        # Send notifications in parallel
        tasks = [
            self._send_to_integration(config, message)
            for config in target_integrations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(IntegrationResult(
                    success=False,
                    integration_name=target_integrations[i].name,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def _should_send_message(self, config: IntegrationConfig, message: NotificationMessage) -> bool:
        """Check if message should be sent to integration based on filters."""
        # Check event type filter
        if config.event_filters and message.event_type not in config.event_filters:
            return False
        
        # Check priority threshold
        priority_levels = {
            MessagePriority.LOW: 0,
            MessagePriority.NORMAL: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.URGENT: 3
        }
        
        if priority_levels[message.priority] < priority_levels[config.priority_threshold]:
            return False
        
        return True
    
    async def _send_to_integration(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send message to specific integration."""
        try:
            if config.integration_type == IntegrationType.SLACK:
                return await self._send_slack_message(config, message)
            elif config.integration_type == IntegrationType.TEAMS:
                return await self._send_teams_message(config, message)
            elif config.integration_type == IntegrationType.EMAIL:
                return await self._send_email_message(config, message)
            elif config.integration_type == IntegrationType.WEBHOOK:
                return await self._send_webhook_message(config, message)
            elif config.integration_type == IntegrationType.DISCORD:
                return await self._send_discord_message(config, message)
            elif config.integration_type == IntegrationType.TELEGRAM:
                return await self._send_telegram_message(config, message)
            elif config.integration_type == IntegrationType.ZAPIER:
                return await self._send_zapier_webhook(config, message)
            elif config.integration_type == IntegrationType.IFTTT:
                return await self._send_ifttt_webhook(config, message)
            else:
                return IntegrationResult(
                    success=False,
                    integration_name=config.name,
                    error=f"Unsupported integration type: {config.integration_type}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                integration_name=config.name,
                error=str(e)
            )
    
    async def _send_slack_message(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send message to Slack."""
        if not config.webhook_url:
            raise ValueError("Slack webhook URL is required")
        
        # Format message for Slack
        color_map = {
            MessagePriority.LOW: "#36a64f",
            MessagePriority.NORMAL: "#2196F3",
            MessagePriority.HIGH: "#ff9800",
            MessagePriority.URGENT: "#f44336"
        }
        
        payload = {
            "text": f"🎯 {message.title}",
            "attachments": [
                {
                    "color": color_map.get(message.priority, "#2196F3"),
                    "fields": [
                        {
                            "title": "Event Type",
                            "value": message.event_type.value.replace('_', ' ').title(),
                            "short": True
                        },
                        {
                            "title": "Priority",
                            "value": message.priority.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Details",
                            "value": message.content,
                            "short": False
                        }
                    ],
                    "footer": "CrediLinq Competitor Intelligence",
                    "ts": int(message.timestamp.timestamp())
                }
            ]
        }
        
        if config.channel:
            payload["channel"] = config.channel
        
        async with self.session.post(config.webhook_url, json=payload) as response:
            if response.status == 200:
                return IntegrationResult(
                    success=True,
                    integration_name=config.name,
                    message_id=f"slack_{int(message.timestamp.timestamp())}",
                    response_data={"status": response.status}
                )
            else:
                error_text = await response.text()
                raise Exception(f"Slack API error {response.status}: {error_text}")
    
    async def _send_teams_message(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send message to Microsoft Teams."""
        if not config.webhook_url:
            raise ValueError("Teams webhook URL is required")
        
        # Format message for Teams
        color_map = {
            MessagePriority.LOW: "good",
            MessagePriority.NORMAL: "default",
            MessagePriority.HIGH: "warning",
            MessagePriority.URGENT: "attention"
        }
        
        payload = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": message.title,
            "themeColor": color_map.get(message.priority, "default"),
            "sections": [
                {
                    "activityTitle": f"🎯 {message.title}",
                    "activitySubtitle": f"Priority: {message.priority.value.upper()}",
                    "facts": [
                        {
                            "name": "Event Type",
                            "value": message.event_type.value.replace('_', ' ').title()
                        },
                        {
                            "name": "Time",
                            "value": message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                        }
                    ],
                    "text": message.content
                }
            ]
        }
        
        async with self.session.post(config.webhook_url, json=payload) as response:
            if response.status == 200:
                return IntegrationResult(
                    success=True,
                    integration_name=config.name,
                    message_id=f"teams_{int(message.timestamp.timestamp())}",
                    response_data={"status": response.status}
                )
            else:
                error_text = await response.text()
                raise Exception(f"Teams API error {response.status}: {error_text}")
    
    async def _send_email_message(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send email notification."""
        if not config.email_settings:
            raise ValueError("Email settings are required")
        
        email_config = config.email_settings
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config.get('from_email', 'noreply@credilinq.com')
        msg['To'] = email_config.get('to_email')
        msg['Subject'] = f"[{message.priority.value.upper()}] {message.title}"
        
        # Create HTML body (precompute values to avoid backslashes in f-string expressions)
        event_type_title = message.event_type.value.replace('_', ' ').title()
        priority_upper = message.priority.value.upper()
        time_str = message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        details_html = (message.content or '').replace('\n', '<br>')

        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background-color: #f0f2f5; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #333;">{message.title}</h2>
                    <p><strong>Event Type:</strong> {event_type_title}</p>
                    <p><strong>Priority:</strong> {priority_upper}</p>
                    <p><strong>Time:</strong> {time_str}</p>
                </div>
                <div style="margin-top: 20px;">
                    <h3>Details:</h3>
                    <p>{details_html}</p>
                </div>
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This notification was sent by CrediLinq Competitor Intelligence System
                </p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Add attachments if any
        if message.attachments:
            for attachment_path in message.attachments:
                try:
                    with open(attachment_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {attachment_path.split("/")[-1]}'
                        )
                        msg.attach(part)
                except Exception as e:
                    logger.warning(f"Failed to attach file {attachment_path}: {e}")
        
        # Send email
        try:
            # Use async SMTP (simplified for demo - in production use aiosmtplib)
            import asyncio
            
            def send_email():
                context = ssl.create_default_context()
                server = smtplib.SMTP(
                    email_config.get('smtp_server', 'smtp.gmail.com'),
                    email_config.get('smtp_port', 587)
                )
                server.starttls(context=context)
                server.login(
                    email_config.get('username'),
                    email_config.get('password')
                )
                server.sendmail(
                    email_config.get('from_email'),
                    email_config.get('to_email'),
                    msg.as_string()
                )
                server.quit()
                return True
            
            # Run in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, send_email)
            
            return IntegrationResult(
                success=True,
                integration_name=config.name,
                message_id=f"email_{int(message.timestamp.timestamp())}",
                response_data={"to": email_config.get('to_email')}
            )
        
        except Exception as e:
            raise Exception(f"Email send failed: {e}")
    
    async def _send_webhook_message(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send generic webhook message."""
        if not config.webhook_url:
            raise ValueError("Webhook URL is required")
        
        payload = {
            "title": message.title,
            "content": message.content,
            "event_type": message.event_type.value,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "data": message.data or {}
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-CrediLinq-Event": message.event_type.value,
            "X-CrediLinq-Priority": message.priority.value
        }
        
        # Add custom headers if specified
        if config.custom_settings and 'headers' in config.custom_settings:
            headers.update(config.custom_settings['headers'])
        
        async with self.session.post(
            config.webhook_url, 
            json=payload, 
            headers=headers
        ) as response:
            response_text = await response.text()
            
            if 200 <= response.status < 300:
                return IntegrationResult(
                    success=True,
                    integration_name=config.name,
                    message_id=f"webhook_{int(message.timestamp.timestamp())}",
                    response_data={
                        "status": response.status,
                        "response": response_text[:500]  # Truncate long responses
                    }
                )
            else:
                raise Exception(f"Webhook error {response.status}: {response_text}")
    
    async def _send_discord_message(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send message to Discord."""
        if not config.webhook_url:
            raise ValueError("Discord webhook URL is required")
        
        # Format message for Discord
        color_map = {
            MessagePriority.LOW: 0x36a64f,
            MessagePriority.NORMAL: 0x2196F3,
            MessagePriority.HIGH: 0xff9800,
            MessagePriority.URGENT: 0xf44336
        }
        
        payload = {
            "embeds": [
                {
                    "title": f"🎯 {message.title}",
                    "description": message.content,
                    "color": color_map.get(message.priority, 0x2196F3),
                    "fields": [
                        {
                            "name": "Event Type",
                            "value": message.event_type.value.replace('_', ' ').title(),
                            "inline": True
                        },
                        {
                            "name": "Priority",
                            "value": message.priority.value.upper(),
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": "CrediLinq Competitor Intelligence"
                    },
                    "timestamp": message.timestamp.isoformat()
                }
            ]
        }
        
        async with self.session.post(config.webhook_url, json=payload) as response:
            if response.status == 204:  # Discord returns 204 on success
                return IntegrationResult(
                    success=True,
                    integration_name=config.name,
                    message_id=f"discord_{int(message.timestamp.timestamp())}",
                    response_data={"status": response.status}
                )
            else:
                error_text = await response.text()
                raise Exception(f"Discord API error {response.status}: {error_text}")
    
    async def _send_telegram_message(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send message to Telegram."""
        if not config.api_token or not config.channel:
            raise ValueError("Telegram bot token and chat ID are required")
        
        # Format message for Telegram
        telegram_text = f"""
🎯 *{message.title}*

📋 *Event:* {message.event_type.value.replace('_', ' ').title()}
⚡ *Priority:* {message.priority.value.upper()}
🕒 *Time:* {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{message.content}

_CrediLinq Competitor Intelligence_
        """.strip()
        
        url = f"https://api.telegram.org/bot{config.api_token}/sendMessage"
        payload = {
            "chat_id": config.channel,
            "text": telegram_text,
            "parse_mode": "Markdown"
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                response_data = await response.json()
                return IntegrationResult(
                    success=True,
                    integration_name=config.name,
                    message_id=f"telegram_{response_data.get('result', {}).get('message_id', 'unknown')}",
                    response_data=response_data
                )
            else:
                error_text = await response.text()
                raise Exception(f"Telegram API error {response.status}: {error_text}")
    
    async def _send_zapier_webhook(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send webhook to Zapier."""
        return await self._send_webhook_message(config, message)
    
    async def _send_ifttt_webhook(
        self, 
        config: IntegrationConfig, 
        message: NotificationMessage
    ) -> IntegrationResult:
        """Send webhook to IFTTT."""
        if not config.webhook_url:
            raise ValueError("IFTTT webhook URL is required")
        
        # IFTTT expects specific format
        payload = {
            "value1": message.title,
            "value2": message.content,
            "value3": f"{message.event_type.value}|{message.priority.value}"
        }
        
        async with self.session.post(config.webhook_url, json=payload) as response:
            if response.status == 200:
                response_text = await response.text()
                return IntegrationResult(
                    success=True,
                    integration_name=config.name,
                    message_id=f"ifttt_{int(message.timestamp.timestamp())}",
                    response_data={"response": response_text}
                )
            else:
                error_text = await response.text()
                raise Exception(f"IFTTT API error {response.status}: {error_text}")
    
    async def test_integration(self, integration_name: str) -> IntegrationResult:
        """Test an integration with a sample message."""
        if integration_name not in self.integrations:
            return IntegrationResult(
                success=False,
                integration_name=integration_name,
                error="Integration not found"
            )
        
        test_message = NotificationMessage(
            title="Integration Test",
            content="This is a test message to verify the integration is working correctly.",
            event_type=EventType.SYSTEM_HEALTH,
            priority=MessagePriority.LOW
        )
        
        return await self._send_to_integration(self.integrations[integration_name], test_message)
    
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered integrations."""
        status = {}
        for name, config in self.integrations.items():
            status[name] = {
                "type": config.integration_type.value,
                "enabled": config.enabled,
                "event_filters": [ef.value for ef in config.event_filters] if config.event_filters else None,
                "priority_threshold": config.priority_threshold.value,
                "has_webhook_url": bool(config.webhook_url),
                "has_api_token": bool(config.api_token),
                "channel": config.channel
            }
        return status
    
    async def broadcast_system_health(self, health_data: Dict[str, Any]):
        """Broadcast system health status to all relevant integrations."""
        message = NotificationMessage(
            title="System Health Update",
            content=f"System status: {health_data.get('status', 'unknown')}",
            event_type=EventType.SYSTEM_HEALTH,
            priority=MessagePriority.LOW,
            data=health_data
        )
        
        return await self.send_notification(message)
    
    async def notify_new_trend(self, trend_data: Dict[str, Any]):
        """Notify about new trend detection."""
        message = NotificationMessage(
            title=f"New Trend Detected: {trend_data.get('title', 'Unknown')}",
            content=f"Strength: {trend_data.get('strength', 'unknown')}, Confidence: {trend_data.get('confidence', 0):.2f}",
            event_type=EventType.NEW_TREND,
            priority=MessagePriority.NORMAL,
            data=trend_data
        )
        
        return await self.send_notification(message)
    
    async def notify_high_priority_alert(self, alert_data: Dict[str, Any]):
        """Notify about high priority alerts."""
        message = NotificationMessage(
            title=f"High Priority Alert: {alert_data.get('title', 'Alert')}",
            content=alert_data.get('description', 'No description available'),
            event_type=EventType.HIGH_PRIORITY_ALERT,
            priority=MessagePriority.HIGH,
            data=alert_data
        )
        
        return await self.send_notification(message)
    
    async def notify_report_generated(self, report_data: Dict[str, Any]):
        """Notify about generated reports."""
        message = NotificationMessage(
            title=f"Report Generated: {report_data.get('title', 'Report')}",
            content=f"Report type: {report_data.get('type', 'unknown')}, Format: {report_data.get('format', 'unknown')}",
            event_type=EventType.REPORT_GENERATED,
            priority=MessagePriority.NORMAL,
            data=report_data,
            attachments=[report_data.get('file_path')] if report_data.get('file_path') else None
        )
        
        return await self.send_notification(message)

# Global instance
external_integrations_service = ExternalIntegrationsService()