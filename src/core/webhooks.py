"""
Comprehensive webhook system for third-party integrations.
Supports event-driven notifications, retry logic, and security verification.
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, HttpUrl, validator
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config.settings import settings
from .cache import cache
from .monitoring import metrics, async_performance_tracker

class WebhookEvent(str, Enum):
    """Supported webhook event types."""
    BLOG_CREATED = "blog.created"
    BLOG_UPDATED = "blog.updated"
    BLOG_PUBLISHED = "blog.published"
    BLOG_DELETED = "blog.deleted"
    
    CAMPAIGN_CREATED = "campaign.created"
    CAMPAIGN_UPDATED = "campaign.updated"
    CAMPAIGN_COMPLETED = "campaign.completed"
    CAMPAIGN_FAILED = "campaign.failed"
    
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    ANALYTICS_THRESHOLD = "analytics.threshold_reached"
    SYSTEM_HEALTH = "system.health_alert"
    
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"

class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"

@dataclass
class WebhookPayload:
    """Webhook payload data structure."""
    event: WebhookEvent
    data: Dict[str, Any]
    timestamp: datetime
    webhook_id: str
    delivery_id: str
    version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "webhook_id": self.webhook_id,
            "delivery_id": self.delivery_id,
            "version": self.version
        }

class WebhookEndpoint(BaseModel):
    """Webhook endpoint configuration."""
    id: str
    url: HttpUrl
    secret: str
    events: List[WebhookEvent]
    active: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('url')
    def validate_url(cls, v):
        """Validate webhook URL."""
        parsed = urlparse(str(v))
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('URL must use HTTP or HTTPS')
        if parsed.scheme == 'http' and not settings.debug:
            raise ValueError('HTTP URLs not allowed in production')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    delivery_id: str
    webhook_id: str
    endpoint_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    status: WebhookStatus
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    attempt_count: int = 0
    max_retries: int = 3
    next_retry: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class WebhookManager:
    """Manages webhook endpoints, delivery, and retry logic."""
    
    def __init__(self):
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.event_listeners: Dict[WebhookEvent, Set[str]] = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        self._retry_queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None
    
    async def start_retry_processor(self):
        """Start background task for processing webhook retries."""
        if self._retry_task is None or self._retry_task.done():
            self._retry_task = asyncio.create_task(self._process_retries())
    
    async def stop_retry_processor(self):
        """Stop background retry processor."""
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
    
    def register_endpoint(
        self,
        url: str,
        secret: str,
        events: List[WebhookEvent],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new webhook endpoint."""
        endpoint_id = str(uuid.uuid4())
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            url=url,
            secret=secret,
            events=events,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.endpoints[endpoint_id] = endpoint
        
        # Update event listeners
        for event in events:
            if event not in self.event_listeners:
                self.event_listeners[event] = set()
            self.event_listeners[event].add(endpoint_id)
        
        return endpoint_id
    
    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Remove a webhook endpoint."""
        if endpoint_id not in self.endpoints:
            return False
        
        endpoint = self.endpoints[endpoint_id]
        
        # Remove from event listeners
        for event in endpoint.events:
            if event in self.event_listeners:
                self.event_listeners[event].discard(endpoint_id)
                if not self.event_listeners[event]:
                    del self.event_listeners[event]
        
        del self.endpoints[endpoint_id]
        return True
    
    def update_endpoint(
        self,
        endpoint_id: str,
        url: Optional[str] = None,
        secret: Optional[str] = None,
        events: Optional[List[WebhookEvent]] = None,
        active: Optional[bool] = None
    ) -> bool:
        """Update an existing webhook endpoint."""
        if endpoint_id not in self.endpoints:
            return False
        
        endpoint = self.endpoints[endpoint_id]
        
        # Remove from old event listeners if events are changing
        if events is not None:
            for event in endpoint.events:
                if event in self.event_listeners:
                    self.event_listeners[event].discard(endpoint_id)
        
        # Update endpoint
        if url is not None:
            endpoint.url = url
        if secret is not None:
            endpoint.secret = secret
        if events is not None:
            endpoint.events = events
        if active is not None:
            endpoint.active = active
        
        endpoint.updated_at = datetime.utcnow()
        
        # Add to new event listeners
        if events is not None:
            for event in events:
                if event not in self.event_listeners:
                    self.event_listeners[event] = set()
                self.event_listeners[event].add(endpoint_id)
        
        return True
    
    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint by ID."""
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all webhook endpoints."""
        return list(self.endpoints.values())
    
    async def emit_event(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        webhook_id: Optional[str] = None
    ) -> List[str]:
        """Emit an event to all registered webhooks."""
        if event not in self.event_listeners:
            return []
        
        delivery_ids = []
        payload = WebhookPayload(
            event=event,
            data=data,
            timestamp=datetime.utcnow(),
            webhook_id=webhook_id or str(uuid.uuid4()),
            delivery_id=str(uuid.uuid4())
        )
        
        # Get all endpoints listening for this event
        listener_ids = self.event_listeners[event].copy()
        
        for endpoint_id in listener_ids:
            endpoint = self.endpoints.get(endpoint_id)
            if endpoint and endpoint.active:
                delivery_id = await self._deliver_webhook(endpoint, payload)
                if delivery_id:
                    delivery_ids.append(delivery_id)
        
        return delivery_ids
    
    async def _deliver_webhook(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> Optional[str]:
        """Deliver webhook to a specific endpoint."""
        delivery_id = str(uuid.uuid4())
        
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=payload.webhook_id,
            endpoint_id=endpoint.id,
            event=payload.event,
            payload=payload.to_dict(),
            status=WebhookStatus.PENDING,
            max_retries=endpoint.max_retries,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.deliveries[delivery_id] = delivery
        
        # Attempt delivery
        success = await self._attempt_delivery(endpoint, payload, delivery)
        
        if success:
            delivery.status = WebhookStatus.DELIVERED
            delivery.completed_at = datetime.utcnow()
        else:
            delivery.status = WebhookStatus.FAILED
            # Schedule retry if attempts remain
            if delivery.attempt_count < delivery.max_retries:
                await self._schedule_retry(delivery)
        
        delivery.updated_at = datetime.utcnow()
        
        # Track metrics
        metrics.increment_counter(
            "webhook.delivery",
            tags={
                "event": payload.event,
                "status": delivery.status,
                "endpoint": endpoint.id
            }
        )
        
        return delivery_id
    
    async def _attempt_delivery(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload,
        delivery: WebhookDelivery
    ) -> bool:
        """Attempt to deliver webhook to endpoint."""
        delivery.attempt_count += 1
        
        try:
            async with async_performance_tracker(f"webhook.delivery.{payload.event}"):
                # Generate signature
                signature = self._generate_signature(
                    payload.to_dict(),
                    endpoint.secret
                )
                
                headers = {
                    "content-type": "application/json",
                    "user-agent": f"CrediLinq-Webhooks/{settings.api_version}",
                    "x-webhook-event": payload.event,
                    "x-webhook-delivery": payload.delivery_id,
                    "x-webhook-signature": signature,
                    "x-webhook-timestamp": str(int(time.time()))
                }
                
                # Make HTTP request
                response = await self.client.post(
                    str(endpoint.url),
                    json=payload.to_dict(),
                    headers=headers,
                    timeout=endpoint.timeout_seconds
                )
                
                delivery.response_status = response.status_code
                delivery.response_body = response.text[:1000]  # Limit response size
                
                # Consider 2xx status codes as success
                return 200 <= response.status_code < 300
                
        except httpx.TimeoutException:
            delivery.error_message = "Request timeout"
            return False
        except httpx.RequestError as e:
            delivery.error_message = f"Request error: {str(e)}"
            return False
        except Exception as e:
            delivery.error_message = f"Unexpected error: {str(e)}"
            return False
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def _schedule_retry(self, delivery: WebhookDelivery):
        """Schedule webhook delivery retry."""
        # Exponential backoff: 2^attempt minutes
        delay_minutes = 2 ** delivery.attempt_count
        delivery.next_retry = datetime.utcnow() + timedelta(minutes=delay_minutes)
        delivery.status = WebhookStatus.RETRYING
        
        # Add to retry queue
        await self._retry_queue.put(delivery.delivery_id)
    
    async def _process_retries(self):
        """Background task to process webhook retries."""
        while True:
            try:
                # Get delivery ID from retry queue
                delivery_id = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=60.0  # Check every minute
                )
                
                delivery = self.deliveries.get(delivery_id)
                if not delivery or delivery.status != WebhookStatus.RETRYING:
                    continue
                
                # Check if it's time to retry
                if delivery.next_retry and datetime.utcnow() < delivery.next_retry:
                    # Put back in queue for later
                    await self._retry_queue.put(delivery_id)
                    await asyncio.sleep(60)  # Wait a minute before checking again
                    continue
                
                # Get endpoint
                endpoint = self.endpoints.get(delivery.endpoint_id)
                if not endpoint or not endpoint.active:
                    delivery.status = WebhookStatus.EXPIRED
                    continue
                
                # Reconstruct payload
                payload = WebhookPayload(
                    event=delivery.event,
                    data=delivery.payload.get("data", {}),
                    timestamp=datetime.fromisoformat(delivery.payload["timestamp"]),
                    webhook_id=delivery.webhook_id,
                    delivery_id=delivery.delivery_id
                )
                
                # Reattempt delivery
                success = await self._attempt_delivery(endpoint, payload, delivery)
                
                if success:
                    delivery.status = WebhookStatus.DELIVERED
                    delivery.completed_at = datetime.utcnow()
                elif delivery.attempt_count >= delivery.max_retries:
                    delivery.status = WebhookStatus.EXPIRED
                else:
                    # Schedule another retry
                    await self._schedule_retry(delivery)
                
                delivery.updated_at = datetime.utcnow()
                
            except asyncio.TimeoutError:
                # No retries in queue, continue loop
                continue
            except Exception as e:
                # Log error and continue
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error processing webhook retry: {e}")
                await asyncio.sleep(60)
    
    def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get webhook delivery by ID."""
        return self.deliveries.get(delivery_id)
    
    def list_deliveries(
        self,
        endpoint_id: Optional[str] = None,
        event: Optional[WebhookEvent] = None,
        status: Optional[WebhookStatus] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """List webhook deliveries with optional filters."""
        deliveries = list(self.deliveries.values())
        
        # Apply filters
        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]
        if event:
            deliveries = [d for d in deliveries if d.event == event]
        if status:
            deliveries = [d for d in deliveries if d.status == status]
        
        # Sort by creation time (newest first) and limit
        deliveries.sort(key=lambda d: d.created_at, reverse=True)
        return deliveries[:limit]
    
    async def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Verify webhook signature from incoming request."""
        if not signature.startswith('sha256='):
            return False
        
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        received_signature = signature[7:]  # Remove 'sha256=' prefix
        
        return hmac.compare_digest(expected_signature, received_signature)

# Global webhook manager instance
webhook_manager = WebhookManager()

# FastAPI router for webhook management
router = APIRouter()

class WebhookEndpointCreate(BaseModel):
    """Request model for creating webhook endpoints."""
    url: HttpUrl
    events: List[WebhookEvent]
    secret: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('secret', pre=True, always=True)
    def generate_secret(cls, v):
        if v is None:
            return str(uuid.uuid4())
        return v

class WebhookEndpointUpdate(BaseModel):
    """Request model for updating webhook endpoints."""
    url: Optional[HttpUrl] = None
    events: Optional[List[WebhookEvent]] = None
    active: Optional[bool] = None

@router.post("/webhooks", status_code=201)
async def create_webhook_endpoint(
    webhook_data: WebhookEndpointCreate,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Create a new webhook endpoint."""
    endpoint_id = webhook_manager.register_endpoint(
        url=str(webhook_data.url),
        secret=webhook_data.secret,
        events=webhook_data.events,
        metadata=webhook_data.metadata
    )
    
    # Start retry processor if not already running
    background_tasks.add_task(webhook_manager.start_retry_processor)
    
    endpoint = webhook_manager.get_endpoint(endpoint_id)
    return {
        "id": endpoint_id,
        "url": str(endpoint.url),
        "events": endpoint.events,
        "active": endpoint.active,
        "created_at": endpoint.created_at.isoformat()
    }

@router.get("/webhooks")
async def list_webhook_endpoints() -> Dict[str, Any]:
    """List all webhook endpoints."""
    endpoints = webhook_manager.list_endpoints()
    return {
        "endpoints": [
            {
                "id": e.id,
                "url": str(e.url),
                "events": e.events,
                "active": e.active,
                "created_at": e.created_at.isoformat(),
                "metadata": e.metadata
            }
            for e in endpoints
        ]
    }

@router.get("/webhooks/{endpoint_id}")
async def get_webhook_endpoint(endpoint_id: str) -> Dict[str, Any]:
    """Get webhook endpoint details."""
    endpoint = webhook_manager.get_endpoint(endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    
    return {
        "id": endpoint.id,
        "url": str(endpoint.url),
        "events": endpoint.events,
        "active": endpoint.active,
        "created_at": endpoint.created_at.isoformat(),
        "updated_at": endpoint.updated_at.isoformat(),
        "metadata": endpoint.metadata
    }

@router.put("/webhooks/{endpoint_id}")
async def update_webhook_endpoint(
    endpoint_id: str,
    update_data: WebhookEndpointUpdate
) -> Dict[str, Any]:
    """Update webhook endpoint."""
    success = webhook_manager.update_endpoint(
        endpoint_id=endpoint_id,
        url=str(update_data.url) if update_data.url else None,
        events=update_data.events,
        active=update_data.active
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    
    endpoint = webhook_manager.get_endpoint(endpoint_id)
    return {
        "id": endpoint.id,
        "url": str(endpoint.url),
        "events": endpoint.events,
        "active": endpoint.active,
        "updated_at": endpoint.updated_at.isoformat()
    }

@router.delete("/webhooks/{endpoint_id}")
async def delete_webhook_endpoint(endpoint_id: str) -> Dict[str, Any]:
    """Delete webhook endpoint."""
    success = webhook_manager.unregister_endpoint(endpoint_id)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    
    return {"message": "Webhook endpoint deleted successfully"}

@router.get("/webhooks/{endpoint_id}/deliveries")
async def list_webhook_deliveries(
    endpoint_id: str,
    limit: int = 50
) -> Dict[str, Any]:
    """List webhook deliveries for an endpoint."""
    deliveries = webhook_manager.list_deliveries(endpoint_id=endpoint_id, limit=limit)
    
    return {
        "deliveries": [
            {
                "delivery_id": d.delivery_id,
                "event": d.event,
                "status": d.status,
                "attempt_count": d.attempt_count,
                "response_status": d.response_status,
                "created_at": d.created_at.isoformat(),
                "completed_at": d.completed_at.isoformat() if d.completed_at else None,
                "error_message": d.error_message
            }
            for d in deliveries
        ]
    }

@router.post("/webhooks/test/{endpoint_id}")
async def test_webhook_endpoint(endpoint_id: str) -> Dict[str, Any]:
    """Send a test event to webhook endpoint."""
    endpoint = webhook_manager.get_endpoint(endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")
    
    # Send test event
    delivery_ids = await webhook_manager.emit_event(
        event=WebhookEvent.SYSTEM_HEALTH,
        data={
            "test": True,
            "message": "This is a test webhook",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {
        "message": "Test webhook sent",
        "delivery_ids": delivery_ids
    }

# Helper functions for emitting events from other parts of the application
async def emit_blog_created(blog_id: str, blog_data: Dict[str, Any]):
    """Emit blog created event."""
    await webhook_manager.emit_event(
        WebhookEvent.BLOG_CREATED,
        {"blog_id": blog_id, **blog_data}
    )

async def emit_campaign_completed(campaign_id: str, campaign_data: Dict[str, Any]):
    """Emit campaign completed event."""
    await webhook_manager.emit_event(
        WebhookEvent.CAMPAIGN_COMPLETED,
        {"campaign_id": campaign_id, **campaign_data}
    )

async def emit_analytics_threshold(metric: str, threshold: float, current_value: float):
    """Emit analytics threshold reached event."""
    await webhook_manager.emit_event(
        WebhookEvent.ANALYTICS_THRESHOLD,
        {
            "metric": metric,
            "threshold": threshold,
            "current_value": current_value,
            "timestamp": datetime.utcnow().isoformat()
        }
    )