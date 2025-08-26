"""
Event Bus System for Agent Communication

This module provides a centralized event bus for publishing and subscribing
to events across the agent ecosystem, enabling loose coupling and reactive
communication patterns.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system."""
    # Campaign events
    CAMPAIGN_CREATED = "campaign_created"
    CAMPAIGN_STARTED = "campaign_started"
    CAMPAIGN_PAUSED = "campaign_paused"
    CAMPAIGN_COMPLETED = "campaign_completed"
    CAMPAIGN_FAILED = "campaign_failed"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_NODE_ENTERED = "workflow_node_entered"
    WORKFLOW_NODE_COMPLETED = "workflow_node_completed"
    WORKFLOW_NODE_FAILED = "workflow_node_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    
    # Agent events
    AGENT_REGISTERED = "agent_registered"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_HEALTH_CHECK = "agent_health_check"
    
    # Task events
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    
    # Content events
    CONTENT_CREATED = "content_created"
    CONTENT_UPDATED = "content_updated"
    CONTENT_PUBLISHED = "content_published"
    CONTENT_REVIEWED = "content_reviewed"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    
    # Performance events
    PERFORMANCE_METRIC = "performance_metric"
    PERFORMANCE_ALERT = "performance_alert"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"


@dataclass
class Event:
    """Represents an event in the system."""
    event_id: str
    event_type: EventType
    source: str
    target: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'target': self.target,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'trace_id': self.trace_id,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            source=data['source'],
            target=data.get('target'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            trace_id=data.get('trace_id'),
            priority=data.get('priority', 5)
        )


@dataclass
class EventSubscription:
    """Represents a subscription to events."""
    subscription_id: str
    subscriber_id: str
    event_types: Set[EventType]
    handler: Callable
    filter_condition: Optional[Callable] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Can return awaitable


class EventBus:
    """
    Centralized event bus for agent communication.
    
    Provides publish/subscribe messaging with filtering, priority handling,
    and both synchronous and asynchronous event processing.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.subscribers: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        self.global_subscribers: List[EventSubscription] = []
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.event_history: List[Event] = []
        self.max_history_size = 1000
        self.processing_task: Optional[asyncio.Task] = None
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'active_subscriptions': 0
        }
        self._running = False
        
    async def start(self):
        """Start the event bus processing."""
        if self._running:
            return
            
        self._running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        
        # Publish system startup event
        await self.publish(Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SYSTEM_STARTUP,
            source="event_bus",
            data={'message': 'Event bus started'}
        ))
        
    async def stop(self):
        """Stop the event bus processing."""
        if not self._running:
            return
            
        self._running = False
        
        # Publish system shutdown event
        await self.publish(Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SYSTEM_SHUTDOWN,
            source="event_bus",
            data={'message': 'Event bus stopping'}
        ))
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Event bus stopped")
        
    async def publish(self, event: Event):
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
        """
        try:
            if not self._running:
                logger.warning("Event bus not running, starting it")
                await self.start()
                
            await self.event_queue.put(event)
            self.stats['events_published'] += 1
            
            logger.debug(f"Published event: {event.event_type.value} from {event.source}")
            
        except asyncio.QueueFull:
            logger.error("Event queue is full, dropping event")
            self.stats['events_failed'] += 1
        except Exception as e:
            logger.error(f"Failed to publish event: {str(e)}")
            self.stats['events_failed'] += 1
            
    async def subscribe(
        self,
        subscriber_id: str,
        event_types: Union[EventType, List[EventType]],
        handler: Union[EventHandler, AsyncEventHandler],
        filter_condition: Optional[Callable[[Event], bool]] = None
    ) -> str:
        """
        Subscribe to events.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            event_types: Event type(s) to subscribe to
            handler: Function to handle events
            filter_condition: Optional filter for events
            
        Returns:
            Subscription ID
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]
            
        subscription_id = str(uuid.uuid4())
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_types=set(event_types),
            handler=handler,
            filter_condition=filter_condition
        )
        
        # Add to appropriate subscriber lists
        for event_type in event_types:
            self.subscribers[event_type].append(subscription)
            
        self.stats['active_subscriptions'] += 1
        
        logger.info(f"Subscribed {subscriber_id} to {[et.value for et in event_types]}")
        return subscription_id
        
    async def subscribe_all(
        self,
        subscriber_id: str,
        handler: Union[EventHandler, AsyncEventHandler],
        filter_condition: Optional[Callable[[Event], bool]] = None
    ) -> str:
        """
        Subscribe to all events.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            handler: Function to handle events
            filter_condition: Optional filter for events
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_types=set(),  # Empty set means all events
            handler=handler,
            filter_condition=filter_condition
        )
        
        self.global_subscribers.append(subscription)
        self.stats['active_subscriptions'] += 1
        
        logger.info(f"Subscribed {subscriber_id} to all events")
        return subscription_id
        
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID of the subscription to remove
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        # Remove from specific event type subscribers
        for event_type, subscriptions in self.subscribers.items():
            self.subscribers[event_type] = [
                sub for sub in subscriptions 
                if sub.subscription_id != subscription_id
            ]
            
        # Remove from global subscribers
        original_count = len(self.global_subscribers)
        self.global_subscribers = [
            sub for sub in self.global_subscribers
            if sub.subscription_id != subscription_id
        ]
        
        removed = len(self.global_subscribers) < original_count
        if removed:
            self.stats['active_subscriptions'] -= 1
            
        logger.info(f"Unsubscribed: {subscription_id}")
        return removed
        
    async def unsubscribe_all(self, subscriber_id: str) -> int:
        """
        Unsubscribe all subscriptions for a subscriber.
        
        Args:
            subscriber_id: ID of the subscriber
            
        Returns:
            Number of subscriptions removed
        """
        removed_count = 0
        
        # Remove from specific event type subscribers
        for event_type, subscriptions in self.subscribers.items():
            original_count = len(subscriptions)
            self.subscribers[event_type] = [
                sub for sub in subscriptions 
                if sub.subscriber_id != subscriber_id
            ]
            removed_count += original_count - len(self.subscribers[event_type])
            
        # Remove from global subscribers
        original_count = len(self.global_subscribers)
        self.global_subscribers = [
            sub for sub in self.global_subscribers
            if sub.subscriber_id != subscriber_id
        ]
        removed_count += original_count - len(self.global_subscribers)
        
        self.stats['active_subscriptions'] -= removed_count
        
        logger.info(f"Unsubscribed all for {subscriber_id}: {removed_count} subscriptions")
        return removed_count
        
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                # Get event with timeout to allow for shutdown
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                await self._handle_event(event)
                self.stats['events_processed'] += 1
                
                # Add to history
                self.event_history.append(event)
                if len(self.event_history) > self.max_history_size:
                    self.event_history.pop(0)
                    
            except asyncio.TimeoutError:
                # Timeout is expected, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                self.stats['events_failed'] += 1
                
    async def _handle_event(self, event: Event):
        """Handle an event by notifying all relevant subscribers."""
        # Get subscribers for this specific event type
        subscribers = self.subscribers.get(event.event_type, [])
        
        # Add global subscribers
        subscribers.extend(self.global_subscribers)
        
        # Process each subscription
        for subscription in subscribers:
            if not subscription.active:
                continue
                
            try:
                # Apply filter if present
                if subscription.filter_condition:
                    if not subscription.filter_condition(event):
                        continue
                        
                # Call handler
                if asyncio.iscoroutinefunction(subscription.handler):
                    await subscription.handler(event)
                else:
                    subscription.handler(event)
                    
            except Exception as e:
                logger.error(f"Error in event handler for {subscription.subscriber_id}: {str(e)}")
                
    def create_event(
        self,
        event_type: EventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        target: Optional[str] = None,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        priority: int = 5
    ) -> Event:
        """
        Create a new event.
        
        Args:
            event_type: Type of the event
            source: Source of the event
            data: Event data
            target: Optional target for the event
            correlation_id: Optional correlation ID
            trace_id: Optional trace ID
            priority: Event priority (1-10)
            
        Returns:
            Created event
        """
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            target=target,
            data=data or {},
            correlation_id=correlation_id,
            trace_id=trace_id,
            priority=priority
        )
        
    async def publish_campaign_event(
        self,
        event_type: EventType,
        campaign_id: str,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """Publish a campaign-related event."""
        event_data = {'campaign_id': campaign_id}
        if data:
            event_data.update(data)
            
        event = self.create_event(
            event_type=event_type,
            source=source,
            data=event_data,
            correlation_id=correlation_id or campaign_id
        )
        
        await self.publish(event)
        
    async def publish_workflow_event(
        self,
        event_type: EventType,
        workflow_id: str,
        campaign_id: str,
        source: str,
        node_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """Publish a workflow-related event."""
        event_data = {
            'workflow_id': workflow_id,
            'campaign_id': campaign_id
        }
        if node_id:
            event_data['node_id'] = node_id
        if data:
            event_data.update(data)
            
        event = self.create_event(
            event_type=event_type,
            source=source,
            data=event_data,
            correlation_id=campaign_id
        )
        
        await self.publish(event)
        
    async def publish_agent_event(
        self,
        event_type: EventType,
        agent_id: str,
        agent_type: str,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        campaign_id: Optional[str] = None
    ):
        """Publish an agent-related event."""
        event_data = {
            'agent_id': agent_id,
            'agent_type': agent_type
        }
        if campaign_id:
            event_data['campaign_id'] = campaign_id
        if data:
            event_data.update(data)
            
        event = self.create_event(
            event_type=event_type,
            source=source,
            data=event_data,
            correlation_id=campaign_id
        )
        
        await self.publish(event)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self.stats,
            'queue_size': self.event_queue.qsize(),
            'history_size': len(self.event_history),
            'is_running': self._running,
            'subscriber_count_by_type': {
                event_type.value: len(subs) 
                for event_type, subs in self.subscribers.items()
            },
            'global_subscribers': len(self.global_subscribers)
        }
        
    def get_recent_events(self, limit: int = 50) -> List[Event]:
        """Get recent events from history."""
        return self.event_history[-limit:]
        
    def get_events_by_type(self, event_type: EventType, limit: int = 50) -> List[Event]:
        """Get recent events of a specific type."""
        events = [e for e in self.event_history if e.event_type == event_type]
        return events[-limit:]
        
    def get_events_by_source(self, source: str, limit: int = 50) -> List[Event]:
        """Get recent events from a specific source."""
        events = [e for e in self.event_history if e.source == source]
        return events[-limit:]


# Global event bus instance
event_bus = EventBus()


async def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    if not event_bus._running:
        await event_bus.start()
    return event_bus