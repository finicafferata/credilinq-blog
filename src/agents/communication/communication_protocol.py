"""
Communication Protocol for Standardized Agent Interaction

This module provides standardized communication protocols for agent interaction,
including message formatting, routing, and coordination patterns.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json

from .event_bus import EventBus, EventType, get_event_bus
from .message_broker import MessageBroker, MessageType, MessagePriority, get_message_broker

logger = logging.getLogger(__name__)


class CommunicationChannel(Enum):
    """Communication channels for different types of interaction."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    EVENT_DRIVEN = "event_driven"
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"


class ProtocolType(Enum):
    """Types of communication protocols."""
    COMMAND = "command"
    QUERY = "query"
    NOTIFICATION = "notification"
    COORDINATION = "coordination"
    WORKFLOW = "workflow"
    HEALTH_CHECK = "health_check"


@dataclass
class AgentMessage:
    """Standardized agent message format."""
    protocol_type: ProtocolType
    sender: str
    recipients: List[str]
    channel: CommunicationChannel
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_required: bool = False
    timeout_seconds: Optional[int] = None
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'protocol_type': self.protocol_type.value,
            'sender': self.sender,
            'recipients': self.recipients,
            'channel': self.channel.value,
            'payload': self.payload,
            'headers': self.headers,
            'correlation_id': self.correlation_id,
            'reply_required': self.reply_required,
            'timeout_seconds': self.timeout_seconds,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat()
        }


class ProtocolHandler(ABC):
    """Abstract base class for protocol handlers."""
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Handle an incoming message."""
        pass
    
    @abstractmethod
    def get_supported_protocols(self) -> List[ProtocolType]:
        """Get list of supported protocol types."""
        pass


class CommunicationProtocol:
    """
    Standardized communication protocol for agent interaction.
    
    Provides high-level communication abstractions over the event bus
    and message broker for standardized agent coordination.
    """
    
    def __init__(
        self,
        agent_id: str,
        event_bus: Optional[EventBus] = None,
        message_broker: Optional[MessageBroker] = None
    ):
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.message_broker = message_broker
        self.protocol_handlers: Dict[ProtocolType, ProtocolHandler] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'conversations_started': 0,
            'conversations_completed': 0
        }
        
    async def initialize(self):
        """Initialize the communication protocol."""
        if self.event_bus is None:
            self.event_bus = await get_event_bus()
        if self.message_broker is None:
            self.message_broker = await get_message_broker()
            
        # Register message handler
        await self.message_broker.register_handler(
            self.agent_id,
            self._handle_incoming_message
        )
        
        logger.info(f"Communication protocol initialized for {self.agent_id}")
        
    async def shutdown(self):
        """Shutdown the communication protocol."""
        if self.message_broker:
            await self.message_broker.unregister_handler(self.agent_id)
            
        logger.info(f"Communication protocol shutdown for {self.agent_id}")
        
    def register_protocol_handler(
        self,
        protocol_type: ProtocolType,
        handler: ProtocolHandler
    ):
        """Register a protocol handler."""
        self.protocol_handlers[protocol_type] = handler
        logger.info(f"Registered {protocol_type.value} handler for {self.agent_id}")
        
    async def send_command(
        self,
        recipient: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        reply_required: bool = True,
        timeout: Optional[int] = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Send a command to another agent.
        
        Args:
            recipient: Target agent ID
            command: Command to execute
            parameters: Command parameters
            reply_required: Whether a reply is expected
            timeout: Timeout for reply
            
        Returns:
            Reply data if reply_required, None otherwise
        """
        message = AgentMessage(
            protocol_type=ProtocolType.COMMAND,
            sender=self.agent_id,
            recipients=[recipient],
            channel=CommunicationChannel.DIRECT,
            payload={
                'command': command,
                'parameters': parameters or {}
            },
            reply_required=reply_required,
            timeout_seconds=timeout,
            priority=MessagePriority.HIGH
        )
        
        return await self._send_message(message)
        
    async def send_query(
        self,
        recipient: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Send a query to another agent.
        
        Args:
            recipient: Target agent ID
            query: Query to execute
            filters: Query filters
            timeout: Timeout for response
            
        Returns:
            Query result
        """
        message = AgentMessage(
            protocol_type=ProtocolType.QUERY,
            sender=self.agent_id,
            recipients=[recipient],
            channel=CommunicationChannel.REQUEST_RESPONSE,
            payload={
                'query': query,
                'filters': filters or {}
            },
            reply_required=True,
            timeout_seconds=timeout,
            priority=MessagePriority.NORMAL
        )
        
        return await self._send_message(message)
        
    async def send_notification(
        self,
        recipients: List[str],
        notification_type: str,
        data: Dict[str, Any],
        channel: CommunicationChannel = CommunicationChannel.MULTICAST
    ):
        """
        Send a notification to multiple agents.
        
        Args:
            recipients: List of recipient agent IDs
            notification_type: Type of notification
            data: Notification data
            channel: Communication channel
        """
        message = AgentMessage(
            protocol_type=ProtocolType.NOTIFICATION,
            sender=self.agent_id,
            recipients=recipients,
            channel=channel,
            payload={
                'notification_type': notification_type,
                'data': data
            },
            reply_required=False,
            priority=MessagePriority.NORMAL
        )
        
        await self._send_message(message)
        
    async def broadcast_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast status to all interested agents.
        
        Args:
            status: Status message
            details: Additional status details
        """
        await self.event_bus.publish_agent_event(
            event_type=EventType.AGENT_HEALTH_CHECK,
            agent_id=self.agent_id,
            agent_type="unknown",  # Could be enhanced to track agent type
            source=self.agent_id,
            data={
                'status': status,
                'details': details or {},
                'timestamp': datetime.now().isoformat()
            }
        )
        
    async def coordinate_workflow(
        self,
        participants: List[str],
        workflow_id: str,
        coordination_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate a workflow with multiple agents.
        
        Args:
            participants: List of participating agent IDs
            workflow_id: Unique workflow identifier
            coordination_data: Workflow coordination data
            
        Returns:
            Coordination results
        """
        conversation_id = f"workflow_{workflow_id}_{datetime.now().timestamp()}"
        
        # Start conversation tracking
        self.active_conversations[conversation_id] = {
            'type': 'workflow_coordination',
            'participants': participants,
            'workflow_id': workflow_id,
            'started_at': datetime.now(),
            'responses': {}
        }
        
        message = AgentMessage(
            protocol_type=ProtocolType.COORDINATION,
            sender=self.agent_id,
            recipients=participants,
            channel=CommunicationChannel.MULTICAST,
            payload={
                'coordination_type': 'workflow',
                'workflow_id': workflow_id,
                'data': coordination_data,
                'conversation_id': conversation_id
            },
            reply_required=True,
            timeout_seconds=60,
            priority=MessagePriority.HIGH,
            correlation_id=conversation_id
        )
        
        # Send to all participants
        responses = {}
        for participant in participants:
            message.recipients = [participant]
            response = await self._send_message(message)
            responses[participant] = response
            
        # Update conversation
        self.active_conversations[conversation_id]['responses'] = responses
        self.active_conversations[conversation_id]['completed_at'] = datetime.now()
        
        return responses
        
    async def request_agent_status(
        self,
        target_agents: List[str],
        timeout: int = 15
    ) -> Dict[str, Dict[str, Any]]:
        """
        Request status from multiple agents.
        
        Args:
            target_agents: List of agent IDs to query
            timeout: Timeout for responses
            
        Returns:
            Status responses by agent ID
        """
        message = AgentMessage(
            protocol_type=ProtocolType.HEALTH_CHECK,
            sender=self.agent_id,
            recipients=target_agents,
            channel=CommunicationChannel.MULTICAST,
            payload={
                'check_type': 'status_request',
                'requested_fields': ['health', 'load', 'capabilities', 'current_tasks']
            },
            reply_required=True,
            timeout_seconds=timeout,
            priority=MessagePriority.NORMAL
        )
        
        responses = {}
        for agent in target_agents:
            message.recipients = [agent]
            try:
                response = await self._send_message(message)
                responses[agent] = response
            except Exception as e:
                responses[agent] = {'error': str(e), 'status': 'unreachable'}
                
        return responses
        
    async def subscribe_to_agent_events(
        self,
        event_types: List[EventType],
        source_agents: Optional[List[str]] = None
    ):
        """
        Subscribe to events from specific agents.
        
        Args:
            event_types: List of event types to subscribe to
            source_agents: Optional list of source agent IDs to filter by
        """
        def event_filter(event):
            if source_agents:
                return event.source in source_agents
            return True
            
        for event_type in event_types:
            await self.event_bus.subscribe(
                subscriber_id=self.agent_id,
                event_types=event_type,
                handler=self._handle_agent_event,
                filter_condition=event_filter if source_agents else None
            )
            
    async def _send_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Send a message using the appropriate channel."""
        if message.channel == CommunicationChannel.DIRECT:
            return await self._send_direct_message(message)
        elif message.channel == CommunicationChannel.MULTICAST:
            return await self._send_multicast_message(message)
        elif message.channel == CommunicationChannel.BROADCAST:
            return await self._send_broadcast_message(message)
        elif message.channel == CommunicationChannel.EVENT_DRIVEN:
            return await self._send_event_message(message)
        elif message.channel == CommunicationChannel.REQUEST_RESPONSE:
            return await self._send_request_response_message(message)
        else:
            raise ValueError(f"Unsupported channel: {message.channel}")
            
    async def _send_direct_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Send a direct message to a single recipient."""
        if len(message.recipients) != 1:
            raise ValueError("Direct messages must have exactly one recipient")
            
        recipient = message.recipients[0]
        
        if message.reply_required:
            # Use request/response pattern
            response = await self.message_broker.send_request(
                sender=self.agent_id,
                recipient=recipient,
                request_payload=message.payload,
                timeout=message.timeout_seconds,
                priority=message.priority
            )
            self.stats['messages_sent'] += 1
            return response
        else:
            # Fire and forget
            await self.message_broker.send_message(
                message_type=MessageType.COMMAND if message.protocol_type == ProtocolType.COMMAND else MessageType.NOTIFICATION,
                sender=self.agent_id,
                recipient=recipient,
                payload=message.payload,
                priority=message.priority,
                correlation_id=message.correlation_id
            )
            self.stats['messages_sent'] += 1
            return None
            
    async def _send_multicast_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Send a message to multiple recipients."""
        responses = {}
        
        for recipient in message.recipients:
            try:
                if message.reply_required:
                    response = await self.message_broker.send_request(
                        sender=self.agent_id,
                        recipient=recipient,
                        request_payload=message.payload,
                        timeout=message.timeout_seconds,
                        priority=message.priority
                    )
                    responses[recipient] = response
                else:
                    await self.message_broker.send_message(
                        message_type=MessageType.NOTIFICATION,
                        sender=self.agent_id,
                        recipient=recipient,
                        payload=message.payload,
                        priority=message.priority,
                        correlation_id=message.correlation_id
                    )
                    
                self.stats['messages_sent'] += 1
                
            except Exception as e:
                logger.error(f"Failed to send message to {recipient}: {str(e)}")
                if message.reply_required:
                    responses[recipient] = {'error': str(e)}
                    
        return responses if message.reply_required else None
        
    async def _send_broadcast_message(self, message: AgentMessage) -> None:
        """Send a broadcast message via event bus."""
        event_type = EventType.SYSTEM_HEALTH_CHECK
        if message.protocol_type == ProtocolType.NOTIFICATION:
            event_type = EventType.AGENT_HEALTH_CHECK
            
        await self.event_bus.publish_agent_event(
            event_type=event_type,
            agent_id=self.agent_id,
            agent_type="communicator",
            source=self.agent_id,
            data=message.payload
        )
        
        self.stats['messages_sent'] += 1
        
    async def _send_event_message(self, message: AgentMessage) -> None:
        """Send a message via event bus."""
        event_type = EventType.AGENT_HEALTH_CHECK  # Default, could be parameterized
        
        await self.event_bus.publish_agent_event(
            event_type=event_type,
            agent_id=self.agent_id,
            agent_type="communicator",
            source=self.agent_id,
            data=message.payload
        )
        
        self.stats['messages_sent'] += 1
        
    async def _send_request_response_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Send a request/response message."""
        if len(message.recipients) != 1:
            raise ValueError("Request/response messages must have exactly one recipient")
            
        recipient = message.recipients[0]
        
        response = await self.message_broker.send_request(
            sender=self.agent_id,
            recipient=recipient,
            request_payload=message.payload,
            timeout=message.timeout_seconds,
            priority=message.priority
        )
        
        self.stats['messages_sent'] += 1
        return response
        
    async def _handle_incoming_message(self, message) -> Optional[Dict[str, Any]]:
        """Handle incoming messages from the message broker."""
        try:
            # Parse message payload to determine protocol type
            payload = message.payload
            protocol_type_str = payload.get('protocol_type')
            
            if protocol_type_str:
                protocol_type = ProtocolType(protocol_type_str)
                
                # Check if we have a handler for this protocol
                if protocol_type in self.protocol_handlers:
                    handler = self.protocol_handlers[protocol_type]
                    
                    # Create AgentMessage from broker message
                    agent_message = AgentMessage(
                        protocol_type=protocol_type,
                        sender=message.sender,
                        recipients=[self.agent_id],
                        channel=CommunicationChannel.DIRECT,
                        payload=payload,
                        correlation_id=message.correlation_id
                    )
                    
                    # Handle the message
                    response = await handler.handle_message(agent_message)
                    self.stats['messages_received'] += 1
                    
                    return response
                else:
                    logger.warning(f"No handler for protocol type: {protocol_type}")
            else:
                # Handle as generic message
                logger.debug(f"Received generic message from {message.sender}")
                self.stats['messages_received'] += 1
                
        except Exception as e:
            logger.error(f"Error handling incoming message: {str(e)}")
            
        return None
        
    async def _handle_agent_event(self, event):
        """Handle agent events from the event bus."""
        logger.debug(f"Received agent event: {event.event_type.value} from {event.source}")
        
        # Could implement specific event handling logic here
        # For now, just log the event
        
    def get_stats(self) -> Dict[str, Any]:
        """Get communication protocol statistics."""
        return {
            **self.stats,
            'registered_handlers': len(self.protocol_handlers),
            'active_conversations': len(self.active_conversations),
            'agent_id': self.agent_id
        }
        
    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active conversations."""
        return self.active_conversations.copy()


class DefaultProtocolHandler(ProtocolHandler):
    """Default protocol handler for basic message types."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
    async def handle_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming messages with default behavior."""
        if message.protocol_type == ProtocolType.HEALTH_CHECK:
            return await self._handle_health_check(message)
        elif message.protocol_type == ProtocolType.COMMAND:
            return await self._handle_command(message)
        elif message.protocol_type == ProtocolType.QUERY:
            return await self._handle_query(message)
        elif message.protocol_type == ProtocolType.NOTIFICATION:
            await self._handle_notification(message)
            return None
        else:
            logger.warning(f"Unhandled protocol type: {message.protocol_type}")
            return {'error': f'Unsupported protocol type: {message.protocol_type.value}'}
            
    def get_supported_protocols(self) -> List[ProtocolType]:
        """Get supported protocol types."""
        return [
            ProtocolType.HEALTH_CHECK,
            ProtocolType.COMMAND,
            ProtocolType.QUERY,
            ProtocolType.NOTIFICATION
        ]
        
    async def _handle_health_check(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle health check messages."""
        return {
            'agent_id': self.agent_id,
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'capabilities': ['communication', 'health_check'],
            'load': 'normal'
        }
        
    async def _handle_command(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle command messages."""
        command = message.payload.get('command')
        parameters = message.payload.get('parameters', {})
        
        # Default command handling - just acknowledge
        return {
            'command': command,
            'status': 'acknowledged',
            'message': f'Command {command} received but not implemented',
            'timestamp': datetime.now().isoformat()
        }
        
    async def _handle_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle query messages."""
        query = message.payload.get('query')
        
        # Default query handling - return empty result
        return {
            'query': query,
            'results': [],
            'message': f'Query {query} received but not implemented',
            'timestamp': datetime.now().isoformat()
        }
        
    async def _handle_notification(self, message: AgentMessage):
        """Handle notification messages."""
        notification_type = message.payload.get('notification_type')
        logger.info(f"Received notification: {notification_type} from {message.sender}")