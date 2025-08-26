"""
Agent Communication System for Campaign Orchestration

This package provides event-driven communication protocols for agent coordination,
real-time messaging, and campaign workflow communication.

Key Components:
- EventBus: Central event publishing and subscription system
- MessageBroker: Reliable message delivery between agents
- CommunicationProtocol: Standardized communication interfaces
- EventLogger: Event tracking and audit capabilities
"""

from .event_bus import (
    EventBus,
    Event,
    EventType,
    EventHandler,
    EventSubscription
)

from .message_broker import (
    MessageBroker,
    Message,
    MessageType,
    MessagePriority,
    DeliveryResult
)

from .communication_protocol import (
    CommunicationProtocol,
    AgentMessage,
    ProtocolHandler,
    CommunicationChannel
)

from .event_logger import (
    EventLogger,
    LogEntry,
    LogLevel,
    AuditTrail
)

__version__ = "1.0.0"
__author__ = "CrediLinq Development Team"

__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "EventType", 
    "EventHandler",
    "EventSubscription",
    
    # Message Broker
    "MessageBroker",
    "Message",
    "MessageType",
    "MessagePriority",
    "DeliveryResult",
    
    # Communication Protocol
    "CommunicationProtocol",
    "AgentMessage",
    "ProtocolHandler",
    "CommunicationChannel",
    
    # Event Logger
    "EventLogger",
    "LogEntry",
    "LogLevel",
    "AuditTrail"
]