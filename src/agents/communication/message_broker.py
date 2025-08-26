"""
Message Broker for Reliable Agent Communication

This module provides a message broker for reliable, ordered message delivery
between agents with support for priorities, persistence, and delivery guarantees.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from collections import defaultdict
import heapq

from src.agents.core.database_service import DatabaseService, get_db_service

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system."""
    COMMAND = "command"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    DATA_TRANSFER = "data_transfer"


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class DeliveryStatus(Enum):
    """Message delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Message:
    """Represents a message in the system."""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Support for priority queue ordering."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.priority.value < other.priority.value
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender': self.sender,
            'recipient': self.recipient,
            'payload': self.payload,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'headers': self.headers,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            sender=data['sender'],
            recipient=data['recipient'],
            payload=data['payload'],
            priority=MessagePriority(data['priority']),
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            headers=data.get('headers', {}),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3)
        )


@dataclass
class DeliveryResult:
    """Result of message delivery attempt."""
    message_id: str
    status: DeliveryStatus
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    acknowledged_at: Optional[datetime] = None


MessageHandler = Callable[[Message], Union[Any, Dict[str, Any]]]
AsyncMessageHandler = Callable[[Message], Any]  # Can return awaitable


class MessageBroker:
    """
    Reliable message broker for agent communication.
    
    Provides guaranteed message delivery, ordering, priority handling,
    and persistence for agent-to-agent communication.
    """
    
    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db_service = db_service or get_db_service()
        self.message_queues: Dict[str, List[Message]] = defaultdict(list)
        self.handlers: Dict[str, MessageHandler] = {}
        self.delivery_results: Dict[str, DeliveryResult] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'active_queues': 0
        }
        self._running = False
        
    async def start(self):
        """Start the message broker."""
        if self._running:
            return
            
        self._running = True
        logger.info("Message broker started")
        
        # Restore messages from database
        await self._restore_persistent_messages()
        
    async def stop(self):
        """Stop the message broker."""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
            
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
            
        # Persist pending messages
        await self._persist_pending_messages()
        
        logger.info("Message broker stopped")
        
    async def send_message(
        self,
        message_type: MessageType,
        sender: str,
        recipient: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        expires_in: Optional[timedelta] = None,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        persistent: bool = True
    ) -> str:
        """
        Send a message to a recipient.
        
        Args:
            message_type: Type of the message
            sender: Sender identifier
            recipient: Recipient identifier
            payload: Message payload
            priority: Message priority
            expires_in: Message expiration time
            correlation_id: Optional correlation ID
            reply_to: Optional reply-to address
            headers: Optional message headers
            persistent: Whether to persist the message
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + expires_in
            
        message = Message(
            message_id=message_id,
            message_type=message_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=priority,
            expires_at=expires_at,
            correlation_id=correlation_id,
            reply_to=reply_to,
            headers=headers or {}
        )
        
        # Add to recipient's queue
        heapq.heappush(self.message_queues[recipient], message)
        
        # Persist message if requested
        if persistent:
            await self._persist_message(message)
            
        # Start processing for this recipient if not already running
        if recipient not in self.processing_tasks:
            self.processing_tasks[recipient] = asyncio.create_task(
                self._process_messages_for_recipient(recipient)
            )
            
        self.stats['messages_sent'] += 1
        logger.debug(f"Sent message {message_id} from {sender} to {recipient}")
        
        return message_id
        
    async def register_handler(
        self,
        recipient: str,
        handler: Union[MessageHandler, AsyncMessageHandler]
    ):
        """
        Register a message handler for a recipient.
        
        Args:
            recipient: Recipient identifier
            handler: Function to handle messages
        """
        self.handlers[recipient] = handler
        logger.info(f"Registered handler for {recipient}")
        
        # Start processing if there are pending messages
        if recipient in self.message_queues and self.message_queues[recipient]:
            if recipient not in self.processing_tasks:
                self.processing_tasks[recipient] = asyncio.create_task(
                    self._process_messages_for_recipient(recipient)
                )
                
    async def unregister_handler(self, recipient: str):
        """
        Unregister a message handler.
        
        Args:
            recipient: Recipient identifier
        """
        if recipient in self.handlers:
            del self.handlers[recipient]
            
        # Cancel processing task
        if recipient in self.processing_tasks:
            self.processing_tasks[recipient].cancel()
            try:
                await self.processing_tasks[recipient]
            except asyncio.CancelledError:
                pass
            del self.processing_tasks[recipient]
            
        logger.info(f"Unregistered handler for {recipient}")
        
    async def send_request(
        self,
        sender: str,
        recipient: str,
        request_payload: Dict[str, Any],
        timeout: Optional[float] = 30.0,
        priority: MessagePriority = MessagePriority.HIGH
    ) -> Dict[str, Any]:
        """
        Send a request and wait for response.
        
        Args:
            sender: Sender identifier
            recipient: Recipient identifier
            request_payload: Request payload
            timeout: Response timeout in seconds
            priority: Message priority
            
        Returns:
            Response payload
        """
        correlation_id = str(uuid.uuid4())
        response_future = asyncio.Future()
        
        # Store future for response correlation
        response_key = f"{sender}_{correlation_id}"
        self._response_futures = getattr(self, '_response_futures', {})
        self._response_futures[response_key] = response_future
        
        # Send request
        await self.send_message(
            message_type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            payload=request_payload,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=sender,
            expires_in=timedelta(seconds=timeout) if timeout else None
        )
        
        try:
            # Wait for response
            if timeout:
                response = await asyncio.wait_for(response_future, timeout=timeout)
            else:
                response = await response_future
                
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {recipient} timed out")
        finally:
            # Cleanup
            if response_key in self._response_futures:
                del self._response_futures[response_key]
                
    async def send_response(
        self,
        sender: str,
        original_message: Message,
        response_payload: Dict[str, Any]
    ):
        """
        Send a response to a request message.
        
        Args:
            sender: Sender identifier
            original_message: Original request message
            response_payload: Response payload
        """
        if not original_message.reply_to:
            logger.warning("Cannot send response: no reply_to address")
            return
            
        await self.send_message(
            message_type=MessageType.RESPONSE,
            sender=sender,
            recipient=original_message.reply_to,
            payload=response_payload,
            correlation_id=original_message.correlation_id,
            priority=MessagePriority.HIGH
        )
        
    async def acknowledge_message(self, message_id: str, recipient: str):
        """
        Acknowledge receipt of a message.
        
        Args:
            message_id: ID of the message to acknowledge
            recipient: Recipient who is acknowledging
        """
        if message_id in self.delivery_results:
            self.delivery_results[message_id].status = DeliveryStatus.ACKNOWLEDGED
            self.delivery_results[message_id].acknowledged_at = datetime.now()
            
            # Update in database
            await self._update_delivery_status(message_id, DeliveryStatus.ACKNOWLEDGED)
            
        logger.debug(f"Message {message_id} acknowledged by {recipient}")
        
    async def get_delivery_status(self, message_id: str) -> Optional[DeliveryResult]:
        """
        Get the delivery status of a message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            Delivery result or None if not found
        """
        return self.delivery_results.get(message_id)
        
    async def get_pending_messages(self, recipient: str) -> List[Message]:
        """
        Get pending messages for a recipient.
        
        Args:
            recipient: Recipient identifier
            
        Returns:
            List of pending messages
        """
        return list(self.message_queues.get(recipient, []))
        
    async def _process_messages_for_recipient(self, recipient: str):
        """Process messages for a specific recipient."""
        while self._running and recipient in self.message_queues:
            try:
                # Get next message
                if not self.message_queues[recipient]:
                    await asyncio.sleep(0.1)
                    continue
                    
                message = heapq.heappop(self.message_queues[recipient])
                
                # Check if message has expired
                if message.is_expired():
                    await self._mark_message_expired(message)
                    continue
                    
                # Attempt delivery
                await self._deliver_message(message)
                
            except Exception as e:
                logger.error(f"Error processing messages for {recipient}: {str(e)}")
                await asyncio.sleep(1)
                
        # Cleanup
        if recipient in self.processing_tasks:
            del self.processing_tasks[recipient]
            
    async def _deliver_message(self, message: Message):
        """Deliver a message to its recipient."""
        try:
            # Check if handler is registered
            if message.recipient not in self.handlers:
                # Requeue message and wait
                heapq.heappush(self.message_queues[message.recipient], message)
                await asyncio.sleep(1)
                return
                
            handler = self.handlers[message.recipient]
            
            # Handle response correlation
            if message.message_type == MessageType.RESPONSE and message.correlation_id:
                await self._handle_response_message(message)
                return
                
            # Call handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message)
            else:
                result = handler(message)
                
            # Mark as delivered
            await self._mark_message_delivered(message, result)
            
        except Exception as e:
            logger.error(f"Failed to deliver message {message.message_id}: {str(e)}")
            await self._handle_delivery_failure(message, str(e))
            
    async def _handle_response_message(self, message: Message):
        """Handle a response message by resolving the corresponding future."""
        response_futures = getattr(self, '_response_futures', {})
        response_key = f"{message.recipient}_{message.correlation_id}"
        
        if response_key in response_futures:
            future = response_futures[response_key]
            if not future.done():
                future.set_result(message.payload)
                
        await self._mark_message_delivered(message)
        
    async def _mark_message_delivered(self, message: Message, result: Any = None):
        """Mark a message as delivered."""
        delivery_result = DeliveryResult(
            message_id=message.message_id,
            status=DeliveryStatus.DELIVERED,
            delivered_at=datetime.now(),
            retry_count=message.retry_count
        )
        
        self.delivery_results[message.message_id] = delivery_result
        await self._update_delivery_status(message.message_id, DeliveryStatus.DELIVERED)
        
        self.stats['messages_delivered'] += 1
        logger.debug(f"Message {message.message_id} delivered successfully")
        
    async def _mark_message_expired(self, message: Message):
        """Mark a message as expired."""
        delivery_result = DeliveryResult(
            message_id=message.message_id,
            status=DeliveryStatus.EXPIRED,
            error_message="Message expired",
            retry_count=message.retry_count
        )
        
        self.delivery_results[message.message_id] = delivery_result
        await self._update_delivery_status(message.message_id, DeliveryStatus.EXPIRED)
        
        logger.warning(f"Message {message.message_id} expired")
        
    async def _handle_delivery_failure(self, message: Message, error: str):
        """Handle message delivery failure."""
        message.retry_count += 1
        
        if message.retry_count <= message.max_retries:
            # Requeue for retry
            heapq.heappush(self.message_queues[message.recipient], message)
            logger.info(f"Retrying message {message.message_id} (attempt {message.retry_count})")
        else:
            # Mark as failed
            delivery_result = DeliveryResult(
                message_id=message.message_id,
                status=DeliveryStatus.FAILED,
                error_message=error,
                retry_count=message.retry_count
            )
            
            self.delivery_results[message.message_id] = delivery_result
            await self._update_delivery_status(message.message_id, DeliveryStatus.FAILED)
            
            self.stats['messages_failed'] += 1
            logger.error(f"Message {message.message_id} failed after {message.retry_count} attempts")
            
    async def _persist_message(self, message: Message):
        """Persist a message to database."""
        try:
            async with self.db_service.get_connection() as conn:
                query = """
                INSERT INTO agent_messages 
                (message_id, message_type, sender, recipient, payload, priority,
                 created_at, expires_at, correlation_id, reply_to, headers,
                 retry_count, max_retries, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """
                
                await conn.execute(
                    query,
                    message.message_id,
                    message.message_type.value,
                    message.sender,
                    message.recipient,
                    json.dumps(message.payload),
                    message.priority.value,
                    message.created_at,
                    message.expires_at,
                    message.correlation_id,
                    message.reply_to,
                    json.dumps(message.headers),
                    message.retry_count,
                    message.max_retries,
                    DeliveryStatus.PENDING.value
                )
                
        except Exception as e:
            logger.error(f"Failed to persist message: {str(e)}")
            
    async def _update_delivery_status(self, message_id: str, status: DeliveryStatus):
        """Update message delivery status in database."""
        try:
            async with self.db_service.get_connection() as conn:
                query = """
                UPDATE agent_messages 
                SET status = $1, updated_at = $2
                WHERE message_id = $3
                """
                
                await conn.execute(query, status.value, datetime.now(), message_id)
                
        except Exception as e:
            logger.error(f"Failed to update delivery status: {str(e)}")
            
    async def _restore_persistent_messages(self):
        """Restore pending messages from database."""
        try:
            async with self.db_service.get_connection() as conn:
                query = """
                SELECT message_id, message_type, sender, recipient, payload, priority,
                       created_at, expires_at, correlation_id, reply_to, headers,
                       retry_count, max_retries
                FROM agent_messages 
                WHERE status = $1
                ORDER BY priority, created_at
                """
                
                rows = await conn.fetch(query, DeliveryStatus.PENDING.value)
                
                for row in rows:
                    message = Message(
                        message_id=row['message_id'],
                        message_type=MessageType(row['message_type']),
                        sender=row['sender'],
                        recipient=row['recipient'],
                        payload=json.loads(row['payload']),
                        priority=MessagePriority(row['priority']),
                        created_at=row['created_at'],
                        expires_at=row['expires_at'],
                        correlation_id=row['correlation_id'],
                        reply_to=row['reply_to'],
                        headers=json.loads(row['headers'] or '{}'),
                        retry_count=row['retry_count'],
                        max_retries=row['max_retries']
                    )
                    
                    # Skip expired messages
                    if message.is_expired():
                        await self._mark_message_expired(message)
                        continue
                        
                    # Add to queue
                    heapq.heappush(self.message_queues[message.recipient], message)
                    
            logger.info(f"Restored {len(rows)} pending messages from database")
            
        except Exception as e:
            logger.error(f"Failed to restore messages: {str(e)}")
            
    async def _persist_pending_messages(self):
        """Persist all pending messages to database."""
        try:
            for recipient, messages in self.message_queues.items():
                for message in messages:
                    await self._persist_message(message)
                    
        except Exception as e:
            logger.error(f"Failed to persist pending messages: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get message broker statistics."""
        queue_sizes = {
            recipient: len(messages) 
            for recipient, messages in self.message_queues.items()
        }
        
        return {
            **self.stats,
            'active_queues': len([q for q in self.message_queues.values() if q]),
            'total_queued_messages': sum(len(q) for q in self.message_queues.values()),
            'registered_handlers': len(self.handlers),
            'queue_sizes': queue_sizes,
            'processing_tasks': len(self.processing_tasks),
            'is_running': self._running
        }


# Global message broker instance
message_broker = MessageBroker()


async def get_message_broker() -> MessageBroker:
    """Get the global message broker instance."""
    if not message_broker._running:
        await message_broker.start()
    return message_broker