"""
Event Logger for Communication Audit and Monitoring

This module provides comprehensive logging and auditing capabilities for
agent communication events, enabling debugging, monitoring, and compliance.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from src.agents.core.database_service import DatabaseService, get_db_service
from .event_bus import Event, EventType

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    """Categories for log entries."""
    COMMUNICATION = "communication"
    WORKFLOW = "workflow"
    AGENT = "agent"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR = "error"


@dataclass
class LogEntry:
    """Represents a log entry in the system."""
    log_id: str
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    source: str
    event_type: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'source': self.source,
            'event_type': self.event_type,
            'message': self.message,
            'data': self.data,
            'correlation_id': self.correlation_id,
            'trace_id': self.trace_id,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create from dictionary."""
        return cls(
            log_id=data['log_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            level=LogLevel(data['level']),
            category=LogCategory(data['category']),
            source=data['source'],
            event_type=data.get('event_type'),
            message=data.get('message', ''),
            data=data.get('data', {}),
            correlation_id=data.get('correlation_id'),
            trace_id=data.get('trace_id'),
            tags=data.get('tags', [])
        )


@dataclass
class AuditTrail:
    """Represents an audit trail for a specific process or entity."""
    trail_id: str
    entity_type: str
    entity_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "active"
    entries: List[LogEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entry(self, entry: LogEntry):
        """Add a log entry to the trail."""
        self.entries.append(entry)
        
    def complete(self, status: str = "completed"):
        """Mark the audit trail as completed."""
        self.completed_at = datetime.now()
        self.status = status
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trail_id': self.trail_id,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'entries': [entry.to_dict() for entry in self.entries],
            'metadata': self.metadata
        }


class EventLogger:
    """
    Comprehensive event logger for agent communication and system events.
    
    Provides structured logging, audit trails, and monitoring capabilities
    for debugging and compliance purposes.
    """
    
    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db_service = db_service or get_db_service()
        self.log_buffer: List[LogEntry] = []
        self.audit_trails: Dict[str, AuditTrail] = {}
        self.buffer_size = 1000
        self.flush_interval = 60  # seconds
        self.stats = {
            'entries_logged': 0,
            'entries_flushed': 0,
            'audit_trails_created': 0,
            'audit_trails_completed': 0
        }
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the event logger."""
        if self._running:
            return
            
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Event logger started")
        
    async def stop(self):
        """Stop the event logger."""
        if not self._running:
            return
            
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
                
        # Flush remaining entries
        await self._flush_logs()
        
        logger.info("Event logger stopped")
        
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        source: str,
        message: str,
        event_type: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Log an event.
        
        Args:
            level: Log level
            category: Log category
            source: Source of the event
            message: Log message
            event_type: Optional event type
            data: Optional event data
            correlation_id: Optional correlation ID
            trace_id: Optional trace ID
            tags: Optional tags
            
        Returns:
            Log entry ID
        """
        log_id = str(uuid.uuid4())
        
        entry = LogEntry(
            log_id=log_id,
            timestamp=datetime.now(),
            level=level,
            category=category,
            source=source,
            event_type=event_type,
            message=message,
            data=data or {},
            correlation_id=correlation_id,
            trace_id=trace_id,
            tags=tags or []
        )
        
        self.log_buffer.append(entry)
        self.stats['entries_logged'] += 1
        
        # Trigger flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_logs())
            
        return log_id
        
    def log_event(self, event: Event, additional_data: Optional[Dict[str, Any]] = None):
        """
        Log an event from the event bus.
        
        Args:
            event: Event to log
            additional_data: Additional data to include
        """
        data = event.data.copy()
        if additional_data:
            data.update(additional_data)
            
        # Determine log level based on event type
        level = LogLevel.INFO
        if 'error' in event.event_type.value or 'failed' in event.event_type.value:
            level = LogLevel.ERROR
        elif 'warning' in event.event_type.value:
            level = LogLevel.WARNING
            
        # Determine category based on event type
        category = LogCategory.SYSTEM
        if 'agent' in event.event_type.value:
            category = LogCategory.AGENT
        elif 'workflow' in event.event_type.value:
            category = LogCategory.WORKFLOW
        elif 'campaign' in event.event_type.value:
            category = LogCategory.WORKFLOW
        elif 'performance' in event.event_type.value:
            category = LogCategory.PERFORMANCE
            
        self.log(
            level=level,
            category=category,
            source=event.source,
            message=f"Event: {event.event_type.value}",
            event_type=event.event_type.value,
            data=data,
            correlation_id=event.correlation_id,
            trace_id=event.trace_id
        )
        
    def log_communication(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Log a communication event.
        
        Args:
            sender: Message sender
            recipient: Message recipient
            message_type: Type of message
            success: Whether communication was successful
            details: Additional details
            correlation_id: Optional correlation ID
        """
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Communication {message_type}: {sender} -> {recipient}"
        
        data = {
            'sender': sender,
            'recipient': recipient,
            'message_type': message_type,
            'success': success
        }
        if details:
            data.update(details)
            
        self.log(
            level=level,
            category=LogCategory.COMMUNICATION,
            source=sender,
            message=message,
            event_type="communication",
            data=data,
            correlation_id=correlation_id,
            tags=['communication', message_type]
        )
        
    def log_workflow_step(
        self,
        workflow_id: str,
        step_name: str,
        agent_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Log a workflow step.
        
        Args:
            workflow_id: Workflow identifier
            step_name: Name of the workflow step
            agent_id: Agent executing the step
            status: Step status
            details: Additional details
            correlation_id: Optional correlation ID
        """
        level = LogLevel.INFO
        if status in ['failed', 'error']:
            level = LogLevel.ERROR
        elif status == 'warning':
            level = LogLevel.WARNING
            
        message = f"Workflow step {step_name} in {workflow_id}: {status}"
        
        data = {
            'workflow_id': workflow_id,
            'step_name': step_name,
            'agent_id': agent_id,
            'status': status
        }
        if details:
            data.update(details)
            
        self.log(
            level=level,
            category=LogCategory.WORKFLOW,
            source=agent_id,
            message=message,
            event_type="workflow_step",
            data=data,
            correlation_id=correlation_id or workflow_id,
            tags=['workflow', step_name, status]
        )
        
    def log_performance(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: str,
        source: str,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            source: Source of the metric
            threshold: Optional threshold for alerts
            metadata: Optional metadata
        """
        level = LogLevel.INFO
        if threshold and value > threshold:
            level = LogLevel.WARNING
            
        message = f"Performance metric {metric_name}: {value} {unit}"
        
        data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'threshold': threshold
        }
        if metadata:
            data.update(metadata)
            
        self.log(
            level=level,
            category=LogCategory.PERFORMANCE,
            source=source,
            message=message,
            event_type="performance_metric",
            data=data,
            tags=['performance', metric_name]
        )
        
    def log_error(
        self,
        source: str,
        error_message: str,
        error_type: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Log an error.
        
        Args:
            source: Source of the error
            error_message: Error message
            error_type: Type of error
            stack_trace: Optional stack trace
            context: Optional error context
            correlation_id: Optional correlation ID
        """
        data = {
            'error_message': error_message,
            'error_type': error_type,
            'stack_trace': stack_trace
        }
        if context:
            data.update(context)
            
        self.log(
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            source=source,
            message=f"Error: {error_message}",
            event_type="error",
            data=data,
            correlation_id=correlation_id,
            tags=['error', error_type] if error_type else ['error']
        )
        
    def start_audit_trail(
        self,
        entity_type: str,
        entity_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start an audit trail for an entity.
        
        Args:
            entity_type: Type of entity being audited
            entity_id: Unique identifier for the entity
            metadata: Optional metadata
            
        Returns:
            Audit trail ID
        """
        trail_id = str(uuid.uuid4())
        
        trail = AuditTrail(
            trail_id=trail_id,
            entity_type=entity_type,
            entity_id=entity_id,
            started_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.audit_trails[trail_id] = trail
        self.stats['audit_trails_created'] += 1
        
        # Log the start of the audit trail
        self.log(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            source="event_logger",
            message=f"Started audit trail for {entity_type} {entity_id}",
            event_type="audit_trail_started",
            data={
                'trail_id': trail_id,
                'entity_type': entity_type,
                'entity_id': entity_id
            },
            correlation_id=entity_id,
            tags=['audit', entity_type]
        )
        
        return trail_id
        
    def add_to_audit_trail(
        self,
        trail_id: str,
        entry: LogEntry
    ):
        """
        Add an entry to an audit trail.
        
        Args:
            trail_id: Audit trail ID
            entry: Log entry to add
        """
        if trail_id in self.audit_trails:
            self.audit_trails[trail_id].add_entry(entry)
            
    def complete_audit_trail(
        self,
        trail_id: str,
        status: str = "completed"
    ):
        """
        Complete an audit trail.
        
        Args:
            trail_id: Audit trail ID
            status: Final status
        """
        if trail_id in self.audit_trails:
            trail = self.audit_trails[trail_id]
            trail.complete(status)
            self.stats['audit_trails_completed'] += 1
            
            # Log the completion
            self.log(
                level=LogLevel.INFO,
                category=LogCategory.SYSTEM,
                source="event_logger",
                message=f"Completed audit trail {trail_id} with status {status}",
                event_type="audit_trail_completed",
                data={
                    'trail_id': trail_id,
                    'entity_type': trail.entity_type,
                    'entity_id': trail.entity_id,
                    'status': status,
                    'duration_seconds': (trail.completed_at - trail.started_at).total_seconds() if trail.completed_at else None,
                    'entry_count': len(trail.entries)
                },
                correlation_id=trail.entity_id,
                tags=['audit', trail.entity_type, status]
            )
            
    async def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        levels: Optional[List[LogLevel]] = None,
        categories: Optional[List[LogCategory]] = None,
        sources: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[LogEntry]:
        """
        Query log entries.
        
        Args:
            start_time: Start time for query
            end_time: End time for query
            levels: Log levels to include
            categories: Categories to include
            sources: Sources to include
            correlation_id: Correlation ID to filter by
            tags: Tags to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of matching log entries
        """
        try:
            async with self.db_service.get_connection() as conn:
                query_parts = ["SELECT * FROM communication_logs WHERE 1=1"]
                params = []
                param_count = 0
                
                if start_time:
                    param_count += 1
                    query_parts.append(f"AND timestamp >= ${param_count}")
                    params.append(start_time)
                    
                if end_time:
                    param_count += 1
                    query_parts.append(f"AND timestamp <= ${param_count}")
                    params.append(end_time)
                    
                if levels:
                    param_count += 1
                    query_parts.append(f"AND level = ANY(${param_count})")
                    params.append([level.value for level in levels])
                    
                if categories:
                    param_count += 1
                    query_parts.append(f"AND category = ANY(${param_count})")
                    params.append([cat.value for cat in categories])
                    
                if sources:
                    param_count += 1
                    query_parts.append(f"AND source = ANY(${param_count})")
                    params.append(sources)
                    
                if correlation_id:
                    param_count += 1
                    query_parts.append(f"AND correlation_id = ${param_count}")
                    params.append(correlation_id)
                    
                if tags:
                    param_count += 1
                    query_parts.append(f"AND tags && ${param_count}")
                    params.append(tags)
                    
                query_parts.append("ORDER BY timestamp DESC")
                query_parts.append(f"LIMIT {limit}")
                
                query = " ".join(query_parts)
                rows = await conn.fetch(query, *params)
                
                entries = []
                for row in rows:
                    entry = LogEntry(
                        log_id=row['log_id'],
                        timestamp=row['timestamp'],
                        level=LogLevel(row['level']),
                        category=LogCategory(row['category']),
                        source=row['source'],
                        event_type=row['event_type'],
                        message=row['message'],
                        data=json.loads(row['data'] or '{}'),
                        correlation_id=row['correlation_id'],
                        trace_id=row['trace_id'],
                        tags=row['tags'] or []
                    )
                    entries.append(entry)
                    
                return entries
                
        except Exception as e:
            logger.error(f"Failed to query logs: {str(e)}")
            return []
            
    async def get_audit_trail(self, trail_id: str) -> Optional[AuditTrail]:
        """
        Get an audit trail by ID.
        
        Args:
            trail_id: Audit trail ID
            
        Returns:
            Audit trail or None if not found
        """
        # Check in-memory trails first
        if trail_id in self.audit_trails:
            return self.audit_trails[trail_id]
            
        # Query database for completed trails
        try:
            async with self.db_service.get_connection() as conn:
                query = """
                SELECT trail_id, entity_type, entity_id, started_at, completed_at,
                       status, metadata
                FROM audit_trails 
                WHERE trail_id = $1
                """
                
                row = await conn.fetchrow(query, trail_id)
                if not row:
                    return None
                    
                # Get entries for this trail
                entries_query = """
                SELECT log_id, timestamp, level, category, source, event_type,
                       message, data, correlation_id, trace_id, tags
                FROM communication_logs 
                WHERE correlation_id = $1
                ORDER BY timestamp
                """
                
                entry_rows = await conn.fetch(entries_query, trail_id)
                entries = []
                
                for entry_row in entry_rows:
                    entry = LogEntry(
                        log_id=entry_row['log_id'],
                        timestamp=entry_row['timestamp'],
                        level=LogLevel(entry_row['level']),
                        category=LogCategory(entry_row['category']),
                        source=entry_row['source'],
                        event_type=entry_row['event_type'],
                        message=entry_row['message'],
                        data=json.loads(entry_row['data'] or '{}'),
                        correlation_id=entry_row['correlation_id'],
                        trace_id=entry_row['trace_id'],
                        tags=entry_row['tags'] or []
                    )
                    entries.append(entry)
                    
                trail = AuditTrail(
                    trail_id=row['trail_id'],
                    entity_type=row['entity_type'],
                    entity_id=row['entity_id'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    status=row['status'],
                    entries=entries,
                    metadata=json.loads(row['metadata'] or '{}')
                )
                
                return trail
                
        except Exception as e:
            logger.error(f"Failed to get audit trail: {str(e)}")
            return None
            
    async def _periodic_flush(self):
        """Periodically flush logs to database."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {str(e)}")
                
    async def _flush_logs(self):
        """Flush log buffer to database."""
        if not self.log_buffer:
            return
            
        entries_to_flush = self.log_buffer.copy()
        self.log_buffer.clear()
        
        try:
            async with self.db_service.get_connection() as conn:
                # Insert log entries
                for entry in entries_to_flush:
                    query = """
                    INSERT INTO communication_logs 
                    (log_id, timestamp, level, category, source, event_type,
                     message, data, correlation_id, trace_id, tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """
                    
                    await conn.execute(
                        query,
                        entry.log_id,
                        entry.timestamp,
                        entry.level.value,
                        entry.category.value,
                        entry.source,
                        entry.event_type,
                        entry.message,
                        json.dumps(entry.data),
                        entry.correlation_id,
                        entry.trace_id,
                        entry.tags
                    )
                    
                # Insert/update audit trails
                for trail in self.audit_trails.values():
                    if trail.completed_at:  # Only persist completed trails
                        trail_query = """
                        INSERT INTO audit_trails 
                        (trail_id, entity_type, entity_id, started_at, completed_at,
                         status, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (trail_id) DO UPDATE SET
                            completed_at = EXCLUDED.completed_at,
                            status = EXCLUDED.status,
                            metadata = EXCLUDED.metadata
                        """
                        
                        await conn.execute(
                            trail_query,
                            trail.trail_id,
                            trail.entity_type,
                            trail.entity_id,
                            trail.started_at,
                            trail.completed_at,
                            trail.status,
                            json.dumps(trail.metadata)
                        )
                        
            self.stats['entries_flushed'] += len(entries_to_flush)
            
        except Exception as e:
            logger.error(f"Failed to flush logs: {str(e)}")
            # Put entries back in buffer on failure
            self.log_buffer.extend(entries_to_flush)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get event logger statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.log_buffer),
            'active_audit_trails': len([t for t in self.audit_trails.values() if not t.completed_at]),
            'completed_audit_trails': len([t for t in self.audit_trails.values() if t.completed_at]),
            'is_running': self._running
        }


# Global event logger instance
event_logger = EventLogger()


async def get_event_logger() -> EventLogger:
    """Get the global event logger instance."""
    if not event_logger._running:
        await event_logger.start()
    return event_logger