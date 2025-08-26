#!/usr/bin/env python3
"""
Agent Communication Bus
Enables intelligent communication and coordination between agents in workflows
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of inter-agent messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    QUALITY_FEEDBACK = "quality_feedback"
    COLLABORATION_REQUEST = "collaboration_request"
    STATUS_UPDATE = "status_update"
    ERROR_NOTIFICATION = "error_notification"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentMessage:
    """Structured message between agents"""
    message_id: str
    sender_agent: str
    recipient_agent: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class AgentCapability:
    """Agent capability description for discovery"""
    agent_type: str
    capabilities: List[str]
    specializations: List[str]
    input_types: List[str]
    output_types: List[str]
    quality_score: float
    performance_metrics: Dict[str, Any]

class AgentCommunicationBus:
    """
    Central communication hub for agent coordination and collaboration
    """
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.agent_registry = {}  # agent_id -> AgentCapability
        self.message_history = {}  # conversation_id -> List[AgentMessage]
        self.active_conversations = {}  # conversation_id -> metadata
        self.performance_tracker = {}
        self.knowledge_cache = {}  # Shared knowledge between agents
        
    async def register_agent(self, agent_id: str, capabilities: AgentCapability):
        """Register an agent with its capabilities"""
        self.agent_registry[agent_id] = capabilities
        logger.info(f"🤖 Agent registered: {agent_id} with {len(capabilities.capabilities)} capabilities")
        
    async def send_message(self, message: AgentMessage) -> str:
        """Send a message to another agent"""
        message.message_id = str(uuid.uuid4())
        message.timestamp = datetime.now()
        
        # Add to queue
        await self.message_queue.put(message)
        
        # Track conversation
        if message.correlation_id:
            if message.correlation_id not in self.message_history:
                self.message_history[message.correlation_id] = []
            self.message_history[message.correlation_id].append(message)
        
        logger.debug(f"📤 Message sent: {message.sender_agent} -> {message.recipient_agent}")
        return message.message_id
        
    async def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get messages for a specific agent"""
        messages = []
        temp_messages = []
        
        # Collect all messages from queue
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                if message.recipient_agent == agent_id or message.recipient_agent == "all":
                    messages.append(message)
                else:
                    temp_messages.append(message)
            except asyncio.QueueEmpty:
                break
        
        # Put back messages not for this agent
        for msg in temp_messages:
            await self.message_queue.put(msg)
            
        return messages
    
    async def request_collaboration(self, 
                                 requesting_agent: str,
                                 task_description: str,
                                 required_capabilities: List[str],
                                 context: Dict[str, Any]) -> List[str]:
        """Request collaboration from agents with specific capabilities"""
        
        # Find suitable agents
        suitable_agents = []
        for agent_id, capabilities in self.agent_registry.items():
            if agent_id != requesting_agent:
                # Check if agent has required capabilities
                if any(cap in capabilities.capabilities for cap in required_capabilities):
                    suitable_agents.append(agent_id)
        
        # Send collaboration requests
        collaboration_id = str(uuid.uuid4())
        for agent_id in suitable_agents:
            message = AgentMessage(
                message_id="",  # Will be set by send_message
                sender_agent=requesting_agent,
                recipient_agent=agent_id,
                message_type=MessageType.COLLABORATION_REQUEST,
                priority=MessagePriority.NORMAL,
                content={
                    "task_description": task_description,
                    "required_capabilities": required_capabilities,
                    "collaboration_id": collaboration_id
                },
                context=context,
                timestamp=datetime.now(),
                requires_response=True,
                correlation_id=collaboration_id
            )
            await self.send_message(message)
        
        logger.info(f"🤝 Collaboration requested by {requesting_agent} from {len(suitable_agents)} agents")
        return suitable_agents
    
    async def share_knowledge(self, 
                            sharing_agent: str,
                            knowledge_type: str,
                            knowledge_data: Dict[str, Any],
                            target_agents: Optional[List[str]] = None):
        """Share knowledge between agents"""
        
        # Store in knowledge cache
        knowledge_key = f"{knowledge_type}_{datetime.now().isoformat()}"
        self.knowledge_cache[knowledge_key] = {
            'source_agent': sharing_agent,
            'type': knowledge_type,
            'data': knowledge_data,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        # Send knowledge share messages
        recipients = target_agents or ["all"]
        for recipient in recipients:
            message = AgentMessage(
                message_id="",
                sender_agent=sharing_agent,
                recipient_agent=recipient,
                message_type=MessageType.KNOWLEDGE_SHARE,
                priority=MessagePriority.NORMAL,
                content={
                    "knowledge_key": knowledge_key,
                    "knowledge_type": knowledge_type,
                    "summary": knowledge_data.get("summary", "Shared knowledge")
                },
                context={"source": sharing_agent},
                timestamp=datetime.now()
            )
            await self.send_message(message)
        
        logger.info(f"🧠 Knowledge shared by {sharing_agent}: {knowledge_type}")
    
    async def get_shared_knowledge(self, 
                                 requesting_agent: str,
                                 knowledge_type: str) -> List[Dict[str, Any]]:
        """Retrieve shared knowledge of a specific type"""
        relevant_knowledge = []
        
        for key, knowledge in self.knowledge_cache.items():
            if knowledge['type'] == knowledge_type:
                # Update access count
                knowledge['access_count'] += 1
                relevant_knowledge.append(knowledge)
        
        logger.debug(f"🔍 {requesting_agent} retrieved {len(relevant_knowledge)} knowledge items")
        return relevant_knowledge
    
    async def provide_quality_feedback(self,
                                     reviewer_agent: str,
                                     target_agent: str,
                                     content_id: str,
                                     feedback: Dict[str, Any],
                                     quality_score: float):
        """Provide quality feedback on agent output"""
        
        message = AgentMessage(
            message_id="",
            sender_agent=reviewer_agent,
            recipient_agent=target_agent,
            message_type=MessageType.QUALITY_FEEDBACK,
            priority=MessagePriority.HIGH if quality_score < 7.0 else MessagePriority.NORMAL,
            content={
                "content_id": content_id,
                "quality_score": quality_score,
                "feedback": feedback,
                "suggestions": feedback.get("improvements", [])
            },
            context={"review_type": "quality_assessment"},
            timestamp=datetime.now(),
            requires_response=True
        )
        
        await self.send_message(message)
        logger.info(f"📝 Quality feedback provided: {reviewer_agent} -> {target_agent} (score: {quality_score})")
    
    async def discover_agents(self, required_capabilities: List[str]) -> List[AgentCapability]:
        """Discover agents with specific capabilities"""
        matching_agents = []
        
        for agent_id, capabilities in self.agent_registry.items():
            match_score = 0
            for req_cap in required_capabilities:
                if req_cap in capabilities.capabilities:
                    match_score += 1
                if req_cap in capabilities.specializations:
                    match_score += 2  # Higher weight for specializations
            
            if match_score > 0:
                matching_agents.append((capabilities, match_score))
        
        # Sort by match score and quality
        matching_agents.sort(key=lambda x: (x[1], x[0].quality_score), reverse=True)
        
        return [agent for agent, _ in matching_agents]
    
    async def coordinate_parallel_tasks(self,
                                      coordinator_agent: str,
                                      tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate parallel task execution across multiple agents"""
        
        coordination_id = str(uuid.uuid4())
        task_assignments = {}
        
        # Assign tasks to best-suited agents
        for i, task in enumerate(tasks):
            required_caps = task.get('required_capabilities', [])
            suitable_agents = await self.discover_agents(required_caps)
            
            if suitable_agents:
                assigned_agent = suitable_agents[0].agent_type
                task_id = f"task_{i}_{coordination_id}"
                
                task_assignments[task_id] = {
                    'agent': assigned_agent,
                    'task': task,
                    'status': 'assigned',
                    'assignment_time': datetime.now()
                }
                
                # Send task request
                message = AgentMessage(
                    message_id="",
                    sender_agent=coordinator_agent,
                    recipient_agent=assigned_agent,
                    message_type=MessageType.TASK_REQUEST,
                    priority=MessagePriority.NORMAL,
                    content={
                        "task_id": task_id,
                        "coordination_id": coordination_id,
                        "task_details": task,
                        "deadline": task.get('deadline')
                    },
                    context={"parallel_coordination": True},
                    timestamp=datetime.now(),
                    requires_response=True,
                    correlation_id=coordination_id
                )
                
                await self.send_message(message)
        
        logger.info(f"🎯 Parallel coordination initiated: {len(task_assignments)} tasks assigned")
        return {
            'coordination_id': coordination_id,
            'task_assignments': task_assignments,
            'status': 'coordinating'
        }
    
    async def get_agent_performance_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        return self.performance_tracker.get(agent_id, {
            'messages_sent': 0,
            'messages_received': 0,
            'tasks_completed': 0,
            'average_response_time': 0.0,
            'quality_score': 0.0,
            'collaboration_count': 0
        })
    
    async def update_agent_performance(self, 
                                     agent_id: str,
                                     metric: str,
                                     value: Any):
        """Update performance metrics for an agent"""
        if agent_id not in self.performance_tracker:
            self.performance_tracker[agent_id] = {}
        
        self.performance_tracker[agent_id][metric] = value
        logger.debug(f"📊 Performance updated: {agent_id}.{metric} = {value}")
    
    async def get_conversation_history(self, conversation_id: str) -> List[AgentMessage]:
        """Get the full conversation history"""
        return self.message_history.get(conversation_id, [])
    
    async def cleanup_expired_messages(self):
        """Clean up expired messages and old conversations"""
        current_time = datetime.now()
        cleaned_count = 0
        
        # Clean message history (keep last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        for conv_id in list(self.message_history.keys()):
            messages = self.message_history[conv_id]
            if messages and messages[-1].timestamp < cutoff_time:
                del self.message_history[conv_id]
                cleaned_count += 1
        
        # Clean knowledge cache (keep last 1 hour for most items)
        knowledge_cutoff = current_time - timedelta(hours=1)
        for key in list(self.knowledge_cache.keys()):
            knowledge = self.knowledge_cache[key]
            if knowledge['timestamp'] < knowledge_cutoff and knowledge['access_count'] == 0:
                del self.knowledge_cache[key]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} expired items")

# Global communication bus instance
communication_bus = AgentCommunicationBus()