"""
Agent Management API Routes
Provides endpoints for managing and monitoring AI agents in the system.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import importlib
import pkgutil
import inspect
from pathlib import Path

from src.agents.core.base_agent import BaseAgent
from src.agents.core.agent_factory import AgentFactory
from src.agents.core.database_service import DatabaseService

router = APIRouter()

# Pydantic models for API responses
class AgentCapability(BaseModel):
    id: str
    name: str
    category: str
    proficiency: float = Field(ge=0, le=100)

class AgentTask(BaseModel):
    id: str
    name: str
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    priority: str
    status: str
    assigned_at: datetime
    estimated_duration: Optional[int] = None  # milliseconds
    
class AgentMetrics(BaseModel):
    tasks_completed: int = 0
    tasks_failed: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0  # milliseconds
    total_runtime: float = 0.0  # milliseconds
    uptime_percentage: float = 0.0
    
class AgentResourceUtilization(BaseModel):
    cpu: float = Field(default=0.0, ge=0, le=100)
    memory: float = Field(default=0.0, ge=0, le=100) 
    network: float = Field(default=0.0, ge=0)
    storage: float = Field(default=0.0, ge=0, le=100)
    max_concurrency: int = 1
    current_concurrency: int = 0
    
class AgentRetryPolicy(BaseModel):
    enabled: bool = True
    max_attempts: int = 3
    backoff_ms: int = 1000
    
class AgentConfig(BaseModel):
    auto_scale: bool = False
    max_tasks: int = 5
    priority: str = "normal"
    timeout: int = 300000  # milliseconds
    retry_policy: AgentRetryPolicy = AgentRetryPolicy()
    
class AgentHealth(BaseModel):
    status: str = "unknown"  # healthy, warning, error, offline
    last_check: datetime = Field(default_factory=datetime.now)
    issues: List[str] = []
    
class AgentInfo(BaseModel):
    id: str
    name: str
    type: str
    status: str = "offline"  # online, busy, idle, offline, error, maintenance
    version: str = "1.0.0"
    deployment: str = "local"  # local, cloud, edge
    capabilities: List[AgentCapability] = []
    current_tasks: List[AgentTask] = []
    metrics: AgentMetrics = AgentMetrics()
    resource_utilization: AgentResourceUtilization = Field(default_factory=AgentResourceUtilization)
    config: AgentConfig = AgentConfig()
    health: AgentHealth = AgentHealth()
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)

class AgentStatusUpdate(BaseModel):
    agent_id: str
    status: str
    
class AgentConfigUpdate(BaseModel):
    agent_id: str
    config: AgentConfig

class AgentPoolStats(BaseModel):
    total_agents: int
    active_agents: int
    busy_agents: int
    idle_agents: int
    offline_agents: int
    error_agents: int
    total_tasks_queued: int
    total_tasks_running: int
    average_response_time: float
    system_load: float

# In-memory agent registry for now (would be database-backed in production)
_agent_registry: Dict[str, AgentInfo] = {}
_agent_factory = None

async def get_agent_factory():
    """Get or create agent factory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory

async def discover_available_agents():
    """Discover all available agent classes from the agents directory."""
    agents_info = []
    
    # Hardcoded agent definitions based on your backend structure
    agent_definitions = [
        # Specialized Agents
        {"type": "content_agent", "name": "Content Writer Agent", "category": "content"},
        {"type": "editor_agent", "name": "Editor Agent", "category": "editing"},
        {"type": "writer_agent", "name": "Writer Agent", "category": "content"},
        {"type": "planner_agent", "name": "Planning Agent", "category": "strategy"},
        {"type": "researcher_agent", "name": "Research Agent", "category": "research"},
        {"type": "seo_agent", "name": "SEO Specialist", "category": "seo"},
        {"type": "social_media_agent", "name": "Social Media Agent", "category": "social"},
        {"type": "image_agent", "name": "Image Generation Agent", "category": "visual"},
        {"type": "repurpose_agent", "name": "Content Repurposing Agent", "category": "content"},
        {"type": "campaign_manager", "name": "Campaign Manager", "category": "management"},
        {"type": "distribution_agent", "name": "Distribution Agent", "category": "distribution"},
        {"type": "document_processor", "name": "Document Processor", "category": "processing"},
        {"type": "geo_analysis_agent", "name": "Geo Analysis Agent", "category": "analytics"},
        {"type": "task_scheduler", "name": "Task Scheduler", "category": "coordination"},
        {"type": "ai_content_generator", "name": "AI Content Generator", "category": "content"},
        {"type": "content_repurposer", "name": "Advanced Content Repurposer", "category": "content"},
        {"type": "search_agent", "name": "Web Search Agent", "category": "research"},
        
        # Competitor Intelligence Agents
        {"type": "content_monitoring_agent", "name": "Content Monitoring Agent", "category": "intelligence"},
        {"type": "trend_analysis_agent", "name": "Trend Analysis Agent", "category": "analytics"},
        {"type": "performance_analysis_agent", "name": "Performance Analysis Agent", "category": "analytics"},
        {"type": "gap_identification_agent", "name": "Gap Identification Agent", "category": "intelligence"},
        {"type": "strategic_insights_agent", "name": "Strategic Insights Agent", "category": "intelligence"},
        {"type": "alert_orchestration_agent", "name": "Alert Orchestration Agent", "category": "intelligence"},
    ]
    
    for i, agent_def in enumerate(agent_definitions):
        agent_info = AgentInfo(
            id=f"agent_{agent_def['type']}_{i + 1:03d}",
            name=agent_def["name"],
            type=agent_def["type"],
            status="offline",
            capabilities=await _generate_capabilities_for_agent(agent_def["type"]),
            health=AgentHealth(status="healthy")
        )
        agents_info.append(agent_info)
    
    return agents_info

async def _generate_capabilities_for_agent(agent_type: str) -> List[AgentCapability]:
    """Generate capabilities based on agent type."""
    capabilities_map = {
        # Specialized agents
        "content_agent": [
            AgentCapability(id="cap_content_1", name="Blog Writing", category="content", proficiency=95),
            AgentCapability(id="cap_content_2", name="Article Creation", category="content", proficiency=90),
            AgentCapability(id="cap_content_3", name="Content Strategy", category="content", proficiency=85)
        ],
        "editor_agent": [
            AgentCapability(id="cap_editor_1", name="Content Editing", category="editing", proficiency=98),
            AgentCapability(id="cap_editor_2", name="Proofreading", category="editing", proficiency=95),
            AgentCapability(id="cap_editor_3", name="Style Guide Compliance", category="editing", proficiency=92)
        ],
        "writer_agent": [
            AgentCapability(id="cap_writer_1", name="Creative Writing", category="content", proficiency=88),
            AgentCapability(id="cap_writer_2", name="Technical Writing", category="content", proficiency=90),
            AgentCapability(id="cap_writer_3", name="Copywriting", category="content", proficiency=85)
        ],
        "seo_agent": [
            AgentCapability(id="cap_seo_1", name="Keyword Research", category="seo", proficiency=96),
            AgentCapability(id="cap_seo_2", name="Content Optimization", category="seo", proficiency=94),
            AgentCapability(id="cap_seo_3", name="Technical SEO", category="seo", proficiency=88)
        ],
        "social_media_agent": [
            AgentCapability(id="cap_social_1", name="LinkedIn Posts", category="social", proficiency=92),
            AgentCapability(id="cap_social_2", name="Twitter Threads", category="social", proficiency=88),
            AgentCapability(id="cap_social_3", name="Content Scheduling", category="social", proficiency=90)
        ],
        "image_agent": [
            AgentCapability(id="cap_image_1", name="Image Generation Prompts", category="visual", proficiency=85),
            AgentCapability(id="cap_image_2", name="Visual Content Strategy", category="visual", proficiency=80),
            AgentCapability(id="cap_image_3", name="Brand Visual Consistency", category="visual", proficiency=88)
        ],
        "researcher_agent": [
            AgentCapability(id="cap_research_1", name="Market Research", category="research", proficiency=94),
            AgentCapability(id="cap_research_2", name="Fact Checking", category="research", proficiency=96),
            AgentCapability(id="cap_research_3", name="Trend Analysis", category="research", proficiency=87)
        ],
        "planner_agent": [
            AgentCapability(id="cap_planner_1", name="Content Planning", category="strategy", proficiency=92),
            AgentCapability(id="cap_planner_2", name="Campaign Strategy", category="strategy", proficiency=89),
            AgentCapability(id="cap_planner_3", name="Timeline Management", category="strategy", proficiency=85)
        ],
        "campaign_manager": [
            AgentCapability(id="cap_campaign_1", name="Campaign Orchestration", category="management", proficiency=93),
            AgentCapability(id="cap_campaign_2", name="Resource Allocation", category="management", proficiency=88),
            AgentCapability(id="cap_campaign_3", name="Performance Tracking", category="management", proficiency=91)
        ],
        # Competitor Intelligence agents
        "content_monitoring_agent": [
            AgentCapability(id="cap_monitor_1", name="Content Monitoring", category="intelligence", proficiency=95),
            AgentCapability(id="cap_monitor_2", name="Change Detection", category="intelligence", proficiency=92),
            AgentCapability(id="cap_monitor_3", name="Alert Generation", category="intelligence", proficiency=89)
        ],
        "trend_analysis_agent": [
            AgentCapability(id="cap_trend_1", name="Trend Identification", category="analytics", proficiency=94),
            AgentCapability(id="cap_trend_2", name="Market Analysis", category="analytics", proficiency=91),
            AgentCapability(id="cap_trend_3", name="Predictive Insights", category="analytics", proficiency=87)
        ],
        "performance_analysis_agent": [
            AgentCapability(id="cap_perf_1", name="Performance Analysis", category="analytics", proficiency=96),
            AgentCapability(id="cap_perf_2", name="Competitive Benchmarking", category="analytics", proficiency=93),
            AgentCapability(id="cap_perf_3", name="Metrics Tracking", category="analytics", proficiency=90)
        ],
    }
    
    return capabilities_map.get(agent_type, [
        AgentCapability(id=f"cap_{agent_type}_1", name=f"{agent_type.replace('_', ' ').title()}", 
                       category="general", proficiency=80)
    ])

async def initialize_agent_registry():
    """Initialize the agent registry with discovered agents."""
    global _agent_registry
    
    if not _agent_registry:
        discovered_agents = await discover_available_agents()
        for agent in discovered_agents:
            _agent_registry[agent.id] = agent
    
    return _agent_registry

# API Endpoints
@router.get("/agents", response_model=List[AgentInfo])
async def list_agents(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type")
):
    """Get list of all agents with their current status and information."""
    await initialize_agent_registry()
    
    agents = list(_agent_registry.values())
    
    # Apply filters
    if status_filter:
        agents = [a for a in agents if a.status == status_filter]
    
    if agent_type:
        agents = [a for a in agents if a.type == agent_type]
    
    return agents

@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent."""
    await initialize_agent_registry()
    
    if agent_id not in _agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return _agent_registry[agent_id]

@router.post("/agents/{agent_id}/start")
async def start_agent(agent_id: str):
    """Start a specific agent."""
    await initialize_agent_registry()
    
    if agent_id not in _agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = _agent_registry[agent_id]
    agent.status = "online"
    agent.last_seen = datetime.now()
    
    return {"message": f"Agent {agent_id} started successfully", "agent_id": agent_id}

@router.post("/agents/{agent_id}/stop")
async def stop_agent(agent_id: str):
    """Stop a specific agent."""
    await initialize_agent_registry()
    
    if agent_id not in _agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = _agent_registry[agent_id]
    agent.status = "offline"
    agent.last_seen = datetime.now()
    
    return {"message": f"Agent {agent_id} stopped successfully", "agent_id": agent_id}

@router.post("/agents/{agent_id}/restart")
async def restart_agent(agent_id: str):
    """Restart a specific agent."""
    await initialize_agent_registry()
    
    if agent_id not in _agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = _agent_registry[agent_id]
    agent.status = "maintenance"
    agent.last_seen = datetime.now()
    
    # Simulate restart delay
    await asyncio.sleep(0.1)
    
    agent.status = "online"
    agent.last_seen = datetime.now()
    
    return {"message": f"Agent {agent_id} restarted successfully", "agent_id": agent_id}

@router.put("/agents/{agent_id}/config")
async def update_agent_config(agent_id: str, config_update: AgentConfigUpdate):
    """Update agent configuration."""
    await initialize_agent_registry()
    
    if agent_id not in _agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = _agent_registry[agent_id]
    agent.config = config_update.config
    agent.last_seen = datetime.now()
    
    return {"message": f"Agent {agent_id} configuration updated", "agent_id": agent_id}

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete a specific agent from the registry."""
    await initialize_agent_registry()
    
    if agent_id not in _agent_registry:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del _agent_registry[agent_id]
    
    return {"message": f"Agent {agent_id} deleted successfully", "agent_id": agent_id}

@router.get("/agents/pool/stats", response_model=AgentPoolStats)
async def get_agent_pool_stats():
    """Get overall statistics about the agent pool."""
    await initialize_agent_registry()
    
    agents = list(_agent_registry.values())
    
    total_agents = len(agents)
    active_agents = len([a for a in agents if a.status in ["online", "busy", "idle"]])
    busy_agents = len([a for a in agents if a.status == "busy"])
    idle_agents = len([a for a in agents if a.status in ["online", "idle"]])
    offline_agents = len([a for a in agents if a.status == "offline"])
    error_agents = len([a for a in agents if a.status == "error"])
    
    total_tasks_queued = sum(len(a.current_tasks) for a in agents)
    total_tasks_running = sum(len([t for t in a.current_tasks if t.status == "running"]) for a in agents)
    
    avg_response_time = sum(a.metrics.avg_execution_time for a in agents) / max(total_agents, 1)
    system_load = (busy_agents / max(total_agents, 1)) * 100
    
    return AgentPoolStats(
        total_agents=total_agents,
        active_agents=active_agents,
        busy_agents=busy_agents,
        idle_agents=idle_agents,
        offline_agents=offline_agents,
        error_agents=error_agents,
        total_tasks_queued=total_tasks_queued,
        total_tasks_running=total_tasks_running,
        average_response_time=avg_response_time,
        system_load=system_load
    )

@router.post("/agents/pool/scale-up")
async def scale_up_agent_pool(agent_type: str, count: int = 1):
    """Scale up the agent pool by adding new agents of specified type."""
    await initialize_agent_registry()
    
    new_agents = []
    for i in range(count):
        agent_id = f"agent_{agent_type}_{len(_agent_registry) + i + 1:03d}"
        
        agent_info = AgentInfo(
            id=agent_id,
            name=f"{agent_type.replace('_', ' ').title()} {i + 1}",
            type=agent_type,
            status="online",
            capabilities=await _generate_capabilities_for_agent(agent_type),
            health=AgentHealth(status="healthy")
        )
        
        _agent_registry[agent_id] = agent_info
        new_agents.append(agent_id)
    
    return {
        "message": f"Scaled up {count} {agent_type} agents", 
        "new_agents": new_agents,
        "total_agents": len(_agent_registry)
    }

@router.post("/agents/pool/scale-down")
async def scale_down_agent_pool(agent_type: str, count: int = 1):
    """Scale down the agent pool by removing agents of specified type."""
    await initialize_agent_registry()
    
    # Find agents of the specified type that are not busy
    available_agents = [
        agent_id for agent_id, agent in _agent_registry.items()
        if agent.type == agent_type and agent.status != "busy"
    ]
    
    removed_agents = []
    for agent_id in available_agents[:count]:
        del _agent_registry[agent_id]
        removed_agents.append(agent_id)
    
    return {
        "message": f"Scaled down {len(removed_agents)} {agent_type} agents",
        "removed_agents": removed_agents,
        "total_agents": len(_agent_registry)
    }