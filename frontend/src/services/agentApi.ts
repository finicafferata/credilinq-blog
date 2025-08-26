import api from '../lib/api';

// Types for agent management - matches backend Pydantic models
export interface AgentCapability {
  id: string;
  name: string;
  category: string;
  proficiency: number;
}

export interface AgentTask {
  id: string;
  name: string;
  campaign_id?: string;
  campaign_name?: string;
  priority: string;
  status: string;
  assigned_at: string;
  estimated_duration?: number;
}

export interface AgentMetrics {
  tasks_completed: number;
  tasks_failed: number;
  success_rate: number;
  avg_execution_time: number;
  total_runtime: number;
  uptime_percentage: number;
}

export interface AgentResourceUtilization {
  cpu: number;
  memory: number;
  network: number;
  storage: number;
  max_concurrency: number;
  current_concurrency: number;
}

export interface AgentRetryPolicy {
  enabled: boolean;
  max_attempts: number;
  backoff_ms: number;
}

export interface AgentConfig {
  auto_scale: boolean;
  max_tasks: number;
  priority: string;
  timeout: number;
  retry_policy: AgentRetryPolicy;
}

export interface AgentHealth {
  status: string;
  last_check: string;
  issues: string[];
}

export interface AgentInfo {
  id: string;
  name: string;
  type: string;
  status: string;
  version: string;
  deployment: string;
  capabilities: AgentCapability[];
  current_tasks: AgentTask[];
  metrics: AgentMetrics;
  resource_utilization: AgentResourceUtilization;
  config: AgentConfig;
  health: AgentHealth;
  created_at: string;
  last_seen: string;
}

export interface AgentPoolStats {
  total_agents: number;
  active_agents: number;
  busy_agents: number;
  idle_agents: number;
  offline_agents: number;
  error_agents: number;
  total_tasks_queued: number;
  total_tasks_running: number;
  average_response_time: number;
  system_load: number;
}

export interface AgentStatusUpdate {
  agent_id: string;
  status: string;
}

export interface AgentConfigUpdate {
  agent_id: string;
  config: AgentConfig;
}

// Agent API service
export const agentApi = {
  // Get all agents
  listAgents: async (statusFilter?: string, agentType?: string): Promise<AgentInfo[]> => {
    const params = new URLSearchParams();
    if (statusFilter) params.append('status_filter', statusFilter);
    if (agentType) params.append('agent_type', agentType);
    
    const response = await api.get(`/api/v2/agents?${params.toString()}`);
    return response.data;
  },

  // Get specific agent details
  getAgent: async (agentId: string): Promise<AgentInfo> => {
    const response = await api.get(`/api/v2/agents/${agentId}`);
    return response.data;
  },

  // Start an agent
  startAgent: async (agentId: string) => {
    const response = await api.post(`/api/v2/agents/${agentId}/start`);
    return response.data;
  },

  // Stop an agent
  stopAgent: async (agentId: string) => {
    const response = await api.post(`/api/v2/agents/${agentId}/stop`);
    return response.data;
  },

  // Restart an agent
  restartAgent: async (agentId: string) => {
    const response = await api.post(`/api/v2/agents/${agentId}/restart`);
    return response.data;
  },

  // Update agent configuration
  updateAgentConfig: async (agentId: string, config: AgentConfig) => {
    const response = await api.put(`/api/v2/agents/${agentId}/config`, {
      agent_id: agentId,
      config
    });
    return response.data;
  },

  // Delete an agent
  deleteAgent: async (agentId: string) => {
    const response = await api.delete(`/api/v2/agents/${agentId}`);
    return response.data;
  },

  // Get agent pool statistics
  getAgentPoolStats: async (): Promise<AgentPoolStats> => {
    const response = await api.get('/api/v2/agents/pool/stats');
    return response.data;
  },

  // Scale up agent pool
  scaleUpAgentPool: async (agentType: string, count: number = 1) => {
    const response = await api.post('/api/v2/agents/pool/scale-up', {
      agent_type: agentType,
      count
    });
    return response.data;
  },

  // Scale down agent pool
  scaleDownAgentPool: async (agentType: string, count: number = 1) => {
    const response = await api.post('/api/v2/agents/pool/scale-down', {
      agent_type: agentType,
      count
    });
    return response.data;
  }
};