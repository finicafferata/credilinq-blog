import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// WebSocket connection for real-time updates
class OrchestrationWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Function[]> = new Map();

  connect() {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws/orchestration';
    
    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('🔗 Orchestration WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connection', { status: 'connected' });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit(data.type, data.payload);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 Orchestration WebSocket disconnected');
        this.emit('connection', { status: 'disconnected' });
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('🚨 WebSocket error:', error);
        this.emit('error', { error });
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.attemptReconnect();
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      console.log(`🔄 Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      console.error('❌ Max reconnection attempts reached');
      this.emit('connection', { status: 'failed' });
    }
  }

  on(eventType: string, callback: Function) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType)!.push(callback);
  }

  off(eventType: string, callback: Function) {
    const callbacks = this.listeners.get(eventType);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(eventType: string, data: any) {
    const callbacks = this.listeners.get(eventType);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('⚠️ WebSocket not connected, cannot send data');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }
}

// Create singleton instance
export const orchestrationWS = new OrchestrationWebSocket();

// API client
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface CampaignOrchestrationData {
  id: string;
  name: string;
  type: 'content_marketing' | 'blog_series' | 'seo_content' | 'email_sequence';
  status: 'draft' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  targetChannels: string[];
  assignedAgents: string[];
  currentStep?: string;
  estimatedCompletion?: string;
  metrics: {
    tasksCompleted: number;
    totalTasks: number;
    contentGenerated: number;
    agentsActive: number;
  };
  workflow?: {
    steps: WorkflowStep[];
    currentStepIndex: number;
  };
}

export interface AgentData {
  id: string;
  name: string;
  type: 'writer' | 'editor' | 'social_media' | 'seo' | 'planner';
  status: 'active' | 'idle' | 'busy' | 'offline';
  currentTask?: string;
  campaignId?: string;
  performance: {
    tasksCompleted: number;
    averageTime: number;
    successRate: number;
  };
  capabilities: string[];
  load: number;
  healthMetrics: {
    uptime: number;
    memoryUsage: number;
    responseTime: number;
    errorRate: number;
  };
}

export interface WorkflowStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  agentId?: string;
  duration?: number;
  startTime?: string;
  endTime?: string;
  output?: any;
  dependencies?: string[];
}

export interface SystemMetrics {
  totalCampaigns: number;
  activeCampaigns: number;
  totalAgents: number;
  activeAgents: number;
  averageResponseTime: number;
  systemLoad: number;
  eventsPerSecond: number;
  messagesInQueue: number;
  queuedTasks: number;
  completedTasksToday: number;
  errorRate: number;
  uptime: number;
}

export interface CampaignCreateRequest {
  name: string;
  type: string;
  description?: string;
  targetChannels: string[];
  priority: 'low' | 'normal' | 'high' | 'urgent';
  deadline?: string;
  requirements: {
    contentTypes: string[];
    tone: string;
    targetAudience: string;
    keywords?: string[];
  };
  workflow?: {
    templateId?: string;
    customSteps?: any[];
  };
}

export interface AgentAssignmentRequest {
  campaignId: string;
  agentRequirements: {
    type: string;
    capabilities: string[];
    minPerformanceScore?: number;
  };
  taskPriority: 'low' | 'normal' | 'high' | 'urgent';
}

// Campaign Management API
export const campaignOrchestrationAPI = {
  // Get all campaigns with real-time status
  async getCampaigns(): Promise<CampaignOrchestrationData[]> {
    const response = await apiClient.get('/api/v2/campaigns/orchestration');
    return response.data;
  },

  // Get campaign details with workflow information
  async getCampaign(campaignId: string): Promise<CampaignOrchestrationData> {
    const response = await apiClient.get(`/api/v2/campaigns/${campaignId}/orchestration`);
    return response.data;
  },

  // Create new campaign with orchestration
  async createCampaign(campaignData: CampaignCreateRequest): Promise<CampaignOrchestrationData> {
    const response = await apiClient.post('/api/v2/campaigns/orchestration', campaignData);
    return response.data;
  },

  // Control campaign execution
  async startCampaign(campaignId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`/api/v2/campaigns/${campaignId}/start`);
    return response.data;
  },

  async pauseCampaign(campaignId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`/api/v2/campaigns/${campaignId}/pause`);
    return response.data;
  },

  async resumeCampaign(campaignId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`/api/v2/campaigns/${campaignId}/resume`);
    return response.data;
  },

  async stopCampaign(campaignId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`/api/v2/campaigns/${campaignId}/stop`);
    return response.data;
  },

  // Get campaign workflow status
  async getCampaignWorkflow(campaignId: string): Promise<WorkflowStep[]> {
    const response = await apiClient.get(`/api/v2/campaigns/${campaignId}/workflow`);
    return response.data;
  },

  // Get campaign performance metrics
  async getCampaignMetrics(campaignId: string, timeRange?: string): Promise<any> {
    const params = timeRange ? { timeRange } : {};
    const response = await apiClient.get(`/api/v2/campaigns/${campaignId}/metrics`, { params });
    return response.data;
  }
};

// Agent Management API
export const agentOrchestrationAPI = {
  // Get all agents with real-time status
  async getAgents(): Promise<AgentData[]> {
    const response = await apiClient.get('/api/v2/agents/pool');
    return response.data;
  },

  // Get agent details
  async getAgent(agentId: string): Promise<AgentData> {
    const response = await apiClient.get(`/api/v2/agents/${agentId}`);
    return response.data;
  },

  // Get available agents for assignment
  async getAvailableAgents(requirements: {
    type?: string;
    capabilities?: string[];
    minLoad?: number;
    maxLoad?: number;
  }): Promise<AgentData[]> {
    const response = await apiClient.get('/api/v2/agents/available', { params: requirements });
    return response.data;
  },

  // Assign agent to campaign
  async assignAgent(assignmentData: AgentAssignmentRequest): Promise<{ success: boolean; agentId: string }> {
    const response = await apiClient.post('/api/v2/agents/assign', assignmentData);
    return response.data;
  },

  // Release agent from campaign
  async releaseAgent(agentId: string, campaignId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`/api/v2/agents/${agentId}/release`, { campaignId });
    return response.data;
  },

  // Get agent performance history
  async getAgentPerformance(agentId: string, timeRange?: string): Promise<any> {
    const params = timeRange ? { timeRange } : {};
    const response = await apiClient.get(`/api/v2/agents/${agentId}/performance`, { params });
    return response.data;
  }
};

// System Monitoring API
export const systemMonitoringAPI = {
  // Get system metrics
  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await apiClient.get('/api/v2/system/metrics');
    return response.data;
  },

  // Get system health status
  async getSystemHealth(): Promise<{
    status: 'healthy' | 'warning' | 'critical';
    components: Record<string, any>;
    issues: any[];
  }> {
    const response = await apiClient.get('/api/v2/system/health');
    return response.data;
  },

  // Get event logs
  async getEventLogs(filters?: {
    level?: string;
    category?: string;
    source?: string;
    startTime?: string;
    endTime?: string;
    limit?: number;
  }): Promise<any[]> {
    const response = await apiClient.get('/api/v2/system/logs', { params: filters });
    return response.data;
  },

  // Get performance analytics
  async getPerformanceAnalytics(timeRange?: string): Promise<{
    campaigns: any[];
    agents: any[];
    system: any;
    trends: any[];
  }> {
    const params = timeRange ? { timeRange } : {};
    const response = await apiClient.get('/api/v2/system/analytics', { params });
    return response.data;
  }
};

// Workflow Management API
export const workflowAPI = {
  // Get workflow templates
  async getWorkflowTemplates(): Promise<any[]> {
    const response = await apiClient.get('/api/v2/workflows/templates');
    return response.data;
  },

  // Create custom workflow
  async createWorkflow(workflowData: {
    name: string;
    description: string;
    steps: any[];
    campaignTypes: string[];
  }): Promise<{ id: string; success: boolean }> {
    const response = await apiClient.post('/api/v2/workflows', workflowData);
    return response.data;
  },

  // Execute workflow
  async executeWorkflow(workflowId: string, input: any): Promise<{ executionId: string }> {
    const response = await apiClient.post(`/api/v2/workflows/${workflowId}/execute`, input);
    return response.data;
  },

  // Get workflow execution status
  async getWorkflowExecution(executionId: string): Promise<{
    status: string;
    currentStep: string;
    progress: number;
    output: any;
    errors: any[];
  }> {
    const response = await apiClient.get(`/api/v2/workflows/executions/${executionId}`);
    return response.data;
  }
};

// Real-time event handlers
export const setupRealTimeListeners = (callbacks: {
  onCampaignUpdate?: (campaign: CampaignOrchestrationData) => void;
  onAgentUpdate?: (agent: AgentData) => void;
  onSystemMetrics?: (metrics: SystemMetrics) => void;
  onWorkflowStep?: (step: WorkflowStep) => void;
  onError?: (error: any) => void;
  onConnection?: (status: { status: string }) => void;
}) => {
  // Setup WebSocket listeners
  if (callbacks.onCampaignUpdate) {
    orchestrationWS.on('campaign_update', callbacks.onCampaignUpdate);
  }
  
  if (callbacks.onAgentUpdate) {
    orchestrationWS.on('agent_update', callbacks.onAgentUpdate);
  }
  
  if (callbacks.onSystemMetrics) {
    orchestrationWS.on('system_metrics', callbacks.onSystemMetrics);
  }
  
  if (callbacks.onWorkflowStep) {
    orchestrationWS.on('workflow_step', callbacks.onWorkflowStep);
  }
  
  if (callbacks.onError) {
    orchestrationWS.on('error', callbacks.onError);
  }
  
  if (callbacks.onConnection) {
    orchestrationWS.on('connection', callbacks.onConnection);
  }

  // Connect if not already connected
  orchestrationWS.connect();
};

// Cleanup listeners
export const cleanupRealTimeListeners = () => {
  orchestrationWS.disconnect();
};

// Export default API object
export default {
  campaigns: campaignOrchestrationAPI,
  agents: agentOrchestrationAPI,
  system: systemMonitoringAPI,
  workflows: workflowAPI,
  realTime: {
    setup: setupRealTimeListeners,
    cleanup: cleanupRealTimeListeners,
    ws: orchestrationWS
  }
};