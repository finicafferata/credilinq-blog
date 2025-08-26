import api from '../lib/api';

// Content workflow types - Updated 2025-08-19
export interface ContentGenerationRequest {
  campaign_id: string;
  trigger: 'campaign_created' | 'content_requested' | 'scheduled_execution' | 'manual_trigger' | 'strategy_updated';
  execution_mode: 'synchronous' | 'asynchronous' | 'step_by_step' | 'debug';
  max_concurrent_tasks?: number;
  auto_approve_threshold?: number;
  require_human_review?: boolean;
  deadline?: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  custom_settings?: Record<string, any>;
}

export interface WorkflowStatus {
  workflow_id: string;
  campaign_id: string;
  current_phase: string;
  total_phases: number;
  completed_phases: number;
  is_active: boolean;
  started_at?: string;
  total_tasks?: number;
  completed_tasks?: number;
  failed_tasks?: number;
  progress_percentage?: number;
}

export interface ContentTask {
  task_id: string;
  campaign_id: string;
  content_type: string;
  channel: string;
  status: string;
  priority: string;
  title?: string;
  themes: string[];
  word_count?: number;
  quality_score?: number;
  created_at: string;
  updated_at: string;
}

export interface GeneratedContent {
  content_id: string;
  task_id: string;
  content_type: string;
  channel: string;
  title: string;
  content: string;
  word_count: number;
  quality_score: number;
  seo_score?: number;
  estimated_engagement: string;
  metadata: Record<string, any>;
  created_at: string;
}

export interface ContentApprovalRequest {
  content_id: string;
  task_id: string;
  approved: boolean;
  feedback?: string;
  reviewer_id?: string;
  quality_rating?: number;
}

export interface ContentRevisionRequest {
  content_id: string;
  task_id: string;
  revision_notes: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

export interface TaskManagementStatus {
  is_running: boolean;
  scheduling_strategy: string;
  execution_mode: string;
  max_concurrent_tasks: number;
  queue_status: Record<string, any>;
  resource_utilization: Record<string, number>;
  system_metrics: Record<string, any>;
}

export interface PendingApproval {
  content_id: string;
  task_id: string;
  campaign_id: string;
  content_type: string;
  title: string;
  quality_score: number;
  created_at: string;
  reviewer_assigned?: string;
  priority: string;
}

export interface WorkflowAnalytics {
  workflow_performance: {
    active_workflows: number;
    total_campaigns_processed: number;
    workflow_success_rate: number;
    average_workflow_duration_minutes: number;
    content_quality_trend: number[];
  };
  task_analytics: Record<string, any>;
  system_performance: Record<string, any>;
}

// Content Workflow API
export const contentWorkflowApi = {
  // Workflow Management
  initiateWorkflow: async (request: ContentGenerationRequest) => {
    const response = await api.post('/api/v2/content-workflows/initiate', request);
    return response.data;
  },

  getWorkflowStatus: async (workflowId: string): Promise<WorkflowStatus> => {
    const response = await api.get(`/api/v2/content-workflows/status/${workflowId}`);
    return response.data;
  },

  pauseWorkflow: async (workflowId: string) => {
    const response = await api.post(`/api/v2/content-workflows/${workflowId}/pause`);
    return response.data;
  },

  resumeWorkflow: async (workflowId: string) => {
    const response = await api.post(`/api/v2/content-workflows/${workflowId}/resume`);
    return response.data;
  },

  cancelWorkflow: async (workflowId: string) => {
    const response = await api.delete(`/api/v2/content-workflows/${workflowId}/cancel`);
    return response.data;
  },

  // Content Generation
  generateContentForCampaign: async (campaignId: string, strategyOverride?: Record<string, any>) => {
    const response = await api.post(`/api/v2/content-workflows/generate/${campaignId}`, strategyOverride);
    return response.data;
  },

  getCampaignTasks: async (campaignId: string): Promise<ContentTask[]> => {
    const response = await api.get(`/api/v2/content-workflows/campaign/${campaignId}/tasks`);
    return response.data;
  },

  getCampaignContent: async (campaignId: string, statusFilter?: string): Promise<GeneratedContent[]> => {
    const params = statusFilter ? `?status_filter=${statusFilter}` : '';
    const response = await api.get(`/api/v2/content-workflows/campaign/${campaignId}/content${params}`);
    return response.data;
  },

  // Task Management
  getTaskManagerStatus: async (): Promise<TaskManagementStatus> => {
    const response = await api.get('/api/v2/content-workflows/task-manager/status');
    return response.data;
  },

  startTaskProcessing: async () => {
    const response = await api.post('/api/v2/content-workflows/task-manager/start');
    return response.data;
  },

  stopTaskProcessing: async () => {
    const response = await api.post('/api/v2/content-workflows/task-manager/stop');
    return response.data;
  },

  getTaskStatus: async (taskId: string) => {
    const response = await api.get(`/api/v2/content-workflows/task/${taskId}/status`);
    return response.data;
  },

  cancelTask: async (taskId: string) => {
    const response = await api.delete(`/api/v2/content-workflows/task/${taskId}/cancel`);
    return response.data;
  },

  // Content Review and Approval
  approveContent: async (request: ContentApprovalRequest) => {
    const response = await api.post('/api/v2/content-workflows/content/approve', request);
    return response.data;
  },

  requestContentRevision: async (request: ContentRevisionRequest) => {
    const response = await api.post('/api/v2/content-workflows/content/request-revision', request);
    return response.data;
  },

  getPendingApprovals: async (campaignId?: string, priority?: string): Promise<PendingApproval[]> => {
    const params = new URLSearchParams();
    if (campaignId) params.append('campaign_id', campaignId);
    if (priority) params.append('priority', priority);
    
    const response = await api.get(`/api/v2/content-workflows/content/pending-approval?${params.toString()}`);
    return response.data;
  },

  // Analytics
  getWorkflowPerformanceAnalytics: async (): Promise<WorkflowAnalytics> => {
    const response = await api.get('/api/v2/content-workflows/analytics/performance');
    return response.data;
  },

  getCampaignWorkflowAnalytics: async (campaignId: string) => {
    const response = await api.get(`/api/v2/content-workflows/analytics/campaign/${campaignId}`);
    return response.data;
  },

  // Health Check
  healthCheck: async () => {
    const response = await api.get('/api/v2/content-workflows/health');
    return response.data;
  },
};