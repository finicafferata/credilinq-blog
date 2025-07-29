import axios from 'axios';
import type { CampaignResponse, CampaignTask, TaskStatus } from '../types/campaign';
import type { 
  BlogAnalytics, 
  BlogAnalyticsResponse, 
  MarketingMetric, 
  AgentFeedback, 
  DashboardAnalytics, 
  AgentAnalytics 
} from '../types/analytics';
import { parseApiError, AppError } from './errors';

const isDev = import.meta.env.DEV;
// Use VITE_API_BASE_URL environment variable if available
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 
  (isDev ? 'http://localhost:8000' : 'https://credilinq-blog-production.up.railway.app');

const api = axios.create({
  baseURL: apiBaseUrl,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for consistent error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError = parseApiError(error);
    throw new AppError(apiError.message, apiError.status, apiError.code, apiError.details);
  }
);

export interface BlogSummary {
  id: string;
  title: string;
  status: string;
  created_at: string;
}

export interface BlogDetail extends BlogSummary {
  content_markdown: string;
  initial_prompt: unknown;
}

export interface BlogCreateRequest {
  title: string;
  company_context: string;
  content_type: string; // "blog" or "linkedin"
}

export interface BlogEditRequest {
  content_markdown: string;
}


// Blog API endpoints
export const blogApi = {
  // Create a new blog post
  create: async (data: { title: string; company_context: string; content_type?: string }) => {
    console.log('Sending blog creation request:', data);
    
    const response = await api.post('/api/blogs', {
      title: data.title,
      company_context: data.company_context,
      content_type: data.content_type || 'blog'
    });
    
    return response.data;
  },

  // Get all blogs
  list: async (): Promise<BlogSummary[]> => {
    const response = await api.get('/api/blogs');
    return response.data;
  },

  // Get single blog by ID
  get: async (id: string): Promise<BlogDetail> => {
    const response = await api.get(`/api/blogs/${id}`);
    return response.data;
  },

  // Update blog content
  update: async (id: string, data: BlogEditRequest): Promise<BlogDetail> => {
    const response = await api.put(`/api/blogs/${id}`, data);
    return response.data;
  },


  // Delete blog post
  delete: async (id: string): Promise<{ message: string; id: string }> => {
    const response = await api.delete(`/api/blogs/${id}`);
    return response.data;
  },

  // Publish blog post
  publish: async (id: string): Promise<BlogDetail> => {
    const response = await api.post(`/api/blogs/${id}/publish`);
    return response.data;
  },

};


// Campaign API endpoints
export const campaignApi = {
  // Create a new campaign for a blog post
  createCampaign: async (blogId: string): Promise<CampaignResponse> => {
    const response = await api.post('/api/campaigns', { blog_id: blogId });
    return response.data;
  },

  // Get the campaign and its tasks for a blog post
  getCampaign: async (blogId: string): Promise<CampaignResponse> => {
    const response = await api.get(`/api/campaigns/${blogId}`);
    return response.data;
  },

  // Execute a specific campaign task
  executeCampaignTask: async (taskId: string): Promise<any> => {
    const response = await api.post('/api/campaigns/tasks/execute', { task_id: taskId });
    return response.data;
  },

  // Update a campaign task (approve, edit, change status)
  updateCampaignTask: async (taskId: string, content: string | undefined, status: TaskStatus): Promise<CampaignTask> => {
    const response = await api.put(`/api/campaigns/tasks/${taskId}`, { content, status });
    return response.data;
  },
};

// Analytics API endpoints
export const analyticsApi = {
  // Get dashboard analytics
  getDashboardAnalytics: async (days: number = 30): Promise<DashboardAnalytics> => {
    const response = await api.get(`/api/analytics/dashboard?days=${days}`);
    return response.data;
  },

  // Get blog-specific analytics
  getBlogAnalytics: async (blogId: string): Promise<BlogAnalyticsResponse> => {
    const response = await api.get(`/api/blogs/${blogId}/analytics`);
    return response.data;
  },

  // Update blog analytics
  updateBlogAnalytics: async (blogId: string, analytics: BlogAnalytics): Promise<{ message: string; record_id: string; blog_id: string }> => {
    const response = await api.post(`/api/blogs/${blogId}/analytics`, analytics);
    return response.data;
  },

  // Record marketing metric
  recordMarketingMetric: async (blogId: string, metric: MarketingMetric): Promise<{ message: string; record_id: string; blog_id: string }> => {
    const response = await api.post(`/api/blogs/${blogId}/metrics`, metric);
    return response.data;
  },

  // Get agent analytics
  getAgentAnalytics: async (agentType?: string, days: number = 30): Promise<AgentAnalytics> => {
    const params = new URLSearchParams();
    if (agentType) params.append('agent_type', agentType);
    params.append('days', days.toString());
    
    const response = await api.get(`/api/analytics/agents?${params.toString()}`);
    return response.data;
  },

  // Record agent feedback
  recordAgentFeedback: async (feedback: AgentFeedback): Promise<{ message: string; record_id: string }> => {
    const response = await api.post(`/api/agents/feedback`, feedback);
    return response.data;
  },
};

export default api; 