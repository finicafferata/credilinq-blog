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


// Campaign types
export interface CampaignSummary {
  id: string;
  name: string;
  status: string;
  progress: number;
  total_tasks: number;
  completed_tasks: number;
  created_at: string;
}

export interface CampaignCreateRequest {
  blog_id: string;
  campaign_name: string;
  company_context: string;
  content_type?: string;
}

export interface CampaignDetail {
  id: string;
  name: string;
  status: string;
  strategy: any;
  timeline: any[];
  tasks: any[];
  scheduled_posts: any[];
  performance: any;
}

// Campaign API endpoints
export const campaignApi = {
  // Get all campaigns
  list: async (): Promise<CampaignSummary[]> => {
    const response = await api.get('/api/campaigns');
    return response.data;
  },

  // Create a new campaign
  create: async (data: CampaignCreateRequest): Promise<any> => {
    const response = await api.post('/api/campaigns', data);
    return response.data;
  },

  // Create campaign from blog
  createFromBlog: async (blogId: string, campaignName?: string): Promise<any> => {
    const response = await api.post(`/api/blogs/${blogId}/create-campaign`, {
      campaign_name: campaignName || `Campaign for ${blogId}`
    });
    return response.data;
  },

  // Create quick campaign with template
  createQuickCampaign: async (templateId: string, blogId: string, campaignName: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/quick/${templateId}`, {
      blog_id: blogId,
      campaign_name: campaignName
    });
    return response.data;
  },

  // Get campaign details
  get: async (id: string): Promise<CampaignDetail> => {
    const response = await api.get(`/api/campaigns/${id}`);
    return response.data;
  },

  // Schedule campaign tasks
  schedule: async (id: string): Promise<any> => {
    const response = await api.post(`/api/campaigns/${id}/schedule`);
    return response.data;
  },

  // Distribute campaign posts
  distribute: async (id: string): Promise<any> => {
    const response = await api.post(`/api/campaigns/${id}/distribute`);
    return response.data;
  },

  // Update task status
  updateTaskStatus: async (campaignId: string, taskId: string, status: string): Promise<any> => {
    const response = await api.put(`/api/campaigns/${campaignId}/tasks/${taskId}/status`, {
      task_id: taskId,
      status: status
    });
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