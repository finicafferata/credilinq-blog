import axios from 'axios';
import type { CampaignResponse, CampaignTask, TaskStatus } from '../types/campaign';
import type { 
  BlogAnalytics, 
  BlogAnalyticsResponse, 
  MarketingMetric, 
  AgentFeedback, 
  DashboardAnalytics, 
  AgentAnalytics,
  CompetitorIntelligenceAnalytics
} from '../types/analytics';
import { parseApiError, AppError } from './errors';

const isDev = import.meta.env.DEV;
const isProduction = import.meta.env.PROD;

// Use VITE_API_BASE_URL environment variable if available, otherwise determine based on environment
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 
  (isDev ? 'http://localhost:8000' : 'https://credilinq-blog-production.up.railway.app');

// Note: avoid noisy console logs in production; use network tab if needed

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

// Comments types
export interface CommentPosition {
  start: number;
  end: number;
  selectedText: string;
}

export interface CommentItem {
  id: string;
  author: string;
  content: string;
  timestamp: string;
  resolved: boolean;
  position?: CommentPosition;
  replies?: CommentItem[];
}

export interface SuggestionItem {
  id: string;
  author: string;
  originalText: string;
  suggestedText: string;
  reason: string;
  timestamp: string;
  status: 'pending' | 'accepted' | 'rejected';
  position: { start: number; end: number };
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
    try {
      const response = await api.post('/api/blogs', {
        title: data.title,
        company_context: data.company_context,
        content_type: data.content_type || 'blog'
      });
      return response.data;
    } catch (err) {
      throw err;
    }
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
    try {
      const response = await api.put(`/api/blogs/${id}`, data);
      return response.data;
    } catch (err) {
      throw err;
    }
  },


  // Delete blog post
  delete: async (id: string): Promise<{ message: string; id: string }> => {
    const response = await api.delete(`/api/blogs/${id}`);
    return response.data;
  },

  // Publish blog post
  publish: async (id: string): Promise<BlogDetail> => {
    try {
      const response = await api.post(`/api/blogs/${id}/publish`);
      return response.data;
    } catch (err) {
      throw err;
    }
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
  // Get competitor intelligence analytics
  getCompetitorIntelligenceAnalytics: async (days: number = 30): Promise<CompetitorIntelligenceAnalytics> => {
    const response = await api.get(`/api/competitor-intelligence/dashboard?days=${days}`);
    return response.data;
  },
};

export default api; 
 
// Comments API
export const commentsApi = {
  list: async (blogId: string): Promise<CommentItem[]> => {
    const res = await api.get(`/api/blogs/${blogId}/comments`);
    return res.data;
  },
  add: async (
    blogId: string,
    data: { content: string; author?: string; position?: CommentPosition }
  ): Promise<CommentItem> => {
    const res = await api.post(`/api/blogs/${blogId}/comments`, data);
    return res.data;
  },
  reply: async (
    blogId: string,
    commentId: string,
    data: { content: string; author?: string }
  ): Promise<CommentItem> => {
    const res = await api.post(`/api/blogs/${blogId}/comments/${commentId}/reply`, data);
    return res.data;
  },
  resolve: async (blogId: string, commentId: string): Promise<CommentItem> => {
    const res = await api.post(`/api/blogs/${blogId}/comments/${commentId}/resolve`);
    return res.data;
  },
};

export const suggestionsApi = {
  list: async (blogId: string): Promise<SuggestionItem[]> => {
    const res = await api.get(`/api/blogs/${blogId}/suggestions`);
    return res.data;
  },
  add: async (
    blogId: string,
    data: { author?: string; originalText: string; suggestedText: string; reason: string; position: { start: number; end: number } }
  ): Promise<SuggestionItem> => {
    const res = await api.post(`/api/blogs/${blogId}/suggestions`, data);
    return res.data;
  },
  accept: async (blogId: string, suggestionId: string): Promise<SuggestionItem> => {
    const res = await api.post(`/api/blogs/${blogId}/suggestions/${suggestionId}/accept`);
    return res.data;
  },
  reject: async (blogId: string, suggestionId: string): Promise<SuggestionItem> => {
    const res = await api.post(`/api/blogs/${blogId}/suggestions/${suggestionId}/reject`);
    return res.data;
  },
};

// Settings API (lightweight, colocated for now)
export type CompanyProfile = {
  companyName?: string;
  companyContext: string;
  brandVoice?: string;
  valueProposition?: string;
  industries: string[];
  targetAudiences: string[];
  tonePresets: string[];
  keywords: string[];
  styleGuidelines?: string;
  prohibitedTopics: string[];
  complianceNotes?: string;
  links: { label: string; url: string }[];
  defaultCTA?: string;
  updatedAt?: string;
};

export const settingsApi = {
  getCompanyProfile: async (): Promise<CompanyProfile> => {
    const res = await api.get('/api/settings/company-profile');
    return res.data as CompanyProfile;
  },
  updateCompanyProfile: async (profile: CompanyProfile): Promise<void> => {
    await api.put('/api/settings/company-profile', profile);
  }
};