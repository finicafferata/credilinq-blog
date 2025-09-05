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
// In production, ALWAYS use relative URLs for Vercel proxy to avoid mixed content issues
let apiBaseUrl: string;

if (isProduction) {
  // Force relative URLs in production to use Vercel's HTTPS proxy
  apiBaseUrl = '';
  console.log('🔧 PRODUCTION: Using relative URLs for Vercel proxy');
} else {
  // In development, use environment variable or localhost
  apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  console.log('🔧 DEVELOPMENT: Using API base URL:', apiBaseUrl);
}

console.log('🔧 Final API base URL:', apiBaseUrl);
console.log('🔧 Environment check - isDev:', isDev, 'isProduction:', isProduction);
console.log('🔧 VITE_API_BASE_URL env var:', import.meta.env.VITE_API_BASE_URL);

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
      const response = await api.post('/api/v2/blogs', {
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
    const response = await api.get('/api/v2/blogs');
    // Handle both response formats: array or object with blogs property
    if (Array.isArray(response.data)) {
      return response.data;
    }
    return response.data.blogs || [];
  },

  // Get single blog by ID
  get: async (id: string): Promise<BlogDetail> => {
    const response = await api.get(`/api/v2/blogs/${id}`);
    return response.data;
  },

  // Update blog content
  update: async (id: string, data: BlogEditRequest): Promise<BlogDetail> => {
    try {
      const response = await api.put(`/api/v2/blogs/${id}`, data);
      return response.data;
    } catch (err) {
      throw err;
    }
  },


  // Delete blog post
  delete: async (id: string): Promise<{ message: string; id: string }> => {
    const response = await api.delete(`/api/v2/blogs/${id}`);
    return response.data;
  },

  // Publish blog post
  publish: async (id: string): Promise<BlogDetail> => {
    try {
      const response = await api.post(`/api/v2/blogs/${id}/publish`);
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
  blog_id?: string; // Optional for orchestration campaigns
  campaign_name: string;
  company_context: string;
  content_type?: string;
  template_config?: any;
  description?: string;
  strategy_type?: string;
  priority?: string;
  target_audience?: string;
  distribution_channels?: string[];
  timeline_weeks?: number;
  success_metrics?: any;
  budget_allocation?: any;
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
    const response = await api.get('/api/v2/campaigns/');
    // Handle both response formats: array or object with campaigns property
    if (Array.isArray(response.data)) {
      return response.data;
    }
    return response.data.campaigns || [];
  },

  // Create a new campaign
  create: async (data: CampaignCreateRequest): Promise<any> => {
    const response = await api.post('/api/v2/campaigns/', data);
    return response.data;
  },

  // Create campaign from blog
  createFromBlog: async (blogId: string, campaignName?: string): Promise<any> => {
    const response = await api.post(`/api/v2/blogs/${blogId}/create-campaign`, {
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
    const response = await api.get(`/api/v2/campaigns/${id}`);
    return response.data;
  },

  // Schedule campaign tasks
  schedule: async (id: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/${id}/schedule`, { campaign_id: id });
    return response.data;
  },

  // Distribute campaign posts
  distribute: async (id: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/${id}/distribute`, { campaign_id: id });
    return response.data;
  },

  // Update task status
  updateTaskStatus: async (campaignId: string, taskId: string, status: string): Promise<any> => {
    const response = await api.put(`/api/v2/campaigns/${campaignId}/tasks/${taskId}/status`, {
      task_id: taskId,
      status: status
    });
    return response.data;
  },

  // Create orchestration campaign (campaign-first, no blog dependency)
  createOrchestrationCampaign: async (campaignData: {
    campaign_name: string;
    company_context: string;
    strategy_type?: string;
    target_audience?: string;
    distribution_channels?: string[];
    timeline_weeks?: number;
    description?: string;
    priority?: string;
    success_metrics?: any;
    budget_allocation?: any;
  }): Promise<any> => {
    const response = await api.post('/api/v2/campaigns/', {
      // No blog_id for orchestration campaigns
      campaign_name: campaignData.campaign_name,
      company_context: campaignData.company_context,
      content_type: 'orchestration',
      template_id: 'orchestration_enhanced',
      template_config: {
        orchestration_mode: true
      },
      description: campaignData.description,
      strategy_type: campaignData.strategy_type,
      priority: campaignData.priority,
      target_audience: campaignData.target_audience,
      distribution_channels: campaignData.distribution_channels,
      timeline_weeks: campaignData.timeline_weeks,
      success_metrics: campaignData.success_metrics,
      budget_allocation: campaignData.budget_allocation
    });
    return response.data;
  },

  // Campaign Orchestration Dashboard API
  getOrchestrationDashboard: async (): Promise<{
    campaigns: any[];
    agents: any[];
    systemMetrics: any;
    lastUpdated: string;
  }> => {
    const response = await api.get('/api/v2/campaigns/orchestration/dashboard');
    return response.data;
  },

  // Control campaign (play, pause, stop, restart)
  controlCampaign: async (campaignId: string, action: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/control?action=${action}`);
    return response.data;
  },

  // Get agent performance details
  getAgentPerformance: async (agentId: string): Promise<any> => {
    const response = await api.get(`/api/v2/campaigns/orchestration/agents/${agentId}/performance`);
    return response.data;
  },

  // Execute individual task
  executeTask: async (campaignId: string, taskId: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/tasks/${taskId}/execute`, {});
    return response.data;
  },

  // Execute all tasks for a campaign
  executeAllTasks: async (campaignId: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/execute-all`, {});
    return response.data;
  },

  // Rerun agents to generate new tasks for a campaign
  rerunCampaignAgents: async (campaignId: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/${campaignId}/rerun-agents`);
    return response.data;
  },
  // Generate tasks from campaign wizard data
  generateTasks: async (campaignId: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/${campaignId}/generate-tasks`);
    return response.data;
  },

  // Review task content (approve, reject, request revision)
  reviewTask: async (campaignId: string, taskId: string, action: 'approve' | 'reject' | 'request_revision', notes?: string): Promise<any> => {
    const params = new URLSearchParams();
    params.append('action', action);
    if (notes) {
      params.append('notes', notes);
    }
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/tasks/${taskId}/review?${params}`, 
      null, 
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    );
    return response.data;
  },

  // Get review queue for a campaign
  getReviewQueue: async (campaignId: string): Promise<any[]> => {
    const response = await api.get(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/review-queue`);
    return response.data;
  },

  // Schedule approved content with smart timing
  scheduleApprovedContent: async (campaignId: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/schedule-approved-content`);
    return response.data;
  },

  // Get scheduled content calendar
  getScheduledContent: async (campaignId: string): Promise<any[]> => {
    const response = await api.get(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/scheduled-content`);
    return response.data;
  },

  // Request detailed revision with feedback
  requestRevision: async (campaignId: string, taskId: string, feedback: {
    type?: string;
    issues?: string[];
    suggestions?: string[];
    quality_score?: number;
    notes?: string;
    changes?: string[];
    priority?: string;
    revision_round?: number;
  }): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/tasks/${taskId}/request-revision`, feedback);
    return response.data;
  },

  // Regenerate task with previous feedback
  regenerateTask: async (campaignId: string, taskId: string): Promise<any> => {
    const response = await api.post(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/tasks/${taskId}/regenerate`);
    return response.data;
  },

  // Get feedback analytics for continuous improvement
  getFeedbackAnalytics: async (campaignId: string): Promise<any> => {
    const response = await api.get(`/api/v2/campaigns/orchestration/campaigns/${campaignId}/feedback-analytics`);
    return response.data;
  },

  // Get AI-powered content recommendations using PlannerAgent
  getAIRecommendations: async (campaignData: {
    campaign_objective: string;
    target_market: string;
    campaign_purpose: string;
    campaign_duration_weeks: number;
    company_context?: string;
  }): Promise<{
    recommended_content_mix: {
      blog_posts: number;
      social_posts: number;
      email_sequences: number;
      infographics: number;
    };
    suggested_themes: string[];
    optimal_channels: string[];
    recommended_posting_frequency: string;
    ai_reasoning?: string;
    generated_by: string;
    timestamp: string;
  }> => {
    const response = await api.post('/api/v2/campaigns/ai-recommendations', campaignData);
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
    const response = await api.get(`/api/v2/blogs/${blogId}/analytics`);
    return response.data;
  },

  // Update blog analytics
  updateBlogAnalytics: async (blogId: string, analytics: BlogAnalytics): Promise<{ message: string; record_id: string; blog_id: string }> => {
    const response = await api.post(`/api/v2/blogs/${blogId}/analytics`, analytics);
    return response.data;
  },

  // Record marketing metric
  recordMarketingMetric: async (blogId: string, metric: MarketingMetric): Promise<{ message: string; record_id: string; blog_id: string }> => {
    const response = await api.post(`/api/v2/blogs/${blogId}/metrics`, metric);
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
    const response = await api.get(`/api/analytics/competitor-intelligence?days=${days}`);
    return response.data;
  },
};

export default api;
export { api };
 
// Comments API
export const commentsApi = {
  list: async (blogId: string): Promise<CommentItem[]> => {
    const res = await api.get(`/api/v2/blogs/${blogId}/comments`);
    return res.data;
  },
  add: async (
    blogId: string,
    data: { content: string; author?: string; position?: CommentPosition }
  ): Promise<CommentItem> => {
    const res = await api.post(`/api/v2/blogs/${blogId}/comments`, data);
    return res.data;
  },
  reply: async (
    blogId: string,
    commentId: string,
    data: { content: string; author?: string }
  ): Promise<CommentItem> => {
    const res = await api.post(`/api/v2/blogs/${blogId}/comments/${commentId}/reply`, data);
    return res.data;
  },
  resolve: async (blogId: string, commentId: string): Promise<CommentItem> => {
    const res = await api.post(`/api/v2/blogs/${blogId}/comments/${commentId}/resolve`);
    return res.data;
  },
};

export const suggestionsApi = {
  list: async (blogId: string): Promise<SuggestionItem[]> => {
    const res = await api.get(`/api/v2/blogs/${blogId}/suggestions`);
    return res.data;
  },
  add: async (
    blogId: string,
    data: { author?: string; originalText: string; suggestedText: string; reason: string; position: { start: number; end: number } }
  ): Promise<SuggestionItem> => {
    const res = await api.post(`/api/v2/blogs/${blogId}/suggestions`, data);
    return res.data;
  },
  accept: async (blogId: string, suggestionId: string): Promise<SuggestionItem> => {
    const res = await api.post(`/api/v2/blogs/${blogId}/suggestions/${suggestionId}/accept`);
    return res.data;
  },
  reject: async (blogId: string, suggestionId: string): Promise<SuggestionItem> => {
    const res = await api.post(`/api/v2/blogs/${blogId}/suggestions/${suggestionId}/reject`);
    return res.data;
  },
};

// Settings API (lightweight, colocated for now) - Updated 2025-08-19
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