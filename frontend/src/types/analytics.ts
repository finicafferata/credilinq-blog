// Analytics types to match backend models

export interface BlogAnalytics {
  views: number;
  unique_visitors: number;
  engagement_rate: number;
  avg_time_on_page?: number;
  bounce_rate: number;
  social_shares: number;
  comments_count: number;
  conversion_rate: number;
  seo_score: number;
  readability_score: number;
}

export interface BlogAnalyticsResponse extends BlogAnalytics {
  blog_id: string;
  created_at?: string;
  updated_at?: string;
}

export interface MarketingMetric {
  metric_type: string;
  metric_value: number;
  source?: string;
  medium?: string;
  campaign_name?: string;
}

export interface AgentFeedback {
  agent_type: string;
  feedback_type: string;
  feedback_value?: number;
  feedback_text?: string;
  user_id?: string;
}

export interface DashboardAnalytics {
  total_blogs: number;
  total_campaigns: number;
  total_agent_executions: number;
  avg_execution_time: number;
  success_rate: number;
  top_performing_agents: Array<{
    agent_type: string;
    execution_count: number;
    avg_success_rate: number;
  }>;
  recent_performance: Array<{
    date: string;
    executions: number;
    success_rate: number;
  }>;
  blog_performance: Array<{
    blog_id: string;
    title: string;
    views: number;
    engagement_rate: number;
  }>;
}

export interface AgentAnalytics {
  agent_type?: string;
  days: number;
  performance_data: Array<{
    id: string;
    agent_type: string;
    task_type: string;
    execution_time_ms: number;
    success_rate: number;
    quality_score: number;
    input_tokens: number;
    output_tokens: number;
    cost_usd: number;
    created_at: string;
  }>;
}

export interface CompetitorIntelligenceAnalytics {
  total_competitors: number;
  active_monitoring: number;
  content_analyzed: number;
  trends_identified: number;
  alerts_generated: number;
  top_competitors: Array<{
    id: string;
    name: string;
    domain: string;
    content_count: number;
    last_activity: string;
  }>;
  content_types_distribution: Array<{
    type: string;
    count: number;
    percentage: number;
  }>;
  platform_activity: Array<{
    platform: string;
    posts: number;
    engagement: number;
  }>;
  trending_topics: Array<{
    topic: string;
    mentions: number;
    growth_rate: number;
  }>;
  alerts_by_priority: Array<{
    priority: string;
    count: number;
  }>;
}