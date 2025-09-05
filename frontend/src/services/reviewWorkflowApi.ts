import api from '../lib/api';

// Types for Review Workflow - matches backend models
export interface ReviewStage {
  stage: string;
  status: 'pending' | 'in_progress' | 'completed' | 'requires_human_review' | 'rejected';
  automated_score?: number;
  human_score?: number;
  feedback: string[];
  suggestions: string[];
  reviewer_id?: string;
  completed_at?: string;
}

export interface ReviewWorkflowStatus {
  workflow_execution_id: string;
  content_id: string;
  workflow_status: string;
  overall_approval_status: string;
  current_stage: string | null;
  overall_progress: number;
  is_paused: boolean;
  completed_stages: string[];
  failed_stages: string[];
  pending_human_reviews: string[];
  estimated_completion?: string;
  started_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface WorkflowMetrics {
  period: {
    days: number;
    start_date: string;
    end_date: string;
  };
  workflow_stats: {
    total_workflows: number;
    completed_workflows: number;
    in_progress_workflows: number;
    paused_workflows: number;
    completion_rate: number;
  };
  approval_stats: {
    approved: number;
    needs_revision: number;
    rejected: number;
    approval_rate: number;
  };
  performance_stats: {
    average_completion_hours: number;
    median_completion_hours: number;
    average_human_review_hours: number;
    automation_rate: number;
  };
  quality_stats: {
    average_quality_score: number;
    average_seo_score: number;
    average_brand_score: number;
    improvement_rate: number;
  };
  stage_stats: {
    [stageName: string]: {
      completion_rate: number;
      average_score?: number;
    };
  };
}

export interface PendingReview {
  workflow_execution_id: string;
  content_id: string;
  content_type: string;
  title: string;
  stage: string;
  priority: string;
  assigned_at: string;
  expected_completion_at: string;
  automated_score?: number;
  campaign_id?: string;
  is_overdue: boolean;
}

export interface StartWorkflowRequest {
  content_id: string;
  content_type: string;
  content: string;
  title?: string;
  campaign_id?: string;
  content_metadata?: Record<string, any>;
  review_config?: {
    require_content_quality?: boolean;
    require_editorial_review?: boolean;
    require_brand_check?: boolean;
    require_seo_analysis?: boolean;
    require_geo_analysis?: boolean;
    require_visual_review?: boolean;
    require_social_media_review?: boolean;
    require_final_approval?: boolean;
    content_quality_auto_approve_threshold?: number;
    editorial_auto_approve_threshold?: number;
    brand_auto_approve_threshold?: number;
    seo_auto_approve_threshold?: number;
    geo_auto_approve_threshold?: number;
    visual_auto_approve_threshold?: number;
    social_media_auto_approve_threshold?: number;
    allow_parallel_reviews?: boolean;
  };
  integration_mode?: string;
  content_source?: string;
  context?: Record<string, any>;
}

export interface HumanReviewDecision {
  stage: string;
  reviewer_id: string;
  status: 'approved' | 'rejected' | 'needs_revision';
  score?: number;
  feedback: string;
  suggestions?: string[];
  revision_requests?: string[];
}

export interface WorkflowSummary {
  workflow_execution_id: string;
  content_id: string;
  overall_approval_status: string;
  quality_metrics: Record<string, any>;
  approval_summary: Record<string, any>;
  total_execution_time_seconds?: number;
  stage_decisions: Array<{
    stage: string;
    reviewer_id: string;
    reviewer_type: string;
    status: string;
    score?: number;
    feedback: string;
    suggestions: string[];
    decision_timestamp: string;
  }>;
}

export const reviewWorkflowApi = {
  /**
   * Start a new review workflow for content
   */
  async startWorkflow(request: StartWorkflowRequest): Promise<{
    success: boolean;
    message: string;
    review_workflow_id: string;
    workflow_status: string;
    current_stage: string;
    is_paused: boolean;
    metadata: Record<string, any>;
  }> {
    const response = await api.post('/api/v2/review-workflow/start', request);
    return response.data;
  },

  /**
   * Get workflow status by ID
   */
  async getWorkflowStatus(workflowId: string): Promise<ReviewWorkflowStatus> {
    const response = await api.get(`/api/v2/review-workflow/${workflowId}/status`);
    return response.data;
  },

  /**
   * Resume a paused workflow with human review decisions
   */
  async resumeWorkflow(
    workflowId: string,
    humanReviewUpdates: Record<string, HumanReviewDecision>,
    context?: Record<string, any>
  ): Promise<{
    success: boolean;
    message: string;
    review_workflow_id: string;
    workflow_status: string;
    current_stage: string;
    is_paused: boolean;
    metadata: Record<string, any>;
  }> {
    const response = await api.post(`/api/v2/review-workflow/${workflowId}/resume`, {
      human_review_updates: humanReviewUpdates,
      context: context || {}
    });
    return response.data;
  },

  /**
   * Submit human review decision for a specific stage
   */
  async submitHumanReview(
    workflowId: string,
    decision: HumanReviewDecision
  ): Promise<{
    success: boolean;
    message: string;
    stage: string;
    reviewer_id: string;
    status: string;
    submitted_at: string;
  }> {
    const response = await api.post(`/api/v2/review-workflow/${workflowId}/human-review`, decision);
    return response.data;
  },

  /**
   * Get pending reviews for the current user or specific reviewer
   */
  async getPendingReviews(options?: {
    reviewer_id?: string;
    stage?: string;
    priority?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    pending_reviews: PendingReview[];
    total_count: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }> {
    const params = new URLSearchParams();
    if (options?.reviewer_id) params.append('reviewer_id', options.reviewer_id);
    if (options?.stage) params.append('stage', options.stage);
    if (options?.priority) params.append('priority', options.priority);
    if (options?.limit) params.append('limit', options.limit.toString());
    if (options?.offset) params.append('offset', options.offset.toString());

    const response = await api.get(`/api/v2/review-workflow/pending-reviews?${params}`);
    return response.data;
  },

  /**
   * Get review workflow metrics and analytics
   */
  async getMetrics(days: number = 30): Promise<WorkflowMetrics> {
    const response = await api.get(`/api/v2/review-workflow/metrics?days=${days}`);
    return response.data;
  },

  /**
   * Get workflow summary with detailed results
   */
  async getWorkflowSummary(workflowId: string): Promise<WorkflowSummary> {
    const response = await api.get(`/api/v2/review-workflow/${workflowId}/summary`);
    return response.data;
  },

  /**
   * Assign reviewers to a workflow stage
   */
  async assignReviewers(
    workflowId: string,
    stage: string,
    reviewerIds: string[],
    options?: {
      priority?: string;
      expected_completion_hours?: number;
      instructions?: string;
    }
  ): Promise<{
    success: boolean;
    message: string;
    workflow_id: string;
    stage: string;
    assignments: any[];
    notifications_sent: number;
  }> {
    const response = await api.post(`/api/v2/review-workflow/${workflowId}/assign-reviewers`, {
      stage,
      reviewer_ids: reviewerIds,
      priority: options?.priority || 'medium',
      expected_completion_hours: options?.expected_completion_hours || 48,
      instructions: options?.instructions
    });
    return response.data;
  },

  /**
   * Cancel an active workflow
   */
  async cancelWorkflow(
    workflowId: string,
    reason: string
  ): Promise<{
    success: boolean;
    message: string;
    cancellation_data: any;
  }> {
    const response = await api.delete(`/api/v2/review-workflow/${workflowId}?reason=${encodeURIComponent(reason)}`);
    return response.data;
  },

  /**
   * Get health status of review workflow service
   */
  async getHealth(): Promise<{
    status: string;
    timestamp: string;
    services: Record<string, string>;
    version: string;
  }> {
    const response = await api.get('/api/v2/review-workflow/health');
    return response.data;
  }
};

// Utility functions for the UI
export const ReviewWorkflowUtils = {
  /**
   * Get display name for review stages
   */
  getStageDisplayName(stage: string): string {
    const stageNames: Record<string, string> = {
      'content_quality': 'Content Quality',
      'editorial_review': 'Editorial Review',
      'brand_check': 'Brand Check',
      'seo_analysis': 'SEO Analysis',
      'geo_analysis': 'GEO Analysis',
      'visual_review': 'Visual Review',
      'social_media_review': 'Social Media Review',
      'final_approval': 'Final Approval'
    };
    return stageNames[stage] || stage;
  },

  /**
   * Get stage progress (1-8 for 8-stage workflow)
   */
  getStageProgress(stage: string): number {
    const stageOrder = [
      'content_quality',
      'editorial_review', 
      'brand_check',
      'seo_analysis',
      'geo_analysis',
      'visual_review',
      'social_media_review',
      'final_approval'
    ];
    return stageOrder.indexOf(stage) + 1;
  },

  /**
   * Calculate overall progress percentage
   */
  calculateProgress(completedStages: string[]): number {
    return Math.round((completedStages.length / 8) * 100);
  },

  /**
   * Get status color for UI
   */
  getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      'pending': 'bg-gray-100 text-gray-800',
      'in_progress': 'bg-blue-100 text-blue-800',
      'completed': 'bg-green-100 text-green-800',
      'requires_human_review': 'bg-yellow-100 text-yellow-800',
      'rejected': 'bg-red-100 text-red-800',
      'approved': 'bg-green-100 text-green-800'
    };
    return colors[status] || colors['pending'];
  },

  /**
   * Format review score for display
   */
  formatScore(score: number): string {
    return `${score.toFixed(1)}/10.0`;
  },

  /**
   * Get score color based on value
   */
  getScoreColor(score: number): string {
    if (score >= 8.5) return 'text-green-600';
    if (score >= 7.0) return 'text-yellow-600';
    return 'text-red-600';
  },

  /**
   * Check if stage requires human review
   */
  requiresHumanReview(stage: string, score?: number, threshold: number = 8.0): boolean {
    return !score || score < threshold;
  }
};

export default reviewWorkflowApi;