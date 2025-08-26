/**
 * Content Deliverables API Service
 * Handles content-first deliverable operations, replacing task-centric API calls
 */

import api from '../lib/api';
import {
  ContentType,
  ContentFormat, 
  DeliverableStatus
} from '../types/contentTypes';
import type {
  ContentDeliverable,
  CampaignGenerationRequest,
  ContentGenerationResponse
} from '../types/contentTypes';

// Re-export types and enums for backward compatibility
export {
  ContentType,
  ContentFormat,
  DeliverableStatus
} from '../types/contentTypes';
export type {
  ContentDeliverable,
  CampaignGenerationRequest,
  ContentGenerationResponse
} from '../types/contentTypes';

export interface ContentNarrative {
  id: string;
  campaign_id: string;
  title: string;
  description?: string;
  narrative_theme: string;
  key_story_arc: string[];
  content_flow: Record<string, any>;
  total_pieces: number;
  completed_pieces: number;
  created_at: string;
  updated_at: string;
}

export interface ContentDeliverableCreate {
  title: string;
  content: string;
  summary?: string;
  content_type: ContentType;
  format?: ContentFormat;
  status?: DeliverableStatus;
  campaign_id: string;
  narrative_order?: number;
  key_messages?: string[];
  target_audience?: string;
  tone?: string;
  platform?: string;
  word_count?: number;
  reading_time?: number;
  created_by?: string;
  metadata?: Record<string, any>;
}

export interface ContentDeliverableUpdate {
  title?: string;
  content?: string;
  summary?: string;
  status?: DeliverableStatus;
  key_messages?: string[];
  target_audience?: string;
  tone?: string;
  platform?: string;
  metadata?: Record<string, any>;
}

// API Service Class
export class ContentDeliverableApi {
  private baseUrl = '/api/v2/deliverables';

  /**
   * Generate content deliverables for a campaign
   * This replaces task-based generation with deliverable-focused creation
   */
  async generateCampaignDeliverables(request: CampaignGenerationRequest): Promise<ContentGenerationResponse> {
    const response = await api.post<ContentGenerationResponse>(`${this.baseUrl}/generate`, request);
    return response.data;
  }

  /**
   * Get all content deliverables for a campaign
   * This replaces task listing with actual deliverable listing
   */
  async getCampaignDeliverables(
    campaignId: string,
    filters?: {
      content_type?: ContentType;
      status?: DeliverableStatus;
      platform?: string;
    }
  ): Promise<ContentDeliverable[]> {
    const params = new URLSearchParams();
    if (filters?.content_type) params.append('content_type', filters.content_type);
    if (filters?.status) params.append('status', filters.status);
    if (filters?.platform) params.append('platform', filters.platform);

    const query = params.toString() ? `?${params.toString()}` : '';
    const response = await api.get<ContentDeliverable[]>(`${this.baseUrl}/campaign/${campaignId}${query}`);
    return response.data;
  }

  /**
   * Get a specific content deliverable by ID
   */
  async getDeliverable(deliverableId: string): Promise<ContentDeliverable> {
    const response = await api.get<ContentDeliverable>(`${this.baseUrl}/${deliverableId}`);
    return response.data;
  }

  /**
   * Create a new content deliverable
   */
  async createDeliverable(deliverable: ContentDeliverableCreate): Promise<ContentDeliverable> {
    const response = await api.post<ContentDeliverable>(`${this.baseUrl}/`, deliverable);
    return response.data;
  }

  /**
   * Update a content deliverable
   */
  async updateDeliverable(deliverableId: string, updates: ContentDeliverableUpdate): Promise<ContentDeliverable> {
    const response = await api.patch<ContentDeliverable>(`${this.baseUrl}/${deliverableId}`, updates);
    return response.data;
  }

  /**
   * Update the status of a content deliverable
   */
  async updateDeliverableStatus(
    deliverableId: string,
    status: DeliverableStatus,
    notes?: string
  ): Promise<{ deliverable_id: string; new_status: string; notes?: string; updated_at: string }> {
    const response = await api.patch(`${this.baseUrl}/${deliverableId}/status`, null, {
      params: { status, notes }
    });
    return response.data;
  }

  /**
   * Get the content narrative for a campaign
   */
  async getCampaignNarrative(campaignId: string): Promise<ContentNarrative> {
    const response = await api.get<ContentNarrative>(`${this.baseUrl}/campaign/${campaignId}/narrative`);
    return response.data;
  }

  /**
   * Delete a content deliverable
   */
  async deleteDeliverable(deliverableId: string): Promise<void> {
    await api.delete(`${this.baseUrl}/${deliverableId}`);
  }

  /**
   * Health check for content deliverables system
   */
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    database: string;
    features: string[];
  }> {
    const response = await api.get(`${this.baseUrl}/health`);
    return response.data;
  }
}

// Export singleton instance
export const contentDeliverableApi = new ContentDeliverableApi();

// Content Type Configuration for UI
export const CONTENT_TYPE_CONFIG = {
  [ContentType.blog_post]: {
    label: 'Blog Posts',
    icon: '📝',
    color: 'blue',
    description: 'In-depth articles and thought leadership content'
  },
  [ContentType.social_media_post]: {
    label: 'Social Media',
    icon: '📱',
    color: 'purple',
    description: 'Social media posts and engagement content'
  },
  [ContentType.email_campaign]: {
    label: 'Email Campaigns',
    icon: '📧',
    color: 'green',
    description: 'Email marketing and newsletter content'
  },
  [ContentType.newsletter]: {
    label: 'Newsletters',
    icon: '📰',
    color: 'indigo',
    description: 'Regular newsletter content and updates'
  },
  [ContentType.whitepaper]: {
    label: 'Whitepapers',
    icon: '📄',
    color: 'gray',
    description: 'Comprehensive research and industry reports'
  },
  [ContentType.case_study]: {
    label: 'Case Studies',
    icon: '📊',
    color: 'orange',
    description: 'Customer success stories and use cases'
  },
  [ContentType.video_script]: {
    label: 'Video Scripts',
    icon: '🎬',
    color: 'red',
    description: 'Video content scripts and storyboards'
  },
  [ContentType.podcast_script]: {
    label: 'Podcast Scripts',
    icon: '🎙️',
    color: 'pink',
    description: 'Podcast episode scripts and show notes'
  },
  [ContentType.press_release]: {
    label: 'Press Releases',
    icon: '📢',
    color: 'yellow',
    description: 'Company announcements and press coverage'
  },
  [ContentType.product_description]: {
    label: 'Product Descriptions',
    icon: '🏷️',
    color: 'teal',
    description: 'Product marketing and feature descriptions'
  },
  [ContentType.landing_page]: {
    label: 'Landing Pages',
    icon: '🎯',
    color: 'cyan',
    description: 'Conversion-focused landing page content'
  },
  [ContentType.ad_copy]: {
    label: 'Ad Copy',
    icon: '💰',
    color: 'emerald',
    description: 'Advertising copy and promotional content'
  },
  [ContentType.infographic_concept]: {
    label: 'Infographic Concepts',
    icon: '📈',
    color: 'lime',
    description: 'Visual content concepts and infographic ideas'
  },
  [ContentType.webinar_outline]: {
    label: 'Webinar Outlines',
    icon: '🖥️',
    color: 'violet',
    description: 'Webinar content structure and presentation outlines'
  },
};

// Status Configuration for UI
export const STATUS_CONFIG = {
  [DeliverableStatus.draft]: {
    label: 'Draft',
    color: 'bg-gray-100 text-gray-800',
    icon: '📝'
  },
  [DeliverableStatus.in_review]: {
    label: 'In Review',
    color: 'bg-yellow-100 text-yellow-800',
    icon: '👀'
  },
  [DeliverableStatus.approved]: {
    label: 'Approved',
    color: 'bg-green-100 text-green-800',
    icon: '✅'
  },
  [DeliverableStatus.published]: {
    label: 'Published',
    color: 'bg-blue-100 text-blue-800',
    icon: '🌐'
  },
  [DeliverableStatus.archived]: {
    label: 'Archived',
    color: 'bg-gray-100 text-gray-600',
    icon: '📦'
  },
  [DeliverableStatus.needs_revision]: {
    label: 'Needs Revision',
    color: 'bg-orange-100 text-orange-800',
    icon: '🔄'
  },
};

export default contentDeliverableApi;