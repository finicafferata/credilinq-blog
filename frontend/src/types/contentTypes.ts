// Content Deliverable Types - Separated for better import resolution

export enum ContentType {
  blog_post = 'blog_post',
  social_media_post = 'social_media_post',
  email_campaign = 'email_campaign',
  newsletter = 'newsletter',
  whitepaper = 'whitepaper',
  case_study = 'case_study',
  video_script = 'video_script',
  podcast_script = 'podcast_script',
  press_release = 'press_release',
  product_description = 'product_description',
  landing_page = 'landing_page',
  ad_copy = 'ad_copy',
  infographic_concept = 'infographic_concept',
  webinar_outline = 'webinar_outline',
}

export enum ContentFormat {
  markdown = 'markdown',
  html = 'html',
  plain_text = 'plain_text',
  json = 'json',
  structured_data = 'structured_data',
}

export enum DeliverableStatus {
  draft = 'draft',
  in_review = 'in_review',
  approved = 'approved',
  published = 'published',
  archived = 'archived',
  needs_revision = 'needs_revision',
}

export interface CampaignGenerationRequest {
  campaign_id: string;
  briefing: Record<string, any>;
  content_strategy?: Record<string, any>;
  deliverable_count?: number;
  content_types?: ContentType[];
}

export interface ContentGenerationResponse {
  campaign_id: string;
  narrative_theme: string;
  story_arc: string[];
  deliverables_created: number;
  deliverable_ids: string[];
  content_relationships: Record<string, string[]>;
  generation_summary: string;
}

export interface ContentDeliverable {
  id: string;
  title: string;
  content: string;
  summary?: string;
  content_type: ContentType;
  format: ContentFormat;
  status: DeliverableStatus;
  campaign_id: string;
  narrative_order?: number;
  key_messages: string[];
  target_audience?: string;
  tone?: string;
  platform?: string;
  word_count?: number;
  reading_time?: number;
  seo_score?: number;
  engagement_score?: number;
  created_by?: string;
  last_edited_by?: string;
  version: number;
  is_published: boolean;
  published_at?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}