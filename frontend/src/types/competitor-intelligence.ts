/**
 * TypeScript types for Competitor Intelligence system
 */

export enum ContentType {
  BLOG_POST = 'blog_post',
  SOCIAL_MEDIA_POST = 'social_media_post',
  VIDEO = 'video',
  PODCAST = 'podcast',
  WHITEPAPER = 'whitepaper',
  CASE_STUDY = 'case_study',
  WEBINAR = 'webinar',
  EMAIL_NEWSLETTER = 'email_newsletter',
  PRESS_RELEASE = 'press_release',
  PRODUCT_UPDATE = 'product_update'
}

export enum Platform {
  WEBSITE = 'website',
  LINKEDIN = 'linkedin',
  TWITTER = 'twitter',
  FACEBOOK = 'facebook',
  INSTAGRAM = 'instagram',
  YOUTUBE = 'youtube',
  MEDIUM = 'medium',
  SUBSTACK = 'substack',
  TIKTOK = 'tiktok',
  REDDIT = 'reddit'
}

export enum Industry {
  FINTECH = 'fintech',
  SAAS = 'saas',
  ECOMMERCE = 'ecommerce',
  HEALTHCARE = 'healthcare',
  EDUCATION = 'education',
  MARKETING = 'marketing',
  TECHNOLOGY = 'technology',
  FINANCE = 'finance',
  RETAIL = 'retail',
  MEDIA = 'media'
}

export enum TrendStrength {
  WEAK = 'weak',
  MODERATE = 'moderate',
  STRONG = 'strong',
  VIRAL = 'viral'
}

export enum AlertPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum CompetitorTier {
  DIRECT = 'direct',
  INDIRECT = 'indirect',
  ASPIRATIONAL = 'aspirational',
  ADJACENT = 'adjacent'
}

// Data Models
export interface Competitor {
  id: string;
  name: string;
  domain: string;
  tier: CompetitorTier;
  industry: Industry;
  description: string;
  platforms: Platform[];
  monitoringKeywords: string[];
  lastMonitored?: string;
  isActive: boolean;
  metadata?: any;
  createdAt: string;
  updatedAt: string;
}

export interface ContentItem {
  id: string;
  competitorId: string;
  title: string;
  content: string;
  contentType: ContentType;
  platform: Platform;
  url: string;
  publishedAt: string;
  discoveredAt: string;
  author?: string;
  engagementMetrics?: {
    likes?: number;
    shares?: number;
    comments?: number;
    views?: number;
  };
  keywords: string[];
  sentimentScore?: number;
  qualityScore?: number;
  metadata?: any;
}

export interface Trend {
  id: string;
  topic: string;
  keywords: string[];
  industry: Industry;
  strength: TrendStrength;
  growthRate: number;
  firstDetected: string;
  lastUpdated: string;
  peakDate?: string;
  competitorsUsing: string[];
  opportunityScore?: number;
  metadata?: any;
}

export interface ContentGap {
  id: string;
  topic: string;
  description: string;
  opportunityScore: number;
  difficultyScore: number;
  potentialReach: number;
  contentTypesMissing: ContentType[];
  platformsMissing: Platform[];
  keywords: string[];
  competitorsCovering: string[];
  suggestedApproach?: string;
  identifiedAt: string;
  metadata?: any;
}

export interface CompetitorInsight {
  id: string;
  competitorId: string;
  insightType: string;
  title: string;
  description: string;
  confidenceScore: number;
  impactLevel: 'low' | 'medium' | 'high';
  supportingEvidence: string[];
  recommendations: string[];
  generatedAt: string;
  expiresAt?: string;
  metadata?: any;
}

export interface Alert {
  id: string;
  alertType: string;
  priority: AlertPriority;
  title: string;
  message: string;
  data: any;
  competitorIds: string[];
  trendIds: string[];
  contentIds: string[];
  createdAt: string;
  sentAt?: string;
  acknowledgedAt?: string;
  expiresAt?: string;
  recipients: string[];
  metadata?: any;
}

// API Request/Response Types
export interface CompetitorCreate {
  name: string;
  domain: string;
  tier: CompetitorTier;
  industry: Industry;
  description: string;
  platforms?: Platform[];
  monitoringKeywords?: string[];
}

export interface MonitoringConfig {
  checkFrequencyHours?: number;
  contentTypes?: ContentType[];
  platforms?: Platform[];
  keywords?: string[];
  sentimentAnalysis?: boolean;
  qualityScoring?: boolean;
  maxContentAgeDays?: number;
}

export interface AlertSubscription {
  userId: string;
  alertTypes: string[];
  competitors?: string[];
  keywords?: string[];
  priorityThreshold?: AlertPriority;
  deliveryChannels?: string[];
  frequencyLimit?: number;
}

// Summary Types for UI
export interface CompetitorSummary {
  id: string;
  name: string;
  tier: CompetitorTier;
  industry: Industry;
  contentCount: number;
  lastActivity?: string;
  avgEngagement: number;
  trendingScore: number;
}

export interface TrendSummary {
  id: string;
  topic: string;
  strength: TrendStrength;
  growthRate: number;
  opportunityScore?: number;
  competitorsCount: number;
  contentCount: number;
}

export interface InsightSummary {
  id: string;
  competitorName: string;
  insightType: string;
  title: string;
  confidenceScore: number;
  impactLevel: string;
  generatedAt: string;
}

// Dashboard Data Types
export interface DashboardOverview {
  competitorsMonitored: number;
  industry: string;
  lastUpdated: string;
  analysisStatus: 'ready' | 'running' | 'up_to_date' | 'stale';
}

export interface KeyMetrics {
  totalContentAnalyzed: number;
  trendsIdentified: number;
  opportunitiesFound: number;
  insightsGenerated: number;
  highPriorityOpportunities: number;
}

export interface DashboardData {
  overview: DashboardOverview;
  keyMetrics: KeyMetrics;
  topTrends?: TrendSummary[];
  topOpportunities?: ContentGap[];
  competitorPerformance?: any[];
  recentInsights?: InsightSummary[];
  message?: string;
}

// Analysis Types
export interface AnalysisRequest {
  competitorIds: string[];
  industry: Industry;
  yourContentTopics?: string[];
  analysisDepth?: 'basic' | 'standard' | 'deep';
}

export interface AnalysisJob {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  estimatedCompletion?: string;
  competitorsCount: number;
  analysisDepth: string;
  createdAt: string;
  completedAt?: string;
  error?: string;
}

export interface AnalysisResults {
  reportId: string;
  generatedAt: string;
  analysisPeriod: {
    start: string;
    end: string;
  };
  competitorsAnalyzed: CompetitorSummary[];
  keyInsights: InsightSummary[];
  trendingTopics: TrendSummary[];
  contentGaps: ContentGap[];
  performanceBenchmarks: any;
  recommendations: string[];
  marketOverview: any;
}

// System Status
export interface SystemStatus {
  orchestratorStatus: string;
  lastFullAnalysis?: string;  
  lastIncrementalAnalysis?: string;
  cacheStatus: 'valid' | 'stale';
  agentStatus: {
    contentMonitor: any;
    trendAnalyzer: string;
    gapIdentifier: string;
    performanceAnalyzer: string;
    strategicInsights: string;
    alertOrchestrator: any;
  };
  configuration: any;
  systemHealth: string;
}

// Change Events for Recent Activity
export interface ChangeEvent {
  id?: string | number;
  competitor_id: string;
  competitor_name: string;
  source: string; // website | blog | pricing | news | social
  change_type: string; // pricing | product | plan_copy | news | social
  url?: string;
  detected_at: string;
  confidence?: number;
  sentiment?: string;
  old_value?: any;
  new_value?: any;
  metadata?: any;
}