/**
 * API service for Competitor Intelligence system
 */

import axios from 'axios';
import { parseApiError, AppError } from './errors';
import type {
  Competitor,
  CompetitorCreate,
  CompetitorSummary,
  ContentGap,
  Alert,
  AlertSubscription,
  DashboardData,
  AnalysisRequest,
  AnalysisJob,
  AnalysisResults,
  SystemStatus,
  Industry,
  CompetitorTier,
  ContentItem,
  Trend
} from '../types/competitor-intelligence';

const isDev = import.meta.env.DEV;
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 
  (isDev ? 'http://localhost:8000' : 'https://credilinq-blog-production.up.railway.app');

const ciApi = axios.create({
  baseURL: `${apiBaseUrl}/api/competitor-intelligence`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds for longer-running analysis operations
});

// Add response interceptor for error handling
ciApi.interceptors.response.use(
  (response) => response,
  (error) => {
    const appError = parseApiError(error);
    return Promise.reject(appError);
  }
);

export class CompetitorIntelligenceAPI {
  // Health and Status
  static async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await ciApi.get('/health');
      return response.data;
    } catch (error) {
      throw new AppError('Failed to check service health', 503, 'SERVICE_UNAVAILABLE');
    }
  }

  static async getSystemStatus(): Promise<SystemStatus> {
    try {
      const response = await ciApi.get('/status');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Competitor Management
  static async createCompetitor(competitorData: CompetitorCreate): Promise<{ success: boolean; competitor: any }> {
    try {
      const response = await ciApi.post('/competitors', competitorData);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async listCompetitors(params?: {
    industry?: Industry;
    tier?: CompetitorTier;
    activeOnly?: boolean;
  }): Promise<CompetitorSummary[]> {
    try {
      const response = await ciApi.get('/competitors', { params });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getCompetitor(competitorId: string): Promise<Competitor> {
    try {
      const response = await ciApi.get(`/competitors/${competitorId}`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async updateCompetitor(competitorId: string, updates: Partial<CompetitorCreate>): Promise<Competitor> {
    try {
      const response = await ciApi.put(`/competitors/${competitorId}`, updates);
      // API returns { success: true, message: string, competitor: Competitor }
      return response.data.competitor || response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async deleteCompetitor(competitorId: string): Promise<{ success: boolean }> {
    try {
      const response = await ciApi.delete(`/competitors/${competitorId}`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Analysis Operations
  static async startComprehensiveAnalysis(
    analysisRequest: AnalysisRequest
  ): Promise<{
    success: boolean;
    jobId: string;
    estimatedCompletion: string;
    competitorsCount: number;
    analysisDepth: string;
  }> {
    try {
      const response = await ciApi.post('/analyze/comprehensive', {
        competitor_ids: analysisRequest.competitorIds,
        industry: analysisRequest.industry,
        your_content_topics: analysisRequest.yourContentTopics || [],
        analysis_depth: analysisRequest.analysisDepth || 'standard'
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async runIncrementalMonitoring(
    competitorIds: string[],
    hoursSinceLastCheck: number = 4
  ): Promise<{
    success: boolean;
    newContentFound: number;
    alertsGenerated: number;
    nextCheckIn: string;
  }> {
    try {
      const response = await ciApi.post('/analyze/incremental', {
        competitor_ids: competitorIds,
        hours_since_last_check: hoursSinceLastCheck
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Data Retrieval
  static async getTrends(params?: {
    industry?: Industry;
    timeRangeDays?: number;
    trendType?: string;
  }): Promise<any[]> {
    try {
      const queryParams: any = {};
      if (params?.industry) queryParams.industry = params.industry;
      if (params?.timeRangeDays) queryParams.time_range_days = params.timeRangeDays;
      if (params?.trendType) queryParams.trend_type = params.trendType;
      
      const response = await ciApi.get('/trends', { params: queryParams });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getInsights(params?: {
    industry?: Industry;
    competitorId?: string;
    insightType?: string;
    daysBack?: number;
  }): Promise<any[]> {
    try {
      const queryParams: any = {};
      if (params?.industry) queryParams.industry = params.industry;
      if (params?.competitorId) queryParams.competitor_id = params.competitorId;
      if (params?.insightType) queryParams.insight_type = params.insightType;
      if (params?.daysBack) queryParams.days_back = params.daysBack;
      
      const response = await ciApi.get('/insights', { params: queryParams });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getContentGaps(params?: {
    industry?: Industry;
    minOpportunityScore?: number;
  }): Promise<ContentGap[]> {
    try {
      const response = await ciApi.get('/gaps', { params });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getCompetitorContent(
    competitorId: string,
    params?: {
      contentType?: string;
      platform?: string;
      daysBack?: number;
      limit?: number;
    }
  ): Promise<ContentItem[]> {
    try {
      const response = await ciApi.get(`/competitors/${competitorId}/content`, { params });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Dashboard
  static async getDashboardData(params: {
    industry: Industry;
    competitorIds?: string[];
    timeRangeDays?: number;
  }): Promise<DashboardData> {
    try {
      const response = await ciApi.get('/dashboard', { params });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Content Monitoring
  static async monitorCompetitor(competitorId: string): Promise<any> {
    try {
      const response = await ciApi.post(`/competitors/${competitorId}/monitor`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async runMonitoringForAll(): Promise<any> {
    try {
      const response = await ciApi.post('/monitoring/run-all');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async runPricingDetectionAll(): Promise<any> {
    try {
      const response = await ciApi.post('/detect/pricing/run-all');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async runCopyDetectionAll(): Promise<any> {
    try {
      const response = await ciApi.post('/detect/copy/run-all');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getRecentChanges(params?: {
    competitorId?: string;
    changeType?: string;
    limit?: number;
  }): Promise<import('../types/competitor-intelligence').ChangeEvent[]> {
    try {
      const queryParams: any = {}
      if (params?.competitorId) queryParams.competitor_id = params.competitorId
      if (params?.changeType) queryParams.change_type = params.changeType
      if (params?.limit) queryParams.limit = params.limit
      const response = await ciApi.get('/changes/recent', { params: queryParams })
      return response.data
    } catch (error) {
      throw parseApiError(error)
    }
  }

  // Alerts
  static async getAlerts(params?: {
    limit?: number;
    priority?: string;
    alertType?: string;
    competitorId?: string;
    unreadOnly?: boolean;
  }): Promise<Alert[]> {
    try {
      const queryParams: any = {};
      if (typeof params?.limit === 'number') queryParams.limit = params.limit;
      if (params?.priority) queryParams.priority = params.priority;
      if (params?.alertType) queryParams.alert_type = params.alertType;
      if (params?.competitorId) queryParams.competitor_id = params.competitorId;
      if (typeof params?.unreadOnly === 'boolean') queryParams.unread_only = params.unreadOnly;
      const response = await ciApi.get('/alerts', { params: queryParams });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getAlertsGrouped(params?: {
    daysBack?: number;
    limitPerCompetitor?: number;
    priority?: string;
    alertType?: string;
    competitorIds?: string[];
    unreadOnly?: boolean;
  }): Promise<Array<{ competitorId: string | null; competitorName: string | null; alerts: Alert[] }>> {
    try {
      const queryParams: any = {};
      if (params?.daysBack) queryParams.days_back = params.daysBack;
      if (params?.limitPerCompetitor) queryParams.limit_per_competitor = params.limitPerCompetitor;
      if (params?.priority) queryParams.priority = params.priority;
      if (params?.alertType) queryParams.alert_type = params.alertType;
      if (params?.competitorIds && params.competitorIds.length) queryParams.competitor_ids = params.competitorIds;
      if (typeof params?.unreadOnly === 'boolean') queryParams.unread_only = params.unreadOnly;
      const response = await ciApi.get('/alerts/grouped', { params: queryParams });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async markAlertRead(alertId: string): Promise<{ success: boolean }> {
    try {
      const response = await ciApi.post(`/alerts/${alertId}/read`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async dismissAlert(alertId: string): Promise<{ success: boolean }> {
    try {
      const response = await ciApi.post(`/alerts/${alertId}/dismiss`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getAlert(alertId: string): Promise<any> {
    try {
      const response = await ciApi.get(`/alerts/${alertId}`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getAlertSummary(): Promise<any> {
    try {
      const response = await ciApi.get('/alerts/summary');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async createAlertSubscription(subscription: AlertSubscription): Promise<{
    success: boolean;
    subscriptionId: string;
  }> {
    try {
      const response = await ciApi.post('/alerts/subscribe', subscription);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async acknowledgeAlert(alertId: string): Promise<{ success: boolean }> {
    try {
      const response = await ciApi.post(`/alerts/${alertId}/acknowledge`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Analysis Jobs & Results
  static async getAnalysisJob(jobId: string): Promise<AnalysisJob> {
    try {
      const response = await ciApi.get(`/analysis/jobs/${jobId}`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getAnalysisResults(reportId: string): Promise<AnalysisResults> {
    try {
      const response = await ciApi.get(`/analysis/results/${reportId}`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async exportAnalysisResults(
    reportId: string,
    format: 'pdf' | 'csv' | 'json' = 'pdf'
  ): Promise<Blob> {
    try {
      const response = await ciApi.get(`/analysis/results/${reportId}/export`, {
        params: { format },
        responseType: 'blob'
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Search and Filtering
  static async searchContent(query: string, params?: {
    competitorIds?: string[];
    contentTypes?: string[];
    platforms?: string[];
    dateRange?: { start: string; end: string };
  }): Promise<ContentItem[]> {
    try {
      const response = await ciApi.post('/search/content', { query, ...params });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async searchTrends(query: string, params?: {
    industry?: Industry;
    minStrength?: string;
  }): Promise<Trend[]> {
    try {
      const response = await ciApi.post('/search/trends', { query, ...params });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Phase 4: AI Content Analysis
  static async analyzeContent(params: {
    content: string;
    contentUrl?: string;
    competitorName?: string;
    contentType?: string;
  }): Promise<any> {
    try {
      const response = await ciApi.post('/analyze/content', params);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async batchAnalyzeContent(contentItems: Array<{
    content: string;
    content_url?: string;
    competitor_name?: string;
    content_type?: string;
  }>): Promise<any> {
    try {
      const response = await ciApi.post('/analyze/batch-content', { content_items: contentItems });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Phase 4: Advanced Reporting
  static async generateReport(params: {
    reportType: string;
    format: string;
    title: string;
    description?: string;
    includeCharts?: boolean;
    includeRawData?: boolean;
    dateRangeDays?: number;
    competitorIds?: string[];
    industry?: string;
    customSections?: string[];
  }): Promise<any> {
    try {
      const response = await ciApi.post('/reports/generate', {
        report_type: params.reportType,
        format: params.format,
        title: params.title,
        description: params.description,
        include_charts: params.includeCharts,
        include_raw_data: params.includeRawData,
        date_range_days: params.dateRangeDays,
        competitor_ids: params.competitorIds,
        industry: params.industry,
        custom_sections: params.customSections
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async downloadReport(reportId: string): Promise<Blob> {
    try {
      const response = await ciApi.get(`/reports/download/${reportId}`, {
        responseType: 'blob'
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  // Phase 4: External Integrations
  static async registerIntegration(params: {
    integrationType: string;
    name: string;
    webhookUrl?: string;
    apiToken?: string;
    channel?: string;
    emailSettings?: Record<string, string>;
    eventFilters?: string[];
    priorityThreshold?: string;
    enabled?: boolean;
  }): Promise<any> {
    try {
      const response = await ciApi.post('/integrations/register', {
        integration_type: params.integrationType,
        name: params.name,
        webhook_url: params.webhookUrl,
        api_token: params.apiToken,
        channel: params.channel,
        email_settings: params.emailSettings,
        event_filters: params.eventFilters,
        priority_threshold: params.priorityThreshold,
        enabled: params.enabled
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async getIntegrationsStatus(): Promise<any> {
    try {
      const response = await ciApi.get('/integrations/status');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async testIntegration(integrationName: string): Promise<any> {
    try {
      const response = await ciApi.post(`/integrations/${integrationName}/test`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async removeIntegration(integrationName: string): Promise<any> {
    try {
      const response = await ciApi.delete(`/integrations/${integrationName}`);
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async sendCustomNotification(params: {
    title: string;
    content: string;
    eventType?: string;
    priority?: string;
    integrationNames?: string[];
    data?: Record<string, any>;
  }): Promise<any> {
    try {
      const response = await ciApi.post('/integrations/notify', {
        title: params.title,
        content: params.content,
        event_type: params.eventType,
        priority: params.priority,
        integration_names: params.integrationNames,
        data: params.data
      });
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }

  static async broadcastSystemHealth(): Promise<any> {
    try {
      const response = await ciApi.post('/integrations/broadcast/health');
      return response.data;
    } catch (error) {
      throw parseApiError(error);
    }
  }
}

// Export individual methods for easier testing
export const {
  healthCheck,
  getSystemStatus,
  createCompetitor,
  listCompetitors,
  getCompetitor,
  updateCompetitor,
  deleteCompetitor,
  startComprehensiveAnalysis,
  runIncrementalMonitoring,
  getTrends,
  getContentGaps,
  getInsights,
  getCompetitorContent,
  getDashboardData,
  getAlerts,
  createAlertSubscription,
  acknowledgeAlert,
  getAnalysisJob,
  getAnalysisResults,
  exportAnalysisResults,
  searchContent,
  searchTrends
} = CompetitorIntelligenceAPI;

// Default export
export default CompetitorIntelligenceAPI;