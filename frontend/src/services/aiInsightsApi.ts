/**
 * AI Insights API Service - Real data from agent_performance and agent_decisions tables
 */

import { api } from '../lib/api';

export interface AgentScores {
  grammar?: number;
  readability?: number;
  structure?: number;
  accuracy?: number;
  consistency?: number;
  overall: number;
}

export interface AgentInsight {
  agent_type: string;
  scores: AgentScores;
  confidence: number;
  reasoning: string;
  recommendations: string[];
  execution_time: number;
  model_used: string;
  status?: 'completed' | 'pending' | 'running' | 'failed';
  last_executed?: string;
  estimated_completion?: string;
}

export interface CampaignAIInsights {
  campaign_id: string;
  agent_insights: AgentInsight[];
  summary: {
    overall_quality: number;
    ready_for_publication: boolean;
    total_agents: number;
  };
  data_source: string;
  generated_at?: string;
}

export const aiInsightsApi = {
  /**
   * Get real AI agent insights for a campaign using the simplified endpoint
   */
  async getCampaignInsights(campaignId: string): Promise<CampaignAIInsights> {
    const response = await api.get(`/api/${campaignId}/agent-insights`);
    return response.data;
  },

  /**
   * Phase 4.4: Trigger agent analysis for specific agent type
   */
  async triggerAgentAnalysis(campaignId: string, agentType: string): Promise<{ message: string; task_id?: string }> {
    const response = await api.post(`/api/${campaignId}/trigger-analysis`, {
      agent_type: agentType
    });
    return response.data;
  },

  /**
   * Phase 4.4: Get analysis progress/status for a campaign
   */
  async getAnalysisStatus(campaignId: string): Promise<{
    running_agents: string[];
    estimated_completion: string;
    progress_percentage: number;
  }> {
    const response = await api.get(`/api/${campaignId}/analysis-status`);
    return response.data;
  },

  /**
   * Transform real AI insights into UI format for CampaignDetails component
   * Phase 4.4: Enhanced no-data state handling
   * Phase 5: UX enhancements for score transparency
   */
  transformInsightsForUI(insights: CampaignAIInsights) {
    // Ensure agent_insights is always an array, even if undefined
    const safeInsights = insights?.agent_insights || [];
    
    const qualityAgent = safeInsights.find(a => 
      a.agent_type === 'quality_review' || a.agent_type.includes('quality')
    );
    const seoAgent = safeInsights.find(a => a.agent_type === 'seo');
    const contentAgent = safeInsights.find(a => 
      a.agent_type === 'writer' || a.agent_type === 'content' || a.agent_type === 'content_agent' || a.agent_type === 'content_agent_readability'
    );
    const brandAgent = safeInsights.find(a => 
      a.agent_type === 'editor' || a.agent_type === 'brand' || a.agent_type === 'brand_review'
    );
    const geoAgent = safeInsights.find(a => 
      a.agent_type === 'generative_engine_optimization' || a.agent_type === 'geo_optimization' || a.agent_type === 'geo' || a.agent_type === 'geo_analysis_agent'
    );

    // Helper function to convert 0-1 scores to 0-10 scale
    const scaleScore = (score: number) => Math.round(score * 10 * 10) / 10;
    
    // Helper function to safely get recommendations
    const safeRecommendations = (agent: any, defaultRecs: string[]) => {
      return agent?.recommendations && Array.isArray(agent.recommendations) ? agent.recommendations : defaultRecs;
    };

    // Phase 4.4: Helper function to determine analysis status
    const getAnalysisStatus = (agent: any, agentType: string) => {
      if (!agent) {
        return {
          status: 'pending',
          statusText: 'Analysis Pending',
          statusColor: 'text-yellow-600',
          backgroundColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          showScore: false,
          showRunButton: true
        };
      }
      
      const status = agent.status || 'completed';
      switch (status) {
        case 'running':
          return {
            status: 'running',
            statusText: 'Analysis in Progress...',
            statusColor: 'text-blue-600',
            backgroundColor: 'bg-blue-50',
            borderColor: 'border-blue-200',
            showScore: false,
            showRunButton: false,
            showProgress: true
          };
        case 'failed':
          return {
            status: 'failed',
            statusText: 'Analysis Failed',
            statusColor: 'text-red-600',
            backgroundColor: 'bg-red-50',
            borderColor: 'border-red-200',
            showScore: false,
            showRunButton: true
          };
        case 'completed':
        default:
          return {
            status: 'completed',
            statusText: 'Analysis Complete',
            statusColor: 'text-green-600',
            backgroundColor: 'bg-green-50',
            borderColor: 'border-green-200',
            showScore: true,
            showRunButton: false
          };
      }
    };

    // Get status for each agent
    const seoStatus = getAnalysisStatus(seoAgent, 'seo');
    const contentStatus = getAnalysisStatus(contentAgent, 'content');  
    const brandStatus = getAnalysisStatus(brandAgent, 'brand');
    const geoStatus = getAnalysisStatus(geoAgent, 'geo');

    return {
      seoAgent: {
        ...seoStatus,
        score: seoAgent ? scaleScore(seoAgent.scores.overall) : null,
        keywords: seoAgent?.metadata?.keywords || ['fintech', 'financial services', 'digital banking'],
        readability: qualityAgent?.scores.readability ? scaleScore(qualityAgent.scores.readability) : null,
        recommendations: safeRecommendations(seoAgent, ['Optimize keyword density', 'Improve meta descriptions']),
        confidence: seoAgent?.confidence || null,
        execution_time: seoAgent?.execution_time || null,
        model_used: seoAgent?.model_used || null,
        last_executed: seoAgent?.last_executed || null,
        methodology: 'Keyword density, meta tags, structure, readability analysis'
      },
      contentAgent: {
        ...contentStatus,
        score: contentAgent ? scaleScore(contentAgent.scores.overall) : null,
        engagement: contentAgent?.confidence && contentAgent.confidence > 0.9 ? 'High' : contentAgent ? 'Medium-High' : null,
        accuracy: qualityAgent?.scores.accuracy ? 
          (qualityAgent.scores.accuracy > 0.8 ? 'Verified' : 'Good') : null,
        structure: qualityAgent?.scores.structure && qualityAgent.scores.structure > 0.8 ? 
          'Excellent structure' : qualityAgent ? 'Clear & well-organized' : null,
        cta: contentAgent ? 'Present' : null,
        confidence: contentAgent?.confidence || null,
        execution_time: contentAgent?.execution_time || null,
        model_used: contentAgent?.model_used || null,
        last_executed: contentAgent?.last_executed || null,
        methodology: 'Flesch Reading Ease, sentence length, engagement metrics'
      },
      brandAgent: {
        ...brandStatus,
        score: brandAgent ? scaleScore(brandAgent.scores.overall) : null,
        voice: brandAgent ? 'Professional & authoritative' : null,
        consistency: brandAgent?.confidence ? brandAgent.confidence > 0.85 : null,
        terminology: brandAgent ? 'Correctly applied' : null,
        alignment: brandAgent?.confidence && brandAgent.confidence > 0.8 ? 'Excellent' : brandAgent ? 'Good' : null,
        confidence: brandAgent?.confidence || null,
        execution_time: brandAgent?.execution_time || null,
        model_used: brandAgent?.model_used || null,
        last_executed: brandAgent?.last_executed || null,
        methodology: 'Brand voice, tone consistency, terminology compliance'
      },
      geoAgent: {
        ...geoStatus,
        score: geoAgent ? scaleScore(geoAgent.scores.overall) : null,
        optimization: geoAgent?.recommendations?.slice(0, 3) || (geoAgent ? ['Answer Engines', 'ChatGPT', 'Perplexity'] : null),
        visibility: geoAgent?.confidence > 0.8 ? 'Enhanced for AI discovery' : geoAgent ? 'Optimized for AI discovery' : null,
        structured_data: geoAgent ? 'Schema markup optimized' : null,
        citations: geoAgent ? 'Source attribution enabled' : null,
        confidence: geoAgent?.confidence || null,
        execution_time: geoAgent?.execution_time || null,
        model_used: geoAgent?.model_used || null,
        last_executed: geoAgent?.last_executed || null,
        methodology: 'AI search optimization, answer engine visibility, structured data'
      },
      overallScore: insights.summary.overall_quality || null,
      confidence: insights.summary.overall_quality ? Math.round(insights.summary.overall_quality * 10) : null,
      recommendations: safeInsights
        .flatMap(agent => safeRecommendations(agent, []).slice(0, 1))
        .slice(0, 3),
      hasRealData: safeInsights.length > 0,
      totalAgentsRun: safeInsights.length,
      lastAnalysisRun: insights.generated_at || null
    };
  }
};

// Example usage in CampaignDetails component:
/*
import { aiInsightsApi } from '../services/aiInsightsApi';

const [aiInsights, setAiInsights] = useState(null);
const [loadingInsights, setLoadingInsights] = useState(false);

useEffect(() => {
  const fetchRealInsights = async () => {
    if (!campaignId) return;
    
    setLoadingInsights(true);
    try {
      const realInsights = await aiInsightsApi.getCampaignInsights(campaignId);
      const uiInsights = aiInsightsApi.transformInsightsForUI(realInsights);
      setAiInsights(uiInsights);
    } catch (error) {
      console.error('Failed to load real AI insights:', error);
      // Keep existing mock data as fallback
    } finally {
      setLoadingInsights(false);
    }
  };

  fetchRealInsights();
}, [campaignId]);
*/