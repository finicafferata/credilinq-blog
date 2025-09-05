/**
 * Enhanced Agent Insights Component
 * Phase 4.4: Better no-data state handling
 * Phase 5: UX enhancements for score transparency
 */

import React, { useState } from 'react';
import { 
  Search, 
  FileText, 
  Shield, 
  Target, 
  Play, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  RefreshCw,
  Info,
  Loader2,
  TrendingUp,
  Eye,
  Star
} from 'lucide-react';
import { aiInsightsApi } from '../services/aiInsightsApi';

interface AgentInsightProps {
  agent: any;
  agentType: string;
  icon: React.ComponentType<any>;
  title: string;
  campaignId: string;
  onRefresh: () => void;
}

const AgentInsightCard: React.FC<AgentInsightProps> = ({ 
  agent, 
  agentType, 
  icon: Icon, 
  title, 
  campaignId,
  onRefresh 
}) => {
  const [isRunning, setIsRunning] = useState(false);

  const handleRunAnalysis = async () => {
    setIsRunning(true);
    try {
      await aiInsightsApi.triggerAgentAnalysis(campaignId, agentType);
      // Refresh after a short delay to allow backend processing
      setTimeout(() => {
        onRefresh();
        setIsRunning(false);
      }, 2000);
    } catch (error) {
      console.error('Failed to trigger analysis:', error);
      setIsRunning(false);
    }
  };

  const getScoreColor = (score: number | null) => {
    if (score === null) return 'text-gray-400';
    if (score >= 8.5) return 'text-green-600';
    if (score >= 7.0) return 'text-yellow-600'; 
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number | null) => {
    if (score === null) return 'bg-gray-50';
    if (score >= 8.5) return 'bg-green-50';
    if (score >= 7.0) return 'bg-yellow-50';
    return 'bg-red-50';
  };

  const formatLastExecuted = (lastExecuted: string | null) => {
    if (!lastExecuted) return null;
    const date = new Date(lastExecuted);
    return date.toLocaleString();
  };

  return (
    <div className={`${agent?.backgroundColor || 'bg-white'} p-4 rounded-lg ${agent?.borderColor || 'border border-gray-200'}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Icon className={`w-4 h-4 ${agent?.statusColor || 'text-gray-600'}`} />
          <span className="font-medium text-sm">{title}</span>
        </div>
        
        {/* Status indicator and actions */}
        <div className="flex items-center space-x-2">
          {agent?.status === 'running' && (
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
          )}
          
          {agent?.showRunButton && (
            <button
              onClick={handleRunAnalysis}
              disabled={isRunning}
              className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50"
            >
              {isRunning ? 'Running...' : 'Run Analysis'}
            </button>
          )}
          
          {agent?.showProgress && (
            <div className="w-16 bg-gray-200 rounded-full h-2">
              <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
            </div>
          )}
        </div>
      </div>

      {/* Status message */}
      <div className={`text-xs font-medium mb-3 ${agent?.statusColor || 'text-gray-600'}`}>
        {agent?.statusText || 'Ready'}
      </div>

      {/* Score display or pending state */}
      {agent?.showScore && agent?.score !== null ? (
        <div className="space-y-2">
          <div className={`text-2xl font-bold ${getScoreColor(agent.score)}`}>
            {agent.score}/10
            {agent.confidence && (
              <span className="text-sm font-normal text-gray-500 ml-2">
                ({Math.round(agent.confidence * 100)}% confidence)
              </span>
            )}
          </div>
          
          {/* Agent-specific metrics */}
          <div className="text-sm text-gray-600 space-y-1">
            {agentType === 'seo' && (
              <>
                <div>Keywords: {agent.keywords?.join(', ') || 'None'}</div>
                {agent.readability && <div>Readability: {agent.readability}/10</div>}
              </>
            )}
            
            {agentType === 'content' && (
              <>
                {agent.engagement && <div>Engagement: {agent.engagement}</div>}
                {agent.accuracy && <div>Accuracy: {agent.accuracy}</div>}
                {agent.structure && <div>Structure: {agent.structure}</div>}
              </>
            )}
            
            {agentType === 'brand' && (
              <>
                {agent.voice && <div>Voice: {agent.voice}</div>}
                {agent.alignment && <div>Alignment: {agent.alignment}</div>}
              </>
            )}
            
            {agentType === 'geo' && (
              <>
                {agent.visibility && <div>AI Visibility: {agent.visibility}</div>}
                {agent.structured_data && <div>Structured Data: {agent.structured_data}</div>}
              </>
            )}
          </div>
          
          {/* Recommendations */}
          {agent.recommendations && agent.recommendations.length > 0 && (
            <div className="mt-3">
              <div className="text-xs font-medium text-gray-700 mb-1">Recommendations:</div>
              <ul className="text-xs text-gray-600 space-y-1">
                {agent.recommendations.slice(0, 2).map((rec: string, idx: number) => (
                  <li key={idx} className="flex items-start space-x-1">
                    <span className="text-blue-500 mt-0.5">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Phase 5: Methodology tooltip */}
          {agent.methodology && (
            <div className="mt-3 p-2 bg-gray-50 rounded text-xs">
              <div className="font-medium text-gray-700 mb-1 flex items-center">
                <Info className="w-3 h-3 mr-1" />
                Methodology
              </div>
              <div className="text-gray-600">{agent.methodology}</div>
            </div>
          )}
          
          {/* Execution details */}
          {(agent.execution_time || agent.model_used || agent.last_executed) && (
            <div className="mt-3 pt-2 border-t border-gray-100 text-xs text-gray-500 space-y-1">
              {agent.execution_time && <div>Execution: {agent.execution_time}ms</div>}
              {agent.model_used && <div>Model: {agent.model_used}</div>}
              {agent.last_executed && <div>Last run: {formatLastExecuted(agent.last_executed)}</div>}
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-4">
          <div className="text-gray-400 mb-2">
            <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <div className="text-sm">No analysis available</div>
            <div className="text-xs">Click "Run Analysis" to start</div>
          </div>
        </div>
      )}
    </div>
  );
};

interface EnhancedAgentInsightsProps {
  aiInsights: any;
  campaignId: string;
  loadingInsights: boolean;
  onRefresh: () => void;
}

export const EnhancedAgentInsights: React.FC<EnhancedAgentInsightsProps> = ({
  aiInsights,
  campaignId,
  loadingInsights,
  onRefresh
}) => {
  if (loadingInsights) {
    return (
      <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
        <div className="flex items-center space-x-2 mb-4">
          <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
          <span className="text-lg font-medium text-gray-900">Loading AI Agent Insights...</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="bg-white p-4 rounded-lg border border-gray-200 animate-pulse">
              <div className="h-4 bg-gray-200 rounded mb-2"></div>
              <div className="h-8 bg-gray-200 rounded mb-2"></div>
              <div className="h-3 bg-gray-200 rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-lg font-medium text-gray-900 flex items-center space-x-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span>AI Agent Insights</span>
          {aiInsights?.hasRealData && (
            <span className="text-sm text-green-600 font-normal">
              ({aiInsights.totalAgentsRun} agents analyzed)
            </span>
          )}
        </h4>
        
        <button
          onClick={onRefresh}
          className="text-sm px-3 py-1 bg-white border border-gray-200 rounded hover:bg-gray-50 flex items-center space-x-1"
        >
          <RefreshCw className="w-3 h-3" />
          <span>Refresh</span>
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <AgentInsightCard
          agent={aiInsights?.seoAgent}
          agentType="seo"
          icon={Search}
          title="SEO Agent"
          campaignId={campaignId}
          onRefresh={onRefresh}
        />
        
        <AgentInsightCard
          agent={aiInsights?.contentAgent}
          agentType="content"
          icon={FileText}
          title="Content Agent"
          campaignId={campaignId}
          onRefresh={onRefresh}
        />
        
        <AgentInsightCard
          agent={aiInsights?.brandAgent}
          agentType="brand"
          icon={Shield}
          title="Brand Agent"
          campaignId={campaignId}
          onRefresh={onRefresh}
        />
        
        <AgentInsightCard
          agent={aiInsights?.geoAgent}
          agentType="geo"
          icon={Target}
          title="GEO Agent"
          campaignId={campaignId}
          onRefresh={onRefresh}
        />
      </div>
      
      {/* Overall insights summary */}
      {aiInsights?.hasRealData && (
        <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-gray-700">Overall Analysis</div>
              {aiInsights.overallScore !== null && (
                <div className="text-lg font-bold text-green-600">
                  {(aiInsights.overallScore * 10).toFixed(1)}/10
                  <span className="text-sm font-normal text-gray-500 ml-2">
                    Quality Score
                  </span>
                </div>
              )}
            </div>
            
            {aiInsights.lastAnalysisRun && (
              <div className="text-xs text-gray-500">
                Last analysis: {new Date(aiInsights.lastAnalysisRun).toLocaleString()}
              </div>
            )}
          </div>
          
          {aiInsights.recommendations && aiInsights.recommendations.length > 0 && (
            <div className="mt-2">
              <div className="text-xs font-medium text-gray-700 mb-1">Top Recommendations:</div>
              <div className="text-xs text-gray-600 space-y-1">
                {aiInsights.recommendations.map((rec: string, idx: number) => (
                  <div key={idx} className="flex items-start space-x-1">
                    <Star className="w-3 h-3 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span>{rec}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EnhancedAgentInsights;