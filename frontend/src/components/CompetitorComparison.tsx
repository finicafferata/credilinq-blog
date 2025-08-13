/**
 * Interactive Competitor Comparison Component
 * Allows users to compare multiple competitors side-by-side with various metrics
 */

import React, { useState, useEffect } from 'react';
import {
  ChartBarIcon,
  ScaleIcon,
  TrophyIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  EyeIcon,
  HeartIcon,
  ChatBubbleLeftRightIcon,
  GlobeAltIcon,
  XMarkIcon,
  PlusIcon,
  SwatchIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import type { CompetitorSummary } from '../types/competitor-intelligence';

interface CompetitorMetrics {
  id: string;
  name: string;
  tier: string;
  contentCount: number;
  avgEngagement: number;
  socialFollowers: number;
  trendingScore: number;
  contentQuality: number;
  publishingFrequency: number;
  sentiment: number;
  topKeywords: string[];
  platformDistribution: { [key: string]: number };
  recentGrowth: number;
}

interface ComparisonProps {
  initialCompetitors?: string[];
  maxCompetitors?: number;
  showMetrics?: string[];
}

export function CompetitorComparison({ 
  initialCompetitors = [], 
  maxCompetitors = 4,
  showMetrics = ['contentCount', 'engagement', 'trending', 'quality', 'sentiment']
}: ComparisonProps) {
  const [availableCompetitors, setAvailableCompetitors] = useState<CompetitorSummary[]>([]);
  const [selectedCompetitors, setSelectedCompetitors] = useState<string[]>(initialCompetitors);
  const [comparisonData, setComparisonData] = useState<CompetitorMetrics[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');
  const [sortBy, setSortBy] = useState<keyof CompetitorMetrics>('trendingScore');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    loadAvailableCompetitors();
  }, []);

  useEffect(() => {
    if (selectedCompetitors.length > 0) {
      loadComparisonData();
    }
  }, [selectedCompetitors]);

  const loadAvailableCompetitors = async () => {
    try {
      const competitors = await CompetitorIntelligenceAPI.listCompetitors({
        activeOnly: true
      });
      setAvailableCompetitors(competitors);
    } catch (err: any) {
      setError(err.message || 'Failed to load competitors');
    }
  };

  const loadComparisonData = async () => {
    try {
      setLoading(true);
      setError(null);

      // In a real implementation, this would make API calls to get detailed metrics
      // For now, we'll generate mock comparison data
      const mockData: CompetitorMetrics[] = selectedCompetitors.map((competitorId, index) => {
        const competitor = availableCompetitors.find(c => c.id === competitorId);
        return {
          id: competitorId,
          name: competitor?.name || `Competitor ${index + 1}`,
          tier: competitor?.tier || 'direct',
          contentCount: Math.floor(Math.random() * 100) + 20,
          avgEngagement: Math.random() * 8 + 2, // 2-10 range
          socialFollowers: Math.floor(Math.random() * 50000) + 5000,
          trendingScore: Math.floor(Math.random() * 100),
          contentQuality: Math.random() * 3 + 7, // 7-10 range
          publishingFrequency: Math.random() * 5 + 1, // posts per week
          sentiment: Math.random() * 0.8 + 0.1, // 0.1-0.9 range
          topKeywords: ['AI', 'fintech', 'innovation', 'automation', 'security'].slice(0, 3 + Math.floor(Math.random() * 2)),
          platformDistribution: {
            linkedin: Math.floor(Math.random() * 50) + 10,
            twitter: Math.floor(Math.random() * 40) + 5,
            medium: Math.floor(Math.random() * 30) + 5,
            youtube: Math.floor(Math.random() * 20) + 2,
          },
          recentGrowth: Math.random() * 50 - 25 // -25% to +25%
        };
      });

      setComparisonData(mockData);
    } catch (err: any) {
      setError(err.message || 'Failed to load comparison data');
    } finally {
      setLoading(false);
    }
  };

  const addCompetitor = (competitorId: string) => {
    if (selectedCompetitors.length < maxCompetitors && !selectedCompetitors.includes(competitorId)) {
      setSelectedCompetitors([...selectedCompetitors, competitorId]);
    }
  };

  const removeCompetitor = (competitorId: string) => {
    setSelectedCompetitors(selectedCompetitors.filter(id => id !== competitorId));
    setComparisonData(comparisonData.filter(data => data.id !== competitorId));
  };

  const getSortedData = () => {
    return [...comparisonData].sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      const multiplier = sortDirection === 'desc' ? -1 : 1;
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return (aVal - bVal) * multiplier;
      }
      return String(aVal).localeCompare(String(bVal)) * multiplier;
    });
  };

  const getMetricColor = (metric: string, value: number, maxValue: number) => {
    const percentage = (value / maxValue) * 100;
    if (percentage >= 80) return 'bg-green-500';
    if (percentage >= 60) return 'bg-blue-500';
    if (percentage >= 40) return 'bg-yellow-500';
    return 'bg-gray-400';
  };

  const formatMetricValue = (key: keyof CompetitorMetrics, value: any) => {
    switch (key) {
      case 'avgEngagement':
      case 'contentQuality':
        return typeof value === 'number' ? value.toFixed(1) : value;
      case 'publishingFrequency':
        return typeof value === 'number' ? `${value.toFixed(1)}/week` : value;
      case 'sentiment':
        return typeof value === 'number' ? (value > 0.5 ? 'Positive' : value > 0.3 ? 'Neutral' : 'Negative') : value;
      case 'socialFollowers':
        return typeof value === 'number' ? value.toLocaleString() : value;
      case 'recentGrowth':
        return typeof value === 'number' ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : value;
      default:
        return value;
    }
  };

  const getMetricIcon = (key: string) => {
    switch (key) {
      case 'contentCount':
        return <GlobeAltIcon className="h-4 w-4" />;
      case 'avgEngagement':
        return <HeartIcon className="h-4 w-4" />;
      case 'trendingScore':
        return <ArrowTrendingUpIcon className="h-4 w-4" />;
      case 'contentQuality':
        return <TrophyIcon className="h-4 w-4" />;
      case 'sentiment':
        return <ChatBubbleLeftRightIcon className="h-4 w-4" />;
      default:
        return <ChartBarIcon className="h-4 w-4" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <ScaleIcon className="h-6 w-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Competitor Comparison</h2>
          <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
            {selectedCompetitors.length}/{maxCompetitors}
          </span>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setViewMode(viewMode === 'chart' ? 'table' : 'chart')}
            className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
          >
            {viewMode === 'chart' ? 'Table View' : 'Chart View'}
          </button>
          
          <button
            onClick={loadComparisonData}
            disabled={loading}
            className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-1"
          >
            <ArrowPathIcon className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Competitor Selection */}
      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {selectedCompetitors.map((competitorId) => {
            const competitor = availableCompetitors.find(c => c.id === competitorId);
            return (
              <div key={competitorId} className="flex items-center bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                <span>{competitor?.name || 'Unknown'}</span>
                <button
                  onClick={() => removeCompetitor(competitorId)}
                  className="ml-2 hover:text-blue-600"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            );
          })}
          
          {selectedCompetitors.length < maxCompetitors && (
            <div className="relative">
              <select
                onChange={(e) => {
                  if (e.target.value) {
                    addCompetitor(e.target.value);
                    e.target.value = '';
                  }
                }}
                className="appearance-none bg-gray-50 border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">+ Add Competitor</option>
                {availableCompetitors
                  .filter(c => !selectedCompetitors.includes(c.id))
                  .map(competitor => (
                    <option key={competitor.id} value={competitor.id}>
                      {competitor.name}
                    </option>
                  ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading comparison data...</span>
        </div>
      ) : comparisonData.length === 0 ? (
        <div className="text-center py-8">
          <EyeIcon className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">Select competitors to start comparing</p>
        </div>
      ) : (
        <div>
          {viewMode === 'chart' ? (
            <div className="space-y-6">
              {/* Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {showMetrics.map((metricKey) => {
                  const maxValue = Math.max(...comparisonData.map(d => d[metricKey as keyof CompetitorMetrics] as number));
                  
                  return (
                    <div key={metricKey} className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-3">
                        {getMetricIcon(metricKey)}
                        <h3 className="font-medium text-gray-900 capitalize">
                          {metricKey.replace(/([A-Z])/g, ' $1').trim()}
                        </h3>
                      </div>
                      
                      <div className="space-y-3">
                        {getSortedData().map((competitor) => (
                          <div key={competitor.id}>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm text-gray-700 font-medium">
                                {competitor.name}
                              </span>
                              <span className="text-sm font-semibold text-gray-900">
                                {formatMetricValue(metricKey as keyof CompetitorMetrics, competitor[metricKey as keyof CompetitorMetrics])}
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${getMetricColor(metricKey, competitor[metricKey as keyof CompetitorMetrics] as number, maxValue)}`}
                                style={{
                                  width: `${Math.max(((competitor[metricKey as keyof CompetitorMetrics] as number) / maxValue) * 100, 5)}%`
                                }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Platform Distribution */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-medium text-gray-900 mb-4 flex items-center">
                  <SwatchIcon className="h-4 w-4 mr-2" />
                  Platform Distribution
                </h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 px-3 text-sm font-medium text-gray-700">Competitor</th>
                        <th className="text-center py-2 px-3 text-sm font-medium text-gray-700">LinkedIn</th>
                        <th className="text-center py-2 px-3 text-sm font-medium text-gray-700">Twitter</th>
                        <th className="text-center py-2 px-3 text-sm font-medium text-gray-700">Medium</th>
                        <th className="text-center py-2 px-3 text-sm font-medium text-gray-700">YouTube</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonData.map((competitor) => (
                        <tr key={competitor.id} className="border-b border-gray-100">
                          <td className="py-2 px-3 text-sm font-medium text-gray-900">{competitor.name}</td>
                          <td className="text-center py-2 px-3 text-sm text-gray-600">{competitor.platformDistribution.linkedin}</td>
                          <td className="text-center py-2 px-3 text-sm text-gray-600">{competitor.platformDistribution.twitter}</td>
                          <td className="text-center py-2 px-3 text-sm text-gray-600">{competitor.platformDistribution.medium}</td>
                          <td className="text-center py-2 px-3 text-sm text-gray-600">{competitor.platformDistribution.youtube}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            /* Table View */
            <div className="overflow-x-auto">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Sort by:</span>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as keyof CompetitorMetrics)}
                    className="border border-gray-300 rounded px-2 py-1 text-sm"
                  >
                    <option value="name">Name</option>
                    <option value="contentCount">Content Count</option>
                    <option value="avgEngagement">Engagement</option>
                    <option value="trendingScore">Trending Score</option>
                    <option value="contentQuality">Quality</option>
                    <option value="recentGrowth">Recent Growth</option>
                  </select>
                  <button
                    onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
                    className="p-1 hover:bg-gray-100 rounded"
                  >
                    {sortDirection === 'desc' ? 
                      <ArrowTrendingDownIcon className="h-4 w-4" /> : 
                      <ArrowTrendingUpIcon className="h-4 w-4" />
                    }
                  </button>
                </div>
              </div>

              <table className="min-w-full bg-white border border-gray-200 rounded-lg overflow-hidden">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">Competitor</th>
                    <th className="text-center py-3 px-4 font-medium text-gray-700">Tier</th>
                    <th className="text-center py-3 px-4 font-medium text-gray-700">Content</th>
                    <th className="text-center py-3 px-4 font-medium text-gray-700">Engagement</th>
                    <th className="text-center py-3 px-4 font-medium text-gray-700">Quality</th>
                    <th className="text-center py-3 px-4 font-medium text-gray-700">Sentiment</th>
                    <th className="text-center py-3 px-4 font-medium text-gray-700">Growth</th>
                  </tr>
                </thead>
                <tbody>
                  {getSortedData().map((competitor, index) => (
                    <tr key={competitor.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="py-3 px-4 font-medium text-gray-900">{competitor.name}</td>
                      <td className="text-center py-3 px-4">
                        <span className="capitalize text-sm text-gray-600">{competitor.tier}</span>
                      </td>
                      <td className="text-center py-3 px-4 text-gray-600">{competitor.contentCount}</td>
                      <td className="text-center py-3 px-4 text-gray-600">{competitor.avgEngagement.toFixed(1)}</td>
                      <td className="text-center py-3 px-4 text-gray-600">{competitor.contentQuality.toFixed(1)}/10</td>
                      <td className="text-center py-3 px-4">
                        <span className={`text-sm ${
                          competitor.sentiment > 0.5 ? 'text-green-600' : 
                          competitor.sentiment > 0.3 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {formatMetricValue('sentiment', competitor.sentiment)}
                        </span>
                      </td>
                      <td className="text-center py-3 px-4">
                        <span className={`text-sm ${
                          competitor.recentGrowth > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatMetricValue('recentGrowth', competitor.recentGrowth)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default CompetitorComparison;