/**
 * Advanced Trend Analysis Component
 * Provides interactive trend visualization and analysis tools
 */

import React, { useState, useEffect } from 'react';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  FireIcon,
  LightBulbIcon,
  ClockIcon,
  EyeIcon,
  ChartBarIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  ShareIcon,
  BookmarkIcon,
  StarIcon,
  TagIcon,
} from '@heroicons/react/24/outline';
import {
  FireIcon as FireIconSolid,
  StarIcon as StarIconSolid,
} from '@heroicons/react/24/solid';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import type { Industry } from '../types/competitor-intelligence';

interface TrendData {
  id: string;
  title: string;
  description: string;
  trend_type: string;
  strength: 'weak' | 'moderate' | 'strong' | 'viral';
  confidence: number;
  growth_rate: number;
  peak_date?: string;
  competitors_involved: string[];
  keywords: string[];
  data_points: Array<{
    date: string;
    value: number;
    mentions: number;
  }>;
  sentiment: number;
  opportunity_score: number;
  threat_level: 'low' | 'medium' | 'high';
  created_at: string;
  metadata: {
    category: string;
    industry: string;
    geographical_focus?: string;
    audience_size?: number;
  };
}

interface TrendFilter {
  timeRange: number;
  industry?: Industry;
  strength?: string;
  trendType?: string;
  minConfidence: number;
  sortBy: 'growth_rate' | 'confidence' | 'opportunity_score' | 'created_at';
  sortDirection: 'asc' | 'desc';
  searchQuery?: string;
}

export function TrendAnalysis() {
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTrend, setSelectedTrend] = useState<TrendData | null>(null);
  const [bookmarkedTrends, setBookmarkedTrends] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'chart'>('grid');

  const [filter, setFilter] = useState<TrendFilter>({
    timeRange: 30,
    minConfidence: 0.5,
    sortBy: 'opportunity_score',
    sortDirection: 'desc'
  });

  useEffect(() => {
    loadTrends();
  }, [filter]);

  const loadTrends = async () => {
    try {
      setLoading(true);
      setError(null);

      const trendsData = await CompetitorIntelligenceAPI.getTrends({
        industry: filter.industry,
        timeRangeDays: filter.timeRange,
        trendType: filter.trendType
      });

      // Generate enhanced mock trend data
      const enhancedTrends: TrendData[] = trendsData.map((trend, index) => ({
        ...trend,
        growth_rate: Math.random() * 100 - 20, // -20% to +80%
        opportunity_score: Math.random() * 100,
        threat_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as 'low' | 'medium' | 'high',
        sentiment: Math.random() * 0.8 + 0.1, // 0.1 to 0.9
        keywords: ['AI', 'automation', 'fintech', 'blockchain', 'security', 'innovation', 'digital', 'mobile'].slice(0, 3 + Math.floor(Math.random() * 3)),
        data_points: Array.from({ length: filter.timeRange }, (_, i) => ({
          date: new Date(Date.now() - (filter.timeRange - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          value: Math.max(0, 50 + Math.sin(i * 0.3) * 30 + Math.random() * 20),
          mentions: Math.floor(Math.random() * 100) + 10
        })),
        metadata: {
          category: ['Technology', 'Finance', 'Marketing', 'Innovation'][Math.floor(Math.random() * 4)],
          industry: filter.industry || 'fintech',
          geographical_focus: ['Global', 'North America', 'Europe', 'Asia'][Math.floor(Math.random() * 4)],
          audience_size: Math.floor(Math.random() * 1000000) + 10000
        }
      }));

      // Apply filters
      let filteredTrends = enhancedTrends.filter(trend => {
        if (trend.confidence < filter.minConfidence) return false;
        if (filter.strength && trend.strength !== filter.strength) return false;
        if (filter.searchQuery) {
          const searchLower = filter.searchQuery.toLowerCase();
          return (
            trend.title.toLowerCase().includes(searchLower) ||
            trend.description.toLowerCase().includes(searchLower) ||
            trend.keywords.some(keyword => keyword.toLowerCase().includes(searchLower))
          );
        }
        return true;
      });

      // Apply sorting
      filteredTrends.sort((a, b) => {
        const aVal = a[filter.sortBy];
        const bVal = b[filter.sortBy];
        const multiplier = filter.sortDirection === 'desc' ? -1 : 1;
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return (aVal - bVal) * multiplier;
        }
        return String(aVal).localeCompare(String(bVal)) * multiplier;
      });

      setTrends(filteredTrends);
    } catch (err: any) {
      setError(err.message || 'Failed to load trends');
    } finally {
      setLoading(false);
    }
  };

  const getStrengthColor = (strength: string) => {
    switch (strength) {
      case 'viral':
        return 'text-red-600 bg-red-100';
      case 'strong':
        return 'text-orange-600 bg-orange-100';
      case 'moderate':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getThreatColor = (threat: string) => {
    switch (threat) {
      case 'high':
        return 'text-red-600';
      case 'medium':
        return 'text-yellow-600';
      default:
        return 'text-green-600';
    }
  };

  const getGrowthIcon = (growthRate: number) => {
    return growthRate > 0 ? 
      <ArrowTrendingUpIcon className="h-4 w-4 text-green-500" /> : 
      <ArrowTrendingDownIcon className="h-4 w-4 text-red-500" />;
  };

  const toggleBookmark = (trendId: string) => {
    setBookmarkedTrends(prev => 
      prev.includes(trendId) 
        ? prev.filter(id => id !== trendId)
        : [...prev, trendId]
    );
  };

  const MiniChart = ({ dataPoints }: { dataPoints: TrendData['data_points'] }) => (
    <svg width="100" height="30" className="inline-block">
      <polyline
        points={dataPoints.map((point, index) => 
          `${(index / (dataPoints.length - 1)) * 95 + 2.5},${30 - (point.value / 100) * 25}`
        ).join(' ')}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        className="text-blue-500"
      />
    </svg>
  );

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <FireIconSolid className="h-6 w-6 text-orange-500" />
            <h2 className="text-xl font-semibold text-gray-900">Trend Analysis</h2>
            <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
              {trends.length} trends
            </span>
          </div>

          <div className="flex items-center space-x-3">
            <div className="flex border border-gray-300 rounded-md">
              {(['grid', 'list', 'chart'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-2 text-sm capitalize ${
                    viewMode === mode 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  } ${mode === 'grid' ? 'rounded-l-md' : mode === 'chart' ? 'rounded-r-md' : ''}`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Advanced Filters */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4 mb-6">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Time Range</label>
            <select
              value={filter.timeRange}
              onChange={(e) => setFilter({ ...filter, timeRange: Number(e.target.value) })}
              className="w-full border border-gray-300 rounded px-2 py-1 text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value={7}>7 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
              <option value={180}>6 months</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Industry</label>
            <select
              value={filter.industry || ''}
              onChange={(e) => setFilter({ ...filter, industry: (e.target.value as Industry) || undefined })}
              className="w-full border border-gray-300 rounded px-2 py-1 text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Industries</option>
              <option value="fintech">FinTech</option>
              <option value="saas">SaaS</option>
              <option value="ecommerce">E-commerce</option>
              <option value="healthcare">Healthcare</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Strength</label>
            <select
              value={filter.strength || ''}
              onChange={(e) => setFilter({ ...filter, strength: e.target.value || undefined })}
              className="w-full border border-gray-300 rounded px-2 py-1 text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Strengths</option>
              <option value="viral">Viral</option>
              <option value="strong">Strong</option>
              <option value="moderate">Moderate</option>
              <option value="weak">Weak</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Sort By</label>
            <select
              value={filter.sortBy}
              onChange={(e) => setFilter({ ...filter, sortBy: e.target.value as any })}
              className="w-full border border-gray-300 rounded px-2 py-1 text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value="opportunity_score">Opportunity</option>
              <option value="growth_rate">Growth Rate</option>
              <option value="confidence">Confidence</option>
              <option value="created_at">Date</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Min Confidence</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filter.minConfidence}
              onChange={(e) => setFilter({ ...filter, minConfidence: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="text-xs text-gray-500 text-center">{Math.round(filter.minConfidence * 100)}%</div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Search</label>
            <div className="relative">
              <input
                type="text"
                placeholder="Search trends..."
                value={filter.searchQuery || ''}
                onChange={(e) => setFilter({ ...filter, searchQuery: e.target.value || undefined })}
                className="w-full border border-gray-300 rounded px-2 py-1 pl-6 text-sm focus:ring-2 focus:ring-blue-500"
              />
              <MagnifyingGlassIcon className="h-3 w-3 text-gray-400 absolute left-2 top-2" />
            </div>
          </div>
        </div>
      </div>

      {/* Content Area */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Analyzing trends...</span>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600">{error}</p>
        </div>
      ) : trends.length === 0 ? (
        <div className="text-center py-12">
          <FireIcon className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">No trends match your current filters</p>
        </div>
      ) : (
        <>
          {viewMode === 'grid' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {trends.map((trend) => (
                <div key={trend.id} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 mb-1">{trend.title}</h3>
                      <p className="text-sm text-gray-600 mb-2 line-clamp-2">{trend.description}</p>
                    </div>
                    <button
                      onClick={() => toggleBookmark(trend.id)}
                      className="ml-2 p-1 hover:bg-gray-100 rounded"
                    >
                      {bookmarkedTrends.includes(trend.id) ? (
                        <StarIconSolid className="h-4 w-4 text-yellow-500" />
                      ) : (
                        <StarIcon className="h-4 w-4 text-gray-400" />
                      )}
                    </button>
                  </div>

                  <div className="flex items-center justify-between mb-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStrengthColor(trend.strength)}`}>
                      {trend.strength}
                    </span>
                    <div className="flex items-center space-x-1">
                      {getGrowthIcon(trend.growth_rate)}
                      <span className={`text-sm font-medium ${trend.growth_rate > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {trend.growth_rate > 0 ? '+' : ''}{trend.growth_rate.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="mb-3">
                    <MiniChart dataPoints={trend.data_points} />
                  </div>

                  <div className="space-y-2 text-xs text-gray-500">
                    <div className="flex justify-between">
                      <span>Opportunity Score:</span>
                      <span className="font-medium">{trend.opportunity_score.toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Confidence:</span>
                      <span className="font-medium">{Math.round(trend.confidence * 100)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Threat Level:</span>
                      <span className={`font-medium capitalize ${getThreatColor(trend.threat_level)}`}>
                        {trend.threat_level}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Competitors:</span>
                      <span className="font-medium">{trend.competitors_involved.length}</span>
                    </div>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-1">
                    {trend.keywords.slice(0, 3).map((keyword) => (
                      <span key={keyword} className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-blue-100 text-blue-800">
                        <TagIcon className="h-3 w-3 mr-1" />
                        {keyword}
                      </span>
                    ))}
                  </div>

                  <div className="mt-4 flex justify-between items-center">
                    <button
                      onClick={() => setSelectedTrend(trend)}
                      className="text-sm text-blue-600 hover:text-blue-800 font-medium flex items-center"
                    >
                      <EyeIcon className="h-4 w-4 mr-1" />
                      View Details
                    </button>
                    <div className="flex items-center space-x-2">
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <ShareIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {viewMode === 'list' && (
            <div className="bg-white rounded-lg shadow overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Strength</th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Growth</th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Opportunity</th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Chart</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {trends.map((trend, index) => (
                      <tr key={trend.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-6 py-4">
                          <div className="flex items-center">
                            <div className="flex-1">
                              <div className="text-sm font-medium text-gray-900">{trend.title}</div>
                              <div className="text-sm text-gray-500">{trend.metadata.category}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-center">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStrengthColor(trend.strength)}`}>
                            {trend.strength}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            {getGrowthIcon(trend.growth_rate)}
                            <span className={`text-sm font-medium ${trend.growth_rate > 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {trend.growth_rate > 0 ? '+' : ''}{trend.growth_rate.toFixed(1)}%
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-center">
                          <div className="text-sm text-gray-900">{trend.opportunity_score.toFixed(0)}/100</div>
                          <div className="w-16 bg-gray-200 rounded-full h-1.5 mx-auto mt-1">
                            <div
                              className="bg-blue-600 h-1.5 rounded-full"
                              style={{ width: `${trend.opportunity_score}%` }}
                            ></div>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-center text-sm text-gray-900">
                          {Math.round(trend.confidence * 100)}%
                        </td>
                        <td className="px-6 py-4 text-center">
                          <MiniChart dataPoints={trend.data_points} />
                        </td>
                        <td className="px-6 py-4 text-right text-sm font-medium">
                          <div className="flex items-center justify-end space-x-2">
                            <button
                              onClick={() => toggleBookmark(trend.id)}
                              className="text-gray-400 hover:text-yellow-500"
                            >
                              {bookmarkedTrends.includes(trend.id) ? (
                                <StarIconSolid className="h-4 w-4 text-yellow-500" />
                              ) : (
                                <StarIcon className="h-4 w-4" />
                              )}
                            </button>
                            <button
                              onClick={() => setSelectedTrend(trend)}
                              className="text-blue-600 hover:text-blue-800"
                            >
                              <EyeIcon className="h-4 w-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {viewMode === 'chart' && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-900 mb-2">Trend Opportunity vs Growth Rate</h3>
                <p className="text-sm text-gray-600">Bubble size represents confidence level</p>
              </div>
              
              <div className="relative w-full h-96 border border-gray-200 rounded-lg p-4">
                <svg width="100%" height="100%" className="overflow-visible">
                  {/* Grid lines */}
                  <defs>
                    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
                    </pattern>
                  </defs>
                  <rect width="100%" height="100%" fill="url(#grid)" />
                  
                  {/* Axes */}
                  <line x1="50" y1="340" x2="650" y2="340" stroke="#6b7280" strokeWidth="2" />
                  <line x1="50" y1="340" x2="50" y2="20" stroke="#6b7280" strokeWidth="2" />
                  
                  {/* Data points */}
                  {trends.map((trend, index) => {
                    const x = 50 + (trend.opportunity_score / 100) * 600;
                    const y = 340 - ((trend.growth_rate + 20) / 100) * 320;
                    const radius = 4 + (trend.confidence * 15);
                    
                    return (
                      <g key={trend.id}>
                        <circle
                          cx={x}
                          cy={y}
                          r={radius}
                          fill={trend.strength === 'viral' ? '#ef4444' : 
                                trend.strength === 'strong' ? '#f97316' : 
                                trend.strength === 'moderate' ? '#eab308' : '#6b7280'}
                          fillOpacity="0.7"
                          stroke="#ffffff"
                          strokeWidth="2"
                          className="cursor-pointer hover:fill-opacity-100"
                          onClick={() => setSelectedTrend(trend)}
                        />
                        <text
                          x={x}
                          y={y - radius - 5}
                          textAnchor="middle"
                          className="text-xs fill-gray-700 pointer-events-none"
                        >
                          {trend.title.split(' ').slice(0, 2).join(' ')}
                        </text>
                      </g>
                    );
                  })}
                  
                  {/* Labels */}
                  <text x="350" y="370" textAnchor="middle" className="text-sm fill-gray-600">
                    Opportunity Score
                  </text>
                  <text x="25" y="180" textAnchor="middle" transform="rotate(-90, 25, 180)" className="text-sm fill-gray-600">
                    Growth Rate (%)
                  </text>
                </svg>
              </div>
              
              <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Viral</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                  <span>Strong</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>Moderate</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                  <span>Weak</span>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Trend Detail Modal */}
      {selectedTrend && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-900">{selectedTrend.title}</h2>
              <button
                onClick={() => setSelectedTrend(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium text-gray-900 mb-2">Description</h3>
                  <p className="text-gray-600 mb-4">{selectedTrend.description}</p>
                  
                  <h3 className="font-medium text-gray-900 mb-2">Key Metrics</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Strength:</span>
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStrengthColor(selectedTrend.strength)}`}>
                        {selectedTrend.strength}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Growth Rate:</span>
                      <span className={`font-medium ${selectedTrend.growth_rate > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {selectedTrend.growth_rate > 0 ? '+' : ''}{selectedTrend.growth_rate.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Opportunity Score:</span>
                      <span className="font-medium">{selectedTrend.opportunity_score.toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Confidence:</span>
                      <span className="font-medium">{Math.round(selectedTrend.confidence * 100)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Threat Level:</span>
                      <span className={`font-medium capitalize ${getThreatColor(selectedTrend.threat_level)}`}>
                        {selectedTrend.threat_level}
                      </span>
                    </div>
                  </div>

                  <h3 className="font-medium text-gray-900 mt-4 mb-2">Keywords</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedTrend.keywords.map((keyword) => (
                      <span key={keyword} className="inline-flex items-center px-2 py-1 rounded-md text-xs bg-blue-100 text-blue-800">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-gray-900 mb-2">Trend Over Time</h3>
                  <div className="bg-gray-50 rounded-lg p-4 mb-4">
                    <svg width="100%" height="200" viewBox="0 0 400 200">
                      <polyline
                        points={selectedTrend.data_points.map((point, index) => 
                          `${(index / (selectedTrend.data_points.length - 1)) * 380 + 10},${190 - (point.value / 100) * 170}`
                        ).join(' ')}
                        fill="none"
                        stroke="#3b82f6"
                        strokeWidth="2"
                      />
                      <line x1="10" y1="190" x2="390" y2="190" stroke="#e5e7eb" strokeWidth="1" />
                    </svg>
                  </div>
                  
                  <h3 className="font-medium text-gray-900 mb-2">Metadata</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Category:</span>
                      <span>{selectedTrend.metadata.category}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Industry:</span>
                      <span className="capitalize">{selectedTrend.metadata.industry}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Geographic Focus:</span>
                      <span>{selectedTrend.metadata.geographical_focus}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Audience Size:</span>
                      <span>{selectedTrend.metadata.audience_size?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Competitors Involved:</span>
                      <span>{selectedTrend.competitors_involved.length}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TrendAnalysis;