/**
 * Comprehensive Analytics Dashboard for Competitor Intelligence
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  EyeIcon,
  BellIcon,
  GlobeAltIcon,
  CalendarIcon,
  UsersIcon,
  ChatBubbleLeftRightIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  FireIcon,
  LightBulbIcon,
  ArrowDownTrayIcon,
  FunnelIcon,
  ChartPieIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';

interface DashboardData {
  overview: {
    competitorsMonitored: number;
    totalContentAnalyzed: number;
    trendsIdentified: number;
    alertsGenerated: number;
    socialPostsTracked: number;
    lastUpdated: string;
  };
  trends: Array<{
    id: string;
    title: string;
    trend_type: string;
    strength: string;
    confidence: number;
    competitors_involved: string[];
  }>;
  insights: Array<{
    id: string;
    title: string;
    insight_type: string;
    impact_level: string;
    confidence: number;
    recommendations: string[];
  }>;
  topCompetitors: Array<{
    id: string;
    name: string;
    contentCount: number;
    socialPosts: number;
    activityScore: number;
    trend: 'up' | 'down' | 'stable';
  }>;
  contentAnalytics: {
    totalPosts: number;
    avgQuality: number;
    topKeywords: Array<{ keyword: string; count: number }>;
    contentTypes: Array<{ type: string; count: number }>;
  };
  socialAnalytics: {
    totalPosts: number;
    avgVirality: number;
    topPlatforms: Array<{ platform: string; count: number }>;
    avgSentiment: number;
  };
}

export function AnalyticsDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState(30);
  const [selectedIndustry, setSelectedIndustry] = useState<string>('');
  const [customDateRange, setCustomDateRange] = useState<{start: string; end: string}>({start: '', end: ''});
  const [useCustomRange, setUseCustomRange] = useState(false);
  const [exportFormat, setExportFormat] = useState<'csv' | 'pdf' | 'json'>('csv');
  const [isExporting, setIsExporting] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['trends', 'insights', 'competitors', 'content']);

  useEffect(() => {
    loadDashboardData();
  }, [timeRange, selectedIndustry]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Calculate actual date range
      const actualTimeRange = useCustomRange && customDateRange.start && customDateRange.end 
        ? Math.ceil((new Date(customDateRange.end).getTime() - new Date(customDateRange.start).getTime()) / (1000 * 60 * 60 * 24))
        : timeRange;

      // Load all dashboard data in parallel
      const [trends, insights, competitors, alertSummary] = await Promise.all([
        CompetitorIntelligenceAPI.getTrends({
          industry: selectedIndustry || undefined,
          timeRangeDays: actualTimeRange
        }),
        CompetitorIntelligenceAPI.getInsights({
          daysBack: actualTimeRange
        }),
        CompetitorIntelligenceAPI.listCompetitors({
          industry: selectedIndustry || undefined
        }),
        CompetitorIntelligenceAPI.getAlertSummary()
      ]);

      // Derive deterministic metrics from real data (no randomness)
      const totalDataPoints = trends.reduce((sum: number, t: any) => {
        const pts = Array.isArray(t?.data_points) ? t.data_points.length : 1;
        return sum + pts;
      }, 0);

      const socialTrendCount = trends.filter((t: any) => {
        const type = (t?.trend_type || t?.type || '').toString().toLowerCase();
        return type.includes('social');
      }).length;

      // Aggregate keywords from trends
      const keywordCounts: Record<string, number> = {};
      trends.forEach((t: any) => {
        const kws: string[] = Array.isArray(t?.keywords) ? t.keywords : [];
        kws.forEach((k) => {
          const key = (k || '').toString().toLowerCase();
          if (!key) return;
          keywordCounts[key] = (keywordCounts[key] || 0) + 1;
        });
      });
      const topKeywords = Object.entries(keywordCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([keyword, count]) => ({ keyword, count }));

      // Content types proxy using trend type buckets
      const contentTypeBuckets: Record<string, number> = {};
      trends.forEach((t: any) => {
        const bucket = ((t?.trend_type || 'content') as string).toString();
        contentTypeBuckets[bucket] = (contentTypeBuckets[bucket] || 0) + 1;
      });
      const contentTypes = Object.entries(contentTypeBuckets).map(([type, count]) => ({ type, count }));

      // Build top competitors (deterministic, using available fields)
      const topCompetitors = competitors.slice(0, 6).map((comp: any) => ({
        id: comp.id,
        name: comp.name,
        contentCount: comp.contentCount ?? 0,
        socialPosts: 0,
        activityScore: comp.trendingScore ?? 0,
        trend: 'stable' as 'up' | 'down' | 'stable'
      }));

      const realDashboardData: DashboardData = {
        overview: {
          competitorsMonitored: competitors.length,
          totalContentAnalyzed: totalDataPoints,
          trendsIdentified: trends.length,
          alertsGenerated: alertSummary?.total_alerts ?? 0,
          socialPostsTracked: socialTrendCount,
          lastUpdated: new Date().toISOString()
        },
        trends: trends.slice(0, 10),
        insights: insights.slice(0, 8),
        topCompetitors,
        contentAnalytics: {
          totalPosts: totalDataPoints,
          avgQuality: 0,
          topKeywords,
          contentTypes
        },
        socialAnalytics: {
          totalPosts: socialTrendCount,
          avgVirality: 0,
          topPlatforms: [],
          avgSentiment: 0.5
        }
      };

      setDashboardData(realDashboardData);
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };
  
  const handleExportData = async () => {
    try {
      setIsExporting(true);
      
      // Prepare export data based on selected metrics
      const exportData: any = {};
      
      if (selectedMetrics.includes('trends')) {
        exportData.trends = dashboardData?.trends || [];
      }
      if (selectedMetrics.includes('insights')) {
        exportData.insights = dashboardData?.insights || [];
      }
      if (selectedMetrics.includes('competitors')) {
        exportData.competitors = dashboardData?.topCompetitors || [];
      }
      if (selectedMetrics.includes('content')) {
        exportData.contentAnalytics = dashboardData?.contentAnalytics || {};
        exportData.socialAnalytics = dashboardData?.socialAnalytics || {};
      }
      
      // Add metadata
      exportData.metadata = {
        exportedAt: new Date().toISOString(),
        timeRange: useCustomRange ? customDateRange : `${timeRange} days`,
        industry: selectedIndustry || 'All Industries',
        format: exportFormat
      };
      
      if (exportFormat === 'json') {
        // Download as JSON
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `competitor-analytics-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else if (exportFormat === 'csv') {
        // Convert to CSV format
        let csvContent = 'Type,Title,Value,Details,Timestamp\n';
        
        if (exportData.trends) {
          exportData.trends.forEach((trend: any) => {
            csvContent += `Trend,"${trend.title}",${trend.confidence},"${trend.trend_type}",${trend.created_at || new Date().toISOString()}\n`;
          });
        }
        
        if (exportData.insights) {
          exportData.insights.forEach((insight: any) => {
            csvContent += `Insight,"${insight.title}",${insight.confidence},"${insight.impact_level}",${insight.created_at || new Date().toISOString()}\n`;
          });
        }
        
        if (exportData.competitors) {
          exportData.competitors.forEach((comp: any) => {
            csvContent += `Competitor,"${comp.name}",${comp.activityScore},"${comp.contentCount} posts",${new Date().toISOString()}\n`;
          });
        }
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `competitor-analytics-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else {
        // For PDF, we'd typically use a service like jsPDF or call backend API
        // For now, show a message that PDF export would be implemented
        alert('PDF export functionality would be implemented with a proper PDF generation service.');
      }
      
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export data. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };
  
  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics(prev => 
      prev.includes(metric) 
        ? prev.filter(m => m !== metric)
        : [...prev, metric]
    );
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <ArrowUpIcon className="h-4 w-4 text-green-500" />;
      case 'down':
        return <ArrowDownIcon className="h-4 w-4 text-red-500" />;
      default:
        return <div className="h-4 w-4 bg-gray-400 rounded-full"></div>;
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

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical':
        return 'text-red-600 bg-red-100';
      case 'high':
        return 'text-orange-600 bg-orange-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-blue-600 bg-blue-100';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (error || !dashboardData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 text-lg">{error || 'Failed to load dashboard'}</p>
          <button
            onClick={loadDashboardData}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <nav className="flex" aria-label="Breadcrumb">
                <ol className="flex items-center space-x-4">
                  <li>
                    <Link to="/competitor-intelligence" className="text-gray-400 hover:text-gray-500">
                      Competitor Intelligence
                    </Link>
                  </li>
                  <li>
                    <span className="text-gray-400">/</span>
                  </li>
                  <li>
                    <span className="text-gray-900 font-medium">Analytics Dashboard</span>
                  </li>
                </ol>
              </nav>
              <h1 className="text-3xl font-bold text-gray-900 mt-2">Analytics Dashboard</h1>
              <p className="mt-2 text-gray-600">
                Comprehensive insights into competitor activity and market trends
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(Number(e.target.value))}
                className="border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value={7}>Last 7 days</option>
                <option value={30}>Last 30 days</option>
                <option value={90}>Last 90 days</option>
              </select>
              
              <select
                value={selectedIndustry}
                onChange={(e) => setSelectedIndustry(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All Industries</option>
                <option value="fintech">FinTech</option>
                <option value="saas">SaaS</option>
                <option value="ecommerce">E-commerce</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <UsersIcon className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Competitors</p>
                <p className="text-2xl font-bold text-gray-900">{dashboardData.overview.competitorsMonitored}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <GlobeAltIcon className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Content Items</p>
                <p className="text-2xl font-bold text-gray-900">{dashboardData.overview.totalContentAnalyzed}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <ChatBubbleLeftRightIcon className="h-8 w-8 text-purple-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Social Posts</p>
                <p className="text-2xl font-bold text-gray-900">{dashboardData.overview.socialPostsTracked}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <ArrowTrendingUpIcon className="h-8 w-8 text-yellow-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Trends</p>
                <p className="text-2xl font-bold text-gray-900">{dashboardData.overview.trendsIdentified}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <BellIcon className="h-8 w-8 text-red-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Alerts</p>
                <p className="text-2xl font-bold text-gray-900">{dashboardData.overview.alertsGenerated}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <CalendarIcon className="h-8 w-8 text-indigo-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Last Update</p>
                <p className="text-sm font-bold text-gray-900">
                  {new Date(dashboardData.overview.lastUpdated).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-8">
            {/* Trending Topics */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900 flex items-center">
                  <FireIcon className="h-5 w-5 mr-2 text-orange-500" />
                  Trending Topics
                </h2>
              </div>
              <div className="p-6">
                {dashboardData.trends.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No trends identified yet</p>
                ) : (
                  <div className="space-y-4">
                    {dashboardData.trends.slice(0, 6).map((trend) => (
                      <div key={trend.id} className="border rounded-lg p-4 hover:bg-gray-50">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h3 className="font-medium text-gray-900">{trend.title}</h3>
                            <p className="text-sm text-gray-600 mt-1">
                              {trend.competitors_involved?.length} competitors involved
                            </p>
                          </div>
                          <div className="ml-4 flex flex-col items-end space-y-1">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStrengthColor(trend.strength)}`}>
                              {trend.strength}
                            </span>
                            <span className="text-xs text-gray-500">
                              {Math.round(trend.confidence * 100)}% confidence
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Market Insights */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900 flex items-center">
                  <LightBulbIcon className="h-5 w-5 mr-2 text-yellow-500" />
                  Strategic Insights
                </h2>
              </div>
              <div className="p-6">
                {dashboardData.insights.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No insights available yet</p>
                ) : (
                  <div className="space-y-4">
                    {dashboardData.insights.slice(0, 4).map((insight) => (
                      <div key={insight.id} className="border rounded-lg p-4 hover:bg-gray-50">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h3 className="font-medium text-gray-900">{insight.title}</h3>
                            <p className="text-sm text-gray-600 mt-1 capitalize">
                              {insight.insight_type}
                            </p>
                            {insight.recommendations && insight.recommendations.length > 0 && (
                              <ul className="mt-2 text-xs text-gray-600 list-disc list-inside">
                                {insight.recommendations.slice(0, 2).map((rec, i) => (
                                  <li key={i}>{rec}</li>
                                ))}
                              </ul>
                            )}
                          </div>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getImpactColor(insight.impact_level)}`}>
                            {insight.impact_level}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Top Competitors */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900">Most Active Competitors</h2>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {dashboardData.topCompetitors.map((competitor) => (
                    <div key={competitor.id} className="flex items-center justify-between">
                      <div className="flex-1">
                        <Link
                          to={`/competitor-intelligence/competitors/${competitor.id}`}
                          className="font-medium text-gray-900 hover:text-blue-600"
                        >
                          {competitor.name}
                        </Link>
                        <p className="text-sm text-gray-500">
                          {competitor.contentCount} posts, {competitor.socialPosts} social
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-gray-900">
                          {competitor.activityScore}
                        </span>
                        {getTrendIcon(competitor.trend)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Content Analytics */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900">Content Analytics</h2>
              </div>
              <div className="p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Posts</span>
                  <span className="font-medium">{dashboardData.contentAnalytics.totalPosts}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg Quality</span>
                  <span className="font-medium">{dashboardData.contentAnalytics.avgQuality.toFixed(1)}/10</span>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Top Keywords</h4>
                  <div className="space-y-1">
                    {dashboardData.contentAnalytics.topKeywords.slice(0, 5).map((keyword) => (
                      <div key={keyword.keyword} className="flex justify-between text-sm">
                        <span className="text-gray-600">{keyword.keyword}</span>
                        <span className="font-medium">{keyword.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Social Analytics */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900">Social Media Analytics</h2>
              </div>
              <div className="p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Social Posts</span>
                  <span className="font-medium">{dashboardData.socialAnalytics.totalPosts}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg Virality</span>
                  <span className="font-medium">{dashboardData.socialAnalytics.avgVirality.toFixed(1)}/10</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg Sentiment</span>
                  <span className={`font-medium ${dashboardData.socialAnalytics.avgSentiment > 0.5 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {dashboardData.socialAnalytics.avgSentiment > 0.5 ? 'Positive' : 'Neutral'}
                  </span>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Top Platforms</h4>
                  <div className="space-y-1">
                    {dashboardData.socialAnalytics.topPlatforms.slice(0, 4).map((platform) => (
                      <div key={platform.platform} className="flex justify-between text-sm">
                        <span className="text-gray-600 capitalize">{platform.platform}</span>
                        <span className="font-medium">{platform.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalyticsDashboard;