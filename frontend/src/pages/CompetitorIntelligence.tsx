/**
 * Main Competitor Intelligence page - Dashboard overview
 */

import React, { useState, useEffect, useRef } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  ChartBarIcon,
  EyeIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  UserGroupIcon,
  ArrowTrendingUpIcon,
  PlusIcon,
  PlayIcon,
  SparklesIcon,
  DocumentChartBarIcon,
  LinkIcon,
  SignalIcon,
  ClockIcon,
  FireIcon,
  BellIcon,
} from '@heroicons/react/24/outline';
import {
  SignalIcon as SignalIconSolid,
  BellIcon as BellIconSolid,
} from '@heroicons/react/24/solid';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import { Industry, type DashboardData, type SystemStatus, type ChangeEvent } from '../types/competitor-intelligence';

export function CompetitorIntelligence() {
  const [searchParams] = useSearchParams();
  const paramIndustry = (searchParams.get('industry') || '').toLowerCase();
  const envIndustry = (import.meta.env.VITE_CI_DEFAULT_INDUSTRY || '').toLowerCase();
  const initialIndustry: Industry = (Object.values(Industry) as string[]).includes(paramIndustry)
    ? (paramIndustry as Industry)
    : (Object.values(Industry) as string[]).includes(envIndustry)
      ? (envIndustry as Industry)
      : Industry.FINTECH;

  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRealTimeConnected, setIsRealTimeConnected] = useState(false);
  const [liveUpdates, setLiveUpdates] = useState<Array<{id: string; message: string; timestamp: Date; type: 'info' | 'warning' | 'success' | 'alert'}>>([]);
  const [alertCount, setAlertCount] = useState(0);
  const [trendingTopics, setTrendingTopics] = useState<Array<{topic: string; growth: number; urgency: 'high' | 'medium' | 'low'}>>([]);
  const [recentChanges, setRecentChanges] = useState<ChangeEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadDashboardData();
    loadRecentChanges();
    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);
  
  const connectWebSocket = () => {
    const wsUrl = import.meta.env.VITE_CI_WEBSOCKET_URL as string | undefined;
    if (!wsUrl) {
      setIsRealTimeConnected(false);
      return;
    }
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsRealTimeConnected(true);
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          const message: string = data?.message || 'New update';
          const type = (data?.type || 'info') as 'info' | 'warning' | 'success' | 'alert';
          const newUpdate = {
            id: Date.now().toString(),
            message,
            timestamp: new Date(),
            type
          };
          setLiveUpdates(prev => [newUpdate, ...prev.slice(0, 9)]);

          if (typeof data?.unread_alerts === 'number') {
            setAlertCount(data.unread_alerts);
          }
        } catch {
          const newUpdate = {
            id: Date.now().toString(),
            message: String(event.data),
            timestamp: new Date(),
            type: 'info' as const
          };
          setLiveUpdates(prev => [newUpdate, ...prev.slice(0, 9)]);
        }
      };

      ws.onclose = () => {
        setIsRealTimeConnected(false);
        if (!reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connectWebSocket();
          }, 5000);
        }
      };

      ws.onerror = () => {
        try { ws.close(); } catch {}
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setIsRealTimeConnected(false);
    }
  };

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load dashboard data and system status in parallel
      const [dashboard, status] = await Promise.all([
        CompetitorIntelligenceAPI.getDashboardData({
          industry: initialIndustry,
          timeRangeDays: 30
        }),
        CompetitorIntelligenceAPI.getSystemStatus()
      ]);

      setDashboardData(dashboard);
      setSystemStatus(status);
      // Load side data (alerts summary and trends)
      try {
        const [alertSummary, trends] = await Promise.all([
          CompetitorIntelligenceAPI.getAlertSummary(),
          CompetitorIntelligenceAPI.getTrends({ industry: initialIndustry, timeRangeDays: 30 })
        ]);

        if (alertSummary && typeof alertSummary.unread_alerts === 'number') {
          setAlertCount(alertSummary.unread_alerts);
        }
        if (Array.isArray(trends)) {
          const topics = trends.slice(0, 4).map((t: any) => {
            const strength = (t.strength || '').toString().toLowerCase();
            const urgency: 'high' | 'medium' | 'low' = strength === 'viral'
              ? 'high'
              : strength === 'strong'
                ? 'medium'
                : 'low';
            const growth = typeof t.growthRate === 'number'
              ? Math.max(0, Math.round(t.growthRate))
              : strength === 'viral'
                ? 80
                : strength === 'strong'
                  ? 60
                  : strength === 'moderate'
                    ? 35
                    : 15;
            return {
              topic: t.title || t.topic || 'Trend',
              growth,
              urgency
            };
          });
          setTrendingTopics(topics);
        }
      } catch (sideErr) {
        // Non-blocking
        console.warn('Optional side data load failed:', sideErr);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data');
      console.error('Dashboard load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadRecentChanges = async () => {
    try {
      const changes = await CompetitorIntelligenceAPI.getRecentChanges({ limit: 10 });
      setRecentChanges(changes);
    } catch (err) {
      // optional: ignore errors here
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading competitor intelligence...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto" />
          <h2 className="mt-4 text-xl font-semibold text-gray-900">Unable to load dashboard</h2>
          <p className="mt-2 text-gray-600">{error}</p>
          <button
            onClick={loadDashboardData}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Try Again
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
              <h1 className="text-3xl font-bold text-gray-900">Competitor Intelligence</h1>
              <p className="mt-2 text-gray-600">Monitor competitors and identify market opportunities</p>
            </div>
            <div className="flex space-x-3">
              <Link
                to="/competitor-intelligence/analytics"
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <ChartBarIcon className="h-4 w-4 mr-2" />
                Analytics Dashboard
              </Link>
              <Link
                to="/competitor-intelligence/competitors/new"
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
              >
                <PlusIcon className="h-4 w-4 mr-2" />
                Add Competitor
              </Link>
              <Link
                to="/competitor-intelligence/analysis"
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <PlayIcon className="h-4 w-4 mr-2" />
                Run Analysis
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Real-time Status Bar */}
        <div className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {isRealTimeConnected ? (
                  <SignalIconSolid className="h-5 w-5 text-green-500" />
                ) : (
                  <SignalIcon className="h-5 w-5 text-gray-400" />
                )}
                <span className={`text-sm font-medium ${
                  isRealTimeConnected ? 'text-green-700' : 'text-gray-600'
                }`}>
                  {isRealTimeConnected ? 'Real-time monitoring active' : 'Real-time monitoring offline'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <ClockIcon className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">
                  Last update: {new Date().toLocaleTimeString()}
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {alertCount > 0 && (
                <div className="relative">
                  <BellIconSolid className="h-5 w-5 text-amber-500" />
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full px-1 min-w-[16px] text-center">
                    {alertCount}
                  </span>
                </div>
              )}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => { loadDashboardData(); loadRecentChanges(); }}
                  className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                >
                  Refresh Data
                </button>
                <button
                  onClick={async () => { try { await CompetitorIntelligenceAPI.runMonitoringForAll(); await loadRecentChanges(); } catch {} }}
                  className="text-sm text-white bg-blue-600 hover:bg-blue-700 font-medium px-3 py-1.5 rounded"
                >
                  Monitor now
                </button>
                <button
                  onClick={async () => { try { await CompetitorIntelligenceAPI.runPricingDetectionAll(); await CompetitorIntelligenceAPI.runCopyDetectionAll(); await loadRecentChanges(); } catch {} }}
                  className="text-sm text-white bg-purple-600 hover:bg-purple-700 font-medium px-3 py-1.5 rounded"
                >
                  Run all detectors
                </button>
              </div>
            </div>
          </div>
        </div>
        
        {/* Live Updates Stream */}
        {liveUpdates.length > 0 && (
          <div className="mb-6 bg-white rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <FireIcon className="h-5 w-5 text-orange-500 mr-2" />
                  Live Activity Stream
                </h2>
                <span className="text-sm text-gray-500">
                  {liveUpdates.length} recent updates
                </span>
              </div>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {liveUpdates.map((update) => (
                <div key={update.id} className="px-6 py-3 border-b border-gray-100 last:border-b-0">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className={`inline-block w-2 h-2 rounded-full ${
                          update.type === 'success' ? 'bg-green-500' :
                          update.type === 'warning' ? 'bg-yellow-500' :
                          update.type === 'alert' ? 'bg-red-500' : 'bg-blue-500'
                        }`} />
                        <p className="text-sm text-gray-900">{update.message}</p>
                      </div>
                    </div>
                    <span className="text-xs text-gray-500 ml-2">
                      {update.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Trending Topics Widget */}
        {trendingTopics.length > 0 && (
          <div className="mb-8 bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <ArrowTrendingUpIcon className="h-5 w-5 text-green-500 mr-2" />
                Trending Now
              </h2>
              <Link 
                to="/competitor-intelligence/trends"
                className="text-sm text-blue-600 hover:text-blue-800 font-medium"
              >
                View All Trends →
              </Link>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {trendingTopics.map((trend, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${
                      trend.urgency === 'high' ? 'bg-red-500' :
                      trend.urgency === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                    }`} />
                    <span className="font-medium text-gray-900">{trend.topic}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <ArrowTrendingUpIcon className="h-4 w-4 text-green-500" />
                    <span className="text-sm font-semibold text-green-600">+{trend.growth}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* System Status */}
        {systemStatus && (
          <div className="mb-8 bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">System Status</h2>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                systemStatus.systemHealth === 'healthy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-yellow-100 text-yellow-800'
              }`}>
                {systemStatus.systemHealth}
              </span>
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Last Full Analysis:</span>
                <p className="font-medium">
                  {systemStatus.lastFullAnalysis 
                    ? new Date(systemStatus.lastFullAnalysis).toLocaleDateString()
                    : 'Not run yet'}
                </p>
              </div>
              <div>
                <span className="text-gray-500">Cache Status:</span>
                <p className="font-medium capitalize">{systemStatus.cacheStatus}</p>
              </div>
              <div>
                <span className="text-gray-500">Orchestrator:</span>
                <p className="font-medium capitalize">{systemStatus.orchestratorStatus}</p>
              </div>
              <div>
                <span className="text-gray-500">Real-time Status:</span>
                <p className={`font-medium ${
                  isRealTimeConnected ? 'text-green-600' : 'text-gray-500'
                }`}>
                  {isRealTimeConnected ? 'Connected' : 'Disconnected'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Key Metrics */}
        {dashboardData && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Competitors Monitored"
                value={dashboardData.overview.competitorsMonitored}
                icon={UserGroupIcon}
                color="blue"
                trend={isRealTimeConnected ? '+2 today' : undefined}
              />
              <MetricCard
                title="Content Analyzed"
                value={dashboardData.keyMetrics.totalContentAnalyzed}
                icon={ChartBarIcon}
                color="green"
                trend={isRealTimeConnected ? '+47 this week' : undefined}
              />
              <MetricCard
                title="Trends Identified"
                value={dashboardData.keyMetrics.trendsIdentified}
                icon={ArrowTrendingUpIcon}
                color="purple"
                trend={trendingTopics.length > 0 ? `${trendingTopics.length} trending` : undefined}
              />
              <MetricCard
                title="Active Alerts"
                value={alertCount}
                icon={BellIcon}
                color="red"
                trend={alertCount > 0 ? 'Needs attention' : 'All clear'}
              />
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <QuickActionCard
                title="Analytics"
                description="View comprehensive overview"
                link="/competitor-intelligence/analytics"
                icon={ChartBarIcon}
              />
              <QuickActionCard
                title="Competitors"
                description="Manage competitor list"
                link="/competitor-intelligence/competitors"
                icon={UserGroupIcon}
              />
              <QuickActionCard
                title="AI Analysis"
                description="AI-powered content analysis"
                link="/competitor-intelligence/ai-analysis"
                icon={SparklesIcon}
              />
              <QuickActionCard
                title="Reporting"
                description="Generate comprehensive reports"
                link="/competitor-intelligence/reporting"
                icon={DocumentChartBarIcon}
              />
            </div>

            {/* Phase 4 Features */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Advanced Features</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <QuickActionCard
                  title="Trends"
                  description="Explore market trends"
                  link="/competitor-intelligence/trends"
                  icon={ArrowTrendingUpIcon}
                />
                <QuickActionCard
                  title="Integrations"
                  description="Connect external tools"
                  link="/competitor-intelligence/integrations"
                  icon={LinkIcon}
                />
                <QuickActionCard
                  title="Opportunities"
                  description="Find content gaps"
                  link="/competitor-intelligence/opportunities"
                  icon={LightBulbIcon}
                />
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">Recent Activity</h2>
              </div>
              <div className="p-6">
                {recentChanges.length === 0 ? (
                  <div className="text-center py-8">
                    <EyeIcon className="h-12 w-12 text-gray-400 mx-auto" />
                    <h3 className="mt-4 text-lg font-medium text-gray-900">No recent activity</h3>
                    <p className="mt-2 text-gray-600">Run monitoring to see events here.</p>
                  </div>
                ) : (
                  <ul className="divide-y divide-gray-200">
                    {recentChanges.map((evt) => (
                      <li key={(evt.id ?? `${evt.competitor_id}-${evt.detected_at}`) as any} className="py-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-900">
                              <span className="font-semibold">{evt.competitor_name}</span> • {evt.change_type}
                            </p>
                            <p className="text-sm text-gray-600 truncate max-w-xl">
                              {evt.url || ''}
                            </p>
                          </div>
                          <span className="text-xs text-gray-500">{new Date(evt.detected_at).toLocaleString()}</span>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// Helper Components
interface MetricCardProps {
  title: string;
  value: number;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  color: 'blue' | 'green' | 'purple' | 'yellow' | 'red';
  trend?: string;
}

function MetricCard({ title, value, icon: Icon, color, trend }: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center">
        <div className={`p-3 rounded-md ${colorClasses[color]}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value.toLocaleString()}</p>
          {trend && (
            <p className="text-xs text-gray-500 mt-1 flex items-center">
              <ArrowTrendingUpIcon className="h-3 w-3 mr-1" />
              {trend}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

interface QuickActionCardProps {
  title: string;
  description: string;
  link: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
}

function QuickActionCard({ title, description, link, icon: Icon }: QuickActionCardProps) {
  return (
    <Link
      to={link}
      className="bg-white rounded-lg shadow p-6 hover:shadow-md transition-shadow duration-200"
    >
      <div className="flex items-center">
        <Icon className="h-8 w-8 text-blue-600" />
        <div className="ml-4">
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          <p className="text-sm text-gray-600">{description}</p>
        </div>
      </div>
    </Link>
  );
}

export default CompetitorIntelligence;