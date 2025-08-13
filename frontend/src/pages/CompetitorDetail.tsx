/**
 * Competitor Detail page - Full view with analytics and content history
 */

import React, { useState, useEffect } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  PencilIcon,
  TrashIcon,
  GlobeAltIcon,
  ChartBarIcon,
  ClockIcon,
  EyeIcon,
  TagIcon,
  CalendarIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import { confirmAction } from '../lib/toast';
import { showErrorNotification } from '../lib/errors';
import type { Industry, CompetitorTier } from '../types/competitor-intelligence';

interface CompetitorDetail {
  id: string;
  name: string;
  domain: string;
  tier: CompetitorTier;
  industry: Industry;
  description: string;
  platforms: string[];
  monitoringKeywords: string[];
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
  lastMonitored?: string;
}

export function CompetitorDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [competitor, setCompetitor] = useState<CompetitorDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [content, setContent] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [monitoring, setMonitoring] = useState(false);

  useEffect(() => {
    if (id) {
      loadCompetitor();
    }
  }, [id]);

  const loadCompetitor = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load competitor details
      const competitorData = await CompetitorIntelligenceAPI.getCompetitor(id!);
      setCompetitor(competitorData);

      // Load competitor content
      try {
        const contentData = await CompetitorIntelligenceAPI.getCompetitorContent(id!, {
          daysBack: 30,
          limit: 20
        });
        setContent(contentData);
      } catch (err) {
        console.error('Failed to load content:', err);
      }

      // Load alerts for this competitor
      try {
        const alertsData = await CompetitorIntelligenceAPI.getAlerts({
          competitorId: id!,
          limit: 10
        });
        setAlerts(alertsData);
      } catch (err) {
        console.error('Failed to load alerts:', err);
      }

    } catch (err: any) {
      setError(err.message || 'Failed to load competitor');
    } finally {
      setLoading(false);
    }
  };

  const handleMonitorNow = async () => {
    try {
      setMonitoring(true);
      await CompetitorIntelligenceAPI.monitorCompetitor(id!);
      // Reload content after monitoring
      await loadCompetitor();
    } catch (err: any) {
      alert('Failed to monitor competitor: ' + err.message);
    } finally {
      setMonitoring(false);
    }
  };

  const handleDelete = async () => {
    if (!competitor) return;
    
    const confirmed = await confirmAction(
      `Are you sure you want to delete ${competitor.name}? This action cannot be undone.`,
      async () => {
        try {
          await CompetitorIntelligenceAPI.deleteCompetitor(competitor.id);
          navigate('/competitor-intelligence/competitors');
        } catch (err: any) {
          showErrorNotification(new Error('Failed to delete competitor: ' + err.message));
        }
      },
      {
        confirmText: 'Delete',
        cancelText: 'Cancel',
        type: 'danger'
      }
    );
  };

  const getTierColor = (tier: CompetitorTier) => {
    switch (tier) {
      case 'direct':
        return 'bg-red-100 text-red-800';
      case 'indirect':
        return 'bg-yellow-100 text-yellow-800';
      case 'aspirational':
        return 'bg-purple-100 text-purple-800';
      case 'adjacent':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getIndustryColor = (industry: Industry) => {
    const colors = [
      'bg-green-100 text-green-800',
      'bg-blue-100 text-blue-800',
      'bg-purple-100 text-purple-800',
      'bg-pink-100 text-pink-800',
      'bg-indigo-100 text-indigo-800',
    ];
    return colors[industry.length % colors.length];
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading competitor...</p>
        </div>
      </div>
    );
  }

  if (error || !competitor) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 text-lg">{error || 'Competitor not found'}</p>
          <Link
            to="/competitor-intelligence/competitors"
            className="mt-4 inline-flex items-center text-blue-600 hover:text-blue-500"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" />
            Back to Competitors
          </Link>
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
            <div className="flex items-center">
              <Link
                to="/competitor-intelligence/competitors"
                className="mr-4 p-2 text-gray-400 hover:text-gray-600"
              >
                <ArrowLeftIcon className="h-5 w-5" />
              </Link>
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
                      <Link to="/competitor-intelligence/competitors" className="text-gray-400 hover:text-gray-500">
                        Competitors
                      </Link>
                    </li>
                    <li>
                      <span className="text-gray-400">/</span>
                    </li>
                    <li>
                      <span className="text-gray-900 font-medium">{competitor.name}</span>
                    </li>
                  </ol>
                </nav>
                <div className="flex items-center mt-2">
                  <h1 className="text-3xl font-bold text-gray-900">{competitor.name}</h1>
                  <div className="ml-4 flex items-center space-x-2">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTierColor(competitor.tier)}`}>
                      {competitor.tier}
                    </span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getIndustryColor(competitor.industry)}`}>
                      {competitor.industry}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={handleMonitorNow}
                disabled={monitoring}
                className="inline-flex items-center px-3 py-2 border border-blue-300 rounded-md text-sm font-medium text-blue-700 bg-white hover:bg-blue-50 disabled:opacity-50"
              >
                <EyeIcon className="h-4 w-4 mr-2" />
                {monitoring ? 'Monitoring...' : 'Monitor Now'}
              </button>
              <Link
                to={`/competitor-intelligence/competitors/${competitor.id}/edit`}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <PencilIcon className="h-4 w-4 mr-2" />
                Edit
              </Link>
              <button
                onClick={handleDelete}
                className="inline-flex items-center px-3 py-2 border border-red-300 rounded-md text-sm font-medium text-red-700 bg-white hover:bg-red-50"
              >
                <TrashIcon className="h-4 w-4 mr-2" />
                Delete
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Overview Card */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Overview</h2>
                <div className="space-y-4">
                  <div className="flex items-start">
                    <GlobeAltIcon className="h-5 w-5 text-gray-400 mt-0.5 mr-3" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">Website</p>
                      <a
                        href={competitor.domain}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-500"
                      >
                        {competitor.domain}
                      </a>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-sm font-medium text-gray-900 mb-2">Description</p>
                    <p className="text-gray-600">{competitor.description}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Analytics Card */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <ChartBarIcon className="h-5 w-5 mr-2" />
                  Analytics & Performance
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">0</div>
                    <div className="text-sm text-gray-500">Content Items</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">0.0</div>
                    <div className="text-sm text-gray-500">Avg Engagement</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">0</div>
                    <div className="text-sm text-gray-500">Trending Score</div>
                  </div>
                </div>
                
                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  <p className="text-center text-gray-500">
                    Analytics data will appear here once content monitoring is active.
                  </p>
                </div>
              </div>
            </div>

            {/* Content History Card */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <ClockIcon className="h-5 w-5 mr-2" />
                  Recent Content ({content.length})
                </h2>
                
                {content.length === 0 ? (
                  <div className="text-center py-12">
                    <EyeIcon className="h-12 w-12 text-gray-400 mx-auto" />
                    <h3 className="mt-4 text-lg font-medium text-gray-900">No content tracked yet</h3>
                    <p className="mt-2 text-gray-600">
                      Content from this competitor will appear here once monitoring begins.
                    </p>
                    <button
                      onClick={handleMonitorNow}
                      disabled={monitoring}
                      className="mt-4 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
                    >
                      {monitoring ? 'Monitoring...' : 'Start Monitoring'}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {content.slice(0, 5).map((item) => (
                      <div key={item.id} className="border rounded-lg p-4 hover:bg-gray-50">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 mb-1">
                              <a
                                href={item.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:text-blue-500"
                              >
                                {item.title}
                              </a>
                            </h4>
                            <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                              {item.content?.substring(0, 150)}...
                            </p>
                            <div className="flex items-center space-x-4 text-xs text-gray-500">
                              <span className="capitalize">{item.contentType?.replace('_', ' ')}</span>
                              <span className="capitalize">{item.platform}</span>
                              {item.publishedAt && (
                                <span>{new Date(item.publishedAt).toLocaleDateString()}</span>
                              )}
                              {item.author && <span>by {item.author}</span>}
                            </div>
                          </div>
                          {item.qualityScore && (
                            <div className="ml-4">
                              <div className="text-xs text-gray-500">Quality</div>
                              <div className="text-sm font-medium">{item.qualityScore.toFixed(1)}/10</div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    
                    {content.length > 5 && (
                      <div className="text-center pt-4">
                        <button className="text-blue-600 hover:text-blue-500 text-sm font-medium">
                          View all {content.length} items
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Alerts Card */}
            {alerts.length > 0 && (
              <div className="bg-white rounded-lg shadow">
                <div className="p-6">
                  <h2 className="text-lg font-medium text-gray-900 mb-4">
                    Recent Alerts ({alerts.length})
                  </h2>
                  
                  <div className="space-y-3">
                    {alerts.slice(0, 5).map((alert) => (
                      <div key={alert.id} className="border-l-4 border-yellow-400 bg-yellow-50 p-4">
                        <div className="flex">
                          <div className="flex-1">
                            <h4 className="text-sm font-medium text-yellow-800">
                              {alert.title}
                            </h4>
                            <p className="text-sm text-yellow-700 mt-1">
                              {alert.message}
                            </p>
                            <p className="text-xs text-yellow-600 mt-2">
                              {new Date(alert.created_at).toLocaleString()}
                            </p>
                          </div>
                          <div className="ml-4">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              alert.priority === 'high' ? 'bg-red-100 text-red-800' :
                              alert.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {alert.priority}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Monitoring Configuration */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Monitoring Configuration</h3>
                
                {/* Platforms */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Platforms</h4>
                  {competitor.platforms && Array.isArray(competitor.platforms) && competitor.platforms.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {competitor.platforms.map((platform) => (
                        <span
                          key={platform}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                        >
                          {platform.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No platforms configured</p>
                  )}
                </div>

                {/* Keywords */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <TagIcon className="h-4 w-4 mr-1" />
                    Keywords
                  </h4>
                  {competitor.monitoringKeywords && Array.isArray(competitor.monitoringKeywords) && competitor.monitoringKeywords.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {competitor.monitoringKeywords.map((keyword) => (
                        <span
                          key={keyword}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800"
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No keywords configured</p>
                  )}
                </div>
              </div>
            </div>

            {/* Metadata */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <CalendarIcon className="h-5 w-5 mr-2" />
                  Information
                </h3>
                
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Status:</span>
                    <span className={`ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                      competitor.isActive ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {competitor.isActive ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  
                  <div>
                    <span className="font-medium text-gray-700">Added:</span>
                    <span className="ml-2 text-gray-600">
                      {new Date(competitor.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <div>
                    <span className="font-medium text-gray-700">Last Updated:</span>
                    <span className="ml-2 text-gray-600">
                      {new Date(competitor.updatedAt).toLocaleDateString()}
                    </span>
                  </div>
                  
                  {competitor.lastMonitored && (
                    <div>
                      <span className="font-medium text-gray-700">Last Monitored:</span>
                      <span className="ml-2 text-gray-600">
                        {new Date(competitor.lastMonitored).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CompetitorDetail;