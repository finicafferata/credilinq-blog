import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { campaignApi, blogApi } from '../lib/api';
import type { CampaignSummary, CampaignCreateRequest, BlogSummary } from '../lib/api';
import { AppError } from '../lib/errors';
import CampaignOrchestrationWizard from '../components/CampaignOrchestrationWizard';
import { 
  PlayIcon, 
  PauseIcon, 
  EyeIcon, 
  CalendarIcon,
  ClockIcon,
  ChartBarIcon,
  RocketLaunchIcon,
  Squares2X2Icon,
  ListBulletIcon,
  FunnelIcon,
  PlusIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';

interface CampaignsProps {
  onNavigate: (page: string) => void;
}

type ViewMode = 'grid' | 'list';
type FilterStatus = 'all' | 'active' | 'draft' | 'completed' | 'paused';

const Campaigns: React.FC<CampaignsProps> = ({ onNavigate }) => {
  const navigate = useNavigate();
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);
  const [blogs, setBlogs] = useState<BlogSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showWizard, setShowWizard] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');

  useEffect(() => {
    const initializeData = async () => {
      await loadCampaigns();
    };
    initializeData();
  }, []);

  const loadCampaigns = async () => {
    try {
      setLoading(true);
      const data = await campaignApi.list();
      setCampaigns(data);
      setError(null);
    } catch (err) {
      const error = err as AppError;
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCampaignCreated = async () => {
    await loadCampaigns();
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      case 'draft':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'completed':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'paused':
        return 'bg-gray-50 text-gray-700 border-gray-200';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return <PlayIcon className="w-3 h-3" />;
      case 'draft':
        return <ExclamationTriangleIcon className="w-3 h-3" />;
      case 'completed':
        return <CheckCircleIcon className="w-3 h-3" />;
      case 'paused':
        return <PauseIcon className="w-3 h-3" />;
      default:
        return null;
    }
  };

  const filteredCampaigns = campaigns.filter(campaign => 
    filterStatus === 'all' || campaign.status.toLowerCase() === filterStatus
  );

  const campaignStats = {
    total: campaigns.length,
    active: campaigns.filter(c => c.status.toLowerCase() === 'active').length,
    completed: campaigns.filter(c => c.status.toLowerCase() === 'completed').length,
    draft: campaigns.filter(c => c.status.toLowerCase() === 'draft').length,
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (error) {
      return 'Invalid date';
    }
  };

  if (loading && campaigns.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-gray-600 font-medium">Loading campaigns...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Modern Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-8">
            <div>
              <div className="flex items-center space-x-3 mb-2">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                  <RocketLaunchIcon className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Campaigns
                </h1>
              </div>
              <p className="text-gray-600">Create, manage and track your AI-powered marketing campaigns</p>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowWizard(true)}
                className="inline-flex items-center px-6 py-3 border border-transparent rounded-xl shadow-sm text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200"
              >
                <PlusIcon className="w-5 h-5 mr-2" />
                Launch Campaign
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50/80 backdrop-blur-sm border border-red-200/50 rounded-xl p-4 shadow-sm">
            <div className="flex">
              <div className="flex-shrink-0">
                <ExclamationTriangleIcon className="h-5 w-5 text-red-400" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* Campaign Stats */}
        {campaigns.length > 0 && (
          <div className="mb-8 grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-gray-200/50">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <ChartBarIcon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Campaigns</p>
                  <p className="text-2xl font-bold text-gray-900">{campaignStats.total}</p>
                </div>
              </div>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-gray-200/50">
              <div className="flex items-center">
                <div className="p-2 bg-emerald-100 rounded-lg">
                  <PlayIcon className="w-6 h-6 text-emerald-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Active</p>
                  <p className="text-2xl font-bold text-emerald-600">{campaignStats.active}</p>
                </div>
              </div>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-gray-200/50">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <CheckCircleIcon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Completed</p>
                  <p className="text-2xl font-bold text-blue-600">{campaignStats.completed}</p>
                </div>
              </div>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-gray-200/50">
              <div className="flex items-center">
                <div className="p-2 bg-amber-100 rounded-lg">
                  <ClockIcon className="w-6 h-6 text-amber-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Draft</p>
                  <p className="text-2xl font-bold text-amber-600">{campaignStats.draft}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Campaign Orchestration Wizard */}
        <CampaignOrchestrationWizard
          isOpen={showWizard}
          onClose={() => setShowWizard(false)}
          onCampaignCreated={handleCampaignCreated}
        />

        {/* Filters and View Controls */}
        {campaigns.length > 0 && (
          <div className="mb-8 bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-gray-200/50">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <FunnelIcon className="w-5 h-5 text-gray-400" />
                  <span className="text-sm font-medium text-gray-700">Filter:</span>
                </div>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value as FilterStatus)}
                  className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="draft">Draft</option>
                  <option value="completed">Completed</option>
                  <option value="paused">Paused</option>
                </select>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-700">View:</span>
                <div className="flex bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-1.5 rounded-md transition-colors ${
                      viewMode === 'grid'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <Squares2X2Icon className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={`p-1.5 rounded-md transition-colors ${
                      viewMode === 'list'
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <ListBulletIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Campaigns Content */}
        {filteredCampaigns.length === 0 && !loading ? (
          <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-sm border border-gray-200/50">
            <div className="text-center py-16 px-6">
              <div className="mx-auto w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center mb-6">
                <RocketLaunchIcon className="w-12 h-12 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {filterStatus === 'all' ? 'No campaigns yet' : `No ${filterStatus} campaigns`}
              </h3>
              <p className="text-gray-600 mb-8 max-w-md mx-auto">
                {filterStatus === 'all' 
                  ? 'Get started by creating your first AI-powered marketing campaign with intelligent content generation.'
                  : `There are no campaigns with ${filterStatus} status at the moment.`
                }
              </p>
              {filterStatus === 'all' && (
                <button
                  onClick={() => setShowWizard(true)}
                  className="inline-flex items-center px-6 py-3 border border-transparent shadow-sm text-sm font-medium rounded-xl text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200"
                >
                  <PlusIcon className="w-5 h-5 mr-2" />
                  Create Your First Campaign
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className={
            viewMode === 'grid' 
              ? "grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6"
              : "space-y-4"
          }>
            {filteredCampaigns.map((campaign) => (
              viewMode === 'grid' ? (
                <div key={campaign.id} className="group bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-xl shadow-sm hover:shadow-lg hover:shadow-blue-100/50 transition-all duration-300 hover:-translate-y-1">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors line-clamp-2">
                          {campaign.name}
                        </h3>
                        <div className="flex items-center text-sm text-gray-500 mb-3">
                          <CalendarIcon className="w-4 h-4 mr-1" />
                          <span>Created {formatDate(campaign.created_at)}</span>
                        </div>
                      </div>
                      <div className={`inline-flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-medium border ${getStatusColor(campaign.status)}`}>
                        {getStatusIcon(campaign.status)}
                        <span className="capitalize">{campaign.status}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between items-center text-sm text-gray-600 mb-2">
                          <span className="flex items-center">
                            <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
                            Progress
                          </span>
                          <span className="font-medium">{(campaign.progress || 0).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2.5 rounded-full transition-all duration-500"
                            style={{ width: `${campaign.progress || 0}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center text-gray-600">
                          <CheckCircleIcon className="w-4 h-4 mr-1" />
                          <span>{campaign.completed_tasks || 0} of {campaign.total_tasks || 0} tasks</span>
                        </div>
                        <div className="text-gray-500">
                          {(campaign.total_tasks || 0) > 0 ? Math.round((((campaign.total_tasks || 0) - (campaign.completed_tasks || 0)) / (campaign.total_tasks || 1)) * 100) : 0}% remaining
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-6 flex space-x-3">
                      <button
                        onClick={() => {
                          console.log('View campaign:', campaign.id);
                          navigate(`/campaigns/${campaign.id}`);
                        }}
                        className="flex-1 inline-flex items-center justify-center px-4 py-2 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 transition-all duration-200"
                      >
                        <EyeIcon className="w-4 h-4 mr-2" />
                        View Details
                      </button>
                      <button
                        onClick={async () => {
                          try {
                            await campaignApi.schedule(campaign.id);
                            await loadCampaigns();
                          } catch (err) {
                            const error = err as AppError;
                            setError(error.message);
                          }
                        }}
                        className="flex-1 inline-flex items-center justify-center px-4 py-2 border border-gray-300 rounded-lg shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
                      >
                        <CalendarIcon className="w-4 h-4 mr-2" />
                        Schedule
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div key={campaign.id} className="bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-xl shadow-sm hover:shadow-md transition-all duration-200">
                  <div className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-4">
                          <h3 className="text-lg font-semibold text-gray-900 hover:text-blue-600 cursor-pointer transition-colors">
                            {campaign.name}
                          </h3>
                          <div className={`inline-flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-medium border ${getStatusColor(campaign.status)}`}>
                            {getStatusIcon(campaign.status)}
                            <span className="capitalize">{campaign.status}</span>
                          </div>
                        </div>
                        <div className="mt-2 flex items-center space-x-6 text-sm text-gray-600">
                          <span className="flex items-center">
                            <CalendarIcon className="w-4 h-4 mr-1" />
                            Created {formatDate(campaign.created_at)}
                          </span>
                          <span className="flex items-center">
                            <CheckCircleIcon className="w-4 h-4 mr-1" />
                            {campaign.completed_tasks || 0}/{campaign.total_tasks || 0} tasks
                          </span>
                          <span className="flex items-center">
                            <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
                            {(campaign.progress || 0).toFixed(0)}% complete
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <div className="w-32">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${campaign.progress || 0}%` }}
                            ></div>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <button
                            onClick={() => navigate(`/campaigns/${campaign.id}`)}
                            className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          >
                            <EyeIcon className="w-5 h-5" />
                          </button>
                          <button
                            onClick={async () => {
                              try {
                                await campaignApi.schedule(campaign.id);
                                await loadCampaigns();
                              } catch (err) {
                                const error = err as AppError;
                                setError(error.message);
                              }
                            }}
                            className="p-2 text-gray-600 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                          >
                            <CalendarIcon className="w-5 h-5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Campaigns; 