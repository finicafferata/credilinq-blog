import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  BarChart3, 
  Users, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Activity,
  Zap,
  TrendingUp,
  Settings,
  Eye,
  Plus,
  Filter,
  RefreshCw
} from 'lucide-react';
import { CampaignProgressWidget } from '../components/dashboard/CampaignProgressWidget';
import { AgentStatusPanel } from '../components/dashboard/AgentStatusPanel';
import CampaignOrchestrationWizard from '../components/CampaignOrchestrationWizard';
import { CampaignDetails } from '../components/CampaignDetails';
import { campaignApi } from '../lib/api';

// Types for campaign orchestration
interface Campaign {
  id: string;
  name: string;
  type: 'content_marketing' | 'blog_series' | 'seo_content' | 'email_sequence';
  status: 'draft' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  createdAt: string;
  targetChannels: string[];
  assignedAgents: string[];
  currentStep?: string;
  estimatedCompletion?: string;
  metrics: {
    tasksCompleted: number;
    totalTasks: number;
    contentGenerated: number;
    agentsActive: number;
  };
}

interface Agent {
  id: string;
  name: string;
  type: 'writer' | 'editor' | 'social_media' | 'seo' | 'planner' | 'researcher';
  status: 'active' | 'idle' | 'busy' | 'offline' | 'error';
  currentTask?: string;
  campaignId?: string;
  campaignName?: string;
  performance: {
    tasksCompleted: number;
    averageTime: number;
    successRate: number;
    uptime: number;
    memoryUsage: number;
    responseTime: number;
    errorRate: number;
  };
  capabilities: string[];
  load: number; // 0-100
  queuedTasks: number;
  lastActivity?: string;
}

interface WorkflowStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  agentId?: string;
  duration?: number;
  startTime?: string;
  endTime?: string;
}

interface SystemMetrics {
  totalCampaigns: number;
  activeCampaigns: number;
  totalAgents: number;
  activeAgents: number;
  averageResponseTime: number;
  systemLoad: number;
  eventsPerSecond: number;
  messagesInQueue: number;
}

export function CampaignOrchestrationDashboard() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [selectedCampaign, setSelectedCampaign] = useState<string | null>(null);
  const [selectedCampaignDetails, setSelectedCampaignDetails] = useState<any | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [showWizard, setShowWizard] = useState(false);

  // Handle campaign creation success
  const handleCampaignCreated = () => {
    // Refresh campaigns data
    fetchDashboardData();
    console.log('Campaign created successfully - refreshing dashboard');
  };

  // Fetch detailed campaign data
  const fetchCampaignDetails = async (campaignId: string) => {
    try {
      const details = await campaignApi.get(campaignId);
      setSelectedCampaignDetails(details);
      setSelectedCampaign(campaignId);
    } catch (error) {
      console.error('Error fetching campaign details:', error);
      // Show basic campaign info from dashboard data if API fails
      const campaign = campaigns.find(c => c.id === campaignId);
      if (campaign) {
        setSelectedCampaignDetails({
          id: campaign.id,
          name: campaign.name,
          status: campaign.status,
          tasks: [],
          scheduled_posts: [],
          strategy: {},
          timeline: [],
          performance: {}
        });
        setSelectedCampaign(campaignId);
      }
    }
  };

  // Fetch real data from API
  const fetchDashboardData = async () => {
    try {
      const data = await campaignApi.getOrchestrationDashboard();
      setCampaigns(data.campaigns);
      setAgents(data.agents);
      setSystemMetrics(data.systemMetrics);
      setIsConnected(true);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setIsConnected(false);
    }
  };

  // Initialize data and set up refresh
  useEffect(() => {
    // Fetch initial data
    fetchDashboardData();

    // Set up auto-refresh
    const interval = setInterval(() => {
      if (autoRefresh) {
        fetchDashboardData();
      }
    }, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': case 'active': case 'busy': return 'text-green-600 bg-green-100';
      case 'paused': case 'idle': return 'text-yellow-600 bg-yellow-100';
      case 'failed': case 'offline': return 'text-red-600 bg-red-100';
      case 'completed': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'content_marketing': return <BarChart3 className="w-4 h-4" />;
      case 'blog_series': return <Activity className="w-4 h-4" />;
      case 'seo_content': return <TrendingUp className="w-4 h-4" />;
      case 'email_sequence': return <Zap className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const filteredCampaigns = campaigns.filter(campaign => 
    filterStatus === 'all' || campaign.status === filterStatus
  );

  const activeCampaignsCount = campaigns.filter(c => c.status === 'running').length;
  const busyAgentsCount = agents.filter(a => a.status === 'busy').length;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Campaign Orchestration</h1>
            <p className="text-gray-600 mt-1">Real-time campaign monitoring and agent management</p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {/* Auto Refresh Toggle */}
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm text-gray-600">Auto refresh</span>
            </label>

            {/* Actions */}
            <button 
              onClick={() => setShowWizard(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              <Plus className="w-4 h-4" />
              <span>🚀 New Campaign</span>
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* System Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Active Campaigns */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Campaigns</p>
                <p className="text-3xl font-bold text-gray-900">{activeCampaignsCount}</p>
                <p className="text-sm text-green-600 mt-1">
                  +2 this week
                </p>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <BarChart3 className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>

          {/* Active Agents */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Agents</p>
                <p className="text-3xl font-bold text-gray-900">{busyAgentsCount}</p>
                <p className="text-sm text-gray-600 mt-1">
                  of {agents.length} total
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg">
                <Users className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          {/* System Performance */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Response</p>
                <p className="text-3xl font-bold text-gray-900">
                  {systemMetrics?.averageResponseTime.toFixed(0)}ms
                </p>
                <p className="text-sm text-green-600 mt-1">
                  -12ms improved
                </p>
              </div>
              <div className="p-3 bg-yellow-100 rounded-lg">
                <Clock className="w-6 h-6 text-yellow-600" />
              </div>
            </div>
          </div>

          {/* Event Throughput */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Events/sec</p>
                <p className="text-3xl font-bold text-gray-900">
                  {systemMetrics?.eventsPerSecond.toFixed(0)}
                </p>
                <p className="text-sm text-blue-600 mt-1">
                  Real-time processing
                </p>
              </div>
              <div className="p-3 bg-purple-100 rounded-lg">
                <Activity className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Enhanced Campaign Progress Widgets */}
          <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Active Campaigns</h2>
              <div className="flex items-center space-x-3">
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="border border-gray-300 rounded-lg px-3 py-1 text-sm"
                >
                  <option value="all">All Status</option>
                  <option value="running">Running</option>
                  <option value="paused">Paused</option>
                  <option value="completed">Completed</option>
                </select>
                <button 
                  onClick={fetchDashboardData}
                  className="p-2 text-gray-600 hover:text-gray-900"
                  title="Refresh data"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>
            
            {/* Enhanced Campaign Progress Widgets */}
            {filteredCampaigns.map((campaign) => {
              // Convert campaign to workflow steps format
              const workflowSteps = [
                {
                  id: 'step_1',
                  name: 'Content Planning',
                  status: 'completed' as const,
                  agentName: 'PlannerBot',
                  duration: 15000
                },
                {
                  id: 'step_2', 
                  name: campaign.currentStep || 'Content Creation',
                  status: campaign.status === 'running' ? 'running' as const : 
                          campaign.status === 'paused' ? 'pending' as const : 'completed' as const,
                  agentName: campaign.assignedAgents[0] || 'Content Writer Agent',
                  progress: campaign.status === 'running' ? 75 : undefined
                },
                {
                  id: 'step_3',
                  name: 'Content Review & Optimization',
                  status: 'pending' as const
                },
                {
                  id: 'step_4',
                  name: 'Publishing & Distribution',
                  status: 'pending' as const
                }
              ];

              return (
                <CampaignProgressWidget
                  key={campaign.id}
                  campaignId={campaign.id}
                  campaignName={campaign.name}
                  status={campaign.status}
                  progress={campaign.progress}
                  currentStep={campaign.currentStep}
                  workflow={workflowSteps}
                  estimatedCompletion={campaign.estimatedCompletion}
                  onViewDetails={() => fetchCampaignDetails(campaign.id)}
                  onControlCampaign={async (action) => {
                    try {
                      await campaignApi.controlCampaign(campaign.id, action);
                      console.log(`${action} campaign:`, campaign.id);
                      // Refresh data to show updated status
                      fetchDashboardData();
                    } catch (error) {
                      console.error(`Error ${action} campaign:`, error);
                    }
                  }}
                />
              );
            })}
          </div>

          {/* Enhanced Agent Status Panel */}
          <div className="space-y-6">
            {/* System Health */}
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">System Load</span>
                    <span className="font-medium">{systemMetrics?.systemLoad.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        (systemMetrics?.systemLoad || 0) > 80 ? 'bg-red-500' :
                        (systemMetrics?.systemLoad || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${systemMetrics?.systemLoad || 0}%` }}
                    ></div>
                  </div>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Messages in Queue</span>
                  <span className="text-sm font-medium">
                    {systemMetrics?.messagesInQueue || 0}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Agent Utilization</span>
                  <span className="text-sm font-medium">
                    {((busyAgentsCount / agents.length) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Enhanced Agent Status Panel */}
            <AgentStatusPanel
              agents={agents.map(agent => ({
                ...agent,
                campaignName: campaigns.find(c => c.id === agent.campaignId)?.name
              }))}
              onAgentSelect={async (agentId) => {
                try {
                  const performance = await campaignApi.getAgentPerformance(agentId);
                  console.log('Agent performance:', performance);
                } catch (error) {
                  console.error('Error fetching agent performance:', error);
                }
              }}
              onAgentAction={(agentId, action) => {
                console.log('Agent action:', agentId, action);
                // For now, just log the action. Could be extended to handle specific agent actions
                if (action === 'refresh') {
                  fetchDashboardData();
                }
              }}
              showDetailed={true}
            />
          </div>
        </div>
      </div>

      {/* Campaign Orchestration Wizard */}
      <CampaignOrchestrationWizard
        isOpen={showWizard}
        onClose={() => setShowWizard(false)}
        onCampaignCreated={handleCampaignCreated}
      />

      {/* Campaign Details Modal */}
      {selectedCampaignDetails && (
        <CampaignDetails
          campaign={selectedCampaignDetails}
          onClose={() => {
            setSelectedCampaignDetails(null);
            setSelectedCampaign(null);
          }}
        />
      )}
    </div>
  );
}

export default CampaignOrchestrationDashboard;