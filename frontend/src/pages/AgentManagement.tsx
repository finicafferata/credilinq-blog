import React, { useState, useEffect, useCallback } from 'react';
import { agentApi, type AgentInfo, type AgentPoolStats } from '../services/agentApi';
import {
  Users,
  Activity,
  Zap,
  Clock,
  BarChart3,
  AlertCircle,
  CheckCircle,
  Pause,
  Play,
  Settings,
  Plus,
  Minus,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Shield,
  Cpu,
  Database,
  Cloud,
  Globe,
  Terminal,
  Eye,
  Filter,
  Download,
  Upload,
  Search,
  ChevronDown,
  ChevronUp,
  MoreVertical,
  Power,
  Trash2,
  Edit3,
  Copy
} from 'lucide-react';

// Agent types and interfaces
export interface AgentCapability {
  id: string;
  name: string;
  category: 'content' | 'seo' | 'social' | 'email' | 'analytics' | 'data' | 'custom';
  proficiency: number; // 0-100
}

export interface AgentTask {
  id: string;
  name: string;
  campaignId: string;
  campaignName: string;
  priority: 'low' | 'normal' | 'high' | 'critical';
  status: 'queued' | 'processing' | 'completed' | 'failed';
  assignedAt: string;
  startedAt?: string;
  completedAt?: string;
  estimatedDuration: number;
  actualDuration?: number;
  progress?: number;
}

export interface AgentMetrics {
  tasksCompleted: number;
  tasksInProgress: number;
  tasksQueued: number;
  avgResponseTime: number;
  avgCompletionTime: number;
  successRate: number;
  errorRate: number;
  uptime: number;
  lastError?: string;
  lastErrorTime?: string;
}

export interface AgentResource {
  cpu: number; // 0-100
  memory: number; // 0-100
  network: number; // Mbps
  storage: number; // GB
  maxConcurrency: number;
  currentConcurrency: number;
}

export interface Agent {
  id: string;
  name: string;
  type: 'content_writer' | 'editor' | 'seo_specialist' | 'social_media' | 'analyst' | 'researcher' | 'coordinator';
  status: 'online' | 'busy' | 'idle' | 'offline' | 'error' | 'maintenance';
  version: string;
  deployment: 'cloud' | 'local' | 'edge';
  capabilities: AgentCapability[];
  currentTasks: AgentTask[];
  metrics: AgentMetrics;
  resources: AgentResource;
  config: {
    autoScale: boolean;
    maxTasks: number;
    priority: 'low' | 'normal' | 'high';
    timeout: number;
    retryPolicy: {
      enabled: boolean;
      maxAttempts: number;
      backoffMs: number;
    };
  };
  health: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    lastCheck: string;
    issues: string[];
  };
}

// Mock data generator
const generateMockAgents = (): Agent[] => {
  return [
    {
      id: 'agent_001',
      name: 'ContentMaster Pro',
      type: 'content_writer',
      status: 'busy',
      version: '2.3.1',
      deployment: 'cloud',
      capabilities: [
        { id: 'cap_1', name: 'Blog Writing', category: 'content', proficiency: 95 },
        { id: 'cap_2', name: 'Technical Documentation', category: 'content', proficiency: 88 },
        { id: 'cap_3', name: 'Long-form Content', category: 'content', proficiency: 92 }
      ],
      currentTasks: [
        {
          id: 'task_1',
          name: 'Write Q1 Marketing Blog Post',
          campaignId: 'camp_1',
          campaignName: 'Q1 Content Marketing',
          priority: 'high',
          status: 'processing',
          assignedAt: new Date(Date.now() - 900000).toISOString(),
          startedAt: new Date(Date.now() - 600000).toISOString(),
          estimatedDuration: 1800000,
          progress: 65
        }
      ],
      metrics: {
        tasksCompleted: 156,
        tasksInProgress: 1,
        tasksQueued: 3,
        avgResponseTime: 1200,
        avgCompletionTime: 25000,
        successRate: 96.5,
        errorRate: 3.5,
        uptime: 172800
      },
      resources: {
        cpu: 68,
        memory: 72,
        network: 12.5,
        storage: 45,
        maxConcurrency: 5,
        currentConcurrency: 1
      },
      config: {
        autoScale: true,
        maxTasks: 5,
        priority: 'high',
        timeout: 300000,
        retryPolicy: {
          enabled: true,
          maxAttempts: 3,
          backoffMs: 5000
        }
      },
      health: {
        status: 'healthy',
        lastCheck: new Date().toISOString(),
        issues: []
      }
    },
    {
      id: 'agent_002',
      name: 'SEO Optimizer Elite',
      type: 'seo_specialist',
      status: 'idle',
      version: '1.8.4',
      deployment: 'cloud',
      capabilities: [
        { id: 'cap_4', name: 'Keyword Research', category: 'seo', proficiency: 98 },
        { id: 'cap_5', name: 'Content Optimization', category: 'seo', proficiency: 94 },
        { id: 'cap_6', name: 'Technical SEO', category: 'seo', proficiency: 90 }
      ],
      currentTasks: [],
      metrics: {
        tasksCompleted: 203,
        tasksInProgress: 0,
        tasksQueued: 0,
        avgResponseTime: 800,
        avgCompletionTime: 15000,
        successRate: 98.2,
        errorRate: 1.8,
        uptime: 259200
      },
      resources: {
        cpu: 15,
        memory: 22,
        network: 5.2,
        storage: 12,
        maxConcurrency: 3,
        currentConcurrency: 0
      },
      config: {
        autoScale: false,
        maxTasks: 3,
        priority: 'normal',
        timeout: 180000,
        retryPolicy: {
          enabled: true,
          maxAttempts: 2,
          backoffMs: 3000
        }
      },
      health: {
        status: 'healthy',
        lastCheck: new Date().toISOString(),
        issues: []
      }
    },
    {
      id: 'agent_003',
      name: 'Social Media Genius',
      type: 'social_media',
      status: 'online',
      version: '3.1.0',
      deployment: 'edge',
      capabilities: [
        { id: 'cap_7', name: 'LinkedIn Posts', category: 'social', proficiency: 92 },
        { id: 'cap_8', name: 'Twitter Threads', category: 'social', proficiency: 88 },
        { id: 'cap_9', name: 'Content Scheduling', category: 'social', proficiency: 95 }
      ],
      currentTasks: [
        {
          id: 'task_2',
          name: 'Create LinkedIn Campaign Posts',
          campaignId: 'camp_2',
          campaignName: 'Product Launch',
          priority: 'normal',
          status: 'queued',
          assignedAt: new Date(Date.now() - 300000).toISOString(),
          estimatedDuration: 600000
        }
      ],
      metrics: {
        tasksCompleted: 89,
        tasksInProgress: 0,
        tasksQueued: 1,
        avgResponseTime: 600,
        avgCompletionTime: 8000,
        successRate: 94.3,
        errorRate: 5.7,
        uptime: 86400
      },
      resources: {
        cpu: 25,
        memory: 30,
        network: 8.3,
        storage: 8,
        maxConcurrency: 10,
        currentConcurrency: 0
      },
      config: {
        autoScale: true,
        maxTasks: 10,
        priority: 'normal',
        timeout: 120000,
        retryPolicy: {
          enabled: false,
          maxAttempts: 1,
          backoffMs: 0
        }
      },
      health: {
        status: 'healthy',
        lastCheck: new Date().toISOString(),
        issues: []
      }
    },
    {
      id: 'agent_004',
      name: 'Data Analyst Pro',
      type: 'analyst',
      status: 'error',
      version: '2.0.3',
      deployment: 'local',
      capabilities: [
        { id: 'cap_10', name: 'Performance Analytics', category: 'analytics', proficiency: 96 },
        { id: 'cap_11', name: 'Predictive Modeling', category: 'analytics', proficiency: 85 },
        { id: 'cap_12', name: 'Report Generation', category: 'analytics', proficiency: 91 }
      ],
      currentTasks: [],
      metrics: {
        tasksCompleted: 67,
        tasksInProgress: 0,
        tasksQueued: 5,
        avgResponseTime: 2000,
        avgCompletionTime: 45000,
        successRate: 88.5,
        errorRate: 11.5,
        uptime: 43200,
        lastError: 'Database connection timeout',
        lastErrorTime: new Date(Date.now() - 1800000).toISOString()
      },
      resources: {
        cpu: 5,
        memory: 10,
        network: 0,
        storage: 120,
        maxConcurrency: 2,
        currentConcurrency: 0
      },
      config: {
        autoScale: false,
        maxTasks: 2,
        priority: 'low',
        timeout: 600000,
        retryPolicy: {
          enabled: true,
          maxAttempts: 5,
          backoffMs: 10000
        }
      },
      health: {
        status: 'unhealthy',
        lastCheck: new Date().toISOString(),
        issues: ['Database connection failed', 'High error rate detected']
      }
    }
  ];
};

export function AgentManagement() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterType, setFilterType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'performance'>('grid');
  const [showAddAgent, setShowAddAgent] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [sortBy, setSortBy] = useState<'name' | 'status' | 'load' | 'performance'>('status');

  // Load agents from API
  const loadAgents = useCallback(async () => {
    try {
      const agentList = await agentApi.listAgents(filterStatus === 'all' ? undefined : filterStatus, filterType === 'all' ? undefined : filterType);
      setAgents(agentList);
    } catch (error) {
      console.error('Error loading agents:', error);
    }
  }, [filterStatus, filterType]);

  useEffect(() => {
    loadAgents();
  }, [loadAgents]);

  // Auto-refresh from API
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadAgents();
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, loadAgents]);

  // Filter and sort agents
  const filteredAgents = agents
    .filter(agent => {
      if (filterStatus !== 'all' && agent.status !== filterStatus) return false;
      if (filterType !== 'all' && agent.type !== filterType) return false;
      if (searchQuery && !agent.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name': return a.name.localeCompare(b.name);
        case 'status': {
          const statusOrder = { error: 0, offline: 1, maintenance: 2, busy: 3, online: 4, idle: 5 };
          return (statusOrder[a.status as keyof typeof statusOrder] || 0) - 
                 (statusOrder[b.status as keyof typeof statusOrder] || 0);
        }
        case 'load': return b.resource_utilization.cpu - a.resource_utilization.cpu;
        case 'performance': return b.metrics.success_rate - a.metrics.success_rate;
        default: return 0;
      }
    });

  // Agent actions
  const handleAgentAction = async (agentId: string, action: 'start' | 'stop' | 'restart' | 'delete' | 'configure') => {
    console.log(`Agent ${agentId}: ${action}`);
    
    try {
      switch (action) {
        case 'start':
          await agentApi.startAgent(agentId);
          break;
        case 'stop':
          await agentApi.stopAgent(agentId);
          break;
        case 'restart':
          await agentApi.restartAgent(agentId);
          break;
        case 'delete':
          await agentApi.deleteAgent(agentId);
          setSelectedAgent(null);
          break;
        case 'configure':
          console.log('Configuration modal not implemented yet');
          break;
      }
      
      // Refresh agents after action
      await loadAgents();
    } catch (error) {
      console.error(`Error ${action} agent:`, error);
    }
  };

  // Scale agent pool
  const scaleAgentPool = (direction: 'up' | 'down') => {
    if (direction === 'up') {
      const newAgent: Agent = {
        id: `agent_${Date.now()}`,
        name: `New Agent ${agents.length + 1}`,
        type: 'content_writer',
        status: 'idle',
        version: '2.3.1',
        deployment: 'cloud',
        capabilities: [],
        currentTasks: [],
        metrics: {
          tasksCompleted: 0,
          tasksInProgress: 0,
          tasksQueued: 0,
          avgResponseTime: 0,
          avgCompletionTime: 0,
          successRate: 100,
          errorRate: 0,
          uptime: 0
        },
        resources: {
          cpu: 0,
          memory: 0,
          network: 0,
          storage: 0,
          maxConcurrency: 5,
          currentConcurrency: 0
        },
        config: {
          autoScale: true,
          maxTasks: 5,
          priority: 'normal',
          timeout: 300000,
          retryPolicy: {
            enabled: true,
            maxAttempts: 3,
            backoffMs: 5000
          }
        },
        health: {
          status: 'healthy',
          lastCheck: new Date().toISOString(),
          issues: []
        }
      };
      setAgents(prev => [...prev, newAgent]);
    } else if (direction === 'down' && agents.length > 1) {
      const idleAgents = agents.filter(a => a.status === 'idle');
      if (idleAgents.length > 0) {
        setAgents(prev => prev.filter(a => a.id !== idleAgents[0].id));
      }
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'busy': return <Zap className="w-4 h-4 text-blue-600 animate-pulse" />;
      case 'idle': return <Clock className="w-4 h-4 text-gray-600" />;
      case 'offline': return <Power className="w-4 h-4 text-gray-400" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-600" />;
      case 'maintenance': return <Settings className="w-4 h-4 text-yellow-600" />;
      default: return null;
    }
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'degraded': return 'text-yellow-600 bg-yellow-100';
      case 'unhealthy': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getDeploymentIcon = (deployment: string) => {
    switch (deployment) {
      case 'cloud': return <Cloud className="w-4 h-4" />;
      case 'local': return <Database className="w-4 h-4" />;
      case 'edge': return <Globe className="w-4 h-4" />;
      default: return null;
    }
  };

  // Calculate pool metrics
  const poolMetrics = {
    totalAgents: agents.length,
    onlineAgents: agents.filter(a => ['online', 'busy', 'idle'].includes(a.status)).length,
    busyAgents: agents.filter(a => a.status === 'busy').length,
    errorAgents: agents.filter(a => a.status === 'error').length,
    avgCpu: agents.reduce((sum, a) => sum + a.resources.cpu, 0) / agents.length,
    avgMemory: agents.reduce((sum, a) => sum + a.resources.memory, 0) / agents.length,
    totalTasks: agents.reduce((sum, a) => sum + a.currentTasks.length, 0),
    queuedTasks: agents.reduce((sum, a) => sum + a.metrics.tasksQueued, 0)
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Agent Management</h1>
              <p className="text-gray-600 mt-1">Monitor and manage your AI agent pool</p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Auto Refresh */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm text-gray-600">Auto refresh</span>
              </label>

              {/* Scale Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => scaleAgentPool('down')}
                  className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                  title="Scale down"
                >
                  <Minus className="w-4 h-4" />
                </button>
                <span className="text-sm font-medium">{poolMetrics.totalAgents} Agents</span>
                <button
                  onClick={() => scaleAgentPool('up')}
                  className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                  title="Scale up"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>

              {/* Add Agent Button */}
              <button
                onClick={() => setShowAddAgent(true)}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <Plus className="w-4 h-4" />
                <span>Add Agent</span>
              </button>
            </div>
          </div>
        </div>

        {/* Pool Metrics Bar */}
        <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
          <div className="grid grid-cols-8 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{poolMetrics.totalAgents}</p>
              <p className="text-xs text-gray-600">Total</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{poolMetrics.onlineAgents}</p>
              <p className="text-xs text-gray-600">Online</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{poolMetrics.busyAgents}</p>
              <p className="text-xs text-gray-600">Busy</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-red-600">{poolMetrics.errorAgents}</p>
              <p className="text-xs text-gray-600">Errors</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">{poolMetrics.avgCpu.toFixed(0)}%</p>
              <p className="text-xs text-gray-600">Avg CPU</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-indigo-600">{poolMetrics.avgMemory.toFixed(0)}%</p>
              <p className="text-xs text-gray-600">Avg Memory</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-orange-600">{poolMetrics.totalTasks}</p>
              <p className="text-xs text-gray-600">Active Tasks</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-yellow-600">{poolMetrics.queuedTasks}</p>
              <p className="text-xs text-gray-600">Queued</p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Controls */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search agents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            {/* Status Filter */}
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg"
            >
              <option value="all">All Status</option>
              <option value="online">Online</option>
              <option value="busy">Busy</option>
              <option value="idle">Idle</option>
              <option value="offline">Offline</option>
              <option value="error">Error</option>
            </select>

            {/* Type Filter */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg"
            >
              <option value="all">All Types</option>
              <option value="content_writer">Content Writer</option>
              <option value="editor">Editor</option>
              <option value="seo_specialist">SEO Specialist</option>
              <option value="social_media">Social Media</option>
              <option value="analyst">Analyst</option>
            </select>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-4 py-2 border border-gray-300 rounded-lg"
            >
              <option value="status">Sort by Status</option>
              <option value="name">Sort by Name</option>
              <option value="load">Sort by Load</option>
              <option value="performance">Sort by Performance</option>
            </select>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-lg ${viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-lg ${viewMode === 'list' ? 'bg-blue-100 text-blue-600' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              <Users className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('performance')}
              className={`p-2 rounded-lg ${viewMode === 'performance' ? 'bg-blue-100 text-blue-600' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              <Activity className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {viewMode === 'grid' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredAgents.map(agent => (
              <div
                key={agent.id}
                className={`bg-white rounded-xl border-2 overflow-hidden hover:shadow-lg transition-all cursor-pointer ${
                  selectedAgent === agent.id ? 'border-blue-500 shadow-lg' : 'border-gray-200'
                }`}
                onClick={() => setSelectedAgent(agent.id === selectedAgent ? null : agent.id)}
              >
                {/* Agent Header */}
                <div className="p-4 border-b border-gray-200">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <Terminal className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900">{agent.name}</h3>
                        <p className="text-sm text-gray-600 capitalize">{agent.type.replace('_', ' ')}</p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end space-y-1">
                      {getStatusIcon(agent.status)}
                      {getDeploymentIcon(agent.deployment)}
                    </div>
                  </div>

                  {/* Status and Health */}
                  <div className="flex items-center justify-between">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      agent.status === 'online' || agent.status === 'busy' ? 'bg-green-100 text-green-700' :
                      agent.status === 'idle' ? 'bg-blue-100 text-blue-700' :
                      agent.status === 'error' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {agent.status}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getHealthColor(agent.health.status)}`}>
                      {agent.health.status}
                    </span>
                  </div>
                </div>

                {/* Resource Usage */}
                <div className="p-4 space-y-3">
                  {/* CPU */}
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-600">CPU</span>
                      <span className="font-medium">{agent.resources.cpu.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          agent.resources.cpu > 80 ? 'bg-red-500' :
                          agent.resources.cpu > 60 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${agent.resources.cpu}%` }}
                      />
                    </div>
                  </div>

                  {/* Memory */}
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-600">Memory</span>
                      <span className="font-medium">{agent.resources.memory.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          agent.resources.memory > 80 ? 'bg-red-500' :
                          agent.resources.memory > 60 ? 'bg-yellow-500' : 'bg-blue-500'
                        }`}
                        style={{ width: `${agent.resources.memory}%` }}
                      />
                    </div>
                  </div>

                  {/* Tasks */}
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Tasks</span>
                    <span className="font-medium">
                      {agent.currentTasks.length}/{agent.config.maxTasks}
                    </span>
                  </div>
                </div>

                {/* Metrics */}
                <div className="p-4 bg-gray-50 border-t border-gray-200">
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div>
                      <p className="text-lg font-bold text-blue-600">{agent.metrics.tasksCompleted}</p>
                      <p className="text-xs text-gray-600">Completed</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-green-600">{agent.metrics.successRate.toFixed(0)}%</p>
                      <p className="text-xs text-gray-600">Success</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-purple-600">{(agent.metrics.avgResponseTime / 1000).toFixed(1)}s</p>
                      <p className="text-xs text-gray-600">Avg Time</p>
                    </div>
                  </div>
                </div>

                {/* Actions (shown when selected) */}
                {selectedAgent === agent.id && (
                  <div className="p-4 border-t border-gray-200 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {agent.status === 'offline' || agent.status === 'error' ? (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAgentAction(agent.id, 'restart');
                          }}
                          className="p-2 text-green-600 hover:bg-green-50 rounded-lg"
                          title="Start agent"
                        >
                          <Play className="w-4 h-4" />
                        </button>
                      ) : (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAgentAction(agent.id, 'stop');
                          }}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                          title="Stop agent"
                        >
                          <Pause className="w-4 h-4" />
                        </button>
                      )}
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAgentAction(agent.id, 'restart');
                        }}
                        className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg"
                        title="Restart agent"
                      >
                        <RefreshCw className="w-4 h-4" />
                      </button>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAgentAction(agent.id, 'configure');
                        }}
                        className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                        title="Configure agent"
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAgentAction(agent.id, 'delete');
                      }}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                      title="Delete agent"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {viewMode === 'list' && (
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Agent
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Resources
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tasks
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Performance
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredAgents.map(agent => (
                  <tr key={agent.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="p-2 bg-blue-100 rounded-lg mr-3">
                          <Terminal className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-900">{agent.name}</div>
                          <div className="text-sm text-gray-500">{agent.type}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(agent.status)}
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          agent.status === 'online' || agent.status === 'busy' ? 'bg-green-100 text-green-700' :
                          agent.status === 'error' ? 'bg-red-100 text-red-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {agent.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2">
                          <Cpu className="w-3 h-3 text-gray-400" />
                          <span className="text-sm">{agent.resources.cpu.toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Database className="w-3 h-3 text-gray-400" />
                          <span className="text-sm">{agent.resources.memory.toFixed(0)}%</span>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm">
                        <div>Active: {agent.currentTasks.length}</div>
                        <div>Queued: {agent.metrics.tasksQueued}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-4">
                        <div className="text-sm">
                          <div>Success: {agent.metrics.successRate.toFixed(1)}%</div>
                          <div>Completed: {agent.metrics.tasksCompleted}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        <button className="p-1 text-gray-600 hover:text-gray-900">
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="p-1 text-gray-600 hover:text-gray-900">
                          <Settings className="w-4 h-4" />
                        </button>
                        <button className="p-1 text-gray-600 hover:text-gray-900">
                          <MoreVertical className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {viewMode === 'performance' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Performance Charts */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Performance</h3>
              <div className="space-y-4">
                {filteredAgents.map(agent => (
                  <div key={agent.id} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3 flex-1">
                      <span className="text-sm font-medium text-gray-900 w-32 truncate">
                        {agent.name}
                      </span>
                      <div className="flex-1 bg-gray-200 rounded-full h-4">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-green-500 h-4 rounded-full"
                          style={{ width: `${agent.metrics.successRate}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-sm font-medium ml-3">
                      {agent.metrics.successRate.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Resource Utilization */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Resource Utilization</h3>
              <div className="space-y-4">
                {filteredAgents.map(agent => (
                  <div key={agent.id} className="border-b border-gray-200 pb-3 last:border-0">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-900">{agent.name}</span>
                      <div className="flex items-center space-x-2">
                        {agent.resources.cpu > 80 && (
                          <AlertCircle className="w-4 h-4 text-red-600" />
                        )}
                        <span className="text-xs text-gray-500">
                          {agent.currentTasks.length}/{agent.config.maxTasks} tasks
                        </span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-600">CPU</span>
                          <span>{agent.resources.cpu.toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full ${
                              agent.resources.cpu > 80 ? 'bg-red-500' :
                              agent.resources.cpu > 60 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${agent.resources.cpu}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-600">Memory</span>
                          <span>{agent.resources.memory.toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full ${
                              agent.resources.memory > 80 ? 'bg-red-500' :
                              agent.resources.memory > 60 ? 'bg-yellow-500' : 'bg-blue-500'
                            }`}
                            style={{ width: `${agent.resources.memory}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {filteredAgents.length === 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
            <Users className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No agents found</h3>
            <p className="text-gray-500">Try adjusting your filters or add a new agent to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AgentManagement;