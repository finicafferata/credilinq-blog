import React, { useState, useEffect } from 'react';
import { 
  Users, 
  Activity, 
  Clock, 
  Zap, 
  AlertCircle, 
  CheckCircle,
  Cpu,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Pause,
  Play,
  Settings,
  Eye,
  ArrowUpRight
} from 'lucide-react';

interface AgentMetrics {
  tasksCompleted: number;
  averageTime: number;
  successRate: number;
  uptime: number;
  memoryUsage: number;
  responseTime: number;
  errorRate: number;
}

interface Agent {
  id: string;
  name: string;
  type: 'writer' | 'editor' | 'social_media' | 'seo' | 'planner' | 'researcher';
  status: 'active' | 'idle' | 'busy' | 'offline' | 'error';
  currentTask?: string;
  campaignId?: string;
  campaignName?: string;
  performance: AgentMetrics;
  capabilities: string[];
  load: number; // 0-100
  queuedTasks: number;
  lastActivity?: string;
}

interface AgentStatusPanelProps {
  agents: Agent[];
  onAgentSelect?: (agentId: string) => void;
  onAgentAction?: (agentId: string, action: 'pause' | 'resume' | 'restart' | 'configure') => void;
  showDetailed?: boolean;
}

export function AgentStatusPanel({
  agents,
  onAgentSelect,
  onAgentAction,
  showDetailed = false
}: AgentStatusPanelProps) {
  const [sortBy, setSortBy] = useState<'name' | 'load' | 'performance' | 'status'>('status');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': case 'busy': return 'text-green-600 bg-green-100';
      case 'idle': return 'text-blue-600 bg-blue-100';
      case 'offline': return 'text-gray-600 bg-gray-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Activity className="w-4 h-4 text-green-600" />;
      case 'busy': return <Zap className="w-4 h-4 text-yellow-600 animate-pulse" />;
      case 'idle': return <Pause className="w-4 h-4 text-blue-600" />;
      case 'offline': return <Clock className="w-4 h-4 text-gray-600" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-600" />;
      default: return <CheckCircle className="w-4 h-4 text-gray-600" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'writer': return '✍️';
      case 'editor': return '📝';
      case 'social_media': return '📱';
      case 'seo': return '🔍';
      case 'planner': return '📋';
      case 'researcher': return '🔬';
      default: return '🤖';
    }
  };

  const getLoadColor = (load: number) => {
    if (load > 80) return 'bg-red-500';
    if (load > 60) return 'bg-yellow-500';
    if (load > 30) return 'bg-blue-500';
    return 'bg-green-500';
  };

  const getPerformanceIndicator = (agent: Agent) => {
    const score = agent.performance.successRate;
    if (score >= 95) return { icon: TrendingUp, color: 'text-green-600', label: 'Excellent' };
    if (score >= 85) return { icon: TrendingUp, color: 'text-blue-600', label: 'Good' };
    if (score >= 70) return { icon: BarChart3, color: 'text-yellow-600', label: 'Fair' };
    return { icon: TrendingDown, color: 'text-red-600', label: 'Poor' };
  };

  // Filter and sort agents
  const filteredAgents = agents
    .filter(agent => filterStatus === 'all' || agent.status === filterStatus)
    .sort((a, b) => {
      switch (sortBy) {
        case 'name': return a.name.localeCompare(b.name);
        case 'load': return b.load - a.load;
        case 'performance': return b.performance.successRate - a.performance.successRate;
        case 'status': 
          const statusOrder = { busy: 4, active: 3, idle: 2, offline: 1, error: 0 };
          return (statusOrder[b.status as keyof typeof statusOrder] || 0) - 
                 (statusOrder[a.status as keyof typeof statusOrder] || 0);
        default: return 0;
      }
    });

  // Calculate aggregate metrics
  const totalAgents = agents.length;
  const activeAgents = agents.filter(a => a.status === 'active' || a.status === 'busy').length;
  const averageLoad = agents.reduce((sum, agent) => sum + agent.load, 0) / totalAgents;
  const averagePerformance = agents.reduce((sum, agent) => sum + agent.performance.successRate, 0) / totalAgents;

  return (
    <div className="bg-white rounded-xl border border-gray-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <Users className="w-5 h-5" />
            <span>Agent Pool</span>
          </h2>
          
          <div className="flex items-center space-x-3">
            {/* Filter */}
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-1 text-sm"
            >
              <option value="all">All Status</option>
              <option value="busy">Busy</option>
              <option value="active">Active</option>
              <option value="idle">Idle</option>
              <option value="offline">Offline</option>
            </select>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="border border-gray-300 rounded-lg px-3 py-1 text-sm"
            >
              <option value="status">Sort by Status</option>
              <option value="name">Sort by Name</option>
              <option value="load">Sort by Load</option>
              <option value="performance">Sort by Performance</option>
            </select>
          </div>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">{activeAgents}</p>
            <p className="text-sm text-gray-600">Active</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{averageLoad.toFixed(0)}%</p>
            <p className="text-sm text-gray-600">Avg Load</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">{averagePerformance.toFixed(0)}%</p>
            <p className="text-sm text-gray-600">Avg Performance</p>
          </div>
        </div>
      </div>

      {/* Agent List */}
      <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
        {filteredAgents.map((agent) => {
          const performanceIndicator = getPerformanceIndicator(agent);
          const isExpanded = expandedAgent === agent.id;

          return (
            <div key={agent.id} className="p-4 hover:bg-gray-50 transition-colors">
              <div 
                className="cursor-pointer"
                onClick={() => setExpandedAgent(isExpanded ? null : agent.id)}
              >
                {/* Agent Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getTypeIcon(agent.type)}</span>
                      {getStatusIcon(agent.status)}
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">{agent.name}</h3>
                      <p className="text-sm text-gray-600 capitalize">{agent.type}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
                      {agent.status}
                    </span>
                    
                    {agent.queuedTasks > 0 && (
                      <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                        {agent.queuedTasks} queued
                      </span>
                    )}
                  </div>
                </div>

                {/* Current Task */}
                {agent.currentTask && (
                  <div className="mb-3">
                    <p className="text-sm text-blue-600 truncate">
                      🎯 {agent.currentTask}
                    </p>
                    {agent.campaignName && (
                      <p className="text-xs text-gray-500">
                        Campaign: {agent.campaignName}
                      </p>
                    )}
                  </div>
                )}

                {/* Load Bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Load</span>
                    <span className="font-medium">{agent.load.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${getLoadColor(agent.load)}`}
                      style={{ width: `${agent.load}%` }}
                    ></div>
                  </div>
                </div>

                {/* Performance Indicator */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 text-sm">
                    <performanceIndicator.icon className={`w-4 h-4 ${performanceIndicator.color}`} />
                    <span className={performanceIndicator.color}>
                      {performanceIndicator.label} ({agent.performance.successRate.toFixed(0)}%)
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    {onAgentSelect && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onAgentSelect(agent.id);
                        }}
                        className="p-1 text-gray-600 hover:text-gray-900"
                        title="View details"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                    )}
                    
                    {onAgentAction && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onAgentAction(agent.id, 'configure');
                        }}
                        className="p-1 text-gray-600 hover:text-gray-900"
                        title="Configure"
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              {isExpanded && showDetailed && (
                <div className="mt-4 pt-4 border-t border-gray-200 space-y-4">
                  {/* Performance Metrics */}
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Performance Metrics</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Tasks Completed:</span>
                        <span className="font-medium ml-2">{agent.performance.tasksCompleted}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Avg Response:</span>
                        <span className="font-medium ml-2">{agent.performance.responseTime}ms</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Success Rate:</span>
                        <span className="font-medium ml-2">{agent.performance.successRate.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Error Rate:</span>
                        <span className="font-medium ml-2">{agent.performance.errorRate.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Capabilities */}
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Capabilities</h4>
                    <div className="flex flex-wrap gap-2">
                      {agent.capabilities.map((capability) => (
                        <span
                          key={capability}
                          className="bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded-full"
                        >
                          {capability}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* System Health */}
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">System Health</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Memory Usage:</span>
                        <span className="font-medium">{agent.performance.memoryUsage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full ${
                            agent.performance.memoryUsage > 80 ? 'bg-red-500' :
                            agent.performance.memoryUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${agent.performance.memoryUsage}%` }}
                        ></div>
                      </div>
                      
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Uptime:</span>
                        <span className="font-medium">{Math.floor(agent.performance.uptime / 3600)}h</span>
                      </div>
                    </div>
                  </div>

                  {/* Agent Actions */}
                  {onAgentAction && (
                    <div className="flex items-center space-x-2 pt-2">
                      {agent.status === 'busy' && (
                        <button
                          onClick={() => onAgentAction(agent.id, 'pause')}
                          className="flex items-center space-x-1 px-3 py-1 text-sm text-yellow-600 hover:text-yellow-700 hover:bg-yellow-50 rounded-lg transition-colors"
                        >
                          <Pause className="w-3 h-3" />
                          <span>Pause</span>
                        </button>
                      )}
                      
                      {agent.status === 'idle' && (
                        <button
                          onClick={() => onAgentAction(agent.id, 'resume')}
                          className="flex items-center space-x-1 px-3 py-1 text-sm text-green-600 hover:text-green-700 hover:bg-green-50 rounded-lg transition-colors"
                        >
                          <Play className="w-3 h-3" />
                          <span>Activate</span>
                        </button>
                      )}
                      
                      {agent.status === 'error' && (
                        <button
                          onClick={() => onAgentAction(agent.id, 'restart')}
                          className="flex items-center space-x-1 px-3 py-1 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                        >
                          <ArrowUpRight className="w-3 h-3" />
                          <span>Restart</span>
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {filteredAgents.length === 0 && (
          <div className="p-8 text-center">
            <Users className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">No agents found matching the current filter.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AgentStatusPanel;