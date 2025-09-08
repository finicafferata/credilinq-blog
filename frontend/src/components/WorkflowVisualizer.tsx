import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  CheckCircle,
  AlertCircle,
  Clock,
  ArrowRight,
  Zap,
  RefreshCw,
  Timer,
  Activity
} from 'lucide-react';

interface AgentNode {
  id: string;
  name: string;
  type: string;
  status: 'waiting' | 'running' | 'completed' | 'failed';
  executionTime?: number;
  estimatedTime?: number;
  dependencies: string[];
  parallelGroupId?: number;
  output?: any;
}

interface WorkflowVisualizerProps {
  workflowId: string;
  agents: AgentNode[];
  status: string;
  progressPercentage: number;
  onAgentClick?: (agentId: string) => void;
  className?: string;
}

const WorkflowVisualizer: React.FC<WorkflowVisualizerProps> = ({
  workflowId,
  agents,
  status,
  progressPercentage,
  onAgentClick,
  className = ""
}) => {
  const [animationKey, setAnimationKey] = useState(0);

  // Trigger animation when status changes
  useEffect(() => {
    setAnimationKey(prev => prev + 1);
  }, [agents]);

  const getStatusIcon = (status: string, isRunning: boolean = false) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      case 'running':
        return <RefreshCw className={`w-5 h-5 text-blue-600 ${isRunning ? 'animate-spin' : ''}`} />;
      case 'waiting':
        return <Clock className="w-5 h-5 text-gray-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 border-green-300 text-green-800';
      case 'failed':
        return 'bg-red-100 border-red-300 text-red-800';
      case 'running':
        return 'bg-blue-100 border-blue-300 text-blue-800 animate-pulse';
      case 'waiting':
        return 'bg-gray-100 border-gray-300 text-gray-600';
      default:
        return 'bg-gray-100 border-gray-300 text-gray-600';
    }
  };

  const getProgressWidth = (agent: AgentNode) => {
    if (agent.status === 'completed') return 100;
    if (agent.status === 'failed') return 100;
    if (agent.status === 'running' && agent.executionTime && agent.estimatedTime) {
      return Math.min((agent.executionTime / agent.estimatedTime) * 100, 95);
    }
    return 0;
  };

  // Group agents by parallel groups
  const groupedAgents = agents.reduce((groups, agent) => {
    const groupId = agent.parallelGroupId || `sequential_${agent.id}`;
    if (!groups[groupId]) {
      groups[groupId] = [];
    }
    groups[groupId].push(agent);
    return groups;
  }, {} as Record<string, AgentNode[]>);

  const renderAgentNode = (agent: AgentNode, index: number) => (
    <div
      key={`${agent.id}_${animationKey}`}
      className={`relative rounded-lg border-2 p-4 transition-all duration-500 cursor-pointer hover:scale-105 hover:shadow-lg ${getStatusColor(agent.status)}`}
      onClick={() => onAgentClick?.(agent.id)}
      style={{
        animationDelay: `${index * 100}ms`
      }}
    >
      {/* Agent Status Indicator */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getStatusIcon(agent.status, agent.status === 'running')}
          <span className="font-semibold text-sm capitalize">
            {agent.name}
          </span>
        </div>
        {agent.status === 'running' && (
          <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
        )}
      </div>

      {/* Progress Bar for Running Agents */}
      {agent.status === 'running' && (
        <div className="mb-2">
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className="bg-blue-600 h-1.5 rounded-full transition-all duration-1000"
              style={{ width: `${getProgressWidth(agent)}%` }}
            />
          </div>
        </div>
      )}

      {/* Execution Time */}
      <div className="text-xs text-gray-600 flex items-center space-x-1">
        <Timer className="w-3 h-3" />
        {agent.status === 'completed' && agent.executionTime && (
          <span>{agent.executionTime.toFixed(1)}s</span>
        )}
        {agent.status === 'running' && agent.executionTime && agent.estimatedTime && (
          <span>{agent.executionTime.toFixed(0)}s / {agent.estimatedTime}s</span>
        )}
        {agent.status === 'waiting' && agent.estimatedTime && (
          <span>~{agent.estimatedTime}s</span>
        )}
        {agent.status === 'failed' && (
          <span className="text-red-600">Failed</span>
        )}
      </div>

      {/* Output Indicator */}
      {agent.output && agent.status === 'completed' && (
        <div className="mt-1">
          <div className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded">
            ✓ Output Generated
          </div>
        </div>
      )}

      {/* Pulse animation for running agents */}
      {agent.status === 'running' && (
        <div className="absolute inset-0 rounded-lg border-2 border-blue-400 animate-ping opacity-25" />
      )}
    </div>
  );

  const renderArrow = (index: number) => (
    <div key={`arrow_${index}`} className="flex items-center justify-center mx-2">
      <ArrowRight className="w-5 h-5 text-gray-400" />
    </div>
  );

  // Create workflow visualization layout
  const renderWorkflowFlow = () => {
    const sequentialGroups = Object.entries(groupedAgents).filter(
      ([groupId]) => groupId.startsWith('sequential_')
    );
    const parallelGroups = Object.entries(groupedAgents).filter(
      ([groupId]) => !groupId.startsWith('sequential_')
    );

    const elements: React.ReactNode[] = [];

    // Add sequential agents
    sequentialGroups.forEach(([groupId, groupAgents], groupIndex) => {
      if (elements.length > 0) {
        elements.push(renderArrow(elements.length));
      }
      elements.push(
        <div key={groupId} className="flex-shrink-0">
          {renderAgentNode(groupAgents[0], elements.length)}
        </div>
      );
    });

    // Add parallel groups
    parallelGroups.forEach(([groupId, groupAgents], groupIndex) => {
      if (elements.length > 0) {
        elements.push(renderArrow(elements.length));
      }
      
      elements.push(
        <div key={groupId} className="flex-shrink-0">
          <div className="text-xs text-gray-500 mb-2 text-center">
            Parallel Group {groupId}
          </div>
          <div className="grid grid-cols-1 gap-2">
            {groupAgents.map((agent, agentIndex) => (
              <div key={agent.id}>
                {renderAgentNode(agent, elements.length + agentIndex)}
              </div>
            ))}
          </div>
        </div>
      );
    });

    return elements;
  };

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}>
      {/* Workflow Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-gray-900">
            Workflow Execution
          </h3>
          <div className="flex items-center space-x-2">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              status === 'running' ? 'bg-blue-100 text-blue-800' :
              status === 'completed' ? 'bg-green-100 text-green-800' :
              status === 'failed' ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {status}
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
          <div
            className={`h-3 rounded-full transition-all duration-1000 ${
              status === 'failed' ? 'bg-red-500' :
              status === 'completed' ? 'bg-green-500' :
              'bg-blue-500'
            }`}
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
        <div className="text-sm text-gray-600 text-center">
          {progressPercentage.toFixed(1)}% Complete
        </div>
      </div>

      {/* Agent Flow Visualization */}
      <div className="overflow-x-auto">
        <div className="flex items-center space-x-4 min-w-max pb-4">
          {renderWorkflowFlow()}
        </div>
      </div>

      {/* Workflow Statistics */}
      <div className="mt-6 grid grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {agents.filter(a => a.status === 'completed').length}
          </div>
          <div className="text-xs text-gray-600">Completed</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {agents.filter(a => a.status === 'running').length}
          </div>
          <div className="text-xs text-gray-600">Running</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-600">
            {agents.filter(a => a.status === 'waiting').length}
          </div>
          <div className="text-xs text-gray-600">Waiting</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">
            {agents.filter(a => a.status === 'failed').length}
          </div>
          <div className="text-xs text-gray-600">Failed</div>
        </div>
      </div>
    </div>
  );
};

export default WorkflowVisualizer;