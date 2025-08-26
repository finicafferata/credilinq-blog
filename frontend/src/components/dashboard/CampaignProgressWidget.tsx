import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Activity,
  Users,
  ArrowRight,
  MoreHorizontal,
  TrendingUp,
  Calendar
} from 'lucide-react';

interface WorkflowStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  agentId?: string;
  agentName?: string;
  duration?: number;
  startTime?: string;
  endTime?: string;
  progress?: number;
  output?: any;
  dependencies?: string[];
}

interface CampaignProgressProps {
  campaignId: string;
  campaignName: string;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  currentStep?: string;
  workflow: WorkflowStep[];
  estimatedCompletion?: string;
  onViewDetails?: () => void;
  onControlCampaign?: (action: 'start' | 'pause' | 'resume' | 'stop') => void;
}

export function CampaignProgressWidget({
  campaignId,
  campaignName,
  status,
  progress,
  currentStep,
  workflow,
  estimatedCompletion,
  onViewDetails,
  onControlCampaign
}: CampaignProgressProps) {
  const [expandedView, setExpandedView] = useState(false);
  const [realtimeProgress, setRealtimeProgress] = useState(progress);

  // Simulate real-time progress updates for running campaigns
  useEffect(() => {
    if (status === 'running') {
      const interval = setInterval(() => {
        setRealtimeProgress(prev => {
          const increment = Math.random() * 2;
          return Math.min(100, prev + increment);
        });
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [status]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-600 bg-green-100';
      case 'paused': return 'text-yellow-600 bg-yellow-100';
      case 'completed': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'running': return <Activity className="w-4 h-4 text-blue-600 animate-pulse" />;
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-600" />;
      case 'pending': return <Clock className="w-4 h-4 text-gray-400" />;
      default: return <div className="w-4 h-4 rounded-full border-2 border-gray-300"></div>;
    }
  };

  const getControlIcon = () => {
    switch (status) {
      case 'running': return <Pause className="w-4 h-4" />;
      case 'paused': return <Play className="w-4 h-4" />;
      default: return <Play className="w-4 h-4" />;
    }
  };

  const getControlAction = () => {
    switch (status) {
      case 'running': return 'pause';
      case 'paused': return 'resume';
      case 'draft': return 'start';
      default: return 'start';
    }
  };

  const completedSteps = workflow.filter(step => step.status === 'completed').length;
  const runningSteps = workflow.filter(step => step.status === 'running').length;
  const failedSteps = workflow.filter(step => step.status === 'failed').length;

  const estimatedTimeRemaining = estimatedCompletion ? 
    new Date(estimatedCompletion).getTime() - new Date().getTime() : null;

  const formatTimeRemaining = (ms: number) => {
    const hours = Math.floor(ms / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h`;
    return '<1h';
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow duration-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <h3 className="font-semibold text-gray-900 text-lg">{campaignName}</h3>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(status)}`}>
                {status}
              </span>
            </div>
            
            {currentStep && (
              <p className="text-sm text-blue-600 mb-3">
                📍 {currentStep}
              </p>
            )}

            {/* Progress Bar */}
            <div className="flex items-center space-x-4">
              <div className="flex-1 bg-gray-200 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-1000 ease-out relative overflow-hidden"
                  style={{ width: `${realtimeProgress}%` }}
                >
                  {status === 'running' && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
                  )}
                </div>
              </div>
              <span className="text-sm font-medium text-gray-900 min-w-[3rem]">
                {realtimeProgress.toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2 ml-4">
            {(status === 'running' || status === 'paused') && onControlCampaign && (
              <button
                onClick={() => onControlCampaign(getControlAction() as any)}
                className={`p-2 rounded-lg transition-colors ${
                  status === 'running' 
                    ? 'text-yellow-600 hover:text-yellow-700 hover:bg-yellow-50' 
                    : 'text-green-600 hover:text-green-700 hover:bg-green-50'
                }`}
                title={getControlAction()}
              >
                {getControlIcon()}
              </button>
            )}
            
            {status === 'running' && onControlCampaign && (
              <button
                onClick={() => onControlCampaign('stop')}
                className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                title="Stop campaign"
              >
                <Square className="w-4 h-4" />
              </button>
            )}

            <button
              onClick={() => setExpandedView(!expandedView)}
              className="p-2 text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="Toggle details"
            >
              <MoreHorizontal className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{completedSteps}</p>
            <p className="text-xs text-gray-600">Completed</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">{runningSteps}</p>
            <p className="text-xs text-gray-600">Active</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-600">{workflow.length}</p>
            <p className="text-xs text-gray-600">Total Steps</p>
          </div>
        </div>

        {/* Time Estimate */}
        {estimatedTimeRemaining && estimatedTimeRemaining > 0 && (
          <div className="mt-4 flex items-center space-x-2 text-sm text-gray-600">
            <Calendar className="w-4 h-4" />
            <span>Est. {formatTimeRemaining(estimatedTimeRemaining)} remaining</span>
          </div>
        )}
      </div>

      {/* Expanded Workflow Details */}
      {expandedView && (
        <div className="p-6 bg-gray-50">
          <h4 className="font-medium text-gray-900 mb-4 flex items-center space-x-2">
            <Activity className="w-4 h-4" />
            <span>Workflow Progress</span>
          </h4>

          <div className="space-y-3">
            {workflow.map((step, index) => (
              <div
                key={step.id}
                className={`flex items-center space-x-4 p-3 rounded-lg transition-all duration-200 ${
                  step.status === 'running' 
                    ? 'bg-blue-50 border border-blue-200' 
                    : 'bg-white border border-gray-200'
                }`}
              >
                <div className="flex-shrink-0">
                  {getStepStatusIcon(step.status)}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className={`text-sm font-medium truncate ${
                      step.status === 'completed' ? 'text-gray-600' : 'text-gray-900'
                    }`}>
                      {step.name}
                    </p>
                    
                    {step.agentName && (
                      <div className="flex items-center space-x-1 text-xs text-gray-500">
                        <Users className="w-3 h-3" />
                        <span>{step.agentName}</span>
                      </div>
                    )}
                  </div>

                  {step.status === 'running' && step.progress !== undefined && (
                    <div className="mt-2">
                      <div className="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Progress</span>
                        <span>{step.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${step.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {step.duration && step.status === 'completed' && (
                    <p className="text-xs text-gray-500 mt-1">
                      ⏱️ Completed in {Math.round(step.duration / 1000)}s
                    </p>
                  )}

                  {step.status === 'failed' && (
                    <p className="text-xs text-red-600 mt-1">
                      ❌ Step failed - check logs for details
                    </p>
                  )}
                </div>

                {index < workflow.length - 1 && (
                  <div className="flex-shrink-0">
                    <ArrowRight className="w-4 h-4 text-gray-400" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Workflow Actions */}
          <div className="mt-6 flex items-center justify-between">
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              {failedSteps > 0 && (
                <div className="flex items-center space-x-1 text-red-600">
                  <AlertCircle className="w-4 h-4" />
                  <span>{failedSteps} failed</span>
                </div>
              )}
              
              <div className="flex items-center space-x-1">
                <TrendingUp className="w-4 h-4" />
                <span>{((completedSteps / workflow.length) * 100).toFixed(0)}% complete</span>
              </div>
            </div>

            {onViewDetails && (
              <button
                onClick={onViewDetails}
                className="text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                View Full Details →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default CampaignProgressWidget;