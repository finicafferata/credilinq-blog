import React, { useState, useEffect } from 'react';
import {
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  PlayIcon,
  PauseIcon,
  UserIcon,
  ComputerDesktopIcon
} from '@heroicons/react/24/outline';
import { reviewWorkflowApi, ReviewWorkflowStatus, ReviewWorkflowUtils } from '../services/reviewWorkflowApi';

interface ReviewWorkflowProgressProps {
  workflowId: string;
  onStatusChange?: (status: ReviewWorkflowStatus) => void;
  showDetails?: boolean;
  refreshInterval?: number; // milliseconds
}

const ReviewWorkflowProgress: React.FC<ReviewWorkflowProgressProps> = ({
  workflowId,
  onStatusChange,
  showDetails = true,
  refreshInterval = 30000 // 30 seconds
}) => {
  const [status, setStatus] = useState<ReviewWorkflowStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // All 8 stages in order
  const allStages = [
    { key: 'content_quality', name: 'Content Quality', icon: '📝' },
    { key: 'editorial_review', name: 'Editorial Review', icon: '✏️' },
    { key: 'brand_check', name: 'Brand Check', icon: '🏢' },
    { key: 'seo_analysis', name: 'SEO Analysis', icon: '🔍' },
    { key: 'geo_analysis', name: 'GEO Analysis', icon: '🌍' },
    { key: 'visual_review', name: 'Visual Review', icon: '🎨' },
    { key: 'social_media_review', name: 'Social Media Review', icon: '📱' },
    { key: 'final_approval', name: 'Final Approval', icon: '✅' }
  ];

  useEffect(() => {
    loadWorkflowStatus();

    const interval = setInterval(() => {
      if (!status?.is_paused && status?.workflow_status !== 'completed') {
        loadWorkflowStatus();
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [workflowId, refreshInterval]);

  const loadWorkflowStatus = async () => {
    try {
      setError(null);
      const workflowStatus = await reviewWorkflowApi.getWorkflowStatus(workflowId);
      setStatus(workflowStatus);
      onStatusChange?.(workflowStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workflow status');
      console.error('Error loading workflow status:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStageStatus = (stageKey: string) => {
    if (!status) return 'pending';
    
    if (status.completed_stages.includes(stageKey)) return 'completed';
    if (status.failed_stages.includes(stageKey)) return 'failed';
    if (status.pending_human_reviews.includes(stageKey)) return 'requires_human_review';
    if (status.current_stage === stageKey) return 'in_progress';
    
    return 'pending';
  };

  const getStageIcon = (stageKey: string, stageStatus: string) => {
    switch (stageStatus) {
      case 'completed':
        return <CheckCircleIcon className="h-6 w-6 text-green-600" />;
      case 'failed':
        return <ExclamationTriangleIcon className="h-6 w-6 text-red-600" />;
      case 'requires_human_review':
        return <UserIcon className="h-6 w-6 text-yellow-600" />;
      case 'in_progress':
        return <div className="h-6 w-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />;
      default:
        return <ClockIcon className="h-6 w-6 text-gray-400" />;
    }
  };

  const getStageStyle = (stageKey: string, stageStatus: string) => {
    const baseStyle = "flex items-center justify-between p-4 rounded-lg border-2 transition-all duration-200";
    
    switch (stageStatus) {
      case 'completed':
        return `${baseStyle} border-green-200 bg-green-50`;
      case 'failed':
        return `${baseStyle} border-red-200 bg-red-50`;
      case 'requires_human_review':
        return `${baseStyle} border-yellow-200 bg-yellow-50 shadow-lg`;
      case 'in_progress':
        return `${baseStyle} border-blue-200 bg-blue-50 shadow-lg`;
      default:
        return `${baseStyle} border-gray-200 bg-white`;
    }
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-8 w-8 bg-gray-200 rounded animate-pulse" />
          <div>
            <div className="h-6 w-48 bg-gray-200 rounded animate-pulse mb-2" />
            <div className="h-4 w-32 bg-gray-200 rounded animate-pulse" />
          </div>
        </div>
        {[...Array(8)].map((_, i) => (
          <div key={i} className="h-16 bg-gray-100 rounded-lg animate-pulse" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center gap-3 text-red-800">
          <ExclamationTriangleIcon className="h-6 w-6" />
          <div>
            <h3 className="font-semibold">Failed to Load Workflow</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
        <button
          onClick={loadWorkflowStatus}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!status) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            {status.is_paused ? (
              <PauseIcon className="h-8 w-8 text-yellow-600" />
            ) : status.workflow_status === 'completed' ? (
              <CheckCircleIcon className="h-8 w-8 text-green-600" />
            ) : (
              <PlayIcon className="h-8 w-8 text-blue-600" />
            )}
            <div>
              <h3 className="text-xl font-semibold text-gray-900">
                Review Workflow Progress
              </h3>
              <p className="text-sm text-gray-600">
                {status.is_paused ? 'Paused' : status.workflow_status === 'completed' ? 'Completed' : 'In Progress'}
                {status.current_stage && ` • Current: ${ReviewWorkflowUtils.getStageDisplayName(status.current_stage)}`}
              </p>
            </div>
          </div>
        </div>
        
        {/* Progress Bar */}
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-2xl font-bold text-gray-900">
              {status.overall_progress}%
            </div>
            <div className="text-sm text-gray-600">
              {status.completed_stages.length}/8 stages
            </div>
          </div>
          <div className="w-32 bg-gray-200 rounded-full h-3">
            <div 
              className="bg-blue-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${status.overall_progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Workflow Status Banner */}
      {status.is_paused && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <PauseIcon className="h-5 w-5 text-yellow-600" />
            <div>
              <h4 className="font-medium text-yellow-800">Workflow Paused</h4>
              <p className="text-sm text-yellow-700">
                Waiting for human review on {status.pending_human_reviews.length} stage(s)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stage Progress */}
      <div className="space-y-3">
        {allStages.map((stage, index) => {
          const stageStatus = getStageStatus(stage.key);
          const isActive = status.current_stage === stage.key;
          
          return (
            <div key={stage.key} className={getStageStyle(stage.key, stageStatus)}>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-3">
                  <div className="text-sm font-medium text-gray-500 w-8 text-center">
                    {index + 1}
                  </div>
                  <div className="text-2xl">{stage.icon}</div>
                  <div>
                    <h4 className="font-medium text-gray-900">{stage.name}</h4>
                    {showDetails && (
                      <p className="text-sm text-gray-600 capitalize">
                        {stageStatus.replace(/_/g, ' ')}
                        {isActive && ' (Current)'}
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                {stageStatus === 'requires_human_review' && (
                  <div className="flex items-center gap-1 text-sm text-yellow-700 bg-yellow-100 px-2 py-1 rounded">
                    <UserIcon className="h-4 w-4" />
                    Human Review
                  </div>
                )}
                {stageStatus === 'in_progress' && (
                  <div className="flex items-center gap-1 text-sm text-blue-700 bg-blue-100 px-2 py-1 rounded">
                    <ComputerDesktopIcon className="h-4 w-4" />
                    AI Processing
                  </div>
                )}
                {getStageIcon(stage.key, stageStatus)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Workflow Details */}
      {showDetails && (
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Started:</span>
              <div className="font-medium">
                {new Date(status.started_at).toLocaleDateString()}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Status:</span>
              <div className={`font-medium capitalize ${ReviewWorkflowUtils.getStatusColor(status.workflow_status)}`}>
                {status.workflow_status.replace(/_/g, ' ')}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Content ID:</span>
              <div className="font-medium text-xs">{status.content_id}</div>
            </div>
            <div>
              <span className="text-gray-600">Workflow ID:</span>
              <div className="font-medium text-xs">{status.workflow_execution_id}</div>
            </div>
          </div>
        </div>
      )}

      {/* Failed Stages Alert */}
      {status.failed_stages.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-600" />
            <div>
              <h4 className="font-medium text-red-800">Failed Stages</h4>
              <p className="text-sm text-red-700">
                {status.failed_stages.map(stage => ReviewWorkflowUtils.getStageDisplayName(stage)).join(', ')}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReviewWorkflowProgress;