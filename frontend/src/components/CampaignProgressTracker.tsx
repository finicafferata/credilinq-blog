import React, { useEffect, useState } from 'react';
import { useCampaignWebSocket } from '../hooks/useCampaignWebSocket';
import type { CampaignStatusMessage } from '../hooks/useCampaignWebSocket';
import { 
  PlayCircleIcon, 
  ClockIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  CogIcon,
  DocumentTextIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

interface CampaignProgressTrackerProps {
  campaignId: string;
  campaignName?: string;
  onProgressComplete?: (contentCreated: any) => void;
}

const CampaignProgressTracker: React.FC<CampaignProgressTrackerProps> = ({
  campaignId,
  campaignName = 'Campaign',
  onProgressComplete
}) => {
  const { 
    isConnected, 
    lastMessage, 
    messages, 
    connectionState, 
    progress, 
    currentStatus
  } = useCampaignWebSocket(campaignId);

  const [stages, setStages] = useState([
    { id: 'workflow_started', name: 'Initializing Workflow', status: 'pending', icon: PlayCircleIcon },
    { id: 'agents_starting', name: 'AI Agents Processing', status: 'pending', icon: CogIcon },
    { id: 'workflow_completed', name: 'Content Generation', status: 'pending', icon: DocumentTextIcon },
    { id: 'campaign_completed', name: 'Ready for Review', status: 'pending', icon: CheckCircleIcon }
  ]);

  const [startTime, setStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState('0:00');

  // Update stages based on received messages
  useEffect(() => {
    if (!lastMessage) return;

    setStages(prev => prev.map(stage => {
      if (lastMessage.type === stage.id) {
        return {
          ...stage,
          status: lastMessage.status === 'failed' ? 'failed' : 'completed'
        };
      }
      return stage;
    }));

    // Set start time when workflow begins
    if (lastMessage.type === 'workflow_started' && !startTime) {
      setStartTime(new Date());
    }

    // Handle completion
    if (lastMessage.type === 'campaign_completed' && onProgressComplete && lastMessage.content_created) {
      onProgressComplete(lastMessage.content_created);
    }
  }, [lastMessage, startTime, onProgressComplete]);

  // Update elapsed time
  useEffect(() => {
    if (!startTime) return;

    const interval = setInterval(() => {
      const now = new Date();
      const elapsed = Math.floor((now.getTime() - startTime.getTime()) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      setElapsedTime(`${minutes}:${seconds.toString().padStart(2, '0')}`);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  const getProgressColor = () => {
    if (connectionState === 'error') return 'bg-red-500';
    if (progress === 100) return 'bg-green-500';
    if (progress > 0) return 'bg-blue-500';
    return 'bg-gray-300';
  };

  const getConnectionStatus = () => {
    switch (connectionState) {
      case 'connecting': return { text: 'Connecting...', color: 'text-yellow-600' };
      case 'connected': return { text: 'Connected', color: 'text-green-600' };
      case 'disconnected': return { text: 'Disconnected', color: 'text-gray-600' };
      case 'error': return { text: 'Connection Error', color: 'text-red-600' };
      default: return { text: 'Unknown', color: 'text-gray-600' };
    }
  };

  const connectionStatus = getConnectionStatus();

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <SparklesIcon className="w-5 h-5 text-blue-500" />
            Content Generation Progress
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            {campaignName} • {elapsedTime} elapsed
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-gray-400'}`} />
          <span className={`text-sm ${connectionStatus.color}`}>
            {connectionStatus.text}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Overall Progress</span>
          <span className="text-sm text-gray-600">{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className={`${getProgressColor()} h-3 rounded-full transition-all duration-500 ease-out`}
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-sm text-gray-600 mt-2">{currentStatus}</p>
      </div>

      {/* Stage Indicators */}
      <div className="space-y-3">
        {stages.map((stage, index) => {
          const Icon = stage.icon;
          const isActive = stages.findIndex(s => s.status === 'pending') === index;
          const isCompleted = stage.status === 'completed';
          const isFailed = stage.status === 'failed';

          return (
            <div
              key={stage.id}
              className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
                isActive ? 'bg-blue-50 border border-blue-200' :
                isCompleted ? 'bg-green-50 border border-green-200' :
                isFailed ? 'bg-red-50 border border-red-200' :
                'bg-gray-50'
              }`}
            >
              <div className={`p-2 rounded-full ${
                isActive ? 'bg-blue-100 text-blue-600' :
                isCompleted ? 'bg-green-100 text-green-600' :
                isFailed ? 'bg-red-100 text-red-600' :
                'bg-gray-100 text-gray-400'
              }`}>
                {isCompleted ? (
                  <CheckCircleIcon className="w-5 h-5" />
                ) : isFailed ? (
                  <ExclamationTriangleIcon className="w-5 h-5" />
                ) : isActive ? (
                  <Icon className="w-5 h-5 animate-pulse" />
                ) : (
                  <Icon className="w-5 h-5" />
                )}
              </div>
              <div className="flex-1">
                <h4 className={`text-sm font-medium ${
                  isCompleted ? 'text-green-800' :
                  isFailed ? 'text-red-800' :
                  isActive ? 'text-blue-800' :
                  'text-gray-600'
                }`}>
                  {stage.name}
                </h4>
                {isActive && (
                  <p className="text-xs text-blue-600 mt-1">
                    In progress...
                  </p>
                )}
                {isCompleted && (
                  <p className="text-xs text-green-600 mt-1">
                    ✓ Completed
                  </p>
                )}
                {isFailed && (
                  <p className="text-xs text-red-600 mt-1">
                    ✗ Failed
                  </p>
                )}
              </div>
              {isActive && (
                <ClockIcon className="w-4 h-4 text-blue-500 animate-spin" />
              )}
            </div>
          );
        })}
      </div>

      {/* Recent Messages (Debug/Development) */}
      {process.env.NODE_ENV === 'development' && messages.length > 0 && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          <details className="group">
            <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
              WebSocket Messages ({messages.length})
            </summary>
            <div className="mt-3 max-h-32 overflow-y-auto space-y-2">
              {messages.slice(-5).map((message, index) => (
                <div key={index} className="text-xs bg-gray-100 p-2 rounded">
                  <div className="font-medium">{message.type}</div>
                  <div className="text-gray-600 truncate">{message.message}</div>
                  <div className="text-gray-500 text-[10px]">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
};

export default CampaignProgressTracker;