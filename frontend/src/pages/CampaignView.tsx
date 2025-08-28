import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useCampaignPlan } from '../hooks/useCampaignPlan';
import { CampaignDetails } from '../components/CampaignDetails';
import { RepurposeModal } from '../components/RepurposeModal';
import type { CampaignDetail } from '../lib/api';

// CampaignView is the main page for managing a campaign workflow for a blog post
function CampaignView() {
  // Get the blogId or campaignId from the route params
  const { blogId, campaignId } = useParams<{ blogId?: string; campaignId?: string }>();
  
  // Use campaignId if available, otherwise use a mock ID
  const actualCampaignId = campaignId || "test-campaign-1";
  
  // Modal states
  const [showDetails, setShowDetails] = useState(false);
  const [showRepurpose, setShowRepurpose] = useState(false);
  
  // Use the custom hook to manage campaign state
  const {
    campaign,
    isLoading,
    error,
    fetchCampaign,
    schedule,
    distribute,
    updateTaskStatus,
  } = useCampaignPlan(actualCampaignId);

  // Fetch the campaign on mount
  useEffect(() => {
    fetchCampaign();
  }, [fetchCampaign]);

  // Handler for scheduling campaign
  const handleSchedule = () => {
    schedule();
  };

  // Handler for distributing campaign
  const handleDistribute = () => {
    distribute();
  };

  // Handler for repurposed content
  const handleRepurpose = (repurposedContent: any) => {
    console.log('Repurposed content:', repurposedContent);
    // Here you would typically save the repurposed content to the campaign
    // For now, we'll just log it
  };

  if (isLoading) {
    return (
      <div className="max-w-2xl mx-auto py-8">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-sm font-medium text-red-800">Error</h3>
          <div className="mt-2 text-sm text-red-700">{error}</div>
        </div>
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="max-w-2xl mx-auto py-8">
        <div className="text-center py-12">
          <h3 className="text-lg font-medium text-gray-900">No campaign found</h3>
          <p className="mt-2 text-sm text-gray-500">No campaign was found for this blog.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">{campaign.name}</h1>
        <p className="text-gray-600">Campaign ID: {campaign.id}</p>
        <div className="mt-4">
          <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
            campaign.status === 'active' ? 'bg-green-100 text-green-800' :
            campaign.status === 'draft' ? 'bg-yellow-100 text-yellow-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {campaign.status}
          </span>
        </div>
      </div>

      {/* Strategy Section */}
      {campaign.strategy && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Strategy</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {campaign.strategy.target_audience && (
              <div>
                <h3 className="font-medium text-gray-900">Target Audience</h3>
                <p className="text-gray-600">{campaign.strategy.target_audience}</p>
              </div>
            )}
            {campaign.strategy.key_messages && (
              <div>
                <h3 className="font-medium text-gray-900">Key Messages</h3>
                <ul className="list-disc list-inside text-gray-600">
                  {campaign.strategy.key_messages.map((message: string, index: number) => (
                    <li key={index}>{message}</li>
                  ))}
                </ul>
              </div>
            )}
            {campaign.strategy.distribution_channels && (
              <div>
                <h3 className="font-medium text-gray-900">Distribution Channels</h3>
                <ul className="list-disc list-inside text-gray-600">
                  {campaign.strategy.distribution_channels.map((channel: string, index: number) => (
                    <li key={index}>{channel}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Enhanced Tasks Section */}
      {campaign.tasks && campaign.tasks.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Tasks</h2>
            <div className="text-sm text-gray-600">
              {campaign.tasks.filter((t: any) => t.status === 'completed').length} of{' '}
              {campaign.tasks.length} completed
            </div>
          </div>
          <div className="space-y-3">
            {campaign.tasks.map((task: any, index: number) => {
              const getStatusColor = (status: string) => {
                switch (status) {
                  case 'completed': return 'bg-green-100 text-green-800 border-green-200';
                  case 'in_progress': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
                  default: return 'bg-gray-100 text-gray-800 border-gray-200';
                }
              };

              const getTaskIcon = (taskType: string) => {
                switch (taskType) {
                  case 'content_creation': return '✏️';
                  case 'distribution': return '📤';
                  case 'analytics': return '📊';
                  default: return '📋';
                }
              };

              const currentStatus = task.status || 'pending';
              const nextStatus = currentStatus === 'pending' ? 'in_progress' : 
                               currentStatus === 'in_progress' ? 'completed' : 'pending';

              return (
                <div 
                  key={index} 
                  className={`border-2 rounded-lg p-4 transition-all duration-200 hover:shadow-md cursor-pointer ${
                    currentStatus === 'completed' ? 'bg-green-50 border-green-200' :
                    currentStatus === 'in_progress' ? 'bg-yellow-50 border-yellow-200' :
                    'bg-gray-50 border-gray-200 hover:border-blue-300'
                  }`}
                  onClick={() => {
                    console.log(`Updating task ${task.id} to ${nextStatus}`);
                    updateTaskStatus(task.id, nextStatus);
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-lg">{getTaskIcon(task.task_type || task.type)}</span>
                        <h3 className="font-semibold text-gray-900 capitalize">
                          {(task.task_type || task.type || 'Unknown').replace('_', ' ')}
                        </h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(task.status)}`}>
                          {(task.status || 'Unknown').replace('_', ' ')}
                        </span>
                      </div>
                      
                      <p className="text-sm text-gray-600 mb-2">{task.content || task.title || 'No description available'}</p>
                      
                      {/* Task Metadata */}
                      {task.metadata && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-gray-500">
                          {task.metadata.platform && (
                            <div>
                              <span className="font-medium">Platform:</span> {task.metadata.platform}
                            </div>
                          )}
                          {task.metadata.estimated_duration_hours && (
                            <div>
                              <span className="font-medium">Duration:</span> {task.metadata.estimated_duration_hours}h
                            </div>
                          )}
                          {task.metadata.assigned_agent && (
                            <div>
                              <span className="font-medium">Agent:</span> {task.metadata.assigned_agent}
                            </div>
                          )}
                          {task.metadata.priority && (
                            <div>
                              <span className="font-medium">Priority:</span> 
                              <span className={`ml-1 capitalize ${
                                task.metadata.priority === 'high' ? 'text-red-600' :
                                task.metadata.priority === 'medium' ? 'text-yellow-600' :
                                'text-gray-600'
                              }`}>
                                {task.metadata.priority}
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Dependencies */}
                      {task.metadata?.dependencies && task.metadata.dependencies.length > 0 && (
                        <div className="mt-2 text-xs text-gray-500">
                          <span className="font-medium">Dependencies:</span> {task.metadata.dependencies.join(', ')}
                        </div>
                      )}
                    </div>
                    
                    {/* Action hint */}
                    <div className="ml-4 text-xs text-gray-400">
                      Click to {nextStatus === 'in_progress' ? 'start' : 
                               nextStatus === 'completed' ? 'complete' : 'reset'}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Progress Bar */}
          <div className="mt-6 pt-4 border-t">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">Overall Progress</span>
              <span className="text-sm text-gray-600">
                {Math.round((campaign.tasks.filter((t: any) => t.status === 'completed').length / campaign.tasks.length) * 100)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-500"
                style={{ 
                  width: `${(campaign.tasks.filter((t: any) => t.status === 'completed').length / campaign.tasks.length) * 100}%` 
                }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Section */}
      {campaign.performance && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Performance</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h3 className="font-medium text-gray-900">Total Posts</h3>
              <p className="text-2xl font-bold text-blue-600">{campaign.performance.total_posts || 0}</p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Published Posts</h3>
              <p className="text-2xl font-bold text-green-600">{campaign.performance.published_posts || 0}</p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Success Rate</h3>
              <p className="text-2xl font-bold text-purple-600">{campaign.performance.success_rate || 0}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <button
            onClick={() => setShowDetails(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
          >
            View Details
          </button>
          <button
            onClick={() => setShowRepurpose(true)}
            className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 transition-colors"
          >
            Repurpose
          </button>
          <button
            onClick={handleSchedule}
            disabled={isLoading}
            className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            {isLoading ? 'Scheduling...' : 'Schedule'}
          </button>
          <button
            onClick={handleDistribute}
            disabled={isLoading}
            className="bg-orange-600 text-white px-4 py-2 rounded-md hover:bg-orange-700 disabled:opacity-50 transition-colors"
          >
            {isLoading ? 'Distributing...' : 'Distribute'}
          </button>
        </div>
      </div>

      {/* Modals */}
      {showDetails && campaign && (
        <CampaignDetails
          campaign={campaign}
          onClose={() => setShowDetails(false)}
        />
      )}

      {showRepurpose && campaign && (
        <RepurposeModal
          campaign={campaign}
          onClose={() => setShowRepurpose(false)}
          onRepurpose={handleRepurpose}
        />
      )}
    </div>
  );
}

export { CampaignView };
export default CampaignView;