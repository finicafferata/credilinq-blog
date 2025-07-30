import React from 'react';
import type { CampaignDetail } from '../lib/api';

interface CampaignDetailsProps {
  campaign: CampaignDetail;
  onClose: () => void;
}

export function CampaignDetails({ campaign, onClose }: CampaignDetailsProps) {
  const formatDate = (dateString: string) => {
    if (!dateString) return "No date";
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (error) {
      return "Invalid date";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'draft':
        return 'bg-yellow-100 text-yellow-800';
      case 'completed':
        return 'bg-blue-100 text-blue-800';
      case 'paused':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{campaign.name}</h2>
            <p className="text-gray-600">Campaign ID: {campaign.id}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Status and Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">Status</h3>
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(campaign.status)}`}>
                {campaign.status}
              </span>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">Total Tasks</h3>
              <p className="text-2xl font-bold text-blue-600">{campaign.tasks?.length || 0}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">Scheduled Posts</h3>
              <p className="text-2xl font-bold text-green-600">{campaign.scheduled_posts?.length || 0}</p>
            </div>
          </div>

          {/* Strategy Section */}
          {campaign.strategy && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Strategy</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {campaign.strategy.target_audience && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Target Audience</h4>
                    <p className="text-gray-600">{campaign.strategy.target_audience}</p>
                  </div>
                )}
                {campaign.strategy.key_messages && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Key Messages</h4>
                    <ul className="list-disc list-inside text-gray-600 space-y-1">
                      {campaign.strategy.key_messages.map((message: string, index: number) => (
                        <li key={index}>{message}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {campaign.strategy.distribution_channels && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Distribution Channels</h4>
                    <div className="flex flex-wrap gap-2">
                      {campaign.strategy.distribution_channels.map((channel: string, index: number) => (
                        <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm">
                          {channel}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tasks Section */}
          {campaign.tasks && campaign.tasks.length > 0 && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Tasks</h3>
              <div className="space-y-3">
                {campaign.tasks.map((task: any, index: number) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900">{task.task_type}</h4>
                      <p className="text-sm text-gray-600">{task.content}</p>
                      {task.created_at && (
                        <p className="text-xs text-gray-500 mt-1">
                          Created: {formatDate(task.created_at)}
                        </p>
                      )}
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      task.status === 'completed' ? 'bg-green-100 text-green-800' :
                      task.status === 'in_progress' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {task.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scheduled Posts Section */}
          {campaign.scheduled_posts && campaign.scheduled_posts.length > 0 && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Scheduled Posts</h3>
              <div className="space-y-3">
                {campaign.scheduled_posts.map((post: any, index: number) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900">{post.platform}</h4>
                      <p className="text-sm text-gray-600 line-clamp-2">{post.content}</p>
                      {post.scheduled_at && (
                        <p className="text-xs text-gray-500 mt-1">
                          Scheduled: {formatDate(post.scheduled_at)}
                        </p>
                      )}
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      post.status === 'published' ? 'bg-green-100 text-green-800' :
                      post.status === 'scheduled' ? 'bg-blue-100 text-blue-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {post.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Performance Section */}
          {campaign.performance && (
            <div className="bg-white border rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <h4 className="font-medium text-gray-900">Total Posts</h4>
                  <p className="text-2xl font-bold text-blue-600">{campaign.performance.total_posts || 0}</p>
                </div>
                <div className="text-center">
                  <h4 className="font-medium text-gray-900">Published</h4>
                  <p className="text-2xl font-bold text-green-600">{campaign.performance.published_posts || 0}</p>
                </div>
                <div className="text-center">
                  <h4 className="font-medium text-gray-900">Success Rate</h4>
                  <p className="text-2xl font-bold text-purple-600">{campaign.performance.success_rate || 0}%</p>
                </div>
                <div className="text-center">
                  <h4 className="font-medium text-gray-900">Engagement</h4>
                  <p className="text-2xl font-bold text-orange-600">{campaign.performance.engagement || 0}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 