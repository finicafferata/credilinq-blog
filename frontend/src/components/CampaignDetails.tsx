import React, { useEffect, useRef, useState } from 'react';
import { Play, RefreshCw, CheckCircle, XCircle, Edit3, RotateCcw, BarChart3 } from 'lucide-react';
import type { CampaignDetail } from '../lib/api';
import { campaignApi } from '../lib/api';
import { RevisionFeedbackDialog } from './RevisionFeedbackDialog';
import { ContentNarrativeViewer } from './ContentNarrativeViewer';

interface CampaignDetailsProps {
  campaign: CampaignDetail;
  onClose: () => void;
}

export function CampaignDetails({ campaign, onClose }: CampaignDetailsProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [executingTasks, setExecutingTasks] = useState<Set<string>>(new Set());
  const [campaignTasks, setCampaignTasks] = useState(campaign.tasks || []);
  const [executingAll, setExecutingAll] = useState(false);
  const [schedulingContent, setSchedulingContent] = useState(false);
  const [scheduledContent, setScheduledContent] = useState<any[]>([]);
  const [revisionDialog, setRevisionDialog] = useState<{
    isOpen: boolean;
    taskId: string;
    taskType: string;
    currentContent: string;
  }>({
    isOpen: false,
    taskId: '',
    taskType: '',
    currentContent: ''
  });
  const [submittingRevision, setSubmittingRevision] = useState(false);
  const [regeneratingTasks, setRegeneratingTasks] = useState<Set<string>>(new Set());
  const [feedbackAnalytics, setFeedbackAnalytics] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'deliverables' | 'tasks'>('tasks');

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  // Execute individual task
  const executeTask = async (taskId: string) => {
    try {
      setExecutingTasks(prev => new Set([...prev, taskId]));
      
      const result = await campaignApi.executeTask(campaign.id, taskId);
      
      // Update task status in local state
      setCampaignTasks(prev => prev.map(task => 
        task.id === taskId 
          ? { ...task, status: result.status, result: result.result }
          : task
      ));
      
      console.log('Task executed successfully:', result);
    } catch (error) {
      console.error('Error executing task:', error);
    } finally {
      setExecutingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskId);
        return next;
      });
    }
  };

  // Execute all tasks
  const executeAllTasks = async () => {
    try {
      setExecutingAll(true);
      
      const result = await campaignApi.executeAllTasks(campaign.id);
      
      // Update tasks to 'generated' status (not completed - they need review)
      setCampaignTasks(prev => prev.map(task => 
        task.status === 'pending' 
          ? { ...task, status: 'generated' }
          : task
      ));
      
      console.log('All tasks executed:', result);
    } catch (error) {
      console.error('Error executing all tasks:', error);
    } finally {
      setExecutingAll(false);
    }
  };

  // Review task content
  const reviewTask = async (taskId: string, action: 'approve' | 'reject' | 'request_revision', notes?: string) => {
    try {
      const result = await campaignApi.reviewTask(campaign.id, taskId, action, notes);
      
      // Update task status in local state
      setCampaignTasks(prev => prev.map(task => 
        task.id === taskId 
          ? { ...task, status: result.new_status, quality_score: result.quality_score, review_notes: notes }
          : task
      ));
      
      console.log('Task reviewed successfully:', result);
    } catch (error) {
      console.error('Error reviewing task:', error);
    }
  };

  // Schedule approved content
  const scheduleApprovedContent = async () => {
    try {
      setSchedulingContent(true);
      const result = await campaignApi.scheduleApprovedContent(campaign.id);
      
      // Fetch updated scheduled content
      const scheduledData = await campaignApi.getScheduledContent(campaign.id);
      setScheduledContent(scheduledData);
      
      console.log('Content scheduled successfully:', result);
    } catch (error) {
      console.error('Error scheduling content:', error);
    } finally {
      setSchedulingContent(false);
    }
  };

  // Load scheduled content and feedback analytics on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [scheduledData, analyticsData] = await Promise.all([
          campaignApi.getScheduledContent(campaign.id),
          campaignApi.getFeedbackAnalytics(campaign.id)
        ]);
        setScheduledContent(scheduledData);
        setFeedbackAnalytics(analyticsData);
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };
    
    loadData();
  }, [campaign.id]);

  // Open revision dialog
  const openRevisionDialog = (task: any) => {
    setRevisionDialog({
      isOpen: true,
      taskId: task.id,
      taskType: task.task_type,
      currentContent: typeof task.result === 'object' && task.result?.title 
        ? task.result.title 
        : task.result || task.content || 'No content available'
    });
  };

  // Submit revision feedback
  const submitRevisionFeedback = async (feedback: any) => {
    try {
      setSubmittingRevision(true);
      
      const result = await campaignApi.requestRevision(campaign.id, revisionDialog.taskId, feedback);
      
      // Update task status in local state
      setCampaignTasks(prev => prev.map(task => 
        task.id === revisionDialog.taskId 
          ? { ...task, status: 'revision_needed', review_notes: JSON.stringify(feedback) }
          : task
      ));
      
      setRevisionDialog({ isOpen: false, taskId: '', taskType: '', currentContent: '' });
      console.log('Revision feedback submitted:', result);
    } catch (error) {
      console.error('Error submitting revision feedback:', error);
    } finally {
      setSubmittingRevision(false);
    }
  };

  // Regenerate task with feedback
  const regenerateTask = async (taskId: string) => {
    try {
      setRegeneratingTasks(prev => new Set([...prev, taskId]));
      
      const result = await campaignApi.regenerateTask(campaign.id, taskId);
      
      // Update task status and content in local state
      setCampaignTasks(prev => prev.map(task => 
        task.id === taskId 
          ? { 
              ...task, 
              status: 'generated', 
              result: result.improved_content,
              quality_score: 85 // Estimated improved score
            }
          : task
      ));
      
      console.log('Task regenerated successfully:', result);
    } catch (error) {
      console.error('Error regenerating task:', error);
    } finally {
      setRegeneratingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskId);
        return next;
      });
    }
  };
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
    <>
      <div
        ref={overlayRef}
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        onClick={(e) => {
          if (e.target === overlayRef.current) {
            onClose()
          }
        }}
      >
        <div
          className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-modal="true"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-2xl font-bold text-gray-900">{campaign.name}</h2>
                <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm font-medium rounded-full">
                  {campaign.status}
                </span>
                <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-medium rounded-full flex items-center gap-1">
                  🤖 AI-Powered
                </span>
              </div>
              <p className="text-gray-600 mb-4">Campaign ID: {campaign.id}</p>
              
              {/* Campaign Configuration Overview */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Target Market</h4>
                  <p className="text-sm text-gray-900">
                    {campaign.target_audience?.includes('embedded') || campaign.description?.includes('embedded') 
                      ? '🏢 Embedded Partners' 
                      : '🏪 Direct Merchants'}
                  </p>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Campaign Type</h4>
                  <p className="text-sm text-gray-900 capitalize">
                    {campaign.strategy_type?.replace('_', ' ') || 'Lead Generation'}
                  </p>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Focus</h4>
                  <p className="text-sm text-gray-900">
                    {campaign.description?.includes('partnership') ? 'Partnership Acquisition' :
                     campaign.description?.includes('education') ? 'Credit Education' :
                     campaign.description?.includes('launch') ? 'Product Launch' : 'Business Growth'}
                  </p>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Content Status</h4>
                  <p className="text-sm text-gray-900">
                    {campaignTasks.filter(t => t.status === 'completed').length} of {campaignTasks.length} complete
                  </p>
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Close"
              title="Close"
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
                <h3 className="font-medium text-gray-900 mb-2">Scheduled</h3>
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
                            {channel.toLowerCase()}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Content & Tasks Section with Tabs */}
            {campaignTasks && campaignTasks.length > 0 && (
              <div className="bg-white border rounded-lg">
                {/* Tab Navigation */}
                <div className="border-b border-gray-200">
                  <nav className="flex space-x-8 px-6">
                    <button
                      onClick={() => setActiveTab('deliverables')}
                      className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                        activeTab === 'deliverables'
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      Content Narrative
                    </button>
                    <button
                      onClick={() => setActiveTab('tasks')}
                      className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                        activeTab === 'tasks'
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      Content Gallery ({campaignTasks.length})
                    </button>
                  </nav>
                </div>

                {/* Tab Content */}
                <div className="p-6">
                  {activeTab === 'deliverables' ? (
                    /* Enhanced Content Narrative Viewer */
                    <ContentNarrativeViewer
                      campaignId={campaign.id}
                      campaignName={campaign.name}
                    />
                  ) : (
                    /* Enhanced Content Gallery View */
                    <div>
                      <div className="flex items-center justify-between mb-6">
                        <h3 className="text-xl font-semibold">Generated Content</h3>
                        <div className="flex items-center space-x-2">
                          {campaignTasks.some(task => task.status === 'approved') && (
                            <button
                              onClick={scheduleApprovedContent}
                              disabled={schedulingContent}
                              className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                              {schedulingContent ? (
                                <RefreshCw className="w-4 h-4 animate-spin" />
                              ) : (
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                              )}
                              <span>{schedulingContent ? 'Scheduling...' : 'Schedule Approved'}</span>
                            </button>
                          )}
                          <button
                            onClick={executeAllTasks}
                            disabled={executingAll || campaignTasks.every(task => task.status === 'completed')}
                            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          >
                            {executingAll ? (
                              <RefreshCw className="w-4 h-4 animate-spin" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                            <span>{executingAll ? 'Executing...' : 'Execute All Tasks'}</span>
                          </button>
                        </div>
                      </div>
                      {/* Content Gallery Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {campaignTasks.map((task: any, index: number) => {
                          const getContentIcon = (taskType: string) => {
                            if (taskType?.includes('blog') || taskType?.includes('article')) return '📝';
                            if (taskType?.includes('social') || taskType?.includes('twitter') || taskType?.includes('linkedin')) return '📱';
                            if (taskType?.includes('email')) return '📧';
                            if (taskType?.includes('image') || taskType?.includes('visual')) return '🎨';
                            if (taskType?.includes('video')) return '🎥';
                            return '📄';
                          };

                          const getContentPreview = (task: any) => {
                            if (typeof task.result === 'object' && task.result?.content) {
                              return task.result.content.substring(0, 200) + '...';
                            }
                            if (typeof task.result === 'string') {
                              return task.result.substring(0, 200) + (task.result.length > 200 ? '...' : '');
                            }
                            if (task.content) {
                              return task.content.substring(0, 200) + (task.content.length > 200 ? '...' : '');
                            }
                            return task.status === 'pending' ? 'Content will be generated when task is executed' : 'No content available';
                          };

                          const getContentTitle = (task: any) => {
                            if (typeof task.result === 'object' && task.result?.title) {
                              return task.result.title;
                            }
                            return task.title || task.task_type || 'Untitled Content';
                          };

                          return (
                            <div key={index} className="bg-white border-2 rounded-xl p-6 hover:shadow-lg transition-all duration-200">
                              {/* Content Header */}
                              <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center space-x-3">
                                  <span className="text-2xl">{getContentIcon(task.task_type)}</span>
                                  <div>
                                    <h4 className="font-semibold text-gray-900 text-lg leading-tight">
                                      {getContentTitle(task)}
                                    </h4>
                                    <div className="flex items-center space-x-2 mt-1">
                                      <span className="text-sm text-gray-500 capitalize">
                                        {task.task_type?.replace('_', ' ') || 'Content'}
                                      </span>
                                      {task.channel && task.channel !== 'all' && (
                                        <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
                                          {task.channel}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                </div>
                                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                                  task.status === 'approved' ? 'bg-green-100 text-green-800' :
                                  task.status === 'scheduled' ? 'bg-purple-100 text-purple-800' :
                                  task.status === 'generated' ? 'bg-blue-100 text-blue-800' :
                                  task.status === 'revision_needed' ? 'bg-orange-100 text-orange-800' :
                                  task.status === 'completed' ? 'bg-green-100 text-green-800' :
                                  task.status === 'in_progress' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-gray-100 text-gray-800'
                                }`}>
                                  {task.status}
                                </span>
                              </div>

                              {/* Content Preview */}
                              <div className="mb-4">
                                <div className="bg-gray-50 rounded-lg p-4 min-h-[120px]">
                                  <p className="text-sm text-gray-700 leading-relaxed">
                                    {getContentPreview(task)}
                                  </p>
                                </div>
                              </div>

                              {/* Error Display */}
                              {task.error && (
                                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                                  <p className="text-sm text-red-600">{task.error}</p>
                                </div>
                              )}

                              {/* Metadata */}
                              <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                                {task.created_at && (
                                  <span>Created: {formatDate(task.created_at)}</span>
                                )}
                                {task.quality_score && (
                                  <div className="flex items-center space-x-1">
                                    <span>Quality:</span>
                                    <span className={`font-medium ${
                                      task.quality_score >= 80 ? 'text-green-600' :
                                      task.quality_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                                    }`}>
                                      {task.quality_score.toFixed(0)}%
                                    </span>
                                  </div>
                                )}
                              </div>

                              {/* Action Buttons */}
                              <div className="flex flex-wrap gap-2">
                                {/* Execute button for pending tasks */}
                                {task.status === 'pending' && (
                                  <button
                                    onClick={() => executeTask(task.id)}
                                    disabled={executingTasks.has(task.id)}
                                    className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                  >
                                    {executingTasks.has(task.id) ? (
                                      <RefreshCw className="w-4 h-4 animate-spin" />
                                    ) : (
                                      <Play className="w-4 h-4" />
                                    )}
                                    <span>{executingTasks.has(task.id) ? 'Generating...' : 'Generate Content'}</span>
                                  </button>
                                )}

                                {/* Review buttons for generated content */}
                                {task.status === 'generated' && (
                                  <>
                                    <button
                                      onClick={() => reviewTask(task.id, 'approve')}
                                      className="flex items-center space-x-1 px-3 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
                                    >
                                      <CheckCircle className="w-4 h-4" />
                                      <span>Approve</span>
                                    </button>
                                    <button
                                      onClick={() => openRevisionDialog(task)}
                                      className="flex items-center space-x-1 px-3 py-2 bg-orange-600 text-white text-sm font-medium rounded-lg hover:bg-orange-700 transition-colors"
                                    >
                                      <Edit3 className="w-4 h-4" />
                                      <span>Revise</span>
                                    </button>
                                    <button
                                      onClick={() => reviewTask(task.id, 'reject', 'Content rejected')}
                                      className="flex items-center space-x-1 px-3 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 transition-colors"
                                    >
                                      <XCircle className="w-4 h-4" />
                                      <span>Reject</span>
                                    </button>
                                  </>
                                )}

                                {/* Regenerate button for revision_needed tasks */}
                                {task.status === 'revision_needed' && (
                                  <>
                                    <button
                                      onClick={() => regenerateTask(task.id)}
                                      disabled={regeneratingTasks.has(task.id)}
                                      className="flex items-center space-x-1 px-3 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                    >
                                      {regeneratingTasks.has(task.id) ? (
                                        <RefreshCw className="w-4 h-4 animate-spin" />
                                      ) : (
                                        <RotateCcw className="w-4 h-4" />
                                      )}
                                      <span>{regeneratingTasks.has(task.id) ? 'Regenerating...' : 'Regenerate'}</span>
                                    </button>
                                    <button
                                      onClick={() => openRevisionDialog(task)}
                                      className="flex items-center space-x-1 px-3 py-2 bg-gray-600 text-white text-sm font-medium rounded-lg hover:bg-gray-700 transition-colors"
                                    >
                                      <Edit3 className="w-4 h-4" />
                                      <span>Update</span>
                                    </button>
                                  </>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Empty State */}
                      {campaignTasks.length === 0 && (
                        <div className="text-center py-12">
                          <div className="text-6xl mb-4">📝</div>
                          <h3 className="text-lg font-medium text-gray-900 mb-2">No Content Generated Yet</h3>
                          <p className="text-gray-500">Content will appear here once the campaign tasks are executed.</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Scheduled Content Calendar */}
            {(scheduledContent.length > 0 || (campaign.scheduled_posts && campaign.scheduled_posts.length > 0)) && (
              <div className="bg-white border rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Scheduled Content Calendar</h3>
                <div className="space-y-3">
                  {/* Smart Scheduled Content */}
                  {scheduledContent.map((item: any, index: number) => (
                    <div key={`scheduled-${index}`} className="flex items-center justify-between p-4 border rounded-lg bg-purple-50">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h4 className="font-medium text-gray-900">{item.platform}</h4>
                          <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
                            Optimal Time
                          </span>
                          {item.optimal_score && (
                            <span className="text-xs text-purple-600 font-medium">
                              Score: {(item.optimal_score * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-gray-600 line-clamp-2">{item.content}</p>
                        <div className="flex items-center space-x-4 mt-2">
                          {item.scheduled_at && (
                            <p className="text-xs text-gray-500">
                              📅 {formatDate(item.scheduled_at)}
                            </p>
                          )}
                          {item.reasoning && (
                            <p className="text-xs text-purple-600" title={item.reasoning}>
                              💡 Smart scheduling
                            </p>
                          )}
                        </div>
                      </div>
                      <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                        scheduled
                      </span>
                    </div>
                  ))}
                  
                  {/* Legacy Scheduled Posts */}
                  {campaign.scheduled_posts && campaign.scheduled_posts.map((post: any, index: number) => (
                    <div key={`legacy-${index}`} className="flex items-center justify-between p-4 border rounded-lg">
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
                    <h4 className="font-medium text-gray-900">Views</h4>
                    <p className="text-2xl font-bold text-blue-600">{campaign.performance.views || 0}</p>
                  </div>
                  <div className="text-center">
                    <h4 className="font-medium text-gray-900">Clicks</h4>
                    <p className="text-2xl font-bold text-green-600">{campaign.performance.clicks || 0}</p>
                  </div>
                  <div className="text-center">
                    <h4 className="font-medium text-gray-900">Engagement Rate</h4>
                    <p className="text-2xl font-bold text-purple-600">
                      {campaign.performance.engagement_rate !== undefined
                        ? `${(campaign.performance.engagement_rate * 100).toFixed(2)}%`
                        : '0.00%'}
                    </p>
                  </div>
                  <div className="text-center" />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Revision Feedback Dialog */}
      <RevisionFeedbackDialog
        isOpen={revisionDialog.isOpen}
        onClose={() => setRevisionDialog({ isOpen: false, taskId: '', taskType: '', currentContent: '' })}
        onSubmit={submitRevisionFeedback}
        taskId={revisionDialog.taskId}
        taskType={revisionDialog.taskType}
        currentContent={revisionDialog.currentContent}
        isSubmitting={submittingRevision}
      />
    </>
  );
} 