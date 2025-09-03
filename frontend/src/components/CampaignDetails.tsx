/**
 * CONTENT REVIEW SYSTEM - IMPLEMENTATION STATUS
 * 
 * ✅ REAL DATA:
 * - Campaign metadata (name, status, description, strategy, target audience, channels)
 * - Campaign tasks (task_type, target_format, status, result content)
 * - Full content display (actual generated content from database)
 * - Review actions (approve/reject functionality)
 * - Campaign progress calculations
 * 
 * 🎭 MOCKED DATA (for demo purposes):
 * - AI agent insights (SEO scores, content quality, brand consistency, GEO analysis)
 * - Agent confidence scores and recommendations
 * - Review form pre-population with AI suggestions
 * - Agent performance metrics (duration, cost, tokens)
 * 
 * 🚀 TO INTEGRATE WITH REAL AGENT DATA:
 * - Connect to agent_performance table for actual scores
 * - Connect to agent_decisions table for reasoning data
 * - Implement real SEO analysis from SEOAgent
 * - Integrate actual GEO analysis from GEOAnalysisAgent
 * - Connect brand consistency from BrandAgent
 */

import React, { useEffect, useRef, useState } from 'react';
import { Play, RefreshCw, CheckCircle, XCircle, RotateCcw, Eye, Copy, Download, X, Target, Users, Calendar, TrendingUp, Filter, BarChart3, Clock, Star, FileText, Shield, Search, UserCheck, MessageSquare, AlertCircle, ArrowRight } from 'lucide-react';
import type { CampaignDetail } from '../lib/api';
import { campaignApi } from '../lib/api';

interface CampaignDetailsProps {
  campaign: CampaignDetail;
  onClose: () => void;
  fullPage?: boolean;
}

export function CampaignDetails({ campaign, onClose, fullPage = false }: CampaignDetailsProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [executingTasks, setExecutingTasks] = useState<Set<string>>(new Set());
  const [campaignTasks, setCampaignTasks] = useState(campaign.tasks || []);
  const [executingAll, setExecutingAll] = useState(false);
  const [schedulingContent, setSchedulingContent] = useState(false);
  const [scheduledContent, setScheduledContent] = useState<any[]>([]);
  const [feedbackAnalytics, setFeedbackAnalytics] = useState<any>(null);
  const [fullContentModal, setFullContentModal] = useState<{
    isOpen: boolean;
    task: any;
    content: string;
    title: string;
  }>({ isOpen: false, task: null, content: '', title: '' });
  const [reviewModal, setReviewModal] = useState<{
    isOpen: boolean;
    task: any;
    stage: string;
  }>({ isOpen: false, task: null, stage: '' });
  const [reviewForm, setReviewForm] = useState({
    brandConsistency: { score: 0, comments: '', checked: false },
    contentQuality: { score: 0, comments: '', checked: false },
    seoOptimization: { score: 0, comments: '', checked: false },
    complianceCheck: { score: 0, comments: '', checked: false },
    overallRating: 0,
    finalComments: '',
    action: 'approve' as 'approve' | 'reject' | 'request_revision'
  });

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

  // Rerun agents on all campaign content
  const handleRerunAgents = async () => {
    try {
      setExecutingAll(true);
      
      // Show confirmation dialog
      const confirmed = window.confirm(
        campaignTasks.length === 0 ?
          'This will run all AI agents to generate campaign content with the latest AI improvements and optimizations. Are you sure you want to continue?' :
          'This will rerun all AI agents on your campaign content with the latest improvements and optimizations. Existing content will be regenerated. Are you sure you want to continue?'
      );
      
      if (!confirmed) {
        setExecutingAll(false);
        return;
      }

      console.log('Rerunning agents for campaign:', campaign.id);
      
      // Call the campaign orchestration API to rerun the full workflow
      const response = await fetch(`/api/v2/campaigns/${campaign.id}/rerun-agents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          rerun_all: true,
          include_optimization: true,
          preserve_approved: false // Set to true if you want to preserve already approved content
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to rerun agents: ${response.status}`);
      }

      const result = await response.json();
      
      // Reset all tasks to 'pending' status so they get regenerated
      setCampaignTasks(prev => prev.map(task => ({
        ...task,
        status: 'pending',
        result: null // Clear previous results
      })));
      
      // Show success message
      alert(
        campaignTasks.length === 0 ?
          'AI agents are now running with the latest improvements. Content will be generated shortly.' :
          'AI agents are now rerunning with the latest improvements. Content will be regenerated shortly.'
      );
      
      console.log('Agents rerun successfully:', result);
    } catch (error) {
      console.error('Error rerunning agents:', error);
      alert('Failed to rerun agents. Please try again or check the logs for details.');
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

  // Open structured review form
  const openReviewForm = (task: any, stage: string) => {
    const aiInsights = getAIInsights(task);
    setReviewModal({ isOpen: true, task, stage });
    
    // Pre-populate form with AI-suggested scores (converted from 10-point to 5-point scale)
    setReviewForm({
      brandConsistency: { 
        score: Math.round(aiInsights.brandAgent.score / 2), 
        comments: `AI Analysis: ${aiInsights.brandAgent.voice}, ${aiInsights.brandAgent.terminology}`, 
        checked: false 
      },
      contentQuality: { 
        score: Math.round(aiInsights.contentAgent.score / 2), 
        comments: `AI Analysis: ${aiInsights.contentAgent.structure}, Engagement: ${aiInsights.contentAgent.engagement}`, 
        checked: false 
      },
      seoOptimization: { 
        score: Math.round(aiInsights.seoAgent.score / 2), 
        comments: `AI Analysis: Keywords: ${aiInsights.seoAgent.keywords.join(', ')}, Readability: ${aiInsights.seoAgent.readability}/10`, 
        checked: false 
      },
      complianceCheck: { 
        score: Math.round(aiInsights.geoAgent.score / 2), 
        comments: `AI Analysis: Markets: ${aiInsights.geoAgent.markets.join(', ')}, Compliance: ${aiInsights.geoAgent.compliance.join(', ')}`, 
        checked: false 
      },
      overallRating: Math.round(aiInsights.overallScore / 2),
      finalComments: `AI Confidence: ${aiInsights.confidence}%. Top recommendation: ${aiInsights.recommendations[0]}`,
      action: aiInsights.overallScore >= 9.0 ? 'approve' : aiInsights.overallScore >= 7.5 ? 'approve' : 'request_revision'
    });
  };

  // Submit structured review
  const submitStructuredReview = async () => {
    if (!reviewModal.task) return;
    
    try {
      const reviewData = {
        taskId: reviewModal.task.id,
        stage: reviewModal.stage,
        criteria: {
          brandConsistency: reviewForm.brandConsistency,
          contentQuality: reviewForm.contentQuality,
          seoOptimization: reviewForm.seoOptimization,
          complianceCheck: reviewForm.complianceCheck
        },
        overallRating: reviewForm.overallRating,
        comments: reviewForm.finalComments,
        action: reviewForm.action
      };

      // For now, we'll use the existing reviewTask function
      // In a full implementation, this would call a new structured review API
      await reviewTask(reviewModal.task.id, reviewForm.action, reviewForm.finalComments);
      
      setReviewModal({ isOpen: false, task: null, stage: '' });
      console.log('Structured review submitted:', reviewData);
    } catch (error) {
      console.error('Error submitting structured review:', error);
    }
  };

  // Load basic campaign data on component mount
  useEffect(() => {
    const loadData = async () => {
      // Only load essential data for simplified view
      try {
        const scheduledData = await campaignApi.getScheduledContent(campaign.id);
        setScheduledContent(scheduledData);
      } catch (error) {
        console.warn('Scheduled content not available:', error);
        setScheduledContent([]);
      }
    };
    
    loadData();
  }, [campaign.id]);

  // Open full content modal
  const openFullContent = (task: any) => {
    let content = '';
    let title = '';
    
    // Extract content with better JSON handling
    if (typeof task.result === 'string') {
      // Check if this is JSON metadata or actual content
      if (task.result.startsWith('{') && task.result.includes('"content_type"')) {
        // This is just metadata, not actual content yet
        content = `Content Generation Details:\n\n${task.result}`;
        title = 'Content Metadata';
      } else {
        // This should be actual content
        try {
          const parsed = JSON.parse(task.result);
          content = parsed.content || parsed.description || task.result;
          title = parsed.title || '';
        } catch {
          // Not JSON, use as-is
          content = task.result;
        }
      }
    } else if (typeof task.result === 'object') {
      content = task.result?.content || JSON.stringify(task.result, null, 2);
      title = task.result?.title || '';
    } else if (task.content) {
      content = task.content;
    } else {
      content = 'No content available yet. Click "Generate Content" to create content for this task.';
    }
    
    // Extract title from content if not found
    if (!title && content && !content.includes('"content_type"')) {
      const lines = content.split('\n').filter(line => line.trim());
      if (lines.length > 0) {
        const firstLine = lines[0].replace(/^#+\s*/, '').trim();
        if (firstLine.length > 0 && firstLine.length < 100) {
          title = firstLine;
        }
      }
    }
    
    setFullContentModal({
      isOpen: true,
      task,
      content,
      title: title || getContentTitle(task)
    });
  };

  // Copy content to clipboard
  const copyToClipboard = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      // You could add a toast notification here
      console.log('Content copied to clipboard');
    } catch (err) {
      console.error('Failed to copy content:', err);
    }
  };

  // Download content as text file
  const downloadContent = (content: string, title: string) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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

  // Review workflow stages
  const getReviewStages = () => [
    { 
      id: 'initial_review', 
      name: 'Quality Check', 
      icon: FileText, 
      description: 'Accuracy, relevance, structure',
      color: 'blue'
    },
    { 
      id: 'brand_review', 
      name: 'Brand Check', 
      icon: Shield, 
      description: 'Voice, tone, guidelines',
      color: 'purple'
    },
    { 
      id: 'seo_review', 
      name: 'SEO Review', 
      icon: Search, 
      description: 'Keywords, optimization',
      color: 'green'
    },
    { 
      id: 'final_approval', 
      name: 'Final Approval', 
      icon: UserCheck, 
      description: 'Executive sign-off',
      color: 'orange'
    }
  ];

  const getTaskReviewStage = (task: any) => {
    // Map task status to review stage
    switch (task.status?.toLowerCase()) {
      case 'pending':
        return 'not_started';
      case 'in_progress':
        return 'initial_review';
      case 'completed':
        return 'brand_review'; // Content is generated, needs brand review
      case 'brand_approved':
        return 'seo_review';
      case 'seo_approved':
        return 'final_approval';
      case 'approved':
        return 'completed';
      case 'rejected':
        return 'revision_needed';
      default:
        return 'initial_review';
    }
  };

  const getStageProgress = (currentStage: string, stages: any[]) => {
    const stageIndex = stages.findIndex(s => s.id === currentStage);
    return stageIndex >= 0 ? (stageIndex + 1) / stages.length * 100 : 0;
  };

  // Extract full content from task result
  const getFullContent = (task: any) => {
    if (!task?.result) return 'No content available';
    
    if (typeof task.result === 'string') {
      // Check if this looks like JSON or actual content
      if (task.result.trim().startsWith('{') && task.result.trim().endsWith('}')) {
        try {
          const parsed = JSON.parse(task.result);
          return parsed.content || parsed.description || parsed.text || task.result;
        } catch {
          return task.result;
        }
      } else {
        return task.result;
      }
    } else if (typeof task.result === 'object') {
      return task.result?.content || task.result?.description || JSON.stringify(task.result, null, 2);
    }
    
    return 'Content format not recognized';
  };

  // Generate AI insights based on task type and content
  const getAIInsights = (task: any) => {
    // TODO: Replace with real API call to /orchestration/campaigns/{campaignId}/ai-insights
    // This mock function can now be replaced with actual database insights
    const baseInsights = {
      seoAgent: {
        score: Math.round((Math.random() * 2 + 8) * 10) / 10, // 8.0-10.0 range
        keywords: task.task_type?.includes('blog') ? ['embedded finance', 'AI platform', 'fintech'] : ['CrediLinq', 'automation', 'efficiency'],
        readability: Math.round((Math.random() * 2 + 7) * 10) / 10,
        recommendations: ['Optimize meta description length', 'Add more subheadings', 'Include target keywords naturally']
      },
      contentAgent: {
        score: Math.round((Math.random() * 1.5 + 8.5) * 10) / 10,
        engagement: task.target_format?.includes('social') ? 'High' : 'Medium-High',
        accuracy: 'Verified',
        structure: 'Clear & well-organized',
        cta: task.task_type?.includes('email') ? 'Compelling' : 'Present'
      },
      brandAgent: {
        score: Math.round((Math.random() * 1.5 + 8.5) * 10) / 10,
        voice: 'Professional & authoritative',
        consistency: true,
        terminology: 'Correctly applied',
        alignment: task.task_type?.includes('blog') ? 'Excellent' : 'Good'
      },
      geoAgent: {
        score: Math.round((Math.random() * 1.5 + 8.2) * 10) / 10,
        markets: ['North America', 'Europe'],
        compliance: ['GDPR', 'CCPA'],
        localization: 'Appropriate',
        sensitivity: 'Reviewed'
      }
    };

    const overallScore = Math.round((baseInsights.seoAgent.score + baseInsights.contentAgent.score + 
                                   baseInsights.brandAgent.score + baseInsights.geoAgent.score) / 4 * 10) / 10;

    /* 
    TODO: Replace above mock data with real API call like:
    
    const realInsights = await fetch(`/api/campaigns/orchestration/campaigns/${campaignId}/ai-insights`);
    const data = await realInsights.json();
    
    return {
      seoAgent: {
        score: data.seo_geo_analysis.seo_analysis.avg_confidence * 10,
        keywords: data.seo_geo_analysis.keyword_extraction.keywords,
        readability: data.seo_geo_analysis.readability_metrics.avg_score,
        recommendations: data.agent_insights.find(a => a.agent_type === 'seo')?.recent_decisions[0]?.reasoning
      },
      geoAgent: {
        score: data.seo_geo_analysis.geo_analysis.avg_confidence * 10,
        // ... other real data mappings
      },
      confidence: data.summary.avg_success_rate,
      recommendations: data.agent_insights.flatMap(a => a.recommendations.map(r => r.description))
    };
    */

    return {
      ...baseInsights,
      overallScore,
      confidence: Math.round(overallScore * 10),
      recommendations: [
        'Consider adding industry statistics for enhanced credibility',
        'Social media adaptation: Excellent for LinkedIn, trim for Twitter',
        task.target_format?.includes('email') ? 'Add personalization placeholder' : 'Optimize for mobile viewing'
      ]
    };
  };

  // Main content that can be rendered in both modal and full-page modes
  const mainContent = (
    <div className={`bg-white ${fullPage ? 'min-h-screen' : 'rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto'}`}>
      {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-2xl font-bold text-gray-900">{campaign.name}</h2>
                <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm font-medium rounded-full">
                  {campaign.status}
                </span>
              </div>
              <p className="text-gray-600 mb-4">Campaign ID: {campaign.id}</p>
              
              {/* Campaign Overview Dashboard */}
              <div className="space-y-4">
                {/* Campaign Progress Bar */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900">Campaign Progress</span>
                    <span className="text-sm text-gray-600">
                      {campaignTasks.filter(t => t.status === 'completed' || t.status === 'approved').length} of {campaignTasks.length} content pieces
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${campaignTasks.length > 0 
                          ? ((campaignTasks.filter(t => t.status === 'completed' || t.status === 'approved').length / campaignTasks.length) * 100)
                          : 0}%`
                      }}
                    ></div>
                  </div>
                </div>

                {/* Campaign Context Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Strategy & Objectives */}
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-3">
                      <Target className="w-4 h-4 text-blue-600" />
                      <h4 className="font-medium text-gray-900">Strategy</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                      {campaign.metadata?.strategy_type && (
                        <p><span className="font-medium">Type:</span> <span className="capitalize">{campaign.metadata.strategy_type.replace('_', ' ')}</span></p>
                      )}
                      {campaign.metadata?.description && (
                        <p className="text-gray-600">{campaign.metadata.description}</p>
                      )}
                      {campaign.metadata?.timeline_weeks && (
                        <p><span className="font-medium">Timeline:</span> {campaign.metadata.timeline_weeks} weeks</p>
                      )}
                    </div>
                  </div>

                  {/* Target Audience */}
                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-3">
                      <Users className="w-4 h-4 text-green-600" />
                      <h4 className="font-medium text-gray-900">Audience</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                      {campaign.metadata?.target_audience && (
                        <p className="text-gray-600">{campaign.metadata.target_audience}</p>
                      )}
                      {campaign.metadata?.company_context && (
                        <p className="text-gray-600 text-xs">{campaign.metadata.company_context.substring(0, 100)}...</p>
                      )}
                    </div>
                  </div>

                  {/* Success Metrics */}
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-3">
                      <TrendingUp className="w-4 h-4 text-purple-600" />
                      <h4 className="font-medium text-gray-900">Metrics</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                      {campaign.metadata?.success_metrics?.content_pieces && (
                        <p><span className="font-medium">Target:</span> {campaign.metadata.success_metrics.content_pieces} pieces</p>
                      )}
                      {campaign.metadata?.success_metrics?.target_channels && (
                        <p><span className="font-medium">Channels:</span> {campaign.metadata.success_metrics.target_channels}</p>
                      )}
                      {campaign.metadata?.priority && (
                        <p><span className="font-medium">Priority:</span> <span className="capitalize text-orange-600">{campaign.metadata.priority}</span></p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Distribution Channels */}
                {campaign.metadata?.distribution_channels && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Distribution Channels</h4>
                    <div className="flex flex-wrap gap-2">
                      {campaign.metadata.distribution_channels.map((channel: string, index: number) => (
                        <span
                          key={index}
                          className="px-3 py-1 bg-white border border-gray-200 rounded-full text-sm text-gray-700 capitalize"
                        >
                          {channel.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Content Pipeline Visualization */}
                <div className="mt-6 p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                      <BarChart3 className="w-5 h-5 text-blue-600" />
                      <span>Content Pipeline</span>
                    </h3>
                    <div className="flex space-x-4 text-xs">
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 rounded-full bg-gray-400"></div>
                        <span>Pending</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                        <span>Generating</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                        <span>Review</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span>Approved</span>
                      </div>
                    </div>
                  </div>

                  {/* Pipeline Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    {[
                      { label: 'Total', count: campaignTasks.length, color: 'bg-gray-100 text-gray-800' },
                      { label: 'Completed', count: campaignTasks.filter(t => t.status === 'completed').length, color: 'bg-yellow-100 text-yellow-800' },
                      { label: 'Approved', count: campaignTasks.filter(t => t.status === 'approved').length, color: 'bg-green-100 text-green-800' },
                      { label: 'Pending', count: campaignTasks.filter(t => t.status === 'pending').length, color: 'bg-gray-100 text-gray-600' }
                    ].map((stat, index) => (
                      <div key={index} className="text-center">
                        <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full ${stat.color} font-bold text-lg mb-1`}>
                          {stat.count}
                        </div>
                        <p className="text-xs text-gray-600">{stat.label}</p>
                      </div>
                    ))}
                  </div>

                  {/* Content Type Breakdown */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {[
                      { type: 'blog', icon: '📝', label: 'Blog Posts' },
                      { type: 'social', icon: '📱', label: 'Social Media' },
                      { type: 'email', icon: '📧', label: 'Email' },
                      { type: 'image', icon: '🎨', label: 'Visual' }
                    ].map((contentType, index) => {
                      const count = campaignTasks.filter(task => 
                        task.task_type?.toLowerCase().includes(contentType.type) ||
                        task.target_format?.toLowerCase().includes(contentType.type)
                      ).length;
                      const completed = campaignTasks.filter(task => 
                        (task.task_type?.toLowerCase().includes(contentType.type) ||
                         task.target_format?.toLowerCase().includes(contentType.type)) &&
                        (task.status === 'completed' || task.status === 'approved')
                      ).length;
                      
                      return count > 0 ? (
                        <div key={index} className="bg-white rounded-lg p-3 border border-gray-200">
                          <div className="flex items-center space-x-2 mb-2">
                            <span className="text-lg">{contentType.icon}</span>
                            <span className="text-sm font-medium text-gray-900">{contentType.label}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-600">{completed}/{count}</span>
                            <div className="w-12 bg-gray-200 rounded-full h-1">
                              <div
                                className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                                style={{ width: `${count > 0 ? (completed / count) * 100 : 0}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      ) : null;
                    })}
                  </div>
                </div>

                {/* Quick Actions Hub */}
                <div className="mt-6 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                      <Clock className="w-5 h-5 text-green-600" />
                      <span>Quick Actions</span>
                    </h3>
                    <div className="text-xs text-gray-500">
                      {campaignTasks.filter(t => t.status === 'completed').length} items ready for review
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                    {/* Generate All Missing */}
                    <button
                      onClick={executeAllTasks}
                      disabled={executingAll || campaignTasks.filter(t => t.status === 'pending').length === 0}
                      className="flex items-center justify-center space-x-2 p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {executingAll ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <Play className="w-4 h-4" />
                      )}
                      <span className="text-sm font-medium">
                        {executingAll ? 'Generating...' : `Generate All (${campaignTasks.filter(t => t.status === 'pending').length})`}
                      </span>
                    </button>

                    {/* Rerun Agents Button */}
                    <button
                      onClick={handleRerunAgents}
                      disabled={executingAll}
                      className="flex items-center justify-center space-x-2 p-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      title={campaignTasks.length === 0 ? 
                        "Run AI agents to generate campaign content with latest improvements" : 
                        "Rerun AI agents to regenerate all content with latest improvements"}
                    >
                      <RotateCcw className="w-4 h-4" />
                      <span className="text-sm font-medium">
                        {campaignTasks.length === 0 ? 'Run Agents' : 'Rerun Agents'}
                      </span>
                    </button>

                    {/* Approve All Completed */}
                    <button
                      onClick={() => {
                        const completedTasks = campaignTasks.filter(t => t.status === 'completed');
                        completedTasks.forEach(task => reviewTask(task.id, 'approve'));
                      }}
                      disabled={campaignTasks.filter(t => t.status === 'completed').length === 0}
                      className="flex items-center justify-center space-x-2 p-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span className="text-sm font-medium">
                        Approve All ({campaignTasks.filter(t => t.status === 'completed').length})
                      </span>
                    </button>

                    {/* Review Pending */}
                    <button
                      onClick={() => {
                        const firstCompleted = campaignTasks.find(t => t.status === 'completed');
                        if (firstCompleted) {
                          document.getElementById(`task-${firstCompleted.id}`)?.scrollIntoView({ behavior: 'smooth' });
                        }
                      }}
                      disabled={campaignTasks.filter(t => t.status === 'completed').length === 0}
                      className="flex items-center justify-center space-x-2 p-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      <Eye className="w-4 h-4" />
                      <span className="text-sm font-medium">Review Queue</span>
                    </button>

                    {/* Schedule Approved */}
                    <button
                      onClick={scheduleApprovedContent}
                      disabled={schedulingContent || campaignTasks.filter(t => t.status === 'approved').length === 0}
                      className="flex items-center justify-center space-x-2 p-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {schedulingContent ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <Calendar className="w-4 h-4" />
                      )}
                      <span className="text-sm font-medium">
                        {schedulingContent ? 'Scheduling...' : `Schedule (${campaignTasks.filter(t => t.status === 'approved').length})`}
                      </span>
                    </button>
                  </div>

                  {/* Bulk Selection Info */}
                  {campaignTasks.some(t => t.status === 'completed') && (
                    <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-xs text-blue-800">
                        💡 <strong>Quick Tip:</strong> Use "Approve All" for content that meets your standards, or review individually using the content cards below.
                      </p>
                    </div>
                  )}
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
            {/* Simple Status Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-900 mb-2">Campaign Status</h3>
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(campaign.status)}`}>
                  {campaign.status}
                </span>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-medium text-gray-900 mb-2">Content Progress</h3>
                <p className="text-lg font-bold text-blue-600">
                  {campaignTasks.filter(t => t.status === 'completed').length} / {campaign.tasks?.length || 0} completed
                </p>
              </div>
            </div>


            {/* Simplified Content Section */}
            {campaignTasks && campaignTasks.length > 0 && (
              <div className="bg-white border rounded-lg">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-semibold">Campaign Content</h3>
                    <div className="flex items-center space-x-2">
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
                        <span>{executingAll ? 'Generating...' : 'Generate Content'}</span>
                      </button>
                    </div>
                  </div>
                  {/* Content Gallery */}
                  <div>
                    {/* Simplified Content Grid */}
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
                            // Parse the content from different possible formats
                            let content = '';
                            
                            if (typeof task.result === 'string') {
                              // Check if this looks like JSON or actual content
                              if (task.result.trim().startsWith('{') && task.result.trim().endsWith('}')) {
                                // This might be JSON, try to parse it
                                try {
                                  const parsed = JSON.parse(task.result);
                                  // If it has content fields, use those; otherwise it's metadata
                                  content = parsed.content || parsed.description || parsed.text || 'Content generation in progress...';
                                } catch {
                                  // If JSON parsing fails, treat as regular content
                                  content = task.result;
                                }
                              } else {
                                // This is likely actual content (like blog posts with Title:, Summary:, Content: format)
                                content = task.result;
                              }
                            } else if (typeof task.result === 'object') {
                              content = task.result?.content || task.result?.description || JSON.stringify(task.result, null, 2);
                            } else if (task.content) {
                              content = task.content;
                            }
                            
                            if (content && content.length > 0 && !content.includes('"content_type"')) {
                              // Clean up content - remove excessive whitespace and line breaks
                              const cleanContent = content.replace(/\s+/g, ' ').trim();
                              return cleanContent.substring(0, 200) + (cleanContent.length > 200 ? '...' : '');
                            }
                            
                            return task.status === 'pending' ? 'Content will be generated when task is executed' : 
                                   task.status === 'generated' ? 'Content ready for review' : 'No content available';
                          };

                          const getContentTitle = (task: any) => {
                            // Extract title from different sources
                            if (typeof task.result === 'string') {
                              // Check if it's JSON metadata
                              if (task.result.startsWith('{') && task.result.includes('"title"')) {
                                try {
                                  const parsed = JSON.parse(task.result);
                                  if (parsed.title) return parsed.title;
                                } catch {}
                              } else if (task.result.length > 0 && !task.result.includes('"content_type"')) {
                                // Try to extract title from actual content
                                const lines = task.result.split('\n').filter(line => line.trim());
                                if (lines.length > 0) {
                                  const firstLine = lines[0].replace(/^#+\s*/, '').trim();
                                  if (firstLine.length > 0 && firstLine.length < 100) {
                                    return firstLine;
                                  }
                                }
                              }
                            } else if (typeof task.result === 'object' && task.result?.title) {
                              return task.result.title;
                            }
                            
                            // Generate title from task type with better formatting
                            const taskType = task.task_type?.replace(/_/g, ' ').toLowerCase() || 'content';
                            const formattedType = taskType.charAt(0).toUpperCase() + taskType.slice(1);
                            return task.title || formattedType;
                          };

                          return (
                            <div key={index} id={`task-${task.id}`} className="bg-white border-2 rounded-xl p-6 hover:shadow-lg transition-all duration-200">
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
                                  task.status === 'completed' ? (task.result && task.result.includes('content_type') ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800') :
                                  task.status === 'in_progress' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-gray-100 text-gray-800'
                                }`}>
                                  {task.status === 'completed' && task.result && task.result.includes('content_type') ? 'needs review' : task.status}
                                </span>
                              </div>

                              {/* Review Stage Indicator */}
                              {task.result && (
                                <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                  <div className="flex items-center justify-between mb-3">
                                    <h5 className="text-sm font-medium text-gray-900 flex items-center space-x-2">
                                      <MessageSquare className="w-4 h-4 text-blue-600" />
                                      <span>Review Workflow</span>
                                    </h5>
                                    <span className="text-xs text-gray-500">
                                      Stage {getReviewStages().findIndex(s => s.id === getTaskReviewStage(task)) + 1} of {getReviewStages().length}
                                    </span>
                                  </div>
                                  
                                  {/* Progress Bar */}
                                  <div className="mb-3">
                                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                                      <span>Progress</span>
                                      <span>{Math.round(getStageProgress(getTaskReviewStage(task), getReviewStages()))}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div
                                        className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${getStageProgress(getTaskReviewStage(task), getReviewStages())}%` }}
                                      ></div>
                                    </div>
                                  </div>

                                  {/* Review Stages */}
                                  <div className="grid grid-cols-4 gap-2 text-center">
                                    {getReviewStages().map((stage, stageIndex) => {
                                      const currentStage = getTaskReviewStage(task);
                                      const currentIndex = getReviewStages().findIndex(s => s.id === currentStage);
                                      const isCompleted = stageIndex < currentIndex;
                                      const isCurrent = stageIndex === currentIndex;
                                      
                                      return (
                                        <div key={stage.id} className="flex flex-col items-center">
                                          <button
                                            onClick={() => isCurrent ? openReviewForm(task, stage.id) : null}
                                            disabled={!isCurrent}
                                            className={`flex items-center justify-center w-8 h-8 rounded-full border-2 transition-all duration-300 mb-1 ${
                                              isCompleted 
                                                ? 'bg-green-600 border-green-600 text-white' 
                                                : isCurrent 
                                                ? `border-${stage.color}-500 bg-${stage.color}-50 text-${stage.color}-700 hover:bg-${stage.color}-100 cursor-pointer`
                                                : 'border-gray-300 bg-white text-gray-400 cursor-not-allowed'
                                            }`}
                                            title={isCurrent ? `Start ${stage.name}` : isCompleted ? `${stage.name} Complete` : `${stage.name} Pending`}
                                          >
                                            {isCompleted ? (
                                              <CheckCircle className="w-4 h-4" />
                                            ) : (
                                              <stage.icon className="w-4 h-4" />
                                            )}
                                          </button>
                                          <p className={`text-xs font-medium ${
                                            isCurrent ? `text-${stage.color}-700` : isCompleted ? 'text-gray-700' : 'text-gray-400'
                                          }`}>
                                            {stage.name}
                                          </p>
                                          <p className="text-xs text-gray-500 hidden md:block">
                                            {stage.description}
                                          </p>
                                        </div>
                                      );
                                    })}
                                  </div>
                                  
                                  {/* Quick Review Action */}
                                  <div className="mt-3 pt-3 border-t border-gray-200">
                                    <button
                                      onClick={() => openReviewForm(task, getTaskReviewStage(task))}
                                      className="w-full px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2"
                                    >
                                      <MessageSquare className="w-4 h-4" />
                                      <span>Start Structured Review</span>
                                    </button>
                                  </div>
                                </div>
                              )}

                              {/* AI Agent Summary (for completed tasks) */}
                              {task.result && task.status === 'completed' && (() => {
                                const aiInsights = getAIInsights(task);
                                return (
                                  <div className="mb-4 p-3 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                                    <div className="flex items-center justify-between mb-2">
                                      <h5 className="text-sm font-medium text-gray-900 flex items-center space-x-2">
                                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                        <span>AI Analysis Summary</span>
                                      </h5>
                                      <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded-full">
                                        Confidence: {aiInsights.confidence}%
                                      </span>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-3 text-xs">
                                      <div className="flex items-center space-x-2">
                                        <Search className="w-3 h-3 text-green-600" />
                                        <span className="text-gray-600">SEO:</span>
                                        <span className="font-medium text-green-700">{aiInsights.seoAgent.score}/10</span>
                                      </div>
                                      <div className="flex items-center space-x-2">
                                        <Shield className="w-3 h-3 text-purple-600" />
                                        <span className="text-gray-600">Brand:</span>
                                        <span className="font-medium text-purple-700">{aiInsights.brandAgent.score}/10</span>
                                      </div>
                                      <div className="flex items-center space-x-2">
                                        <FileText className="w-3 h-3 text-blue-600" />
                                        <span className="text-gray-600">Quality:</span>
                                        <span className="font-medium text-blue-700">{aiInsights.contentAgent.score}/10</span>
                                      </div>
                                      <div className="flex items-center space-x-2">
                                        <Target className="w-3 h-3 text-orange-600" />
                                        <span className="text-gray-600">GEO:</span>
                                        <span className="font-medium text-orange-700">{aiInsights.geoAgent.score}/10</span>
                                      </div>
                                    </div>
                                    
                                    <div className="mt-2 text-xs text-gray-600">
                                      <strong>Top AI Recommendation:</strong> {aiInsights.recommendations[0]}
                                    </div>
                                    
                                  </div>
                                );
                              })()}

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

                                {/* Simple review buttons for completed content that needs review */}
                                {(task.status === 'generated' || task.status === 'completed') && (
                                  <>
                                    <button
                                      onClick={() => reviewTask(task.id, 'approve')}
                                      className="flex items-center space-x-1 px-3 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
                                    >
                                      <CheckCircle className="w-4 h-4" />
                                      <span>Approve</span>
                                    </button>
                                    <button
                                      onClick={() => reviewTask(task.id, 'reject', 'Needs revision')}
                                      className="flex items-center space-x-1 px-3 py-2 bg-gray-600 text-white text-sm font-medium rounded-lg hover:bg-gray-700 transition-colors"
                                    >
                                      <XCircle className="w-4 h-4" />
                                      <span>Reject</span>
                                    </button>
                                  </>
                                )}

                                {/* Simple regenerate for rejected tasks or re-generate completed tasks */}
                                {(task.status === 'revision_needed' || task.status === 'rejected' || (task.status === 'completed' && task.result && task.result.includes('content_type'))) && (
                                  <button
                                    onClick={() => executeTask(task.id)}
                                    disabled={executingTasks.has(task.id)}
                                    className="flex items-center space-x-1 px-3 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                  >
                                    {executingTasks.has(task.id) ? (
                                      <RefreshCw className="w-4 h-4 animate-spin" />
                                    ) : (
                                      <RotateCcw className="w-4 h-4" />
                                    )}
                                    <span>{executingTasks.has(task.id) ? 'Regenerating...' : 'Regenerate'}</span>
                                  </button>
                                )}
                                
                                {/* Read Full Content button - always visible if content exists */}
                                {(task.result || task.content) && (
                                  <button
                                    onClick={() => openFullContent(task)}
                                    className="flex items-center space-x-1 px-3 py-2 bg-gray-100 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors"
                                  >
                                    <Eye className="w-4 h-4" />
                                    <span>Read Full Content</span>
                                  </button>
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
                        <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Create Content</h3>
                        <p className="text-gray-500 mb-4">Click "Generate Content" above to start creating campaign content with AI.</p>
                        <div className="bg-blue-50 p-4 rounded-lg mt-4 max-w-md mx-auto">
                          <p className="text-sm text-blue-800">
                            💡 <strong>After generating:</strong> Content will appear below as cards with "Approve" and "Reject" buttons for review.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Basic Scheduled Content */}
            {scheduledContent.length > 0 && (
              <div className="bg-white border rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Scheduled Content</h3>
                <div className="space-y-3">
                  {scheduledContent.map((item: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900">{item.platform}</h4>
                        <p className="text-sm text-gray-600 line-clamp-2">{item.content}</p>
                        {item.scheduled_at && (
                          <p className="text-xs text-gray-500 mt-1">
                            📅 {formatDate(item.scheduled_at)}
                          </p>
                        )}
                      </div>
                      <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        scheduled
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
    </div>
  );

  return (
    <>
      {fullPage ? (
        // Full page mode - render content directly
        <>
          {mainContent}
        </>
      ) : (
        // Modal mode - wrap content in modal overlay
        <div
          ref={overlayRef}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={(e) => {
            if (e.target === overlayRef.current) {
              onClose()
            }
          }}
        >
          <div onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
            {mainContent}
          </div>
        </div>
      )}
      
      {/* Full Content Modal */}
      {fullContentModal.isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-gray-900">{fullContentModal.title}</h3>
                <p className="text-sm text-gray-500 mt-1">
                  {fullContentModal.task?.task_type?.replace(/_/g, ' ') || 'Generated Content'} • 
                  {fullContentModal.content?.length || 0} characters
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => copyToClipboard(fullContentModal.content)}
                  className="flex items-center space-x-1 px-3 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Copy className="w-4 h-4" />
                  <span>Copy</span>
                </button>
                <button
                  onClick={() => downloadContent(fullContentModal.content, fullContentModal.title)}
                  className="flex items-center space-x-1 px-3 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
                <button
                  onClick={() => setFullContentModal({ isOpen: false, task: null, content: '', title: '' })}
                  className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
            
            {/* Modal Content */}
            <div className="p-6 overflow-y-auto max-h-[70vh]">
              <div className="bg-gray-50 rounded-lg p-6">
                <pre className="whitespace-pre-wrap font-sans text-gray-800 text-sm leading-relaxed">
                  {fullContentModal.content}
                </pre>
              </div>
              
              {/* Content Metadata */}
              {fullContentModal.task && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <h4 className="text-sm font-medium text-gray-900 mb-3">Content Details</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-500">Type:</span>
                      <span className="ml-2 text-gray-900">{fullContentModal.task.task_type?.replace(/_/g, ' ')}</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-500">Status:</span>
                      <span className="ml-2 text-gray-900 capitalize">{fullContentModal.task.status}</span>
                    </div>
                    {fullContentModal.task.channel && (
                      <div>
                        <span className="font-medium text-gray-500">Channel:</span>
                        <span className="ml-2 text-gray-900">{fullContentModal.task.channel}</span>
                      </div>
                    )}
                    {fullContentModal.task.created_at && (
                      <div>
                        <span className="font-medium text-gray-500">Created:</span>
                        <span className="ml-2 text-gray-900">{formatDate(fullContentModal.task.created_at)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Structured Review Modal */}
      {reviewModal.isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-gray-900">Structured Content Review</h3>
                <p className="text-sm text-gray-500 mt-1">
                  {getReviewStages().find(s => s.id === reviewModal.stage)?.name} • 
                  {reviewModal.task?.task_type?.replace(/_/g, ' ') || 'Content'}
                </p>
              </div>
              <button
                onClick={() => setReviewModal({ isOpen: false, task: null, stage: '' })}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Modal Content */}
            <div className="p-6 overflow-y-auto max-h-[70vh]">
              {/* Content Being Reviewed */}
              <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-lg font-medium text-gray-900">Content Under Review</h4>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => copyToClipboard(getFullContent(reviewModal.task))}
                      className="flex items-center space-x-1 px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                    >
                      <Copy className="w-3 h-3" />
                      <span>Copy</span>
                    </button>
                    <span className="text-xs text-gray-500">
                      {getFullContent(reviewModal.task).length} chars
                    </span>
                  </div>
                </div>
                
                <div className="bg-white rounded-lg p-4 max-h-64 overflow-y-auto border border-gray-200">
                  <div className="prose prose-sm max-w-none">
                    <pre className="whitespace-pre-wrap font-sans text-gray-800 text-sm leading-relaxed">
                      {getFullContent(reviewModal.task)}
                    </pre>
                  </div>
                </div>
              </div>

              {/* AI Insights Panel */}
              {(() => {
                const aiInsights = getAIInsights(reviewModal.task);
                return (
                  <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                    <h4 className="text-lg font-medium text-gray-900 mb-4 flex items-center space-x-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                      <span>AI Agent Insights</span>
                    </h4>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* SEO Agent Analysis */}
                      <div className="bg-white p-3 rounded-lg border border-green-200">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Search className="w-4 h-4 text-green-600" />
                            <span className="font-medium text-sm">SEO Agent</span>
                          </div>
                          <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded-full">Score: {aiInsights.seoAgent.score}/10</span>
                        </div>
                        <div className="text-xs text-gray-600 space-y-1">
                          <p>• Target keywords: {aiInsights.seoAgent.keywords.map(k => `"${k}"`).join(', ')}</p>
                          <p>• Readability score: {aiInsights.seoAgent.readability}/10 (Good)</p>
                          <p>• Meta description optimized for length</p>
                          <p>• H2/H3 structure follows SEO best practices</p>
                        </div>
                      </div>

                      {/* Content Quality Agent */}
                      <div className="bg-white p-3 rounded-lg border border-blue-200">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <FileText className="w-4 h-4 text-blue-600" />
                            <span className="font-medium text-sm">Content Agent</span>
                          </div>
                          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full">Score: {aiInsights.contentAgent.score}/10</span>
                        </div>
                        <div className="text-xs text-gray-600 space-y-1">
                          <p>• Engagement potential: {aiInsights.contentAgent.engagement}</p>
                          <p>• Factual accuracy: {aiInsights.contentAgent.accuracy}</p>
                          <p>• Sentence structure: {aiInsights.contentAgent.structure}</p>
                          <p>• Call-to-action: {aiInsights.contentAgent.cta}</p>
                        </div>
                      </div>

                      {/* Brand Consistency Agent */}
                      <div className="bg-white p-3 rounded-lg border border-purple-200">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Shield className="w-4 h-4 text-purple-600" />
                            <span className="font-medium text-sm">Brand Agent</span>
                          </div>
                          <span className="text-xs px-2 py-1 bg-purple-100 text-purple-800 rounded-full">Score: {aiInsights.brandAgent.score}/10</span>
                        </div>
                        <div className="text-xs text-gray-600 space-y-1">
                          <p>• Voice alignment: {aiInsights.brandAgent.voice}</p>
                          <p>• Tone consistency: {aiInsights.brandAgent.consistency ? '✓ Maintained throughout' : '⚠ Needs review'}</p>
                          <p>• Brand terminology: {aiInsights.brandAgent.terminology}</p>
                          <p>• Value proposition: Clearly communicated</p>
                        </div>
                      </div>

                      {/* GEO Analysis Agent */}
                      <div className="bg-white p-3 rounded-lg border border-orange-200">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Target className="w-4 h-4 text-orange-600" />
                            <span className="font-medium text-sm">GEO Agent</span>
                          </div>
                          <span className="text-xs px-2 py-1 bg-orange-100 text-orange-800 rounded-full">Score: {aiInsights.geoAgent.score}/10</span>
                        </div>
                        <div className="text-xs text-gray-600 space-y-1">
                          <p>• Target market: {aiInsights.geoAgent.markets.join(', ')}</p>
                          <p>• Regional compliance: {aiInsights.geoAgent.compliance.join(', ')} aware</p>
                          <p>• Local terminology: {aiInsights.geoAgent.localization}</p>
                          <p>• Cultural sensitivity: {aiInsights.geoAgent.sensitivity}</p>
                        </div>
                      </div>
                    </div>

                    {/* AI Confidence Summary */}
                    <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200">
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="font-medium text-sm text-gray-900">Overall AI Confidence</h5>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div className="bg-green-500 h-2 rounded-full" style={{ width: `${aiInsights.confidence}%` }}></div>
                          </div>
                          <span className="text-sm font-medium text-green-600">{aiInsights.confidence}%</span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-600">
                        {aiInsights.confidence >= 85 ? 'High confidence across all analysis dimensions. Content meets quality standards and brand guidelines.' :
                         aiInsights.confidence >= 70 ? 'Good confidence with some areas for improvement. Review recommendations below.' :
                         'Lower confidence detected. Manual review recommended for quality assurance.'}
                      </p>
                    </div>

                    {/* Quick AI Recommendations */}
                    <div className="mt-3 text-xs text-gray-600">
                      <strong className="text-gray-900">AI Recommendations:</strong>
                      <ul className="mt-1 ml-4 space-y-1">
                        {aiInsights.recommendations.map((rec, index) => (
                          <li key={index}>• {rec}</li>
                        ))}
                      </ul>
                    </div>

                  </div>
                );
              })()}

              {/* Review Criteria */}
              <div className="space-y-6">
                <div>
                  <h4 className="text-lg font-medium text-gray-900 mb-4">Manual Review Criteria</h4>
                  
                  {/* Brand Consistency */}
                  <div className="mb-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Shield className="w-5 h-5 text-purple-600" />
                        <h5 className="font-medium text-gray-900">Brand Consistency</h5>
                      </div>
                      <div className="flex items-center space-x-2">
                        {[1, 2, 3, 4, 5].map((rating) => (
                          <button
                            key={rating}
                            onClick={() => setReviewForm(prev => ({
                              ...prev,
                              brandConsistency: { ...prev.brandConsistency, score: rating }
                            }))}
                            className={`w-6 h-6 rounded-full border-2 ${
                              reviewForm.brandConsistency.score >= rating
                                ? 'bg-purple-600 border-purple-600'
                                : 'border-gray-300 hover:border-purple-400'
                            }`}
                          >
                            {reviewForm.brandConsistency.score >= rating && (
                              <span className="text-white text-xs">✓</span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">Voice, tone, terminology, and visual identity alignment</p>
                    <textarea
                      placeholder="Comments or feedback..."
                      className="w-full p-3 border border-gray-200 rounded-lg text-sm resize-none"
                      rows={2}
                      value={reviewForm.brandConsistency.comments}
                      onChange={(e) => setReviewForm(prev => ({
                        ...prev,
                        brandConsistency: { ...prev.brandConsistency, comments: e.target.value }
                      }))}
                    />
                  </div>

                  {/* Content Quality */}
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-5 h-5 text-blue-600" />
                        <h5 className="font-medium text-gray-900">Content Quality</h5>
                      </div>
                      <div className="flex items-center space-x-2">
                        {[1, 2, 3, 4, 5].map((rating) => (
                          <button
                            key={rating}
                            onClick={() => setReviewForm(prev => ({
                              ...prev,
                              contentQuality: { ...prev.contentQuality, score: rating }
                            }))}
                            className={`w-6 h-6 rounded-full border-2 ${
                              reviewForm.contentQuality.score >= rating
                                ? 'bg-blue-600 border-blue-600'
                                : 'border-gray-300 hover:border-blue-400'
                            }`}
                          >
                            {reviewForm.contentQuality.score >= rating && (
                              <span className="text-white text-xs">✓</span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">Accuracy, relevance, engagement, and structure</p>
                    <textarea
                      placeholder="Comments or feedback..."
                      className="w-full p-3 border border-gray-200 rounded-lg text-sm resize-none"
                      rows={2}
                      value={reviewForm.contentQuality.comments}
                      onChange={(e) => setReviewForm(prev => ({
                        ...prev,
                        contentQuality: { ...prev.contentQuality, comments: e.target.value }
                      }))}
                    />
                  </div>

                  {/* SEO Optimization */}
                  <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Search className="w-5 h-5 text-green-600" />
                        <h5 className="font-medium text-gray-900">SEO Optimization</h5>
                      </div>
                      <div className="flex items-center space-x-2">
                        {[1, 2, 3, 4, 5].map((rating) => (
                          <button
                            key={rating}
                            onClick={() => setReviewForm(prev => ({
                              ...prev,
                              seoOptimization: { ...prev.seoOptimization, score: rating }
                            }))}
                            className={`w-6 h-6 rounded-full border-2 ${
                              reviewForm.seoOptimization.score >= rating
                                ? 'bg-green-600 border-green-600'
                                : 'border-gray-300 hover:border-green-400'
                            }`}
                          >
                            {reviewForm.seoOptimization.score >= rating && (
                              <span className="text-white text-xs">✓</span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">Keywords, meta data, readability, and structure</p>
                    <textarea
                      placeholder="Comments or feedback..."
                      className="w-full p-3 border border-gray-200 rounded-lg text-sm resize-none"
                      rows={2}
                      value={reviewForm.seoOptimization.comments}
                      onChange={(e) => setReviewForm(prev => ({
                        ...prev,
                        seoOptimization: { ...prev.seoOptimization, comments: e.target.value }
                      }))}
                    />
                  </div>

                  {/* Compliance Check */}
                  <div className="mb-6 p-4 bg-orange-50 border border-orange-200 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <AlertCircle className="w-5 h-5 text-orange-600" />
                        <h5 className="font-medium text-gray-900">Compliance & Guidelines</h5>
                      </div>
                      <div className="flex items-center space-x-2">
                        {[1, 2, 3, 4, 5].map((rating) => (
                          <button
                            key={rating}
                            onClick={() => setReviewForm(prev => ({
                              ...prev,
                              complianceCheck: { ...prev.complianceCheck, score: rating }
                            }))}
                            className={`w-6 h-6 rounded-full border-2 ${
                              reviewForm.complianceCheck.score >= rating
                                ? 'bg-orange-600 border-orange-600'
                                : 'border-gray-300 hover:border-orange-400'
                            }`}
                          >
                            {reviewForm.complianceCheck.score >= rating && (
                              <span className="text-white text-xs">✓</span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">Legal, industry regulations, and platform guidelines</p>
                    <textarea
                      placeholder="Comments or feedback..."
                      className="w-full p-3 border border-gray-200 rounded-lg text-sm resize-none"
                      rows={2}
                      value={reviewForm.complianceCheck.comments}
                      onChange={(e) => setReviewForm(prev => ({
                        ...prev,
                        complianceCheck: { ...prev.complianceCheck, comments: e.target.value }
                      }))}
                    />
                  </div>
                </div>

                {/* Overall Rating & Comments */}
                <div className="pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h5 className="font-medium text-gray-900">Overall Rating</h5>
                    <div className="flex items-center space-x-1">
                      {[1, 2, 3, 4, 5].map((rating) => (
                        <button
                          key={rating}
                          onClick={() => setReviewForm(prev => ({ ...prev, overallRating: rating }))}
                          className={`w-8 h-8 rounded-full ${
                            reviewForm.overallRating >= rating
                              ? 'text-yellow-500 fill-current'
                              : 'text-gray-300 hover:text-yellow-400'
                          }`}
                        >
                          <Star className="w-full h-full" fill={reviewForm.overallRating >= rating ? 'currentColor' : 'none'} />
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <textarea
                    placeholder="Final comments and recommendations..."
                    className="w-full p-4 border border-gray-200 rounded-lg text-sm"
                    rows={3}
                    value={reviewForm.finalComments}
                    onChange={(e) => setReviewForm(prev => ({ ...prev, finalComments: e.target.value }))}
                  />
                </div>

                {/* Review Actions */}
                <div className="pt-4 border-t border-gray-200">
                  <div className="flex items-center space-x-3 mb-4">
                    <span className="text-sm font-medium text-gray-700">Decision:</span>
                    {[
                      { value: 'approve', label: 'Approve', color: 'green', icon: CheckCircle },
                      { value: 'request_revision', label: 'Request Revision', color: 'yellow', icon: RotateCcw },
                      { value: 'reject', label: 'Reject', color: 'red', icon: XCircle }
                    ].map((action) => (
                      <button
                        key={action.value}
                        onClick={() => setReviewForm(prev => ({ ...prev, action: action.value as any }))}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 transition-all ${
                          reviewForm.action === action.value
                            ? `border-${action.color}-500 bg-${action.color}-50 text-${action.color}-700`
                            : 'border-gray-200 text-gray-600 hover:border-gray-300'
                        }`}
                      >
                        <action.icon className="w-4 h-4" />
                        <span className="text-sm font-medium">{action.label}</span>
                      </button>
                    ))}
                  </div>
                  
                  <div className="flex justify-end space-x-3">
                    <button
                      onClick={() => setReviewModal({ isOpen: false, task: null, stage: '' })}
                      className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={submitStructuredReview}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Submit Review
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
} 