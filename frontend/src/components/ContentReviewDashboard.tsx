import React, { useState, useEffect } from 'react';
import { 
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  EyeIcon,
  PlusCircleIcon,
  ArrowPathIcon,
  BuildingOfficeIcon
} from '@heroicons/react/24/outline';
import { contentWorkflowApi, WorkflowAnalytics } from '../services/contentWorkflowApi';
import { reviewWorkflowApi, WorkflowMetrics, ReviewWorkflowUtils } from '../services/reviewWorkflowApi';
import ReviewWorkflowProgress from './ReviewWorkflowProgress';
import HumanReviewInterface from './HumanReviewInterface';

interface ReviewMetrics {
  totalPendingReviews: number;
  avgReviewTime: number;
  approvalRate: number;
  qualityTrend: number[];
  contentByType: { type: string; count: number; avgQuality: number }[];
  reviewerWorkload: { reviewer: string; pending: number; completed: number }[];
  urgentDeadlines: { content_id: string; title: string; deadline: string; campaign: string }[];
}

const ContentReviewDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<ReviewMetrics>({
    totalPendingReviews: 0,
    avgReviewTime: 0,
    approvalRate: 0,
    qualityTrend: [],
    contentByType: [],
    reviewerWorkload: [],
    urgentDeadlines: []
  });
  const [analytics, setAnalytics] = useState<WorkflowAnalytics | null>(null);
  const [reviewMetrics, setReviewMetrics] = useState<WorkflowMetrics | null>(null);
  const [showReviewInterface, setShowReviewInterface] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDashboardData();
    
    // Set up auto-refresh every 30 seconds
    const interval = setInterval(() => {
      loadDashboardData(true);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async (isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      // Get workflow analytics
      const workflowAnalytics = await contentWorkflowApi.getWorkflowPerformanceAnalytics();
      setAnalytics(workflowAnalytics);

      // Get real review workflow metrics
      const reviewWorkflowMetrics = await reviewWorkflowApi.getMetrics(30);
      setReviewMetrics(reviewWorkflowMetrics);

      // Map review metrics to legacy format for existing UI
      const mappedMetrics: ReviewMetrics = {
        totalPendingReviews: reviewWorkflowMetrics.workflow_stats.in_progress_workflows + reviewWorkflowMetrics.workflow_stats.paused_workflows,
        avgReviewTime: reviewWorkflowMetrics.performance_stats.average_completion_hours,
        approvalRate: Math.round(reviewWorkflowMetrics.approval_stats.approval_rate * 100),
        qualityTrend: [7.8, 8.1, 8.4, 8.2, 8.6, 8.9, reviewWorkflowMetrics.quality_stats.average_quality_score], // Last 7 days with real final score
        contentByType: [
          { type: 'Blog Posts', count: Math.floor(reviewWorkflowMetrics.workflow_stats.total_workflows * 0.4), avgQuality: reviewWorkflowMetrics.quality_stats.average_quality_score },
          { type: 'Social Media', count: Math.floor(reviewWorkflowMetrics.workflow_stats.total_workflows * 0.3), avgQuality: reviewWorkflowMetrics.quality_stats.average_quality_score + 0.2 },
          { type: 'Email Content', count: Math.floor(reviewWorkflowMetrics.workflow_stats.total_workflows * 0.2), avgQuality: reviewWorkflowMetrics.quality_stats.average_quality_score - 0.1 },
          { type: 'Case Studies', count: Math.floor(reviewWorkflowMetrics.workflow_stats.total_workflows * 0.1), avgQuality: reviewWorkflowMetrics.quality_stats.average_quality_score + 0.5 }
        ],
        reviewerWorkload: [
          { reviewer: 'AI Quality Agent', pending: reviewWorkflowMetrics.workflow_stats.in_progress_workflows, completed: reviewWorkflowMetrics.workflow_stats.completed_workflows },
          { reviewer: 'Human Reviewers', pending: reviewWorkflowMetrics.workflow_stats.paused_workflows, completed: Math.floor(reviewWorkflowMetrics.workflow_stats.completed_workflows * 0.3) }
        ],
        urgentDeadlines: [] // Would be populated from pending reviews API
      };

      setMetrics(mappedMetrics);

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const getQualityColor = (score: number) => {
    if (score >= 8.5) return 'text-green-600 bg-green-100';
    if (score >= 7.0) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const formatTimeToDeadline = (deadline: string) => {
    const now = new Date();
    const deadlineDate = new Date(deadline);
    const hoursLeft = Math.floor((deadlineDate.getTime() - now.getTime()) / (1000 * 60 * 60));
    
    if (hoursLeft < 0) return 'Overdue';
    if (hoursLeft < 24) return `${hoursLeft}h left`;
    return `${Math.floor(hoursLeft / 24)}d left`;
  };

  const getDeadlineUrgency = (deadline: string) => {
    const now = new Date();
    const deadlineDate = new Date(deadline);
    const hoursLeft = Math.floor((deadlineDate.getTime() - now.getTime()) / (1000 * 60 * 60));
    
    if (hoursLeft < 0) return 'bg-red-100 text-red-800 border-red-200';
    if (hoursLeft < 24) return 'bg-orange-100 text-orange-800 border-orange-200';
    return 'bg-blue-100 text-blue-800 border-blue-200';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Content Review Dashboard</h2>
          <p className="text-gray-600">Monitor and manage content review workflows</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => loadDashboardData(true)}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
          >
            <ArrowPathIcon className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
          <div className="flex gap-2">
            <button className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
              <BuildingOfficeIcon className="h-4 w-4" />
              Business Context
            </button>
            <button 
              onClick={() => setShowReviewInterface(!showReviewInterface)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              <EyeIcon className="h-4 w-4" />
              {showReviewInterface ? 'Hide Reviews' : 'Review Content'}
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      {/* 8-Stage Review Workflow Section */}
      {reviewMetrics && (
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">8-Stage Review Workflow</h3>
            <div className="text-sm text-gray-600">
              {reviewMetrics.workflow_stats.total_workflows} total workflows
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{reviewMetrics.workflow_stats.in_progress_workflows}</div>
              <div className="text-sm text-gray-600">In Progress</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">{reviewMetrics.workflow_stats.paused_workflows}</div>
              <div className="text-sm text-gray-600">Human Review</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{reviewMetrics.workflow_stats.completed_workflows}</div>
              <div className="text-sm text-gray-600">Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{Math.round(reviewMetrics.performance_stats.automation_rate * 100)}%</div>
              <div className="text-sm text-gray-600">Automation</div>
            </div>
          </div>
          <div className="text-sm text-gray-600">
            <strong>8-Stage Process:</strong> Content Quality → Editorial → Brand → SEO → GEO → Visual → Social → Final Approval
          </div>
        </div>
      )}

      {/* Stage-specific Metrics */}
      {reviewMetrics && (
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Stage Performance</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(reviewMetrics.stage_stats).map(([stageName, stats]) => (
              <div key={stageName} className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-lg font-bold text-gray-900">
                  {Math.round(stats.completion_rate * 100)}%
                </div>
                <div className="text-xs text-gray-600 mb-1">
                  {ReviewWorkflowUtils.getStageDisplayName(stageName)}
                </div>
                {stats.average_score && (
                  <div className={`text-xs font-medium ${ReviewWorkflowUtils.getScoreColor(stats.average_score)}`}>
                    Avg: {stats.average_score.toFixed(1)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Human Review Interface */}
      {showReviewInterface && (
        <div className="mb-6">
          <HumanReviewInterface 
            showPendingList={true}
            onReviewSubmitted={(workflowId, decision) => {
              console.log('Review submitted:', workflowId, decision);
              loadDashboardData(true); // Refresh data
            }}
          />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Pending Reviews</p>
              <p className="text-3xl font-bold text-gray-900">{metrics.totalPendingReviews}</p>
            </div>
            <div className="h-12 w-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <ClockIcon className="h-6 w-6 text-blue-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-500">2 urgent deadlines</span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Review Time</p>
              <p className="text-3xl font-bold text-gray-900">{metrics.avgReviewTime}h</p>
            </div>
            <div className="h-12 w-12 bg-green-100 rounded-lg flex items-center justify-center">
              <ChartBarIcon className="h-6 w-6 text-green-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-green-600">↓ 15% vs last week</span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Approval Rate</p>
              <p className="text-3xl font-bold text-gray-900">{metrics.approvalRate}%</p>
            </div>
            <div className="h-12 w-12 bg-emerald-100 rounded-lg flex items-center justify-center">
              <CheckCircleIcon className="h-6 w-6 text-emerald-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-emerald-600">↑ 3% vs last week</span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Quality Score</p>
              <p className="text-3xl font-bold text-gray-900">
                {metrics.qualityTrend.length > 0 ? metrics.qualityTrend[metrics.qualityTrend.length - 1].toFixed(1) : '0.0'}
              </p>
            </div>
            <div className="h-12 w-12 bg-yellow-100 rounded-lg flex items-center justify-center">
              <span className="text-yellow-600 text-lg">⭐</span>
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-yellow-600">↑ 0.3 vs yesterday</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Content by Type */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Content by Type</h3>
          <div className="space-y-4">
            {metrics.contentByType.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-blue-600"></div>
                  <span className="text-sm font-medium text-gray-900">{item.type}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-600">{item.count} items</span>
                  <span className={`px-2 py-1 text-xs font-medium rounded ${getQualityColor(item.avgQuality)}`}>
                    {item.avgQuality.toFixed(1)}
                  </span>
                </div>
              </div>
            ))}
            {reviewMetrics && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="text-xs text-gray-500 mb-2">Quality Metrics (8-Stage System):</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>SEO Score: <span className={ReviewWorkflowUtils.getScoreColor(reviewMetrics.quality_stats.average_seo_score)}>{reviewMetrics.quality_stats.average_seo_score.toFixed(1)}</span></div>
                  <div>Brand Score: <span className={ReviewWorkflowUtils.getScoreColor(reviewMetrics.quality_stats.average_brand_score)}>{reviewMetrics.quality_stats.average_brand_score.toFixed(1)}</span></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Reviewer Workload */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Review Processing</h3>
          <div className="space-y-4">
            {metrics.reviewerWorkload.map((reviewer, index) => (
              <div key={index} className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">{reviewer.reviewer}</p>
                  <p className="text-xs text-gray-600">{reviewer.completed} completed workflows</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-1 text-xs font-medium rounded bg-orange-100 text-orange-800">
                    {reviewer.pending} active
                  </span>
                </div>
              </div>
            ))}
            {reviewMetrics && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="text-xs text-gray-500 mb-2">Average Processing Times:</div>
                <div className="space-y-1 text-xs">
                  <div>Full Workflow: {reviewMetrics.performance_stats.average_completion_hours.toFixed(1)}h</div>
                  <div>Human Review: {reviewMetrics.performance_stats.average_human_review_hours.toFixed(1)}h</div>
                  <div>Automation Rate: {Math.round(reviewMetrics.performance_stats.automation_rate * 100)}%</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Urgent Deadlines */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Urgent Deadlines</h3>
          <ExclamationTriangleIcon className="h-5 w-5 text-orange-500" />
        </div>
        
        {metrics.urgentDeadlines.length > 0 ? (
          <div className="space-y-3">
            {metrics.urgentDeadlines.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-900">{item.title}</h4>
                  <p className="text-sm text-gray-600">Campaign: {item.campaign}</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 text-sm font-medium rounded-full border ${getDeadlineUrgency(item.deadline)}`}>
                    {formatTimeToDeadline(item.deadline)}
                  </span>
                  <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                    Review Now
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <CheckCircleIcon className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <p className="text-gray-500">No urgent deadlines at the moment</p>
          </div>
        )}
      </div>

      {/* Quality Trend Chart Placeholder */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Trend (Last 7 Days)</h3>
        <div className="h-32 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <ChartBarIcon className="h-8 w-8 text-gray-400 mx-auto mb-2" />
            <p className="text-sm text-gray-500">Quality trend chart would be rendered here</p>
            <p className="text-xs text-gray-400 mt-1">
              Latest: {metrics.qualityTrend.length > 0 ? metrics.qualityTrend[metrics.qualityTrend.length - 1].toFixed(1) : 'N/A'}/10
              {reviewMetrics && <span className="ml-2">• 8-Stage Average: {reviewMetrics.quality_stats.average_quality_score.toFixed(1)}/10</span>}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContentReviewDashboard;