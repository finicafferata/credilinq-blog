import React, { useState, useEffect } from 'react';
import {
  CheckCircleIcon,
  XCircleIcon,
  PencilIcon,
  StarIcon,
  ClockIcon,
  UserIcon,
  ChatBubbleLeftRightIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { reviewWorkflowApi, PendingReview, HumanReviewDecision, ReviewWorkflowUtils } from '../services/reviewWorkflowApi';

interface HumanReviewInterfaceProps {
  workflowId?: string; // If provided, show review for specific workflow
  stage?: string; // If provided, filter to specific stage
  onReviewSubmitted?: (workflowId: string, decision: HumanReviewDecision) => void;
  showPendingList?: boolean; // Whether to show pending reviews list
}

const HumanReviewInterface: React.FC<HumanReviewInterfaceProps> = ({
  workflowId,
  stage,
  onReviewSubmitted,
  showPendingList = true
}) => {
  const [pendingReviews, setPendingReviews] = useState<PendingReview[]>([]);
  const [selectedReview, setSelectedReview] = useState<PendingReview | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Review form state
  const [reviewDecision, setReviewDecision] = useState<'approved' | 'rejected' | 'needs_revision'>('approved');
  const [reviewScore, setReviewScore] = useState<number>(8);
  const [reviewFeedback, setReviewFeedback] = useState<string>('');
  const [suggestions, setSuggestions] = useState<string[]>(['']);
  const [revisionRequests, setRevisionRequests] = useState<string[]>(['']);

  useEffect(() => {
    loadPendingReviews();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadPendingReviews, 30000);
    return () => clearInterval(interval);
  }, [workflowId, stage]);

  const loadPendingReviews = async () => {
    try {
      setError(null);
      const response = await reviewWorkflowApi.getPendingReviews({
        stage,
        limit: 50
      });
      
      let reviews = response.pending_reviews;
      
      // If workflowId is specified, filter to that workflow
      if (workflowId) {
        reviews = reviews.filter(review => review.workflow_execution_id === workflowId);
      }
      
      setPendingReviews(reviews);
      
      // Auto-select first review if none selected
      if (reviews.length > 0 && !selectedReview) {
        setSelectedReview(reviews[0]);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pending reviews');
      console.error('Error loading pending reviews:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReviewSubmit = async () => {
    if (!selectedReview) return;

    try {
      setSubmitting(true);
      setError(null);

      const decision: HumanReviewDecision = {
        stage: selectedReview.stage,
        reviewer_id: 'current_user', // This would come from auth context
        status: reviewDecision,
        score: reviewScore,
        feedback: reviewFeedback,
        suggestions: suggestions.filter(s => s.trim() !== ''),
        revision_requests: reviewDecision === 'needs_revision' 
          ? revisionRequests.filter(r => r.trim() !== '') 
          : undefined
      };

      await reviewWorkflowApi.submitHumanReview(selectedReview.workflow_execution_id, decision);
      
      // Notify parent component
      onReviewSubmitted?.(selectedReview.workflow_execution_id, decision);
      
      // Reset form and remove from pending list
      resetForm();
      setPendingReviews(prev => prev.filter(r => 
        r.workflow_execution_id !== selectedReview.workflow_execution_id || 
        r.stage !== selectedReview.stage
      ));
      setSelectedReview(null);
      
      // Select next review if available
      const remainingReviews = pendingReviews.filter(r => 
        r.workflow_execution_id !== selectedReview.workflow_execution_id || 
        r.stage !== selectedReview.stage
      );
      if (remainingReviews.length > 0) {
        setSelectedReview(remainingReviews[0]);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit review');
      console.error('Error submitting review:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setReviewDecision('approved');
    setReviewScore(8);
    setReviewFeedback('');
    setSuggestions(['']);
    setRevisionRequests(['']);
  };

  const addSuggestion = () => {
    setSuggestions([...suggestions, '']);
  };

  const updateSuggestion = (index: number, value: string) => {
    const updated = [...suggestions];
    updated[index] = value;
    setSuggestions(updated);
  };

  const removeSuggestion = (index: number) => {
    setSuggestions(suggestions.filter((_, i) => i !== index));
  };

  const addRevisionRequest = () => {
    setRevisionRequests([...revisionRequests, '']);
  };

  const updateRevisionRequest = (index: number, value: string) => {
    const updated = [...revisionRequests];
    updated[index] = value;
    setRevisionRequests(updated);
  };

  const removeRevisionRequest = (index: number) => {
    setRevisionRequests(revisionRequests.filter((_, i) => i !== index));
  };

  const getUrgencyColor = (review: PendingReview) => {
    if (review.is_overdue) return 'border-red-200 bg-red-50';
    const hoursLeft = Math.floor(
      (new Date(review.expected_completion_at).getTime() - new Date().getTime()) / (1000 * 60 * 60)
    );
    if (hoursLeft < 24) return 'border-orange-200 bg-orange-50';
    return 'border-blue-200 bg-blue-50';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading reviews...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center gap-3 text-red-800">
          <ExclamationTriangleIcon className="h-6 w-6" />
          <div>
            <h3 className="font-semibold">Error Loading Reviews</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
        <button
          onClick={loadPendingReviews}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (pendingReviews.length === 0) {
    return (
      <div className="text-center py-12">
        <CheckCircleIcon className="h-16 w-16 text-green-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">All Caught Up!</h3>
        <p className="text-gray-600">No pending reviews at this time.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Pending Reviews List */}
      {showPendingList && (
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-4 border-b">
              <h3 className="text-lg font-semibold text-gray-900">
                Pending Reviews ({pendingReviews.length})
              </h3>
            </div>
            <div className="max-h-96 overflow-y-auto">
              {pendingReviews.map((review, index) => (
                <div
                  key={`${review.workflow_execution_id}-${review.stage}`}
                  className={`p-4 border-b cursor-pointer hover:bg-gray-50 transition-colors ${
                    selectedReview?.workflow_execution_id === review.workflow_execution_id &&
                    selectedReview?.stage === review.stage
                      ? 'bg-blue-50 border-l-4 border-l-blue-500'
                      : ''
                  } ${getUrgencyColor(review)}`}
                  onClick={() => setSelectedReview(review)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 text-sm mb-1">
                        {review.title}
                      </h4>
                      <div className="flex items-center gap-2 text-xs text-gray-600 mb-2">
                        <span className="bg-gray-100 px-2 py-1 rounded">
                          {ReviewWorkflowUtils.getStageDisplayName(review.stage)}
                        </span>
                        <span>{review.content_type}</span>
                        {review.priority === 'high' && (
                          <span className="bg-red-100 text-red-800 px-2 py-1 rounded">
                            High Priority
                          </span>
                        )}
                      </div>
                      {review.automated_score && (
                        <div className="flex items-center gap-1 text-xs">
                          <StarIcon className="h-3 w-3 text-yellow-500" />
                          <span className={ReviewWorkflowUtils.getScoreColor(review.automated_score)}>
                            {ReviewWorkflowUtils.formatScore(review.automated_score)}
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      {review.is_overdue ? (
                        <span className="text-red-600 font-medium">Overdue</span>
                      ) : (
                        <ClockIcon className="h-4 w-4" />
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Review Form */}
      <div className={showPendingList ? 'lg:col-span-2' : 'col-span-1'}>
        {selectedReview ? (
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900 mb-2">
                    {selectedReview.title}
                  </h2>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span className="flex items-center gap-1">
                      <UserIcon className="h-4 w-4" />
                      {ReviewWorkflowUtils.getStageDisplayName(selectedReview.stage)}
                    </span>
                    <span>Content ID: {selectedReview.content_id}</span>
                    {selectedReview.automated_score && (
                      <span className="flex items-center gap-1">
                        <StarIcon className="h-4 w-4 text-yellow-500" />
                        AI Score: {ReviewWorkflowUtils.formatScore(selectedReview.automated_score)}
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">Assigned</div>
                  <div className="text-sm font-medium">
                    {new Date(selectedReview.assigned_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            </div>

            <form onSubmit={(e) => { e.preventDefault(); handleReviewSubmit(); }} className="p-6">
              {/* Decision Buttons */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Review Decision
                </label>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => setReviewDecision('approved')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-colors ${
                      reviewDecision === 'approved'
                        ? 'border-green-500 bg-green-50 text-green-800'
                        : 'border-gray-200 hover:border-green-300'
                    }`}
                  >
                    <CheckCircleIcon className="h-5 w-5" />
                    Approve
                  </button>
                  <button
                    type="button"
                    onClick={() => setReviewDecision('needs_revision')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-colors ${
                      reviewDecision === 'needs_revision'
                        ? 'border-yellow-500 bg-yellow-50 text-yellow-800'
                        : 'border-gray-200 hover:border-yellow-300'
                    }`}
                  >
                    <PencilIcon className="h-5 w-5" />
                    Needs Revision
                  </button>
                  <button
                    type="button"
                    onClick={() => setReviewDecision('rejected')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-colors ${
                      reviewDecision === 'rejected'
                        ? 'border-red-500 bg-red-50 text-red-800'
                        : 'border-gray-200 hover:border-red-300'
                    }`}
                  >
                    <XCircleIcon className="h-5 w-5" />
                    Reject
                  </button>
                </div>
              </div>

              {/* Score */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Quality Score (0-10)
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.1"
                    value={reviewScore}
                    onChange={(e) => setReviewScore(parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <div className={`text-lg font-bold ${ReviewWorkflowUtils.getScoreColor(reviewScore)}`}>
                    {reviewScore.toFixed(1)}
                  </div>
                </div>
              </div>

              {/* Feedback */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Review Feedback
                </label>
                <textarea
                  value={reviewFeedback}
                  onChange={(e) => setReviewFeedback(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                  placeholder="Provide detailed feedback about the content..."
                  required
                />
              </div>

              {/* Suggestions */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Suggestions for Improvement
                  </label>
                  <button
                    type="button"
                    onClick={addSuggestion}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    + Add Suggestion
                  </button>
                </div>
                {suggestions.map((suggestion, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={suggestion}
                      onChange={(e) => updateSuggestion(index, e.target.value)}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="Enter a suggestion..."
                    />
                    {suggestions.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeSuggestion(index)}
                        className="px-3 py-2 text-red-600 hover:text-red-800"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* Revision Requests (only for needs_revision) */}
              {reviewDecision === 'needs_revision' && (
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-gray-700">
                      Specific Revision Requests
                    </label>
                    <button
                      type="button"
                      onClick={addRevisionRequest}
                      className="text-sm text-blue-600 hover:text-blue-800"
                    >
                      + Add Request
                    </button>
                  </div>
                  {revisionRequests.map((request, index) => (
                    <div key={index} className="flex gap-2 mb-2">
                      <input
                        type="text"
                        value={request}
                        onChange={(e) => updateRevisionRequest(index, e.target.value)}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Enter a specific revision request..."
                      />
                      {revisionRequests.length > 1 && (
                        <button
                          type="button"
                          onClick={() => removeRevisionRequest(index)}
                          className="px-3 py-2 text-red-600 hover:text-red-800"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Submit Button */}
              <div className="flex items-center justify-end gap-3">
                <button
                  type="button"
                  onClick={resetForm}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
                  disabled={submitting}
                >
                  Reset
                </button>
                <button
                  type="submit"
                  disabled={submitting || !reviewFeedback.trim()}
                  className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {submitting ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Submitting...
                    </>
                  ) : (
                    <>
                      <ChatBubbleLeftRightIcon className="h-4 w-4" />
                      Submit Review
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
            <InformationCircleIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Select a Review
            </h3>
            <p className="text-gray-600">
              Choose a pending review from the list to get started.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default HumanReviewInterface;