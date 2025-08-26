import React, { useState } from 'react';
import { X, AlertCircle, Lightbulb, Edit3, RefreshCw } from 'lucide-react';

interface RevisionFeedbackDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (feedback: any) => void;
  taskId: string;
  taskType: string;
  currentContent: string;
  isSubmitting?: boolean;
}

const commonIssues = [
  'Tone not aligned with brand voice',
  'Content too long or verbose',
  'Missing key messaging points',
  'Poor call-to-action',
  'Lacks engagement factor',
  'Grammar or spelling errors',
  'Factual inaccuracies',
  'Formatting issues',
  'Missing target audience consideration',
  'Weak opening or hook'
];

const improvementSuggestions = [
  'Add more specific examples',
  'Include industry statistics',
  'Strengthen the call-to-action',
  'Make it more conversational',
  'Add emotional hooks',
  'Include customer testimonials',
  'Use bullet points for clarity',
  'Add relevant hashtags',
  'Include compelling visuals description',
  'Focus on benefits over features'
];

export function RevisionFeedbackDialog({
  isOpen,
  onClose,
  onSubmit,
  taskId,
  taskType,
  currentContent,
  isSubmitting = false
}: RevisionFeedbackDialogProps) {
  const [feedback, setFeedback] = useState({
    type: 'content_improvement',
    issues: [] as string[],
    suggestions: [] as string[],
    quality_score: 50,
    notes: '',
    changes: [] as string[],
    priority: 'medium',
    revision_round: 1
  });

  const [customIssue, setCustomIssue] = useState('');
  const [customSuggestion, setCustomSuggestion] = useState('');
  const [customChange, setCustomChange] = useState('');

  if (!isOpen) return null;

  const handleIssueToggle = (issue: string) => {
    setFeedback(prev => ({
      ...prev,
      issues: prev.issues.includes(issue)
        ? prev.issues.filter(i => i !== issue)
        : [...prev.issues, issue]
    }));
  };

  const handleSuggestionToggle = (suggestion: string) => {
    setFeedback(prev => ({
      ...prev,
      suggestions: prev.suggestions.includes(suggestion)
        ? prev.suggestions.filter(s => s !== suggestion)
        : [...prev.suggestions, suggestion]
    }));
  };

  const addCustomIssue = () => {
    if (customIssue.trim()) {
      setFeedback(prev => ({
        ...prev,
        issues: [...prev.issues, customIssue.trim()]
      }));
      setCustomIssue('');
    }
  };

  const addCustomSuggestion = () => {
    if (customSuggestion.trim()) {
      setFeedback(prev => ({
        ...prev,
        suggestions: [...prev.suggestions, customSuggestion.trim()]
      }));
      setCustomSuggestion('');
    }
  };

  const addCustomChange = () => {
    if (customChange.trim()) {
      setFeedback(prev => ({
        ...prev,
        changes: [...prev.changes, customChange.trim()]
      }));
      setCustomChange('');
    }
  };

  const handleSubmit = () => {
    onSubmit(feedback);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Request Revision</h2>
            <p className="text-gray-600">Task: {taskType} (ID: {taskId.slice(0, 8)}...)</p>
          </div>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="text-gray-400 hover:text-gray-600 transition-colors disabled:opacity-50"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Current Content Preview */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Current Content</h3>
            <div className="text-sm text-gray-700 max-h-32 overflow-y-auto bg-white p-3 rounded border">
              {currentContent || 'No content available'}
            </div>
          </div>

          {/* Quality Score */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quality Score (1-100)
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min="1"
                max="100"
                value={feedback.quality_score}
                onChange={(e) => setFeedback(prev => ({ ...prev, quality_score: parseInt(e.target.value) }))}
                className="flex-1"
                disabled={isSubmitting}
              />
              <span className={`text-lg font-bold ${
                feedback.quality_score >= 80 ? 'text-green-600' :
                feedback.quality_score >= 60 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {feedback.quality_score}
              </span>
            </div>
          </div>

          {/* Priority */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Priority Level
            </label>
            <select
              value={feedback.priority}
              onChange={(e) => setFeedback(prev => ({ ...prev, priority: e.target.value }))}
              className="w-full border border-gray-300 rounded-lg px-3 py-2"
              disabled={isSubmitting}
            >
              <option value="low">Low - Minor adjustments needed</option>
              <option value="medium">Medium - Moderate improvements required</option>
              <option value="high">High - Significant changes needed</option>
              <option value="critical">Critical - Major revision required</option>
            </select>
          </div>

          {/* Issues Identified */}
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <h3 className="text-lg font-medium text-gray-900">Issues Identified</h3>
            </div>
            <div className="grid grid-cols-2 gap-2 mb-3">
              {commonIssues.map((issue) => (
                <label key={issue} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={feedback.issues.includes(issue)}
                    onChange={() => handleIssueToggle(issue)}
                    disabled={isSubmitting}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm text-gray-700">{issue}</span>
                </label>
              ))}
            </div>
            <div className="flex space-x-2">
              <input
                type="text"
                placeholder="Add custom issue..."
                value={customIssue}
                onChange={(e) => setCustomIssue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addCustomIssue()}
                disabled={isSubmitting}
                className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm"
              />
              <button
                onClick={addCustomIssue}
                disabled={!customIssue.trim() || isSubmitting}
                className="px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
              >
                Add
              </button>
            </div>
          </div>

          {/* Improvement Suggestions */}
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Lightbulb className="w-5 h-5 text-yellow-500" />
              <h3 className="text-lg font-medium text-gray-900">Improvement Suggestions</h3>
            </div>
            <div className="grid grid-cols-2 gap-2 mb-3">
              {improvementSuggestions.map((suggestion) => (
                <label key={suggestion} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={feedback.suggestions.includes(suggestion)}
                    onChange={() => handleSuggestionToggle(suggestion)}
                    disabled={isSubmitting}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm text-gray-700">{suggestion}</span>
                </label>
              ))}
            </div>
            <div className="flex space-x-2">
              <input
                type="text"
                placeholder="Add custom suggestion..."
                value={customSuggestion}
                onChange={(e) => setCustomSuggestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addCustomSuggestion()}
                disabled={isSubmitting}
                className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm"
              />
              <button
                onClick={addCustomSuggestion}
                disabled={!customSuggestion.trim() || isSubmitting}
                className="px-3 py-2 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 disabled:opacity-50"
              >
                Add
              </button>
            </div>
          </div>

          {/* Specific Changes Requested */}
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Edit3 className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-medium text-gray-900">Specific Changes Requested</h3>
            </div>
            {feedback.changes.length > 0 && (
              <div className="mb-3 space-y-1">
                {feedback.changes.map((change, index) => (
                  <div key={index} className="flex items-center justify-between bg-blue-50 p-2 rounded">
                    <span className="text-sm text-gray-700">{change}</span>
                    <button
                      onClick={() => setFeedback(prev => ({
                        ...prev,
                        changes: prev.changes.filter((_, i) => i !== index)
                      }))}
                      disabled={isSubmitting}
                      className="text-red-500 hover:text-red-700 disabled:opacity-50"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex space-x-2">
              <input
                type="text"
                placeholder="Describe specific change needed..."
                value={customChange}
                onChange={(e) => setCustomChange(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addCustomChange()}
                disabled={isSubmitting}
                className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm"
              />
              <button
                onClick={addCustomChange}
                disabled={!customChange.trim() || isSubmitting}
                className="px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50"
              >
                Add
              </button>
            </div>
          </div>

          {/* Additional Notes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Additional Notes
            </label>
            <textarea
              value={feedback.notes}
              onChange={(e) => setFeedback(prev => ({ ...prev, notes: e.target.value }))}
              placeholder="Any additional context or specific requirements..."
              disabled={isSubmitting}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 h-24 resize-none"
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            This feedback will help the AI agent learn and improve future content generation.
          </div>
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || (feedback.issues.length === 0 && feedback.suggestions.length === 0)}
              className="flex items-center space-x-2 px-6 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Edit3 className="w-4 h-4" />
              )}
              <span>{isSubmitting ? 'Submitting...' : 'Request Revision'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}