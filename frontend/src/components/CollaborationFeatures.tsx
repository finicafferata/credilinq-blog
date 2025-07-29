import { useState } from 'react';
import { StatusIndicator } from './LoadingStates';

export interface Comment {
  id: string;
  author: string;
  content: string;
  timestamp: Date;
  resolved: boolean;
  position?: {
    start: number;
    end: number;
    selectedText: string;
  };
  replies?: Comment[];
}

export interface Suggestion {
  id: string;
  author: string;
  originalText: string;
  suggestedText: string;
  reason: string;
  timestamp: Date;
  status: 'pending' | 'accepted' | 'rejected';
  position: {
    start: number;
    end: number;
  };
}

interface CollaborationPanelProps {
  comments: Comment[];
  suggestions: Suggestion[];
  onAddComment: (content: string, position?: Comment['position']) => void;
  onResolveComment: (commentId: string) => void;
  onReplyToComment: (commentId: string, content: string) => void;
  onAcceptSuggestion: (suggestionId: string) => void;
  onRejectSuggestion: (suggestionId: string) => void;
  selectedText?: string;
  selectionPosition?: { start: number; end: number };
}

export function CollaborationPanel({
  comments,
  suggestions,
  onAddComment,
  onResolveComment,
  onReplyToComment,
  onAcceptSuggestion,
  onRejectSuggestion,
  selectedText,
  selectionPosition
}: CollaborationPanelProps) {
  const [activeTab, setActiveTab] = useState<'comments' | 'suggestions'>('comments');
  const [newComment, setNewComment] = useState('');
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyContent, setReplyContent] = useState('');
  const [showNewCommentForm, setShowNewCommentForm] = useState(false);

  const unresolvedComments = comments.filter(c => !c.resolved);
  const resolvedComments = comments.filter(c => c.resolved);
  const pendingSuggestions = suggestions.filter(s => s.status === 'pending');
  const reviewedSuggestions = suggestions.filter(s => s.status !== 'pending');

  const handleAddComment = () => {
    if (!newComment.trim()) return;

    const position = selectedText && selectionPosition ? {
      start: selectionPosition.start,
      end: selectionPosition.end,
      selectedText
    } : undefined;

    onAddComment(newComment.trim(), position);
    setNewComment('');
    setShowNewCommentForm(false);
  };

  const handleReply = (commentId: string) => {
    if (!replyContent.trim()) return;
    onReplyToComment(commentId, replyContent.trim());
    setReplyContent('');
    setReplyingTo(null);
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return timestamp.toLocaleDateString();
  };

  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Collaboration</h3>
        
        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('comments')}
            className={`flex-1 py-2 px-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'comments'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Comments ({unresolvedComments.length})
          </button>
          <button
            onClick={() => setActiveTab('suggestions')}
            className={`flex-1 py-2 px-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'suggestions'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Suggestions ({pendingSuggestions.length})
          </button>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="p-4 bg-gray-50 border-b border-gray-200">
        {selectedText ? (
          <div className="mb-3">
            <div className="text-xs text-gray-600 mb-2">Selected text:</div>
            <div className="bg-blue-50 border border-blue-200 rounded p-2 text-sm">
              "{selectedText.length > 50 ? selectedText.substring(0, 50) + '...' : selectedText}"
            </div>
          </div>
        ) : null}
        
        <button
          onClick={() => setShowNewCommentForm(true)}
          className="w-full btn-primary text-sm"
        >
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Add Comment
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'comments' ? (
          <div className="p-4 space-y-4">
            {/* New Comment Form */}
            {showNewCommentForm && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <textarea
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  placeholder={selectedText ? "Comment on selected text..." : "Add a general comment..."}
                  className="w-full p-2 border border-gray-300 rounded text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary-500"
                  rows={3}
                />
                <div className="flex justify-end space-x-2 mt-2">
                  <button
                    onClick={() => {
                      setShowNewCommentForm(false);
                      setNewComment('');
                    }}
                    className="px-3 py-1 text-xs text-gray-600 hover:text-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleAddComment}
                    disabled={!newComment.trim()}
                    className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
                  >
                    Add Comment
                  </button>
                </div>
              </div>
            )}

            {/* Unresolved Comments */}
            {unresolvedComments.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Active Comments</h4>
                {unresolvedComments.map((comment) => (
                  <CommentItem
                    key={comment.id}
                    comment={comment}
                    onResolve={onResolveComment}
                    onReply={onReplyToComment}
                    replyingTo={replyingTo}
                    setReplyingTo={setReplyingTo}
                    replyContent={replyContent}
                    setReplyContent={setReplyContent}
                    handleReply={handleReply}
                    formatTimestamp={formatTimestamp}
                  />
                ))}
              </div>
            )}

            {/* Resolved Comments */}
            {resolvedComments.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-500 mb-2">Resolved ({resolvedComments.length})</h4>
                <div className="space-y-2">
                  {resolvedComments.map((comment) => (
                    <div key={comment.id} className="bg-gray-50 rounded p-2 opacity-75">
                      <div className="text-xs text-gray-500 mb-1">
                        {comment.author} • {formatTimestamp(comment.timestamp)} • Resolved
                      </div>
                      <div className="text-sm text-gray-700">{comment.content}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {unresolvedComments.length === 0 && !showNewCommentForm && (
              <div className="text-center py-8">
                <div className="w-12 h-12 mx-auto mb-3 bg-gray-100 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-500">No active comments</p>
                <p className="text-xs text-gray-400 mt-1">Select text to add contextual comments</p>
              </div>
            )}
          </div>
        ) : (
          <div className="p-4 space-y-4">
            {/* Pending Suggestions */}
            {pendingSuggestions.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Pending Suggestions</h4>
                {pendingSuggestions.map((suggestion) => (
                  <SuggestionItem
                    key={suggestion.id}
                    suggestion={suggestion}
                    onAccept={onAcceptSuggestion}
                    onReject={onRejectSuggestion}
                    formatTimestamp={formatTimestamp}
                  />
                ))}
              </div>
            )}

            {/* Reviewed Suggestions */}
            {reviewedSuggestions.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-500 mb-2">Reviewed ({reviewedSuggestions.length})</h4>
                <div className="space-y-2">
                  {reviewedSuggestions.map((suggestion) => (
                    <div key={suggestion.id} className="bg-gray-50 rounded p-3 opacity-75">
                      <div className="text-xs text-gray-500 mb-1">
                        {suggestion.author} • {formatTimestamp(suggestion.timestamp)} • {suggestion.status}
                      </div>
                      <div className="text-sm">
                        <div className="text-red-600 line-through">{suggestion.originalText}</div>
                        <div className="text-green-600">{suggestion.suggestedText}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {pendingSuggestions.length === 0 && (
              <div className="text-center py-8">
                <div className="w-12 h-12 mx-auto mb-3 bg-gray-100 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-500">No pending suggestions</p>
                <p className="text-xs text-gray-400 mt-1">AI suggestions will appear here</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface CommentItemProps {
  comment: Comment;
  onResolve: (commentId: string) => void;
  onReply: (commentId: string, content: string) => void;
  replyingTo: string | null;
  setReplyingTo: (id: string | null) => void;
  replyContent: string;
  setReplyContent: (content: string) => void;
  handleReply: (commentId: string) => void;
  formatTimestamp: (timestamp: Date) => string;
}

function CommentItem({
  comment,
  onResolve,
  replyingTo,
  setReplyingTo,
  replyContent,
  setReplyContent,
  handleReply,
  formatTimestamp
}: CommentItemProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-3 mb-3">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-sm font-medium text-gray-900">{comment.author}</div>
          <div className="text-xs text-gray-500">{formatTimestamp(comment.timestamp)}</div>
        </div>
        <button
          onClick={() => onResolve(comment.id)}
          className="text-xs text-green-600 hover:text-green-800"
        >
          Resolve
        </button>
      </div>
      
      {comment.position && (
        <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-xs">
          <div className="text-blue-600 font-medium mb-1">Commenting on:</div>
          <div className="text-blue-800">"{comment.position.selectedText}"</div>
        </div>
      )}
      
      <div className="text-sm text-gray-700 mb-2">{comment.content}</div>
      
      {comment.replies && comment.replies.length > 0 && (
        <div className="ml-4 border-l-2 border-gray-200 pl-3 space-y-2">
          {comment.replies.map((reply) => (
            <div key={reply.id} className="bg-gray-50 rounded p-2">
              <div className="text-xs text-gray-500 mb-1">
                {reply.author} • {formatTimestamp(reply.timestamp)}
              </div>
              <div className="text-sm text-gray-700">{reply.content}</div>
            </div>
          ))}
        </div>
      )}
      
      <div className="flex items-center space-x-2 mt-2">
        <button
          onClick={() => setReplyingTo(replyingTo === comment.id ? null : comment.id)}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          Reply
        </button>
      </div>
      
      {replyingTo === comment.id && (
        <div className="mt-2">
          <textarea
            value={replyContent}
            onChange={(e) => setReplyContent(e.target.value)}
            placeholder="Write a reply..."
            className="w-full p-2 border border-gray-300 rounded text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary-500"
            rows={2}
          />
          <div className="flex justify-end space-x-2 mt-1">
            <button
              onClick={() => setReplyingTo(null)}
              className="px-2 py-1 text-xs text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              onClick={() => handleReply(comment.id)}
              disabled={!replyContent.trim()}
              className="px-2 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
            >
              Reply
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

interface SuggestionItemProps {
  suggestion: Suggestion;
  onAccept: (suggestionId: string) => void;
  onReject: (suggestionId: string) => void;
  formatTimestamp: (timestamp: Date) => string;
}

function SuggestionItem({ suggestion, onAccept, onReject, formatTimestamp }: SuggestionItemProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-3 mb-3">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-sm font-medium text-gray-900">{suggestion.author}</div>
          <div className="text-xs text-gray-500">{formatTimestamp(suggestion.timestamp)}</div>
        </div>
        <StatusIndicator status="info" size="sm" />
      </div>
      
      <div className="mb-3">
        <div className="text-xs text-gray-600 mb-1">Suggested change:</div>
        <div className="bg-red-50 border border-red-200 rounded p-2 mb-1">
          <div className="text-xs text-red-600 mb-1">Remove:</div>
          <div className="text-sm text-red-800 line-through">{suggestion.originalText}</div>
        </div>
        <div className="bg-green-50 border border-green-200 rounded p-2">
          <div className="text-xs text-green-600 mb-1">Replace with:</div>
          <div className="text-sm text-green-800">{suggestion.suggestedText}</div>
        </div>
      </div>
      
      {suggestion.reason && (
        <div className="mb-3 p-2 bg-blue-50 border border-blue-200 rounded">
          <div className="text-xs text-blue-600 mb-1">Reason:</div>
          <div className="text-sm text-blue-800">{suggestion.reason}</div>
        </div>
      )}
      
      <div className="flex justify-end space-x-2">
        <button
          onClick={() => onReject(suggestion.id)}
          className="px-3 py-1 text-xs border border-gray-300 text-gray-700 rounded hover:bg-gray-50"
        >
          Reject
        </button>
        <button
          onClick={() => onAccept(suggestion.id)}
          className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
        >
          Accept
        </button>
      </div>
    </div>
  );
}