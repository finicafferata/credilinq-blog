import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { blogApi } from '../lib/api';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';
import type { BlogDetail } from '../lib/api';
import { Breadcrumbs } from '../components/Breadcrumbs';
import { KeyboardShortcutsHelp } from '../components/KeyboardShortcutsHelp';
import { SplitViewEditor } from '../components/MarkdownPreview';
import { CollaborationPanel, type Comment, type Suggestion } from '../components/CollaborationFeatures';
import { useAutoSave, useAutoSaveStatus } from '../hooks/useAutoSave';
import { StatusIndicator, LoadingSpinner } from '../components/LoadingStates';

export function EnhancedBlogEditor() {
  const { blogId } = useParams<{ blogId: string }>();
  const navigate = useNavigate();
  
  // Blog state
  const [blog, setBlog] = useState<BlogDetail | null>(null);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  
  // Editor state
  const [selectedText, setSelectedText] = useState('');
  const [selectionPosition, setSelectionPosition] = useState<{ start: number; end: number } | undefined>();
  const [showCollaboration, setShowCollaboration] = useState(false);
  
  // Collaboration state
  const [comments, setComments] = useState<Comment[]>([
    // Mock data for demonstration
    {
      id: '1',
      author: 'AI Assistant',
      content: 'This section could benefit from more specific examples to support the main points.',
      timestamp: new Date(Date.now() - 3600000), // 1 hour ago
      resolved: false,
      position: {
        start: 150,
        end: 200,
        selectedText: 'the importance of digital transformation'
      }
    },
    {
      id: '2',
      author: 'Content Reviewer',
      content: 'Great introduction! The hook is engaging and sets up the article well.',
      timestamp: new Date(Date.now() - 1800000), // 30 minutes ago
      resolved: false
    }
  ]);
  
  const [suggestions, setSuggestions] = useState<Suggestion[]>([
    // Mock data for demonstration
    {
      id: '1',
      author: 'AI Editor',
      originalText: 'In today\'s rapidly changing business landscape',
      suggestedText: 'In today\'s dynamic business environment',
      reason: 'More concise and impactful phrasing',
      timestamp: new Date(Date.now() - 900000), // 15 minutes ago
      status: 'pending',
      position: { start: 0, end: 45 }
    }
  ]);

  // Auto-save setup
  const autoSave = useAutoSave(blogId, content, {
    delay: 3000, // 3 seconds
    onSaveStart: () => setSaving(true),
    onSaveSuccess: () => {
      setSaving(false);
      // Could show a subtle success indicator
    },
    onSaveError: (error) => {
      setSaving(false);
      console.error('Auto-save error:', error);
    }
  });

  const autoSaveStatus = useAutoSaveStatus(autoSave);

  // Fetch blog data
  const fetchBlog = useCallback(async () => {
    if (!blogId) return;
    
    try {
      setLoading(true);
      const data = await blogApi.get(blogId);
      setBlog(data);
      setContent(data.content_markdown);
    } catch (error) {
      if (error instanceof AppError && error.status === 404) {
        showErrorNotification(new AppError('Blog post not found'));
      } else {
        showErrorNotification(error instanceof AppError ? error : new AppError('Failed to load blog'));
      }
      navigate('/dashboard');
    } finally {
      setLoading(false);
    }
  }, [blogId, navigate]);

  useEffect(() => {
    if (blogId) {
      fetchBlog();
    }
  }, [blogId, fetchBlog]);

  // Handle text selection for collaboration
  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const selectedText = selection.toString();
      // const range = selection.getRangeAt(0);
      
      // Calculate position in the content
      const textArea = document.querySelector('textarea');
      if (textArea) {
        const start = textArea.selectionStart;
        const end = textArea.selectionEnd;
        
        setSelectedText(selectedText);
        setSelectionPosition({ start, end });
      }
    } else {
      setSelectedText('');
      setSelectionPosition(undefined);
    }
  }, []);

  // Manual save
  const handleManualSave = async () => {
    if (!blogId) return;
    
    try {
      setSaving(true);
      await autoSave.triggerSave();
      showSuccessNotification('Blog saved successfully!');
    } catch (error) {
      showErrorNotification(error instanceof AppError ? error : new AppError('Failed to save blog'));
    } finally {
      setSaving(false);
    }
  };

  // Collaboration handlers
  const handleAddComment = (commentContent: string, position?: Comment['position']) => {
    const newComment: Comment = {
      id: Date.now().toString(),
      author: 'You',
      content: commentContent,
      timestamp: new Date(),
      resolved: false,
      position
    };
    setComments([...comments, newComment]);
  };

  const handleResolveComment = (commentId: string) => {
    setComments(comments.map(comment => 
      comment.id === commentId ? { ...comment, resolved: true } : comment
    ));
  };

  const handleReplyToComment = (commentId: string, replyContent: string) => {
    const reply: Comment = {
      id: Date.now().toString(),
      author: 'You',
      content: replyContent,
      timestamp: new Date(),
      resolved: false
    };
    
    setComments(comments.map(comment => 
      comment.id === commentId 
        ? { ...comment, replies: [...(comment.replies || []), reply] }
        : comment
    ));
  };

  const handleAcceptSuggestion = (suggestionId: string) => {
    const suggestion = suggestions.find(s => s.id === suggestionId);
    if (suggestion) {
      // Apply the suggestion to the content
      const newContent = content.replace(suggestion.originalText, suggestion.suggestedText);
      setContent(newContent);
      
      // Mark suggestion as accepted
      setSuggestions(suggestions.map(s => 
        s.id === suggestionId ? { ...s, status: 'accepted' as const } : s
      ));
    }
  };

  const handleRejectSuggestion = (suggestionId: string) => {
    setSuggestions(suggestions.map(s => 
      s.id === suggestionId ? { ...s, status: 'rejected' as const } : s
    ));
  };

  const handlePublish = async () => {
    if (!blogId) return;
    
    try {
      setSaving(true);
      await autoSave.triggerSave(); // Save current changes first
      await blogApi.publish(blogId);
      showSuccessNotification('Blog published successfully!');
      navigate('/dashboard');
    } catch (error) {
      showErrorNotification(error instanceof AppError ? error : new AppError('Failed to publish blog'));
    } finally {
      setSaving(false);
    }
  };

  // Handle beforeunload to warn about unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (autoSave.hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [autoSave.hasUnsavedChanges]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="xl" className="mx-auto mb-4" />
          <p className="text-gray-600">Loading blog editor...</p>
        </div>
      </div>
    );
  }

  if (!blog) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 mb-4">Blog not found</div>
          <button onClick={() => navigate('/dashboard')} className="btn-primary">
            Go to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <Breadcrumbs />
        
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-gray-900 mb-1">{blog.title}</h1>
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <span>Status: {blog.status}</span>
              <span>•</span>
              <span className={autoSaveStatus.statusColor}>
                {autoSaveStatus.statusIcon} {autoSaveStatus.statusText}
              </span>
              {autoSave.saveCount > 0 && (
                <>
                  <span>•</span>
                  <span>{autoSave.saveCount} auto-saves</span>
                </>
              )}
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCollaboration(!showCollaboration)}
              className={`btn-secondary ${showCollaboration ? 'bg-primary-100 text-primary-700' : ''}`}
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a2 2 0 01-2-2v-2M9 4h8a2 2 0 012 2v6a2 2 0 01-2 2H9l-4 4V4z" />
              </svg>
              Comments ({comments.filter(c => !c.resolved).length})
            </button>
            
            <button
              onClick={handleManualSave}
              disabled={saving || autoSave.isSaving}
              className="btn-secondary"
            >
              {saving || autoSave.isSaving ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Saving...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Save
                </>
              )}
            </button>
            
            <button
              onClick={handlePublish}
              disabled={saving || autoSave.isSaving}
              className="btn-primary"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
              Publish
            </button>
          </div>
        </div>

        {/* Warning for unsaved changes */}
        {autoSave.hasUnsavedChanges && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <StatusIndicator status="warning" size="sm" />
              <span className="ml-2 text-sm text-yellow-800">
                You have unsaved changes. They will be automatically saved in a few seconds.
              </span>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="flex gap-6">
          {/* Editor */}
          <div className={`transition-all duration-300 ${showCollaboration ? 'flex-1' : 'w-full'}`}>
            <SplitViewEditor
              content={content}
              onChange={(newContent) => {
                setContent(newContent);
                handleTextSelection();
              }}
              placeholder="Start writing your blog post content here..."
              autoSaveStatus={autoSaveStatus}
            />
          </div>

          {/* Collaboration Panel */}
          {showCollaboration && (
            <div className="w-80 transition-all duration-300">
              <CollaborationPanel
                comments={comments}
                suggestions={suggestions}
                onAddComment={handleAddComment}
                onResolveComment={handleResolveComment}
                onReplyToComment={handleReplyToComment}
                onAcceptSuggestion={handleAcceptSuggestion}
                onRejectSuggestion={handleRejectSuggestion}
                selectedText={selectedText}
                selectionPosition={selectionPosition}
              />
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 p-4 bg-blue-50 rounded-lg">
          <h3 className="text-sm font-medium text-blue-900 mb-2">💡 Pro Tips:</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use **bold** and *italic* formatting with markdown syntax</li>
            <li>• Select text to add contextual comments or see AI suggestions</li>
            <li>• Your work is automatically saved every few seconds</li>
            <li>• Press ? for keyboard shortcuts</li>
          </ul>
        </div>
      </div>
      
      <KeyboardShortcutsHelp />
    </div>
  );
}