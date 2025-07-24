import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { blogApi } from '../lib/api';
import type { BlogDetail } from '../lib/api';

interface RevisionPopupProps {
  isOpen: boolean;
  onClose: () => void;
  onRevise: (instruction: string) => void;
  loading: boolean;
}

function RevisionPopup({ isOpen, onClose, onRevise, loading }: RevisionPopupProps) {
  const [instruction, setInstruction] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (instruction.trim()) {
      onRevise(instruction.trim());
      setInstruction('');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Revision Assistant</h3>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              How should I revise this text?
            </label>
            <input
              type="text"
              className="input"
              placeholder="e.g., Make this more concise, Add more technical details..."
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              autoFocus
            />
          </div>
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={loading || !instruction.trim()}
            >
              {loading ? 'Revising...' : 'Revise'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function BlogEditor() {
  const { blogId } = useParams<{ blogId: string }>();
  const navigate = useNavigate();
  const [blog, setBlog] = useState<BlogDetail | null>(null);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showRevisionPopup, setShowRevisionPopup] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [revisionLoading, setRevisionLoading] = useState(false);
  const [revisedText, setRevisedText] = useState('');
  const [showRevisionResult, setShowRevisionResult] = useState(false);

  useEffect(() => {
    if (blogId) {
      fetchBlog();
    }
  }, [blogId]);

  const fetchBlog = async () => {
    try {
      setLoading(true);
      const data = await blogApi.get(blogId!);
      setBlog(data);
      setContent(data.content_markdown);
    } catch (error) {
      console.error('Failed to fetch blog:', error);
      navigate('/');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!blogId) return;
    
    try {
      setSaving(true);
      await blogApi.update(blogId, { content_markdown: content });
      alert('Blog saved successfully!');
    } catch (error) {
      console.error('Failed to save blog:', error);
      alert('Failed to save blog. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      setSelectedText(selection.toString());
    }
  };

  const handleRevise = async (instruction: string) => {
    if (!blogId || !selectedText) return;

    try {
      setRevisionLoading(true);
      const response = await blogApi.revise(blogId, {
        instruction,
        text_to_revise: selectedText,
      });
      setRevisedText(response.revised_text);
      setShowRevisionPopup(false);
      setShowRevisionResult(true);
    } catch (error) {
      console.error('Failed to revise text:', error);
      alert('Failed to revise text. Please try again.');
    } finally {
      setRevisionLoading(false);
    }
  };

  const acceptRevision = () => {
    const newContent = content.replace(selectedText, revisedText);
    setContent(newContent);
    setShowRevisionResult(false);
    setSelectedText('');
    setRevisedText('');
  };

  const rejectRevision = () => {
    setShowRevisionResult(false);
    setSelectedText('');
    setRevisedText('');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!blog) {
    return (
      <div className="text-center py-12">
        <div className="text-red-600 mb-4">Blog not found</div>
        <button onClick={() => navigate('/')} className="btn-primary">
          Go to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{blog.title}</h1>
          <p className="text-gray-600">Edit your AI-generated blog post</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => navigate('/')}
            className="btn-secondary"
          >
            Back to Dashboard
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="btn-primary"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      <div className="card">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">Content Editor</h3>
          {selectedText && (
            <button
              onClick={() => setShowRevisionPopup(true)}
              className="flex items-center space-x-2 text-sm bg-primary-100 text-primary-700 px-3 py-1 rounded-full hover:bg-primary-200 transition-colors"
            >
              <span>✨</span>
              <span>Revise Selected Text</span>
            </button>
          )}
        </div>
        
        <textarea
          className="textarea w-full"
          rows={25}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onMouseUp={handleTextSelection}
          placeholder="Start writing your blog post content here..."
        />
        
        <p className="mt-2 text-sm text-gray-500">
          Select text to use AI revision assistance, or edit directly in the textarea.
        </p>
      </div>

      <RevisionPopup
        isOpen={showRevisionPopup}
        onClose={() => setShowRevisionPopup(false)}
        onRevise={handleRevise}
        loading={revisionLoading}
      />

      {showRevisionResult && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Revision Result</h3>
            
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Original Text:</h4>
                <div className="p-3 bg-red-50 border border-red-200 rounded text-sm">
                  {selectedText}
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Revised Text:</h4>
                <div className="p-3 bg-green-50 border border-green-200 rounded text-sm">
                  {revisedText}
                </div>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={rejectRevision}
                className="btn-secondary"
              >
                Reject
              </button>
              <button
                onClick={acceptRevision}
                className="btn-primary"
              >
                Accept Revision
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 