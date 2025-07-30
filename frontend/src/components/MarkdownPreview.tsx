import { useState, useEffect, useMemo } from 'react';
import { StatusIndicator } from './LoadingStates';

interface MarkdownPreviewProps {
  content: string;
  className?: string;
}

// Simple markdown parser for basic formatting
function parseMarkdown(content: string): string {
  if (!content) return '';

  let html = content;

  // Headers
  html = html.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold text-gray-900 mt-6 mb-3">$1</h3>');
  html = html.replace(/^## (.*$)/gm, '<h2 class="text-xl font-bold text-gray-900 mt-8 mb-4">$1</h2>');
  html = html.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold text-gray-900 mt-8 mb-6">$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong class="font-bold"><em class="italic">$1</em></strong>');
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold">$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em class="italic">$1</em>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">$1</a>');

  // Code blocks
  html = html.replace(/```([^`]+)```/g, '<pre class="bg-gray-100 rounded-lg p-4 overflow-x-auto my-4"><code class="text-sm">$1</code></pre>');
  
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-2 py-1 rounded text-sm font-mono">$1</code>');

  // Lists
  html = html.replace(/^\* (.+)$/gm, '<li class="mb-1">$1</li>');
  html = html.replace(/^- (.+)$/gm, '<li class="mb-1">$1</li>');
  html = html.replace(/^\d+\. (.+)$/gm, '<li class="mb-1">$1</li>');

  // Wrap consecutive list items in ul/ol tags
  html = html.replace(/(<li class="mb-1">.*<\/li>\s*)+/g, (match) => {
    return `<ul class="list-disc list-inside space-y-1 my-4 pl-4">${match}</ul>`;
  });

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote class="border-l-4 border-gray-300 pl-4 italic text-gray-700 my-4">$1</blockquote>');

  // Line breaks
  html = html.replace(/\n\n/g, '</p><p class="mb-4">');
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraphs
  if (html && !html.startsWith('<')) {
    html = `<p class="mb-4">${html}</p>`;
  }

  return html;
}

export function MarkdownPreview({ content, className = '' }: MarkdownPreviewProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Debounce processing for performance
  useEffect(() => {
    if (content) {
      setIsProcessing(true);
      const timer = setTimeout(() => {
        setIsProcessing(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [content]);

  const parsedContent = useMemo(() => {
    return parseMarkdown(content);
  }, [content]);

  const isEmpty = !content.trim();

  return (
    <div className={`relative flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Preview</h3>
        <div className="flex items-center space-x-2">
          <StatusIndicator
            status={isProcessing ? 'loading' : 'success'}
            message={isProcessing ? 'Updating...' : 'Live preview'}
            size="sm"
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <div className="flex items-center justify-center h-64 text-center">
            <div>
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <p className="text-gray-500 text-sm">
                Start typing in the editor to see your content preview here
              </p>
            </div>
          </div>
        ) : (
          <div 
            className="prose max-w-none"
            dangerouslySetInnerHTML={{ __html: parsedContent }}
          />
        )}
      </div>

      {/* Footer with stats */}
      {!isEmpty && (
        <div className="mt-4 pt-2 border-t border-gray-200">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center space-x-4">
              <span>{content.trim().split(/\s+/).length} words</span>
              <span>{content.length} characters</span>
              <span>{content.split('\n').length} lines</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-green-400 rounded-full"></span>
              <span>Live preview active</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface SplitViewEditorProps {
  content: string;
  onChange: (content: string) => void;
  placeholder?: string;
  autoSaveStatus?: {
    statusText: string;
    statusColor: string;
    statusIcon: string;
  };
}

export function SplitViewEditor({ 
  content, 
  onChange, 
  placeholder = "Start writing your content here...",
  autoSaveStatus 
}: SplitViewEditorProps) {
  const [viewMode, setViewMode] = useState<'split' | 'editor' | 'preview'>('split');

  const handleTextSelection = () => {
    // Handle text selection for collaboration features
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      // Could trigger collaboration features here
      console.log('Text selected:', selection.toString());
    }
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white">
      {/* Toolbar */}
      <div className="bg-gray-50 border-b border-gray-200 px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">View:</span>
            <div className="flex border border-gray-300 rounded-md overflow-hidden">
              <button
                onClick={() => setViewMode('editor')}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  viewMode === 'editor'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                Editor
              </button>
              <button
                onClick={() => setViewMode('split')}
                className={`px-3 py-1 text-xs font-medium transition-colors border-x border-gray-300 ${
                  viewMode === 'split'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                Split
              </button>
              <button
                onClick={() => setViewMode('preview')}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  viewMode === 'preview'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                Preview
              </button>
            </div>
          </div>
          
          {autoSaveStatus && (
            <div className="flex items-center space-x-2">
              <span className="text-lg">{autoSaveStatus.statusIcon}</span>
              <span className={`text-sm ${autoSaveStatus.statusColor}`}>
                {autoSaveStatus.statusText}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex" style={{ height: '700px' }}>
        {/* Editor */}
        {(viewMode === 'editor' || viewMode === 'split') && (
          <div className={`${viewMode === 'split' ? 'w-1/2 border-r border-gray-200' : 'w-full'}`}>
            <textarea
              className="w-full h-full p-4 border-none resize-none focus:outline-none focus:ring-0 text-sm font-mono leading-relaxed"
              value={content}
              onChange={(e) => onChange(e.target.value)}
              onMouseUp={handleTextSelection}
              placeholder={placeholder}
            />
          </div>
        )}

        {/* Preview */}
        {(viewMode === 'preview' || viewMode === 'split') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} bg-white flex flex-col`}>
            <MarkdownPreview 
              content={content} 
              className="flex-1 p-4"
            />
          </div>
        )}
      </div>
    </div>
  );
}