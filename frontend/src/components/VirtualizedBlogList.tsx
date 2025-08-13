import { useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Link } from 'react-router-dom';
import type { BlogSummary } from '../lib/api';
import { blogApi } from '../lib/api';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';
import { confirmAction } from '../lib/toast';
import { QuickCampaignActions } from './QuickCampaignActions';

interface VirtualizedBlogListProps {
  blogs: BlogSummary[];
  onDelete?: (blogId: string) => void;
  onRefresh?: () => void;
  height?: number;
}

export function VirtualizedBlogList({ 
  blogs, 
  onDelete, 
  onRefresh, 
  height = 600 
}: VirtualizedBlogListProps) {
  const parentRef = useRef<HTMLDivElement>(null);

  // Filter out duplicates
  const uniqueBlogs = blogs.filter((blog, index, self) => 
    blog.id && self.findIndex(b => b.id === blog.id) === index
  );

  const virtualizer = useVirtualizer({
    count: uniqueBlogs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200, // Estimated height of each blog card
    overscan: 5, // Render 5 items outside the visible area
  });

  const formatDate = (dateString: string) => {
    if (!dateString) return "No date";
    
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        return "Invalid date";
      }
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (error) {
      return "Invalid date";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'draft':
        return 'bg-yellow-100 text-yellow-800';
      case 'edited':
        return 'bg-blue-100 text-blue-800';
      case 'published':
        return 'bg-green-100 text-green-800';
      case 'completed':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const canLaunchCampaign = (status: string) => {
    const allowedStatuses = ['edited', 'completed', 'published'];
    return allowedStatuses.includes(status.toLowerCase());
  };

  const canPublish = (status: string) => {
    return ['draft', 'edited'].includes(status.toLowerCase());
  };

  const handlePublish = async (blogId: string) => {
    const confirmed = await confirmAction(
      'Are you sure you want to publish this blog post?',
      async () => {
        try {
          await blogApi.publish(blogId);
          showSuccessNotification('Blog published successfully!');
          if (onRefresh) {
            onRefresh();
          }
        } catch (err) {
          showErrorNotification(err instanceof AppError ? err : new AppError('Failed to publish blog post'));
        }
      },
      {
        confirmText: 'Publish',
        cancelText: 'Cancel',
        type: 'info'
      }
    );
  };

  if (uniqueBlogs.length === 0) {
    return (
      <div className="text-center py-12" role="region" aria-label="No blogs found">
        <div className="w-24 h-24 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center" aria-hidden="true">
          <svg 
            className="w-12 h-12 text-gray-400" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
            aria-hidden="true"
            role="img"
          >
            <title>Document icon</title>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No blogs yet</h3>
        <p className="text-gray-600 mb-6">Get started by creating your first AI-generated blog post.</p>
        <Link 
          to="/workflow" 
          className="btn-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
          aria-label="Create your first blog post"
        >
          Create First Blog
        </Link>
      </div>
    );
  }

  return (
    <div 
      ref={parentRef}
      className="h-full overflow-auto"
      style={{ height }}
      role="list" 
      aria-label="Blog posts"
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => {
          const blog = uniqueBlogs[virtualItem.index];
          
          return (
            <article
              key={`${blog.id}-${virtualItem.index}`}
              className="absolute top-0 left-0 w-full px-4"
              style={{
                height: `${virtualItem.size}px`,
                transform: `translateY(${virtualItem.start}px)`,
              }}
              role="listitem"
            >
              <div className="card hover:shadow-md transition-shadow h-full">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
                      {blog.title || 'Untitled'}
                    </h3>
                    <div className="flex items-center space-x-2 text-sm text-gray-500">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(blog.status)}`}>
                        {blog.status || 'draft'}
                      </span>
                      <span>•</span>
                      <span>{formatDate(blog.created_at)}</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex space-x-2 flex-wrap">
                    <Link
                      to={`/edit/${blog.id}`}
                      className="btn-primary text-sm mb-1 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                      aria-label={`Edit blog post: ${blog.title || 'Untitled'}`}
                    >
                      Edit
                    </Link>
                    
                    {canPublish(blog.status) && (
                      <button
                        onClick={() => handlePublish(blog.id)}
                        className="btn-secondary text-sm mb-1 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                        aria-label={`Publish blog post: ${blog.title || 'Untitled'}`}
                      >
                        Publish
                      </button>
                    )}
                    
                    {canLaunchCampaign(blog.status) && (
                      <>
                        <Link
                          to={`/campaign/${blog.id}`}
                          className="btn-secondary text-sm mb-1 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                          aria-label={`Create campaign for blog post: ${blog.title || 'Untitled'}`}
                        >
                          Create Campaign
                        </Link>
                        <QuickCampaignActions 
                          blog={blog} 
                          onRefresh={onRefresh}
                        />
                      </>
                    )}
                  </div>
                  
                  {onDelete && (
                    <button 
                      onClick={() => onDelete(blog.id)}
                      className="text-sm text-gray-500 hover:text-red-600 transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 rounded"
                      aria-label={`Delete blog post: ${blog.title || 'Untitled'}`}
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}