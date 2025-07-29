import { Link } from 'react-router-dom';
import type { BlogSummary } from '../lib/api';
import { blogApi } from '../lib/api';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';

interface BlogListProps {
  blogs: BlogSummary[];
  onDelete?: (blogId: string) => void;
  onRefresh?: () => void;
}

export function BlogList({ blogs, onDelete, onRefresh }: BlogListProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
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

  const canCreateCampaign = (status: string) => {
    // Allow creating campaigns for edited, completed or published blogs
    const allowedStatuses = ['edited', 'completed', 'published'];
    return allowedStatuses.includes(status.toLowerCase());
  };

  const canPublish = (status: string) => {
    // Only draft or edited blogs can be published
    return ['draft', 'edited'].includes(status.toLowerCase());
  };

  const handlePublish = async (blogId: string) => {
    if (!confirm('Are you sure you want to publish this blog post?')) {
      return;
    }

    try {
      await blogApi.publish(blogId);
      showSuccessNotification('Blog published successfully!');
      if (onRefresh) {
        onRefresh(); // Refresh list to show new status
      }
    } catch (err) {
      showErrorNotification(err instanceof AppError ? err : new AppError('Failed to publish blog post'));
    }
  };

  if (blogs.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="w-24 h-24 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
          <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No blogs yet</h3>
        <p className="text-gray-600 mb-6">Get started by creating your first AI-generated blog post.</p>
        <Link to="/new" className="btn-primary">
          Create First Blog
        </Link>
      </div>
    );
  }

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {blogs.map((blog) => (
        <div key={blog.id} className="card hover:shadow-md transition-shadow">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
                {blog.title}
              </h3>
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(blog.status)}`}>
                  {blog.status}
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
                className="btn-primary text-sm mb-1"
              >
                Edit
              </Link>
              
              {canPublish(blog.status) && (
                <button
                  onClick={() => handlePublish(blog.id)}
                  className="btn-secondary text-sm mb-1"
                >
                  Publish
                </button>
              )}
              
              {canCreateCampaign(blog.status) && (
                <Link
                  to={`/campaign/${blog.id}`}
                  className="btn-secondary text-sm mb-1"
                >
                  Create Campaign
                </Link>
              )}
            </div>
            
            {onDelete && (
              <button 
                onClick={() => onDelete(blog.id)}
                className="text-sm text-gray-500 hover:text-red-600 transition-colors"
              >
                Delete
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
} 