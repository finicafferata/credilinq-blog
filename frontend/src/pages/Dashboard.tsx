import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { blogApi } from '../lib/api';
import { showErrorNotification, AppError } from '../lib/errors';
import type { BlogSummary } from '../lib/api';
import { BlogList } from '../components/BlogList';

export function Dashboard() {
  const [blogs, setBlogs] = useState<BlogSummary[]>([]);
  const [allBlogs, setAllBlogs] = useState<BlogSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // States for search and filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [isSearching, setIsSearching] = useState(false);

  useEffect(() => {
    fetchBlogs();
  }, []);

  const fetchBlogs = async () => {
    try {
      setLoading(true);
      const data = await blogApi.list();
      setAllBlogs(data);
      setBlogs(data);
    } catch (err) {
      const errorMessage = 'Failed to fetch blogs';
      setError(errorMessage);
      showErrorNotification(err instanceof AppError ? err : new AppError(errorMessage));
    } finally {
      setLoading(false);
    }
  };

  const performSearch = () => {
    // Client-side filtering since backend doesn't have search endpoint
    setIsSearching(true);
    setError(null);
    
    let filtered = allBlogs;
    
    // Filter by search query
    if (searchQuery.trim()) {
      filtered = filtered.filter(blog =>
        blog.title.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    // Filter by status
    if (statusFilter !== 'all') {
      filtered = filtered.filter(blog => blog.status === statusFilter);
    }
    
    setBlogs(filtered);
    
    setTimeout(() => {
      setIsSearching(false);
    }, 300); // Add small delay for UX
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    performSearch();
  };

  const handleStatusFilterChange = (newStatus: string) => {
    setStatusFilter(newStatus);
    // Auto-search when filter changes
    setTimeout(() => {
      if (newStatus !== statusFilter) {
        performSearch();
      }
    }, 100);
  };

  const clearFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setBlogs(allBlogs); // Return to original list
  };

  const handleDelete = async (blogId: string) => {
    if (!confirm('Are you sure you want to delete this blog post?')) {
      return;
    }

    try {
      await blogApi.delete(blogId);
      // Refresh the blogs list and reapply filters
      await fetchBlogs();
      performSearch();
    } catch (err) {
      showErrorNotification(err instanceof AppError ? err : new AppError('Failed to delete blog post'));
    }
  };

  const handleRefresh = async () => {
    // Refresh the blogs list and reapply filters
    await fetchBlogs();
    performSearch();
  };

  if (loading && !isSearching) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error && !isSearching) {
    return (
      <div className="text-center py-12">
        <div className="text-red-600 mb-4">{error}</div>
        <button onClick={fetchBlogs} className="btn-primary">
          Retry
        </button>
      </div>
    );
  }

  const hasActiveFilters = searchQuery.trim() || statusFilter !== 'all';

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Blog Dashboard</h1>
          <p className="text-gray-600 mt-1">Manage your AI-generated content and campaigns</p>
        </div>
        <Link to="/new" className="btn-primary">
          + Create New Blog
        </Link>
      </div>

      {/* Search and Filters */}
      <div className="mb-8 bg-gray-50 p-6 rounded-lg">
        <form onSubmit={handleSearchSubmit} className="space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search bar */}
            <div className="flex-1">
              <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-2">
                Search blogs
              </label>
              <div className="relative">
                <input
                  type="text"
                  id="search"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by title..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <div className="absolute left-3 top-2.5">
                  <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Status filter */}
            <div className="md:w-48">
              <label htmlFor="status" className="block text-sm font-medium text-gray-700 mb-2">
                Status
              </label>
              <select
                id="status"
                value={statusFilter}
                onChange={(e) => handleStatusFilterChange(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="all">All Statuses</option>
                <option value="draft">Draft</option>
                <option value="edited">Edited</option>
                <option value="published">Published</option>
                <option value="completed">Completed</option>
              </select>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center space-x-4">
            <button
              type="submit"
              disabled={isSearching}
              className="btn-primary disabled:opacity-50"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
            
            {hasActiveFilters && (
              <button
                type="button"
                onClick={clearFilters}
                className="btn-secondary"
              >
                Clear Filters
              </button>
            )}
            
            <div className="text-sm text-gray-500">
              {blogs.length} blog{blogs.length !== 1 ? 's' : ''} found
              {hasActiveFilters && ' (filtered)'}
            </div>
          </div>
        </form>
      </div>

      <BlogList blogs={blogs} onDelete={handleDelete} onRefresh={handleRefresh} />
    </div>
  );
} 