import { useState, useEffect, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { blogApi } from '../lib/api';
import { showErrorNotification, AppError } from '../lib/errors';
import type { BlogSummary } from '../lib/api';
import { BlogList } from '../components/BlogList';
import { QuickStartWizard } from '../components/QuickStartWizard';
import { Breadcrumbs } from '../components/Breadcrumbs';
import { KeyboardShortcutsHelp } from '../components/KeyboardShortcutsHelp';
import { BlogCardSkeleton, EmptyState, StatusIndicator } from '../components/LoadingStates';

interface DashboardStats {
  totalBlogs: number;
  draftBlogs: number;
  publishedBlogs: number;
  recentActivity: number;
}

const calculateStats = (blogs: BlogSummary[]) => {
  const totalBlogs = blogs.length;
  const draftBlogs = blogs.filter(blog => blog.status === 'draft').length;
  const publishedBlogs = blogs.filter(blog => blog.status === 'published').length;
  const recentActivity = blogs.filter(blog => {
    const blogDate = new Date(blog.created_at);
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    return blogDate > weekAgo;
  }).length;

  return {
    totalBlogs,
    draftBlogs,
    publishedBlogs,
    recentActivity
  };
};

export function ImprovedDashboard() {
  const [blogs, setBlogs] = useState<BlogSummary[]>([]);
  const [allBlogs, setAllBlogs] = useState<BlogSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // States for search and filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [isSearching, setIsSearching] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  
  // Dashboard stats
  const [stats, setStats] = useState<DashboardStats>({
    totalBlogs: 0,
    draftBlogs: 0,
    publishedBlogs: 0,
    recentActivity: 0
  });

  useEffect(() => {
    fetchBlogs();
  }, []);

  useEffect(() => {
    const currentStats = calculateStats(allBlogs);
    setStats(currentStats);
  }, [allBlogs]);
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
    }, 300);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    performSearch();
  };

  const handleStatusFilterChange = (newStatus: string) => {
    setStatusFilter(newStatus);
    setTimeout(() => {
      if (newStatus !== statusFilter) {
        performSearch();
      }
    }, 100);
  };

  const clearFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setBlogs(allBlogs);
  };

  const handleDelete = async (blogId: string) => {
    if (!confirm('Are you sure you want to delete this blog post?')) {
      return;
    }

    try {
      await blogApi.delete(blogId);
      await fetchBlogs();
      performSearch();
    } catch (err) {
      showErrorNotification(err instanceof AppError ? err : new AppError('Failed to delete blog post'));
    }
  };

  const handleRefresh = async () => {
    await fetchBlogs();
    performSearch();
  };

  const hasActiveFilters = searchQuery.trim() || statusFilter !== 'all';

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <Breadcrumbs />
          
          {/* Header Skeleton */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <div className="h-8 w-64 bg-gray-200 rounded animate-pulse mb-2"></div>
              <div className="h-4 w-96 bg-gray-200 rounded animate-pulse"></div>
            </div>
            <div className="h-10 w-40 bg-gray-200 rounded animate-pulse"></div>
          </div>

          {/* Stats Skeleton */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="bg-white rounded-xl shadow-sm p-6">
                <div className="h-4 w-20 bg-gray-200 rounded animate-pulse mb-4"></div>
                <div className="h-8 w-16 bg-gray-200 rounded animate-pulse mb-2"></div>
                <div className="h-3 w-24 bg-gray-200 rounded animate-pulse"></div>
              </div>
            ))}
          </div>

          <BlogCardSkeleton />
        </div>
      </div>
    );
  }

  if (error && blogs.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <Breadcrumbs />
          <EmptyState
            title="Failed to load dashboard"
            description={error}
            action={{
              label: "Retry",
              onClick: fetchBlogs
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <Breadcrumbs />
        
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center">
              Content Dashboard
              <span className="ml-3 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                {stats.totalBlogs} {stats.totalBlogs === 1 ? 'post' : 'posts'}
              </span>
            </h1>
            <p className="text-gray-600 mt-1 flex items-center">
              Manage your AI-generated content and campaigns
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <Link to="/workflow" className="btn-primary">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Create Content
            </Link>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Posts</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalBlogs}</p>
                <p className="text-sm text-gray-500 mt-1">All time</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Drafts</p>
                <p className="text-3xl font-bold text-yellow-600 mt-2">{stats.draftBlogs}</p>
                <p className="text-sm text-gray-500 mt-1">In progress</p>
              </div>
              <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Published</p>
                <p className="text-3xl font-bold text-green-600 mt-2">{stats.publishedBlogs}</p>
                <p className="text-sm text-gray-500 mt-1">Live content</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">This Week</p>
                <p className="text-3xl font-bold text-purple-600 mt-2">{stats.recentActivity}</p>
                <p className="text-sm text-gray-500 mt-1">New posts</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Create Content */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 mb-8">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Create Content</h3>
            <p className="text-gray-600 mb-6">Choose your content creation approach</p>
            <Link
              to="/workflow"
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Start Creating
            </Link>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 mb-8">
          <form onSubmit={handleSearchSubmit} className="space-y-4">
            <div className="flex flex-col lg:flex-row gap-4">
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
                    placeholder="Search by title... (Ctrl+F)"
                    className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  />
                  <div className="absolute left-3 top-3.5">
                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  {isSearching && (
                    <div className="absolute right-3 top-3.5">
                      <StatusIndicator status="loading" size="sm" />
                    </div>
                  )}
                </div>
              </div>

              {/* Status filter */}
              <div className="lg:w-48">
                <label htmlFor="status" className="block text-sm font-medium text-gray-700 mb-2">
                  Status
                </label>
                <select
                  id="status"
                  value={statusFilter}
                  onChange={(e) => handleStatusFilterChange(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                >
                  <option value="all">All Statuses</option>
                  <option value="draft">Draft</option>
                  <option value="edited">Edited</option>
                  <option value="published">Published</option>
                  <option value="completed">Completed</option>
                </select>
              </div>

              {/* View mode toggle */}
              <div className="lg:w-32">
                <label className="block text-sm font-medium text-gray-700 mb-2">View</label>
                <div className="flex border border-gray-300 rounded-lg overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setViewMode('grid')}
                    className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                      viewMode === 'grid'
                        ? 'bg-primary-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <svg className="w-4 h-4 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode('list')}
                    className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                      viewMode === 'list'
                        ? 'bg-primary-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <svg className="w-4 h-4 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex items-center justify-between">
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
                
                <button
                  type="button"
                  onClick={handleRefresh}
                  className="text-gray-500 hover:text-gray-700 p-2 rounded-lg hover:bg-gray-100 transition-colors"
                  title="Refresh (Ctrl+R)"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              </div>
              
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <StatusIndicator
                  status={isSearching ? 'loading' : blogs.length > 0 ? 'success' : 'info'}
                  message={`${blogs.length} blog${blogs.length !== 1 ? 's' : ''} ${hasActiveFilters ? '(filtered)' : 'total'}`}
                />
              </div>
            </div>
          </form>
        </div>

        {/* Content */}
        {blogs.length === 0 && !loading ? (
          hasActiveFilters ? (
            <EmptyState
              title="No blogs match your search"
              description="Try adjusting your search terms or filters to find what you're looking for."
              action={{
                label: "Clear Filters",
                onClick: clearFilters
              }}
            />
          ) : (
            <EmptyState
              title="No blogs yet"
              description="Get started by creating your first AI-generated blog post. Our intelligent agents will help you create amazing content."
              action={{
                label: "Create First Blog",
                onClick: () => window.location.href = "/new"
              }}
            />
          )
        ) : (
          <BlogList blogs={blogs} onDelete={handleDelete} onRefresh={handleRefresh} />
        )}

        <KeyboardShortcutsHelp />
      </div>
    </div>
  );
}