import { useState, useEffect, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  DocumentTextIcon,
  PencilIcon,
  CheckIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';
import { blogApi } from '../lib/api';
import { showErrorNotification, AppError } from '../lib/errors';
import { confirmAction } from '../lib/toast';
import type { BlogSummary } from '../lib/api';
import { useBlogStore } from '../store';
import { BlogList } from '../components/BlogList';
import { VirtualizedBlogList } from '../components/VirtualizedBlogList';
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
  // TODO: Refactor to use Zustand store for state management
  // Example usage:
  // const { blogs, isLoading, searchQuery, statusFilter, fetchBlogs, deleteBlog, setSearchQuery, setStatusFilter, filteredBlogs, stats } = useBlogStore();
  
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
    const confirmed = await confirmAction(
      'Are you sure you want to delete this blog post? This action cannot be undone.',
      async () => {
        try {
          await blogApi.delete(blogId);
          await fetchBlogs();
          performSearch();
        } catch (err) {
          showErrorNotification(err instanceof AppError ? err : new AppError('Failed to delete blog post'));
        }
      },
      {
        confirmText: 'Delete',
        cancelText: 'Cancel',
        type: 'danger'
      }
    );
  };

  const handleRefresh = async () => {
    await fetchBlogs();
    performSearch();
  };

  const hasActiveFilters = searchQuery.trim() || statusFilter !== 'all';

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error && blogs.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <svg className="h-12 w-12 text-red-500 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <h2 className="mt-4 text-xl font-semibold text-gray-900">Unable to load dashboard</h2>
          <p className="mt-2 text-gray-600">{error}</p>
          <button
            onClick={fetchBlogs}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Content Dashboard</h1>
              <p className="mt-2 text-gray-600">Manage your AI-generated content and campaigns</p>
            </div>
            <div className="flex space-x-3">
              <Link
                to="/workflow"
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Create Content
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Total Posts"
            value={stats.totalBlogs}
            icon={DocumentTextIcon}
            color="blue"
            trend="All time"
          />
          <MetricCard
            title="Drafts"
            value={stats.draftBlogs}
            icon={PencilIcon}
            color="yellow"
            trend="In progress"
          />
          <MetricCard
            title="Published"
            value={stats.publishedBlogs}
            icon={CheckIcon}
            color="green"
            trend="Live content"
          />
          <MetricCard
            title="This Week"
            value={stats.recentActivity}
            icon={ArrowTrendingUpIcon}
            color="purple"
            trend="New posts"
          />
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
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
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Blog Posts</h2>
          </div>
          <div className="p-6">
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
              blogs.length > 20 ? (
                <VirtualizedBlogList 
                  blogs={blogs} 
                  onDelete={handleDelete} 
                  onRefresh={handleRefresh}
                  height={600}
                />
              ) : (
                <BlogList blogs={blogs} onDelete={handleDelete} onRefresh={handleRefresh} />
              )
            )}
          </div>
        </div>

        <KeyboardShortcutsHelp />
      </div>
    </div>
  );
}

// Helper Components
interface MetricCardProps {
  title: string;
  value: number;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  color: 'blue' | 'green' | 'purple' | 'yellow' | 'red';
  trend?: string;
}

function MetricCard({ title, value, icon: Icon, color, trend }: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center">
        <div className={`p-3 rounded-md ${colorClasses[color]}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value.toLocaleString()}</p>
          {trend && (
            <p className="text-xs text-gray-500 mt-1 flex items-center">
              <ArrowTrendingUpIcon className="h-3 w-3 mr-1" />
              {trend}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImprovedDashboard;