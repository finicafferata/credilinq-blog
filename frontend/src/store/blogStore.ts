/**
 * Blog state management using Zustand
 */
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { blogApi } from '../lib/api';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';
import type { BlogSummary, BlogData } from '../lib/api';

export interface BlogState {
  // State
  blogs: BlogSummary[];
  currentBlog: BlogData | null;
  isLoading: boolean;
  isPublishing: boolean;
  isDeleting: boolean;
  error: string | null;
  searchQuery: string;
  statusFilter: 'all' | 'draft' | 'edited' | 'published' | 'completed';
  
  // Actions
  setBlogs: (blogs: BlogSummary[]) => void;
  setCurrentBlog: (blog: BlogData | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSearchQuery: (query: string) => void;
  setStatusFilter: (filter: BlogState['statusFilter']) => void;
  
  // Async actions
  fetchBlogs: () => Promise<void>;
  fetchBlog: (id: string) => Promise<void>;
  createBlog: (data: Partial<BlogData>) => Promise<string | null>;
  updateBlog: (id: string, data: Partial<BlogData>) => Promise<void>;
  publishBlog: (id: string) => Promise<void>;
  deleteBlog: (id: string) => Promise<void>;
  
  // Computed state
  filteredBlogs: () => BlogSummary[];
  stats: () => {
    total: number;
    draft: number;
    published: number;
    recentActivity: number;
  };
}

export const useBlogStore = create<BlogState>()(
  devtools(
    (set, get) => ({
      // Initial state
      blogs: [],
      currentBlog: null,
      isLoading: false,
      isPublishing: false,
      isDeleting: false,
      error: null,
      searchQuery: '',
      statusFilter: 'all',

      // Basic setters
      setBlogs: (blogs) => set({ blogs }),
      setCurrentBlog: (blog) => set({ currentBlog: blog }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
      setSearchQuery: (searchQuery) => set({ searchQuery }),
      setStatusFilter: (statusFilter) => set({ statusFilter }),

      // Async actions
      fetchBlogs: async () => {
        set({ isLoading: true, error: null });
        try {
          const blogs = await blogApi.getAll();
          set({ blogs, isLoading: false });
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to fetch blogs';
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      fetchBlog: async (id: string) => {
        set({ isLoading: true, error: null });
        try {
          const blog = await blogApi.get(id);
          set({ currentBlog: blog, isLoading: false });
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : `Failed to fetch blog ${id}`;
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      createBlog: async (data: Partial<BlogData>) => {
        set({ isLoading: true, error: null });
        try {
          const result = await blogApi.create(data);
          const blogId = typeof result === 'string' ? result : result.id;
          
          // Refresh blogs list
          await get().fetchBlogs();
          
          showSuccessNotification('Blog created successfully!');
          set({ isLoading: false });
          return blogId;
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to create blog';
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
          return null;
        }
      },

      updateBlog: async (id: string, data: Partial<BlogData>) => {
        set({ isLoading: true, error: null });
        try {
          await blogApi.update(id, data);
          
          // Update current blog if it's the one being updated
          const { currentBlog } = get();
          if (currentBlog && currentBlog.id === id) {
            set({ currentBlog: { ...currentBlog, ...data } });
          }
          
          // Refresh blogs list
          await get().fetchBlogs();
          
          showSuccessNotification('Blog updated successfully!');
          set({ isLoading: false });
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to update blog';
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      publishBlog: async (id: string) => {
        set({ isPublishing: true, error: null });
        try {
          await blogApi.publish(id);
          
          // Update blog status in the list
          const { blogs } = get();
          const updatedBlogs = blogs.map(blog => 
            blog.id === id ? { ...blog, status: 'published' } : blog
          );
          set({ blogs: updatedBlogs, isPublishing: false });
          
          showSuccessNotification('Blog published successfully!');
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to publish blog';
          set({ error: errorMessage, isPublishing: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      deleteBlog: async (id: string) => {
        set({ isDeleting: true, error: null });
        try {
          await blogApi.delete(id);
          
          // Remove blog from the list
          const { blogs } = get();
          const updatedBlogs = blogs.filter(blog => blog.id !== id);
          set({ blogs: updatedBlogs, isDeleting: false });
          
          showSuccessNotification('Blog deleted successfully!');
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to delete blog';
          set({ error: errorMessage, isDeleting: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      // Computed state
      filteredBlogs: () => {
        const { blogs, searchQuery, statusFilter } = get();
        
        return blogs.filter(blog => {
          // Filter by search query
          const matchesSearch = !searchQuery || 
            blog.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            blog.content?.toLowerCase().includes(searchQuery.toLowerCase());
          
          // Filter by status
          const matchesStatus = statusFilter === 'all' || blog.status === statusFilter;
          
          return matchesSearch && matchesStatus;
        });
      },

      stats: () => {
        const { blogs } = get();
        const now = new Date();
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        
        return {
          total: blogs.length,
          draft: blogs.filter(blog => blog.status === 'draft').length,
          published: blogs.filter(blog => blog.status === 'published').length,
          recentActivity: blogs.filter(blog => 
            blog.created_at && new Date(blog.created_at) > weekAgo
          ).length,
        };
      },
    }),
    {
      name: 'blog-store',
    }
  )
);