import { useState, useCallback, useEffect } from 'react';
import type { DashboardAnalytics, AgentAnalytics, BlogAnalyticsResponse } from '../types/analytics';
import { analyticsApi } from '../lib/api';

// Custom hook for dashboard analytics
export function useDashboardAnalytics(days: number = 30) {
  const [analytics, setAnalytics] = useState<DashboardAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyticsApi.getDashboardAnalytics(days);
      setAnalytics(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to fetch analytics');
    } finally {
      setIsLoading(false);
    }
  }, [days]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    analytics,
    isLoading,
    error,
    refetch: fetchAnalytics,
  };
}

// Custom hook for agent analytics
export function useAgentAnalytics(agentType?: string, days: number = 30) {
  const [analytics, setAnalytics] = useState<AgentAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyticsApi.getAgentAnalytics(agentType, days);
      setAnalytics(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to fetch agent analytics');
    } finally {
      setIsLoading(false);
    }
  }, [agentType, days]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    analytics,
    isLoading,
    error,
    refetch: fetchAnalytics,
  };
}

// Custom hook for blog analytics
export function useBlogAnalytics(blogId: string) {
  const [analytics, setAnalytics] = useState<BlogAnalyticsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async () => {
    if (!blogId) return;
    
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyticsApi.getBlogAnalytics(blogId);
      setAnalytics(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to fetch blog analytics');
    } finally {
      setIsLoading(false);
    }
  }, [blogId]);

  const updateAnalytics = useCallback(async (newAnalytics: Partial<BlogAnalyticsResponse>) => {
    if (!blogId) return;
    
    setIsLoading(true);
    setError(null);
    try {
      await analyticsApi.updateBlogAnalytics(blogId, newAnalytics as any);
      await fetchAnalytics(); // Refresh data
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to update analytics');
    } finally {
      setIsLoading(false);
    }
  }, [blogId, fetchAnalytics]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    analytics,
    isLoading,
    error,
    refetch: fetchAnalytics,
    updateAnalytics,
  };
}