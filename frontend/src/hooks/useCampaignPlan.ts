import { useState, useCallback } from 'react';
import type { CampaignPlan, TaskStatus } from '../types/campaign';
import { campaignApi } from '../lib/api';

// Custom hook to manage the campaign plan state and API interactions
// Usage: const { plan, isLoading, error, executeTask, approveAsset, updateStatus, fetchPlan } = useCampaignPlan(blogId);
export function useCampaignPlan(blogId: string) {
  // State for the campaign plan
  const [plan, setPlan] = useState<CampaignPlan>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the campaign plan from the backend
  const fetchPlan = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const campaign = await campaignApi.getCampaign(blogId);
      setPlan(campaign.tasks);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch campaign plan');
    } finally {
      setIsLoading(false);
    }
  }, [blogId]);

  // Execute a specific task
  const executeTask = useCallback(async (taskId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await campaignApi.executeCampaignTask(taskId);
      await fetchPlan(); // Refresh plan after execution starts
    } catch (err: any) {
      setError(err?.message || 'Failed to execute task');
    } finally {
      setIsLoading(false);
    }
  }, [fetchPlan]);

  // Approve an asset (update content and set status to Approved)
  const approveAsset = useCallback(async (taskId: string, newContent?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await campaignApi.updateCampaignTask(taskId, newContent, 'Approved');
      await fetchPlan();
    } catch (err: any) {
      setError(err?.message || 'Failed to approve asset');
    } finally {
      setIsLoading(false);
    }
  }, [fetchPlan]);

  // Update the status of a task (e.g., to Posted)
  const updateStatus = useCallback(async (taskId: string, status: TaskStatus) => {
    setIsLoading(true);
    setError(null);
    try {
      await campaignApi.updateCampaignTask(taskId, undefined, status);
      await fetchPlan();
    } catch (err: any) {
      setError(err?.message || 'Failed to update status');
    } finally {
      setIsLoading(false);
    }
  }, [fetchPlan]);

  // Optionally, fetch the plan on mount (not required for SSR)
  // useEffect(() => { fetchPlan(); }, [fetchPlan]);

  return {
    plan,
    isLoading,
    error,
    fetchPlan,
    executeTask,
    approveAsset,
    updateStatus,
  };
} 