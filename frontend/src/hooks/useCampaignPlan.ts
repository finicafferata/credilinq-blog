import { useState, useCallback } from 'react';
import { campaignApi } from '../lib/api';
import type { CampaignDetail } from '../lib/api';
import { showErrorNotification, showSuccessNotification } from '../lib/errors';

// Custom hook to manage the campaign plan state and API interactions
// Usage: const { campaign, isLoading, error, schedule, distribute, fetchCampaign } = useCampaignPlan(campaignId);
export function useCampaignPlan(campaignId: string) {
  // State for the campaign
  const [campaign, setCampaign] = useState<CampaignDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the campaign from the backend
  const fetchCampaign = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await campaignApi.get(campaignId);
      setCampaign(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch campaign');
      const message = err?.message || 'Failed to fetch campaign';
      showErrorNotification(`Failed to fetch campaign: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [campaignId]);

  // Schedule campaign tasks
  const schedule = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      await campaignApi.schedule(campaignId);
      showSuccessNotification('Campaign scheduled successfully!');
      await fetchCampaign(); // Refresh campaign after scheduling
    } catch (err: any) {
      setError(err?.message || 'Failed to schedule campaign');
      const message = err?.message || 'Failed to schedule';
      showErrorNotification(`Failed to schedule campaign: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [campaignId, fetchCampaign]);

  // Distribute campaign posts
  const distribute = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      await campaignApi.distribute(campaignId);
      showSuccessNotification('Campaign distributed successfully!');
      await fetchCampaign(); // Refresh campaign after distribution
    } catch (err: any) {
      setError(err?.message || 'Failed to distribute campaign');
      const message = err?.message || 'Failed to distribute';
      showErrorNotification(`Failed to distribute campaign: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [campaignId, fetchCampaign]);

  // Update task status
  const updateTaskStatus = useCallback(async (taskId: string, status: string) => {
    setError(null);
    try {
      await campaignApi.updateTaskStatus(campaignId, taskId, status);
      showSuccessNotification('Task status updated successfully!');
      await fetchCampaign(); // Refresh campaign after task update
    } catch (err: any) {
      setError(err?.message || 'Failed to update task status');
      const message = err?.message || 'Failed to update task';
      showErrorNotification(`Failed to update task status: ${message}`);
    }
  }, [campaignId, fetchCampaign]);

  return {
    campaign,
    isLoading,
    error,
    fetchCampaign,
    schedule,
    distribute,
    updateTaskStatus,
  };
} 