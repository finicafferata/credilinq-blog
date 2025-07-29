import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useCampaignPlan } from '../hooks/useCampaignPlan';
import { ChecklistItem } from '../components/ChecklistItem';
import { AssetPreviewModal } from '../components/AssetPreviewModal';
import type { CampaignTask, TaskStatus } from '../types/campaign';

// CampaignView is the main page for managing a campaign workflow for a blog post
export function CampaignView() {
  // Get the blogId from the route params
  const { blogId } = useParams<{ blogId: string }>();
  // Use the custom hook to manage campaign state
  const {
    plan,
    isLoading,
    error,
    fetchPlan,
    executeTask,
    approveAsset,
    updateStatus,
  } = useCampaignPlan(blogId!);

  // Modal state for asset review
  const [reviewTask, setReviewTask] = useState<CampaignTask | null>(null);

  // Fetch the campaign plan on mount
  useEffect(() => {
    fetchPlan();
    // Optionally, set up polling here if needed
  }, [fetchPlan]);

  // Handler for executing a task
  const handleExecute = (taskId: string) => {
    executeTask(taskId);
  };

  // Handler for opening the review modal
  const handleReview = (task: CampaignTask) => {
    setReviewTask(task);
  };

  // Handler for approving an asset
  const handleApprove = async (newContent?: string) => {
    if (reviewTask) {
      await approveAsset(reviewTask.id, newContent);
      setReviewTask(null);
    }
  };

  // Handler for requesting revision
  const handleRequestRevision = async () => {
    if (reviewTask) {
      // For now, just close the modal. You can implement revision logic here.
      setReviewTask(null);
    }
  };

  // Handler for updating status (e.g., to Posted)
  const handleStatusChange = (taskId: string, status: TaskStatus) => {
    updateStatus(taskId, status);
  };

  return (
    <div className="max-w-2xl mx-auto py-8">
      <h1 className="text-2xl font-bold mb-6">Campaign Workflow</h1>
      {isLoading && <div className="mb-4">Loading campaign plan...</div>}
      {error && <div className="mb-4 text-red-500">{error}</div>}
      <div className="space-y-4">
        {plan.map((task) => (
          <ChecklistItem
            key={task.id}
            task={task}
            onExecute={() => handleExecute(task.id)}
            onReview={() => handleReview(task)}
            onStatusChange={(status) => handleStatusChange(task.id, status)}
            isLoading={isLoading}
          />
        ))}
      </div>
      <AssetPreviewModal
        open={!!reviewTask}
        task={reviewTask}
        onClose={() => setReviewTask(null)}
        onApprove={handleApprove}
        onRequestRevision={handleRequestRevision}
      />
    </div>
  );
} 