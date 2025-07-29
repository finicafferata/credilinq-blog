import type { CampaignTask, TaskStatus } from '../types/campaign';

// Props for the ChecklistItem component
interface ChecklistItemProps {
  task: CampaignTask;
  onExecute: () => void;
  onReview: () => void;
  onStatusChange: (status: TaskStatus) => void;
  isLoading?: boolean;
}

// Helper to get a human-readable label for the task
function getTaskLabel(task: CampaignTask) {
  if (task.taskType === 'repurpose') {
    return `Generate ${task.targetFormat}`;
  }
  if (task.taskType === 'create_image_prompt') {
    return `Create ${task.targetAsset}`;
  }
  return task.taskType;
}

// ChecklistItem displays a single campaign task with status, actions, and manual status update
export function ChecklistItem({ task, onExecute, onReview, onStatusChange, isLoading }: ChecklistItemProps) {
  // Status badge color
  const statusColor = {
    'Pending': 'bg-gray-200 text-gray-800',
    'In Progress': 'bg-blue-200 text-blue-800',
    'Needs Review': 'bg-yellow-200 text-yellow-800',
    'Approved': 'bg-green-200 text-green-800',
    'Posted': 'bg-purple-200 text-purple-800',
    'Error': 'bg-red-200 text-red-800',
  }[task.status] || 'bg-gray-100 text-gray-700';

  // Show Execute button if Pending
  const showExecute = task.status === 'Pending';
  // Show Review button if Needs Review
  const showReview = task.status === 'Needs Review';
  // Allow manual status change if Approved
  const showStatusDropdown = task.status === 'Approved';

  return (
    <div className="flex items-center justify-between p-4 border rounded-lg bg-white shadow-sm">
      <div>
        <div className="font-medium">{getTaskLabel(task)}</div>
        <span className={statusColor + " ml-1 px-2 py-1 rounded text-xs font-semibold"}>{task.status}</span>
        {task.error && <span className="text-xs text-red-500 ml-2">{task.error}</span>}
      </div>
      <div className="flex items-center gap-2">
        {showExecute && (
          <button disabled={isLoading} onClick={onExecute} className="px-3 py-1 bg-blue-600 text-white rounded text-sm disabled:opacity-50">
            Execute
          </button>
        )}
        {showReview && (
          <button disabled={isLoading} onClick={onReview} className="px-3 py-1 border border-gray-400 rounded text-sm disabled:opacity-50">
            Review
          </button>
        )}
        {showStatusDropdown && (
          <select
            className="w-28 h-8 text-xs border rounded"
            value={task.status}
            onChange={e => onStatusChange(e.target.value as TaskStatus)}
          >
            <option value="Approved">Approved</option>
            <option value="Posted">Posted</option>
            {/* You can add more manual statuses here if needed */}
          </select>
        )}
      </div>
    </div>
  );
} 