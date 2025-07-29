import type { CampaignTask } from '../types/campaign';
import React from 'react';

// Props for the AssetPreviewModal component
interface AssetPreviewModalProps {
  open: boolean;
  task: CampaignTask | null;
  onClose: () => void;
  onApprove: (newContent?: string) => void;
  onRequestRevision: (newContent?: string) => void;
}

// AssetPreviewModal displays the generated asset for review and approval
export function AssetPreviewModal({ open, task, onClose, onApprove, onRequestRevision }: AssetPreviewModalProps) {
  const [editedContent, setEditedContent] = React.useState<string>("");

  React.useEffect(() => {
    if (task) {
      setEditedContent(task.result || "");
    }
  }, [task]);

  if (!open || !task) return null;

  // Determine if the asset is text or image prompt
  const isTextAsset = task.taskType === 'repurpose';
  const isImagePrompt = task.taskType === 'create_image_prompt';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-lg relative">
        <button className="absolute top-2 right-2 text-gray-500 hover:text-black" onClick={onClose}>&times;</button>
        <h2 className="text-lg font-semibold mb-4">Review Asset</h2>
        {isTextAsset && (
          <textarea
            className="w-full min-h-[180px] border rounded p-2 mb-4"
            value={editedContent}
            onChange={e => setEditedContent(e.target.value)}
          />
        )}
        {isImagePrompt && (
          <div className="mb-4">
            <div className="font-mono bg-gray-100 p-2 rounded text-sm mb-2">
              {task.result}
            </div>
            {task.imageUrl && (
              <img src={task.imageUrl} alt="Generated asset" className="w-full rounded border" />
            )}
          </div>
        )}
        <div className="flex gap-2 justify-end">
          <button
            className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-50"
            onClick={() => onApprove(isTextAsset ? editedContent : undefined)}
          >
            Approve
          </button>
          <button
            className="px-4 py-2 bg-yellow-500 text-white rounded disabled:opacity-50"
            onClick={() => onRequestRevision(isTextAsset ? editedContent : undefined)}
          >
            Request Revision
          </button>
        </div>
      </div>
    </div>
  );
} 