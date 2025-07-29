// Types for the Campaign Workflow feature

// Possible status values for a campaign task
export type TaskStatus = 'Pending' | 'In Progress' | 'Needs Review' | 'Approved' | 'Posted' | 'Error';

// Structure of a single campaign task
export interface CampaignTask {
  id: string;
  taskType: string; // e.g., 'repurpose', 'create_image_prompt'
  targetFormat?: string; // e.g., 'LinkedIn Post', 'Tweet Thread'
  targetAsset?: string;  // e.g., 'Blog Header', 'LinkedIn Post Image'
  status: TaskStatus;
  result?: string | null; // The generated asset or null
  imageUrl?: string;    // For image assets, if available
  error?: string | null;
  createdAt: string;
  updatedAt: string;
}

// Structure of the campaign plan (list of tasks)
export type CampaignPlan = CampaignTask[];

// Structure of the campaign response from the backend
export interface CampaignResponse {
  id: string;
  blogId: string;
  createdAt: string;
  tasks: CampaignTask[];
} 