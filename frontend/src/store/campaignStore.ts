/**
 * Campaign state management using Zustand
 */
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';
import { toastService } from '../lib/toast';

// Define types (these should match your API types)
interface Campaign {
  id: string;
  blog_id: string;
  title: string;
  description?: string;
  status: 'planning' | 'active' | 'completed' | 'paused';
  created_at: string;
  updated_at: string;
}

interface CampaignTask {
  id: string;
  campaign_id: string;
  title: string;
  type: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  created_at: string;
  result?: any;
}

export interface CampaignState {
  // State
  campaigns: Campaign[];
  currentCampaign: Campaign | null;
  campaignTasks: CampaignTask[];
  isLoading: boolean;
  isCreating: boolean;
  error: string | null;
  
  // Actions
  setCampaigns: (campaigns: Campaign[]) => void;
  setCurrentCampaign: (campaign: Campaign | null) => void;
  setCampaignTasks: (tasks: CampaignTask[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Async actions
  fetchCampaigns: () => Promise<void>;
  fetchCampaign: (id: string) => Promise<void>;
  fetchCampaignTasks: (campaignId: string) => Promise<void>;
  createCampaign: (data: Partial<Campaign>) => Promise<string | null>;
  updateCampaign: (id: string, data: Partial<Campaign>) => Promise<void>;
  deleteCampaign: (id: string) => Promise<void>;
  
  // Optimistic updates
  updateCampaignOptimistic: (id: string, data: Partial<Campaign>) => void;
  updateTaskOptimistic: (id: string, data: Partial<CampaignTask>) => void;
  
  // Computed state
  activeCampaigns: () => Campaign[];
  completedCampaigns: () => Campaign[];
  pendingTasks: () => CampaignTask[];
}

export const useCampaignStore = create<CampaignState>()(
  devtools(
    (set, get) => ({
      // Initial state
      campaigns: [],
      currentCampaign: null,
      campaignTasks: [],
      isLoading: false,
      isCreating: false,
      error: null,

      // Basic setters
      setCampaigns: (campaigns) => set({ campaigns }),
      setCurrentCampaign: (currentCampaign) => set({ currentCampaign }),
      setCampaignTasks: (campaignTasks) => set({ campaignTasks }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),

      // Async actions (placeholder implementations - would need actual API)
      fetchCampaigns: async () => {
        set({ isLoading: true, error: null });
        try {
          // TODO: Replace with actual API call
          // const campaigns = await campaignApi.getAll();
          const campaigns: Campaign[] = []; // Placeholder
          set({ campaigns, isLoading: false });
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to fetch campaigns';
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      fetchCampaign: async (id: string) => {
        set({ isLoading: true, error: null });
        try {
          // TODO: Replace with actual API call
          // const campaign = await campaignApi.get(id);
          const campaign: Campaign | null = null; // Placeholder
          set({ currentCampaign: campaign, isLoading: false });
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : `Failed to fetch campaign ${id}`;
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      fetchCampaignTasks: async (campaignId: string) => {
        set({ isLoading: true, error: null });
        try {
          // TODO: Replace with actual API call
          // const tasks = await campaignApi.getTasks(campaignId);
          const tasks: CampaignTask[] = []; // Placeholder
          set({ campaignTasks: tasks, isLoading: false });
        } catch (error) {
          const errorMessage = error instanceof AppError ? error.message : 'Failed to fetch campaign tasks';
          set({ error: errorMessage, isLoading: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      createCampaign: async (data: Partial<Campaign>) => {
        set({ isCreating: true, error: null });
        
        // Optimistic update
        const optimisticCampaign: Campaign = {
          id: `temp-${Date.now()}`,
          blog_id: data.blog_id || '',
          title: data.title || 'New Campaign',
          description: data.description,
          status: 'planning',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          ...data,
        };
        
        const { campaigns } = get();
        set({ campaigns: [...campaigns, optimisticCampaign] });
        
        const loadingToast = toastService.loading('Creating campaign...');
        
        try {
          // TODO: Replace with actual API call
          // const result = await campaignApi.create(data);
          const result = optimisticCampaign.id; // Placeholder
          
          toastService.dismiss(loadingToast);
          showSuccessNotification('Campaign created successfully!');
          
          // Replace optimistic update with real data
          await get().fetchCampaigns();
          
          set({ isCreating: false });
          return result;
        } catch (error) {
          // Remove optimistic update
          set({ campaigns: campaigns });
          
          toastService.dismiss(loadingToast);
          const errorMessage = error instanceof AppError ? error.message : 'Failed to create campaign';
          set({ error: errorMessage, isCreating: false });
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
          return null;
        }
      },

      updateCampaign: async (id: string, data: Partial<Campaign>) => {
        const { campaigns, currentCampaign } = get();
        
        // Optimistic update
        const updatedCampaigns = campaigns.map(campaign => 
          campaign.id === id ? { ...campaign, ...data } : campaign
        );
        set({ campaigns: updatedCampaigns });
        
        if (currentCampaign && currentCampaign.id === id) {
          set({ currentCampaign: { ...currentCampaign, ...data } });
        }
        
        try {
          // TODO: Replace with actual API call
          // await campaignApi.update(id, data);
          showSuccessNotification('Campaign updated successfully!');
        } catch (error) {
          // Revert optimistic update
          set({ campaigns });
          if (currentCampaign && currentCampaign.id === id) {
            set({ currentCampaign });
          }
          
          const errorMessage = error instanceof AppError ? error.message : 'Failed to update campaign';
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      deleteCampaign: async (id: string) => {
        const { campaigns } = get();
        
        // Optimistic update
        const updatedCampaigns = campaigns.filter(campaign => campaign.id !== id);
        set({ campaigns: updatedCampaigns });
        
        try {
          // TODO: Replace with actual API call
          // await campaignApi.delete(id);
          showSuccessNotification('Campaign deleted successfully!');
        } catch (error) {
          // Revert optimistic update
          set({ campaigns });
          
          const errorMessage = error instanceof AppError ? error.message : 'Failed to delete campaign';
          showErrorNotification(error instanceof AppError ? error : new AppError(errorMessage));
        }
      },

      // Optimistic update helpers
      updateCampaignOptimistic: (id: string, data: Partial<Campaign>) => {
        const { campaigns, currentCampaign } = get();
        
        const updatedCampaigns = campaigns.map(campaign => 
          campaign.id === id ? { ...campaign, ...data } : campaign
        );
        set({ campaigns: updatedCampaigns });
        
        if (currentCampaign && currentCampaign.id === id) {
          set({ currentCampaign: { ...currentCampaign, ...data } });
        }
      },

      updateTaskOptimistic: (id: string, data: Partial<CampaignTask>) => {
        const { campaignTasks } = get();
        
        const updatedTasks = campaignTasks.map(task => 
          task.id === id ? { ...task, ...data } : task
        );
        set({ campaignTasks: updatedTasks });
      },

      // Computed state
      activeCampaigns: () => {
        const { campaigns } = get();
        return campaigns.filter(campaign => 
          campaign.status === 'active' || campaign.status === 'planning'
        );
      },

      completedCampaigns: () => {
        const { campaigns } = get();
        return campaigns.filter(campaign => campaign.status === 'completed');
      },

      pendingTasks: () => {
        const { campaignTasks } = get();
        return campaignTasks.filter(task => 
          task.status === 'pending' || task.status === 'in_progress'
        );
      },
    }),
    {
      name: 'campaign-store',
    }
  )
);