/**
 * UI state management using Zustand
 */
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

export interface UIState {
  // Sidebar state
  sidebarOpen: boolean;
  
  // Modal states
  quickStartWizardOpen: boolean;
  keyboardShortcutsOpen: boolean;
  
  // Loading states (global)
  globalLoading: boolean;
  
  // Preferences
  theme: 'light' | 'dark' | 'system';
  
  // Optimistic updates tracking
  optimisticUpdates: Map<string, any>;
  
  // Actions
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  
  setQuickStartWizardOpen: (open: boolean) => void;
  setKeyboardShortcutsOpen: (open: boolean) => void;
  
  setGlobalLoading: (loading: boolean) => void;
  
  setTheme: (theme: UIState['theme']) => void;
  
  // Optimistic updates
  addOptimisticUpdate: (key: string, value: any) => void;
  removeOptimisticUpdate: (key: string) => void;
  clearOptimisticUpdates: () => void;
  
  // Reset all UI state
  reset: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        sidebarOpen: false,
        quickStartWizardOpen: false,
        keyboardShortcutsOpen: false,
        globalLoading: false,
        theme: 'system',
        optimisticUpdates: new Map(),

        // Sidebar actions
        setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

        // Modal actions
        setQuickStartWizardOpen: (quickStartWizardOpen) => set({ quickStartWizardOpen }),
        setKeyboardShortcutsOpen: (keyboardShortcutsOpen) => set({ keyboardShortcutsOpen }),

        // Loading actions
        setGlobalLoading: (globalLoading) => set({ globalLoading }),

        // Theme actions
        setTheme: (theme) => {
          set({ theme });
          
          // Apply theme to document
          const root = document.documentElement;
          if (theme === 'dark') {
            root.classList.add('dark');
          } else if (theme === 'light') {
            root.classList.remove('dark');
          } else {
            // System preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (prefersDark) {
              root.classList.add('dark');
            } else {
              root.classList.remove('dark');
            }
          }
        },

        // Optimistic updates
        addOptimisticUpdate: (key, value) => {
          const { optimisticUpdates } = get();
          const newUpdates = new Map(optimisticUpdates);
          newUpdates.set(key, value);
          set({ optimisticUpdates: newUpdates });
        },

        removeOptimisticUpdate: (key) => {
          const { optimisticUpdates } = get();
          const newUpdates = new Map(optimisticUpdates);
          newUpdates.delete(key);
          set({ optimisticUpdates: newUpdates });
        },

        clearOptimisticUpdates: () => {
          set({ optimisticUpdates: new Map() });
        },

        // Reset
        reset: () => set({
          sidebarOpen: false,
          quickStartWizardOpen: false,
          keyboardShortcutsOpen: false,
          globalLoading: false,
          optimisticUpdates: new Map(),
        }),
      }),
      {
        name: 'ui-store',
        partialize: (state) => ({
          theme: state.theme,
          sidebarOpen: state.sidebarOpen,
        }),
      }
    ),
    {
      name: 'ui-store',
    }
  )
);