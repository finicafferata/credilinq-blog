import { useEffect, useRef, useCallback, useState } from 'react';
import { blogApi } from '../lib/api';
import { showErrorNotification, AppError } from '../lib/errors';

interface AutoSaveOptions {
  delay?: number; // Delay in milliseconds before saving (default: 2000)
  enabled?: boolean; // Whether auto-save is enabled (default: true)
  onSaveStart?: () => void;
  onSaveSuccess?: () => void;
  onSaveError?: (error: Error) => void;
}

interface AutoSaveState {
  isSaving: boolean;
  lastSaved: Date | null;
  hasUnsavedChanges: boolean;
  saveCount: number;
}

export interface AutoSaveReturn extends AutoSaveState {
  triggerSave: () => Promise<void>;
  enableAutoSave: () => void;
  disableAutoSave: () => void;
  resetUnsavedChanges: () => void;
}

export function useAutoSave(
  blogId: string | undefined,
  content: string,
  options: AutoSaveOptions = {}
): AutoSaveReturn {
  const {
    delay = 2000,
    enabled = true,
    onSaveStart,
    onSaveSuccess,
    onSaveError
  } = options;

  const [state, setState] = useState<AutoSaveState>({
    isSaving: false,
    lastSaved: null,
    hasUnsavedChanges: false,
    saveCount: 0
  });

  const timeoutRef = useRef<NodeJS.Timeout>();
  const previousContentRef = useRef(content);
  const contentRef = useRef(content);
  const isEnabledRef = useRef(enabled);
  const isMountedRef = useRef(true);
  const onSaveStartRef = useRef(onSaveStart);
  const onSaveSuccessRef = useRef(onSaveSuccess);
  const onSaveErrorRef = useRef(onSaveError);

  // Update refs with latest values
  useEffect(() => {
    contentRef.current = content;
    isEnabledRef.current = enabled;
    onSaveStartRef.current = onSaveStart;
    onSaveSuccessRef.current = onSaveSuccess;
    onSaveErrorRef.current = onSaveError;
  }, [content, enabled, onSaveStart, onSaveSuccess, onSaveError]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  // Save function
  const save = useCallback(async () => {
    if (!blogId || !isMountedRef.current) return;

    try {
      setState(prev => ({ ...prev, isSaving: true }));
      onSaveStartRef.current?.();

      await blogApi.update(blogId, { content_markdown: contentRef.current });

      if (isMountedRef.current) {
        setState(prev => ({
          ...prev,
          isSaving: false,
          lastSaved: new Date(),
          hasUnsavedChanges: false,
          saveCount: prev.saveCount + 1
        }));
        previousContentRef.current = contentRef.current;
        onSaveSuccessRef.current?.();
      }
    } catch (error) {
      if (isMountedRef.current) {
        setState(prev => ({ ...prev, isSaving: false }));
        const appError = error instanceof AppError ? error : new AppError('Failed to auto-save content');
        onSaveErrorRef.current?.(appError);
        
        // Only show error notification for non-network errors
        if (!(error instanceof AppError) || error.status !== 0) {
          showErrorNotification(appError);
        }
      }
    }
  }, [blogId]); // Use refs for content and callbacks

  // Manual save trigger
  const triggerSave = useCallback(async () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    await save();
  }, [save]);

  // Content change effect
  useEffect(() => {
    // Don't trigger auto-save on first render or if content hasn't changed
    if (previousContentRef.current === content) {
      return;
    }

    // Mark as having unsaved changes
    setState(prev => ({ ...prev, hasUnsavedChanges: true }));

    // Skip auto-save if disabled or no blog ID
    if (!isEnabledRef.current || !blogId) {
      return;
    }

    // Clear existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Set new timeout for auto-save
    timeoutRef.current = setTimeout(() => {
      save();
    }, delay);

    // Cleanup function
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [content, blogId, delay]); // Removed 'save' from dependencies

  // Control functions
  const enableAutoSave = useCallback(() => {
    isEnabledRef.current = true;
  }, []);

  const disableAutoSave = useCallback(() => {
    isEnabledRef.current = false;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  }, []);

  const resetUnsavedChanges = useCallback(() => {
    setState(prev => ({ ...prev, hasUnsavedChanges: false }));
    previousContentRef.current = contentRef.current;
  }, []);

  return {
    ...state,
    triggerSave,
    enableAutoSave,
    disableAutoSave,
    resetUnsavedChanges
  };
}

// Hook for managing auto-save status display
export function useAutoSaveStatus(autoSave: AutoSaveReturn) {
  const getStatusText = useCallback((): string => {
    if (autoSave.isSaving) {
      return 'Saving...';
    }
    if (autoSave.hasUnsavedChanges) {
      return 'Unsaved changes';
    }
    if (autoSave.lastSaved) {
      const timeDiff = Date.now() - autoSave.lastSaved.getTime();
      if (timeDiff < 60000) { // Less than 1 minute
        return 'Saved just now';
      } else if (timeDiff < 3600000) { // Less than 1 hour
        const minutes = Math.floor(timeDiff / 60000);
        return `Saved ${minutes} minute${minutes > 1 ? 's' : ''} ago`;
      } else {
        return `Saved at ${autoSave.lastSaved.toLocaleTimeString()}`;
      }
    }
    return 'Not saved';
  }, [autoSave]);

  const getStatusColor = useCallback((): string => {
    if (autoSave.isSaving) {
      return 'text-blue-600';
    }
    if (autoSave.hasUnsavedChanges) {
      return 'text-yellow-600';
    }
    if (autoSave.lastSaved) {
      return 'text-green-600';
    }
    return 'text-gray-600';
  }, [autoSave]);

  const getStatusIcon = useCallback((): string => {
    if (autoSave.isSaving) {
      return '⏳';
    }
    if (autoSave.hasUnsavedChanges) {
      return '⚠️';
    }
    if (autoSave.lastSaved) {
      return '✅';
    }
    return '❌';
  }, [autoSave]);

  return {
    statusText: getStatusText(),
    statusColor: getStatusColor(),
    statusIcon: getStatusIcon()
  };
}