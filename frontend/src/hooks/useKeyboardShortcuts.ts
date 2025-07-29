import { useEffect, useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';

export interface KeyboardShortcut {
  key: string;
  description: string;
  action: () => void;
  category?: string;
  ctrlKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
  metaKey?: boolean;
}

export function useKeyboardShortcuts() {
  const navigate = useNavigate();
  const [showHelp, setShowHelp] = useState(false);

  const shortcuts: KeyboardShortcut[] = [
    // Navigation shortcuts
    {
      key: 'd',
      description: 'Go to Dashboard',
      action: () => navigate('/dashboard'),
      category: 'Navigation',
      altKey: true
    },
    {
      key: 'n',
      description: 'Create New Blog',
      action: () => navigate('/new'),
      category: 'Actions',
      altKey: true
    },
    {
      key: 'a',
      description: 'View Analytics',
      action: () => navigate('/analytics'),
      category: 'Navigation',
      altKey: true
    },
    {
      key: 'k',
      description: 'Knowledge Base',
      action: () => navigate('/knowledge-base'),
      category: 'Navigation',
      altKey: true
    },
    // General shortcuts
    {
      key: '?',
      description: 'Show/Hide Keyboard Shortcuts',
      action: () => setShowHelp(prev => !prev),
      category: 'General'
    },
    {
      key: 'Escape',
      description: 'Close Modal/Dialog',
      action: () => setShowHelp(false),
      category: 'General'
    },
    // Search shortcuts
    {
      key: 'f',
      description: 'Focus Search',
      action: () => {
        const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
        if (searchInput) {
          searchInput.focus();
          searchInput.select();
        }
      },
      category: 'Actions',
      ctrlKey: true
    },
    {
      key: 'r',
      description: 'Refresh Page',
      action: () => window.location.reload(),
      category: 'General',
      ctrlKey: true
    }
  ];

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLTextAreaElement ||
      event.target instanceof HTMLSelectElement ||
      (event.target as HTMLElement).contentEditable === 'true'
    ) {
      // Only allow escape and ctrl+f when in inputs
      if (event.key === 'Escape' || (event.ctrlKey && event.key.toLowerCase() === 'f')) {
        // Continue to handle these shortcuts
      } else {
        return;
      }
    }

    const matchingShortcut = shortcuts.find(shortcut => {
      const keyMatches = shortcut.key.toLowerCase() === event.key.toLowerCase();
      const ctrlMatches = !!shortcut.ctrlKey === event.ctrlKey;
      const altMatches = !!shortcut.altKey === event.altKey;
      const shiftMatches = !!shortcut.shiftKey === event.shiftKey;
      const metaMatches = !!shortcut.metaKey === event.metaKey;

      return keyMatches && ctrlMatches && altMatches && shiftMatches && metaMatches;
    });

    if (matchingShortcut) {
      event.preventDefault();
      matchingShortcut.action();
    }
  }, [shortcuts]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const formatShortcut = (shortcut: KeyboardShortcut): string => {
    const parts: string[] = [];
    
    if (shortcut.ctrlKey) parts.push('Ctrl');
    if (shortcut.altKey) parts.push('Alt');
    if (shortcut.shiftKey) parts.push('Shift');
    if (shortcut.metaKey) parts.push('Cmd');
    
    parts.push(shortcut.key === ' ' ? 'Space' : shortcut.key.toUpperCase());
    
    return parts.join(' + ');
  };

  return {
    shortcuts,
    showHelp,
    setShowHelp,
    formatShortcut
  };
}