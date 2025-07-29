import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';

export function KeyboardShortcutsHelp() {
  const { shortcuts, showHelp, setShowHelp, formatShortcut } = useKeyboardShortcuts();

  if (!showHelp) {
    return null;
  }

  // Group shortcuts by category
  const shortcutsByCategory = shortcuts.reduce((acc, shortcut) => {
    const category = shortcut.category || 'Other';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(shortcut);
    return acc;
  }, {} as Record<string, typeof shortcuts>);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-gray-900 to-blue-900 text-white p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold">Keyboard Shortcuts</h2>
                <p className="text-blue-100 text-sm">Power user features for faster navigation</p>
              </div>
            </div>
            <button
              onClick={() => setShowHelp(false)}
              className="text-blue-100 hover:text-white transition-colors p-2 rounded-lg hover:bg-white hover:bg-opacity-10"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-96">
          <div className="space-y-6">
            {Object.entries(shortcutsByCategory).map(([category, categoryShortcuts]) => (
              <div key={category}>
                <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                  <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                  {category}
                </h3>
                <div className="grid gap-2">
                  {categoryShortcuts.map((shortcut, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                      <span className="text-gray-700 font-medium">
                        {shortcut.description}
                      </span>
                      <kbd className="inline-flex items-center px-3 py-1 bg-white border border-gray-300 rounded-md text-sm font-mono text-gray-900 shadow-sm">
                        {formatShortcut(shortcut)}
                      </kbd>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Press <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">?</kbd> anytime to toggle this help</span>
            </div>
            <div className="flex items-center space-x-1">
              <span>Press <kbd className="px-2 py-1 bg-gray-200 rounded text-xs">Esc</kbd> to close</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}