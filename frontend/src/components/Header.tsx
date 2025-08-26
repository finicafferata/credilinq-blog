import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

interface MenuItem {
  path: string;
  label: string;
  icon: string;
  isActive?: (pathname: string) => boolean;
}

export function Header() {
  const location = useLocation();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Menu items configuration
  const menuItems: MenuItem[] = [
    // {
    //   path: '/dashboard',
    //   label: 'Orchestration',
    //   icon: '🎛️'
    // },
    // {
    //   path: '/analytics',
    //   label: 'Analytics',
    //   icon: '📈'
    // },
    {
      path: '/campaigns',
      label: 'Campaigns',
      icon: '🎯'
    },
    // {
    //   path: '/content-review',
    //   label: 'Content Review',
    //   icon: '📝'
    // },
    // {
    //   path: '/content-brief',
    //   label: 'Content Brief',
    //   icon: '📋'
    // },
    // {
    //   path: '/agents',
    //   label: 'Agents',
    //   icon: '🤖'
    // },
    // {
    //   path: '/monitoring',
    //   label: 'Monitoring',
    //   icon: '📊'
    // },
    // {
    //   path: '/competitor-intelligence',
    //   label: 'Intelligence',
    //   icon: '🔍',
    //   isActive: (pathname: string) => pathname.startsWith('/competitor-intelligence')
    // },
    {
      path: '/knowledge-base',
      label: 'Knowledge Base',
      icon: '📚'
    },
    // {
    //   path: '/testing',
    //   label: 'Testing',
    //   icon: '🧪'
    // },
    {
      path: '/settings',
      label: 'Settings',
      icon: '⚙️'
    }
  ];

  const isActive = (item: MenuItem) => {
    if (item.isActive) {
      return item.isActive(location.pathname);
    }
    return location.pathname === item.path;
  };

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  // Check if we're on mobile
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
      // On mobile, collapse by default
      if (window.innerWidth < 768) {
        setIsSidebarCollapsed(true);
      } else {
        setIsSidebarCollapsed(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const sidebarWidth = isSidebarCollapsed ? (isMobile ? 0 : 70) : 280;

  return (
    <>
      {/* Left Sidebar - Always visible on desktop */}
      <aside
        className={`fixed top-0 left-0 h-full bg-white border-r border-gray-200 shadow-lg z-40 transition-all duration-300 ease-in-out ${
          isMobile && isSidebarCollapsed ? '-translate-x-full' : 'translate-x-0'
        }`}
        style={{ width: `${sidebarWidth}px` }}
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 h-16">
          {!isSidebarCollapsed && (
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">C</span>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-gray-900">CrediLinq</h1>
                <p className="text-sm text-gray-500">Content Agent</p>
              </div>
            </div>
          )}
          
          <button
            onClick={toggleSidebar}
            className={`p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${isSidebarCollapsed ? 'mx-auto' : ''}`}
            aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <svg 
              className={`w-5 h-5 transition-transform duration-200 ${isSidebarCollapsed ? 'rotate-180' : ''}`}
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
        </div>

        {/* Navigation Menu */}
        <nav className={`p-3 ${isSidebarCollapsed ? 'px-2' : ''}`}>
          <ul className="space-y-2">
            {menuItems.map((item) => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center space-x-3 px-3 py-3 rounded-xl text-sm font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                    isActive(item)
                      ? 'bg-blue-50 text-blue-700 border-l-4 border-blue-500 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 hover:shadow-sm'
                  } ${isSidebarCollapsed ? 'justify-center' : ''}`}
                  aria-current={isActive(item) ? 'page' : undefined}
                  title={isSidebarCollapsed ? item.label : undefined}
                >
                  <span className="text-lg flex-shrink-0" role="img" aria-hidden="true">
                    {item.icon}
                  </span>
                  {!isSidebarCollapsed && (
                    <>
                      <span className="flex-1">{item.label}</span>
                      {isActive(item) && (
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      )}
                    </>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* Sidebar Footer - only show when expanded */}
        {!isSidebarCollapsed && (
          <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
            <div className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-gray-50">
              <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-gray-600 text-sm font-medium">U</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">User</p>
                <p className="text-xs text-gray-500 truncate">user@credilinq.com</p>
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* Mobile Overlay */}
      {isMobile && !isSidebarCollapsed && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setIsSidebarCollapsed(true)}
          aria-hidden="true"
        />
      )}

      {/* Header - Adjusted for sidebar */}
      <header 
        className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-30 transition-all duration-300" 
        style={{ marginLeft: isMobile ? '0px' : `${sidebarWidth}px` }}
        role="banner"
      >
        <div className="px-4">
          <div className="flex items-center justify-between h-16">
            {/* Mobile hamburger - only show on mobile when collapsed */}
            {isMobile && isSidebarCollapsed && (
              <button
                onClick={toggleSidebar}
                className="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                aria-label="Open navigation menu"
              >
                <svg 
                  className="w-6 h-6" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}

            {/* Page Title or Breadcrumbs */}
            <div className="flex-1">
              <h1 className="text-xl font-semibold text-gray-900">
                {menuItems.find(item => isActive(item))?.label || 'CrediLinq Content Agent'}
              </h1>
            </div>

            {/* Header Actions */}
            <div className="flex items-center space-x-4">
              <button className="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 6h11m-11 6h11m-11 6h11" />
                </svg>
              </button>
              <button className="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5-5 5-5v10z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Global style to push main content */}
      <style>
        {`.main-content {
          margin-left: ${isMobile ? '0px' : `${sidebarWidth}px`};
          transition: margin-left 0.3s ease-in-out;
          min-height: calc(100vh - 64px);
        }`}
      </style>
    </>
  );
}