import { Link, useLocation } from 'react-router-dom';

export function Header() {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <Link to="/dashboard" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">C</span>
              </div>
              <span className="text-xl font-semibold text-gray-900">Credilinq Content Agent</span>
            </Link>
            
            <nav className="flex space-x-6">
              <Link
                to="/dashboard"
                className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  isActive('/dashboard') 
                    ? 'bg-primary-100 text-primary-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </Link>
              <Link
                to="/analytics"
                className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  isActive('/analytics') 
                    ? 'bg-primary-100 text-primary-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Analytics
              </Link>
              <Link
                to="/knowledge-base"
                className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  isActive('/knowledge-base') 
                    ? 'bg-primary-100 text-primary-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Knowledge Base
              </Link>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
} 