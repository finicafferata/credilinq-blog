import { Link, useLocation } from 'react-router-dom';
import { useMemo } from 'react';

interface BreadcrumbItem {
  label: string;
  path: string;
  icon?: React.ReactNode;
}

interface BreadcrumbsProps {
  customItems?: BreadcrumbItem[];
}

export function Breadcrumbs({ customItems }: BreadcrumbsProps) {
  const location = useLocation();

  const breadcrumbs = useMemo(() => {
    if (customItems) {
      return customItems;
    }

    const pathSegments = location.pathname.split('/').filter(Boolean);
    const items: BreadcrumbItem[] = [
      {
        label: 'Dashboard',
        path: '/dashboard',
        icon: (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5a2 2 0 012-2h4a2 2 0 012 2v2H8V5z" />
          </svg>
        )
      }
    ];

    pathSegments.forEach((segment, index) => {
      const path = '/' + pathSegments.slice(0, index + 1).join('/');
      
      let label = segment;
      let icon: React.ReactNode = undefined;

      // Map path segments to human-readable labels and icons
      switch (segment) {
        case 'dashboard':
          return; // Already added as root
        case 'new':
          label = 'Create New Blog';
          icon = (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          );
          break;
        case 'edit':
          label = 'Edit Blog';
          icon = (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          );
          break;
        case 'campaign':
          label = 'Campaign';
          icon = (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
            </svg>
          );
          break;
        case 'analytics':
          label = 'Analytics';
          icon = (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          );
          break;
        case 'knowledge-base':
          label = 'Knowledge Base';
          icon = (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          );
          break;
        default:
          // For dynamic segments like blog IDs, check if it's a UUID-like string
          if (segment.match(/^[a-f0-9-]{36}$/i)) {
            label = `Blog ${segment.slice(0, 8)}...`;
          } else {
            label = segment.charAt(0).toUpperCase() + segment.slice(1);
          }
      }

      items.push({ label, path, icon });
    });

    return items;
  }, [location.pathname, customItems]);

  if (breadcrumbs.length <= 1) {
    return null;
  }

  return (
    <nav className="flex items-center space-x-2 text-sm text-gray-600 mb-6" aria-label="Breadcrumb">
      <ol className="flex items-center space-x-2">
        {breadcrumbs.map((item, index) => (
          <li key={item.path} className="flex items-center">
            {index > 0 && (
              <svg className="w-4 h-4 text-gray-400 mx-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
            {index === breadcrumbs.length - 1 ? (
              <span className="flex items-center space-x-1 text-gray-900 font-medium">
                {item.icon}
                <span>{item.label}</span>
              </span>
            ) : (
              <Link
                to={item.path}
                className="flex items-center space-x-1 hover:text-blue-600 transition-colors"
              >
                {item.icon}
                <span>{item.label}</span>
              </Link>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}