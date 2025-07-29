import { ReactNode } from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'white' | 'gray';
  className?: string;
}

export function LoadingSpinner({ size = 'md', color = 'primary', className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const colorClasses = {
    primary: 'border-primary-600',
    white: 'border-white',
    gray: 'border-gray-600'
  };

  return (
    <div
      className={`animate-spin rounded-full border-2 border-transparent ${sizeClasses[size]} ${colorClasses[color]} ${className}`}
      style={{
        borderTopColor: 'currentColor',
        borderRightColor: 'currentColor'
      }}
    />
  );
}

interface SkeletonProps {
  className?: string;
  animate?: boolean;
}

export function Skeleton({ className = 'h-4 w-full', animate = true }: SkeletonProps) {
  return (
    <div
      className={`bg-gray-200 rounded ${animate ? 'animate-pulse' : ''} ${className}`}
    />
  );
}

interface BlogCardSkeletonProps {
  count?: number;
}

export function BlogCardSkeleton({ count = 6 }: BlogCardSkeletonProps) {
  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="card">
          <div className="animate-pulse">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <Skeleton className="h-6 w-3/4 mb-2" />
                <div className="flex items-center space-x-2">
                  <Skeleton className="h-5 w-16 rounded-full" />
                  <Skeleton className="h-2 w-2 rounded-full" />
                  <Skeleton className="h-4 w-20" />
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex space-x-2">
                <Skeleton className="h-8 w-16 rounded" />
                <Skeleton className="h-8 w-20 rounded" />
                <Skeleton className="h-8 w-24 rounded" />
              </div>
              <Skeleton className="h-4 w-12" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

interface ProgressBarProps {
  progress: number;
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'green' | 'yellow' | 'red';
  showPercentage?: boolean;
  label?: string;
  className?: string;
}

export function ProgressBar({
  progress,
  size = 'md',
  color = 'primary',
  showPercentage = false,
  label,
  className = ''
}: ProgressBarProps) {
  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3'
  };

  const colorClasses = {
    primary: 'bg-primary-600',
    green: 'bg-green-600',
    yellow: 'bg-yellow-600',
    red: 'bg-red-600'
  };

  const clampedProgress = Math.min(Math.max(progress, 0), 100);

  return (
    <div className={className}>
      {(label || showPercentage) && (
        <div className="flex items-center justify-between mb-2 text-sm">
          {label && <span className="font-medium text-gray-700">{label}</span>}
          {showPercentage && <span className="text-gray-500">{Math.round(clampedProgress)}%</span>}
        </div>
      )}
      <div className={`w-full bg-gray-200 rounded-full ${sizeClasses[size]} overflow-hidden`}>
        <div
          className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full transition-all duration-300 ease-out`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
    </div>
  );
}

interface LoadingOverlayProps {
  isVisible: boolean;
  message?: string;
  progress?: number;
  children?: ReactNode;
}

export function LoadingOverlay({ isVisible, message, progress, children }: LoadingOverlayProps) {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-40">
      <div className="bg-white rounded-xl shadow-2xl p-8 max-w-sm w-full mx-4">
        <div className="text-center">
          <LoadingSpinner size="xl" className="mx-auto mb-4" />
          {message && (
            <p className="text-gray-700 font-medium mb-4">{message}</p>
          )}
          {typeof progress === 'number' && (
            <ProgressBar
              progress={progress}
              showPercentage
              className="mb-4"
            />
          )}
          {children}
        </div>
      </div>
    </div>
  );
}

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({ icon, title, description, action, className = '' }: EmptyStateProps) {
  const defaultIcon = (
    <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  );

  return (
    <div className={`text-center py-12 ${className}`}>
      <div className="w-24 h-24 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
        {icon || defaultIcon}
      </div>
      <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
      {description && (
        <p className="text-gray-600 mb-6 max-w-md mx-auto">{description}</p>
      )}
      {action && (
        <button onClick={action.onClick} className="btn-primary">
          {action.label}
        </button>
      )}
    </div>
  );
}

interface StatusIndicatorProps {
  status: 'loading' | 'success' | 'error' | 'warning' | 'info';
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function StatusIndicator({ status, message, size = 'md' }: StatusIndicatorProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'loading':
        return <LoadingSpinner size={size === 'lg' ? 'md' : 'sm'} />;
      case 'success':
        return (
          <svg className={`${sizeClasses[size]} text-green-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case 'error':
        return (
          <svg className={`${sizeClasses[size]} text-red-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
      case 'warning':
        return (
          <svg className={`${sizeClasses[size]} text-yellow-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case 'info':
        return (
          <svg className={`${sizeClasses[size]} text-blue-500`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  return (
    <div className="flex items-center space-x-2">
      {getStatusIcon()}
      {message && (
        <span className={`text-sm ${
          status === 'error' ? 'text-red-700' :
          status === 'success' ? 'text-green-700' :
          status === 'warning' ? 'text-yellow-700' :
          status === 'info' ? 'text-blue-700' :
          'text-gray-700'
        }`}>
          {message}
        </span>
      )}
    </div>
  );
}