/**
 * Lazy loading utilities for React components with loading states and error boundaries
 */
import { lazy, Suspense, Component, ReactNode, ComponentType } from 'react'

/**
 * Loading component with spinner
 */
export const LoadingSpinner = ({ message = 'Loading...' }: { message?: string }) => (
  <div className="flex items-center justify-center min-h-[200px] bg-gray-50 rounded-lg">
    <div className="flex flex-col items-center space-y-4">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <p className="text-gray-600 text-sm">{message}</p>
    </div>
  </div>
)

/**
 * Full page loading component
 */
export const FullPageLoader = ({ message = 'Loading page...' }: { message?: string }) => (
  <div className="min-h-screen bg-gray-50 flex items-center justify-center">
    <div className="flex flex-col items-center space-y-4">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      <p className="text-gray-600">{message}</p>
    </div>
  </div>
)

/**
 * Error boundary for lazy loaded components
 */
interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

class LazyLoadErrorBoundary extends Component<
  { children: ReactNode; fallback?: ComponentType<{ error: Error }> },
  ErrorBoundaryState
> {
  private retryTimeoutId: number | null = null

  constructor(props: any) {
    super(props)
    this.state = { hasError: false }
    this.handleRetry = this.handleRetry.bind(this)
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // Ensure error is properly serialized to avoid object-to-primitive conversion issues
    const safeError = error instanceof Error ? error : new Error(String(error))
    return { hasError: true, error: safeError }
  }

  componentDidCatch(error: Error, errorInfo: any) {
    // Safe error logging with proper error serialization
    const errorMessage = error instanceof Error ? error.message : String(error)
    const errorStack = error instanceof Error ? error.stack : 'No stack trace available'
    
    console.error('Lazy load error:', {
      message: errorMessage,
      stack: errorStack,
      componentStack: errorInfo.componentStack
    })

    // Report to error tracking service if available
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'exception', {
        description: errorMessage,
        fatal: false
      })
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId)
    }
  }

  handleRetry() {
    this.setState({ hasError: false, error: undefined })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback && this.state.error) {
        const FallbackComponent = this.props.fallback
        return <FallbackComponent error={this.state.error} />
      }

      const errorMessage = this.state.error?.message || 'An error occurred while loading this component.'

      return (
        <div className="min-h-[200px] bg-red-50 border border-red-200 rounded-lg p-6 m-4">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0">
              <svg className="h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-medium text-red-800">Failed to load component</h3>
              <p className="text-red-600 text-sm mt-1">
                {errorMessage}
              </p>
              <div className="flex space-x-2 mt-3">
                <button 
                  className="px-4 py-2 bg-red-600 text-white rounded-md text-sm hover:bg-red-700 transition-colors"
                  onClick={this.handleRetry}
                >
                  Retry
                </button>
                <button 
                  className="px-4 py-2 border border-red-300 text-red-600 rounded-md text-sm hover:bg-red-50 transition-colors"
                  onClick={() => window.location.reload()}
                >
                  Reload Page
                </button>
              </div>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Enhanced lazy loading wrapper with error boundary and custom loading
 */
export const lazyWithFallback = <P extends object>(
  importFunc: () => Promise<{ default: ComponentType<P> }>,
  options: {
    fallback?: ComponentType
    errorFallback?: ComponentType<{ error: Error }>
    loadingMessage?: string
    name?: string
  } = {}
) => {
  // Add proper error handling and type guards
  const wrappedImportFunc = async () => {
    try {
      const module = await importFunc()
      
      // Type guard to ensure proper module structure
      if (!module || typeof module !== 'object') {
        throw new Error(`Invalid module structure for component: ${options.name || 'Unknown'}`)
      }
      
      if (!module.default || typeof module.default !== 'function') {
        throw new Error(`Component ${options.name || 'Unknown'} does not have a valid default export`)
      }
      
      return module
    } catch (error) {
      console.error(`Failed to load component ${options.name || 'Unknown'}:`, error)
      throw error
    }
  }

  const LazyComponent = lazy(wrappedImportFunc)
  
  // Add displayName for better debugging
  const WrappedComponent = (props: P) => (
    <LazyLoadErrorBoundary fallback={options.errorFallback}>
      <Suspense fallback={
        options.fallback ? 
          <options.fallback /> : 
          <LoadingSpinner message={options.loadingMessage || 'Loading...'} />
      }>
        <LazyComponent {...props} />
      </Suspense>
    </LazyLoadErrorBoundary>
  )
  
  WrappedComponent.displayName = `LazyWithFallback(${options.name || 'Component'})`
  
  return WrappedComponent
}

/**
 * Preload a lazy component
 */
export const preloadComponent = (importFunc: () => Promise<any>) => {
  const componentImport = importFunc()
  return componentImport
}

/**
 * Hook for preloading components on hover or other events
 */
export const usePreloadOnHover = (importFunc: () => Promise<any>) => {
  let preloadPromise: Promise<any> | null = null

  const preload = () => {
    if (!preloadPromise) {
      preloadPromise = importFunc()
    }
    return preloadPromise
  }

  return {
    onMouseEnter: preload,
    onFocus: preload
  }
}

/**
 * Route-based lazy loading with loading states
 */
export const LazyRoute = ({ 
  children, 
  loading = <FullPageLoader />,
  error
}: { 
  children: ReactNode
  loading?: ReactNode
  error?: ComponentType<{ error: Error }>
}) => (
  <LazyLoadErrorBoundary fallback={error}>
    <Suspense fallback={loading}>
      {children}
    </Suspense>
  </LazyLoadErrorBoundary>
)

/**
 * Performance monitoring for lazy loaded components
 */
export const withPerformanceTracking = <P extends object>(
  Component: ComponentType<P>,
  componentName: string
) => {
  return (props: P) => {
    const startTime = performance.now()

    // Track component mount time
    const trackMount = () => {
      const loadTime = performance.now() - startTime
      if (loadTime > 100) { // Only track if over 100ms
        console.log(`Component ${componentName} loaded in ${loadTime.toFixed(2)}ms`)
      }
    }

    return (
      <div onLoad={trackMount}>
        <Component {...props} />
      </div>
    )
  }
}