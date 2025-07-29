/**
 * Error handling utilities for consistent frontend error responses
 */

export interface ApiError {
  message: string;
  status: number;
  code?: string;
  details?: any;
}

export class AppError extends Error {
  public readonly status: number;
  public readonly code?: string;
  public readonly details?: any;

  constructor(message: string, status: number = 500, code?: string, details?: any) {
    super(message);
    this.name = 'AppError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/**
 * Extract meaningful error information from API responses
 */
export function parseApiError(error: any): ApiError {
  // Handle axios errors
  if (error.response) {
    const { status, data } = error.response;
    
    // Backend returns structured error response
    if (data && typeof data === 'object') {
      return {
        message: data.detail || data.message || 'An error occurred',
        status: status,
        code: data.code,
        details: data.details
      };
    }
    
    // Fallback for non-structured responses
    return {
      message: getStatusMessage(status),
      status: status
    };
  }

  // Handle network errors
  if (error.request) {
    return {
      message: 'Network error - please check your connection',
      status: 0
    };
  }

  // Handle other errors
  return {
    message: error.message || 'An unexpected error occurred',
    status: 500
  };
}

/**
 * Get user-friendly message for HTTP status codes
 */
function getStatusMessage(status: number): string {
  switch (status) {
    case 400:
      return 'Invalid request - please check your input';
    case 401:
      return 'Authentication required';
    case 403:
      return 'Access denied';
    case 404:
      return 'Resource not found';
    case 409:
      return 'Conflict - resource already exists';
    case 422:
      return 'Validation error - please check your input';
    case 429:
      return 'Too many requests - please try again later';
    case 500:
      return 'Server error - please try again';
    case 502:
      return 'Service temporarily unavailable';
    case 503:
      return 'Service unavailable - please try again later';
    default:
      return `Request failed with status ${status}`;
  }
}

/**
 * Show user-friendly error notification
 */
export function showErrorNotification(error: ApiError | AppError | Error) {
  let message: string;
  
  if (error instanceof AppError) {
    message = error.message;
  } else if ('status' in error) {
    message = error.message;
  } else {
    message = error.message || 'An unexpected error occurred';
  }

  // For now, use alert - could be replaced with toast notifications
  alert(`Error: ${message}`);
}

/**
 * Show success notification
 */
export function showSuccessNotification(message: string) {
  // For now, use alert - could be replaced with toast notifications
  alert(`Success: ${message}`);
}

/**
 * Utility to handle common API call patterns with error handling
 */
export async function handleApiCall<T>(
  apiCall: () => Promise<T>,
  options?: {
    successMessage?: string;
    errorMessage?: string;
    showSuccess?: boolean;
    showError?: boolean;
  }
): Promise<T> {
  const {
    successMessage,
    errorMessage,
    showSuccess = false,
    showError = true
  } = options || {};

  try {
    const result = await apiCall();
    
    if (showSuccess && successMessage) {
      showSuccessNotification(successMessage);
    }
    
    return result;
  } catch (error) {
    const apiError = parseApiError(error);
    
    if (showError) {
      const displayMessage = errorMessage || apiError.message;
      showErrorNotification(new AppError(displayMessage, apiError.status));
    }
    
    throw new AppError(
      errorMessage || apiError.message,
      apiError.status,
      apiError.code,
      apiError.details
    );
  }
}