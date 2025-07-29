import { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'

// Custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      {children}
    </BrowserRouter>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

export * from '@testing-library/react'
export { customRender as render }

// Test utilities
export const createMockNavigate = () => vi.fn()

export const createMockLocation = (pathname: string = '/', search: string = '') => ({
  pathname,
  search,
  hash: '',
  state: null,
  key: 'default'
})

// Mock data factories
export const createMockBlog = (overrides = {}) => ({
  id: 'test-blog-123',
  title: 'Test Blog Post',
  status: 'draft',
  created_at: '2024-01-15T10:30:00Z',
  ...overrides
})

export const createMockBlogDetail = (overrides = {}) => ({
  id: 'test-blog-123',
  title: 'Test Blog Post',
  status: 'draft',
  created_at: '2024-01-15T10:30:00Z',
  content_markdown: '# Test Blog\n\nThis is test content.',
  initial_prompt: {
    title: 'Test Blog Post',
    company_context: 'Test context',
    content_type: 'blog'
  },
  ...overrides
})

export const createMockCampaign = (overrides = {}) => ({
  id: 'test-campaign-123',
  blog_id: 'test-blog-123',
  created_at: '2024-01-15T11:00:00Z',
  tasks: [
    {
      id: 'task-1',
      task_type: 'repurpose',
      target_format: 'linkedin_post',
      status: 'pending',
      result: null,
      error: null,
      created_at: '2024-01-15T11:00:00Z',
      updated_at: '2024-01-15T11:00:00Z'
    }
  ],
  ...overrides
})

// Async testing utilities
export const waitForLoadingToFinish = async () => {
  const { findByText } = await import('@testing-library/react')
  // Wait for loading spinners to disappear
  const loadingIndicators = document.querySelectorAll('.animate-spin')
  if (loadingIndicators.length > 0) {
    await new Promise(resolve => setTimeout(resolve, 100))
  }
}

// Event simulation utilities
export const simulateApiDelay = (ms: number = 100) => {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Form testing utilities
export const fillForm = async (fields: Record<string, string>) => {
  const { userEvent } = await import('@testing-library/user-event')
  const user = userEvent.setup()
  
  for (const [name, value] of Object.entries(fields)) {
    const field = document.querySelector(`[name="${name}"]`) as HTMLInputElement | HTMLTextAreaElement
    if (field) {
      await user.clear(field)
      await user.type(field, value)
    }
  }
}

// Error boundary for testing error states
export const TestErrorBoundary = ({ children }: { children: React.ReactNode }) => {
  return (
    <div data-testid="error-boundary">
      {children}
    </div>
  )
}