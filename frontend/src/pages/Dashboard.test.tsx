import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Dashboard } from './Dashboard'
import { render, createMockBlog } from '../test/utils'

// Mock the API module
vi.mock('../lib/api', () => ({
  blogApi: {
    list: vi.fn(),
    delete: vi.fn(),
    publish: vi.fn(),
  }
}))

// Mock the error handling module
vi.mock('../lib/errors', () => ({
  showErrorNotification: vi.fn(),
  showSuccessNotification: vi.fn(),
  AppError: class AppError extends Error {
    constructor(message: string, public status: number = 500) {
      super(message)
    }
  }
}))

import { blogApi } from '../lib/api'

describe('Dashboard', () => {
  const mockBlogs = [
    createMockBlog({ id: 'blog-1', title: 'First Blog', status: 'draft' }),
    createMockBlog({ id: 'blog-2', title: 'Second Blog', status: 'published' }),
    createMockBlog({ id: 'blog-3', title: 'Third Blog', status: 'edited' }),
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    // Default successful API response
    vi.mocked(blogApi.list).mockResolvedValue(mockBlogs)
  })

  it('renders dashboard with blogs', async () => {
    render(<Dashboard />)
    
    expect(screen.getByText('Blog Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Manage your AI-generated content and campaigns')).toBeInTheDocument()
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
      expect(screen.getByText('Second Blog')).toBeInTheDocument()
      expect(screen.getByText('Third Blog')).toBeInTheDocument()
    })
  })

  it('shows loading state initially', () => {
    render(<Dashboard />)
    
    expect(screen.getByTestId('loading-spinner') || screen.getByRole('status')).toBeInTheDocument()
  })

  it('handles empty blog list', async () => {
    vi.mocked(blogApi.list).mockResolvedValue([])
    
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('No blogs yet')).toBeInTheDocument()
      expect(screen.getByText('Get started by creating your first AI-generated blog post.')).toBeInTheDocument()
      expect(screen.getByText('Create First Blog')).toBeInTheDocument()
    })
  })

  it('handles API error gracefully', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.mocked(blogApi.list).mockRejectedValue(new Error('API Error'))
    
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch blogs')).toBeInTheDocument()
      expect(screen.getByText('Retry')).toBeInTheDocument()
    })
    
    consoleError.mockRestore()
  })

  it('filters blogs by search query', async () => {
    const user = userEvent.setup()
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    const searchInput = screen.getByPlaceholderText('Search by title...')
    await user.type(searchInput, 'First')
    
    const searchButton = screen.getByText('Search')
    await user.click(searchButton)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
      expect(screen.queryByText('Second Blog')).not.toBeInTheDocument()
    })
  })

  it('filters blogs by status', async () => {
    const user = userEvent.setup()
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    const statusSelect = screen.getByDisplayValue('All Statuses')
    await user.selectOptions(statusSelect, 'published')
    
    await waitFor(() => {
      expect(screen.getByText('Second Blog')).toBeInTheDocument()
      expect(screen.queryByText('First Blog')).not.toBeInTheDocument()
    })
  })

  it('clears filters when clear button is clicked', async () => {
    const user = userEvent.setup()
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    // Apply search filter
    const searchInput = screen.getByPlaceholderText('Search by title...')
    await user.type(searchInput, 'First')
    
    const searchButton = screen.getByText('Search')
    await user.click(searchButton)
    
    // Clear filters
    const clearButton = screen.getByText('Clear Filters')
    await user.click(clearButton)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
      expect(screen.getByText('Second Blog')).toBeInTheDocument()
      expect(screen.getByText('Third Blog')).toBeInTheDocument()
    })
  })

  it('deletes blog when delete is confirmed', async () => {
    const user = userEvent.setup()
    vi.mocked(blogApi.delete).mockResolvedValue({ message: 'Blog deleted successfully', id: 'blog-1' })
    
    // Mock window.confirm to return true
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    const deleteButton = screen.getAllByText('Delete')[0]
    await user.click(deleteButton)
    
    expect(confirmSpy).toHaveBeenCalledWith('Are you sure you want to delete this blog post?')
    expect(blogApi.delete).toHaveBeenCalledWith('blog-1')
    
    confirmSpy.mockRestore()
  })

  it('does not delete blog when delete is cancelled', async () => {
    const user = userEvent.setup()
    
    // Mock window.confirm to return false
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    const deleteButton = screen.getAllByText('Delete')[0]
    await user.click(deleteButton)
    
    expect(confirmSpy).toHaveBeenCalled()
    expect(blogApi.delete).not.toHaveBeenCalled()
    
    confirmSpy.mockRestore()
  })

  it('shows create new blog button', async () => {
    render(<Dashboard />)
    
    const createButton = screen.getByText('+ Create New Blog')
    expect(createButton).toBeInTheDocument()
    expect(createButton.closest('a')).toHaveAttribute('href', '/new')
  })

  it('shows correct blog count', async () => {
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('3 blogs found')).toBeInTheDocument()
    })
  })

  it('shows filtered count when filters are applied', async () => {
    const user = userEvent.setup()
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    // Apply status filter
    const statusSelect = screen.getByDisplayValue('All Statuses')
    await user.selectOptions(statusSelect, 'published')
    
    await waitFor(() => {
      expect(screen.getByText('1 blog found (filtered)')).toBeInTheDocument()
    })
  })

  it('handles search form submission', async () => {
    const user = userEvent.setup()
    render(<Dashboard />)
    
    await waitFor(() => {
      expect(screen.getByText('First Blog')).toBeInTheDocument()
    })
    
    const searchInput = screen.getByPlaceholderText('Search by title...')
    await user.type(searchInput, 'Second')
    
    // Submit form using Enter key
    await user.type(searchInput, '{enter}')
    
    await waitFor(() => {
      expect(screen.getByText('Second Blog')).toBeInTheDocument()
      expect(screen.queryByText('First Blog')).not.toBeInTheDocument()
    })
  })