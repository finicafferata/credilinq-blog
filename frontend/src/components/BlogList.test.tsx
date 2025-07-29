import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { BlogList } from './BlogList'
import { render, createMockBlog } from '../test/utils'

// Mock the API module
vi.mock('../lib/api', () => ({
  blogApi: {
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
import { showSuccessNotification, showErrorNotification } from '../lib/errors'

describe('BlogList', () => {
  const mockOnDelete = vi.fn()
  const mockOnRefresh = vi.fn()
  
  const mockBlogs = [
    createMockBlog({ id: 'blog-1', title: 'Draft Blog', status: 'draft' }),
    createMockBlog({ id: 'blog-2', title: 'Published Blog', status: 'published' }),
    createMockBlog({ id: 'blog-3', title: 'Edited Blog', status: 'edited' }),
    createMockBlog({ id: 'blog-4', title: 'Completed Blog', status: 'completed' }),
  ]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders blog list correctly', () => {
    render(<BlogList blogs={mockBlogs} onDelete={mockOnDelete} onRefresh={mockOnRefresh} />)
    
    expect(screen.getByText('Draft Blog')).toBeInTheDocument()
    expect(screen.getByText('Published Blog')).toBeInTheDocument() 
    expect(screen.getByText('Edited Blog')).toBeInTheDocument()
    expect(screen.getByText('Completed Blog')).toBeInTheDocument()
  })

  it('shows empty state when no blogs', () => {
    render(<BlogList blogs={[]} />)
    
    expect(screen.getByText('No blogs yet')).toBeInTheDocument()
    expect(screen.getByText('Get started by creating your first AI-generated blog post.')).toBeInTheDocument()
    expect(screen.getByText('Create First Blog')).toBeInTheDocument()
  })

  it('displays correct status badges', () => {
    render(<BlogList blogs={mockBlogs} />)
    
    const draftBadge = screen.getByText('draft')
    const publishedBadge = screen.getByText('published')
    const editedBadge = screen.getByText('edited')
    const completedBadge = screen.getByText('completed')
    
    expect(draftBadge).toHaveClass('bg-yellow-100', 'text-yellow-800')
    expect(publishedBadge).toHaveClass('bg-green-100', 'text-green-800')
    expect(editedBadge).toHaveClass('bg-blue-100', 'text-blue-800')
    expect(completedBadge).toHaveClass('bg-purple-100', 'text-purple-800')
  })

  it('shows edit button for all blogs', () => {
    render(<BlogList blogs={mockBlogs} />)
    
    const editButtons = screen.getAllByText('Edit')
    expect(editButtons).toHaveLength(4)
    
    editButtons.forEach((button, index) => {
      expect(button.closest('a')).toHaveAttribute('href', `/edit/blog-${index + 1}`)
    })
  })

  it('shows publish button only for draft and edited blogs', () => {
    render(<BlogList blogs={mockBlogs} onRefresh={mockOnRefresh} />)
    
    const publishButtons = screen.getAllByText('Publish')
    expect(publishButtons).toHaveLength(2) // Only draft and edited blogs
  })

  it('shows create campaign button for edited, completed, and published blogs', () => {
    render(<BlogList blogs={mockBlogs} />)
    
    const campaignButtons = screen.getAllByText('Create Campaign')
    expect(campaignButtons).toHaveLength(3) // edited, completed, and published blogs
  })

  it('handles blog publishing successfully', async () => {
    const user = userEvent.setup()
    vi.mocked(blogApi.publish).mockResolvedValue(createMockBlog({ status: 'published' }))
    
    // Mock window.confirm to return true
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<BlogList blogs={[mockBlogs[0]]} onRefresh={mockOnRefresh} />)
    
    const publishButton = screen.getByText('Publish')
    await user.click(publishButton)
    
    expect(confirmSpy).toHaveBeenCalledWith('Are you sure you want to publish this blog post?')
    expect(blogApi.publish).toHaveBeenCalledWith('blog-1')
    
    await waitFor(() => {
      expect(showSuccessNotification).toHaveBeenCalledWith('Blog published successfully!')
      expect(mockOnRefresh).toHaveBeenCalled()
    })
    
    confirmSpy.mockRestore()
  })

  it('does not publish when confirmation is cancelled', async () => {
    const user = userEvent.setup()
    
    // Mock window.confirm to return false
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    
    render(<BlogList blogs={[mockBlogs[0]]} onRefresh={mockOnRefresh} />)
    
    const publishButton = screen.getByText('Publish')
    await user.click(publishButton)
    
    expect(confirmSpy).toHaveBeenCalled()
    expect(blogApi.publish).not.toHaveBeenCalled()
    expect(mockOnRefresh).not.toHaveBeenCalled()
    
    confirmSpy.mockRestore()
  })

  it('handles publish error gracefully', async () => {
    const user = userEvent.setup()
    vi.mocked(blogApi.publish).mockRejectedValue(new Error('Publish failed'))
    
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<BlogList blogs={[mockBlogs[0]]} onRefresh={mockOnRefresh} />)
    
    const publishButton = screen.getByText('Publish')
    await user.click(publishButton)
    
    await waitFor(() => {
      expect(showErrorNotification).toHaveBeenCalled()
    })
    
    confirmSpy.mockRestore()
  })

  it('calls onDelete when delete button is clicked', async () => {
    const user = userEvent.setup()
    
    render(<BlogList blogs={mockBlogs} onDelete={mockOnDelete} />)
    
    const deleteButtons = screen.getAllByText('Delete')
    await user.click(deleteButtons[0])
    
    expect(mockOnDelete).toHaveBeenCalledWith('blog-1')
  })

  it('does not show delete button when onDelete is not provided', () => {
    render(<BlogList blogs={mockBlogs} />)
    
    expect(screen.queryByText('Delete')).not.toBeInTheDocument()
  })

  it('formats dates correctly', () => {
    const blogWithDate = createMockBlog({
      created_at: '2024-01-15T10:30:00Z'
    })
    
    render(<BlogList blogs={[blogWithDate]} />)
    
    expect(screen.getByText('Jan 15, 2024')).toBeInTheDocument()
  })

  it('truncates long blog titles', () => {
    const blogWithLongTitle = createMockBlog({
      title: 'This is a very long blog title that should be truncated when displayed in the blog list component'
    })
    
    render(<BlogList blogs={[blogWithLongTitle]} />)
    
    const titleElement = screen.getByText('This is a very long blog title that should be truncated when displayed in the blog list component')
    expect(titleElement).toHaveClass('line-clamp-2')
  })

  it('shows correct campaign link URLs', () => {
    render(<BlogList blogs={[mockBlogs[1]]} />) // Published blog
    
    const campaignLink = screen.getByText('Create Campaign')
    expect(campaignLink.closest('a')).toHaveAttribute('href', '/campaign/blog-2')
  })

  it('handles blogs without onRefresh callback', async () => {
    const user = userEvent.setup()
    vi.mocked(blogApi.publish).mockResolvedValue(createMockBlog({ status: 'published' }))
    
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<BlogList blogs={[mockBlogs[0]]} />) // No onRefresh provided
    
    const publishButton = screen.getByText('Publish')
    await user.click(publishButton)
    
    await waitFor(() => {
      expect(showSuccessNotification).toHaveBeenCalled()
    })
    
    // Should not crash when onRefresh is not provided
    confirmSpy.mockRestore()
  })

  it('displays proper hover effects', () => {
    render(<BlogList blogs={[mockBlogs[0]]} />)
    
    const cardElement = screen.getByText('Draft Blog').closest('.card')
    expect(cardElement).toHaveClass('hover:shadow-md', 'transition-shadow')
  })

  it('shows correct button styles', () => {
    render(<BlogList blogs={mockBlogs} onDelete={mockOnDelete} onRefresh={mockOnRefresh} />)
    
    const editButton = screen.getAllByText('Edit')[0]
    const publishButton = screen.getAllByText('Publish')[0]
    const deleteButton = screen.getAllByText('Delete')[0]
    
    expect(editButton).toHaveClass('btn-primary')
    expect(publishButton).toHaveClass('btn-secondary')
    expect(deleteButton).toHaveClass('text-gray-500', 'hover:text-red-600')
  })