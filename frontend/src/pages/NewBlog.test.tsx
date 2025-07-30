import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { NewBlog } from './NewBlog'
import { render, createMockBlog } from '../test/utils'

// Mock react-router-dom
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

// Mock the API module
vi.mock('../lib/api', () => ({
  blogApi: {
    create: vi.fn(),
  }
}))

// Mock the error handling module
vi.mock('../lib/errors', () => ({
  showErrorNotification: vi.fn(),
  AppError: class AppError extends Error {
    constructor(message: string, public status: number = 500) {
      super(message)
    }
  }
}))

import { blogApi } from '../lib/api'
import { showErrorNotification } from '../lib/errors'

describe('NewBlog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders new blog form', () => {
    render(<NewBlog />)
    
    expect(screen.getByText('Create New Content')).toBeInTheDocument()
    expect(screen.getByText('Let our AI agents create professional content for LinkedIn or comprehensive blog posts')).toBeInTheDocument()
    
    expect(screen.getByText('Content Type')).toBeInTheDocument()
    expect(screen.getByText('📝 Blog Post')).toBeInTheDocument()
    expect(screen.getByText('💼 LinkedIn Post')).toBeInTheDocument()
    
    expect(screen.getByLabelText('Blog Title')).toBeInTheDocument()
    expect(screen.getByLabelText('Company Context')).toBeInTheDocument()
  })

  it('has default company context pre-filled', () => {
    render(<NewBlog />)
    
    const contextTextarea = screen.getByLabelText('Company Context')
    expect(contextTextarea).toHaveValue('Credilinq.ai is a global fintech leader in embedded lending and B2B credit solutions, operating across Southeast Asia, Europe, and the United States. We empower businesses to access funding through embedded financial products and cutting-edge credit infrastructure tailored to digital platforms and marketplaces.')
  })

  it('defaults to blog content type', () => {
    render(<NewBlog />)
    
    const blogButton = screen.getByText('📝 Blog Post').closest('button')
    const linkedinButton = screen.getByText('💼 LinkedIn Post').closest('button')
    
    expect(blogButton).toHaveClass('border-primary-500', 'bg-primary-50', 'text-primary-700')
    expect(linkedinButton).not.toHaveClass('border-primary-500')
  })

  it('switches content type when clicked', async () => {
    const user = userEvent.setup()
    render(<NewBlog />)
    
    const linkedinButton = screen.getByText('💼 LinkedIn Post').closest('button')
    await user.click(linkedinButton!)
    
    expect(linkedinButton).toHaveClass('border-primary-500', 'bg-primary-50', 'text-primary-700')
    expect(screen.getByLabelText('Post Title')).toBeInTheDocument()
  })

  it('updates form labels based on content type', async () => {
    const user = userEvent.setup()
    render(<NewBlog />)
    
    // Initially shows blog labels
    expect(screen.getByLabelText('Blog Title')).toBeInTheDocument()
    
    // Switch to LinkedIn
    const linkedinButton = screen.getByText('💼 LinkedIn Post').closest('button')
    await user.click(linkedinButton!)
    
    expect(screen.getByLabelText('Post Title')).toBeInTheDocument()
    expect(screen.queryByLabelText('Blog Title')).not.toBeInTheDocument()
  })

  it('creates blog successfully', async () => {
    const user = userEvent.setup()
    const mockBlog = createMockBlog({ id: 'new-blog-123', title: 'Test Blog' })
    vi.mocked(blogApi.create).mockResolvedValue(mockBlog)
    
    render(<NewBlog />)
    
    // Fill form
    const titleInput = screen.getByLabelText('Blog Title')
    await user.clear(titleInput)
    await user.type(titleInput, 'Test Blog')
    
    const contextTextarea = screen.getByLabelText('Company Context')
    await user.clear(contextTextarea)
    await user.type(contextTextarea, 'Test company context')
    
    // Submit form
    const submitButton = screen.getByText('Create Blog')
    await user.click(submitButton)
    
    expect(blogApi.create).toHaveBeenCalledWith({
      title: 'Test Blog',
      company_context: 'Test company context',
      content_type: 'blog'
    })
    
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/edit/new-blog-123')
    })
  })

  it('creates LinkedIn post successfully', async () => {
    const user = userEvent.setup()
    const mockBlog = createMockBlog({ id: 'new-linkedin-123', title: 'Test LinkedIn Post' })
    vi.mocked(blogApi.create).mockResolvedValue(mockBlog)
    
    render(<NewBlog />)
    
    // Switch to LinkedIn content type
    const linkedinButton = screen.getByText('💼 LinkedIn Post').closest('button')
    await user.click(linkedinButton!)
    
    // Fill form
    const titleInput = screen.getByLabelText('Post Title')
    await user.clear(titleInput)
    await user.type(titleInput, 'Test LinkedIn Post')
    
    // Submit form
    const submitButton = screen.getByText('Create LinkedIn Post')
    await user.click(submitButton)
    
    expect(blogApi.create).toHaveBeenCalledWith({
      title: 'Test LinkedIn Post',
      company_context: expect.any(String),
      content_type: 'linkedin'
    })
    
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/edit/new-linkedin-123')
    })
  })

  it('shows loading state during creation', async () => {
    const user = userEvent.setup()
    // Mock slow API response
    vi.mocked(blogApi.create).mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve(createMockBlog()), 1000))
    )
    
    render(<NewBlog />)
    
    const titleInput = screen.getByLabelText('Blog Title')
    await user.type(titleInput, 'Test Blog')
    
    const submitButton = screen.getByText('Create Blog')
    await user.click(submitButton)
    
    expect(screen.getByText('Creating...')).toBeInTheDocument()
    expect(submitButton).toBeDisabled()
  })

  it('disables submit when title is empty', async () => {
    const user = userEvent.setup()
    render(<NewBlog />)
    
    const titleInput = screen.getByLabelText('Blog Title')
    await user.clear(titleInput)
    
    const submitButton = screen.getByText('Create Blog')
    expect(submitButton).toBeDisabled()
    expect(submitButton).toHaveClass('disabled:opacity-50', 'disabled:cursor-not-allowed')
  })

  it('handles API error gracefully', async () => {
    const user = userEvent.setup()
    vi.mocked(blogApi.create).mockRejectedValue(new Error('API Error'))
    
    render(<NewBlog />)
    
    const titleInput = screen.getByLabelText('Blog Title')
    await user.type(titleInput, 'Test Blog')
    
    const submitButton = screen.getByText('Create Blog')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(showErrorNotification).toHaveBeenCalled()
    })
    
    // Should not navigate on error
    expect(mockNavigate).not.toHaveBeenCalled()
  })

  it('navigates back to dashboard when cancel is clicked', async () => {
    const user = userEvent.setup()
    render(<NewBlog />)
    
    const cancelButton = screen.getByText('Cancel')
    await user.click(cancelButton)
    
    expect(mockNavigate).toHaveBeenCalledWith('/')
  })

  it('shows correct instructions for selected content type', async () => {
    const user = userEvent.setup()
    render(<NewBlog />)
    
    // Blog instructions
    expect(screen.getByText(/Our Planner Agent creates a structured outline optimized for comprehensive blog content/)).toBeInTheDocument()
    expect(screen.getByText(/crafts content tailored for in-depth blog reading/)).toBeInTheDocument()
    
    // Switch to LinkedIn
    const linkedinButton = screen.getByText('💼 LinkedIn Post').closest('button')
    await user.click(linkedinButton!)
    
    // LinkedIn instructions
    expect(screen.getByText(/Our Planner Agent creates a structured outline optimized for LinkedIn engagement/)).toBeInTheDocument()
    expect(screen.getByText(/crafts content tailored for professional social media/)).toBeInTheDocument()
  })

  it('shows word count hints for content types', () => {
    render(<NewBlog />)
    
    expect(screen.getByText('Comprehensive, detailed articles (1500-2500 words)')).toBeInTheDocument()
    expect(screen.getByText('Professional, engaging posts (800-1200 words)')).toBeInTheDocument()
  })

  it('validates form submission without title', async () => {
    const user = userEvent.setup()
    render(<NewBlog />)
    
    // Clear title (it should be empty by default, but let's be explicit)
    const titleInput = screen.getByLabelText('Blog Title')
    await user.clear(titleInput)
    
    // Try to submit form with Enter key
    await user.type(titleInput, '{enter}')
    
    // Should not call API
    expect(blogApi.create).not.toHaveBeenCalled()
  })
})