import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import type { BlogSummary, BlogDetail } from '../../lib/api'
import type { CampaignResponse } from '../../types/campaign'

// Mock data
const mockBlogs: BlogSummary[] = [
  {
    id: 'blog-1',
    title: 'The Future of Embedded Finance',
    status: 'published',
    created_at: '2024-01-15T10:30:00Z',
  },
  {
    id: 'blog-2', 
    title: 'Digital Transformation in Fintech',
    status: 'draft',
    created_at: '2024-01-14T09:15:00Z',
  },
]

const mockBlogDetail: BlogDetail = {
  id: 'blog-1',
  title: 'The Future of Embedded Finance',
  status: 'published',
  created_at: '2024-01-15T10:30:00Z',
  content_markdown: '# The Future of Embedded Finance\n\nEmbedded finance is revolutionizing...',
  initial_prompt: {
    title: 'The Future of Embedded Finance',
    company_context: 'Credilinq.ai fintech expertise',
    content_type: 'blog'
  }
}

const mockCampaign: CampaignResponse = {
  id: 'campaign-1',
  blog_id: 'blog-1',
  created_at: '2024-01-15T11:00:00Z',
  tasks: [
    {
      id: 'task-1',
      task_type: 'repurpose',
      target_format: 'linkedin_post',
      target_asset: undefined,
      status: 'pending',
      result: null,
      image_url: null,
      error: null,
      created_at: '2024-01-15T11:00:00Z',
      updated_at: '2024-01-15T11:00:00Z'
    },
    {
      id: 'task-2',
      task_type: 'create_image_prompt',
      target_format: undefined,
      target_asset: 'Blog Header',
      status: 'pending',
      result: null,
      image_url: null,
      error: null,
      created_at: '2024-01-15T11:00:00Z',
      updated_at: '2024-01-15T11:00:00Z'
    }
  ]
}

// Define handlers
export const handlers = [
  // Blog endpoints
  http.get('/api/blogs', () => {
    return HttpResponse.json(mockBlogs)
  }),

  http.get('/api/blogs/:id', ({ params }) => {
    const { id } = params
    if (id === 'blog-1') {
      return HttpResponse.json(mockBlogDetail)
    }
    return new HttpResponse(null, { status: 404 })
  }),

  http.post('/api/blogs', async ({ request }) => {
    const requestBody = await request.json() as any
    const newBlog: BlogSummary = {
      id: 'new-blog-123',
      title: requestBody.title,
      status: 'draft',
      created_at: new Date().toISOString(),
    }
    return HttpResponse.json(newBlog, { status: 201 })
  }),

  http.put('/api/blogs/:id', async ({ params, request }) => {
    const { id } = params
    const requestBody = await request.json() as any
    
    if (id === 'blog-1') {
      const updatedBlog: BlogDetail = {
        ...mockBlogDetail,
        content_markdown: requestBody.content_markdown || mockBlogDetail.content_markdown
      }
      return HttpResponse.json(updatedBlog)
    }
    return new HttpResponse(null, { status: 404 })
  }),

  http.delete('/api/blogs/:id', ({ params }) => {
    const { id } = params
    if (id === 'blog-1') {
      return HttpResponse.json({ message: 'Blog deleted successfully', id })
    }
    return new HttpResponse(null, { status: 404 })
  }),

  http.post('/api/blogs/:id/publish', ({ params }) => {
    const { id } = params
    if (id === 'blog-1') {
      return HttpResponse.json({ ...mockBlogDetail, status: 'published' })
    }
    return new HttpResponse(null, { status: 404 })
  }),

  // Campaign endpoints
  http.post('/api/campaigns', async ({ request }) => {
    const requestBody = await request.json() as any
    return HttpResponse.json({
      ...mockCampaign,
      blog_id: requestBody.blog_id
    })
  }),

  http.get('/api/campaigns/:blogId', ({ params }) => {
    const { blogId } = params
    if (blogId === 'blog-1') {
      return HttpResponse.json(mockCampaign)
    }
    return new HttpResponse(null, { status: 404 })
  }),

  http.post('/api/campaigns/tasks/execute', async ({ request }) => {
    const requestBody = await request.json() as any
    return HttpResponse.json({
      message: 'Task execution started',
      task_id: requestBody.task_id
    }, { status: 202 })
  }),

  http.put('/api/campaigns/tasks/:taskId', async ({ params, request }) => {
    const { taskId } = params
    const requestBody = await request.json() as any
    
    return HttpResponse.json({
      id: taskId,
      task_type: 'repurpose',
      target_format: 'linkedin_post',
      target_asset: undefined,
      status: requestBody.status,
      result: requestBody.content,
      image_url: null,
      error: null,
      created_at: '2024-01-15T11:00:00Z',
      updated_at: new Date().toISOString()
    })
  }),

  // Analytics endpoints
  http.get('/api/analytics/dashboard', () => {
    return HttpResponse.json({
      total_blogs: 10,
      published_blogs: 8,
      total_views: 1500,
      avg_engagement: 0.25,
      top_performing_blogs: mockBlogs.slice(0, 2)
    })
  }),

  http.get('/api/blogs/:id/analytics', ({ params }) => {
    const { id } = params
    return HttpResponse.json({
      blog_id: id,
      views: 100,
      unique_visitors: 75,
      engagement_rate: 0.25,
      social_shares: 15,
      message: 'Analytics data found'
    })
  }),

  // Error simulation handlers (for error testing)
  http.get('/api/error/500', () => {
    return new HttpResponse(null, { status: 500 })
  }),

  http.get('/api/error/404', () => {
    return new HttpResponse(null, { status: 404 })
  }),
]

// Setup server
export const server = setupServer(...handlers)