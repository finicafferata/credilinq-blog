import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CampaignDetails } from './CampaignDetails'
import { render } from '../test/utils'
import type { CampaignDetail } from '../lib/api'

describe('CampaignDetails', () => {
  const mockOnClose = vi.fn()
  
  const mockCampaign: CampaignDetail = {
    id: 'campaign-123',
    name: 'Test Campaign',
    status: 'active',
    strategy: {
      target_audience: 'B2B professionals',
      key_messages: ['Innovation', 'Growth'],
      distribution_channels: ['LinkedIn', 'Email']
    },
    timeline: [
      { date: '2025-01-01', task: 'Create content', status: 'completed' },
      { date: '2025-01-02', task: 'Schedule posts', status: 'pending' }
    ],
    tasks: [
      {
        id: 'task-1',
        task_type: 'content_repurposing',
        status: 'completed',
        result: 'LinkedIn post created',
        error: null
      },
      {
        id: 'task-2',
        task_type: 'image_generation',
        status: 'pending',
        result: null,
        error: null
      }
    ],
    scheduled_posts: [
      {
        id: 'post-1',
        platform: 'LinkedIn',
        content: 'Test post content',
        scheduled_at: '2025-01-05T10:00:00Z',
        status: 'scheduled'
      }
    ],
    performance: {
      views: 1000,
      clicks: 50,
      engagement_rate: 0.05
    }
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders campaign details correctly', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    expect(screen.getByText('Test Campaign')).toBeInTheDocument()
    expect(screen.getByText('Campaign ID: campaign-123')).toBeInTheDocument()
    expect(screen.getByText('active')).toBeInTheDocument()
  })

  it('displays strategy information when available', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    expect(screen.getByText('Strategy')).toBeInTheDocument()
    expect(screen.getByText('B2B professionals')).toBeInTheDocument()
    expect(screen.getByText('Innovation')).toBeInTheDocument()
    expect(screen.getByText('Growth')).toBeInTheDocument()
  })

  it('displays tasks section when tasks exist', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    expect(screen.getByText('Tasks')).toBeInTheDocument()
    expect(screen.getByText('content_repurposing')).toBeInTheDocument()
    expect(screen.getByText('image_generation')).toBeInTheDocument()
    expect(screen.getByText('LinkedIn post created')).toBeInTheDocument()
  })

  it('displays scheduled posts when available', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    expect(screen.getByText('Scheduled Posts')).toBeInTheDocument()
    expect(screen.getByText('LinkedIn')).toBeInTheDocument()
    expect(screen.getByText('Test post content')).toBeInTheDocument()
  })

  it('displays performance metrics', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    expect(screen.getByText('Performance')).toBeInTheDocument()
    expect(screen.getByText('1000')).toBeInTheDocument() // views
    expect(screen.getByText('50')).toBeInTheDocument() // clicks
    expect(screen.getByText('5.00%')).toBeInTheDocument() // engagement rate
  })

  it('handles close button click', async () => {
    const user = userEvent.setup()
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    const closeButton = screen.getByRole('button', { name: /close/i })
    await user.click(closeButton)
    
    expect(mockOnClose).toHaveBeenCalledTimes(1)
  })

  it('handles modal overlay click', async () => {
    const user = userEvent.setup()
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    const overlay = screen.getByRole('dialog').parentElement
    if (overlay) {
      await user.click(overlay)
      expect(mockOnClose).toHaveBeenCalledTimes(1)
    }
  })

  it('shows task status badges correctly', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    const completedBadge = screen.getByText('completed')
    const pendingBadge = screen.getByText('pending')
    
    expect(completedBadge).toHaveClass('bg-green-100', 'text-green-800')
    expect(pendingBadge).toHaveClass('bg-gray-100', 'text-gray-800')
  })

  it('handles campaign with no tasks', () => {
    const campaignWithoutTasks = { ...mockCampaign, tasks: [] }
    render(<CampaignDetails campaign={campaignWithoutTasks} onClose={mockOnClose} />)
    
    expect(screen.queryByText('Tasks')).not.toBeInTheDocument()
  })

  it('handles campaign with no scheduled posts', () => {
    const campaignWithoutPosts = { ...mockCampaign, scheduled_posts: [] }
    render(<CampaignDetails campaign={campaignWithoutPosts} onClose={mockOnClose} />)
    
    expect(screen.queryByText('Scheduled Posts')).not.toBeInTheDocument()
  })

  it('handles campaign with no strategy', () => {
    const campaignWithoutStrategy = { ...mockCampaign, strategy: {} }
    render(<CampaignDetails campaign={campaignWithoutStrategy} onClose={mockOnClose} />)
    
    expect(screen.queryByText('B2B professionals')).not.toBeInTheDocument()
  })

  it('formats dates correctly', () => {
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    // Should format the scheduled date
    expect(screen.getByText(/Jan 5, 2025/)).toBeInTheDocument()
  })

  it('displays task errors when present', () => {
    const campaignWithErrors = {
      ...mockCampaign,
      tasks: [
        {
          id: 'task-error',
          task_type: 'failed_task',
          status: 'error' as const,
          result: null,
          error: 'Task failed due to API error'
        }
      ]
    }
    
    render(<CampaignDetails campaign={campaignWithErrors} onClose={mockOnClose} />)
    
    expect(screen.getByText('Task failed due to API error')).toBeInTheDocument()
  })

  it('shows empty states gracefully', () => {
    const emptyCampaign: CampaignDetail = {
      id: 'empty-campaign',
      name: 'Empty Campaign',
      status: 'draft',
      strategy: {},
      timeline: [],
      tasks: [],
      scheduled_posts: [],
      performance: {}
    }
    
    render(<CampaignDetails campaign={emptyCampaign} onClose={mockOnClose} />)
    
    expect(screen.getByText('Empty Campaign')).toBeInTheDocument()
    expect(screen.queryByText('Tasks')).not.toBeInTheDocument()
    expect(screen.queryByText('Scheduled Posts')).not.toBeInTheDocument()
  })

  it('handles keyboard navigation', async () => {
    const user = userEvent.setup()
    render(<CampaignDetails campaign={mockCampaign} onClose={mockOnClose} />)
    
    // Should be able to navigate with keyboard
    await user.keyboard('{Escape}')
    expect(mockOnClose).toHaveBeenCalledTimes(1)
  })
})