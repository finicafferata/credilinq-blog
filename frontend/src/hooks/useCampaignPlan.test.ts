import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { useCampaignPlan } from './useCampaignPlan'

// Mock the API
vi.mock('../lib/api', () => ({
  campaignApi: {
    get: vi.fn(),
    schedule: vi.fn(),
    distribute: vi.fn(),
    updateTaskStatus: vi.fn(),
  }
}))

vi.mock('../lib/errors', () => ({
  showErrorNotification: vi.fn(),
  showSuccessNotification: vi.fn(),
}))

import { campaignApi } from '../lib/api'
import { showErrorNotification, showSuccessNotification } from '../lib/errors'

describe('useCampaignPlan', () => {
  const mockCampaignId = 'test-campaign-123'
  const mockCampaign = {
    id: mockCampaignId,
    name: 'Test Campaign',
    status: 'active',
    strategy: { target_audience: 'B2B professionals' },
    timeline: [],
    tasks: [
      {
        id: 'task-1',
        task_type: 'content_repurposing',
        status: 'pending',
        result: null,
        error: null
      }
    ],
    scheduled_posts: [],
    performance: {}
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('initializes with correct default state', () => {
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    expect(result.current.campaign).toBe(null)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBe(null)
  })

  it('fetches campaign successfully', async () => {
    vi.mocked(campaignApi.get).mockResolvedValue(mockCampaign)
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    act(() => {
      result.current.fetchCampaign()
    })
    
    expect(result.current.isLoading).toBe(true)
    
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false)
    })
    
    expect(result.current.campaign).toEqual(mockCampaign)
    expect(result.current.error).toBe(null)
    expect(campaignApi.get).toHaveBeenCalledWith(mockCampaignId)
  })

  it('handles fetch campaign error', async () => {
    const error = new Error('Failed to fetch campaign')
    vi.mocked(campaignApi.get).mockRejectedValue(error)
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    act(() => {
      result.current.fetchCampaign()
    })
    
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false)
    })
    
    expect(result.current.campaign).toBe(null)
    expect(result.current.error).toBe('Failed to fetch campaign')
    expect(showErrorNotification).toHaveBeenCalledWith('Failed to fetch campaign: Failed to fetch campaign')
  })

  it('schedules campaign successfully', async () => {
    vi.mocked(campaignApi.schedule).mockResolvedValue({ success: true })
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    await act(async () => {
      await result.current.schedule()
    })
    
    expect(campaignApi.schedule).toHaveBeenCalledWith(mockCampaignId)
    expect(showSuccessNotification).toHaveBeenCalledWith('Campaign scheduled successfully!')
  })

  it('handles schedule campaign error', async () => {
    const error = new Error('Failed to schedule')
    vi.mocked(campaignApi.schedule).mockRejectedValue(error)
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    await act(async () => {
      await result.current.schedule()
    })
    
    expect(showErrorNotification).toHaveBeenCalledWith('Failed to schedule campaign: Failed to schedule')
  })

  it('distributes campaign successfully', async () => {
    vi.mocked(campaignApi.distribute).mockResolvedValue({ success: true })
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    await act(async () => {
      await result.current.distribute()
    })
    
    expect(campaignApi.distribute).toHaveBeenCalledWith(mockCampaignId)
    expect(showSuccessNotification).toHaveBeenCalledWith('Campaign distributed successfully!')
  })

  it('handles distribute campaign error', async () => {
    const error = new Error('Failed to distribute')
    vi.mocked(campaignApi.distribute).mockRejectedValue(error)
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    await act(async () => {
      await result.current.distribute()
    })
    
    expect(showErrorNotification).toHaveBeenCalledWith('Failed to distribute campaign: Failed to distribute')
  })

  it('updates task status successfully', async () => {
    const updatedTask = { 
      id: 'task-1', 
      task_type: 'content_repurposing', 
      status: 'completed', 
      result: 'Task completed',
      error: null 
    }
    
    vi.mocked(campaignApi.updateTaskStatus).mockResolvedValue({ 
      success: true, 
      task: updatedTask 
    })
    vi.mocked(campaignApi.get).mockResolvedValue({
      ...mockCampaign,
      tasks: [updatedTask]
    })
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    await act(async () => {
      await result.current.updateTaskStatus('task-1', 'completed')
    })
    
    expect(campaignApi.updateTaskStatus).toHaveBeenCalledWith(mockCampaignId, 'task-1', 'completed')
    expect(showSuccessNotification).toHaveBeenCalledWith('Task status updated successfully!')
    expect(campaignApi.get).toHaveBeenCalledWith(mockCampaignId) // Should refetch
  })

  it('handles update task status error', async () => {
    const error = new Error('Failed to update task')
    vi.mocked(campaignApi.updateTaskStatus).mockRejectedValue(error)
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    await act(async () => {
      await result.current.updateTaskStatus('task-1', 'completed')
    })
    
    expect(showErrorNotification).toHaveBeenCalledWith('Failed to update task status: Failed to update task')
  })

  it('handles loading states correctly during multiple operations', async () => {
    vi.mocked(campaignApi.get).mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve(mockCampaign), 100))
    )
    vi.mocked(campaignApi.schedule).mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve({ success: true }), 50))
    )
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    // Start fetch
    act(() => {
      result.current.fetchCampaign()
    })
    expect(result.current.isLoading).toBe(true)
    
    // Start schedule while fetch is still running
    act(() => {
      result.current.schedule()
    })
    
    // Wait for schedule to complete first
    await waitFor(() => {
      expect(showSuccessNotification).toHaveBeenCalledWith('Campaign scheduled successfully!')
    }, { timeout: 100 })
    
    // Wait for fetch to complete
    await waitFor(() => {
      expect(result.current.campaign).toEqual(mockCampaign)
      expect(result.current.isLoading).toBe(false)
    }, { timeout: 200 })
  })

  it('clears error when new operation starts', async () => {
    // First operation fails
    vi.mocked(campaignApi.get).mockRejectedValueOnce(new Error('First error'))
    
    const { result } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    act(() => {
      result.current.fetchCampaign()
    })
    
    await waitFor(() => {
      expect(result.current.error).toBe('First error')
    })
    
    // Second operation should clear the error
    vi.mocked(campaignApi.get).mockResolvedValue(mockCampaign)
    
    act(() => {
      result.current.fetchCampaign()
    })
    
    expect(result.current.error).toBe(null)
    
    await waitFor(() => {
      expect(result.current.campaign).toEqual(mockCampaign)
    })
  })

  it('provides stable function references', () => {
    const { result, rerender } = renderHook(() => useCampaignPlan(mockCampaignId))
    
    const initialFunctions = {
      fetchCampaign: result.current.fetchCampaign,
      schedule: result.current.schedule,
      distribute: result.current.distribute,
      updateTaskStatus: result.current.updateTaskStatus,
    }
    
    rerender()
    
    expect(result.current.fetchCampaign).toBe(initialFunctions.fetchCampaign)
    expect(result.current.schedule).toBe(initialFunctions.schedule)
    expect(result.current.distribute).toBe(initialFunctions.distribute)
    expect(result.current.updateTaskStatus).toBe(initialFunctions.updateTaskStatus)
  })
})