import { useState, useEffect, useCallback, useRef } from 'react';
import {
  campaignOrchestrationAPI,
  agentOrchestrationAPI,
  systemMonitoringAPI,
  setupRealTimeListeners,
  cleanupRealTimeListeners,
  CampaignOrchestrationData,
  AgentData,
  SystemMetrics,
  WorkflowStep
} from '../services/orchestrationApi';

// Hook for managing campaign orchestration state
export function useCampaignOrchestration() {
  const [campaigns, setCampaigns] = useState<CampaignOrchestrationData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCampaign, setSelectedCampaign] = useState<string | null>(null);

  // Load campaigns
  const loadCampaigns = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await campaignOrchestrationAPI.getCampaigns();
      setCampaigns(data);
    } catch (err) {
      console.error('Failed to load campaigns:', err);
      setError('Failed to load campaigns');
    } finally {
      setLoading(false);
    }
  }, []);

  // Campaign control actions
  const startCampaign = useCallback(async (campaignId: string) => {
    try {
      const result = await campaignOrchestrationAPI.startCampaign(campaignId);
      if (result.success) {
        await loadCampaigns(); // Refresh data
      }
      return result;
    } catch (err) {
      console.error('Failed to start campaign:', err);
      throw err;
    }
  }, [loadCampaigns]);

  const pauseCampaign = useCallback(async (campaignId: string) => {
    try {
      const result = await campaignOrchestrationAPI.pauseCampaign(campaignId);
      if (result.success) {
        await loadCampaigns(); // Refresh data
      }
      return result;
    } catch (err) {
      console.error('Failed to pause campaign:', err);
      throw err;
    }
  }, [loadCampaigns]);

  const resumeCampaign = useCallback(async (campaignId: string) => {
    try {
      const result = await campaignOrchestrationAPI.resumeCampaign(campaignId);
      if (result.success) {
        await loadCampaigns(); // Refresh data
      }
      return result;
    } catch (err) {
      console.error('Failed to resume campaign:', err);
      throw err;
    }
  }, [loadCampaigns]);

  const stopCampaign = useCallback(async (campaignId: string) => {
    try {
      const result = await campaignOrchestrationAPI.stopCampaign(campaignId);
      if (result.success) {
        await loadCampaigns(); // Refresh data
      }
      return result;
    } catch (err) {
      console.error('Failed to stop campaign:', err);
      throw err;
    }
  }, [loadCampaigns]);

  // Real-time updates
  useEffect(() => {
    setupRealTimeListeners({
      onCampaignUpdate: (updatedCampaign) => {
        setCampaigns(prev => 
          prev.map(campaign => 
            campaign.id === updatedCampaign.id ? updatedCampaign : campaign
          )
        );
      },
      onError: (error) => {
        console.error('Real-time campaign error:', error);
      }
    });

    return () => {
      cleanupRealTimeListeners();
    };
  }, []);

  // Load initial data
  useEffect(() => {
    loadCampaigns();
  }, [loadCampaigns]);

  const activeCampaigns = campaigns.filter(c => c.status === 'running');
  const pausedCampaigns = campaigns.filter(c => c.status === 'paused');
  const completedCampaigns = campaigns.filter(c => c.status === 'completed');

  return {
    campaigns,
    activeCampaigns,
    pausedCampaigns,
    completedCampaigns,
    selectedCampaign,
    setSelectedCampaign,
    loading,
    error,
    actions: {
      reload: loadCampaigns,
      start: startCampaign,
      pause: pauseCampaign,
      resume: resumeCampaign,
      stop: stopCampaign
    }
  };
}

// Hook for managing agent pool state
export function useAgentPool() {
  const [agents, setAgents] = useState<AgentData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load agents
  const loadAgents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await agentOrchestrationAPI.getAgents();
      setAgents(data);
    } catch (err) {
      console.error('Failed to load agents:', err);
      setError('Failed to load agents');
    } finally {
      setLoading(false);
    }
  }, []);

  // Agent management actions
  const assignAgent = useCallback(async (campaignId: string, requirements: any) => {
    try {
      const result = await agentOrchestrationAPI.assignAgent({
        campaignId,
        agentRequirements: requirements,
        taskPriority: 'normal'
      });
      if (result.success) {
        await loadAgents(); // Refresh data
      }
      return result;
    } catch (err) {
      console.error('Failed to assign agent:', err);
      throw err;
    }
  }, [loadAgents]);

  const releaseAgent = useCallback(async (agentId: string, campaignId: string) => {
    try {
      const result = await agentOrchestrationAPI.releaseAgent(agentId, campaignId);
      if (result.success) {
        await loadAgents(); // Refresh data
      }
      return result;
    } catch (err) {
      console.error('Failed to release agent:', err);
      throw err;
    }
  }, [loadAgents]);

  // Real-time updates
  useEffect(() => {
    setupRealTimeListeners({
      onAgentUpdate: (updatedAgent) => {
        setAgents(prev => 
          prev.map(agent => 
            agent.id === updatedAgent.id ? updatedAgent : agent
          )
        );
      },
      onError: (error) => {
        console.error('Real-time agent error:', error);
      }
    });
  }, []);

  // Load initial data
  useEffect(() => {
    loadAgents();
  }, [loadAgents]);

  const activeAgents = agents.filter(a => a.status === 'active' || a.status === 'busy');
  const idleAgents = agents.filter(a => a.status === 'idle');
  const offlineAgents = agents.filter(a => a.status === 'offline');
  const busyAgents = agents.filter(a => a.status === 'busy');

  return {
    agents,
    activeAgents,
    idleAgents,
    offlineAgents,
    busyAgents,
    loading,
    error,
    actions: {
      reload: loadAgents,
      assign: assignAgent,
      release: releaseAgent
    }
  };
}

// Hook for system monitoring
export function useSystemMonitoring(refreshInterval: number = 5000) {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load system metrics
  const loadMetrics = useCallback(async () => {
    try {
      const [metricsData, healthData] = await Promise.all([
        systemMonitoringAPI.getSystemMetrics(),
        systemMonitoringAPI.getSystemHealth()
      ]);
      
      setMetrics(metricsData);
      setHealth(healthData);
      setError(null);
    } catch (err) {
      console.error('Failed to load system metrics:', err);
      setError('Failed to load system metrics');
    }
  }, []);

  // Setup real-time monitoring
  useEffect(() => {
    setupRealTimeListeners({
      onSystemMetrics: (updatedMetrics) => {
        setMetrics(updatedMetrics);
      },
      onConnection: (status) => {
        setIsConnected(status.status === 'connected');
      },
      onError: (error) => {
        console.error('Real-time system error:', error);
        setError('Real-time connection error');
      }
    });

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Periodic refresh fallback
  useEffect(() => {
    // Initial load
    loadMetrics().then(() => setLoading(false));

    // Setup interval for fallback refresh
    intervalRef.current = setInterval(loadMetrics, refreshInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [loadMetrics, refreshInterval]);

  return {
    metrics,
    health,
    loading,
    error,
    isConnected,
    actions: {
      reload: loadMetrics
    }
  };
}

// Hook for workflow monitoring
export function useWorkflowMonitoring(campaignId?: string) {
  const [workflows, setWorkflows] = useState<WorkflowStep[]>([]);
  const [currentStep, setCurrentStep] = useState<WorkflowStep | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load workflow data
  const loadWorkflow = useCallback(async (id: string) => {
    try {
      setLoading(true);
      setError(null);
      const workflowData = await campaignOrchestrationAPI.getCampaignWorkflow(id);
      setWorkflows(workflowData);
      
      // Find current step
      const runningStep = workflowData.find(step => step.status === 'running');
      setCurrentStep(runningStep || null);
    } catch (err) {
      console.error('Failed to load workflow:', err);
      setError('Failed to load workflow');
    } finally {
      setLoading(false);
    }
  }, []);

  // Real-time workflow updates
  useEffect(() => {
    setupRealTimeListeners({
      onWorkflowStep: (step) => {
        setWorkflows(prev => 
          prev.map(s => s.id === step.id ? step : s)
        );
        
        if (step.status === 'running') {
          setCurrentStep(step);
        }
      }
    });
  }, []);

  // Load workflow when campaign changes
  useEffect(() => {
    if (campaignId) {
      loadWorkflow(campaignId);
    }
  }, [campaignId, loadWorkflow]);

  const completedSteps = workflows.filter(s => s.status === 'completed');
  const failedSteps = workflows.filter(s => s.status === 'failed');
  const pendingSteps = workflows.filter(s => s.status === 'pending');
  
  const progress = workflows.length > 0 ? 
    (completedSteps.length / workflows.length) * 100 : 0;

  return {
    workflows,
    currentStep,
    completedSteps,
    failedSteps,
    pendingSteps,
    progress,
    loading,
    error,
    actions: {
      reload: () => campaignId && loadWorkflow(campaignId)
    }
  };
}

// Hook for performance analytics
export function usePerformanceAnalytics(timeRange: string = '24h') {
  const [analytics, setAnalytics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAnalytics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await systemMonitoringAPI.getPerformanceAnalytics(timeRange);
      setAnalytics(data);
    } catch (err) {
      console.error('Failed to load analytics:', err);
      setError('Failed to load analytics');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  return {
    analytics,
    loading,
    error,
    actions: {
      reload: loadAnalytics
    }
  };
}

// Combined hook for dashboard data
export function useOrchestrationDashboard() {
  const campaigns = useCampaignOrchestration();
  const agents = useAgentPool();
  const system = useSystemMonitoring();
  const performance = usePerformanceAnalytics();

  const isLoading = campaigns.loading || agents.loading || system.loading;
  const hasError = campaigns.error || agents.error || system.error;

  const reloadAll = useCallback(async () => {
    await Promise.all([
      campaigns.actions.reload(),
      agents.actions.reload(),
      system.actions.reload(),
      performance.actions.reload()
    ]);
  }, [campaigns.actions, agents.actions, system.actions, performance.actions]);

  return {
    campaigns,
    agents,
    system,
    performance,
    isLoading,
    hasError,
    actions: {
      reloadAll
    }
  };
}