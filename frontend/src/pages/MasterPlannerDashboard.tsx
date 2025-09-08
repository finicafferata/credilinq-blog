import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity,
  Users,
  Clock,
  CheckCircle,
  AlertCircle,
  Play,
  Pause,
  Square,
  RefreshCw,
  Settings,
  Eye,
  ArrowRight,
  TrendingUp,
  Zap,
  Database,
  Network,
  BarChart3,
  GitBranch,
  Timer,
  Target,
  Layers,
  PlayCircle,
  PauseCircle,
  StopCircle,
  AlertTriangle,
  Info,
  ChevronRight,
  ChevronDown,
  Calendar,
  FileText,
  MessageSquare,
  Workflow
} from 'lucide-react';
import WorkflowVisualizer from '../components/WorkflowVisualizer';

// Types for Master Planner Dashboard
interface AgentKnowledge {
  type: string;
  dependencies: string[];
  execution_time_estimate: number;
  priority: number;
  description: string;
}

interface AgentSequenceStep {
  agent_name: string;
  agent_type: string;
  execution_order: number;
  dependencies: string[];
  parallel_group_id?: number;
  status?: 'waiting' | 'running' | 'completed' | 'failed';
}

interface ExecutionPlan {
  success: boolean;
  plan_id: string;
  workflow_execution_id: string;
  strategy: string;
  total_agents: number;
  estimated_duration_minutes: number;
  agent_sequence: AgentSequenceStep[];
  created_at: string;
  message: string;
}

interface WorkflowStatus {
  workflow_execution_id: string;
  status: string;
  progress_percentage: number;
  current_step: number;
  total_steps: number;
  agents_status: {
    waiting: string[];
    running: string[];
    completed: string[];
    failed: string[];
  };
  start_time?: string;
  estimated_completion_time?: string;
  actual_completion_time?: string;
  execution_metadata: Record<string, any>;
  intermediate_outputs: Record<string, any>;
  last_heartbeat: string;
}

interface ActiveWorkflow {
  workflow_execution_id: string;
  campaign_id?: string;
  status: string;
  progress_percentage: number;
  total_agents: number;
  completed_agents: number;
  start_time?: string;
  estimated_completion_time?: string;
  last_heartbeat?: string;
}

const MasterPlannerDashboard: React.FC = () => {
  // State management for workflow execution
  const [executionForm, setExecutionForm] = useState({
    campaign_id: '',
    blog_title: '',
    company_context: '',
    content_type: 'blog',
    execution_strategy: 'adaptive',
    required_agents: ['planner', 'researcher', 'writer', 'editor']
  });
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResults, setExecutionResults] = useState<any>(null);
  const [activeExecutions, setActiveExecutions] = useState<string[]>([]);
  const [agentResults, setAgentResults] = useState<Record<string, any>>({});

  // State management
  const [agentKnowledgeBase, setAgentKnowledgeBase] = useState<Record<string, AgentKnowledge>>({});
  const [activeWorkflows, setActiveWorkflows] = useState<ActiveWorkflow[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // WebSocket for real-time updates
  const wsRef = useRef<WebSocket | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Create execution plan state
  const [createPlanForm, setCreatePlanForm] = useState({
    campaign_id: '',
    strategy: 'adaptive',
    required_agents: ['planner', 'researcher', 'writer', 'editor', 'seo']
  });
  const [showCreateForm, setShowCreateForm] = useState(false);

  const API_BASE = 'http://localhost:8001/api/v2/workflow-orchestration';

  // Load initial data
  useEffect(() => {
    loadAgentKnowledgeBase();
    loadActiveWorkflows();
  }, []);

  // Auto-refresh active workflows
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      loadActiveWorkflows();
      if (selectedWorkflow) {
        loadWorkflowStatus(selectedWorkflow);
      }
    }, 3000);
    
    return () => clearInterval(interval);
  }, [autoRefresh, selectedWorkflow]);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback((workflowId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const wsUrl = `ws://localhost:8001/api/v2/workflow-orchestration/workflows/${workflowId}/stream`;
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setWsConnected(true);
        console.log(`WebSocket connected for workflow: ${workflowId}`);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status_update') {
            setWorkflowStatus(data.data);
          } else if (data.type === 'workflow_finished') {
            setWsConnected(false);
            loadActiveWorkflows(); // Refresh the list
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };
      
      wsRef.current.onclose = () => {
        setWsConnected(false);
        console.log('WebSocket disconnected');
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }, []);

  // API calls
  const loadAgentKnowledgeBase = async () => {
    try {
      const response = await fetch(`${API_BASE}/agents/knowledge-base`);
      const data = await response.json();
      if (data.success) {
        setAgentKnowledgeBase(data.agents);
      }
    } catch (error) {
      console.error('Failed to load agent knowledge base:', error);
    }
  };

  const loadActiveWorkflows = async () => {
    try {
      // Fetch both Master Planner workflows and campaign workflows
      const [masterPlannerResponse, campaignResponse] = await Promise.all([
        fetch(`${API_BASE}/workflows/active`),
        fetch(`${API_BASE}/campaigns/workflow/active`)
      ]);
      
      const masterPlannerWorkflows = masterPlannerResponse.ok ? await masterPlannerResponse.json() : [];
      const campaignWorkflows = campaignResponse.ok ? await campaignResponse.json() : [];
      
      // Combine both types of workflows
      const allWorkflows = [
        ...masterPlannerWorkflows,
        ...campaignWorkflows.map((workflow: any) => ({
          ...workflow,
          workflow_type: 'campaign',
          is_campaign: true
        }))
      ];
      
      setActiveWorkflows(allWorkflows);
      console.log('✅ Loaded active workflows:', {
        masterPlanner: masterPlannerWorkflows.length,
        campaigns: campaignWorkflows.length,
        total: allWorkflows.length
      });
    } catch (error) {
      console.error('Failed to load active workflows:', error);
    }
  };

  const loadWorkflowStatus = async (workflowId: string) => {
    try {
      // Check if this is a campaign workflow
      const isCampaignWorkflow = workflowId.startsWith('campaign_');
      let response;
      
      if (isCampaignWorkflow) {
        // Extract campaign ID from workflow ID
        const campaignId = workflowId.replace('campaign_', '');
        response = await fetch(`${API_BASE}/campaigns/${campaignId}/workflow-status`);
      } else {
        // Use Master Planner workflow status endpoint
        response = await fetch(`${API_BASE}/workflows/${workflowId}/status`);
      }
      
      if (response.ok) {
        const data = await response.json();
        setWorkflowStatus(data);
        console.log('✅ Loaded workflow status:', { workflowId, isCampaignWorkflow, status: data.status });
      }
    } catch (error) {
      console.error('Failed to load workflow status:', error);
    }
  };

  // Real workflow execution functions
  const executeWorkflow = async () => {
    if (!executionForm.campaign_id || !executionForm.blog_title) {
      setError('Campaign ID and Blog Title are required for execution');
      return;
    }

    setIsExecuting(true);
    setError(null);
    setExecutionResults(null);

    try {
      const response = await fetch(`${API_BASE}/workflows/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          campaign_id: executionForm.campaign_id,
          execution_strategy: executionForm.execution_strategy,
          required_agents: executionForm.required_agents,
          blog_title: executionForm.blog_title,
          company_context: executionForm.company_context,
          content_type: executionForm.content_type,
          context_data: {
            timestamp: new Date().toISOString()
          }
        })
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setExecutionResults(data);
        setActiveExecutions(prev => [...prev, data.workflow_execution_id]);
        
        // Start monitoring this workflow
        startRealtimeMonitoring(data.workflow_execution_id);
        
        console.log('✅ Workflow execution started:', data);
      } else {
        throw new Error(data.detail || 'Failed to execute workflow');
      }
    } catch (error: any) {
      setError(error.message);
      console.error('Workflow execution failed:', error);
    } finally {
      setIsExecuting(false);
    }
  };

  const startRealtimeMonitoring = (workflowId: string) => {
    const wsUrl = `ws://localhost:8000/workflow-orchestration/workflows/${workflowId}/stream`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`🔗 Connected to workflow stream: ${workflowId}`);
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      console.log('📨 Workflow update:', message);

      if (message.type === 'status_update') {
        // Update workflow status in real-time
        setWorkflowStatus(message.data);
        
        // If this is the selected workflow, update detailed view
        if (selectedWorkflow === workflowId) {
          loadWorkflowStatus(workflowId);
        }
      } else if (message.type === 'workflow_finished') {
        console.log('🏁 Workflow completed:', message);
        setActiveExecutions(prev => prev.filter(id => id !== workflowId));
        
        // Load final results
        loadAgentResults(workflowId);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log(`🔌 Disconnected from workflow stream: ${workflowId}`);
    };

    return ws;
  };

  const loadAgentResults = async (workflowId: string) => {
    try {
      const agentNames = executionForm.required_agents;
      const results: Record<string, any> = {};

      for (const agentName of agentNames) {
        try {
          const response = await fetch(`${API_BASE}/workflows/${workflowId}/agents/${agentName}/results`);
          if (response.ok) {
            const data = await response.json();
            if (data.success) {
              results[agentName] = data.results;
            }
          }
        } catch (error) {
          console.error(`Failed to load results for agent ${agentName}:`, error);
        }
      }

      setAgentResults(prev => ({
        ...prev,
        [workflowId]: results
      }));
    } catch (error) {
      console.error('Failed to load agent results:', error);
    }
  };

  const cancelWorkflow = async (workflowId: string) => {
    try {
      const response = await fetch(`${API_BASE}/workflows/${workflowId}/cancel`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        console.log('❌ Workflow cancelled:', data);
        setActiveExecutions(prev => prev.filter(id => id !== workflowId));
        loadActiveWorkflows(); // Refresh the list
      }
    } catch (error) {
      console.error('Failed to cancel workflow:', error);
    }
  };

  // Convert workflow status to visualizer format
  const convertToVisualizerAgents = (workflowStatus: any, requiredAgents?: string[]) => {
    if (!workflowStatus) return [];
    
    // Handle campaign workflow format
    if (workflowStatus.agents_status) {
      const allAgents = [
        ...workflowStatus.agents_status.waiting.map((agent: any) => ({ ...agent, status: 'waiting' })),
        ...workflowStatus.agents_status.running.map((agent: any) => ({ ...agent, status: 'running' })),
        ...workflowStatus.agents_status.completed.map((agent: any) => ({ ...agent, status: 'completed' })),
        ...workflowStatus.agents_status.failed.map((agent: any) => ({ ...agent, status: 'failed' }))
      ];
      
      return allAgents.map((agent, index) => ({
        id: agent.agent_type || agent.task_type || `agent_${index}`,
        name: agent.agent_type || agent.task_type || `Agent ${index + 1}`,
        type: agent.agent_type || agent.task_type || 'unknown',
        status: agent.status,
        executionTime: 0,
        estimatedTime: getEstimatedTime(agent.agent_type || agent.task_type),
        dependencies: getAgentDependencies(agent.agent_type || agent.task_type, index),
        parallelGroupId: getParallelGroup(agent.agent_type || agent.task_type),
        output: agent.output_preview ? { preview: agent.output_preview } : null,
        startTime: agent.start_time,
        updatedTime: agent.updated_time
      }));
    }
    
    // Handle Master Planner workflow format
    if (!workflowStatus.agent_results || !requiredAgents) return [];
    
    return requiredAgents.map((agentName, index) => {
      const agentResult = workflowStatus.agent_results[agentName];
      const isCompleted = workflowStatus.completed_agents?.includes(agentName);
      const isFailed = workflowStatus.failed_agents?.includes(agentName);
      const isRunning = workflowStatus.current_agents?.includes(agentName);
      
      let status = 'waiting';
      if (isCompleted) status = 'completed';
      else if (isFailed) status = 'failed';  
      else if (isRunning) status = 'running';
      
      return {
        id: agentName,
        name: agentName,
        type: agentName,
        status,
        executionTime: agentResult?.execution_time || 0,
        estimatedTime: getEstimatedTime(agentName),
        dependencies: getAgentDependencies(agentName, index),
        parallelGroupId: getParallelGroup(agentName),
        output: agentResult?.output_keys?.length > 0 ? agentResult : null
      };
    });
  };

  const getEstimatedTime = (agentName: string): number => {
    const estimates: Record<string, number> = {
      'planner': 3,
      'researcher': 5, 
      'writer': 8,
      'editor': 4,
      'seo': 2,
      'image': 3,
      'social_media': 2,
      'campaign_manager': 4,
      'content_repurposer': 3
    };
    return estimates[agentName] || 5;
  };

  const getAgentDependencies = (agentName: string, index: number): string[] => {
    const dependencies: Record<string, string[]> = {
      'planner': [],
      'researcher': ['planner'],
      'writer': ['planner', 'researcher'],
      'editor': ['writer'],
      'seo': ['editor'],
      'image': ['writer'],
      'social_media': ['editor'],
      'campaign_manager': ['seo'],
      'content_repurposer': ['editor']
    };
    return dependencies[agentName] || (index > 0 ? [executionForm.required_agents[index - 1]] : []);
  };

  const getParallelGroup = (agentName: string): number | undefined => {
    // Define which agents can run in parallel
    if (['image', 'social_media'].includes(agentName)) return 1;
    if (['seo', 'content_repurposer'].includes(agentName)) return 2;
    return undefined;
  };

  const createExecutionPlan = async () => {
    if (!createPlanForm.campaign_id) {
      setError('Campaign ID is required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/execution-plans`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(createPlanForm)
      });

      if (response.ok) {
        const data = await response.json();
        setShowCreateForm(false);
        setCreatePlanForm({
          campaign_id: '',
          strategy: 'adaptive',
          required_agents: ['planner', 'researcher', 'writer', 'editor', 'seo']
        });
        loadActiveWorkflows();
        
        // Auto-select the new workflow
        setTimeout(() => {
          setSelectedWorkflow(data.workflow_execution_id);
          loadWorkflowStatus(data.workflow_execution_id);
          connectWebSocket(data.workflow_execution_id);
        }, 1000);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to create execution plan');
      }
    } catch (error) {
      setError(`Failed to create execution plan: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle workflow selection
  const handleWorkflowSelect = (workflowId: string) => {
    setSelectedWorkflow(workflowId);
    loadWorkflowStatus(workflowId);
    connectWebSocket(workflowId);
  };

  // Utility functions
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'running': return 'text-blue-600 bg-blue-100';
      case 'waiting': return 'text-yellow-600 bg-yellow-100';
      case 'failed': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'running': return <PlayCircle className="w-4 h-4" />;
      case 'waiting': return <Clock className="w-4 h-4" />;
      case 'failed': return <AlertCircle className="w-4 h-4" />;
      default: return <PauseCircle className="w-4 h-4" />;
    }
  };

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="bg-white shadow-sm rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Workflow className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Master Planner Dashboard</h1>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Database className="w-4 h-4" />
                <span>{Object.keys(agentKnowledgeBase).length} agents in knowledge base</span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                <span className="text-sm text-gray-500">
                  {wsConnected ? 'Real-time connected' : 'Polling mode'}
                </span>
              </div>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`flex items-center space-x-2 px-3 py-1 rounded ${
                  autoRefresh ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                }`}
              >
                <RefreshCw className="w-4 h-4" />
                <span>Auto-refresh</span>
              </button>
              <button
                onClick={() => setShowCreateForm(true)}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>Create Workflow</span>
              </button>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <span className="text-red-700">{error}</span>
              <button onClick={() => setError(null)} className="ml-auto text-red-600 hover:text-red-800">
                ×
              </button>
            </div>
          </div>
        )}

        {/* Real Workflow Execution Section */}
        <div className="bg-white shadow-sm rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
              <PlayCircle className="w-6 h-6 text-green-600" />
              <span>Execute Real Workflow</span>
            </h2>
            {activeExecutions.length > 0 && (
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-green-700 font-medium">
                    {activeExecutions.length} active execution{activeExecutions.length > 1 ? 's' : ''}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Execution Form */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Campaign ID *
                </label>
                <input
                  type="text"
                  value={executionForm.campaign_id}
                  onChange={(e) => setExecutionForm(prev => ({ ...prev, campaign_id: e.target.value }))}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  placeholder="e.g., campaign_2024_fintech_01"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Blog Title *
                </label>
                <input
                  type="text"
                  value={executionForm.blog_title}
                  onChange={(e) => setExecutionForm(prev => ({ ...prev, blog_title: e.target.value }))}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  placeholder="e.g., The Future of Digital Banking and AI"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Company Context
                </label>
                <textarea
                  value={executionForm.company_context}
                  onChange={(e) => setExecutionForm(prev => ({ ...prev, company_context: e.target.value }))}
                  rows={3}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  placeholder="CrediLinq is a B2B fintech platform specializing in alternative lending and credit solutions for small businesses..."
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Content Type
                  </label>
                  <select
                    value={executionForm.content_type}
                    onChange={(e) => setExecutionForm(prev => ({ ...prev, content_type: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  >
                    <option value="blog">Blog Post</option>
                    <option value="linkedin">LinkedIn Post</option>
                    <option value="article">Article</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Execution Strategy
                  </label>
                  <select
                    value={executionForm.execution_strategy}
                    onChange={(e) => setExecutionForm(prev => ({ ...prev, execution_strategy: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  >
                    <option value="adaptive">Adaptive (Recommended)</option>
                    <option value="parallel">Maximum Parallel</option>
                    <option value="sequential">Sequential</option>
                  </select>
                </div>
              </div>

              <button
                onClick={executeWorkflow}
                disabled={isExecuting || !executionForm.campaign_id || !executionForm.blog_title}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg flex items-center justify-center space-x-2 transition-all duration-200 font-medium"
              >
                {isExecuting ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Executing Workflow...</span>
                  </>
                ) : (
                  <>
                    <PlayCircle className="w-5 h-5" />
                    <span>Execute Real Workflow with {executionForm.required_agents.length} Agents</span>
                  </>
                )}
              </button>
            </div>

            {/* Results/Status Display */}
            <div className="space-y-4">
              {executionResults && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center space-x-2 mb-3">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="text-lg font-medium text-green-800">Workflow Execution Started!</span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-green-700">Workflow ID:</span>
                      <span className="font-mono text-green-800">{executionResults.workflow_execution_id.split('-')[0]}...</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-700">Total Agents:</span>
                      <span className="font-medium text-green-800">{executionResults.total_agents}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-700">Strategy:</span>
                      <span className="font-medium text-green-800 capitalize">{executionResults.status}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-700">Est. Duration:</span>
                      <span className="font-medium text-green-800">{executionResults.estimated_duration_minutes}min</span>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-green-200">
                    <p className="text-xs text-green-600">
                      🔄 Real-time status updates will appear below. Agents will execute: {executionForm.required_agents.join(' → ')}
                    </p>
                  </div>
                </div>
              )}

              {/* Agent Results Display */}
              {Object.keys(agentResults).length > 0 && (
                <div className="space-y-3">
                  <h4 className="font-medium text-gray-900">Agent Results</h4>
                  {Object.entries(agentResults).map(([workflowId, results]) => (
                    <div key={workflowId} className="border border-gray-200 rounded-lg p-3">
                      <h5 className="text-sm font-medium text-gray-800 mb-2">Workflow: {workflowId.split('-')[0]}...</h5>
                      <div className="space-y-2">
                        {Object.entries(results).map(([agentName, result]: [string, any]) => (
                          <div key={agentName} className="bg-gray-50 p-2 rounded text-xs">
                            <div className="font-medium text-gray-800">{agentName}</div>
                            <div className="text-gray-600 mt-1">
                              {result.content ? `Generated ${result.word_count || 'N/A'} words` : 'Processing...'}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Live Execution Status */}
              {workflowStatus && activeExecutions.includes(workflowStatus.workflow_execution_id) && (
                <div className="border border-blue-200 rounded-lg p-4 bg-blue-50">
                  <div className="flex items-center space-x-2 mb-3">
                    <Activity className="w-5 h-5 text-blue-600 animate-pulse" />
                    <span className="font-medium text-blue-800">Live Execution Status</span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-blue-700">Progress:</span>
                      <span className="font-medium text-blue-800">{workflowStatus.progress_percentage.toFixed(1)}%</span>
                    </div>
                    
                    {/* Progress Bar */}
                    <div className="w-full bg-blue-100 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${workflowStatus.progress_percentage}%` }}
                      ></div>
                    </div>

                    {/* Agent Status */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {workflowStatus.agents_status.running.length > 0 && (
                        <div className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                          ⚡ Running: {workflowStatus.agents_status.running.join(', ')}
                        </div>
                      )}
                      {workflowStatus.agents_status.completed.length > 0 && (
                        <div className="bg-green-100 text-green-800 px-2 py-1 rounded">
                          ✅ Done: {workflowStatus.agents_status.completed.length}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Enhanced Visual Workflow Representation */}
        {(executionResults || (workflowStatus && activeExecutions.includes(workflowStatus.workflow_execution_id))) && (
          <div className="mb-6">
            <WorkflowVisualizer
              workflowId={executionResults?.workflow_execution_id || workflowStatus?.workflow_execution_id}
              agents={convertToVisualizerAgents(workflowStatus, workflowStatus?.is_campaign ? undefined : executionForm.required_agents)}
              status={workflowStatus?.status || 'planning'}
              progressPercentage={workflowStatus?.progress_percentage || 0}
              onAgentClick={(agentId) => {
                console.log('Agent clicked:', agentId);
                // TODO: Show agent details modal
              }}
              className="animate-fadeIn"
            />
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Active Workflows Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white shadow-sm rounded-lg">
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-gray-900">Active Workflows</h2>
                  <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded">
                    {activeWorkflows.length}
                  </span>
                </div>
              </div>
              <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                {activeWorkflows.length === 0 ? (
                  <div className="p-6 text-center text-gray-500">
                    <Workflow className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                    <p>No active workflows</p>
                    <p className="text-sm">Create a new workflow to get started</p>
                  </div>
                ) : (
                  activeWorkflows.map((workflow) => (
                    <div
                      key={workflow.workflow_execution_id}
                      onClick={() => handleWorkflowSelect(workflow.workflow_execution_id)}
                      className={`p-4 cursor-pointer transition-colors ${
                        selectedWorkflow === workflow.workflow_execution_id
                          ? 'bg-blue-50 border-r-4 border-blue-500'
                          : 'hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className={`flex items-center space-x-2 px-2 py-1 rounded text-xs font-medium ${getStatusColor(workflow.status)}`}>
                          {getStatusIcon(workflow.status)}
                          <span>{workflow.status.toUpperCase()}</span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {workflow.progress_percentage.toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${workflow.progress_percentage}%` }}
                        ></div>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {workflow.is_campaign ? (
                            <>
                              <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded mr-2">
                                Campaign
                              </span>
                              {workflow.campaign_name || `Campaign ${workflow.campaign_id}`}
                            </>
                          ) : (
                            `Workflow ${workflow.workflow_execution_id.slice(0, 8)}`
                          )}
                        </p>
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>{workflow.completed_agents}/{workflow.total_agents} agents</span>
                          {workflow.start_time && (
                            <span>{new Date(workflow.start_time).toLocaleTimeString()}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Workflow Details */}
            {selectedWorkflow && workflowStatus ? (
              <>
                {/* Status Overview */}
                <div className="bg-white shadow-sm rounded-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Workflow Status</h3>
                    <div className={`flex items-center space-x-2 px-3 py-1 rounded-lg text-sm font-medium ${getStatusColor(workflowStatus.status)}`}>
                      {getStatusIcon(workflowStatus.status)}
                      <span>{workflowStatus.status.toUpperCase()}</span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{workflowStatus.progress_percentage.toFixed(0)}%</div>
                      <div className="text-sm text-gray-500">Progress</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{workflowStatus.agents_status.completed.length}</div>
                      <div className="text-sm text-gray-500">Completed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-600">{workflowStatus.agents_status.running.length}</div>
                      <div className="text-sm text-gray-500">Running</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-gray-600">{workflowStatus.agents_status.waiting.length}</div>
                      <div className="text-sm text-gray-500">Waiting</div>
                    </div>
                  </div>
                  
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-1000"
                      style={{ width: `${workflowStatus.progress_percentage}%` }}
                    ></div>
                  </div>
                </div>

                {/* Agent Status Grid */}
                <div className="bg-white shadow-sm rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Status</h3>
                  <div className="space-y-3">
                    {['waiting', 'running', 'completed', 'failed'].map((statusType) => {
                      const agents = workflowStatus.agents_status[statusType as keyof typeof workflowStatus.agents_status] || [];
                      return (
                        <div key={statusType} className="flex items-center space-x-3">
                          <div className={`w-16 text-xs font-medium ${getStatusColor(statusType === 'waiting' ? 'waiting' : statusType)} px-2 py-1 rounded text-center`}>
                            {statusType.toUpperCase()}
                          </div>
                          <div className="flex-1">
                            {agents.length === 0 ? (
                              <span className="text-sm text-gray-400">None</span>
                            ) : (
                              <div className="flex flex-wrap gap-2">
                                {agents.map((agent) => (
                                  <span key={agent} className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-sm">
                                    {agent}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Timing Information */}
                {(workflowStatus.start_time || workflowStatus.estimated_completion_time) && (
                  <div className="bg-white shadow-sm rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Timing</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {workflowStatus.start_time && (
                        <div>
                          <div className="text-sm text-gray-500">Started</div>
                          <div className="font-medium">{new Date(workflowStatus.start_time).toLocaleString()}</div>
                        </div>
                      )}
                      {workflowStatus.estimated_completion_time && (
                        <div>
                          <div className="text-sm text-gray-500">Estimated Completion</div>
                          <div className="font-medium">{new Date(workflowStatus.estimated_completion_time).toLocaleString()}</div>
                        </div>
                      )}
                      <div>
                        <div className="text-sm text-gray-500">Last Update</div>
                        <div className="font-medium">{new Date(workflowStatus.last_heartbeat).toLocaleString()}</div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              /* Agent Knowledge Base Display */
              <div className="bg-white shadow-sm rounded-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Knowledge Base</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(agentKnowledgeBase).map(([name, info]) => (
                    <div key={name} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-900">{name}</h4>
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                          Priority {info.priority}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{info.description}</p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{formatDuration(Math.floor(info.execution_time_estimate / 60))}</span>
                        <span>{info.dependencies.length} dependencies</span>
                      </div>
                      {info.dependencies.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {info.dependencies.map((dep) => (
                            <span key={dep} className="bg-blue-100 text-blue-700 text-xs px-1.5 py-0.5 rounded">
                              {dep}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Create Workflow Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Create New Workflow</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Campaign ID (UUID format)
                </label>
                <input
                  type="text"
                  value={createPlanForm.campaign_id}
                  onChange={(e) => setCreatePlanForm({...createPlanForm, campaign_id: e.target.value})}
                  placeholder="550e8400-e29b-41d4-a716-446655440001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Strategy</label>
                <select
                  value={createPlanForm.strategy}
                  onChange={(e) => setCreatePlanForm({...createPlanForm, strategy: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="sequential">Sequential</option>
                  <option value="parallel">Parallel</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Required Agents</label>
                <p className="text-sm text-gray-500 mb-2">
                  Selected: {createPlanForm.required_agents.join(', ')}
                </p>
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={createExecutionPlan}
                disabled={isLoading || !createPlanForm.campaign_id}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Creating...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Create Workflow</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MasterPlannerDashboard;