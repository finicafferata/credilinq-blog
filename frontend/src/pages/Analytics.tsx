import { useState, useEffect } from 'react';
import { useDashboardAnalytics, useAgentAnalytics, useCompetitorIntelligenceAnalytics } from '../hooks/useAnalytics';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Users, Activity, DollarSign, Clock, Target, RefreshCw, Eye, AlertTriangle, Zap } from 'lucide-react';
import { MetricTooltip } from '../components/Tooltip';

export function Analytics() {
  const [selectedDays, setSelectedDays] = useState(30);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [refreshInterval, setRefreshInterval] = useState(30000); // 30 seconds
  const [lastRefresh, setLastRefresh] = useState(new Date());
  
  // Color palette for charts
  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#84CC16', '#F97316'];
  
  const { 
    analytics: dashboardData, 
    isLoading: dashboardLoading, 
    error: dashboardError,
    refetch: refetchDashboard 
  } = useDashboardAnalytics(selectedDays);
  
  const { 
    analytics: agentData, 
    isLoading: agentLoading, 
    error: agentError,
    refetch: refetchAgent 
  } = useAgentAnalytics(selectedAgent || undefined, selectedDays);
  
  const { 
    analytics: ciData, 
    isLoading: ciLoading, 
    error: ciError,
    refetch: refetchCI 
  } = useCompetitorIntelligenceAnalytics(selectedDays);

  const handleDaysChange = (days: number) => {
    setSelectedDays(days);
  };

  // Auto-refresh functionality
  useEffect(() => {
    const interval = setInterval(() => {
      refetchDashboard();
      refetchAgent();
      refetchCI();
      setLastRefresh(new Date());
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval, refetchDashboard, refetchAgent, refetchCI]);

  // Format data for charts
  const formatPerformanceData = () => {
    if (!dashboardData?.recent_performance) return [];
    return dashboardData.recent_performance.map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      executions: item.executions,
      successRate: (item.success_rate * 100).toFixed(1),
    }));
  };

  const formatAgentData = () => {
    if (!dashboardData?.top_performing_agents) return [];
    return dashboardData.top_performing_agents.map(agent => ({
      name: agent.agent_type,
      executions: agent.execution_count,
      successRate: (agent.avg_success_rate * 100).toFixed(1),
    }));
  };

  const formatCostData = () => {
    if (!agentData?.performance_data) return [];
    const costByAgent = agentData.performance_data.reduce((acc, item) => {
      const agent = item.agent_type;
      acc[agent] = (acc[agent] || 0) + item.cost_usd;
      return acc;
    }, {} as Record<string, number>);
    
    return Object.entries(costByAgent).map(([name, cost]) => ({
      name,
      cost: Number(cost.toFixed(3))
    }));
  };

  if (dashboardLoading && !dashboardData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (dashboardError) {
    return (
      <div className="text-center py-12">
        <div className="text-red-600 mb-4">{dashboardError}</div>
        <button onClick={refetchDashboard} className="btn-primary">
          Retry
        </button>
      </div>
    );
  }

  // Show no data state if everything is zero or empty
  const hasData = dashboardData && (
    dashboardData.total_blogs > 0 || 
    dashboardData.total_campaigns > 0 || 
    dashboardData.total_agent_executions > 0
  );

  if (dashboardData && !hasData) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-600 mb-4">
          No analytics data available yet. Start by creating some blog posts or campaigns.
        </div>
        {dashboardData.data_notes && (
          <div className="text-sm text-gray-500 mb-4">
            {dashboardData.data_notes.blog_views && <div>• {dashboardData.data_notes.blog_views}</div>}
            {dashboardData.data_notes.engagement_metrics && <div>• {dashboardData.data_notes.engagement_metrics}</div>}
            {dashboardData.data_notes.status && <div>• {dashboardData.data_notes.status}</div>}
          </div>
        )}
        <button onClick={refetchDashboard} className="btn-secondary">
          Refresh Data
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
              <p className="mt-2 text-gray-600">Monitor your AI agents and content performance</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="flex items-center text-sm text-gray-500">
                <RefreshCw className="w-4 h-4 mr-1" />
                Last updated: {lastRefresh.toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div></div>
        
        <div className="flex items-center space-x-4">
          {/* Auto-refresh toggle */}
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Auto-refresh:</label>
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value={0}>Off</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
              <option value={60000}>1m</option>
            </select>
          </div>
          
          {/* Manual refresh */}
          <button
            onClick={() => {
              refetchDashboard();
              refetchAgent();
              refetchCI();
              setLastRefresh(new Date());
            }}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          
          {/* Time period selector */}
          <div className="flex space-x-2">
            {[7, 30, 90].map((days) => (
              <button
                key={days}
                onClick={() => handleDaysChange(days)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  selectedDays === days
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {days}d
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Enhanced Overview Stats */}
      {dashboardData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="card bg-gradient-to-r from-blue-50 to-blue-100 border-blue-200">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-blue-600">Total Blogs</span>
                  <MetricTooltip
                    title="Total Blog Posts"
                    calculation="Count of all blog posts created in the system across all campaigns and content initiatives."
                    example="Includes published, draft, and archived posts"
                    className="ml-1"
                  />
                </div>
                <div className="text-2xl font-bold text-blue-900">{dashboardData.total_blogs}</div>
              </div>
              <div className="p-3 bg-blue-500 rounded-full">
                <Target className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
          
          <div className="card bg-gradient-to-r from-green-50 to-green-100 border-green-200">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-green-600">Total Campaigns</span>
                  <MetricTooltip
                    title="Total Marketing Campaigns"
                    calculation="Count of all marketing campaigns created, including active, completed, and draft campaigns."
                    example="Multi-channel content campaigns with associated tasks and goals"
                    className="ml-1"
                  />
                </div>
                <div className="text-2xl font-bold text-green-900">{dashboardData.total_campaigns}</div>
              </div>
              <div className="p-3 bg-green-500 rounded-full">
                <Users className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
          
          <div className="card bg-gradient-to-r from-purple-50 to-purple-100 border-purple-200">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-purple-600">Agent Executions</span>
                  <MetricTooltip
                    title="AI Agent Executions"
                    calculation="Total number of AI agent tasks executed across all agent types (planner, researcher, writer, editor, SEO, etc.)."
                    example="Each agent run counts as one execution, regardless of duration or success"
                    className="ml-1"
                  />
                </div>
                <div className="text-2xl font-bold text-purple-900">{dashboardData.total_agent_executions}</div>
              </div>
              <div className="p-3 bg-purple-500 rounded-full">
                <Activity className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
          
          <div className="card bg-gradient-to-r from-emerald-50 to-emerald-100 border-emerald-200">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-emerald-600">Success Rate</span>
                  <MetricTooltip
                    title="Overall Success Rate"
                    calculation="Percentage of successful agent executions out of total executions.

Formula: (Successful Executions ÷ Total Executions) × 100"
                    example="87.6% = 93 successful tasks out of 104 total executions"
                    className="ml-1"
                  />
                </div>
                <div className="text-2xl font-bold text-emerald-900">
                  {(dashboardData.success_rate * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 bg-emerald-500 rounded-full">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Trends Chart */}
      {dashboardData?.recent_performance && (
        <div className="card mb-8">
          <div className="flex items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Performance Trends</h3>
            <MetricTooltip
              title="Performance Trends Over Time"
              calculation="Daily tracking of agent execution count and success rate over the selected time period.

• Executions: Number of agent tasks completed each day
• Success Rate: Percentage of successful tasks per day

Formula: Daily Success Rate = (Daily Successful Tasks ÷ Daily Total Tasks) × 100"
              example="Aug 13: 20 executions with 90.0% success rate means 18 tasks succeeded"
              className="ml-2"
            />
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={formatPerformanceData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ color: '#374151' }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="executions"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.2}
                  name="Executions"
                />
                <Area
                  type="monotone"
                  dataKey="successRate"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.2}
                  name="Success Rate (%)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Agent Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Agent Executions Chart */}
        {dashboardData?.top_performing_agents && (
          <div className="card">
            <div className="flex items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Agent Executions</h3>
              <MetricTooltip
                title="Agent Execution Count by Type"
                calculation="Number of tasks executed by each AI agent type during the selected period.

Agent Types:
• Planner: Content strategy and planning tasks
• Researcher: Information gathering and analysis
• Writer: Content creation and drafting
• Editor: Content review and improvement
• SEO: Search engine optimization analysis
• Image Prompt Generator: Visual content prompts"
                example="Researcher: 25 executions means the research agent completed 25 analysis tasks"
                className="ml-2"
              />
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={formatAgentData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" stroke="#6b7280" fontSize={10} angle={-45} textAnchor="end" height={60} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  />
                  <Bar dataKey="executions" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Cost Distribution Chart */}
        {agentData?.performance_data && (
          <div className="card">
            <div className="flex items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Cost Distribution by Agent</h3>
              <MetricTooltip
                title="AI API Cost Distribution"
                calculation="Distribution of AI model API costs across different agent types based on token usage.

Cost Calculation:
• Input Tokens × Input Rate + Output Tokens × Output Rate
• Rates vary by model (typically $0.001-$0.003 per 1000 tokens)
• Costs accumulate across all executions per agent type

Total Cost = Σ(Input Tokens × Input Rate + Output Tokens × Output Rate) for each agent"
                example="Writer agent: $0.045 total cost from processing 25,000 tokens across all writing tasks"
                className="ml-2"
              />
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={formatCostData()}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="cost"
                  >
                    {formatCostData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`$${value}`, 'Cost']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Top Performing Agents */}
      {dashboardData?.top_performing_agents && (
        <div className="card mb-8">
          <div className="flex items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Top Performing Agents</h3>
            <MetricTooltip
              title="Agent Performance Ranking"
              calculation="Agents ranked by average success rate over the selected period.

Performance Metrics:
• Execution Count: Total number of tasks completed
• Success Rate: Percentage of successful task completions

Formula: Success Rate = (Successful Tasks ÷ Total Tasks) × 100

Ranking is based on success rate with execution count as secondary factor."
              example="Planner: 91.3% success rate from 23 executions means 21 successful planning tasks"
              className="ml-2"
            />
          </div>
          <div className="space-y-4">
            {dashboardData.top_performing_agents.map((agent, index) => (
              <div key={agent.agent_type} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                    <span className="text-primary-600 font-medium text-sm">{index + 1}</span>
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">{agent.agent_type}</div>
                    <div className="text-sm text-gray-500">{agent.execution_count} executions</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-green-600">
                    {(agent.avg_success_rate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">success rate</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Agent Analytics Section */}
      <div className="card mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <h3 className="text-lg font-semibold text-gray-900">Agent Performance</h3>
            <MetricTooltip
              title="Detailed Agent Performance Analytics"
              calculation="Individual execution records showing detailed performance metrics for each agent task.

Table Columns:
• Duration: Task execution time in seconds
• Quality: Performance quality score (0-100%)
• Tokens: Input→Output token usage for AI model calls
• Cost: API cost in USD based on token usage
• Status: Success rate percentage for that execution"
              example="Writer task: 12.8s duration, 88% quality, 186→293 tokens, $0.0008 cost, 87% success"
              className="ml-2"
            />
          </div>
          <select
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="">All Agents</option>
            {dashboardData?.top_performing_agents?.map((agent) => (
              <option key={agent.agent_type} value={agent.agent_type}>
                {agent.agent_type}
              </option>
            ))}
          </select>
        </div>
        
        {agentLoading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
          </div>
        ) : agentError ? (
          <div className="text-center py-8">
            <div className="text-red-600 mb-2">{agentError}</div>
            <button onClick={refetchAgent} className="btn-secondary text-sm">
              Retry
            </button>
          </div>
        ) : agentData?.performance_data?.length ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Task</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tokens</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cost</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {agentData.performance_data.slice(0, 10).map((perf) => (
                  <tr key={perf.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{perf.agent_type}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{perf.task_type}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-900">
                        <Clock className="w-4 h-4 mr-1 text-gray-400" />
                        {(perf.execution_time_ms / 1000).toFixed(1)}s
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-sm font-medium text-gray-900">{(perf.quality_score * 100).toFixed(0)}%</div>
                        <div className={`ml-2 w-2 h-2 rounded-full ${
                          perf.quality_score >= 0.8 ? 'bg-green-400' : 
                          perf.quality_score >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}></div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {perf.input_tokens + perf.output_tokens}
                        <div className="text-xs text-gray-500">
                          {perf.input_tokens}→{perf.output_tokens}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-900">
                        <DollarSign className="w-4 h-4 mr-1 text-gray-400" />
                        {perf.cost_usd.toFixed(3)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        perf.success_rate >= 0.9 ? 'bg-green-100 text-green-800' :
                        perf.success_rate >= 0.7 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {(perf.success_rate * 100).toFixed(0)}% Success
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            No agent performance data available
          </div>
        )}
      </div>

      {/* Enhanced Blog Performance */}
      {dashboardData?.blog_performance && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <h3 className="text-lg font-semibold text-gray-900">Top Blog Performance</h3>
              <MetricTooltip
                title="Blog Content Performance Metrics"
                calculation="Performance analytics for individual blog posts based on audience engagement.

Metrics Explained:
• Views: Total page views and unique visitors
• Engagement Rate: Percentage of visitors who interact with content
• Trend: Performance direction based on recent activity

Formula: Engagement Rate = (Interactions ÷ Total Views) × 100
Interactions include: comments, shares, time on page, scroll depth"
                example="AI Marketing post: 2,847 views with 5.2% engagement = ~148 meaningful interactions"
                className="ml-2"
              />
            </div>
            <button className="px-3 py-1 text-sm bg-primary-100 text-primary-700 rounded-md hover:bg-primary-200 transition-colors">
              View All
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Title</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Views</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Engagement</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {dashboardData.blog_performance.slice(0, 5).map((blog, index) => (
                  <tr key={blog.blog_id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900 truncate max-w-xs">{blog.title}</div>
                      <div className="text-xs text-gray-500">ID: {blog.blog_id.substring(0, 8)}...</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{blog.views.toLocaleString()}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-sm font-medium text-gray-900">
                          {(blog.engagement_rate * 100).toFixed(1)}%
                        </div>
                        <div className={`ml-2 w-2 h-2 rounded-full ${
                          blog.engagement_rate >= 0.05 ? 'bg-green-400' : 
                          blog.engagement_rate >= 0.02 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}></div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {index < 2 ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {/* Competitive Intelligence Analytics */}
      {ciData && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <h2 className="text-xl font-semibold text-gray-900">Competitive Intelligence</h2>
              <MetricTooltip
                title="Competitive Intelligence Analytics"
                calculation="Automated monitoring and analysis of competitor content and marketing activities.

Key Metrics:
• Competitors: Number of companies being monitored
• Active Monitoring: Currently tracked competitors with recent activity
• Content Analyzed: Total pieces of competitor content processed
• Trends Identified: Emerging patterns and topics detected
• Alerts Generated: Notifications for significant competitor activities

Data is collected through web scraping, social media monitoring, and content analysis algorithms."
                example="12 competitors monitored, 347 content pieces analyzed, 23 trends identified in the last 30 days"
                className="ml-2"
              />
            </div>
            <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
              {selectedDays} days
            </span>
          </div>

          {/* CI Overview Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
            <div className="card bg-gradient-to-r from-cyan-50 to-cyan-100 border-cyan-200">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-cyan-600">Competitors</span>
                    <MetricTooltip
                      title="Total Competitors Monitored"
                      calculation="Total number of competitor companies actively tracked in the competitive intelligence system."
                      example="12 companies including TechCorp, InnovatePlus, DigitalEdge"
                      className="ml-1"
                    />
                  </div>
                  <div className="text-2xl font-bold text-cyan-900">{ciData.total_competitors}</div>
                </div>
                <div className="p-3 bg-cyan-500 rounded-full">
                  <Eye className="w-5 h-5 text-white" />
                </div>
              </div>
            </div>
            
            <div className="card bg-gradient-to-r from-orange-50 to-orange-100 border-orange-200">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-orange-600">Active Monitoring</span>
                    <MetricTooltip
                      title="Active Monitoring Status"
                      calculation="Number of competitors with recent activity and current monitoring status."
                      example="8 out of 12 competitors have published content in the last 7 days"
                      className="ml-1"
                    />
                  </div>
                  <div className="text-2xl font-bold text-orange-900">{ciData.active_monitoring}</div>
                </div>
                <div className="p-3 bg-orange-500 rounded-full">
                  <Activity className="w-5 h-5 text-white" />
                </div>
              </div>
            </div>
            
            <div className="card bg-gradient-to-r from-violet-50 to-violet-100 border-violet-200">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-violet-600">Content Analyzed</span>
                    <MetricTooltip
                      title="Content Analysis Volume"
                      calculation="Total number of competitor content pieces processed by AI analysis algorithms during the selected period."
                      example="347 pieces including blog posts, social media, press releases, and videos"
                      className="ml-1"
                    />
                  </div>
                  <div className="text-2xl font-bold text-violet-900">{ciData.content_analyzed}</div>
                </div>
                <div className="p-3 bg-violet-500 rounded-full">
                  <Target className="w-5 h-5 text-white" />
                </div>
              </div>
            </div>
            
            <div className="card bg-gradient-to-r from-yellow-50 to-yellow-100 border-yellow-200">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-yellow-600">Trends</span>
                    <MetricTooltip
                      title="Trend Identification"
                      calculation="Number of emerging topics and patterns identified through content analysis and sentiment tracking."
                      example="23 trends like 'AI & Machine Learning' growth at 28% vs 'Remote Work' decline at -10%"
                      className="ml-1"
                    />
                  </div>
                  <div className="text-2xl font-bold text-yellow-900">{ciData.trends_identified}</div>
                </div>
                <div className="p-3 bg-yellow-500 rounded-full">
                  <Zap className="w-5 h-5 text-white" />
                </div>
              </div>
            </div>
            
            <div className="card bg-gradient-to-r from-red-50 to-red-100 border-red-200">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-red-600">Alerts</span>
                    <MetricTooltip
                      title="Intelligence Alerts"
                      calculation="Number of significant competitor activities that triggered automated alerts during the monitoring period."
                      example="5 alerts: major product launches, significant content campaigns, or market positioning changes"
                      className="ml-1"
                    />
                  </div>
                  <div className="text-2xl font-bold text-red-900">{ciData.alerts_generated}</div>
                </div>
                <div className="p-3 bg-red-500 rounded-full">
                  <AlertTriangle className="w-5 h-5 text-white" />
                </div>
              </div>
            </div>
          </div>

          {/* CI Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Content Types Distribution */}
            {ciData.content_types_distribution && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Content Types Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={ciData.content_types_distribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ type, percentage }) => `${type}: ${percentage}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="count"
                      >
                        {ciData.content_types_distribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Platform Activity */}
            {ciData.platform_activity && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Platform Activity</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={ciData.platform_activity}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="platform" stroke="#6b7280" fontSize={10} angle={-45} textAnchor="end" height={60} />
                      <YAxis stroke="#6b7280" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                      />
                      <Bar dataKey="posts" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {/* Top Competitors and Trending Topics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Competitors */}
            {ciData.top_competitors && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Competitors</h3>
                <div className="space-y-3">
                  {ciData.top_competitors.slice(0, 5).map((competitor, index) => (
                    <div key={competitor.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                          <span className="text-primary-600 font-medium text-sm">{index + 1}</span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{competitor.name}</div>
                          <div className="text-sm text-gray-500">{competitor.domain}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium text-gray-900">{competitor.content_count}</div>
                        <div className="text-xs text-gray-500">content items</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Trending Topics */}
            {ciData.trending_topics && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Trending Topics</h3>
                <div className="space-y-3">
                  {ciData.trending_topics.slice(0, 5).map((topic, index) => (
                    <div key={topic.topic} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{topic.topic}</div>
                        <div className="text-sm text-gray-500">{topic.mentions} mentions</div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className={`text-sm font-medium ${ 
                          topic.growth_rate > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {topic.growth_rate > 0 ? '+' : ''}{topic.growth_rate.toFixed(1)}%
                        </div>
                        {topic.growth_rate > 0 ? (
                          <TrendingUp className="w-4 h-4 text-green-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

        {/* Export and Actions */}
        <div className="flex justify-end mt-8">
          <div className="flex space-x-3">
            <button className="px-4 py-2 text-sm border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors">
              Export CSV
            </button>
            <button className="px-4 py-2 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors">
              Generate Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Analytics;