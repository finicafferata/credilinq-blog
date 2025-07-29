import { useState } from 'react';
import { useDashboardAnalytics, useAgentAnalytics } from '../hooks/useAnalytics';

export function Analytics() {
  const [selectedDays, setSelectedDays] = useState(30);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  
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

  const handleDaysChange = (days: number) => {
    setSelectedDays(days);
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

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="text-gray-600 mt-1">Monitor your AI agents and content performance</p>
        </div>
        
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

      {/* Overview Stats */}
      {dashboardData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Total Blogs</div>
            <div className="text-2xl font-bold text-gray-900">{dashboardData.total_blogs}</div>
          </div>
          
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Total Campaigns</div>
            <div className="text-2xl font-bold text-gray-900">{dashboardData.total_campaigns}</div>
          </div>
          
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Agent Executions</div>
            <div className="text-2xl font-bold text-gray-900">{dashboardData.total_agent_executions}</div>
          </div>
          
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Success Rate</div>
            <div className="text-2xl font-bold text-green-600">
              {(dashboardData.success_rate * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Top Performing Agents */}
      {dashboardData?.top_performing_agents && (
        <div className="card mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Performing Agents</h3>
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
          <h3 className="text-lg font-semibold text-gray-900">Agent Performance</h3>
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
          <div className="space-y-3">
            {agentData.performance_data.slice(0, 10).map((perf) => (
              <div key={perf.id} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-b-0">
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{perf.agent_type}</div>
                  <div className="text-sm text-gray-500">{perf.task_type}</div>
                </div>
                <div className="text-right mr-4">
                  <div className="text-sm font-medium">{(perf.execution_time_ms / 1000).toFixed(1)}s</div>
                  <div className="text-xs text-gray-500">execution time</div>
                </div>
                <div className="text-right mr-4">
                  <div className="text-sm font-medium">{(perf.quality_score * 100).toFixed(0)}%</div>
                  <div className="text-xs text-gray-500">quality</div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium">${perf.cost_usd.toFixed(3)}</div>
                  <div className="text-xs text-gray-500">cost</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            No agent performance data available
          </div>
        )}
      </div>

      {/* Blog Performance */}
      {dashboardData?.blog_performance && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Blog Performance</h3>
          <div className="space-y-3">
            {dashboardData.blog_performance.slice(0, 5).map((blog) => (
              <div key={blog.blog_id} className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="font-medium text-gray-900 truncate">{blog.title}</div>
                  <div className="text-sm text-gray-500">Blog ID: {blog.blog_id.substring(0, 8)}...</div>
                </div>
                <div className="text-right mr-4">
                  <div className="font-medium text-gray-900">{blog.views}</div>
                  <div className="text-xs text-gray-500">views</div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-green-600">
                    {(blog.engagement_rate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">engagement</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}