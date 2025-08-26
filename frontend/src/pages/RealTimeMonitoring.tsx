import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Activity,
  Zap,
  Clock,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Eye,
  EyeOff,
  Play,
  Pause,
  Square,
  Filter,
  Download,
  Bell,
  Settings,
  Maximize2,
  Minimize2,
  Users,
  BarChart3,
  LineChart,
  PieChart,
  Cpu,
  Database,
  Network,
  Globe,
  Wifi,
  WifiOff,
  MessageSquare
} from 'lucide-react';

// Types for real-time data
interface MetricPoint {
  timestamp: number;
  value: number;
  label?: string;
}

interface CampaignMetrics {
  id: string;
  name: string;
  progress: number;
  tasksCompleted: number;
  tasksTotal: number;
  activeAgents: number;
  throughput: MetricPoint[];
  latency: MetricPoint[];
  errorRate: MetricPoint[];
  status: 'running' | 'paused' | 'completed' | 'failed';
}

interface AgentMetrics {
  id: string;
  name: string;
  type: string;
  status: 'online' | 'busy' | 'idle' | 'offline';
  cpu: MetricPoint[];
  memory: MetricPoint[];
  taskQueue: number;
  currentTask?: string;
  performance: {
    successRate: number;
    avgResponseTime: number;
    tasksPerHour: number;
  };
}

interface SystemMetrics {
  timestamp: number;
  totalCampaigns: number;
  activeCampaigns: number;
  totalAgents: number;
  activeAgents: number;
  systemLoad: MetricPoint[];
  networkLatency: MetricPoint[];
  queueDepth: MetricPoint[];
  throughput: MetricPoint[];
  errorRate: number;
  warningCount: number;
}

interface Event {
  id: string;
  timestamp: number;
  type: 'info' | 'warning' | 'error' | 'success';
  category: 'campaign' | 'agent' | 'system';
  title: string;
  message: string;
  metadata?: any;
}

// Mock WebSocket service
class MonitoringWebSocket {
  private listeners: Map<string, Set<Function>> = new Map();
  private interval: NodeJS.Timeout | null = null;
  
  connect() {
    console.log('Connecting to monitoring WebSocket...');
    
    // Simulate real-time data updates
    this.interval = setInterval(() => {
      this.simulateDataUpdate();
    }, 1000);
  }
  
  disconnect() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }
  
  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }
  
  off(event: string, callback: Function) {
    this.listeners.get(event)?.delete(callback);
  }
  
  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }
  
  private simulateDataUpdate() {
    const now = Date.now();
    
    // Emit system metrics
    this.emit('system:metrics', {
      timestamp: now,
      systemLoad: Math.random() * 100,
      networkLatency: Math.random() * 50,
      queueDepth: Math.floor(Math.random() * 20),
      throughput: Math.random() * 1000
    });
    
    // Emit campaign updates
    if (Math.random() > 0.7) {
      this.emit('campaign:update', {
        id: `camp_${Math.floor(Math.random() * 3) + 1}`,
        progress: Math.random() * 100,
        tasksCompleted: Math.floor(Math.random() * 50)
      });
    }
    
    // Emit agent updates
    if (Math.random() > 0.6) {
      this.emit('agent:update', {
        id: `agent_${Math.floor(Math.random() * 5) + 1}`,
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        status: ['online', 'busy', 'idle'][Math.floor(Math.random() * 3)]
      });
    }
    
    // Emit events
    if (Math.random() > 0.9) {
      this.emit('event:new', {
        id: `evt_${now}`,
        timestamp: now,
        type: ['info', 'warning', 'error', 'success'][Math.floor(Math.random() * 4)],
        category: ['campaign', 'agent', 'system'][Math.floor(Math.random() * 3)],
        title: 'System Event',
        message: 'An event occurred in the system'
      });
    }
  }
}

// Chart component for metrics visualization
function MetricsChart({ 
  data, 
  color = '#3B82F6', 
  height = 60,
  showAxis = false 
}: { 
  data: MetricPoint[]; 
  color?: string; 
  height?: number;
  showAxis?: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;
    
    const svg = svgRef.current;
    const width = svg.clientWidth;
    const padding = showAxis ? 20 : 5;
    
    const minValue = Math.min(...data.map(d => d.value));
    const maxValue = Math.max(...data.map(d => d.value));
    const valueRange = maxValue - minValue || 1;
    
    const points = data.map((point, index) => {
      const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((point.value - minValue) / valueRange) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');
    
    // Clear previous content
    svg.innerHTML = '';
    
    // Create gradient
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const linearGradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    linearGradient.setAttribute('id', 'gradient');
    linearGradient.setAttribute('x1', '0%');
    linearGradient.setAttribute('y1', '0%');
    linearGradient.setAttribute('x2', '0%');
    linearGradient.setAttribute('y2', '100%');
    
    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', color);
    stop1.setAttribute('stop-opacity', '0.3');
    
    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', color);
    stop2.setAttribute('stop-opacity', '0');
    
    linearGradient.appendChild(stop1);
    linearGradient.appendChild(stop2);
    gradient.appendChild(linearGradient);
    svg.appendChild(gradient);
    
    // Create area
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    area.setAttribute('points', `${points} ${width - padding},${height - padding} ${padding},${height - padding}`);
    area.setAttribute('fill', 'url(#gradient)');
    svg.appendChild(area);
    
    // Create line
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', points);
    line.setAttribute('fill', 'none');
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', '2');
    svg.appendChild(line);
    
    // Add axis if needed
    if (showAxis) {
      const axis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axis.setAttribute('x1', padding.toString());
      axis.setAttribute('y1', (height - padding).toString());
      axis.setAttribute('x2', (width - padding).toString());
      axis.setAttribute('y2', (height - padding).toString());
      axis.setAttribute('stroke', '#E5E7EB');
      axis.setAttribute('stroke-width', '1');
      svg.appendChild(axis);
    }
  }, [data, color, height, showAxis]);
  
  return (
    <svg ref={svgRef} className="w-full" style={{ height: `${height}px` }} />
  );
}

export function RealTimeMonitoring() {
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [selectedView, setSelectedView] = useState<'overview' | 'campaigns' | 'agents' | 'events'>('overview');
  const [timeRange, setTimeRange] = useState<'1m' | '5m' | '15m' | '1h'>('5m');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showAlerts, setShowAlerts] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  
  // Metrics state
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    timestamp: Date.now(),
    totalCampaigns: 12,
    activeCampaigns: 3,
    totalAgents: 8,
    activeAgents: 5,
    systemLoad: [],
    networkLatency: [],
    queueDepth: [],
    throughput: [],
    errorRate: 0.02,
    warningCount: 3
  });
  
  const [campaigns, setCampaigns] = useState<CampaignMetrics[]>([]);
  const [agents, setAgents] = useState<AgentMetrics[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  
  const wsRef = useRef<MonitoringWebSocket | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new MonitoringWebSocket();
    wsRef.current = ws;
    
    ws.connect();
    setIsConnected(true);
    
    // Subscribe to events
    ws.on('system:metrics', (data: any) => {
      if (!isPaused) {
        setSystemMetrics(prev => ({
          ...prev,
          timestamp: data.timestamp,
          systemLoad: [...prev.systemLoad.slice(-59), { timestamp: data.timestamp, value: data.systemLoad }],
          networkLatency: [...prev.networkLatency.slice(-59), { timestamp: data.timestamp, value: data.networkLatency }],
          queueDepth: [...prev.queueDepth.slice(-59), { timestamp: data.timestamp, value: data.queueDepth }],
          throughput: [...prev.throughput.slice(-59), { timestamp: data.timestamp, value: data.throughput }]
        }));
      }
    });
    
    ws.on('campaign:update', (data: any) => {
      if (!isPaused) {
        setCampaigns(prev => {
          const existing = prev.find(c => c.id === data.id);
          if (existing) {
            return prev.map(c => c.id === data.id ? { ...c, ...data } : c);
          }
          return prev;
        });
      }
    });
    
    ws.on('agent:update', (data: any) => {
      if (!isPaused) {
        setAgents(prev => {
          const existing = prev.find(a => a.id === data.id);
          if (existing) {
            return prev.map(a => a.id === data.id ? {
              ...a,
              status: data.status,
              cpu: [...a.cpu.slice(-59), { timestamp: Date.now(), value: data.cpu }],
              memory: [...a.memory.slice(-59), { timestamp: Date.now(), value: data.memory }]
            } : a);
          }
          return prev;
        });
      }
    });
    
    ws.on('event:new', (event: Event) => {
      if (!isPaused) {
        setEvents(prev => [event, ...prev.slice(0, 99)]);
      }
    });
    
    // Initialize with mock data
    initializeMockData();
    
    return () => {
      ws.disconnect();
    };
  }, [isPaused]);

  // Auto-scroll to latest events
  useEffect(() => {
    if (autoScroll && eventsEndRef.current) {
      eventsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScroll]);

  const initializeMockData = () => {
    // Mock campaigns
    setCampaigns([
      {
        id: 'camp_1',
        name: 'Q1 Content Marketing',
        progress: 65,
        tasksCompleted: 13,
        tasksTotal: 20,
        activeAgents: 3,
        throughput: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: Math.random() * 100
        })),
        latency: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: Math.random() * 50
        })),
        errorRate: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: Math.random() * 5
        })),
        status: 'running'
      },
      {
        id: 'camp_2',
        name: 'SEO Optimization Sprint',
        progress: 30,
        tasksCompleted: 6,
        tasksTotal: 20,
        activeAgents: 2,
        throughput: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: Math.random() * 80
        })),
        latency: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: Math.random() * 40
        })),
        errorRate: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: Math.random() * 3
        })),
        status: 'running'
      }
    ]);

    // Mock agents
    setAgents([
      {
        id: 'agent_1',
        name: 'ContentMaster Pro',
        type: 'content_writer',
        status: 'busy',
        cpu: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: 60 + Math.random() * 30
        })),
        memory: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: 50 + Math.random() * 20
        })),
        taskQueue: 3,
        currentTask: 'Writing blog post',
        performance: {
          successRate: 96,
          avgResponseTime: 1200,
          tasksPerHour: 4
        }
      },
      {
        id: 'agent_2',
        name: 'SEO Optimizer',
        type: 'seo_specialist',
        status: 'online',
        cpu: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: 30 + Math.random() * 20
        })),
        memory: Array.from({ length: 60 }, (_, i) => ({
          timestamp: Date.now() - (60 - i) * 1000,
          value: 40 + Math.random() * 15
        })),
        taskQueue: 1,
        performance: {
          successRate: 98,
          avgResponseTime: 800,
          tasksPerHour: 6
        }
      }
    ]);

    // Mock events
    setEvents([
      {
        id: 'evt_1',
        timestamp: Date.now() - 5000,
        type: 'success',
        category: 'campaign',
        title: 'Campaign Started',
        message: 'Q1 Content Marketing campaign has been started successfully'
      },
      {
        id: 'evt_2',
        timestamp: Date.now() - 15000,
        type: 'info',
        category: 'agent',
        title: 'Agent Online',
        message: 'ContentMaster Pro agent is now online and ready'
      },
      {
        id: 'evt_3',
        timestamp: Date.now() - 30000,
        type: 'warning',
        category: 'system',
        title: 'High Memory Usage',
        message: 'System memory usage is above 80%'
      }
    ]);
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-600" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-600" />;
      default: return <Activity className="w-4 h-4 text-blue-600" />;
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': case 'online': case 'busy': return 'text-green-600';
      case 'paused': case 'idle': return 'text-yellow-600';
      case 'failed': case 'offline': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className={`min-h-screen bg-gray-900 text-white ${fullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold flex items-center space-x-2">
                <Activity className="w-6 h-6 text-blue-500" />
                <span>Real-Time Monitoring</span>
              </h1>
              
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                {isConnected ? (
                  <>
                    <Wifi className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-green-500">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-4 h-4 text-red-500" />
                    <span className="text-sm text-red-500">Disconnected</span>
                  </>
                )}
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Time Range Selector */}
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as any)}
                className="bg-gray-700 text-white px-3 py-1 rounded-lg text-sm"
              >
                <option value="1m">Last 1 min</option>
                <option value="5m">Last 5 min</option>
                <option value="15m">Last 15 min</option>
                <option value="1h">Last 1 hour</option>
              </select>

              {/* Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsPaused(!isPaused)}
                  className={`p-2 rounded-lg ${isPaused ? 'bg-yellow-600' : 'bg-gray-700'} hover:bg-gray-600`}
                  title={isPaused ? 'Resume' : 'Pause'}
                >
                  {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                </button>
                
                <button
                  onClick={() => initializeMockData()}
                  className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
                  title="Refresh"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
                
                <button
                  onClick={() => setShowAlerts(!showAlerts)}
                  className={`p-2 rounded-lg ${showAlerts ? 'bg-blue-600' : 'bg-gray-700'} hover:bg-gray-600`}
                  title="Toggle alerts"
                >
                  <Bell className="w-4 h-4" />
                </button>
                
                <button
                  onClick={() => setFullscreen(!fullscreen)}
                  className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
                  title={fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                  {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>

          {/* View Tabs */}
          <div className="flex items-center space-x-4 mt-4">
            {['overview', 'campaigns', 'agents', 'events'].map(view => (
              <button
                key={view}
                onClick={() => setSelectedView(view as any)}
                className={`px-4 py-2 rounded-lg capitalize ${
                  selectedView === view 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {view}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {selectedView === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {/* System Health */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center justify-between">
                <span>System Health</span>
                <Activity className="w-5 h-5 text-blue-500" />
              </h3>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-400">System Load</span>
                    <span>{systemMetrics.systemLoad[systemMetrics.systemLoad.length - 1]?.value.toFixed(0)}%</span>
                  </div>
                  <MetricsChart data={systemMetrics.systemLoad} color="#3B82F6" />
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-400">Network Latency</span>
                    <span>{systemMetrics.networkLatency[systemMetrics.networkLatency.length - 1]?.value.toFixed(0)}ms</span>
                  </div>
                  <MetricsChart data={systemMetrics.networkLatency} color="#10B981" />
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-400">Queue Depth</span>
                    <span>{systemMetrics.queueDepth[systemMetrics.queueDepth.length - 1]?.value.toFixed(0)}</span>
                  </div>
                  <MetricsChart data={systemMetrics.queueDepth} color="#F59E0B" />
                </div>
              </div>
            </div>

            {/* Campaign Status */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center justify-between">
                <span>Active Campaigns</span>
                <BarChart3 className="w-5 h-5 text-green-500" />
              </h3>
              
              <div className="space-y-3">
                {campaigns.slice(0, 3).map(campaign => (
                  <div key={campaign.id} className="border border-gray-700 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm">{campaign.name}</span>
                      <span className={`text-xs ${getStatusColor(campaign.status)}`}>
                        {campaign.status}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-400">
                      <span>{campaign.tasksCompleted}/{campaign.tasksTotal} tasks</span>
                      <span>{campaign.activeAgents} agents</span>
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${campaign.progress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Live Events */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 flex items-center justify-between">
                <span>Live Events</span>
                <MessageSquare className="w-5 h-5 text-purple-500" />
              </h3>
              
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {events.slice(0, 10).map(event => (
                  <div key={event.id} className="flex items-start space-x-2 p-2 hover:bg-gray-700 rounded">
                    {getEventIcon(event.type)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{event.title}</p>
                      <p className="text-xs text-gray-400">{formatTime(event.timestamp)}</p>
                    </div>
                  </div>
                ))}
                <div ref={eventsEndRef} />
              </div>
            </div>
          </div>
        )}

        {selectedView === 'campaigns' && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {campaigns.map(campaign => (
              <div key={campaign.id} className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">{campaign.name}</h3>
                  <span className={`px-3 py-1 rounded-full text-sm ${
                    campaign.status === 'running' ? 'bg-green-600' :
                    campaign.status === 'paused' ? 'bg-yellow-600' :
                    'bg-gray-600'
                  }`}>
                    {campaign.status}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-gray-400 text-sm">Progress</p>
                    <p className="text-2xl font-bold">{campaign.progress.toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Tasks</p>
                    <p className="text-2xl font-bold">{campaign.tasksCompleted}/{campaign.tasksTotal}</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Throughput</p>
                    <MetricsChart data={campaign.throughput} color="#10B981" height={40} />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Latency</p>
                    <MetricsChart data={campaign.latency} color="#F59E0B" height={40} />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Error Rate</p>
                    <MetricsChart data={campaign.errorRate} color="#EF4444" height={40} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'agents' && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {agents.map(agent => (
              <div key={agent.id} className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold">{agent.name}</h3>
                    <p className="text-sm text-gray-400">{agent.type}</p>
                  </div>
                  <span className={`w-3 h-3 rounded-full ${
                    agent.status === 'online' || agent.status === 'busy' ? 'bg-green-500' :
                    agent.status === 'idle' ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`} />
                </div>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">CPU</span>
                      <span>{agent.cpu[agent.cpu.length - 1]?.value.toFixed(0)}%</span>
                    </div>
                    <MetricsChart data={agent.cpu} color="#3B82F6" height={30} />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Memory</span>
                      <span>{agent.memory[agent.memory.length - 1]?.value.toFixed(0)}%</span>
                    </div>
                    <MetricsChart data={agent.memory} color="#8B5CF6" height={30} />
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 text-center pt-2 border-t border-gray-700">
                    <div>
                      <p className="text-lg font-bold text-green-500">{agent.performance.successRate}%</p>
                      <p className="text-xs text-gray-400">Success</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-blue-500">{agent.performance.tasksPerHour}</p>
                      <p className="text-xs text-gray-400">Tasks/hr</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-yellow-500">{agent.taskQueue}</p>
                      <p className="text-xs text-gray-400">Queued</p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'events' && (
          <div className="bg-gray-800 rounded-xl border border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Event Stream</h3>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Auto-scroll</span>
                </label>
              </div>
            </div>
            
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full">
                <thead className="bg-gray-900 sticky top-0">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Time</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Type</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Category</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400">Event</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {events.map(event => (
                    <tr key={event.id} className="hover:bg-gray-700">
                      <td className="px-4 py-2 text-sm text-gray-400">
                        {formatTime(event.timestamp)}
                      </td>
                      <td className="px-4 py-2">
                        {getEventIcon(event.type)}
                      </td>
                      <td className="px-4 py-2 text-sm text-gray-400 capitalize">
                        {event.category}
                      </td>
                      <td className="px-4 py-2">
                        <p className="text-sm font-medium">{event.title}</p>
                        <p className="text-xs text-gray-400">{event.message}</p>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div ref={eventsEndRef} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RealTimeMonitoring;