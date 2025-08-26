import React, { useState, useEffect, useMemo } from 'react';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Target,
  Zap,
  Gem,
  Lightbulb,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  BarChart3,
  LineChart,
  PieChart,
  Activity,
  Award,
  Shield,
  Rocket,
  Calendar,
  RefreshCw,
  Download,
  Settings,
  Eye,
  ArrowRight,
  Star,
  Flame,
  Layers,
  GitBranch,
  Cpu,
  Database,
  Globe,
  Gauge
} from 'lucide-react';

// Types for predictive analytics
interface PredictionModel {
  id: string;
  name: string;
  type: 'campaign_performance' | 'agent_utilization' | 'cost_optimization' | 'quality_prediction' | 'demand_forecasting';
  accuracy: number; // 0-100
  lastTrained: string;
  dataPoints: number;
  status: 'active' | 'training' | 'inactive';
  confidence: number; // 0-100
}

interface Forecast {
  metric: string;
  currentValue: number;
  predictions: Array<{
    date: string;
    value: number;
    confidence: number;
    scenario: 'best' | 'expected' | 'worst';
  }>;
  trend: 'increasing' | 'decreasing' | 'stable';
  seasonality: {
    detected: boolean;
    pattern?: string;
    impact: number;
  };
}

interface SmartRecommendation {
  id: string;
  title: string;
  category: 'optimization' | 'scaling' | 'cost_reduction' | 'quality_improvement' | 'risk_mitigation';
  priority: 'low' | 'medium' | 'high' | 'critical';
  impact: {
    metric: string;
    expectedImprovement: number;
    timeToRealization: string;
    confidence: number;
  };
  aiReasoning: string;
  implementation: {
    steps: string[];
    resources: string[];
    risks: string[];
    dependencies: string[];
  };
  basedOn: string[];
  learningSource: 'historical_data' | 'similar_patterns' | 'industry_benchmarks' | 'ml_model';
}

interface AnomalyDetection {
  id: string;
  timestamp: string;
  metric: string;
  value: number;
  expectedValue: number;
  deviation: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'performance' | 'cost' | 'quality' | 'security' | 'capacity';
  description: string;
  possibleCauses: string[];
  suggestedActions: string[];
  autoResolved: boolean;
}

interface ScenarioAnalysis {
  id: string;
  name: string;
  description: string;
  assumptions: Array<{
    parameter: string;
    baseValue: number;
    scenarioValue: number;
    impact: number;
  }>;
  outcomes: Array<{
    metric: string;
    baselineValue: number;
    scenarioValue: number;
    change: number;
  }>;
  probability: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export function PredictiveAnalytics() {
  const [selectedTimeHorizon, setSelectedTimeHorizon] = useState<'1w' | '1m' | '3m' | '6m' | '1y'>('1m');
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [selectedView, setSelectedView] = useState<'forecasts' | 'recommendations' | 'anomalies' | 'scenarios'>('forecasts');
  const [confidenceThreshold, setConfidenceThreshold] = useState(75);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // State for predictive data
  const [models, setModels] = useState<PredictionModel[]>([]);
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [recommendations, setRecommendations] = useState<SmartRecommendation[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyDetection[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioAnalysis[]>([]);

  // Initialize mock data
  useEffect(() => {
    initializeMockData();
  }, [selectedTimeHorizon]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      updatePredictions();
    }, 10000); // Update every 10 seconds
    
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const initializeMockData = () => {
    // Mock ML models
    setModels([
      {
        id: 'model_1',
        name: 'Campaign Success Predictor',
        type: 'campaign_performance',
        accuracy: 92.5,
        lastTrained: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        dataPoints: 15420,
        status: 'active',
        confidence: 88
      },
      {
        id: 'model_2',
        name: 'Agent Workload Optimizer',
        type: 'agent_utilization',
        accuracy: 89.3,
        lastTrained: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
        dataPoints: 8950,
        status: 'active',
        confidence: 85
      },
      {
        id: 'model_3',
        name: 'Cost Prediction Engine',
        type: 'cost_optimization',
        accuracy: 87.8,
        lastTrained: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        dataPoints: 12340,
        status: 'active',
        confidence: 82
      },
      {
        id: 'model_4',
        name: 'Quality Forecaster',
        type: 'quality_prediction',
        accuracy: 94.1,
        lastTrained: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
        dataPoints: 9876,
        status: 'training',
        confidence: 91
      },
      {
        id: 'model_5',
        name: 'Demand Intelligence',
        type: 'demand_forecasting',
        accuracy: 86.7,
        lastTrained: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
        dataPoints: 18750,
        status: 'active',
        confidence: 79
      }
    ]);

    // Mock forecasts
    setForecasts([
      {
        metric: 'Campaign Success Rate',
        currentValue: 94.2,
        predictions: generateTimeSeries(94.2, 30, 'increasing'),
        trend: 'increasing',
        seasonality: {
          detected: true,
          pattern: 'Weekly peak on Tuesdays',
          impact: 12
        }
      },
      {
        metric: 'Agent Utilization',
        currentValue: 72.5,
        predictions: generateTimeSeries(72.5, 30, 'increasing'),
        trend: 'increasing',
        seasonality: {
          detected: false,
          impact: 0
        }
      },
      {
        metric: 'Cost per Task',
        currentValue: 4.85,
        predictions: generateTimeSeries(4.85, 30, 'decreasing'),
        trend: 'decreasing',
        seasonality: {
          detected: true,
          pattern: 'Monthly billing cycle',
          impact: 8
        }
      },
      {
        metric: 'Quality Score',
        currentValue: 91.8,
        predictions: generateTimeSeries(91.8, 30, 'stable'),
        trend: 'stable',
        seasonality: {
          detected: false,
          impact: 0
        }
      }
    ]);

    // Mock recommendations
    setRecommendations([
      {
        id: 'rec_1',
        title: 'Implement Dynamic Agent Scaling',
        category: 'scaling',
        priority: 'high',
        impact: {
          metric: 'System Throughput',
          expectedImprovement: 23.5,
          timeToRealization: '2-3 weeks',
          confidence: 87
        },
        aiReasoning: 'Analysis of workload patterns shows 40% capacity utilization variance across peak/off-peak hours. ML models predict 23.5% throughput improvement with dynamic scaling.',
        implementation: {
          steps: [
            'Analyze current workload patterns and peak usage times',
            'Implement auto-scaling triggers based on queue depth',
            'Set up monitoring for scaling events',
            'Optimize scaling parameters based on performance'
          ],
          resources: ['DevOps Engineer', 'ML Engineer', 'System Architect'],
          risks: ['Potential service disruption during deployment', 'Increased infrastructure costs'],
          dependencies: ['Load balancer configuration', 'Monitoring infrastructure']
        },
        basedOn: ['Historical utilization patterns', 'Queue depth analysis', 'Response time metrics'],
        learningSource: 'ml_model'
      },
      {
        id: 'rec_2',
        title: 'Optimize Content Quality Routing',
        category: 'quality_improvement',
        priority: 'medium',
        impact: {
          metric: 'Content Quality Score',
          expectedImprovement: 8.3,
          timeToRealization: '1-2 weeks',
          confidence: 92
        },
        aiReasoning: 'Pattern analysis reveals that agent specialization alignment with content types increases quality scores by 8.3% on average.',
        implementation: {
          steps: [
            'Map agent capabilities to content types',
            'Implement intelligent routing algorithm',
            'A/B test routing performance',
            'Monitor quality improvements'
          ],
          resources: ['Content Strategist', 'ML Engineer'],
          risks: ['Learning curve for new routing system'],
          dependencies: ['Agent capability assessment', 'Content classification system']
        },
        basedOn: ['Content type analysis', 'Agent performance data', 'Quality score correlation'],
        learningSource: 'historical_data'
      },
      {
        id: 'rec_3',
        title: 'Predictive Maintenance for AI Models',
        category: 'optimization',
        priority: 'medium',
        impact: {
          metric: 'Model Accuracy',
          expectedImprovement: 5.7,
          timeToRealization: '3-4 weeks',
          confidence: 78
        },
        aiReasoning: 'Model drift detection indicates accuracy degradation patterns. Predictive retraining schedule could maintain optimal performance.',
        implementation: {
          steps: [
            'Set up model performance monitoring',
            'Implement drift detection algorithms',
            'Create automated retraining pipelines',
            'Establish performance thresholds'
          ],
          resources: ['ML Engineer', 'Data Scientist'],
          risks: ['Model downtime during retraining'],
          dependencies: ['Training data pipeline', 'Model versioning system']
        },
        basedOn: ['Model accuracy trends', 'Drift detection metrics', 'Performance degradation patterns'],
        learningSource: 'ml_model'
      }
    ]);

    // Mock anomalies
    setAnomalies([
      {
        id: 'anom_1',
        timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
        metric: 'Agent Response Time',
        value: 2850,
        expectedValue: 1200,
        deviation: 137.5,
        severity: 'high',
        category: 'performance',
        description: 'Agent response time increased significantly above normal range',
        possibleCauses: ['High system load', 'Network latency', 'Agent overutilization'],
        suggestedActions: ['Scale up agents', 'Check network connectivity', 'Redistribute workload'],
        autoResolved: false
      },
      {
        id: 'anom_2',
        timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
        metric: 'Content Quality Score',
        value: 76.2,
        expectedValue: 91.8,
        deviation: -17.0,
        severity: 'medium',
        category: 'quality',
        description: 'Quality score dropped below expected threshold',
        possibleCauses: ['New content type', 'Agent configuration change', 'Training data shift'],
        suggestedActions: ['Review recent configurations', 'Retrain quality models', 'Analyze content patterns'],
        autoResolved: false
      },
      {
        id: 'anom_3',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        metric: 'System Memory Usage',
        value: 94.2,
        expectedValue: 68.5,
        deviation: 37.6,
        severity: 'critical',
        category: 'capacity',
        description: 'Memory usage approaching critical levels',
        possibleCauses: ['Memory leak', 'Increased workload', 'Inefficient processes'],
        suggestedActions: ['Restart affected services', 'Scale up resources', 'Investigate memory leaks'],
        autoResolved: true
      }
    ]);

    // Mock scenarios
    setScenarios([
      {
        id: 'scenario_1',
        name: 'Peak Season Demand Surge',
        description: 'Simulates 150% increase in campaign volume during peak business season',
        assumptions: [
          { parameter: 'Campaign Volume', baseValue: 100, scenarioValue: 250, impact: 8.5 },
          { parameter: 'Agent Pool Size', baseValue: 12, scenarioValue: 18, impact: 6.2 },
          { parameter: 'Quality Threshold', baseValue: 90, scenarioValue: 88, impact: -2.3 }
        ],
        outcomes: [
          { metric: 'System Throughput', baselineValue: 450, scenarioValue: 675, change: 50 },
          { metric: 'Response Time', baselineValue: 1200, scenarioValue: 1650, change: 37.5 },
          { metric: 'Cost per Task', baselineValue: 4.85, scenarioValue: 5.20, change: 7.2 },
          { metric: 'Quality Score', baselineValue: 91.8, scenarioValue: 89.2, change: -2.8 }
        ],
        probability: 73,
        riskLevel: 'medium'
      },
      {
        id: 'scenario_2',
        name: 'AI Model Performance Degradation',
        description: 'Models experience gradual accuracy decline due to data drift',
        assumptions: [
          { parameter: 'Model Accuracy', baseValue: 92, scenarioValue: 78, impact: -15.2 },
          { parameter: 'Retraining Frequency', baseValue: 30, scenarioValue: 14, impact: 8.7 },
          { parameter: 'Quality Control', baseValue: 85, scenarioValue: 95, impact: 11.8 }
        ],
        outcomes: [
          { metric: 'Content Quality', baselineValue: 91.8, scenarioValue: 84.2, change: -8.3 },
          { metric: 'Error Rate', baselineValue: 2.1, scenarioValue: 5.8, change: 176.2 },
          { metric: 'Operational Cost', baselineValue: 12400, scenarioValue: 15800, change: 27.4 }
        ],
        probability: 45,
        riskLevel: 'high'
      }
    ]);
  };

  // Helper function to generate time series data
  function generateTimeSeries(baseValue: number, days: number, trend: 'increasing' | 'decreasing' | 'stable') {
    const data = [];
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i);
      
      let trendMultiplier = 1;
      if (trend === 'increasing') trendMultiplier = 1 + (i * 0.002);
      if (trend === 'decreasing') trendMultiplier = 1 - (i * 0.002);
      
      const noise = (Math.random() - 0.5) * 0.1;
      const seasonality = Math.sin(i * 0.4) * 0.05;
      
      data.push({
        date: date.toISOString().split('T')[0],
        value: baseValue * trendMultiplier * (1 + noise + seasonality),
        confidence: 85 + Math.random() * 10,
        scenario: 'expected' as const
      });
    }
    return data;
  }

  const updatePredictions = () => {
    // Simulate real-time updates to predictions
    setForecasts(prev => prev.map(forecast => ({
      ...forecast,
      currentValue: forecast.currentValue + (Math.random() - 0.5) * 2,
      predictions: forecast.predictions.map(p => ({
        ...p,
        value: p.value + (Math.random() - 0.5) * 1,
        confidence: Math.max(70, Math.min(95, p.confidence + (Math.random() - 0.5) * 5))
      }))
    })));
  };

  // Filter data based on confidence threshold
  const filteredRecommendations = useMemo(() => {
    return recommendations.filter(rec => rec.impact.confidence >= confidenceThreshold);
  }, [recommendations, confidenceThreshold]);

  const getModelStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'training': return 'text-blue-600 bg-blue-100';
      case 'inactive': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-600 bg-red-100 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-100 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-100 border-yellow-200';
      case 'low': return 'text-blue-600 bg-blue-100 border-blue-200';
      default: return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now.getTime() - time.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white">
        <div className="px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold flex items-center space-x-3">
                <Gem className="w-8 h-8" />
                <span>Predictive Analytics</span>
              </h1>
              <p className="text-purple-100 mt-2 text-lg">
                AI-powered forecasting, anomaly detection, and intelligent recommendations
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Time Horizon */}
              <div className="bg-white/20 rounded-lg p-1">
                <select
                  value={selectedTimeHorizon}
                  onChange={(e) => setSelectedTimeHorizon(e.target.value as any)}
                  className="bg-transparent text-white px-3 py-1 rounded text-sm font-medium"
                >
                  <option value="1w" className="text-gray-900">1 Week</option>
                  <option value="1m" className="text-gray-900">1 Month</option>
                  <option value="3m" className="text-gray-900">3 Months</option>
                  <option value="6m" className="text-gray-900">6 Months</option>
                  <option value="1y" className="text-gray-900">1 Year</option>
                </select>
              </div>

              {/* Confidence Threshold */}
              <div className="bg-white/20 rounded-lg px-3 py-2">
                <label className="text-sm font-medium block mb-1">Confidence ≥ {confidenceThreshold}%</label>
                <input
                  type="range"
                  min="50"
                  max="95"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                  className="w-20"
                />
              </div>

              {/* Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`p-2 rounded-lg transition-colors ${
                    autoRefresh ? 'bg-white/20 text-white' : 'bg-white/10 text-purple-200'
                  }`}
                  title="Toggle auto-refresh"
                >
                  <RefreshCw className="w-5 h-5" />
                </button>
                
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className={`p-2 rounded-lg transition-colors ${
                    showAdvanced ? 'bg-white/20 text-white' : 'bg-white/10 text-purple-200'
                  }`}
                  title="Toggle advanced view"
                >
                  <Settings className="w-5 h-5" />
                </button>
                
                <button className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors">
                  <Download className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Model Status Bar */}
          <div className="mt-6 grid grid-cols-5 gap-4">
            {models.map(model => (
              <div key={model.id} className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-sm truncate">{model.name}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs ${getModelStatusColor(model.status)}`}>
                    {model.status}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-purple-100">Accuracy:</span>
                  <span className="font-medium">{model.accuracy.toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-purple-100">Confidence:</span>
                  <span className="font-medium">{model.confidence}%</span>
                </div>
              </div>
            ))}
          </div>

          {/* View Tabs */}
          <div className="flex items-center space-x-4 mt-6">
            {[
              { key: 'forecasts', label: 'Forecasts', icon: TrendingUp },
              { key: 'recommendations', label: 'AI Recommendations', icon: Lightbulb },
              { key: 'anomalies', label: 'Anomaly Detection', icon: AlertTriangle },
              { key: 'scenarios', label: 'Scenario Analysis', icon: GitBranch }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setSelectedView(key as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedView === key 
                    ? 'bg-white text-purple-600' 
                    : 'text-purple-100 hover:text-white hover:bg-white/20'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {selectedView === 'forecasts' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {forecasts.map((forecast, index) => (
              <div key={index} className="bg-white rounded-xl border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{forecast.metric}</h3>
                    <div className="flex items-center space-x-4 mt-1">
                      <span className="text-2xl font-bold text-blue-600">
                        {forecast.currentValue.toFixed(forecast.metric.includes('Rate') ? 1 : 2)}
                        {forecast.metric.includes('Rate') && '%'}
                      </span>
                      <div className={`flex items-center space-x-1 ${
                        forecast.trend === 'increasing' ? 'text-green-600' :
                        forecast.trend === 'decreasing' ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {forecast.trend === 'increasing' ? <TrendingUp className="w-4 h-4" /> :
                         forecast.trend === 'decreasing' ? <TrendingDown className="w-4 h-4" /> :
                         <Activity className="w-4 h-4" />}
                        <span className="text-sm capitalize">{forecast.trend}</span>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center space-x-1 text-gray-600">
                      <Brain className="w-4 h-4" />
                      <span className="text-sm">AI Forecast</span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {selectedTimeHorizon} horizon
                    </span>
                  </div>
                </div>

                {/* Chart Placeholder */}
                <div className="h-32 bg-gray-50 rounded-lg mb-4 flex items-center justify-center">
                  <div className="text-center text-gray-500">
                    <LineChart className="w-8 h-8 mx-auto mb-2" />
                    <p className="text-sm">Forecast Chart</p>
                  </div>
                </div>

                {/* Seasonality Info */}
                {forecast.seasonality.detected && (
                  <div className="bg-blue-50 rounded-lg p-3 mb-4">
                    <div className="flex items-center space-x-2 mb-1">
                      <Calendar className="w-4 h-4 text-blue-600" />
                      <span className="text-sm font-medium text-blue-900">Seasonality Detected</span>
                    </div>
                    <p className="text-sm text-blue-800">{forecast.seasonality.pattern}</p>
                    <p className="text-xs text-blue-600 mt-1">
                      Impact: {forecast.seasonality.impact}% variance
                    </p>
                  </div>
                )}

                {/* Prediction Summary */}
                <div className="space-y-2">
                  {forecast.predictions.slice(-3).map((pred, idx) => {
                    const dateObj = new Date(pred.date);
                    const isNearTerm = idx === 0;
                    return (
                      <div key={idx} className={`flex items-center justify-between p-2 rounded ${
                        isNearTerm ? 'bg-purple-50' : 'bg-gray-50'
                      }`}>
                        <span className="text-sm text-gray-700">
                          {dateObj.toLocaleDateString()}
                        </span>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium">
                            {pred.value.toFixed(forecast.metric.includes('Rate') ? 1 : 2)}
                            {forecast.metric.includes('Rate') && '%'}
                          </span>
                          <div className="flex items-center space-x-1">
                            <Brain className="w-3 h-3 text-purple-600" />
                            <span className="text-xs text-purple-600">{pred.confidence.toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'recommendations' && (
          <div className="space-y-6">
            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Lightbulb className="w-5 h-5 text-yellow-600" />
                  <span className="font-medium text-gray-900">{filteredRecommendations.length}</span>
                </div>
                <p className="text-sm text-gray-600">Active Recommendations</p>
              </div>
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Star className="w-5 h-5 text-purple-600" />
                  <span className="font-medium text-gray-900">
                    {filteredRecommendations.filter(r => r.priority === 'high').length}
                  </span>
                </div>
                <p className="text-sm text-gray-600">High Priority</p>
              </div>
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  <span className="font-medium text-gray-900">
                    {filteredRecommendations.reduce((sum, r) => sum + r.impact.expectedImprovement, 0).toFixed(1)}%
                  </span>
                </div>
                <p className="text-sm text-gray-600">Total Potential Gain</p>
              </div>
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Brain className="w-5 h-5 text-blue-600" />
                  <span className="font-medium text-gray-900">
                    {(filteredRecommendations.reduce((sum, r) => sum + r.impact.confidence, 0) / filteredRecommendations.length).toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-gray-600">Avg Confidence</p>
              </div>
            </div>

            {/* Recommendations List */}
            {filteredRecommendations.map(rec => (
              <div key={rec.id} className={`bg-white rounded-xl border-2 p-6 ${getPriorityColor(rec.priority)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-lg ${getPriorityColor(rec.priority)}`}>
                      <Rocket className="w-6 h-6" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">{rec.title}</h3>
                      <div className="flex items-center space-x-4 mb-3">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getPriorityColor(rec.priority)}`}>
                          {rec.priority} Priority
                        </span>
                        <span className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm font-medium capitalize">
                          {rec.category.replace('_', ' ')}
                        </span>
                        <div className="flex items-center space-x-1">
                          <Brain className="w-4 h-4 text-purple-600" />
                          <span className="text-sm text-purple-600">{rec.impact.confidence}% confidence</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-green-600">
                      +{rec.impact.expectedImprovement.toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">{rec.impact.metric}</div>
                  </div>
                </div>

                {/* AI Reasoning */}
                <div className="bg-blue-50 rounded-lg p-4 mb-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Brain className="w-5 h-5 text-blue-600" />
                    <span className="font-medium text-blue-900">AI Analysis</span>
                  </div>
                  <p className="text-blue-800 text-sm">{rec.aiReasoning}</p>
                </div>

                {/* Implementation Details */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Implementation Steps</h4>
                    <ol className="space-y-2">
                      {rec.implementation.steps.map((step, index) => (
                        <li key={index} className="flex items-start space-x-3">
                          <span className="flex-shrink-0 w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-sm font-medium">
                            {index + 1}
                          </span>
                          <span className="text-sm text-gray-700">{step}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h5 className="font-medium text-gray-900 mb-2">Required Resources</h5>
                      <div className="flex flex-wrap gap-2">
                        {rec.implementation.resources.map((resource, index) => (
                          <span key={index} className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                            {resource}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="font-medium text-gray-900 mb-2">Timeline</h5>
                      <p className="text-sm text-gray-600">{rec.impact.timeToRealization}</p>
                    </div>
                    
                    {rec.implementation.risks.length > 0 && (
                      <div>
                        <h5 className="font-medium text-gray-900 mb-2">Potential Risks</h5>
                        <ul className="space-y-1">
                          {rec.implementation.risks.map((risk, index) => (
                            <li key={index} className="text-sm text-gray-600 flex items-start space-x-2">
                              <AlertTriangle className="w-3 h-3 text-yellow-600 mt-0.5 flex-shrink-0" />
                              <span>{risk}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>

                {/* Based On */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-gray-600">Based on: </span>
                      <span className="text-sm font-medium text-gray-900">
                        {rec.basedOn.join(', ')}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <Database className="w-4 h-4" />
                      <span className="capitalize">{rec.learningSource.replace('_', ' ')}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'anomalies' && (
          <div className="space-y-6">
            {/* Anomaly Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                  <span className="font-medium text-gray-900">
                    {anomalies.filter(a => a.severity === 'critical').length}
                  </span>
                </div>
                <p className="text-sm text-gray-600">Critical Anomalies</p>
              </div>
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-orange-600" />
                  <span className="font-medium text-gray-900">
                    {anomalies.filter(a => a.severity === 'high').length}
                  </span>
                </div>
                <p className="text-sm text-gray-600">High Priority</p>
              </div>
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span className="font-medium text-gray-900">
                    {anomalies.filter(a => a.autoResolved).length}
                  </span>
                </div>
                <p className="text-sm text-gray-600">Auto-Resolved</p>
              </div>
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Clock className="w-5 h-5 text-blue-600" />
                  <span className="font-medium text-gray-900">
                    {formatTimeAgo(anomalies[0]?.timestamp)}
                  </span>
                </div>
                <p className="text-sm text-gray-600">Latest Detection</p>
              </div>
            </div>

            {/* Anomalies List */}
            {anomalies.map(anomaly => (
              <div key={anomaly.id} className={`bg-white rounded-xl border-2 p-6 ${
                anomaly.autoResolved ? 'border-green-200' : getSeverityColor(anomaly.severity).replace('text-', 'border-').replace('-600', '-200')
              }`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-lg ${getSeverityColor(anomaly.severity)}`}>
                      {anomaly.autoResolved ? (
                        <CheckCircle className="w-6 h-6" />
                      ) : (
                        <AlertTriangle className="w-6 h-6" />
                      )}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-1">{anomaly.metric}</h3>
                      <p className="text-gray-600 mb-2">{anomaly.description}</p>
                      <div className="flex items-center space-x-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(anomaly.severity)}`}>
                          {anomaly.severity.toUpperCase()}
                        </span>
                        <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs font-medium capitalize">
                          {anomaly.category}
                        </span>
                        <span className="text-xs text-gray-500">
                          {formatTimeAgo(anomaly.timestamp)}
                        </span>
                        {anomaly.autoResolved && (
                          <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                            Auto-Resolved
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-red-600">
                      {anomaly.deviation.toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Deviation</div>
                  </div>
                </div>

                {/* Values Comparison */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-sm text-gray-600 mb-1">Expected Value</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {anomaly.expectedValue.toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-red-50 rounded-lg p-3">
                    <p className="text-sm text-gray-600 mb-1">Actual Value</p>
                    <p className="text-lg font-semibold text-red-600">
                      {anomaly.value.toLocaleString()}
                    </p>
                  </div>
                </div>

                {/* Possible Causes & Actions */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Possible Causes</h4>
                    <ul className="space-y-2">
                      {anomaly.possibleCauses.map((cause, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{cause}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Suggested Actions</h4>
                    <ul className="space-y-2">
                      {anomaly.suggestedActions.map((action, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <ArrowRight className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'scenarios' && (
          <div className="space-y-6">
            {scenarios.map(scenario => (
              <div key={scenario.id} className="bg-white rounded-xl border border-gray-200 p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">{scenario.name}</h3>
                    <p className="text-gray-600 mb-3">{scenario.description}</p>
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-600">Probability:</span>
                        <span className="font-medium text-blue-600">{scenario.probability}%</span>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        scenario.riskLevel === 'low' ? 'bg-green-100 text-green-700' :
                        scenario.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {scenario.riskLevel.toUpperCase()} RISK
                      </span>
                    </div>
                  </div>
                  <div className={`p-3 rounded-lg ${
                    scenario.riskLevel === 'low' ? 'bg-green-100' :
                    scenario.riskLevel === 'medium' ? 'bg-yellow-100' : 'bg-red-100'
                  }`}>
                    <GitBranch className={`w-6 h-6 ${
                      scenario.riskLevel === 'low' ? 'text-green-600' :
                      scenario.riskLevel === 'medium' ? 'text-yellow-600' : 'text-red-600'
                    }`} />
                  </div>
                </div>

                {/* Assumptions */}
                <div className="mb-6">
                  <h4 className="font-medium text-gray-900 mb-3">Scenario Assumptions</h4>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    {scenario.assumptions.map((assumption, index) => (
                      <div key={index} className="bg-blue-50 rounded-lg p-3">
                        <p className="font-medium text-blue-900 text-sm">{assumption.parameter}</p>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className="text-sm text-gray-600">
                            {assumption.baseValue} → {assumption.scenarioValue}
                          </span>
                          <span className={`text-sm font-medium ${
                            assumption.impact > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {assumption.impact > 0 ? '+' : ''}{assumption.impact.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Outcomes */}
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Projected Outcomes</h4>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {scenario.outcomes.map((outcome, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">{outcome.metric}</span>
                          <span className={`text-lg font-bold ${
                            outcome.change > 0 ? 'text-green-600' : 
                            outcome.change < 0 ? 'text-red-600' : 'text-gray-600'
                          }`}>
                            {outcome.change > 0 ? '+' : ''}{outcome.change.toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span>Current: {outcome.baselineValue.toLocaleString()}</span>
                          <span>Projected: {outcome.scenarioValue.toLocaleString()}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictiveAnalytics;