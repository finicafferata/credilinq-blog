import React, { useState, useEffect, useRef } from 'react';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Play,
  Pause,
  RotateCcw,
  Activity,
  Zap,
  Database,
  Globe,
  Settings,
  FileText,
  Users,
  BarChart3,
  Clock,
  TrendingUp,
  Shield,
  Link,
  Terminal,
  Code,
  RefreshCw,
  Download,
  Upload,
  Eye,
  AlertCircle,
  Loader
} from 'lucide-react';

// Types for integration testing
interface TestResult {
  id: string;
  name: string;
  category: 'ui' | 'api' | 'integration' | 'performance' | 'security';
  status: 'pending' | 'running' | 'passed' | 'failed' | 'warning';
  duration: number;
  error?: string;
  details?: string;
  timestamp: number;
  coverage?: number;
  metrics?: {
    responseTime?: number;
    throughput?: number;
    errorRate?: number;
    memoryUsage?: number;
  };
}

interface TestSuite {
  id: string;
  name: string;
  description: string;
  category: 'ui' | 'api' | 'integration' | 'performance' | 'security';
  tests: TestResult[];
  progress: number;
  status: 'idle' | 'running' | 'completed' | 'failed';
  estimatedDuration: number;
  actualDuration: number;
}

interface ValidationReport {
  id: string;
  timestamp: number;
  overallStatus: 'passed' | 'failed' | 'warning';
  totalTests: number;
  passedTests: number;
  failedTests: number;
  warningTests: number;
  coverage: {
    ui: number;
    api: number;
    integration: number;
  };
  performance: {
    avgResponseTime: number;
    throughput: number;
    errorRate: number;
  };
  recommendations: Array<{
    type: 'critical' | 'warning' | 'info';
    component: string;
    issue: string;
    solution: string;
    priority: number;
  }>;
}

// Mock test execution engine
class TestExecutionEngine {
  private onUpdate: (result: TestResult) => void;
  
  constructor(onUpdate: (result: TestResult) => void) {
    this.onUpdate = onUpdate;
  }

  async executeTest(test: TestResult): Promise<TestResult> {
    // Update status to running
    this.onUpdate({ ...test, status: 'running' });
    
    // Simulate test execution
    const duration = Math.random() * 3000 + 500;
    await new Promise(resolve => setTimeout(resolve, duration));
    
    // Determine result based on test category and name
    const success = this.determineTestSuccess(test);
    const updatedTest: TestResult = {
      ...test,
      status: success ? 'passed' : Math.random() > 0.8 ? 'warning' : 'failed',
      duration: Math.round(duration),
      timestamp: Date.now(),
      coverage: Math.random() * 100,
      metrics: {
        responseTime: Math.random() * 1000,
        throughput: Math.random() * 100,
        errorRate: Math.random() * 5,
        memoryUsage: Math.random() * 80
      }
    };

    if (!success) {
      updatedTest.error = this.generateErrorMessage(test);
      updatedTest.details = this.generateErrorDetails(test);
    }

    this.onUpdate(updatedTest);
    return updatedTest;
  }

  private determineTestSuccess(test: TestResult): boolean {
    // Campaign-centric components should have higher success rates
    const campaignComponents = [
      'CampaignOrchestrationDashboard',
      'AgentManagement', 
      'RealTimeMonitoring',
      'PerformanceAnalytics',
      'PredictiveAnalytics'
    ];
    
    const isCampaignComponent = campaignComponents.some(comp => 
      test.name.toLowerCase().includes(comp.toLowerCase())
    );
    
    if (isCampaignComponent) {
      return Math.random() > 0.1; // 90% success rate for campaign components
    }
    
    return Math.random() > 0.2; // 80% success rate for other components
  }

  private generateErrorMessage(test: TestResult): string {
    const errors = [
      'Component failed to render properly',
      'API endpoint returned 500 error',
      'WebSocket connection timeout',
      'Database query exceeded timeout',
      'Authentication token expired',
      'Memory usage exceeded threshold',
      'Network latency too high',
      'Validation error in form submission'
    ];
    return errors[Math.floor(Math.random() * errors.length)];
  }

  private generateErrorDetails(test: TestResult): string {
    return `Error occurred in ${test.category} test "${test.name}". Check logs for detailed stack trace and error context.`;
  }
}

export function IntegrationTesting() {
  const [testSuites, setTestSuites] = useState<TestSuite[]>([]);
  const [selectedSuite, setSelectedSuite] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [currentReport, setCurrentReport] = useState<ValidationReport | null>(null);
  const [selectedView, setSelectedView] = useState<'overview' | 'suites' | 'results' | 'reports'>('overview');
  const [autoRun, setAutoRun] = useState(false);
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [showOnlyFailed, setShowOnlyFailed] = useState(false);
  
  const testEngineRef = useRef<TestExecutionEngine | null>(null);
  const runningTestsRef = useRef<Set<string>>(new Set());

  // Initialize test suites
  useEffect(() => {
    initializeTestSuites();
    testEngineRef.current = new TestExecutionEngine(handleTestUpdate);
  }, []);

  const initializeTestSuites = () => {
    const suites: TestSuite[] = [
      {
        id: 'ui-components',
        name: 'UI Component Testing',
        description: 'Test all React components for rendering, interactions, and state management',
        category: 'ui',
        progress: 0,
        status: 'idle',
        estimatedDuration: 45000,
        actualDuration: 0,
        tests: [
          { id: 'ui-1', name: 'CampaignOrchestrationDashboard Render', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-2', name: 'AgentManagement Grid View', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-3', name: 'RealTimeMonitoring WebSocket', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-4', name: 'PerformanceAnalytics Charts', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-5', name: 'PredictiveAnalytics ML Models', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-6', name: 'WorkflowCanvas Drag & Drop', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-7', name: 'Header Navigation', category: 'ui', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'ui-8', name: 'Responsive Design Mobile', category: 'ui', status: 'pending', duration: 0, timestamp: 0 }
        ]
      },
      {
        id: 'api-endpoints',
        name: 'API Endpoint Testing',
        description: 'Validate all REST API endpoints for campaigns, agents, and analytics',
        category: 'api',
        progress: 0,
        status: 'idle',
        estimatedDuration: 30000,
        actualDuration: 0,
        tests: [
          { id: 'api-1', name: 'GET /api/v2/campaigns', category: 'api', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'api-2', name: 'POST /api/v2/campaigns', category: 'api', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'api-3', name: 'GET /api/v2/agents', category: 'api', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'api-4', name: 'GET /api/v2/analytics/performance', category: 'api', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'api-5', name: 'GET /api/v2/monitoring/metrics', category: 'api', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'api-6', name: 'WebSocket /ws/monitoring', category: 'api', status: 'pending', duration: 0, timestamp: 0 }
        ]
      },
      {
        id: 'integration-flows',
        name: 'Integration Flow Testing',
        description: 'End-to-end testing of complete user workflows and data flows',
        category: 'integration',
        progress: 0,
        status: 'idle',
        estimatedDuration: 60000,
        actualDuration: 0,
        tests: [
          { id: 'int-1', name: 'Campaign Creation to Execution', category: 'integration', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'int-2', name: 'Agent Assignment to Task', category: 'integration', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'int-3', name: 'Real-time Monitoring Updates', category: 'integration', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'int-4', name: 'Analytics Data Pipeline', category: 'integration', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'int-5', name: 'Workflow Builder to Execution', category: 'integration', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'int-6', name: 'Predictive Analytics Training', category: 'integration', status: 'pending', duration: 0, timestamp: 0 }
        ]
      },
      {
        id: 'performance',
        name: 'Performance Testing',
        description: 'Load testing, response times, and resource utilization validation',
        category: 'performance',
        progress: 0,
        status: 'idle',
        estimatedDuration: 90000,
        actualDuration: 0,
        tests: [
          { id: 'perf-1', name: 'Dashboard Load Time < 2s', category: 'performance', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'perf-2', name: 'Concurrent Users (100+)', category: 'performance', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'perf-3', name: 'Memory Usage < 100MB', category: 'performance', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'perf-4', name: 'WebSocket Throughput', category: 'performance', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'perf-5', name: 'Database Query Performance', category: 'performance', status: 'pending', duration: 0, timestamp: 0 }
        ]
      },
      {
        id: 'security',
        name: 'Security Testing',
        description: 'Authentication, authorization, and security vulnerability testing',
        category: 'security',
        progress: 0,
        status: 'idle',
        estimatedDuration: 40000,
        actualDuration: 0,
        tests: [
          { id: 'sec-1', name: 'JWT Token Validation', category: 'security', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'sec-2', name: 'API Rate Limiting', category: 'security', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'sec-3', name: 'XSS Prevention', category: 'security', status: 'pending', duration: 0, timestamp: 0 },
          { id: 'sec-4', name: 'CSRF Protection', category: 'security', status: 'pending', duration: 0, timestamp: 0 }
        ]
      }
    ];

    setTestSuites(suites);
    generateReport(suites);
  };

  const handleTestUpdate = (updatedTest: TestResult) => {
    setTestSuites(prev => prev.map(suite => ({
      ...suite,
      tests: suite.tests.map(test => 
        test.id === updatedTest.id ? updatedTest : test
      )
    })));
  };

  const runAllTests = async () => {
    if (!testEngineRef.current) return;

    setIsRunning(true);
    runningTestsRef.current.clear();

    for (const suite of testSuites) {
      await runTestSuite(suite);
    }

    setIsRunning(false);
    generateReport(testSuites);
  };

  const runTestSuite = async (suite: TestSuite) => {
    if (!testEngineRef.current) return;

    // Update suite status
    setTestSuites(prev => prev.map(s => 
      s.id === suite.id ? { ...s, status: 'running', progress: 0 } : s
    ));

    const startTime = Date.now();
    
    for (let i = 0; i < suite.tests.length; i++) {
      const test = suite.tests[i];
      runningTestsRef.current.add(test.id);
      
      try {
        await testEngineRef.current.executeTest(test);
      } catch (error) {
        console.error(`Test ${test.id} failed:`, error);
      }
      
      runningTestsRef.current.delete(test.id);
      
      // Update progress
      const progress = ((i + 1) / suite.tests.length) * 100;
      setTestSuites(prev => prev.map(s => 
        s.id === suite.id ? { ...s, progress } : s
      ));
    }

    const actualDuration = Date.now() - startTime;
    const hasFailures = suite.tests.some(t => t.status === 'failed');
    
    setTestSuites(prev => prev.map(s => 
      s.id === suite.id ? { 
        ...s, 
        status: hasFailures ? 'failed' : 'completed',
        progress: 100,
        actualDuration
      } : s
    ));
  };

  const generateReport = (suites: TestSuite[]) => {
    const allTests = suites.flatMap(s => s.tests);
    const totalTests = allTests.length;
    const passedTests = allTests.filter(t => t.status === 'passed').length;
    const failedTests = allTests.filter(t => t.status === 'failed').length;
    const warningTests = allTests.filter(t => t.status === 'warning').length;

    const report: ValidationReport = {
      id: `report_${Date.now()}`,
      timestamp: Date.now(),
      overallStatus: failedTests > 0 ? 'failed' : warningTests > 0 ? 'warning' : 'passed',
      totalTests,
      passedTests,
      failedTests,
      warningTests,
      coverage: {
        ui: calculateCoverage(allTests, 'ui'),
        api: calculateCoverage(allTests, 'api'),
        integration: calculateCoverage(allTests, 'integration')
      },
      performance: {
        avgResponseTime: allTests.reduce((sum, t) => sum + (t.metrics?.responseTime || 0), 0) / totalTests,
        throughput: allTests.reduce((sum, t) => sum + (t.metrics?.throughput || 0), 0) / totalTests,
        errorRate: allTests.reduce((sum, t) => sum + (t.metrics?.errorRate || 0), 0) / totalTests
      },
      recommendations: generateRecommendations(allTests)
    };

    setCurrentReport(report);
  };

  const calculateCoverage = (tests: TestResult[], category: string) => {
    const categoryTests = tests.filter(t => t.category === category);
    const passedCategoryTests = categoryTests.filter(t => t.status === 'passed');
    return categoryTests.length > 0 ? (passedCategoryTests.length / categoryTests.length) * 100 : 0;
  };

  const generateRecommendations = (tests: TestResult[]) => {
    const recommendations = [];
    const failedTests = tests.filter(t => t.status === 'failed');
    
    if (failedTests.length > 0) {
      recommendations.push({
        type: 'critical' as const,
        component: 'Test Suite',
        issue: `${failedTests.length} tests are failing`,
        solution: 'Review and fix failing tests before deployment',
        priority: 1
      });
    }

    const slowTests = tests.filter(t => (t.metrics?.responseTime || 0) > 2000);
    if (slowTests.length > 0) {
      recommendations.push({
        type: 'warning' as const,
        component: 'Performance',
        issue: 'Some components have slow response times',
        solution: 'Optimize component rendering and API calls',
        priority: 2
      });
    }

    return recommendations.slice(0, 5);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'running': return <Loader className="w-4 h-4 text-blue-500 animate-spin" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'ui': return <Eye className="w-4 h-4" />;
      case 'api': return <Globe className="w-4 h-4" />;
      case 'integration': return <Link className="w-4 h-4" />;
      case 'performance': return <Zap className="w-4 h-4" />;
      case 'security': return <Shield className="w-4 h-4" />;
      default: return <Code className="w-4 h-4" />;
    }
  };

  const filteredSuites = testSuites.filter(suite => 
    filterCategory === 'all' || suite.category === filterCategory
  );

  const filteredTests = testSuites.flatMap(s => s.tests).filter(test => {
    if (showOnlyFailed && test.status !== 'failed') return false;
    if (filterCategory !== 'all' && test.category !== filterCategory) return false;
    return true;
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl flex items-center justify-center">
                  <Terminal className="w-6 h-6 text-white" />
                </div>
                <span>Integration Testing</span>
              </h1>
              
              {currentReport && (
                <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-gray-100">
                  {getStatusIcon(currentReport.overallStatus)}
                  <span className="text-sm font-medium">
                    {currentReport.passedTests}/{currentReport.totalTests} Passed
                  </span>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-4">
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
              >
                <option value="all">All Categories</option>
                <option value="ui">UI Components</option>
                <option value="api">API Endpoints</option>
                <option value="integration">Integration</option>
                <option value="performance">Performance</option>
                <option value="security">Security</option>
              </select>

              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showOnlyFailed}
                  onChange={(e) => setShowOnlyFailed(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm">Failed Only</span>
              </label>

              <button
                onClick={runAllTests}
                disabled={isRunning}
                className={`px-4 py-2 rounded-lg font-medium flex items-center space-x-2 ${
                  isRunning
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {isRunning ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    <span>Running...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Run All Tests</span>
                  </>
                )}
              </button>

              <button
                onClick={() => initializeTestSuites()}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center space-x-2"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Reset</span>
              </button>
            </div>
          </div>

          {/* View Tabs */}
          <div className="flex items-center space-x-4 mt-4">
            {['overview', 'suites', 'results', 'reports'].map(view => (
              <button
                key={view}
                onClick={() => setSelectedView(view as any)}
                className={`px-4 py-2 rounded-lg capitalize font-medium ${
                  selectedView === view 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {view}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="p-6">
        {selectedView === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
            {/* Test Suites Overview */}
            {filteredSuites.map(suite => (
              <div key={suite.id} className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getCategoryIcon(suite.category)}
                    <div>
                      <h3 className="font-semibold text-gray-900">{suite.name}</h3>
                      <p className="text-sm text-gray-500">{suite.tests.length} tests</p>
                    </div>
                  </div>
                  {getStatusIcon(suite.status)}
                </div>
                
                <p className="text-sm text-gray-600 mb-4">{suite.description}</p>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Progress</span>
                    <span className="font-medium">{Math.round(suite.progress)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${suite.progress}%` }}
                    />
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-gray-100">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">
                      {suite.actualDuration > 0 ? 
                        `${Math.round(suite.actualDuration / 1000)}s` : 
                        `Est. ${Math.round(suite.estimatedDuration / 1000)}s`
                      }
                    </span>
                    <button 
                      onClick={() => setSelectedSuite(suite.id)}
                      className="text-blue-600 hover:text-blue-700 font-medium"
                    >
                      View Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'suites' && (
          <div className="space-y-6">
            {filteredSuites.map(suite => (
              <div key={suite.id} className="bg-white rounded-xl border border-gray-200">
                <div className="p-6 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      {getCategoryIcon(suite.category)}
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">{suite.name}</h3>
                        <p className="text-sm text-gray-600">{suite.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      {getStatusIcon(suite.status)}
                      <button
                        onClick={() => runTestSuite(suite)}
                        disabled={isRunning || suite.status === 'running'}
                        className="px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
                      >
                        Run Suite
                      </button>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${suite.progress}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="divide-y divide-gray-100">
                  {suite.tests.map(test => (
                    <div key={test.id} className="p-4 hover:bg-gray-50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          {getStatusIcon(test.status)}
                          <div>
                            <p className="font-medium text-gray-900">{test.name}</p>
                            {test.error && (
                              <p className="text-sm text-red-600">{test.error}</p>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          {test.duration > 0 && (
                            <p className="text-sm text-gray-500">{test.duration}ms</p>
                          )}
                          {test.coverage && (
                            <p className="text-sm text-gray-500">{Math.round(test.coverage)}% coverage</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedView === 'results' && (
          <div className="bg-white rounded-xl border border-gray-200">
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold">Test Results</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Status</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Test Name</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Category</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Duration</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Coverage</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Response Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {filteredTests.map(test => (
                    <tr key={test.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3">{getStatusIcon(test.status)}</td>
                      <td className="px-4 py-3 font-medium text-gray-900">{test.name}</td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center space-x-1">
                          {getCategoryIcon(test.category)}
                          <span className="capitalize text-sm text-gray-600">{test.category}</span>
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {test.duration > 0 ? `${test.duration}ms` : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {test.coverage ? `${Math.round(test.coverage)}%` : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {test.metrics?.responseTime ? `${Math.round(test.metrics.responseTime)}ms` : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {selectedView === 'reports' && currentReport && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Overall Status</p>
                    <p className="text-2xl font-bold text-gray-900 capitalize">
                      {currentReport.overallStatus}
                    </p>
                  </div>
                  {getStatusIcon(currentReport.overallStatus)}
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Tests Passed</p>
                    <p className="text-2xl font-bold text-green-600">
                      {currentReport.passedTests}/{currentReport.totalTests}
                    </p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-green-500" />
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Avg Response</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {Math.round(currentReport.performance.avgResponseTime)}ms
                    </p>
                  </div>
                  <Zap className="w-8 h-8 text-blue-500" />
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Error Rate</p>
                    <p className="text-2xl font-bold text-red-600">
                      {currentReport.performance.errorRate.toFixed(2)}%
                    </p>
                  </div>
                  <AlertCircle className="w-8 h-8 text-red-500" />
                </div>
              </div>
            </div>

            {/* Coverage Report */}
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Test Coverage</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {Object.entries(currentReport.coverage).map(([category, coverage]) => (
                  <div key={category} className="space-y-2">
                    <div className="flex justify-between">
                      <span className="capitalize text-gray-600">{category}</span>
                      <span className="font-medium">{Math.round(coverage)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-500 ${
                          coverage >= 80 ? 'bg-green-500' : 
                          coverage >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${coverage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            {currentReport.recommendations.length > 0 && (
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
                <div className="space-y-3">
                  {currentReport.recommendations.map((rec, index) => (
                    <div key={index} className={`p-4 rounded-lg border-l-4 ${
                      rec.type === 'critical' ? 'bg-red-50 border-red-400' :
                      rec.type === 'warning' ? 'bg-yellow-50 border-yellow-400' :
                      'bg-blue-50 border-blue-400'
                    }`}>
                      <div className="flex items-start space-x-3">
                        {rec.type === 'critical' ? 
                          <XCircle className="w-5 h-5 text-red-500 mt-0.5" /> :
                          rec.type === 'warning' ?
                          <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" /> :
                          <AlertCircle className="w-5 h-5 text-blue-500 mt-0.5" />
                        }
                        <div className="flex-1">
                          <p className="font-medium text-gray-900">{rec.component}: {rec.issue}</p>
                          <p className="text-sm text-gray-600 mt-1">{rec.solution}</p>
                        </div>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          rec.priority === 1 ? 'bg-red-100 text-red-700' :
                          rec.priority === 2 ? 'bg-yellow-100 text-yellow-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          P{rec.priority}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default IntegrationTesting;