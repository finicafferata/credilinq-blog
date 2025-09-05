import React, { useState } from 'react';
import {
  CheckCircleIcon,
  ClipboardDocumentCheckIcon,
  UserGroupIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import ContentReviewDashboard from '../components/ContentReviewDashboard';
import HumanReviewInterface from '../components/HumanReviewInterface';
import ReviewWorkflowProgress from '../components/ReviewWorkflowProgress';

type TabType = 'dashboard' | 'review' | 'progress' | 'settings';

const ContentReviewApproval: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);

  const tabs = [
    { 
      id: 'dashboard' as TabType, 
      name: '8-Stage Dashboard', 
      icon: ChartBarIcon,
      description: 'Overview of all review workflows'
    },
    { 
      id: 'review' as TabType, 
      name: 'Human Review', 
      icon: UserGroupIcon,
      description: 'Pending human reviews and decisions'
    },
    { 
      id: 'progress' as TabType, 
      name: 'Workflow Progress', 
      icon: ClipboardDocumentCheckIcon,
      description: 'Track specific workflow progress'
    },
    { 
      id: 'settings' as TabType, 
      name: 'Settings', 
      icon: Cog6ToothIcon,
      description: 'Configure review workflow settings'
    }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <ContentReviewDashboard />;
      
      case 'review':
        return (
          <HumanReviewInterface
            showPendingList={true}
            onReviewSubmitted={(workflowId, decision) => {
              console.log('Review submitted:', workflowId, decision);
            }}
          />
        );
      
      case 'progress':
        return (
          <div className="space-y-6">
            {!selectedWorkflowId ? (
              <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
                <ClipboardDocumentCheckIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Track Workflow Progress
                </h3>
                <p className="text-gray-600 mb-4">
                  Enter a workflow ID to view detailed 8-stage progress
                </p>
                <div className="max-w-md mx-auto">
                  <div className="flex gap-3">
                    <input
                      type="text"
                      placeholder="Enter workflow execution ID..."
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          const target = e.target as HTMLInputElement;
                          if (target.value.trim()) {
                            setSelectedWorkflowId(target.value.trim());
                          }
                        }
                      }}
                    />
                    <button
                      onClick={() => {
                        const input = document.querySelector('input[placeholder="Enter workflow execution ID..."]') as HTMLInputElement;
                        if (input?.value.trim()) {
                          setSelectedWorkflowId(input.value.trim());
                        }
                      }}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                    >
                      <EyeIcon className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-gray-900">
                    Workflow Progress
                  </h2>
                  <button
                    onClick={() => setSelectedWorkflowId(null)}
                    className="text-sm text-gray-600 hover:text-gray-800"
                  >
                    ← Back to search
                  </button>
                </div>
                <ReviewWorkflowProgress
                  workflowId={selectedWorkflowId}
                  showDetails={true}
                  onStatusChange={(status) => {
                    console.log('Workflow status updated:', status);
                  }}
                />
              </div>
            )}
          </div>
        );
      
      case 'settings':
        return (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                8-Stage Review Workflow Settings
              </h3>
              
              <div className="space-y-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Auto-Approval Thresholds</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { name: 'Content Quality', key: 'content_quality', default: 8.0 },
                      { name: 'Editorial Review', key: 'editorial_review', default: 8.0 },
                      { name: 'Brand Check', key: 'brand_check', default: 8.5 },
                      { name: 'SEO Analysis', key: 'seo_analysis', default: 7.5 },
                      { name: 'GEO Analysis', key: 'geo_analysis', default: 8.0 },
                      { name: 'Visual Review', key: 'visual_review', default: 7.5 },
                      { name: 'Social Media', key: 'social_media', default: 8.0 },
                      { name: 'Final Approval', key: 'final_approval', default: 8.5 }
                    ].map((stage) => (
                      <div key={stage.key} className="space-y-2">
                        <label className="block text-sm font-medium text-gray-700">
                          {stage.name}
                        </label>
                        <input
                          type="number"
                          min="0"
                          max="10"
                          step="0.1"
                          defaultValue={stage.default}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Workflow Configuration</h4>
                  <div className="space-y-4">
                    <label className="flex items-center">
                      <input type="checkbox" defaultChecked className="mr-2" />
                      <span className="text-sm text-gray-700">Require human approval for all stages below threshold</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" />
                      <span className="text-sm text-gray-700">Allow parallel review processing</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" defaultChecked className="mr-2" />
                      <span className="text-sm text-gray-700">Send notifications for human review assignments</span>
                    </label>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Review Timeouts</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Human Review Timeout (hours)
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="168"
                        defaultValue={48}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Escalation Timeout (hours)
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="168"
                        defaultValue={72}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end">
                <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  Save Settings
                </button>
              </div>
            </div>

            <div className="bg-blue-50 rounded-lg p-6">
              <h4 className="font-medium text-blue-900 mb-2">8-Stage Review Process</h4>
              <p className="text-sm text-blue-800 mb-3">
                Your content goes through 8 comprehensive review stages:
              </p>
              <div className="text-sm text-blue-700 space-y-1">
                <div>📝 <strong>Content Quality</strong> - Grammar, readability, structure</div>
                <div>✏️ <strong>Editorial Review</strong> - Style, tone, coherence</div>
                <div>🏢 <strong>Brand Check</strong> - Brand consistency and messaging</div>
                <div>🔍 <strong>SEO Analysis</strong> - Search optimization and keywords</div>
                <div>🌍 <strong>GEO Analysis</strong> - Geographic targeting and relevance</div>
                <div>🎨 <strong>Visual Review</strong> - Image and visual content alignment</div>
                <div>📱 <strong>Social Media Review</strong> - Platform optimization</div>
                <div>✅ <strong>Final Approval</strong> - Publication readiness validation</div>
              </div>
            </div>
          </div>
        );
      
      default:
        return <ContentReviewDashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <CheckCircleIcon className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Content Review & Approval
                </h1>
                <p className="text-sm text-gray-600">
                  8-Stage AI-Powered Review Workflow System
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="h-5 w-5" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>
          
          {/* Tab description */}
          <div className="mt-2">
            <p className="text-sm text-gray-600">
              {tabs.find(tab => tab.id === activeTab)?.description}
            </p>
          </div>
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
};

export default ContentReviewApproval;