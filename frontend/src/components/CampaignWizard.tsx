import React, { useState, useEffect } from 'react';
import { campaignApi, blogApi } from '../lib/api';
import type { BlogSummary } from '../lib/api';
import { AppError } from '../lib/errors';

interface CampaignWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onCampaignCreated: () => void;
}

interface WizardData {
  // Step 1: Basic Info
  blog_id: string;
  campaign_name: string;
  description: string;
  
  // Step 2: Strategy
  strategy_type: string;
  priority: string;
  target_audience: string;
  
  // Step 3: Channels & Timeline
  distribution_channels: string[];
  timeline_weeks: number;
  scheduled_start: string;
  deadline: string;
  
  // Step 4: Goals & Budget
  success_metrics: {
    impressions: number;
    engagement_rate: number;
    conversions: number;
    leads: number;
  };
  budget_allocation: {
    content_creation: number;
    promotion: number;
    analytics: number;
  };
}

const CampaignWizard: React.FC<CampaignWizardProps> = ({ isOpen, onClose, onCampaignCreated }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blogs, setBlogs] = useState<BlogSummary[]>([]);
  const [wizardData, setWizardData] = useState<WizardData>({
    blog_id: '',
    campaign_name: '',
    description: '',
    strategy_type: 'thought_leadership',
    priority: 'medium',
    target_audience: '',
    distribution_channels: ['linkedin'],
    timeline_weeks: 2,
    scheduled_start: '',
    deadline: '',
    success_metrics: {
      impressions: 5000,
      engagement_rate: 0.05,
      conversions: 50,
      leads: 25
    },
    budget_allocation: {
      content_creation: 60,
      promotion: 25,
      analytics: 15
    }
  });

  useEffect(() => {
    if (isOpen) {
      loadBlogs();
      // Reset form when opening
      setCurrentStep(1);
      setError(null);
    }
  }, [isOpen]);

  const loadBlogs = async () => {
    try {
      const data = await blogApi.list();
      const eligibleBlogs = data.filter(blog => 
        ['edited', 'completed', 'published'].includes(blog.status.toLowerCase())
      );
      setBlogs(eligibleBlogs);
    } catch (err) {
      console.error('Error loading blogs:', err);
    }
  };

  const updateWizardData = (updates: Partial<WizardData>) => {
    setWizardData(prev => ({ ...prev, ...updates }));
  };

  const nextStep = () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError(null);

      // Create campaign with wizard data
      const response = await campaignApi.create({
        blog_id: wizardData.blog_id,
        campaign_name: wizardData.campaign_name,
        company_context: wizardData.description,
        content_type: 'blog',
        template_config: {
          strategy_type: wizardData.strategy_type,
          priority: wizardData.priority,
          target_audience: wizardData.target_audience,
          distribution_channels: wizardData.distribution_channels,
          timeline_weeks: wizardData.timeline_weeks,
          success_metrics: wizardData.success_metrics,
          budget_allocation: wizardData.budget_allocation
        }
      });

      onCampaignCreated();
      onClose();
    } catch (err) {
      const error = err as AppError;
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const isStepValid = (step: number) => {
    switch (step) {
      case 1:
        return wizardData.blog_id && wizardData.campaign_name && wizardData.description;
      case 2:
        return wizardData.strategy_type && wizardData.target_audience;
      case 3:
        return wizardData.distribution_channels.length > 0 && wizardData.timeline_weeks > 0;
      case 4:
        return wizardData.success_metrics.impressions > 0;
      default:
        return true;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Create AI-Powered Campaign</h2>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          {/* Progress Bar */}
          <div className="mt-4">
            <div className="flex justify-between text-sm mb-2">
              <span>Step {currentStep} of 4</span>
              <span>{Math.round((currentStep / 4) * 100)}% Complete</span>
            </div>
            <div className="w-full bg-blue-800 rounded-full h-2">
              <div 
                className="bg-white h-2 rounded-full transition-all duration-300"
                style={{ width: `${(currentStep / 4) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {error && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <div className="mt-2 text-sm text-red-700">{error}</div>
                </div>
              </div>
            </div>
          )}

          {/* Step 1: Basic Information */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">Campaign Foundation</h3>
                <p className="text-gray-600 mt-2">Let's start with the basics of your campaign</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select Blog Post
                    </label>
                    <select
                      value={wizardData.blog_id}
                      onChange={(e) => {
                        const selectedBlog = blogs.find(blog => blog.id === e.target.value);
                        updateWizardData({
                          blog_id: e.target.value,
                          campaign_name: selectedBlog ? `Campaign: ${selectedBlog.title}` : '',
                          description: selectedBlog ? `Marketing campaign for: ${selectedBlog.title}` : ''
                        });
                      }}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Choose a blog post...</option>
                      {blogs.map((blog) => (
                        <option key={blog.id} value={blog.id}>
                          {blog.title} ({blog.status})
                        </option>
                      ))}
                    </select>
                    {blogs.length === 0 && (
                      <p className="text-sm text-gray-500 mt-1">
                        No eligible blogs found. Create and publish a blog post first.
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Name
                    </label>
                    <input
                      type="text"
                      value={wizardData.campaign_name}
                      onChange={(e) => updateWizardData({ campaign_name: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      placeholder="Enter a memorable campaign name..."
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Campaign Description
                  </label>
                  <textarea
                    value={wizardData.description}
                    onChange={(e) => updateWizardData({ description: e.target.value })}
                    rows={6}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Describe your campaign goals, target outcome, and key context..."
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Strategy & Audience */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">Strategy & Audience</h3>
                <p className="text-gray-600 mt-2">Define your campaign strategy and target audience</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Strategy Type
                    </label>
                    <select
                      value={wizardData.strategy_type}
                      onChange={(e) => updateWizardData({ strategy_type: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="thought_leadership">Thought Leadership</option>
                      <option value="product_marketing">Product Marketing</option>
                      <option value="brand_awareness">Brand Awareness</option>
                      <option value="lead_generation">Lead Generation</option>
                      <option value="customer_retention">Customer Retention</option>
                      <option value="market_education">Market Education</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Priority Level
                    </label>
                    <select
                      value={wizardData.priority}
                      onChange={(e) => updateWizardData({ priority: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="low">Low Priority</option>
                      <option value="medium">Medium Priority</option>
                      <option value="high">High Priority</option>
                      <option value="critical">Critical Priority</option>
                      <option value="urgent">Urgent Priority</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Target Audience
                  </label>
                  <textarea
                    value={wizardData.target_audience}
                    onChange={(e) => updateWizardData({ target_audience: e.target.value })}
                    rows={6}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Describe your ideal audience: their roles, pain points, interests, and demographics..."
                  />
                </div>
              </div>

              {/* Strategy Preview */}
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <h4 className="font-medium text-blue-900 mb-2">🎯 Strategy Preview</h4>
                <p className="text-blue-800 text-sm">
                  {wizardData.strategy_type === 'thought_leadership' && 
                    "Position your brand as an industry authority through expert insights and analysis."}
                  {wizardData.strategy_type === 'product_marketing' && 
                    "Drive product awareness and adoption through strategic messaging and demonstrations."}
                  {wizardData.strategy_type === 'brand_awareness' && 
                    "Increase brand visibility and recognition across your target market."}
                  {wizardData.strategy_type === 'lead_generation' && 
                    "Generate qualified leads through compelling content and strategic calls-to-action."}
                  {wizardData.strategy_type === 'customer_retention' && 
                    "Strengthen relationships with existing customers through valuable content."}
                  {wizardData.strategy_type === 'market_education' && 
                    "Educate your market about industry trends and best practices."}
                </p>
              </div>
            </div>
          )}

          {/* Step 3: Channels & Timeline */}
          {currentStep === 3 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">Channels & Timeline</h3>
                <p className="text-gray-600 mt-2">Choose your distribution channels and set your timeline</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Distribution Channels
                  </label>
                  <div className="space-y-3">
                    {[
                      { value: 'linkedin', label: 'LinkedIn', desc: 'Professional networking and B2B reach' },
                      { value: 'twitter', label: 'Twitter/X', desc: 'Real-time engagement and viral potential' },
                      { value: 'email', label: 'Email Marketing', desc: 'Direct audience communication' },
                      { value: 'website', label: 'Website/Blog', desc: 'SEO benefits and owned media' },
                      { value: 'facebook', label: 'Facebook', desc: 'Broader social media reach' },
                      { value: 'youtube', label: 'YouTube', desc: 'Video content distribution' }
                    ].map((channel) => (
                      <label key={channel.value} className="flex items-start space-x-3 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={wizardData.distribution_channels.includes(channel.value)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              updateWizardData({
                                distribution_channels: [...wizardData.distribution_channels, channel.value]
                              });
                            } else {
                              updateWizardData({
                                distribution_channels: wizardData.distribution_channels.filter(c => c !== channel.value)
                              });
                            }
                          }}
                          className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <div>
                          <div className="font-medium text-gray-900">{channel.label}</div>
                          <div className="text-sm text-gray-600">{channel.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Duration (weeks)
                    </label>
                    <select
                      value={wizardData.timeline_weeks}
                      onChange={(e) => updateWizardData({ timeline_weeks: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value={1}>1 Week - Quick Blast</option>
                      <option value={2}>2 Weeks - Standard Campaign</option>
                      <option value={3}>3 Weeks - Extended Reach</option>
                      <option value={4}>4 Weeks - Comprehensive Campaign</option>
                      <option value={6}>6 Weeks - Long-term Strategy</option>
                      <option value={8}>8 Weeks - Maximum Impact</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Scheduled Start Date
                    </label>
                    <input
                      type="date"
                      value={wizardData.scheduled_start}
                      onChange={(e) => updateWizardData({ scheduled_start: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Deadline
                    </label>
                    <input
                      type="date"
                      value={wizardData.deadline}
                      onChange={(e) => updateWizardData({ deadline: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  {/* Timeline Preview */}
                  <div className="bg-green-50 rounded-lg p-4 border border-green-200 mt-4">
                    <h4 className="font-medium text-green-900 mb-2">📅 Timeline Preview</h4>
                    <p className="text-green-800 text-sm">
                      {wizardData.timeline_weeks} week campaign across {wizardData.distribution_channels.length} channels
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Goals & Budget */}
          {currentStep === 4 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">Goals & Budget</h3>
                <p className="text-gray-600 mt-2">Set your success metrics and budget allocation</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-4">Success Metrics</h4>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Target Impressions
                      </label>
                      <input
                        type="number"
                        value={wizardData.success_metrics.impressions}
                        onChange={(e) => updateWizardData({
                          success_metrics: {
                            ...wizardData.success_metrics,
                            impressions: parseInt(e.target.value) || 0
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Engagement Rate (%)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={wizardData.success_metrics.engagement_rate * 100}
                        onChange={(e) => updateWizardData({
                          success_metrics: {
                            ...wizardData.success_metrics,
                            engagement_rate: parseFloat(e.target.value) / 100 || 0
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Target Conversions
                      </label>
                      <input
                        type="number"
                        value={wizardData.success_metrics.conversions}
                        onChange={(e) => updateWizardData({
                          success_metrics: {
                            ...wizardData.success_metrics,
                            conversions: parseInt(e.target.value) || 0
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Lead Generation Goal
                      </label>
                      <input
                        type="number"
                        value={wizardData.success_metrics.leads}
                        onChange={(e) => updateWizardData({
                          success_metrics: {
                            ...wizardData.success_metrics,
                            leads: parseInt(e.target.value) || 0
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 mb-4">Budget Allocation (%)</h4>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Content Creation ({wizardData.budget_allocation.content_creation}%)
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={wizardData.budget_allocation.content_creation}
                        onChange={(e) => {
                          const value = parseInt(e.target.value);
                          const remaining = 100 - value;
                          const promotionRatio = wizardData.budget_allocation.promotion / (wizardData.budget_allocation.promotion + wizardData.budget_allocation.analytics);
                          updateWizardData({
                            budget_allocation: {
                              content_creation: value,
                              promotion: Math.round(remaining * promotionRatio),
                              analytics: Math.round(remaining * (1 - promotionRatio))
                            }
                          });
                        }}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Promotion & Advertising ({wizardData.budget_allocation.promotion}%)
                      </label>
                      <input
                        type="range"
                        min="0"
                        max={100 - wizardData.budget_allocation.content_creation}
                        value={wizardData.budget_allocation.promotion}
                        onChange={(e) => {
                          const value = parseInt(e.target.value);
                          updateWizardData({
                            budget_allocation: {
                              ...wizardData.budget_allocation,
                              promotion: value,
                              analytics: 100 - wizardData.budget_allocation.content_creation - value
                            }
                          });
                        }}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Analytics & Optimization ({wizardData.budget_allocation.analytics}%)
                      </label>
                      <input
                        type="range"
                        min="0"
                        max={100 - wizardData.budget_allocation.content_creation - wizardData.budget_allocation.promotion}
                        value={wizardData.budget_allocation.analytics}
                        onChange={(e) => {
                          const value = parseInt(e.target.value);
                          updateWizardData({
                            budget_allocation: {
                              ...wizardData.budget_allocation,
                              analytics: value,
                              promotion: 100 - wizardData.budget_allocation.content_creation - value
                            }
                          });
                        }}
                        className="w-full"
                      />
                    </div>
                  </div>

                  {/* Budget Visualization */}
                  <div className="mt-6 bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <h4 className="font-medium text-purple-900 mb-3">💰 Budget Breakdown</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Content Creation</span>
                        <span className="font-medium">{wizardData.budget_allocation.content_creation}%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Promotion</span>
                        <span className="font-medium">{wizardData.budget_allocation.promotion}%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Analytics</span>
                        <span className="font-medium">{wizardData.budget_allocation.analytics}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-between items-center">
          <div>
            {currentStep > 1 && (
              <button
                onClick={prevStep}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
              >
                ← Previous
              </button>
            )}
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              Cancel
            </button>
            
            {currentStep < 4 ? (
              <button
                onClick={nextStep}
                disabled={!isStepValid(currentStep)}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Next Step →
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={loading || !isStepValid(currentStep)}
                className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-md hover:from-blue-700 hover:to-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all duration-200 font-medium"
              >
                {loading ? 'Creating Campaign...' : '🚀 Create Campaign'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CampaignWizard;