import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { campaignApi } from '../lib/api';
import type { CampaignDetail } from '../lib/api';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';

interface WizardStep {
  id: string;
  title: string;
  description: string;
  completed: boolean;
}

export function CampaignWizard() {
  const { campaignId } = useParams<{ campaignId: string }>();
  const navigate = useNavigate();
  const [campaign, setCampaign] = useState<CampaignDetail | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Wizard steps
  const [steps] = useState<WizardStep[]>([
    {
      id: 'strategy',
      title: 'Campaign Strategy',
      description: 'Define your target audience and key messages',
      completed: false
    },
    {
      id: 'channels',
      title: 'Distribution Channels',
      description: 'Select platforms and content types',
      completed: false
    },
    {
      id: 'content',
      title: 'Content Adaptation',
      description: 'Adapt content for each platform',
      completed: false
    },
    {
      id: 'schedule',
      title: 'Schedule & Timing',
      description: 'Set posting schedule and timing',
      completed: false
    },
    {
      id: 'review',
      title: 'Review & Launch',
      description: 'Review campaign and launch',
      completed: false
    }
  ]);

  // Form data
  const [formData, setFormData] = useState({
    strategy: {
      target_audience: '',
      key_messages: [''],
      goals: ''
    },
    channels: {
      selected: [] as string[],
      content_types: {} as Record<string, string[]>
    },
    content: {
      adaptations: {} as Record<string, string>
    },
    schedule: {
      start_date: '',
      frequency: 'daily',
      best_times: [] as string[]
    }
  });

  useEffect(() => {
    if (campaignId) {
      loadCampaign();
    }
  }, [campaignId]);

  const loadCampaign = async () => {
    try {
      setLoading(true);
      const data = await campaignApi.get(campaignId!);
      setCampaign(data);
    } catch (err) {
      setError(err instanceof AppError ? err.message : 'Failed to load campaign');
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = async () => {
    try {
      setLoading(true);
      // Here you would save the campaign configuration
      await campaignApi.schedule(campaignId!);
      showSuccessNotification('Campaign configured successfully!');
      navigate(`/campaigns/${campaignId}`);
    } catch (err) {
      showErrorNotification(err instanceof AppError ? err : new AppError('Failed to complete campaign setup'));
    } finally {
      setLoading(false);
    }
  };

  const renderStrategyStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-2">Target Audience</h3>
        <textarea
          value={formData.strategy.target_audience}
          onChange={(e) => setFormData({
            ...formData,
            strategy: { ...formData.strategy, target_audience: e.target.value }
          })}
          className="w-full p-3 border rounded-lg"
          placeholder="Describe your target audience..."
          rows={3}
        />
      </div>
      
      <div>
        <h3 className="text-lg font-semibold mb-2">Key Messages</h3>
        {formData.strategy.key_messages.map((message, index) => (
          <div key={index} className="flex gap-2 mb-2">
            <input
              type="text"
              value={message}
              onChange={(e) => {
                const newMessages = [...formData.strategy.key_messages];
                newMessages[index] = e.target.value;
                setFormData({
                  ...formData,
                  strategy: { ...formData.strategy, key_messages: newMessages }
                });
              }}
              className="flex-1 p-3 border rounded-lg"
              placeholder={`Key message ${index + 1}`}
            />
            {formData.strategy.key_messages.length > 1 && (
              <button
                onClick={() => {
                  const newMessages = formData.strategy.key_messages.filter((_, i) => i !== index);
                  setFormData({
                    ...formData,
                    strategy: { ...formData.strategy, key_messages: newMessages }
                  });
                }}
                className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg"
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <button
          onClick={() => setFormData({
            ...formData,
            strategy: { 
              ...formData.strategy, 
              key_messages: [...formData.strategy.key_messages, ''] 
            }
          })}
          className="text-blue-600 hover:text-blue-700"
        >
          + Add Message
        </button>
      </div>
    </div>
  );

  const renderChannelsStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">Select Distribution Channels</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {['LinkedIn', 'Twitter', 'Facebook', 'Instagram', 'Email', 'Blog'].map((channel) => (
            <label key={channel} className="flex items-center p-4 border rounded-lg cursor-pointer hover:bg-gray-50">
              <input
                type="checkbox"
                checked={formData.channels.selected.includes(channel.toLowerCase())}
                onChange={(e) => {
                  const newSelected = e.target.checked
                    ? [...formData.channels.selected, channel.toLowerCase()]
                    : formData.channels.selected.filter(c => c !== channel.toLowerCase());
                  setFormData({
                    ...formData,
                    channels: { ...formData.channels, selected: newSelected }
                  });
                }}
                className="mr-3"
              />
              <span className="font-medium">{channel}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );

  const renderContentStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">Content Adaptation</h3>
        <p className="text-gray-600 mb-4">Adapt your content for each selected platform</p>
        
        {formData.channels.selected.map((channel) => (
          <div key={channel} className="mb-6 p-4 border rounded-lg">
            <h4 className="font-medium mb-2 capitalize">{channel}</h4>
            <textarea
              value={formData.content.adaptations[channel] || ''}
              onChange={(e) => setFormData({
                ...formData,
                content: {
                  ...formData.content,
                  adaptations: {
                    ...formData.content.adaptations,
                    [channel]: e.target.value
                  }
                }
              })}
              className="w-full p-3 border rounded-lg"
              placeholder={`Adapt content for ${channel}...`}
              rows={4}
            />
          </div>
        ))}
      </div>
    </div>
  );

  const renderScheduleStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">Campaign Schedule</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">Start Date</label>
            <input
              type="date"
              value={formData.schedule.start_date}
              onChange={(e) => setFormData({
                ...formData,
                schedule: { ...formData.schedule, start_date: e.target.value }
              })}
              className="w-full p-3 border rounded-lg"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Posting Frequency</label>
            <select
              value={formData.schedule.frequency}
              onChange={(e) => setFormData({
                ...formData,
                schedule: { ...formData.schedule, frequency: e.target.value }
              })}
              className="w-full p-3 border rounded-lg"
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="biweekly">Bi-weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );

  const renderReviewStep = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">Campaign Summary</h3>
        
        <div className="bg-gray-50 p-6 rounded-lg space-y-4">
          <div>
            <h4 className="font-medium">Strategy</h4>
            <p className="text-gray-600">{formData.strategy.target_audience || 'Not specified'}</p>
          </div>
          
          <div>
            <h4 className="font-medium">Channels</h4>
            <p className="text-gray-600">
              {formData.channels.selected.length > 0 
                ? formData.channels.selected.map(c => c.charAt(0).toUpperCase() + c.slice(1)).join(', ')
                : 'No channels selected'
              }
            </p>
          </div>
          
          <div>
            <h4 className="font-medium">Schedule</h4>
            <p className="text-gray-600">
              {formData.schedule.start_date ? `Starting ${formData.schedule.start_date}` : 'No start date set'} - {formData.schedule.frequency}
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return renderStrategyStep();
      case 1:
        return renderChannelsStep();
      case 2:
        return renderContentStep();
      case 3:
        return renderScheduleStep();
      case 4:
        return renderReviewStep();
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Error</h3>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Campaign Wizard
          </h1>
          <p className="text-gray-600">
            {campaign?.name ? `Setting up campaign: ${campaign.name}` : 'Configure your campaign'}
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  index <= currentStep 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-600'
                }`}>
                  {index + 1}
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-gray-900">{step.title}</h3>
                  <p className="text-xs text-gray-500">{step.description}</p>
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-16 h-1 mx-4 ${
                    index < currentStep ? 'bg-blue-600' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="bg-white rounded-lg shadow p-8">
          {renderCurrentStep()}
        </div>

        {/* Navigation */}
        <div className="mt-8 flex justify-between">
          <button
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className="px-6 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            Previous
          </button>
          
          <div className="flex space-x-3">
            <button
              onClick={() => navigate('/campaigns')}
              className="px-6 py-2 text-gray-600 hover:text-gray-800"
            >
              Save Draft
            </button>
            
            {currentStep === steps.length - 1 ? (
              <button
                onClick={handleComplete}
                disabled={loading}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? 'Launching...' : 'Launch Campaign'}
              </button>
            ) : (
              <button
                onClick={handleNext}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Next
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CampaignWizard; 