import React, { useState, useEffect } from 'react';
import { ArrowLeftIcon, DocumentTextIcon, LightBulbIcon, ChartBarIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { Link, useNavigate } from 'react-router-dom';
import api from '../lib/api';

interface ContentType {
  value: string;
  label: string;
  description: string;
}

interface ContentPurpose {
  value: string;
  label: string;
  description: string;
}

interface ContentBriefRequest {
  topic: string;
  content_type: string;
  primary_purpose: string;
  target_audience: string;
  company_context: string;
  competitive_focus: string;
  distribution_channels: string[];
  success_metrics: string[];
}

interface ContentBriefResponse {
  brief_id: string;
  title: string;
  content_type: string;
  primary_purpose: string;
  marketing_objective: string;
  target_audience: string;
  primary_keyword: {
    keyword: string;
    search_volume: number;
    difficulty: string;
    intent: string;
  };
  secondary_keywords: Array<{
    keyword: string;
    search_volume: number;
    difficulty: string;
    intent: string;
  }>;
  content_structure: {
    estimated_word_count: number;
    suggested_headlines: string[];
    content_outline: Array<{
      section: string;
      description: string;
    }>;
    call_to_actions: string[];
  };
  success_kpis: string[];
  estimated_creation_time: string;
  summary: string;
  created_at: string;
}

const ContentBriefCreator: React.FC = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [contentTypes, setContentTypes] = useState<ContentType[]>([]);
  const [contentPurposes, setContentPurposes] = useState<ContentPurpose[]>([]);
  const [audienceSegments, setAudienceSegments] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedBrief, setGeneratedBrief] = useState<ContentBriefResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isGeneratingBlog, setIsGeneratingBlog] = useState(false);

  const [formData, setFormData] = useState<ContentBriefRequest>({
    topic: '',
    content_type: 'blog_post',
    primary_purpose: 'lead_generation',
    target_audience: 'B2B finance professionals',
    company_context: '',
    competitive_focus: '',
    distribution_channels: ['website', 'linkedin', 'email'],
    success_metrics: []
  });

  // Load content types and purposes on component mount
  useEffect(() => {
    loadContentOptions();
  }, []);

  const loadContentOptions = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v2/content-briefs/types/available');
      if (response.ok) {
        const data = await response.json();
        setContentTypes(data.content_types);
        setContentPurposes(data.content_purposes);
        setAudienceSegments(data.default_audience_segments);
      } else {
        setError('Failed to load content options');
      }
    } catch (err) {
      setError('Network error loading options');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field: keyof ContentBriefRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleChannelToggle = (channel: string) => {
    setFormData(prev => ({
      ...prev,
      distribution_channels: prev.distribution_channels.includes(channel)
        ? prev.distribution_channels.filter(c => c !== channel)
        : [...prev.distribution_channels, channel]
    }));
  };

  const generateContentBrief = async () => {
    if (!formData.topic.trim()) {
      setError('Please enter a content topic');
      return;
    }

    setIsGenerating(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v2/content-briefs/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        const brief = await response.json();
        setGeneratedBrief(brief);
        setCurrentStep(4); // Move to results step
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to generate content brief');
      }
    } catch (err) {
      setError('Network error generating brief');
    } finally {
      setIsGenerating(false);
    }
  };

  const nextStep = () => {
    if (currentStep < 4) setCurrentStep(currentStep + 1);
  };

  const prevStep = () => {
    if (currentStep > 1) setCurrentStep(currentStep - 1);
  };

  const resetForm = () => {
    setCurrentStep(1);
    setGeneratedBrief(null);
    setError(null);
    setFormData({
      topic: '',
      content_type: 'blog_post',
      primary_purpose: 'lead_generation',
      target_audience: 'B2B finance professionals',
      company_context: '',
      competitive_focus: '',
      distribution_channels: ['website', 'linkedin', 'email'],
      success_metrics: []
    });
  };

  const generateBlogFromBrief = async () => {
    if (!generatedBrief || !generatedBrief.id) {
      setError('No brief available to generate blog from');
      return;
    }

    setIsGeneratingBlog(true);
    setError(null);

    try {
      const response = await api.post(`/blogs/from-brief/${generatedBrief.id}`);
      const blogData = response.data;
      
      // Navigate to the newly created blog
      navigate(`/blogs/${blogData.id}`, {
        state: { 
          message: 'Blog successfully generated from content brief!',
          briefId: generatedBrief.id
        }
      });
    } catch (error: any) {
      console.error('Error generating blog from brief:', error);
      setError(
        error.response?.data?.detail || 
        'Failed to generate blog from brief. Please try again.'
      );
    } finally {
      setIsGeneratingBlog(false);
    }
  };

  // Step indicators
  const steps = [
    { number: 1, title: 'Content Topic', icon: DocumentTextIcon },
    { number: 2, title: 'Strategy & Audience', icon: LightBulbIcon },
    { number: 3, title: 'Review & Generate', icon: ChartBarIcon },
    { number: 4, title: 'Your Brief', icon: CheckCircleIcon }
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading content options...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center">
            <Link to="/campaigns" className="text-gray-400 hover:text-gray-600">
              <ArrowLeftIcon className="h-6 w-6" />
            </Link>
            <h1 className="ml-4 text-2xl font-bold text-gray-900">
              Create Strategic Content Brief
            </h1>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.number} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                  currentStep >= step.number 
                    ? 'bg-blue-600 border-blue-600 text-white' 
                    : 'border-gray-300 text-gray-500'
                }`}>
                  {currentStep > step.number ? (
                    <CheckCircleIcon className="h-6 w-6" />
                  ) : (
                    <step.icon className="h-5 w-5" />
                  )}
                </div>
                <div className="ml-3 hidden sm:block">
                  <p className={`text-sm font-medium ${
                    currentStep >= step.number ? 'text-blue-600' : 'text-gray-500'
                  }`}>
                    Step {step.number}
                  </p>
                  <p className={`text-xs ${
                    currentStep >= step.number ? 'text-blue-500' : 'text-gray-400'
                  }`}>
                    {step.title}
                  </p>
                </div>
                {index < steps.length - 1 && (
                  <div className={`flex-1 mx-4 h-0.5 ${
                    currentStep > step.number ? 'bg-blue-600' : 'bg-gray-300'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="text-sm text-red-600">{error}</div>
          </div>
        )}

        {/* Step Content */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          {currentStep === 1 && (
            <div className="p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                What content topic do you want to create a brief for?
              </h2>
              <div className="space-y-6">
                <div>
                  <label htmlFor="topic" className="block text-sm font-medium text-gray-700 mb-2">
                    Content Topic *
                  </label>
                  <input
                    type="text"
                    id="topic"
                    value={formData.topic}
                    onChange={(e) => handleInputChange('topic', e.target.value)}
                    placeholder="e.g., Embedded Finance for B2B Marketplaces"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Enter the main topic or subject for your content piece
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="content_type" className="block text-sm font-medium text-gray-700 mb-2">
                      Content Type *
                    </label>
                    <select
                      id="content_type"
                      value={formData.content_type}
                      onChange={(e) => handleInputChange('content_type', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {contentTypes.map((type) => (
                        <option key={type.value} value={type.value}>
                          {type.label} - {type.description}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label htmlFor="primary_purpose" className="block text-sm font-medium text-gray-700 mb-2">
                      Primary Purpose *
                    </label>
                    <select
                      id="primary_purpose"
                      value={formData.primary_purpose}
                      onChange={(e) => handleInputChange('primary_purpose', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {contentPurposes.map((purpose) => (
                        <option key={purpose.value} value={purpose.value}>
                          {purpose.label} - {purpose.description}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <div className="p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Strategy & Audience Details
              </h2>
              <div className="space-y-6">
                <div>
                  <label htmlFor="target_audience" className="block text-sm font-medium text-gray-700 mb-2">
                    Target Audience *
                  </label>
                  <select
                    id="target_audience"
                    value={formData.target_audience}
                    onChange={(e) => handleInputChange('target_audience', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {audienceSegments.map((segment) => (
                      <option key={segment} value={segment}>
                        {segment}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="company_context" className="block text-sm font-medium text-gray-700 mb-2">
                    Company Context
                  </label>
                  <textarea
                    id="company_context"
                    value={formData.company_context}
                    onChange={(e) => handleInputChange('company_context', e.target.value)}
                    placeholder="e.g., CrediLinq.ai is a leading fintech company providing credit-as-a-service solutions for B2B marketplaces..."
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Describe your company's position and unique value proposition
                  </p>
                </div>

                <div>
                  <label htmlFor="competitive_focus" className="block text-sm font-medium text-gray-700 mb-2">
                    Competitive Focus
                  </label>
                  <input
                    type="text"
                    id="competitive_focus"
                    value={formData.competitive_focus}
                    onChange={(e) => handleInputChange('competitive_focus', e.target.value)}
                    placeholder="e.g., Traditional banking solutions, Legacy lending platforms"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    What competitors or solutions should we differentiate against?
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Distribution Channels
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {['website', 'linkedin', 'email', 'twitter', 'facebook', 'instagram'].map((channel) => (
                      <label key={channel} className="inline-flex items-center">
                        <input
                          type="checkbox"
                          checked={formData.distribution_channels.includes(channel)}
                          onChange={() => handleChannelToggle(channel)}
                          className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                        />
                        <span className="ml-2 text-sm text-gray-700 capitalize">{channel}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentStep === 3 && (
            <div className="p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Review & Generate Brief
              </h2>
              <div className="bg-gray-50 rounded-md p-4 space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Topic:</span>
                    <span className="ml-2 text-gray-900">{formData.topic}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Content Type:</span>
                    <span className="ml-2 text-gray-900">{
                      contentTypes.find(t => t.value === formData.content_type)?.label
                    }</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Purpose:</span>
                    <span className="ml-2 text-gray-900">{
                      contentPurposes.find(p => p.value === formData.primary_purpose)?.label
                    }</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Audience:</span>
                    <span className="ml-2 text-gray-900">{formData.target_audience}</span>
                  </div>
                </div>
                {formData.company_context && (
                  <div className="text-sm">
                    <span className="font-medium text-gray-700">Company Context:</span>
                    <p className="mt-1 text-gray-900">{formData.company_context}</p>
                  </div>
                )}
                <div className="text-sm">
                  <span className="font-medium text-gray-700">Distribution:</span>
                  <span className="ml-2 text-gray-900">{formData.distribution_channels.join(', ')}</span>
                </div>
              </div>

              <div className="mt-6 bg-blue-50 border border-blue-200 rounded-md p-4">
                <h3 className="text-sm font-medium text-blue-900 mb-2">What happens next?</h3>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• AI will conduct comprehensive SEO keyword research</li>
                  <li>• Competitive landscape analysis will identify differentiation opportunities</li>
                  <li>• Strategic content structure will be generated with optimal word count</li>
                  <li>• Success metrics will be defined aligned with your business objectives</li>
                </ul>
              </div>
            </div>
          )}

          {currentStep === 4 && generatedBrief && (
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900">
                  Your Strategic Content Brief
                </h2>
                <div className="text-sm text-gray-500">
                  Brief ID: {generatedBrief.brief_id}
                </div>
              </div>

              {/* Brief Summary */}
              <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-md">
                <h3 className="text-sm font-medium text-green-900 mb-2">Executive Summary</h3>
                <div className="text-sm text-green-800 whitespace-pre-line">
                  {generatedBrief.summary}
                </div>
              </div>

              {/* Key Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-sm font-medium text-blue-900">Primary Keyword</div>
                  <div className="text-lg font-semibold text-blue-600">
                    {generatedBrief.primary_keyword.keyword}
                  </div>
                  <div className="text-xs text-blue-700">
                    {generatedBrief.primary_keyword.search_volume.toLocaleString()} searches/month
                    • {generatedBrief.primary_keyword.difficulty} difficulty
                  </div>
                </div>
                
                <div className="bg-purple-50 p-4 rounded-lg">
                  <div className="text-sm font-medium text-purple-900">Estimated Content</div>
                  <div className="text-lg font-semibold text-purple-600">
                    {generatedBrief.content_structure.estimated_word_count} words
                  </div>
                  <div className="text-xs text-purple-700">
                    Creation time: {generatedBrief.estimated_creation_time}
                  </div>
                </div>
                
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-sm font-medium text-green-900">Success Metrics</div>
                  <div className="text-lg font-semibold text-green-600">
                    {generatedBrief.success_kpis.length} KPIs
                  </div>
                  <div className="text-xs text-green-700">
                    {generatedBrief.success_kpis.slice(0, 2).join(', ')}
                  </div>
                </div>
              </div>

              {/* Detailed Sections */}
              <div className="space-y-6">
                {/* SEO Keywords */}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-3">SEO Strategy</h3>
                  <div className="bg-gray-50 p-4 rounded-md">
                    <div className="mb-3">
                      <span className="font-medium text-gray-700">Primary:</span>
                      <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
                        {generatedBrief.primary_keyword.keyword}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Secondary Keywords:</span>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {generatedBrief.secondary_keywords.map((kw, idx) => (
                          <span key={idx} className="px-2 py-1 bg-gray-100 text-gray-800 rounded text-sm">
                            {kw.keyword}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Content Structure */}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Content Structure</h3>
                  <div className="bg-gray-50 p-4 rounded-md space-y-3">
                    <div>
                      <span className="font-medium text-gray-700">Suggested Headlines:</span>
                      <ul className="mt-2 space-y-1">
                        {generatedBrief.content_structure.suggested_headlines.map((headline, idx) => (
                          <li key={idx} className="text-sm text-gray-800">• {headline}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Content Outline:</span>
                      <div className="mt-2 space-y-2">
                        {generatedBrief.content_structure.content_outline.map((section, idx) => (
                          <div key={idx} className="text-sm">
                            <span className="font-medium text-gray-800">{section.section}:</span>
                            <span className="ml-2 text-gray-700">{section.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Call to Actions */}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Recommended CTAs</h3>
                  <div className="bg-gray-50 p-4 rounded-md">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {generatedBrief.content_structure.call_to_actions.map((cta, idx) => (
                        <div key={idx} className="text-sm text-gray-800">• {cta}</div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="mt-8 pt-6 border-t border-gray-200 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={generateBlogFromBrief}
                  disabled={isGeneratingBlog}
                  className="flex-1 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isGeneratingBlog ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating Blog...
                    </>
                  ) : (
                    'Generate Blog Post'
                  )}
                </button>
                <Link
                  to="/campaigns"
                  className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 text-center flex items-center justify-center"
                >
                  Create Campaign
                </Link>
                <button
                  onClick={() => window.print()}
                  className="flex-1 bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700"
                >
                  Export Brief
                </button>
                <button
                  onClick={resetForm}
                  className="flex-1 bg-gray-200 text-gray-800 px-4 py-2 rounded-md hover:bg-gray-300"
                >
                  Create Another Brief
                </button>
              </div>
            </div>
          )}

          {/* Navigation */}
          {currentStep < 4 && (
            <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-between">
              <button
                onClick={prevStep}
                disabled={currentStep === 1}
                className="px-4 py-2 text-gray-600 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              
              {currentStep < 3 ? (
                <button
                  onClick={nextStep}
                  className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Next
                </button>
              ) : (
                <button
                  onClick={generateContentBrief}
                  disabled={isGenerating || !formData.topic.trim()}
                  className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  {isGenerating ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating Brief...
                    </>
                  ) : (
                    'Generate Strategic Brief'
                  )}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContentBriefCreator;