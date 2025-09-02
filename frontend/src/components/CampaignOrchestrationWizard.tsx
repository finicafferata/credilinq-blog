import React, { useState, useEffect } from 'react';
import { campaignApi } from '../lib/api';
import { AppError } from '../lib/errors';

interface CampaignOrchestrationWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onCampaignCreated: () => void;
}

interface CampaignOrchestrationData {
  // Step 1: Campaign Foundation
  campaign_name: string;
  campaign_objective: string; // lead_generation, brand_awareness, customer_retention, etc.
  industry: string; // Campaign purpose/focus area
  company_context: string;
  target_market: 'direct_merchants' | 'embedded_partners'; // CrediLinq's two business models
  
  // Step 2: Strategy & Audience
  target_personas: Array<{
    name: string;
    role: string;
    pain_points: string[];
    channels: string[];
  }>;
  key_messages: string[];
  value_proposition: string;
  competitive_differentiation: string;
  
  // Step 3: Content Planning
  content_mix: {
    blog_posts: number;
    social_posts: number;
    email_sequences: number;
    video_content: number;
    infographics: number;
  };
  content_themes: string[];
  content_tone: string;
  
  // Step 4: Distribution & Timeline  
  distribution_channels: string[];
  campaign_duration_weeks: number;
  content_frequency: string; // daily, weekly, bi-weekly
  budget_range: string;
  
  // Step 5: Launch Configuration
  auto_generate_content: boolean;
  auto_schedule: boolean;
  approval_required: boolean;
}

const CampaignOrchestrationWizard: React.FC<CampaignOrchestrationWizardProps> = ({ 
  isOpen, 
  onClose, 
  onCampaignCreated 
}) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiSuggestions, setAiSuggestions] = useState<any>(null);
  const [wizardData, setWizardData] = useState<CampaignOrchestrationData>({
    campaign_name: '',
    campaign_objective: 'lead_generation',
    industry: '',
    company_context: '',
    target_market: 'direct_merchants',
    target_personas: [],
    key_messages: [],
    value_proposition: '',
    competitive_differentiation: '',
    content_mix: {
      blog_posts: 3,
      social_posts: 10,
      email_sequences: 1,
      video_content: 0,
      infographics: 2
    },
    content_themes: [
      'SME credit access fundamentals',
      'CrediLinq platform benefits',
      'Customer success stories',
      'Industry insights and trends'
    ],
    content_tone: 'professional',
    distribution_channels: ['linkedin', 'website'],
    campaign_duration_weeks: 4,
    content_frequency: 'weekly',
    budget_range: '1000-5000',
    auto_generate_content: true,
    auto_schedule: false,
    approval_required: true
  });

  useEffect(() => {
    if (isOpen) {
      setCurrentStep(1);
      setError(null);
      setAiSuggestions(null);
    }
  }, [isOpen]);

  const updateWizardData = (updates: Partial<CampaignOrchestrationData>) => {
    setWizardData(prev => ({ ...prev, ...updates }));
  };

  const addPersona = () => {
    const newPersona = {
      name: `Target Persona ${wizardData.target_personas.length + 1}`,
      role: '',
      pain_points: [],
      channels: ['linkedin']
    };
    updateWizardData({
      target_personas: [...wizardData.target_personas, newPersona]
    });
  };

  const removePersona = (index: number) => {
    const updatedPersonas = wizardData.target_personas.filter((_, i) => i !== index);
    updateWizardData({ target_personas: updatedPersonas });
  };

  const updatePersona = (index: number, updates: any) => {
    const updatedPersonas = wizardData.target_personas.map((persona, i) => 
      i === index ? { ...persona, ...updates } : persona
    );
    updateWizardData({ target_personas: updatedPersonas });
  };

  // Generate smart content themes based on target market and campaign purpose
  const generateSmartContentThemes = (targetMarket: string, campaignPurpose: string): string[] => {
    const baseThemes = ['CrediLinq platform benefits', 'Customer success stories'];
    
    // Target market specific themes
    const marketThemes = targetMarket === 'direct_merchants' 
      ? ['SME credit access fundamentals', 'Business growth through credit', 'Credit application best practices']
      : ['Embedded finance integration guide', 'API implementation benefits', 'Partner success metrics', 'B2B2B revenue models'];
    
    // Campaign purpose specific themes  
    const purposeThemes: { [key: string]: string[] } = {
      'credit_access_education': ['Credit education fundamentals', 'Understanding business credit', 'Credit myths debunked'],
      'partnership_acquisition': ['Partnership benefits showcase', 'Integration success stories', 'ROI of embedded finance'],
      'product_feature_launch': ['New feature highlights', 'Product update benefits', 'Enhanced capabilities demo'],
      'competitive_positioning': ['CrediLinq vs competitors', 'Unique value propositions', 'Market differentiation'],
      'thought_leadership': ['Industry insights and trends', 'Future of fintech', 'Credit market analysis'],
      'customer_success_stories': ['Customer transformation stories', 'Real-world success metrics', 'Before and after case studies'],
      'market_expansion': ['New market opportunities', 'Geographic expansion benefits', 'Sector-specific solutions']
    };
    
    const selectedPurposeThemes = purposeThemes[campaignPurpose] || ['Industry best practices', 'Market insights'];
    
    // Combine and return unique themes
    return [...baseThemes, ...marketThemes.slice(0, 2), ...selectedPurposeThemes.slice(0, 2)].slice(0, 4);
  };

  const getAIContentSuggestions = async () => {
    setLoading(true);
    try {
      // Call real AI recommendations API using PlannerAgent
      const { campaignApi } = await import('../lib/api');
      
      const aiRecommendations = await campaignApi.getAIRecommendations({
        campaign_objective: wizardData.campaign_objective,
        target_market: wizardData.target_market,
        campaign_purpose: wizardData.industry,
        campaign_duration_weeks: wizardData.campaign_duration_weeks,
        company_context: wizardData.company_context
      });
      
      console.log('🤖 AI Recommendations received:', aiRecommendations);
      
      // Structure suggestions for the UI
      const suggestions = {
        recommended_content_mix: aiRecommendations.recommended_content_mix,
        suggested_themes: aiRecommendations.suggested_themes,
        optimal_channels: aiRecommendations.optimal_channels,
        recommended_posting_frequency: aiRecommendations.recommended_posting_frequency,
        ai_reasoning: aiRecommendations.ai_reasoning,
        generated_by: aiRecommendations.generated_by
      };
      
      setAiSuggestions(suggestions);
      
      // Auto-apply AI suggestions
      updateWizardData({
        content_mix: suggestions.recommended_content_mix,
        content_themes: suggestions.suggested_themes,
        distribution_channels: suggestions.optimal_channels,
        content_frequency: suggestions.recommended_posting_frequency
      });
      
    } catch (err) {
      console.error('Error getting AI suggestions:', err);
      
      // Fallback to intelligent defaults if API fails
      const fallbackSuggestions = {
        recommended_content_mix: {
          blog_posts: Math.max(2, Math.floor(wizardData.campaign_duration_weeks / 2)),
          social_posts: wizardData.campaign_duration_weeks * 2,
          email_sequences: 1,
          infographics: Math.max(1, Math.floor(wizardData.campaign_duration_weeks / 3))
        },
        suggested_themes: generateSmartContentThemes(wizardData.target_market, wizardData.industry),
        optimal_channels: wizardData.target_market === 'direct_merchants' 
          ? ['linkedin', 'email', 'website', 'industry_publications'] 
          : ['linkedin', 'email', 'website', 'partner_portals', 'webinars'],
        recommended_posting_frequency: wizardData.campaign_duration_weeks > 6 ? 'bi-weekly' : 'weekly',
        ai_reasoning: 'Using intelligent fallbacks due to AI service unavailability',
        generated_by: 'IntelligentFallback'
      };
      
      setAiSuggestions(fallbackSuggestions);
      updateWizardData({
        content_mix: fallbackSuggestions.recommended_content_mix,
        content_themes: fallbackSuggestions.suggested_themes,
        distribution_channels: fallbackSuggestions.optimal_channels,
        content_frequency: fallbackSuggestions.recommended_posting_frequency
      });
      
    } finally {
      setLoading(false);
    }
  };

  const nextStep = () => {
    if (currentStep < 5) {
      setCurrentStep(currentStep + 1);
      
      // Get AI suggestions when moving to content planning step
      if (currentStep === 2) {
        getAIContentSuggestions();
      }
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleLaunchCampaign = async () => {
    try {
      setLoading(true);
      setError(null);

      // Create orchestration campaign (no blog dependency)
      const response = await campaignApi.createOrchestrationCampaign({
        campaign_name: wizardData.campaign_name,
        company_context: wizardData.company_context,
        description: `${wizardData.campaign_objective.replace('_', ' ')} campaign targeting ${wizardData.target_market === 'direct_merchants' ? 'businesses seeking credit access' : 'companies interested in embedded finance solutions'}${wizardData.industry ? ` with focus on ${wizardData.industry.replace(/_/g, ' ')}` : ''}`,
        strategy_type: wizardData.campaign_objective,
        priority: 'high',
        target_audience: wizardData.target_personas.map(p => p.name).join(', '),
        distribution_channels: wizardData.distribution_channels,
        timeline_weeks: wizardData.campaign_duration_weeks,
        success_metrics: {
          campaign_objective: wizardData.campaign_objective,
          content_pieces: wizardData.content_mix ? Object.values(wizardData.content_mix).reduce((a, b) => a + b, 0) : 0,
          target_channels: wizardData.distribution_channels.length,
          content_themes_count: wizardData.content_themes.length
        },
        budget_allocation: {
          content_creation: 0.6,
          distribution: 0.25,
          promotion: 0.10,
          analytics: 0.05
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
        return wizardData.campaign_name && wizardData.industry && wizardData.company_context;
      case 2:
        return wizardData.target_personas.length > 0 && wizardData.value_proposition;
      case 3:
        return wizardData.content_themes.length > 0;
      case 4:
        return wizardData.distribution_channels.length > 0;
      case 5:
        return true;
      default:
        return true;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-purple-600 to-blue-600 text-white">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">🚀 Campaign Orchestration Wizard</h2>
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
              <span>Step {currentStep} of 5</span>
              <span>{Math.round((currentStep / 5) * 100)}% Complete</span>
            </div>
            <div className="w-full bg-purple-800 rounded-full h-2">
              <div 
                className="bg-white h-2 rounded-full transition-all duration-300"
                style={{ width: `${(currentStep / 5) * 100}%` }}
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

          {/* Step 1: Campaign Foundation */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">🎯 Campaign Foundation</h3>
                <p className="text-gray-600 mt-2">Define your campaign's core purpose and business context</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Name
                    </label>
                    <input
                      type="text"
                      value={wizardData.campaign_name}
                      onChange={(e) => updateWizardData({ campaign_name: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      placeholder="e.g., Q1 Lead Generation Campaign"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Primary Campaign Objective
                    </label>
                    <select
                      value={wizardData.campaign_objective}
                      onChange={(e) => updateWizardData({ campaign_objective: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    >
                      <option value="lead_generation">Lead Generation</option>
                      <option value="brand_awareness">Brand Awareness</option>
                      <option value="customer_retention">Customer Retention</option>
                      <option value="product_launch">Product Launch</option>
                      <option value="thought_leadership">Thought Leadership</option>
                      <option value="market_education">Market Education</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Purpose
                    </label>
                    <select
                      value={wizardData.industry}
                      onChange={(e) => updateWizardData({ industry: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    >
                      <option value="">Select campaign purpose...</option>
                      <option value="credit_access_education">📚 Credit Access Education & Awareness</option>
                      <option value="partnership_acquisition">🤝 Partnership & Integration Acquisition</option>
                      <option value="product_feature_launch">🚀 Product Feature Launch & Updates</option>
                      <option value="competitive_positioning">⚔️ Competitive Positioning & Differentiation</option>
                      <option value="thought_leadership">💡 Thought Leadership & Industry Expertise</option>
                      <option value="customer_success_stories">🎯 Customer Success & Case Studies</option>
                      <option value="market_expansion">🌍 Market Expansion & New Segments</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Target Market
                    </label>
                    <select
                      value={wizardData.target_market}
                      onChange={(e) => updateWizardData({ target_market: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    >
                      <option value="direct_merchants">🏪 Direct Merchant Acquisition (Businesses seeking credit)</option>
                      <option value="embedded_partners">🏢 Partner/Embedded Solutions (Companies wanting to embed credit)</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Company & Business Context
                  </label>
                  <textarea
                    value={wizardData.company_context}
                    onChange={(e) => updateWizardData({ company_context: e.target.value })}
                    rows={8}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    placeholder="Describe your company, products/services, market position, and what makes you unique. This context will help AI generate relevant content..."
                  />
                </div>
              </div>

              {/* Foundation Preview */}
              <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                <h4 className="font-medium text-purple-900 mb-2">🔍 Campaign Overview</h4>
                <p className="text-purple-800 text-sm">
                  {wizardData.campaign_objective && (
                    <span className="capitalize">{wizardData.campaign_objective.replace('_', ' ')}</span>
                  )} campaign for {wizardData.target_market === 'direct_merchants' ? 'businesses seeking credit' : 'embedded finance partners'}{wizardData.industry && (
                    <span> - Focus: {wizardData.industry.replace(/_/g, ' ')}</span>
                  )}.
                </p>
              </div>
            </div>
          )}

          {/* Step 2: Strategy & Audience */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">👥 Strategy & Target Audience</h3>
                <p className="text-gray-600 mt-2">Define your target personas and strategic messaging</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Unique Angle
                      <span className="text-xs text-gray-500 ml-2">(What makes THIS campaign special?)</span>
                    </label>
                    <textarea
                      value={wizardData.value_proposition}
                      onChange={(e) => updateWizardData({ value_proposition: e.target.value })}
                      rows={3}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      placeholder={
                        wizardData.target_market === 'direct_merchants' 
                          ? "e.g., 'Fast credit decisions in 24 hours for growing SMEs' or 'Zero hidden fees, transparent pricing'"
                          : "e.g., 'White-label credit API ready in 2 weeks' or 'Revenue share model with guaranteed ROI'"
                      }
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Campaign Focus Message
                      <span className="text-xs text-gray-500 ml-2">(Primary message for this campaign)</span>
                    </label>
                    <textarea
                      value={wizardData.competitive_differentiation}
                      onChange={(e) => updateWizardData({ competitive_differentiation: e.target.value })}
                      rows={3}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      placeholder={
                        wizardData.industry === 'credit_access_education' 
                          ? "e.g., 'Demystifying business credit - your guide to financial growth'"
                          : wizardData.industry === 'partnership_acquisition'
                          ? "e.g., 'Transform your platform into a credit powerhouse with our embedded solutions'"
                          : wizardData.industry === 'product_feature_launch'
                          ? "e.g., 'Introducing faster approvals and enhanced dashboard analytics'"
                          : "e.g., 'The specific message this campaign will emphasize'"
                      }
                    />
                  </div>

                  {/* Smart Campaign Suggestions */}
                  <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <h5 className="font-medium text-purple-900 mb-3">💡 Smart Campaign Suggestions</h5>
                    <div className="grid grid-cols-1 gap-2 text-sm">
                      <button
                        onClick={() => {
                          const angle = wizardData.target_market === 'direct_merchants' 
                            ? 'Fast, transparent credit decisions for growing businesses'
                            : 'Seamless credit integration with full API control';
                          const message = wizardData.industry === 'credit_access_education'
                            ? 'Unlock your business potential with smart credit solutions'
                            : wizardData.industry === 'partnership_acquisition' 
                            ? 'Partner with CrediLinq and transform your customer experience'
                            : 'Discover how CrediLinq accelerates business growth';
                          updateWizardData({ 
                            value_proposition: angle,
                            competitive_differentiation: message
                          });
                        }}
                        className="text-left p-2 bg-white rounded border hover:bg-purple-100 transition-colors"
                      >
                        <div className="font-medium text-purple-800">✨ Use Campaign-Optimized Messaging</div>
                        <div className="text-purple-600 text-xs">
                          {wizardData.target_market === 'direct_merchants' ? 'Direct merchant' : 'Embedded partner'} + {wizardData.industry.replace(/_/g, ' ')} focus
                        </div>
                      </button>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="block text-sm font-medium text-gray-700">
                        Key Messages
                      </label>
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            const smartMessages = wizardData.target_market === 'direct_merchants' 
                              ? ['Quick approval process', 'Transparent pricing', 'Dedicated support']
                              : ['Easy API integration', 'White-label solution', 'Revenue sharing model'];
                            updateWizardData({ key_messages: smartMessages });
                          }}
                          className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full hover:bg-purple-200 transition-colors"
                        >
                          ⚡ Quick Start
                        </button>
                        <button
                          onClick={() => {
                            const newMessages = [...wizardData.key_messages, ''];
                            updateWizardData({ key_messages: newMessages });
                          }}
                          className="text-sm text-purple-600 hover:text-purple-700"
                        >
                          + Add Message
                        </button>
                      </div>
                    </div>
                    {wizardData.key_messages.map((message, index) => (
                      <div key={index} className="flex gap-2 mb-2">
                        <input
                          type="text"
                          value={message}
                          onChange={(e) => {
                            const updated = [...wizardData.key_messages];
                            updated[index] = e.target.value;
                            updateWizardData({ key_messages: updated });
                          }}
                          className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                          placeholder="Enter key message"
                        />
                        <button
                          onClick={() => {
                            const updated = wizardData.key_messages.filter((_, i) => i !== index);
                            updateWizardData({ key_messages: updated });
                          }}
                          className="text-red-500 hover:text-red-700"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    {wizardData.key_messages.length === 0 && (
                      <button
                        onClick={() => updateWizardData({ key_messages: [''] })}
                        className="w-full py-2 border-2 border-dashed border-gray-300 rounded-md text-gray-500 hover:border-purple-300 hover:text-purple-600"
                      >
                        + Add your first key message
                      </button>
                    )}
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-4">
                    <h4 className="font-medium text-gray-900">Target Personas</h4>
                    <button
                      onClick={addPersona}
                      className="px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700"
                    >
                      + Add Persona
                    </button>
                  </div>

                  {wizardData.target_personas.length === 0 && (
                    <div className="text-center py-8 border-2 border-dashed border-gray-300 rounded-lg">
                      <p className="text-gray-500 mb-4">No target personas defined yet</p>
                      <button
                        onClick={addPersona}
                        className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
                      >
                        Create First Persona
                      </button>
                    </div>
                  )}

                  {wizardData.target_personas.map((persona, index) => (
                    <div key={index} className="mb-4 p-4 border border-gray-200 rounded-lg">
                      <div className="flex justify-between items-center mb-3">
                        <input
                          type="text"
                          value={persona.name}
                          onChange={(e) => updatePersona(index, { name: e.target.value })}
                          className="font-medium text-gray-900 bg-transparent border-none focus:outline-none focus:ring-0 p-0"
                          placeholder="Persona Name"
                        />
                        <button
                          onClick={() => removePersona(index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          ×
                        </button>
                      </div>
                      
                      <input
                        type="text"
                        value={persona.role}
                        onChange={(e) => updatePersona(index, { role: e.target.value })}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded mb-2"
                        placeholder="Job Role/Title"
                      />
                      
                      <textarea
                        value={persona.pain_points.join(', ')}
                        onChange={(e) => updatePersona(index, { 
                          pain_points: e.target.value.split(',').map(p => p.trim()).filter(p => p) 
                        })}
                        rows={2}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded"
                        placeholder="Pain points (comma-separated)"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Step 3: AI Content Planning */}
          {currentStep === 3 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">🤖 AI Content Planning</h3>
                <p className="text-gray-600 mt-2">Let AI suggest the optimal content mix for your campaign</p>
              </div>

              {aiSuggestions && (
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200 mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-blue-900">🎯 AI Recommendations</h4>
                    <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                      {aiSuggestions.generated_by === 'PlannerAgent' ? '🤖 AI Generated' : '⚡ Smart Default'}
                    </span>
                  </div>
                  <p className="text-blue-800 text-sm mb-2">
                    Based on your {wizardData.campaign_objective.replace('_', ' ')} objective and {wizardData.campaign_duration_weeks}-week timeline,
                    I recommend {aiSuggestions.recommended_content_mix ? Object.values(aiSuggestions.recommended_content_mix).reduce((a, b) => a + b, 0) : 0} total content pieces
                    across {aiSuggestions.optimal_channels?.length || 0} channels.
                  </p>
                  {aiSuggestions.ai_reasoning && (
                    <div className="mt-2 p-2 bg-blue-100 rounded text-xs text-blue-700">
                      <strong>AI Insight:</strong> {aiSuggestions.ai_reasoning}
                    </div>
                  )}
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-4">Content Mix</h4>
                  <div className="space-y-4">
                    {Object.entries(wizardData.content_mix).map(([type, count]) => (
                      <div key={type} className="flex items-center justify-between">
                        <label className="text-sm font-medium text-gray-700 capitalize">
                          {type.replace('_', ' ')}
                        </label>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => {
                              const updated = { ...wizardData.content_mix };
                              updated[type as keyof typeof wizardData.content_mix] = Math.max(0, count - 1);
                              updateWizardData({ content_mix: updated });
                            }}
                            className="w-8 h-8 rounded-full bg-gray-200 hover:bg-gray-300 flex items-center justify-center"
                          >
                            −
                          </button>
                          <span className="w-8 text-center font-medium">{count}</span>
                          <button
                            onClick={() => {
                              const updated = { ...wizardData.content_mix };
                              updated[type as keyof typeof wizardData.content_mix] = count + 1;
                              updateWizardData({ content_mix: updated });
                            }}
                            className="w-8 h-8 rounded-full bg-purple-200 hover:bg-purple-300 flex items-center justify-center"
                          >
                            +
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Content Tone
                    </label>
                    <select
                      value={wizardData.content_tone}
                      onChange={(e) => updateWizardData({ content_tone: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="professional">Professional</option>
                      <option value="friendly">Friendly & Approachable</option>
                      <option value="authoritative">Authoritative</option>
                      <option value="conversational">Conversational</option>
                      <option value="technical">Technical & Detailed</option>
                    </select>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-4">
                    <h4 className="font-medium text-gray-900">Content Themes</h4>
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          const smartThemes = generateSmartContentThemes(wizardData.target_market, wizardData.industry);
                          updateWizardData({ content_themes: smartThemes });
                        }}
                        className="text-sm bg-purple-100 text-purple-700 px-3 py-1 rounded-full hover:bg-purple-200 transition-colors"
                        title="Generate themes based on your target market and campaign purpose"
                      >
                        ⚡ Quick Start
                      </button>
                      <button
                        onClick={() => {
                          const newThemes = [...wizardData.content_themes, ''];
                          updateWizardData({ content_themes: newThemes });
                        }}
                        className="text-sm text-purple-600 hover:text-purple-700"
                      >
                        + Add Theme
                      </button>
                    </div>
                  </div>

                  {wizardData.content_themes.map((theme, index) => (
                    <div key={index} className="flex gap-2 mb-2">
                      <input
                        type="text"
                        value={theme}
                        onChange={(e) => {
                          const updated = [...wizardData.content_themes];
                          updated[index] = e.target.value;
                          updateWizardData({ content_themes: updated });
                        }}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                        placeholder="Enter content theme"
                      />
                      <button
                        onClick={() => {
                          const updated = wizardData.content_themes.filter((_, i) => i !== index);
                          updateWizardData({ content_themes: updated });
                        }}
                        className="text-red-500 hover:text-red-700"
                      >
                        ×
                      </button>
                    </div>
                  ))}

                  {wizardData.content_themes.length === 0 && (
                    <div className="text-center py-4 border-2 border-dashed border-gray-300 rounded-lg">
                      <p className="text-gray-500 text-sm mb-2">Get started with CrediLinq-focused themes</p>
                      <div className="flex justify-center gap-2">
                        <button
                          onClick={() => {
                            const smartThemes = generateSmartContentThemes(wizardData.target_market, wizardData.industry);
                            updateWizardData({ content_themes: smartThemes });
                          }}
                          className="text-sm bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-colors"
                        >
                          ⚡ Quick Start Themes
                        </button>
                        <button
                          onClick={() => updateWizardData({ 
                            content_themes: aiSuggestions?.suggested_themes || generateSmartContentThemes(wizardData.target_market, wizardData.industry)
                          })}
                          className="text-sm text-purple-600 hover:text-purple-700 px-4 py-2 border border-purple-300 rounded hover:bg-purple-50 transition-colors"
                        >
                          Use AI Suggestions
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="mt-6 bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium text-gray-900 mb-2">📊 Content Planning Summary</h5>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div>Total content pieces: <span className="font-medium">{wizardData.content_mix ? Object.values(wizardData.content_mix).reduce((a, b) => a + b, 0) : 0}</span></div>
                      <div>Content themes: <span className="font-medium">{wizardData.content_themes.length}</span></div>
                      <div>Tone: <span className="font-medium capitalize">{wizardData.content_tone}</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Distribution & Timeline */}
          {currentStep === 4 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">📅 Distribution & Timeline</h3>
                <p className="text-gray-600 mt-2">Configure your campaign's distribution strategy and schedule</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Distribution Channels
                  </label>
                  <div className="space-y-3">
                    {[
                      { value: 'website', label: 'Company Website', desc: 'Blog posts and landing pages' },
                      { value: 'linkedin', label: 'LinkedIn', desc: 'Professional networking and B2B reach' },
                      { value: 'twitter', label: 'Twitter/X', desc: 'Real-time updates and thought leadership' },
                      { value: 'email', label: 'Email Marketing', desc: 'Direct subscriber communication' },
                      { value: 'facebook', label: 'Facebook', desc: 'Broader social media reach' },
                      { value: 'instagram', label: 'Instagram', desc: 'Visual content and brand awareness' },
                      { value: 'youtube', label: 'YouTube', desc: 'Video content distribution' },
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
                          className="mt-1 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
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
                      Campaign Duration
                    </label>
                    <select
                      value={wizardData.campaign_duration_weeks}
                      onChange={(e) => updateWizardData({ campaign_duration_weeks: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value={2}>2 Weeks - Sprint Campaign</option>
                      <option value={4}>4 Weeks - Standard Campaign</option>
                      <option value={6}>6 Weeks - Extended Campaign</option>
                      <option value={8}>8 Weeks - Comprehensive Campaign</option>
                      <option value={12}>12 Weeks - Long-term Strategy</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Content Publishing Frequency
                    </label>
                    <select
                      value={wizardData.content_frequency}
                      onChange={(e) => updateWizardData({ content_frequency: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="daily">Daily</option>
                      <option value="every_other_day">Every Other Day</option>
                      <option value="weekly">Weekly</option>
                      <option value="bi-weekly">Bi-weekly</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Budget Range (USD)
                    </label>
                    <select
                      value={wizardData.budget_range}
                      onChange={(e) => updateWizardData({ budget_range: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="0-1000">$0 - $1,000</option>
                      <option value="1000-5000">$1,000 - $5,000</option>
                      <option value="5000-15000">$5,000 - $15,000</option>
                      <option value="15000-50000">$15,000 - $50,000</option>
                      <option value="50000+">$50,000+</option>
                    </select>
                  </div>

                  {/* Campaign Timeline Preview */}
                  <div className="bg-green-50 rounded-lg p-4 border border-green-200 mt-4">
                    <h4 className="font-medium text-green-900 mb-2">📈 Campaign Timeline</h4>
                    <div className="text-green-800 text-sm space-y-1">
                      <div>Duration: <span className="font-medium">{wizardData.campaign_duration_weeks} weeks</span></div>
                      <div>Content pieces: <span className="font-medium">{wizardData.content_mix ? Object.values(wizardData.content_mix).reduce((a, b) => a + b, 0) : 0}</span></div>
                      <div>Channels: <span className="font-medium">{wizardData.distribution_channels.length}</span></div>
                      <div>Frequency: <span className="font-medium capitalize">{wizardData.content_frequency.replace('_', ' ')}</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 5: Launch Configuration */}
          {currentStep === 5 && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-900">🚀 Launch Configuration</h3>
                <p className="text-gray-600 mt-2">Configure how your campaign will be executed</p>
              </div>

              <div className="max-w-2xl mx-auto space-y-6">
                <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6 border">
                  <h4 className="font-semibold text-gray-900 mb-4">Campaign Summary</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-700">Campaign:</span>
                      <p className="text-gray-600">{wizardData.campaign_name}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Objective:</span>
                      <p className="text-gray-600 capitalize">{wizardData.campaign_objective.replace('_', ' ')}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Duration:</span>
                      <p className="text-gray-600">{wizardData.campaign_duration_weeks} weeks</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Content Pieces:</span>
                      <p className="text-gray-600">{wizardData.content_mix ? Object.values(wizardData.content_mix).reduce((a, b) => a + b, 0) : 0} total</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Channels:</span>
                      <p className="text-gray-600">{wizardData.distribution_channels.join(', ')}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Target Personas:</span>
                      <p className="text-gray-600">{wizardData.target_personas.length} defined</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900">Automation Settings</h4>
                  
                  <label className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={wizardData.auto_generate_content}
                      onChange={(e) => updateWizardData({ auto_generate_content: e.target.checked })}
                      className="mt-1 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                    />
                    <div>
                      <div className="font-medium text-gray-900">Auto-Generate Content</div>
                      <div className="text-sm text-gray-600">AI will automatically create all content pieces based on your campaign strategy</div>
                    </div>
                  </label>

                  <label className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={wizardData.auto_schedule}
                      onChange={(e) => updateWizardData({ auto_schedule: e.target.checked })}
                      className="mt-1 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                    />
                    <div>
                      <div className="font-medium text-gray-900">Auto-Schedule Publishing</div>
                      <div className="text-sm text-gray-600">Automatically schedule content across channels based on optimal timing</div>
                    </div>
                  </label>

                  <label className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={wizardData.approval_required}
                      onChange={(e) => updateWizardData({ approval_required: e.target.checked })}
                      className="mt-1 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                    />
                    <div>
                      <div className="font-medium text-gray-900">Require Approval</div>
                      <div className="text-sm text-gray-600">Review and approve content before publishing (recommended)</div>
                    </div>
                  </label>
                </div>

                <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                  <h4 className="font-medium text-yellow-800 mb-2">🎯 Next Steps</h4>
                  <p className="text-yellow-700 text-sm">
                    After launching, your AI agents will begin creating content according to your campaign strategy. 
                    {wizardData.approval_required && ' You\'ll be notified to review each piece before publication.'}
                  </p>
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
            
            {currentStep < 5 ? (
              <button
                onClick={nextStep}
                disabled={!isStepValid(currentStep)}
                className="px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Next Step →
              </button>
            ) : (
              <button
                onClick={handleLaunchCampaign}
                disabled={loading || !isStepValid(currentStep)}
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-md hover:from-purple-700 hover:to-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all duration-200 font-medium"
              >
                {loading ? 'Launching Campaign...' : '🚀 Launch Campaign'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CampaignOrchestrationWizard;