import React, { useState } from 'react';
// Content generation wizard component
import { 
  SparklesIcon,
  XMarkIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  EnvelopeIcon,
  PhotoIcon,
  PresentationChartLineIcon,
  CheckIcon
} from '@heroicons/react/24/outline';
import type { 
  ContentType, 
  CampaignGenerationRequest 
} from '../types/contentTypes';
import { 
  contentDeliverableApi, 
  CONTENT_TYPE_CONFIG 
} from '../services/contentDeliverableApi';

interface ContentGenerationWizardProps {
  isOpen: boolean;
  onClose: () => void;
  campaignId: string;
  campaignName?: string;
  briefing?: Record<string, any>;
  onGenerationComplete?: () => void;
}

const DEFAULT_BRIEFING = {
  marketing_objective: "Generate high-quality content for the campaign",
  target_audience: "Business professionals and decision makers",
  channels: ["blog", "linkedin", "email"],
  desired_tone: "professional",
  company_context: "Expert content creation for business growth"
};

export const ContentGenerationWizard: React.FC<ContentGenerationWizardProps> = ({
  isOpen,
  onClose,
  campaignId,
  campaignName,
  briefing = DEFAULT_BRIEFING,
  onGenerationComplete
}) => {
  const [step, setStep] = useState(1);
  const [generating, setGenerating] = useState(false);
  const [selectedTypes, setSelectedTypes] = useState<ContentType[]>([
    ContentType.blog_post,
    ContentType.social_media_post,
    ContentType.email_campaign
  ]);
  const [deliverableCount, setDeliverableCount] = useState(3);
  const [generationResult, setGenerationResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTypeToggle = (type: ContentType) => {
    setSelectedTypes(prev => 
      prev.includes(type)
        ? prev.filter(t => t !== type)
        : [...prev, type]
    );
  };

  const handleGenerate = async () => {
    try {
      setGenerating(true);
      setError(null);
      
      const request: CampaignGenerationRequest = {
        campaign_id: campaignId,
        briefing,
        deliverable_count: deliverableCount,
        content_types: selectedTypes.length > 0 ? selectedTypes : undefined
      };
      
      const result = await contentDeliverableApi.generateCampaignDeliverables(request);
      setGenerationResult(result);
      setStep(3); // Move to success step
      
      if (onGenerationComplete) {
        onGenerationComplete();
      }
      
    } catch (err) {
      console.error('Generation failed:', err);
      setError('Failed to generate content. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const handleClose = () => {
    setStep(1);
    setGenerationResult(null);
    setError(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            <SparklesIcon className="h-6 w-6 text-blue-600" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Generate Content Deliverables</h2>
              <p className="text-sm text-gray-600">{campaignName || 'Campaign'}</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>

        <div className="p-6">
          {/* Step 1: Configuration */}
          {step === 1 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Configure Content Generation
                </h3>
                <p className="text-sm text-gray-600 mb-6">
                  Select the types of content you want to generate and how many pieces.
                </p>
              </div>

              {/* Content Types Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Content Types
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {Object.values(ContentType).map(type => {
                    const config = CONTENT_TYPE_CONFIG[type];
                    const isSelected = selectedTypes.includes(type);
                    
                    return (
                      <button
                        key={type}
                        onClick={() => handleTypeToggle(type)}
                        className={`p-3 border rounded-lg text-left transition-colors ${
                          isSelected 
                            ? 'border-blue-500 bg-blue-50' 
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{config?.icon}</span>
                          <div className="flex-1">
                            <div className="font-medium text-sm text-gray-900">
                              {config?.label || type}
                            </div>
                            <div className="text-xs text-gray-500">
                              {config?.description}
                            </div>
                          </div>
                          {isSelected && (
                            <CheckIcon className="h-4 w-4 text-blue-600" />
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
                {selectedTypes.length === 0 && (
                  <p className="text-sm text-red-600 mt-2">
                    Please select at least one content type.
                  </p>
                )}
              </div>

              {/* Count Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Content Pieces
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={deliverableCount}
                  onChange={(e) => setDeliverableCount(parseInt(e.target.value) || 1)}
                  className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Generate 1-10 content pieces (recommended: 3-5)
                </p>
              </div>

              {/* Briefing Preview */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Campaign Briefing
                </label>
                <div className="bg-gray-50 p-3 rounded-lg text-sm">
                  <div><strong>Objective:</strong> {briefing.marketing_objective}</div>
                  <div><strong>Audience:</strong> {briefing.target_audience}</div>
                  <div><strong>Tone:</strong> {briefing.desired_tone}</div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex justify-end gap-3 pt-4">
                <button
                  onClick={handleClose}
                  className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => setStep(2)}
                  disabled={selectedTypes.length === 0}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Continue
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Generation in Progress */}
          {step === 2 && (
            <div className="space-y-6">
              <div className="text-center">
                <SparklesIcon className="mx-auto h-12 w-12 text-blue-500 animate-pulse mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {generating ? 'Generating Content...' : 'Ready to Generate'}
                </h3>
                <p className="text-sm text-gray-600 mb-6">
                  {generating 
                    ? 'Our AI agents are creating your content deliverables. This may take a few minutes.'
                    : 'Click Generate to start creating your content deliverables.'
                  }
                </p>
              </div>

              {/* Generation Summary */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">Generation Summary</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• {deliverableCount} content pieces</li>
                  <li>• Types: {selectedTypes.map(t => CONTENT_TYPE_CONFIG[t]?.label).join(', ')}</li>
                  <li>• Campaign: {campaignName}</li>
                </ul>
              </div>

              {error && (
                <div className="bg-red-50 p-4 rounded-lg">
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-between pt-4">
                <button
                  onClick={() => setStep(1)}
                  disabled={generating}
                  className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                >
                  Back
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={generating}
                  className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {generating ? (
                    <>
                      <SparklesIcon className="h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <SparklesIcon className="h-4 w-4" />
                      Generate Content
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Success */}
          {step === 3 && generationResult && (
            <div className="space-y-6">
              <div className="text-center">
                <CheckIcon className="mx-auto h-12 w-12 text-green-500 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Content Generated Successfully!
                </h3>
                <p className="text-sm text-gray-600">
                  {generationResult.generation_summary}
                </p>
              </div>

              {/* Generation Results */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-medium text-green-900 mb-3">Generated Content</h4>
                <div className="space-y-2 text-sm text-green-800">
                  <div>✅ {generationResult.deliverables_created} content pieces created</div>
                  <div>🎭 Narrative theme: {generationResult.narrative_theme}</div>
                  <div>📖 Story flow: {generationResult.story_arc.join(' → ')}</div>
                  <div>🔗 {Object.keys(generationResult.content_relationships).length} content relationships established</div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex justify-center pt-4">
                <button
                  onClick={handleClose}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  View Generated Content
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContentGenerationWizard;