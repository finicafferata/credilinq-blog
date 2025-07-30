import React, { useState } from 'react';
import type { CampaignDetail } from '../lib/api';

interface RepurposeModalProps {
  campaign: CampaignDetail;
  onClose: () => void;
  onRepurpose: (repurposedContent: RepurposedContent) => void;
}

interface RepurposedContent {
  platform: string;
  content: string;
  type: string;
  variations: string[];
}

export function RepurposeModal({ campaign, onClose, onRepurpose }: RepurposeModalProps) {
  const [selectedPlatform, setSelectedPlatform] = useState('linkedin');
  const [contentType, setContentType] = useState('post');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedContent, setGeneratedContent] = useState<RepurposedContent | null>(null);

  const platforms = [
    { id: 'linkedin', name: 'LinkedIn', icon: '💼' },
    { id: 'twitter', name: 'Twitter', icon: '🐦' },
    { id: 'facebook', name: 'Facebook', icon: '📘' },
    { id: 'instagram', name: 'Instagram', icon: '📷' },
    { id: 'tiktok', name: 'TikTok', icon: '🎵' }
  ];

  const contentTypes = [
    { id: 'post', name: 'Post', description: 'Standard social media post' },
    { id: 'thread', name: 'Thread', description: 'Series of connected posts' },
    { id: 'story', name: 'Story', description: 'Short-form content' },
    { id: 'video', name: 'Video Script', description: 'Video content script' },
    { id: 'carousel', name: 'Carousel', description: 'Multi-slide content' }
  ];

  const handleGenerate = async () => {
    setIsGenerating(true);
    
    // Simulate API call
    setTimeout(() => {
      const mockContent: RepurposedContent = {
        platform: selectedPlatform,
        content: `Repurposed content for ${selectedPlatform} - ${contentType}`,
        type: contentType,
        variations: [
          `Variation 1: ${selectedPlatform} ${contentType} content`,
          `Variation 2: Alternative ${selectedPlatform} ${contentType} approach`,
          `Variation 3: Creative ${selectedPlatform} ${contentType} version`
        ]
      };
      
      setGeneratedContent(mockContent);
      setIsGenerating(false);
    }, 2000);
  };

  const handleSave = () => {
    if (generatedContent) {
      onRepurpose(generatedContent);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Repurpose Content</h2>
            <p className="text-gray-600">Create variations for different platforms</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Campaign Info */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-medium text-gray-900 mb-2">Campaign: {campaign.name}</h3>
            <p className="text-sm text-gray-600">Creating repurposed content from this campaign</p>
          </div>

          {/* Platform Selection */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Select Platform</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {platforms.map((platform) => (
                <button
                  key={platform.id}
                  onClick={() => setSelectedPlatform(platform.id)}
                  className={`p-4 border rounded-lg text-center transition-colors ${
                    selectedPlatform === platform.id
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="text-2xl mb-2">{platform.icon}</div>
                  <div className="font-medium">{platform.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Content Type Selection */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Content Type</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {contentTypes.map((type) => (
                <button
                  key={type.id}
                  onClick={() => setContentType(type.id)}
                  className={`p-4 border rounded-lg text-left transition-colors ${
                    contentType === type.id
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">{type.name}</div>
                  <div className="text-sm text-gray-600">{type.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <div className="text-center">
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Generating...
                </div>
              ) : (
                'Generate Repurposed Content'
              )}
            </button>
          </div>

          {/* Generated Content */}
          {generatedContent && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Generated Content</h3>
              
              {/* Main Content */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-2">Main Content</h4>
                <p className="text-gray-700">{generatedContent.content}</p>
              </div>

              {/* Variations */}
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Variations</h4>
                <div className="space-y-3">
                  {generatedContent.variations.map((variation, index) => (
                    <div key={index} className="bg-white border p-4 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-600">Variation {index + 1}</span>
                        <button className="text-blue-600 hover:text-blue-700 text-sm">
                          Copy
                        </button>
                      </div>
                      <p className="text-gray-700">{variation}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-3 pt-4">
                <button
                  onClick={handleSave}
                  className="flex-1 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
                >
                  Save to Campaign
                </button>
                <button
                  onClick={handleGenerate}
                  className="flex-1 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700"
                >
                  Generate More
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 