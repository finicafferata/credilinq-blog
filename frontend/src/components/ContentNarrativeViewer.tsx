import React, { useState, useEffect } from 'react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  EyeIcon,
  DocumentTextIcon,
  CalendarIcon,
  UserIcon,
  ClockIcon,
  TagIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import type { 
  ContentDeliverable
} from '../types/contentTypes';
import { 
  contentDeliverableApi,
  ContentType,
  DeliverableStatus,
  CONTENT_TYPE_CONFIG,
  STATUS_CONFIG
} from '../services/contentDeliverableApi';

interface ContentNarrativeViewerProps {
  campaignId: string;
  campaignName?: string;
}

interface GroupedContent {
  [contentType: string]: ContentDeliverable[];
}

export const ContentNarrativeViewer: React.FC<ContentNarrativeViewerProps> = ({
  campaignId,
  campaignName
}) => {
  const [content, setContent] = useState<ContentDeliverable[]>([]);
  const [groupedContent, setGroupedContent] = useState<GroupedContent>({});
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set(['blog_post']));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    fetchCampaignContent();
  }, [campaignId]);

  const fetchCampaignContent = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const deliverables = await contentDeliverableApi.getCampaignDeliverables(campaignId);
      setContent(deliverables);

      // Group content by type for organized display
      const grouped = deliverables.reduce((acc: GroupedContent, item) => {
        const type = item.content_type;
        if (!acc[type]) {
          acc[type] = [];
        }
        acc[type].push(item);
        return acc;
      }, {});

      // Sort within each group by narrative_order
      Object.keys(grouped).forEach(type => {
        grouped[type].sort((a, b) => (a.narrative_order || 0) - (b.narrative_order || 0));
      });

      setGroupedContent(grouped);
    } catch (err) {
      console.error('Failed to fetch campaign content:', err);
      setError('Failed to load campaign content');
    } finally {
      setLoading(false);
    }
  };

  const toggleExpanded = (itemId: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(itemId)) {
      newExpanded.delete(itemId);
    } else {
      newExpanded.add(itemId);
    }
    setExpandedItems(newExpanded);
  };

  const toggleTypeExpanded = (type: string) => {
    const newExpanded = new Set(expandedTypes);
    if (newExpanded.has(type)) {
      newExpanded.delete(type);
    } else {
      newExpanded.add(type);
    }
    setExpandedTypes(newExpanded);
  };

  const formatContent = (content: string, maxLength: number = 300) => {
    if (!content) return 'No content available';
    
    // Strip markdown formatting for preview
    const stripped = content
      .replace(/^#{1,6}\s+/gm, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italics
      .replace(/\[(.*?)\]\(.*?\)/g, '$1') // Remove links
      .replace(/```[\s\S]*?```/g, '[Code Block]') // Replace code blocks
      .trim();
    
    return stripped.length > maxLength ? 
      stripped.substring(0, maxLength) + '...' : 
      stripped;
  };

  const getWordCount = (content: string) => {
    return content.split(/\s+/).filter(word => word.length > 0).length;
  };

  const getEstimatedReadingTime = (wordCount: number) => {
    // Average reading speed: 200 words per minute
    const minutes = Math.ceil(wordCount / 200);
    return minutes === 1 ? '1 min read' : `${minutes} min read`;
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded mb-4 w-1/3"></div>
          <div className="space-y-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="border rounded-lg p-4">
                <div className="h-4 bg-gray-200 rounded mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center text-red-600">
          <DocumentTextIcon className="mx-auto h-12 w-12 mb-4" />
          <p>{error}</p>
          <button 
            onClick={fetchCampaignContent}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (content.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center text-gray-500 max-w-md mx-auto">
          <DocumentTextIcon className="mx-auto h-12 w-12 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Content Deliverables</h3>
          <p className="text-sm mb-6">This campaign has completed tasks but no content deliverables. Choose how you'd like to proceed:</p>
          
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}
          
          <div className="space-y-3">
            <button
              onClick={async () => {
                setGenerating(true);
                setError(null);
                
                try {
                  const result = await contentDeliverableApi.generateCampaignDeliverables({
                    campaign_id: campaignId,
                    briefing: {
                      marketing_objective: "Generate high-quality content deliverables",
                      target_audience: "Business professionals and decision makers",
                      channels: ["blog", "linkedin", "email"],
                      desired_tone: "professional",
                      company_context: "AI-powered content marketing platform"
                    },
                    deliverable_count: 5,
                    content_types: ['blog_post', 'social_media_post', 'email_campaign']
                  });
                  
                  console.log('Content generated:', result);
                  
                  // Refresh the content list
                  await fetchCampaignContent();
                  
                } catch (err) {
                  console.error('Content generation failed:', err);
                  setError('Failed to generate content. Please try again.');
                } finally {
                  setGenerating(false);
                }
              }}
              disabled={generating}
              className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 justify-center disabled:opacity-50"
            >
              {generating ? (
                <>
                  <SparklesIcon className="h-5 w-5 animate-spin" />
                  Generating Content...
                </>
              ) : (
                <>
                  <SparklesIcon className="h-5 w-5" />
                  Generate New Content Deliverables
                </>
              )}
            </button>
            
            <button
              onClick={async () => {
                try {
                  // Call migration API
                  alert('🔄 Task migration feature coming soon!\n\nThis will convert your existing 21 completed tasks into viewable content deliverables while preserving all the generated content.');
                } catch (err) {
                  alert('Migration failed. Please try again.');
                }
              }}
              className="w-full px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2 justify-center border border-gray-300"
            >
              <DocumentTextIcon className="h-5 w-5" />
              Convert Existing Tasks to Content
            </button>
          </div>
          
          <p className="text-xs text-gray-400 mt-4">
            💡 New generation creates fresh content with narrative flow, while conversion preserves existing task content.
          </p>
        </div>
      </div>
    );
  }

  const totalPieces = content.length;
  const publishedPieces = content.filter(item => item.status === 'published').length;
  const draftPieces = content.filter(item => item.status === 'draft').length;
  const reviewPieces = content.filter(item => item.status === 'in_review' || item.status === 'needs_revision').length;

  return (
    <div className="space-y-6">
      {/* Campaign Content Summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Campaign Content</h2>
            <p className="text-gray-600">{campaignName || `Campaign ${campaignId}`}</p>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{totalPieces}</div>
              <div className="text-gray-500">Total Pieces</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{publishedPieces}</div>
              <div className="text-gray-500">Published</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">{reviewPieces}</div>
              <div className="text-gray-500">In Review</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">{draftPieces}</div>
              <div className="text-gray-500">Draft</div>
            </div>
          </div>
        </div>

        {/* Content Type Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {Object.entries(groupedContent).map(([type, items]) => {
            const config = CONTENT_TYPE_CONFIG[type as ContentType] || {
              label: type,
              icon: '📄',
              color: 'gray'
            };
            return (
              <div key={type} className="bg-gray-50 rounded-lg p-3 text-center">
                <div className="text-2xl mb-1">{config.icon}</div>
                <div className="font-semibold text-lg">{items.length}</div>
                <div className="text-sm text-gray-600">{config.label}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Content by Type */}
      {Object.entries(groupedContent).map(([contentType, items]) => {
        const config = CONTENT_TYPE_CONFIG[contentType as ContentType] || {
          label: contentType,
          icon: '📄',
          color: 'gray',
          description: ''
        };
        const isTypeExpanded = expandedTypes.has(contentType);

        return (
          <div key={contentType} className="bg-white rounded-lg shadow">
            {/* Content Type Header */}
            <div 
              className="flex items-center justify-between p-6 border-b cursor-pointer hover:bg-gray-50"
              onClick={() => toggleTypeExpanded(contentType)}
            >
              <div className="flex items-center gap-3">
                <span className="text-2xl">{config.icon}</span>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    {config.label} ({items.length})
                  </h3>
                  <p className="text-sm text-gray-600">{config.description}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">
                  {isTypeExpanded ? 'Collapse' : 'Expand'}
                </span>
                {isTypeExpanded ? (
                  <ChevronDownIcon className="h-5 w-5 text-gray-400" />
                ) : (
                  <ChevronRightIcon className="h-5 w-5 text-gray-400" />
                )}
              </div>
            </div>

            {/* Content Items */}
            {isTypeExpanded && (
              <div className="divide-y">
                {items.map((item, index) => {
                  const isExpanded = expandedItems.has(item.id);
                  const wordCount = getWordCount(item.content);
                  const readingTime = getEstimatedReadingTime(wordCount);
                  const statusConfig = STATUS_CONFIG[item.status as DeliverableStatus] || {
                    label: item.status,
                    color: 'bg-gray-100 text-gray-800',
                    icon: '❓'
                  };

                  return (
                    <div key={item.id} className="p-6">
                      {/* Content Header */}
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h4 className="text-lg font-medium text-gray-900">
                              {item.title || `Untitled ${config.label}`}
                            </h4>
                            <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${statusConfig.color}`}>
                              <span className="mr-1">{statusConfig.icon}</span>
                              {statusConfig.label}
                            </span>
                          </div>
                          
                          {/* Content Metadata */}
                          <div className="flex items-center gap-4 text-sm text-gray-500 mb-3">
                            <div className="flex items-center gap-1">
                              <ClockIcon className="h-4 w-4" />
                              {readingTime}
                            </div>
                            <div className="flex items-center gap-1">
                              <DocumentTextIcon className="h-4 w-4" />
                              {wordCount.toLocaleString()} words
                            </div>
                            {item.target_audience && (
                              <div className="flex items-center gap-1">
                                <UserIcon className="h-4 w-4" />
                                {item.target_audience}
                              </div>
                            )}
                            {item.platform && (
                              <div className="flex items-center gap-1">
                                <TagIcon className="h-4 w-4" />
                                {item.platform}
                              </div>
                            )}
                          </div>

                          {/* Content Preview/Full */}
                          {isExpanded ? (
                            <div className="prose max-w-none">
                              <div className="bg-gray-50 p-4 rounded-lg">
                                <pre className="whitespace-pre-wrap font-sans text-gray-800 text-sm leading-relaxed">
                                  {item.content}
                                </pre>
                              </div>
                              {item.summary && (
                                <div className="mt-4 p-3 bg-blue-50 rounded border-l-4 border-blue-400">
                                  <h5 className="text-sm font-medium text-blue-900 mb-1">Summary</h5>
                                  <p className="text-sm text-blue-800">{item.summary}</p>
                                </div>
                              )}
                              {item.key_messages && item.key_messages.length > 0 && (
                                <div className="mt-4">
                                  <h5 className="text-sm font-medium text-gray-900 mb-2">Key Messages</h5>
                                  <ul className="text-sm text-gray-700 space-y-1">
                                    {item.key_messages.map((message, i) => (
                                      <li key={i} className="flex items-start gap-2">
                                        <span className="text-blue-500 mt-1">•</span>
                                        {message}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="text-gray-700 text-sm leading-relaxed">
                              {formatContent(item.content)}
                            </div>
                          )}

                          {/* Narrative Connection */}
                          {index < items.length - 1 && (
                            <div className="mt-4 pt-4 border-t border-gray-100">
                              <p className="text-xs text-blue-600 italic">
                                → Continues in "{items[index + 1]?.title || `Next ${config.label}`}"
                              </p>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Expand/Collapse Button */}
                      <button
                        onClick={() => toggleExpanded(item.id)}
                        className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
                      >
                        <EyeIcon className="h-4 w-4" />
                        {isExpanded ? 'Show Less' : 'Read Full Content'}
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ContentNarrativeViewer;