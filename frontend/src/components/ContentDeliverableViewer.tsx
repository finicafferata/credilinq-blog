import React, { useState, useMemo, useEffect } from 'react';
import { 
  DocumentTextIcon, 
  ChatBubbleLeftRightIcon, 
  EnvelopeIcon,
  PhotoIcon,
  PencilSquareIcon,
  ArrowsRightLeftIcon,
  MagnifyingGlassIcon,
  EyeIcon,
  EyeSlashIcon,
  SparklesIcon,
  ClockIcon,
  UserIcon,
  TagIcon,
  CalendarIcon,
  PresentationChartLineIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';
import { CheckCircleIcon, ExclamationCircleIcon } from '@heroicons/react/24/solid';
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
import { ContentGenerationWizard } from './ContentGenerationWizard';

interface ContentDeliverableViewerProps {
  campaignId: string;
  campaignName?: string;
  onGenerateContent?: () => void;
  onEditDeliverable?: (deliverable: ContentDeliverable) => void;
}

// Helper function to get content type icon
const getContentTypeIcon = (contentType: ContentType) => {
  const iconMap = {
    [ContentType.blog_post]: DocumentTextIcon,
    [ContentType.social_media_post]: ChatBubbleLeftRightIcon,
    [ContentType.email_campaign]: EnvelopeIcon,
    [ContentType.newsletter]: EnvelopeIcon,
    [ContentType.whitepaper]: DocumentTextIcon,
    [ContentType.case_study]: PresentationChartLineIcon,
    [ContentType.video_script]: PhotoIcon,
    [ContentType.podcast_script]: PhotoIcon,
    [ContentType.press_release]: DocumentTextIcon,
    [ContentType.product_description]: TagIcon,
    [ContentType.landing_page]: GlobeAltIcon,
    [ContentType.ad_copy]: SparklesIcon,
    [ContentType.infographic_concept]: PhotoIcon,
    [ContentType.webinar_outline]: PresentationChartLineIcon,
  };
  
  return iconMap[contentType] || DocumentTextIcon;
};

// Helper function to format content for display
const formatContentPreview = (content: string, maxLength: number = 300): string => {
  if (!content) return 'No content available';
  
  // Remove markdown headers and formatting for preview
  const cleanContent = content
    .replace(/#{1,6}\s+/g, '') // Remove markdown headers
    .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold formatting
    .replace(/\*(.*?)\*/g, '$1') // Remove italic formatting
    .replace(/\[(.*?)\]\(.*?\)/g, '$1') // Remove links, keep text
    .replace(/\n\s*\n/g, ' ') // Replace double newlines with space
    .replace(/\n/g, ' ') // Replace single newlines with space
    .trim();
  
  return cleanContent.length > maxLength 
    ? cleanContent.substring(0, maxLength) + '...'
    : cleanContent;
};

// Helper function to extract title from content
const extractTitle = (deliverable: ContentDeliverable): string => {
  if (deliverable.title) return deliverable.title;
  
  // Try to extract title from content
  const lines = deliverable.content.split('\n');
  const firstLine = lines[0]?.trim();
  
  if (firstLine?.startsWith('#')) {
    return firstLine.replace(/^#+\s*/, '');
  }
  
  return `${CONTENT_TYPE_CONFIG[deliverable.content_type]?.label || deliverable.content_type}`;
};

export const ContentDeliverableViewer: React.FC<ContentDeliverableViewerProps> = ({
  campaignId,
  campaignName,
  onGenerateContent,
  onEditDeliverable
}) => {
  const [deliverables, setDeliverables] = useState<ContentDeliverable[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedDeliverables, setExpandedDeliverables] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<ContentType | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<DeliverableStatus | null>(null);
  const [showGenerationWizard, setShowGenerationWizard] = useState(false);

  // Load deliverables
  useEffect(() => {
    loadDeliverables();
  }, [campaignId]);

  const loadDeliverables = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await contentDeliverableApi.getCampaignDeliverables(campaignId);
      setDeliverables(data);
      
    } catch (err) {
      console.error('Error loading deliverables:', err);
      setError('Failed to load content deliverables');
      setDeliverables([]);
    } finally {
      setLoading(false);
    }
  };

  // Filter deliverables based on search and filters
  const filteredDeliverables = useMemo(() => {
    let filtered = deliverables;

    // Apply search filter
    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter(d => 
        extractTitle(d).toLowerCase().includes(search) ||
        formatContentPreview(d.content).toLowerCase().includes(search) ||
        d.key_messages.some(msg => msg.toLowerCase().includes(search)) ||
        d.platform?.toLowerCase().includes(search) ||
        d.target_audience?.toLowerCase().includes(search)
      );
    }

    // Apply type filter
    if (selectedType) {
      filtered = filtered.filter(d => d.content_type === selectedType);
    }

    // Apply status filter
    if (selectedStatus) {
      filtered = filtered.filter(d => d.status === selectedStatus);
    }

    // Sort by narrative order, then by creation date
    return filtered.sort((a, b) => {
      if (a.narrative_order && b.narrative_order) {
        return a.narrative_order - b.narrative_order;
      }
      if (a.narrative_order) return -1;
      if (b.narrative_order) return 1;
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
  }, [deliverables, searchTerm, selectedType, selectedStatus]);

  // Group deliverables by content type for better organization
  const groupedDeliverables = useMemo(() => {
    const groups: Record<ContentType, ContentDeliverable[]> = {} as Record<ContentType, ContentDeliverable[]>;
    
    filteredDeliverables.forEach(deliverable => {
      if (!groups[deliverable.content_type]) {
        groups[deliverable.content_type] = [];
      }
      groups[deliverable.content_type].push(deliverable);
    });
    
    return groups;
  }, [filteredDeliverables]);

  const toggleDeliverableExpansion = (deliverableId: string) => {
    setExpandedDeliverables(prev => {
      const next = new Set(prev);
      if (next.has(deliverableId)) {
        next.delete(deliverableId);
      } else {
        next.add(deliverableId);
      }
      return next;
    });
  };

  const handleUpdateStatus = async (deliverableId: string, newStatus: DeliverableStatus) => {
    try {
      await contentDeliverableApi.updateDeliverableStatus(deliverableId, newStatus);
      await loadDeliverables(); // Refresh the data
    } catch (err) {
      console.error('Error updating status:', err);
    }
  };

  // Statistics
  const stats = useMemo(() => {
    const total = deliverables.length;
    const published = deliverables.filter(d => d.status === DeliverableStatus.published).length;
    const approved = deliverables.filter(d => d.status === DeliverableStatus.approved).length;
    const draft = deliverables.filter(d => d.status === DeliverableStatus.draft).length;
    const totalWordCount = deliverables.reduce((sum, d) => sum + (d.word_count || 0), 0);
    
    return { total, published, approved, draft, totalWordCount };
  }, [deliverables]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <SparklesIcon className="mx-auto h-12 w-12 text-blue-400 animate-pulse" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">Loading Content Deliverables</h3>
          <p className="mt-1 text-sm text-gray-500">Fetching your generated content...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <ExclamationCircleIcon className="mx-auto h-12 w-12 text-red-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">Error Loading Content</h3>
          <p className="mt-1 text-sm text-gray-500">{error}</p>
          <button
            onClick={loadDeliverables}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Statistics */}
      <div className="border-b border-gray-200 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <SparklesIcon className="h-6 w-6 text-blue-600" />
              Content Deliverables - {campaignName || 'Campaign'}
            </h2>
            <p className="mt-1 text-sm text-gray-600">
              Ready-to-use content pieces generated for your campaign
            </p>
          </div>
          
          <button
            onClick={() => setShowGenerationWizard(true)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <SparklesIcon className="h-4 w-4" />
            Generate Content
          </button>
        </div>
        
        {/* Statistics */}
        <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
            <div className="text-sm text-blue-800">Total Pieces</div>
          </div>
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{stats.published}</div>
            <div className="text-sm text-green-800">Published</div>
          </div>
          <div className="bg-purple-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{stats.approved}</div>
            <div className="text-sm text-purple-800">Approved</div>
          </div>
          <div className="bg-yellow-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">{stats.draft}</div>
            <div className="text-sm text-yellow-800">Drafts</div>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-gray-600">
              {stats.totalWordCount.toLocaleString()}
            </div>
            <div className="text-sm text-gray-800">Total Words</div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="space-y-4">
        {/* Search */}
        <div className="max-w-md">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search content, messages, audience..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>

        {/* Type and Status Filters */}
        <div className="flex flex-wrap gap-2">
          <select
            value={selectedType || ''}
            onChange={(e) => setSelectedType(e.target.value as ContentType || null)}
            className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
          >
            <option value="">All Content Types</option>
            {Object.values(ContentType).map(type => (
              <option key={type} value={type}>
                {CONTENT_TYPE_CONFIG[type]?.label || type}
              </option>
            ))}
          </select>
          
          <select
            value={selectedStatus || ''}
            onChange={(e) => setSelectedStatus(e.target.value as DeliverableStatus || null)}
            className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
          >
            <option value="">All Statuses</option>
            {Object.values(DeliverableStatus).map(status => (
              <option key={status} value={status}>
                {STATUS_CONFIG[status]?.label || status}
              </option>
            ))}
          </select>
          
          {(selectedType || selectedStatus || searchTerm) && (
            <button
              onClick={() => {
                setSelectedType(null);
                setSelectedStatus(null);
                setSearchTerm('');
              }}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900"
            >
              Clear Filters
            </button>
          )}
        </div>
      </div>

      {/* Content Display */}
      {deliverables.length === 0 ? (
        <div className="text-center py-12">
          <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No Content Generated Yet</h3>
          <p className="mt-1 text-sm text-gray-500">
            Generate content deliverables to see your campaign content here.
          </p>
          <button
            onClick={() => setShowGenerationWizard(true)}
            className="mt-4 flex items-center gap-2 mx-auto px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <SparklesIcon className="h-4 w-4" />
            Generate Content
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedDeliverables).map(([contentType, typeDeliverables]) => {
            const config = CONTENT_TYPE_CONFIG[contentType as ContentType];
            const Icon = getContentTypeIcon(contentType as ContentType);
            
            return (
              <div key={contentType} className="bg-white border border-gray-200 rounded-lg">
                <div className={`px-4 py-3 border-b border-gray-200 bg-${config?.color || 'gray'}-50`}>
                  <div className="flex items-center space-x-2">
                    <Icon className={`h-5 w-5 text-${config?.color || 'gray'}-600`} />
                    <h3 className="font-medium text-gray-900">
                      {config?.label || contentType} ({typeDeliverables.length})
                    </h3>
                  </div>
                  {config?.description && (
                    <p className="text-sm text-gray-600 mt-1">{config.description}</p>
                  )}
                </div>

                <div className="divide-y divide-gray-200">
                  {typeDeliverables.map((deliverable) => {
                    const isExpanded = expandedDeliverables.has(deliverable.id);
                    const title = extractTitle(deliverable);
                    const preview = formatContentPreview(deliverable.content);
                    const statusConfig = STATUS_CONFIG[deliverable.status];
                    
                    return (
                      <div key={deliverable.id} className="p-4 hover:bg-gray-50">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            {/* Title and Status */}
                            <div className="flex items-center gap-3 mb-2">
                              <h4 className="font-medium text-gray-900 truncate">
                                {title}
                              </h4>
                              <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusConfig.color}`}>
                                {statusConfig.icon} {statusConfig.label}
                              </span>
                              {deliverable.narrative_order && (
                                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                                  #{deliverable.narrative_order}
                                </span>
                              )}
                            </div>
                            
                            {/* Metadata */}
                            <div className="flex flex-wrap gap-4 text-xs text-gray-500 mb-3">
                              {deliverable.platform && (
                                <div className="flex items-center gap-1">
                                  <GlobeAltIcon className="h-3 w-3" />
                                  {deliverable.platform}
                                </div>
                              )}
                              {deliverable.word_count && (
                                <div className="flex items-center gap-1">
                                  <DocumentTextIcon className="h-3 w-3" />
                                  {deliverable.word_count.toLocaleString()} words
                                </div>
                              )}
                              {deliverable.reading_time && (
                                <div className="flex items-center gap-1">
                                  <ClockIcon className="h-3 w-3" />
                                  {deliverable.reading_time} min read
                                </div>
                              )}
                              {deliverable.created_by && (
                                <div className="flex items-center gap-1">
                                  <UserIcon className="h-3 w-3" />
                                  {deliverable.created_by}
                                </div>
                              )}
                              <div className="flex items-center gap-1">
                                <CalendarIcon className="h-3 w-3" />
                                {new Date(deliverable.created_at).toLocaleDateString()}
                              </div>
                            </div>
                            
                            {/* Key Messages */}
                            {deliverable.key_messages.length > 0 && (
                              <div className="mb-3">
                                <div className="text-xs text-gray-500 mb-1">Key Messages:</div>
                                <div className="flex flex-wrap gap-1">
                                  {deliverable.key_messages.map((message, idx) => (
                                    <span
                                      key={idx}
                                      className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-blue-100 text-blue-800"
                                    >
                                      <TagIcon className="h-3 w-3 mr-1" />
                                      {message}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {/* Content Preview/Full */}
                            <div className="mt-3">
                              <div className={`prose prose-sm max-w-none ${isExpanded ? '' : 'line-clamp-3'}`}>
                                {isExpanded ? (
                                  <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                                    {deliverable.content}
                                  </div>
                                ) : (
                                  <p className="text-gray-700">{preview}</p>
                                )}
                              </div>
                            </div>
                            
                            {/* Actions */}
                            <div className="mt-4 flex items-center gap-2">
                              <button
                                onClick={() => toggleDeliverableExpansion(deliverable.id)}
                                className="flex items-center gap-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900"
                              >
                                {isExpanded ? (
                                  <>
                                    <EyeSlashIcon className="h-4 w-4" />
                                    Show Less
                                  </>
                                ) : (
                                  <>
                                    <EyeIcon className="h-4 w-4" />
                                    Read Full Content
                                  </>
                                )}
                              </button>
                              
                              {onEditDeliverable && (
                                <button
                                  onClick={() => onEditDeliverable(deliverable)}
                                  className="flex items-center gap-1 px-3 py-1 text-sm text-blue-600 hover:text-blue-900"
                                >
                                  <PencilSquareIcon className="h-4 w-4" />
                                  Edit
                                </button>
                              )}
                              
                              {/* Status Update Buttons */}
                              {deliverable.status === DeliverableStatus.draft && (
                                <button
                                  onClick={() => handleUpdateStatus(deliverable.id, DeliverableStatus.in_review)}
                                  className="flex items-center gap-1 px-3 py-1 text-sm text-yellow-600 hover:text-yellow-900"
                                >
                                  Submit for Review
                                </button>
                              )}
                              
                              {deliverable.status === DeliverableStatus.in_review && (
                                <button
                                  onClick={() => handleUpdateStatus(deliverable.id, DeliverableStatus.approved)}
                                  className="flex items-center gap-1 px-3 py-1 text-sm text-green-600 hover:text-green-900"
                                >
                                  <CheckCircleIcon className="h-4 w-4" />
                                  Approve
                                </button>
                              )}
                              
                              {deliverable.status === DeliverableStatus.approved && (
                                <button
                                  onClick={() => handleUpdateStatus(deliverable.id, DeliverableStatus.published)}
                                  className="flex items-center gap-1 px-3 py-1 text-sm text-blue-600 hover:text-blue-900"
                                >
                                  <GlobeAltIcon className="h-4 w-4" />
                                  Publish
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* No Results Message */}
      {deliverables.length > 0 && filteredDeliverables.length === 0 && (
        <div className="text-center py-12">
          <MagnifyingGlassIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No Content Found</h3>
          <p className="mt-1 text-sm text-gray-500">
            Try adjusting your search terms or filters.
          </p>
        </div>
      )}

      {/* Content Generation Wizard */}
      <ContentGenerationWizard
        isOpen={showGenerationWizard}
        onClose={() => setShowGenerationWizard(false)}
        campaignId={campaignId}
        campaignName={campaignName}
        onGenerationComplete={() => {
          setShowGenerationWizard(false);
          loadDeliverables(); // Refresh the data
        }}
      />
    </div>
  );
};

export default ContentDeliverableViewer;