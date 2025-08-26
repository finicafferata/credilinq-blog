import React, { useState, useEffect, useRef } from 'react';
import { 
  CheckCircleIcon, 
  XCircleIcon, 
  PencilSquareIcon,
  EyeIcon,
  ClockIcon,
  StarIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  ExclamationTriangleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  DocumentTextIcon,
  SparklesIcon,
  ArrowTrendingUpIcon,
  ChartBarIcon,
  Squares2X2Icon,
  ListBulletIcon
} from '@heroicons/react/24/outline';
import { CheckCircleIcon as CheckCircleSolid, StarIcon as StarSolid } from '@heroicons/react/24/solid';
import toast from 'react-hot-toast';
import { contentWorkflowApi } from '../services/contentWorkflowApi';

// Local type definitions to avoid import issues
interface GeneratedContent {
  content_id: string;
  task_id: string;
  content_type: string;
  channel: string;
  title: string;
  content: string;
  word_count: number;
  quality_score: number;
  seo_score?: number;
  estimated_engagement: string;
  metadata: Record<string, any>;
  created_at: string;
}

interface ContentApprovalRequest {
  content_id: string;
  task_id: string;
  approved: boolean;
  feedback?: string;
  reviewer_id?: string;
  quality_rating?: number;
}

interface ContentRevisionRequest {
  content_id: string;
  task_id: string;
  revision_notes: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

interface PendingApproval {
  content_id: string;
  task_id: string;
  campaign_id: string;
  content_type: string;
  title: string;
  quality_score: number;
  created_at: string;
  reviewer_assigned?: string;
  priority: string;
}
// import BusinessContextPanel from '../components/BusinessContextPanel';
// import BrandConsistencyChecker from '../components/BrandConsistencyChecker';

interface ContentReviewItem extends GeneratedContent {
  campaign_name?: string;
  deadline?: string;
  reviewer_assigned?: string;
  priority_level?: 'low' | 'medium' | 'high' | 'urgent';
}

interface FilterOptions {
  campaign?: string;
  contentType?: string;
  priority?: string;
  qualityRange?: [number, number];
  dateRange?: string;
}

const PRIORITY_COLORS = {
  urgent: 'bg-red-100 text-red-800 border-red-200',
  high: 'bg-orange-100 text-orange-800 border-orange-200',
  medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  low: 'bg-green-100 text-green-800 border-green-200'
};

const CONTENT_TYPE_ICONS = {
  blog_post: DocumentTextIcon,
  social_post: SparklesIcon,
  email_content: '📧',
  linkedin_article: '💼',
  twitter_thread: '🐦',
  case_study: '📊',
  newsletter: '📬'
};

const ContentReviewApproval: React.FC = () => {
  const [contentItems, setContentItems] = useState<ContentReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedContent, setSelectedContent] = useState<ContentReviewItem | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [filters, setFilters] = useState<FilterOptions>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'created_at' | 'quality_score' | 'priority'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'dashboard' | 'list'>('dashboard');
  const [showBusinessContext, setShowBusinessContext] = useState(false);
  
  // Review state
  const [reviewComment, setReviewComment] = useState('');
  const [qualityRating, setQualityRating] = useState(0);
  const [revisionNotes, setRevisionNotes] = useState('');
  const [showRevisionModal, setShowRevisionModal] = useState(false);
  const [processingApproval, setProcessingApproval] = useState(false);

  // Stats
  const [stats, setStats] = useState({
    total: 0,
    pendingReview: 0,
    approved: 0,
    needsRevision: 0,
    averageQuality: 0
  });

  const previewRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadContentForReview();
  }, []);

  const loadContentForReview = async () => {
    try {
      setLoading(true);
      
      // Get all campaigns and their content (in a real implementation, this would be more efficient)
      // For now, we'll simulate getting content that needs review
      const mockContentItems: ContentReviewItem[] = [
        {
          content_id: '1',
          task_id: 'task_1',
          content_type: 'blog_post',
          channel: 'blog',
          title: 'The Future of Fintech: AI-Driven Credit Solutions',
          content: `# The Future of Fintech: AI-Driven Credit Solutions

The financial technology landscape is rapidly evolving, with artificial intelligence at the forefront of innovation. As we look toward the future, AI-driven credit solutions are reshaping how businesses and individuals access funding.

## Key Innovations in AI Credit Solutions

1. **Automated Risk Assessment**: Machine learning algorithms analyze vast amounts of data to provide more accurate risk assessments
2. **Real-time Decision Making**: AI enables instant credit decisions, reducing wait times from weeks to minutes
3. **Alternative Data Sources**: Beyond traditional credit scores, AI leverages social media, transaction history, and behavioral patterns

## The CrediLinq Advantage

At CrediLinq, we're pioneering next-generation AI solutions that make credit more accessible and fair. Our proprietary algorithms consider hundreds of variables to provide comprehensive credit assessments.

*Ready to transform your lending process? Contact our team today.*`,
          word_count: 189,
          quality_score: 8.7,
          seo_score: 85,
          estimated_engagement: 'High',
          metadata: {
            target_keywords: ['fintech', 'AI credit', 'lending technology'],
            readability_score: 'Good',
            brand_consistency: 95
          },
          created_at: '2025-01-19T14:30:00Z',
          campaign_name: 'Q1 Thought Leadership',
          deadline: '2025-01-20T18:00:00Z',
          priority_level: 'high'
        },
        {
          content_id: '2',
          task_id: 'task_2',
          content_type: 'linkedin_article',
          channel: 'linkedin',
          title: '5 Ways AI is Transforming Small Business Lending',
          content: `Small business lending is experiencing a revolution thanks to AI technology. Here are 5 key transformations:

🚀 **Faster Approvals**: What used to take weeks now happens in hours
📊 **Better Risk Analysis**: AI looks at 1000+ data points vs traditional 10-20
💡 **Personalized Solutions**: Custom loan products based on business needs
🎯 **Fraud Prevention**: Advanced detection algorithms protect both lenders and borrowers
📈 **Improved Success Rates**: Better matching leads to higher loan success rates

At CrediLinq, we're at the forefront of this transformation. Our AI-powered platform has helped thousands of businesses access the funding they need to grow.

#Fintech #AILending #SmallBusiness #CrediLinq`,
          word_count: 134,
          quality_score: 9.1,
          seo_score: 78,
          estimated_engagement: 'Very High',
          metadata: {
            platform_optimized: true,
            hashtags: ['#Fintech', '#AILending', '#SmallBusiness'],
            cta_included: true
          },
          created_at: '2025-01-19T15:45:00Z',
          campaign_name: 'Social Media Boost',
          deadline: '2025-01-21T12:00:00Z',
          priority_level: 'urgent'
        },
        {
          content_id: '3',
          task_id: 'task_3',
          content_type: 'email_content',
          channel: 'email',
          title: 'Your Credit Application Status Update',
          content: `Subject: Great news about your credit application!

Hi [First Name],

We have exciting news about your recent credit application with CrediLinq. Our AI-powered review process has completed its analysis, and we're pleased to move forward with the next steps.

**What happens next:**
• Review your personalized credit terms (attached)
• Schedule a brief consultation call
• Complete final documentation
• Receive your funding (typically within 48 hours)

Our team will reach out within the next business day to guide you through the process. If you have any immediate questions, don't hesitate to reply to this email or call us at (555) 123-4567.

Thank you for choosing CrediLinq for your financing needs.

Best regards,
The CrediLinq Team

P.S. Check out our latest blog post on AI in lending: [blog link]`,
          word_count: 156,
          quality_score: 7.8,
          seo_score: 65,
          estimated_engagement: 'Medium',
          metadata: {
            personalization_fields: ['First Name'],
            cta_count: 2,
            tone: 'Professional, Friendly'
          },
          created_at: '2025-01-19T16:20:00Z',
          campaign_name: 'Customer Onboarding',
          deadline: '2025-01-22T09:00:00Z',
          priority_level: 'medium'
        }
      ];

      setContentItems(mockContentItems);
      
      // Calculate stats
      setStats({
        total: mockContentItems.length,
        pendingReview: mockContentItems.length,
        approved: 0,
        needsRevision: 0,
        averageQuality: mockContentItems.reduce((sum, item) => sum + item.quality_score, 0) / mockContentItems.length
      });

    } catch (error) {
      console.error('Error loading content for review:', error);
      toast.error('Failed to load content for review');
    } finally {
      setLoading(false);
    }
  };

  const handleApproval = async (contentItem: ContentReviewItem, approved: boolean) => {
    if (!contentItem) return;

    try {
      setProcessingApproval(true);

      const approvalRequest: ContentApprovalRequest = {
        content_id: contentItem.content_id,
        task_id: contentItem.task_id,
        approved,
        feedback: reviewComment,
        quality_rating: qualityRating || undefined,
        reviewer_id: 'current-user-id' // This would come from auth context
      };

      await contentWorkflowApi.approveContent(approvalRequest);

      // Update local state
      setContentItems(prev => prev.filter(item => item.content_id !== contentItem.content_id));
      
      // Update stats
      setStats(prev => ({
        ...prev,
        total: prev.total - 1,
        pendingReview: prev.pendingReview - 1,
        approved: approved ? prev.approved + 1 : prev.approved,
        needsRevision: !approved ? prev.needsRevision + 1 : prev.needsRevision
      }));

      toast.success(approved ? 'Content approved successfully!' : 'Content marked for revision');
      
      // Reset form
      setReviewComment('');
      setQualityRating(0);
      setSelectedContent(null);
      setShowPreview(false);

    } catch (error) {
      console.error('Error processing approval:', error);
      toast.error('Failed to process approval');
    } finally {
      setProcessingApproval(false);
    }
  };

  const handleRevisionRequest = async () => {
    if (!selectedContent || !revisionNotes.trim()) return;

    try {
      const revisionRequest: ContentRevisionRequest = {
        content_id: selectedContent.content_id,
        task_id: selectedContent.task_id,
        revision_notes: revisionNotes,
        priority: selectedContent.priority_level || 'medium'
      };

      await contentWorkflowApi.requestContentRevision(revisionRequest);

      // Update local state
      setContentItems(prev => prev.filter(item => item.content_id !== selectedContent.content_id));
      setStats(prev => ({
        ...prev,
        total: prev.total - 1,
        pendingReview: prev.pendingReview - 1,
        needsRevision: prev.needsRevision + 1
      }));

      toast.success('Revision requested successfully!');
      
      // Reset and close modal
      setRevisionNotes('');
      setShowRevisionModal(false);
      setSelectedContent(null);
      setShowPreview(false);

    } catch (error) {
      console.error('Error requesting revision:', error);
      toast.error('Failed to request revision');
    }
  };

  const filteredAndSortedContent = contentItems
    .filter(item => {
      if (searchQuery && !item.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !item.content.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      
      if (filters.campaign && item.campaign_name !== filters.campaign) return false;
      if (filters.contentType && item.content_type !== filters.contentType) return false;
      if (filters.priority && item.priority_level !== filters.priority) return false;
      
      if (filters.qualityRange) {
        const [min, max] = filters.qualityRange;
        if (item.quality_score < min || item.quality_score > max) return false;
      }

      return true;
    })
    .sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'quality_score':
          aValue = a.quality_score;
          bValue = b.quality_score;
          break;
        case 'priority':
          const priorityOrder = { urgent: 4, high: 3, medium: 2, low: 1 };
          aValue = priorityOrder[a.priority_level || 'medium'];
          bValue = priorityOrder[b.priority_level || 'medium'];
          break;
        default: // created_at
          aValue = new Date(a.created_at).getTime();
          bValue = new Date(b.created_at).getTime();
      }

      return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
    });

  const toggleItemExpansion = (contentId: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(contentId)) {
        newSet.delete(contentId);
      } else {
        newSet.add(contentId);
      }
      return newSet;
    });
  };

  const getQualityColor = (score: number) => {
    if (score >= 9) return 'text-green-600';
    if (score >= 7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityBadge = (score: number) => {
    if (score >= 9) return 'bg-green-100 text-green-800';
    if (score >= 7) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading content for review...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/30">
      {/* Modern Header */}
      <div className="bg-white/80 backdrop-blur-sm shadow-sm border-b border-gray-200/50 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div>
              <div className="flex items-center space-x-3 mb-2">
                <div className="p-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl shadow-lg">
                  <DocumentTextIcon className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Content Review & Approval
                </h1>
              </div>
              <p className="text-gray-600 ml-12">Review and approve AI-generated content for campaigns with intelligent analysis</p>
            </div>
            
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              {/* View Mode Toggle */}
              <div className="flex items-center bg-gray-100/80 backdrop-blur-sm p-1.5 rounded-xl shadow-sm">
                <button
                  onClick={() => setViewMode('dashboard')}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                    viewMode === 'dashboard' 
                      ? 'bg-white text-gray-900 shadow-md transform scale-105' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                  }`}
                >
                  <ChartBarIcon className="h-4 w-4" />
                  Dashboard
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                    viewMode === 'list' 
                      ? 'bg-white text-gray-900 shadow-md transform scale-105' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                  }`}
                >
                  <ListBulletIcon className="h-4 w-4" />
                  Review List
                </button>
              </div>
              
              {/* Enhanced Stats Cards */}
              <div className="flex gap-3">
                <div className="group bg-gradient-to-r from-blue-500 to-blue-600 p-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{stats.total}</div>
                    <div className="text-xs text-blue-100 font-medium">Pending Review</div>
                  </div>
                </div>
                <div className="group bg-gradient-to-r from-emerald-500 to-emerald-600 p-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{stats.approved}</div>
                    <div className="text-xs text-emerald-100 font-medium">Approved</div>
                  </div>
                </div>
                <div className="group bg-gradient-to-r from-amber-500 to-orange-500 p-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{stats.averageQuality.toFixed(1)}</div>
                    <div className="text-xs text-orange-100 font-medium">Avg Quality</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {viewMode === 'dashboard' ? (
          <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200/50 p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Content Review Dashboard</h2>
            <p className="text-gray-600 mb-6">Analytics and insights for content review workflow</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-6 rounded-xl text-white">
                <h3 className="text-lg font-semibold mb-2">Pending Reviews</h3>
                <p className="text-3xl font-bold">3</p>
              </div>
              <div className="bg-gradient-to-r from-emerald-500 to-emerald-600 p-6 rounded-xl text-white">
                <h3 className="text-lg font-semibold mb-2">Approved Today</h3>
                <p className="text-3xl font-bold">8</p>
              </div>
              <div className="bg-gradient-to-r from-amber-500 to-orange-500 p-6 rounded-xl text-white">
                <h3 className="text-lg font-semibold mb-2">Avg Quality</h3>
                <p className="text-3xl font-bold">8.4</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex gap-6 relative">
            {/* Business Context Panel - Temporarily disabled */}
            {/* {showBusinessContext && (
              <div className="absolute left-0 top-0 z-10">
                <BusinessContextPanel
                  isVisible={showBusinessContext}
                  onToggle={() => setShowBusinessContext(!showBusinessContext)}
                />
              </div>
            )} */}
            
            {/* Main Content List */}
            <div className={`flex-1 transition-all duration-300 ${showBusinessContext ? 'ml-84' : ''}`}>
            {/* Enhanced Filters and Search */}
            <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200/50 mb-8 p-6">
              <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
                <div className="flex flex-col sm:flex-row flex-1 gap-4 items-start sm:items-center">
                  <div className="flex-1 min-w-64 max-w-md">
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
                      </div>
                      <input
                        type="text"
                        placeholder="Search content by title or keywords..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="block w-full pl-10 pr-4 py-3 border border-gray-300/50 rounded-xl bg-white/70 backdrop-blur-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm transition-all duration-200 text-sm"
                      />
                    </div>
                  </div>
                  
                  <div className="flex gap-3">
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value as any)}
                      className="px-4 py-3 border border-gray-300/50 rounded-xl bg-white/70 backdrop-blur-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm text-sm font-medium"
                    >
                      <option value="created_at">Sort by Date</option>
                      <option value="quality_score">Sort by Quality</option>
                      <option value="priority">Sort by Priority</option>
                    </select>
                    
                    <button
                      onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                      className="p-3 border border-gray-300/50 rounded-xl bg-white/70 backdrop-blur-sm hover:bg-white/90 shadow-sm transition-all duration-200 group"
                    >
                      {sortOrder === 'asc' ? 
                        <ChevronUpIcon className="h-5 w-5 text-gray-600 group-hover:text-gray-900" /> : 
                        <ChevronDownIcon className="h-5 w-5 text-gray-600 group-hover:text-gray-900" />
                      }
                    </button>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <button className="flex items-center gap-2 px-4 py-3 border border-gray-300/50 rounded-xl bg-white/70 backdrop-blur-sm hover:bg-white/90 shadow-sm transition-all duration-200 text-sm font-medium text-gray-700 hover:text-gray-900">
                    <FunnelIcon className="h-4 w-4" />
                    Advanced Filters
                  </button>
                  
                  <button
                    disabled
                    className="flex items-center gap-2 px-4 py-3 border border-gray-300/50 rounded-xl bg-gray-100/50 backdrop-blur-sm shadow-sm transition-all duration-200 text-sm font-medium text-gray-400 cursor-not-allowed"
                  >
                    <DocumentTextIcon className="h-4 w-4" />
                    Business Context (Coming Soon)
                  </button>
                </div>
              </div>
            </div>

            {/* Content Items */}
            <div className="space-y-4">
              {filteredAndSortedContent.map((item) => {
                const isExpanded = expandedItems.has(item.content_id);
                const IconComponent = CONTENT_TYPE_ICONS[item.content_type as keyof typeof CONTENT_TYPE_ICONS];
                
                return (
                  <div key={item.content_id} className="group bg-white/80 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200/50 hover:shadow-xl hover:shadow-blue-100/20 transition-all duration-300 hover:-translate-y-1">
                    <div className="p-6">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-4 flex-1">
                          <div className="flex-shrink-0">
                            <div className={`p-3 rounded-xl shadow-sm ${
                              item.priority_level === 'urgent' ? 'bg-gradient-to-r from-red-500 to-pink-500' :
                              item.priority_level === 'high' ? 'bg-gradient-to-r from-orange-500 to-amber-500' :
                              item.priority_level === 'medium' ? 'bg-gradient-to-r from-blue-500 to-indigo-500' :
                              'bg-gradient-to-r from-emerald-500 to-green-500'
                            }`}>
                              {typeof IconComponent === 'string' ? (
                                <span className="text-xl text-white">{IconComponent}</span>
                              ) : (
                                <IconComponent className="h-6 w-6 text-white" />
                              )}
                            </div>
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex-1 min-w-0 pr-4">
                                <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600 transition-colors mb-2 line-clamp-1">
                                  {item.title}
                                </h3>
                                <div className="flex items-center gap-3 text-sm text-gray-600 mb-2">
                                  <span className="flex items-center gap-1">
                                    <ArrowTrendingUpIcon className="h-4 w-4 text-purple-500" />
                                    {item.campaign_name}
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <DocumentTextIcon className="h-4 w-4 text-blue-500" />
                                    {item.content_type.replace('_', ' ')}
                                  </span>
                                  <span className="flex items-center gap-1 text-gray-500">
                                    📝 {item.word_count} words
                                  </span>
                                </div>
                              </div>
                              <span className={`px-3 py-1.5 text-xs font-semibold rounded-full border shadow-sm ${PRIORITY_COLORS[item.priority_level || 'medium']}`}>
                                {item.priority_level?.toUpperCase() || 'MEDIUM'}
                              </span>
                            </div>
                            
                            {/* Enhanced Metrics Row */}
                            <div className="flex items-center gap-3 flex-wrap">
                              <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold shadow-sm ${
                                item.quality_score >= 9 ? 'bg-gradient-to-r from-emerald-100 to-green-100 text-emerald-700 border border-emerald-200' :
                                item.quality_score >= 7 ? 'bg-gradient-to-r from-amber-100 to-yellow-100 text-amber-700 border border-amber-200' :
                                'bg-gradient-to-r from-red-100 to-pink-100 text-red-700 border border-red-200'
                              }`}>
                                <StarIcon className="h-3.5 w-3.5" />
                                Quality {item.quality_score}/10
                              </div>
                              
                              {item.seo_score && (
                                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 border border-blue-200 text-xs font-semibold shadow-sm">
                                  <ChartBarIcon className="h-3.5 w-3.5" />
                                  SEO {item.seo_score}%
                                </div>
                              )}
                              
                              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-purple-100 to-violet-100 text-purple-700 border border-purple-200 text-xs font-semibold shadow-sm">
                                <SparklesIcon className="h-3.5 w-3.5" />
                                {item.estimated_engagement} Engagement
                              </div>
                              
                              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-green-100 to-emerald-100 text-green-700 border border-green-200 text-xs font-semibold shadow-sm">
                                <CheckCircleIcon className="h-3.5 w-3.5" />
                                Brand 95%
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-2 ml-4">
                          <button
                            onClick={() => toggleItemExpansion(item.content_id)}
                            className="p-2.5 text-gray-400 hover:text-gray-600 rounded-xl hover:bg-gray-100/80 backdrop-blur-sm transition-all duration-200 group/btn"
                          >
                            {isExpanded ? 
                              <ChevronUpIcon className="h-5 w-5 group-hover/btn:scale-110 transition-transform" /> : 
                              <ChevronDownIcon className="h-5 w-5 group-hover/btn:scale-110 transition-transform" />
                            }
                          </button>
                          
                          <button
                            onClick={() => {
                              setSelectedContent(item);
                              setShowPreview(true);
                            }}
                            className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 rounded-xl hover:from-blue-100 hover:to-indigo-100 shadow-sm hover:shadow-md transition-all duration-200 font-medium border border-blue-200/50"
                          >
                            <EyeIcon className="h-4 w-4" />
                            Preview
                          </button>
                        </div>
                      </div>
                      
                      {isExpanded && (
                        <div className="mt-6 pt-6 border-t border-gray-200/50">
                          <div className="mb-6">
                            <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                              <DocumentTextIcon className="h-4 w-4 text-blue-500" />
                              Content Preview
                            </h4>
                            <div className="bg-gradient-to-r from-gray-50 to-blue-50/30 p-6 rounded-xl border border-gray-200/50 backdrop-blur-sm">
                              <p className="text-sm text-gray-700 line-clamp-4 leading-relaxed">{item.content}</p>
                            </div>
                          </div>
                          
                          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 text-sm">
                              <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50/80 rounded-lg">
                                <ClockIcon className="h-4 w-4 text-blue-500" />
                                <span className="text-gray-700">Created: <span className="font-medium">{new Date(item.created_at).toLocaleDateString()}</span></span>
                              </div>
                              {item.deadline && (
                                <div className="flex items-center gap-2 px-3 py-1.5 bg-orange-50/80 rounded-lg">
                                  <ExclamationTriangleIcon className="h-4 w-4 text-orange-500" />
                                  <span className="text-orange-700">Deadline: <span className="font-medium">{new Date(item.deadline).toLocaleDateString()}</span></span>
                                </div>
                              )}
                            </div>
                            
                            <div className="flex items-center gap-3">
                              <button
                                onClick={() => {
                                  setSelectedContent(item);
                                  setShowRevisionModal(true);
                                }}
                                className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-amber-50 to-yellow-50 text-amber-700 rounded-xl hover:from-amber-100 hover:to-yellow-100 shadow-sm hover:shadow-md transition-all duration-200 font-medium border border-amber-200/50"
                              >
                                <PencilSquareIcon className="h-4 w-4" />
                                Request Revision
                              </button>
                              
                              <button
                                onClick={() => handleApproval(item, false)}
                                className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-red-50 to-pink-50 text-red-700 rounded-xl hover:from-red-100 hover:to-pink-100 shadow-sm hover:shadow-md transition-all duration-200 font-medium border border-red-200/50"
                              >
                                <XCircleIcon className="h-4 w-4" />
                                Reject
                              </button>
                              
                              <button
                                onClick={() => handleApproval(item, true)}
                                className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-emerald-50 to-green-50 text-emerald-700 rounded-xl hover:from-emerald-100 hover:to-green-100 shadow-sm hover:shadow-md transition-all duration-200 font-medium border border-emerald-200/50"
                              >
                                <CheckCircleIcon className="h-4 w-4" />
                                Approve
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              
              {filteredAndSortedContent.length === 0 && (
                <div className="text-center py-12">
                  <DocumentTextIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No content items found for review</p>
                </div>
              )}
            </div>
          </div>

          {/* Enhanced Preview Panel */}
          {showPreview && selectedContent && (
            <div className="w-96 bg-white/90 backdrop-blur-lg rounded-xl shadow-2xl border border-gray-200/50 h-fit sticky top-6">
              <div className="p-6 border-b border-gray-200/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg">
                      <EyeIcon className="h-5 w-5 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                      Content Preview
                    </h3>
                  </div>
                  <button
                    onClick={() => setShowPreview(false)}
                    className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100/80 transition-all duration-200"
                  >
                    <XCircleIcon className="h-5 w-5" />
                  </button>
                </div>
              </div>
              
              <div className="p-6 max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
                <h4 className="font-semibold text-gray-900 mb-4 text-base leading-relaxed">{selectedContent.title}</h4>
                <div className="prose prose-sm max-w-none">
                  <div className="bg-gradient-to-r from-gray-50 to-blue-50/30 p-4 rounded-xl border border-gray-200/50">
                    <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans leading-relaxed">
                      {selectedContent.content}
                    </pre>
                  </div>
                </div>
              </div>
              
              <div className="p-6 border-t border-gray-200/50 space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Review Comments
                  </label>
                  <textarea
                    value={reviewComment}
                    onChange={(e) => setReviewComment(e.target.value)}
                    placeholder="Share your detailed feedback and suggestions..."
                    rows={3}
                    className="w-full px-4 py-3 border border-gray-300/50 rounded-xl bg-white/70 backdrop-blur-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm transition-all duration-200 text-sm"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Quality Rating
                  </label>
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      {[1, 2, 3, 4, 5].map((rating) => (
                        <button
                          key={rating}
                          onClick={() => setQualityRating(rating * 2)}
                          className="p-1.5 rounded-lg hover:bg-yellow-50 transition-colors"
                        >
                          {qualityRating >= rating * 2 ? (
                            <StarSolid className="h-6 w-6 text-yellow-400 hover:text-yellow-500 transition-colors" />
                          ) : (
                            <StarIcon className="h-6 w-6 text-gray-300 hover:text-yellow-300 transition-colors" />
                          )}
                        </button>
                      ))}
                    </div>
                    <div className="ml-3 px-3 py-1 bg-gray-100/80 rounded-lg">
                      <span className="text-sm font-semibold text-gray-700">
                        {qualityRating}/10
                      </span>
                    </div>
                  </div>
                </div>
                
                {/* Brand Consistency Check - Temporarily disabled */}
                <div className="border-t border-gray-200/50 pt-5">
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-xl border border-green-200/50">
                    <h4 className="text-sm font-semibold text-green-800 mb-2">Brand Consistency Analysis</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-green-700">Tone Alignment</span>
                        <span className="font-semibold text-green-800">95%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-green-700">Keyword Usage</span>
                        <span className="font-semibold text-green-800">88%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-green-700">Brand Guidelines</span>
                        <span className="font-semibold text-green-800">92%</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={() => handleApproval(selectedContent, false)}
                    disabled={processingApproval}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl hover:from-red-600 hover:to-red-700 disabled:opacity-50 shadow-lg hover:shadow-xl transition-all duration-200 font-semibold"
                  >
                    <XCircleIcon className="h-4 w-4" />
                    Reject
                  </button>
                  
                  <button
                    onClick={() => handleApproval(selectedContent, true)}
                    disabled={processingApproval}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl hover:from-emerald-600 hover:to-emerald-700 disabled:opacity-50 shadow-lg hover:shadow-xl transition-all duration-200 font-semibold"
                  >
                    <CheckCircleIcon className="h-4 w-4" />
                    Approve
                  </button>
                </div>
              </div>
            </div>
          )}
          </div>
        )}
      </div>

      {/* Revision Modal */}
      {showRevisionModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Request Content Revision</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Revision Notes *
                  </label>
                  <textarea
                    value={revisionNotes}
                    onChange={(e) => setRevisionNotes(e.target.value)}
                    placeholder="Describe what changes are needed..."
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
              
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowRevisionModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleRevisionRequest}
                  disabled={!revisionNotes.trim()}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                >
                  Request Revision
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ContentReviewApproval;