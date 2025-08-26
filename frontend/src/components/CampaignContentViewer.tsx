import React, { useState, useMemo } from 'react';
import { 
  DocumentTextIcon, 
  ChatBubbleLeftRightIcon, 
  EnvelopeIcon,
  PhotoIcon,
  PencilSquareIcon,
  ArrowsRightLeftIcon,
  MagnifyingGlassIcon,
  EyeIcon,
  EyeSlashIcon
} from '@heroicons/react/24/outline';
import { CheckCircleIcon, ClockIcon, ExclamationCircleIcon } from '@heroicons/react/24/solid';

interface Task {
  id: string;
  task_type: string;
  status: string;
  title: string;
  result?: string;
  error?: string;
  channel?: string;
  content_type?: string;
  assigned_agent?: string;
  created_at: string;
  completed_at?: string;
}

interface CampaignContentViewerProps {
  campaignId: string;
  tasks: Task[];
  campaignName?: string;
}

// Content type configurations
const CONTENT_TYPES = {
  content_creation: {
    label: 'Blog Posts & Articles',
    icon: DocumentTextIcon,
    color: 'blue',
    description: 'Long-form content pieces'
  },
  social_media_adaptation: {
    label: 'Social Media Content',
    icon: ChatBubbleLeftRightIcon,
    color: 'purple',
    description: 'Social media posts and adaptations'
  },
  email_formatting: {
    label: 'Email Campaigns',
    icon: EnvelopeIcon,
    color: 'green',
    description: 'Email marketing content'
  },
  image_generation: {
    label: 'Visual Content',
    icon: PhotoIcon,
    color: 'orange',
    description: 'Images, graphics, and visual concepts'
  },
  content_editing: {
    label: 'Editorial Review',
    icon: PencilSquareIcon,
    color: 'red',
    description: 'Content review and quality assurance'
  },
  content_repurposing: {
    label: 'Repurposed Content',
    icon: ArrowsRightLeftIcon,
    color: 'indigo',
    description: 'Multi-format content adaptations'
  },
  seo_optimization: {
    label: 'SEO Optimization',
    icon: MagnifyingGlassIcon,
    color: 'yellow',
    description: 'Search engine optimization'
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
    case 'pending':
      return <ClockIcon className="h-5 w-5 text-yellow-500" />;
    case 'error':
    case 'failed':
      return <ExclamationCircleIcon className="h-5 w-5 text-red-500" />;
    default:
      return <ClockIcon className="h-5 w-5 text-gray-500" />;
  }
};

const formatContent = (result: string): { title: string; content: string; isMarkdown: boolean } => {
  if (!result) return { title: 'No content available', content: '', isMarkdown: false };
  
  // Check if content starts with markdown header
  const lines = result.split('\\n');
  const firstLine = lines[0];
  
  if (firstLine.startsWith('# ')) {
    return {
      title: firstLine.replace('# ', ''),
      content: lines.slice(1).join('\\n').trim(),
      isMarkdown: true
    };
  }
  
  // Check for other title patterns
  if (firstLine.includes(':')) {
    const [title, ...contentParts] = result.split('\\n');
    return {
      title: title.replace(/^(Subject:|Title:|.*?:)\\s*/, ''),
      content: contentParts.join('\\n').trim(),
      isMarkdown: false
    };
  }
  
  return {
    title: result.length > 60 ? result.substring(0, 60) + '...' : result,
    content: result,
    isMarkdown: false
  };
};

export const CampaignContentViewer: React.FC<CampaignContentViewerProps> = ({
  campaignId,
  tasks,
  campaignName
}) => {
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');

  // Group tasks by type
  const groupedTasks = useMemo(() => {
    const groups: Record<string, Task[]> = {};
    tasks.forEach(task => {
      const type = task.task_type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(task);
    });
    return groups;
  }, [tasks]);

  // Filter tasks based on search
  const filteredGroups = useMemo(() => {
    if (!searchTerm) return groupedTasks;
    
    const filtered: Record<string, Task[]> = {};
    Object.entries(groupedTasks).forEach(([type, typeTasks]) => {
      const matchingTasks = typeTasks.filter(task => {
        const { title, content } = formatContent(task.result || '');
        return title.toLowerCase().includes(searchTerm.toLowerCase()) ||
               content.toLowerCase().includes(searchTerm.toLowerCase()) ||
               type.toLowerCase().includes(searchTerm.toLowerCase());
      });
      if (matchingTasks.length > 0) {
        filtered[type] = matchingTasks;
      }
    });
    return filtered;
  }, [groupedTasks, searchTerm]);

  const toggleCardExpansion = (taskId: string) => {
    setExpandedCards(prev => {
      const next = new Set(prev);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else {
        next.add(taskId);
      }
      return next;
    });
  };

  const totalTasks = tasks.length;
  const completedTasks = tasks.filter(task => task.status === 'completed').length;
  const completionRate = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">
          Generated Content - {campaignName || 'Campaign'}
        </h2>
        <div className="mt-2 flex items-center space-x-4 text-sm text-gray-600">
          <span>{totalTasks} total pieces</span>
          <span>{completedTasks} completed</span>
          <span>{completionRate}% complete</span>
        </div>
        
        {/* Search */}
        <div className="mt-4 max-w-md">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search content..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Content Type Filters */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        <button
          onClick={() => setSelectedType(null)}
          className={`p-3 text-left rounded-lg border transition-colors ${ 
            selectedType === null 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <div className="font-medium text-sm text-gray-900">All Content</div>
          <div className="text-xs text-gray-500">{totalTasks} items</div>
        </button>
        
        {Object.entries(filteredGroups).map(([type, typeTasks]) => {
          const config = CONTENT_TYPES[type as keyof typeof CONTENT_TYPES];
          const Icon = config?.icon || DocumentTextIcon;
          const completedCount = typeTasks.filter(t => t.status === 'completed').length;
          
          return (
            <button
              key={type}
              onClick={() => setSelectedType(selectedType === type ? null : type)}
              className={`p-3 text-left rounded-lg border transition-colors ${
                selectedType === type 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <Icon className={`h-4 w-4 text-${config?.color || 'gray'}-500`} />
                <span className="font-medium text-sm text-gray-900">
                  {config?.label || type}
                </span>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {completedCount}/{typeTasks.length} completed
              </div>
            </button>
          );
        })}
      </div>

      {/* Content Display */}
      <div className="space-y-6">
        {Object.entries(filteredGroups)
          .filter(([type]) => selectedType === null || selectedType === type)
          .map(([type, typeTasks]) => {
            const config = CONTENT_TYPES[type as keyof typeof CONTENT_TYPES];
            const Icon = config?.icon || DocumentTextIcon;

            return (
              <div key={type} className="bg-white border border-gray-200 rounded-lg">
                <div className={`px-4 py-3 border-b border-gray-200 bg-${config?.color || 'gray'}-50`}>
                  <div className="flex items-center space-x-2">
                    <Icon className={`h-5 w-5 text-${config?.color || 'gray'}-600`} />
                    <h3 className="font-medium text-gray-900">
                      {config?.label || type} ({typeTasks.length})
                    </h3>
                  </div>
                  {config?.description && (
                    <p className="text-sm text-gray-600 mt-1">{config.description}</p>
                  )}
                </div>

                <div className="divide-y divide-gray-200">
                  {typeTasks.map((task) => {
                    const { title, content, isMarkdown } = formatContent(task.result || '');
                    const isExpanded = expandedCards.has(task.id);

                    return (
                      <div key={task.id} className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-2">
                              {getStatusIcon(task.status)}
                              <h4 className="font-medium text-gray-900 truncate">
                                {title}
                              </h4>
                            </div>
                            
                            <div className="text-sm text-gray-600 mb-2">
                              {task.channel && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 mr-2">
                                  {task.channel}
                                </span>
                              )}
                              {task.assigned_agent && (
                                <span className="text-gray-500">
                                  by {task.assigned_agent}
                                </span>
                              )}
                            </div>

                            {/* Content Preview/Full */}
                            <div className="mt-3">
                              {content && (
                                <div className={`prose prose-sm max-w-none ${
                                  isExpanded ? '' : 'line-clamp-3'
                                }`}>
                                  {isMarkdown ? (
                                    <div className="whitespace-pre-wrap">{content}</div>
                                  ) : (
                                    <p className="text-gray-700 whitespace-pre-wrap">{content}</p>
                                  )}
                                </div>
                              )}
                              
                              {task.error && (
                                <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded">
                                  <p className="text-sm text-red-700">{task.error}</p>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Expand/Collapse Button */}
                          {content && content.length > 200 && (
                            <button
                              onClick={() => toggleCardExpansion(task.id)}
                              className="ml-4 flex-shrink-0 p-2 text-gray-400 hover:text-gray-600 transition-colors"
                              title={isExpanded ? "Show less" : "Show more"}
                            >
                              {isExpanded ? (
                                <EyeSlashIcon className="h-4 w-4" />
                              ) : (
                                <EyeIcon className="h-4 w-4" />
                              )}
                            </button>
                          )}
                        </div>

                        {/* Metadata */}
                        <div className="mt-3 text-xs text-gray-500 flex items-center space-x-4">
                          <span>Created: {new Date(task.created_at).toLocaleDateString()}</span>
                          {task.completed_at && (
                            <span>Completed: {new Date(task.completed_at).toLocaleDateString()}</span>
                          )}
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            task.status === 'completed' ? 'bg-green-100 text-green-800' :
                            task.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {task.status}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
      </div>

      {/* No results */}
      {Object.keys(filteredGroups).length === 0 && (
        <div className="text-center py-12">
          <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No content found</h3>
          <p className="mt-1 text-sm text-gray-500">
            {searchTerm ? 'Try a different search term.' : 'No content has been generated yet.'}
          </p>
        </div>
      )}
    </div>
  );
};