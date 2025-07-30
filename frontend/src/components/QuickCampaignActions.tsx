import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { campaignApi } from '../lib/api';
import { showErrorNotification, showSuccessNotification, AppError } from '../lib/errors';
import type { BlogSummary } from '../lib/api';

interface QuickCampaignActionsProps {
  blog: BlogSummary;
  onRefresh?: () => void;
}

interface CampaignTemplate {
  id: string;
  title: string;
  description: string;
  icon: string;
  channels: string[];
  color: string;
}

const campaignTemplates: CampaignTemplate[] = [
  {
    id: 'social-blast',
    title: 'Social Media Blast',
    description: 'Share across LinkedIn, Twitter, and Facebook',
    icon: '📱',
    channels: ['linkedin', 'twitter', 'facebook'],
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'professional-share',
    title: 'Professional Networks',
    description: 'LinkedIn article and professional updates',
    icon: '💼',
    channels: ['linkedin'],
    color: 'from-blue-600 to-blue-700'
  },
  {
    id: 'email-campaign',
    title: 'Email Newsletter',
    description: 'Convert to email newsletter format',
    icon: '📧',
    channels: ['email'],
    color: 'from-green-500 to-emerald-500'
  },
  {
    id: 'full-campaign',
    title: 'Full Marketing Push',
    description: 'Multi-channel campaign with scheduling',
    icon: '🎯',
    channels: ['linkedin', 'twitter', 'facebook', 'email'],
    color: 'from-purple-500 to-pink-500'
  }
];

export function QuickCampaignActions({ blog, onRefresh }: QuickCampaignActionsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleQuickCampaign = async (template: CampaignTemplate) => {
    setIsLoading(true);
    setIsOpen(false);

    try {
      const campaignName = `${template.title}: ${blog.title}`;
      
      let result;
      
      if (template.id === 'full-campaign') {
        // Full campaign uses the regular API
        result = await campaignApi.createFromBlog(blog.id, campaignName);
        showSuccessNotification(`${template.title} campaign created! 🚀`);
        navigate(`/campaign-wizard/${result.campaign_id}`);
      } else {
        // Quick campaigns use the new template API
        result = await campaignApi.createQuickCampaign(template.id, blog.id, campaignName);
        
        if (result.auto_executed) {
          showSuccessNotification(`${template.title} campaign created and scheduled! 🎉`);
          navigate(`/campaigns/${result.campaign_id}?success=auto-scheduled`);
        } else {
          showSuccessNotification(`${template.title} campaign created! 🚀`);
          navigate(`/campaigns/${result.campaign_id}?template=${template.id}`);
        }
      }

      if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      showErrorNotification(
        err instanceof AppError 
          ? err 
          : new AppError(`Failed to create ${template.title.toLowerCase()}`)
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleAdvancedSetup = async () => {
    setIsLoading(true);
    setIsOpen(false);

    try {
      const campaignName = `Advanced Campaign: ${blog.title}`;
      const result = await campaignApi.createFromBlog(blog.id, campaignName);
      
      // Navigate to full wizard
      navigate(`/campaign-wizard/${result.campaign_id}`);
      
      if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      showErrorNotification(
        err instanceof AppError 
          ? err 
          : new AppError('Failed to create campaign')
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
        className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-3 py-2 rounded-md text-sm font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:transform-none flex items-center space-x-1"
      >
        <span>🚀</span>
        <span>{isLoading ? 'Creating...' : 'Go Multi-Channel'}</span>
        <svg 
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-80 bg-white rounded-lg shadow-xl border border-gray-200 z-50 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900">Choose Campaign Type</h3>
            <p className="text-xs text-gray-600 mt-1">AI will adapt your content automatically</p>
          </div>

          {/* Quick Templates */}
          <div className="py-2">
            {campaignTemplates.map((template) => (
              <button
                key={template.id}
                onClick={() => handleQuickCampaign(template)}
                disabled={isLoading}
                className="w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors disabled:opacity-50 group"
              >
                <div className="flex items-start space-x-3">
                  <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${template.color} flex items-center justify-center text-white text-sm font-medium group-hover:scale-110 transition-transform`}>
                    {template.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                      {template.title}
                    </h4>
                    <p className="text-xs text-gray-600 mt-1">
                      {template.description}
                    </p>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {template.channels.map((channel) => (
                        <span
                          key={channel}
                          className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700 capitalize"
                        >
                          {channel}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>

          {/* Divider */}
          <div className="border-t border-gray-200"></div>

          {/* Advanced Option */}
          <div className="p-2">
            <button
              onClick={handleAdvancedSetup}
              disabled={isLoading}
              className="w-full px-3 py-2 text-left hover:bg-gray-50 rounded-md transition-colors disabled:opacity-50 group"
            >
              <div className="flex items-center space-x-2">
                <div className="w-6 h-6 rounded bg-gray-200 flex items-center justify-center text-gray-600">
                  ⚙️
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-900 group-hover:text-blue-600">
                    Advanced Setup
                  </span>
                  <p className="text-xs text-gray-500">
                    Custom strategy, timing, and channels
                  </p>
                </div>
              </div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}