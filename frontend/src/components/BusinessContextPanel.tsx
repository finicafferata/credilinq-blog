import React, { useState, useEffect } from 'react';
import { 
  BuildingOfficeIcon,
  SpeakerWaveIcon,
  TagIcon,
  UsersIcon,
  ExclamationTriangleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  LinkIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';
import { settingsApi, CompanyProfile } from '../lib/api';

interface BusinessContextPanelProps {
  isVisible: boolean;
  onToggle: () => void;
}

const BusinessContextPanel: React.FC<BusinessContextPanelProps> = ({ 
  isVisible, 
  onToggle 
}) => {
  const [profile, setProfile] = useState<CompanyProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['overview']));

  useEffect(() => {
    loadCompanyProfile();
  }, []);

  const loadCompanyProfile = async () => {
    try {
      setLoading(true);
      const companyProfile = await settingsApi.getCompanyProfile();
      setProfile(companyProfile);
    } catch (error) {
      console.error('Error loading company profile:', error);
      setError('Failed to load company profile');
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  };

  const renderArrayField = (items: string[], emptyText: string = 'None specified') => {
    if (!items || items.length === 0) {
      return <span className="text-gray-400 italic">{emptyText}</span>;
    }
    return (
      <div className="flex flex-wrap gap-2">
        {items.map((item, index) => (
          <span 
            key={index}
            className="px-2 py-1 bg-blue-50 text-blue-700 rounded-full text-xs"
          >
            {item}
          </span>
        ))}
      </div>
    );
  };

  if (!isVisible) return null;

  if (loading) {
    return (
      <div className="w-80 bg-white rounded-lg shadow-lg border p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Business Context</h3>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-gray-600"
          >
            <ChevronUpIcon className="h-5 w-5" />
          </button>
        </div>
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Loading business context...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-80 bg-white rounded-lg shadow-lg border p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Business Context</h3>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-gray-600"
          >
            <ChevronUpIcon className="h-5 w-5" />
          </button>
        </div>
        <div className="text-center py-4">
          <ExclamationTriangleIcon className="h-8 w-8 text-red-500 mx-auto mb-2" />
          <p className="text-sm text-red-600">{error}</p>
          <button 
            onClick={loadCompanyProfile}
            className="mt-2 text-sm text-blue-600 hover:text-blue-800"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="w-80 bg-white rounded-lg shadow-lg border p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Business Context</h3>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-gray-600"
          >
            <ChevronUpIcon className="h-5 w-5" />
          </button>
        </div>
        <div className="text-center py-4">
          <BuildingOfficeIcon className="h-8 w-8 text-gray-400 mx-auto mb-2" />
          <p className="text-sm text-gray-600">No company profile configured</p>
          <p className="text-xs text-gray-500 mt-1">Go to Settings to set up your business context</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-white rounded-lg shadow-lg border max-h-[80vh] overflow-y-auto">
      <div className="sticky top-0 bg-white border-b p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Business Context</h3>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-gray-600"
          >
            <ChevronUpIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Company Overview */}
        <div>
          <button
            onClick={() => toggleSection('overview')}
            className="flex items-center justify-between w-full p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
          >
            <div className="flex items-center gap-2">
              <BuildingOfficeIcon className="h-5 w-5 text-gray-600" />
              <span className="font-medium text-gray-900">Company Overview</span>
            </div>
            {expandedSections.has('overview') ? (
              <ChevronUpIcon className="h-4 w-4 text-gray-600" />
            ) : (
              <ChevronDownIcon className="h-4 w-4 text-gray-600" />
            )}
          </button>
          
          {expandedSections.has('overview') && (
            <div className="mt-3 space-y-3">
              {profile.companyName && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700">{profile.companyName}</h4>
                </div>
              )}
              
              {profile.companyContext && (
                <div>
                  <h5 className="text-xs font-medium text-gray-600 mb-1">Company Context</h5>
                  <p className="text-sm text-gray-700 bg-gray-50 p-2 rounded text-wrap">
                    {profile.companyContext}
                  </p>
                </div>
              )}
              
              {profile.valueProposition && (
                <div>
                  <h5 className="text-xs font-medium text-gray-600 mb-1">Value Proposition</h5>
                  <p className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                    {profile.valueProposition}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Brand Voice & Tone */}
        <div>
          <button
            onClick={() => toggleSection('voice')}
            className="flex items-center justify-between w-full p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
          >
            <div className="flex items-center gap-2">
              <SpeakerWaveIcon className="h-5 w-5 text-gray-600" />
              <span className="font-medium text-gray-900">Brand Voice & Tone</span>
            </div>
            {expandedSections.has('voice') ? (
              <ChevronUpIcon className="h-4 w-4 text-gray-600" />
            ) : (
              <ChevronDownIcon className="h-4 w-4 text-gray-600" />
            )}
          </button>
          
          {expandedSections.has('voice') && (
            <div className="mt-3 space-y-3">
              {profile.brandVoice && (
                <div>
                  <h5 className="text-xs font-medium text-gray-600 mb-1">Brand Voice</h5>
                  <p className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                    {profile.brandVoice}
                  </p>
                </div>
              )}
              
              <div>
                <h5 className="text-xs font-medium text-gray-600 mb-2">Tone Presets</h5>
                {renderArrayField(profile.tonePresets, 'No tone presets specified')}
              </div>
              
              {profile.styleGuidelines && (
                <div>
                  <h5 className="text-xs font-medium text-gray-600 mb-1">Style Guidelines</h5>
                  <p className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                    {profile.styleGuidelines}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Target Audience */}
        <div>
          <button
            onClick={() => toggleSection('audience')}
            className="flex items-center justify-between w-full p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
          >
            <div className="flex items-center gap-2">
              <UsersIcon className="h-5 w-5 text-gray-600" />
              <span className="font-medium text-gray-900">Target Audience</span>
            </div>
            {expandedSections.has('audience') ? (
              <ChevronUpIcon className="h-4 w-4 text-gray-600" />
            ) : (
              <ChevronDownIcon className="h-4 w-4 text-gray-600" />
            )}
          </button>
          
          {expandedSections.has('audience') && (
            <div className="mt-3 space-y-3">
              <div>
                <h5 className="text-xs font-medium text-gray-600 mb-2">Target Audiences</h5>
                {renderArrayField(profile.targetAudiences, 'No target audiences specified')}
              </div>
              
              <div>
                <h5 className="text-xs font-medium text-gray-600 mb-2">Industries</h5>
                {renderArrayField(profile.industries, 'No industries specified')}
              </div>
            </div>
          )}
        </div>

        {/* Keywords & SEO */}
        <div>
          <button
            onClick={() => toggleSection('keywords')}
            className="flex items-center justify-between w-full p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
          >
            <div className="flex items-center gap-2">
              <TagIcon className="h-5 w-5 text-gray-600" />
              <span className="font-medium text-gray-900">Keywords & SEO</span>
            </div>
            {expandedSections.has('keywords') ? (
              <ChevronUpIcon className="h-4 w-4 text-gray-600" />
            ) : (
              <ChevronDownIcon className="h-4 w-4 text-gray-600" />
            )}
          </button>
          
          {expandedSections.has('keywords') && (
            <div className="mt-3 space-y-3">
              <div>
                <h5 className="text-xs font-medium text-gray-600 mb-2">Brand Keywords</h5>
                {renderArrayField(profile.keywords, 'No keywords specified')}
              </div>
              
              {profile.defaultCTA && (
                <div>
                  <h5 className="text-xs font-medium text-gray-600 mb-1">Default CTA</h5>
                  <span className="text-sm text-gray-700 bg-blue-50 px-2 py-1 rounded">
                    {profile.defaultCTA}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Compliance & Guidelines */}
        <div>
          <button
            onClick={() => toggleSection('compliance')}
            className="flex items-center justify-between w-full p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
          >
            <div className="flex items-center gap-2">
              <ShieldCheckIcon className="h-5 w-5 text-gray-600" />
              <span className="font-medium text-gray-900">Compliance</span>
            </div>
            {expandedSections.has('compliance') ? (
              <ChevronUpIcon className="h-4 w-4 text-gray-600" />
            ) : (
              <ChevronDownIcon className="h-4 w-4 text-gray-600" />
            )}
          </button>
          
          {expandedSections.has('compliance') && (
            <div className="mt-3 space-y-3">
              <div>
                <h5 className="text-xs font-medium text-gray-600 mb-2">Prohibited Topics</h5>
                {renderArrayField(profile.prohibitedTopics, 'No prohibited topics specified')}
              </div>
              
              {profile.complianceNotes && (
                <div>
                  <h5 className="text-xs font-medium text-gray-600 mb-1">Compliance Notes</h5>
                  <p className="text-sm text-gray-700 bg-yellow-50 p-2 rounded border-l-4 border-yellow-400">
                    {profile.complianceNotes}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Useful Links */}
        {profile.links && profile.links.length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('links')}
              className="flex items-center justify-between w-full p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
            >
              <div className="flex items-center gap-2">
                <LinkIcon className="h-5 w-5 text-gray-600" />
                <span className="font-medium text-gray-900">Useful Links</span>
              </div>
              {expandedSections.has('links') ? (
                <ChevronUpIcon className="h-4 w-4 text-gray-600" />
              ) : (
                <ChevronDownIcon className="h-4 w-4 text-gray-600" />
              )}
            </button>
            
            {expandedSections.has('links') && (
              <div className="mt-3 space-y-2">
                {profile.links.map((link, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <LinkIcon className="h-4 w-4 text-gray-400 flex-shrink-0" />
                    <a 
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:text-blue-800 truncate"
                      title={link.url}
                    >
                      {link.label}
                    </a>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {profile.updatedAt && (
          <div className="pt-4 border-t text-xs text-gray-500">
            Last updated: {new Date(profile.updatedAt).toLocaleDateString()}
          </div>
        )}
      </div>
    </div>
  );
};

export default BusinessContextPanel;