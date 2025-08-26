import React, { useState, useEffect } from 'react';
import { CheckCircleIcon, ExclamationTriangleIcon, XCircleIcon } from '@heroicons/react/24/outline';
import { settingsApi, CompanyProfile } from '../lib/api';
import { GeneratedContent } from '../services/contentWorkflowApi';

interface BrandConsistencyCheckerProps {
  content: GeneratedContent;
  className?: string;
}

interface ConsistencyCheck {
  category: string;
  score: number;
  status: 'pass' | 'warning' | 'fail';
  details: string[];
  recommendations?: string[];
}

const BrandConsistencyChecker: React.FC<BrandConsistencyCheckerProps> = ({ 
  content, 
  className = '' 
}) => {
  const [profile, setProfile] = useState<CompanyProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [consistencyChecks, setConsistencyChecks] = useState<ConsistencyCheck[]>([]);

  useEffect(() => {
    loadProfileAndAnalyze();
  }, [content]);

  const loadProfileAndAnalyze = async () => {
    try {
      setLoading(true);
      const companyProfile = await settingsApi.getCompanyProfile();
      setProfile(companyProfile);
      
      // Perform brand consistency analysis
      const checks = analyzeBrandConsistency(content, companyProfile);
      setConsistencyChecks(checks);
    } catch (error) {
      console.error('Error analyzing brand consistency:', error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeBrandConsistency = (
    content: GeneratedContent, 
    profile: CompanyProfile
  ): ConsistencyCheck[] => {
    const checks: ConsistencyCheck[] = [];
    const contentText = content.content.toLowerCase();

    // 1. Tone Alignment Check
    if (profile.tonePresets && profile.tonePresets.length > 0) {
      const toneScore = calculateToneAlignment(contentText, profile.tonePresets);
      checks.push({
        category: 'Tone Alignment',
        score: toneScore,
        status: toneScore >= 80 ? 'pass' : toneScore >= 60 ? 'warning' : 'fail',
        details: [
          `Expected tones: ${profile.tonePresets.join(', ')}`,
          `Content tone appears ${toneScore >= 80 ? 'well-aligned' : 'misaligned'} with brand voice`
        ],
        recommendations: toneScore < 80 ? [
          'Consider adjusting language to match brand tone presets',
          'Review style guidelines for tone examples'
        ] : undefined
      });
    }

    // 2. Keyword Usage Check
    if (profile.keywords && profile.keywords.length > 0) {
      const keywordScore = calculateKeywordUsage(contentText, profile.keywords);
      const usedKeywords = profile.keywords.filter(keyword => 
        contentText.includes(keyword.toLowerCase())
      );
      
      checks.push({
        category: 'Brand Keywords',
        score: keywordScore,
        status: keywordScore >= 70 ? 'pass' : keywordScore >= 40 ? 'warning' : 'fail',
        details: [
          `Used ${usedKeywords.length} of ${profile.keywords.length} brand keywords`,
          `Keywords found: ${usedKeywords.join(', ') || 'None'}`
        ],
        recommendations: keywordScore < 70 ? [
          'Consider incorporating more brand-specific keywords',
          `Missing keywords: ${profile.keywords.filter(k => !usedKeywords.includes(k)).join(', ')}`
        ] : undefined
      });
    }

    // 3. Prohibited Topics Check
    if (profile.prohibitedTopics && profile.prohibitedTopics.length > 0) {
      const prohibitedFound = profile.prohibitedTopics.filter(topic =>
        contentText.includes(topic.toLowerCase())
      );
      
      checks.push({
        category: 'Compliance Check',
        score: prohibitedFound.length === 0 ? 100 : 0,
        status: prohibitedFound.length === 0 ? 'pass' : 'fail',
        details: prohibitedFound.length === 0 
          ? ['No prohibited topics detected']
          : [`Found prohibited topics: ${prohibitedFound.join(', ')}`],
        recommendations: prohibitedFound.length > 0 ? [
          'Remove or rephrase content containing prohibited topics',
          'Review compliance guidelines before publishing'
        ] : undefined
      });
    }

    // 4. Target Audience Alignment
    if (profile.targetAudiences && profile.targetAudiences.length > 0) {
      const audienceScore = calculateAudienceAlignment(contentText, profile.targetAudiences);
      checks.push({
        category: 'Audience Alignment',
        score: audienceScore,
        status: audienceScore >= 75 ? 'pass' : audienceScore >= 50 ? 'warning' : 'fail',
        details: [
          `Target audiences: ${profile.targetAudiences.join(', ')}`,
          `Content appears ${audienceScore >= 75 ? 'well-suited' : 'poorly-suited'} for target audience`
        ],
        recommendations: audienceScore < 75 ? [
          'Adjust language complexity and terminology for target audience',
          'Include examples relevant to target audience'
        ] : undefined
      });
    }

    // 5. Call-to-Action Check
    if (profile.defaultCTA) {
      const ctaPresent = contentText.includes(profile.defaultCTA.toLowerCase()) ||
        hasGenericCTA(contentText);
      
      checks.push({
        category: 'Call-to-Action',
        score: ctaPresent ? 100 : 60,
        status: ctaPresent ? 'pass' : 'warning',
        details: ctaPresent 
          ? ['Call-to-action is present']
          : ['No clear call-to-action found'],
        recommendations: !ctaPresent ? [
          `Consider adding default CTA: "${profile.defaultCTA}"`,
          'Ensure content guides readers to next steps'
        ] : undefined
      });
    }

    return checks;
  };

  const calculateToneAlignment = (contentText: string, tonePresets: string[]): number => {
    // Simplified tone analysis - in a real implementation, this would use NLP
    const professionalWords = ['professional', 'business', 'industry', 'expertise', 'solution'];
    const friendlyWords = ['help', 'easy', 'simple', 'great', 'amazing', 'welcome'];
    const formalWords = ['therefore', 'furthermore', 'consequently', 'nevertheless'];
    
    let score = 50; // Base score
    
    if (tonePresets.some(tone => tone.toLowerCase().includes('professional'))) {
      const professionalCount = professionalWords.filter(word => contentText.includes(word)).length;
      score += Math.min(professionalCount * 10, 30);
    }
    
    if (tonePresets.some(tone => tone.toLowerCase().includes('friendly'))) {
      const friendlyCount = friendlyWords.filter(word => contentText.includes(word)).length;
      score += Math.min(friendlyCount * 8, 25);
    }
    
    if (tonePresets.some(tone => tone.toLowerCase().includes('formal'))) {
      const formalCount = formalWords.filter(word => contentText.includes(word)).length;
      score += Math.min(formalCount * 12, 25);
    }
    
    return Math.min(score, 100);
  };

  const calculateKeywordUsage = (contentText: string, keywords: string[]): number => {
    const usedKeywords = keywords.filter(keyword => 
      contentText.includes(keyword.toLowerCase())
    );
    return Math.round((usedKeywords.length / keywords.length) * 100);
  };

  const calculateAudienceAlignment = (contentText: string, audiences: string[]): number => {
    // Simplified audience alignment - check for technical vs. general language
    const technicalTerms = ['api', 'integration', 'algorithm', 'analytics', 'optimization'];
    const generalTerms = ['easy', 'simple', 'help', 'guide', 'learn', 'understand'];
    
    const hasTechnical = technicalTerms.some(term => contentText.includes(term));
    const hasGeneral = generalTerms.some(term => contentText.includes(term));
    
    // Adjust score based on audience type
    if (audiences.some(aud => aud.toLowerCase().includes('technical') || aud.toLowerCase().includes('developer'))) {
      return hasTechnical ? 85 : 65;
    } else if (audiences.some(aud => aud.toLowerCase().includes('business') || aud.toLowerCase().includes('executive'))) {
      return hasGeneral && hasTechnical ? 90 : 70;
    }
    
    return 75; // Default score
  };

  const hasGenericCTA = (contentText: string): boolean => {
    const commonCTAs = ['contact us', 'learn more', 'get started', 'book a demo', 'sign up', 'try now'];
    return commonCTAs.some(cta => contentText.includes(cta));
  };

  const getStatusIcon = (status: 'pass' | 'warning' | 'fail') => {
    switch (status) {
      case 'pass':
        return <CheckCircleIcon className="h-4 w-4 text-green-600" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-4 w-4 text-yellow-600" />;
      case 'fail':
        return <XCircleIcon className="h-4 w-4 text-red-600" />;
    }
  };

  const getStatusColor = (status: 'pass' | 'warning' | 'fail') => {
    switch (status) {
      case 'pass':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'fail':
        return 'text-red-600';
    }
  };

  const getOverallScore = () => {
    if (consistencyChecks.length === 0) return 0;
    return Math.round(
      consistencyChecks.reduce((sum, check) => sum + check.score, 0) / consistencyChecks.length
    );
  };

  if (loading) {
    return (
      <div className={`bg-gray-50 p-4 rounded-lg ${className}`}>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
          <div className="h-3 bg-gray-200 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  if (!profile || consistencyChecks.length === 0) {
    return (
      <div className={`bg-gray-50 p-4 rounded-lg ${className}`}>
        <p className="text-sm text-gray-600">No business context configured</p>
        <p className="text-xs text-gray-500 mt-1">
          Set up your company profile in Settings to enable brand consistency checks
        </p>
      </div>
    );
  }

  const overallScore = getOverallScore();

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Overall Score */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-700">Brand Consistency</h4>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-medium ${
            overallScore >= 80 ? 'text-green-600' : 
            overallScore >= 60 ? 'text-yellow-600' : 'text-red-600'
          }`}>
            {overallScore}%
          </span>
          {getStatusIcon(overallScore >= 80 ? 'pass' : overallScore >= 60 ? 'warning' : 'fail')}
        </div>
      </div>

      {/* Individual Checks */}
      <div className="space-y-3">
        {consistencyChecks.map((check, index) => (
          <div key={index} className="border border-gray-200 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">{check.category}</span>
              <div className="flex items-center gap-2">
                <span className={`text-sm font-medium ${getStatusColor(check.status)}`}>
                  {check.score}%
                </span>
                {getStatusIcon(check.status)}
              </div>
            </div>
            
            <div className="text-xs text-gray-600 space-y-1">
              {check.details.map((detail, idx) => (
                <div key={idx}>• {detail}</div>
              ))}
            </div>
            
            {check.recommendations && (
              <div className="mt-2 text-xs text-orange-600 bg-orange-50 p-2 rounded">
                <div className="font-medium mb-1">Recommendations:</div>
                {check.recommendations.map((rec, idx) => (
                  <div key={idx}>• {rec}</div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="text-xs text-gray-500 border-t pt-2">
        💡 Based on company profile from Settings
      </div>
    </div>
  );
};

export default BrandConsistencyChecker;