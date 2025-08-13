/**
 * AI Content Analysis Page
 * Provides AI-powered content analysis with quality scoring, topic extraction, and competitive insights
 */

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  DocumentMagnifyingGlassIcon,
  SparklesIcon,
  ChartBarIcon,
  EyeIcon,
  HeartIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';

interface AnalysisResult {
  content_id: string;
  content_type: string;
  processing_time_ms: number;
  analysis_timestamp: string;
  topics: {
    primary_topics: string[];
    secondary_topics: string[];
    entities: string[];
    keywords: string[];
    themes: string[];
    confidence: number;
  };
  quality: {
    overall_score: number;
    readability_score: number;
    engagement_potential: number;
    seo_score: number;
    information_density: number;
    originality_score: number;
    structure_score: number;
    quality_rating: string;
  };
  sentiment: {
    overall_sentiment: string;
    sentiment_score: number;
    emotional_tone: string[];
    confidence: number;
    key_phrases: string[];
  };
  competitive_insights: {
    content_strategy: string;
    target_audience: string;
    positioning: string;
    strengths: string[];
    weaknesses: string[];
    differentiation_opportunities: string[];
    threat_level: string;
  };
}

export function AIContentAnalysis() {
  const [content, setContent] = useState('');
  const [contentUrl, setContentUrl] = useState('');
  const [competitorName, setCompetitorName] = useState('');
  const [contentType, setContentType] = useState('article');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!content.trim()) {
      setError('Please enter content to analyze');
      return;
    }

    try {
      setAnalyzing(true);
      setError(null);
      
      const response = await CompetitorIntelligenceAPI.analyzeContent({
        content: content.trim(),
        contentUrl: contentUrl.trim() || undefined,
        competitorName: competitorName.trim() || undefined,
        contentType: contentType
      });

      setResult(response.analysis);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze content');
    } finally {
      setAnalyzing(false);
    }
  };

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-100';
      case 'negative': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getThreatColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-green-600 bg-green-100';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <nav className="flex" aria-label="Breadcrumb">
                <ol className="flex items-center space-x-4">
                  <li>
                    <Link to="/competitor-intelligence" className="text-gray-400 hover:text-gray-500">
                      Competitor Intelligence
                    </Link>
                  </li>
                  <li>
                    <span className="text-gray-400">/</span>
                  </li>
                  <li>
                    <span className="text-gray-900 font-medium">AI Content Analysis</span>
                  </li>
                </ol>
              </nav>
              <h1 className="text-3xl font-bold text-gray-900 mt-2 flex items-center">
                <SparklesIcon className="h-8 w-8 mr-3 text-purple-600" />
                AI Content Analysis
              </h1>
              <p className="mt-2 text-gray-600">
                Analyze content with AI for quality scoring, topic extraction, and competitive insights
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-6 flex items-center">
              <DocumentMagnifyingGlassIcon className="h-5 w-5 mr-2 text-blue-600" />
              Content Input
            </h2>
            
            <div className="space-y-4">
              {/* Content Type Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Content Type
                </label>
                <select
                  value={contentType}
                  onChange={(e) => setContentType(e.target.value)}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="article">Article</option>
                  <option value="blog_post">Blog Post</option>
                  <option value="press_release">Press Release</option>
                  <option value="case_study">Case Study</option>
                  <option value="whitepaper">Whitepaper</option>
                  <option value="social_post">Social Post</option>
                </select>
              </div>

              {/* Optional Fields */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Content URL (optional)
                </label>
                <input
                  type="url"
                  value={contentUrl}
                  onChange={(e) => setContentUrl(e.target.value)}
                  placeholder="https://example.com/article"
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Competitor Name (optional)
                </label>
                <input
                  type="text"
                  value={competitorName}
                  onChange={(e) => setCompetitorName(e.target.value)}
                  placeholder="Company name"
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Content Text Area */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Content to Analyze *
                </label>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Paste the content you want to analyze here..."
                  rows={12}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="mt-1 text-sm text-gray-500">
                  {content.length} characters
                </p>
              </div>

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={analyzing || !content.trim()}
                className={`w-full flex items-center justify-center px-4 py-3 rounded-md font-medium ${
                  analyzing || !content.trim()
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {analyzing ? (
                  <>
                    <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <SparklesIcon className="h-5 w-5 mr-2" />
                    Analyze Content
                  </>
                )}
              </button>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-4">
                  <div className="flex">
                    <ExclamationTriangleIcon className="h-5 w-5 text-red-400" />
                    <div className="ml-3">
                      <p className="text-sm text-red-800">{error}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Analysis Overview */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Analysis Overview</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Content ID:</span>
                      <p className="font-medium">{result.content_id}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Type:</span>
                      <p className="font-medium capitalize">{result.content_type.replace('_', ' ')}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Processing Time:</span>
                      <p className="font-medium">{result.processing_time_ms}ms</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Analyzed:</span>
                      <p className="font-medium">{new Date(result.analysis_timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                </div>

                {/* Quality Metrics */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                    <ChartBarIcon className="h-5 w-5 mr-2 text-blue-600" />
                    Quality Metrics
                  </h3>
                  
                  <div className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Overall Score</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getQualityColor(result.quality.overall_score)}`}>
                        {result.quality.overall_score.toFixed(1)}/100
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${result.quality.overall_score}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Readability:</span>
                      <p className="font-medium">{result.quality.readability_score.toFixed(1)}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Engagement:</span>
                      <p className="font-medium">{result.quality.engagement_potential.toFixed(1)}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">SEO Score:</span>
                      <p className="font-medium">{result.quality.seo_score.toFixed(1)}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Originality:</span>
                      <p className="font-medium">{result.quality.originality_score.toFixed(1)}</p>
                    </div>
                  </div>

                  <div className="mt-4">
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getQualityColor(result.quality.overall_score)}`}>
                      {result.quality.quality_rating.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Topics & Keywords */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                    <EyeIcon className="h-5 w-5 mr-2 text-green-600" />
                    Topics & Keywords
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Primary Topics</h4>
                      <div className="flex flex-wrap gap-2">
                        {result.topics.primary_topics.map((topic, index) => (
                          <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-md text-xs">
                            {topic}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Keywords</h4>
                      <div className="flex flex-wrap gap-2">
                        {result.topics.keywords.slice(0, 10).map((keyword, index) => (
                          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-md text-xs">
                            {keyword}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Entities</h4>
                      <div className="flex flex-wrap gap-2">
                        {result.topics.entities.slice(0, 8).map((entity, index) => (
                          <span key={index} className="px-2 py-1 bg-purple-100 text-purple-800 rounded-md text-xs">
                            {entity}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Sentiment Analysis */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                    <HeartIcon className="h-5 w-5 mr-2 text-pink-600" />
                    Sentiment Analysis
                  </h3>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-gray-500 text-sm">Overall Sentiment:</span>
                      <div className="mt-1">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(result.sentiment.overall_sentiment)}`}>
                          {result.sentiment.overall_sentiment.toUpperCase()}
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-gray-500 text-sm">Sentiment Score:</span>
                      <p className="font-medium text-lg">{result.sentiment.sentiment_score.toFixed(2)}</p>
                    </div>
                  </div>

                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Emotional Tone</h4>
                    <div className="flex flex-wrap gap-2">
                      {result.sentiment.emotional_tone.map((tone, index) => (
                        <span key={index} className="px-2 py-1 bg-pink-100 text-pink-800 rounded-md text-xs">
                          {tone}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Competitive Insights */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                    <LightBulbIcon className="h-5 w-5 mr-2 text-yellow-600" />
                    Competitive Insights
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Content Strategy</h4>
                      <p className="text-sm text-gray-600 mt-1">{result.competitive_insights.content_strategy}</p>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Target Audience</h4>
                      <p className="text-sm text-gray-600 mt-1">{result.competitive_insights.target_audience}</p>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Positioning</h4>
                      <p className="text-sm text-gray-600 mt-1">{result.competitive_insights.positioning}</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Strengths</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {result.competitive_insights.strengths.map((strength, index) => (
                            <li key={index} className="flex items-start">
                              <CheckCircleIcon className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                              {strength}
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Weaknesses</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {result.competitive_insights.weaknesses.map((weakness, index) => (
                            <li key={index} className="flex items-start">
                              <ExclamationTriangleIcon className="h-4 w-4 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                              {weakness}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Differentiation Opportunities</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {result.competitive_insights.differentiation_opportunities.map((opportunity, index) => (
                          <li key={index} className="flex items-start">
                            <LightBulbIcon className="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                            {opportunity}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <span className="text-sm text-gray-500">Threat Level:</span>
                      <div className="mt-1">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getThreatColor(result.competitive_insights.threat_level)}`}>
                          {result.competitive_insights.threat_level.toUpperCase()}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white rounded-lg shadow p-12 text-center">
                <DocumentMagnifyingGlassIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Analysis Yet</h3>
                <p className="text-gray-600">
                  Enter content on the left and click "Analyze Content" to see AI-powered insights.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AIContentAnalysis;