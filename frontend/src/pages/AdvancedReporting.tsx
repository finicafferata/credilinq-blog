/**
 * Advanced Reporting Page
 * Generate comprehensive reports in multiple formats (PDF, CSV, JSON, Excel)
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  DocumentChartBarIcon,
  ArrowDownTrayIcon,
  Cog6ToothIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  DocumentTextIcon,
  TableCellsIcon,
  CodeBracketIcon,
  PresentationChartBarIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';

interface ReportConfig {
  reportType: string;
  format: string;
  title: string;
  description: string;
  includeCharts: boolean;
  includeRawData: boolean;
  dateRangeDays: number;
  competitorIds: string[];
  industry: string;
  customSections: string[];
}

interface GeneratedReport {
  report_id: string;
  file_path: string;
  file_size_bytes: number;
  generation_time_ms: number;
  created_at: string;
  sections_count: number;
  metadata: Record<string, any>;
}

const REPORT_TYPES = [
  { value: 'executive_summary', label: 'Executive Summary', description: 'High-level overview and key findings' },
  { value: 'detailed_analysis', label: 'Detailed Analysis', description: 'Comprehensive analysis across all dimensions' },
  { value: 'trend_report', label: 'Trend Report', description: 'Focus on trends and emerging patterns' },
  { value: 'competitive_landscape', label: 'Competitive Landscape', description: 'Market positioning and competitive gaps' },
  { value: 'content_analysis', label: 'Content Analysis', description: 'AI-powered content quality and strategy analysis' },
  { value: 'social_media_report', label: 'Social Media Report', description: 'Social media engagement and viral content analysis' },
];

const REPORT_FORMATS = [
  { value: 'pdf', label: 'PDF', icon: DocumentTextIcon, description: 'Professional formatted document' },
  { value: 'excel', label: 'Excel', icon: TableCellsIcon, description: 'Spreadsheet with charts and data' },
  { value: 'csv', label: 'CSV', icon: TableCellsIcon, description: 'Raw data in CSV format' },
  { value: 'json', label: 'JSON', icon: CodeBracketIcon, description: 'Structured data format' },
  { value: 'html', label: 'HTML', icon: PresentationChartBarIcon, description: 'Web-viewable report' },
];

const CUSTOM_SECTIONS = [
  'competitor_overview',
  'market_positioning',
  'content_gaps',
  'social_overview',
  'engagement_analysis',
  'viral_content',
  'trending_topics',
  'emerging_patterns',
  'trend_predictions'
];

export function AdvancedReporting() {
  const [config, setConfig] = useState<ReportConfig>({
    reportType: 'executive_summary',
    format: 'pdf',
    title: '',
    description: '',
    includeCharts: true,
    includeRawData: false,
    dateRangeDays: 30,
    competitorIds: [],
    industry: '',
    customSections: []
  });

  const [competitors, setCompetitors] = useState<any[]>([]);
  const [generating, setGenerating] = useState(false);
  const [generatedReport, setGeneratedReport] = useState<GeneratedReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCompetitors();
  }, []);

  const loadCompetitors = async () => {
    try {
      const competitorList = await CompetitorIntelligenceAPI.listCompetitors();
      setCompetitors(competitorList);
    } catch (err) {
      console.error('Failed to load competitors:', err);
    }
  };

  const handleGenerateReport = async () => {
    if (!config.title.trim()) {
      setError('Please enter a report title');
      return;
    }

    try {
      setGenerating(true);
      setError(null);

      const response = await CompetitorIntelligenceAPI.generateReport({
        reportType: config.reportType,
        format: config.format,
        title: config.title,
        description: config.description,
        includeCharts: config.includeCharts,
        includeRawData: config.includeRawData,
        dateRangeDays: config.dateRangeDays,
        competitorIds: config.competitorIds,
        industry: config.industry || undefined,
        customSections: config.customSections
      });

      setGeneratedReport(response.report);
    } catch (err: any) {
      setError(err.message || 'Failed to generate report');
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!generatedReport) return;

    try {
      const blob = await CompetitorIntelligenceAPI.downloadReport(generatedReport.report_id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report-${generatedReport.report_id}.${config.format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setError(err.message || 'Failed to download report');
    }
  };

  const updateConfig = (updates: Partial<ReportConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  };

  const toggleCompetitor = (competitorId: string) => {
    const newIds = config.competitorIds.includes(competitorId)
      ? config.competitorIds.filter(id => id !== competitorId)
      : [...config.competitorIds, competitorId];
    updateConfig({ competitorIds: newIds });
  };

  const toggleCustomSection = (section: string) => {
    const newSections = config.customSections.includes(section)
      ? config.customSections.filter(s => s !== section)
      : [...config.customSections, section];
    updateConfig({ customSections: newSections });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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
                    <span className="text-gray-900 font-medium">Advanced Reporting</span>
                  </li>
                </ol>
              </nav>
              <h1 className="text-3xl font-bold text-gray-900 mt-2 flex items-center">
                <DocumentChartBarIcon className="h-8 w-8 mr-3 text-blue-600" />
                Advanced Reporting
              </h1>
              <p className="mt-2 text-gray-600">
                Generate comprehensive reports in multiple formats with customizable sections and insights
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Configuration Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Basic Configuration */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-6 flex items-center">
                <Cog6ToothIcon className="h-5 w-5 mr-2 text-blue-600" />
                Report Configuration
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Report Type */}
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Report Type
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {REPORT_TYPES.map((type) => (
                      <div
                        key={type.value}
                        className={`border rounded-lg p-3 cursor-pointer transition-all ${
                          config.reportType === type.value
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => updateConfig({ reportType: type.value })}
                      >
                        <div className="font-medium text-sm">{type.label}</div>
                        <div className="text-xs text-gray-500 mt-1">{type.description}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Format Selection */}
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Output Format
                  </label>
                  <div className="grid grid-cols-5 gap-3">
                    {REPORT_FORMATS.map((format) => {
                      const IconComponent = format.icon;
                      return (
                        <div
                          key={format.value}
                          className={`border rounded-lg p-3 text-center cursor-pointer transition-all ${
                            config.format === format.value
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                          onClick={() => updateConfig({ format: format.value })}
                        >
                          <IconComponent className="h-6 w-6 mx-auto mb-1 text-gray-600" />
                          <div className="text-xs font-medium">{format.label}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Title and Description */}
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Report Title *
                  </label>
                  <input
                    type="text"
                    value={config.title}
                    onChange={(e) => updateConfig({ title: e.target.value })}
                    placeholder="Q4 2024 Competitive Intelligence Report"
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description (optional)
                  </label>
                  <textarea
                    value={config.description}
                    onChange={(e) => updateConfig({ description: e.target.value })}
                    placeholder="Comprehensive analysis of competitive landscape and market trends..."
                    rows={3}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Date Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Date Range
                  </label>
                  <select
                    value={config.dateRangeDays}
                    onChange={(e) => updateConfig({ dateRangeDays: parseInt(e.target.value) })}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value={7}>Last 7 days</option>
                    <option value={30}>Last 30 days</option>
                    <option value={90}>Last 90 days</option>
                    <option value={180}>Last 6 months</option>
                    <option value={365}>Last year</option>
                  </select>
                </div>

                {/* Industry Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Industry (optional)
                  </label>
                  <select
                    value={config.industry}
                    onChange={(e) => updateConfig({ industry: e.target.value })}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">All Industries</option>
                    <option value="fintech">FinTech</option>
                    <option value="saas">SaaS</option>
                    <option value="ecommerce">E-commerce</option>
                    <option value="healthcare">Healthcare</option>
                    <option value="education">Education</option>
                  </select>
                </div>

                {/* Options */}
                <div className="md:col-span-2 space-y-3">
                  <label className="block text-sm font-medium text-gray-700">
                    Report Options
                  </label>
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={config.includeCharts}
                        onChange={(e) => updateConfig({ includeCharts: e.target.checked })}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">Include charts and visualizations</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={config.includeRawData}
                        onChange={(e) => updateConfig({ includeRawData: e.target.checked })}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">Include raw data tables</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Competitor Selection */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Competitor Selection (optional)
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                Select specific competitors to include in the report. If none selected, all competitors will be included.
              </p>
              
              {competitors.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {competitors.map((competitor) => (
                    <label key={competitor.id} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={config.competitorIds.includes(competitor.id)}
                        onChange={() => toggleCompetitor(competitor.id)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">{competitor.name}</span>
                      <span className="ml-auto text-xs text-gray-500 capitalize">{competitor.tier}</span>
                    </label>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>No competitors found. Add competitors to enable selection.</p>
                </div>
              )}
            </div>

            {/* Custom Sections */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Additional Sections
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                Add custom sections to your report for specialized analysis.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {CUSTOM_SECTIONS.map((section) => (
                  <label key={section} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.customSections.includes(section)}
                      onChange={() => toggleCustomSection(section)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-700 capitalize">
                      {section.replace('_', ' ')}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Generation Panel */}
          <div className="space-y-6">
            {/* Generate Report */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Generate Report</h3>
              
              <div className="space-y-4">
                <div className="text-sm text-gray-600">
                  <p><strong>Type:</strong> {REPORT_TYPES.find(t => t.value === config.reportType)?.label}</p>
                  <p><strong>Format:</strong> {config.format.toUpperCase()}</p>
                  <p><strong>Date Range:</strong> Last {config.dateRangeDays} days</p>
                  <p><strong>Competitors:</strong> {config.competitorIds.length || 'All'}</p>
                </div>

                <button
                  onClick={handleGenerateReport}
                  disabled={generating || !config.title.trim()}
                  className={`w-full flex items-center justify-center px-4 py-3 rounded-md font-medium ${
                    generating || !config.title.trim()
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {generating ? (
                    <>
                      <ClockIcon className="h-5 w-5 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <DocumentChartBarIcon className="h-5 w-5 mr-2" />
                      Generate Report
                    </>
                  )}
                </button>

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-md p-4">
                    <div className="flex">
                      <XCircleIcon className="h-5 w-5 text-red-400" />
                      <div className="ml-3">
                        <p className="text-sm text-red-800">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Generated Report */}
            {generatedReport && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <CheckCircleIcon className="h-5 w-5 mr-2 text-green-600" />
                  Report Generated
                </h3>
                
                <div className="space-y-3">
                  <div className="text-sm text-gray-600">
                    <p><strong>Report ID:</strong> {generatedReport.report_id}</p>
                    <p><strong>Size:</strong> {formatFileSize(generatedReport.file_size_bytes)}</p>
                    <p><strong>Generation Time:</strong> {generatedReport.generation_time_ms}ms</p>
                    <p><strong>Sections:</strong> {generatedReport.sections_count}</p>
                    <p><strong>Created:</strong> {new Date(generatedReport.created_at).toLocaleString()}</p>
                  </div>

                  <button
                    onClick={handleDownloadReport}
                    className="w-full flex items-center justify-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                  >
                    <ArrowDownTrayIcon className="h-5 w-5 mr-2" />
                    Download Report
                  </button>
                </div>
              </div>
            )}

            {/* Report Examples */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Report Examples</h3>
              
              <div className="space-y-3 text-sm">
                <div>
                  <p className="font-medium">Executive Summary</p>
                  <p className="text-gray-600">Key findings, metrics, and strategic recommendations</p>
                </div>
                <div>
                  <p className="font-medium">Trend Report</p>
                  <p className="text-gray-600">Emerging patterns and predictive analysis</p>
                </div>
                <div>
                  <p className="font-medium">Content Analysis</p>
                  <p className="text-gray-600">AI-powered quality scoring and topic insights</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdvancedReporting;