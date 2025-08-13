/**
 * Competitor Management page - List and manage competitors
 */

import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import { confirmAction } from '../lib/toast';
import { showErrorNotification } from '../lib/errors';
import type { CompetitorSummary, Industry, CompetitorTier } from '../types/competitor-intelligence';

export function CompetitorManagement() {
  const navigate = useNavigate();
  const [competitors, setCompetitors] = useState<CompetitorSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndustry, setSelectedIndustry] = useState<Industry | ''>('');
  const [selectedTier, setSelectedTier] = useState<CompetitorTier | ''>('');

  useEffect(() => {
    loadCompetitors();
  }, [selectedIndustry, selectedTier]);

  const loadCompetitors = async () => {
    try {
      setLoading(true);
      setError(null);

      const filters: any = { activeOnly: true };
      if (selectedIndustry) filters.industry = selectedIndustry;
      if (selectedTier) filters.tier = selectedTier;

      const data = await CompetitorIntelligenceAPI.listCompetitors(filters);
      setCompetitors(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load competitors');
      console.error('Failed to load competitors:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteCompetitor = async (competitorId: string, competitorName: string) => {
    const confirmed = await confirmAction(
      `Are you sure you want to delete "${competitorName}"? This action cannot be undone.`,
      async () => {
        try {
          await CompetitorIntelligenceAPI.deleteCompetitor(competitorId);
          // Refresh the list
          await loadCompetitors();
        } catch (err: any) {
          showErrorNotification(new Error('Failed to delete competitor: ' + err.message));
        }
      },
      {
        confirmText: 'Delete',
        cancelText: 'Cancel',
        type: 'danger'
      }
    );
  };

  const filteredCompetitors = competitors.filter(competitor =>
    competitor.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading competitors...</p>
        </div>
      </div>
    );
  }

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
                    <span className="text-gray-900 font-medium">Competitors</span>
                  </li>
                </ol>
              </nav>
              <h1 className="text-3xl font-bold text-gray-900 mt-2">Competitor Management</h1>
              <p className="mt-2 text-gray-600">
                Manage your competitor list and monitoring settings
              </p>
            </div>
            <Link
              to="/competitor-intelligence/competitors/new"
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              Add Competitor
            </Link>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-4">
            <div className="flex flex-col sm:flex-row gap-4">
              {/* Search */}
              <div className="flex-1">
                <div className="relative">
                  <MagnifyingGlassIcon className="h-5 w-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search competitors..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Industry Filter */}
              <div className="sm:w-48">
                <select
                  value={selectedIndustry}
                  onChange={(e) => setSelectedIndustry(e.target.value as Industry | '')}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">All Industries</option>
                  <option value="fintech">FinTech</option>
                  <option value="saas">SaaS</option>
                  <option value="ecommerce">E-commerce</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="education">Education</option>
                  <option value="marketing">Marketing</option>
                  <option value="technology">Technology</option>
                  <option value="finance">Finance</option>
                  <option value="retail">Retail</option>
                  <option value="media">Media</option>
                </select>
              </div>

              {/* Tier Filter */}
              <div className="sm:w-48">
                <select
                  value={selectedTier}
                  onChange={(e) => setSelectedTier(e.target.value as CompetitorTier | '')}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">All Tiers</option>
                  <option value="direct">Direct</option>
                  <option value="indirect">Indirect</option>
                  <option value="aspirational">Aspirational</option>
                  <option value="adjacent">Adjacent</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <p className="text-red-700">{error}</p>
            <button
              onClick={loadCompetitors}
              className="mt-2 text-red-600 hover:text-red-800 underline"
            >
              Try again
            </button>
          </div>
        )}

        {/* Competitors Grid */}
        {filteredCompetitors.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <EyeIcon className="h-12 w-12 text-gray-400 mx-auto" />
            <h3 className="mt-4 text-lg font-medium text-gray-900">
              {searchQuery || selectedIndustry || selectedTier ? 'No competitors found' : 'No competitors yet'}
            </h3>
            <p className="mt-2 text-gray-600">
              {searchQuery || selectedIndustry || selectedTier 
                ? 'Try adjusting your search or filters.'
                : 'Get started by adding your first competitor to monitor.'}
            </p>
            {!searchQuery && !selectedIndustry && !selectedTier && (
              <div className="mt-6">
                <Link
                  to="/competitor-intelligence/competitors/new"
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
                >
                  <PlusIcon className="h-4 w-4 mr-2" />
                  Add First Competitor
                </Link>
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredCompetitors.map((competitor) => (
              <CompetitorCard
                key={competitor.id}
                competitor={competitor}
                onDelete={() => handleDeleteCompetitor(competitor.id, competitor.name)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Competitor Card Component
interface CompetitorCardProps {
  competitor: CompetitorSummary;
  onDelete: () => void;
}

function CompetitorCard({ competitor, onDelete }: CompetitorCardProps) {
  const [showMenu, setShowMenu] = useState(false);

  const getTierColor = (tier: CompetitorTier) => {
    switch (tier) {
      case 'direct':
        return 'bg-red-100 text-red-800';
      case 'indirect':
        return 'bg-yellow-100 text-yellow-800';
      case 'aspirational':
        return 'bg-purple-100 text-purple-800';
      case 'adjacent':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getIndustryColor = (industry: Industry) => {
    const colors = [
      'bg-green-100 text-green-800',
      'bg-blue-100 text-blue-800',
      'bg-purple-100 text-purple-800',
      'bg-pink-100 text-pink-800',
      'bg-indigo-100 text-indigo-800',
    ];
    return colors[industry.length % colors.length];
  };

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-md transition-shadow duration-200">
      <div className="p-6">
        {/* Header with Menu */}
        <div className="flex justify-between items-start mb-4">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 truncate">
              {competitor.name}
            </h3>
            <div className="flex items-center gap-2 mt-2">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTierColor(competitor.tier)}`}>
                {competitor.tier}
              </span>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getIndustryColor(competitor.industry)}`}>
                {competitor.industry}
              </span>
            </div>
          </div>
          
          {/* Menu Dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-1 rounded-full hover:bg-gray-100"
            >
              <EllipsisVerticalIcon className="h-5 w-5 text-gray-400" />
            </button>
            
            {showMenu && (
              <div className="absolute right-0 mt-1 w-48 bg-white rounded-md shadow-lg z-10 border">
                <div className="py-1">
                  <Link
                    to={`/competitor-intelligence/competitors/${competitor.id}`}
                    className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    <EyeIcon className="h-4 w-4 mr-2" />
                    View Details
                  </Link>
                  <Link
                    to={`/competitor-intelligence/competitors/${competitor.id}/edit`}
                    className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    <PencilIcon className="h-4 w-4 mr-2" />
                    Edit
                  </Link>
                  <button
                    onClick={() => {
                      setShowMenu(false);
                      onDelete();
                    }}
                    className="flex items-center w-full px-4 py-2 text-sm text-red-700 hover:bg-red-50"
                  >
                    <TrashIcon className="h-4 w-4 mr-2" />
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-sm text-gray-500">Content Items</p>
            <p className="text-lg font-semibold text-gray-900">{competitor.contentCount}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Avg Engagement</p>
            <p className="text-lg font-semibold text-gray-900">{(competitor.avgEngagement || 0).toFixed(1)}</p>
          </div>
        </div>

        {/* Last Activity */}
        {competitor.lastActivity && (
          <div className="text-sm text-gray-500">
            Last activity: {new Date(competitor.lastActivity).toLocaleDateString()}
          </div>
        )}

        {/* Trending Score */}
        <div className="mt-4 flex items-center">
          <span className="text-sm text-gray-500 mr-2">Trending Score:</span>
          <div className="flex-1 bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full"
              style={{ width: `${Math.min(competitor.trendingScore || 0, 100)}%` }}
            ></div>
          </div>
          <span className="text-sm font-medium text-gray-900 ml-2">
            {(competitor.trendingScore || 0).toFixed(0)}
          </span>
        </div>
      </div>
    </div>
  );
}

export default CompetitorManagement;