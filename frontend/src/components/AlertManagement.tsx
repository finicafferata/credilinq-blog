/**
 * Advanced Alert Management System
 * Provides customizable alerts, filtering, and notification management
 */

import React, { useState, useEffect } from 'react';
import {
  BellIcon,
  XMarkIcon,
  EyeIcon,
  EyeSlashIcon,
  FunnelIcon,
  Cog6ToothIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
  PlusIcon,
  TrashIcon,
  PencilIcon,
} from '@heroicons/react/24/outline';
import {
  BellIcon as BellIconSolid,
  ExclamationTriangleIcon as ExclamationTriangleIconSolid,
} from '@heroicons/react/24/solid';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import type { Alert, AlertSubscription } from '../types/competitor-intelligence';

interface AlertFilter {
  priority?: string;
  type?: string;
  competitor?: string;
  dateRange?: { start: string; end: string };
  unreadOnly?: boolean;
}

interface AlertRule {
  id?: string;
  name: string;
  description: string;
  conditions: {
    triggerType: 'content_change' | 'trend_spike' | 'competitor_activity' | 'sentiment_change' | 'keyword_mention';
    threshold?: number;
    keywords?: string[];
    competitors?: string[];
  };
  priority: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  deliveryChannels: string[];
}

export function AlertManagement() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<AlertFilter>({});
  const [showCreateRule, setShowCreateRule] = useState(false);
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null);
  const [selectedAlerts, setSelectedAlerts] = useState<string[]>([]);
  const [alertSummary, setAlertSummary] = useState<{
    total: number;
    unread: number;
    byPriority: { [key: string]: number };
    recentActivity: number;
  } | null>(null);

  const [newRule, setNewRule] = useState<AlertRule>({
    name: '',
    description: '',
    conditions: {
      triggerType: 'content_change',
      threshold: 5,
      keywords: [],
      competitors: []
    },
    priority: 'medium',
    enabled: true,
    deliveryChannels: ['email']
  });

  useEffect(() => {
    loadAlerts();
    loadAlertSummary();
    loadAlertRules();
  }, [filter]);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const alertsData = await CompetitorIntelligenceAPI.getAlerts({
        limit: 50,
        priority: filter.priority,
        alertType: filter.type,
        competitorId: filter.competitor,
        unreadOnly: filter.unreadOnly || false
      });
      
      // Mock alert data since the API might not have real alerts
      const mockAlerts: Alert[] = [
        {
          id: '1',
          alertType: 'content_spike',
          priority: 'high' as any,
          title: 'Content Spike Detected',
          message: 'TechCorp published 5 new blog posts in the last 24 hours, 3x their usual rate',
          data: { competitor: 'TechCorp', count: 5, threshold: 2 },
          competitorIds: ['comp-1'],
          trendIds: [],
          contentIds: ['content-1', 'content-2'],
          createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          recipients: ['user@example.com'],
          metadata: { category: 'content_monitoring' }
        },
        {
          id: '2',
          alertType: 'trend_emergence',
          priority: 'medium' as any,
          title: 'New Trend Emerging',
          message: 'AI Automation is gaining traction with 25% increase in mentions',
          data: { trend: 'AI Automation', growth: 25 },
          competitorIds: ['comp-1', 'comp-2'],
          trendIds: ['trend-1'],
          contentIds: [],
          createdAt: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
          recipients: ['user@example.com'],
          metadata: { category: 'trend_analysis' }
        },
        {
          id: '3',
          alertType: 'sentiment_change',
          priority: 'critical' as any,
          title: 'Negative Sentiment Spike',
          message: 'FinanceFlow showing 40% negative sentiment increase in social mentions',
          data: { competitor: 'FinanceFlow', sentiment_change: -40 },
          competitorIds: ['comp-3'],
          trendIds: [],
          contentIds: [],
          createdAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
          acknowledgedAt: new Date().toISOString(),
          recipients: ['user@example.com'],
          metadata: { category: 'sentiment_monitoring' }
        }
      ];
      
      setAlerts(mockAlerts);
    } catch (err: any) {
      setError(err.message || 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  const loadAlertSummary = async () => {
    try {
      const summary = await CompetitorIntelligenceAPI.getAlertSummary();
      
      // Mock summary data
      const mockSummary = {
        total: 15,
        unread: 8,
        byPriority: {
          critical: 2,
          high: 5,
          medium: 6,
          low: 2
        },
        recentActivity: 3
      };
      
      setAlertSummary(mockSummary);
    } catch (err) {
      console.error('Failed to load alert summary:', err);
    }
  };

  const loadAlertRules = () => {
    // Mock alert rules - in a real app, these would come from an API
    const mockRules: AlertRule[] = [
      {
        id: 'rule-1',
        name: 'High Content Volume',
        description: 'Alert when competitor publishes more than 3 posts in 24 hours',
        conditions: {
          triggerType: 'content_change',
          threshold: 3,
          keywords: [],
          competitors: []
        },
        priority: 'high',
        enabled: true,
        deliveryChannels: ['email', 'slack']
      },
      {
        id: 'rule-2',
        name: 'Trending Keywords',
        description: 'Alert when specific keywords show trending behavior',
        conditions: {
          triggerType: 'keyword_mention',
          threshold: 50,
          keywords: ['AI', 'automation', 'fintech'],
          competitors: []
        },
        priority: 'medium',
        enabled: true,
        deliveryChannels: ['email']
      }
    ];
    
    setAlertRules(mockRules);
  };

  const handleAlertAction = async (alertId: string, action: 'read' | 'dismiss' | 'acknowledge') => {
    try {
      switch (action) {
        case 'read':
          await CompetitorIntelligenceAPI.markAlertRead(alertId);
          break;
        case 'dismiss':
          await CompetitorIntelligenceAPI.dismissAlert(alertId);
          break;
        case 'acknowledge':
          await CompetitorIntelligenceAPI.acknowledgeAlert(alertId);
          break;
      }
      
      // Update local state
      setAlerts(alerts.map(alert => 
        alert.id === alertId 
          ? { ...alert, acknowledgedAt: action === 'acknowledge' ? new Date().toISOString() : alert.acknowledgedAt }
          : alert
      ));
      
      if (action === 'dismiss') {
        setAlerts(alerts.filter(alert => alert.id !== alertId));
      }
    } catch (err: any) {
      setError(err.message || `Failed to ${action} alert`);
    }
  };

  const handleBulkAction = async (action: 'read' | 'dismiss' | 'acknowledge') => {
    try {
      await Promise.all(selectedAlerts.map(alertId => handleAlertAction(alertId, action)));
      setSelectedAlerts([]);
    } catch (err: any) {
      setError(err.message || `Failed to perform bulk ${action}`);
    }
  };

  const createAlertRule = async () => {
    try {
      // In a real app, this would make an API call
      const ruleWithId = { ...newRule, id: `rule-${Date.now()}` };
      setAlertRules([...alertRules, ruleWithId]);
      
      setShowCreateRule(false);
      setNewRule({
        name: '',
        description: '',
        conditions: {
          triggerType: 'content_change',
          threshold: 5,
          keywords: [],
          competitors: []
        },
        priority: 'medium',
        enabled: true,
        deliveryChannels: ['email']
      });
    } catch (err: any) {
      setError(err.message || 'Failed to create alert rule');
    }
  };

  const toggleRuleEnabled = (ruleId: string) => {
    setAlertRules(alertRules.map(rule => 
      rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
    ));
  };

  const deleteRule = (ruleId: string) => {
    setAlertRules(alertRules.filter(rule => rule.id !== ruleId));
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'high':
        return <ExclamationTriangleIconSolid className="h-5 w-5 text-orange-500" />;
      case 'medium':
        return <InformationCircleIcon className="h-5 w-5 text-blue-500" />;
      case 'low':
        return <CheckCircleIcon className="h-5 w-5 text-gray-500" />;
      default:
        return <BellIconSolid className="h-5 w-5 text-gray-400" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'high':
        return 'bg-orange-50 border-orange-200 text-orange-800';
      case 'medium':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'low':
        return 'bg-gray-50 border-gray-200 text-gray-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Alert Summary */}
      {alertSummary && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <BellIconSolid className="h-5 w-5 mr-2 text-blue-600" />
              Alert Overview
            </h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{alertSummary.total}</div>
              <div className="text-sm text-gray-600">Total Alerts</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{alertSummary.unread}</div>
              <div className="text-sm text-gray-600">Unread</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{alertSummary.byPriority.critical || 0}</div>
              <div className="text-sm text-gray-600">Critical</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{alertSummary.recentActivity}</div>
              <div className="text-sm text-gray-600">Last Hour</div>
            </div>
          </div>
        </div>
      )}

      {/* Alert Management */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Recent Alerts</h2>
            
            <div className="flex items-center space-x-3">
              {/* Filters */}
              <div className="flex items-center space-x-2">
                <FunnelIcon className="h-4 w-4 text-gray-500" />
                <select
                  value={filter.priority || ''}
                  onChange={(e) => setFilter({ ...filter, priority: e.target.value || undefined })}
                  className="border border-gray-300 rounded px-2 py-1 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Priorities</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                
                <select
                  value={filter.type || ''}
                  onChange={(e) => setFilter({ ...filter, type: e.target.value || undefined })}
                  className="border border-gray-300 rounded px-2 py-1 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Types</option>
                  <option value="content_spike">Content Spike</option>
                  <option value="trend_emergence">Trend Emergence</option>
                  <option value="sentiment_change">Sentiment Change</option>
                  <option value="keyword_mention">Keyword Mention</option>
                </select>
                
                <label className="flex items-center text-sm">
                  <input
                    type="checkbox"
                    checked={filter.unreadOnly || false}
                    onChange={(e) => setFilter({ ...filter, unreadOnly: e.target.checked })}
                    className="mr-1"
                  />
                  Unread only
                </label>
              </div>
              
              <button
                onClick={() => setShowCreateRule(true)}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 flex items-center space-x-1"
              >
                <PlusIcon className="h-4 w-4" />
                <span>New Rule</span>
              </button>
            </div>
          </div>
          
          {/* Bulk Actions */}
          {selectedAlerts.length > 0 && (
            <div className="mt-3 flex items-center space-x-3 p-3 bg-blue-50 rounded">
              <span className="text-sm text-blue-800">
                {selectedAlerts.length} alert{selectedAlerts.length > 1 ? 's' : ''} selected
              </span>
              <button
                onClick={() => handleBulkAction('acknowledge')}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Acknowledge
              </button>
              <button
                onClick={() => handleBulkAction('dismiss')}
                className="text-sm text-red-600 hover:text-red-800"
              >
                Dismiss
              </button>
              <button
                onClick={() => setSelectedAlerts([])}
                className="text-sm text-gray-600 hover:text-gray-800"
              >
                Clear Selection
              </button>
            </div>
          )}
        </div>
        
        <div className="max-h-96 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span className="ml-2 text-gray-600">Loading alerts...</span>
            </div>
          ) : alerts.length === 0 ? (
            <div className="text-center py-8">
              <BellIcon className="h-12 w-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600">No alerts match your filters</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-4 hover:bg-gray-50 ${alert.acknowledgedAt ? 'opacity-75' : ''}`}
                >
                  <div className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      checked={selectedAlerts.includes(alert.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedAlerts([...selectedAlerts, alert.id]);
                        } else {
                          setSelectedAlerts(selectedAlerts.filter(id => id !== alert.id));
                        }
                      }}
                      className="mt-1"
                    />
                    
                    <div className="flex-shrink-0 mt-1">
                      {getPriorityIcon(alert.priority)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-medium text-gray-900">{alert.title}</h3>
                        <div className="flex items-center space-x-2">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getPriorityColor(alert.priority)}`}>
                            {alert.priority}
                          </span>
                          <span className="text-xs text-gray-500">
                            {new Date(alert.createdAt).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{alert.message}</p>
                      
                      <div className="mt-2 flex items-center space-x-4">
                        {!alert.acknowledgedAt && (
                          <button
                            onClick={() => handleAlertAction(alert.id, 'acknowledge')}
                            className="text-xs text-blue-600 hover:text-blue-800"
                          >
                            Acknowledge
                          </button>
                        )}
                        <button
                          onClick={() => handleAlertAction(alert.id, 'dismiss')}
                          className="text-xs text-red-600 hover:text-red-800"
                        >
                          Dismiss
                        </button>
                        <span className="text-xs text-gray-500">
                          Type: {alert.alertType.replace('_', ' ')}
                        </span>
                        {alert.competitorIds.length > 0 && (
                          <span className="text-xs text-gray-500">
                            Competitors: {alert.competitorIds.length}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Alert Rules Management */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <Cog6ToothIcon className="h-5 w-5 mr-2 text-gray-600" />
            Alert Rules
          </h2>
        </div>
        
        <div className="p-6">
          <div className="space-y-4">
            {alertRules.map((rule) => (
              <div key={rule.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <h3 className="font-medium text-gray-900">{rule.name}</h3>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                        rule.enabled 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {rule.enabled ? 'Active' : 'Disabled'}
                      </span>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getPriorityColor(rule.priority)}`}>
                        {rule.priority}
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-gray-600">{rule.description}</p>
                    <div className="mt-2 text-xs text-gray-500">
                      Trigger: {rule.conditions.triggerType.replace('_', ' ')} 
                      {rule.conditions.threshold && ` (threshold: ${rule.conditions.threshold})`}
                      {rule.conditions.keywords && rule.conditions.keywords.length > 0 && (
                        <span> • Keywords: {rule.conditions.keywords.join(', ')}</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => toggleRuleEnabled(rule.id!)}
                      className="p-1 text-gray-400 hover:text-gray-600"
                    >
                      {rule.enabled ? <EyeIcon className="h-4 w-4" /> : <EyeSlashIcon className="h-4 w-4" />}
                    </button>
                    <button
                      onClick={() => setEditingRule(rule)}
                      className="p-1 text-gray-400 hover:text-gray-600"
                    >
                      <PencilIcon className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => deleteRule(rule.id!)}
                      className="p-1 text-gray-400 hover:text-red-600"
                    >
                      <TrashIcon className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Create/Edit Rule Modal */}
      {(showCreateRule || editingRule) && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">
                {editingRule ? 'Edit Alert Rule' : 'Create Alert Rule'}
              </h2>
            </div>
            
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={editingRule ? editingRule.name : newRule.name}
                  onChange={(e) => editingRule 
                    ? setEditingRule({ ...editingRule, name: e.target.value })
                    : setNewRule({ ...newRule, name: e.target.value })}
                  className="w-full border border-gray-300 rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter rule name"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={editingRule ? editingRule.description : newRule.description}
                  onChange={(e) => editingRule
                    ? setEditingRule({ ...editingRule, description: e.target.value })
                    : setNewRule({ ...newRule, description: e.target.value })}
                  className="w-full border border-gray-300 rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="Describe when this alert should trigger"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Trigger Type</label>
                <select
                  value={editingRule ? editingRule.conditions.triggerType : newRule.conditions.triggerType}
                  onChange={(e) => {
                    const triggerType = e.target.value as any;
                    if (editingRule) {
                      setEditingRule({ ...editingRule, conditions: { ...editingRule.conditions, triggerType }});
                    } else {
                      setNewRule({ ...newRule, conditions: { ...newRule.conditions, triggerType }});
                    }
                  }}
                  className="w-full border border-gray-300 rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                >
                  <option value="content_change">Content Change</option>
                  <option value="trend_spike">Trend Spike</option>
                  <option value="competitor_activity">Competitor Activity</option>
                  <option value="sentiment_change">Sentiment Change</option>
                  <option value="keyword_mention">Keyword Mention</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                <select
                  value={editingRule ? editingRule.priority : newRule.priority}
                  onChange={(e) => editingRule
                    ? setEditingRule({ ...editingRule, priority: e.target.value as any })
                    : setNewRule({ ...newRule, priority: e.target.value as any })}
                  className="w-full border border-gray-300 rounded px-3 py-2 focus:ring-2 focus:ring-blue-500"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>
            
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowCreateRule(false);
                  setEditingRule(null);
                }}
                className="px-4 py-2 text-sm text-gray-700 border border-gray-300 rounded hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={createAlertRule}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                {editingRule ? 'Update Rule' : 'Create Rule'}
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="fixed bottom-4 right-4 bg-red-50 border border-red-200 rounded-lg p-4 shadow-lg">
          <div className="flex items-center">
            <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-sm text-red-700">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-3 text-red-400 hover:text-red-600"
            >
              <XMarkIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AlertManagement;