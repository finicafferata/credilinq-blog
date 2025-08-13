/**
 * Integration Management Page
 * Manage external integrations (Slack, Teams, email, webhooks, etc.)
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  LinkIcon,
  PlusIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  PaperAirplaneIcon,
  TrashIcon,
  PlayIcon,
  Cog6ToothIcon,
  BellAlertIcon,
} from '@heroicons/react/24/outline';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import { confirmAction } from '../lib/toast';

interface Integration {
  name: string;
  type: string;
  enabled: boolean;
  event_filters: string[] | null;
  priority_threshold: string;
  has_webhook_url: boolean;
  has_api_token: boolean;
  channel: string | null;
}

interface IntegrationConfig {
  integrationType: string;
  name: string;
  webhookUrl: string;
  apiToken: string;
  channel: string;
  emailSettings: Record<string, string>;
  eventFilters: string[];
  priorityThreshold: string;
  enabled: boolean;
}

const INTEGRATION_TYPES = [
  { value: 'slack', label: 'Slack', description: 'Send notifications to Slack channels' },
  { value: 'teams', label: 'Microsoft Teams', description: 'Send notifications to Teams channels' },
  { value: 'email', label: 'Email', description: 'Send email notifications' },
  { value: 'webhook', label: 'Generic Webhook', description: 'HTTP POST to custom endpoint' },
  { value: 'discord', label: 'Discord', description: 'Send notifications to Discord channels' },
  { value: 'telegram', label: 'Telegram', description: 'Send notifications to Telegram chats' },
  { value: 'zapier', label: 'Zapier', description: 'Integrate with Zapier workflows' },
  { value: 'ifttt', label: 'IFTTT', description: 'Trigger IFTTT applets' },
];

const EVENT_TYPES = [
  { value: 'new_trend', label: 'New Trends', description: 'When new trends are detected' },
  { value: 'high_priority_alert', label: 'High Priority Alerts', description: 'Critical competitive alerts' },
  { value: 'competitor_update', label: 'Competitor Updates', description: 'New competitor content or changes' },
  { value: 'report_generated', label: 'Report Generated', description: 'When reports are completed' },
  { value: 'system_health', label: 'System Health', description: 'System status updates' },
];

const PRIORITY_LEVELS = [
  { value: 'low', label: 'Low', description: 'All notifications' },
  { value: 'normal', label: 'Normal', description: 'Normal and higher priority' },
  { value: 'high', label: 'High', description: 'High and urgent only' },
  { value: 'urgent', label: 'Urgent', description: 'Urgent notifications only' },
];

export function IntegrationManagement() {
  const [integrations, setIntegrations] = useState<Record<string, Integration>>({});
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [testingIntegration, setTestingIntegration] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [newConfig, setNewConfig] = useState<IntegrationConfig>({
    integrationType: 'slack',
    name: '',
    webhookUrl: '',
    apiToken: '',
    channel: '',
    emailSettings: {},
    eventFilters: [],
    priorityThreshold: 'normal',
    enabled: true
  });

  useEffect(() => {
    loadIntegrations();
  }, []);

  const loadIntegrations = async () => {
    try {
      setLoading(true);
      const response = await CompetitorIntelligenceAPI.getIntegrationsStatus();
      setIntegrations(response.integrations);
    } catch (err: any) {
      setError('Failed to load integrations');
    } finally {
      setLoading(false);
    }
  };

  const handleAddIntegration = async () => {
    if (!newConfig.name.trim()) {
      setError('Please enter an integration name');
      return;
    }

    try {
      setError(null);
      await CompetitorIntelligenceAPI.registerIntegration({
        integrationType: newConfig.integrationType,
        name: newConfig.name,
        webhookUrl: newConfig.webhookUrl || undefined,
        apiToken: newConfig.apiToken || undefined,
        channel: newConfig.channel || undefined,
        emailSettings: Object.keys(newConfig.emailSettings).length > 0 ? newConfig.emailSettings : undefined,
        eventFilters: newConfig.eventFilters.length > 0 ? newConfig.eventFilters : undefined,
        priorityThreshold: newConfig.priorityThreshold,
        enabled: newConfig.enabled
      });

      setSuccess('Integration added successfully');
      setShowAddForm(false);
      setNewConfig({
        integrationType: 'slack',
        name: '',
        webhookUrl: '',
        apiToken: '',
        channel: '',
        emailSettings: {},
        eventFilters: [],
        priorityThreshold: 'normal',
        enabled: true
      });
      
      // Reload integrations
      await loadIntegrations();
    } catch (err: any) {
      setError(err.message || 'Failed to add integration');
    }
  };

  const handleTestIntegration = async (integrationName: string) => {
    try {
      setTestingIntegration(integrationName);
      setError(null);
      
      const result = await CompetitorIntelligenceAPI.testIntegration(integrationName);
      
      if (result.success) {
        setSuccess(`Integration "${integrationName}" test successful`);
      } else {
        setError(`Integration test failed: ${result.error}`);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to test integration');
    } finally {
      setTestingIntegration(null);
    }
  };

  const handleRemoveIntegration = async (integrationName: string) => {
    const confirmed = await confirmAction(
      `Are you sure you want to remove the integration "${integrationName}"?`,
      async () => {
        try {
          setError(null);
          await CompetitorIntelligenceAPI.removeIntegration(integrationName);
          setSuccess(`Integration "${integrationName}" removed successfully`);
          await loadIntegrations();
        } catch (err: any) {
          setError(err.message || 'Failed to remove integration');
        }
      },
      {
        confirmText: 'Remove',
        cancelText: 'Cancel',
        type: 'warning'
      }
    );
  };

  const handleSendTestNotification = async () => {
    try {
      setError(null);
      await CompetitorIntelligenceAPI.sendCustomNotification({
        title: 'Test Notification',
        content: 'This is a test notification from the Integration Management page to verify all integrations are working correctly.',
        eventType: 'custom',
        priority: 'normal'
      });
      setSuccess('Test notification sent to all enabled integrations');
    } catch (err: any) {
      setError(err.message || 'Failed to send test notification');
    }
  };

  const updateNewConfig = (updates: Partial<IntegrationConfig>) => {
    setNewConfig(prev => ({ ...prev, ...updates }));
  };

  const toggleEventFilter = (eventType: string) => {
    const newFilters = newConfig.eventFilters.includes(eventType)
      ? newConfig.eventFilters.filter(e => e !== eventType)
      : [...newConfig.eventFilters, eventType];
    updateNewConfig({ eventFilters: newFilters });
  };

  const getIntegrationType = (type: string) => {
    return INTEGRATION_TYPES.find(t => t.value === type) || { label: type, description: '' };
  };

  const getStatusIcon = (integration: Integration) => {
    if (!integration.enabled) {
      return <XCircleIcon className="h-5 w-5 text-gray-400" />;
    }
    
    const hasRequiredFields = integration.has_webhook_url || integration.has_api_token;
    if (hasRequiredFields) {
      return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
    }
    
    return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading integrations...</p>
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
                    <span className="text-gray-900 font-medium">Integration Management</span>
                  </li>
                </ol>
              </nav>
              <h1 className="text-3xl font-bold text-gray-900 mt-2 flex items-center">
                <LinkIcon className="h-8 w-8 mr-3 text-green-600" />
                Integration Management
              </h1>
              <p className="mt-2 text-gray-600">
                Connect external tools and services to receive notifications and automate workflows
              </p>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={handleSendTestNotification}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <PaperAirplaneIcon className="h-4 w-4 mr-2" />
                Test All
              </button>
              <button
                onClick={() => setShowAddForm(true)}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
              >
                <PlusIcon className="h-4 w-4 mr-2" />
                Add Integration
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Alerts */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <XCircleIcon className="h-5 w-5 text-red-400" />
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-400 hover:text-red-600"
              >
                <XCircleIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        {success && (
          <div className="mb-6 bg-green-50 border border-green-200 rounded-md p-4">
            <div className="flex">
              <CheckCircleIcon className="h-5 w-5 text-green-400" />
              <div className="ml-3">
                <p className="text-sm text-green-800">{success}</p>
              </div>
              <button
                onClick={() => setSuccess(null)}
                className="ml-auto text-green-400 hover:text-green-600"
              >
                <XCircleIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        {/* Add Integration Form */}
        {showAddForm && (
          <div className="mb-8 bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-medium text-gray-900">Add New Integration</h2>
              <button
                onClick={() => setShowAddForm(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <XCircleIcon className="h-6 w-6" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Integration Type */}
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Integration Type
                </label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {INTEGRATION_TYPES.map((type) => (
                    <div
                      key={type.value}
                      className={`border rounded-lg p-3 cursor-pointer ${
                        newConfig.integrationType === type.value
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => updateNewConfig({ integrationType: type.value })}
                    >
                      <div className="font-medium text-sm">{type.label}</div>
                      <div className="text-xs text-gray-500 mt-1">{type.description}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Basic Settings */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Integration Name *
                </label>
                <input
                  type="text"
                  value={newConfig.name}
                  onChange={(e) => updateNewConfig({ name: e.target.value })}
                  placeholder="My Slack Integration"
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Priority Threshold
                </label>
                <select
                  value={newConfig.priorityThreshold}
                  onChange={(e) => updateNewConfig({ priorityThreshold: e.target.value })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {PRIORITY_LEVELS.map((level) => (
                    <option key={level.value} value={level.value}>
                      {level.label} - {level.description}
                    </option>
                  ))}
                </select>
              </div>

              {/* Connection Settings */}
              {['slack', 'teams', 'webhook', 'discord', 'zapier', 'ifttt'].includes(newConfig.integrationType) && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Webhook URL *
                  </label>
                  <input
                    type="url"
                    value={newConfig.webhookUrl}
                    onChange={(e) => updateNewConfig({ webhookUrl: e.target.value })}
                    placeholder="https://hooks.slack.com/services/..."
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {newConfig.integrationType === 'telegram' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Bot Token *
                  </label>
                  <input
                    type="text"
                    value={newConfig.apiToken}
                    onChange={(e) => updateNewConfig({ apiToken: e.target.value })}
                    placeholder="123456789:ABCdef..."
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {['slack', 'teams', 'discord', 'telegram'].includes(newConfig.integrationType) && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Channel/Chat ID
                  </label>
                  <input
                    type="text"
                    value={newConfig.channel}
                    onChange={(e) => updateNewConfig({ channel: e.target.value })}
                    placeholder="#notifications or @username"
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {/* Event Filters */}
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Event Filters (optional)
                </label>
                <p className="text-sm text-gray-600 mb-3">
                  Select which events should trigger notifications. If none selected, all events will be sent.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {EVENT_TYPES.map((event) => (
                    <label key={event.value} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={newConfig.eventFilters.includes(event.value)}
                        onChange={() => toggleEventFilter(event.value)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">{event.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Enable Toggle */}
              <div className="md:col-span-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={newConfig.enabled}
                    onChange={(e) => updateNewConfig({ enabled: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Enable this integration</span>
                </label>
              </div>
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleAddIntegration}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
              >
                Add Integration
              </button>
            </div>
          </div>
        )}

        {/* Integrations List */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Active Integrations</h2>
          </div>
          
          {Object.keys(integrations).length === 0 ? (
            <div className="text-center py-12">
              <LinkIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Integrations</h3>
              <p className="text-gray-600 mb-6">
                Connect external tools and services to receive notifications and automate workflows.
              </p>
              <button
                onClick={() => setShowAddForm(true)}
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                <PlusIcon className="h-4 w-4 mr-2" />
                Add Your First Integration
              </button>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {Object.entries(integrations).map(([name, integration]) => (
                <div key={name} className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      {getStatusIcon(integration)}
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">{name}</h3>
                        <p className="text-sm text-gray-600">
                          {getIntegrationType(integration.type).label}
                          {integration.channel && ` • ${integration.channel}`}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        integration.enabled
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {integration.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                      
                      <button
                        onClick={() => handleTestIntegration(name)}
                        disabled={testingIntegration === name}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                      >
                        {testingIntegration === name ? (
                          <div className="flex items-center">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                            Testing...
                          </div>
                        ) : (
                          <div className="flex items-center">
                            <PlayIcon className="h-4 w-4 mr-1" />
                            Test
                          </div>
                        )}
                      </button>
                      
                      <button
                        onClick={() => handleRemoveIntegration(name)}
                        className="text-red-600 hover:text-red-800"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Priority Threshold:</span>
                      <p className="font-medium capitalize">{integration.priority_threshold}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Event Filters:</span>
                      <p className="font-medium">
                        {integration.event_filters ? integration.event_filters.length : 'All Events'}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500">Webhook:</span>
                      <p className="font-medium">{integration.has_webhook_url ? 'Configured' : 'Not Set'}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">API Token:</span>
                      <p className="font-medium">{integration.has_api_token ? 'Configured' : 'Not Set'}</p>
                    </div>
                  </div>
                  
                  {integration.event_filters && integration.event_filters.length > 0 && (
                    <div className="mt-3">
                      <span className="text-gray-500 text-sm">Filtered Events:</span>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {integration.event_filters.map((event) => (
                          <span key={event} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                            {event.replace('_', ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default IntegrationManagement;