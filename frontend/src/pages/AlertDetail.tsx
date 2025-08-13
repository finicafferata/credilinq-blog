import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';
import { showErrorNotification } from '../lib/errors';

interface AlertDetailData {
  id: string;
  title: string;
  message: string;
  alert_type: string;
  priority: string;
  competitor_id?: string | null;
  competitor_name?: string | null;
  content_id?: string | null;
  metadata?: any;
  created_at: string;
  is_read: boolean;
  is_dismissed: boolean;
}

export function AlertDetail() {
  const { alertId } = useParams<{ alertId: string }>();
  const [alert, setAlert] = useState<AlertDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    if (alertId) {
      loadAlert();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alertId]);

  const loadAlert = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await CompetitorIntelligenceAPI.getAlert(alertId!);
      setAlert(data as any);
    } catch (err: any) {
      setError(err.message || 'Failed to load alert');
    } finally {
      setLoading(false);
    }
  };

  const markRead = async () => {
    if (!alert) return;
    try {
      setUpdating(true);
      await CompetitorIntelligenceAPI.markAlertRead(alert.id);
      await loadAlert();
    } catch (err: any) {
      showErrorNotification(new Error('Failed to mark alert as read: ' + err.message));
    } finally {
      setUpdating(false);
    }
  };

  const dismiss = async () => {
    if (!alert) return;
    try {
      setUpdating(true);
      await CompetitorIntelligenceAPI.dismissAlert(alert.id);
      await loadAlert();
    } catch (err: any) {
      showErrorNotification(new Error('Failed to dismiss alert: ' + err.message));
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading alert...</p>
        </div>
      </div>
    );
  }

  if (error || !alert) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 text-lg">{error || 'Alert not found'}</p>
          <Link to="/competitor-intelligence" className="mt-4 inline-flex items-center text-blue-600 hover:text-blue-500">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  const priorityClass = alert.priority === 'critical'
    ? 'bg-red-100 text-red-800'
    : alert.priority === 'high'
    ? 'bg-orange-100 text-orange-800'
    : alert.priority === 'medium'
    ? 'bg-yellow-100 text-yellow-800'
    : 'bg-gray-100 text-gray-800';

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <nav className="flex" aria-label="Breadcrumb">
                <ol className="flex items-center space-x-4">
                  <li>
                    <Link to="/competitor-intelligence" className="text-gray-400 hover:text-gray-500">Competitor Intelligence</Link>
                  </li>
                  <li><span className="text-gray-400">/</span></li>
                  <li><span className="text-gray-900 font-medium">Alert</span></li>
                </ol>
              </nav>
              <h1 className="mt-2 text-2xl font-bold text-gray-900">{alert.title}</h1>
              <div className="mt-2 flex items-center space-x-2">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${priorityClass}`}>{alert.priority}</span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">{alert.alert_type}</span>
                <span className="text-sm text-gray-500">{new Date(alert.created_at).toLocaleString()}</span>
              </div>
            </div>
            <div className="space-x-2">
              <button onClick={markRead} disabled={updating || alert.is_read} className="px-3 py-2 rounded-md border text-sm disabled:opacity-50">
                Mark as Read
              </button>
              <button onClick={dismiss} disabled={updating || alert.is_dismissed} className="px-3 py-2 rounded-md border text-sm disabled:opacity-50">
                Dismiss
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-2">Message</h2>
              <p className="text-gray-700 whitespace-pre-wrap">{alert.message}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-2">Metadata</h2>
              <pre className="text-sm bg-gray-50 p-3 rounded overflow-auto">{JSON.stringify(alert.metadata || {}, null, 2)}</pre>
            </div>
          </div>
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-2">Context</h3>
              <div className="space-y-2 text-sm text-gray-700">
                <div><span className="font-medium">Competitor:</span> {alert.competitor_name || 'N/A'}</div>
                {alert.competitor_id && (
                  <Link to={`/competitor-intelligence/competitors/${alert.competitor_id}`} className="text-blue-600 hover:text-blue-500">View competitor</Link>
                )}
                {alert.content_id && (
                  <div className="mt-2">
                    <span className="font-medium">Content ID:</span> {String(alert.content_id)}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AlertDetail;

