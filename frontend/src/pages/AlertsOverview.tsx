import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CompetitorIntelligenceAPI } from '../lib/competitor-intelligence-api';

interface GroupedAlerts {
  competitorId: string | null;
  competitorName: string | null;
  alerts: Array<{
    id: string;
    title: string;
    message: string;
    priority: string;
    alert_type: string;
    created_at: string;
  }>;
}

export function AlertsOverview() {
  const [groups, setGroups] = useState<GroupedAlerts[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const [grouped, trendsData] = await Promise.all([
          CompetitorIntelligenceAPI.getAlertsGrouped({ daysBack: 7, limitPerCompetitor: 10 }),
          CompetitorIntelligenceAPI.getTrends({ timeRangeDays: 14 })
        ]);
        setGroups(grouped);
        setTrends(trendsData.slice(0, 10));
      } catch (e: any) {
        setError(e.message || 'Failed to load alerts overview');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading alerts...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 text-lg">{error}</p>
          <Link to="/competitor-intelligence" className="mt-4 inline-flex items-center text-blue-600 hover:text-blue-500">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-bold text-gray-900">Alerts & Trends</h1>
          <p className="text-gray-600 mt-1">Resumen de alertas recientes por competidor y tendencias destacadas.</p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          {groups.length === 0 ? (
            <div className="bg-white rounded-lg shadow p-8 text-center text-gray-600">No alerts in the last days.</div>
          ) : (
            groups.map(group => (
              <div key={group.competitorId || 'none'} className="bg-white rounded-lg shadow">
                <div className="p-6 border-b">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-gray-900">
                      {group.competitorName || 'Sin competidor asociado'}
                    </h2>
                    {group.competitorId && (
                      <Link to={`/competitor-intelligence/competitors/${group.competitorId}`} className="text-blue-600 hover:text-blue-500 text-sm">
                        Ver competidor
                      </Link>
                    )}
                  </div>
                </div>
                <ul className="divide-y">
                  {group.alerts.map(a => (
                    <li key={a.id} className="p-4 hover:bg-gray-50 flex items-start justify-between">
                      <div className="pr-4">
                        <Link to={`/competitor-intelligence/alerts/${a.id}`} className="text-sm font-medium text-gray-900 hover:text-blue-600">
                          {a.title}
                        </Link>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">{a.message}</p>
                        <div className="text-xs text-gray-500 mt-1">{new Date(a.created_at).toLocaleString()}</div>
                      </div>
                      <div>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          a.priority === 'critical' ? 'bg-red-100 text-red-800' :
                          a.priority === 'high' ? 'bg-orange-100 text-orange-800' :
                          a.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {a.priority}
                        </span>
                      </div>
                    </li>)
                  )}
                </ul>
              </div>
            ))
          )}
        </div>
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b">
              <h3 className="text-lg font-semibold text-gray-900">Tendencias recientes</h3>
              <p className="text-sm text-gray-600">Últimos 14 días</p>
            </div>
            <ul className="divide-y">
              {trends.map((t: any) => (
                <li key={t.id} className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-gray-900">{t.title || t.topic}</div>
                      <div className="text-xs text-gray-500 mt-1">Fuerza: {String(t.strength || t.strength?.value || '')}</div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AlertsOverview;


