import { useEffect, useState } from 'react';
import api from '../lib/api';

type LinkItem = { label: string; url: string };

type CompanyProfile = {
  companyName?: string;
  companyContext: string;
  brandVoice?: string;
  valueProposition?: string;
  industries: string[];
  targetAudiences: string[];
  tonePresets: string[];
  keywords: string[];
  styleGuidelines?: string;
  prohibitedTopics: string[];
  complianceNotes?: string;
  links: LinkItem[];
  defaultCTA?: string;
  updatedAt?: string;
};

async function fetchCompanyProfile(): Promise<CompanyProfile> {
  const res = await api.get('/api/settings/company-profile');
  return res.data as CompanyProfile;
}

async function saveCompanyProfile(profile: CompanyProfile): Promise<void> {
  await api.put('/api/settings/company-profile', profile);
}

export default function Settings() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<CompanyProfile>({
    companyContext: '',
    industries: [],
    targetAudiences: [],
    tonePresets: [],
    keywords: [],
    prohibitedTopics: [],
    links: [],
  });

  useEffect(() => {
    (async () => {
      try {
        const data = await fetchCompanyProfile();
        setProfile({
          companyContext: '',
          industries: [],
          targetAudiences: [],
          tonePresets: [],
          keywords: [],
          prohibitedTopics: [],
          links: [],
          ...data,
        });
      } catch (e: any) {
        setError(e?.message || 'Failed to load settings');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const updateArray = (key: keyof CompanyProfile) => (value: string) => {
    const items = value
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    setProfile((p) => ({ ...p, [key]: items } as CompanyProfile));
  };

  const arrayToString = (arr?: string[]) => (arr && arr.length ? arr.join(', ') : '');

  const onSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await saveCompanyProfile(profile);
    } catch (e: any) {
      setError(e?.message || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading settings...</p>
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
              <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
              <p className="mt-2 text-gray-600">Configure your company profile and content preferences</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">{error}</div>
              </div>
            </div>
          </div>
        )}

      <div className="grid grid-cols-1 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Company Profile</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Company Name</label>
              <input
                className="w-full border rounded-md px-3 py-2"
                value={profile.companyName || ''}
                onChange={(e) => setProfile({ ...profile, companyName: e.target.value })}
                placeholder="Acme Inc."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Company Context</label>
              <textarea
                className="w-full border rounded-md px-3 py-2"
                rows={6}
                value={profile.companyContext}
                onChange={(e) => setProfile({ ...profile, companyContext: e.target.value })}
                placeholder="Describe your company, audience, tone, differentiators…"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Brand Voice</label>
                <textarea
                  className="w-full border rounded-md px-3 py-2"
                  rows={3}
                  value={profile.brandVoice || ''}
                  onChange={(e) => setProfile({ ...profile, brandVoice: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Value Proposition</label>
                <textarea
                  className="w-full border rounded-md px-3 py-2"
                  rows={3}
                  value={profile.valueProposition || ''}
                  onChange={(e) => setProfile({ ...profile, valueProposition: e.target.value })}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Industries (comma-separated)</label>
                <input
                  className="w-full border rounded-md px-3 py-2"
                  value={arrayToString(profile.industries)}
                  onChange={(e) => updateArray('industries')(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target Audiences (comma-separated)</label>
                <input
                  className="w-full border rounded-md px-3 py-2"
                  value={arrayToString(profile.targetAudiences)}
                  onChange={(e) => updateArray('targetAudiences')(e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Tone Presets (comma-separated)</label>
                <input
                  className="w-full border rounded-md px-3 py-2"
                  value={arrayToString(profile.tonePresets)}
                  onChange={(e) => updateArray('tonePresets')(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Brand Keywords (comma-separated)</label>
                <input
                  className="w-full border rounded-md px-3 py-2"
                  value={arrayToString(profile.keywords)}
                  onChange={(e) => updateArray('keywords')(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Style Guidelines</label>
              <textarea
                className="w-full border rounded-md px-3 py-2"
                rows={4}
                value={profile.styleGuidelines || ''}
                onChange={(e) => setProfile({ ...profile, styleGuidelines: e.target.value })}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Prohibited Topics (comma-separated)</label>
              <input
                className="w-full border rounded-md px-3 py-2"
                value={arrayToString(profile.prohibitedTopics)}
                onChange={(e) => updateArray('prohibitedTopics')(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Compliance Notes</label>
              <textarea
                className="w-full border rounded-md px-3 py-2"
                rows={3}
                value={profile.complianceNotes || ''}
                onChange={(e) => setProfile({ ...profile, complianceNotes: e.target.value })}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Useful Links</label>
              {(profile.links || []).map((link, idx) => (
                <div key={idx} className="flex items-center gap-2 mb-2">
                  <input
                    className="flex-1 border rounded-md px-3 py-2"
                    placeholder="Label"
                    value={link.label}
                    onChange={(e) => {
                      const next = [...profile.links];
                      next[idx] = { ...next[idx], label: e.target.value };
                      setProfile({ ...profile, links: next });
                    }}
                  />
                  <input
                    className="flex-1 border rounded-md px-3 py-2"
                    placeholder="https://example.com"
                    value={link.url}
                    onChange={(e) => {
                      const next = [...profile.links];
                      next[idx] = { ...next[idx], url: e.target.value };
                      setProfile({ ...profile, links: next });
                    }}
                  />
                  <button
                    className="text-sm text-red-600"
                    onClick={() => setProfile({ ...profile, links: profile.links.filter((_, i) => i !== idx) })}
                  >
                    Remove
                  </button>
                </div>
              ))}
              <button
                className="text-sm text-blue-600"
                onClick={() => setProfile({ ...profile, links: [...profile.links, { label: '', url: '' }] })}
              >
                + Add link
              </button>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Default Call To Action</label>
              <input
                className="w-full border rounded-md px-3 py-2"
                value={profile.defaultCTA || ''}
                onChange={(e) => setProfile({ ...profile, defaultCTA: e.target.value })}
                placeholder="Book a demo, Start free trial, Contact sales…"
              />
            </div>
          </div>

          <div className="mt-6 flex items-center gap-3">
            <button
              onClick={onSave}
              disabled={saving}
              className="bg-blue-600 text-white px-4 py-2 rounded-md disabled:opacity-60"
            >
              {saving ? 'Saving…' : 'Save Settings'}
            </button>
            {profile.updatedAt && (
              <span className="text-sm text-gray-500">Last updated {new Date(profile.updatedAt).toLocaleString()}</span>
            )}
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

