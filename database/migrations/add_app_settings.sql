-- App settings table to store single-tenant configuration
CREATE TABLE IF NOT EXISTS app_settings (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed empty company profile if not present
INSERT INTO app_settings(key, value)
VALUES ('company_profile', jsonb_build_object())
ON CONFLICT (key) DO NOTHING;

