-- CI Webhooks, Digests and Change Events (snake_case)

BEGIN;

-- Change events table
CREATE TABLE IF NOT EXISTS ci_change_events (
    id BIGSERIAL PRIMARY KEY,
    competitor_id UUID NOT NULL,
    source TEXT NOT NULL, -- 'website' | 'blog' | 'pricing' | 'news' | 'social'
    change_type TEXT NOT NULL, -- 'pricing' | 'product' | 'plan_copy' | 'news' | 'social'
    old_value JSONB,
    new_value JSONB,
    url TEXT,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidence REAL,
    sentiment TEXT, -- 'positive' | 'neutral' | 'negative'
    metadata JSONB,
    CONSTRAINT ci_change_events_competitor_fk FOREIGN KEY (competitor_id)
        REFERENCES ci_competitors(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ci_change_events_competitor ON ci_change_events(competitor_id);
CREATE INDEX IF NOT EXISTS idx_ci_change_events_detected_at ON ci_change_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_ci_change_events_type ON ci_change_events(change_type);

-- Webhook subscriptions
CREATE TABLE IF NOT EXISTS ci_webhook_subscriptions (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    target_url TEXT NOT NULL,
    secret_hmac TEXT,
    event_types TEXT[] NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ci_webhook_active ON ci_webhook_subscriptions(is_active);

-- Delivery log
CREATE TABLE IF NOT EXISTS ci_delivery_log (
    id BIGSERIAL PRIMARY KEY,
    subscription_id UUID NOT NULL,
    event_id TEXT,
    event_type TEXT,
    status TEXT NOT NULL, -- 'success' | 'failed'
    http_code INTEGER,
    attempts INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    delivered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ci_delivery_log_subscription_fk FOREIGN KEY (subscription_id)
        REFERENCES ci_webhook_subscriptions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ci_delivery_log_subscription ON ci_delivery_log(subscription_id);
CREATE INDEX IF NOT EXISTS idx_ci_delivery_log_delivered_at ON ci_delivery_log(delivered_at DESC);

-- Digest subscriptions
CREATE TABLE IF NOT EXISTS ci_digest_subscriptions (
    id UUID PRIMARY KEY,
    channel TEXT NOT NULL, -- 'email' | 'teams'
    address_or_webhook TEXT NOT NULL,
    frequency TEXT NOT NULL DEFAULT 'daily', -- 'daily' | 'weekly'
    timezone TEXT NOT NULL DEFAULT 'UTC',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ci_digest_active ON ci_digest_subscriptions(is_active);

COMMIT;


