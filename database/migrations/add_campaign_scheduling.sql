-- Add Campaign Scheduling Tables Migration
-- This migration adds tables for scheduling and distributing campaign content

-- 1. Create Scheduled Posts table
CREATE TABLE IF NOT EXISTS scheduled_post (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaign(id) ON DELETE CASCADE,
    task_id UUID REFERENCES campaign_task(id) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL, -- 'linkedin', 'twitter', 'instagram', 'email'
    content TEXT NOT NULL,
    image_url TEXT,
    scheduled_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(50) DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'published', 'failed', 'cancelled')),
    published_at TIMESTAMPTZ,
    post_url TEXT, -- URL of the published post
    metadata JSONB, -- Platform-specific data (post ID, engagement, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create Campaign Timeline table
CREATE TABLE IF NOT EXISTS campaign_timeline (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaign(id) ON DELETE CASCADE,
    phase VARCHAR(100) NOT NULL, -- 'awareness', 'engagement', 'conversion'
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    description TEXT,
    goals JSONB, -- Specific goals for this phase
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create Distribution Channels table
CREATE TABLE IF NOT EXISTS distribution_channel (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaign(id) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL, -- 'linkedin', 'twitter', 'instagram', 'email', 'website'
    channel_name VARCHAR(255), -- Custom channel name
    is_active BOOLEAN DEFAULT true,
    settings JSONB, -- Platform-specific settings (API keys, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Create Content Calendar table
CREATE TABLE IF NOT EXISTS content_calendar (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaign(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    content_type VARCHAR(100) NOT NULL, -- 'blog', 'social', 'email', 'video'
    title VARCHAR(255),
    description TEXT,
    status VARCHAR(50) DEFAULT 'planned' CHECK (status IN ('planned', 'in_progress', 'completed', 'published')),
    assigned_to VARCHAR(255), -- Team member assigned
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Create Engagement Tracking table
CREATE TABLE IF NOT EXISTS engagement_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scheduled_post_id UUID REFERENCES scheduled_post(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL, -- 'views', 'likes', 'shares', 'comments', 'clicks'
    metric_value INTEGER NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    source VARCHAR(100), -- Where the metric came from
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_scheduled_post_campaign_id ON scheduled_post(campaign_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_post_platform ON scheduled_post(platform);
CREATE INDEX IF NOT EXISTS idx_scheduled_post_status ON scheduled_post(status);
CREATE INDEX IF NOT EXISTS idx_scheduled_post_scheduled_at ON scheduled_post(scheduled_at);

CREATE INDEX IF NOT EXISTS idx_campaign_timeline_campaign_id ON campaign_timeline(campaign_id);
CREATE INDEX IF NOT EXISTS idx_campaign_timeline_dates ON campaign_timeline(start_date, end_date);

CREATE INDEX IF NOT EXISTS idx_distribution_channel_campaign_id ON distribution_channel(campaign_id);
CREATE INDEX IF NOT EXISTS idx_distribution_channel_platform ON distribution_channel(platform);

CREATE INDEX IF NOT EXISTS idx_content_calendar_campaign_id ON content_calendar(campaign_id);
CREATE INDEX IF NOT EXISTS idx_content_calendar_date ON content_calendar(date);
CREATE INDEX IF NOT EXISTS idx_content_calendar_status ON content_calendar(status);

CREATE INDEX IF NOT EXISTS idx_engagement_tracking_scheduled_post_id ON engagement_tracking(scheduled_post_id);
CREATE INDEX IF NOT EXISTS idx_engagement_tracking_recorded_at ON engagement_tracking(recorded_at);

-- 7. Add triggers for updated_at
CREATE TRIGGER update_scheduled_post_updated_at BEFORE UPDATE ON scheduled_post
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_distribution_channel_updated_at BEFORE UPDATE ON distribution_channel
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_calendar_updated_at BEFORE UPDATE ON content_calendar
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 8. Insert sample scheduling data
INSERT INTO distribution_channel (campaign_id, platform, channel_name, settings) VALUES
(
    (SELECT id FROM campaign LIMIT 1),
    'linkedin',
    'Company LinkedIn',
    '{"posting_frequency": "3x per week", "best_times": ["9:00", "12:00", "17:00"]}'
),
(
    (SELECT id FROM campaign LIMIT 1),
    'twitter',
    'Company Twitter',
    '{"posting_frequency": "5x per week", "best_times": ["8:00", "12:00", "16:00", "20:00"]}'
) ON CONFLICT DO NOTHING;

-- 9. Insert sample timeline
INSERT INTO campaign_timeline (campaign_id, phase, start_date, end_date, description, goals) VALUES
(
    (SELECT id FROM campaign LIMIT 1),
    'awareness',
    CURRENT_DATE,
    CURRENT_DATE + INTERVAL '7 days',
    'Build awareness about the new guide',
    '{"target_impressions": 10000, "target_engagement": 500}'
),
(
    (SELECT id FROM campaign LIMIT 1),
    'engagement',
    CURRENT_DATE + INTERVAL '8 days',
    CURRENT_DATE + INTERVAL '14 days',
    'Drive engagement and discussion',
    '{"target_comments": 100, "target_shares": 50}'
) ON CONFLICT DO NOTHING; 