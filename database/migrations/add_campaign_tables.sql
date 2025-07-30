-- Add Campaign Tables Migration
-- This migration adds the necessary tables for campaign management

-- 1. Create Campaign table
CREATE TABLE IF NOT EXISTS campaign (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blog_id UUID REFERENCES "BlogPost"(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'completed', 'paused')),
    strategy JSONB, -- Campaign strategy and settings
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create Campaign Task table
CREATE TABLE IF NOT EXISTS campaign_task (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaign(id) ON DELETE CASCADE,
    task_type VARCHAR(100) NOT NULL, -- 'linkedin_post', 'twitter_thread', 'email', 'image', 'schedule'
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'approved', 'rejected')),
    content TEXT, -- Generated content
    image_url TEXT, -- URL of generated image
    metadata JSONB, -- Additional data (channels, timing, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create Campaign Analytics table
CREATE TABLE IF NOT EXISTS campaign_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaign(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL, -- 'views', 'engagement', 'clicks', 'shares'
    metric_value FLOAT NOT NULL,
    source VARCHAR(100), -- 'linkedin', 'twitter', 'email', 'website'
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_campaign_blog_id ON campaign(blog_id);
CREATE INDEX IF NOT EXISTS idx_campaign_status ON campaign(status);
CREATE INDEX IF NOT EXISTS idx_campaign_task_campaign_id ON campaign_task(campaign_id);
CREATE INDEX IF NOT EXISTS idx_campaign_task_status ON campaign_task(status);
CREATE INDEX IF NOT EXISTS idx_campaign_task_type ON campaign_task(task_type);
CREATE INDEX IF NOT EXISTS idx_campaign_analytics_campaign_id ON campaign_analytics(campaign_id);
CREATE INDEX IF NOT EXISTS idx_campaign_analytics_date ON campaign_analytics(date);

-- 5. Add trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_campaign_updated_at BEFORE UPDATE ON campaign
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_campaign_task_updated_at BEFORE UPDATE ON campaign_task
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 6. Insert sample data (optional)
INSERT INTO campaign (id, blog_id, name, status, strategy) VALUES
(
    gen_random_uuid(),
    (SELECT id FROM "BlogPost" LIMIT 1),
    'Sample Campaign',
    'draft',
    '{"channels": ["linkedin", "twitter"], "timeline": "2 weeks"}'
) ON CONFLICT DO NOTHING; 