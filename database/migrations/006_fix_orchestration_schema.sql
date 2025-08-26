-- Fix orchestration schema issues
-- Add missing columns and tables for campaign orchestration

-- 1. Add quality_score column to agent_feedback table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'agent_feedback' AND column_name = 'quality_score') THEN
        ALTER TABLE agent_feedback ADD COLUMN quality_score DECIMAL(3,2) DEFAULT 0.0;
    END IF;
END $$;

-- 2. Create scheduled_post table if it doesn't exist
CREATE TABLE IF NOT EXISTS scheduled_post (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    platform VARCHAR(50) NOT NULL,
    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'published', 'failed', 'cancelled')),
    post_id VARCHAR(255), -- External platform post ID
    engagement_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create campaign_performance table if it doesn't exist
CREATE TABLE IF NOT EXISTS campaign_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,2) NOT NULL,
    measurement_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    platform VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_scheduled_post_campaign_id ON scheduled_post(campaign_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_post_scheduled_time ON scheduled_post(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_campaign_performance_campaign_id ON campaign_performance(campaign_id);
CREATE INDEX IF NOT EXISTS idx_campaign_performance_date ON campaign_performance(measurement_date);

-- 5. Update existing campaigns to have proper orchestration data
UPDATE campaigns 
SET orchestration_data = COALESCE(orchestration_data, '{}')::jsonb
WHERE orchestration_data IS NULL;

-- 6. Ensure campaign_tasks has task_details column
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'campaign_tasks' AND column_name = 'task_details') THEN
        ALTER TABLE campaign_tasks ADD COLUMN task_details JSONB DEFAULT '{}';
    END IF;
END $$;

-- 7. Add updated_at trigger for scheduled_post
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_scheduled_post_updated_at') THEN
        CREATE TRIGGER update_scheduled_post_updated_at 
        BEFORE UPDATE ON scheduled_post 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;