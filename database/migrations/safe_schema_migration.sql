-- Safe Schema Migration for Database Optimization
-- This migration renames columns and adds new ones while preserving data

BEGIN;

-- 1. Add new columns with default values first
ALTER TABLE blog_posts 
ADD COLUMN IF NOT EXISTS content_markdown TEXT,
ADD COLUMN IF NOT EXISTS initial_prompt JSONB,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS seo_score FLOAT,
ADD COLUMN IF NOT EXISTS word_count INTEGER,
ADD COLUMN IF NOT EXISTS reading_time INTEGER,
ADD COLUMN IF NOT EXISTS geo_metadata JSONB,
ADD COLUMN IF NOT EXISTS geo_optimized BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS geo_score SMALLINT;

-- 2. Copy data from old columns to new columns
UPDATE blog_posts SET 
    content_markdown = "contentMarkdown" WHERE content_markdown IS NULL,
    initial_prompt = "initialPrompt" WHERE initial_prompt IS NULL,
    created_at = "createdAt" WHERE created_at = NOW(),
    updated_at = "updatedAt" WHERE updated_at = NOW(),
    geo_metadata = "geoMetadata" WHERE geo_metadata IS NULL,
    geo_optimized = "geoOptimized" WHERE geo_optimized = FALSE,
    geo_score = "geoScore" WHERE geo_score IS NULL;

-- 3. Update null content_markdown fields (safety check)
UPDATE blog_posts SET content_markdown = COALESCE("contentMarkdown", '') WHERE content_markdown IS NULL;
UPDATE blog_posts SET initial_prompt = COALESCE("initialPrompt", '{}') WHERE initial_prompt IS NULL;

-- 4. Add NOT NULL constraints after data migration
ALTER TABLE blog_posts 
ALTER COLUMN content_markdown SET NOT NULL,
ALTER COLUMN created_at SET NOT NULL,
ALTER COLUMN updated_at SET NOT NULL;

-- 5. Drop old camelCase columns
ALTER TABLE blog_posts 
DROP COLUMN IF EXISTS "contentMarkdown",
DROP COLUMN IF EXISTS "initialPrompt", 
DROP COLUMN IF EXISTS "createdAt",
DROP COLUMN IF EXISTS "updatedAt",
DROP COLUMN IF EXISTS "geoMetadata",
DROP COLUMN IF EXISTS "geoOptimized",
DROP COLUMN IF EXISTS "geoScore";

-- 6. Update campaigns table
ALTER TABLE campaigns 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS blog_post_id UUID,
ADD COLUMN IF NOT EXISTS name VARCHAR(255),
ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'draft';

-- Copy campaign data
UPDATE campaigns SET 
    created_at = "createdAt" WHERE created_at = NOW(),
    blog_post_id = "blogPostId" WHERE blog_post_id IS NULL;

-- Drop old campaign columns
ALTER TABLE campaigns 
DROP COLUMN IF EXISTS "createdAt",
DROP COLUMN IF EXISTS "blogPostId";

-- 7. Update briefings table
ALTER TABLE briefings 
ADD COLUMN IF NOT EXISTS campaign_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS marketing_objective TEXT,
ADD COLUMN IF NOT EXISTS target_audience VARCHAR(500),
ADD COLUMN IF NOT EXISTS desired_tone VARCHAR(100),
ADD COLUMN IF NOT EXISTS company_context TEXT,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS campaign_id UUID;

-- Copy briefing data
UPDATE briefings SET 
    campaign_name = "campaignName" WHERE campaign_name IS NULL,
    marketing_objective = "marketingObjective" WHERE marketing_objective IS NULL,
    target_audience = "targetAudience" WHERE target_audience IS NULL,
    desired_tone = "desiredTone" WHERE desired_tone IS NULL,
    company_context = "companyContext" WHERE company_context IS NULL,
    created_at = "createdAt" WHERE created_at = NOW(),
    updated_at = "updatedAt" WHERE updated_at = NOW(),
    campaign_id = "campaignId" WHERE campaign_id IS NULL;

-- Drop old briefing columns
ALTER TABLE briefings 
DROP COLUMN IF EXISTS "campaignName",
DROP COLUMN IF EXISTS "marketingObjective",
DROP COLUMN IF EXISTS "targetAudience",
DROP COLUMN IF EXISTS "desiredTone",
DROP COLUMN IF EXISTS "companyContext",
DROP COLUMN IF EXISTS "createdAt",
DROP COLUMN IF EXISTS "updatedAt",
DROP COLUMN IF EXISTS "campaignId";

-- 8. Update campaign_tasks table
ALTER TABLE campaign_tasks 
ADD COLUMN IF NOT EXISTS campaign_id UUID,
ADD COLUMN IF NOT EXISTS task_type VARCHAR(100),
ADD COLUMN IF NOT EXISTS target_format VARCHAR(100),
ADD COLUMN IF NOT EXISTS target_asset VARCHAR(200),
ADD COLUMN IF NOT EXISTS image_url VARCHAR(500),
ADD COLUMN IF NOT EXISTS execution_time INTEGER,
ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 5,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

-- Copy campaign task data
UPDATE campaign_tasks SET 
    campaign_id = "campaignId" WHERE campaign_id IS NULL,
    task_type = "taskType" WHERE task_type IS NULL,
    target_format = "targetFormat" WHERE target_format IS NULL,
    target_asset = "targetAsset" WHERE target_asset IS NULL,
    image_url = "imageUrl" WHERE image_url IS NULL,
    created_at = "createdAt" WHERE created_at = NOW(),
    updated_at = "updatedAt" WHERE updated_at = NOW();

-- Drop old campaign task columns
ALTER TABLE campaign_tasks 
DROP COLUMN IF EXISTS "campaignId",
DROP COLUMN IF EXISTS "taskType",
DROP COLUMN IF EXISTS "targetFormat",
DROP COLUMN IF EXISTS "targetAsset",
DROP COLUMN IF EXISTS "imageUrl",
DROP COLUMN IF EXISTS "createdAt",
DROP COLUMN IF EXISTS "updatedAt";

-- 9. Update content_strategies table (if exists)
ALTER TABLE content_strategies 
ADD COLUMN IF NOT EXISTS campaign_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS narrative_approach TEXT,
ADD COLUMN IF NOT EXISTS tone_by_channel JSONB,
ADD COLUMN IF NOT EXISTS key_phrases JSONB,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS campaign_id UUID;

-- Copy content strategy data if columns exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='content_strategies' AND column_name='campaignName') THEN
        UPDATE content_strategies SET 
            campaign_name = "campaignName" WHERE campaign_name IS NULL,
            narrative_approach = "narrativeApproach" WHERE narrative_approach IS NULL,
            tone_by_channel = "toneByChannel" WHERE tone_by_channel IS NULL,
            key_phrases = "keyPhrases" WHERE key_phrases IS NULL,
            created_at = "createdAt" WHERE created_at = NOW(),
            updated_at = "updatedAt" WHERE updated_at = NOW(),
            campaign_id = "campaignId" WHERE campaign_id IS NULL;
    END IF;
END $$;

-- 10. Update agent_performance table
ALTER TABLE agent_performance 
ADD COLUMN IF NOT EXISTS agent_name VARCHAR(100),
ADD COLUMN IF NOT EXISTS agent_type VARCHAR(100),
ADD COLUMN IF NOT EXISTS execution_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS blog_post_id UUID,
ADD COLUMN IF NOT EXISTS campaign_id UUID,
ADD COLUMN IF NOT EXISTS start_time TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS end_time TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS input_tokens INTEGER,
ADD COLUMN IF NOT EXISTS output_tokens INTEGER,
ADD COLUMN IF NOT EXISTS total_tokens INTEGER,
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS error_code VARCHAR(50),
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS max_retries INTEGER DEFAULT 3,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- Copy agent performance data
UPDATE agent_performance SET 
    agent_name = "agentName" WHERE agent_name IS NULL,
    agent_type = "agentType" WHERE agent_type IS NULL,
    execution_id = "executionId" WHERE execution_id IS NULL,
    start_time = "startTime" WHERE start_time IS NULL,
    end_time = "endTime" WHERE end_time IS NULL,
    input_tokens = "inputTokens" WHERE input_tokens IS NULL,
    output_tokens = "outputTokens" WHERE output_tokens IS NULL,
    error_message = "errorMessage" WHERE error_message IS NULL,
    created_at = "createdAt" WHERE created_at = NOW();

-- Add unique constraint for execution_id
ALTER TABLE agent_performance 
ADD CONSTRAINT unique_execution_id UNIQUE (execution_id);

-- Drop old agent performance columns  
ALTER TABLE agent_performance 
DROP COLUMN IF EXISTS "agentName",
DROP COLUMN IF EXISTS "agentType",
DROP COLUMN IF EXISTS "executionId",
DROP COLUMN IF EXISTS "startTime",
DROP COLUMN IF EXISTS "endTime",
DROP COLUMN IF EXISTS "inputTokens",
DROP COLUMN IF EXISTS "outputTokens",
DROP COLUMN IF EXISTS "errorMessage",
DROP COLUMN IF EXISTS "createdAt";

-- 11. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_blog_posts_status_created ON blog_posts(status, created_at);
CREATE INDEX IF NOT EXISTS idx_blog_posts_published_at ON blog_posts(published_at);
CREATE INDEX IF NOT EXISTS idx_blog_posts_word_count ON blog_posts(word_count);
CREATE INDEX IF NOT EXISTS idx_blog_posts_seo_score ON blog_posts(seo_score);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_priority ON campaign_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_agent_performance_blog_post ON agent_performance(blog_post_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_campaign ON agent_performance(campaign_id);

COMMIT;

-- Verification queries
SELECT 'Migration completed successfully' as status;