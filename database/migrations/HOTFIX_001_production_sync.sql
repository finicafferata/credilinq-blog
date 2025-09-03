-- HOTFIX 001: Critical Production Database Synchronization
-- This script addresses the immediate production issues
-- Run this AFTER taking a full database backup

-- =============================================================================
-- 1. FIX AGENT_PERFORMANCE TABLE FOR FEEDBACK ANALYTICS
-- =============================================================================

-- Add missing columns to agent_performance table if they don't exist
DO $$ 
BEGIN
    -- Add quality_score column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'agent_performance' AND column_name = 'quality_score') THEN
        ALTER TABLE agent_performance ADD COLUMN quality_score DECIMAL(4,2) DEFAULT 0.0;
        RAISE NOTICE 'Added quality_score column to agent_performance';
    END IF;
    
    -- Add success boolean column (derived from status)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'agent_performance' AND column_name = 'success') THEN
        ALTER TABLE agent_performance ADD COLUMN success BOOLEAN DEFAULT false;
        RAISE NOTICE 'Added success column to agent_performance';
    END IF;
    
    -- Add feedback_data column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'agent_performance' AND column_name = 'feedback_data') THEN
        ALTER TABLE agent_performance ADD COLUMN feedback_data JSONB;
        RAISE NOTICE 'Added feedback_data column to agent_performance';
    END IF;
END $$;

-- Update success column based on status
UPDATE agent_performance 
SET success = (status = 'success') 
WHERE success IS NULL;

-- Set default quality scores based on success status
UPDATE agent_performance 
SET quality_score = CASE 
    WHEN success = true THEN 0.85 
    ELSE 0.40 
END 
WHERE quality_score IS NULL OR quality_score = 0.0;

-- =============================================================================
-- 2. FIX CAMPAIGNS TABLE - ADD NAME COLUMN AND UPDATE NULLS
-- =============================================================================

-- Add name column to campaigns if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'campaigns' AND column_name = 'name') THEN
        ALTER TABLE campaigns ADD COLUMN name VARCHAR(255);
        RAISE NOTICE 'Added name column to campaigns';
    END IF;
END $$;

-- Update NULL or empty campaign names
UPDATE campaigns 
SET name = 'Campaign ' || SUBSTRING(id::text, 1, 8)
WHERE name IS NULL OR name = '' OR TRIM(name) = '';

-- Make name column NOT NULL after populating data
ALTER TABLE campaigns ALTER COLUMN name SET NOT NULL;

-- =============================================================================
-- 3. ENSURE BLOG_POSTS TABLE HAS REQUIRED COLUMNS
-- =============================================================================

-- Add missing columns to blog_posts if they don't exist
DO $$ 
BEGIN
    -- Add updated_at column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'updated_at') THEN
        ALTER TABLE blog_posts ADD COLUMN updated_at TIMESTAMPTZ(6) DEFAULT NOW();
        RAISE NOTICE 'Added updated_at column to blog_posts';
    END IF;
    
    -- Add published_at column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'published_at') THEN
        ALTER TABLE blog_posts ADD COLUMN published_at TIMESTAMPTZ(6);
        RAISE NOTICE 'Added published_at column to blog_posts';
    END IF;
    
    -- Add campaign_id column for relationship
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'campaign_id') THEN
        ALTER TABLE blog_posts ADD COLUMN campaign_id UUID;
        RAISE NOTICE 'Added campaign_id column to blog_posts';
    END IF;
    
    -- Add geo metadata columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'geo_metadata') THEN
        ALTER TABLE blog_posts ADD COLUMN geo_metadata JSONB;
        RAISE NOTICE 'Added geo_metadata column to blog_posts';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'geo_optimized') THEN
        ALTER TABLE blog_posts ADD COLUMN geo_optimized BOOLEAN DEFAULT false;
        RAISE NOTICE 'Added geo_optimized column to blog_posts';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'geo_score') THEN
        ALTER TABLE blog_posts ADD COLUMN geo_score SMALLINT;
        RAISE NOTICE 'Added geo_score column to blog_posts';
    END IF;
    
    -- Add SEO and content metrics
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'seo_score') THEN
        ALTER TABLE blog_posts ADD COLUMN seo_score FLOAT;
        RAISE NOTICE 'Added seo_score column to blog_posts';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'word_count') THEN
        ALTER TABLE blog_posts ADD COLUMN word_count INTEGER;
        RAISE NOTICE 'Added word_count column to blog_posts';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'blog_posts' AND column_name = 'reading_time') THEN
        ALTER TABLE blog_posts ADD COLUMN reading_time INTEGER;
        RAISE NOTICE 'Added reading_time column to blog_posts';
    END IF;
END $$;

-- =============================================================================
-- 4. FIX CAMPAIGN_TASKS TABLE STRUCTURE
-- =============================================================================

-- Rename columns to match expected schema
DO $$ 
BEGIN
    -- Rename taskType to task_type if needed
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'campaign_task' AND column_name = 'taskType') THEN
        ALTER TABLE campaign_task RENAME COLUMN "taskType" TO task_type;
        RAISE NOTICE 'Renamed taskType to task_type in campaign_task';
    END IF;
    
    -- Rename targetFormat to target_format if needed
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'campaign_task' AND column_name = 'targetFormat') THEN
        ALTER TABLE campaign_task RENAME COLUMN "targetFormat" TO target_format;
        RAISE NOTICE 'Renamed targetFormat to target_format in campaign_task';
    END IF;
    
    -- Rename targetAsset to target_asset if needed
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'campaign_task' AND column_name = 'targetAsset') THEN
        ALTER TABLE campaign_task RENAME COLUMN "targetAsset" TO target_asset;
        RAISE NOTICE 'Renamed targetAsset to target_asset in campaign_task';
    END IF;
    
    -- Rename imageUrl to image_url if needed
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'campaign_task' AND column_name = 'imageUrl') THEN
        ALTER TABLE campaign_task RENAME COLUMN "imageUrl" TO image_url;
        RAISE NOTICE 'Renamed imageUrl to image_url in campaign_task';
    END IF;
    
    -- Add missing columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'campaign_task' AND column_name = 'execution_time') THEN
        ALTER TABLE campaign_task ADD COLUMN execution_time INTEGER;
        RAISE NOTICE 'Added execution_time column to campaign_task';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'campaign_task' AND column_name = 'priority') THEN
        ALTER TABLE campaign_task ADD COLUMN priority INTEGER DEFAULT 5;
        RAISE NOTICE 'Added priority column to campaign_task';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'campaign_task' AND column_name = 'started_at') THEN
        ALTER TABLE campaign_task ADD COLUMN started_at TIMESTAMPTZ(6);
        RAISE NOTICE 'Added started_at column to campaign_task';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'campaign_task' AND column_name = 'completed_at') THEN
        ALTER TABLE campaign_task ADD COLUMN completed_at TIMESTAMPTZ(6);
        RAISE NOTICE 'Added completed_at column to campaign_task';
    END IF;
END $$;

-- Rename table to match expected name
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'campaign_task') 
       AND NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'campaign_tasks') THEN
        ALTER TABLE campaign_task RENAME TO campaign_tasks;
        RAISE NOTICE 'Renamed campaign_task table to campaign_tasks';
    END IF;
END $$;

-- =============================================================================
-- 5. CREATE MISSING ESSENTIAL TABLES
-- =============================================================================

-- Create briefings table if it doesn't exist
CREATE TABLE IF NOT EXISTS briefings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_name VARCHAR(255) NOT NULL,
    marketing_objective TEXT NOT NULL,
    target_audience VARCHAR(500) NOT NULL,
    channels JSONB NOT NULL,
    desired_tone VARCHAR(100) NOT NULL,
    language VARCHAR(50) NOT NULL,
    company_context TEXT,
    created_at TIMESTAMPTZ(6) DEFAULT NOW(),
    updated_at TIMESTAMPTZ(6) DEFAULT NOW(),
    campaign_id UUID UNIQUE REFERENCES campaigns(id) ON DELETE CASCADE
);

-- Create content_strategies table if it doesn't exist
CREATE TABLE IF NOT EXISTS content_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_name VARCHAR(255) NOT NULL,
    narrative_approach TEXT NOT NULL,
    hooks JSONB NOT NULL,
    themes JSONB NOT NULL,
    tone_by_channel JSONB NOT NULL,
    key_phrases JSONB NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ(6) DEFAULT NOW(),
    updated_at TIMESTAMPTZ(6) DEFAULT NOW(),
    campaign_id UUID UNIQUE REFERENCES campaigns(id) ON DELETE CASCADE
);

-- =============================================================================
-- 6. ADD CRITICAL INDEXES
-- =============================================================================

-- Add indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_status ON blog_posts(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_campaign_id ON blog_posts(campaign_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_name ON campaigns(name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_campaign_id ON agent_performance(campaign_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_quality_score ON agent_performance(quality_score);

-- =============================================================================
-- 7. UPDATE TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- =============================================================================

-- Create or replace the update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at columns
DO $$ 
BEGIN
    -- Add trigger for blog_posts if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_blog_posts_updated_at') THEN
        CREATE TRIGGER update_blog_posts_updated_at 
            BEFORE UPDATE ON blog_posts
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        RAISE NOTICE 'Added updated_at trigger for blog_posts';
    END IF;
    
    -- Add trigger for campaigns if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_campaigns_updated_at') THEN
        CREATE TRIGGER update_campaigns_updated_at 
            BEFORE UPDATE ON campaigns
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        RAISE NOTICE 'Added updated_at trigger for campaigns';
    END IF;
END $$;

-- =============================================================================
-- 8. POPULATE DEFAULT DATA WHERE NEEDED
-- =============================================================================

-- Set default word counts for blog posts that don't have them
UPDATE blog_posts 
SET word_count = LENGTH(content_markdown) / 5  -- Rough estimate: 5 chars per word
WHERE word_count IS NULL AND content_markdown IS NOT NULL;

-- Set default reading times (assuming 200 words per minute)
UPDATE blog_posts 
SET reading_time = GREATEST(1, word_count / 200)
WHERE reading_time IS NULL AND word_count IS NOT NULL;

-- Set published_at for published posts
UPDATE blog_posts 
SET published_at = created_at
WHERE status = 'published' AND published_at IS NULL;

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

DO $$ 
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'HOTFIX 001 COMPLETED SUCCESSFULLY';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Fixed:';
    RAISE NOTICE '- agent_performance table (quality_score, success, feedback_data)';
    RAISE NOTICE '- campaigns table (name column, NULL handling)';
    RAISE NOTICE '- blog_posts table (missing columns, relationships)';
    RAISE NOTICE '- campaign_tasks table (column names, missing fields)';
    RAISE NOTICE '- Added essential indexes for performance';
    RAISE NOTICE '- Created briefings and content_strategies tables';
    RAISE NOTICE '- Added updated_at triggers';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'NEXT STEPS:';
    RAISE NOTICE '1. Restart the application to use new schema';
    RAISE NOTICE '2. Test the feedback-analytics endpoint';
    RAISE NOTICE '3. Verify campaign names are populated';
    RAISE NOTICE '4. Deploy the content-deliverables API fix';
    RAISE NOTICE '============================================';
END $$;