-- Database Schema Optimization Migration
-- This migration implements the database-architect recommendations
-- Apply these changes carefully in the correct order

BEGIN;

-- 1. Add vector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Update existing tables with new columns (safe additions)

-- BlogPost enhancements
ALTER TABLE blog_posts 
ADD COLUMN IF NOT EXISTS seo_score FLOAT,
ADD COLUMN IF NOT EXISTS word_count INTEGER,
ADD COLUMN IF NOT EXISTS reading_time INTEGER,
ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;

-- Campaign enhancements
ALTER TABLE campaigns 
ADD COLUMN IF NOT EXISTS name VARCHAR(255),
ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'draft';

-- Document enhancements
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS file_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS file_size BIGINT,
ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100),
ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS chunk_count INTEGER;

-- CampaignTask enhancements
ALTER TABLE campaign_tasks 
ADD COLUMN IF NOT EXISTS execution_time INTEGER,
ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 5,
ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

-- AgentPerformance enhancements
ALTER TABLE agent_performance 
ADD COLUMN IF NOT EXISTS blog_post_id UUID,
ADD COLUMN IF NOT EXISTS campaign_id UUID,
ADD COLUMN IF NOT EXISTS total_tokens INTEGER,
ADD COLUMN IF NOT EXISTS error_code VARCHAR(50),
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS max_retries INTEGER DEFAULT 3;

-- AgentDecision enhancements
ALTER TABLE agent_decisions 
ADD COLUMN IF NOT EXISTS tokens_used INTEGER,
ADD COLUMN IF NOT EXISTS decision_latency FLOAT;

-- CIContentItem enhancements  
ALTER TABLE ci_content_items 
ADD COLUMN IF NOT EXISTS word_count INTEGER,
ADD COLUMN IF NOT EXISTS read_time INTEGER,
ADD COLUMN IF NOT EXISTS share_count INTEGER,
ADD COLUMN IF NOT EXISTS like_count INTEGER,
ADD COLUMN IF NOT EXISTS comment_count INTEGER,
ADD COLUMN IF NOT EXISTS view_count INTEGER;

-- 3. Add check constraints for data validation

-- BlogPost constraints
ALTER TABLE blog_posts ADD CONSTRAINT IF NOT EXISTS valid_geo_score 
CHECK (geo_score IS NULL OR (geo_score >= 0 AND geo_score <= 100));

ALTER TABLE blog_posts ADD CONSTRAINT IF NOT EXISTS valid_seo_score 
CHECK (seo_score IS NULL OR (seo_score >= 0 AND seo_score <= 100));

ALTER TABLE blog_posts ADD CONSTRAINT IF NOT EXISTS valid_word_count 
CHECK (word_count IS NULL OR word_count >= 0);

ALTER TABLE blog_posts ADD CONSTRAINT IF NOT EXISTS valid_reading_time 
CHECK (reading_time IS NULL OR reading_time >= 0);

-- CampaignTask constraints
ALTER TABLE campaign_tasks ADD CONSTRAINT IF NOT EXISTS valid_priority 
CHECK (priority >= 1 AND priority <= 10);

ALTER TABLE campaign_tasks ADD CONSTRAINT IF NOT EXISTS valid_execution_time 
CHECK (execution_time IS NULL OR execution_time >= 0);

ALTER TABLE campaign_tasks ADD CONSTRAINT IF NOT EXISTS valid_task_dates 
CHECK (started_at IS NULL OR completed_at IS NULL OR started_at <= completed_at);

-- AgentPerformance constraints
ALTER TABLE agent_performance ADD CONSTRAINT IF NOT EXISTS valid_duration 
CHECK (duration IS NULL OR duration >= 0);

ALTER TABLE agent_performance ADD CONSTRAINT IF NOT EXISTS valid_input_tokens 
CHECK (input_tokens IS NULL OR input_tokens >= 0);

ALTER TABLE agent_performance ADD CONSTRAINT IF NOT EXISTS valid_output_tokens 
CHECK (output_tokens IS NULL OR output_tokens >= 0);

ALTER TABLE agent_performance ADD CONSTRAINT IF NOT EXISTS valid_cost 
CHECK (cost IS NULL OR cost >= 0);

ALTER TABLE agent_performance ADD CONSTRAINT IF NOT EXISTS valid_retry_count 
CHECK (retry_count >= 0 AND retry_count <= max_retries);

ALTER TABLE agent_performance ADD CONSTRAINT IF NOT EXISTS valid_agent_dates 
CHECK (end_time IS NULL OR start_time <= end_time);

-- AgentDecision constraints
ALTER TABLE agent_decisions ADD CONSTRAINT IF NOT EXISTS valid_confidence 
CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1));

ALTER TABLE agent_decisions ADD CONSTRAINT IF NOT EXISTS valid_decision_execution_time 
CHECK (execution_time IS NULL OR execution_time >= 0);

ALTER TABLE agent_decisions ADD CONSTRAINT IF NOT EXISTS valid_tokens 
CHECK (tokens_used IS NULL OR tokens_used >= 0);

ALTER TABLE agent_decisions ADD CONSTRAINT IF NOT EXISTS valid_latency 
CHECK (decision_latency IS NULL OR decision_latency >= 0);

-- CIContentItem constraints
ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_sentiment 
CHECK (sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1));

ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_quality 
CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100));

ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_ci_word_count 
CHECK (word_count IS NULL OR word_count >= 0);

ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_engagement 
CHECK (share_count IS NULL OR share_count >= 0);

ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_likes 
CHECK (like_count IS NULL OR like_count >= 0);

ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_comments 
CHECK (comment_count IS NULL OR comment_count >= 0);

ALTER TABLE ci_content_items ADD CONSTRAINT IF NOT EXISTS valid_views 
CHECK (view_count IS NULL OR view_count >= 0);

-- Cache constraints
ALTER TABLE cache_entries ADD CONSTRAINT IF NOT EXISTS valid_hit_count 
CHECK (hit_count >= 0);

ALTER TABLE cache_entries ADD CONSTRAINT IF NOT EXISTS valid_cache_expiry 
CHECK (expires_at IS NULL OR expires_at > created_at);

-- API Key constraints
ALTER TABLE api_keys ADD CONSTRAINT IF NOT EXISTS valid_rate_limit 
CHECK (rate_limit_per_hour > 0);

ALTER TABLE api_keys ADD CONSTRAINT IF NOT EXISTS valid_key_expiry 
CHECK (expires_at IS NULL OR expires_at > created_at);

-- 4. Create optimized indexes

-- BlogPost performance indexes
CREATE INDEX IF NOT EXISTS idx_blog_posts_geo_optimized ON blog_posts(geo_optimized, geo_score);
CREATE INDEX IF NOT EXISTS idx_blog_posts_status_published ON blog_posts(status, published_at);
CREATE INDEX IF NOT EXISTS idx_blog_posts_seo_score ON blog_posts(seo_score) WHERE seo_score IS NOT NULL;

-- Campaign performance indexes
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaigns_status_created ON campaigns(status, created_at);

-- CampaignTask performance indexes
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_priority ON campaign_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_status_priority ON campaign_tasks(status, priority);

-- Document performance indexes
CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents(file_name);
CREATE INDEX IF NOT EXISTS idx_documents_processed_at ON documents(processed_at);

-- Vector search optimization for document_chunks
-- Note: This will be handled by Prisma schema changes

-- AgentPerformance enhanced indexes
CREATE INDEX IF NOT EXISTS idx_agent_performance_blog_post ON agent_performance(blog_post_id) WHERE blog_post_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_performance_campaign ON agent_performance(campaign_id) WHERE campaign_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_performance_cost ON agent_performance(cost) WHERE cost IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_performance_type_status ON agent_performance(agent_type, status);

-- AgentDecision enhanced indexes
CREATE INDEX IF NOT EXISTS idx_agent_decisions_execution_time ON agent_decisions(execution_time) WHERE execution_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_decisions_perf_timestamp ON agent_decisions(performance_id, timestamp);

-- CI enhanced indexes
CREATE INDEX IF NOT EXISTS idx_ci_content_sentiment ON ci_content_items(sentiment_score) WHERE sentiment_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ci_content_quality ON ci_content_items(quality_score) WHERE quality_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ci_content_competitor_type ON ci_content_items(competitor_id, content_type);
CREATE INDEX IF NOT EXISTS idx_ci_content_platform_published ON ci_content_items(platform, published_at);
CREATE INDEX IF NOT EXISTS idx_ci_content_type_published ON ci_content_items(content_type, published_at);

-- CI Competitor enhanced indexes
CREATE INDEX IF NOT EXISTS idx_ci_competitors_industry_tier ON ci_competitors(industry, tier);
CREATE INDEX IF NOT EXISTS idx_ci_competitors_active_monitored ON ci_competitors(is_active, last_monitored);

-- 5. Create materialized views for analytics (commented out for now - can be enabled later)
/*
CREATE MATERIALIZED VIEW IF NOT EXISTS blog_performance_summary AS
SELECT 
    bp.id,
    bp.title,
    bp.status,
    bp.geo_score,
    bp.seo_score,
    bp.word_count,
    COUNT(ap.id) as agent_interactions,
    AVG(ap.execution_time_seconds) as avg_execution_time,
    bp.created_at,
    bp.updated_at
FROM blog_posts bp
LEFT JOIN agent_performance ap ON bp.id = ap.blog_post_id
GROUP BY bp.id, bp.title, bp.status, bp.geo_score, bp.seo_score, bp.word_count, bp.created_at, bp.updated_at;

CREATE INDEX IF NOT EXISTS idx_blog_perf_summary_status ON blog_performance_summary(status);
CREATE INDEX IF NOT EXISTS idx_blog_perf_summary_geo_score ON blog_performance_summary(geo_score);
*/

COMMIT;

-- Post-migration verification queries (run separately)
-- SELECT 'Migration completed successfully' as status;
-- SELECT table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name;
-- SELECT schemaname, tablename, indexname FROM pg_indexes WHERE schemaname = 'public' ORDER BY tablename;