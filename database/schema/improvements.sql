-- =====================================================
-- DATABASE IMPROVEMENTS FOR AI AGENTS MARKETING PLATFORM
-- Phase 1: Foundation - Schema Fixes and Performance
-- FIXED VERSION - Proper SQL statement separation
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===== STEP 1: FIX SCHEMA INCONSISTENCIES =====

-- Standardize campaign table field naming (snake_case) - Single DO block
DO $campaign_rename$ 
BEGIN
    -- Check if columns exist before altering
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign' AND column_name = 'blogId') THEN
        ALTER TABLE campaign RENAME COLUMN "blogId" TO blog_id;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign' AND column_name = 'createdAt') THEN
        ALTER TABLE campaign RENAME COLUMN "createdAt" TO created_at;
    END IF;
END $campaign_rename$;

-- Standardize campaign_task table field naming - Single DO block
DO $campaign_task_rename$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'campaignId') THEN
        ALTER TABLE campaign_task RENAME COLUMN "campaignId" TO campaign_id;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'taskType') THEN
        ALTER TABLE campaign_task RENAME COLUMN "taskType" TO task_type;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'targetFormat') THEN
        ALTER TABLE campaign_task RENAME COLUMN "targetFormat" TO target_format;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'targetAsset') THEN
        ALTER TABLE campaign_task RENAME COLUMN "targetAsset" TO target_asset;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'imageUrl') THEN
        ALTER TABLE campaign_task RENAME COLUMN "imageUrl" TO image_url;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'createdAt') THEN
        ALTER TABLE campaign_task RENAME COLUMN "createdAt" TO created_at;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'campaign_task' AND column_name = 'updatedAt') THEN
        ALTER TABLE campaign_task RENAME COLUMN "updatedAt" TO updated_at;
    END IF;
END $campaign_task_rename$;

-- Add missing updated_at column to blog_posts if it doesn't exist
DO $blog_posts_update$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'blog_posts' AND column_name = 'updated_at') THEN
        ALTER TABLE blog_posts ADD COLUMN updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;
    END IF;
END $blog_posts_update$;

-- ===== STEP 2: CREATE PERFORMANCE TRACKING TABLES =====

-- Agent performance tracking
CREATE TABLE IF NOT EXISTS agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    execution_time_ms INTEGER,
    success_rate FLOAT CHECK (success_rate >= 0 AND success_rate <= 1),
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 10),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd DECIMAL(10,6),
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Agent decision logging
CREATE TABLE IF NOT EXISTS agent_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    blog_id UUID REFERENCES blog_posts(id),
    campaign_id UUID REFERENCES campaign(id),
    decision_context JSONB,
    reasoning TEXT,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    outcome VARCHAR(50), -- 'success', 'failure', 'partial'
    execution_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Blog analytics and performance
CREATE TABLE IF NOT EXISTS blog_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blog_id UUID NOT NULL REFERENCES blog_posts(id) ON DELETE CASCADE,
    views INTEGER DEFAULT 0,
    unique_visitors INTEGER DEFAULT 0,
    engagement_rate FLOAT DEFAULT 0,
    avg_time_on_page INTEGER, -- in seconds
    bounce_rate FLOAT DEFAULT 0,
    social_shares INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    conversion_rate FLOAT DEFAULT 0,
    seo_score FLOAT DEFAULT 0,
    readability_score FLOAT DEFAULT 0,
    recorded_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Campaign attribution and marketing metrics
CREATE TABLE IF NOT EXISTS marketing_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blog_id UUID REFERENCES blog_posts(id),
    campaign_id UUID REFERENCES campaign(id),
    metric_type VARCHAR(50) NOT NULL, -- 'views', 'clicks', 'conversions', 'revenue'
    metric_value FLOAT NOT NULL,
    source VARCHAR(100), -- 'organic', 'social', 'email', 'paid'
    medium VARCHAR(100), -- 'linkedin', 'twitter', 'instagram', 'email'
    campaign_name VARCHAR(200),
    recorded_at TIMESTAMPTZ DEFAULT now()
);

-- Content optimization tracking
CREATE TABLE IF NOT EXISTS content_optimization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blog_id UUID NOT NULL REFERENCES blog_posts(id) ON DELETE CASCADE,
    optimization_type VARCHAR(50) NOT NULL, -- 'seo', 'readability', 'engagement', 'conversion'
    before_score FLOAT,
    after_score FLOAT,
    optimization_prompt TEXT,
    agent_type VARCHAR(100),
    changes_made JSONB,
    improvement_percentage FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO and content metadata
CREATE TABLE IF NOT EXISTS seo_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blog_id UUID NOT NULL REFERENCES blog_posts(id) ON DELETE CASCADE,
    target_keywords JSONB,
    meta_description TEXT,
    meta_title TEXT,
    canonical_url VARCHAR(500),
    og_title VARCHAR(200),
    og_description TEXT,
    og_image_url VARCHAR(500),
    search_ranking INTEGER,
    organic_traffic INTEGER DEFAULT 0,
    keyword_density JSONB,
    internal_links_count INTEGER DEFAULT 0,
    external_links_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Content variations for A/B testing
CREATE TABLE IF NOT EXISTS content_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    blog_id UUID NOT NULL REFERENCES blog_posts(id) ON DELETE CASCADE,
    variant_type VARCHAR(50) NOT NULL, -- 'title', 'intro', 'cta', 'full_content'
    original_content TEXT,
    variant_content TEXT,
    performance_score FLOAT DEFAULT 0,
    traffic_allocation FLOAT DEFAULT 0.5, -- percentage of traffic to this variant
    conversion_rate FLOAT DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Feedback and learning system
CREATE TABLE IF NOT EXISTS agent_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    blog_id UUID REFERENCES blog_posts(id),
    campaign_id UUID REFERENCES campaign(id),
    task_id UUID REFERENCES campaign_task(id),
    feedback_type VARCHAR(50), -- 'user_rating', 'performance_metric', 'error_correction'
    feedback_value FLOAT,
    feedback_text TEXT,
    user_id VARCHAR(100), -- for user feedback
    is_positive BOOLEAN,
    learning_context JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ===== STEP 3: CREATE PERFORMANCE INDEXES =====

-- Blog search and performance optimization
CREATE INDEX IF NOT EXISTS idx_blog_posts_status_created ON blog_posts(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_blog_posts_title_trgm ON blog_posts USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_blog_posts_content_trgm ON blog_posts USING gin (content_markdown gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_blog_posts_updated_at ON blog_posts(updated_at DESC);

-- Campaign optimization
CREATE INDEX IF NOT EXISTS idx_campaign_blog_id ON campaign(blog_id);
CREATE INDEX IF NOT EXISTS idx_campaign_created_at ON campaign(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_campaign_task_status ON campaign_task(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_campaign_task_type ON campaign_task(task_type);
CREATE INDEX IF NOT EXISTS idx_campaign_task_campaign_id ON campaign_task(campaign_id);

-- Vector search optimization (improved from default)
DROP INDEX IF EXISTS document_chunks_embedding_idx;
CREATE INDEX IF NOT EXISTS document_chunks_embedding_cosine_idx 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Document search optimization
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at DESC);

-- Agent performance indexes
CREATE INDEX IF NOT EXISTS idx_agent_performance_type_task ON agent_performance(agent_type, task_type);
CREATE INDEX IF NOT EXISTS idx_agent_performance_created_at ON agent_performance(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent_type ON agent_decisions(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_blog_id ON agent_decisions(blog_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_created_at ON agent_decisions(created_at DESC);

-- Analytics and metrics indexes
CREATE INDEX IF NOT EXISTS idx_blog_analytics_blog_id ON blog_analytics(blog_id);
CREATE INDEX IF NOT EXISTS idx_blog_analytics_recorded_at ON blog_analytics(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_marketing_metrics_blog_id ON marketing_metrics(blog_id);
CREATE INDEX IF NOT EXISTS idx_marketing_metrics_campaign_id ON marketing_metrics(campaign_id);
CREATE INDEX IF NOT EXISTS idx_marketing_metrics_type_recorded ON marketing_metrics(metric_type, recorded_at DESC);

-- Content optimization indexes
CREATE INDEX IF NOT EXISTS idx_content_optimization_blog_id ON content_optimization(blog_id);
CREATE INDEX IF NOT EXISTS idx_content_optimization_type ON content_optimization(optimization_type);
CREATE INDEX IF NOT EXISTS idx_seo_metadata_blog_id ON seo_metadata(blog_id);
CREATE INDEX IF NOT EXISTS idx_content_variants_blog_id ON content_variants(blog_id);
CREATE INDEX IF NOT EXISTS idx_content_variants_active ON content_variants(is_active, variant_type);

-- Feedback system indexes
CREATE INDEX IF NOT EXISTS idx_agent_feedback_agent_type ON agent_feedback(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_feedback_blog_id ON agent_feedback(blog_id);
CREATE INDEX IF NOT EXISTS idx_agent_feedback_created_at ON agent_feedback(created_at DESC);

-- ===== STEP 4: CREATE AUTOMATED TRIGGERS =====

-- Auto-update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers individually
DROP TRIGGER IF EXISTS update_blog_posts_updated_at ON blog_posts;
CREATE TRIGGER update_blog_posts_updated_at
    BEFORE UPDATE ON blog_posts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_campaign_task_updated_at ON campaign_task;
CREATE TRIGGER update_campaign_task_updated_at
    BEFORE UPDATE ON campaign_task
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_blog_analytics_updated_at ON blog_analytics;
CREATE TRIGGER update_blog_analytics_updated_at
    BEFORE UPDATE ON blog_analytics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_seo_metadata_updated_at ON seo_metadata;
CREATE TRIGGER update_seo_metadata_updated_at
    BEFORE UPDATE ON seo_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_content_variants_updated_at ON content_variants;
CREATE TRIGGER update_content_variants_updated_at
    BEFORE UPDATE ON content_variants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===== STEP 5: CREATE USEFUL VIEWS =====

-- Campaign performance view
CREATE OR REPLACE VIEW campaign_performance AS
SELECT 
    c.id as campaign_id,
    c.blog_id,
    bp.title as blog_title,
    COUNT(ct.id) as total_tasks,
    COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN ct.status = 'failed' THEN 1 END) as failed_tasks,
    ROUND(
        COUNT(CASE WHEN ct.status = 'completed' THEN 1 END)::FLOAT / 
        NULLIF(COUNT(ct.id), 0) * 100, 2
    ) as completion_rate,
    COALESCE(AVG(ba.engagement_rate), 0) as avg_engagement_rate,
    COALESCE(SUM(mm.metric_value) FILTER (WHERE mm.metric_type = 'conversions'), 0) as total_conversions,
    c.created_at
FROM campaign c
LEFT JOIN blog_posts bp ON c.blog_id = bp.id
LEFT JOIN campaign_task ct ON c.id = ct.campaign_id
LEFT JOIN blog_analytics ba ON c.blog_id = ba.blog_id
LEFT JOIN marketing_metrics mm ON c.id = mm.campaign_id
GROUP BY c.id, c.blog_id, bp.title, c.created_at;

-- Agent efficiency view
CREATE OR REPLACE VIEW agent_efficiency AS
SELECT 
    agent_type,
    task_type,
    COUNT(*) as total_executions,
    AVG(execution_time_ms) as avg_execution_time,
    AVG(success_rate) as avg_success_rate,
    AVG(quality_score) as avg_quality_score,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(cost_usd) as total_cost,
    MAX(created_at) as last_execution
FROM agent_performance
GROUP BY agent_type, task_type
ORDER BY avg_quality_score DESC, avg_execution_time ASC;

-- Blog content performance view
CREATE OR REPLACE VIEW blog_content_performance AS
SELECT 
    bp.id,
    bp.title,
    bp.status,
    bp.created_at,
    COALESCE(ba.views, 0) as views,
    COALESCE(ba.engagement_rate, 0) as engagement_rate,
    COALESCE(ba.seo_score, 0) as seo_score,
    COALESCE(ba.social_shares, 0) as social_shares,
    COALESCE(COUNT(cv.id), 0) as variant_count,
    COALESCE(AVG(co.improvement_percentage), 0) as avg_optimization_improvement
FROM blog_posts bp
LEFT JOIN blog_analytics ba ON bp.id = ba.blog_id
LEFT JOIN content_variants cv ON bp.id = cv.blog_id AND cv.is_active = true
LEFT JOIN content_optimization co ON bp.id = co.blog_id
GROUP BY bp.id, bp.title, bp.status, bp.created_at, ba.views, ba.engagement_rate, ba.seo_score, ba.social_shares;

-- ===== STEP 6: UPDATE RLS POLICIES FOR NEW TABLES =====

-- Enable RLS on new tables
ALTER TABLE agent_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE blog_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketing_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_optimization ENABLE ROW LEVEL SECURITY;
ALTER TABLE seo_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_variants ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_feedback ENABLE ROW LEVEL SECURITY;

-- Create public access policies for development (modify for production)
-- agent_performance policies
DROP POLICY IF EXISTS "Public access SELECT on agent_performance" ON agent_performance;
CREATE POLICY "Public access SELECT on agent_performance" ON agent_performance FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on agent_performance" ON agent_performance;
CREATE POLICY "Public access INSERT on agent_performance" ON agent_performance FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on agent_performance" ON agent_performance;
CREATE POLICY "Public access UPDATE on agent_performance" ON agent_performance FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on agent_performance" ON agent_performance;
CREATE POLICY "Public access DELETE on agent_performance" ON agent_performance FOR DELETE USING (true);

-- agent_decisions policies
DROP POLICY IF EXISTS "Public access SELECT on agent_decisions" ON agent_decisions;
CREATE POLICY "Public access SELECT on agent_decisions" ON agent_decisions FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on agent_decisions" ON agent_decisions;
CREATE POLICY "Public access INSERT on agent_decisions" ON agent_decisions FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on agent_decisions" ON agent_decisions;
CREATE POLICY "Public access UPDATE on agent_decisions" ON agent_decisions FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on agent_decisions" ON agent_decisions;
CREATE POLICY "Public access DELETE on agent_decisions" ON agent_decisions FOR DELETE USING (true);

-- blog_analytics policies
DROP POLICY IF EXISTS "Public access SELECT on blog_analytics" ON blog_analytics;
CREATE POLICY "Public access SELECT on blog_analytics" ON blog_analytics FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on blog_analytics" ON blog_analytics;
CREATE POLICY "Public access INSERT on blog_analytics" ON blog_analytics FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on blog_analytics" ON blog_analytics;
CREATE POLICY "Public access UPDATE on blog_analytics" ON blog_analytics FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on blog_analytics" ON blog_analytics;
CREATE POLICY "Public access DELETE on blog_analytics" ON blog_analytics FOR DELETE USING (true);

-- marketing_metrics policies
DROP POLICY IF EXISTS "Public access SELECT on marketing_metrics" ON marketing_metrics;
CREATE POLICY "Public access SELECT on marketing_metrics" ON marketing_metrics FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on marketing_metrics" ON marketing_metrics;
CREATE POLICY "Public access INSERT on marketing_metrics" ON marketing_metrics FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on marketing_metrics" ON marketing_metrics;
CREATE POLICY "Public access UPDATE on marketing_metrics" ON marketing_metrics FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on marketing_metrics" ON marketing_metrics;
CREATE POLICY "Public access DELETE on marketing_metrics" ON marketing_metrics FOR DELETE USING (true);

-- content_optimization policies
DROP POLICY IF EXISTS "Public access SELECT on content_optimization" ON content_optimization;
CREATE POLICY "Public access SELECT on content_optimization" ON content_optimization FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on content_optimization" ON content_optimization;
CREATE POLICY "Public access INSERT on content_optimization" ON content_optimization FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on content_optimization" ON content_optimization;
CREATE POLICY "Public access UPDATE on content_optimization" ON content_optimization FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on content_optimization" ON content_optimization;
CREATE POLICY "Public access DELETE on content_optimization" ON content_optimization FOR DELETE USING (true);

-- seo_metadata policies
DROP POLICY IF EXISTS "Public access SELECT on seo_metadata" ON seo_metadata;
CREATE POLICY "Public access SELECT on seo_metadata" ON seo_metadata FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on seo_metadata" ON seo_metadata;
CREATE POLICY "Public access INSERT on seo_metadata" ON seo_metadata FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on seo_metadata" ON seo_metadata;
CREATE POLICY "Public access UPDATE on seo_metadata" ON seo_metadata FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on seo_metadata" ON seo_metadata;
CREATE POLICY "Public access DELETE on seo_metadata" ON seo_metadata FOR DELETE USING (true);

-- content_variants policies
DROP POLICY IF EXISTS "Public access SELECT on content_variants" ON content_variants;
CREATE POLICY "Public access SELECT on content_variants" ON content_variants FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on content_variants" ON content_variants;
CREATE POLICY "Public access INSERT on content_variants" ON content_variants FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on content_variants" ON content_variants;
CREATE POLICY "Public access UPDATE on content_variants" ON content_variants FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on content_variants" ON content_variants;
CREATE POLICY "Public access DELETE on content_variants" ON content_variants FOR DELETE USING (true);

-- agent_feedback policies
DROP POLICY IF EXISTS "Public access SELECT on agent_feedback" ON agent_feedback;
CREATE POLICY "Public access SELECT on agent_feedback" ON agent_feedback FOR SELECT USING (true);
DROP POLICY IF EXISTS "Public access INSERT on agent_feedback" ON agent_feedback;
CREATE POLICY "Public access INSERT on agent_feedback" ON agent_feedback FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Public access UPDATE on agent_feedback" ON agent_feedback;
CREATE POLICY "Public access UPDATE on agent_feedback" ON agent_feedback FOR UPDATE USING (true);
DROP POLICY IF EXISTS "Public access DELETE on agent_feedback" ON agent_feedback;
CREATE POLICY "Public access DELETE on agent_feedback" ON agent_feedback FOR DELETE USING (true);

-- ===== VERIFICATION =====

-- Verify new tables were created
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_name IN (
        'agent_performance', 'agent_decisions', 'blog_analytics', 
        'marketing_metrics', 'content_optimization', 'seo_metadata',
        'content_variants', 'agent_feedback'
    )
ORDER BY table_name;

-- Success message
SELECT 'Database improvements successfully applied!' as status;