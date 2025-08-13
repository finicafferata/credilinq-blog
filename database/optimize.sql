-- Performance optimization queries for CrediLinq database
-- Includes indexes, query optimizations, and maintenance procedures

-- ====================================
-- INDEXES FOR PERFORMANCE
-- ====================================

-- Blog posts indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_status 
ON blog_posts (status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_created_at 
ON blog_posts (created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_user_created 
ON blog_posts (user_id, created_at DESC) 
WHERE user_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_status_created 
ON blog_posts (status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_title_search 
ON blog_posts USING gin(to_tsvector('english', title));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blog_posts_content_search 
ON blog_posts USING gin(to_tsvector('english', content_markdown));

-- Campaign indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_blog_id 
ON "Campaign" (blog_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_status 
ON "Campaign" (status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaigns_created_at 
ON "Campaign" (created_at DESC);

-- Campaign tasks indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_tasks_campaign_id 
ON "CampaignTask" (campaign_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_tasks_status 
ON "CampaignTask" (status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_tasks_type 
ON "CampaignTask" (task_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_tasks_campaign_status 
ON "CampaignTask" (campaign_id, status);

-- Agent performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_agent_type 
ON "AgentPerformance" (agent_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_created_at 
ON "AgentPerformance" (created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_blog_id 
ON "AgentPerformance" (blog_id) 
WHERE blog_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_performance_agent_created 
ON "AgentPerformance" (agent_type, created_at DESC);

-- Agent decisions indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_decisions_agent_id 
ON "AgentDecision" (agent_performance_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_decisions_decision_type 
ON "AgentDecision" (decision_type);

-- Document indexes (for knowledge base)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_status 
ON "Document" (status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_type 
ON "Document" (document_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_created_at 
ON "Document" (created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_title_search 
ON "Document" USING gin(to_tsvector('english', title));

-- Document chunks indexes (for vector search)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id 
ON "DocumentChunk" (document_id);

-- Vector similarity index (if using pgvector)
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding 
-- ON "DocumentChunk" USING ivfflat (embedding vector_cosine_ops);

-- ====================================
-- MATERIALIZED VIEWS FOR ANALYTICS
-- ====================================

-- Blog analytics summary
CREATE MATERIALIZED VIEW IF NOT EXISTS blog_analytics_summary AS
SELECT 
    bp.id,
    bp.title,
    bp.status,
    bp.created_at,
    COUNT(ap.id) as agent_interactions,
    AVG(ap.execution_time_seconds) as avg_execution_time,
    SUM(CASE WHEN ap.agent_type = 'content_agent' THEN ap.execution_time_seconds ELSE 0 END) as content_time,
    SUM(CASE WHEN ap.agent_type = 'editor_agent' THEN ap.execution_time_seconds ELSE 0 END) as editor_time,
    COUNT(c.id) as campaign_count,
    COUNT(ct.id) as total_tasks
FROM blog_posts bp
LEFT JOIN "AgentPerformance" ap ON bp.id = ap.blog_id
LEFT JOIN "Campaign" c ON bp.id = c.blog_id
LEFT JOIN "CampaignTask" ct ON c.id = ct.campaign_id
GROUP BY bp.id, bp.title, bp.status, bp.created_at;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_blog_analytics_summary_status 
ON blog_analytics_summary (status);

CREATE INDEX IF NOT EXISTS idx_blog_analytics_summary_created 
ON blog_analytics_summary (created_at DESC);

-- Agent performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS agent_performance_summary AS
SELECT 
    agent_type,
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as total_executions,
    AVG(execution_time_seconds) as avg_execution_time,
    MIN(execution_time_seconds) as min_execution_time,
    MAX(execution_time_seconds) as max_execution_time,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_executions,
    COUNT(*) - SUM(CASE WHEN success THEN 1 ELSE 0 END) as failed_executions,
    AVG(CASE WHEN success THEN execution_time_seconds ELSE NULL END) as avg_success_time
FROM "AgentPerformance"
GROUP BY agent_type, DATE_TRUNC('day', created_at);

-- Create indexes on agent performance summary
CREATE INDEX IF NOT EXISTS idx_agent_performance_summary_type_date 
ON agent_performance_summary (agent_type, date DESC);

-- ====================================
-- QUERY OPTIMIZATION FUNCTIONS
-- ====================================

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY blog_analytics_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY agent_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Function to get blog performance metrics
CREATE OR REPLACE FUNCTION get_blog_performance(blog_id_param UUID)
RETURNS TABLE(
    total_time NUMERIC,
    agent_count BIGINT,
    success_rate NUMERIC,
    average_time NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(execution_time_seconds), 0) as total_time,
        COUNT(*) as agent_count,
        CASE 
            WHEN COUNT(*) = 0 THEN 0
            ELSE ROUND(SUM(CASE WHEN success THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100, 2)
        END as success_rate,
        COALESCE(AVG(execution_time_seconds), 0) as average_time
    FROM "AgentPerformance"
    WHERE blog_id = blog_id_param;
END;
$$ LANGUAGE plpgsql;

-- Function to get recent blogs with performance data
CREATE OR REPLACE FUNCTION get_recent_blogs_with_performance(limit_param INTEGER DEFAULT 50)
RETURNS TABLE(
    id UUID,
    title TEXT,
    status TEXT,
    created_at TIMESTAMP,
    agent_interactions BIGINT,
    avg_execution_time NUMERIC,
    campaign_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        bas.id,
        bas.title,
        bas.status,
        bas.created_at,
        bas.agent_interactions,
        bas.avg_execution_time,
        bas.campaign_count
    FROM blog_analytics_summary bas
    ORDER BY bas.created_at DESC
    LIMIT limit_param;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- MAINTENANCE PROCEDURES
-- ====================================

-- Function to clean up old performance data
CREATE OR REPLACE FUNCTION cleanup_old_performance_data(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old agent performance records
    DELETE FROM "AgentPerformance" 
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up orphaned agent decisions
    DELETE FROM "AgentDecision" 
    WHERE agent_performance_id NOT IN (SELECT id FROM "AgentPerformance");
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze table statistics
CREATE OR REPLACE FUNCTION analyze_table_stats()
RETURNS void AS $$
BEGIN
    ANALYZE blog_posts;
    ANALYZE "Campaign";
    ANALYZE "CampaignTask";
    ANALYZE "AgentPerformance";
    ANALYZE "AgentDecision";
    ANALYZE "Document";
    ANALYZE "DocumentChunk";
END;
$$ LANGUAGE plpgsql;

-- Function to get database size information
CREATE OR REPLACE FUNCTION get_database_size_info()
RETURNS TABLE(
    table_name TEXT,
    row_count BIGINT,
    total_size TEXT,
    index_size TEXT,
    table_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        n_tup_ins + n_tup_upd + n_tup_del as row_count,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size
    FROM pg_stat_user_tables 
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- AUTOMATIC MAINTENANCE SETUP
-- ====================================

-- Create a function to run daily maintenance
CREATE OR REPLACE FUNCTION daily_maintenance()
RETURNS void AS $$
BEGIN
    -- Refresh materialized views
    PERFORM refresh_analytics_views();
    
    -- Update table statistics
    PERFORM analyze_table_stats();
    
    -- Clean up old data (keep 90 days)
    PERFORM cleanup_old_performance_data(90);
    
    -- Log maintenance completion
    INSERT INTO "AgentPerformance" (
        agent_type, 
        execution_time_seconds, 
        success, 
        metadata,
        created_at
    ) VALUES (
        'maintenance_agent',
        0.1,
        true,
        '{"task": "daily_maintenance", "completed_at": "' || NOW()::TEXT || '"}',
        NOW()
    );
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- QUERY PERFORMANCE HINTS
-- ====================================

-- Common optimized queries

-- Get blogs with pagination and performance data
-- SELECT * FROM get_recent_blogs_with_performance(20);

-- Get agent performance for specific blog
-- SELECT * FROM get_blog_performance('blog-uuid-here');

-- Search blogs by title with full-text search
-- SELECT id, title, status, ts_rank(to_tsvector('english', title), query) as rank
-- FROM blog_posts, to_tsquery('english', 'search terms') query
-- WHERE to_tsvector('english', title) @@ query
-- ORDER BY rank DESC, created_at DESC;

-- Get campaign completion statistics
-- SELECT 
--     c.id,
--     c.status,
--     COUNT(ct.id) as total_tasks,
--     SUM(CASE WHEN ct.status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
--     ROUND(SUM(CASE WHEN ct.status = 'completed' THEN 1 ELSE 0 END)::NUMERIC / COUNT(ct.id) * 100, 2) as completion_rate
-- FROM "Campaign" c
-- LEFT JOIN "CampaignTask" ct ON c.id = ct.campaign_id
-- GROUP BY c.id, c.status;

COMMIT;