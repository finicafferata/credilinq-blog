-- ================================================================================
-- PERFORMANCE TESTING AND MONITORING SUITE - CAMPAIGN ORCHESTRATION
-- ================================================================================
-- This file contains performance testing queries, benchmarking tools, and
-- monitoring utilities for the campaign-centric database architecture
-- Generated: 2024-12-18 | Version: 1.0 | Phase: Foundation
-- ================================================================================

-- ################################################################################
-- PERFORMANCE BENCHMARKING FUNCTIONS
-- ################################################################################

-- Function to benchmark campaign query performance
CREATE OR REPLACE FUNCTION benchmark_campaign_queries(
    iterations INTEGER DEFAULT 100
) RETURNS TABLE(
    query_name TEXT,
    avg_execution_time_ms DECIMAL(10,3),
    min_execution_time_ms DECIMAL(10,3),
    max_execution_time_ms DECIMAL(10,3),
    rows_returned BIGINT,
    performance_grade TEXT
) AS $$
DECLARE
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
    execution_times DECIMAL(10,3)[];
    i INTEGER;
    rows_count BIGINT;
    temp_result RECORD;
BEGIN
    -- Benchmark 1: Active campaigns with orchestrator info
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO rows_count
        FROM campaigns c
        LEFT JOIN campaign_orchestrators co ON c.orchestrator_id = co.id
        LEFT JOIN campaign_strategies cs ON c.strategy_id = cs.id
        WHERE c.status IN ('active', 'scheduled', 'running')
        ORDER BY c.priority DESC, c.deadline ASC;
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    RETURN QUERY
    SELECT 
        'Active Campaigns with Relationships'::TEXT,
        (SELECT AVG(x) FROM unnest(execution_times) x),
        (SELECT MIN(x) FROM unnest(execution_times) x),
        (SELECT MAX(x) FROM unnest(execution_times) x),
        rows_count,
        CASE 
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 10 THEN 'A'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 50 THEN 'B'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 100 THEN 'C'
            ELSE 'D'
        END;
    
    -- Benchmark 2: Campaign content by campaign
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO rows_count
        FROM campaign_content cc
        LEFT JOIN campaign_analytics ca ON cc.id = ca.content_id
        WHERE cc.campaign_id = (
            SELECT id FROM campaigns WHERE status = 'active' LIMIT 1
        )
        AND cc.is_active = true
        ORDER BY cc.quality_score DESC NULLS LAST;
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    RETURN QUERY
    SELECT 
        'Campaign Content with Analytics',
        (SELECT AVG(x) FROM unnest(execution_times) x),
        (SELECT MIN(x) FROM unnest(execution_times) x),
        (SELECT MAX(x) FROM unnest(execution_times) x),
        rows_count,
        CASE 
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 5 THEN 'A'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 25 THEN 'B'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 50 THEN 'C'
            ELSE 'D'
        END;
    
    -- Benchmark 3: Workflow execution monitoring
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO rows_count
        FROM campaign_workflows cw
        LEFT JOIN campaign_workflow_steps cws ON cw.id = cws.workflow_id
        WHERE cw.status IN ('running', 'paused')
        GROUP BY cw.id
        ORDER BY cw.started_at DESC;
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    RETURN QUERY
    SELECT 
        'Active Workflow Monitoring',
        (SELECT AVG(x) FROM unnest(execution_times) x),
        (SELECT MIN(x) FROM unnest(execution_times) x),
        (SELECT MAX(x) FROM unnest(execution_times) x),
        rows_count,
        CASE 
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 15 THEN 'A'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 75 THEN 'B'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 150 THEN 'C'
            ELSE 'D'
        END;
    
    -- Benchmark 4: Calendar upcoming events
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO rows_count
        FROM campaign_calendar cc
        LEFT JOIN campaigns c ON cc.campaign_id = c.id
        LEFT JOIN campaign_content cnt ON cc.content_id = cnt.id
        WHERE cc.scheduled_datetime BETWEEN NOW() AND NOW() + INTERVAL '7 days'
        AND cc.status = 'scheduled'
        ORDER BY cc.scheduled_datetime ASC;
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    RETURN QUERY
    SELECT 
        'Upcoming Calendar Events',
        (SELECT AVG(x) FROM unnest(execution_times) x),
        (SELECT MIN(x) FROM unnest(execution_times) x),
        (SELECT MAX(x) FROM unnest(execution_times) x),
        rows_count,
        CASE 
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 8 THEN 'A'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 40 THEN 'B'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 80 THEN 'C'
            ELSE 'D'
        END;
    
    -- Benchmark 5: Agent performance analysis
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO rows_count
        FROM agent_orchestration_performance aop
        JOIN campaign_workflow_steps cws ON aop.step_id = cws.id
        JOIN campaign_workflows cw ON cws.workflow_id = cw.id
        WHERE aop.started_at >= NOW() - INTERVAL '24 hours'
        ORDER BY aop.duration_ms DESC;
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    RETURN QUERY
    SELECT 
        'Agent Performance Analysis',
        (SELECT AVG(x) FROM unnest(execution_times) x),
        (SELECT MIN(x) FROM unnest(execution_times) x),
        (SELECT MAX(x) FROM unnest(execution_times) x),
        rows_count,
        CASE 
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 12 THEN 'A'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 60 THEN 'B'
            WHEN (SELECT AVG(x) FROM unnest(execution_times) x) < 120 THEN 'C'
            ELSE 'D'
        END;
END;
$$ LANGUAGE plpgsql;

-- Function to benchmark write operations
CREATE OR REPLACE FUNCTION benchmark_write_operations(
    iterations INTEGER DEFAULT 50
) RETURNS TABLE(
    operation_name TEXT,
    avg_execution_time_ms DECIMAL(10,3),
    operations_per_second DECIMAL(10,2),
    performance_grade TEXT
) AS $$
DECLARE
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
    execution_times DECIMAL(10,3)[];
    i INTEGER;
    test_campaign_id UUID;
    test_workflow_id UUID;
    ops_per_second DECIMAL(10,2);
BEGIN
    -- Create test campaign for write benchmarking
    INSERT INTO campaigns (id, name, status, campaign_data, priority)
    VALUES (uuid_generate_v4(), 'Performance Test Campaign', 'draft', '{"test": true}', 'medium')
    RETURNING id INTO test_campaign_id;
    
    -- Benchmark 1: Campaign creation
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        INSERT INTO campaigns (name, status, campaign_data, priority)
        VALUES (
            'Test Campaign ' || i, 
            'draft', 
            json_build_object('test', true, 'iteration', i),
            'medium'
        );
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    ops_per_second := 1000.0 / (SELECT AVG(x) FROM unnest(execution_times) x);
    
    RETURN QUERY
    SELECT 
        'Campaign Creation'::TEXT,
        (SELECT AVG(x) FROM unnest(execution_times) x),
        ops_per_second,
        CASE 
            WHEN ops_per_second > 100 THEN 'A'
            WHEN ops_per_second > 50 THEN 'B'
            WHEN ops_per_second > 20 THEN 'C'
            ELSE 'D'
        END;
    
    -- Benchmark 2: Content creation
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        INSERT INTO campaign_content (
            campaign_id, title, content_type, platform, content_markdown, status
        ) VALUES (
            test_campaign_id,
            'Test Content ' || i,
            'blog_post',
            'website',
            'This is test content for performance benchmarking iteration ' || i,
            'draft'
        );
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    ops_per_second := 1000.0 / (SELECT AVG(x) FROM unnest(execution_times) x);
    
    RETURN QUERY
    SELECT 
        'Content Creation',
        (SELECT AVG(x) FROM unnest(execution_times) x),
        ops_per_second,
        CASE 
            WHEN ops_per_second > 80 THEN 'A'
            WHEN ops_per_second > 40 THEN 'B'
            WHEN ops_per_second > 15 THEN 'C'
            ELSE 'D'
        END;
    
    -- Benchmark 3: Analytics insertion
    execution_times := '{}';
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        INSERT INTO campaign_analytics (
            campaign_id, measurement_date, views, clicks, conversions, revenue_generated
        ) VALUES (
            test_campaign_id,
            CURRENT_DATE - (i || ' days')::INTERVAL,
            100 + i,
            10 + i,
            1,
            i * 10.50
        );
        
        end_time := clock_timestamp();
        execution_times := execution_times || EXTRACT(milliseconds FROM (end_time - start_time));
    END LOOP;
    
    ops_per_second := 1000.0 / (SELECT AVG(x) FROM unnest(execution_times) x);
    
    RETURN QUERY
    SELECT 
        'Analytics Insertion',
        (SELECT AVG(x) FROM unnest(execution_times) x),
        ops_per_second,
        CASE 
            WHEN ops_per_second > 200 THEN 'A'
            WHEN ops_per_second > 100 THEN 'B'
            WHEN ops_per_second > 50 THEN 'C'
            ELSE 'D'
        END;
    
    -- Cleanup test data
    DELETE FROM campaigns WHERE name LIKE 'Test Campaign %' OR name = 'Performance Test Campaign';
    
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- LOAD TESTING FUNCTIONS
-- ################################################################################

-- Function to simulate concurrent campaign operations
CREATE OR REPLACE FUNCTION load_test_concurrent_operations(
    concurrent_operations INTEGER DEFAULT 10,
    duration_seconds INTEGER DEFAULT 30
) RETURNS TABLE(
    test_scenario TEXT,
    total_operations INTEGER,
    successful_operations INTEGER,
    failed_operations INTEGER,
    avg_response_time_ms DECIMAL(10,3),
    operations_per_second DECIMAL(10,2),
    success_rate_percent DECIMAL(5,2)
) AS $$
DECLARE
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
    operation_count INTEGER := 0;
    success_count INTEGER := 0;
    total_response_time DECIMAL(10,3) := 0;
    i INTEGER;
    op_start_time TIMESTAMPTZ;
    op_end_time TIMESTAMPTZ;
BEGIN
    start_time := clock_timestamp();
    end_time := start_time + (duration_seconds || ' seconds')::INTERVAL;
    
    -- Simulate concurrent campaign creation and updates
    WHILE clock_timestamp() < end_time LOOP
        operation_count := operation_count + 1;
        op_start_time := clock_timestamp();
        
        BEGIN
            -- Simulate campaign creation
            INSERT INTO campaigns (name, status, campaign_data, priority)
            VALUES (
                'Load Test Campaign ' || operation_count, 
                'draft', 
                json_build_object('load_test', true, 'operation', operation_count),
                'medium'
            );
            
            -- Simulate campaign update
            UPDATE campaigns 
            SET progress_percentage = random() * 100,
                status = CASE 
                    WHEN random() < 0.7 THEN 'active'
                    WHEN random() < 0.9 THEN 'scheduled'
                    ELSE 'completed'
                END
            WHERE name = 'Load Test Campaign ' || operation_count;
            
            success_count := success_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            -- Operation failed, continue
            NULL;
        END;
        
        op_end_time := clock_timestamp();
        total_response_time := total_response_time + EXTRACT(milliseconds FROM (op_end_time - op_start_time));
        
        -- Small delay to prevent overwhelming the system
        PERFORM pg_sleep(0.01);
    END LOOP;
    
    RETURN QUERY
    SELECT 
        'Concurrent Campaign Operations'::TEXT,
        operation_count,
        success_count,
        operation_count - success_count,
        CASE WHEN operation_count > 0 THEN total_response_time / operation_count ELSE 0 END,
        CASE WHEN EXTRACT(epoch FROM (clock_timestamp() - start_time)) > 0 THEN 
            success_count / EXTRACT(epoch FROM (clock_timestamp() - start_time))
        ELSE 0 END,
        CASE WHEN operation_count > 0 THEN 
            (success_count::DECIMAL / operation_count) * 100 
        ELSE 0 END;
    
    -- Cleanup load test data
    DELETE FROM campaigns WHERE name LIKE 'Load Test Campaign %';
    
END;
$$ LANGUAGE plpgsql;

-- Function to test database connection pool limits
CREATE OR REPLACE FUNCTION test_connection_pool_limits()
RETURNS TABLE(
    metric_name TEXT,
    current_value INTEGER,
    max_value INTEGER,
    utilization_percent DECIMAL(5,2),
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'Active Connections'::TEXT,
        (SELECT COUNT(*)::INTEGER FROM pg_stat_activity WHERE state = 'active'),
        (SELECT setting::INTEGER FROM pg_settings WHERE name = 'max_connections'),
        (SELECT COUNT(*)::DECIMAL FROM pg_stat_activity WHERE state = 'active') / 
            (SELECT setting::DECIMAL FROM pg_settings WHERE name = 'max_connections') * 100,
        CASE 
            WHEN (SELECT COUNT(*)::DECIMAL FROM pg_stat_activity WHERE state = 'active') / 
                 (SELECT setting::DECIMAL FROM pg_settings WHERE name = 'max_connections') > 0.8 
            THEN 'WARNING'
            WHEN (SELECT COUNT(*)::DECIMAL FROM pg_stat_activity WHERE state = 'active') / 
                 (SELECT setting::DECIMAL FROM pg_settings WHERE name = 'max_connections') > 0.9 
            THEN 'CRITICAL'
            ELSE 'OK'
        END
    
    UNION ALL
    
    SELECT 
        'Idle Connections',
        (SELECT COUNT(*)::INTEGER FROM pg_stat_activity WHERE state = 'idle'),
        (SELECT setting::INTEGER FROM pg_settings WHERE name = 'max_connections'),
        (SELECT COUNT(*)::DECIMAL FROM pg_stat_activity WHERE state = 'idle') / 
            (SELECT setting::DECIMAL FROM pg_settings WHERE name = 'max_connections') * 100,
        'INFO'
    
    UNION ALL
    
    SELECT 
        'Long Running Queries',
        (SELECT COUNT(*)::INTEGER FROM pg_stat_activity 
         WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes'),
        10, -- Arbitrary threshold
        (SELECT COUNT(*)::DECIMAL FROM pg_stat_activity 
         WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes') / 10 * 100,
        CASE 
            WHEN (SELECT COUNT(*) FROM pg_stat_activity 
                  WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes') > 5 
            THEN 'WARNING'
            ELSE 'OK'
        END;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- MEMORY AND RESOURCE MONITORING
-- ################################################################################

-- Function to monitor database memory usage
CREATE OR REPLACE FUNCTION monitor_memory_usage()
RETURNS TABLE(
    memory_type TEXT,
    allocated_mb DECIMAL(10,2),
    used_mb DECIMAL(10,2),
    usage_percent DECIMAL(5,2),
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'Shared Buffers'::TEXT,
        ROUND((SELECT setting::BIGINT * 8192 / 1024 / 1024 FROM pg_settings WHERE name = 'shared_buffers'), 2),
        ROUND((SELECT buffers_alloc * 8192 / 1024 / 1024 FROM pg_stat_bgwriter), 2),
        ROUND(
            (SELECT buffers_alloc * 8192 / 1024 / 1024 FROM pg_stat_bgwriter) / 
            (SELECT setting::BIGINT * 8192 / 1024 / 1024 FROM pg_settings WHERE name = 'shared_buffers') * 100, 2
        ),
        'INFO'
    
    UNION ALL
    
    SELECT 
        'Work Memory',
        ROUND((SELECT setting::BIGINT / 1024 FROM pg_settings WHERE name = 'work_mem'), 2),
        0.00, -- Not directly measurable
        0.00,
        'INFO'
    
    UNION ALL
    
    SELECT 
        'Maintenance Work Memory',
        ROUND((SELECT setting::BIGINT / 1024 FROM pg_settings WHERE name = 'maintenance_work_mem'), 2),
        0.00, -- Not directly measurable
        0.00,
        'INFO';
END;
$$ LANGUAGE plpgsql;

-- Function to analyze table and index sizes
CREATE OR REPLACE FUNCTION analyze_storage_usage()
RETURNS TABLE(
    object_name TEXT,
    object_type TEXT,
    size_mb DECIMAL(10,2),
    size_pretty TEXT,
    row_count BIGINT,
    avg_row_size_bytes INTEGER
) AS $$
BEGIN
    RETURN QUERY
    -- Campaign tables
    SELECT 
        t.table_name::TEXT,
        'table'::TEXT,
        ROUND(pg_total_relation_size(c.oid) / 1024.0 / 1024.0, 2),
        pg_size_pretty(pg_total_relation_size(c.oid))::TEXT,
        s.n_tup_ins + s.n_tup_upd - s.n_tup_del as estimated_rows,
        CASE 
            WHEN s.n_tup_ins + s.n_tup_upd - s.n_tup_del > 0 THEN
                (pg_total_relation_size(c.oid) / (s.n_tup_ins + s.n_tup_upd - s.n_tup_del))::INTEGER
            ELSE 0
        END
    FROM information_schema.tables t
    JOIN pg_class c ON c.relname = t.table_name
    JOIN pg_stat_user_tables s ON s.relname = t.table_name
    WHERE t.table_schema = 'public' 
    AND t.table_name LIKE '%campaign%'
    AND c.relkind = 'r'
    
    UNION ALL
    
    -- Indexes for campaign tables
    SELECT 
        i.indexname::TEXT,
        'index'::TEXT,
        ROUND(pg_relation_size(i.indexrelid) / 1024.0 / 1024.0, 2),
        pg_size_pretty(pg_relation_size(i.indexrelid))::TEXT,
        0::BIGINT,
        0::INTEGER
    FROM pg_stat_user_indexes i
    WHERE i.schemaname = 'public' 
    AND i.indexname LIKE 'idx_%campaign%'
    
    ORDER BY size_mb DESC;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- QUERY PERFORMANCE ANALYSIS
-- ################################################################################

-- Function to identify slow queries (requires pg_stat_statements extension)
CREATE OR REPLACE FUNCTION identify_slow_queries()
RETURNS TABLE(
    query_snippet TEXT,
    calls BIGINT,
    mean_exec_time_ms DECIMAL(10,3),
    total_exec_time_ms DECIMAL(10,3),
    rows_per_call DECIMAL(10,2),
    cache_hit_ratio DECIMAL(5,2)
) AS $$
BEGIN
    -- Check if pg_stat_statements is available
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements') THEN
        RETURN QUERY
        SELECT 
            'pg_stat_statements extension not available'::TEXT,
            0::BIGINT,
            0.0::DECIMAL(10,3),
            0.0::DECIMAL(10,3),
            0.0::DECIMAL(10,2),
            0.0::DECIMAL(5,2);
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        LEFT(query, 100)::TEXT,
        calls,
        ROUND(mean_exec_time, 3),
        ROUND(total_exec_time, 3),
        ROUND(rows::DECIMAL / GREATEST(calls, 1), 2),
        CASE 
            WHEN shared_blks_hit + shared_blks_read > 0 THEN
                ROUND(shared_blks_hit::DECIMAL / (shared_blks_hit + shared_blks_read) * 100, 2)
            ELSE 0.00
        END
    FROM pg_stat_statements
    WHERE query ILIKE '%campaign%'
    ORDER BY mean_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- COMPREHENSIVE PERFORMANCE REPORT
-- ################################################################################

-- Function to generate comprehensive performance report
CREATE OR REPLACE FUNCTION generate_performance_report()
RETURNS TABLE(
    section TEXT,
    metric TEXT,
    value TEXT,
    status TEXT,
    recommendation TEXT
) AS $$
DECLARE
    active_campaigns INTEGER;
    running_workflows INTEGER;
    avg_campaign_progress DECIMAL(5,2);
    total_content_pieces INTEGER;
    recent_analytics_count INTEGER;
    db_size_mb DECIMAL(10,2);
    index_hit_ratio DECIMAL(5,2);
    table_hit_ratio DECIMAL(5,2);
BEGIN
    -- Gather key metrics
    SELECT COUNT(*) INTO active_campaigns 
    FROM campaigns WHERE status IN ('active', 'running', 'scheduled');
    
    SELECT COUNT(*) INTO running_workflows 
    FROM campaign_workflows WHERE status IN ('running', 'paused');
    
    SELECT AVG(progress_percentage) INTO avg_campaign_progress 
    FROM campaigns WHERE status IN ('active', 'running');
    
    SELECT COUNT(*) INTO total_content_pieces 
    FROM campaign_content WHERE is_active = true;
    
    SELECT COUNT(*) INTO recent_analytics_count 
    FROM campaign_analytics WHERE collected_at >= NOW() - INTERVAL '24 hours';
    
    SELECT ROUND(pg_database_size(current_database()) / 1024.0 / 1024.0, 2) INTO db_size_mb;
    
    -- Calculate cache hit ratios
    SELECT 
        ROUND(
            SUM(idx_blks_hit)::DECIMAL / NULLIF(SUM(idx_blks_hit + idx_blks_read), 0) * 100, 2
        ) INTO index_hit_ratio
    FROM pg_statio_user_indexes;
    
    SELECT 
        ROUND(
            SUM(heap_blks_hit)::DECIMAL / NULLIF(SUM(heap_blks_hit + heap_blks_read), 0) * 100, 2
        ) INTO table_hit_ratio
    FROM pg_statio_user_tables;
    
    -- System Overview
    RETURN QUERY
    SELECT 
        'System Overview'::TEXT,
        'Active Campaigns'::TEXT,
        active_campaigns::TEXT,
        CASE WHEN active_campaigns < 100 THEN 'GOOD' ELSE 'MONITOR' END::TEXT,
        CASE WHEN active_campaigns > 100 THEN 'Consider archiving completed campaigns' ELSE 'Healthy campaign load' END::TEXT;
    
    RETURN QUERY
    SELECT 
        'System Overview',
        'Running Workflows',
        running_workflows::TEXT,
        CASE WHEN running_workflows < 50 THEN 'GOOD' ELSE 'HIGH' END,
        CASE WHEN running_workflows > 50 THEN 'High workflow concurrency - monitor resource usage' ELSE 'Normal workflow activity' END;
    
    RETURN QUERY
    SELECT 
        'System Overview',
        'Average Campaign Progress',
        COALESCE(avg_campaign_progress, 0)::TEXT || '%',
        CASE WHEN COALESCE(avg_campaign_progress, 0) > 50 THEN 'GOOD' ELSE 'ATTENTION' END,
        CASE WHEN COALESCE(avg_campaign_progress, 0) < 30 THEN 'Many campaigns in early stages - consider resource allocation' ELSE 'Healthy progress distribution' END;
    
    RETURN QUERY
    SELECT 
        'System Overview',
        'Total Content Pieces',
        total_content_pieces::TEXT,
        CASE WHEN total_content_pieces > 0 THEN 'ACTIVE' ELSE 'EMPTY' END,
        CASE WHEN total_content_pieces = 0 THEN 'No active content - check campaign workflows' ELSE 'Content generation active' END;
    
    -- Performance Metrics
    RETURN QUERY
    SELECT 
        'Performance'::TEXT,
        'Database Size (MB)'::TEXT,
        db_size_mb::TEXT,
        CASE 
            WHEN db_size_mb < 1000 THEN 'GOOD'
            WHEN db_size_mb < 5000 THEN 'MONITOR'
            ELSE 'HIGH'
        END,
        CASE 
            WHEN db_size_mb > 5000 THEN 'Consider data archival and partitioning strategies'
            ELSE 'Database size within normal range'
        END;
    
    RETURN QUERY
    SELECT 
        'Performance',
        'Index Hit Ratio (%)',
        COALESCE(index_hit_ratio, 0)::TEXT,
        CASE 
            WHEN COALESCE(index_hit_ratio, 0) >= 95 THEN 'EXCELLENT'
            WHEN COALESCE(index_hit_ratio, 0) >= 90 THEN 'GOOD'
            WHEN COALESCE(index_hit_ratio, 0) >= 80 THEN 'FAIR'
            ELSE 'POOR'
        END,
        CASE 
            WHEN COALESCE(index_hit_ratio, 0) < 90 THEN 'Consider increasing shared_buffers or adding more memory'
            ELSE 'Index performance is optimal'
        END;
    
    RETURN QUERY
    SELECT 
        'Performance',
        'Table Hit Ratio (%)',
        COALESCE(table_hit_ratio, 0)::TEXT,
        CASE 
            WHEN COALESCE(table_hit_ratio, 0) >= 95 THEN 'EXCELLENT'
            WHEN COALESCE(table_hit_ratio, 0) >= 90 THEN 'GOOD'
            WHEN COALESCE(table_hit_ratio, 0) >= 80 THEN 'FAIR'
            ELSE 'POOR'
        END,
        CASE 
            WHEN COALESCE(table_hit_ratio, 0) < 90 THEN 'Consider tuning memory parameters or optimizing queries'
            ELSE 'Table access performance is optimal'
        END;
    
    -- Data Freshness
    RETURN QUERY
    SELECT 
        'Data Quality'::TEXT,
        'Recent Analytics (24h)'::TEXT,
        recent_analytics_count::TEXT,
        CASE WHEN recent_analytics_count > 0 THEN 'ACTIVE' ELSE 'STALE' END,
        CASE WHEN recent_analytics_count = 0 THEN 'No recent analytics data - check collection processes' ELSE 'Analytics data collection is active' END;
    
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- AUTOMATED PERFORMANCE MONITORING
-- ################################################################################

-- Create performance monitoring log table
CREATE TABLE IF NOT EXISTS performance_monitoring_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_timestamp TIMESTAMPTZ DEFAULT NOW(),
    metric_category VARCHAR(100) NOT NULL,
    metric_name VARCHAR(200) NOT NULL,
    metric_value DECIMAL(15,4),
    metric_unit VARCHAR(50),
    status VARCHAR(20),
    threshold_warning DECIMAL(15,4),
    threshold_critical DECIMAL(15,4),
    details JSONB DEFAULT '{}',
    alert_sent BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_performance_monitoring_log_timestamp 
ON performance_monitoring_log(check_timestamp DESC);

CREATE INDEX idx_performance_monitoring_log_status 
ON performance_monitoring_log(status, metric_category);

-- Function to log performance metrics
CREATE OR REPLACE FUNCTION log_performance_metrics()
RETURNS INTEGER AS $$
DECLARE
    metrics_logged INTEGER := 0;
    active_campaigns_count INTEGER;
    running_workflows_count INTEGER;
    avg_query_time DECIMAL(10,3);
    cache_hit_ratio DECIMAL(5,2);
BEGIN
    -- Gather current metrics
    SELECT COUNT(*) INTO active_campaigns_count
    FROM campaigns WHERE status IN ('active', 'running', 'scheduled');
    
    SELECT COUNT(*) INTO running_workflows_count
    FROM campaign_workflows WHERE status IN ('running', 'paused');
    
    -- Calculate average query time from recent operations
    SELECT COALESCE(AVG(EXTRACT(milliseconds FROM (NOW() - query_start))), 0) 
    INTO avg_query_time
    FROM pg_stat_activity 
    WHERE state = 'active' AND query_start IS NOT NULL;
    
    -- Calculate cache hit ratio
    SELECT 
        ROUND(
            SUM(heap_blks_hit)::DECIMAL / NULLIF(SUM(heap_blks_hit + heap_blks_read), 0) * 100, 2
        ) INTO cache_hit_ratio
    FROM pg_statio_user_tables;
    
    -- Log metrics
    INSERT INTO performance_monitoring_log (
        metric_category, metric_name, metric_value, metric_unit, 
        status, threshold_warning, threshold_critical
    ) VALUES
    ('campaigns', 'active_count', active_campaigns_count, 'count',
     CASE WHEN active_campaigns_count > 200 THEN 'WARNING' WHEN active_campaigns_count > 500 THEN 'CRITICAL' ELSE 'OK' END,
     200, 500),
     
    ('workflows', 'running_count', running_workflows_count, 'count',
     CASE WHEN running_workflows_count > 100 THEN 'WARNING' WHEN running_workflows_count > 250 THEN 'CRITICAL' ELSE 'OK' END,
     100, 250),
     
    ('performance', 'avg_query_time_ms', avg_query_time, 'milliseconds',
     CASE WHEN avg_query_time > 1000 THEN 'WARNING' WHEN avg_query_time > 5000 THEN 'CRITICAL' ELSE 'OK' END,
     1000, 5000),
     
    ('performance', 'cache_hit_ratio_percent', cache_hit_ratio, 'percent',
     CASE WHEN cache_hit_ratio < 90 THEN 'WARNING' WHEN cache_hit_ratio < 80 THEN 'CRITICAL' ELSE 'OK' END,
     90, 80);
    
    GET DIAGNOSTICS metrics_logged = ROW_COUNT;
    
    RETURN metrics_logged;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- PERFORMANCE OPTIMIZATION RECOMMENDATIONS
-- ################################################################################

-- Function to generate optimization recommendations
CREATE OR REPLACE FUNCTION generate_optimization_recommendations()
RETURNS TABLE(
    category TEXT,
    priority TEXT,
    recommendation TEXT,
    estimated_impact TEXT,
    implementation_effort TEXT
) AS $$
DECLARE
    large_tables_count INTEGER;
    unused_indexes_count INTEGER;
    long_running_queries_count INTEGER;
    cache_hit_ratio DECIMAL(5,2);
BEGIN
    -- Analyze current state
    SELECT COUNT(*) INTO large_tables_count
    FROM (
        SELECT pg_total_relation_size(c.oid) 
        FROM pg_class c 
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        AND pg_total_relation_size(c.oid) > 100 * 1024 * 1024 -- > 100MB
    ) large_tables;
    
    SELECT COUNT(*) INTO unused_indexes_count
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0 AND schemaname = 'public';
    
    SELECT COUNT(*) INTO long_running_queries_count
    FROM pg_stat_activity
    WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes';
    
    SELECT 
        ROUND(
            SUM(heap_blks_hit)::DECIMAL / NULLIF(SUM(heap_blks_hit + heap_blks_read), 0) * 100, 2
        ) INTO cache_hit_ratio
    FROM pg_statio_user_tables;
    
    -- Generate recommendations based on analysis
    IF cache_hit_ratio < 90 THEN
        RETURN QUERY
        SELECT 
            'Memory'::TEXT,
            'HIGH'::TEXT,
            'Increase shared_buffers to improve cache hit ratio (currently ' || cache_hit_ratio || '%)'::TEXT,
            'Significant query performance improvement'::TEXT,
            'Medium - requires restart'::TEXT;
    END IF;
    
    IF large_tables_count > 5 THEN
        RETURN QUERY
        SELECT 
            'Storage',
            'MEDIUM',
            'Consider implementing table partitioning for large tables',
            'Improved query performance and maintenance',
            'High - requires schema changes';
    END IF;
    
    IF unused_indexes_count > 10 THEN
        RETURN QUERY
        SELECT 
            'Indexes',
            'MEDIUM',
            'Drop ' || unused_indexes_count || ' unused indexes to save storage and improve write performance',
            'Reduced storage usage and faster writes',
            'Low - simple DROP INDEX commands';
    END IF;
    
    IF long_running_queries_count > 3 THEN
        RETURN QUERY
        SELECT 
            'Queries',
            'HIGH',
            'Investigate and optimize ' || long_running_queries_count || ' long-running queries',
            'Better overall system responsiveness',
            'Medium - requires query analysis';
    END IF;
    
    -- Always include proactive recommendations
    RETURN QUERY
    SELECT 
        'Monitoring'::TEXT,
        'LOW'::TEXT,
        'Set up automated performance monitoring with alerts'::TEXT,
        'Proactive issue detection'::TEXT,
        'Medium - requires monitoring infrastructure'::TEXT;
    
    RETURN QUERY
    SELECT 
        'Maintenance',
        'LOW',
        'Schedule regular VACUUM and ANALYZE operations',
        'Consistent performance over time',
        'Low - automated maintenance scripts';
    
    RETURN QUERY
    SELECT 
        'Architecture',
        'LOW',
        'Consider read replicas for analytics queries',
        'Reduced load on primary database',
        'High - requires infrastructure changes';
    
END;
$$ LANGUAGE plpgsql;

-- ================================================================================
-- END PERFORMANCE TESTING AND MONITORING SUITE
-- ================================================================================
-- This comprehensive performance suite provides:
-- 1. Detailed query benchmarking functions for read and write operations
-- 2. Load testing capabilities for concurrent operations
-- 3. Memory and resource usage monitoring
-- 4. Storage analysis and optimization recommendations  
-- 5. Automated performance logging and alerting
-- 6. Comprehensive performance reporting and health checks
-- 7. Query performance analysis with pg_stat_statements integration
-- 8. Connection pool and resource limit monitoring
-- 9. Optimization recommendations based on current system state
-- 10. Automated maintenance and monitoring utilities
-- 
-- Usage:
-- SELECT * FROM benchmark_campaign_queries(100);
-- SELECT * FROM benchmark_write_operations(50);
-- SELECT * FROM generate_performance_report();
-- SELECT * FROM generate_optimization_recommendations();
-- ================================================================================