-- ================================================================================
-- ADVANCED DATABASE FUNCTIONS AND TRIGGERS - CAMPAIGN ORCHESTRATION
-- ================================================================================
-- This file contains sophisticated database functions, triggers, and procedures
-- for the campaign-centric architecture with orchestrated workflows
-- Generated: 2024-12-18 | Version: 1.0 | Phase: Foundation
-- ================================================================================

-- ################################################################################
-- CAMPAIGN MANAGEMENT FUNCTIONS
-- ################################################################################

-- Function to calculate campaign progress based on workflow completion
CREATE OR REPLACE FUNCTION calculate_campaign_progress(campaign_id_param UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    total_workflows INTEGER;
    completed_workflows INTEGER;
    running_workflows INTEGER;
    total_content INTEGER;
    published_content INTEGER;
    progress_percentage DECIMAL(5,2);
BEGIN
    -- Count workflows
    SELECT 
        COUNT(*),
        COUNT(*) FILTER (WHERE status = 'completed'),
        COUNT(*) FILTER (WHERE status IN ('running', 'paused'))
    INTO total_workflows, completed_workflows, running_workflows
    FROM campaign_workflows 
    WHERE campaign_id = campaign_id_param;
    
    -- Count content pieces
    SELECT 
        COUNT(*),
        COUNT(*) FILTER (WHERE status = 'published')
    INTO total_content, published_content
    FROM campaign_content 
    WHERE campaign_id = campaign_id_param AND is_active = true;
    
    -- Calculate progress based on multiple factors
    IF total_workflows = 0 AND total_content = 0 THEN
        progress_percentage := 0.00;
    ELSIF total_workflows > 0 AND total_content > 0 THEN
        -- Weighted calculation: 60% workflow completion + 40% content publication
        progress_percentage := 
            (completed_workflows::DECIMAL / total_workflows * 60) +
            (published_content::DECIMAL / total_content * 40);
    ELSIF total_workflows > 0 THEN
        -- Only workflows exist
        progress_percentage := completed_workflows::DECIMAL / total_workflows * 100;
    ELSE
        -- Only content exists
        progress_percentage := published_content::DECIMAL / total_content * 100;
    END IF;
    
    -- Add partial credit for running workflows
    IF running_workflows > 0 AND total_workflows > 0 THEN
        progress_percentage := progress_percentage + (running_workflows::DECIMAL / total_workflows * 15);
    END IF;
    
    -- Ensure progress doesn't exceed 100%
    progress_percentage := LEAST(progress_percentage, 100.00);
    
    RETURN progress_percentage;
END;
$$ LANGUAGE plpgsql;

-- Function to update campaign status based on progress and deadlines
CREATE OR REPLACE FUNCTION update_campaign_status(campaign_id_param UUID)
RETURNS TEXT AS $$
DECLARE
    current_status TEXT;
    current_progress DECIMAL(5,2);
    deadline_date TIMESTAMPTZ;
    scheduled_start_date TIMESTAMPTZ;
    has_active_workflows BOOLEAN;
    new_status TEXT;
BEGIN
    -- Get current campaign data
    SELECT status, progress_percentage, deadline, scheduled_start
    INTO current_status, current_progress, deadline_date, scheduled_start_date
    FROM campaigns 
    WHERE id = campaign_id_param;
    
    -- Check if there are active workflows
    SELECT COUNT(*) > 0 
    INTO has_active_workflows
    FROM campaign_workflows 
    WHERE campaign_id = campaign_id_param 
    AND status IN ('running', 'paused');
    
    -- Determine new status
    new_status := current_status;
    
    -- Check for completion
    IF current_progress >= 100.00 THEN
        new_status := 'completed';
    -- Check for overdue campaigns
    ELSIF deadline_date IS NOT NULL AND NOW() > deadline_date AND current_status NOT IN ('completed', 'cancelled') THEN
        new_status := 'overdue';
    -- Check for active campaigns
    ELSIF has_active_workflows AND current_status = 'scheduled' THEN
        new_status := 'active';
    -- Check for scheduled campaigns
    ELSIF scheduled_start_date IS NOT NULL AND NOW() >= scheduled_start_date AND current_status = 'draft' THEN
        new_status := 'scheduled';
    END IF;
    
    -- Update campaign status and progress
    UPDATE campaigns 
    SET 
        status = new_status,
        progress_percentage = current_progress,
        actual_start = CASE 
            WHEN new_status = 'active' AND actual_start IS NULL THEN NOW()
            ELSE actual_start 
        END,
        actual_completion = CASE 
            WHEN new_status = 'completed' AND actual_completion IS NULL THEN NOW()
            ELSE actual_completion 
        END,
        updated_at = NOW()
    WHERE id = campaign_id_param;
    
    RETURN new_status;
END;
$$ LANGUAGE plpgsql;

-- Function to get campaign performance summary
CREATE OR REPLACE FUNCTION get_campaign_performance_summary(campaign_id_param UUID)
RETURNS JSON AS $$
DECLARE
    performance_data JSON;
BEGIN
    SELECT json_build_object(
        'campaign_id', campaign_id_param,
        'total_content', (
            SELECT COUNT(*) FROM campaign_content 
            WHERE campaign_id = campaign_id_param AND is_active = true
        ),
        'published_content', (
            SELECT COUNT(*) FROM campaign_content 
            WHERE campaign_id = campaign_id_param AND status = 'published' AND is_active = true
        ),
        'total_workflows', (
            SELECT COUNT(*) FROM campaign_workflows 
            WHERE campaign_id = campaign_id_param
        ),
        'completed_workflows', (
            SELECT COUNT(*) FROM campaign_workflows 
            WHERE campaign_id = campaign_id_param AND status = 'completed'
        ),
        'total_views', COALESCE((
            SELECT SUM(views) FROM campaign_analytics 
            WHERE campaign_id = campaign_id_param
        ), 0),
        'total_conversions', COALESCE((
            SELECT SUM(conversions) FROM campaign_analytics 
            WHERE campaign_id = campaign_id_param
        ), 0),
        'total_revenue', COALESCE((
            SELECT SUM(revenue_generated) FROM campaign_analytics 
            WHERE campaign_id = campaign_id_param
        ), 0.00),
        'average_content_quality', COALESCE((
            SELECT AVG(quality_score) FROM campaign_content 
            WHERE campaign_id = campaign_id_param AND quality_score IS NOT NULL
        ), 0.00),
        'calendar_events', (
            SELECT COUNT(*) FROM campaign_calendar 
            WHERE campaign_id = campaign_id_param
        ),
        'overdue_events', (
            SELECT COUNT(*) FROM campaign_calendar 
            WHERE campaign_id = campaign_id_param 
            AND scheduled_datetime < NOW() 
            AND status = 'scheduled'
        )
    ) INTO performance_data;
    
    RETURN performance_data;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- WORKFLOW MANAGEMENT FUNCTIONS
-- ################################################################################

-- Function to validate workflow step dependencies
CREATE OR REPLACE FUNCTION validate_workflow_dependencies(workflow_id_param UUID)
RETURNS BOOLEAN AS $$
DECLARE
    step_record RECORD;
    dependency_name TEXT;
    dependency_exists BOOLEAN;
BEGIN
    -- Check each step's dependencies
    FOR step_record IN 
        SELECT id, step_name, dependencies 
        FROM campaign_workflow_steps 
        WHERE workflow_id = workflow_id_param
    LOOP
        -- Check each dependency
        FOR dependency_name IN 
            SELECT jsonb_array_elements_text(step_record.dependencies)
        LOOP
            SELECT EXISTS(
                SELECT 1 FROM campaign_workflow_steps 
                WHERE workflow_id = workflow_id_param 
                AND step_name = dependency_name
            ) INTO dependency_exists;
            
            IF NOT dependency_exists THEN
                RAISE EXCEPTION 'Step % has invalid dependency: %', step_record.step_name, dependency_name;
            END IF;
        END LOOP;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to get next executable workflow steps
CREATE OR REPLACE FUNCTION get_next_executable_steps(workflow_id_param UUID)
RETURNS TABLE(step_id UUID, step_name TEXT, step_order INTEGER) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cws.id,
        cws.step_name,
        cws.step_order
    FROM campaign_workflow_steps cws
    WHERE cws.workflow_id = workflow_id_param
    AND cws.status = 'pending'
    AND NOT EXISTS (
        -- Check if any dependencies are not completed
        SELECT 1 
        FROM jsonb_array_elements_text(cws.dependencies) dep_name
        WHERE NOT EXISTS (
            SELECT 1 
            FROM campaign_workflow_steps dep_step
            WHERE dep_step.workflow_id = workflow_id_param
            AND dep_step.step_name = dep_name
            AND dep_step.status = 'completed'
        )
    )
    ORDER BY cws.step_order;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate workflow execution metrics
CREATE OR REPLACE FUNCTION calculate_workflow_metrics(workflow_id_param UUID)
RETURNS JSON AS $$
DECLARE
    metrics_data JSON;
BEGIN
    SELECT json_build_object(
        'workflow_id', workflow_id_param,
        'total_steps', (
            SELECT COUNT(*) FROM campaign_workflow_steps 
            WHERE workflow_id = workflow_id_param
        ),
        'completed_steps', (
            SELECT COUNT(*) FROM campaign_workflow_steps 
            WHERE workflow_id = workflow_id_param AND status = 'completed'
        ),
        'failed_steps', (
            SELECT COUNT(*) FROM campaign_workflow_steps 
            WHERE workflow_id = workflow_id_param AND status = 'failed'
        ),
        'total_execution_time_ms', COALESCE((
            SELECT SUM(execution_time_ms) FROM campaign_workflow_steps 
            WHERE workflow_id = workflow_id_param AND execution_time_ms IS NOT NULL
        ), 0),
        'average_step_time_ms', COALESCE((
            SELECT AVG(execution_time_ms) FROM campaign_workflow_steps 
            WHERE workflow_id = workflow_id_param AND execution_time_ms IS NOT NULL
        ), 0),
        'total_cost_usd', COALESCE((
            SELECT SUM(estimated_cost_usd) FROM agent_orchestration_performance 
            WHERE workflow_id = workflow_id_param
        ), 0.00),
        'agent_performance', (
            SELECT json_object_agg(
                agent_name,
                json_build_object(
                    'executions', COUNT(*),
                    'avg_duration_ms', AVG(duration_ms),
                    'success_rate', AVG(CASE WHEN step.status = 'completed' THEN 1.0 ELSE 0.0 END),
                    'total_cost', SUM(estimated_cost_usd)
                )
            )
            FROM agent_orchestration_performance aop
            JOIN campaign_workflow_steps step ON aop.step_id = step.id
            WHERE aop.workflow_id = workflow_id_param
            GROUP BY agent_name
        )
    ) INTO metrics_data;
    
    RETURN metrics_data;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- CONTENT MANAGEMENT FUNCTIONS
-- ################################################################################

-- Function to calculate content quality score
CREATE OR REPLACE FUNCTION calculate_content_quality_score(
    content_id_param UUID,
    word_count_param INTEGER DEFAULT NULL,
    seo_score_param DECIMAL DEFAULT NULL,
    readability_score_param DECIMAL DEFAULT NULL,
    engagement_metrics_param JSON DEFAULT NULL
) RETURNS DECIMAL(4,2) AS $$
DECLARE
    quality_score DECIMAL(4,2) := 0.00;
    word_count_score DECIMAL(4,2) := 0.00;
    seo_score_normalized DECIMAL(4,2) := 0.00;
    readability_score_normalized DECIMAL(4,2) := 0.00;
    engagement_score DECIMAL(4,2) := 0.00;
    content_type_val TEXT;
BEGIN
    -- Get content type for type-specific scoring
    SELECT content_type INTO content_type_val
    FROM campaign_content 
    WHERE id = content_id_param;
    
    -- Calculate word count score based on content type
    IF word_count_param IS NOT NULL THEN
        CASE content_type_val
            WHEN 'blog_post' THEN
                word_count_score := LEAST(word_count_param / 20.0, 100.00); -- Target: 2000+ words
            WHEN 'social_post' THEN
                word_count_score := CASE 
                    WHEN word_count_param BETWEEN 50 AND 150 THEN 100.00
                    ELSE GREATEST(0, 100 - ABS(word_count_param - 100) * 0.5)
                END;
            ELSE
                word_count_score := LEAST(word_count_param / 10.0, 100.00); -- Default scoring
        END CASE;
    END IF;
    
    -- Normalize SEO score (assuming input is 0-100)
    IF seo_score_param IS NOT NULL THEN
        seo_score_normalized := LEAST(seo_score_param, 100.00);
    END IF;
    
    -- Normalize readability score (assuming input is 0-100)
    IF readability_score_param IS NOT NULL THEN
        readability_score_normalized := LEAST(readability_score_param, 100.00);
    END IF;
    
    -- Calculate engagement score from metrics
    IF engagement_metrics_param IS NOT NULL THEN
        engagement_score := LEAST(
            COALESCE((engagement_metrics_param->>'engagement_rate')::DECIMAL * 1000, 0) + 
            COALESCE((engagement_metrics_param->>'click_rate')::DECIMAL * 2000, 0) +
            COALESCE(LN(GREATEST((engagement_metrics_param->>'views')::INTEGER, 1)) * 5, 0),
            100.00
        );
    END IF;
    
    -- Calculate weighted quality score
    quality_score := (
        word_count_score * 0.20 +           -- 20% word count appropriateness
        seo_score_normalized * 0.30 +       -- 30% SEO optimization
        readability_score_normalized * 0.25 + -- 25% readability
        engagement_score * 0.25              -- 25% engagement performance
    );
    
    RETURN LEAST(quality_score, 100.00);
END;
$$ LANGUAGE plpgsql;

-- Function to generate content recommendations
CREATE OR REPLACE FUNCTION generate_content_recommendations(campaign_id_param UUID)
RETURNS JSON AS $$
DECLARE
    recommendations JSON;
    avg_quality DECIMAL(4,2);
    low_performing_content INTEGER;
    missing_platforms TEXT[];
BEGIN
    -- Calculate average quality for campaign
    SELECT AVG(quality_score) INTO avg_quality
    FROM campaign_content 
    WHERE campaign_id = campaign_id_param AND quality_score IS NOT NULL;
    
    -- Count low-performing content
    SELECT COUNT(*) INTO low_performing_content
    FROM campaign_content 
    WHERE campaign_id = campaign_id_param 
    AND quality_score < 70.00;
    
    -- Find missing platforms
    WITH campaign_platforms AS (
        SELECT DISTINCT platform 
        FROM campaign_content 
        WHERE campaign_id = campaign_id_param
    ),
    target_platforms AS (
        SELECT UNNEST(ARRAY['website', 'linkedin', 'twitter', 'facebook', 'instagram']) as platform
    )
    SELECT ARRAY_AGG(tp.platform) INTO missing_platforms
    FROM target_platforms tp
    LEFT JOIN campaign_platforms cp ON tp.platform = cp.platform
    WHERE cp.platform IS NULL;
    
    -- Build recommendations
    SELECT json_build_object(
        'campaign_id', campaign_id_param,
        'average_quality_score', COALESCE(avg_quality, 0.00),
        'recommendations', json_build_array(
            CASE 
                WHEN avg_quality < 75.00 THEN 
                    json_build_object(
                        'type', 'quality_improvement',
                        'priority', 'high',
                        'message', 'Campaign content quality is below target (75+). Focus on SEO optimization and readability.',
                        'affected_content_count', low_performing_content
                    )
                ELSE NULL
            END,
            CASE 
                WHEN array_length(missing_platforms, 1) > 0 THEN
                    json_build_object(
                        'type', 'platform_expansion',
                        'priority', 'medium',
                        'message', 'Consider expanding to additional platforms for better reach.',
                        'missing_platforms', missing_platforms
                    )
                ELSE NULL
            END,
            CASE 
                WHEN (SELECT COUNT(*) FROM campaign_content WHERE campaign_id = campaign_id_param AND status = 'draft') > 5 THEN
                    json_build_object(
                        'type', 'content_review',
                        'priority', 'medium',
                        'message', 'Multiple draft content pieces need review and publishing.',
                        'draft_count', (SELECT COUNT(*) FROM campaign_content WHERE campaign_id = campaign_id_param AND status = 'draft')
                    )
                ELSE NULL
            END
        )
    ) INTO recommendations;
    
    RETURN recommendations;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- ADVANCED TRIGGER FUNCTIONS
-- ################################################################################

-- Trigger function to automatically update campaign progress
CREATE OR REPLACE FUNCTION trigger_update_campaign_progress()
RETURNS TRIGGER AS $$
DECLARE
    affected_campaign_id UUID;
    new_progress DECIMAL(5,2);
    new_status TEXT;
BEGIN
    -- Get campaign ID from the affected row
    IF TG_TABLE_NAME = 'campaign_workflows' THEN
        affected_campaign_id := COALESCE(NEW.campaign_id, OLD.campaign_id);
    ELSIF TG_TABLE_NAME = 'campaign_content' THEN
        affected_campaign_id := COALESCE(NEW.campaign_id, OLD.campaign_id);
    ELSIF TG_TABLE_NAME = 'campaign_workflow_steps' THEN
        SELECT cw.campaign_id INTO affected_campaign_id
        FROM campaign_workflows cw
        WHERE cw.id = COALESCE(NEW.workflow_id, OLD.workflow_id);
    END IF;
    
    -- Calculate new progress
    new_progress := calculate_campaign_progress(affected_campaign_id);
    
    -- Update campaign status
    new_status := update_campaign_status(affected_campaign_id);
    
    -- Log the update
    INSERT INTO campaign_progress_log (
        campaign_id, 
        previous_progress,
        new_progress,
        previous_status,
        new_status,
        trigger_table,
        trigger_operation,
        updated_at
    ) VALUES (
        affected_campaign_id,
        (SELECT progress_percentage FROM campaigns WHERE id = affected_campaign_id),
        new_progress,
        (SELECT status FROM campaigns WHERE id = affected_campaign_id),
        new_status,
        TG_TABLE_NAME,
        TG_OP,
        NOW()
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger function for workflow step dependency validation
CREATE OR REPLACE FUNCTION trigger_validate_step_execution()
RETURNS TRIGGER AS $$
DECLARE
    dependency_name TEXT;
    dependency_completed BOOLEAN;
BEGIN
    -- Only validate when transitioning to 'running' status
    IF NEW.status = 'running' AND (OLD.status IS NULL OR OLD.status != 'running') THEN
        -- Check each dependency
        FOR dependency_name IN 
            SELECT jsonb_array_elements_text(NEW.dependencies)
        LOOP
            SELECT EXISTS(
                SELECT 1 FROM campaign_workflow_steps 
                WHERE workflow_id = NEW.workflow_id 
                AND step_name = dependency_name
                AND status = 'completed'
            ) INTO dependency_completed;
            
            IF NOT dependency_completed THEN
                RAISE EXCEPTION 'Cannot start step %: dependency % not completed', 
                    NEW.step_name, dependency_name;
            END IF;
        END LOOP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger function for automatic content quality scoring
CREATE OR REPLACE FUNCTION trigger_calculate_content_quality()
RETURNS TRIGGER AS $$
DECLARE
    calculated_quality DECIMAL(4,2);
BEGIN
    -- Calculate quality when content is created or updated
    IF (NEW.word_count IS NOT NULL OR NEW.seo_score IS NOT NULL OR 
        NEW.readability_score IS NOT NULL OR NEW.engagement_metrics IS NOT NULL) THEN
        
        calculated_quality := calculate_content_quality_score(
            NEW.id,
            NEW.word_count,
            NEW.seo_score,
            NEW.readability_score,
            NEW.engagement_metrics
        );
        
        NEW.quality_score := calculated_quality;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger function for orchestrator performance updates
CREATE OR REPLACE FUNCTION trigger_update_orchestrator_performance()
RETURNS TRIGGER AS $$
DECLARE
    workflow_duration_ms INTEGER;
BEGIN
    -- Update orchestrator statistics when workflow completes
    IF NEW.status IN ('completed', 'failed') AND 
       (OLD.status IS NULL OR OLD.status NOT IN ('completed', 'failed')) THEN
        
        -- Calculate workflow duration
        IF NEW.started_at IS NOT NULL AND NEW.completed_at IS NOT NULL THEN
            workflow_duration_ms := EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at)) * 1000;
        END IF;
        
        UPDATE campaign_orchestrators 
        SET 
            total_executions = total_executions + 1,
            successful_executions = successful_executions + CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
            average_execution_time_ms = CASE 
                WHEN workflow_duration_ms IS NOT NULL THEN 
                    ((average_execution_time_ms * total_executions) + workflow_duration_ms) / (total_executions + 1)
                ELSE average_execution_time_ms
            END,
            updated_at = NOW()
        WHERE id = NEW.orchestrator_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- CREATE TRIGGERS
-- ################################################################################

-- Campaign progress update triggers
CREATE TRIGGER trigger_campaign_progress_on_workflow_change
    AFTER INSERT OR UPDATE OR DELETE ON campaign_workflows
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_campaign_progress();

CREATE TRIGGER trigger_campaign_progress_on_content_change
    AFTER INSERT OR UPDATE OR DELETE ON campaign_content
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_campaign_progress();

CREATE TRIGGER trigger_campaign_progress_on_step_change
    AFTER UPDATE ON campaign_workflow_steps
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_campaign_progress();

-- Workflow step validation trigger
CREATE TRIGGER trigger_validate_step_dependencies
    BEFORE UPDATE OF status ON campaign_workflow_steps
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_step_execution();

-- Content quality calculation trigger
CREATE TRIGGER trigger_content_quality_calculation
    BEFORE INSERT OR UPDATE ON campaign_content
    FOR EACH ROW
    EXECUTE FUNCTION trigger_calculate_content_quality();

-- Orchestrator performance update trigger
CREATE TRIGGER trigger_orchestrator_performance_update
    AFTER UPDATE OF status ON campaign_workflows
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_orchestrator_performance();

-- ################################################################################
-- UTILITY AND MAINTENANCE FUNCTIONS
-- ################################################################################

-- Function to clean up old performance data
CREATE OR REPLACE FUNCTION cleanup_old_performance_data(
    days_to_keep INTEGER DEFAULT 90
) RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_date TIMESTAMPTZ;
BEGIN
    cutoff_date := NOW() - INTERVAL '1 day' * days_to_keep;
    
    -- Clean up old agent performance records
    WITH deleted AS (
        DELETE FROM agent_orchestration_performance 
        WHERE recorded_at < cutoff_date
        AND workflow_id IN (
            SELECT id FROM campaign_workflows 
            WHERE status IN ('completed', 'failed', 'cancelled')
            AND completed_at < cutoff_date
        )
        RETURNING *
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    -- Log cleanup operation
    INSERT INTO maintenance_log (
        operation_type,
        affected_records,
        cutoff_date,
        executed_at
    ) VALUES (
        'performance_data_cleanup',
        deleted_count,
        cutoff_date,
        NOW()
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze campaign performance trends
CREATE OR REPLACE FUNCTION analyze_campaign_performance_trends(
    date_range_days INTEGER DEFAULT 30
) RETURNS TABLE(
    metric_name TEXT,
    current_period_value DECIMAL,
    previous_period_value DECIMAL,
    change_percentage DECIMAL,
    trend_direction TEXT
) AS $$
DECLARE
    current_start_date DATE;
    current_end_date DATE;
    previous_start_date DATE;
    previous_end_date DATE;
BEGIN
    -- Define date ranges
    current_end_date := CURRENT_DATE;
    current_start_date := current_end_date - INTERVAL '1 day' * date_range_days;
    previous_end_date := current_start_date - INTERVAL '1 day';
    previous_start_date := previous_end_date - INTERVAL '1 day' * date_range_days;
    
    RETURN QUERY
    WITH current_metrics AS (
        SELECT
            COUNT(DISTINCT c.id) as active_campaigns,
            COUNT(DISTINCT cc.id) as content_pieces,
            COALESCE(AVG(cc.quality_score), 0) as avg_quality,
            COALESCE(SUM(ca.views), 0) as total_views,
            COALESCE(SUM(ca.conversions), 0) as total_conversions,
            COALESCE(SUM(ca.revenue_generated), 0) as total_revenue
        FROM campaigns c
        LEFT JOIN campaign_content cc ON c.id = cc.campaign_id
        LEFT JOIN campaign_analytics ca ON c.id = ca.campaign_id
        WHERE c.created_at::DATE BETWEEN current_start_date AND current_end_date
    ),
    previous_metrics AS (
        SELECT
            COUNT(DISTINCT c.id) as active_campaigns,
            COUNT(DISTINCT cc.id) as content_pieces,
            COALESCE(AVG(cc.quality_score), 0) as avg_quality,
            COALESCE(SUM(ca.views), 0) as total_views,
            COALESCE(SUM(ca.conversions), 0) as total_conversions,
            COALESCE(SUM(ca.revenue_generated), 0) as total_revenue
        FROM campaigns c
        LEFT JOIN campaign_content cc ON c.id = cc.campaign_id
        LEFT JOIN campaign_analytics ca ON c.id = ca.campaign_id
        WHERE c.created_at::DATE BETWEEN previous_start_date AND previous_end_date
    )
    SELECT 
        'Active Campaigns'::TEXT,
        cm.active_campaigns::DECIMAL,
        pm.active_campaigns::DECIMAL,
        CASE WHEN pm.active_campaigns = 0 THEN NULL 
             ELSE ((cm.active_campaigns - pm.active_campaigns) * 100.0 / pm.active_campaigns) 
        END,
        CASE WHEN pm.active_campaigns = 0 THEN 'unknown'
             WHEN cm.active_campaigns > pm.active_campaigns THEN 'up'
             WHEN cm.active_campaigns < pm.active_campaigns THEN 'down'
             ELSE 'stable'
        END
    FROM current_metrics cm, previous_metrics pm
    UNION ALL
    SELECT 
        'Content Pieces'::TEXT,
        cm.content_pieces::DECIMAL,
        pm.content_pieces::DECIMAL,
        CASE WHEN pm.content_pieces = 0 THEN NULL 
             ELSE ((cm.content_pieces - pm.content_pieces) * 100.0 / pm.content_pieces) 
        END,
        CASE WHEN pm.content_pieces = 0 THEN 'unknown'
             WHEN cm.content_pieces > pm.content_pieces THEN 'up'
             WHEN cm.content_pieces < pm.content_pieces THEN 'down'
             ELSE 'stable'
        END
    FROM current_metrics cm, previous_metrics pm
    UNION ALL
    SELECT 
        'Average Quality'::TEXT,
        cm.avg_quality,
        pm.avg_quality,
        CASE WHEN pm.avg_quality = 0 THEN NULL 
             ELSE ((cm.avg_quality - pm.avg_quality) * 100.0 / pm.avg_quality) 
        END,
        CASE WHEN pm.avg_quality = 0 THEN 'unknown'
             WHEN cm.avg_quality > pm.avg_quality THEN 'up'
             WHEN cm.avg_quality < pm.avg_quality THEN 'down'
             ELSE 'stable'
        END
    FROM current_metrics cm, previous_metrics pm;
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- LOGGING AND AUDIT TABLES
-- ################################################################################

-- Campaign progress log for audit trail
CREATE TABLE IF NOT EXISTS campaign_progress_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL,
    previous_progress DECIMAL(5,2),
    new_progress DECIMAL(5,2),
    previous_status TEXT,
    new_status TEXT,
    trigger_table TEXT,
    trigger_operation TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_campaign_progress_log_campaign ON campaign_progress_log(campaign_id, updated_at);
CREATE INDEX idx_campaign_progress_log_date ON campaign_progress_log(updated_at);

-- Maintenance operations log
CREATE TABLE IF NOT EXISTS maintenance_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_type TEXT NOT NULL,
    affected_records INTEGER,
    cutoff_date TIMESTAMPTZ,
    execution_duration_ms INTEGER,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    details JSON DEFAULT '{}'
);

CREATE INDEX idx_maintenance_log_type ON maintenance_log(operation_type, executed_at);

-- ================================================================================
-- END ADVANCED FUNCTIONS AND TRIGGERS
-- ================================================================================
-- These functions provide:
-- 1. Automated campaign progress calculation and status updates
-- 2. Workflow dependency validation and step execution control
-- 3. Content quality scoring and recommendations
-- 4. Performance monitoring and trend analysis
-- 5. Maintenance operations and cleanup utilities
-- 6. Comprehensive audit trail and logging
-- ================================================================================