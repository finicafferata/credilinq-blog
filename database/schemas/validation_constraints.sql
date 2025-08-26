-- ================================================================================
-- DATA VALIDATION AND CONSTRAINT SYSTEM - CAMPAIGN ORCHESTRATION
-- ================================================================================
-- This file contains comprehensive data validation rules, constraints, and
-- business logic enforcement for the campaign-centric architecture
-- Generated: 2024-12-18 | Version: 1.0 | Phase: Foundation
-- ================================================================================

-- ################################################################################
-- CAMPAIGN VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure campaign dates are logical
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_date_logic 
CHECK (
    -- Scheduled start must be before deadline
    (scheduled_start IS NULL OR deadline IS NULL OR scheduled_start < deadline) AND
    -- Actual start must be before deadline (with some tolerance)
    (actual_start IS NULL OR deadline IS NULL OR actual_start <= deadline + INTERVAL '1 day') AND
    -- Actual completion should be after actual start
    (actual_completion IS NULL OR actual_start IS NULL OR actual_completion >= actual_start) AND
    -- Estimated completion should be reasonable
    (estimated_completion IS NULL OR created_at IS NULL OR estimated_completion >= created_at)
);

-- Ensure campaign priority is valid
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_priority_valid 
CHECK (priority IN ('low', 'medium', 'high', 'critical', 'urgent'));

-- Ensure campaign progress is within valid range
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_progress_range 
CHECK (progress_percentage >= 0.00 AND progress_percentage <= 100.00);

-- Ensure campaign status transitions are valid
CREATE OR REPLACE FUNCTION validate_campaign_status_transition(
    old_status TEXT, 
    new_status TEXT
) RETURNS BOOLEAN AS $$
BEGIN
    -- Define valid status transitions
    RETURN CASE old_status
        WHEN 'draft' THEN new_status IN ('scheduled', 'active', 'cancelled')
        WHEN 'scheduled' THEN new_status IN ('active', 'paused', 'cancelled')
        WHEN 'active' THEN new_status IN ('paused', 'completed', 'cancelled', 'overdue')
        WHEN 'paused' THEN new_status IN ('active', 'cancelled')
        WHEN 'overdue' THEN new_status IN ('active', 'completed', 'cancelled')
        WHEN 'completed' THEN new_status IN ('archived')  -- Limited transitions from completed
        WHEN 'cancelled' THEN new_status IN ('draft', 'scheduled')  -- Can restart cancelled campaigns
        WHEN 'archived' THEN FALSE  -- No transitions from archived
        ELSE TRUE  -- Allow new statuses for flexibility
    END;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate campaign status transitions
CREATE OR REPLACE FUNCTION trigger_validate_campaign_status()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS NOT NULL AND NEW.status != OLD.status THEN
        IF NOT validate_campaign_status_transition(OLD.status, NEW.status) THEN
            RAISE EXCEPTION 'Invalid campaign status transition from % to %', OLD.status, NEW.status;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_campaign_status_validation
    BEFORE UPDATE OF status ON campaigns
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_campaign_status();

-- ################################################################################
-- WORKFLOW VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure workflow timing is logical
ALTER TABLE campaign_workflows 
ADD CONSTRAINT chk_workflow_timing_logic 
CHECK (
    (completed_at IS NULL OR started_at IS NULL OR completed_at > started_at) AND
    (actual_duration_ms IS NULL OR actual_duration_ms > 0) AND
    (estimated_duration_ms IS NULL OR estimated_duration_ms > 0)
);

-- Ensure workflow retry logic is valid
ALTER TABLE campaign_workflows 
ADD CONSTRAINT chk_workflow_retry_logic 
CHECK (
    retry_count >= 0 AND 
    max_retries >= 0 AND 
    retry_count <= max_retries
);

-- Ensure workflow status is valid
ALTER TABLE campaign_workflows 
ADD CONSTRAINT chk_workflow_status_valid 
CHECK (status IN ('pending', 'running', 'paused', 'completed', 'failed', 'cancelled', 'timeout'));

-- Validate workflow step configuration
CREATE OR REPLACE FUNCTION validate_workflow_step_config(
    step_order_param INTEGER,
    dependencies_param JSONB,
    workflow_id_param UUID
) RETURNS BOOLEAN AS $$
DECLARE
    dep_name TEXT;
    dep_order INTEGER;
BEGIN
    -- Step order must be positive
    IF step_order_param <= 0 THEN
        RETURN FALSE;
    END IF;
    
    -- Validate each dependency
    FOR dep_name IN SELECT jsonb_array_elements_text(dependencies_param)
    LOOP
        -- Check if dependency exists and has lower order
        SELECT step_order INTO dep_order
        FROM campaign_workflow_steps
        WHERE workflow_id = workflow_id_param 
        AND step_name = dep_name;
        
        IF dep_order IS NULL OR dep_order >= step_order_param THEN
            RAISE EXCEPTION 'Invalid dependency %: must exist and have lower step order than %', 
                dep_name, step_order_param;
        END IF;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate workflow step configuration
CREATE OR REPLACE FUNCTION trigger_validate_workflow_step_config()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT validate_workflow_step_config(NEW.step_order, NEW.dependencies, NEW.workflow_id) THEN
        RAISE EXCEPTION 'Invalid workflow step configuration';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_workflow_step_config_validation
    BEFORE INSERT OR UPDATE ON campaign_workflow_steps
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_workflow_step_config();

-- Ensure workflow step timing is logical
ALTER TABLE campaign_workflow_steps 
ADD CONSTRAINT chk_workflow_step_timing 
CHECK (
    (completed_at IS NULL OR started_at IS NULL OR completed_at > started_at) AND
    (execution_time_ms IS NULL OR execution_time_ms >= 0)
);

-- Ensure workflow step status is valid
ALTER TABLE campaign_workflow_steps 
ADD CONSTRAINT chk_workflow_step_status_valid 
CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'retry'));

-- Ensure workflow step retry logic is valid
ALTER TABLE campaign_workflow_steps 
ADD CONSTRAINT chk_workflow_step_retry_logic 
CHECK (
    retry_count >= 0 AND 
    max_retries >= 0 AND 
    retry_count <= max_retries
);

-- ################################################################################
-- CONTENT VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure content has required fields based on type
CREATE OR REPLACE FUNCTION validate_content_requirements(
    content_type_param TEXT,
    title_param TEXT,
    content_markdown_param TEXT,
    platform_param TEXT
) RETURNS BOOLEAN AS $$
BEGIN
    -- Basic validation for all content types
    IF title_param IS NULL OR LENGTH(TRIM(title_param)) = 0 THEN
        RAISE EXCEPTION 'Content title is required';
    END IF;
    
    -- Content type specific validation
    CASE content_type_param
        WHEN 'blog_post' THEN
            IF content_markdown_param IS NULL OR LENGTH(TRIM(content_markdown_param)) < 100 THEN
                RAISE EXCEPTION 'Blog posts must have substantial content (minimum 100 characters)';
            END IF;
            IF platform_param IS NULL THEN
                RAISE EXCEPTION 'Blog posts must specify a platform';
            END IF;
            
        WHEN 'social_post' THEN
            IF content_markdown_param IS NULL OR LENGTH(TRIM(content_markdown_param)) = 0 THEN
                RAISE EXCEPTION 'Social posts must have content';
            END IF;
            IF LENGTH(content_markdown_param) > 2000 THEN
                RAISE EXCEPTION 'Social posts cannot exceed 2000 characters';
            END IF;
            IF platform_param IS NULL THEN
                RAISE EXCEPTION 'Social posts must specify a platform';
            END IF;
            
        WHEN 'email' THEN
            IF content_markdown_param IS NULL OR LENGTH(TRIM(content_markdown_param)) < 50 THEN
                RAISE EXCEPTION 'Email content must be at least 50 characters';
            END IF;
            
        WHEN 'video_script' THEN
            IF content_markdown_param IS NULL OR LENGTH(TRIM(content_markdown_param)) < 200 THEN
                RAISE EXCEPTION 'Video scripts must be at least 200 characters';
            END IF;
    END CASE;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate content requirements
CREATE OR REPLACE FUNCTION trigger_validate_content_requirements()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT validate_content_requirements(
        NEW.content_type, 
        NEW.title, 
        NEW.content_markdown, 
        NEW.platform
    ) THEN
        RAISE EXCEPTION 'Content validation failed';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_content_requirements_validation
    BEFORE INSERT OR UPDATE ON campaign_content
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_content_requirements();

-- Ensure content publishing dates are logical
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_publishing_dates 
CHECK (
    (published_at IS NULL OR published_at >= created_at) AND
    (scheduled_publish_at IS NULL OR scheduled_publish_at >= created_at) AND
    (published_at IS NULL OR scheduled_publish_at IS NULL OR published_at >= scheduled_publish_at - INTERVAL '1 hour')
);

-- Ensure content quality scores are within valid range
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_quality_scores 
CHECK (
    (quality_score IS NULL OR (quality_score >= 0.00 AND quality_score <= 100.00)) AND
    (seo_score IS NULL OR (seo_score >= 0.00 AND seo_score <= 100.00)) AND
    (readability_score IS NULL OR (readability_score >= 0.00 AND readability_score <= 100.00)) AND
    (sentiment_score IS NULL OR (sentiment_score >= -100.00 AND sentiment_score <= 100.00))
);

-- Ensure content metrics are non-negative
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_metrics_non_negative 
CHECK (
    view_count >= 0 AND
    share_count >= 0 AND
    word_count IS NULL OR word_count >= 0 AND
    character_count IS NULL OR character_count >= 0 AND
    reading_time_minutes IS NULL OR reading_time_minutes >= 0
);

-- Ensure content status is valid
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_status_valid 
CHECK (status IN ('draft', 'in_review', 'approved', 'published', 'archived', 'deprecated'));

-- Prevent circular content relationships
ALTER TABLE campaign_content_relationships 
ADD CONSTRAINT chk_content_relationship_no_circular 
CHECK (source_content_id != target_content_id);

-- Ensure content relationship strength is valid
ALTER TABLE campaign_content_relationships 
ADD CONSTRAINT chk_content_relationship_strength 
CHECK (strength >= 0.00 AND strength <= 1.00);

-- ################################################################################
-- CALENDAR AND SCHEDULING CONSTRAINTS
-- ################################################################################

-- Ensure calendar event timing is logical
ALTER TABLE campaign_calendar 
ADD CONSTRAINT chk_calendar_event_timing 
CHECK (
    (actual_completion_datetime IS NULL OR actual_completion_datetime >= scheduled_datetime) AND
    (duration_minutes IS NULL OR duration_minutes > 0) AND
    (completion_percentage >= 0.00 AND completion_percentage <= 100.00)
);

-- Ensure calendar event status is valid
ALTER TABLE campaign_calendar 
ADD CONSTRAINT chk_calendar_event_status_valid 
CHECK (status IN ('scheduled', 'in_progress', 'completed', 'cancelled', 'postponed'));

-- Ensure reminder intervals are positive
ALTER TABLE campaign_calendar 
ADD CONSTRAINT chk_calendar_reminder_intervals 
CHECK (
    reminder_intervals IS NULL OR 
    (SELECT bool_and(interval_val > 0) FROM unnest(reminder_intervals) AS interval_val)
);

-- Validate calendar event type requirements
CREATE OR REPLACE FUNCTION validate_calendar_event_requirements(
    event_type_param TEXT,
    content_id_param UUID,
    workflow_id_param UUID,
    scheduled_datetime_param TIMESTAMPTZ
) RETURNS BOOLEAN AS $$
BEGIN
    -- Validate based on event type
    CASE event_type_param
        WHEN 'content_publish' THEN
            IF content_id_param IS NULL THEN
                RAISE EXCEPTION 'Content publishing events must reference content';
            END IF;
            
        WHEN 'workflow_milestone' THEN
            IF workflow_id_param IS NULL THEN
                RAISE EXCEPTION 'Workflow milestone events must reference workflow';
            END IF;
            
        WHEN 'campaign_deadline' THEN
            -- Campaign deadlines don't require additional references
            NULL;
            
        WHEN 'review_due' THEN
            -- Review events should have content or workflow reference
            IF content_id_param IS NULL AND workflow_id_param IS NULL THEN
                RAISE EXCEPTION 'Review events should reference content or workflow';
            END IF;
    END CASE;
    
    -- Ensure future events are not scheduled too far in advance (1 year limit)
    IF scheduled_datetime_param > NOW() + INTERVAL '1 year' THEN
        RAISE EXCEPTION 'Events cannot be scheduled more than 1 year in advance';
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate calendar event requirements
CREATE OR REPLACE FUNCTION trigger_validate_calendar_event()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT validate_calendar_event_requirements(
        NEW.event_type,
        NEW.content_id,
        NEW.workflow_id,
        NEW.scheduled_datetime
    ) THEN
        RAISE EXCEPTION 'Calendar event validation failed';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_calendar_event_validation
    BEFORE INSERT OR UPDATE ON campaign_calendar
    FOR EACH ROW
    EXECUTE FUNCTION trigger_validate_calendar_event();

-- ################################################################################
-- ANALYTICS VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure analytics metrics are non-negative
ALTER TABLE campaign_analytics 
ADD CONSTRAINT chk_analytics_metrics_non_negative 
CHECK (
    views >= 0 AND
    unique_views >= 0 AND
    clicks >= 0 AND
    likes >= 0 AND
    shares >= 0 AND
    comments >= 0 AND
    reach >= 0 AND
    impressions >= 0 AND
    conversions >= 0 AND
    organic_traffic >= 0 AND
    backlinks_gained >= 0
);

-- Ensure analytics rates are within valid range
ALTER TABLE campaign_analytics 
ADD CONSTRAINT chk_analytics_rates_valid 
CHECK (
    (click_through_rate IS NULL OR (click_through_rate >= 0.0000 AND click_through_rate <= 1.0000)) AND
    (engagement_rate IS NULL OR (engagement_rate >= 0.0000 AND engagement_rate <= 10.0000)) AND  -- Allow up to 1000%
    (conversion_rate IS NULL OR (conversion_rate >= 0.0000 AND conversion_rate <= 1.0000))
);

-- Ensure financial metrics are reasonable
ALTER TABLE campaign_analytics 
ADD CONSTRAINT chk_analytics_financial_reasonable 
CHECK (
    (revenue_generated IS NULL OR revenue_generated >= 0.00) AND
    (cost_per_conversion IS NULL OR cost_per_conversion >= 0.00) AND
    (attribution_value IS NULL OR attribution_value >= 0.00) AND
    (roi_calculated IS NULL OR roi_calculated >= -100.0000)  -- Allow negative ROI but limit
);

-- Ensure unique views don't exceed total views
ALTER TABLE campaign_analytics 
ADD CONSTRAINT chk_analytics_unique_views_logical 
CHECK (unique_views <= views);

-- Ensure measurement period is valid
ALTER TABLE campaign_analytics 
ADD CONSTRAINT chk_analytics_measurement_period_valid 
CHECK (measurement_period IN ('daily', 'weekly', 'monthly', 'quarterly', 'yearly'));

-- Ensure data quality score is within valid range
ALTER TABLE campaign_analytics 
ADD CONSTRAINT chk_analytics_data_quality_valid 
CHECK (data_quality_score IS NULL OR (data_quality_score >= 0.00 AND data_quality_score <= 1.00));

-- ################################################################################
-- AGENT PERFORMANCE VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure agent performance timing is logical
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_performance_timing 
CHECK (
    (completed_at IS NULL OR completed_at > started_at) AND
    (duration_ms IS NULL OR duration_ms >= 0)
);

-- Ensure agent resource usage is reasonable
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_resource_usage_reasonable 
CHECK (
    (memory_used_mb IS NULL OR memory_used_mb >= 0) AND
    (cpu_time_ms IS NULL OR cpu_time_ms >= 0) AND
    (api_calls_made >= 0) AND
    (tokens_input >= 0) AND
    (tokens_output >= 0) AND
    (tokens_total >= tokens_input + tokens_output)  -- Total should be at least sum of input/output
);

-- Ensure agent cost tracking is reasonable
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_cost_reasonable 
CHECK (
    (estimated_cost_usd IS NULL OR estimated_cost_usd >= 0.000000) AND
    (estimated_cost_usd IS NULL OR estimated_cost_usd <= 1000.00)  -- Prevent extreme costs
);

-- Ensure agent quality scores are within valid range
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_quality_scores_valid 
CHECK (
    (output_quality_score IS NULL OR (output_quality_score >= 0.00 AND output_quality_score <= 100.00)) AND
    (task_completion_rate IS NULL OR (task_completion_rate >= 0.0000 AND task_completion_rate <= 1.0000)) AND
    (accuracy_score IS NULL OR (accuracy_score >= 0.00 AND accuracy_score <= 100.00)) AND
    (efficiency_score IS NULL OR (efficiency_score >= 0.00 AND efficiency_score <= 100.00)) AND
    (success_rate IS NULL OR (success_rate >= 0.0000 AND success_rate <= 1.0000))
);

-- Ensure agent error tracking is reasonable
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_error_tracking_reasonable 
CHECK (
    error_count >= 0 AND
    retry_count >= 0 AND
    timeout_count >= 0 AND
    (error_count + timeout_count) >= 0  -- Basic sanity check
);

-- Ensure agent performance benchmarks are reasonable
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_benchmarks_reasonable 
CHECK (
    (baseline_duration_ms IS NULL OR baseline_duration_ms > 0) AND
    (performance_deviation IS NULL OR performance_deviation >= -10.0000) AND  -- Allow up to 1000% deviation
    (percentile_rank IS NULL OR (percentile_rank >= 0.00 AND percentile_rank <= 100.00))
);

-- Ensure agent data size tracking is reasonable
ALTER TABLE agent_orchestration_performance 
ADD CONSTRAINT chk_agent_data_size_reasonable 
CHECK (
    (input_data_size_bytes IS NULL OR input_data_size_bytes >= 0) AND
    (output_data_size_bytes IS NULL OR output_data_size_bytes >= 0) AND
    (input_data_size_bytes IS NULL OR input_data_size_bytes <= 1000000000) AND  -- 1GB limit
    (output_data_size_bytes IS NULL OR output_data_size_bytes <= 1000000000)    -- 1GB limit
);

-- ################################################################################
-- ORCHESTRATOR VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure orchestrator configuration is reasonable
ALTER TABLE campaign_orchestrators 
ADD CONSTRAINT chk_orchestrator_config_reasonable 
CHECK (
    priority >= 1 AND priority <= 10 AND
    max_concurrent_executions > 0 AND max_concurrent_executions <= 100 AND
    timeout_seconds > 0 AND timeout_seconds <= 86400  -- Max 24 hours
);

-- Ensure orchestrator metrics are non-negative
ALTER TABLE campaign_orchestrators 
ADD CONSTRAINT chk_orchestrator_metrics_non_negative 
CHECK (
    total_executions >= 0 AND
    successful_executions >= 0 AND
    successful_executions <= total_executions AND
    average_execution_time_ms >= 0
);

-- Ensure orchestrator status is valid
ALTER TABLE campaign_orchestrators 
ADD CONSTRAINT chk_orchestrator_status_valid 
CHECK (status IN ('active', 'inactive', 'deprecated', 'maintenance'));

-- Ensure orchestrator type is valid
ALTER TABLE campaign_orchestrators 
ADD CONSTRAINT chk_orchestrator_type_valid 
CHECK (orchestrator_type IN (
    'content_creation', 
    'multi_channel_distribution', 
    'performance_optimization', 
    'audience_engagement', 
    'competitive_intelligence', 
    'custom_workflow'
));

-- ################################################################################
-- STRATEGY VALIDATION CONSTRAINTS
-- ################################################################################

-- Ensure strategy configuration is reasonable
ALTER TABLE campaign_strategies 
ADD CONSTRAINT chk_strategy_config_reasonable 
CHECK (
    usage_count >= 0
);

-- Ensure strategy type is valid
ALTER TABLE campaign_strategies 
ADD CONSTRAINT chk_strategy_type_valid 
CHECK (strategy_type IN (
    'thought_leadership',
    'product_marketing',
    'brand_awareness',
    'lead_generation',
    'customer_retention',
    'competitive_positioning',
    'market_education',
    'community_building'
));

-- ################################################################################
-- CROSS-TABLE VALIDATION FUNCTIONS
-- ################################################################################

-- Function to validate campaign-workflow consistency
CREATE OR REPLACE FUNCTION validate_campaign_workflow_consistency()
RETURNS TRIGGER AS $$
DECLARE
    campaign_status TEXT;
    workflow_count INTEGER;
BEGIN
    -- Get campaign status
    SELECT status INTO campaign_status
    FROM campaigns
    WHERE id = NEW.campaign_id;
    
    -- Count active workflows for this campaign
    SELECT COUNT(*) INTO workflow_count
    FROM campaign_workflows
    WHERE campaign_id = NEW.campaign_id
    AND status IN ('running', 'paused');
    
    -- Validate consistency
    IF campaign_status = 'completed' AND NEW.status IN ('pending', 'running') THEN
        RAISE EXCEPTION 'Cannot start new workflows for completed campaigns';
    END IF;
    
    IF campaign_status = 'cancelled' AND NEW.status IN ('pending', 'running') THEN
        RAISE EXCEPTION 'Cannot start workflows for cancelled campaigns';
    END IF;
    
    -- Limit concurrent workflows per campaign
    IF NEW.status = 'running' AND workflow_count >= 5 THEN
        RAISE EXCEPTION 'Campaign cannot have more than 5 concurrent workflows';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_campaign_workflow_consistency
    BEFORE INSERT OR UPDATE ON campaign_workflows
    FOR EACH ROW
    EXECUTE FUNCTION validate_campaign_workflow_consistency();

-- Function to validate content-campaign consistency
CREATE OR REPLACE FUNCTION validate_content_campaign_consistency()
RETURNS TRIGGER AS $$
DECLARE
    campaign_status TEXT;
BEGIN
    -- Get campaign status
    SELECT status INTO campaign_status
    FROM campaigns
    WHERE id = NEW.campaign_id;
    
    -- Validate consistency
    IF campaign_status = 'completed' AND NEW.status = 'draft' THEN
        -- Allow draft content for completed campaigns (for future use)
        NULL;
    ELSIF campaign_status = 'cancelled' AND NEW.status NOT IN ('archived', 'deprecated') THEN
        RAISE EXCEPTION 'Cannot create active content for cancelled campaigns';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_content_campaign_consistency
    BEFORE INSERT OR UPDATE ON campaign_content
    FOR EACH ROW
    EXECUTE FUNCTION validate_content_campaign_consistency();

-- ################################################################################
-- VALIDATION UTILITY FUNCTIONS
-- ################################################################################

-- Function to check overall database integrity
CREATE OR REPLACE FUNCTION check_database_integrity()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    issue_count INTEGER,
    details TEXT
) AS $$
BEGIN
    -- Check for orphaned workflows
    RETURN QUERY
    SELECT 
        'Orphaned Workflows'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        COUNT(*)::INTEGER,
        'Workflows without valid campaign references'::TEXT
    FROM campaign_workflows cw
    LEFT JOIN campaigns c ON cw.campaign_id = c.id
    WHERE c.id IS NULL;
    
    -- Check for orphaned content
    RETURN QUERY
    SELECT 
        'Orphaned Content'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        COUNT(*)::INTEGER,
        'Content without valid campaign references'::TEXT
    FROM campaign_content cc
    LEFT JOIN campaigns c ON cc.campaign_id = c.id
    WHERE c.id IS NULL;
    
    -- Check for inconsistent campaign progress
    RETURN QUERY
    SELECT 
        'Campaign Progress Consistency'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        COUNT(*)::INTEGER,
        'Campaigns with progress 100% but not completed status'::TEXT
    FROM campaigns
    WHERE progress_percentage >= 100.00 AND status != 'completed';
    
    -- Check for workflow steps without workflows
    RETURN QUERY
    SELECT 
        'Orphaned Workflow Steps'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        COUNT(*)::INTEGER,
        'Workflow steps without valid workflow references'::TEXT
    FROM campaign_workflow_steps cws
    LEFT JOIN campaign_workflows cw ON cws.workflow_id = cw.id
    WHERE cw.id IS NULL;
    
    -- Check for performance records without steps
    RETURN QUERY
    SELECT 
        'Orphaned Performance Records'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        COUNT(*)::INTEGER,
        'Performance records without valid step references'::TEXT
    FROM agent_orchestration_performance aop
    LEFT JOIN campaign_workflow_steps cws ON aop.step_id = cws.id
    WHERE cws.id IS NULL;
    
END;
$$ LANGUAGE plpgsql;

-- Function to repair common data integrity issues
CREATE OR REPLACE FUNCTION repair_data_integrity_issues()
RETURNS TABLE(
    repair_action TEXT,
    affected_records INTEGER,
    status TEXT
) AS $$
DECLARE
    repair_count INTEGER;
BEGIN
    -- Repair campaign progress inconsistencies
    UPDATE campaigns 
    SET status = 'completed', actual_completion = NOW()
    WHERE progress_percentage >= 100.00 
    AND status NOT IN ('completed', 'cancelled', 'archived');
    
    GET DIAGNOSTICS repair_count = ROW_COUNT;
    
    RETURN QUERY
    SELECT 
        'Fix Completed Campaign Status'::TEXT,
        repair_count,
        CASE WHEN repair_count > 0 THEN 'REPAIRED' ELSE 'NO_ISSUES' END::TEXT;
    
    -- Clean up orphaned calendar events
    DELETE FROM campaign_calendar
    WHERE campaign_id NOT IN (SELECT id FROM campaigns);
    
    GET DIAGNOSTICS repair_count = ROW_COUNT;
    
    RETURN QUERY
    SELECT 
        'Remove Orphaned Calendar Events'::TEXT,
        repair_count,
        CASE WHEN repair_count > 0 THEN 'REPAIRED' ELSE 'NO_ISSUES' END::TEXT;
    
    -- Update stale workflow statuses
    UPDATE campaign_workflows
    SET status = 'timeout'
    WHERE status = 'running' 
    AND started_at < NOW() - INTERVAL '24 hours'
    AND completed_at IS NULL;
    
    GET DIAGNOSTICS repair_count = ROW_COUNT;
    
    RETURN QUERY
    SELECT 
        'Fix Stale Running Workflows'::TEXT,
        repair_count,
        CASE WHEN repair_count > 0 THEN 'REPAIRED' ELSE 'NO_ISSUES' END::TEXT;
    
END;
$$ LANGUAGE plpgsql;

-- ################################################################################
-- VALIDATION SUMMARY
-- ################################################################################

-- Create validation summary view
CREATE OR REPLACE VIEW validation_summary AS
SELECT 
    'Campaign Validation' as category,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE status IN ('active', 'scheduled', 'running')) as active_records,
    COUNT(*) FILTER (WHERE progress_percentage >= 100 AND status != 'completed') as inconsistent_records
FROM campaigns
UNION ALL
SELECT 
    'Workflow Validation',
    COUNT(*),
    COUNT(*) FILTER (WHERE status IN ('running', 'paused')),
    COUNT(*) FILTER (WHERE status = 'running' AND started_at < NOW() - INTERVAL '2 hours' AND completed_at IS NULL)
FROM campaign_workflows
UNION ALL
SELECT 
    'Content Validation',
    COUNT(*),
    COUNT(*) FILTER (WHERE status IN ('published', 'approved')),
    COUNT(*) FILTER (WHERE quality_score < 50)
FROM campaign_content
WHERE is_active = true;

-- ================================================================================
-- END DATA VALIDATION AND CONSTRAINT SYSTEM
-- ================================================================================
-- This comprehensive validation system provides:
-- 1. Table-level constraints for data integrity
-- 2. Business logic validation through triggers
-- 3. Cross-table consistency checks
-- 4. Automated repair functions for common issues
-- 5. Integrity monitoring and reporting
-- 6. Status transition validation
-- 7. Resource usage and performance bounds
-- ================================================================================