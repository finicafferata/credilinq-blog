-- ================================================================================
-- Unified Campaign Progress Synchronization Migration
-- ================================================================================
-- This migration creates triggers and functions to ensure data consistency
-- between legacy campaign_tasks and new orchestration workflow schemas.
-- It provides automatic synchronization for consistent progress tracking.
-- ================================================================================

-- Create function to synchronize campaign progress
CREATE OR REPLACE FUNCTION sync_campaign_progress()
RETURNS TRIGGER AS $$
DECLARE
    campaign_id_val UUID;
    total_tasks INTEGER;
    completed_tasks INTEGER;
    in_progress_tasks INTEGER;
    progress_percentage DECIMAL(5,2);
BEGIN
    -- Get campaign_id from the affected table
    IF TG_TABLE_NAME = 'campaign_tasks' THEN
        campaign_id_val := COALESCE(NEW.campaign_id, OLD.campaign_id);
    ELSIF TG_TABLE_NAME = 'campaign_workflow_steps' THEN
        -- Get campaign_id from workflow
        SELECT cw.campaign_id INTO campaign_id_val
        FROM campaign_workflows cw
        WHERE cw.id = COALESCE(NEW.workflow_id, OLD.workflow_id);
    END IF;

    -- Skip if no campaign_id found
    IF campaign_id_val IS NULL THEN
        RETURN COALESCE(NEW, OLD);
    END IF;

    -- Calculate progress from campaign_tasks (legacy)
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN status IN ('completed', 'approved', 'scheduled') THEN 1 END) as completed,
        COUNT(CASE WHEN status IN ('in_progress', 'generated', 'under_review') THEN 1 END) as in_progress
    INTO total_tasks, completed_tasks, in_progress_tasks
    FROM campaign_tasks
    WHERE campaign_id = campaign_id_val;

    -- Calculate progress percentage
    IF total_tasks > 0 THEN
        progress_percentage := ROUND((completed_tasks::DECIMAL / total_tasks::DECIMAL) * 100, 2);
    ELSE
        progress_percentage := 0.00;
    END IF;

    -- Update campaign status based on progress
    UPDATE campaigns 
    SET status = CASE 
        WHEN progress_percentage = 100.00 THEN 'completed'
        WHEN progress_percentage > 0 AND progress_percentage < 100.00 THEN 'active'
        WHEN in_progress_tasks > 0 THEN 'active'
        ELSE status 
    END,
    updated_at = NOW()
    WHERE id = campaign_id_val;

    -- Try to sync with orchestration workflow if it exists
    UPDATE campaign_workflows
    SET steps_total = total_tasks,
        steps_completed = completed_tasks,
        updated_at = NOW()
    WHERE campaign_id = campaign_id_val;

    -- Log progress update
    INSERT INTO campaign_progress_log (campaign_id, total_tasks, completed_tasks, progress_percentage, calculated_at)
    VALUES (campaign_id_val, total_tasks, completed_tasks, progress_percentage, NOW())
    ON CONFLICT (campaign_id) DO UPDATE SET
        total_tasks = EXCLUDED.total_tasks,
        completed_tasks = EXCLUDED.completed_tasks,
        progress_percentage = EXCLUDED.progress_percentage,
        calculated_at = EXCLUDED.calculated_at;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create progress log table for tracking
CREATE TABLE IF NOT EXISTS campaign_progress_log (
    campaign_id UUID PRIMARY KEY,
    total_tasks INTEGER NOT NULL DEFAULT 0,
    completed_tasks INTEGER NOT NULL DEFAULT 0,
    progress_percentage DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_campaign_progress_log_campaign_id ON campaign_progress_log(campaign_id);
CREATE INDEX IF NOT EXISTS idx_campaign_progress_log_calculated_at ON campaign_progress_log(calculated_at);

-- Create triggers for automatic synchronization
DROP TRIGGER IF EXISTS trigger_sync_campaign_progress_on_task_change ON campaign_tasks;
CREATE TRIGGER trigger_sync_campaign_progress_on_task_change
    AFTER INSERT OR UPDATE OR DELETE ON campaign_tasks
    FOR EACH ROW EXECUTE FUNCTION sync_campaign_progress();

-- Create trigger for orchestration workflow steps (if table exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'campaign_workflow_steps') THEN
        DROP TRIGGER IF EXISTS trigger_sync_campaign_progress_on_step_change ON campaign_workflow_steps;
        CREATE TRIGGER trigger_sync_campaign_progress_on_step_change
            AFTER INSERT OR UPDATE OR DELETE ON campaign_workflow_steps
            FOR EACH ROW EXECUTE FUNCTION sync_campaign_progress();
    END IF;
END
$$;

-- Function to manually recalculate all campaign progress
CREATE OR REPLACE FUNCTION recalculate_all_campaign_progress()
RETURNS TABLE(campaign_id UUID, old_progress DECIMAL(5,2), new_progress DECIMAL(5,2)) AS $$
DECLARE
    rec RECORD;
    old_prog DECIMAL(5,2);
    new_prog DECIMAL(5,2);
BEGIN
    FOR rec IN SELECT DISTINCT c.id FROM campaigns c LOOP
        -- Get old progress
        SELECT progress_percentage INTO old_prog
        FROM campaign_progress_log 
        WHERE campaign_progress_log.campaign_id = rec.id;
        
        -- Trigger recalculation
        UPDATE campaign_tasks 
        SET updated_at = updated_at 
        WHERE campaign_tasks.campaign_id = rec.id;
        
        -- Get new progress
        SELECT progress_percentage INTO new_prog
        FROM campaign_progress_log 
        WHERE campaign_progress_log.campaign_id = rec.id;
        
        campaign_id := rec.id;
        old_progress := COALESCE(old_prog, 0.00);
        new_progress := COALESCE(new_prog, 0.00);
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get unified campaign progress
CREATE OR REPLACE FUNCTION get_campaign_progress(p_campaign_id UUID)
RETURNS TABLE(
    campaign_id UUID,
    total_tasks INTEGER,
    completed_tasks INTEGER,
    in_progress_tasks INTEGER,
    pending_tasks INTEGER,
    failed_tasks INTEGER,
    progress_percentage DECIMAL(5,2),
    current_phase TEXT,
    last_updated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cpl.campaign_id,
        cpl.total_tasks,
        cpl.completed_tasks,
        COUNT(CASE WHEN ct.status IN ('in_progress', 'generated', 'under_review') THEN 1 END)::INTEGER as in_progress_tasks,
        COUNT(CASE WHEN ct.status IN ('pending') THEN 1 END)::INTEGER as pending_tasks,
        COUNT(CASE WHEN ct.status IN ('failed', 'error', 'cancelled', 'timeout') THEN 1 END)::INTEGER as failed_tasks,
        cpl.progress_percentage,
        CASE 
            WHEN cpl.progress_percentage = 0 THEN 'planning'
            WHEN cpl.progress_percentage < 30 THEN 'content_creation'
            WHEN cpl.progress_percentage < 70 THEN 'content_review'
            WHEN cpl.progress_percentage < 100 THEN 'distribution_prep'
            ELSE 'campaign_execution'
        END as current_phase,
        cpl.calculated_at as last_updated
    FROM campaign_progress_log cpl
    LEFT JOIN campaign_tasks ct ON ct.campaign_id = cpl.campaign_id
    WHERE cpl.campaign_id = p_campaign_id
    GROUP BY cpl.campaign_id, cpl.total_tasks, cpl.completed_tasks, cpl.progress_percentage, cpl.calculated_at;
END;
$$ LANGUAGE plpgsql;

-- Function to ensure campaign progress consistency
CREATE OR REPLACE FUNCTION ensure_campaign_progress_consistency(p_campaign_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    legacy_count INTEGER;
    orchestration_count INTEGER;
    workflow_id_val UUID;
BEGIN
    -- Get legacy task count
    SELECT COUNT(*) INTO legacy_count
    FROM campaign_tasks
    WHERE campaign_id = p_campaign_id;

    -- Get orchestration workflow if exists
    SELECT id INTO workflow_id_val
    FROM campaign_workflows
    WHERE campaign_id = p_campaign_id
    ORDER BY created_at DESC
    LIMIT 1;

    IF workflow_id_val IS NOT NULL THEN
        -- Get orchestration step count
        SELECT COUNT(*) INTO orchestration_count
        FROM campaign_workflow_steps
        WHERE workflow_id = workflow_id_val;

        -- Sync orchestration to match legacy if different
        IF legacy_count != orchestration_count AND legacy_count > 0 THEN
            -- Update orchestration workflow totals
            UPDATE campaign_workflows
            SET steps_total = legacy_count,
                updated_at = NOW()
            WHERE id = workflow_id_val;

            -- Create missing workflow steps if needed
            INSERT INTO campaign_workflow_steps (
                id, workflow_id, step_name, step_type, step_order, status, created_at, updated_at
            )
            SELECT 
                ct.id,
                workflow_id_val,
                COALESCE(ct.task_type, 'content_creation'),
                'content_creation',
                ROW_NUMBER() OVER (ORDER BY ct.created_at),
                ct.status,
                ct.created_at,
                ct.updated_at
            FROM campaign_tasks ct
            WHERE ct.campaign_id = p_campaign_id
            AND ct.id NOT IN (
                SELECT cws.id 
                FROM campaign_workflow_steps cws 
                WHERE cws.workflow_id = workflow_id_val
            );
        END IF;
    END IF;

    -- Force progress recalculation
    UPDATE campaign_tasks 
    SET updated_at = NOW() 
    WHERE campaign_id = p_campaign_id;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Create function to initialize missing progress records
CREATE OR REPLACE FUNCTION initialize_missing_progress_records()
RETURNS INTEGER AS $$
DECLARE
    missing_count INTEGER := 0;
    rec RECORD;
BEGIN
    FOR rec IN 
        SELECT c.id 
        FROM campaigns c 
        LEFT JOIN campaign_progress_log cpl ON c.id = cpl.campaign_id 
        WHERE cpl.campaign_id IS NULL
    LOOP
        -- Trigger progress calculation for campaigns without progress records
        PERFORM ensure_campaign_progress_consistency(rec.id);
        missing_count := missing_count + 1;
    END LOOP;
    
    RETURN missing_count;
END;
$$ LANGUAGE plpgsql;

-- Initialize progress records for existing campaigns
SELECT initialize_missing_progress_records();

-- Create function to get dashboard progress data (optimized for performance)
CREATE OR REPLACE FUNCTION get_dashboard_campaign_progress()
RETURNS TABLE(
    campaign_id UUID,
    campaign_name TEXT,
    status TEXT,
    progress DECIMAL(5,2),
    total_tasks INTEGER,
    completed_tasks INTEGER,
    created_at TIMESTAMPTZ,
    target_channels TEXT[],
    current_phase TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id as campaign_id,
        COALESCE(b.campaign_name, 'Unnamed Campaign') as campaign_name,
        c.status,
        COALESCE(cpl.progress_percentage, 0.00) as progress,
        COALESCE(cpl.total_tasks, 0) as total_tasks,
        COALESCE(cpl.completed_tasks, 0) as completed_tasks,
        c.created_at,
        CASE 
            WHEN b.channels IS NOT NULL THEN 
                ARRAY(SELECT jsonb_array_elements_text(b.channels))
            ELSE ARRAY['blog', 'linkedin']::TEXT[]
        END as target_channels,
        CASE 
            WHEN COALESCE(cpl.progress_percentage, 0) = 0 THEN 'planning'
            WHEN COALESCE(cpl.progress_percentage, 0) < 30 THEN 'content_creation'
            WHEN COALESCE(cpl.progress_percentage, 0) < 70 THEN 'content_review'
            WHEN COALESCE(cpl.progress_percentage, 0) < 100 THEN 'distribution_prep'
            ELSE 'campaign_execution'
        END as current_phase
    FROM campaigns c
    LEFT JOIN briefings b ON c.id = b.campaign_id
    LEFT JOIN campaign_progress_log cpl ON c.id = cpl.campaign_id
    WHERE c.created_at >= NOW() - INTERVAL '30 days'
    ORDER BY c.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for optimized queries
CREATE INDEX IF NOT EXISTS idx_campaigns_created_at_recent ON campaigns(created_at) WHERE created_at >= NOW() - INTERVAL '30 days';
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_campaign_status ON campaign_tasks(campaign_id, status);
CREATE INDEX IF NOT EXISTS idx_briefings_campaign_id_channels ON briefings(campaign_id, channels);

-- Add comments for documentation
COMMENT ON FUNCTION sync_campaign_progress() IS 'Automatically synchronizes campaign progress between legacy and orchestration schemas';
COMMENT ON FUNCTION get_campaign_progress(UUID) IS 'Returns unified campaign progress data from synchronized sources';
COMMENT ON FUNCTION ensure_campaign_progress_consistency(UUID) IS 'Ensures data consistency between legacy and orchestration schemas for a specific campaign';
COMMENT ON FUNCTION get_dashboard_campaign_progress() IS 'Optimized function for dashboard campaign progress display';
COMMENT ON TABLE campaign_progress_log IS 'Stores calculated campaign progress for consistent reporting across all views';

-- Test the synchronization by triggering a recalculation
SELECT 'Migration completed. Testing synchronization...' as status;
SELECT campaign_id, old_progress, new_progress 
FROM recalculate_all_campaign_progress() 
LIMIT 5;