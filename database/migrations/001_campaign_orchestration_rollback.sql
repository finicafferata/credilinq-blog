-- ================================================================================
-- CAMPAIGN ORCHESTRATION MIGRATION ROLLBACK SCRIPT
-- ================================================================================
-- This script safely rolls back the campaign orchestration migration,
-- restoring the original content-centric architecture
-- Rollback: 001 | Version: 1.0 | Date: 2024-12-18
-- Estimated Duration: 10-15 minutes | Downtime: 1-3 minutes
-- ================================================================================

-- ################################################################################
-- ROLLBACK VALIDATION AND SAFETY CHECKS
-- ################################################################################

-- Start rollback logging
INSERT INTO migration_log (migration_name, version, status, details) 
VALUES (
    'Campaign Orchestration Migration Rollback', 
    '1.0',
    'running',
    json_build_object(
        'description', 'Rollback campaign orchestration migration to content-centric architecture',
        'rollback_started_at', NOW()
    )
);

-- ################################################################################
-- STEP 1: DROP CAMPAIGN ORCHESTRATION COMPONENTS
-- ################################################################################

RAISE NOTICE 'ROLLBACK STEP 1/4: Removing campaign orchestration components...';

-- Drop triggers first
DROP TRIGGER IF EXISTS trigger_campaign_progress_on_workflow_change ON campaign_workflows;
DROP TRIGGER IF EXISTS trigger_campaign_progress_on_content_change ON campaign_content;
DROP TRIGGER IF EXISTS trigger_campaign_progress_on_step_change ON campaign_workflow_steps;

-- Drop functions
DROP FUNCTION IF EXISTS calculate_campaign_progress(UUID);
DROP FUNCTION IF EXISTS update_campaign_status(UUID);
DROP FUNCTION IF EXISTS trigger_update_campaign_progress();

-- Drop views
DROP VIEW IF EXISTS campaign_dashboard;

-- Drop tables in dependency order
DROP TABLE IF EXISTS agent_orchestration_performance CASCADE;
DROP TABLE IF EXISTS campaign_analytics CASCADE;
DROP TABLE IF EXISTS campaign_calendar CASCADE;
DROP TABLE IF EXISTS campaign_content_relationships CASCADE;
DROP TABLE IF EXISTS campaign_content CASCADE;
DROP TABLE IF EXISTS campaign_workflow_steps CASCADE;
DROP TABLE IF EXISTS campaign_workflows CASCADE;

RAISE NOTICE 'ROLLBACK STEP 1/4: Campaign orchestration components removed';

-- ################################################################################
-- STEP 2: RESTORE ORIGINAL CAMPAIGNS TABLE
-- ################################################################################

RAISE NOTICE 'ROLLBACK STEP 2/4: Restoring original campaigns table...';

-- Drop current campaigns table
DROP TABLE IF EXISTS campaigns CASCADE;

-- Restore from backup
ALTER TABLE campaigns_old RENAME TO campaigns;

-- Restore blog_posts foreign key constraint
ALTER TABLE blog_posts DROP CONSTRAINT IF EXISTS blog_posts_campaign_id_fkey;
ALTER TABLE blog_posts 
ADD CONSTRAINT blog_posts_campaign_id_fkey 
FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE SET NULL;

RAISE NOTICE 'ROLLBACK STEP 2/4: Original campaigns table restored';

-- ################################################################################
-- STEP 3: DROP REMAINING ORCHESTRATION TABLES
-- ################################################################################

RAISE NOTICE 'ROLLBACK STEP 3/4: Cleaning up remaining tables...';

DROP TABLE IF EXISTS campaign_strategies CASCADE;
DROP TABLE IF EXISTS campaign_orchestrators CASCADE;

-- Drop custom types
DROP TYPE IF EXISTS content_status CASCADE;
DROP TYPE IF EXISTS campaign_priority CASCADE;
DROP TYPE IF EXISTS strategy_type CASCADE;
DROP TYPE IF EXISTS step_type CASCADE;
DROP TYPE IF EXISTS step_status CASCADE;
DROP TYPE IF EXISTS workflow_status CASCADE;
DROP TYPE IF EXISTS orchestrator_type CASCADE;

RAISE NOTICE 'ROLLBACK STEP 3/4: Cleanup completed';

-- ################################################################################
-- STEP 4: VALIDATION AND COMPLETION
-- ################################################################################

RAISE NOTICE 'ROLLBACK STEP 4/4: Running validation...';

-- Validate rollback
DO $$
DECLARE
    campaign_count INTEGER;
    blog_post_count INTEGER;
    orchestrator_exists BOOLEAN;
BEGIN
    SELECT COUNT(*) INTO campaign_count FROM campaigns;
    SELECT COUNT(*) INTO blog_post_count FROM blog_posts;
    
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'campaign_orchestrators'
    ) INTO orchestrator_exists;
    
    IF orchestrator_exists THEN
        RAISE EXCEPTION 'Rollback failed - orchestration tables still exist';
    END IF;
    
    RAISE NOTICE 'Rollback validation passed - Campaigns: %, Blog Posts: %', 
        campaign_count, blog_post_count;
END;
$$;

-- Mark migration as rolled back
UPDATE migration_log 
SET status = 'rolled_back'
WHERE migration_name = 'Campaign Orchestration Migration' 
AND version = '1.0';

-- Complete rollback logging
UPDATE migration_log 
SET 
    status = 'completed',
    completed_at = NOW()
WHERE migration_name = 'Campaign Orchestration Migration Rollback' 
AND version = '1.0' 
AND status = 'running';

RAISE NOTICE 'ROLLBACK COMPLETED SUCCESSFULLY!';

-- ================================================================================
-- END ROLLBACK SCRIPT
-- ================================================================================