-- Fix campaign_tasks table schema for campaign orchestration
-- Add missing columns needed for smart scheduling and review workflow

-- Add task_details column for structured task data
ALTER TABLE campaign_tasks ADD COLUMN IF NOT EXISTS task_details JSONB;

-- Add review workflow columns
ALTER TABLE campaign_tasks ADD COLUMN IF NOT EXISTS quality_score DECIMAL(5,2);
ALTER TABLE campaign_tasks ADD COLUMN IF NOT EXISTS review_notes TEXT;
ALTER TABLE campaign_tasks ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITH TIME ZONE;

-- Add scheduling column
ALTER TABLE campaign_tasks ADD COLUMN IF NOT EXISTS scheduled_at TIMESTAMP WITH TIME ZONE;

-- Update existing tasks with default task_details if empty
UPDATE campaign_tasks 
SET task_details = jsonb_build_object(
    'task_type', task_type,
    'channel', CASE 
        WHEN task_type LIKE '%linkedin%' THEN 'linkedin'
        WHEN task_type LIKE '%facebook%' THEN 'facebook'
        WHEN task_type LIKE '%twitter%' THEN 'twitter'
        ELSE 'linkedin'
    END,
    'content_type', CASE 
        WHEN task_type LIKE '%blog%' THEN 'blog_posts'
        WHEN task_type LIKE '%social%' THEN 'social_posts'
        ELSE 'social_posts'
    END
)
WHERE task_details IS NULL;

-- Create index on task_details for better performance
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_task_details ON campaign_tasks USING gin(task_details);

-- Create index on status for filtering
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_status ON campaign_tasks(status);

-- Create index on scheduled_at for scheduling queries
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_scheduled_at ON campaign_tasks(scheduled_at) WHERE scheduled_at IS NOT NULL;