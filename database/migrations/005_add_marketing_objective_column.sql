-- Add marketing_objective column to content_briefs table
-- Migration: 005_add_marketing_objective_column
-- Created: 2025-08-26

BEGIN;

-- Add marketing_objective column
ALTER TABLE content_briefs ADD COLUMN IF NOT EXISTS marketing_objective TEXT;

-- Create index for marketing objectives for filtering
CREATE INDEX IF NOT EXISTS idx_content_briefs_marketing_objective ON content_briefs(marketing_objective);

COMMIT;