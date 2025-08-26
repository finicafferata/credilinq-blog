-- Add summary column to content_briefs table
-- Migration: 004_add_summary_column
-- Created: 2025-08-26

BEGIN;

-- Add summary column
ALTER TABLE content_briefs ADD COLUMN IF NOT EXISTS summary TEXT;

-- Add approval related columns that are referenced in code
ALTER TABLE content_briefs ADD COLUMN IF NOT EXISTS approval_notes TEXT;
ALTER TABLE content_briefs ADD COLUMN IF NOT EXISTS approved_at TIMESTAMP WITH TIME ZONE;

-- Create index for approved briefs
CREATE INDEX IF NOT EXISTS idx_content_briefs_approved_at ON content_briefs(approved_at);

COMMIT;