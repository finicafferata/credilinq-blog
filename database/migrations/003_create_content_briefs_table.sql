-- Create content_briefs table for strategic content brief storage
-- Migration: 003_create_content_briefs_table
-- Created: 2025-08-25

BEGIN;

-- Create content_briefs table
CREATE TABLE IF NOT EXISTS content_briefs (
    id VARCHAR(255) PRIMARY KEY,
    brief_data JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Extracted fields for querying
    title VARCHAR(500),
    content_type VARCHAR(100),
    primary_purpose VARCHAR(100),
    target_audience VARCHAR(500),
    company_context TEXT,
    
    -- Indexes for performance
    CONSTRAINT content_briefs_status_check CHECK (status IN ('draft', 'approved', 'rejected', 'blog_generated'))
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_content_briefs_status ON content_briefs(status);
CREATE INDEX IF NOT EXISTS idx_content_briefs_created_at ON content_briefs(created_at);
CREATE INDEX IF NOT EXISTS idx_content_briefs_content_type ON content_briefs(content_type);
CREATE INDEX IF NOT EXISTS idx_content_briefs_primary_purpose ON content_briefs(primary_purpose);
CREATE INDEX IF NOT EXISTS idx_content_briefs_title_gin ON content_briefs USING gin (to_tsvector('english', title));

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_content_briefs_updated_at
    BEFORE UPDATE ON content_briefs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMIT;