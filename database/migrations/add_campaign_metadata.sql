-- Add metadata column to campaigns table to store additional campaign information
ALTER TABLE campaigns 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Add indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_campaigns_metadata ON campaigns USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns (status);
CREATE INDEX IF NOT EXISTS idx_campaigns_created_at ON campaigns (created_at DESC);