-- Content Deliverable System Migration
-- Transforms task-centric to content-centric approach

-- Content Deliverables Table - Stores actual content pieces
CREATE TABLE IF NOT EXISTS content_deliverables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL,
    parent_content_id UUID, -- For repurposed content
    
    -- Content Identity
    content_type VARCHAR(50) NOT NULL, -- blog_post, social_media, email, video_script
    channel VARCHAR(50) NOT NULL, -- linkedin, twitter, email, blog
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT NOT NULL, -- Full content body
    format VARCHAR(20) DEFAULT 'markdown', -- markdown, html, plain
    
    -- Workflow Status
    status VARCHAR(20) DEFAULT 'draft', -- draft, review, approved, published
    quality_score INTEGER,
    
    -- Narrative Structure
    narrative_position INTEGER, -- Order in campaign story
    narrative_theme TEXT, -- Central theme/message
    story_arc VARCHAR(50), -- introduction, development, climax, resolution
    
    -- Content Metadata
    word_count INTEGER,
    reading_time_minutes INTEGER,
    seo_score INTEGER,
    target_keywords TEXT[],
    
    -- Relationships
    related_content_ids UUID[],
    references_content_ids UUID[], -- Content this piece references
    
    -- Publication
    scheduled_for TIMESTAMP,
    published_at TIMESTAMP,
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    
    -- Constraints
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_content_id) REFERENCES content_deliverables(id) ON DELETE SET NULL
);

-- Campaign Narrative Context - Ensures story coherence
CREATE TABLE IF NOT EXISTS campaign_narratives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL UNIQUE,
    
    -- Narrative Strategy
    primary_theme TEXT NOT NULL,
    target_audience TEXT NOT NULL,
    key_messages TEXT[],
    story_arc TEXT, -- Overall campaign narrative structure
    
    -- Content Planning
    content_pillars JSONB, -- Main content themes and supporting topics
    narrative_flow JSONB, -- How content pieces connect
    
    -- Voice and Tone
    brand_voice TEXT,
    tone_guidelines TEXT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

-- Content Revisions - Version control for content
CREATE TABLE IF NOT EXISTS content_revisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_deliverable_id UUID NOT NULL,
    revision_number INTEGER NOT NULL,
    
    -- Revision Content
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    
    -- Revision Metadata
    change_type VARCHAR(50), -- creation, edit, review, approval
    change_reason TEXT,
    quality_score INTEGER,
    
    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    
    FOREIGN KEY (content_deliverable_id) REFERENCES content_deliverables(id) ON DELETE CASCADE,
    UNIQUE(content_deliverable_id, revision_number)
);

-- Content Performance Metrics
CREATE TABLE IF NOT EXISTS content_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_deliverable_id UUID NOT NULL,
    
    -- Engagement Metrics
    views INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    
    -- Performance Scores
    engagement_rate DECIMAL(5,4) DEFAULT 0,
    conversion_rate DECIMAL(5,4) DEFAULT 0,
    click_through_rate DECIMAL(5,4) DEFAULT 0,
    
    -- Platform Specific
    platform VARCHAR(50),
    platform_metrics JSONB,
    
    -- Time Tracking
    measured_at TIMESTAMP DEFAULT NOW(),
    performance_date DATE,
    
    FOREIGN KEY (content_deliverable_id) REFERENCES content_deliverables(id) ON DELETE CASCADE
);

-- Link existing campaign_tasks to content_deliverables for backward compatibility
ALTER TABLE campaign_tasks 
ADD COLUMN IF NOT EXISTS content_deliverable_id UUID;

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_content_deliverables_campaign_id ON content_deliverables(campaign_id);
CREATE INDEX IF NOT EXISTS idx_content_deliverables_content_type ON content_deliverables(content_type);
CREATE INDEX IF NOT EXISTS idx_content_deliverables_status ON content_deliverables(status);
CREATE INDEX IF NOT EXISTS idx_content_deliverables_narrative_position ON content_deliverables(narrative_position);
CREATE INDEX IF NOT EXISTS idx_content_deliverables_created_at ON content_deliverables(created_at);
CREATE INDEX IF NOT EXISTS idx_campaign_narratives_campaign_id ON campaign_narratives(campaign_id);
CREATE INDEX IF NOT EXISTS idx_content_revisions_deliverable_id ON content_revisions(content_deliverable_id);
CREATE INDEX IF NOT EXISTS idx_content_performance_deliverable_id ON content_performance(content_deliverable_id);

-- Full-text search for content
CREATE INDEX IF NOT EXISTS idx_content_deliverables_fulltext 
ON content_deliverables USING gin(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, '') || ' ' || COALESCE(summary, '')));

-- Add foreign key constraint for campaign_tasks after tables are created
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_campaign_task_content' 
        AND table_name = 'campaign_tasks'
    ) THEN
        ALTER TABLE campaign_tasks 
        ADD CONSTRAINT fk_campaign_task_content 
            FOREIGN KEY (content_deliverable_id) 
            REFERENCES content_deliverables(id) ON DELETE SET NULL;
    END IF;
END $$;

COMMENT ON TABLE content_deliverables IS 'Stores actual content deliverables (blog posts, social media, emails) instead of task metadata';
COMMENT ON TABLE campaign_narratives IS 'Manages campaign-level story structure and narrative coherence';
COMMENT ON TABLE content_revisions IS 'Tracks content changes and version history';
COMMENT ON TABLE content_performance IS 'Tracks content engagement and performance metrics';