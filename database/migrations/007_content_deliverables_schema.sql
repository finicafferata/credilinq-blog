-- ================================================================================
-- CONTENT DELIVERABLES SCHEMA MIGRATION
-- ================================================================================
-- Adds support for content-first workflow with deliverable tracking
-- Transforms from task-centric to content-centric data model
-- Generated: 2024-12-18 | Version: 1.0
-- ================================================================================

-- Content Deliverable Types Enum
CREATE TYPE content_deliverable_type AS ENUM (
    'blog_post',
    'linkedin_post',
    'twitter_thread',
    'email_sequence',
    'case_study',
    'whitepaper',
    'infographic_copy',
    'video_script',
    'newsletter',
    'landing_page_copy'
);

-- Content Status Enum  
CREATE TYPE content_status AS ENUM (
    'planned',
    'in_progress',
    'draft_complete',
    'under_review',
    'approved',
    'published',
    'archived'
);

-- Narrative Position Enum
CREATE TYPE narrative_position AS ENUM (
    'foundation',
    'exploration',
    'application',
    'transformation',
    'reinforcement'
);

-- Content Deliverables Table
CREATE TABLE content_deliverables (
    content_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    
    -- Content Identification
    deliverable_type content_deliverable_type NOT NULL,
    title VARCHAR(500) NOT NULL,
    content_body TEXT,
    summary TEXT,
    word_count INTEGER DEFAULT 0,
    status content_status DEFAULT 'planned',
    
    -- Narrative Flow
    narrative_position narrative_position NOT NULL,
    narrative_thread_id VARCHAR(255) NOT NULL,
    key_message TEXT NOT NULL,
    supporting_points JSONB DEFAULT '[]',
    
    -- Content Relationships
    references_content JSONB DEFAULT '[]', -- Array of content IDs this references
    referenced_by_content JSONB DEFAULT '[]', -- Array of content IDs that reference this
    narrative_precedence JSONB DEFAULT '[]', -- Content that should be read before this
    narrative_sequence JSONB DEFAULT '[]', -- Content that flows after this
    
    -- Content Metadata
    target_audience VARCHAR(500) DEFAULT 'B2B professionals',
    tone VARCHAR(100) DEFAULT 'Professional',
    channel VARCHAR(100),
    seo_keywords JSONB DEFAULT '[]',
    call_to_action TEXT,
    
    -- Performance & Quality
    readability_score DECIMAL(3,2),
    engagement_prediction VARCHAR(100),
    quality_score DECIMAL(3,2),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Indexes for performance
    INDEX idx_content_deliverables_campaign_id (campaign_id),
    INDEX idx_content_deliverables_status (status),
    INDEX idx_content_deliverables_type (deliverable_type),
    INDEX idx_content_deliverables_narrative_position (narrative_position),
    INDEX idx_content_deliverables_narrative_thread (narrative_thread_id),
    INDEX idx_content_deliverables_created_at (created_at),
    INDEX idx_content_deliverables_status_campaign (status, campaign_id),
    INDEX idx_content_deliverables_type_status (deliverable_type, status)
);

-- Narrative Contexts Table
CREATE TABLE narrative_contexts (
    context_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    narrative_thread_id VARCHAR(255) NOT NULL,
    
    -- Narrative Definition
    central_theme TEXT NOT NULL,
    supporting_themes JSONB DEFAULT '[]',
    key_messages JSONB DEFAULT '[]',
    target_transformation TEXT,
    brand_voice_guidelines TEXT,
    
    -- Flow Management
    content_journey_map JSONB DEFAULT '{}',
    cross_references JSONB DEFAULT '{}',
    thematic_connections JSONB DEFAULT '{}',
    
    -- Consistency Tracking
    terminology_glossary JSONB DEFAULT '{}',
    recurring_concepts JSONB DEFAULT '[]',
    brand_examples JSONB DEFAULT '[]',
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_narrative_contexts_campaign_id (campaign_id),
    INDEX idx_narrative_contexts_thread_id (narrative_thread_id),
    UNIQUE(campaign_id, narrative_thread_id)
);

-- Content Relationships Table (for complex relationship tracking)
CREATE TABLE content_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_content_id UUID NOT NULL REFERENCES content_deliverables(content_id) ON DELETE CASCADE,
    target_content_id UUID NOT NULL REFERENCES content_deliverables(content_id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- 'references', 'builds_on', 'supports', 'follows'
    relationship_strength DECIMAL(3,2) DEFAULT 0.5, -- 0.0 to 1.0
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_content_relationships_source (source_content_id),
    INDEX idx_content_relationships_target (target_content_id),
    INDEX idx_content_relationships_type (relationship_type),
    UNIQUE(source_content_id, target_content_id, relationship_type)
);

-- Content Generation Workflows Table
CREATE TABLE content_generation_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    
    -- Workflow Status
    workflow_status VARCHAR(50) DEFAULT 'initializing',
    current_phase VARCHAR(100),
    total_deliverables INTEGER DEFAULT 0,
    completed_deliverables INTEGER DEFAULT 0,
    failed_deliverables INTEGER DEFAULT 0,
    
    -- Workflow Configuration
    workflow_config JSONB DEFAULT '{}',
    agent_assignments JSONB DEFAULT '{}',
    priority_order JSONB DEFAULT '[]',
    
    -- Quality Metrics
    average_quality_score DECIMAL(3,2),
    total_word_count INTEGER DEFAULT 0,
    revision_count INTEGER DEFAULT 0,
    
    -- Timestamps
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    estimated_completion TIMESTAMPTZ,
    
    -- Indexes
    INDEX idx_content_workflows_campaign_id (campaign_id),
    INDEX idx_content_workflows_status (workflow_status),
    INDEX idx_content_workflows_started_at (started_at)
);

-- Content Quality Reviews Table
CREATE TABLE content_quality_reviews (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL REFERENCES content_deliverables(content_id) ON DELETE CASCADE,
    reviewer_agent_id VARCHAR(255),
    
    -- Quality Scores (0.0 to 1.0)
    content_quality_score DECIMAL(3,2),
    narrative_alignment_score DECIMAL(3,2),
    message_reinforcement_score DECIMAL(3,2),
    audience_appropriateness_score DECIMAL(3,2),
    brand_voice_consistency_score DECIMAL(3,2),
    overall_quality_score DECIMAL(3,2),
    
    -- Review Feedback
    feedback_summary TEXT,
    improvement_suggestions JSONB DEFAULT '[]',
    revision_required BOOLEAN DEFAULT false,
    
    -- Review Metadata
    reviewed_at TIMESTAMPTZ DEFAULT NOW(),
    review_duration_ms INTEGER,
    
    -- Indexes
    INDEX idx_content_reviews_content_id (content_id),
    INDEX idx_content_reviews_overall_score (overall_quality_score),
    INDEX idx_content_reviews_reviewed_at (reviewed_at)
);

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_content_deliverables_updated_at 
    BEFORE UPDATE ON content_deliverables 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_narrative_contexts_updated_at 
    BEFORE UPDATE ON narrative_contexts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample narrative context for testing
INSERT INTO narrative_contexts (
    campaign_id,
    narrative_thread_id,
    central_theme,
    supporting_themes,
    key_messages,
    target_transformation,
    brand_voice_guidelines,
    terminology_glossary,
    recurring_concepts
) VALUES (
    gen_random_uuid(), -- This would be a real campaign ID in production
    'sample_main_thread',
    'Digital transformation through AI-powered solutions',
    '["Innovation Leadership", "Customer Success", "Technology Excellence"]',
    '["Transform your business", "Achieve competitive advantage", "Drive measurable results"]',
    'Move prospects from awareness to consideration and trial',
    'Professional, authoritative, approachable, results-focused',
    '{"AI": "Artificial Intelligence", "ROI": "Return on Investment", "Digital Transformation": "Technology-enabled business evolution"}',
    '["competitive advantage", "measurable results", "business transformation", "innovation"]'
);

-- Comments for documentation
COMMENT ON TABLE content_deliverables IS 'Stores complete content deliverables with narrative context and relationships';
COMMENT ON TABLE narrative_contexts IS 'Maintains narrative consistency and thematic flow across content portfolios';
COMMENT ON TABLE content_relationships IS 'Tracks complex relationships and dependencies between content pieces';
COMMENT ON TABLE content_generation_workflows IS 'Manages content-first workflow execution and progress';
COMMENT ON TABLE content_quality_reviews IS 'Records quality assessments and improvement feedback for content';

COMMENT ON COLUMN content_deliverables.narrative_thread_id IS 'Links content to its narrative context for consistency';
COMMENT ON COLUMN content_deliverables.references_content IS 'JSON array of content IDs that this content references';
COMMENT ON COLUMN content_deliverables.narrative_precedence IS 'Content that should be consumed before this piece';
COMMENT ON COLUMN narrative_contexts.content_journey_map IS 'Maps content types to their position in the customer journey';
COMMENT ON COLUMN narrative_contexts.terminology_glossary IS 'Consistent terminology definitions for brand voice';

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON content_deliverables TO your_app_user;
-- GRANT ALL PRIVILEGES ON narrative_contexts TO your_app_user;
-- GRANT ALL PRIVILEGES ON content_relationships TO your_app_user;
-- GRANT ALL PRIVILEGES ON content_generation_workflows TO your_app_user;
-- GRANT ALL PRIVILEGES ON content_quality_reviews TO your_app_user;