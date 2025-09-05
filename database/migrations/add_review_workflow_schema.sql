-- Review Workflow Database Schema Migration
-- Adds comprehensive review workflow support to the existing database schema

-- Review Workflow Executions Table
-- Tracks individual workflow executions and their overall state
CREATE TABLE review_workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID UNIQUE NOT NULL,
    content_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    campaign_id UUID REFERENCES campaigns(id) ON DELETE SET NULL,
    blog_post_id UUID REFERENCES blog_posts(id) ON DELETE SET NULL,
    
    -- Workflow configuration
    review_config JSONB NOT NULL DEFAULT '{}',
    required_stages TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    
    -- Workflow state
    workflow_status VARCHAR(50) NOT NULL DEFAULT 'initialized',
    overall_approval_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_stage VARCHAR(50),
    
    -- Stage tracking
    completed_stages TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    failed_stages TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    skipped_stages TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    
    -- Content data
    content_data JSONB NOT NULL DEFAULT '{}',
    content_metadata JSONB NOT NULL DEFAULT '{}',
    original_workflow_state JSONB,
    
    -- Progress tracking
    overall_progress FLOAT NOT NULL DEFAULT 0.0,
    stage_progress JSONB NOT NULL DEFAULT '{}',
    estimated_completion TIMESTAMPTZ,
    
    -- Pause/resume functionality
    is_paused BOOLEAN NOT NULL DEFAULT false,
    pause_reason TEXT,
    paused_at_stage VARCHAR(50),
    resume_token UUID,
    
    -- Final output
    final_content JSONB,
    approval_summary JSONB NOT NULL DEFAULT '{}',
    quality_metrics JSONB NOT NULL DEFAULT '{}',
    
    -- Integration
    next_workflow_step VARCHAR(255),
    integration_callbacks JSONB NOT NULL DEFAULT '[]',
    
    -- Audit and timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Error handling
    error_state TEXT,
    retry_attempts INTEGER NOT NULL DEFAULT 0,
    max_retry_attempts INTEGER NOT NULL DEFAULT 3
);

-- Indexes for review_workflow_executions
CREATE INDEX idx_review_workflow_executions_workflow_id ON review_workflow_executions(workflow_execution_id);
CREATE INDEX idx_review_workflow_executions_content_id ON review_workflow_executions(content_id);
CREATE INDEX idx_review_workflow_executions_campaign_id ON review_workflow_executions(campaign_id);
CREATE INDEX idx_review_workflow_executions_status ON review_workflow_executions(workflow_status);
CREATE INDEX idx_review_workflow_executions_approval_status ON review_workflow_executions(overall_approval_status);
CREATE INDEX idx_review_workflow_executions_paused ON review_workflow_executions(is_paused, pause_reason);
CREATE INDEX idx_review_workflow_executions_created ON review_workflow_executions(started_at);
CREATE INDEX idx_review_workflow_executions_updated ON review_workflow_executions(updated_at);
CREATE INDEX idx_review_workflow_executions_resume_token ON review_workflow_executions(resume_token) WHERE resume_token IS NOT NULL;

-- Review Stage Decisions Table
-- Tracks individual review decisions for each stage
CREATE TABLE review_stage_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,
    
    -- Decision details
    reviewer_id VARCHAR(255) NOT NULL,
    reviewer_type VARCHAR(20) NOT NULL CHECK (reviewer_type IN ('agent', 'human', 'mixed')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'in_agent_review', 'awaiting_human_review', 'human_reviewing', 'approved', 'rejected', 'needs_revision', 'skipped')),
    
    -- Scoring and feedback
    score FLOAT,
    feedback TEXT NOT NULL DEFAULT '',
    suggestions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    revision_requests TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    
    -- Automated analysis
    automated_checks JSONB NOT NULL DEFAULT '{}',
    agent_analysis JSONB,
    
    -- Timing
    decision_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    review_started_at TIMESTAMPTZ,
    review_completed_at TIMESTAMPTZ,
    
    -- Human review specific
    assigned_reviewers TEXT[],
    review_timeout_at TIMESTAMPTZ,
    escalated BOOLEAN NOT NULL DEFAULT false,
    escalation_reason TEXT,
    
    -- Audit trail
    decision_history JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Indexes for review_stage_decisions
CREATE INDEX idx_review_stage_decisions_workflow_id ON review_stage_decisions(workflow_execution_id);
CREATE INDEX idx_review_stage_decisions_stage ON review_stage_decisions(stage);
CREATE INDEX idx_review_stage_decisions_reviewer ON review_stage_decisions(reviewer_id);
CREATE INDEX idx_review_stage_decisions_status ON review_stage_decisions(status);
CREATE INDEX idx_review_stage_decisions_reviewer_type ON review_stage_decisions(reviewer_type);
CREATE INDEX idx_review_stage_decisions_score ON review_stage_decisions(score) WHERE score IS NOT NULL;
CREATE INDEX idx_review_stage_decisions_timestamp ON review_stage_decisions(decision_timestamp);
CREATE INDEX idx_review_stage_decisions_timeout ON review_stage_decisions(review_timeout_at) WHERE review_timeout_at IS NOT NULL;
CREATE INDEX idx_review_stage_decisions_escalated ON review_stage_decisions(escalated) WHERE escalated = true;

-- Human Review Assignments Table
-- Tracks human reviewer assignments and their status
CREATE TABLE human_review_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,
    
    -- Reviewer details
    reviewer_id VARCHAR(255) NOT NULL,
    reviewer_email VARCHAR(255),
    reviewer_name VARCHAR(255),
    reviewer_role VARCHAR(100),
    
    -- Assignment details
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assigned_by VARCHAR(255),
    assignment_status VARCHAR(50) NOT NULL DEFAULT 'assigned' CHECK (assignment_status IN ('assigned', 'accepted', 'declined', 'completed', 'expired', 'reassigned')),
    
    -- Review timeline
    expected_completion_at TIMESTAMPTZ,
    review_started_at TIMESTAMPTZ,
    review_completed_at TIMESTAMPTZ,
    
    -- Notifications
    notification_sent BOOLEAN NOT NULL DEFAULT false,
    notification_sent_at TIMESTAMPTZ,
    reminder_count INTEGER NOT NULL DEFAULT 0,
    last_reminder_sent_at TIMESTAMPTZ,
    
    -- Review context
    review_instructions TEXT,
    automated_score FLOAT,
    automated_suggestions TEXT[],
    priority VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    
    -- Metadata
    assignment_metadata JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for human_review_assignments
CREATE INDEX idx_human_review_assignments_workflow_id ON human_review_assignments(workflow_execution_id);
CREATE INDEX idx_human_review_assignments_reviewer_id ON human_review_assignments(reviewer_id);
CREATE INDEX idx_human_review_assignments_stage ON human_review_assignments(stage);
CREATE INDEX idx_human_review_assignments_status ON human_review_assignments(assignment_status);
CREATE INDEX idx_human_review_assignments_assigned_at ON human_review_assignments(assigned_at);
CREATE INDEX idx_human_review_assignments_expected_completion ON human_review_assignments(expected_completion_at) WHERE expected_completion_at IS NOT NULL;
CREATE INDEX idx_human_review_assignments_pending ON human_review_assignments(assignment_status, expected_completion_at) WHERE assignment_status IN ('assigned', 'accepted');

-- Review Workflow Checkpoints Table
-- Stores workflow state snapshots for pause/resume functionality
CREATE TABLE review_workflow_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE CASCADE,
    checkpoint_name VARCHAR(100) NOT NULL,
    
    -- Checkpoint data
    workflow_state JSONB NOT NULL,
    checkpoint_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Resume functionality
    resume_token UUID,
    is_resumable BOOLEAN NOT NULL DEFAULT true,
    resumed_at TIMESTAMPTZ,
    resumed_by VARCHAR(255),
    
    -- Checkpoint metadata
    checkpoint_type VARCHAR(50) NOT NULL DEFAULT 'manual' CHECK (checkpoint_type IN ('manual', 'automatic', 'pause', 'error', 'completion')),
    description TEXT,
    created_by VARCHAR(255)
);

-- Indexes for review_workflow_checkpoints
CREATE INDEX idx_review_workflow_checkpoints_workflow_id ON review_workflow_checkpoints(workflow_execution_id);
CREATE INDEX idx_review_workflow_checkpoints_resume_token ON review_workflow_checkpoints(resume_token) WHERE resume_token IS NOT NULL;
CREATE INDEX idx_review_workflow_checkpoints_timestamp ON review_workflow_checkpoints(checkpoint_timestamp);
CREATE INDEX idx_review_workflow_checkpoints_resumable ON review_workflow_checkpoints(is_resumable) WHERE is_resumable = true;

-- Review Audit Trail Table
-- Comprehensive audit trail for all review workflow actions
CREATE TABLE review_audit_trail (
    id BIGSERIAL PRIMARY KEY,
    workflow_execution_id UUID NOT NULL REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE CASCADE,
    
    -- Action details
    action_type VARCHAR(100) NOT NULL, -- 'workflow_started', 'stage_completed', 'human_review_assigned', etc.
    actor_id VARCHAR(255) NOT NULL,
    actor_type VARCHAR(50) NOT NULL CHECK (actor_type IN ('system', 'agent', 'human', 'api')),
    
    -- Context
    stage VARCHAR(50),
    previous_state VARCHAR(50),
    new_state VARCHAR(50),
    
    -- Action data
    action_data JSONB NOT NULL DEFAULT '{}',
    action_description TEXT,
    
    -- Timing
    action_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Metadata
    correlation_id UUID,
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT
);

-- Indexes for review_audit_trail
CREATE INDEX idx_review_audit_trail_workflow_id ON review_audit_trail(workflow_execution_id);
CREATE INDEX idx_review_audit_trail_action_type ON review_audit_trail(action_type);
CREATE INDEX idx_review_audit_trail_actor ON review_audit_trail(actor_id, actor_type);
CREATE INDEX idx_review_audit_trail_timestamp ON review_audit_trail(action_timestamp);
CREATE INDEX idx_review_audit_trail_stage ON review_audit_trail(stage) WHERE stage IS NOT NULL;

-- Review Notifications Table
-- Tracks notifications sent during review process
CREATE TABLE review_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE CASCADE,
    assignment_id UUID REFERENCES human_review_assignments(id) ON DELETE SET NULL,
    
    -- Notification details
    notification_type VARCHAR(100) NOT NULL, -- 'assignment', 'reminder', 'escalation', 'completion'
    recipient_id VARCHAR(255) NOT NULL,
    recipient_email VARCHAR(255),
    recipient_type VARCHAR(50) NOT NULL DEFAULT 'human' CHECK (recipient_type IN ('human', 'system', 'webhook')),
    
    -- Message content
    subject VARCHAR(500),
    message_body TEXT,
    notification_channel VARCHAR(50) NOT NULL DEFAULT 'email' CHECK (notification_channel IN ('email', 'slack', 'webhook', 'in_app')),
    
    -- Delivery tracking
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ,
    
    -- Status
    delivery_status VARCHAR(50) NOT NULL DEFAULT 'sent' CHECK (delivery_status IN ('sent', 'delivered', 'failed', 'bounced', 'opened', 'clicked')),
    failure_reason TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    
    -- Context
    stage VARCHAR(50),
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    
    -- Metadata
    notification_metadata JSONB NOT NULL DEFAULT '{}',
    external_message_id VARCHAR(255)
);

-- Indexes for review_notifications
CREATE INDEX idx_review_notifications_workflow_id ON review_notifications(workflow_execution_id);
CREATE INDEX idx_review_notifications_recipient ON review_notifications(recipient_id);
CREATE INDEX idx_review_notifications_type ON review_notifications(notification_type);
CREATE INDEX idx_review_notifications_status ON review_notifications(delivery_status);
CREATE INDEX idx_review_notifications_sent_at ON review_notifications(sent_at);
CREATE INDEX idx_review_notifications_stage ON review_notifications(stage) WHERE stage IS NOT NULL;

-- Content Quality Assessments Table
-- Stores detailed quality assessments from agents
CREATE TABLE content_quality_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,
    
    -- Assessment details
    assessed_by VARCHAR(255) NOT NULL,
    assessment_type VARCHAR(50) NOT NULL, -- 'automated', 'human', 'hybrid'
    
    -- Quality scores
    overall_score FLOAT NOT NULL CHECK (overall_score >= 0 AND overall_score <= 10),
    dimension_scores JSONB NOT NULL DEFAULT '{}',
    
    -- Analysis details
    strengths TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    improvement_areas TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    suggestions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    
    -- Content analysis
    content_statistics JSONB NOT NULL DEFAULT '{}',
    readability_metrics JSONB NOT NULL DEFAULT '{}',
    technical_analysis JSONB NOT NULL DEFAULT '{}',
    
    -- Assessment metadata
    assessment_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assessment_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Detailed analysis
    detailed_analysis JSONB NOT NULL DEFAULT '{}',
    assessment_metadata JSONB NOT NULL DEFAULT '{}'
);

-- Indexes for content_quality_assessments
CREATE INDEX idx_content_quality_assessments_workflow_id ON content_quality_assessments(workflow_execution_id);
CREATE INDEX idx_content_quality_assessments_stage ON content_quality_assessments(stage);
CREATE INDEX idx_content_quality_assessments_overall_score ON content_quality_assessments(overall_score);
CREATE INDEX idx_content_quality_assessments_assessed_by ON content_quality_assessments(assessed_by);
CREATE INDEX idx_content_quality_assessments_timestamp ON content_quality_assessments(assessment_timestamp);

-- Extend existing campaign_tasks table to support review workflow integration
ALTER TABLE campaign_tasks ADD COLUMN IF NOT EXISTS review_workflow_id UUID REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_campaign_tasks_review_workflow ON campaign_tasks(review_workflow_id) WHERE review_workflow_id IS NOT NULL;

-- Extend blog_posts table to support review workflow tracking
ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS review_workflow_id UUID REFERENCES review_workflow_executions(workflow_execution_id) ON DELETE SET NULL;
ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS review_status VARCHAR(50);
ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 10);
ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS last_reviewed_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_blog_posts_review_workflow ON blog_posts(review_workflow_id) WHERE review_workflow_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_blog_posts_review_status ON blog_posts(review_status) WHERE review_status IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_blog_posts_quality_score ON blog_posts(quality_score) WHERE quality_score IS NOT NULL;

-- Functions and Triggers for Review Workflow

-- Function to update review_workflow_executions.updated_at
CREATE OR REPLACE FUNCTION update_review_workflow_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for review_workflow_executions
CREATE TRIGGER trigger_review_workflow_executions_updated_at
    BEFORE UPDATE ON review_workflow_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_review_workflow_updated_at();

-- Function to automatically create audit trail entries
CREATE OR REPLACE FUNCTION create_review_audit_entry()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert audit trail entry for significant changes
    IF TG_OP = 'UPDATE' AND (
        OLD.workflow_status != NEW.workflow_status OR
        OLD.overall_approval_status != NEW.overall_approval_status OR
        OLD.current_stage != NEW.current_stage OR
        OLD.is_paused != NEW.is_paused
    ) THEN
        INSERT INTO review_audit_trail (
            workflow_execution_id,
            action_type,
            actor_id,
            actor_type,
            stage,
            previous_state,
            new_state,
            action_data,
            action_description
        ) VALUES (
            NEW.workflow_execution_id,
            CASE 
                WHEN OLD.workflow_status != NEW.workflow_status THEN 'workflow_status_changed'
                WHEN OLD.overall_approval_status != NEW.overall_approval_status THEN 'approval_status_changed'
                WHEN OLD.current_stage != NEW.current_stage THEN 'stage_changed'
                WHEN OLD.is_paused != NEW.is_paused THEN 'pause_status_changed'
            END,
            'system',
            'system',
            NEW.current_stage,
            COALESCE(OLD.workflow_status, OLD.overall_approval_status, OLD.current_stage::text, OLD.is_paused::text),
            COALESCE(NEW.workflow_status, NEW.overall_approval_status, NEW.current_stage::text, NEW.is_paused::text),
            jsonb_build_object(
                'old_values', jsonb_build_object(
                    'workflow_status', OLD.workflow_status,
                    'overall_approval_status', OLD.overall_approval_status,
                    'current_stage', OLD.current_stage,
                    'is_paused', OLD.is_paused
                ),
                'new_values', jsonb_build_object(
                    'workflow_status', NEW.workflow_status,
                    'overall_approval_status', NEW.overall_approval_status,
                    'current_stage', NEW.current_stage,
                    'is_paused', NEW.is_paused
                )
            ),
            'Automatic audit trail entry for workflow state change'
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic audit trail creation
CREATE TRIGGER trigger_review_workflow_audit_trail
    AFTER UPDATE ON review_workflow_executions
    FOR EACH ROW
    EXECUTE FUNCTION create_review_audit_entry();

-- Function to clean up old completed workflows (retention policy)
CREATE OR REPLACE FUNCTION cleanup_completed_review_workflows(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete completed workflows older than retention_days
    DELETE FROM review_workflow_executions 
    WHERE workflow_status = 'completed' 
    AND completed_at < NOW() - INTERVAL '1 day' * retention_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Views for common review workflow queries

-- View for active review workflows
CREATE OR REPLACE VIEW active_review_workflows AS
SELECT 
    rwe.*,
    ARRAY_LENGTH(rwe.completed_stages, 1) as completed_stage_count,
    ARRAY_LENGTH(rwe.required_stages, 1) as total_stage_count,
    CASE 
        WHEN ARRAY_LENGTH(rwe.required_stages, 1) > 0 
        THEN (ARRAY_LENGTH(rwe.completed_stages, 1)::FLOAT / ARRAY_LENGTH(rwe.required_stages, 1)::FLOAT) * 100
        ELSE 0 
    END as completion_percentage
FROM review_workflow_executions rwe
WHERE workflow_status IN ('initialized', 'in_progress', 'paused')
ORDER BY started_at DESC;

-- View for pending human reviews
CREATE OR REPLACE VIEW pending_human_reviews AS
SELECT 
    hra.*,
    rwe.content_type,
    rwe.campaign_id,
    rsd.score as automated_score,
    rsd.suggestions as automated_suggestions,
    CASE 
        WHEN hra.expected_completion_at < NOW() THEN true
        ELSE false
    END as is_overdue
FROM human_review_assignments hra
JOIN review_workflow_executions rwe ON rwe.workflow_execution_id = hra.workflow_execution_id
LEFT JOIN review_stage_decisions rsd ON rsd.workflow_execution_id = hra.workflow_execution_id AND rsd.stage = hra.stage
WHERE hra.assignment_status IN ('assigned', 'accepted')
ORDER BY 
    CASE WHEN hra.expected_completion_at < NOW() THEN 1 ELSE 2 END,
    hra.expected_completion_at ASC;

-- View for workflow performance metrics
CREATE OR REPLACE VIEW review_workflow_metrics AS
SELECT 
    DATE_TRUNC('day', started_at) as date,
    COUNT(*) as workflows_started,
    COUNT(*) FILTER (WHERE workflow_status = 'completed') as workflows_completed,
    COUNT(*) FILTER (WHERE overall_approval_status = 'approved') as workflows_approved,
    COUNT(*) FILTER (WHERE overall_approval_status = 'needs_revision') as workflows_needing_revision,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at)) / 3600) FILTER (WHERE completed_at IS NOT NULL) as avg_completion_hours,
    AVG(overall_progress) as avg_progress
FROM review_workflow_executions
WHERE started_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', started_at)
ORDER BY date DESC;

-- Grant appropriate permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO review_workflow_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO review_workflow_user;

COMMENT ON TABLE review_workflow_executions IS 'Main table tracking review workflow executions and their overall state';
COMMENT ON TABLE review_stage_decisions IS 'Individual review decisions for each stage of the workflow';
COMMENT ON TABLE human_review_assignments IS 'Human reviewer assignments and their status';
COMMENT ON TABLE review_workflow_checkpoints IS 'Workflow state snapshots for pause/resume functionality';
COMMENT ON TABLE review_audit_trail IS 'Comprehensive audit trail for all review workflow actions';
COMMENT ON TABLE review_notifications IS 'Notifications sent during the review process';
COMMENT ON TABLE content_quality_assessments IS 'Detailed quality assessments from agents and humans';