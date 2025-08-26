-- ================================================================================
-- CAMPAIGN ORCHESTRATION SCHEMA - PHASE 1: FOUNDATION ARCHITECTURE
-- ================================================================================
-- This schema transforms the current content-centric architecture to campaign-centric
-- Campaign becomes the primary entity with orchestrated workflows
-- Generated: 2024-12-18 | Version: 1.0 | Phase: Foundation
-- ================================================================================

-- ################################################################################
-- ENUMS AND TYPES
-- ################################################################################

-- Campaign Orchestration Enums
CREATE TYPE orchestrator_type AS ENUM (
    'content_creation',
    'multi_channel_distribution', 
    'performance_optimization',
    'audience_engagement',
    'competitive_intelligence',
    'custom_workflow'
);

CREATE TYPE workflow_status AS ENUM (
    'pending',
    'running',
    'paused',
    'completed',
    'failed',
    'cancelled',
    'timeout'
);

CREATE TYPE step_status AS ENUM (
    'pending',
    'running', 
    'completed',
    'failed',
    'skipped',
    'retry'
);

CREATE TYPE step_type AS ENUM (
    'content_planning',
    'content_generation',
    'content_optimization',
    'content_review',
    'content_publishing',
    'performance_analysis',
    'audience_analysis',
    'competitive_analysis',
    'custom_task'
);

CREATE TYPE strategy_type AS ENUM (
    'thought_leadership',
    'product_marketing',
    'brand_awareness',
    'lead_generation',
    'customer_retention',
    'competitive_positioning',
    'market_education',
    'community_building'
);

CREATE TYPE campaign_priority AS ENUM (
    'low',
    'medium', 
    'high',
    'critical',
    'urgent'
);

CREATE TYPE content_status AS ENUM (
    'draft',
    'in_review',
    'approved',
    'published',
    'archived',
    'deprecated'
);

-- ################################################################################
-- CORE CAMPAIGN ORCHESTRATION TABLES
-- ################################################################################

-- Campaign Orchestrators - Define reusable workflow templates
CREATE TABLE campaign_orchestrators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    orchestrator_type orchestrator_type NOT NULL,
    
    -- Workflow Configuration
    workflow_definition JSONB NOT NULL, -- Complete workflow structure
    agent_mappings JSONB DEFAULT '{}', -- Agent name to class mappings
    configuration JSONB DEFAULT '{}', -- Runtime configuration
    
    -- Execution Settings
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    max_concurrent_executions INTEGER DEFAULT 5,
    timeout_seconds INTEGER DEFAULT 3600,
    retry_policy JSONB DEFAULT '{"max_retries": 3, "backoff_multiplier": 2}',
    
    -- Status and Metrics
    status VARCHAR(50) DEFAULT 'active', -- active, inactive, deprecated
    version VARCHAR(20) DEFAULT '1.0',
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    average_execution_time_ms INTEGER DEFAULT 0,
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID, -- Reference to user who created this
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Campaign Orchestrators
CREATE INDEX idx_campaign_orchestrators_type ON campaign_orchestrators(orchestrator_type);
CREATE INDEX idx_campaign_orchestrators_status ON campaign_orchestrators(status);
CREATE INDEX idx_campaign_orchestrators_priority ON campaign_orchestrators(priority);
CREATE INDEX idx_campaign_orchestrators_created ON campaign_orchestrators(created_at);
CREATE INDEX idx_campaign_orchestrators_type_status ON campaign_orchestrators(orchestrator_type, status);

-- Campaign Strategies - Strategic framework for campaigns
CREATE TABLE campaign_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    strategy_type strategy_type NOT NULL,
    
    -- Strategic Configuration
    target_channels TEXT[] DEFAULT '{}', -- Distribution channels
    content_pillars JSONB DEFAULT '[]', -- Core content themes
    messaging_framework JSONB DEFAULT '{}', -- Brand voice and messaging
    timeline_template JSONB DEFAULT '{}', -- Template for campaign timeline
    
    -- KPI and Performance Targets
    kpi_targets JSONB DEFAULT '{}', -- Target metrics
    budget_allocation JSONB, -- Budget breakdown by channel/activity
    success_metrics JSONB DEFAULT '{}', -- Success measurement criteria
    
    -- Template Configuration
    is_template BOOLEAN DEFAULT false,
    template_category VARCHAR(100),
    usage_count INTEGER DEFAULT 0,
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Campaign Strategies
CREATE INDEX idx_campaign_strategies_type ON campaign_strategies(strategy_type);
CREATE INDEX idx_campaign_strategies_template ON campaign_strategies(is_template);
CREATE INDEX idx_campaign_strategies_category ON campaign_strategies(template_category);
CREATE INDEX idx_campaign_strategies_usage ON campaign_strategies(usage_count);

-- Campaigns - Central campaign entity (PRIMARY WORKFLOW ENTRY POINT)
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Strategic Association
    strategy_id UUID REFERENCES campaign_strategies(id) ON DELETE SET NULL,
    orchestrator_id UUID REFERENCES campaign_orchestrators(id) ON DELETE SET NULL,
    
    -- Campaign Configuration
    campaign_data JSONB NOT NULL DEFAULT '{}', -- Campaign-specific data
    execution_context JSONB DEFAULT '{}', -- Runtime execution context
    priority campaign_priority DEFAULT 'medium',
    
    -- Scheduling and Timeline
    scheduled_start TIMESTAMPTZ,
    actual_start TIMESTAMPTZ,
    deadline TIMESTAMPTZ,
    estimated_completion TIMESTAMPTZ,
    actual_completion TIMESTAMPTZ,
    
    -- Status Tracking
    status VARCHAR(50) DEFAULT 'draft', -- draft, scheduled, active, paused, completed, cancelled
    progress_percentage DECIMAL(5,2) DEFAULT 0.00 CHECK (progress_percentage BETWEEN 0 AND 100),
    current_phase VARCHAR(100),
    
    -- Performance and Analytics
    performance_metrics JSONB DEFAULT '{}',
    kpi_actual JSONB DEFAULT '{}',
    budget_spent JSONB DEFAULT '{}',
    roi_calculated DECIMAL(10,4),
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID, -- Reference to user
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Campaigns (OPTIMIZED FOR CAMPAIGN-FIRST QUERIES)
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_priority ON campaigns(priority);
CREATE INDEX idx_campaigns_scheduled_start ON campaigns(scheduled_start);
CREATE INDEX idx_campaigns_deadline ON campaigns(deadline);
CREATE INDEX idx_campaigns_created_at ON campaigns(created_at);
CREATE INDEX idx_campaigns_orchestrator_id ON campaigns(orchestrator_id);
CREATE INDEX idx_campaigns_strategy_id ON campaigns(strategy_id);
CREATE INDEX idx_campaigns_status_priority ON campaigns(status, priority);
CREATE INDEX idx_campaigns_active_deadline ON campaigns(status, deadline) WHERE status IN ('active', 'scheduled');
CREATE INDEX idx_campaigns_progress ON campaigns(progress_percentage);
CREATE INDEX idx_campaigns_tags ON campaigns USING GIN(tags);
CREATE INDEX idx_campaigns_search_text ON campaigns USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')));

-- ################################################################################
-- WORKFLOW EXECUTION TABLES
-- ################################################################################

-- Campaign Workflows - Actual workflow execution instances
CREATE TABLE campaign_workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    orchestrator_id UUID NOT NULL REFERENCES campaign_orchestrators(id) ON DELETE RESTRICT,
    workflow_instance_id UUID NOT NULL, -- Unique identifier for this workflow execution
    
    -- Execution State
    current_step VARCHAR(255),
    step_sequence JSONB DEFAULT '[]', -- Ordered list of step execution
    execution_context JSONB DEFAULT '{}', -- Runtime context and variables
    status workflow_status DEFAULT 'pending',
    
    -- Timing and Performance
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_duration_ms INTEGER,
    actual_duration_ms INTEGER,
    
    -- Error Handling and Recovery
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    recovery_point VARCHAR(255), -- Last successful checkpoint
    
    -- Performance Metrics
    steps_total INTEGER DEFAULT 0,
    steps_completed INTEGER DEFAULT 0,
    steps_failed INTEGER DEFAULT 0,
    agent_executions JSONB DEFAULT '{}', -- Per-agent execution stats
    resource_usage JSONB DEFAULT '{}', -- Memory, CPU, etc.
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for Campaign Workflows
CREATE INDEX idx_campaign_workflows_campaign_id ON campaign_workflows(campaign_id);
CREATE INDEX idx_campaign_workflows_orchestrator_id ON campaign_workflows(orchestrator_id);
CREATE INDEX idx_campaign_workflows_status ON campaign_workflows(status);
CREATE INDEX idx_campaign_workflows_started_at ON campaign_workflows(started_at);
CREATE INDEX idx_campaign_workflows_completed_at ON campaign_workflows(completed_at);
CREATE INDEX idx_campaign_workflows_instance_id ON campaign_workflows(workflow_instance_id);
CREATE INDEX idx_campaign_workflows_active ON campaign_workflows(status, started_at) WHERE status IN ('pending', 'running', 'paused');
CREATE INDEX idx_campaign_workflows_duration ON campaign_workflows(actual_duration_ms) WHERE actual_duration_ms IS NOT NULL;

-- Campaign Workflow Steps - Individual step execution tracking
CREATE TABLE campaign_workflow_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    workflow_id UUID NOT NULL REFERENCES campaign_workflows(id) ON DELETE CASCADE,
    step_name VARCHAR(255) NOT NULL,
    step_type step_type NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    
    -- Step Configuration
    step_order INTEGER NOT NULL,
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    configuration JSONB DEFAULT '{}',
    dependencies JSONB DEFAULT '[]', -- List of dependent step names
    
    -- Execution State
    status step_status DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    
    -- Error Handling
    error_message TEXT,
    error_code VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Performance and Context
    execution_metadata JSONB DEFAULT '{}',
    resource_consumption JSONB DEFAULT '{}',
    agent_version VARCHAR(50),
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for Campaign Workflow Steps
CREATE INDEX idx_campaign_workflow_steps_workflow_id ON campaign_workflow_steps(workflow_id);
CREATE INDEX idx_campaign_workflow_steps_status ON campaign_workflow_steps(status);
CREATE INDEX idx_campaign_workflow_steps_step_type ON campaign_workflow_steps(step_type);
CREATE INDEX idx_campaign_workflow_steps_agent_name ON campaign_workflow_steps(agent_name);
CREATE INDEX idx_campaign_workflow_steps_order ON campaign_workflow_steps(workflow_id, step_order);
CREATE INDEX idx_campaign_workflow_steps_execution_time ON campaign_workflow_steps(execution_time_ms) WHERE execution_time_ms IS NOT NULL;
CREATE INDEX idx_campaign_workflow_steps_started_at ON campaign_workflow_steps(started_at);
CREATE INDEX idx_campaign_workflow_steps_completed_at ON campaign_workflow_steps(completed_at);

-- ################################################################################
-- CONTENT MANAGEMENT TABLES (CAMPAIGN-CENTRIC APPROACH)
-- ################################################################################

-- Campaign Content - Content generated within campaigns
CREATE TABLE campaign_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Campaign Association (PRIMARY RELATIONSHIP)
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    workflow_id UUID REFERENCES campaign_workflows(id) ON DELETE SET NULL,
    
    -- Content Identity
    title VARCHAR(500) NOT NULL,
    content_type VARCHAR(100) NOT NULL, -- blog_post, social_post, email, video_script, etc.
    platform VARCHAR(100), -- target platform (website, linkedin, twitter, etc.)
    
    -- Content Data
    content_markdown TEXT,
    content_html TEXT,
    content_summary TEXT,
    content_metadata JSONB DEFAULT '{}',
    
    -- Generation Context
    generation_prompt JSONB, -- Original prompt/context used
    generation_model VARCHAR(100), -- AI model used
    generation_version VARCHAR(20), -- Content version
    parent_content_id UUID REFERENCES campaign_content(id), -- For derived content
    
    -- Status and Quality
    status content_status DEFAULT 'draft',
    quality_score DECIMAL(4,2), -- 0-100 quality rating
    seo_score DECIMAL(4,2), -- SEO optimization score
    readability_score DECIMAL(4,2), -- Content readability
    sentiment_score DECIMAL(4,2), -- Content sentiment
    
    -- Publishing and Distribution
    published_at TIMESTAMPTZ,
    scheduled_publish_at TIMESTAMPTZ,
    distribution_channels TEXT[] DEFAULT '{}',
    publication_urls JSONB DEFAULT '{}', -- URLs where content is published
    
    -- Performance Tracking
    view_count INTEGER DEFAULT 0,
    engagement_metrics JSONB DEFAULT '{}',
    conversion_metrics JSONB DEFAULT '{}',
    share_count INTEGER DEFAULT 0,
    
    -- Content Specifications
    word_count INTEGER,
    character_count INTEGER,
    reading_time_minutes INTEGER,
    language VARCHAR(10) DEFAULT 'en',
    target_keywords TEXT[] DEFAULT '{}',
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID, -- Agent or user ID
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Campaign Content
CREATE INDEX idx_campaign_content_campaign_id ON campaign_content(campaign_id);
CREATE INDEX idx_campaign_content_workflow_id ON campaign_content(workflow_id);
CREATE INDEX idx_campaign_content_type ON campaign_content(content_type);
CREATE INDEX idx_campaign_content_platform ON campaign_content(platform);
CREATE INDEX idx_campaign_content_status ON campaign_content(status);
CREATE INDEX idx_campaign_content_published_at ON campaign_content(published_at);
CREATE INDEX idx_campaign_content_scheduled_publish ON campaign_content(scheduled_publish_at) WHERE scheduled_publish_at IS NOT NULL;
CREATE INDEX idx_campaign_content_quality_score ON campaign_content(quality_score);
CREATE INDEX idx_campaign_content_parent_id ON campaign_content(parent_content_id);
CREATE INDEX idx_campaign_content_keywords ON campaign_content USING GIN(target_keywords);
CREATE INDEX idx_campaign_content_channels ON campaign_content USING GIN(distribution_channels);
CREATE INDEX idx_campaign_content_search ON campaign_content USING GIN(to_tsvector('english', title || ' ' || COALESCE(content_summary, '')));
CREATE INDEX idx_campaign_content_active_campaign ON campaign_content(campaign_id, is_active) WHERE is_active = true;

-- Campaign Content Relationships - Many-to-many relationships between content
CREATE TABLE campaign_content_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_content_id UUID NOT NULL REFERENCES campaign_content(id) ON DELETE CASCADE,
    target_content_id UUID NOT NULL REFERENCES campaign_content(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL, -- derived_from, references, complements, supersedes
    strength DECIMAL(3,2) DEFAULT 1.00, -- Relationship strength (0.00-1.00)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(source_content_id, target_content_id, relationship_type)
);

CREATE INDEX idx_content_relationships_source ON campaign_content_relationships(source_content_id);
CREATE INDEX idx_content_relationships_target ON campaign_content_relationships(target_content_id);
CREATE INDEX idx_content_relationships_type ON campaign_content_relationships(relationship_type);

-- ################################################################################
-- CALENDAR AND SCHEDULING TABLES
-- ################################################################################

-- Campaign Calendar - Unified content calendar for all campaigns
CREATE TABLE campaign_calendar (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    content_id UUID REFERENCES campaign_content(id) ON DELETE CASCADE,
    workflow_id UUID REFERENCES campaign_workflows(id) ON DELETE SET NULL,
    
    -- Event Details
    event_type VARCHAR(100) NOT NULL, -- content_publish, workflow_milestone, campaign_deadline, review_due
    title VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Scheduling
    scheduled_datetime TIMESTAMPTZ NOT NULL,
    duration_minutes INTEGER,
    timezone VARCHAR(50) DEFAULT 'UTC',
    all_day BOOLEAN DEFAULT false,
    
    -- Status and Tracking
    status VARCHAR(50) DEFAULT 'scheduled', -- scheduled, in_progress, completed, cancelled, postponed
    completion_percentage DECIMAL(5,2) DEFAULT 0.00,
    actual_completion_datetime TIMESTAMPTZ,
    
    -- Recurring Events
    recurrence_rule TEXT, -- RRULE format for recurring events
    recurrence_exceptions JSONB DEFAULT '[]', -- Specific dates to skip
    parent_event_id UUID REFERENCES campaign_calendar(id),
    
    -- Notifications and Reminders
    reminder_intervals INTEGER[] DEFAULT '{1440, 60, 15}', -- Minutes before event
    notification_sent JSONB DEFAULT '{}', -- Track which notifications sent
    
    -- Platform and Channel
    platform VARCHAR(100),
    channels TEXT[] DEFAULT '{}',
    assignee_id UUID, -- Who is responsible for this event
    
    -- Audit Fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Campaign Calendar
CREATE INDEX idx_campaign_calendar_campaign_id ON campaign_calendar(campaign_id);
CREATE INDEX idx_campaign_calendar_content_id ON campaign_calendar(content_id);
CREATE INDEX idx_campaign_calendar_scheduled_datetime ON campaign_calendar(scheduled_datetime);
CREATE INDEX idx_campaign_calendar_event_type ON campaign_calendar(event_type);
CREATE INDEX idx_campaign_calendar_status ON campaign_calendar(status);
CREATE INDEX idx_campaign_calendar_assignee_id ON campaign_calendar(assignee_id);
CREATE INDEX idx_campaign_calendar_platform ON campaign_calendar(platform);
CREATE INDEX idx_campaign_calendar_upcoming ON campaign_calendar(scheduled_datetime, status) WHERE status IN ('scheduled', 'in_progress');
CREATE INDEX idx_campaign_calendar_channels ON campaign_calendar USING GIN(channels);
CREATE INDEX idx_campaign_calendar_parent_event ON campaign_calendar(parent_event_id);

-- ################################################################################
-- ANALYTICS AND PERFORMANCE TABLES
-- ################################################################################

-- Campaign Analytics - Performance metrics and KPI tracking
CREATE TABLE campaign_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    content_id UUID REFERENCES campaign_content(id) ON DELETE CASCADE,
    
    -- Time Period
    measurement_date DATE NOT NULL,
    measurement_period VARCHAR(20) DEFAULT 'daily', -- daily, weekly, monthly
    
    -- Engagement Metrics
    views INTEGER DEFAULT 0,
    unique_views INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    click_through_rate DECIMAL(5,4),
    
    -- Social Media Metrics
    likes INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    engagement_rate DECIMAL(5,4),
    reach INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    
    -- Conversion Metrics
    conversions INTEGER DEFAULT 0,
    conversion_rate DECIMAL(5,4),
    revenue_generated DECIMAL(10,2),
    cost_per_conversion DECIMAL(8,2),
    
    -- SEO and Search Metrics
    organic_traffic INTEGER DEFAULT 0,
    keyword_rankings JSONB DEFAULT '{}',
    backlinks_gained INTEGER DEFAULT 0,
    domain_authority_impact DECIMAL(4,2),
    
    -- Platform-Specific Metrics
    platform VARCHAR(100),
    platform_metrics JSONB DEFAULT '{}', -- Platform-specific custom metrics
    
    -- Attribution and Source Tracking
    traffic_sources JSONB DEFAULT '{}',
    attribution_data JSONB DEFAULT '{}',
    referrer_data JSONB DEFAULT '{}',
    
    -- Campaign ROI
    cost_data JSONB DEFAULT '{}',
    roi_calculated DECIMAL(8,4),
    attribution_value DECIMAL(10,2),
    
    -- Audit Fields
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    data_source VARCHAR(100), -- google_analytics, social_platform, manual, etc.
    data_quality_score DECIMAL(3,2), -- Data reliability score
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Campaign Analytics
CREATE INDEX idx_campaign_analytics_campaign_id ON campaign_analytics(campaign_id);
CREATE INDEX idx_campaign_analytics_content_id ON campaign_analytics(content_id);
CREATE INDEX idx_campaign_analytics_measurement_date ON campaign_analytics(measurement_date);
CREATE INDEX idx_campaign_analytics_platform ON campaign_analytics(platform);
CREATE INDEX idx_campaign_analytics_period ON campaign_analytics(measurement_period);
CREATE INDEX idx_campaign_analytics_campaign_date ON campaign_analytics(campaign_id, measurement_date);
CREATE INDEX idx_campaign_analytics_conversions ON campaign_analytics(conversions) WHERE conversions > 0;
CREATE INDEX idx_campaign_analytics_revenue ON campaign_analytics(revenue_generated) WHERE revenue_generated > 0;
CREATE INDEX idx_campaign_analytics_collected_at ON campaign_analytics(collected_at);

-- ################################################################################
-- AGENT PERFORMANCE TRACKING (ENHANCED FOR ORCHESTRATION)
-- ################################################################################

-- Agent Orchestration Performance - Track agent performance within workflows
CREATE TABLE agent_orchestration_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    workflow_id UUID NOT NULL REFERENCES campaign_workflows(id) ON DELETE CASCADE,
    step_id UUID NOT NULL REFERENCES campaign_workflow_steps(id) ON DELETE CASCADE,
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    agent_version VARCHAR(50),
    
    -- Execution Tracking
    execution_id VARCHAR(255) UNIQUE NOT NULL, -- Unique execution identifier
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    
    -- Resource Usage
    memory_used_mb INTEGER,
    cpu_time_ms INTEGER,
    api_calls_made INTEGER DEFAULT 0,
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    tokens_total INTEGER DEFAULT 0,
    
    -- Cost Tracking
    estimated_cost_usd DECIMAL(10,6),
    cost_breakdown JSONB DEFAULT '{}',
    
    -- Quality and Performance Metrics
    output_quality_score DECIMAL(4,2),
    task_completion_rate DECIMAL(5,4),
    accuracy_score DECIMAL(4,2),
    efficiency_score DECIMAL(4,2),
    
    -- Error and Retry Tracking
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    timeout_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4),
    
    -- Context and Configuration
    input_data_size_bytes INTEGER,
    output_data_size_bytes INTEGER,
    configuration_hash VARCHAR(64), -- Hash of agent configuration
    execution_context JSONB DEFAULT '{}',
    
    -- Performance Benchmarks
    baseline_duration_ms INTEGER, -- Expected duration for this task type
    performance_deviation DECIMAL(6,4), -- Actual vs baseline performance
    percentile_rank DECIMAL(5,2), -- Performance rank among similar executions
    
    -- Audit Fields
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for Agent Orchestration Performance
CREATE INDEX idx_agent_orchestration_perf_workflow_id ON agent_orchestration_performance(workflow_id);
CREATE INDEX idx_agent_orchestration_perf_step_id ON agent_orchestration_performance(step_id);
CREATE INDEX idx_agent_orchestration_perf_agent_name ON agent_orchestration_performance(agent_name);
CREATE INDEX idx_agent_orchestration_perf_agent_type ON agent_orchestration_performance(agent_type);
CREATE INDEX idx_agent_orchestration_perf_execution_id ON agent_orchestration_performance(execution_id);
CREATE INDEX idx_agent_orchestration_perf_started_at ON agent_orchestration_performance(started_at);
CREATE INDEX idx_agent_orchestration_perf_duration ON agent_orchestration_performance(duration_ms) WHERE duration_ms IS NOT NULL;
CREATE INDEX idx_agent_orchestration_perf_cost ON agent_orchestration_performance(estimated_cost_usd) WHERE estimated_cost_usd IS NOT NULL;
CREATE INDEX idx_agent_orchestration_perf_quality ON agent_orchestration_performance(output_quality_score) WHERE output_quality_score IS NOT NULL;
CREATE INDEX idx_agent_orchestration_perf_agent_started ON agent_orchestration_performance(agent_name, started_at);

-- ################################################################################
-- MIGRATION COMPATIBILITY VIEWS
-- ################################################################################

-- Legacy BlogPost compatibility view
CREATE VIEW legacy_blog_posts AS
SELECT 
    cc.id,
    cc.title,
    cc.content_markdown,
    cc.generation_prompt as initial_prompt,
    cc.status::TEXT as status,
    '{}' as geo_metadata, -- Placeholder, will be populated during migration
    false as geo_optimized,
    NULL as geo_score,
    cc.seo_score,
    cc.word_count,
    cc.reading_time_minutes as reading_time,
    cc.created_at,
    cc.updated_at,
    cc.published_at,
    cc.campaign_id
FROM campaign_content cc
WHERE cc.content_type = 'blog_post' AND cc.is_active = true;

-- Legacy Campaigns compatibility view
CREATE VIEW legacy_campaigns AS
SELECT 
    c.id,
    c.name,
    c.status,
    c.created_at,
    c.updated_at,
    NULL as blog_post_id -- Will be populated for campaigns with single blog post
FROM campaigns c;

-- ################################################################################
-- TRIGGERS AND FUNCTIONS
-- ################################################################################

-- Function to update campaign progress based on workflow completion
CREATE OR REPLACE FUNCTION update_campaign_progress()
RETURNS TRIGGER AS $$
BEGIN
    -- Update campaign progress when workflow status changes
    IF NEW.status = 'completed' THEN
        UPDATE campaigns 
        SET progress_percentage = 100.0,
            actual_completion = NOW(),
            status = CASE 
                WHEN status != 'completed' THEN 'completed'
                ELSE status
            END
        WHERE id = NEW.campaign_id;
    ELSIF NEW.status = 'failed' THEN
        UPDATE campaigns 
        SET status = 'failed'
        WHERE id = NEW.campaign_id AND status NOT IN ('completed', 'cancelled');
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update campaign progress
CREATE TRIGGER trigger_update_campaign_progress
    AFTER UPDATE OF status ON campaign_workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_campaign_progress();

-- Function to update orchestrator statistics
CREATE OR REPLACE FUNCTION update_orchestrator_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update orchestrator statistics when workflow completes
    IF NEW.status IN ('completed', 'failed') AND (OLD.status IS NULL OR OLD.status NOT IN ('completed', 'failed')) THEN
        UPDATE campaign_orchestrators 
        SET total_executions = total_executions + 1,
            successful_executions = successful_executions + CASE WHEN NEW.status = 'completed' THEN 1 ELSE 0 END,
            average_execution_time_ms = CASE 
                WHEN NEW.actual_duration_ms IS NOT NULL THEN 
                    (average_execution_time_ms * total_executions + NEW.actual_duration_ms) / (total_executions + 1)
                ELSE average_execution_time_ms
            END,
            updated_at = NOW()
        WHERE id = NEW.orchestrator_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update orchestrator statistics
CREATE TRIGGER trigger_update_orchestrator_stats
    AFTER UPDATE OF status ON campaign_workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_orchestrator_stats();

-- Function to auto-update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers to all relevant tables
CREATE TRIGGER trigger_update_campaigns_updated_at
    BEFORE UPDATE ON campaigns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_campaign_orchestrators_updated_at
    BEFORE UPDATE ON campaign_orchestrators
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_campaign_strategies_updated_at
    BEFORE UPDATE ON campaign_strategies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_campaign_workflows_updated_at
    BEFORE UPDATE ON campaign_workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_campaign_workflow_steps_updated_at
    BEFORE UPDATE ON campaign_workflow_steps
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_campaign_content_updated_at
    BEFORE UPDATE ON campaign_content
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_campaign_calendar_updated_at
    BEFORE UPDATE ON campaign_calendar
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ################################################################################
-- CONSTRAINTS AND VALIDATION
-- ################################################################################

-- Add constraint to ensure campaign deadlines are after creation
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_deadline_after_creation 
CHECK (deadline IS NULL OR deadline > created_at);

-- Add constraint to ensure workflow completion time is after start
ALTER TABLE campaign_workflows 
ADD CONSTRAINT chk_workflow_completion_after_start 
CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at > started_at);

-- Add constraint to ensure step completion time is after start
ALTER TABLE campaign_workflow_steps 
ADD CONSTRAINT chk_step_completion_after_start 
CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at > started_at);

-- Add constraint to ensure content published time is valid
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_published_after_creation 
CHECK (published_at IS NULL OR published_at >= created_at);

-- Add constraint to prevent circular content relationships
ALTER TABLE campaign_content_relationships 
ADD CONSTRAINT chk_no_self_reference 
CHECK (source_content_id != target_content_id);

-- ################################################################################
-- PERFORMANCE OPTIMIZATIONS
-- ################################################################################

-- Partial index for active campaigns
CREATE INDEX idx_campaigns_active_status ON campaigns(status, priority, deadline) 
WHERE status IN ('active', 'scheduled') AND deadline IS NOT NULL;

-- Partial index for running workflows
CREATE INDEX idx_workflows_running_status ON campaign_workflows(status, started_at) 
WHERE status IN ('running', 'paused');

-- Partial index for pending workflow steps
CREATE INDEX idx_workflow_steps_pending ON campaign_workflow_steps(workflow_id, step_order) 
WHERE status = 'pending';

-- Composite index for campaign content queries
CREATE INDEX idx_campaign_content_campaign_status_type ON campaign_content(campaign_id, status, content_type);

-- Index for calendar events in date ranges
CREATE INDEX idx_campaign_calendar_date_range ON campaign_calendar(scheduled_datetime, event_type, status);

-- Index for performance analytics queries
CREATE INDEX idx_agent_perf_agent_date ON agent_orchestration_performance(agent_name, started_at DESC);

-- ################################################################################
-- COMMENTS AND DOCUMENTATION
-- ################################################################################

-- Table Comments
COMMENT ON TABLE campaigns IS 'Primary campaign entity - central to all content creation workflows';
COMMENT ON TABLE campaign_orchestrators IS 'Reusable workflow templates that define how campaigns are executed';
COMMENT ON TABLE campaign_strategies IS 'Strategic frameworks and templates for different types of campaigns';
COMMENT ON TABLE campaign_workflows IS 'Active workflow execution instances tied to specific campaigns';
COMMENT ON TABLE campaign_workflow_steps IS 'Individual steps within workflow executions with detailed tracking';
COMMENT ON TABLE campaign_content IS 'All content generated within campaigns with comprehensive metadata';
COMMENT ON TABLE campaign_calendar IS 'Unified calendar for all campaign activities and deadlines';
COMMENT ON TABLE campaign_analytics IS 'Performance metrics and KPIs tracked at campaign and content level';
COMMENT ON TABLE agent_orchestration_performance IS 'Detailed performance tracking for agents within orchestrated workflows';

-- Column Comments for key fields
COMMENT ON COLUMN campaigns.campaign_data IS 'Campaign-specific configuration data including topic, audience, channels, etc.';
COMMENT ON COLUMN campaign_orchestrators.workflow_definition IS 'Complete JSON workflow structure defining steps, dependencies, and agent assignments';
COMMENT ON COLUMN campaign_workflows.execution_context IS 'Runtime variables and state information for workflow execution';
COMMENT ON COLUMN campaign_content.generation_prompt IS 'Original prompt and context used for AI content generation';
COMMENT ON COLUMN campaign_analytics.attribution_data IS 'Multi-touch attribution data for campaign performance analysis';

-- ################################################################################
-- SCHEMA VERSION AND METADATA
-- ################################################################################

-- Schema version tracking
CREATE TABLE schema_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL,
    component VARCHAR(100) NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    description TEXT,
    migration_script VARCHAR(255)
);

INSERT INTO schema_versions (version, component, description, migration_script) 
VALUES ('1.0', 'campaign_orchestration', 'Initial campaign-centric schema with orchestration support', 'campaign_orchestration_schema.sql');

-- Schema validation function
CREATE OR REPLACE FUNCTION validate_campaign_orchestration_schema()
RETURNS BOOLEAN AS $$
DECLARE
    table_count INTEGER;
    trigger_count INTEGER;
    index_count INTEGER;
BEGIN
    -- Count core tables
    SELECT COUNT(*) INTO table_count 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN ('campaigns', 'campaign_orchestrators', 'campaign_workflows', 'campaign_workflow_steps', 'campaign_content', 'campaign_calendar', 'campaign_analytics');
    
    -- Count triggers
    SELECT COUNT(*) INTO trigger_count 
    FROM information_schema.triggers 
    WHERE trigger_schema = 'public' 
    AND trigger_name LIKE 'trigger_%campaign%';
    
    -- Count indexes
    SELECT COUNT(*) INTO index_count 
    FROM pg_indexes 
    WHERE schemaname = 'public' 
    AND indexname LIKE 'idx_%campaign%';
    
    -- Validate counts (adjust based on actual schema)
    RETURN table_count >= 7 AND trigger_count >= 3 AND index_count >= 20;
END;
$$ LANGUAGE plpgsql;

-- ================================================================================
-- END CAMPAIGN ORCHESTRATION SCHEMA
-- ================================================================================
-- This schema provides a complete foundation for campaign-centric content creation
-- with orchestrated multi-agent workflows, comprehensive analytics, and performance tracking.
-- ================================================================================