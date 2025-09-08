# Database Enhancement Migrations for LangGraph Observability

## Migration Overview

This document contains SQL migration scripts to enhance your existing PostgreSQL database schema for improved LangGraph workflow observability and control.

## Prerequisites

- PostgreSQL with UUID extension enabled
- Existing CrediLinq database with current schema
- Database backup completed before migration

## Migration Scripts

### Migration 001: Execution Planning Tables

```sql
-- Migration 001: Add execution planning and dependency management tables
-- File: database/migrations/001_add_execution_planning.sql

-- Create execution plans table
CREATE TABLE execution_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    workflow_execution_id UUID UNIQUE NOT NULL,
    
    -- Execution configuration
    execution_strategy VARCHAR(50) NOT NULL DEFAULT 'sequential', 
    total_agents INTEGER NOT NULL,
    estimated_duration_minutes INTEGER,
    
    -- Plan data
    planned_sequence JSONB NOT NULL, -- Ordered execution plan with dependencies
    agent_configurations JSONB DEFAULT '{}'::JSONB, -- Agent-specific configurations
    
    -- State tracking
    current_step INTEGER DEFAULT 0,
    completed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    failed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Metadata
    created_by_agent VARCHAR(100) NOT NULL DEFAULT 'MasterPlannerAgent',
    planning_reasoning TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_execution_strategy CHECK (execution_strategy IN ('sequential', 'parallel', 'adaptive')),
    CONSTRAINT positive_total_agents CHECK (total_agents > 0),
    CONSTRAINT positive_duration CHECK (estimated_duration_minutes > 0)
);

-- Create agent dependencies table
CREATE TABLE agent_dependencies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_plan_id UUID NOT NULL REFERENCES execution_plans(id) ON DELETE CASCADE,
    
    -- Agent identification
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    
    -- Dependency configuration  
    depends_on_agents TEXT[] DEFAULT ARRAY[]::TEXT[], -- Prerequisite agents
    dependency_type VARCHAR(20) DEFAULT 'hard' CHECK (dependency_type IN ('hard', 'soft', 'optional')),
    execution_order INTEGER NOT NULL,
    
    -- Parallel execution
    parallel_group_id INTEGER, -- Agents in same group can run in parallel
    is_parallel_eligible BOOLEAN DEFAULT FALSE,
    
    -- Execution constraints
    max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0),
    timeout_minutes INTEGER DEFAULT 30 CHECK (timeout_minutes > 0),
    resource_requirements JSONB DEFAULT '{}'::JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT positive_execution_order CHECK (execution_order > 0),
    UNIQUE(execution_plan_id, agent_name), -- One entry per agent per plan
    UNIQUE(execution_plan_id, execution_order) -- Unique order within plan
);

-- Create real-time workflow state tracking table
CREATE TABLE workflow_state_live (
    workflow_execution_id UUID PRIMARY KEY,
    
    -- Current execution state
    current_agents_running TEXT[] DEFAULT ARRAY[]::TEXT[],
    current_step INTEGER DEFAULT 0,
    
    -- Completion tracking
    completed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    failed_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    waiting_agents TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Performance metrics
    start_time TIMESTAMPTZ,
    estimated_completion_time TIMESTAMPTZ,
    actual_completion_time TIMESTAMPTZ,
    
    -- State data
    execution_metadata JSONB DEFAULT '{}'::JSONB,
    intermediate_outputs JSONB DEFAULT '{}'::JSONB, -- Outputs from completed agents
    
    -- Heartbeat for monitoring
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_execution_plans_campaign ON execution_plans(campaign_id);
CREATE INDEX idx_execution_plans_workflow ON execution_plans(workflow_execution_id);
CREATE INDEX idx_execution_plans_strategy ON execution_plans(execution_strategy);
CREATE INDEX idx_execution_plans_created_at ON execution_plans(created_at);

CREATE INDEX idx_agent_dependencies_plan ON agent_dependencies(execution_plan_id);
CREATE INDEX idx_agent_dependencies_order ON agent_dependencies(execution_plan_id, execution_order);
CREATE INDEX idx_agent_dependencies_agent_name ON agent_dependencies(agent_name);
CREATE INDEX idx_agent_dependencies_parallel_group ON agent_dependencies(parallel_group_id) WHERE parallel_group_id IS NOT NULL;

CREATE INDEX idx_workflow_state_live_heartbeat ON workflow_state_live(last_heartbeat);
CREATE INDEX idx_workflow_state_live_start_time ON workflow_state_live(start_time);
CREATE INDEX idx_workflow_state_live_completion ON workflow_state_live(actual_completion_time) WHERE actual_completion_time IS NOT NULL;

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_execution_plans_updated_at BEFORE UPDATE ON execution_plans 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_state_live_updated_at BEFORE UPDATE ON workflow_state_live 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust role names as needed)
GRANT SELECT, INSERT, UPDATE, DELETE ON execution_plans TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_dependencies TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON workflow_state_live TO credilinq_app;
GRANT USAGE ON SEQUENCE execution_plans_id_seq TO credilinq_app;
GRANT USAGE ON SEQUENCE agent_dependencies_id_seq TO credilinq_app;

COMMENT ON TABLE execution_plans IS 'Master execution plans for LangGraph workflows with agent sequencing and dependency management';
COMMENT ON TABLE agent_dependencies IS 'Agent dependency mapping and execution configuration for workflow plans';
COMMENT ON TABLE workflow_state_live IS 'Real-time workflow execution state for monitoring and observability';
```

### Migration 002: Error Recovery and Classification

```sql
-- Migration 002: Add error recovery and classification tables
-- File: database/migrations/002_add_error_recovery.sql

-- Create error classification table
CREATE TABLE error_classifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_pattern TEXT NOT NULL, -- Regex or string pattern to match errors
    error_type VARCHAR(50) NOT NULL CHECK (error_type IN ('transient', 'configuration', 'data', 'critical', 'rate_limit', 'timeout', 'permission')),
    recovery_strategy VARCHAR(50) NOT NULL CHECK (recovery_strategy IN ('retry_immediate', 'retry_backoff', 'retry_exponential', 'reload_config', 'skip_agent', 'manual_intervention', 'alternative_agent')),
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    learning_source VARCHAR(50) DEFAULT 'automatic' CHECK (learning_source IN ('automatic', 'manual', 'ml_trained')),
    
    -- Recovery configuration
    max_retry_attempts INTEGER DEFAULT 3 CHECK (max_retry_attempts >= 0),
    retry_delay_seconds INTEGER DEFAULT 30 CHECK (retry_delay_seconds >= 0),
    backoff_multiplier FLOAT DEFAULT 2.0 CHECK (backoff_multiplier >= 1.0),
    
    -- Metadata
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0 CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    last_used TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(error_pattern, error_type) -- Prevent duplicate patterns
);

-- Create error recovery logs table
CREATE TABLE error_recovery_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_execution_id UUID REFERENCES workflow_state_live(workflow_execution_id) ON DELETE CASCADE,
    agent_execution_id UUID, -- From langgraph_agent_executions table
    
    -- Error details
    original_error TEXT NOT NULL,
    error_classification_id UUID REFERENCES error_classifications(id),
    error_type VARCHAR(50),
    
    -- Recovery attempt details
    recovery_strategy VARCHAR(50),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    recovery_success BOOLEAN,
    recovery_duration_ms INTEGER CHECK (recovery_duration_ms >= 0),
    
    -- Recovery outcome
    final_result VARCHAR(20) CHECK (final_result IN ('success', 'failed', 'skipped', 'manual_required')),
    recovery_notes TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_error_recovery_workflow (workflow_execution_id),
    INDEX idx_error_recovery_classification (error_classification_id),
    INDEX idx_error_recovery_created_at (created_at),
    INDEX idx_error_recovery_success (recovery_success, created_at)
);

-- Create workflow execution revisions table for plan updates
CREATE TABLE execution_plan_revisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_plan_id UUID NOT NULL REFERENCES execution_plans(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL CHECK (revision_number > 0),
    
    -- Change details
    change_reason VARCHAR(20) NOT NULL CHECK (change_reason IN ('agent_failure', 'dependency_update', 'optimization', 'manual_override', 'error_recovery')),
    change_description TEXT,
    
    -- Plan data
    old_sequence JSONB,
    new_sequence JSONB NOT NULL,
    affected_agents TEXT[],
    
    -- Performance impact
    old_estimated_duration INTEGER,
    new_estimated_duration INTEGER,
    
    -- Metadata
    revised_by_agent VARCHAR(100) DEFAULT 'MasterPlannerAgent',
    auto_generated BOOLEAN DEFAULT TRUE,
    
    revised_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(execution_plan_id, revision_number)
);

-- Create indexes
CREATE INDEX idx_error_classifications_pattern ON error_classifications USING gin(to_tsvector('english', error_pattern));
CREATE INDEX idx_error_classifications_type ON error_classifications(error_type, confidence_score DESC);
CREATE INDEX idx_error_classifications_success_rate ON error_classifications(success_rate DESC, usage_count DESC);

CREATE INDEX idx_execution_plan_revisions_plan ON execution_plan_revisions(execution_plan_id, revision_number);
CREATE INDEX idx_execution_plan_revisions_reason ON execution_plan_revisions(change_reason, revised_at);

-- Add triggers
CREATE TRIGGER update_error_classifications_updated_at BEFORE UPDATE ON error_classifications 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON error_classifications TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON error_recovery_logs TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON execution_plan_revisions TO credilinq_app;

-- Insert default error classifications
INSERT INTO error_classifications (error_pattern, error_type, recovery_strategy, confidence_score, learning_source) VALUES
    ('rate limit|too many requests|429', 'rate_limit', 'retry_exponential', 0.95, 'manual'),
    ('timeout|timed out|request timeout', 'timeout', 'retry_backoff', 0.90, 'manual'),
    ('connection|network|dns', 'transient', 'retry_immediate', 0.85, 'manual'),
    ('permission|unauthorized|403|401', 'permission', 'reload_config', 0.80, 'manual'),
    ('not found|404|does not exist', 'configuration', 'skip_agent', 0.75, 'manual'),
    ('out of memory|memory error|oom', 'critical', 'manual_intervention', 0.95, 'manual'),
    ('validation error|invalid input', 'data', 'skip_agent', 0.70, 'manual');

COMMENT ON TABLE error_classifications IS 'Machine learning enhanced error classification for automatic recovery strategy selection';
COMMENT ON TABLE error_recovery_logs IS 'Log of all error recovery attempts with success tracking for continuous improvement';
COMMENT ON TABLE execution_plan_revisions IS 'History of execution plan changes for audit and analysis';
```

### Migration 003: Enhanced Performance Analytics

```sql
-- Migration 003: Enhanced performance analytics and workflow optimization
-- File: database/migrations/003_add_performance_analytics.sql

-- Create workflow performance summary table
CREATE TABLE workflow_performance_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_execution_id UUID UNIQUE REFERENCES workflow_state_live(workflow_execution_id) ON DELETE CASCADE,
    campaign_id UUID REFERENCES campaigns(id) ON DELETE SET NULL,
    
    -- Execution metrics
    execution_strategy VARCHAR(50),
    total_agents INTEGER NOT NULL,
    successful_agents INTEGER DEFAULT 0,
    failed_agents INTEGER DEFAULT 0,
    
    -- Timing metrics
    planned_duration_minutes INTEGER,
    actual_duration_minutes INTEGER,
    duration_variance_percent FLOAT, -- (actual - planned) / planned * 100
    
    -- Quality metrics
    average_agent_quality_score FLOAT CHECK (average_agent_quality_score >= 0 AND average_agent_quality_score <= 1),
    overall_workflow_score FLOAT CHECK (overall_workflow_score >= 0 AND overall_workflow_score <= 1),
    
    -- Cost metrics  
    total_tokens_used INTEGER DEFAULT 0,
    estimated_cost_usd DECIMAL(10, 4) DEFAULT 0.0000,
    cost_per_agent_avg DECIMAL(10, 4) DEFAULT 0.0000,
    
    -- Error and recovery metrics
    error_count INTEGER DEFAULT 0,
    recovery_attempts INTEGER DEFAULT 0,
    successful_recoveries INTEGER DEFAULT 0,
    
    -- Efficiency metrics
    parallel_efficiency_score FLOAT, -- How well parallel execution was utilized
    dependency_satisfaction_rate FLOAT, -- % of dependencies satisfied on first attempt
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create agent performance benchmarks table
CREATE TABLE agent_performance_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    
    -- Performance benchmarks (rolling averages)
    avg_execution_time_seconds FLOAT,
    median_execution_time_seconds FLOAT,
    p95_execution_time_seconds FLOAT,
    
    -- Quality benchmarks
    avg_quality_score FLOAT,
    median_quality_score FLOAT,
    
    -- Reliability benchmarks
    success_rate FLOAT CHECK (success_rate >= 0 AND success_rate <= 1),
    failure_rate FLOAT CHECK (failure_rate >= 0 AND failure_rate <= 1),
    
    -- Cost benchmarks
    avg_tokens_per_execution FLOAT,
    avg_cost_per_execution DECIMAL(8, 4),
    
    -- Sample size and recency
    sample_size INTEGER NOT NULL CHECK (sample_size > 0),
    last_execution TIMESTAMPTZ,
    benchmark_window_days INTEGER DEFAULT 30,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(agent_name, agent_type, benchmark_window_days)
);

-- Create workflow optimization recommendations table
CREATE TABLE workflow_optimization_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_execution_id UUID REFERENCES workflow_state_live(workflow_execution_id) ON DELETE CASCADE,
    
    -- Recommendation details
    recommendation_type VARCHAR(50) NOT NULL CHECK (recommendation_type IN ('execution_order', 'parallel_optimization', 'agent_replacement', 'timeout_adjustment', 'cost_reduction', 'quality_improvement')),
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    
    -- Recommendation content
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    expected_improvement TEXT,
    implementation_effort VARCHAR(20) CHECK (implementation_effort IN ('low', 'medium', 'high')),
    
    -- Quantified benefits (optional)
    expected_time_savings_minutes INTEGER,
    expected_cost_savings_percent FLOAT,
    expected_quality_improvement_percent FLOAT,
    
    -- Implementation details
    affected_agents TEXT[],
    required_changes JSONB,
    
    -- Tracking
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'accepted', 'rejected', 'implemented')),
    implementation_notes TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create workflow comparison analysis table
CREATE TABLE workflow_comparison_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Workflows being compared
    baseline_workflow_id UUID NOT NULL,
    comparison_workflow_id UUID NOT NULL,
    
    -- Comparison metrics
    duration_improvement_percent FLOAT,
    cost_improvement_percent FLOAT,
    quality_improvement_percent FLOAT,
    success_rate_improvement_percent FLOAT,
    
    -- Detailed analysis
    agent_level_comparison JSONB, -- Detailed agent-by-agent comparison
    recommendation_summary TEXT,
    
    -- Metadata
    analysis_type VARCHAR(50) DEFAULT 'automatic' CHECK (analysis_type IN ('automatic', 'manual', 'a_b_test')),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(baseline_workflow_id, comparison_workflow_id)
);

-- Create indexes for performance
CREATE INDEX idx_workflow_performance_campaign ON workflow_performance_summary(campaign_id, created_at DESC);
CREATE INDEX idx_workflow_performance_strategy ON workflow_performance_summary(execution_strategy, overall_workflow_score DESC);
CREATE INDEX idx_workflow_performance_duration ON workflow_performance_summary(actual_duration_minutes, created_at DESC);
CREATE INDEX idx_workflow_performance_cost ON workflow_performance_summary(estimated_cost_usd DESC, created_at DESC);

CREATE INDEX idx_agent_benchmarks_agent ON agent_performance_benchmarks(agent_name, agent_type);
CREATE INDEX idx_agent_benchmarks_success_rate ON agent_performance_benchmarks(success_rate DESC, avg_execution_time_seconds);
CREATE INDEX idx_agent_benchmarks_updated ON agent_performance_benchmarks(updated_at DESC);

CREATE INDEX idx_optimization_recommendations_workflow ON workflow_optimization_recommendations(workflow_execution_id, priority, status);
CREATE INDEX idx_optimization_recommendations_type ON workflow_optimization_recommendations(recommendation_type, priority);
CREATE INDEX idx_optimization_recommendations_status ON workflow_optimization_recommendations(status, created_at DESC);

CREATE INDEX idx_workflow_comparison_baseline ON workflow_comparison_analysis(baseline_workflow_id);
CREATE INDEX idx_workflow_comparison_improvement ON workflow_comparison_analysis(duration_improvement_percent DESC, quality_improvement_percent DESC);

-- Add triggers
CREATE TRIGGER update_workflow_performance_summary_updated_at BEFORE UPDATE ON workflow_performance_summary 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_performance_benchmarks_updated_at BEFORE UPDATE ON agent_performance_benchmarks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_optimization_recommendations_updated_at BEFORE UPDATE ON workflow_optimization_recommendations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON workflow_performance_summary TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_performance_benchmarks TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON workflow_optimization_recommendations TO credilinq_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON workflow_comparison_analysis TO credilinq_app;

COMMENT ON TABLE workflow_performance_summary IS 'Comprehensive performance metrics for completed workflow executions';
COMMENT ON TABLE agent_performance_benchmarks IS 'Rolling performance benchmarks for individual agents to identify trends and outliers';
COMMENT ON TABLE workflow_optimization_recommendations IS 'AI-generated recommendations for workflow optimization based on performance analysis';
COMMENT ON TABLE workflow_comparison_analysis IS 'Comparative analysis between different workflow executions to identify best practices';
```

### Migration 004: Real-Time Monitoring Views

```sql
-- Migration 004: Create views and functions for real-time monitoring
-- File: database/migrations/004_add_monitoring_views.sql

-- Create comprehensive workflow status view
CREATE OR REPLACE VIEW v_workflow_status_dashboard AS
SELECT 
    wsl.workflow_execution_id,
    ep.campaign_id,
    c.name as campaign_name,
    ep.execution_strategy,
    ep.total_agents,
    wsl.current_step,
    
    -- Progress calculation
    CASE 
        WHEN ep.total_agents > 0 
        THEN ROUND((COALESCE(array_length(wsl.completed_agents, 1), 0)::FLOAT / ep.total_agents::FLOAT) * 100, 2)
        ELSE 0 
    END as progress_percentage,
    
    -- Agent counts
    COALESCE(array_length(wsl.current_agents_running, 1), 0) as agents_running,
    COALESCE(array_length(wsl.completed_agents, 1), 0) as agents_completed,
    COALESCE(array_length(wsl.failed_agents, 1), 0) as agents_failed,
    COALESCE(array_length(wsl.waiting_agents, 1), 0) as agents_waiting,
    
    -- Timing information
    wsl.start_time,
    wsl.estimated_completion_time,
    wsl.actual_completion_time,
    
    -- Calculate current status
    CASE 
        WHEN wsl.actual_completion_time IS NOT NULL THEN 'completed'
        WHEN COALESCE(array_length(wsl.current_agents_running, 1), 0) > 0 THEN 'running'
        WHEN COALESCE(array_length(wsl.failed_agents, 1), 0) = ep.total_agents THEN 'failed'
        WHEN COALESCE(array_length(wsl.completed_agents, 1), 0) + COALESCE(array_length(wsl.failed_agents, 1), 0) = ep.total_agents THEN 'completed'
        ELSE 'pending'
    END as workflow_status,
    
    -- Performance metrics
    CASE 
        WHEN wsl.start_time IS NOT NULL AND wsl.actual_completion_time IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (wsl.actual_completion_time - wsl.start_time)) / 60.0
        WHEN wsl.start_time IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (NOW() - wsl.start_time)) / 60.0
        ELSE NULL 
    END as elapsed_minutes,
    
    ep.estimated_duration_minutes,
    wsl.last_heartbeat,
    
    -- Health indicators
    CASE 
        WHEN wsl.last_heartbeat < NOW() - INTERVAL '2 minutes' THEN 'stale'
        WHEN COALESCE(array_length(wsl.failed_agents, 1), 0) > 0 THEN 'degraded'
        ELSE 'healthy'
    END as health_status,
    
    wsl.created_at as workflow_started_at

FROM workflow_state_live wsl
JOIN execution_plans ep ON wsl.workflow_execution_id = ep.workflow_execution_id
LEFT JOIN campaigns c ON ep.campaign_id = c.id
ORDER BY wsl.created_at DESC;

-- Create agent execution timeline view
CREATE OR REPLACE VIEW v_agent_execution_timeline AS
SELECT 
    lae.workflow_id,
    lae.agent_name,
    lae.agent_type,
    lae.step_name,
    lae.start_time,
    lae.end_time,
    lae.duration,
    lae.status,
    lae.input_tokens,
    lae.output_tokens,
    lae.total_tokens,
    lae.cost,
    lae.retry_count,
    
    -- Add dependency information
    ad.depends_on_agents,
    ad.execution_order,
    ad.parallel_group_id,
    
    -- Calculate relative timing
    LAG(lae.end_time) OVER (PARTITION BY lae.workflow_id ORDER BY ad.execution_order) as previous_agent_end_time,
    LEAD(lae.start_time) OVER (PARTITION BY lae.workflow_id ORDER BY ad.execution_order) as next_agent_start_time,
    
    -- Calculate wait time (time between dependency completion and start)
    CASE 
        WHEN LAG(lae.end_time) OVER (PARTITION BY lae.workflow_id ORDER BY ad.execution_order) IS NOT NULL
        THEN EXTRACT(EPOCH FROM (lae.start_time - LAG(lae.end_time) OVER (PARTITION BY lae.workflow_id ORDER BY ad.execution_order))) / 60.0
        ELSE NULL 
    END as wait_time_minutes

FROM langgraph_agent_executions lae
JOIN execution_plans ep ON lae.workflow_id = ep.workflow_execution_id  
JOIN agent_dependencies ad ON ep.id = ad.execution_plan_id AND lae.agent_name = ad.agent_name
ORDER BY lae.workflow_id, ad.execution_order;

-- Create performance analytics view
CREATE OR REPLACE VIEW v_workflow_performance_analytics AS
SELECT 
    DATE_TRUNC('day', wps.created_at) as execution_date,
    wps.execution_strategy,
    
    -- Volume metrics
    COUNT(*) as total_workflows,
    COUNT(*) FILTER (WHERE wps.successful_agents = wps.total_agents) as successful_workflows,
    COUNT(*) FILTER (WHERE wps.failed_agents > 0) as failed_workflows,
    
    -- Performance averages
    ROUND(AVG(wps.actual_duration_minutes), 2) as avg_duration_minutes,
    ROUND(AVG(wps.duration_variance_percent), 2) as avg_duration_variance_percent,
    ROUND(AVG(wps.average_agent_quality_score), 3) as avg_quality_score,
    
    -- Cost metrics
    ROUND(AVG(wps.total_tokens_used)) as avg_tokens_per_workflow,
    ROUND(AVG(wps.estimated_cost_usd), 4) as avg_cost_per_workflow,
    
    -- Error metrics
    ROUND(AVG(wps.error_count), 1) as avg_errors_per_workflow,
    ROUND(AVG(wps.successful_recoveries::FLOAT / NULLIF(wps.recovery_attempts, 0)), 3) as avg_recovery_rate

FROM workflow_performance_summary wps
WHERE wps.created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', wps.created_at), wps.execution_strategy
ORDER BY execution_date DESC;

-- Create agent performance trends view
CREATE OR REPLACE VIEW v_agent_performance_trends AS
SELECT 
    apb.agent_name,
    apb.agent_type,
    apb.avg_execution_time_seconds,
    apb.success_rate,
    apb.avg_quality_score,
    apb.avg_cost_per_execution,
    
    -- Trend calculation (compare with previous period)
    LAG(apb.avg_execution_time_seconds) OVER (
        PARTITION BY apb.agent_name, apb.agent_type 
        ORDER BY apb.updated_at
    ) as prev_avg_execution_time,
    
    LAG(apb.success_rate) OVER (
        PARTITION BY apb.agent_name, apb.agent_type 
        ORDER BY apb.updated_at
    ) as prev_success_rate,
    
    LAG(apb.avg_quality_score) OVER (
        PARTITION BY apb.agent_name, apb.agent_type 
        ORDER BY apb.updated_at
    ) as prev_avg_quality_score,
    
    -- Performance ranking within agent type
    ROW_NUMBER() OVER (
        PARTITION BY apb.agent_type 
        ORDER BY apb.success_rate DESC, apb.avg_execution_time_seconds ASC
    ) as performance_rank,
    
    apb.sample_size,
    apb.last_execution,
    apb.updated_at

FROM agent_performance_benchmarks apb
WHERE apb.benchmark_window_days = 30
ORDER BY apb.agent_type, performance_rank;

-- Create function to get real-time workflow progress
CREATE OR REPLACE FUNCTION get_workflow_progress(p_workflow_execution_id UUID)
RETURNS TABLE (
    workflow_execution_id UUID,
    progress_percentage NUMERIC,
    current_status TEXT,
    agents_running TEXT[],
    agents_completed TEXT[],
    agents_failed TEXT[],
    agents_waiting TEXT[],
    elapsed_minutes NUMERIC,
    estimated_completion_minutes INTEGER,
    health_status TEXT
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT 
        vws.workflow_execution_id,
        vws.progress_percentage,
        vws.workflow_status,
        wsl.current_agents_running,
        wsl.completed_agents,
        wsl.failed_agents,
        wsl.waiting_agents,
        vws.elapsed_minutes,
        vws.estimated_duration_minutes,
        vws.health_status
    FROM v_workflow_status_dashboard vws
    JOIN workflow_state_live wsl ON vws.workflow_execution_id = wsl.workflow_execution_id
    WHERE vws.workflow_execution_id = p_workflow_execution_id;
END;
$$;

-- Create function to update workflow heartbeat
CREATE OR REPLACE FUNCTION update_workflow_heartbeat(p_workflow_execution_id UUID)
RETURNS BOOLEAN LANGUAGE plpgsql AS $$
BEGIN
    UPDATE workflow_state_live 
    SET last_heartbeat = NOW()
    WHERE workflow_execution_id = p_workflow_execution_id;
    
    RETURN FOUND;
END;
$$;

-- Create function to calculate agent performance benchmarks
CREATE OR REPLACE FUNCTION refresh_agent_benchmarks(p_window_days INTEGER DEFAULT 30)
RETURNS INTEGER LANGUAGE plpgsql AS $$
DECLARE
    updated_count INTEGER := 0;
    agent_record RECORD;
BEGIN
    -- Calculate benchmarks for each agent
    FOR agent_record IN 
        SELECT DISTINCT agent_name, agent_type 
        FROM langgraph_agent_executions 
        WHERE start_time >= NOW() - (p_window_days || ' days')::INTERVAL
    LOOP
        INSERT INTO agent_performance_benchmarks (
            agent_name, 
            agent_type, 
            avg_execution_time_seconds,
            median_execution_time_seconds,
            p95_execution_time_seconds,
            avg_quality_score,
            success_rate,
            failure_rate,
            avg_tokens_per_execution,
            avg_cost_per_execution,
            sample_size,
            last_execution,
            benchmark_window_days
        )
        SELECT 
            agent_record.agent_name,
            agent_record.agent_type,
            AVG(lae.duration / 1000.0), -- Convert ms to seconds
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lae.duration / 1000.0),
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lae.duration / 1000.0),
            AVG((lae.metadata->>'quality_score')::FLOAT),
            AVG(CASE WHEN lae.status = 'success' THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN lae.status != 'success' THEN 1.0 ELSE 0.0 END),
            AVG(lae.total_tokens),
            AVG(lae.cost),
            COUNT(*),
            MAX(lae.start_time),
            p_window_days
        FROM langgraph_agent_executions lae
        WHERE lae.agent_name = agent_record.agent_name
          AND lae.agent_type = agent_record.agent_type
          AND lae.start_time >= NOW() - (p_window_days || ' days')::INTERVAL
        GROUP BY agent_record.agent_name, agent_record.agent_type
        
        ON CONFLICT (agent_name, agent_type, benchmark_window_days) 
        DO UPDATE SET
            avg_execution_time_seconds = EXCLUDED.avg_execution_time_seconds,
            median_execution_time_seconds = EXCLUDED.median_execution_time_seconds,
            p95_execution_time_seconds = EXCLUDED.p95_execution_time_seconds,
            avg_quality_score = EXCLUDED.avg_quality_score,
            success_rate = EXCLUDED.success_rate,
            failure_rate = EXCLUDED.failure_rate,
            avg_tokens_per_execution = EXCLUDED.avg_tokens_per_execution,
            avg_cost_per_execution = EXCLUDED.avg_cost_per_execution,
            sample_size = EXCLUDED.sample_size,
            last_execution = EXCLUDED.last_execution,
            updated_at = NOW();
            
        updated_count := updated_count + 1;
    END LOOP;
    
    RETURN updated_count;
END;
$$;

-- Grant permissions on views and functions
GRANT SELECT ON v_workflow_status_dashboard TO credilinq_app;
GRANT SELECT ON v_agent_execution_timeline TO credilinq_app;
GRANT SELECT ON v_workflow_performance_analytics TO credilinq_app;
GRANT SELECT ON v_agent_performance_trends TO credilinq_app;
GRANT EXECUTE ON FUNCTION get_workflow_progress(UUID) TO credilinq_app;
GRANT EXECUTE ON FUNCTION update_workflow_heartbeat(UUID) TO credilinq_app;
GRANT EXECUTE ON FUNCTION refresh_agent_benchmarks(INTEGER) TO credilinq_app;

COMMENT ON VIEW v_workflow_status_dashboard IS 'Real-time dashboard view of all workflow executions with health status';
COMMENT ON VIEW v_agent_execution_timeline IS 'Timeline view of agent executions within workflows showing dependencies and timing';
COMMENT ON VIEW v_workflow_performance_analytics IS 'Aggregated performance analytics for workflow execution trends';
COMMENT ON VIEW v_agent_performance_trends IS 'Performance trends and rankings for individual agents';
```

## Running the Migrations

### Prerequisites Check

```bash
# Check if uuid-ossp extension is available
psql -d your_database -c "SELECT * FROM pg_available_extensions WHERE name = 'uuid-ossp';"

# Enable the extension if not already enabled
psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
```

### Migration Execution Script

```bash
#!/bin/bash
# File: scripts/run_migrations.sh

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-credilinq}
DB_USER=${DB_USER:-credilinq_app}

MIGRATION_DIR="database/migrations"

echo "Starting LangGraph Observability migrations..."

# Migration 001
echo "Running Migration 001: Execution Planning Tables"
psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -f "$MIGRATION_DIR/001_add_execution_planning.sql"

# Migration 002  
echo "Running Migration 002: Error Recovery and Classification"
psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -f "$MIGRATION_DIR/002_add_error_recovery.sql"

# Migration 003
echo "Running Migration 003: Enhanced Performance Analytics"
psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -f "$MIGRATION_DIR/003_add_performance_analytics.sql"

# Migration 004
echo "Running Migration 004: Real-Time Monitoring Views"
psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -f "$MIGRATION_DIR/004_add_monitoring_views.sql"

echo "All migrations completed successfully!"

# Verify migrations
echo "Verifying new tables..."
psql -h $DB_HOST -p $DB_PORT -d $DB_NAME -U $DB_USER -c "
SELECT 
    schemaname,
    tablename,
    tableowner 
FROM pg_tables 
WHERE tablename IN (
    'execution_plans',
    'agent_dependencies', 
    'workflow_state_live',
    'error_classifications',
    'error_recovery_logs',
    'workflow_performance_summary'
) 
ORDER BY tablename;
"
```

### Rollback Scripts

Create rollback scripts for each migration:

```sql
-- File: database/rollbacks/rollback_001.sql
DROP TABLE IF EXISTS execution_plan_revisions;
DROP TABLE IF EXISTS agent_dependencies;
DROP TABLE IF EXISTS workflow_state_live;
DROP TABLE IF EXISTS execution_plans;

-- File: database/rollbacks/rollback_002.sql  
DROP TABLE IF EXISTS error_recovery_logs;
DROP TABLE IF EXISTS error_classifications;

-- File: database/rollbacks/rollback_003.sql
DROP TABLE IF EXISTS workflow_comparison_analysis;
DROP TABLE IF EXISTS workflow_optimization_recommendations;
DROP TABLE IF EXISTS agent_performance_benchmarks;
DROP TABLE IF EXISTS workflow_performance_summary;

-- File: database/rollbacks/rollback_004.sql
DROP FUNCTION IF EXISTS refresh_agent_benchmarks(INTEGER);
DROP FUNCTION IF EXISTS update_workflow_heartbeat(UUID);
DROP FUNCTION IF EXISTS get_workflow_progress(UUID);
DROP VIEW IF EXISTS v_agent_performance_trends;
DROP VIEW IF EXISTS v_workflow_performance_analytics;
DROP VIEW IF EXISTS v_agent_execution_timeline;
DROP VIEW IF EXISTS v_workflow_status_dashboard;
```

## Post-Migration Verification

```sql
-- Verify table creation and constraints
SELECT 
    table_name,
    table_type,
    is_insertable_into,
    is_typed
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name LIKE '%execution%' 
   OR table_name LIKE '%workflow%' 
   OR table_name LIKE '%agent%'
ORDER BY table_name;

-- Verify indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename IN (
    'execution_plans',
    'agent_dependencies',
    'workflow_state_live'
)
ORDER BY tablename, indexname;

-- Test data insertion
INSERT INTO execution_plans (
    campaign_id,
    workflow_execution_id,
    execution_strategy,
    total_agents,
    planned_sequence
) VALUES (
    uuid_generate_v4(),
    uuid_generate_v4(),
    'sequential',
    5,
    '[{"agent_name": "test", "execution_order": 1}]'::jsonb
);

-- Clean up test data
DELETE FROM execution_plans WHERE created_by_agent = 'test';
```

## Monitoring and Maintenance

### Daily Maintenance Tasks

```sql
-- Refresh agent performance benchmarks
SELECT refresh_agent_benchmarks(30);

-- Clean up old workflow states (keep 90 days)
DELETE FROM workflow_state_live 
WHERE created_at < NOW() - INTERVAL '90 days'
  AND actual_completion_time IS NOT NULL;

-- Update workflow heartbeats for active workflows
UPDATE workflow_state_live 
SET last_heartbeat = NOW() 
WHERE actual_completion_time IS NULL 
  AND last_heartbeat > NOW() - INTERVAL '1 hour';
```

### Performance Monitoring

```sql
-- Monitor table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables 
WHERE tablename IN (
    'execution_plans',
    'workflow_state_live', 
    'error_recovery_logs',
    'workflow_performance_summary'
)
ORDER BY size_bytes DESC;

-- Monitor query performance
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
WHERE query LIKE '%execution_plans%' 
   OR query LIKE '%workflow_state_live%'
ORDER BY mean_time DESC;
```

These migrations provide a solid foundation for enhanced LangGraph observability while maintaining compatibility with your existing system.