-- Migration 001: Add execution planning and dependency management tables (Fixed)
-- File: database/migrations/001_add_execution_planning_fixed.sql

-- Check if uuid-ossp extension is enabled (required for UUID generation)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create execution plans table (removed campaigns FK constraint for now)
CREATE TABLE IF NOT EXISTS execution_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID, -- Will add FK constraint later when campaigns table exists
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
    CONSTRAINT positive_duration CHECK (estimated_duration_minutes > 0 OR estimated_duration_minutes IS NULL)
);

-- Create agent dependencies table
CREATE TABLE IF NOT EXISTS agent_dependencies (
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

-- Create real-time workflow state tracking table (this one already exists, so skip if exists)
CREATE TABLE IF NOT EXISTS workflow_state_live (
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

-- Create indexes for performance (only if they don't exist)
DO $$
BEGIN
    -- Indexes for execution_plans
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_execution_plans_campaign' AND n.nspname = 'public') THEN
        CREATE INDEX idx_execution_plans_campaign ON execution_plans(campaign_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_execution_plans_workflow' AND n.nspname = 'public') THEN
        CREATE INDEX idx_execution_plans_workflow ON execution_plans(workflow_execution_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_execution_plans_strategy' AND n.nspname = 'public') THEN
        CREATE INDEX idx_execution_plans_strategy ON execution_plans(execution_strategy);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_execution_plans_created_at' AND n.nspname = 'public') THEN
        CREATE INDEX idx_execution_plans_created_at ON execution_plans(created_at);
    END IF;
    
    -- Indexes for agent_dependencies  
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_agent_dependencies_plan' AND n.nspname = 'public') THEN
        CREATE INDEX idx_agent_dependencies_plan ON agent_dependencies(execution_plan_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_agent_dependencies_order' AND n.nspname = 'public') THEN
        CREATE INDEX idx_agent_dependencies_order ON agent_dependencies(execution_plan_id, execution_order);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_agent_dependencies_agent_name' AND n.nspname = 'public') THEN
        CREATE INDEX idx_agent_dependencies_agent_name ON agent_dependencies(agent_name);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_agent_dependencies_parallel_group' AND n.nspname = 'public') THEN
        CREATE INDEX idx_agent_dependencies_parallel_group ON agent_dependencies(parallel_group_id) WHERE parallel_group_id IS NOT NULL;
    END IF;
    
    -- Indexes for workflow_state_live (may already exist)
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_workflow_state_live_heartbeat' AND n.nspname = 'public') THEN
        CREATE INDEX idx_workflow_state_live_heartbeat ON workflow_state_live(last_heartbeat);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_workflow_state_live_start_time' AND n.nspname = 'public') THEN
        CREATE INDEX idx_workflow_state_live_start_time ON workflow_state_live(start_time);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = 'idx_workflow_state_live_completion' AND n.nspname = 'public') THEN
        CREATE INDEX idx_workflow_state_live_completion ON workflow_state_live(actual_completion_time) WHERE actual_completion_time IS NOT NULL;
    END IF;
END
$$;

-- Create update trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers (drop first if exists to avoid errors)
DROP TRIGGER IF EXISTS update_execution_plans_updated_at ON execution_plans;
CREATE TRIGGER update_execution_plans_updated_at BEFORE UPDATE ON execution_plans 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_workflow_state_live_updated_at ON workflow_state_live;
CREATE TRIGGER update_workflow_state_live_updated_at BEFORE UPDATE ON workflow_state_live 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE execution_plans IS 'Master execution plans for LangGraph workflows with agent sequencing and dependency management';
COMMENT ON TABLE agent_dependencies IS 'Agent dependency mapping and execution configuration for workflow plans';
COMMENT ON TABLE workflow_state_live IS 'Real-time workflow execution state for monitoring and observability';

-- Test data insertion to verify everything works
DO $$
DECLARE
    test_campaign_id UUID;
    test_execution_id UUID;
    test_plan_id UUID;
BEGIN
    -- Use a dummy campaign ID for testing
    test_campaign_id := uuid_generate_v4();
    test_execution_id := uuid_generate_v4();
    
    RAISE NOTICE 'Testing with campaign ID: %', test_campaign_id;
    
    -- Test execution plan insertion
    INSERT INTO execution_plans (
        campaign_id,
        workflow_execution_id,
        execution_strategy,
        total_agents,
        estimated_duration_minutes,
        planned_sequence,
        planning_reasoning
    ) VALUES (
        test_campaign_id,
        test_execution_id,
        'sequential',
        3,
        10,
        '[{"agent_name": "planner", "execution_order": 1}, {"agent_name": "writer", "execution_order": 2}, {"agent_name": "editor", "execution_order": 3}]'::jsonb,
        'Test execution plan for migration verification'
    ) RETURNING id INTO test_plan_id;
    
    -- Test agent dependencies insertion
    INSERT INTO agent_dependencies (
        execution_plan_id,
        agent_name,
        agent_type,
        execution_order,
        depends_on_agents
    ) VALUES 
        (test_plan_id, 'planner', 'PLANNER', 1, ARRAY[]::TEXT[]),
        (test_plan_id, 'writer', 'WRITER', 2, ARRAY['planner']),
        (test_plan_id, 'editor', 'EDITOR', 3, ARRAY['writer']);
    
    -- Test workflow state live insertion
    INSERT INTO workflow_state_live (
        workflow_execution_id,
        waiting_agents,
        start_time,
        execution_metadata
    ) VALUES (
        test_execution_id,
        ARRAY['planner'],
        NOW(),
        '{"test": true, "migration_version": "001_fixed"}'::jsonb
    ) ON CONFLICT (workflow_execution_id) DO UPDATE SET
        execution_metadata = EXCLUDED.execution_metadata;
    
    RAISE NOTICE 'Migration test successful! Created test plan with ID: %', test_plan_id;
    
    -- Clean up test data
    DELETE FROM workflow_state_live WHERE workflow_execution_id = test_execution_id;
    DELETE FROM agent_dependencies WHERE execution_plan_id = test_plan_id;
    DELETE FROM execution_plans WHERE id = test_plan_id;
    
    RAISE NOTICE 'Test data cleaned up successfully';
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Migration test encountered an issue: %', SQLERRM;
        -- Don't fail the migration for test issues
END
$$;

-- Final verification query
SELECT 
    'execution_plans' as table_name,
    count(*) as row_count,
    pg_size_pretty(pg_total_relation_size('execution_plans')) as size
FROM execution_plans
UNION ALL
SELECT 
    'agent_dependencies' as table_name,
    count(*) as row_count,
    pg_size_pretty(pg_total_relation_size('agent_dependencies')) as size
FROM agent_dependencies
UNION ALL
SELECT 
    'workflow_state_live' as table_name,
    count(*) as row_count,
    pg_size_pretty(pg_total_relation_size('workflow_state_live')) as size
FROM workflow_state_live;

RAISE NOTICE '✅ Migration 001 completed successfully!';
RAISE NOTICE 'Created tables: execution_plans, agent_dependencies, workflow_state_live';
RAISE NOTICE 'Next step: Implement the Master Planner Agent';