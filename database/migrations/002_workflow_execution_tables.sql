-- Migration 002: Add workflow execution and orchestration tables
-- These tables support the Master Planner Agent and Workflow Executor functionality

-- Execution Plans table - stores Master Planner execution plans
CREATE TABLE IF NOT EXISTS execution_plans (
    id UUID PRIMARY KEY,
    campaign_id VARCHAR(255) NOT NULL,
    workflow_execution_id VARCHAR(255) UNIQUE NOT NULL,
    execution_strategy VARCHAR(50) NOT NULL DEFAULT 'adaptive',
    total_agents INTEGER NOT NULL,
    estimated_duration_minutes INTEGER,
    planned_sequence JSONB NOT NULL,
    agent_configurations JSONB DEFAULT '{}',
    created_by_agent VARCHAR(255) DEFAULT 'MasterPlannerAgent',
    planning_reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent Dependencies table - tracks agent execution dependencies  
CREATE TABLE IF NOT EXISTS agent_dependencies (
    id SERIAL PRIMARY KEY,
    execution_plan_id UUID REFERENCES execution_plans(id) ON DELETE CASCADE,
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(255) NOT NULL,
    depends_on_agents TEXT[] DEFAULT '{}',
    execution_order INTEGER NOT NULL,
    parallel_group_id INTEGER,
    is_parallel_eligible BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow State Live table - real-time workflow execution tracking
CREATE TABLE IF NOT EXISTS workflow_state_live (
    workflow_execution_id VARCHAR(255) PRIMARY KEY,
    current_step INTEGER DEFAULT 0,
    current_agents_running TEXT[] DEFAULT '{}',
    completed_agents TEXT[] DEFAULT '{}',
    failed_agents TEXT[] DEFAULT '{}',
    waiting_agents TEXT[] DEFAULT '{}',
    start_time TIMESTAMP,
    estimated_completion_time TIMESTAMP,
    actual_completion_time TIMESTAMP,
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    intermediate_outputs JSONB DEFAULT '{}',
    execution_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Execution Results table - stores final workflow results
CREATE TABLE IF NOT EXISTS workflow_execution_results (
    id SERIAL PRIMARY KEY,
    workflow_execution_id VARCHAR(255) UNIQUE NOT NULL,
    execution_plan_id UUID REFERENCES execution_plans(id),
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_agents INTEGER,
    successful_agents INTEGER,
    failed_agents INTEGER,
    execution_time_seconds INTEGER,
    final_outputs JSONB DEFAULT '{}',
    error_details JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent Execution Logs table - detailed agent execution tracking
CREATE TABLE IF NOT EXISTS agent_execution_logs (
    id SERIAL PRIMARY KEY,
    workflow_execution_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(255) NOT NULL,
    execution_order INTEGER,
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_seconds REAL,
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    dependencies_met BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_execution_plans_campaign_id ON execution_plans(campaign_id);
CREATE INDEX IF NOT EXISTS idx_execution_plans_workflow_id ON execution_plans(workflow_execution_id);
CREATE INDEX IF NOT EXISTS idx_agent_dependencies_plan_id ON agent_dependencies(execution_plan_id);
CREATE INDEX IF NOT EXISTS idx_agent_dependencies_order ON agent_dependencies(execution_order);
CREATE INDEX IF NOT EXISTS idx_workflow_state_heartbeat ON workflow_state_live(last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_workflow_results_status ON workflow_execution_results(status);
CREATE INDEX IF NOT EXISTS idx_agent_logs_workflow_id ON agent_execution_logs(workflow_execution_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_status ON agent_execution_logs(status);

-- Add update triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_execution_plans_updated_at BEFORE UPDATE ON execution_plans 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_workflow_state_updated_at BEFORE UPDATE ON workflow_state_live 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE execution_plans IS 'Master Planner execution plans with agent sequences and strategies';
COMMENT ON TABLE agent_dependencies IS 'Agent execution dependencies and ordering for workflows';
COMMENT ON TABLE workflow_state_live IS 'Real-time workflow execution state for live monitoring';
COMMENT ON TABLE workflow_execution_results IS 'Final results and performance metrics for completed workflows';
COMMENT ON TABLE agent_execution_logs IS 'Detailed execution logs for individual agents within workflows';

-- Insert sample data for testing (optional)
-- INSERT INTO execution_plans (id, campaign_id, workflow_execution_id, execution_strategy, total_agents, planned_sequence, planning_reasoning)
-- VALUES (
--     gen_random_uuid(),
--     'test_campaign_001',
--     'test_workflow_001',
--     'adaptive',
--     4,
--     '[{"agent_name": "planner", "agent_type": "PLANNER", "execution_order": 1, "dependencies": []}, {"agent_name": "researcher", "agent_type": "RESEARCHER", "execution_order": 2, "dependencies": ["planner"]}, {"agent_name": "writer", "agent_type": "WRITER", "execution_order": 3, "dependencies": ["planner", "researcher"]}, {"agent_name": "editor", "agent_type": "EDITOR", "execution_order": 4, "dependencies": ["writer"]}]',
--     'Test execution plan for development workflow validation'
-- );

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;