# Campaign Orchestration Database Relationships

## Overview
This document defines the comprehensive relationships between all entities in the campaign-centric architecture.

## Primary Entity Flow
```
CampaignStrategy → Campaign → CampaignWorkflow → CampaignWorkflowStep → AgentOrchestrationPerformance
                      ↓
                CampaignContent → CampaignAnalytics
                      ↓
                CampaignCalendar
```

## Detailed Relationships

### Core Campaign Flow
- **CampaignStrategy** (1) → **Campaign** (*)
  - One strategy can be used for multiple campaigns
  - Templates enable reusable campaign patterns

- **CampaignOrchestrator** (1) → **Campaign** (*)
  - One orchestrator can handle multiple campaigns
  - Defines the workflow template for execution

- **Campaign** (1) → **CampaignWorkflow** (*)
  - One campaign can have multiple workflow executions
  - Supports retry, versioning, and A/B testing

- **CampaignWorkflow** (1) → **CampaignWorkflowStep** (*)
  - Each workflow contains multiple ordered steps
  - Steps can have dependencies on other steps

### Content Management Flow
- **Campaign** (1) → **CampaignContent** (*)
  - All content is generated within campaign context
  - Content can be derived from other content

- **CampaignContent** (1) → **CampaignContent** (*) [Self-referencing]
  - Parent-child relationships for content derivation
  - Tracks repurposing and adaptation chains

- **CampaignContent** (1) → **CampaignContentRelationship** (*)
  - Many-to-many relationships between content pieces
  - Tracks references, dependencies, and connections

### Performance and Analytics Flow
- **Campaign** (1) → **CampaignAnalytics** (*)
  - Performance metrics tracked at campaign level
  - Time-series data for trend analysis

- **CampaignContent** (1) → **CampaignAnalytics** (*)
  - Individual content performance tracking
  - Platform-specific metrics and attribution

- **CampaignWorkflowStep** (1) → **AgentOrchestrationPerformance** (*)
  - Each step tracks agent performance
  - Cost, quality, and efficiency metrics

### Scheduling and Calendar Flow
- **Campaign** (1) → **CampaignCalendar** (*)
  - Campaign deadlines and milestones
  - Strategic planning events

- **CampaignContent** (1) → **CampaignCalendar** (*)
  - Content publishing schedules
  - Review and approval deadlines

- **CampaignWorkflow** (1) → **CampaignCalendar** (*)
  - Workflow execution milestones
  - Step completion deadlines

## Foreign Key Constraints

### Primary Relationships
```sql
-- Campaign relationships
ALTER TABLE campaigns 
  ADD CONSTRAINT fk_campaigns_strategy 
  FOREIGN KEY (strategy_id) REFERENCES campaign_strategies(id) ON DELETE SET NULL;

ALTER TABLE campaigns 
  ADD CONSTRAINT fk_campaigns_orchestrator 
  FOREIGN KEY (orchestrator_id) REFERENCES campaign_orchestrators(id) ON DELETE SET NULL;

-- Workflow relationships
ALTER TABLE campaign_workflows 
  ADD CONSTRAINT fk_workflows_campaign 
  FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;

ALTER TABLE campaign_workflows 
  ADD CONSTRAINT fk_workflows_orchestrator 
  FOREIGN KEY (orchestrator_id) REFERENCES campaign_orchestrators(id) ON DELETE RESTRICT;

ALTER TABLE campaign_workflow_steps 
  ADD CONSTRAINT fk_workflow_steps_workflow 
  FOREIGN KEY (workflow_id) REFERENCES campaign_workflows(id) ON DELETE CASCADE;

-- Content relationships
ALTER TABLE campaign_content 
  ADD CONSTRAINT fk_content_campaign 
  FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;

ALTER TABLE campaign_content 
  ADD CONSTRAINT fk_content_workflow 
  FOREIGN KEY (workflow_id) REFERENCES campaign_workflows(id) ON DELETE SET NULL;

ALTER TABLE campaign_content 
  ADD CONSTRAINT fk_content_parent 
  FOREIGN KEY (parent_content_id) REFERENCES campaign_content(id) ON DELETE SET NULL;

-- Calendar relationships
ALTER TABLE campaign_calendar 
  ADD CONSTRAINT fk_calendar_campaign 
  FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;

ALTER TABLE campaign_calendar 
  ADD CONSTRAINT fk_calendar_content 
  FOREIGN KEY (content_id) REFERENCES campaign_content(id) ON DELETE CASCADE;

ALTER TABLE campaign_calendar 
  ADD CONSTRAINT fk_calendar_workflow 
  FOREIGN KEY (workflow_id) REFERENCES campaign_workflows(id) ON DELETE SET NULL;

-- Analytics relationships
ALTER TABLE campaign_analytics 
  ADD CONSTRAINT fk_analytics_campaign 
  FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;

ALTER TABLE campaign_analytics 
  ADD CONSTRAINT fk_analytics_content 
  FOREIGN KEY (content_id) REFERENCES campaign_content(id) ON DELETE CASCADE;

-- Performance relationships
ALTER TABLE agent_orchestration_performance 
  ADD CONSTRAINT fk_performance_workflow 
  FOREIGN KEY (workflow_id) REFERENCES campaign_workflows(id) ON DELETE CASCADE;

ALTER TABLE agent_orchestration_performance 
  ADD CONSTRAINT fk_performance_step 
  FOREIGN KEY (step_id) REFERENCES campaign_workflow_steps(id) ON DELETE CASCADE;
```

## Relationship Types

### CASCADE Deletions
- When a **Campaign** is deleted, all related **CampaignContent**, **CampaignCalendar**, **CampaignAnalytics** are deleted
- When a **CampaignWorkflow** is deleted, all **CampaignWorkflowStep** and **AgentOrchestrationPerformance** are deleted

### SET NULL Deletions
- When a **CampaignStrategy** is deleted, related **Campaign** strategy_id is set to NULL
- When a **CampaignContent** parent is deleted, child parent_content_id is set to NULL

### RESTRICT Deletions
- **CampaignOrchestrator** cannot be deleted if it has active workflows
- Prevents accidental deletion of critical orchestration templates

## Data Integrity Rules

### Campaign Data Integrity
```sql
-- Ensure campaign deadlines are logical
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_timeline 
CHECK (
  deadline IS NULL OR 
  (scheduled_start IS NULL OR deadline > scheduled_start) AND 
  (actual_start IS NULL OR deadline > actual_start)
);

-- Ensure progress percentage is valid
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_progress 
CHECK (progress_percentage BETWEEN 0 AND 100);

-- Ensure priority is valid
ALTER TABLE campaigns 
ADD CONSTRAINT chk_campaign_priority 
CHECK (priority IN ('low', 'medium', 'high', 'critical', 'urgent'));
```

### Workflow Data Integrity
```sql
-- Ensure workflow completion is logical
ALTER TABLE campaign_workflows 
ADD CONSTRAINT chk_workflow_timing 
CHECK (
  completed_at IS NULL OR 
  started_at IS NULL OR 
  completed_at > started_at
);

-- Ensure step order is positive
ALTER TABLE campaign_workflow_steps 
ADD CONSTRAINT chk_step_order 
CHECK (step_order > 0);

-- Ensure step completion is logical
ALTER TABLE campaign_workflow_steps 
ADD CONSTRAINT chk_step_timing 
CHECK (
  completed_at IS NULL OR 
  started_at IS NULL OR 
  completed_at > started_at
);
```

### Content Data Integrity
```sql
-- Prevent self-referencing parent content
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_no_self_parent 
CHECK (id != parent_content_id);

-- Ensure published content has valid timing
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_publish_timing 
CHECK (
  published_at IS NULL OR 
  published_at >= created_at
);

-- Ensure scheduled publishing is in future when created
ALTER TABLE campaign_content 
ADD CONSTRAINT chk_content_schedule_timing 
CHECK (
  scheduled_publish_at IS NULL OR 
  scheduled_publish_at > created_at
);
```

## Performance Considerations

### Optimized Query Patterns
```sql
-- Campaign-first queries (most common)
SELECT c.*, co.name as orchestrator_name, cs.strategy_type
FROM campaigns c
LEFT JOIN campaign_orchestrators co ON c.orchestrator_id = co.id
LEFT JOIN campaign_strategies cs ON c.strategy_id = cs.id
WHERE c.status = 'active'
ORDER BY c.priority DESC, c.deadline ASC;

-- Content within campaign
SELECT cc.*, ca.views, ca.conversions
FROM campaign_content cc
LEFT JOIN campaign_analytics ca ON cc.id = ca.content_id
WHERE cc.campaign_id = $1
AND cc.is_active = true
ORDER BY cc.created_at DESC;

-- Workflow execution status
SELECT cw.*, COUNT(cws.id) as total_steps, 
       COUNT(cws.id) FILTER (WHERE cws.status = 'completed') as completed_steps
FROM campaign_workflows cw
LEFT JOIN campaign_workflow_steps cws ON cw.id = cws.workflow_id
WHERE cw.campaign_id = $1
GROUP BY cw.id
ORDER BY cw.created_at DESC;
```

### Index Strategy for Relationships
```sql
-- Campaign-centric indexes
CREATE INDEX idx_campaign_relationships ON campaigns(status, priority, deadline);
CREATE INDEX idx_content_campaign_active ON campaign_content(campaign_id, is_active, status);
CREATE INDEX idx_workflow_campaign_status ON campaign_workflows(campaign_id, status);
CREATE INDEX idx_calendar_campaign_date ON campaign_calendar(campaign_id, scheduled_datetime);
CREATE INDEX idx_analytics_campaign_date ON campaign_analytics(campaign_id, measurement_date DESC);

-- Performance tracking indexes
CREATE INDEX idx_agent_performance_workflow ON agent_orchestration_performance(workflow_id, started_at DESC);
CREATE INDEX idx_workflow_steps_execution ON campaign_workflow_steps(workflow_id, step_order, status);

-- Content relationship indexes
CREATE INDEX idx_content_parent_child ON campaign_content(parent_content_id, created_at);
CREATE INDEX idx_content_relationships_source ON campaign_content_relationships(source_content_id, relationship_type);
```

## Migration Considerations

### Legacy Relationship Mapping
```sql
-- Map existing blog posts to campaigns
UPDATE blog_posts 
SET campaign_id = (
  SELECT c.id FROM campaigns c 
  WHERE c.name = 'Legacy Blog Post Campaign' 
  LIMIT 1
)
WHERE campaign_id IS NULL;

-- Create default campaign content entries
INSERT INTO campaign_content (
  id, campaign_id, title, content_type, platform, content_markdown,
  status, created_at, updated_at, is_active
)
SELECT 
  bp.id, bp.campaign_id, bp.title, 'blog_post', 'website', bp.content_markdown,
  bp.status::TEXT, bp.created_at, bp.updated_at, true
FROM blog_posts bp
WHERE bp.campaign_id IS NOT NULL;
```

### Relationship Validation Queries
```sql
-- Validate campaign hierarchy
SELECT 
  c.id,
  c.name,
  COUNT(cw.id) as workflows,
  COUNT(cc.id) as content_pieces,
  COUNT(ce.id) as calendar_events
FROM campaigns c
LEFT JOIN campaign_workflows cw ON c.id = cw.campaign_id
LEFT JOIN campaign_content cc ON c.id = cc.campaign_id AND cc.is_active = true
LEFT JOIN campaign_calendar ce ON c.id = ce.campaign_id
GROUP BY c.id, c.name
HAVING COUNT(cw.id) = 0 OR COUNT(cc.id) = 0  -- Find incomplete campaigns
ORDER BY c.created_at DESC;

-- Validate workflow integrity
SELECT 
  cw.id,
  cw.status,
  COUNT(cws.id) as total_steps,
  COUNT(cws.id) FILTER (WHERE cws.status = 'completed') as completed_steps,
  COUNT(cws.id) FILTER (WHERE cws.status = 'failed') as failed_steps
FROM campaign_workflows cw
LEFT JOIN campaign_workflow_steps cws ON cw.id = cws.workflow_id
GROUP BY cw.id, cw.status
HAVING cw.status = 'completed' AND COUNT(cws.id) FILTER (WHERE cws.status = 'completed') != COUNT(cws.id)  -- Find inconsistent workflows
ORDER BY cw.started_at DESC;
```

## Relationship Documentation

### Entity Relationship Summary
- **8 core tables** with clear hierarchy
- **12 foreign key relationships** ensuring data integrity
- **3 self-referencing relationships** for content derivation
- **5 many-to-many relationships** through junction tables
- **Campaign as root entity** for all operations

### Relationship Benefits
1. **Data Consistency**: Foreign key constraints prevent orphaned records
2. **Query Optimization**: Indexes optimized for campaign-first queries
3. **Scalability**: Hierarchical structure supports large-scale campaigns
4. **Flexibility**: JSON fields allow for dynamic schema evolution
5. **Performance**: Strategic indexing for common query patterns