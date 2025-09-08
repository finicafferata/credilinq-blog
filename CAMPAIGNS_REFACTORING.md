# Campaign Routes Refactoring

## Overview
The large `src/api/routes/campaigns.py` file (4750+ lines) has been refactored into a modular structure for better maintainability and organization.

## New Structure

### Main Router
- `src/api/routes/campaigns_main.py` - Main router that aggregates all sub-modules

### Sub-modules
1. **`campaigns/workflow.py`** - Workflow Management
   - Active workflow monitoring
   - Workflow status tracking
   - WebSocket connections for real-time updates
   - WebSocket manager and connection handling

2. **`campaigns/crud.py`** - Campaign CRUD Operations
   - Campaign creation (standard and quick campaigns)
   - Campaign listing and retrieval
   - Campaign status updates
   - Background agent execution

3. **`campaigns/agents.py`** - Agent Operations
   - Agent insights and performance metrics
   - Agent analysis triggering
   - AI content recommendations
   - Agent performance tracking

4. **`campaigns/tasks.py`** - Task Management
   - Task status updates
   - Task creation and management
   - Task execution and monitoring
   - Task retry functionality

5. **`campaigns/orchestration.py`** - Campaign Orchestration
   - Orchestration dashboard
   - Campaign control (start/pause/resume/stop)
   - Agent performance metrics
   - AI insights and recommendations
   - Advanced agent rerun functionality

6. **`campaigns/scheduling.py`** - Scheduling & Distribution
   - Campaign scheduling
   - Post distribution
   - Performance tracking
   - Engagement metrics

7. **`campaigns/autonomous.py`** - Autonomous Workflows
   - Autonomous workflow management
   - Workflow status tracking
   - Workflow control (pause/resume/stop)

8. **`campaigns/testing.py`** - Testing & Debug Utilities
   - Test endpoints
   - Database connectivity tests
   - System information debugging
   - WebSocket testing
   - Performance statistics

## Benefits

### Maintainability
- Each module focuses on a specific domain
- Easier to locate and modify specific functionality
- Reduced cognitive load when working on specific features

### Team Collaboration
- Different team members can work on different modules simultaneously
- Clear separation of concerns reduces merge conflicts
- Easier code reviews with focused changes

### Testability
- Smaller modules are easier to unit test
- More focused test suites per module
- Better test isolation

### Performance
- Lazy loading of dependencies where appropriate
- More efficient imports and memory usage
- Better error isolation

## Migration Notes

### Import Changes
- Main router import changed from `campaigns` to `campaigns_main as campaigns`
- All existing API endpoints maintain the same URLs
- No breaking changes to API consumers

### File Organization
```
src/api/routes/
├── campaigns/
│   ├── __init__.py
│   ├── workflow.py
│   ├── crud.py
│   ├── agents.py
│   ├── tasks.py
│   ├── orchestration.py
│   ├── scheduling.py
│   ├── autonomous.py
│   └── testing.py
├── campaigns_main.py
└── campaigns.py (original - can be removed after testing)
```

### Endpoint Mapping
All original endpoints are preserved with the same paths:
- `/api/v2/campaigns/*` - All campaign endpoints
- WebSocket endpoints remain at `/ws/campaign/{campaign_id}/status`
- All orchestration endpoints under `/orchestration/`
- All autonomous workflow endpoints under `/autonomous/`

## Testing
- All existing tests should continue to work without modification
- New modular structure allows for more targeted testing
- Debug endpoints available for testing individual modules

## Future Enhancements
- Further sub-module division if needed
- Module-specific middleware
- Independent versioning of modules
- Module-level caching strategies

## Rollback Plan
If issues arise, the original `campaigns.py` file remains available and can be quickly restored by reverting the import change in `main.py`.