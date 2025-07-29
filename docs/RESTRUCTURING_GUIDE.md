# Project Restructuring Guide

## Overview
This document outlines the major restructuring of the CrediLinQ AI Content Platform for improved organization, maintainability, and developer experience.

## New Structure

```
credilinq-agent/
├── src/                          # Main application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py           # Centralized settings with Pydantic
│   │   └── database.py           # Database configuration
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   ├── routes/               # API endpoints organized by feature
│   │   │   ├── __init__.py
│   │   │   ├── blogs.py          # Blog management endpoints
│   │   │   ├── campaigns.py      # Campaign management endpoints
│   │   │   ├── analytics.py      # Analytics and metrics endpoints
│   │   │   └── health.py         # Health check endpoints
│   │   └── models/               # Pydantic models for API
│   │       ├── __init__.py
│   │       ├── blog.py           # Blog-related models
│   │       ├── campaign.py       # Campaign-related models
│   │       └── analytics.py      # Analytics-related models
│   └── agents/                   # AI Agents
│       ├── __init__.py
│       ├── core/                 # Core agent infrastructure
│       │   ├── __init__.py
│       │   ├── base_agent.py     # Base agent class
│       │   └── database_service.py # Enhanced database service
│       ├── workflow/             # Agent workflow orchestration
│       │   ├── __init__.py
│       │   └── blog_workflow.py  # Blog creation workflow
│       └── specialized/          # Specialized agents
│           ├── __init__.py
│           ├── content_agent.py  # Content generation
│           ├── campaign_manager.py # Campaign management
│           ├── image_agent.py    # Image prompt generation
│           ├── repurpose_agent.py # Content repurposing
│           └── search_agent.py   # Web search capabilities
├── database/                     # Database management
│   ├── migrations/               # Database migrations
│   ├── schema/                   # Database schemas
│   │   ├── current.sql          # Current complete schema
│   │   └── improvements.sql     # Latest improvements
│   └── scripts/                  # Database utilities
│       ├── migrate.py           # Migration runner
│       ├── setup_retriever.py   # RAG setup
│       └── test_access.py       # Database testing
├── tools/                        # Development and admin tools
│   ├── database/                # Database tools
│   │   ├── fix_permissions.py   # Permission fixes
│   │   └── diagnose.py          # Database diagnostics
│   ├── scripts/                 # Utility scripts
│   │   ├── create_blogs.py      # Blog creation helpers
│   │   ├── manual_tests.py      # Manual testing
│   │   └── debug_api.py         # API debugging
│   └── environment/             # Environment tools
│       └── check_env.py         # Environment validation
├── tests/                        # Test organization
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test fixtures
├── docs/                        # Documentation
│   ├── api/                     # API documentation
│   ├── deployment/              # Deployment guides
│   └── database/                # Database documentation
├── frontend/                    # React frontend (unchanged)
├── knowledge_base/              # RAG knowledge base
├── prisma/                      # Prisma configuration
├── .env.example                 # Environment template
├── pyproject.toml              # Modern Python project config
├── requirements.txt            # Python dependencies
└── README.md                   # Main documentation
```

## Key Improvements

### 1. **Organized Source Code**
- All application code moved to `src/` directory
- Clear separation of API, configuration, and agent logic
- Modular structure with proper Python packaging

### 2. **Centralized Configuration**
- `src/config/settings.py` - Single source of truth for all settings
- Environment variable management with Pydantic validation
- Database configuration abstraction

### 3. **Structured API Layer**
- Routes organized by feature (blogs, campaigns, analytics, health)
- Pydantic models separated by domain
- Clear separation of concerns

### 4. **Enhanced Agent Organization**
- Core infrastructure in `agents/core/`
- Workflow orchestration in `agents/workflow/`
- Specialized agents in `agents/specialized/`
- Proper import hierarchy

### 5. **Database Organization**
- All database files in `database/` directory
- Migrations, schemas, and scripts properly organized
- Clear versioning and deployment strategy

### 6. **Development Tools**
- Diagnostic and debugging tools in `tools/`
- Environment validation utilities
- Database administration tools

## Migration from Old Structure

### Import Changes
Old imports need to be updated:

```python
# OLD
from agent import app as blog_agent_app
from agents.integrations import db_service

# NEW  
from src.agents.workflow.blog_workflow import blog_agent_app
from src.agents.core.database_service import db_service
```

### Configuration Changes
Environment variables now managed centrally:

```python
# OLD
import os
SUPABASE_URL = os.getenv("SUPABASE_URL")

# NEW
from src.config import settings
supabase_url = settings.supabase_url
```

### Running the Application

```bash
# Development
python -m src.main

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Benefits

1. **Reduced Root Directory Clutter** - 25+ files reduced to essential project files
2. **Clear Module Dependencies** - Hierarchical import structure
3. **Easier Testing** - Organized test structure with proper fixtures
4. **Better Development Workflow** - Centralized tools and utilities
5. **Simplified Onboarding** - Clear project structure for new developers
6. **Modern Python Standards** - pyproject.toml, proper packaging
7. **Scalable Architecture** - Easy to add new features and agents

## Future Enhancements

### Phase 2 Improvements
- Docker containerization
- Advanced testing framework
- CI/CD pipeline configuration
- Performance monitoring integration

### Phase 3 Scalability
- Microservices architecture preparation
- Advanced logging and observability
- Multi-tenant support structure
- Production deployment optimization

## Troubleshooting

### Common Issues After Restructuring

1. **Import Errors**: Update all imports to use new `src.` prefix
2. **Path Issues**: Update any hardcoded file paths
3. **Environment Variables**: Ensure .env file is properly configured
4. **Database Scripts**: Use new `database/scripts/` location

### Testing the New Structure

```bash
# Test application startup
python -m src.main

# Test health endpoint
curl http://localhost:8000/health

# Test API endpoints
curl http://localhost:8000/api/blogs
```

## Conclusion

This restructuring provides a solid foundation for the CrediLinQ AI Content Platform, enabling better maintainability, clearer development workflows, and easier scaling as the platform grows.