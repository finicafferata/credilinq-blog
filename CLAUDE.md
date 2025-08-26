# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrediLinq Content Agent is a full-stack AI-powered content management platform for B2B financial services with:
- **Backend**: Python 3.9+ FastAPI application with multi-agent AI system
- **Frontend**: React 18+ TypeScript SPA with Vite 5+
- **Database**: PostgreSQL with Prisma ORM, vector extensions, and dual schemas
- **AI Architecture**: LangGraph-based agent orchestration with performance tracking

## Development Commands

### Backend (Python)
```bash
# First-time setup: Configure secure admin credentials
python src/scripts/setup_admin.py

# Development server (primary command for backend)
python3 -m src.main             # Use python3 explicitly
python -m src.main              # Alternative if python3 not available

# Production deployment
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Testing (comprehensive test suite with pytest markers)
pytest tests/                                          # All tests
pytest -m unit                                         # Unit tests only
pytest -m integration                                  # Integration tests only
pytest -m "not slow"                                   # Skip slow tests
pytest -m database                                     # Database tests only
pytest -m agent                                        # AI agent tests
pytest tests/api/test_blogs.py::test_create_blog       # Single test
python tests/run_all_tests.py                          # Comprehensive test runner
python scripts/test_runner.py --all                    # All tests with frontend

# Code quality (configured in pyproject.toml)
black src/ --line-length 88     # Format code (line length matches config)
flake8 src/                     # Lint
mypy src/                        # Type checking
pip install -e .[dev]           # Install dev dependencies
```

### Frontend (React/TypeScript)
```bash
cd frontend/
npm run dev          # Development server (typically http://localhost:5173, may auto-increment)
npm run build        # Production build with TypeScript compilation via tsconfig.build.json
npm run lint         # ESLint with React-specific rules
npm run preview      # Preview production build
npm run test         # Run tests with Vitest
npm run test:ui      # Interactive test UI with Vitest
npm run test:coverage # Generate test coverage report
npm run vercel-build # Vercel-optimized build command
```

### Database Operations
```bash
# Prisma operations (dual schema setup)
npx prisma generate                     # Generate client (REQUIRED after schema changes)
npx prisma db push                      # Push schema changes (development only)
npx prisma migrate dev --name <name>    # Create migration (production-ready)
npx prisma studio                       # Database GUI tool

# Migration scripts and database management
python database/scripts/migrate.py               # Run custom migrations
python database/scripts/test_access.py           # Test database connectivity
python run_migration.py                          # Run specific migration files
PYTHONPATH=. python3 -c "import asyncio; ..."  # Database schema inspection
```

## Architecture

### Backend Structure (`src/`)
- **`main.py`** - FastAPI application entry point with CORS and routing
- **`config/`** - Centralized settings management with Pydantic validation
- **`api/`** - RESTful API layer
  - `routes/v1/` - Version 1 endpoints (deprecated)
  - `routes/v2/` - Current API endpoints (blogs, campaigns, analytics, health)
  - `models/` - Pydantic request/response models
- **`agents/`** - Multi-agent AI system:
  - `core/` - Base agent infrastructure (`base_agent.py`, `database_service.py`, `agent_factory.py`, `enhanced_agent_pool.py`)
  - `workflow/` - LangGraph orchestration (`blog_workflow.py`, `content_generation_workflow.py`, `task_management_system.py`)
  - `specialized/` - Domain-specific agents (content, campaign, image, SEO, social media)
  - `competitor_intelligence/` - Market analysis agents with orchestrator
  - `orchestration/` - Campaign orchestration and workflow state management
  - `communication/` - Inter-agent communication protocol and event system
  - `plugins/` - Plugin system for extensible agent functionality
- **`core/`** - Cross-cutting concerns (auth, monitoring, performance)
- **`services/`** - Business logic and external integrations

### Database Architecture (PostgreSQL + Vector)
**Dual Schema Design**: Core models in `prisma/schema.prisma`, orchestration in `prisma/campaign_orchestration.prisma`

**Core Models**:
- **BlogPost** - Content with markdown, status tracking, SEO/geo optimization, and word count analytics
- **Campaign** - Marketing campaigns with status management and blog post relationships  
- **CampaignTask** - Individual tasks (content repurposing, image generation, scheduling, distribution)
- **AgentPerformance/AgentDecision** - Comprehensive AI agent execution tracking and analytics
- **Document/DocumentChunk** - RAG knowledge base with PostgreSQL vector embeddings
- **Briefing** - Campaign briefing data with JSON metadata storage

**Orchestration Models** (Campaign Orchestration Schema):
- **Advanced Campaign Management** - Enhanced workflow orchestration and state tracking
- **Task Dependencies** - Hierarchical task system with scheduling and priority management
- **Agent Communication** - Inter-agent communication logs and message protocols
- **Performance Metrics** - Enhanced analytics and performance tracking

**Key Database Features**:
- Vector search with `uuid-ossp` extension for embeddings
- Comprehensive indexing on status, timestamps, and geo metadata
- JSON column support for flexible metadata storage
- Foreign key relationships with cascade options

### Frontend Architecture
- **React 18** with TypeScript and Vite
- **TailwindCSS** for styling
- **React Router** for navigation
- **Axios** for API communication (configured in `lib/api.ts`)
- **Zustand** for state management (`store/` directory)
- **Testing**: Vitest with React Testing Library and MSW for API mocking
- **Services**: Modular API services (`services/` - agentApi, orchestrationApi, contentWorkflowApi)
- **Hooks**: Custom React hooks for data fetching and state management
- Components organized by feature (Dashboard, Campaign, Analytics, Competitor Intelligence)

### AI Agent System
**LangGraph-Based Multi-Agent Architecture** with comprehensive orchestration and performance tracking:

**Agent Infrastructure**:
- **Base Agent** (`src/agents/core/base_agent.py`) - Abstract base class with status tracking and error handling
- **Agent Factory** (`src/agents/core/agent_factory.py`) - Dynamic agent creation and registration
- **Database Service** (`src/agents/core/database_service.py`) - Centralized database operations for agents
- **Enhanced Agent Pool** (`src/agents/core/enhanced_agent_pool.py`) - Agent lifecycle and resource management
- **Performance Tracker** - Automatic execution tracking in AgentPerformance/AgentDecision tables

**Core Content Pipeline Agents**:
1. **PlannerAgent** (`planner_agent.py`) - Content strategy and planning
2. **ResearcherAgent** (`researcher_agent.py`) - Web research and data gathering  
3. **WriterAgent** (`writer_agent.py`) - Content creation and writing
4. **EditorAgent** (`editor_agent.py`) - Content editing and quality assurance
5. **ContentAgent** (`content_agent.py`) - General content operations
6. **CampaignManager** (`campaign_manager.py`) - Campaign orchestration and management

**Specialized Function Agents**:
- **SEO Agent** (`seo_agent.py`) - Search engine optimization analysis
- **Image Agent** (`image_agent.py`) - Visual content and image prompt generation
- **Social Media Agent** (`social_media_agent.py`) - Platform-specific content adaptation
- **Content Repurposer** (`content_repurposer.py`) - Multi-format content adaptation
- **GEO Analysis Agent** (`geo_analysis_agent.py`) - Location-based content optimization
- **Search Agent** (`search_agent.py`) - Web research and competitive intelligence
- **AI Content Generator** (`ai_content_generator.py`) - Advanced template-based generation
- **Task Scheduler** (`task_scheduler.py`) - Automated workflow scheduling
- **Distribution Agent** (`distribution_agent.py`) - Multi-channel content distribution

**Workflow Orchestration**:
- **Blog Workflow** (`src/agents/workflow/blog_workflow.py`) - End-to-end blog creation pipeline
- **Content Generation Workflow** (`src/agents/workflow/content_generation_workflow.py`) - Advanced content workflows
- **Task Management System** (`src/agents/workflow/task_management_system.py`) - Hierarchical task coordination
- **Campaign Orchestrator** (`src/agents/orchestration/campaign_orchestrator.py`) - Campaign-level workflow management
- **Workflow State Manager** (`src/agents/orchestration/workflow_state_manager.py`) - State persistence and recovery

**Competitor Intelligence System**:
- **Orchestrator** (`competitor_intelligence_orchestrator.py`) - CI workflow coordination
- **Content Monitoring** (`content_monitoring_agent.py`) - Competitor content tracking
- **Performance Analysis** (`performance_analysis_agent.py`) - Competitive benchmarking
- **Trend Analysis** (`trend_analysis_agent.py`) - Market trend identification
- **Gap Identification** (`gap_identification_agent.py`) - Content opportunity analysis
- **Alert Orchestration** (`alert_orchestration_agent.py`) - Real-time competitive alerts

## Development Workflow

### Running the Full Stack
1. **Backend**: `python3 -m src.main` (defaults to port 8000)
2. **Frontend**: `cd frontend && npm run dev` (typically port 5173, may auto-increment to 5174+ if busy)
3. **Access Points**:
   - Frontend application: http://localhost:5173 (or shown port)
   - Backend API: http://localhost:8000
   - API documentation: http://localhost:8000/docs (Swagger UI)
   - Database GUI: `npx prisma studio` (usually port 5555)

### Database Development Workflow
**Dual Schema Management**: Core (`prisma/schema.prisma`) + Orchestration (`prisma/campaign_orchestration.prisma`)

**Schema Development**:
- Edit schemas → `npx prisma generate` → `npx prisma db push` (dev) OR `npx prisma migrate dev` (production)
- Always run `npx prisma generate` after schema changes (required for client updates)
- Vector search requires PostgreSQL with `vector`, `pg_trgm`, and `uuid-ossp` extensions
- Test connectivity: `python database/scripts/test_access.py`
- Complex migrations available in `database/migrations/` directory

**Migration Strategy**:
- Development: Use `npx prisma db push` for rapid prototyping
- Production: Use `npx prisma migrate dev --name descriptive_name` for versioned changes
- Custom migrations: Python scripts in `database/scripts/` and `run_migration.py`

### Agent Development Patterns
**Core Architecture**: All agents inherit from `BaseAgent` with automatic performance tracking

**Creating New Agents**:
1. Extend `src/agents/core/base_agent.py` 
2. Register in `src/agents/core/agent_factory.py`
3. Database operations via `src/agents/core/database_service.py`
4. Add to appropriate workflow in `src/agents/workflow/`

**Agent Communication**: 
- Inter-agent messaging via `src/agents/communication/` protocols
- Event bus for asynchronous communication (`event_bus.py`)
- Performance automatic via `AgentPerformanceTracker`

**Workflow Integration**:
- Blog workflows: `src/agents/workflow/blog_workflow.py` 
- Campaign orchestration: `src/agents/orchestration/campaign_orchestrator.py`
- Task management: `src/agents/workflow/task_management_system.py`
- State management: `src/agents/orchestration/workflow_state_manager.py`

## Environment Configuration

Required environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `DATABASE_URL_DIRECT` - Direct database connection for Prisma
- `SUPABASE_URL` / `SUPABASE_ANON_KEY` - Supabase configuration
- `OPENAI_API_KEY` - OpenAI API for AI agents
- `LANGCHAIN_API_KEY` - LangChain/LangSmith for agent monitoring (optional)

### Security Configuration (Recommended)
- `ADMIN_EMAIL` - Admin user email (default: admin@credilinq.com)
- `ADMIN_PASSWORD` - Secure admin password (auto-generated if not set)
- `SECRET_KEY` - Application secret key (auto-generated if not set)
- `JWT_SECRET` - JWT signing secret (auto-generated if not set)

**Note**: Use `python src/scripts/setup_admin.py` for secure credential generation.

## Package Management

### Python Dependencies
- **Core dependencies**: Listed in `pyproject.toml` under `[project.dependencies]`
- **Development dependencies**: Available as `pip install -e .[dev]`
- **Alternative**: Use `requirements.txt`, `requirements-dev.txt`, `requirements-ci.txt`
- **Virtual environment**: `python -m venv venv && source venv/bin/activate`

### Frontend Dependencies
- **Package manager**: npm (package.json)
- **Testing**: Vitest with React Testing Library and MSW for API mocking
- **Build**: TypeScript compilation with `tsconfig.build.json` for production builds

## Critical Development Notes

### Import and Module Structure
- **Backend imports**: Always use `src.` prefix (e.g., `from src.config import settings`)
- **Python version**: Use `python3` explicitly in commands (macOS/Linux compatibility)  
- **Package structure**: Organized by domain (`agents/`, `api/`, `core/`, `services/`)
- **Agent registration**: Import agents in `src/main.py` to trigger factory registration

### Database and Performance
- **PostgreSQL Extensions**: Requires `vector`, `pg_trgm`, and `uuid-ossp` extensions for full functionality
- **Dual Schema Design**: Core models + orchestration models for separation of concerns
- **Agent Performance**: ALL agent executions automatically tracked in `AgentPerformance`/`AgentDecision` tables
- **Vector Search**: Integrated PostgreSQL vector search for RAG knowledge base
- **Connection Management**: Enhanced connection pooling and lifecycle management

### Testing and Quality Assurance
- **Test Safety**: All tests use mocked database connections - production data is completely safe
- **Test Markers**: pytest markers available (`unit`, `integration`, `database`, `agent`, `slow`)
- **Comprehensive Coverage**: API, workflow, agent, and frontend testing suites
- **Quality Tools**: Black (88-char lines), Flake8, MyPy with strict typing, ESLint for frontend

### API and Versioning
- **Current API**: v2 endpoints are primary, v1 deprecated but functional
- **Documentation**: Auto-generated OpenAPI docs at `/docs` with FastAPI
- **Error Handling**: Enhanced middleware with structured error responses
- **Analytics**: Built-in API analytics and performance monitoring

### Deployment and Infrastructure
- **Frontend**: Vercel deployment with `vercel-build` script
- **Backend**: Railway deployment with `railway.toml` configuration  
- **Environment**: Auto-generated secure keys for development
- **Monitoring**: Built-in performance monitoring, alerting, and logging systems
- **Middleware**: Compression, caching, rate limiting, and CORS configured