# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrediLinQ Content Agent is a full-stack AI-powered content management platform with:
- **Backend**: Python FastAPI application with multi-agent AI system
- **Frontend**: React/TypeScript SPA with Vite
- **Database**: PostgreSQL with Prisma ORM and vector extensions
- **AI Architecture**: LangGraph-based agent workflow for content creation and campaign management

## Development Commands

### Backend (Python)
```bash
# Start development server
python -m src.main

# Production server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Testing
pytest tests/

# Code quality
black src/ --line-length 88
flake8 src/
mypy src/
```

### Frontend (React/TypeScript)
```bash
cd frontend/
npm run dev          # Development server
npm run build        # Production build
npm run lint         # ESLint
npm run preview      # Preview production build
```

### Database
```bash
# Prisma operations
npx prisma generate          # Generate client
npx prisma db push          # Push schema changes
npx prisma migrate dev      # Create and apply migration
npx prisma studio          # Database GUI

# Database scripts
python database/scripts/migrate.py      # Run migrations
python database/scripts/test_access.py  # Test database connectivity
python tools/database/diagnose.py       # Database diagnostics
```

## Architecture

### Backend Structure (`src/`)
- **`main.py`** - FastAPI application entry point with CORS and routing
- **`config/`** - Centralized settings management with Pydantic validation
- **`api/routes/`** - Feature-organized endpoints (blogs, campaigns, analytics, health)
- **`api/models/`** - Pydantic models for API validation
- **`agents/`** - Multi-agent AI system:
  - `core/` - Base agent infrastructure and database service
  - `workflow/` - LangGraph orchestration for blog creation
  - `specialized/` - Domain-specific agents (content, campaign, image, search)

### Database Schema (PostgreSQL + Vector)
Key models in `prisma/schema.prisma`:
- **BlogPost** - Content with markdown, status tracking, and AI metadata
- **Campaign** - Marketing campaigns with associated tasks
- **CampaignTask** - Individual tasks (content repurposing, image generation)
- **AgentPerformance/AgentDecision** - AI agent tracking and analytics
- **Document/DocumentChunk** - RAG knowledge base with vector embeddings

### Frontend Architecture
- **React 18** with TypeScript and Vite
- **TailwindCSS** for styling
- **React Router** for navigation
- **Axios** for API communication
- Components organized by feature (Blog, Campaign, Analytics)

### AI Agent System
Multi-agent workflow using LangGraph:
1. **Content Agent** - Blog writing and optimization
2. **Campaign Manager** - Marketing strategy and planning
3. **Repurpose Agent** - Content adaptation for different formats
4. **Image Agent** - Visual content prompt generation
5. **Search Agent** - Web research capabilities

## Development Workflow

### Running the Full Stack
1. Start backend: `python -m src.main`
2. Start frontend: `cd frontend && npm run dev`
3. Access application at http://localhost:5173
4. API available at http://localhost:8000

### Database Development
- Schema changes: Edit `prisma/schema.prisma` then run `npx prisma db push`
- Migrations: Use `npx prisma migrate dev` for production-ready changes
- Vector search requires PostgreSQL with `vector` and `pg_trgm` extensions

### Agent Development
- Base agents extend `src/agents/core/base_agent.py`
- Database operations use `src/agents/core/database_service.py`
- Workflow orchestration in `src/agents/workflow/blog_workflow.py`

## Environment Configuration

Required environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `DATABASE_URL_DIRECT` - Direct database connection for Prisma
- `SUPABASE_URL` / `SUPABASE_ANON_KEY` - Supabase configuration
- `OPENAI_API_KEY` - OpenAI API for AI agents
- LangChain/LangGraph configuration for agent workflows

## Important Notes

- **Import Structure**: All backend imports use `src.` prefix (e.g., `from src.config import settings`)
- **Database Extensions**: Requires PostgreSQL with vector extensions for RAG functionality
- **Agent Performance**: All agent executions are tracked in `AgentPerformance` and `AgentDecision` tables
- **Content Versioning**: Blog posts support multiple variants for A/B testing
- **Deployment**: Configured for Vercel (frontend) with FastAPI backend deployment