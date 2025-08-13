# CrediLinq AI Content Platform

<div align="center">

![CrediLinq Logo](https://img.shields.io/badge/CrediLinq-AI%20Content%20Platform-blue?style=for-the-badge&logo=robot)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)

**AI-Powered Content Management Platform for B2B Financial Services**

[🚀 Live Demo](https://credilinq-blog.vercel.app) • [📖 Documentation](docs/) • [🐛 Report Bug](https://github.com/finicafferata/credilinq-blog/issues) • [💡 Request Feature](https://github.com/finicafferata/credilinq-blog/issues)

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [💻 Development Setup](#-development-setup)
- [🌐 Deployment](#-deployment)
- [📚 API Documentation](#-api-documentation)
- [🤖 AI Agents](#-ai-agents)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Overview

CrediLinq AI Content Platform is a sophisticated, AI-powered content management system designed specifically for B2B financial services companies. It leverages advanced AI agents to automate content creation, campaign management, and performance analytics.

### 🎯 Built For

- **Financial Technology Companies** - Embedded lending and B2B credit solutions
- **Digital Marketplaces** - Platform-integrated financial products
- **B2B Content Teams** - Automated content creation and optimization
- **Marketing Agencies** - Scalable content production for fintech clients

### 🏢 About CrediLinq

CrediLinq.ai is a global fintech leader in embedded lending and B2B credit solutions, operating across Southeast Asia, Europe, and the United States. We empower businesses to access funding through embedded financial products and cutting-edge credit infrastructure tailored to digital platforms and marketplaces.

---

## ✨ Features

### 🤖 AI-Powered Content Creation
- **Multi-Agent Workflow** - Planner, Researcher, Writer, Editor, SEO, and Social Media agents
- **Intelligent Research** - Automated market research and trend analysis
- **Content Optimization** - SEO analysis and optimization recommendations
- **Multiple Formats** - Blog posts, LinkedIn articles, social media content

### 📊 Campaign Management
- **Campaign Orchestration** - Multi-channel marketing campaign planning
- **Content Repurposing** - Automatic adaptation for different platforms
- **Scheduling & Distribution** - Automated posting and distribution
- **Performance Tracking** - Real-time analytics and insights

### 🎨 Visual Content Generation
- **AI Image Prompts** - Intelligent image prompt generation
- **Multiple Styles** - Professional, creative, minimalist, and corporate styles
- **Brand Consistency** - Maintains visual brand identity across content

### 📈 Advanced Analytics
- **Performance Metrics** - Comprehensive content and campaign analytics
- **Agent Insights** - AI agent performance tracking and optimization
- **ROI Analysis** - Marketing effectiveness measurement
- **Real-time Dashboards** - Live performance monitoring

### 🔧 Developer-Friendly
- **RESTful API** - Comprehensive API with OpenAPI documentation
- **Webhook Support** - Real-time event notifications
- **Rate Limiting** - Built-in API rate limiting and security
- **Version Control** - API versioning with backward compatibility

---

## 🏗️ Architecture

### 🖥️ Tech Stack

#### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Database**: PostgreSQL with vector extensions
- **ORM**: Prisma
- **AI/ML**: OpenAI GPT, LangChain, LangGraph
- **Authentication**: JWT, API Keys
- **Caching**: Redis (optional)
- **Monitoring**: Custom metrics and logging

#### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **State Management**: React Hooks and Context
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **Testing**: Vitest, React Testing Library

#### Infrastructure
- **Backend Hosting**: Railway
- **Frontend Hosting**: Vercel
- **Database**: Supabase PostgreSQL
- **CI/CD**: GitHub Actions
- **Monitoring**: Built-in application monitoring

### 🔄 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (Vercel)      │◄──►│   (Railway)     │◄──►│   (Supabase)    │
│                 │    │                 │    │                 │
│ React + TS      │    │ FastAPI + Py    │    │ PostgreSQL      │
│ TailwindCSS     │    │ AI Agents       │    │ Vector Ext.     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AI Services   │
                       │                 │
                       │ OpenAI GPT      │
                       │ LangChain       │
                       │ LangGraph       │
                       └─────────────────┘
```

### 🤖 AI Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Orchestrator                     │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   Planner   │ │Researcher │ │  Writer   │
        │   Agent     │ │  Agent    │ │  Agent    │
        └─────────────┘ └───────────┘ └───────────┘
                │             │             │
        ┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   Editor    │ │   SEO     │ │   Image   │
        │   Agent     │ │  Agent    │ │  Agent    │
        └─────────────┘ └───────────┘ └───────────┘
                │             │             │
        ┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │Social Media │ │ Campaign  │ │ Analytics │
        │   Agent     │ │ Manager   │ │  Agent    │
        └─────────────┘ └───────────┘ └───────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **PostgreSQL** 14+ (or Supabase account)
- **OpenAI API Key**

### 1. Clone the Repository

```bash
git clone https://github.com/finicafferata/credilinq-blog.git
cd credilinq-blog
```

### 2. Environment Setup

```bash
# Copy environment template
cp env.example .env

# Edit with your configuration
# Required: DATABASE_URL, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

### 3. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run database migrations
npx prisma generate
npx prisma db push

# Start backend server
python -m src.main
```

### 4. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 💻 Development Setup

### Backend Development

#### Project Structure
```
src/
├── agents/           # AI agent implementations
│   ├── core/        # Base agent classes and factory
│   ├── specialized/ # Specific agent implementations
│   └── workflow/    # Workflow orchestration
├── api/             # API routes and models
│   ├── routes/      # FastAPI route handlers
│   └── models/      # Pydantic models
├── config/          # Configuration management
├── core/            # Core utilities and middleware
└── main.py          # Application entry point
```

#### Running Backend

```bash
# Development with auto-reload
python -m src.main

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000

# With custom settings
ENVIRONMENT=development python -m src.main
```

#### Database Operations

```bash
# Generate Prisma client
npx prisma generate

# Push schema changes (development)
npx prisma db push

# Create migration (production)
npx prisma migrate dev --name description

# View database
npx prisma studio

# Reset database
npx prisma migrate reset
```

### Frontend Development

#### Project Structure
```
frontend/src/
├── components/      # Reusable UI components
├── pages/          # Page components
├── hooks/          # Custom React hooks
├── lib/            # Utilities and API client
├── types/          # TypeScript type definitions
└── data/           # Static data and templates
```

#### Running Frontend

```bash
cd frontend

# Development server
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Run tests
npm run test

# Lint code
npm run lint
```

### Development Commands

```bash
# Backend
python -m src.main                  # Start backend
pytest tests/                       # Run tests
black src/ --line-length 88         # Format code
flake8 src/                         # Lint code
mypy src/                           # Type checking

# Frontend
npm run dev                         # Start dev server
npm run build                       # Build for production
npm run test                        # Run tests
npm run lint                        # Lint code

# Database
npx prisma studio                   # Database GUI
npx prisma migrate dev              # Create migration
npx prisma db push                  # Push schema changes
```

---

## 🌐 Deployment

### Production Deployment

The application is configured for deployment on:
- **Frontend**: Vercel
- **Backend**: Railway
- **Database**: Supabase

#### Vercel Deployment (Frontend)

1. **Connect Repository**
   ```bash
   # Connect your GitHub repository to Vercel
   ```

2. **Environment Variables**
   ```bash
   VITE_API_BASE_URL=https://your-backend-url.railway.app
   ```

3. **Build Settings**
   ```json
   {
     "buildCommand": "cd frontend && npm run build",
     "outputDirectory": "frontend/dist"
   }
   ```

#### Railway Deployment (Backend)

1. **Connect Repository**
   ```bash
   # Connect your GitHub repository to Railway
   ```

2. **Environment Variables**
   ```bash
   DATABASE_URL=postgresql://user:password@host:port/database
   OPENAI_API_KEY=sk-...
   SUPABASE_URL=https://...
   SUPABASE_KEY=eyJ...
   ENVIRONMENT=production
   SECRET_KEY=your-production-secret
   JWT_SECRET=your-jwt-secret
   CORS_ORIGINS=https://your-frontend-domain.vercel.app
   ```

3. **Railway Configuration** (`railway.toml`)
   ```toml
   [build]
   command = "pip install -r requirements.txt"

   [deploy]
   startCommand = "uvicorn src.main:app --host 0.0.0.0 --port $PORT"
   ```

#### Database Setup (Supabase)

1. **Create Project**
   - Sign up at [Supabase](https://supabase.com)
   - Create a new project

2. **Enable Extensions**
   ```sql
   -- Enable vector extension for AI features
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_trgm;
   ```

3. **Run Migrations**
   ```bash
   npx prisma db push
   ```

### Docker Deployment (Optional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t credilinq-api .
docker run -p 8000:8000 --env-file .env credilinq-api
```

---

## 📚 API Documentation

### Authentication

All API endpoints require authentication using API keys or JWT tokens:

```bash
# API Key Authentication
curl -H "X-API-Key: your-api-key" https://api.credilinq.com/api/blogs

# JWT Token Authentication
curl -H "Authorization: Bearer your-jwt-token" https://api.credilinq.com/api/blogs
```

### Core Endpoints

#### Blog Management

```bash
# Create blog post
POST /api/blogs
{
  "title": "Your Blog Title",
  "company_context": "Company description",
  "content_type": "blog"
}

# Get all blogs
GET /api/blogs

# Get specific blog
GET /api/blogs/{blog_id}

# Update blog
PUT /api/blogs/{blog_id}
{
  "content_markdown": "Updated content"
}

# Delete blog
DELETE /api/blogs/{blog_id}
```

#### Workflow Management

```bash
# Start AI workflow
POST /api/workflow-fixed/start
{
  "title": "Blog Title",
  "company_context": "Context",
  "mode": "advanced"
}

# Check workflow status
GET /api/workflow-fixed/status/{workflow_id}
```

#### Campaign Management

```bash
# Create campaign
POST /api/campaigns
{
  "blog_id": "uuid",
  "campaign_name": "Campaign Name",
  "company_context": "Context"
}

# Get campaigns
GET /api/campaigns

# Update campaign
PUT /api/campaigns/{campaign_id}
```

#### Analytics

```bash
# Get dashboard analytics
GET /api/analytics/dashboard?days=30

# Get blog analytics
GET /api/blogs/{blog_id}/analytics

# Get agent performance
GET /api/analytics/agents?agent_type=writer&days=30
```

### API Versioning

The API supports multiple versions:
- **v2** (current): `/api/v2/...`
- **v1** (deprecated): `/api/v1/...`
- **Default**: `/api/...` (uses v2)

### Response Format

```json
{
  "data": { /* response data */ },
  "status": "success",
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0"
}
```

### Error Handling

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": { /* error details */ }
  },
  "status": "error",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 🤖 AI Agents

### Agent Types

#### 1. **Planner Agent**
- **Purpose**: Creates content outlines and structure
- **Input**: Title, company context, content type
- **Output**: Detailed content outline with sections

#### 2. **Researcher Agent**
- **Purpose**: Gathers relevant information and data
- **Input**: Content outline, research topics
- **Output**: Research data, statistics, examples

#### 3. **Writer Agent**
- **Purpose**: Generates high-quality content
- **Input**: Outline, research data, company context
- **Output**: Complete blog post in markdown format

#### 4. **Editor Agent**
- **Purpose**: Reviews and improves content quality
- **Input**: Draft content
- **Output**: Edited, polished content

#### 5. **SEO Agent**
- **Purpose**: Optimizes content for search engines
- **Input**: Content, target keywords
- **Output**: SEO recommendations and optimized content

#### 6. **Image Agent**
- **Purpose**: Generates image prompts and descriptions
- **Input**: Content, style preferences
- **Output**: Image prompts, alt text, descriptions

#### 7. **Social Media Agent**
- **Purpose**: Creates social media adaptations
- **Input**: Blog content
- **Output**: Social media posts, hashtags, captions

#### 8. **Campaign Manager Agent**
- **Purpose**: Orchestrates marketing campaigns
- **Input**: Content, campaign goals
- **Output**: Campaign strategy, timeline, tasks

### Agent Configuration

```python
# Custom agent configuration
from src.agents.core.agent_factory import create_agent, AgentType
from src.agents.core.base_agent import AgentMetadata

# Create specialized agent
writer_agent = create_agent(
    AgentType.WRITER,
    metadata=AgentMetadata(
        name="B2B Writer",
        description="Specialized in B2B financial content",
        max_retries=3,
        timeout_seconds=300
    )
)

# Execute agent
result = writer_agent.execute({
    "outline": ["Introduction", "Main Points", "Conclusion"],
    "research": {"industry_data": "..."},
    "company_context": "CrediLinq.ai context"
})
```

### Workflow Orchestration

```python
# Custom workflow
from src.agents.workflow.blog_workflow import BlogWorkflow

workflow = BlogWorkflow()
result = await workflow.execute({
    "title": "Blog Title",
    "company_context": "Company Context",
    "content_type": "blog"
})
```

---

## 🔧 Configuration

### Environment Variables

#### Required Variables

```bash
# Database Configuration
DATABASE_URL="postgresql://user:password@host:port/database"
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-supabase-anon-key"

# AI Configuration
OPENAI_API_KEY="sk-your-openai-api-key"

# Security
SECRET_KEY="your-secret-key"
JWT_SECRET="your-jwt-secret"
```

#### Optional Variables

```bash
# Application Settings
ENVIRONMENT="production"  # or "development"
API_VERSION="2.0.0"
DEBUG="false"

# CORS Settings
CORS_ORIGINS="https://yourapp.com,http://localhost:3000"

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# AI Services (Optional)
GOOGLE_API_KEY="your-google-api-key"
TAVILY_API_KEY="your-tavily-api-key"

# Monitoring
ENABLE_MONITORING="true"
LOG_LEVEL="INFO"
SENTRY_DSN="your-sentry-dsn"

# Cache Configuration
REDIS_HOST="localhost"
REDIS_PORT=6379
CACHE_TTL_SECONDS=300
```

---

## 🧪 Testing

### Backend Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# Run specific test file
pytest tests/unit/test_agents.py

# Run with verbose output
pytest -v tests/
```

### Frontend Testing

```bash
cd frontend

# Run all tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run tests in UI mode
npm run test:ui
```

### Test Structure

```
tests/
├── conftest.py          # Test configuration
├── fixtures/            # Test data fixtures
├── unit/               # Unit tests
│   ├── test_agents.py  # Agent tests
│   └── test_database.py # Database tests
├── integration/        # Integration tests
│   ├── test_api_blogs.py
│   └── test_api_campaigns.py
└── e2e/               # End-to-end tests
```

---

## 🤝 Contributing

We welcome contributions to the CrediLinq AI Content Platform! Please follow these guidelines:

### Development Process

1. **Fork the Repository**
   ```bash
   git fork https://github.com/finicafferata/credilinq-blog.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow coding standards
   - Add tests for new features
   - Update documentation

4. **Commit Changes**
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Coding Standards

#### Python (Backend)
- **Formatting**: Use `black` with line length 88
- **Linting**: Use `flake8`
- **Type Hints**: Use `mypy` for type checking
- **Docstrings**: Use Google-style docstrings

#### TypeScript (Frontend)
- **Formatting**: Use Prettier
- **Linting**: Use ESLint
- **Types**: Prefer explicit typing
- **Components**: Use functional components with hooks

---

## 🚨 Troubleshooting

### Common Issues

#### Backend Issues

**Agent Registration Errors**
```bash
ERROR: Agent type 'writer' is not registered
```
**Solution**: Ensure agents are imported in `src/main.py`

**Database Connection Issues**
```bash
ERROR: connection to server failed
```
**Solution**: Check DATABASE_URL and network connectivity

**OpenAI API Errors**
```bash
ERROR: OpenAI API rate limit exceeded
```
**Solution**: Check API key and usage limits

#### Frontend Issues

**API Connection Issues**
```bash
Failed to fetch: localhost:8000
```
**Solution**: Check VITE_API_BASE_URL environment variable

#### Deployment Issues

**Railway Deployment**
```bash
Build failed: requirements.txt not found
```
**Solution**: Ensure requirements.txt is in root directory

**Vercel Deployment**
```bash
Build command failed
```
**Solution**: Check build configuration and environment variables

---

## 📈 Performance & Monitoring

### Performance Metrics

- **API Response Time**: < 200ms average
- **Agent Processing**: < 30s per workflow
- **Database Queries**: Optimized with indexing
- **Frontend Load Time**: < 2s initial load

### Monitoring

```bash
# Health check endpoint
GET /health

# System metrics
GET /api/analytics/system

# Agent performance
GET /api/analytics/agents
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **LangChain** for AI orchestration
- **FastAPI** for backend framework
- **React** for frontend framework
- **Vercel** for frontend hosting
- **Railway** for backend hosting
- **Supabase** for database services

---

## 📞 Support

- **Email**: support@credilinq.ai
- **GitHub Issues**: [Report bugs or request features](https://github.com/finicafferata/credilinq-blog/issues)
- **Documentation**: [Comprehensive guides](docs/)

---

<div align="center">

**Made with ❤️ by the CrediLinq Team**

[🌐 Website](https://credilinq.ai) • [📧 Contact](mailto:hello@credilinq.ai) • [🐦 Twitter](https://twitter.com/credilinq)

</div>