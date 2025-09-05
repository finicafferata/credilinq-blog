# 🚀 CrediLinq AI Content Platform - Deployment Guide

Complete deployment guide for the CrediLinq AI Content Platform with LangGraph workflows.

## 📋 Overview

This deployment includes:
- **FastAPI Backend** (Port 8000) - Main API and campaign management
- **LangGraph Workflows** (Port 8001) - AI agent orchestration and monitoring
- **React Frontend** (Port 5173/Production) - User interface
- **PostgreSQL Database** - Data persistence and vector storage

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend│    │  FastAPI Backend │    │ LangGraph API   │
│   (Port 5173)   │◄──►│   (Port 8000)    │◄──►│  (Port 8001)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────────────────────────┐
                       │         PostgreSQL Database         │
                       │      (with Vector Extensions)       │
                       └─────────────────────────────────────┘
```

## 🚀 Quick Deployment

### Option 1: Automated Deployment Script

```bash
# Make deployment script executable
chmod +x deploy-production.py

# Run automated deployment
python3 deploy-production.py
```

### Option 2: Manual Deployment Steps

1. **Prerequisites**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login to Railway
   railway login
   ```

2. **Deploy Backend + LangGraph**
   ```bash
   # Deploy main application
   railway up
   ```

3. **Deploy Frontend**
   ```bash
   cd frontend/
   railway up
   ```

## 📦 Files Overview

### Core Application Files
- `src/main.py` - FastAPI application entry point
- `langgraph_workflows.py` - LangGraph workflow definitions
- `langgraph.json` - LangGraph configuration
- `requirements-railway.txt` - Python dependencies

### Deployment Configuration
- `railway.toml` - Railway deployment configuration
- `Dockerfile` - Container configuration
- `scripts/start_production.py` - Production startup script
- `deploy-production.py` - Automated deployment script

### Frontend
- `frontend/src/` - React application source
- `frontend/railway.toml` - Frontend deployment config

## 🔧 Environment Variables

### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@host:port/db
DATABASE_URL_DIRECT=postgresql://user:pass@host:port/db

# AI API Keys
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
LANGCHAIN_API_KEY=your_langsmith_key  # Optional

# Application Settings
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PASSWORD=secure_password

# LangGraph Configuration
ENABLE_LANGGRAPH=true
LANGGRAPH_API_URL=http://localhost:8001

# Railway Specific
RAILWAY_FULL=true
ENABLE_AGENT_LOADING=true
```

### Optional Environment Variables

```bash
# Monitoring
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_ENABLED=true

# Email Configuration
SMTP_SERVER=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password

# Redis (Optional for caching)
REDIS_URL=redis://localhost:6379
```

## 🎯 Production Services

### Service 1: FastAPI Backend (Port 8000)

**Endpoints:**
- `GET /` - API root
- `GET /health/railway` - Health check
- `GET /docs` - API documentation  
- `POST /api/v2/campaigns/` - Create campaign
- `POST /api/v2/campaigns/autonomous/{id}/start` - Start campaign workflow

**Features:**
- Campaign management
- Content generation
- Agent performance tracking
- Database operations

### Service 2: LangGraph API (Port 8001)

**Endpoints:**
- `GET /docs` - LangGraph API docs
- `POST /runs/stream` - Stream workflow execution
- `GET /threads` - Get workflow threads
- `POST /assistants/search` - Search assistants

**Features:**
- Workflow orchestration
- Real-time agent monitoring
- State management
- Performance tracking

## 🌐 LangGraph Studio Integration

### Connecting to Production

1. **Access LangGraph Studio:**
   ```
   https://smith.langchain.com/studio/
   ```

2. **Connect to Your Deployment:**
   - Base URL: `https://your-app.railway.app:8001`
   - Or local: `http://localhost:8001`

3. **Monitor Workflows:**
   - View real-time agent execution
   - Inspect workflow states
   - Debug agent performance

## 📊 Monitoring & Health Checks

### Health Check Endpoints

```bash
# Backend health
curl https://your-app.railway.app/health/railway

# LangGraph health  
curl https://your-app.railway.app:8001/health

# Frontend health
curl https://your-frontend.railway.app/
```

### Monitoring Features

- **Agent Performance Tracking** - Built-in database logging
- **Workflow State Monitoring** - LangGraph Studio integration
- **API Analytics** - Request/response tracking
- **Error Monitoring** - Structured logging with Sentry
- **Resource Monitoring** - Railway dashboard metrics

## 🗄️ Database Setup

### Required Extensions

```sql
-- PostgreSQL extensions for vector search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Database Migration

```bash
# Generate Prisma client
npx prisma generate

# Push schema to database (development)
npx prisma db push

# Run migrations (production)
npx prisma migrate deploy
```

## 🛠️ Development vs Production

### Development
- Single service mode
- Hot reloading enabled
- Debug logging
- In-memory state management

### Production
- Multi-service deployment
- Optimized resource usage
- Structured logging
- Persistent state management
- Health monitoring

## 🔐 Security Configuration

### Authentication
- JWT-based authentication
- Secure password hashing (bcrypt)
- Admin user management
- API key validation

### CORS Configuration
- Frontend domain whitelisting
- Secure headers
- Request validation

### Database Security
- Connection pooling
- Parameterized queries
- Input validation
- SQL injection prevention

## 📈 Scaling Considerations

### Vertical Scaling (Railway)
- Memory: 512MB → 4GB
- CPU: 0.5 → 2.0 cores
- Storage: Auto-scaling

### Horizontal Scaling
- Multiple Railway services
- Load balancer configuration
- Database connection pooling
- Redis caching layer

## 🚨 Troubleshooting

### Common Issues

1. **LangGraph Service Not Starting**
   ```bash
   # Check logs
   railway logs --service your-backend-service
   
   # Verify port configuration
   echo $PORT
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connectivity
   python -c "from src.config.database import db_config; print(db_config.health_check())"
   ```

3. **Agent Loading Failures**
   ```bash
   # Check agent factory logs
   grep "Agent registration" /app/logs/application.log
   ```

4. **Memory Issues**
   ```bash
   # Increase Railway memory limit
   # Update railway.toml memoryLimit setting
   ```

### Debug Commands

```bash
# Check service status
railway status

# View logs in real-time  
railway logs --follow

# Access shell in deployed service
railway shell

# Check environment variables
railway variables
```

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangGraph Studio Guide](https://docs.langchain.com/langgraph-platform/langgraph-studio)

## 🎉 Post-Deployment

### Verification Steps

1. ✅ **Backend API responding** - `curl https://your-app.railway.app/health/railway`
2. ✅ **LangGraph API responding** - `curl https://your-app.railway.app:8001/docs`  
3. ✅ **Frontend loading** - Visit `https://your-frontend.railway.app`
4. ✅ **Database connected** - Check health endpoint response
5. ✅ **Agents loading** - Check orchestration dashboard
6. ✅ **Workflows executing** - Test campaign creation
7. ✅ **LangGraph Studio connected** - Monitor workflow runs

### Next Steps

1. **Configure Domain** - Add custom domain in Railway
2. **Set up Monitoring** - Configure Sentry, analytics
3. **Enable SSL** - Railway auto-provides HTTPS
4. **Add Team Members** - Invite collaborators to Railway project
5. **Set up CI/CD** - GitHub integration for auto-deploy
6. **Configure Backup** - Database backup strategy

---

🎯 **You're all set!** Your CrediLinq AI Content Platform with LangGraph workflows is now deployed and ready for production use!