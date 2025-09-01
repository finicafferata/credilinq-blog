# Railway Deployment Guide for CrediLinq AI Platform

## Overview

This guide provides step-by-step instructions for deploying the CrediLinq AI Content Platform to Railway, including environment setup, configuration, and troubleshooting.

## ✅ Pre-Deployment Status

Based on enhanced diagnostics testing, your application is **ready for Railway deployment**:

- ✅ **Full agent system loads successfully** (1.3s, 350MB memory)
- ✅ **18 AI agents registered** with LangGraph workflows
- ✅ **Database connectivity healthy** with PostgreSQL
- ✅ **LangGraph compatibility layer working** 
- ✅ **Memory usage well within Railway limits** (17% of 2GB limit)

## Required Environment Variables

### Critical Variables (Must Set)
```bash
# Database Connection
DATABASE_URL=postgresql://username:password@host:port/database
DATABASE_URL_DIRECT=postgresql://username:password@host:port/database

# AI Services
OPENAI_API_KEY=your_openai_api_key_here

# Railway Configuration
RAILWAY_FULL=true
ENABLE_AGENT_LOADING=true
PORT=8080
ENVIRONMENT=production

# Application Security
SECRET_KEY=your_secure_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

### Optional But Recommended
```bash
# Railway Environment Detection
RAILWAY_ENVIRONMENT=production
RAILWAY_SERVICE_NAME=credilinq-ai-platform

# Additional AI Providers (optional)
GEMINI_API_KEY=your_gemini_key_here
LANGCHAIN_API_KEY=your_langchain_key_here

# Application Configuration
DEBUG=false
ENABLE_CACHE=true
ENABLE_ANALYTICS=true
ENABLE_PERFORMANCE_TRACKING=true
```

## Railway Configuration

### 1. Resource Limits (railway.toml)

Your current configuration is appropriate:

```toml
[deploy]
healthcheckPath = "/health/railway"
healthcheckTimeout = 120
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 5

# Production memory allocation
[environments.production]
memoryLimit = "2GB"     # Full system needs ~350MB, 2GB provides headroom
cpuLimit = "1500m"      # Sufficient for AI workloads
```

### 2. Startup Command

Your startup script is properly configured:
```toml
startCommand = "python /app/scripts/start_railway.py"
```

### 3. Health Check Endpoint

The application provides Railway-optimized health checks:
- **Primary**: `/health/railway` (configured in railway.toml)
- **Fallback**: `/health/live`, `/ping`

## Deployment Steps

### Step 1: Set Environment Variables in Railway

1. **Access Railway Dashboard**
   ```
   railway login
   railway link [your-project-id]
   ```

2. **Set Critical Variables**
   ```bash
   railway env set DATABASE_URL="your_postgresql_connection_string"
   railway env set OPENAI_API_KEY="your_openai_api_key"
   railway env set RAILWAY_FULL="true"
   railway env set ENABLE_AGENT_LOADING="true"
   railway env set ENVIRONMENT="production"
   ```

3. **Set Security Variables**
   ```bash
   railway env set SECRET_KEY="$(openssl rand -base64 32)"
   railway env set JWT_SECRET="$(openssl rand -base64 32)"
   ```

### Step 2: Database Setup

Your PostgreSQL database must have these extensions:
```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Step 3: Deploy

```bash
railway deploy
```

### Step 4: Verify Deployment

1. **Check Health Endpoint**
   ```bash
   curl https://your-app.railway.app/health/railway
   ```

2. **Expected Response** (Full System):
   ```json
   {
     "message": "CrediLinq AI Content Platform API",
     "version": "2.0.0",
     "status": "operational",
     "environment": "production"
   }
   ```

3. **If Simple Mode** (indicates failure):
   ```json
   {
     "message": "CrediLinq AI Platform (Railway Simple)",
     "version": "2.0.0-railway-simple",
     "status": "operational"
   }
   ```

## Troubleshooting

### Issue: Application Starts in Simple Mode

**Symptoms**: API returns "Railway Simple" in response

**Diagnosis Steps**:
1. Check Railway logs for import errors
2. Run diagnostics script locally:
   ```bash
   python3 scripts/railway_debug_enhanced.py
   ```

**Common Causes & Solutions**:

#### 1. Missing Environment Variables
```bash
railway env set RAILWAY_FULL="true"
railway env set ENABLE_AGENT_LOADING="true"
```

#### 2. Database Connection Issues
- Verify `DATABASE_URL` format: `postgresql://user:pass@host:port/db`
- Ensure database has required extensions
- Test connection with simple query

#### 3. Memory/Resource Constraints
- Current memory usage: ~350MB (well within 2GB limit)
- Monitor Railway metrics dashboard
- Check for memory spikes during startup

### Issue: Import Failures

**Check Railway Logs**:
```bash
railway logs --tail
```

**Look for**:
- Module import errors
- LangGraph compatibility issues
- Agent registration failures

### Issue: Database Problems

**Symptoms**: Database connection errors in logs

**Solutions**:
1. **Check Connection String Format**:
   ```bash
   # Railway format
   postgresql://username:password@host:port/database
   
   # If using postgres:// (needs conversion)
   DATABASE_URL=${DATABASE_URL/postgres:\/\//postgresql:\/\/}
   ```

2. **Verify Extensions**:
   ```sql
   SELECT * FROM pg_extension WHERE extname IN ('uuid-ossp', 'vector', 'pg_trgm');
   ```

3. **Test Connection Health**:
   ```bash
   curl https://your-app.railway.app/health/railway
   ```

## Performance Optimization

### Memory Management
- **Current Usage**: ~350MB for full system
- **Railway Limit**: 2GB production
- **Headroom**: 83% available

### Startup Optimization
- Agent loading time: ~1.3 seconds
- Database connections: Pooled (max 30)
- Health check timeout: 120 seconds

### Monitoring
Your application includes built-in monitoring:
- **Health checks**: Multiple endpoints
- **Performance tracking**: Agent execution metrics
- **Error handling**: Comprehensive logging
- **Graceful degradation**: Fallback to simple mode

## Advanced Configuration

### Custom Startup Options

Environment variables for fine-tuning:

```bash
# Force simple mode (for testing)
RAILWAY_SIMPLE=true

# Custom resource settings
WORKERS=1                    # Single worker for Railway
MAX_DB_CONNECTIONS=30        # Database pool size
HEALTH_CHECK_TIMEOUT=120     # Health check timeout

# Feature toggles
ENABLE_ANALYTICS=true
ENABLE_PERFORMANCE_TRACKING=true
ENABLE_CACHE=true
```

### Load Balancing

Railway automatically handles load balancing. For high-traffic scenarios:

```toml
# railway.toml
[environments.production]
replicas = 2              # Multiple instances
memoryLimit = "2GB"       # Per instance
cpuLimit = "1500m"        # Per instance
```

## Success Indicators

### Full System Running
- ✅ Health check returns "CrediLinq AI Content Platform API"
- ✅ Version shows "2.0.0" (not "railway-simple")
- ✅ Status is "operational"
- ✅ 18 AI agents available in logs
- ✅ Database connection healthy
- ✅ Response time under 2 seconds

### Performance Metrics
- **Memory Usage**: ~350MB (17% of limit)
- **Startup Time**: ~1.3 seconds
- **Agent Loading**: ~18 types registered
- **Database Health**: PostgreSQL with extensions

## Support and Debugging

### Diagnostic Tools

1. **Enhanced Diagnostics** (run locally):
   ```bash
   python3 scripts/railway_debug_enhanced.py
   ```

2. **Railway Logs**:
   ```bash
   railway logs --tail
   railway logs --filter="ERROR"
   ```

3. **Health Monitoring**:
   ```bash
   # Test all health endpoints
   curl https://your-app.railway.app/health/railway
   curl https://your-app.railway.app/health/live
   curl https://your-app.railway.app/ping
   ```

### Common Solutions

| Issue | Quick Fix |
|-------|-----------|
| Simple mode fallback | Set `RAILWAY_FULL=true` |
| Import errors | Check Railway logs for missing dependencies |
| Database errors | Verify connection string and extensions |
| Memory issues | Monitor Railway metrics (unlikely with current usage) |
| Slow startup | Increase health check timeout to 120s |

## Conclusion

Your application is well-prepared for Railway deployment with:
- ✅ Comprehensive agent system (18 types)
- ✅ Optimized memory usage (350MB)
- ✅ Robust fallback mechanisms
- ✅ Health monitoring and diagnostics
- ✅ Production-ready configuration

The main requirement is setting the proper environment variables, particularly `RAILWAY_FULL=true` and ensuring your database connection is properly configured with required extensions.